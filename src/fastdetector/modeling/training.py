import numpy as np
import torch
import torch.nn as nn
import wandb
from accelerate import PartialState
from accelerate.utils import broadcast_object_list
from peft import LoraConfig, get_peft_model
from scipy.special import softmax
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from fastdetector.modeling.training_utils import load_splits
from fastdetector.visualization.metrics import classifier_metrics


def log(message) -> None:
    """Print a message from the main process only.

    Args:
        message: Object to print.
    """
    if PartialState().is_main_process:
        print(message, flush=True)


def bcast(obj):
    """Broadcast an object from the main process to every rank.

    Args:
        obj: Picklable object; only the main process's value is kept.

    Returns:
        The main process's value of ``obj``.
    """
    return broadcast_object_list([obj])[0]


def _tpr_at_fpr(fprs: np.ndarray, tprs: np.ndarray, target: float) -> float:
    """Read the best true positive rate a false positive budget buys.

    Args:
        fprs: False positive rates from ``roc_curve``, ascending.
        tprs: True positive rates, aligned with *fprs*.
        target: Largest false positive rate the operating point may have.

    Returns:
        The highest true positive rate whose false positive rate is within
        *target*, or NaN when the curve is undefined.
    """
    within = np.flatnonzero(fprs <= target)
    return float(tprs[within[-1]]) if within.size else float("nan")


def _build_head(hidden: int, n_labels: int, norm: bool = True,
               bottleneck: int = 0, dropout: float = 0.1, **kw) -> nn.Module:
    """Build the classification head from two orthogonal shape choices.

    The defaults give a normed linear head: a LayerNorm over the pooled hidden
    state, then an unbiased projection to the labels.

    Args:
        hidden: Input width, the base model's hidden size.
        n_labels: Number of output logits.
        norm: Whether to prefix a LayerNorm.
        bottleneck: Width of an intermediate GELU projection; 0 for none.
        dropout: Dropout after the bottleneck. Unused when bottleneck is 0.
        **kw: dtype and device, forwarded to every parameterized layer.

    Returns:
        The head as an ``nn.Sequential``.
    """
    layers = [nn.LayerNorm(hidden, **kw)] if norm else []
    if bottleneck:
        layers += [nn.Linear(hidden, bottleneck, **kw), nn.GELU(), nn.Dropout(dropout)]
        hidden = bottleneck
    layers.append(nn.Linear(hidden, n_labels, bias=False, **kw))
    return nn.Sequential(*layers)


def _quirks(model_name: str) -> tuple[str | None, str]:
    """Register any missing task head, and return this model's build overrides.

    The only place model-specific code belongs. If a model needs a change
    anywhere else, this abstraction is wrong.

    Args:
        model_name: Base model path or HuggingFace repository ID.

    Returns:
        Tuple of (lora_exclude_pattern, attn_implementation), where the pattern
        is None when no module needs excluding from LoRA.
    """
    if "gemma-4" not in model_name.lower() and "gemma4" not in model_name.lower():
        return None, "flash_attention_2"

    from transformers.modeling_layers import GenericForSequenceClassification
    from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
    from transformers.models.gemma4.modeling_gemma4 import Gemma4PreTrainedModel

    class Gemma4ForSequenceClassification(GenericForSequenceClassification, Gemma4PreTrainedModel):
        config: Gemma4Config

    try:
        AutoModelForSequenceClassification.register(Gemma4Config, Gemma4ForSequenceClassification)
    except ValueError:
        pass

    return r".*(vision_tower|audio_tower|visual|embed_vision|embed_audio).*", "sdpa"


def build(
    model: str,
    n_buckets: int = 4,
    head_norm: bool = True,
    head_bottleneck: int = 0,
    head_dropout: float = 0.1,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    use_rslora: bool = False,
) -> tuple:
    """Load the 4bit base model, swap in a fresh head, and wrap it in LoRA.

    Args:
        model: Base model path or Hugging Face repository ID.
        n_buckets: Number of output discrete bucket labels.
        head_norm: Whether the classification head starts with a LayerNorm.
        head_bottleneck: Bottleneck width for the head; 0 for none.
        head_dropout: Dropout after the head's bottleneck.
        lora_r: Rank of LoRA adapters.
        lora_alpha: Scaling parameter of LoRA adapters.
        lora_dropout: Dropout probability for LoRA layers.
        use_rslora: Whether to rank stabilize the adapters, scaling by
            ``lora_alpha / sqrt(lora_r)`` instead of ``lora_alpha / lora_r``.

    Returns:
        Tuple of (peft_model, tokenizer).
    """
    exclude, resolved_attn = _quirks(model)

    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model)
    config.num_labels = n_buckets
    hidden_size = config.get_text_config().hidden_size
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=True
    )

    peft_model = AutoModelForSequenceClassification.from_pretrained(
        model,
        config=config,
        dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map={"": PartialState().local_process_index},
        attn_implementation=resolved_attn
    )
    
    peft_model.config.get_text_config().pad_token_id = tokenizer.pad_token_id

    peft_model.score = _build_head(hidden_size, n_buckets, head_norm, head_bottleneck, head_dropout, dtype=torch.float32, device=peft_model.device)
    for param in peft_model.parameters():
        param.requires_grad = False
        
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        r=lora_r, 
        lora_alpha=lora_alpha, 
        lora_dropout=lora_dropout, 
        bias="none",
        use_rslora=use_rslora,
        target_modules=target_modules,
        exclude_modules=exclude,
        task_type="SEQ_CLS"
    )

    peft_model = get_peft_model(peft_model, lora_config)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    log(f"[build] attn={resolved_attn} norm={head_norm} bn={head_bottleneck} pad={tokenizer.padding_side} trainable={trainable:,}")
    return peft_model, tokenizer


def train(
    wandb_project: str,
    model: str = "meta-llama/Llama-3.2-3B",
    dataset: str = "pangram/editlens_iclr",
    save_dir: str = "/usr/project/xtmp/fastdetector_models",
    bucket_strategy: str = "editlens",
    n_buckets: int = 4,
    lo: float = 0.03,
    hi: float = 0.15,
    max_length: int = 1024,
    min_words: int = 75,
    head_norm: bool = True,
    head_bottleneck: int = 0,
    head_dropout: float = 0.1,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    use_rslora: bool = False,
    lr: float = 3e-5,
    batch: int = 3,
    eval_batch: int = 16,
    grad_accum: int = 1,
    epochs: float = 1,
    eval_rows: int = 3000,
    seed: int = 42,
    trial=None,
) -> tuple[float, bool]:
    """Run one training job and evaluate it.

    Args:
        wandb_project: Project to log the run to, one wandb run per call,
            grouped by model. Required; every run is logged.
        model: Base model path or Hugging Face repository ID.
        dataset: Dataset repository ID, loaded from its train split.
        save_dir: Directory the adapter and head are written to, once, after a
            run that was not pruned. Nothing is written for a pruned run. A
            trial saves to a ``trial_<number>`` subdirectory of it.
        bucket_strategy: Which strategy cuts the edit scores into labels.
        n_buckets: Number of output discrete bucket labels.
        lo: Lower edge of the graded scoring band.
        hi: Upper edge of the graded scoring band.
        max_length: Tokenizer truncation length.
        min_words: Shortest text to keep.
        head_norm: Whether the classification head starts with a LayerNorm.
        head_bottleneck: Bottleneck width for the head; 0 for none.
        head_dropout: Dropout after the head's bottleneck.
        lora_r: Rank of LoRA adapters.
        lora_alpha: Scaling parameter of LoRA adapters.
        lora_dropout: Dropout probability for LoRA layers.
        use_rslora: Whether to rank stabilize the adapters. Off keeps the
            official scaling, ``lora_alpha / lora_r``.
        lr: Learning rate.
        batch: Per-device batch size.
        eval_batch: Per-device batch size for evaluation, which is forward
            only and so need not follow the searched train batch.
        grad_accum: Gradient accumulation steps.
        epochs: Training epochs.
        eval_rows: Rows between evaluations. Held in rows rather than steps so
            that a sweep over the effective batch evaluates every run at the
            same points in the data: an interval counted in optimizer steps is
            worth sixteen times as much text at an effective batch of 48 as at
            3, which lands the runs on different rungs of the pruning ladder
            and has the large batches judged on far more training than the
            small ones.
        seed: Seed for weights, data order, and shuffling.
        trial: Optuna trial to report intermediate scores to, enabling pruning.

    Returns:
        Tuple of (eval score AUROC, whether the run was pruned), identical on
        every rank. The full metrics are printed.
    """
    set_seed(seed)
    PartialState()

    peft_model, tokenizer = build(
        model=model, n_buckets=n_buckets, head_norm=head_norm,
        head_bottleneck=head_bottleneck, head_dropout=head_dropout,
        lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        use_rslora=use_rslora,
    )

    world_size = PartialState().num_processes
    log(f"[train] batch={batch}x{grad_accum}x{world_size}={batch * grad_accum * world_size}")

    if PartialState().is_main_process:
        config = {
            "model": model,
            "dataset": dataset,
            "bucket_strategy": bucket_strategy,
            "n_buckets": n_buckets,
            "lo": lo,
            "hi": hi,
            "max_length": max_length,
            "head_norm": head_norm,
            "head_bottleneck": head_bottleneck,
            "head_dropout": head_dropout,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "use_rslora": use_rslora,
            "lr": lr,
            "batch": batch,
            "grad_accum": grad_accum,
            "epochs": epochs,
            "seed": seed,
        }
        wandb.init(project=wandb_project, config=config,
                   name="single" if trial is None else f"trial_{trial.number}",
                   reinit=True)
        wandb.define_metric("iteration_count")
        wandb.define_metric("train/*", step_metric="iteration_count")
        wandb.define_metric("eval/*", step_metric="iteration_count")
        wandb.define_metric("eval_score/*", step_metric="iteration_count")
        wandb.define_metric("eval_bucket/*", step_metric="iteration_count")

    train_ds, val_ds, val_scores = load_splits(dataset, tokenizer, bucket_strategy,
                                               n_buckets, lo, hi, max_length,
                                               min_words, seed)
    log(f"[data] train={len(train_ds)} val={len(val_ds)}")
    run_state = {"pruned": False, "trained": False}
    rows_per_step = batch * grad_accum * world_size
    eval_steps = max(1, round(eval_rows / rows_per_step))

    def compute_metrics(eval_pred) -> dict:
        """Score the model as the binary detector it is deployed as.

        Args:
            eval_pred: Tuple of (logits, labels) from the Trainer.

        Returns:
            Dict of score_auroc, the swept objective, plus pearson_bucket,
            pearson_score, accuracy, macro_f1, and the per bucket precision,
            recall, f1, predicted count and actual count.
        """
        probs, labels = softmax(eval_pred[0], axis=1), eval_pred[1]
        buckets = np.argmax(probs, axis=1)
        scores = probs @ np.arange(n_buckets) / (n_buckets - 1)
        if len(val_scores) != len(labels):
            raise ValueError("eval predictions are not aligned with the val split")
        is_ai = val_scores > 0
        step, main = trainer.state.global_step, PartialState().is_main_process

        bucket = classifier_metrics(buckets, is_ai, threshold=0, flip=False)
        if main:
            wandb.log({"iteration_count": step,
                       "eval_bucket/auroc": bucket["auroc"],
                       "eval_bucket/f1": bucket["f1"],
                       "eval_bucket/accuracy": bucket["accuracy"],
                       "eval_bucket/tpr": bucket["tpr"],
                       "eval_bucket/fpr": bucket["fpr"]})

        if 0 < int(is_ai.sum()) < is_ai.size:
            score_auroc = float(roc_auc_score(is_ai, scores))
            fprs, tprs, _ = roc_curve(is_ai, scores)
        else:
            score_auroc, fprs, tprs = float("nan"), np.empty(0), np.empty(0)
        if main:
            wandb.log({"iteration_count": step,
                       "eval_score/auroc": score_auroc,
                       "eval_score/tpr_at_fpr_1pct": _tpr_at_fpr(fprs, tprs, 0.01),
                       "eval_score/tpr_at_fpr_0_5pct": _tpr_at_fpr(fprs, tprs, 0.005),
                       "eval_score/tpr_at_fpr_0_1pct": _tpr_at_fpr(fprs, tprs, 0.001)})

        precision, recall, f1, actual = precision_recall_fscore_support(
            labels, buckets, labels=np.arange(n_buckets), zero_division=0)
        predicted = np.bincount(buckets, minlength=n_buckets)

        return {"score_auroc": score_auroc,
                "pearson_bucket": float(np.corrcoef(buckets, val_scores)[0, 1]),
                "pearson_score": float(np.corrcoef(scores, val_scores)[0, 1]),
                "accuracy": float((buckets == labels).mean()),
                "macro_f1": float(f1.mean()),
                **{f"precision_{i}": float(v) for i, v in enumerate(precision)},
                **{f"recall_{i}": float(v) for i, v in enumerate(recall)},
                **{f"f1_{i}": float(v) for i, v in enumerate(f1)},
                **{f"predicted_{i}": int(v) for i, v in enumerate(predicted)},
                **{f"actual_{i}": int(v) for i, v in enumerate(actual)}}

    class Prune(TrainerCallback):
        """Stops training when optuna prunes the trial."""

        def on_evaluate(self, train_args, trainer_state, control, metrics=None, **kwargs):
            """Report the eval score to optuna and stop if it says to prune.

            Args:
                train_args: The run's TrainingArguments.
                trainer_state: The Trainer's state, read for global_step.
                control: TrainerControl to signal the stop through.
                metrics: Metrics just computed by the Trainer.
                **kwargs: Unused Trainer internals.

            Returns:
                The (possibly modified) TrainerControl.
            """
            score = (metrics or {}).get("eval_score_auroc")
            if score is None or run_state["trained"]:
                return control
            stop = False
            if PartialState().is_main_process and trial is not None:
                trial.report(score, trainer_state.global_step * rows_per_step)
                stop = bool(trial.should_prune())
            if bcast(stop):
                run_state["pruned"] = True
                control.should_training_stop = True
            return control

    trainer = Trainer(
        model=peft_model,
        args=TrainingArguments(
            num_train_epochs=epochs,
            learning_rate=lr, per_device_train_batch_size=batch,
            per_device_eval_batch_size=eval_batch,
            gradient_accumulation_steps=grad_accum,
            lr_scheduler_type="constant", bf16=True,
            eval_strategy="steps", eval_steps=eval_steps, save_strategy="no",
            report_to="wandb",
            logging_steps=50,
            seed=seed, data_seed=seed, remove_unused_columns=False,
            dataloader_num_workers=2,
            gradient_checkpointing=True,
            ddp_find_unused_parameters=False,
        ),
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[Prune()],
    )

    train_output = trainer.train()
    run_state["trained"] = True
    metrics = trainer.evaluate()
    log(f"[eval] score_auroc={metrics['eval_score_auroc']:.4f} "
        f"pearson_score={metrics['eval_pearson_score']:.4f} "
        f"pearson_bucket={metrics['eval_pearson_bucket']:.4f} "
        f"macro_f1={metrics['eval_macro_f1']:.4f} accuracy={metrics['eval_accuracy']:.4f} "
        f"loss={metrics['eval_loss']:.4f} train_loss={train_output.training_loss:.4f} "
        f"pruned={run_state['pruned']} "
        f"peak_gb={torch.cuda.max_memory_reserved() / 1024**3:.2f}")

    if not run_state["pruned"] and PartialState().is_main_process:
        path = save_dir if trial is None else f"{save_dir}/trial_{trial.number}"
        peft_model.save_pretrained(path)
        log(f"[save] {path}")

    if PartialState().is_main_process:
        wandb.finish()

    return bcast((float(metrics["eval_score_auroc"]), run_state["pruned"]))

