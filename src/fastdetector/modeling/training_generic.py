import numpy as np
import torch
import torch.nn as nn
from accelerate import PartialState
from accelerate.utils import broadcast_object_list
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
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

from fastdetector.modeling.training_utils import get_data


def _log(message) -> None:
    """Print a message from the main process only.

    Args:
        message: Object to print.
    """
    if PartialState().is_main_process:
        print(message, flush=True)


def _bcast(obj):
    """Broadcast an object from the main process to every rank.

    Args:
        obj: Picklable object; only the main process's value is kept.

    Returns:
        The main process's value of ``obj``.
    """
    return broadcast_object_list([obj])[0]


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
            **Adapters are rank stabilized with rsLoRA.**
        lora_dropout: Dropout probability for LoRA layers.

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
        use_rslora=True,
        target_modules=target_modules,
        exclude_modules=exclude,
        task_type="SEQ_CLS"
    )

    peft_model = get_peft_model(peft_model, lora_config)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    _log(f"[build] attn={resolved_attn} norm={head_norm} bn={head_bottleneck} pad={tokenizer.padding_side} trainable={trainable:,}")
    return peft_model, tokenizer


def train(
    model: str = "meta-llama/Llama-3.2-3B",
    dataset: str = "pangram/editlens_iclr",
    n_rows: int = -1,
    n_buckets: int = 4,
    lo: float = 0.03,
    hi: float = 0.15,
    max_length: int = 1024,
    head_norm: bool = True,
    head_bottleneck: int = 0,
    head_dropout: float = 0.1,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lr: float = 3e-5,
    batch: int = 3,
    grad_accum: int = 1,
    epochs: float = 1,
    eval_steps: int = 200,
    seed: int = 42,
    trial=None,
) -> tuple[float, bool]:
    """Run one training job and evaluate it.

    Args:
        model: Base model path or Hugging Face repository ID.
        dataset: Dataset repository ID, loaded from its train split.
        n_rows: Rows to keep from the dataset; -1 for all.
        n_buckets: Number of output discrete bucket labels.
        lo: Lower edge of the graded scoring band.
        hi: Upper edge of the graded scoring band.
        max_length: Tokenizer truncation length.
        head_norm: Whether the classification head starts with a LayerNorm.
        head_bottleneck: Bottleneck width for the head; 0 for none.
        head_dropout: Dropout after the head's bottleneck.
        lora_r: Rank of LoRA adapters.
        lora_alpha: Scaling parameter of LoRA adapters.
        lora_dropout: Dropout probability for LoRA layers.
        lr: Learning rate.
        batch: Per-device batch size.
        grad_accum: Gradient accumulation steps.
        epochs: Training epochs.
        eval_steps: Steps between evaluations.
        seed: Seed for weights, data order, and shuffling.
        trial: Optuna trial to report intermediate scores to, enabling pruning.

    Returns:
        Tuple of (eval macro F1, whether the run was pruned), identical on
        every rank. The full metrics are printed.
    """
    set_seed(seed)
    PartialState()

    peft_model, tokenizer = build(
        model=model, n_buckets=n_buckets, head_norm=head_norm,
        head_bottleneck=head_bottleneck, head_dropout=head_dropout,
        lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
    )

    world_size = PartialState().num_processes
    _log(f"[train] batch={batch}x{grad_accum}x{world_size}={batch * grad_accum * world_size}")

    raw_ds = load_dataset(dataset, split="train")

    train_ds, val_ds = get_data(
        dataset=raw_ds, tokenizer=tokenizer, n_buckets=n_buckets,
        lo=lo, hi=hi, max_length=max_length, seed=seed, n_rows=n_rows,
    )
    _log(f"[data] train={len(train_ds)} val={len(val_ds)}")

    def compute_metrics(eval_pred) -> dict:
        """Score bucket predictions.

        Args:
            eval_pred: Tuple of (logits, labels) from the Trainer.

        Returns:
            Dict of accuracy, macro_f1, and mae.
        """
        preds, labels = np.argmax(eval_pred[0], -1), eval_pred[1]
        per_class_f1 = []
        for bucket in range(n_buckets):
            true_pos = ((preds == bucket) & (labels == bucket)).sum()
            precision = true_pos / max((preds == bucket).sum(), 1)
            recall = true_pos / max((labels == bucket).sum(), 1)
            per_class_f1.append(2 * precision * recall / (precision + recall)
                                if precision + recall else 0.0)
        return {"accuracy": float((preds == labels).mean()),
                "macro_f1": float(np.mean(per_class_f1)),
                "mae": float(np.abs(preds - labels).mean())}

    run_state = {"pruned": False}

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
            score = (metrics or {}).get("eval_macro_f1")
            if score is None:
                return control
            stop = False
            if PartialState().is_main_process and trial is not None:
                trial.report(score, trainer_state.global_step)
                stop = bool(trial.should_prune())
            if _bcast(stop):
                run_state["pruned"] = True
                control.should_training_stop = True
            return control

    trainer = Trainer(
        model=peft_model,
        args=TrainingArguments(
            num_train_epochs=epochs,
            learning_rate=lr, per_device_train_batch_size=batch,
            per_device_eval_batch_size=batch,
            gradient_accumulation_steps=grad_accum,
            lr_scheduler_type="constant", bf16=True,
            eval_strategy="steps", eval_steps=eval_steps, save_strategy="no",
            logging_steps=max(eval_steps // 4, 1),
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
    metrics = trainer.evaluate()
    _log(f"[eval] macro_f1={metrics['eval_macro_f1']:.4f} accuracy={metrics['eval_accuracy']:.4f} "
         f"mae={metrics['eval_mae']:.4f} loss={metrics['eval_loss']:.4f} "
         f"train_loss={train_output.training_loss:.4f} pruned={run_state['pruned']} "
         f"peak_gb={torch.cuda.max_memory_reserved() / 1024**3:.2f}")
    return _bcast((float(metrics["eval_macro_f1"]), run_state["pruned"]))

