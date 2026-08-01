"""QLoRA sequence-classification training for EditLens bucket prediction.

Subcommands: ``train`` (single run), ``sweep`` (optuna search), ``smoke``
(tiny synthetic run that exercises the whole path).
"""

import argparse
import json
import random
import re
from pathlib import Path

from accelerate import PartialState
from accelerate.utils import broadcast_object_list
from datasets import Dataset, load_dataset
import numpy as np
import optuna
from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn
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
import yaml

from fastdetector.modeling.preprocessing import clean_text


def log(message) -> None:
    """Print a message from the main process only.

    Args:
        message: Object to print.
    """
    if PartialState().is_main_process:
        print(message, flush=True)


def bcast(obj):
    """Broadcast an object from the main process to every rank.

    WHY: if ranks disagree on a control-flow decision (prune, next params) one
    exits while the others wait in all_reduce, and the job hangs until the wall
    clock kills it -- which looks like a timeout, not a pruned trial.

    Args:
        obj: Picklable object; only the main process's value is kept.

    Returns:
        The main process's value of ``obj``.
    """
    return broadcast_object_list([obj])[0]


# --------------------------------------------------------------------------- #
# EditLens preprocessing (transcribed from pangramlabs/EditLens preprocess.py)
# --------------------------------------------------------------------------- #

def count_words(text: str) -> int:
    """Count word characters runs in a string.

    Args:
        text: Input text string.

    Returns:
        Number of words.
    """
    return len(re.findall(r"\b\w+\b", text))


def to_bucket(score: float, n_buckets: int, lo: float, hi: float) -> int:
    """Discretize a cosine score into an ordinal bucket.

    Scores at or below ``lo`` collapse to bucket 0 and scores at or above ``hi``
    to the top bucket; the band between them is split evenly across the rest.

    Args:
        score: Cosine score to bucket.
        n_buckets: Total number of buckets.
        lo: Lower edge of the graded band.
        hi: Upper edge of the graded band.

    Returns:
        Bucket index in ``[0, n_buckets)``.
    """
    if score <= lo:
        return 0
    if score >= hi:
        return n_buckets - 1
    return 1 + int((score - lo) / (hi - lo) * (n_buckets - 2))


# Classification heads, all called as (hidden, n_labels, bottleneck, dropout,
# **module_kwargs); the shallow ones ignore the bottleneck and dropout.
HEADS = {
    "linear": lambda hidden, n_labels, *_, **kw: nn.Linear(hidden, n_labels, bias=False, **kw),
    "normed": lambda hidden, n_labels, *_, **kw: nn.Sequential(
        nn.LayerNorm(hidden, **kw),
        nn.Linear(hidden, n_labels, bias=False, **kw)),
    "mlp": lambda hidden, n_labels, bottleneck, dropout, **kw: nn.Sequential(
        nn.LayerNorm(hidden, **kw),
        nn.Linear(hidden, bottleneck, **kw),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(bottleneck, n_labels, bias=False, **kw)),
}


# --------------------------------------------------------------------------- #

def quirks(model_name: str) -> tuple[str | None, str]:
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

    class Gemma4ForSequenceClassification(GenericForSequenceClassification,
                                          Gemma4PreTrainedModel):
        config: Gemma4Config

    try:
        AutoModelForSequenceClassification.register(Gemma4Config, Gemma4ForSequenceClassification)
    except ValueError:
        pass

    return r".*(vision_tower|audio_tower|visual|embed_vision|embed_audio).*", "sdpa"


def build(args: argparse.Namespace) -> tuple:
    """Load the 4bit base model, swap in a fresh head, and wrap it in LoRA.

    Args:
        args: Parsed CLI/config namespace.

    Returns:
        Tuple of (peft_model, tokenizer, info), where info records the build
        choices worth reporting back to a sweep.
    """
    exclude, attn = quirks(args.model)
    if args.attn != "auto":
        attn = args.attn
    world_size = PartialState().num_processes

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    config = AutoConfig.from_pretrained(args.model)
    config.num_labels = args.n_buckets
    text_config = config.get_text_config()
    if text_config is not config:
        text_config.num_labels = args.n_buckets
        if not hasattr(config, "hidden_size"):
            config.hidden_size = text_config.hidden_size
    hidden_size, dtype = text_config.hidden_size, torch.bfloat16

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, config=config, dtype=dtype,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True),
        device_map={"": PartialState().local_process_index},
        attn_implementation=attn)
    model.config.pad_token_id = tokenizer.pad_token_id

    model.score = HEADS[args.head](hidden_size, args.n_buckets, args.head_bottleneck,
                                   args.head_dropout, dtype=dtype, device=model.device)

    # peft's prepare_model_for_kbit_training inlined, minus its unconditional
    # upcast of every non-4bit param to fp32 -- which the default fp32_upcast=0
    # then had to walk the model a second time to undo.
    for param in model.parameters():
        param.requires_grad = False
    if args.fp32_upcast:
        for param in model.parameters():
            if param.dtype == dtype and type(param).__name__ != "Params4bit":
                param.data = param.data.to(torch.float32)
    if args.grad_ckpt:
        # WHY non-reentrant: the reentrant path needs a grad-requiring input to
        # each checkpointed segment, which a fully frozen 4bit base cannot give.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        exclude_modules=exclude, task_type="SEQ_CLS"))
    model.config.use_cache = False

    checkpointing = any(getattr(m, "gradient_checkpointing", False) for m in model.modules())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[build] attn={attn} head={args.head} ckpt={checkpointing} "
        f"pad={tokenizer.padding_side} trainable={trainable:,} "
        f"batch={args.batch}x{args.grad_accum}x{world_size}"
        f"={args.batch * args.grad_accum * world_size}")
    return model, tokenizer, dict(attn=attn, ckpt=checkpointing,
                                  trainable=trainable, head=args.head)


# --------------------------------------------------------------------------- #

_SYNTHETIC_VOCAB = (
    "the of and to in a is that for it as was with be by on not this are from at which "
    "model research writing student essay analysis however therefore data language").split()


def get_data(args: argparse.Namespace, tokenizer) -> tuple[Dataset, Dataset]:
    """Load, filter, and tokenize the training corpus.

    Args:
        args: Parsed CLI/config namespace.
        tokenizer: Tokenizer matching the base model.

    Returns:
        Tuple of (train_dataset, eval_dataset), each carrying only input_ids,
        attention_mask, and label.
    """
    if args.data == "synthetic":
        rng = random.Random(args.seed)
        rows = {"text": [], "cosine_score": []}
        for _ in range(args.n_rows if args.n_rows > 0 else 2000):
            rows["text"].append(" ".join(rng.choice(_SYNTHETIC_VOCAB)
                                         for _ in range(rng.randint(90, 900))))
            roll = rng.random()
            rows["cosine_score"].append(
                rng.uniform(0, .03) if roll < .35 else
                rng.uniform(.15, 1) if roll < .65 else rng.uniform(.03, .15))
        ds = Dataset.from_dict(rows)
    else:
        ds = load_dataset(args.dataset, split="train")

    def keep(row):  # one pass, and count_words never sees a None
        return (row["text"] is not None and row["cosine_score"] is not None
                and count_words(row["text"]) >= 75)

    def prep(batch):
        encoded = tokenizer([clean_text(text) for text in batch["text"]],
                            truncation=True, max_length=args.max_length)
        encoded["label"] = [to_bucket(score, args.n_buckets, args.lo, args.hi)
                            for score in batch["cosine_score"]]
        return encoded

    # WHY main_process_first: every rank derives the same dataset, so without it
    # all of them tokenize the full corpus at once and race on one cache file.
    # The others then hit the cache datasets fingerprinted for rank 0.
    with PartialState().main_process_first():
        ds = ds.filter(keep).shuffle(seed=args.seed)
        if args.n_rows > 0:
            ds = ds.select(range(min(args.n_rows, len(ds))))
        ds = ds.map(prep, batched=True, remove_columns=ds.column_names)

    split = ds.train_test_split(test_size=0.05, seed=args.seed)
    return split["train"], split["test"]


# --------------------------------------------------------------------------- #

def train(args: argparse.Namespace, trial=None) -> dict:
    """Run one training job and evaluate it.

    Args:
        args: Parsed CLI/config namespace.
        trial: Optuna trial to report intermediate scores to, enabling pruning.
            None for a standalone run. Only the main process holds one.

    Returns:
        The eval metrics, plus train_loss, pruned, peak_gb, and the build info.
        Identical on every rank.
    """
    set_seed(args.seed)
    # Inits the process group and binds this rank's cuda device. Trainer does the
    # same on its first TrainingArguments, but we need both before from_pretrained
    # (and before the sweep's first bcast). It is a singleton, so Trainer reuses it.
    PartialState()

    model, tokenizer, info = build(args)
    train_ds, val_ds = get_data(args, tokenizer)
    log(f"[data] train={len(train_ds)} val={len(val_ds)}")

    def compute_metrics(eval_pred) -> dict:
        """Score bucket predictions.

        Args:
            eval_pred: Tuple of (logits, labels) from the Trainer.

        Returns:
            Dict of accuracy, macro_f1, and mae.
        """
        preds, labels = np.argmax(eval_pred[0], -1), eval_pred[1]
        per_class_f1 = []
        for bucket in range(args.n_buckets):
            true_pos = ((preds == bucket) & (labels == bucket)).sum()
            precision = true_pos / max((preds == bucket).sum(), 1)
            recall = true_pos / max((labels == bucket).sum(), 1)
            per_class_f1.append(2 * precision * recall / (precision + recall)
                                if precision + recall else 0.0)
        # mae because the buckets are ordinal: predicting 0 when the truth is 3
        # is worse than predicting 2, and accuracy is blind to that.
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
            if bcast(stop):
                run_state["pruned"] = True
                control.should_training_stop = True
            return control

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out, max_steps=args.max_steps, num_train_epochs=args.epochs,
            learning_rate=args.lr, per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch * 2,
            gradient_accumulation_steps=args.grad_accum,
            lr_scheduler_type="constant", weight_decay=0.0, bf16=True,
            eval_strategy="steps", eval_steps=args.eval_steps, save_strategy="no",
            logging_steps=max(args.eval_steps // 4, 1), report_to=[],
            seed=args.seed, data_seed=args.seed, remove_unused_columns=False,
            dataloader_num_workers=2,
            # WHY: checkpointing is already enabled on the model in build().
            gradient_checkpointing=False,
            # WHY False: it is the default anyway once checkpointing is on, and
            # any trainable param outside the forward graph then hard-errors.
            ddp_find_unused_parameters=False),
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[Prune()] if trial is not None else [])

    train_output = trainer.train()
    result = dict(trainer.evaluate(),
                  train_loss=float(train_output.training_loss),
                  pruned=run_state["pruned"],
                  peak_gb=round(torch.cuda.max_memory_reserved() / 1024**3, 2), **info)
    if PartialState().is_main_process:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    return bcast(result)


# --------------------------------------------------------------------------- #

def sweep(args: argparse.Namespace) -> None:
    """Search LoRA and head hyperparameters with optuna.

    The main process owns the study; every rank runs each trial's training job
    against parameters broadcast from it.

    Args:
        args: Parsed CLI/config namespace.
    """
    study = None
    if PartialState().is_main_process:
        study = optuna.create_study(
            study_name=args.study or args.model.split("/")[-1], storage=args.storage,
            direction="maximize", load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
            sampler=optuna.samplers.TPESampler(seed=args.seed))

    for _ in range(args.trials):
        trial = payload = None
        if PartialState().is_main_process:
            trial = study.ask()
            lora_r = trial.suggest_categorical("lora_r", [8, 16, 32, 64])
            payload = (trial.number, dict(
                lr=trial.suggest_float("lr", 1e-5, 5e-4, log=True), lora_r=lora_r,
                # WHY a ratio, not an absolute: LoRA scales its update by
                # alpha/r, so sweeping both independently re-samples the
                # same effective scale and wastes trials.
                lora_alpha=lora_r * trial.suggest_categorical("alpha_over_r", [1, 2, 4]),
                lora_dropout=trial.suggest_float("lora_dropout", 0.0, 0.15),
                head=trial.suggest_categorical("head", list(HEADS))))
        number, params = bcast(payload)

        trial_args = argparse.Namespace(**vars(args))
        for key, value in params.items():
            setattr(trial_args, key, value)
        trial_args.max_steps = args.trial_steps
        trial_args.out = f"{args.out}/trial_{number}"
        log(f"\n[trial {number}] {json.dumps(params)}")

        try:
            result = train(trial_args, trial=trial)
        except Exception as e:
            log(f"[trial {number}] FAILED {type(e).__name__}: {e}")
            if PartialState().is_main_process:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            continue

        if PartialState().is_main_process:
            if result["pruned"]:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            else:
                trial.set_user_attr("info", {k: result[k] for k in
                                             ("attn", "ckpt", "trainable", "peak_gb")})
                study.tell(trial, result["eval_macro_f1"])
            log(f"[trial {number}] macro_f1={result.get('eval_macro_f1'):.4f}"
                f"{' (pruned)' if result['pruned'] else ''}")

    if PartialState().is_main_process:
        log("\nbest:")
        for best in study.best_trials[:5]:
            log(f"  #{best.number}  {best.value:.4f}  {best.params}")


# --------------------------------------------------------------------------- #

def main() -> None:
    """Parse arguments and dispatch to the requested subcommand."""
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["train", "sweep", "smoke"])
    parser.add_argument("--config", help="YAML whose keys override these defaults")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--head", default="normed", choices=list(HEADS))
    parser.add_argument("--head-bottleneck", type=int, default=512)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--attn", default="auto",
                        help="override the attn_implementation quirks() pins for the model")
    parser.add_argument("--data", default="hf", choices=["hf", "synthetic"])
    parser.add_argument("--dataset", default="pangram/editlens_iclr")
    parser.add_argument("--n-rows", type=int, default=-1)
    parser.add_argument("--n-buckets", type=int, default=4)
    parser.add_argument("--lo", type=float, default=0.03)
    parser.add_argument("--hi", type=float, default=0.15)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--grad-ckpt", type=int, default=1)
    parser.add_argument("--fp32-upcast", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="runs/default")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--trial-steps", type=int, default=800)
    parser.add_argument("--study")
    parser.add_argument("--storage", default="sqlite:///sweeps.db")
    args = parser.parse_args()

    if args.config:
        for key, value in (yaml.safe_load(open(args.config)) or {}).items():
            setattr(args, key.replace("-", "_"), value)

    if args.cmd == "smoke":
        args.data, args.n_rows, args.max_steps, args.eval_steps = "synthetic", 200, 10, 5
        args.out = f"{args.out}/smoke"
    sweep(args) if args.cmd == "sweep" else train(args)


if __name__ == "__main__":
    main()
