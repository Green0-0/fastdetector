import argparse
import os
import subprocess

import optuna
import torch
from accelerate import PartialState

from fastdetector.modeling.training import bcast, train
from fastdetector.modeling.training_utils import get_or_create_study

# The parts of the official EditLens recipe the search leaves alone.
FIXED = dict(
    model="meta-llama/Llama-3.2-3B",
    dataset="pangram/editlens_iclr",
    lo=0.03,
    hi=0.15,
    max_length=1024,
    min_words=75,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    epochs=1,
    eval_steps=500,
    seed=42,
)
BUCKET_STRATEGIES = ["editlens", "naive", "human", "human_editlens", "human_size"]
DEVICE_BATCH = 3
EFFECTIVE_BATCHES = [3, 6, 12, 24, 48]


def _split_batch(effective: int) -> dict:
    """Reach an effective batch with a per device batch and accumulation.

    Args:
        effective: Rows the optimizer should step on, across every rank.

    Returns:
        Dict of the ``batch`` and ``grad_accum`` that come nearest to it.
    """
    world = PartialState().num_processes
    batch = max(1, min(DEVICE_BATCH, effective // world))
    return {"batch": batch, "grad_accum": max(1, round(effective / (batch * world)))}


def run_trial(args) -> None:
    """Run one trial, in lockstep on every rank of this launch.

    Only the main process holds the study. It suggests the trial's parameters
    and broadcasts them, so every rank trains the same model and reaches the
    same collectives; the other ranks never touch optuna.

    Args:
        args: Parsed command line arguments.
    """
    if not PartialState().is_main_process:
        train(wandb_project=args.wandb_project, **bcast(None), **FIXED)
        return

    study = get_or_create_study(args.study_name, args.journal_path,
                                min_resource=FIXED["eval_steps"],
                                reduction_factor=args.reduction_factor)

    def objective(trial) -> float:
        """Suggest a labelling and the head and optimizer to learn it with.

        How the scores are cut is what the search is really about, so the
        strategy and the bucket count lead. The learning rate, the effective
        batch and the head shape ride along, since the labelling changes the
        target's shape and the best settings for one need not suit another.
        They stay near the official values rather than ranging freely, which
        keeps the official run inside the space at editlens, 4 buckets, lr
        3e-5, an effective batch of 3 and a normed linear head.

        Only the effective batch is searched. How it is reached is a hardware
        question: fill the device up to ``DEVICE_BATCH``, spread it over the
        ranks, and accumulate whatever is left over. Searching the per device
        batch and the accumulation apart would just be searching the same
        optimizer step more than once.

        Args:
            trial: Optuna trial to suggest from.

        Returns:
            The trial's eval score AUROC.

        Raises:
            optuna.TrialPruned: If the pruner stopped the run early.
        """
        params = bcast(dict(
            bucket_strategy=trial.suggest_categorical("bucket_strategy", BUCKET_STRATEGIES),
            n_buckets=trial.suggest_int("n_buckets", 2, 12),
            lr=trial.suggest_float("lr", 1e-5, 1e-4, log=True),
            **_split_batch(trial.suggest_categorical("effective_batch", EFFECTIVE_BATCHES)),
            head_norm=trial.suggest_categorical("head_norm", [False, True]),
            head_bottleneck=trial.suggest_categorical("head_bottleneck", [0, 512])))
        score, pruned = train(wandb_project=args.wandb_project, trial=trial,
                              **params, **FIXED)
        if pruned:
            raise optuna.TrialPruned()
        return score

    study.optimize(objective, n_trials=1)


def main() -> None:
    """Drive the labelling ablation, one launch per trial.

    A trial holds a quantized base model for a whole epoch, so each one gets a
    fresh launch that is then thrown away, keeping one trial's allocator state
    off the next one. The launch is what spreads the trial over the GPUs.
    """
    parser = argparse.ArgumentParser(
        description="Ablate how EditLens labels its edit scores."
    )
    parser.add_argument("--wandb-project", type=str, required=True)
    parser.add_argument("--trials", type=int, default=10, help="Trials to run on this node")
    parser.add_argument("--journal-path", type=str, default="sweeps/ablate_editlens.journal")
    parser.add_argument("--study-name", type=str, default="ablate_editlens")
    parser.add_argument("--reduction-factor", type=int, default=3)
    parser.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--main-process-port", type=int, default=None,
                        help="Rendezvous port for the launch; 0 picks a free one. "
                             "Needed when two drivers share a node, since they "
                             "would otherwise both take accelerate's default.")
    parser.add_argument("--worker", action="store_true",
                        help="Run one trial. Set by the driver, not by hand.")
    args = parser.parse_args()

    if args.worker:
        run_trial(args)
        return

    study = get_or_create_study(args.study_name, args.journal_path,
                                min_resource=FIXED["eval_steps"],
                                reduction_factor=args.reduction_factor)

    print(f"\n#--- Starting EditLens labelling ablation ({args.trials} trials on {args.gpus} GPUs) ---#")
    launch = ["accelerate", "launch", "--num_processes", str(args.gpus)]
    if args.main_process_port is not None:
        launch += ["--main_process_port", str(args.main_process_port)]
    launch += [os.path.abspath(__file__), "--worker",
               "--wandb-project", args.wandb_project, "--journal-path", args.journal_path,
               "--study-name", args.study_name, "--reduction-factor", str(args.reduction_factor)]

    for index in range(args.trials):
        print(f"\n--- Initiating Trial {index + 1}/{args.trials} ---")
        completed = subprocess.run(launch)
        if completed.returncode:
            print(f"Trial launch exited with code {completed.returncode}")

    print("\n#--- Sweep Complete ---#")
    for best in study.best_trials[:5]:
        print(f"  #{best.number}  {best.value:.4f}  {best.params}")


if __name__ == "__main__":
    main()
