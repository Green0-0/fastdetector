import argparse
import os
import subprocess
import uuid

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
    eval_rows=3000,
    seed=42,
)


def _split_batch(effective: int) -> dict:
    """Reach an effective batch with a per device batch and accumulation.

    Args:
        effective: Rows the optimizer should step on, across every rank.

    Returns:
        Dict of the ``batch`` and ``grad_accum`` that reach it exactly.

    Raises:
        ValueError: if the ranks cannot share *effective* evenly.
    """
    world = PartialState().num_processes
    if effective % world:
        raise ValueError(f"Effective batch {effective} does not divide over {world} ranks; every entry of EFFECTIVE_BATCHES must.")
    per_rank = effective // world
    batch = max(d for d in range(1, min(8, per_rank) + 1) if per_rank % d == 0)
    return {"batch": batch, "grad_accum": per_rank // batch}


def run_trial(args) -> None:
    """Run one trial, in lockstep on every rank of this launch.

    Args:
        args: Parsed command line arguments.
    """
    if not PartialState().is_main_process:
        train(wandb_project=args.wandb_project, **bcast(None), **FIXED)
        return

    study = get_or_create_study(args.study_name, args.journal_path, min_resource=3000, reduction_factor=args.reduction_factor)

    def objective(trial) -> float:
        """Suggest a labelling and the head and optimizer to learn it with.
        
        Args:
            trial: Optuna trial to suggest from.

        Returns:
            The trial's eval score AUROC.

        Raises:
            optuna.TrialPruned: If the pruner stopped the run early.
        """
        # Stamped before anything can fail, so a launch that dies without telling optuna anything can still be traced back to its trial.
        trial.set_user_attr("launch_id", os.environ.get("FD_LAUNCH_ID", ""))
        params = bcast(dict(
            bucket_strategy=trial.suggest_categorical("bucket_strategy", ["editlens", "naive", "human", "human_editlens", "human_size"]),
            n_buckets=trial.suggest_int("n_buckets", 2, 12),
            lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            **_split_batch(trial.suggest_categorical("effective_batch", [16, 32, 64, 128])),
            head_norm=trial.suggest_categorical("head_norm", [False, True]),
            head_bottleneck=trial.suggest_categorical("head_bottleneck", [0, 128, 512])
        ))
        score, pruned = train(wandb_project=args.wandb_project, trial=trial, **params, **FIXED)
        if pruned:
            trial.set_user_attr("score_at_prune", score)
            raise optuna.TrialPruned()
        return score

    study.optimize(objective, n_trials=1)
    

def main() -> None:
    """Drive the labelling ablation, one launch per trial."""
    parser = argparse.ArgumentParser(
        description="Ablate how EditLens labels its edit scores."
    )
    parser.add_argument("--wandb-project", type=str, required=True)
    parser.add_argument("--trials", type=int, default=10, help="Trials to run on this node")
    parser.add_argument("--journal-path", type=str, default="sweeps/ablate_editlens.journal")
    parser.add_argument("--study-name", type=str, default="ablate_editlens")
    parser.add_argument("--reduction-factor", type=int, default=3)
    parser.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--main-process-port", type=int, default=None)
    parser.add_argument("--worker", action="store_true", help="Run one trial. Set by the driver, not by hand.")
    args = parser.parse_args()

    if args.worker:
        run_trial(args)
        return

    study = get_or_create_study(args.study_name, args.journal_path, min_resource=3000, reduction_factor=args.reduction_factor)

    print(f"\n#--- Starting EditLens labelling ablation ({args.trials} trials on {args.gpus} GPUs) ---#")
    launch = ["accelerate", "launch", "--num_processes", str(args.gpus)]
    
    if args.main_process_port is not None:
        launch += ["--main_process_port", str(args.main_process_port)]
        
    launch += [
        os.path.abspath(__file__), 
        "--worker",
        "--wandb-project", args.wandb_project, 
        "--journal-path", args.journal_path,
        "--study-name", args.study_name, 
        "--reduction-factor", str(args.reduction_factor)
    ]

    for index in range(args.trials):
        print(f"\n--- Initiating Trial {index + 1}/{args.trials} ---")
        launch_id = uuid.uuid4().hex
        completed = subprocess.run(launch, env={**os.environ, "FD_LAUNCH_ID": launch_id})
        if completed.returncode:
            running = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,))
            where = "no trial claimed"
            for trial in running:
                if trial.user_attrs.get("launch_id") == launch_id:
                    study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
                    where = f"trial {launch_id} marked failed"
                    break
            print(f"Trial launch exited with code {completed.returncode}; {where}")

    print("\n#--- Sweep Complete ---#")
    for best in study.best_trials[:5]:
        print(f"  #{best.number}  {best.value:.4f}  {best.params}")


if __name__ == "__main__":
    main()
