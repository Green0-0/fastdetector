import json
import os

import optuna
from accelerate import PartialState
from optuna.storages import JournalStorage, JournalFileStorage, JournalFileOpenLock

from fastdetector.modeling.training_generic import _bcast, _log, train


def get_or_create_study(study_name, journal_path, min_resource, max_resource, reduction_factor, direction="maximize"):
    """Open the study behind a journal file, creating it if it is not there.

    Args:
        study_name: Optuna study name, the key the journal is resumed under.
        journal_path: Path to the journal file; its directory is created.
        min_resource: Steps a trial runs before Hyperband may prune it.
        max_resource: Steps a trial runs at most, or "auto" to infer it from
            the first completed trial.
        reduction_factor: Fraction of trials Hyperband keeps at each rung.
        direction: Direction to optimize the objective in.

    Returns:
        The optuna study.
    """
    directory = os.path.dirname(journal_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    lock = JournalFileOpenLock(f"{journal_path}.lock")
    storage = JournalStorage(JournalFileStorage(journal_path, lock_obj=lock))
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=10,
        multivariate=True,
        constant_liar=True,
    )
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=min_resource,
        max_resource=max_resource,
        reduction_factor=reduction_factor,
    )
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
    )


def sweep(trials: int = 40, study_name: str | None = None,
          journal_path: str = "sweeps/journal.log",
          min_resource: int | None = None,
          max_resource: int | str = "auto",
          reduction_factor: int = 3,
          **train_kwargs) -> None:
    """Search LoRA and head hyperparameters with optuna.

    The main process owns the study and drives ``study.optimize``; every other
    rank mirrors it trial for trial, training against parameters broadcast from
    the objective.

    Args:
        trials: Number of trials to run.
        study_name: Optuna study name. Defaults to the model's basename.
        journal_path: Path to the study's journal file.
        min_resource: Steps a trial runs before Hyperband may prune it.
            Defaults to the first step a score is reported at, one eval.
        max_resource: Steps a trial runs at most, or "auto" to infer it from
            the first completed trial. An epoch's step count is not known
            ahead of time, so "auto" is the usable setting here.
        reduction_factor: Fraction of trials Hyperband keeps at each rung.
        **train_kwargs: Forwarded to ``train`` and held fixed across trials,
            except for the fields the search itself sets.
    """
    model = train_kwargs.get("model", "meta-llama/Llama-3.2-3B")

    def run_trial(number=None, params=None, trial=None) -> tuple[float, bool]:
        """Run one trial's training job in lockstep on every rank.

        Args:
            number: Trial number. None off the main process, which takes the
                main process's value from the broadcast.
            params: Suggested hyperparameters, likewise.
            trial: Optuna trial to prune against. None off the main process.

        Returns:
            Tuple of (eval macro F1, whether the trial was pruned).
        """
        number, params = _bcast((number, params))
        _log(f"\n[trial {number}] {json.dumps(params)}")
        score, pruned = train(**{**train_kwargs, **params}, trial=trial)
        _log(f"[trial {number}] macro_f1={score:.4f}{' (pruned)' if pruned else ''}")
        return score, pruned

    def objective(trial) -> float:
        """Suggest one trial's hyperparameters and score them.

        Args:
            trial: Optuna trial to suggest from.

        Returns:
            The trial's eval macro F1.

        Raises:
            optuna.TrialPruned: If the pruner stopped the run early.
        """
        lora_r = trial.suggest_categorical("lora_r", [8, 16, 32, 64])
        score, pruned = run_trial(trial.number, dict(
            lr=trial.suggest_float("lr", 1e-5, 5e-4, log=True), lora_r=lora_r,
            lora_alpha=lora_r * trial.suggest_categorical("alpha_over_r", [1, 2, 4]),
            lora_dropout=trial.suggest_float("lora_dropout", 0.0, 0.15),
            head_norm=trial.suggest_categorical("head_norm", [False, True]),
            head_bottleneck=trial.suggest_categorical("head_bottleneck", [0, 512])), trial)
        if pruned:
            raise optuna.TrialPruned()
        return score

    if not PartialState().is_main_process:
        for _ in range(trials):
            try:
                run_trial()
            except Exception:
                pass  # Mirror optimize's catch, so the ranks stay in step.
        return

    study = get_or_create_study(
        study_name=study_name or model.split("/")[-1], journal_path=journal_path,
        min_resource=min_resource or train_kwargs.get("eval_steps", 200),
        max_resource=max_resource, reduction_factor=reduction_factor)
    study.optimize(objective, n_trials=trials, catch=(Exception,))

    _log("\nbest:")
    for best in study.best_trials[:5]:
        _log(f"  #{best.number}  {best.value:.4f}  {best.params}")
