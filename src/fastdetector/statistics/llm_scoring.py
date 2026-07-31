import gc
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from fastdetector.frontend.toml_config import LLMStatConfig


PROGRESS_PRINT_INTERVAL = 500

# Every metric in statistics_llm is a sum over token positions, so per-position
# values never leave the device: each forward pass reduces straight into one
# row of running totals per (text, model), and this is the whole interchange
# format between the scoring here and the metrics there.
SUMS = np.dtype(
    [
        ("n", np.int64),  # scoreable positions, i.e. tokens - 1
        ("lp", np.float64),  # sum of log p(target)
        ("entropy", np.float64),  # sum of H
        ("variance", np.float64),  # sum of max(0, E[(log p)^2] - H^2)
        ("topp", np.float64),  # number of top-p nucleus outliers
        ("topk", np.float64),  # number of top-k rank outliers
        ("ce", np.float64),  # sum of H(p_observer, p_performer)
    ]
)

# The per-model accumulators, in the order _score_batch stacks them. "n" comes
# from the token lengths and "ce" is cross-model, so neither belongs here.
ACCUMULATED = ("lp", "entropy", "variance", "topp", "topk")


def score_columns(
    checkpoints: list[str],
    config: LLMStatConfig,
    columns: Iterable[tuple[str, list[str]]],
) -> dict[str, np.ndarray]:
    """Score dataset columns using loaded model checkpoints across available devices.

    Args:
        checkpoints: List of HuggingFace model IDs or local paths.
        config: Pipeline configuration providing device and scoring settings.
        columns: Iterable of (column name, list of text samples) tuples.

    Returns:
        Dictionary mapping column names to structured SUMS NumPy arrays.

    Raises:
        ValueError: If checkpoint count is unsupported or model vocabs mismatch.
    """
    # Checked before anything downloads, so a misconfiguration fails in
    # seconds rather than after pulling several GB of weights.
    if len(checkpoints) not in (1, 2):
        raise ValueError(f"Scoring supports 1 or 2 checkpoints, got {len(checkpoints)}.")

    if isinstance(config.devices, list):
        devices = config.devices
    elif torch.cuda.is_available():
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        devices = ["cpu"]

    replicas: list[list[torch.nn.Module]] = []
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoints[0])
        print(f"Loading {len(checkpoints)} checkpoint(s) onto {len(devices)} device(s): {devices}")
        for device in devices:
            replicas.append(
                [_load_model(checkpoint, config, device) for checkpoint in checkpoints]
            )

        vocabs = [m.get_output_embeddings().weight.shape[0] for m in replicas[0]]
        if len(set(vocabs)) > 1:
            raise ValueError(
                f"Vocab size mismatch between co-resident models: {vocabs}. "
                f"Cross-model scoring requires a shared tokenizer/vocab."
            )
        if config.topk_threshold > min(vocabs):
            raise ValueError(
                f"topk_threshold ({config.topk_threshold}) exceeds the model "
                f"vocab size ({min(vocabs)})."
            )

        results = {}
        for name, texts in columns:
            print(f"Scoring column '{name}'...")
            # Blank rows tokenize to nothing and keep their all-zero sums row.
            nonempty = [i for i, text in enumerate(texts) if text and text.strip()]
            encoded = tokenizer(
                [texts[i] for i in nonempty],
                truncation=True,
                max_length=config.max_model_len,
            )["input_ids"]
            # Compact int64 arrays keep corpus-scale tokenizations cheap in RAM
            # (~8 bytes/token vs ~28 for lists of Python ints).
            token_arrays = [np.zeros(0, dtype=np.int64)] * len(texts)
            for i, ids in zip(nonempty, encoded):
                token_arrays[i] = np.asarray(ids, dtype=np.int64)

            results[name] = _score_tokens(token_arrays, replicas, devices, config, name)
        return results
    finally:
        print("Freeing scoring model(s)...")
        replicas.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Scoring model(s) freed.")


def _load_model(model_name: str, config: LLMStatConfig, device: str) -> torch.nn.Module:
    """Load causal language model onto specified device.

    Args:
        model_name: HuggingFace model identifier or local path.
        config: Pipeline configuration supplying the scoring settings.
        device: Target execution device.

    Returns:
        Loaded PyTorch model in evaluation mode.

    Raises:
        ValueError: If config.dtype does not name a floating-point torch dtype.
        RuntimeError: If model loading fails with every attention backend.
    """
    torch_dtype = getattr(torch, config.dtype, None)
    if not isinstance(torch_dtype, torch.dtype) or not torch_dtype.is_floating_point:
        raise ValueError(
            f"Unsupported dtype {config.dtype!r}; expected a floating-point torch dtype."
        )

    attn_candidates = (
        [config.attn_implementation]
        if config.attn_implementation is not None
        else ["flash_attention_2", "sdpa"]
    )

    last_error: Optional[Exception] = None
    for attn in attn_candidates:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch_dtype,
                attn_implementation=attn,
            )
            print(f"Loaded {model_name} on {device} with attn_implementation={attn}.")
            model.to(device)
            model.eval()
            return model
        except (ImportError, ValueError) as e:
            print(f"attn_implementation={attn} unavailable for {model_name}: {e}")
            last_error = e
    raise RuntimeError(f"Could not load {model_name} with any attention backend.") from last_error


def _score_tokens(
    token_lists: list,
    replicas: list[list[torch.nn.Module]],
    devices: list[str],
    config: LLMStatConfig,
    progress_label: str = "",
) -> np.ndarray:
    """Batch pre-tokenized texts and score them across every replica.

    Args:
        token_lists: One token ID array per text.
        replicas: Model replicas grouped per target device.
        devices: Target device list, aligned with replicas.
        config: Pipeline configuration supplying the scoring settings.
        progress_label: Optional label for progress output.

    Returns:
        SUMS-dtype array of shape (len(token_lists), models per replica).
    """
    token_arrays = [np.asarray(ids, dtype=np.int64) for ids in token_lists]
    num_models = len(replicas[0])
    # Two co-resident models is exactly the Binoculars setup: nothing else
    # loads a second checkpoint alongside the first, so the cross-model term
    # is derived rather than configured.
    cross_entropy = num_models == 2

    # Texts too short to have a next-token prediction keep their all-zero
    # row, which is exactly the "no positions" sentinel the metrics expect.
    sums = np.zeros((len(token_arrays), num_models), dtype=SUMS)

    scoreable = [i for i, ids in enumerate(token_arrays) if ids.shape[0] >= 2]
    scoreable.sort(key=lambda i: token_arrays[i].shape[0], reverse=True)

    # Greedy packing of the length-sorted texts. A batch is padded out to its
    # longest member, so the cap is on batch_size * max_len, not the token
    # total; sorting first keeps the padding waste inside a batch small.
    batches: list[list[int]] = []
    batch: list[int] = []
    batch_max_len = 0
    for i in scoreable:
        n = token_arrays[i].shape[0]
        new_max = max(batch_max_len, n)
        if batch and new_max * (len(batch) + 1) > config.max_batch_tokens:
            batches.append(batch)
            batch, new_max = [], n
        batch.append(i)
        batch_max_len = new_max
    if batch:
        batches.append(batch)

    completed = 0
    next_report = PROGRESS_PRINT_INTERVAL
    total = len(scoreable)
    label = f" [{progress_label}]" if progress_label else ""

    # Workers check a replica out of the queue per batch, so load stays
    # balanced even when batch runtimes vary. With one replica this
    # degenerates to a single-worker executor.
    idle_replicas: queue.SimpleQueue[int] = queue.SimpleQueue()
    for replica_idx in range(len(replicas)):
        idle_replicas.put(replica_idx)

    def _run_batch(batch_indices: list[int]) -> tuple[list[int], np.ndarray]:
        """Score one batch on whichever replica is idle."""
        replica_idx = idle_replicas.get()
        try:
            batch_sums = _score_batch(
                [token_arrays[i] for i in batch_indices],
                replicas[replica_idx],
                devices[replica_idx],
                config,
                cross_entropy,
            )
        finally:
            idle_replicas.put(replica_idx)
        return batch_indices, batch_sums

    with ThreadPoolExecutor(max_workers=len(replicas)) as executor:
        futures = [executor.submit(_run_batch, batch) for batch in batches]
        for future in as_completed(futures):
            batch_indices, batch_sums = future.result()
            sums[batch_indices] = batch_sums
            completed += len(batch_indices)
            if completed >= next_report or completed == total:
                next_report = completed + PROGRESS_PRINT_INTERVAL
                print(f"  Progress{label}: {completed}/{total} texts scored", flush=True)

    return sums


@torch.inference_mode()
def _score_batch(
    token_arrays: list[np.ndarray],
    models: list[torch.nn.Module],
    device: str,
    config: LLMStatConfig,
    cross_entropy: bool,
) -> np.ndarray:
    """Perform forward pass and metric reduction for a single batch.

    Args:
        token_arrays: Batch token ID arrays.
        models: The co-resident models of one replica.
        device: Device those models sit on.
        config: Pipeline configuration supplying the scoring settings.
        cross_entropy: Whether to accumulate the observer-performer term.

    Returns:
        SUMS-dtype array of shape (len(token_arrays), len(models)).
    """
    lengths = [ids.shape[0] for ids in token_arrays]
    max_len = max(lengths)
    batch_size = len(token_arrays)

    input_ids_np = np.zeros((batch_size, max_len), dtype=np.int64)
    attention_mask_np = np.zeros((batch_size, max_len), dtype=np.int64)
    for row, ids in enumerate(token_arrays):
        input_ids_np[row, : ids.shape[0]] = ids
        attention_mask_np[row, : ids.shape[0]] = 1
    input_ids = torch.from_numpy(input_ids_np).to(device)
    attention_mask = torch.from_numpy(attention_mask_np).to(device)

    # Flat indices of every predicting position: position j of row r
    # predicts token j+1, valid for j in [0, len_r - 2].
    position_counts = [n - 1 for n in lengths]
    rows_np = np.repeat(np.arange(batch_size), position_counts)
    cols_np = np.concatenate([np.arange(m) for m in position_counts])
    targets_np = np.concatenate([ids[1:] for ids in token_arrays])
    rows_t = torch.from_numpy(rows_np).to(device)
    cols_t = torch.from_numpy(cols_np).to(device)
    targets_t = torch.from_numpy(targets_np).to(device)
    num_positions = rows_t.shape[0]

    # Backbone forward per model (padding is right-side, so causal
    # attention never attends to pad tokens at valid positions).
    flat_hiddens = []
    for model in models:
        hidden = model.get_decoder()(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        flat_hiddens.append(hidden[rows_t, cols_t])
        del hidden

    # One running total per (text, accumulator). rows_t already says which
    # text each flat position belongs to, so index_add_ scatters a chunk's
    # positions onto their texts without ever building per-position arrays.
    # float64 preserves the accumulation precision the metrics used to get
    # from summing with NumPy.
    totals = [
        torch.zeros((batch_size, len(ACCUMULATED)), dtype=torch.float64, device=device)
        for _ in models
    ]
    ce_totals = torch.zeros(batch_size, dtype=torch.float64, device=device)

    for start in range(0, num_positions, config.head_chunk_size):
        end = min(start + config.head_chunk_size, num_positions)
        rows_chunk = rows_t[start:end]
        targets_chunk = targets_t[start:end]
        chunk_p_obs: Optional[torch.Tensor] = None
        chunk_logp_perf: Optional[torch.Tensor] = None

        for model_idx, model in enumerate(models):
            logits = model.get_output_embeddings()(
                flat_hiddens[model_idx][start:end]
            ).float()
            logp = F.log_softmax(logits, dim=-1)
            del logits
            p = logp.exp()

            token_lp = logp.gather(1, targets_chunk[:, None]).squeeze(1)
            entropy = -(p * logp).sum(dim=-1)
            e_lp2 = (p * logp.square()).sum(dim=-1)

            # One descending sort serves both outlier metrics: the k-th
            # largest logprob for top-k, and the nucleus boundary for
            # top-p.
            sorted_lp, _ = torch.sort(logp, descending=True, dim=-1)
            kth_lp = sorted_lp[:, config.topk_threshold - 1]

            cum = sorted_lp.exp().cumsum(dim=-1)
            thresholds = torch.full(
                (end - start, 1), config.topp_threshold, device=cum.device
            )
            # First index where cumulative mass reaches the threshold;
            # clamp covers the (float-rounding) case where it never does,
            # falling back to the smallest logprob.
            idx = torch.searchsorted(cum, thresholds).clamp(max=logp.shape[-1] - 1)
            threshold_lp = sorted_lp.gather(1, idx).squeeze(1)
            del sorted_lp, cum

            # E[(log p)^2] lands close to H^2 where the distribution is
            # peaked, so the difference is taken at double precision and
            # clamped: the leftover rounding artefact is not a real
            # negative variance, and fastdetectgpt takes its square root.
            entropy64 = entropy.double()
            variance = (e_lp2.double() - entropy64.square()).clamp(min=0.0)

            totals[model_idx].index_add_(
                0,
                rows_chunk,
                torch.stack(
                    [
                        token_lp.double(),
                        entropy64,
                        variance,
                        # The slack absorbs float32 rounding, so a token that
                        # *is* the nucleus or k-th boundary is not flagged.
                        (token_lp < threshold_lp - 1e-5).double(),
                        (token_lp < kth_lp - 1e-5).double(),
                    ],
                    dim=1,
                ),
            )

            if cross_entropy and model_idx == 0:
                chunk_p_obs = p
            elif cross_entropy and model_idx == 1:
                chunk_logp_perf = logp
            del logp, p

        if cross_entropy:
            # H(p_observer, log p_performer) = -sum_v p_obs(v) log p_perf(v)
            ce_totals.index_add_(
                0, rows_chunk, -(chunk_p_obs * chunk_logp_perf).sum(dim=-1).double()
            )
            del chunk_p_obs, chunk_logp_perf

    del flat_hiddens

    sums = np.zeros((batch_size, len(models)), dtype=SUMS)
    sums["n"] = np.asarray(position_counts, dtype=np.int64)[:, None]
    for model_idx, model_totals in enumerate(totals):
        model_totals_np = model_totals.cpu().numpy()
        for field_idx, name in enumerate(ACCUMULATED):
            sums[name][:, model_idx] = model_totals_np[:, field_idx]
    if cross_entropy:
        # Binoculars divides the performer's total logprob by this, so the
        # cross-model total lives on the performer's row.
        sums["ce"][:, 1] = ce_totals.cpu().numpy()
    return sums
