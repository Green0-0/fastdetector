"""Exact full-vocabulary LLM scoring via in-process transformers forward passes.

This module replaces the previous approach of launching a vLLM server and
fetching top-N logprobs over the OpenAI completions API. Instead, models are
loaded in-process with ``transformers`` and each text is scored with a single
batched forward pass. The full next-token distribution (all V logits) is
materialized only transiently, in small position chunks, and immediately
reduced to a handful of per-position scalars before being freed. As a result:

- All metrics are **exact** (no top-N tail-mass approximations).
- Peak memory is bounded by ``head_chunk_size * vocab_size`` floats per model,
  independent of corpus size; full logprobs never touch RAM in bulk or disk.

Throughput notes: attention runs through FlashAttention 2 when available
(falling back to SDPA), texts are length-sorted into buckets to minimize
padding waste, and the LM head + log-softmax + reduction is applied in
position chunks to avoid materializing a ``[seq_len, vocab]`` logits tensor.

Exactness is verified for standard CausalLM heads (hidden -> lm_head ->
log_softmax). Architectures that post-process logits outside the head (e.g.
final logit softcapping) are not supported.
"""

import gc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Tolerance (in nats) for outlier comparisons. Log-space is monotonic in
# probability space, so `token_lp < threshold_lp - LOG_TOLERANCE` collapses
# only true numerical ties. Matches the historical behaviour of the API-based
# pathway.
LOG_TOLERANCE = 1e-5

PROGRESS_PRINT_INTERVAL = 500


@dataclass
class ScorerSettings:
    """Settings controlling model loading, batching, and metric thresholds.

    Attributes:
        topp_threshold: Probability mass threshold for top-p outlier flags.
        topk_threshold: Rank threshold for top-k outlier flags.
        max_model_len: Maximum tokens per text; longer texts are truncated.
        max_batch_tokens: Cap on padded tokens (batch_size * max_len) per
            forward pass.
        head_chunk_size: Number of positions per LM-head/log-softmax chunk.
        dtype: Model dtype ("bfloat16", "float16", or "float32"). Logits are
            always reduced in float32.
        attn_implementation: Attention backend. None tries
            "flash_attention_2" and falls back to "sdpa".
        device: Device to load models on.
        compute_cross_entropy: Whether to compute the per-position
            observer->performer cross-entropy (requires two models; used by
            the Binoculars score).
    """

    topp_threshold: float = 0.95
    topk_threshold: int = 50
    max_model_len: int = 16000
    max_batch_tokens: int = 16384
    head_chunk_size: int = 512
    dtype: str = "bfloat16"
    attn_implementation: Optional[str] = None
    device: str = "cuda"
    compute_cross_entropy: bool = False


@dataclass
class TextScores:
    """Per-position sufficient statistics for one text.

    All arrays share length ``m`` = (token count - 1), one entry per next-token
    prediction. ``m`` is 0 for empty or single-token texts. ``per_model``
    fields are lists with one array per scored model, in checkpoint order.

    Attributes:
        token_lps: Per model, log p of each actual next token.
        entropies: Per model, exact next-token distribution entropy H.
            Note E[log p] = -H, used by FastDetectGPT.
        e_lp2: Per model, exact second moment E[(log p)^2].
        topp_outlier: Per model, True where the actual token fell outside the
            top-p nucleus.
        topk_outlier: Per model, True where the actual token fell outside the
            top-k ranks.
        cross_entropies: H(p_observer, log p_performer) per position, when
            two models are scored with compute_cross_entropy. Model 0 is the
            observer and model 1 the performer, matching the Binoculars
            convention.
    """

    token_lps: list[np.ndarray] = field(default_factory=list)
    entropies: list[np.ndarray] = field(default_factory=list)
    e_lp2: list[np.ndarray] = field(default_factory=list)
    topp_outlier: list[np.ndarray] = field(default_factory=list)
    topk_outlier: list[np.ndarray] = field(default_factory=list)
    cross_entropies: Optional[np.ndarray] = None


def _empty_scores(num_models: int, with_cross_entropy: bool) -> TextScores:
    """Build a TextScores with zero positions (empty/whitespace/1-token text).

    Args:
        num_models: Number of models being scored.
        with_cross_entropy: Whether a cross-entropy array should be present.

    Returns:
        TextScores whose arrays all have length 0.
    """
    empty_f = np.zeros(0, dtype=np.float32)
    empty_b = np.zeros(0, dtype=bool)
    return TextScores(
        token_lps=[empty_f.copy() for _ in range(num_models)],
        entropies=[empty_f.copy() for _ in range(num_models)],
        e_lp2=[empty_f.copy() for _ in range(num_models)],
        topp_outlier=[empty_b.copy() for _ in range(num_models)],
        topk_outlier=[empty_b.copy() for _ in range(num_models)],
        cross_entropies=empty_f.copy() if with_cross_entropy else None,
    )


def _resolve_dtype(dtype: str) -> torch.dtype:
    """Map a dtype string to a torch dtype.

    Args:
        dtype: One of "bfloat16", "float16", "float32".

    Returns:
        The corresponding torch dtype.

    Raises:
        ValueError: if the dtype string is not recognized.
    """
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype {dtype!r}; expected one of {sorted(mapping)}")
    return mapping[dtype]


def _load_model(model_name: str, settings: ScorerSettings) -> torch.nn.Module:
    """Load a CausalLM in eval mode with the best available attention backend.

    Args:
        model_name: HuggingFace model ID or local path.
        settings: Scorer settings (dtype, device, attention backend).

    Returns:
        The loaded model.
    """
    torch_dtype = _resolve_dtype(settings.dtype)
    attn_candidates = (
        [settings.attn_implementation]
        if settings.attn_implementation is not None
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
            print(f"Loaded {model_name} with attn_implementation={attn}.")
            model.to(settings.device)
            model.eval()
            return model
        except (ImportError, ValueError) as e:
            print(f"attn_implementation={attn} unavailable for {model_name}: {e}")
            last_error = e
    raise RuntimeError(f"Could not load {model_name} with any attention backend.") from last_error


class ExactScorer:
    """Scores texts against one or two CausalLMs with exact full-vocab metrics.

    When two models are provided they are co-resident so that per-position
    cross-model statistics (Binoculars cross-entropy) can be computed without
    persisting either model's distributions. All models are scored against the
    tokenization of the *first* model's tokenizer; when two models are used
    their vocab sizes must match.
    """

    def __init__(
        self,
        models: list[torch.nn.Module],
        tokenizer,
        settings: ScorerSettings,
    ):
        """Initialize the scorer.

        Args:
            models: One or two loaded CausalLM models.
            tokenizer: Tokenizer shared by all models (may be None if only
                the pre-tokenized ``score_token_lists`` entry point is used).
            settings: Scorer settings.

        Raises:
            ValueError: if the model count is invalid, cross-entropy is
                requested without exactly two models, or vocab sizes mismatch.
        """
        if len(models) not in (1, 2):
            raise ValueError(f"ExactScorer supports 1 or 2 models, got {len(models)}.")
        if settings.compute_cross_entropy and len(models) != 2:
            raise ValueError("compute_cross_entropy requires exactly 2 models.")
        if len(models) == 2:
            vocabs = [m.get_output_embeddings().weight.shape[0] for m in models]
            if vocabs[0] != vocabs[1]:
                raise ValueError(
                    f"Vocab size mismatch between co-resident models: {vocabs}. "
                    f"Cross-model scoring requires a shared tokenizer/vocab."
                )
        self.models = models
        self.tokenizer = tokenizer
        self.settings = settings

    def score_texts(self, texts: list[str], progress_label: str = "") -> list[TextScores]:
        """Tokenize and score a list of texts.

        Args:
            texts: Input strings. Empty/whitespace strings yield empty scores.
            progress_label: Optional label for progress prints.

        Returns:
            One TextScores per input text, in input order.

        Raises:
            RuntimeError: if the scorer was constructed without a tokenizer.
        """
        if self.tokenizer is None:
            raise RuntimeError("score_texts requires a tokenizer; use score_token_lists instead.")
        token_lists: list[list[int]] = []
        for text in texts:
            if not text or not text.strip():
                token_lists.append([])
                continue
            token_lists.append(
                self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.settings.max_model_len,
                )["input_ids"]
            )
        return self.score_token_lists(token_lists, progress_label=progress_label)

    def score_token_lists(
        self, token_lists: list[list[int]], progress_label: str = ""
    ) -> list[TextScores]:
        """Score pre-tokenized texts.

        Texts are length-sorted into padded batches capped at
        ``max_batch_tokens`` padded tokens to minimize padding waste; results
        are returned in the original input order.

        Args:
            token_lists: One token-ID list per text. Lists with fewer than 2
                tokens yield empty scores (no next-token prediction exists).
            progress_label: Optional label for progress prints.

        Returns:
            One TextScores per input, in input order.
        """
        num_models = len(self.models)
        with_ce = self.settings.compute_cross_entropy
        results: list[Optional[TextScores]] = [None] * len(token_lists)

        scoreable = []
        for i, ids in enumerate(token_lists):
            if len(ids) < 2:
                results[i] = _empty_scores(num_models, with_ce)
            else:
                scoreable.append(i)
        scoreable.sort(key=lambda i: len(token_lists[i]), reverse=True)

        completed = 0
        total = len(scoreable)
        for batch_indices in self._plan_batches(scoreable, token_lists):
            batch_scores = self._score_batch([token_lists[i] for i in batch_indices])
            for i, scores in zip(batch_indices, batch_scores):
                results[i] = scores
            completed += len(batch_indices)
            if total and (
                completed % PROGRESS_PRINT_INTERVAL < len(batch_indices) or completed == total
            ):
                label = f" [{progress_label}]" if progress_label else ""
                print(f"  Progress{label}: {completed}/{total} texts scored", flush=True)

        return results  # type: ignore[return-value]

    def _plan_batches(
        self, sorted_indices: list[int], token_lists: list[list[int]]
    ) -> Iterator[list[int]]:
        """Greedily group length-sorted texts into padded-token-capped batches.

        Because indices arrive sorted by length (descending), the padded cost
        of a batch is ``len(first_text) * batch_size``.

        Args:
            sorted_indices: Text indices sorted by token count, descending.
            token_lists: All token-ID lists (indexed by the above).

        Yields:
            Lists of text indices forming one batch each.
        """
        batch: list[int] = []
        batch_max_len = 0
        for i in sorted_indices:
            n = len(token_lists[i])
            new_max = max(batch_max_len, n)
            if batch and new_max * (len(batch) + 1) > self.settings.max_batch_tokens:
                yield batch
                batch = []
                new_max = n
            batch.append(i)
            batch_max_len = new_max
        if batch:
            yield batch

    @torch.inference_mode()
    def _score_batch(self, token_lists: list[list[int]]) -> list[TextScores]:
        """Run one padded forward pass per model and reduce to TextScores.

        Args:
            token_lists: Token-ID lists for this batch (each length >= 2).

        Returns:
            One TextScores per input, in batch order.
        """
        settings = self.settings
        device = settings.device
        lengths = [len(ids) for ids in token_lists]
        max_len = max(lengths)
        batch_size = len(token_lists)

        input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        for row, ids in enumerate(token_lists):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, : len(ids)] = 1
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Flat indices of every predicting position: position j of row r
        # predicts token j+1, valid for j in [0, len_r - 2].
        rows, cols, targets = [], [], []
        for row, ids in enumerate(token_lists):
            m = len(ids) - 1
            rows.extend([row] * m)
            cols.extend(range(m))
            targets.extend(ids[1:])
        rows_t = torch.tensor(rows, dtype=torch.long, device=device)
        cols_t = torch.tensor(cols, dtype=torch.long, device=device)
        targets_t = torch.tensor(targets, dtype=torch.long, device=device)
        num_positions = rows_t.shape[0]

        # Backbone forward per model (padding is right-side, so causal
        # attention never attends to pad tokens at valid positions).
        flat_hiddens = []
        for model in self.models:
            hidden = model.get_decoder()(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            flat_hiddens.append(hidden[rows_t, cols_t])
            del hidden

        per_model_chunks: list[dict[str, list[torch.Tensor]]] = [
            {"token_lps": [], "entropies": [], "e_lp2": [], "topp": [], "topk": []}
            for _ in self.models
        ]
        ce_chunks: list[torch.Tensor] = []

        for start in range(0, num_positions, settings.head_chunk_size):
            end = min(start + settings.head_chunk_size, num_positions)
            targets_chunk = targets_t[start:end]
            chunk_p_obs: Optional[torch.Tensor] = None
            chunk_logp_perf: Optional[torch.Tensor] = None

            for model_idx, model in enumerate(self.models):
                logits = self.models[model_idx].get_output_embeddings()(
                    flat_hiddens[model_idx][start:end]
                ).float()
                logp = F.log_softmax(logits, dim=-1)
                del logits
                p = logp.exp()

                out = per_model_chunks[model_idx]
                token_lp = logp.gather(1, targets_chunk[:, None]).squeeze(1)
                out["token_lps"].append(token_lp)
                out["entropies"].append(-(p * logp).sum(dim=-1))
                out["e_lp2"].append((p * logp.square()).sum(dim=-1))

                kth_lp = torch.topk(logp, settings.topk_threshold, dim=-1).values[:, -1]
                out["topk"].append(token_lp < kth_lp - LOG_TOLERANCE)

                sorted_lp, _ = torch.sort(logp, descending=True, dim=-1)
                cum = sorted_lp.exp().cumsum(dim=-1)
                thresholds = torch.full(
                    (end - start, 1), settings.topp_threshold, device=cum.device
                )
                # First index where cumulative mass reaches the threshold;
                # clamp covers the (float-rounding) case where it never does,
                # falling back to the smallest logprob.
                idx = torch.searchsorted(cum, thresholds).clamp(max=logp.shape[-1] - 1)
                threshold_lp = sorted_lp.gather(1, idx).squeeze(1)
                out["topp"].append(token_lp < threshold_lp - LOG_TOLERANCE)
                del sorted_lp, cum

                if settings.compute_cross_entropy and model_idx == 0:
                    chunk_p_obs = p
                elif settings.compute_cross_entropy and model_idx == 1:
                    chunk_logp_perf = logp
                del logp, p

            if settings.compute_cross_entropy:
                # H(p_observer, log p_performer) = -sum_v p_obs(v) log p_perf(v)
                ce_chunks.append(-(chunk_p_obs * chunk_logp_perf).sum(dim=-1))
                del chunk_p_obs, chunk_logp_perf

        del flat_hiddens

        def _to_numpy(chunks: list[torch.Tensor], dtype) -> np.ndarray:
            """Concatenate reduction chunks and move them to a numpy array.

            Args:
                chunks: Per-chunk 1D tensors.
                dtype: Target numpy dtype.

            Returns:
                Concatenated numpy array.
            """
            return torch.cat(chunks).cpu().numpy().astype(dtype)

        token_lps = [_to_numpy(c["token_lps"], np.float32) for c in per_model_chunks]
        entropies = [_to_numpy(c["entropies"], np.float32) for c in per_model_chunks]
        e_lp2 = [_to_numpy(c["e_lp2"], np.float32) for c in per_model_chunks]
        topp = [_to_numpy(c["topp"], bool) for c in per_model_chunks]
        topk = [_to_numpy(c["topk"], bool) for c in per_model_chunks]
        ce = _to_numpy(ce_chunks, np.float32) if settings.compute_cross_entropy else None

        results = []
        offset = 0
        for ids in token_lists:
            m = len(ids) - 1
            sl = slice(offset, offset + m)
            results.append(
                TextScores(
                    token_lps=[arr[sl] for arr in token_lps],
                    entropies=[arr[sl] for arr in entropies],
                    e_lp2=[arr[sl] for arr in e_lp2],
                    topp_outlier=[arr[sl] for arr in topp],
                    topk_outlier=[arr[sl] for arr in topk],
                    cross_entropies=ce[sl] if ce is not None else None,
                )
            )
            offset += m
        return results


@contextmanager
def exact_scorer_context(checkpoints: list[str], settings: ScorerSettings):
    """Load models for scoring and free their GPU memory on exit.

    Args:
        checkpoints: One or two HuggingFace model IDs / local paths. The
            first checkpoint's tokenizer is used for all models.
        settings: Scorer settings.

    Yields:
        A ready ExactScorer.
    """
    models: list[torch.nn.Module] = []
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoints[0])
        for checkpoint in checkpoints:
            models.append(_load_model(checkpoint, settings))
        yield ExactScorer(models, tokenizer, settings)
    finally:
        print("Freeing scoring model(s)...")
        for model in models:
            del model
        models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Scoring model(s) freed.")
