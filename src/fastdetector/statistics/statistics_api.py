"""Backward-compatibility re-export shim.

Functions live in :mod:`logprobs_api`, :mod:`embeddings_api`, and
:mod:`softngram_api`; new code should import from those modules directly.
"""

from fastdetector.statistics.logprobs_api import fetch_logprobs_all  # noqa: F401
from fastdetector.statistics.embeddings_api import (  # noqa: F401
    batch_gen_embeddings,
    generate_token_embeddings_pairs,
    batch_cross_encoder,
)
from fastdetector.statistics.softngram_api import batch_soft_ngram_scores  # noqa: F401

__all__ = [
    "fetch_logprobs_all",
    "batch_gen_embeddings",
    "generate_token_embeddings_pairs",
    "batch_cross_encoder",
    "batch_soft_ngram_scores",
]
