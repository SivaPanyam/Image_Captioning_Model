"""
Evaluation package initialization.
"""

from src.evaluation.metrics import compute_corpus_bleu, compute_corpus_rouge_l

__all__ = ["compute_corpus_bleu", "compute_corpus_rouge_l"]
