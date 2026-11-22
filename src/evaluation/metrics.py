"""
Evaluation Metrics Engine for Image Captioning.
Computes BLEU-1..4, ROUGE-L, and consensus metrics against multiple reference captions.
"""

import math
from collections import Counter
from typing import Dict, List, Tuple


def compute_n_gram_precision(
    candidate: List[str], references: List[List[str]], n: int
) -> Tuple[int, int]:
    """
    Computes clipped n-gram count and candidate n-gram count.
    
    Args:
        candidate: List of tokens in predicted caption.
        references: List of token lists for ground-truth reference captions.
        n: N-gram order (1, 2, 3, 4).
    """
    if len(candidate) < n:
        return 0, 0

    candidate_ngrams = Counter(
        [tuple(candidate[i:i + n]) for i in range(len(candidate) - n + 1)]
    )

    # Max reference frequency for each n-gram
    max_ref_counts = Counter()
    for ref in references:
        if len(ref) < n:
            continue
        ref_ngrams = Counter(
            [tuple(ref[i:i + n]) for i in range(len(ref) - n + 1)]
        )
        for ngram, count in ref_ngrams.items():
            max_ref_counts[ngram] = max(max_ref_counts[ngram], count)

    # Clipped counts
    clipped_count = sum(
        min(count, max_ref_counts[ngram])
        for ngram, count in candidate_ngrams.items()
    )
    total_count = sum(candidate_ngrams.values())

    return clipped_count, total_count


def compute_corpus_bleu(
    hypotheses: Dict[str, List[str]],
    references: Dict[str, List[List[str]]],
    max_order: int = 4,
) -> Dict[str, float]:
    """
    Computes corpus-level BLEU-1, BLEU-2, BLEU-3, BLEU-4 with brevity penalty.
    
    Args:
        hypotheses: Dict mapping image_id -> candidate token list.
        references: Dict mapping image_id -> list of 5 reference token lists.
        
    Returns:
        Dict containing BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores in [0, 100].
    """
    clipped_counts = [0] * max_order
    total_counts = [0] * max_order
    cand_len_sum = 0
    ref_len_sum = 0

    for img_id, cand in hypotheses.items():
        refs = references.get(img_id, [])
        if not refs:
            continue

        cand_len = len(cand)
        cand_len_sum += cand_len

        # Effective reference length (closest reference length to candidate)
        closest_ref_len = min(
            (len(ref) for ref in refs),
            key=lambda ref_len: (abs(ref_len - cand_len), ref_len),
        )
        ref_len_sum += closest_ref_len

        for n in range(1, max_order + 1):
            clipped, total = compute_n_gram_precision(cand, refs, n)
            clipped_counts[n - 1] += clipped
            total_counts[n - 1] += total

    # Brevity Penalty
    if cand_len_sum == 0:
        return {f"BLEU-{n}": 0.0 for n in range(1, max_order + 1)}

    bp = (
        1.0
        if cand_len_sum > ref_len_sum
        else math.exp(1 - (ref_len_sum / max(cand_len_sum, 1)))
    )

    results = {}
    precisions = []

    for n in range(1, max_order + 1):
        if total_counts[n - 1] == 0 or clipped_counts[n - 1] == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped_counts[n - 1] / total_counts[n - 1])

        # Compute cumulative geometric mean
        if min(precisions[:n]) == 0.0:
            score = 0.0
        else:
            log_prec_sum = sum(math.log(p) for p in precisions[:n]) / n
            score = bp * math.exp(log_prec_sum)

        results[f"BLEU-{n}"] = round(score * 100.0, 2)

    return results


def longest_common_subsequence(seq1: List[str], seq2: List[str]) -> int:
    """Computes length of Longest Common Subsequence using dynamic programming."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_corpus_rouge_l(
    hypotheses: Dict[str, List[str]],
    references: Dict[str, List[List[str]]],
    beta: float = 1.2,
) -> float:
    """
    Computes corpus-level ROUGE-L F-measure.
    """
    scores = []

    for img_id, cand in hypotheses.items():
        refs = references.get(img_id, [])
        if not refs or not cand:
            continue

        best_f = 0.0
        for ref in refs:
            if not ref:
                continue
            lcs = longest_common_subsequence(cand, ref)
            prec = lcs / len(cand)
            rec = lcs / len(ref)

            if prec + rec > 0:
                f_measure = ((1 + beta ** 2) * prec * rec) / ((beta ** 2 * prec) + rec)
                best_f = max(best_f, f_measure)

        scores.append(best_f)

    if not scores:
        return 0.0

    return round((sum(scores) / len(scores)) * 100.0, 2)
