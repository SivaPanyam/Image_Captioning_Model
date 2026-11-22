"""
Unit tests for multi-reference evaluation metrics (BLEU-1..4 and ROUGE-L).
"""

from src.evaluation.metrics import compute_corpus_bleu, compute_corpus_rouge_l


def test_perfect_bleu_score():
    references = {
        "img1": [
            ["a", "dog", "runs", "in", "the", "park"],
            ["the", "dog", "is", "running", "outside"],
        ]
    }
    hypotheses = {
        "img1": ["a", "dog", "runs", "in", "the", "park"]
    }

    scores = compute_corpus_bleu(hypotheses, references)
    assert scores["BLEU-1"] == 100.0
    assert scores["BLEU-4"] == 100.0


def test_brevity_penalty_bleu():
    references = {
        "img1": [["a", "dog", "runs", "fast", "in", "the", "park"]]
    }
    hypotheses = {
        "img1": ["a", "dog"]  # Much shorter candidate
    }

    scores = compute_corpus_bleu(hypotheses, references)
    # Precision is 100% for 1-gram, but Brevity Penalty reduces the score
    assert scores["BLEU-1"] < 100.0


def test_rouge_l():
    references = {
        "img1": [["the", "cat", "sat", "on", "the", "mat"]]
    }
    hypotheses = {
        "img1": ["the", "cat", "was", "on", "the", "mat"]
    }

    score = compute_corpus_rouge_l(hypotheses, references)
    assert score > 70.0


if __name__ == "__main__":
    test_perfect_bleu_score()
    test_brevity_penalty_bleu()
    test_rouge_l()
    print("=" * 60)
    print("ALL 3 EVALUATION METRICS UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
