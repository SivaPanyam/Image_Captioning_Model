"""
Unit tests for caption preprocessing, cleaning, tokenization, and vocabulary.
"""

import pytest
from pathlib import Path
from src.preprocessing.caption_cleaner import CaptionCleaner, SimpleTokenizer
from src.preprocessing.vocabulary import (
    Vocabulary,
    PAD_IDX,
    START_IDX,
    END_IDX,
    UNK_IDX,
)
from src.preprocessing.sequence_utils import (
    numericalize_caption,
    prepare_teacher_forcing_pair,
    pad_sequence,
)


def test_caption_cleaning():
    raw = "A Dog, running after a BALL... fast-paced!"
    expected = "a dog running after a ball fast paced"
    cleaned = CaptionCleaner.clean(raw)
    assert cleaned == expected


def test_tokenizer():
    tokenizer = SimpleTokenizer()
    tokens = tokenizer.tokenize("A cat sits on a mat.")
    assert tokens == ["a", "cat", "sits", "on", "a", "mat"]


def test_vocabulary_special_tokens():
    vocab = Vocabulary()
    assert vocab.get_index("<PAD>") == PAD_IDX == 0
    assert vocab.get_index("<START>") == START_IDX == 1
    assert vocab.get_index("<END>") == END_IDX == 2
    assert vocab.get_index("<UNK>") == UNK_IDX == 3
    assert vocab.get_word(0) == "<PAD>"
    assert vocab.get_word(1) == "<START>"
    assert vocab.get_word(2) == "<END>"
    assert vocab.get_word(3) == "<UNK>"


def test_vocabulary_frequency_filtering():
    vocab = Vocabulary()
    captions = [
        ["dog", "runs", "fast"],
        ["dog", "runs"],
        ["dog", "barks"],
        ["cat", "runs"],
        ["dog", "plays"],
    ]
    # "dog" appears 4 times, "runs" appears 3 times, others appear <= 1
    vocab.build_vocabulary(captions, min_freq=3)

    assert "dog" in vocab.word2idx
    assert "runs" in vocab.word2idx
    assert "fast" not in vocab.word2idx
    assert "cat" not in vocab.word2idx
    # Out of vocabulary words map to UNK
    assert vocab.get_index("unknown_token") == UNK_IDX


def test_sequence_numericalization():
    vocab = Vocabulary()
    vocab.build_vocabulary([["a", "dog", "runs"]], min_freq=1)
    tokens = ["a", "dog", "runs"]
    ids = numericalize_caption(tokens, vocab, add_special_tokens=True, max_length=10)

    assert ids[0] == START_IDX
    assert ids[-1] == END_IDX
    assert len(ids) == len(tokens) + 2


def test_teacher_forcing_alignment():
    caption_ids = [START_IDX, 4, 5, 6, END_IDX]
    inp, tgt = prepare_teacher_forcing_pair(caption_ids)

    assert inp == [START_IDX, 4, 5, 6]
    assert tgt == [4, 5, 6, END_IDX]
    assert len(inp) == len(tgt)


def test_pad_sequence():
    seq = [1, 2, 3]
    padded = pad_sequence(seq, max_length=6, pad_idx=PAD_IDX)
    assert padded == [1, 2, 3, 0, 0, 0]


if __name__ == "__main__":
    test_caption_cleaning()
    test_tokenizer()
    test_vocabulary_special_tokens()
    test_vocabulary_frequency_filtering()
    test_sequence_numericalization()
    test_teacher_forcing_alignment()
    test_pad_sequence()
    print("=" * 60)
    print("ALL 7 PREPROCESSING UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

