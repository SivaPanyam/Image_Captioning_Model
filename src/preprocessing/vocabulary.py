"""
Vocabulary representation and builder for Image Captioning.
Guarantees training-only vocabulary construction with zero data leakage.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Union

from configs.config import MIN_WORD_FREQUENCY, PROJECT_ROOT


# Reserved Special Tokens and Indices
PAD_TOKEN = "<PAD>"    # Index 0: Padding for batch collation
START_TOKEN = "<START>"  # Index 1: Sequence start signal
END_TOKEN = "<END>"      # Index 2: Sequence termination signal
UNK_TOKEN = "<UNK>"      # Index 3: Out-of-vocabulary representation

SPECIAL_TOKENS = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]

PAD_IDX = 0
START_IDX = 1
END_IDX = 2
UNK_IDX = 3


class Vocabulary:
    """
    Bidirectional vocabulary mapping words to indices and indices to words.
    """

    def __init__(self):
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.frequencies: Dict[str, int] = {}
        self._init_special_tokens()

    def _init_special_tokens(self) -> None:
        """Initializes reserved special tokens with fixed indexes 0..3."""
        for idx, token in enumerate(SPECIAL_TOKENS):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    def __len__(self) -> int:
        return len(self.word2idx)

    def add_word(self, word: str, count: int = 1) -> None:
        """Adds a regular word token if not already present."""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            self.frequencies[word] = count
        else:
            self.frequencies[word] = self.frequencies.get(word, 0) + count

    def get_index(self, word: str) -> int:
        """Returns token index, falling back to UNK_IDX if not found."""
        return self.word2idx.get(word, UNK_IDX)

    def get_word(self, idx: int) -> str:
        """Returns word token for index, falling back to UNK_TOKEN if not found."""
        return self.idx2word.get(idx, UNK_TOKEN)

    def build_vocabulary(
        self,
        tokenized_captions: List[List[str]],
        min_freq: int = MIN_WORD_FREQUENCY,
    ) -> None:
        """
        Builds vocabulary from a list of tokenized captions (training set only).
        
        Args:
            tokenized_captions: List of token lists, e.g. [["a", "dog", "runs"], ...]
            min_freq: Minimum frequency threshold to retain a word.
        """
        counter = Counter()
        for tokens in tokenized_captions:
            counter.update(tokens)

        # Sort for deterministic vocabulary ordering across platforms
        sorted_words = sorted(
            [word for word, count in counter.items() if count >= min_freq],
            key=lambda w: (-counter[w], w),
        )

        for word in sorted_words:
            self.add_word(word, count=counter[word])

    def save(self, filepath: Union[str, Path]) -> None:
        """Serializes the vocabulary to a JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "word2idx": self.word2idx,
            "idx2word": {str(k): v for k, v in self.idx2word.items()},
            "frequencies": self.frequencies,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "Vocabulary":
        """Loads a vocabulary instance from a saved JSON file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = cls()
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        vocab.frequencies = data.get("frequencies", {})
        return vocab
