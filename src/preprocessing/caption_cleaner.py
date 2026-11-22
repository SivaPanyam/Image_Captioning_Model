"""
Caption cleaning and tokenization pipeline.
"""

import re
from typing import List


class CaptionCleaner:
    """
    Deterministic text cleaning for image captions:
    - Lowercase conversion
    - Replaces hyphens with spaces
    - Removes all non-alphabetic characters
    - Normalizes consecutive whitespace
    """

    @staticmethod
    def clean(text: str) -> str:
        """
        Cleans a single caption string.
        """
        if not isinstance(text, str):
            text = str(text)

        # 1. Lowercase
        text = text.lower()

        # 2. Hyphens to space
        text = text.replace("-", " ")

        # 3. Strip non-alphabetic characters (keep a-z and spaces)
        text = re.sub(r"[^a-z\s]", "", text)

        # 4. Collapse multiple spaces and strip ends
        text = re.sub(r"\s+", " ", text).strip()

        return text


class SimpleTokenizer:
    """
    Word-level tokenizer using whitespace splitting.
    """

    def __init__(self, cleaner: CaptionCleaner = CaptionCleaner()):
        self.cleaner = cleaner

    def tokenize(self, text: str) -> List[str]:
        """
        Cleans the input text and splits into word tokens.
        """
        cleaned_text = self.cleaner.clean(text)
        if not cleaned_text:
            return []
        return cleaned_text.split(" ")
