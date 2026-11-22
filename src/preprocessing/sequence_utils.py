"""
Sequence utility functions for encoding, padding, and teacher-forcing alignment.
"""

from typing import List, Tuple
import torch

from configs.config import MAX_CAPTION_LENGTH
from src.preprocessing.vocabulary import (
    Vocabulary,
    START_IDX,
    END_IDX,
    PAD_IDX,
)


def numericalize_caption(
    tokens: List[str],
    vocab: Vocabulary,
    add_special_tokens: bool = True,
    max_length: int = MAX_CAPTION_LENGTH,
) -> List[int]:
    """
    Converts a tokenized caption into integer indices.
    
    If add_special_tokens is True:
        Returns: [<START>, w1, w2, ..., wn, <END>]
    Truncates if sequence exceeds max_length.
    """
    indices = [vocab.get_index(token) for token in tokens]

    if add_special_tokens:
        # Reserve 2 slots for <START> and <END>
        max_content_length = max_length - 2
        indices = indices[:max_content_length]
        indices = [START_IDX] + indices + [END_IDX]
    else:
        indices = indices[:max_length]

    return indices


def prepare_teacher_forcing_pair(
    caption_ids: List[int],
) -> Tuple[List[int], List[int]]:
    """
    Splits an encoded sequence [<START>, w1, ..., wn, <END>] into:
      input_ids:  [<START>, w1, ..., wn]
      target_ids: [w1, ..., wn, <END>]
    
    Mathematical Purpose:
        At each time step t, the model receives input_ids[t] and is supervised
        to predict target_ids[t].
    """
    if len(caption_ids) < 2:
        raise ValueError(
            f"Caption IDs sequence must contain at least 2 tokens (<START> and <END>), got {caption_ids}"
        )

    input_ids = caption_ids[:-1]
    target_ids = caption_ids[1:]
    return input_ids, target_ids


def pad_sequence(
    sequence: List[int],
    max_length: int,
    pad_idx: int = PAD_IDX,
) -> List[int]:
    """
    Pads or truncates a sequence to a fixed max_length.
    """
    if len(sequence) >= max_length:
        return sequence[:max_length]
    padding = [pad_idx] * (max_length - len(sequence))
    return sequence + padding
