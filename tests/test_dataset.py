"""
Unit tests for Flickr8kDataset and custom CaptionCollate batch collation.
"""

import torch
from pathlib import Path
from src.datasets.flickr8k_dataset import Flickr8kDataset, CaptionCollate
from src.preprocessing.vocabulary import Vocabulary, PAD_IDX, START_IDX, END_IDX


def test_caption_collate():
    collate = CaptionCollate(pad_idx=PAD_IDX)

    # 3 dummy samples with different sequence lengths
    img1 = torch.randn(3, 224, 224)
    inp1 = torch.tensor([START_IDX, 4, 5], dtype=torch.long)
    tgt1 = torch.tensor([4, 5, END_IDX], dtype=torch.long)

    img2 = torch.randn(3, 224, 224)
    inp2 = torch.tensor([START_IDX, 4, 5, 6, 7], dtype=torch.long)
    tgt2 = torch.tensor([4, 5, 6, 7, END_IDX], dtype=torch.long)

    batch = [(img1, inp1, tgt1), (img2, inp2, tgt2)]
    images, inputs_padded, targets_padded, lengths = collate(batch)

    assert images.shape == (2, 3, 224, 224)
    assert inputs_padded.shape == (2, 5), f"Expected max length 5, got {inputs_padded.shape}"
    assert targets_padded.shape == (2, 5)
    assert lengths.tolist() == [3, 5]

    # Verify zero-padding on first sample
    assert inputs_padded[0, 3:].tolist() == [PAD_IDX, PAD_IDX]
    assert targets_padded[0, 3:].tolist() == [PAD_IDX, PAD_IDX]


def test_flickr8k_dataset_length():
    train_dataset = Flickr8kDataset(split="train")
    val_dataset = Flickr8kDataset(split="val")
    test_dataset = Flickr8kDataset(split="test")

    # Train: 6,000 images * 5 captions = 30,000 samples
    assert len(train_dataset) == 30000, f"Expected 30000 train samples, got {len(train_dataset)}"
    # Val: 1,000 images
    assert len(val_dataset) == 1000, f"Expected 1000 val samples, got {len(val_dataset)}"
    # Test: 1,091 images
    assert len(test_dataset) == 1091, f"Expected 1091 test samples, got {len(test_dataset)}"


def test_dataset_item_types():
    train_dataset = Flickr8kDataset(split="train")
    img, inp, tgt = train_dataset[0]

    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert isinstance(inp, torch.Tensor)
    assert isinstance(tgt, torch.Tensor)
    assert inp.dtype == torch.long
    assert tgt.dtype == torch.long
    assert inp[0].item() == START_IDX
    assert tgt[-1].item() == END_IDX
    assert len(inp) == len(tgt)


if __name__ == "__main__":
    test_caption_collate()
    test_flickr8k_dataset_length()
    test_dataset_item_types()
    print("=" * 60)
    print("ALL 3 DATASET & DATALOADER UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
