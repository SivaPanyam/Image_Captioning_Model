"""
PyTorch Dataset and DataLoader for Flickr8k Image Captioning.
Supports training-mode batch collation and evaluation-mode multi-reference evaluation.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from configs.config import (
    IMAGES_DIR,
    CAPTIONS_FILE,
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
    BATCH_SIZE,
    NUM_WORKERS,
    MAX_CAPTION_LENGTH,
    PROJECT_ROOT,
)
from src.preprocessing.caption_loader import load_captions, load_split_captions
from src.preprocessing.caption_cleaner import SimpleTokenizer
from src.preprocessing.vocabulary import Vocabulary, PAD_IDX
from src.preprocessing.sequence_utils import (
    numericalize_caption,
    prepare_teacher_forcing_pair,
    pad_sequence,
)
from src.encoder.feature_extractor import get_image_transforms


class Flickr8kDataset(Dataset):
    """
    Flickr8k Dataset with split isolation and dual-mode functionality.
    
    Modes:
        - "train": Flattens each image into 5 individual (image, caption) pairs.
        - "val" / "test": Groups by image, returning image + list of all reference captions.
    """

    def __init__(
        self,
        split: str = "train",
        vocab: Optional[Vocabulary] = None,
        transform: Optional[Callable] = None,
        images_dir: Path = IMAGES_DIR,
        captions_file: Path = CAPTIONS_FILE,
        max_length: int = MAX_CAPTION_LENGTH,
        feature_cache_path: Optional[Path] = None,
    ):
        self.split = split.lower()
        assert self.split in ["train", "val", "test"], f"Invalid split: {split}"

        self.images_dir = Path(images_dir)
        self.max_length = max_length
        self.tokenizer = SimpleTokenizer()
        self.feature_cache_path = feature_cache_path
        self.cached_features: Optional[Dict[str, torch.Tensor]] = None

        if self.feature_cache_path is not None and Path(self.feature_cache_path).exists():
            self.cached_features = torch.load(self.feature_cache_path, weights_only=False)

        # Resolve split file
        split_map = {
            "train": TRAIN_FILE,
            "val": VAL_FILE,
            "test": TEST_FILE,
        }
        split_file = split_map[self.split]

        # Load isolated split captions
        all_captions = load_captions(captions_file)
        self.captions_dict: Dict[str, List[str]] = load_split_captions(split_file, all_captions)
        self.image_names: List[str] = sorted(list(self.captions_dict.keys()))

        # Vocabulary
        if vocab is None:
            vocab_path = PROJECT_ROOT / "dataset" / "vocab.json"
            if not vocab_path.exists():
                raise FileNotFoundError(f"Vocabulary not found at {vocab_path}. Run build_vocab.py first.")
            self.vocab = Vocabulary.load(vocab_path)
        else:
            self.vocab = vocab

        # Transforms
        if transform is None:
            self.transform = get_image_transforms(is_training=(self.split == "train"))
        else:
            self.transform = transform

        # Build sample index for training (1 image -> 5 samples)
        if self.split == "train":
            self.samples: List[Tuple[str, str]] = []
            for img_name in self.image_names:
                for caption in self.captions_dict[img_name]:
                    self.samples.append((img_name, caption))

    def __len__(self) -> int:
        if self.split == "train":
            return len(self.samples)
        return len(self.image_names)

    def __getitem__(self, idx: int):
        if self.split == "train":
            img_name, caption = self.samples[idx]

            # Use in-memory cached features if available, else load image
            if self.cached_features is not None and img_name in self.cached_features:
                image_tensor = self.cached_features[img_name]
            else:
                img_path = self.images_dir / img_name
                image = Image.open(img_path).convert("RGB")
                image_tensor = self.transform(image)

            # Tokenize & Numericalize
            tokens = self.tokenizer.tokenize(caption)
            full_ids = numericalize_caption(
                tokens, self.vocab, add_special_tokens=True, max_length=self.max_length
            )
            input_ids, target_ids = prepare_teacher_forcing_pair(full_ids)

            return image_tensor, torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)
        else:
            img_name = self.image_names[idx]

            if self.cached_features is not None and img_name in self.cached_features:
                image_tensor = self.cached_features[img_name]
            else:
                img_path = self.images_dir / img_name
                image = Image.open(img_path).convert("RGB")
                image_tensor = self.transform(image)

            # Keep raw reference tokens for BLEU/CIDEr multi-reference scoring
            raw_references = [
                self.tokenizer.tokenize(cap) for cap in self.captions_dict[img_name]
            ]

            return image_tensor, raw_references, img_name


class CaptionCollate:
    """
    Minibatch collate function for dynamic sequence padding.
    Pads input_ids and target_ids dynamically to the longest sequence in the batch.
    """

    def __init__(self, pad_idx: int = PAD_IDX):
        self.pad_idx = pad_idx

    def __call__(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: List of tuples (image_tensor, input_ids, target_ids)
            
        Returns:
            images: Tensor of shape (B, 3, 224, 224)
            inputs_padded: Tensor of shape (B, T_max)
            targets_padded: Tensor of shape (B, T_max)
            lengths: Tensor of shape (B) indicating valid non-padded lengths
        """
        images = [item[0] for item in batch]
        inputs = [item[1] for item in batch]
        targets = [item[2] for item in batch]

        # Stack image tensors: (B, 3, 224, 224)
        images_tensor = torch.stack(images, dim=0)

        # Batch sequence lengths
        lengths = torch.tensor([len(inp) for inp in inputs], dtype=torch.long)
        max_len = int(lengths.max().item())

        # Pad dynamically to max_len
        inputs_padded = torch.full((len(inputs), max_len), self.pad_idx, dtype=torch.long)
        targets_padded = torch.full((len(targets), max_len), self.pad_idx, dtype=torch.long)

        for i, (inp, tgt) in enumerate(zip(inputs, targets)):
            inputs_padded[i, :len(inp)] = inp
            targets_padded[i, :len(tgt)] = tgt

        return images_tensor, inputs_padded, targets_padded, lengths


def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    vocab: Optional[Vocabulary] = None,
    feature_cache_path: Optional[Path] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Factory creating Train, Validation, and Test DataLoaders.
    """
    collate_fn = CaptionCollate(pad_idx=PAD_IDX)

    train_dataset = Flickr8kDataset(split="train", vocab=vocab, feature_cache_path=feature_cache_path)
    val_dataset = Flickr8kDataset(split="val", vocab=vocab, feature_cache_path=feature_cache_path)
    test_dataset = Flickr8kDataset(split="test", vocab=vocab, feature_cache_path=feature_cache_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Val and Test don't pad captions (evaluation handles references individually)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
