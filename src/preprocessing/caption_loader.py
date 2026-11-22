"""
Caption loading utility for Flickr8k dataset.
Provides split-aware mapping between image filenames and caption lists.
"""

from pathlib import Path
from typing import Dict, List, Set
from configs.config import CAPTIONS_FILE, TRAIN_FILE, VAL_FILE, TEST_FILE


def load_captions(captions_path: Path = CAPTIONS_FILE) -> Dict[str, List[str]]:
    """
    Loads all captions from the CSV file into a dictionary mapping.
    
    Args:
        captions_path: Path to captions.txt
        
    Returns:
        Dict mapping image filename to a list of raw caption strings.
        e.g. {"1000268201_693b08cb0e.jpg": ["caption 1", "caption 2", ...]}
    """
    if not captions_path.exists():
        raise FileNotFoundError(f"Captions file not found at: {captions_path}")

    captions_dict: Dict[str, List[str]] = {}

    with open(captions_path, "r", encoding="utf-8") as f:
        # Skip header: image,caption
        header = next(f, None)
        if header is None:
            raise ValueError(f"Captions file {captions_path} is empty.")

        for line_num, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue

            parts = line.split(",", maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f"Malformed row at line {line_num} in {captions_path}: {line}"
                )

            image_name, caption = parts[0].strip(), parts[1].strip()

            if image_name not in captions_dict:
                captions_dict[image_name] = []
            captions_dict[image_name].append(caption)

    return captions_dict


def load_split_image_names(split_path: Path) -> Set[str]:
    """
    Reads image filenames listed in a split text file.
    """
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found at: {split_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        images = {line.strip() for line in f if line.strip()}
    return images


def load_split_captions(
    split_path: Path, captions_dict: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    Filters full captions dictionary to retain only images belonging to the specified split.
    """
    split_images = load_split_image_names(split_path)
    filtered = {
        img: captions_dict[img]
        for img in split_images
        if img in captions_dict
    }

    if len(filtered) != len(split_images):
        missing = split_images - set(filtered.keys())
        raise KeyError(
            f"{len(missing)} images from split {split_path.name} not found in captions dictionary."
        )

    return filtered
