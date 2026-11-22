"""
Feature extraction and image preprocessing pipeline.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
from PIL import Image
import torch
from torchvision import transforms

from configs.config import IMAGE_SIZE


def get_image_transforms(is_training: bool = False) -> transforms.Compose:
    """
    Standard torchvision ImageNet transforms.
    
    Training: Resize, random horizontal flip/crop (optional data augmentation).
    Validation/Inference: Resize, center crop, normalize.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_training:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def load_and_preprocess_image(
    image_path: Union[str, Path],
    transform: Optional[transforms.Compose] = None
) -> torch.Tensor:
    """
    Loads an image from disk, converts to RGB, and applies transformations.
    
    Returns:
        Tensor of shape (3, 224, 224)
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    image = Image.open(image_path).convert("RGB")
    if transform is None:
        transform = get_image_transforms(is_training=False)

    return transform(image)
