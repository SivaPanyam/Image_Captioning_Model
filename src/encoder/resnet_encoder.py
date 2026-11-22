"""
ResNet-50 Encoder for Image Captioning.
Extracts 2048-dimensional visual feature vectors from input images.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNetEncoder(nn.Module):
    """
    Pretrained ResNet-50 Visual Feature Extractor.
    Strips the 1000-class classification head and produces a 2048-dim feature vector.
    """

    def __init__(self, encoded_image_size: int = 14, fine_tune: bool = False):
        """
        Args:
            encoded_image_size: Spatial dimension for attention mechanism (default: 14)
            fine_tune: Whether to allow gradient updates to layer2..layer4 blocks.
        """
        super().__init__()
        
        # Load ResNet-50 with latest ImageNet-1K V2 weights
        weights = ResNet50_Weights.DEFAULT
        resnet = resnet50(weights=weights)

        # Remove classification linear layer and global average pool
        # Keeping layers up to layer4 gives spatial feature maps (B, 2048, 7, 7)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Adaptive pool for fixed spatial resolution (used in Attention)
        self.adaptive_pool_spatial = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        # Global average pool for baseline flat feature vector (B, 2048)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fine_tune(fine_tune)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            images: Tensor of shape (B, 3, 224, 224)
            
        Returns:
            features: Tensor of shape (B, 2048) - Global pooled visual representation
        """
        # (B, 3, 224, 224) -> (B, 2048, 7, 7)
        features = self.resnet(images)
        
        # (B, 2048, 7, 7) -> (B, 2048, 1, 1)
        pooled = self.global_pool(features)
        
        # (B, 2048, 1, 1) -> (B, 2048)
        flat_features = pooled.flatten(start_dim=1)
        
        return flat_features

    def forward_spatial(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass preserving spatial grid for Attention models (Stage 13).
        
        Args:
            images: Tensor of shape (B, 3, 224, 224)
            
        Returns:
            spatial_features: Tensor of shape (B, 2048, H, W)
        """
        features = self.resnet(images)
        return self.adaptive_pool_spatial(features)

    def fine_tune(self, fine_tune: bool = False) -> None:
        """
        Configures parameter freezing.
        If fine_tune is False, freezes all parameters.
        If True, only unfreezes deeper blocks (layer2, layer3, layer4).
        """
        for param in self.resnet.parameters():
            param.requires_grad = False

        if fine_tune:
            for child in list(self.resnet.children())[5:]:  # layer2 onwards
                for param in child.parameters():
                    param.requires_grad = True
