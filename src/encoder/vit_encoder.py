"""
Vision Transformer (ViT-B/16) Visual Encoder.
Extracts 196 contextual patch tokens (D=768) + 1 [CLS] token from raw 224x224 images.
Replaces legacy CNN spatial grids with full self-attention patch representations.
"""

from typing import Tuple
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class VisionTransformerEncoder(nn.Module):
    """
    Vision Transformer Encoder (ViT-B/16).
    
    Transforms input image (B, 3, 224, 224) into:
    - patch_embeddings: (B, 196, 768) - for cross-attention
    - cls_token: (B, 768) - global summary
    """

    def __init__(self, fine_tune: bool = False):
        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT
        vit = vit_b_16(weights=weights)

        # Retain ViT core stages
        self.conv_proj = vit.conv_proj
        self.class_token = vit.class_token
        self.encoder = vit.encoder
        self.embed_dim = 768

        # Freeze/unfreeze
        for param in self.parameters():
            param.requires_grad = fine_tune

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: Tensor of shape (B, 3, 224, 224)
            
        Returns:
            patch_tokens: Tensor of shape (B, 196, 768)
            cls_token: Tensor of shape (B, 768)
        """
        # Patch projection: (B, 3, 224, 224) -> (B, 768, 14, 14) -> (B, 196, 768)
        x = self.conv_proj(images)
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).permute(0, 2, 1)

        # Prepend [CLS] token: (B, 197, 768)
        cls_token = self.class_token.expand(b, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        # Transformer encoder self-attention blocks
        x = self.encoder(x)

        cls_out = x[:, 0]        # (B, 768)
        patch_out = x[:, 1:]     # (B, 196, 768)
        return patch_out, cls_out
