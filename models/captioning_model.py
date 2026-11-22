"""
Complete Image Captioning Architectures.
Provides end-to-end integration for both Baseline (Show and Tell) and Research (Adaptive Spatial Attention) models.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from configs.config import EMBEDDING_DIM, HIDDEN_DIM, FEATURE_DIM
from src.encoder.resnet_encoder import ResNetEncoder
from src.decoder.lstm_decoder import LSTMDecoder
from src.decoder.attention_decoder import AdaptiveAttentionDecoder
from src.preprocessing.vocabulary import START_IDX, END_IDX


class ShowAndTellModel(nn.Module):
    """
    Baseline End-to-End Image Captioning System (Vinyals et al., 2015).
    Pipeline: Image -> ResNet50 (AvgPool) -> 2048-D Vector -> LSTM Decoder -> Vocabulary Logits
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
        feature_dim: int = FEATURE_DIM,
        fine_tune_encoder: bool = False,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.encoder = ResNetEncoder(fine_tune=fine_tune_encoder)
        self.decoder = LSTMDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            dropout=dropout,
        )

    def forward(
        self, images: torch.Tensor, captions: torch.Tensor
    ) -> torch.Tensor:
        """
        Training forward pass with teacher forcing.
        
        Args:
            images: Tensor of shape (B, 3, 224, 224)
            captions: Tensor of token IDs (B, T)
            
        Returns:
            logits: Tensor of shape (B, T, vocab_size)
        """
        # (B, 3, 224, 224) -> (B, 2048)
        features = self.encoder(images)
        # (B, 2048) + (B, T) -> (B, T, vocab_size)
        logits = self.decoder(features, captions)
        return logits

    def generate(
        self,
        images: torch.Tensor,
        max_length: int = 30,
        start_idx: int = START_IDX,
        end_idx: int = END_IDX,
    ) -> torch.Tensor:
        """
        Autoregressive caption generation for inference.
        """
        self.eval()
        with torch.no_grad():
            features = self.encoder(images)
            generated_ids = self.decoder.generate_caption(
                features, max_length=max_length, start_idx=start_idx, end_idx=end_idx
            )
        return generated_ids


class AdaptiveAttentionModel(nn.Module):
    """
    Research-Grade Image Captioning System with Adaptive Multi-Modal Routing.
    Pipeline: Image -> ResNet50 Spatial Grid (14x14) -> Sentinel Gated Attention -> Vocabulary Logits
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
        feature_dim: int = FEATURE_DIM,
        attention_dim: int = 256,
        fine_tune_encoder: bool = False,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.encoder = ResNetEncoder(encoded_image_size=14, fine_tune=fine_tune_encoder)
        self.decoder = AdaptiveAttentionDecoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            attention_dim=attention_dim,
            dropout=dropout,
        )

    def forward(
        self, images: torch.Tensor, captions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: Tensor of shape (B, 3, 224, 224)
            captions: Tensor of shape (B, T)
            
        Returns:
            logits: (B, T, vocab_size)
            alphas: (B, T, 196) attention weights
        """
        # (B, 3, 224, 224) -> (B, 2048, 14, 14)
        spatial_maps = self.encoder.forward_spatial(images)

        # Reshape to (B, 196, 2048) for sequence attention
        batch_size, channels, h, w = spatial_maps.shape
        spatial_features = spatial_maps.view(batch_size, channels, h * w).permute(0, 2, 1)

        logits, alphas = self.decoder(spatial_features, captions)
        return logits, alphas
