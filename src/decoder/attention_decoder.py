"""
Adaptive Spatial Attention Decoder with Visual Sentinel Gating (Research Architecture).
Implements fine-grained spatial grounding over (14x14) feature grids with dynamic
routing between visual context and language memory.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import (
    EMBEDDING_DIM,
    HIDDEN_DIM,
    FEATURE_DIM,
)
from src.preprocessing.vocabulary import PAD_IDX, START_IDX, END_IDX


class BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau) Spatial Attention Mechanism.
    Aligns decoder hidden state with 196 spatial image regions.
    """

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_dim: int = HIDDEN_DIM, attention_dim: int = 256):
        super().__init__()
        self.feature_proj = nn.Linear(feature_dim, attention_dim)
        self.hidden_proj = nn.Linear(hidden_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1)

    def forward(
        self, encoder_features: torch.Tensor, decoder_hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_features: Spatial features (B, num_pixels, feature_dim), e.g. (B, 196, 2048)
            decoder_hidden: Current hidden state (B, hidden_dim)
            
        Returns:
            context: Weighted visual representation (B, feature_dim)
            alpha: Attention weights across regions (B, num_pixels)
        """
        # (B, num_pixels, attention_dim)
        proj_features = self.feature_proj(encoder_features)
        
        # (B, 1, attention_dim)
        proj_hidden = self.hidden_proj(decoder_hidden).unsqueeze(1)

        # Energy scores: (B, num_pixels, 1) -> (B, num_pixels)
        scores = self.v(torch.tanh(proj_features + proj_hidden)).squeeze(-1)

        # Attention distribution
        alpha = F.softmax(scores, dim=-1)

        # Context vector: (B, feature_dim)
        context = (encoder_features * alpha.unsqueeze(-1)).sum(dim=1)

        return context, alpha


class AdaptiveAttentionDecoder(nn.Module):
    """
    Dual-Gated Spatial Attention Decoder with Visual Sentinel Gate (AMSR Paper Model).
    Dynamically balances visual grounding against language model memory.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
        feature_dim: int = FEATURE_DIM,
        attention_dim: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(p=dropout)

        self.attention = BahdanauAttention(feature_dim, hidden_dim, attention_dim)

        # Visual Sentinel Gate: Decides if next word needs visual attention or language memory
        self.sentinel_gate = nn.Linear(hidden_dim + embedding_dim, 1)

        # LSTM Cell operating step-by-step
        self.lstm_cell = nn.LSTMCell(embedding_dim + feature_dim, hidden_dim)

        # State initialization from global mean feature
        self.init_h = nn.Linear(feature_dim, hidden_dim)
        self.init_c = nn.Linear(feature_dim, hidden_dim)

        # Output projection head
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def init_hidden_state(self, encoder_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes (h0, c0) by taking the spatial mean of encoder features."""
        # encoder_features: (B, num_pixels, feature_dim) -> mean (B, feature_dim)
        mean_features = encoder_features.mean(dim=1)
        h = torch.tanh(self.init_h(mean_features))
        c = torch.tanh(self.init_c(mean_features))
        return h, c

    def forward(
        self,
        encoder_features: torch.Tensor,
        captions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_features: Spatial features (B, 196, 2048)
            captions: Input sequence token IDs (B, T)
            
        Returns:
            logits: Vocabulary logits (B, T, vocab_size)
            alphas: Attention distributions (B, T, 196) for visual grounding inspection
        """
        batch_size, seq_len = captions.shape
        device = encoder_features.device

        h, c = self.init_hidden_state(encoder_features)
        embeddings = self.dropout(self.embedding(captions))

        logits = torch.zeros(batch_size, seq_len, self.vocab_size, device=device)
        alphas = torch.zeros(batch_size, seq_len, encoder_features.size(1), device=device)

        for t in range(seq_len):
            current_embed = embeddings[:, t, :]  # (B, embedding_dim)

            # 1. Compute spatial visual context and attention map
            context, alpha = self.attention(encoder_features, h)
            alphas[:, t, :] = alpha

            # 2. Compute Visual Sentinel Gate beta in [0, 1]
            gate_input = torch.cat([h, current_embed], dim=-1)
            beta = torch.sigmoid(self.sentinel_gate(gate_input))

            # 3. Gated visual feature feeding
            gated_context = beta * context

            # 4. Advance LSTM step
            lstm_input = torch.cat([current_embed, gated_context], dim=-1)
            h, c = self.lstm_cell(lstm_input, (h, c))

            # 5. Project to vocabulary logits
            logits[:, t, :] = self.fc(self.dropout(h))

        return logits, alphas
