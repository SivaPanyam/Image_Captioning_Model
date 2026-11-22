"""
Multi-Head Cross-Attention Transformer Decoder with Adaptive Sentinel Gating.
Implements modern Transformer autoregressive decoding for image captioning.
Features:
- Causal Multi-Head Self-Attention for linguistic context (Q, K, V from text tokens)
- Multi-Head Cross-Attention over ViT patch embeddings (Q from text, K, V from image patches)
- Adaptive Visual Sentinel Gate beta_t controlling visual dependency per token
- Rotary/Sinusoidal Positional Embeddings
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.preprocessing.vocabulary import PAD_IDX


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0)) # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return x + self.pe[:, : x.size(1)]


class TransformerDecoderLayerWithSentinel(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        # Adaptive Visual Sentinel Gate: beta_t = sigmoid(W [h_t; c_t])
        self.sentinel_gate = nn.Linear(d_model * 2, d_model)
        
        # Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Causal Self-Attention
        tgt_norm = self.norm1(tgt)
        self_out, _ = self.self_attn(
            tgt_norm, tgt_norm, tgt_norm,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = tgt + self.dropout(self_out)

        # 2. Multi-Head Cross-Attention with Sentinel Gate
        tgt_norm = self.norm2(tgt)
        cross_out, _ = self.cross_attn(query=tgt_norm, key=memory, value=memory)
        
        # Compute visual sentinel gate
        gate_input = torch.cat([tgt_norm, cross_out], dim=-1)
        beta = torch.sigmoid(self.sentinel_gate(gate_input))
        
        # Modulate visual context
        modulated_cross = beta * cross_out
        tgt = tgt + self.dropout(modulated_cross)

        # 3. Position-wise Feedforward
        tgt_norm = self.norm3(tgt)
        ffn_out = self.ffn(tgt_norm)
        tgt = tgt + self.dropout(ffn_out)
        return tgt


class ViTTransformerCaptioningModel(nn.Module):
    """
    Vision Transformer (ViT-B/16) + Multi-Head Cross-Attention Transformer Captioning System.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        vit_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        pad_idx: int = PAD_IDX,
    ):
        super().__init__()
        from src.encoder.vit_encoder import VisionTransformerEncoder
        self.encoder = VisionTransformerEncoder(fine_tune=False)
        self.pad_idx = pad_idx
        self.d_model = d_model

        # Project ViT 768-D patches to decoder d_model (512)
        self.visual_proj = nn.Linear(vit_dim, d_model)

        # Token embedding & Positional Encoding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer Decoder Stack
        self.layers = nn.ModuleList([
            TransformerDecoderLayerWithSentinel(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Output projection head
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz, device=device) * float("-inf"), diagonal=1)
        return mask

    def forward(self, images_or_patches: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images_or_patches: (B, 3, 224, 224) raw images or (B, 196, 768) cached ViT patches
            captions: (B, T) token IDs
        Returns:
            logits: (B, T, vocab_size)
        """
        device = captions.device
        if images_or_patches.dim() == 4:
            patches, _ = self.encoder(images_or_patches) # (B, 196, 768)
        else:
            patches = images_or_patches

        # Project visual patches: (B, 196, d_model)
        memory = self.visual_proj(patches)

        # Target embeddings + positional encoding
        tgt = self.embedding(captions) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)

        seq_len = captions.size(1)
        tgt_mask = self.generate_square_subsequent_mask(seq_len, device)
        padding_mask = (captions == self.pad_idx)

        # Pass through decoder layers
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=padding_mask)

        tgt = self.final_norm(tgt)
        logits = self.fc_out(tgt)
        return logits
