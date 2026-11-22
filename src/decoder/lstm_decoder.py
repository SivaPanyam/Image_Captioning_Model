"""
Baseline LSTM Decoder for Image Captioning (Show and Tell).
Projects visual features into initial hidden/cell states and decodes text autoregressively.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from configs.config import (
    EMBEDDING_DIM,
    HIDDEN_DIM,
    FEATURE_DIM,
    PROJECT_ROOT,
)
from src.preprocessing.vocabulary import Vocabulary, PAD_IDX, START_IDX, END_IDX


class LSTMDecoder(nn.Module):
    """
    Standard LSTM Decoder with visual state initialization and teacher forcing.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
        feature_dim: int = FEATURE_DIM,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim

        # 1. Word Embedding Layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=PAD_IDX,
        )

        # 2. Linear Projections for Initial States (h0, c0) from Image Features
        self.init_h = nn.Linear(feature_dim, hidden_dim)
        self.init_c = nn.Linear(feature_dim, hidden_dim)

        # 3. Recurrent LSTM Cell
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # 4. Regularization & Vocabulary Projection
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def init_hidden_state(
        self, encoder_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Projects 2048-dim visual feature vectors into initial hidden (h0) and cell (c0) states.
        
        Args:
            encoder_features: Tensor of shape (B, 2048)
            
        Returns:
            h0: Tensor of shape (1, B, hidden_dim)
            c0: Tensor of shape (1, B, hidden_dim)
        """
        # (B, 2048) -> (B, hidden_dim)
        h = torch.tanh(self.init_h(encoder_features))
        c = torch.tanh(self.init_c(encoder_features))

        # PyTorch LSTM expects (num_layers, B, hidden_dim)
        return h.unsqueeze(0), c.unsqueeze(0)

    def forward(
        self,
        encoder_features: torch.Tensor,
        captions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass during training with Teacher Forcing.
        
        Args:
            encoder_features: Visual features (B, 2048)
            captions: Input sequence token IDs (B, T)
            
        Returns:
            logits: Output vocabulary logits (B, T, vocab_size)
        """
        # 1. Initialize hidden & cell states from image features
        h0, c0 = self.init_hidden_state(encoder_features)

        # 2. Embed input tokens: (B, T) -> (B, T, embedding_dim)
        embeddings = self.dropout(self.embedding(captions))

        # 3. Pass through LSTM: (B, T, embedding_dim) -> (B, T, hidden_dim)
        lstm_out, _ = self.lstm(embeddings, (h0, c0))

        # 4. Project hidden states to vocabulary logits: (B, T, hidden_dim) -> (B, T, vocab_size)
        logits = self.fc(self.dropout(lstm_out))

        return logits

    def generate_caption(
        self,
        encoder_features: torch.Tensor,
        max_length: int = 30,
        start_idx: int = START_IDX,
        end_idx: int = END_IDX,
    ) -> torch.Tensor:
        """
        Greedy autoregressive inference on a single image or batch.
        
        Args:
            encoder_features: Visual features (B, 2048)
            max_length: Maximum allowed sequence length
            
        Returns:
            generated_ids: Tensor of token IDs (B, max_length)
        """
        self.eval()
        batch_size = encoder_features.size(0)
        device = encoder_features.device

        h, c = self.init_hidden_state(encoder_features)
        current_token = torch.full((batch_size, 1), start_idx, dtype=torch.long, device=device)

        generated = []

        with torch.no_grad():
            for _ in range(max_length):
                # Embed: (B, 1, embedding_dim)
                embed = self.embedding(current_token)

                # LSTM step: (B, 1, hidden_dim)
                lstm_out, (h, c) = self.lstm(embed, (h, c))

                # Logits: (B, 1, vocab_size)
                logits = self.fc(lstm_out)

                # Greedy next token selection: (B, 1)
                next_token = logits.argmax(dim=-1)
                generated.append(next_token)

                current_token = next_token

        return torch.cat(generated, dim=1)
