"""
Unit tests for LSTMDecoder and AdaptiveAttentionDecoder.
"""

import torch
from src.decoder.lstm_decoder import LSTMDecoder
from src.decoder.attention_decoder import AdaptiveAttentionDecoder, BahdanauAttention


def test_lstm_decoder_forward():
    batch_size = 4
    seq_len = 10
    vocab_size = 500

    decoder = LSTMDecoder(
        vocab_size=vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        feature_dim=256,
    )

    dummy_features = torch.randn(batch_size, 256)
    dummy_captions = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits = decoder(dummy_features, dummy_captions)

    assert logits.shape == (batch_size, seq_len, vocab_size), f"Expected shape ({batch_size}, {seq_len}, {vocab_size}), got {logits.shape}"


def test_lstm_decoder_generation():
    batch_size = 2
    vocab_size = 500
    decoder = LSTMDecoder(
        vocab_size=vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        feature_dim=256,
    )

    dummy_features = torch.randn(batch_size, 256)
    gen_ids = decoder.generate_caption(dummy_features, max_length=15)

    assert gen_ids.shape == (batch_size, 15), f"Expected generated shape (2, 15), got {gen_ids.shape}"


def test_adaptive_attention_decoder_forward():
    batch_size = 2
    num_regions = 196  # 14x14
    feature_dim = 256
    seq_len = 8
    vocab_size = 500

    decoder = AdaptiveAttentionDecoder(
        vocab_size=vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        feature_dim=feature_dim,
        attention_dim=64,
    )

    dummy_spatial_features = torch.randn(batch_size, num_regions, feature_dim)
    dummy_captions = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits, alphas = decoder(dummy_spatial_features, dummy_captions)

    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert alphas.shape == (batch_size, seq_len, num_regions)
    # Alphas must sum to 1 over spatial regions for every time step
    alpha_sums = alphas.sum(dim=-1)
    assert torch.allclose(alpha_sums, torch.ones_like(alpha_sums), atol=1e-5), "Attention weights must sum to 1.0"


if __name__ == "__main__":
    test_lstm_decoder_forward()
    test_lstm_decoder_generation()
    test_adaptive_attention_decoder_forward()
    print("=" * 60)
    print("ALL 3 DECODER UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
