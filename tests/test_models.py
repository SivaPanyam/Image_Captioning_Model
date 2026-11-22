"""
Unit tests for end-to-end ShowAndTellModel and AdaptiveAttentionModel.
"""

import torch
from models.captioning_model import ShowAndTellModel, AdaptiveAttentionModel


def test_show_and_tell_model():
    batch_size = 2
    seq_len = 6
    vocab_size = 2563

    model = ShowAndTellModel(
        vocab_size=vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        feature_dim=2048,
    )
    model.eval()

    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_captions = torch.randint(0, vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits = model(dummy_images, dummy_captions)
        gen_ids = model.generate(dummy_images, max_length=10)

    assert logits.shape == (batch_size, seq_len, vocab_size), f"Expected shape ({batch_size}, {seq_len}, {vocab_size}), got {logits.shape}"
    assert gen_ids.shape == (batch_size, 10), f"Expected generated shape (2, 10), got {gen_ids.shape}"


def test_adaptive_attention_model():
    batch_size = 2
    seq_len = 5
    vocab_size = 2563

    model = AdaptiveAttentionModel(
        vocab_size=vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        feature_dim=2048,
        attention_dim=64,
    )
    model.eval()

    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_captions = torch.randint(0, vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits, alphas = model(dummy_images, dummy_captions)

    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert alphas.shape == (batch_size, seq_len, 196)


if __name__ == "__main__":
    test_show_and_tell_model()
    test_adaptive_attention_model()
    print("=" * 60)
    print("ALL END-TO-END MODEL UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
