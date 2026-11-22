"""
Unit tests for ResNet-50 Encoder and feature extractor pipeline.
"""

import torch
import pytest
from src.encoder.resnet_encoder import ResNetEncoder
from src.encoder.feature_extractor import get_image_transforms


def test_encoder_output_shape():
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224)

    encoder = ResNetEncoder(fine_tune=False)
    encoder.eval()

    with torch.no_grad():
        features = encoder(dummy_images)

    assert features.shape == (batch_size, 2048), f"Expected shape ({batch_size}, 2048), got {features.shape}"


def test_encoder_spatial_output_shape():
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 224, 224)

    encoder = ResNetEncoder(encoded_image_size=14, fine_tune=False)
    encoder.eval()

    with torch.no_grad():
        spatial_features = encoder.forward_spatial(dummy_images)

    assert spatial_features.shape == (batch_size, 2048, 14, 14), f"Expected shape ({batch_size}, 2048, 14, 14), got {spatial_features.shape}"


def test_encoder_parameter_freezing():
    encoder = ResNetEncoder(fine_tune=False)
    for param in encoder.parameters():
        assert not param.requires_grad, "Parameters should be frozen when fine_tune=False"


def test_image_transforms_shape():
    from PIL import Image
    # Create synthetic test PIL image
    img = Image.new("RGB", (400, 300), color=(128, 128, 128))
    transform = get_image_transforms(is_training=False)
    tensor = transform(img)

    assert tensor.shape == (3, 224, 224), f"Transformed image shape expected (3, 224, 224), got {tensor.shape}"
    assert tensor.dtype == torch.float32


if __name__ == "__main__":
    test_encoder_output_shape()
    test_encoder_spatial_output_shape()
    test_encoder_parameter_freezing()
    test_image_transforms_shape()
    print("=" * 60)
    print("ALL 4 ENCODER UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
