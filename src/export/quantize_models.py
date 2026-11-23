"""
Native Edge & Mobile NPU Model Optimizer via PyTorch AO Quantization.
Performs Post-Training Dynamic INT8 Quantization (qint8) across Linear and LSTM layers.
Benefits:
- Eliminates external protobuf / onnx runtime incompatibilities
- Reduces model checkpoint weight size by ~70%
- Speeds up CPU and mobile NPU forward inference latency
- Native format deployable with PyTorch Mobile / TorchScript (.pt / .pth)
"""

import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.ao.quantization import quantize_dynamic

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import PROJECT_ROOT
from src.preprocessing.vocabulary import Vocabulary
from models.captioning_model import ShowAndTellModel, AdaptiveAttentionModel
from src.decoder.vit_transformer_decoder import ViTTransformerCaptioningModel


def quantize_all_models(output_dir: Path = PROJECT_ROOT / "checkpoints" / "quantized_mobile"):
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab = Vocabulary.load(PROJECT_ROOT / "dataset" / "vocab.json")
    vocab_size = len(vocab)

    print("=" * 65)
    print("NATIVE PYTORCH EDGE & MOBILE INT8 QUANTIZATION ENGINE")
    print("=" * 65)

    models_to_quantize = [
        ("Baseline (ShowAndTell)", "baseline", ShowAndTellModel(vocab_size=vocab_size), PROJECT_ROOT / "checkpoints" / "baseline" / "best_model.pth"),
        ("Adaptive Attention", "adaptive", AdaptiveAttentionModel(vocab_size=vocab_size), PROJECT_ROOT / "checkpoints" / "adaptive" / "best_model.pth"),
        ("ViT Transformer", "vit_transformer", ViTTransformerCaptioningModel(vocab_size=vocab_size), PROJECT_ROOT / "checkpoints" / "vit_transformer" / "best_model.pth"),
    ]

    for label, name, model, ckpt_p in models_to_quantize:
        if not ckpt_p.exists():
            continue

        print(f"\nProcessing {label} ...")
        ckpt = torch.load(ckpt_p, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Measure original size
        orig_size_mb = ckpt_p.stat().st_size / (1024 * 1024)

        # Apply INT8 dynamic quantization
        quantized_model = quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.LSTMCell},
            dtype=torch.qint8,
        )

        out_path = output_dir / f"{name}_int8.pth"
        torch.save(
            {
                "model_state_dict": quantized_model.state_dict(),
                "vocab_size": vocab_size,
                "model_type": name,
                "quantized": True,
            },
            out_path,
        )

        quant_size_mb = out_path.stat().st_size / (1024 * 1024)
        savings = (1 - quant_size_mb / orig_size_mb) * 100

        print(f"  - Original FP32 Weight Size : {orig_size_mb:.2f} MB")
        print(f"  - Quantized INT8 Weight Size: {quant_size_mb:.2f} MB")
        print(f"  - Memory Compression Ratio  : {savings:.1f}% reduction")
        print(f"  - Saved to: {out_path.name}")

    print("\n" + "=" * 65)
    print("QUANTIZATION BENCHMARK & LATENCY VERIFICATION:")
    print("=" * 65)

    # Benchmark CPU latency
    adaptive_fp32 = AdaptiveAttentionModel(vocab_size=vocab_size)
    adaptive_fp32.load_state_dict(torch.load(PROJECT_ROOT / "checkpoints" / "adaptive" / "best_model.pth", map_location="cpu")["model_state_dict"])
    adaptive_fp32.eval()

    adaptive_int8 = quantize_dynamic(adaptive_fp32, {nn.Linear, nn.LSTM, nn.LSTMCell}, dtype=torch.qint8)

    dummy_spatial = torch.randn(1, 196, 2048)
    dummy_caption = torch.randint(0, 2000, (1, 15))

    # Benchmark FP32
    t0 = time.time()
    for _ in range(20):
        with torch.no_grad():
            _ = adaptive_fp32.decoder(dummy_spatial, dummy_caption)
    fp32_latency = (time.time() - t0) / 20 * 1000

    # Benchmark INT8
    t0 = time.time()
    for _ in range(20):
        with torch.no_grad():
            _ = adaptive_int8.decoder(dummy_spatial, dummy_caption)
    int8_latency = (time.time() - t0) / 20 * 1000

    print(f"FP32 Decoder Latency on CPU/Edge: {fp32_latency:.2f} ms")
    print(f"INT8 Decoder Latency on CPU/Edge: {int8_latency:.2f} ms")
    print(f"Inference Acceleration          : {fp32_latency / max(int8_latency, 0.001):.2f}x faster!")
    print("=" * 65)


if __name__ == "__main__":
    quantize_all_models()
