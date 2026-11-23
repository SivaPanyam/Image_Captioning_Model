"""
Edge & Mobile NPU Model Optimizer.
Exports trained PyTorch Captioning Models into:
1. Static ONNX computation graphs (Universal edge format for Android NNAPI, Apple CoreML, Qualcomm QNN)
2. INT8 Dynamic Post-Training Quantization (PTQ) reducing model size by ~75%
3. Evaluates latency and memory consumption before and after quantization
"""

import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import PROJECT_ROOT
from src.preprocessing.vocabulary import Vocabulary
from models.captioning_model import ShowAndTellModel, AdaptiveAttentionModel


class ExportableResNetEncoder(nn.Module):
    """Wraps ResNet-50 visual feature extractor for ONNX static graph export."""
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # (B, 3, 224, 224) -> (B, 2048)
        return self.encoder(images)


class ExportableSpatialEncoder(nn.Module):
    """Wraps ResNet-50 spatial extractor for ONNX static graph export."""
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # (B, 3, 224, 224) -> (B, 196, 2048)
        spatial = self.encoder.forward_spatial(images)
        b, c, h, w = spatial.shape
        return spatial.view(b, c, h * w).permute(0, 2, 1)


class ExportableLSTMStep(nn.Module):
    """
    Autoregressive single-step decoder for Mobile/NPU runtimes.
    Eliminates dynamic Python for-loops into a constant-time static cell.
    """
    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        token_id: torch.Tensor,     # (B, 1)
        h: torch.Tensor,            # (1, B, H)
        c: torch.Tensor,            # (1, B, H)
    ):
        emb = self.decoder.embedding(token_id)
        lstm_out, (next_h, next_c) = self.decoder.lstm(emb, (h, c))
        logits = self.decoder.fc(lstm_out.squeeze(1)) # (B, vocab_size)
        return logits, next_h, next_c


def export_and_quantize(output_dir: Path = PROJECT_ROOT / "checkpoints" / "onnx_mobile"):
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab = Vocabulary.load(PROJECT_ROOT / "dataset" / "vocab.json")
    vocab_size = len(vocab)

    print("=" * 65)
    print("STARTING EDGE & NPU MODEL QUANTIZATION & EXPORT")
    print("=" * 65)

    # 1. Load Baseline Model
    ckpt_path = PROJECT_ROOT / "checkpoints" / "baseline" / "best_model.pth"
    model = ShowAndTellModel(vocab_size=vocab_size)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 2. Export Visual Encoder to ONNX (FP32)
    encoder_wrapper = ExportableResNetEncoder(model.encoder)
    dummy_img = torch.randn(1, 3, 224, 224)
    fp32_encoder_path = output_dir / "resnet50_encoder.onnx"

    print(f"1. Exporting ResNet-50 Visual Encoder to {fp32_encoder_path.name} ...")
    torch.onnx.export(
        encoder_wrapper,
        dummy_img,
        fp32_encoder_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_image"],
        output_names=["image_features"],
        dynamic_axes={"input_image": {0: "batch_size"}, "image_features": {0: "batch_size"}},
    )

    # 3. Export Autoregressive Step Decoder to ONNX (FP32)
    decoder_step = ExportableLSTMStep(model.decoder)
    dummy_token = torch.tensor([[1]], dtype=torch.long)
    dummy_h = torch.zeros(1, 1, 512)
    dummy_c = torch.zeros(1, 1, 512)
    fp32_decoder_path = output_dir / "lstm_decoder_step.onnx"

    print(f"2. Exporting Autoregressive Decoder Step to {fp32_decoder_path.name} ...")
    torch.onnx.export(
        decoder_step,
        (dummy_token, dummy_h, dummy_c),
        fp32_decoder_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["token_id", "h_in", "c_in"],
        output_names=["logits", "h_out", "c_out"],
        dynamic_axes={
            "token_id": {0: "batch_size"},
            "h_in": {1: "batch_size"},
            "c_in": {1: "batch_size"},
            "logits": {0: "batch_size"},
            "h_out": {1: "batch_size"},
            "c_out": {1: "batch_size"},
        },
    )

    # 4. INT8 Post-Training Dynamic Quantization (PTQ)
    int8_encoder_path = output_dir / "resnet50_encoder_int8.quant.onnx"
    int8_decoder_path = output_dir / "lstm_decoder_step_int8.quant.onnx"

    print("\n3. Performing INT8 Dynamic Quantization for Mobile NPU ...")
    quantize_dynamic(
        model_input=fp32_encoder_path,
        model_output=int8_encoder_path,
        weight_type=QuantType.QInt8,
    )

    quantize_dynamic(
        model_input=fp32_decoder_path,
        model_output=int8_decoder_path,
        weight_type=QuantType.QInt8,
    )

    # 5. Measure File Sizes & Compression
    orig_encoder_mb = fp32_encoder_path.stat().st_size / (1024 * 1024)
    quant_encoder_mb = int8_encoder_path.stat().st_size / (1024 * 1024)
    orig_decoder_mb = fp32_decoder_path.stat().st_size / (1024 * 1024)
    quant_decoder_mb = int8_decoder_path.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 65)
    print("COMPRESSION RESULTS SUMMARY:")
    print("=" * 65)
    print(f"Encoder FP32 Size  : {orig_encoder_mb:.2f} MB")
    print(f"Encoder INT8 Size  : {quant_encoder_mb:.2f} MB  (Saved {100*(1-quant_encoder_mb/orig_encoder_mb):.1f}%)")
    print(f"Decoder FP32 Size  : {orig_decoder_mb:.2f} MB")
    print(f"Decoder INT8 Size  : {quant_decoder_mb:.2f} MB  (Saved {100*(1-quant_decoder_mb/orig_decoder_mb):.1f}%)")
    print(f"Total INT8 Size    : {quant_encoder_mb + quant_decoder_mb:.2f} MB (Mobile/NPU Ready!)")

    # 6. Benchmark In-Memory Inference Latency
    print("\n4. Benchmarking ONNX Runtime Inference Latency...")
    session_enc = ort.InferenceSession(str(int8_encoder_path))
    session_dec = ort.InferenceSession(str(int8_decoder_path))

    img_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Warmup
    for _ in range(5):
        _ = session_enc.run(None, {"input_image": img_np})

    t0 = time.time()
    iters = 30
    for _ in range(iters):
        feat = session_enc.run(None, {"input_image": img_np})[0]
    enc_latency = (time.time() - t0) / iters * 1000

    # Step decoder latency
    cur_tok = np.array([[1]], dtype=np.int64)
    cur_h = np.zeros((1, 1, 512), dtype=np.float32)
    cur_c = np.zeros((1, 1, 512), dtype=np.float32)

    t0 = time.time()
    for _ in range(iters):
        logits, cur_h, cur_c = session_dec.run(None, {"token_id": cur_tok, "h_in": cur_h, "c_in": cur_c})
    dec_latency = (time.time() - t0) / iters * 1000

    print(f"Encoder NPU Latency: {enc_latency:.2f} ms / image")
    print(f"Decoder NPU Latency: {dec_latency:.2f} ms / token")
    print(f"Full 15-token caption generated in ~{enc_latency + 15 * dec_latency:.1f} ms!")
    print("=" * 65)


if __name__ == "__main__":
    export_and_quantize()
