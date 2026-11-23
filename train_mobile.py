"""
Training Pipeline for Ultra-Compact MobileNetV3 Captioner (< 8 MB INT8 Model).
Trains in under 60 seconds using cached 576-D MobileNetV3 features.
"""

import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.ao.quantization import quantize_dynamic

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import PROJECT_ROOT
from src.preprocessing.vocabulary import Vocabulary, PAD_IDX
from src.datasets.flickr8k_dataset import get_dataloaders
from models.mobile_captioner import MobileNetCaptioner


def train_mobile_model(epochs: int = 10, batch_size: int = 128, lr: float = 8e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60, flush=True)
    print(f"Training Ultra-Compact MobileNetV3 Captioner on {device}", flush=True)
    print("=" * 60, flush=True)

    vocab = Vocabulary.load(PROJECT_ROOT / "dataset" / "vocab.json")
    vocab_size = len(vocab)
    cache_path = PROJECT_ROOT / "dataset" / "cached_features" / "mobilenetv3_global.pt"
    assert cache_path.exists(), f"Feature cache missing: {cache_path}"

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size, vocab=vocab, feature_cache_path=cache_path
    )
    print(f"Loaded {len(train_loader.dataset)} training samples ({len(train_loader)} batches/epoch)", flush=True)

    model = MobileNetCaptioner(vocab_size=vocab_size, embedding_dim=128, hidden_dim=256).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    out_dir = PROJECT_ROOT / "checkpoints" / "mobile"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        total_tokens = 0

        for feats, inps, tgts, _ in train_loader:
            feats = feats.to(device, non_blocking=True)
            inps = inps.to(device, non_blocking=True)
            tgts = tgts.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                logits = model(feats, inps)
                loss = criterion(logits.reshape(-1, vocab_size), tgts.reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            valid = (tgts != PAD_IDX).sum().item()
            running_loss += loss.item() * valid
            total_tokens += valid

        train_loss = running_loss / max(total_tokens, 1)
        scheduler.step()

        # Validation
        model.eval()
        val_sum, val_tok = 0.0, 0
        with torch.no_grad():
            for i in range(min(len(val_loader.dataset), 200)):
                feat_s, refs, _ = val_loader.dataset[i]
                from src.preprocessing.sequence_utils import numericalize_caption, prepare_teacher_forcing_pair
                ids = numericalize_caption(refs[0], vocab, add_special_tokens=True)
                inp, tgt = prepare_teacher_forcing_pair(ids)
                inp_t = torch.tensor([inp], dtype=torch.long, device=device)
                tgt_t = torch.tensor([tgt], dtype=torch.long, device=device)

                with autocast(enabled=(device.type == "cuda")):
                    vl = criterion(model(feat_s.unsqueeze(0).to(device), inp_t).reshape(-1, vocab_size), tgt_t.reshape(-1))
                nt = (tgt_t != PAD_IDX).sum().item()
                val_sum += vl.item() * nt
                val_tok += nt

        val_loss = val_sum / max(val_tok, 1)
        elapsed = time.time() - t0
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            torch.save(
                {"model_state_dict": model.state_dict(), "vocab_size": vocab_size, "val_loss": val_loss},
                out_dir / "best_model.pth"
            )

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}s" + (" [*]" if is_best else ""), flush=True)

    print("=" * 60, flush=True)
    print(f"Training Complete! Best Val Loss: {best_loss:.4f}", flush=True)
    
    # INT8 Quantization
    print("\nQuantizing to Mobile NPU INT8 format...", flush=True)
    model.eval()
    model_cpu = model.cpu()
    quant_model = quantize_dynamic(model_cpu, {nn.Linear, nn.LSTM}, dtype=torch.qint8)
    quant_path = out_dir / "mobilenet_captioner_int8.pth"
    torch.save(
        {"model_state_dict": quant_model.state_dict(), "vocab_size": vocab_size, "quantized": True},
        quant_path
    )
    final_size_mb = quant_path.stat().st_size / (1024 * 1024)
    print("=" * 60, flush=True)
    print(f"FINAL MOBILE NPU MODEL SIZE: {final_size_mb:.2f} MB !!", flush=True)
    print(f"Saved: {quant_path}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    train_mobile_model()
