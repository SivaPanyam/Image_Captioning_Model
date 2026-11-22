"""
Training Pipeline for Vision Transformer (ViT-B/16) + Cross-Attention Transformer Decoder.
Trains end-to-end multi-head cross-attention over cached 196-patch visual representations.
Features:
- AdamW with Cosine Annealing learning rate schedule
- Label smoothing cross-entropy (0.1) to prevent overconfident single-token collapse
- Gradient clipping (1.0)
- FP16 Automatic Mixed Precision (AMP)
"""

import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import PROJECT_ROOT
from src.preprocessing.vocabulary import Vocabulary, PAD_IDX
from src.datasets.flickr8k_dataset import get_dataloaders
from src.decoder.vit_transformer_decoder import ViTTransformerCaptioningModel


def train_vit_transformer(epochs: int = 15, batch_size: int = 64, lr: float = 3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60, flush=True)
    print(f"Training SOTA Vision Transformer Captioning Model on {device}", flush=True)
    print("=" * 60, flush=True)

    # 1. Load Vocab
    vocab = Vocabulary.load(PROJECT_ROOT / "dataset" / "vocab.json")
    vocab_size = len(vocab)
    print(f"Loaded Vocabulary: {vocab_size} tokens", flush=True)

    # 2. Load Dataloaders with cached ViT patches
    cache_path = PROJECT_ROOT / "dataset" / "cached_features" / "vit_b16_patches.pt"
    assert cache_path.exists(), f"ViT patch cache not found: {cache_path}"
    print(f"Loading cached ViT patches: {cache_path} ...", flush=True)

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size, vocab=vocab, feature_cache_path=cache_path
    )
    print(f"Loaded {len(train_loader.dataset)} training samples ({len(train_loader)} batches/epoch)", flush=True)

    # 3. Model
    model = ViTTransformerCaptioningModel(
        vocab_size=vocab_size,
        d_model=512,
        vit_dim=768,
        num_layers=4,
        num_heads=8,
        d_ff=2048,
        dropout=0.1,
    ).to(device)

    # 4. Criterion, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "vit_transformer"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        total_tokens = 0

        for patches, input_ids, target_ids, _ in train_loader:
            patches = patches.to(device, dtype=torch.float32, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(device.type == "cuda")):
                logits = model(patches, input_ids) # (B, T, V)
                loss = criterion(logits.reshape(-1, vocab_size), target_ids.reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            valid_tokens = (target_ids != PAD_IDX).sum().item()
            running_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

        train_loss = running_loss / max(total_tokens, 1)
        scheduler.step()

        # Quick validation
        model.eval()
        val_loss_sum = 0.0
        val_tokens = 0
        eval_count = min(len(val_loader.dataset), 200)

        with torch.no_grad():
            for i in range(eval_count):
                patches_sample, raw_refs, _ = val_loader.dataset[i]
                patches_t = patches_sample.unsqueeze(0).to(device, dtype=torch.float32)

                from src.preprocessing.sequence_utils import numericalize_caption, prepare_teacher_forcing_pair
                ref = raw_refs[0]
                ids = numericalize_caption(ref, vocab, add_special_tokens=True)
                inp, tgt = prepare_teacher_forcing_pair(ids)

                inp_t = torch.tensor([inp], dtype=torch.long, device=device)
                tgt_t = torch.tensor([tgt], dtype=torch.long, device=device)

                with autocast(enabled=(device.type == "cuda")):
                    v_logits = model(patches_t, inp_t)
                    v_loss = criterion(v_logits.reshape(-1, vocab_size), tgt_t.reshape(-1))

                nt = (tgt_t != PAD_IDX).sum().item()
                val_loss_sum += v_loss.item() * nt
                val_tokens += nt

        val_loss = val_loss_sum / max(val_tokens, 1)
        elapsed = time.time() - t0

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "vocab_size": vocab_size,
                    "model_type": "vit_transformer",
                },
                checkpoint_dir / "best_model.pth",
            )

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| LR: {scheduler.get_last_lr()[0]:.6f} "
            f"| Time: {elapsed:.1f}s"
            + (" [* New Best]" if is_best else ""),
            flush=True,
        )

    print("=" * 60, flush=True)
    print(f"ViT-Transformer Training Complete! Best Val Loss: {best_val_loss:.4f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    train_vit_transformer()
