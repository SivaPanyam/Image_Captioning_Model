"""
Production Training Pipeline for Image Captioning Models.
Supports both Baseline (ShowAndTell) and Research (AdaptiveAttention) architectures.
Features:
- Cross-Entropy Loss with PAD masking (ignore_index=0)
- AdamW Optimizer with Cosine Annealing Learning Rate Scheduler
- Gradient Clipping (norm=5.0) to prevent RNN gradient explosion
- Automatic Mixed Precision (AMP) for RTX 2050 (4GB VRAM) efficiency
- Comprehensive Checkpointing (best_model.pth, last_model.pth)
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from configs.config import (
    BATCH_SIZE,
    PROJECT_ROOT,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    FEATURE_DIM,
)
from src.preprocessing.vocabulary import Vocabulary, PAD_IDX
from src.datasets.flickr8k_dataset import Flickr8kDataset, CaptionCollate, get_dataloaders
from models.captioning_model import ShowAndTellModel, AdaptiveAttentionModel


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    grad_clip: float = 5.0,
    model_type: str = "baseline",
) -> float:
    """
    Executes one training epoch across the dataset.
    """
    model.train()
    running_loss = 0.0
    total_tokens = 0

    for step, (images, input_ids, target_ids, lengths) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        target_ids = target_ids.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        with autocast(enabled=(device.type == "cuda")):
            if model_type == "baseline":
                logits = model(images, input_ids)
            else:
                logits, alphas = model(images, input_ids)

            # Reshape for cross entropy: (B * T, vocab_size) vs (B * T)
            vocab_size = logits.size(-1)
            loss = criterion(
                logits.reshape(-1, vocab_size),
                target_ids.reshape(-1),
            )

        # Scaled backward pass
        scaler.scale(loss).backward()

        # Unscale for gradient clipping
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * (target_ids != PAD_IDX).sum().item()
        total_tokens += (target_ids != PAD_IDX).sum().item()

    epoch_loss = running_loss / max(total_tokens, 1)
    return epoch_loss


def validate(
    model: nn.Module,
    val_dataset: Flickr8kDataset,
    device: torch.device,
    vocab: Vocabulary,
    max_eval_samples: int = 200,
    model_type: str = "baseline",
) -> float:
    """
    Runs quick validation tracking token-prediction perplexity / loss.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    running_loss = 0.0
    total_tokens = 0

    eval_count = min(len(val_dataset), max_eval_samples)

    with torch.no_grad():
        for i in range(eval_count):
            img_tensor, raw_references, _ = val_dataset[i]
            img_tensor = img_tensor.unsqueeze(0).to(device)

            # Use first reference caption for validation teacher forcing loss
            from src.preprocessing.sequence_utils import numericalize_caption, prepare_teacher_forcing_pair
            first_ref = raw_references[0]
            ids = numericalize_caption(first_ref, vocab, add_special_tokens=True)
            inp, tgt = prepare_teacher_forcing_pair(ids)

            inp_tensor = torch.tensor([inp], dtype=torch.long, device=device)
            tgt_tensor = torch.tensor([tgt], dtype=torch.long, device=device)

            with autocast(enabled=(device.type == "cuda")):
                if model_type == "baseline":
                    logits = model(img_tensor, inp_tensor)
                else:
                    logits, _ = model(img_tensor, inp_tensor)

                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_tensor.reshape(-1))

            num_tokens = (tgt_tensor != PAD_IDX).sum().item()
            running_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return running_loss / max(total_tokens, 1)


def save_checkpoint(
    state: Dict[str, Any], is_best: bool, checkpoint_dir: Path
) -> None:
    """Saves checkpoint to disk."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / "last_model.pth"
    torch.save(state, last_path)
    if is_best:
        best_path = checkpoint_dir / "best_model.pth"
        torch.save(state, best_path)
        print(f"[*] New best validation loss achieved. Checkpoint saved to: {best_path}")


def main():
    parser = argparse.ArgumentParser(description="Image Captioning Training Engine")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "adaptive"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training minibatch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for AdamW")
    parser.add_argument("--grad-clip", type=float, default=5.0, help="Gradient clipping norm")
    parser.add_argument("--fine-tune-encoder", action="store_true", help="Allow gradient updates on ResNet50 blocks")
    args = parser.parse_args()

    # Hardware detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"Starting Training: Model={args.model.upper()} | Device={device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM Available: {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)} GB")
    print("=" * 60)

    # 1. Load Vocabulary
    vocab_path = PROJECT_ROOT / "dataset" / "vocab.json"
    vocab = Vocabulary.load(vocab_path)
    vocab_size = len(vocab)
    print(f"Loaded Vocabulary: {vocab_size} tokens")

    # 2. Build DataLoaders
    train_loader, val_loader, _ = get_dataloaders(batch_size=args.batch_size, vocab=vocab)
    print(f"Loaded {len(train_loader.dataset)} training samples ({len(train_loader)} batches/epoch)")

    # 3. Instantiate Architecture
    if args.model == "baseline":
        model = ShowAndTellModel(
            vocab_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            feature_dim=FEATURE_DIM,
            fine_tune_encoder=args.fine_tune_encoder,
        ).to(device)
    else:
        model = AdaptiveAttentionModel(
            vocab_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            feature_dim=FEATURE_DIM,
            attention_dim=256,
            fine_tune_encoder=args.fine_tune_encoder,
        ).to(device)

    # 4. Loss, Optimizer, Scaler, Scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # 5. Training Loop
    best_val_loss = float("inf")
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / args.model

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip=args.grad_clip,
            model_type=args.model,
        )
        scheduler.step()

        val_loss = validate(
            model=model,
            val_dataset=val_loader.dataset,
            device=device,
            vocab=vocab,
            model_type=args.model,
        )
        elapsed = time.time() - t0

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| LR: {scheduler.get_last_lr()[0]:.6f} "
            f"| Time: {elapsed:.1f}s"
        )

        save_checkpoint(
            state={
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "vocab_size": vocab_size,
                "model_type": args.model,
            },
            is_best=is_best,
            checkpoint_dir=checkpoint_dir,
        )

    print("=" * 60)
    print(f"Training Complete! Best Validation Loss: {best_val_loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
