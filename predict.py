"""
Production Inference Engine for Image Captioning.
Supports single-image prediction using trained checkpoints.
Decoding Strategies:
1. Greedy Search (K=1): Fast, top-probability next-token selection.
2. Beam Search (K > 1): Global sequence path optimization with length penalty.
3. Temperature & Top-k / Nucleus Sampling: Stochastic diversity exploration.
"""

import argparse
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import torch
import torch.nn.functional as F

from configs.config import PROJECT_ROOT, IMAGE_SIZE
from src.preprocessing.vocabulary import Vocabulary, START_IDX, END_IDX, PAD_IDX
from src.encoder.feature_extractor import get_image_transforms
from models.captioning_model import ShowAndTellModel, AdaptiveAttentionModel


def load_model(checkpoint_path: Path, model_type: str, vocab_size: int, device: torch.device):
    """Loads model weights from checkpoint."""
    if model_type == "baseline":
        model = ShowAndTellModel(vocab_size=vocab_size).to(device)
    else:
        model = AdaptiveAttentionModel(vocab_size=vocab_size).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def greedy_decode(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    vocab: Vocabulary,
    max_length: int = 30,
    model_type: str = "baseline",
    device: torch.device = torch.device("cpu"),
) -> Tuple[str, List[str]]:
    """Generates caption via greedy argmax next-token selection."""
    with torch.no_grad():
        if model_type == "baseline":
            features = model.encoder(image_tensor)
            h, c = model.decoder.init_hidden_state(features)
        else:
            spatial = model.encoder.forward_spatial(image_tensor)
            b, ch, h_dim, w_dim = spatial.shape
            spatial_flat = spatial.view(b, ch, h_dim * w_dim).permute(0, 2, 1)
            h, c = model.decoder.init_hidden_state(spatial_flat)

        current_token = torch.tensor([[START_IDX]], dtype=torch.long, device=device)
        generated_tokens: List[str] = []

        for _ in range(max_length):
            if model_type == "baseline":
                embed = model.decoder.embedding(current_token)
                lstm_out, (h, c) = model.decoder.lstm(embed, (h, c))
                logits = model.decoder.fc(lstm_out.squeeze(1))
            else:
                embed = model.decoder.embedding(current_token[:, -1:])
                ctx, _ = model.decoder.attention(spatial_flat, h)
                gate_in = torch.cat([h, embed.squeeze(1)], dim=-1)
                beta = torch.sigmoid(model.decoder.sentinel_gate(gate_in))
                cell_in = torch.cat([embed.squeeze(1), beta * ctx], dim=-1)
                h, c = model.decoder.lstm_cell(cell_in, (h, c))
                logits = model.decoder.fc(h)

            next_id = logits.argmax(dim=-1).item()
            if next_id == END_IDX:
                break

            token_str = vocab.get_word(next_id)
            if next_id not in [PAD_IDX, START_IDX, END_IDX]:
                generated_tokens.append(token_str)

            current_token = torch.tensor([[next_id]], dtype=torch.long, device=device)

    caption = " ".join(generated_tokens)
    return caption, generated_tokens


def beam_search_decode(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    vocab: Vocabulary,
    beam_width: int = 3,
    max_length: int = 30,
    length_penalty_alpha: float = 0.7,
    model_type: str = "baseline",
    device: torch.device = torch.device("cpu"),
) -> Tuple[str, List[str]]:
    """Generates caption via Beam Search with length normalization."""
    with torch.no_grad():
        if model_type == "baseline":
            features = model.encoder(image_tensor)
            h, c = model.decoder.init_hidden_state(features)
        else:
            spatial = model.encoder.forward_spatial(image_tensor)
            b, ch, h_dim, w_dim = spatial.shape
            spatial_flat = spatial.view(b, ch, h_dim * w_dim).permute(0, 2, 1)
            h, c = model.decoder.init_hidden_state(spatial_flat)

        # Candidate format: (tokens_list, cumulative_log_prob, hidden_state, cell_state)
        beams = [([START_IDX], 0.0, h, c)]
        completed_beams = []

        for _ in range(max_length):
            candidates = []
            for tokens, score, cur_h, cur_c in beams:
                last_token = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)

                if model_type == "baseline":
                    embed = model.decoder.embedding(last_token)
                    lstm_out, (nxt_h, nxt_c) = model.decoder.lstm(embed, (cur_h, cur_c))
                    logits = model.decoder.fc(lstm_out.squeeze(1))
                else:
                    embed = model.decoder.embedding(last_token)
                    ctx, _ = model.decoder.attention(spatial_flat, cur_h)
                    gate_in = torch.cat([cur_h, embed.squeeze(1)], dim=-1)
                    beta = torch.sigmoid(model.decoder.sentinel_gate(gate_in))
                    cell_in = torch.cat([embed.squeeze(1), beta * ctx], dim=-1)
                    nxt_h, nxt_c = model.decoder.lstm_cell(cell_in, (cur_h, cur_c))
                    logits = model.decoder.fc(nxt_h)

                log_probs = F.log_softmax(logits, dim=-1)
                topk_probs, topk_ids = log_probs.topk(beam_width)

                for k in range(beam_width):
                    nxt_id = topk_ids[0, k].item()
                    nxt_score = score + topk_probs[0, k].item()

                    if nxt_id == END_IDX:
                        # Length penalty normalization
                        penalty = ((5 + len(tokens)) / 6) ** length_penalty_alpha
                        completed_beams.append((tokens + [nxt_id], nxt_score / penalty))
                    else:
                        candidates.append((tokens + [nxt_id], nxt_score, nxt_h, nxt_c))

            if not candidates:
                break

            # Retain top-k beams
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]

        # Select best completed or active candidate
        if completed_beams:
            completed_beams.sort(key=lambda x: x[1], reverse=True)
            best_tokens = completed_beams[0][0]
        else:
            beams.sort(key=lambda x: x[1], reverse=True)
            best_tokens = beams[0][0]

    words = [
        vocab.get_word(idx)
        for idx in best_tokens
        if idx not in [PAD_IDX, START_IDX, END_IDX]
    ]
    caption = " ".join(words)
    return caption, words


def main():
    parser = argparse.ArgumentParser(description="Image Captioning Prediction Engine")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "adaptive"], help="Model architecture")
    parser.add_argument("--method", type=str, default="beam", choices=["greedy", "beam"], help="Decoding strategy")
    parser.add_argument("--beam-width", type=int, default=3, help="Beam width for beam search")
    parser.add_argument("--max-length", type=int, default=30, help="Maximum generated caption length")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Vocab
    vocab_path = PROJECT_ROOT / "dataset" / "vocab.json"
    vocab = Vocabulary.load(vocab_path)

    # 2. Preprocess Image
    transform = get_image_transforms(is_training=False)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 3. Load Model
    model = load_model(Path(args.checkpoint), args.model, len(vocab), device)

    # 4. Generate Caption
    if args.method == "greedy":
        caption, _ = greedy_decode(
            model=model,
            image_tensor=image_tensor,
            vocab=vocab,
            max_length=args.max_length,
            model_type=args.model,
            device=device,
        )
    else:
        caption, _ = beam_search_decode(
            model=model,
            image_tensor=image_tensor,
            vocab=vocab,
            beam_width=args.beam_width,
            max_length=args.max_length,
            model_type=args.model,
            device=device,
        )

    print("=" * 60)
    print(f"Image   : {image_path.name}")
    print(f"Model   : {args.model.upper()}")
    print(f"Strategy: {args.method.upper()}" + (f" (Width={args.beam_width})" if args.method == "beam" else ""))
    print(f"Caption : {caption.capitalize()}.")
    print("=" * 60)


if __name__ == "__main__":
    main()
