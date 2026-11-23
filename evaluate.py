"""
Evaluation CLI script for Image Captioning Models.
Evaluates a trained checkpoint on the isolated Flickr8k Test Set (1,091 images).
"""

import argparse
from pathlib import Path
import torch

from configs.config import PROJECT_ROOT, BATCH_SIZE
from src.preprocessing.vocabulary import Vocabulary
from src.datasets.flickr8k_dataset import Flickr8kDataset
from models.captioning_model import ShowAndTellModel, AdaptiveAttentionModel
from src.decoder.vit_transformer_decoder import ViTTransformerCaptioningModel
from src.evaluation.metrics import compute_corpus_bleu, compute_corpus_rouge_l


def evaluate(checkpoint_path: Path, model_type: str = "baseline", max_samples: int = 0):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"Evaluating Model: {model_type.upper()} | Checkpoint: {checkpoint_path.name}")
    print(f"Device: {device}")
    print("=" * 60)

    # 1. Load Vocab
    vocab_path = PROJECT_ROOT / "dataset" / "vocab.json"
    vocab = Vocabulary.load(vocab_path)

    # 2. Load Test Dataset with high-speed feature caching
    cache_path = None
    if model_type == "baseline":
        g_cache = PROJECT_ROOT / "dataset" / "cached_features" / "resnet50_global.pt"
        if g_cache.exists():
            cache_path = g_cache
    elif model_type == "adaptive":
        s_cache = PROJECT_ROOT / "dataset" / "cached_features" / "resnet50_spatial.pt"
        if s_cache.exists():
            cache_path = s_cache
    elif model_type == "vit_transformer":
        v_cache = PROJECT_ROOT / "dataset" / "cached_features" / "vit_b16_patches.pt"
        if v_cache.exists():
            cache_path = v_cache

    test_dataset = Flickr8kDataset(split="test", vocab=vocab, feature_cache_path=cache_path)
    total_samples = len(test_dataset)
    if max_samples > 0:
        total_samples = min(total_samples, max_samples)
    print(f"Loaded Test Set: {total_samples} images to evaluate (cached features: {cache_path is not None})")

    # 3. Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab_size = checkpoint.get("vocab_size", len(vocab))

    if model_type == "baseline":
        model = ShowAndTellModel(vocab_size=vocab_size).to(device)
    elif model_type == "adaptive":
        model = AdaptiveAttentionModel(vocab_size=vocab_size).to(device)
    else:
        model = ViTTransformerCaptioningModel(vocab_size=vocab_size).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 4. Generate Predictions
    hypotheses = {}
    references = {}

    print("Generating candidate captions across test split...")
    with torch.no_grad():
        for i in range(total_samples):
            item_tensor, raw_refs, img_name = test_dataset[i]
            item_tensor = item_tensor.unsqueeze(0).to(device)

            # Generate token sequence
            if model_type == "baseline":
                if item_tensor.dim() == 2 or (item_tensor.dim() == 4 and item_tensor.size(1) != 3):
                    # Pre-extracted global feature (1, 2048)
                    feat = item_tensor
                    gen_ids = model.decoder.generate_caption(feat, max_length=25)[0].cpu().tolist()
                else:
                    gen_ids = model.generate(item_tensor, max_length=25)[0].cpu().tolist()
            elif model_type == "adaptive":
                # Attention model greedy decoding
                gen_ids = []
                if item_tensor.dim() == 4 and item_tensor.size(1) != 3:
                    # Pre-extracted spatial feature (1, 2048, 7, 7) float16 -> float32
                    b, c, h, w = item_tensor.shape
                    spatial_flat = item_tensor.view(b, c, h * w).permute(0, 2, 1).to(torch.float32)
                else:
                    features = model.encoder.forward_spatial(item_tensor)
                    b, c, h, w = features.shape
                    spatial_flat = features.view(b, c, h * w).permute(0, 2, 1).to(torch.float32)
                # Decode steps
                curr = torch.tensor([[1]], dtype=torch.long, device=device) # <START>
                h_s, c_s = model.decoder.init_hidden_state(spatial_flat)
                for _ in range(25):
                    emb = model.decoder.embedding(curr[:, -1:])
                    ctx, _ = model.decoder.attention(spatial_flat, h_s)
                    gate = torch.sigmoid(model.decoder.sentinel_gate(torch.cat([h_s, emb.squeeze(1)], dim=-1)))
                    h_s, c_s = model.decoder.lstm_cell(torch.cat([emb.squeeze(1), gate * ctx], dim=-1), (h_s, c_s))
                    next_id = model.decoder.fc(h_s).argmax(dim=-1).item()
                    if next_id == 2: # <END>
                        break
                    gen_ids.append(next_id)
                    curr = torch.tensor([[next_id]], dtype=torch.long, device=device)
            else:
                # ViT-Transformer autoregressive greedy decoding
                gen_ids = []
                if item_tensor.dim() == 3:
                    # Pre-extracted patches (1, 196, 768)
                    patches = item_tensor.to(torch.float32)
                else:
                    patches, _ = model.encoder(item_tensor)
                memory = model.visual_proj(patches)

                generated = [1] # <START>
                for step in range(25):
                    tgt_in = torch.tensor([generated], dtype=torch.long, device=device)
                    seq_len = tgt_in.size(1)
                    tgt_mask = model.generate_square_subsequent_mask(seq_len, device)
                    tgt_emb = model.embedding(tgt_in) * (model.d_model ** 0.5)
                    tgt_emb = model.pos_encoder(tgt_emb)

                    for layer in model.layers:
                        tgt_emb = layer(tgt_emb, memory, tgt_mask=tgt_mask)
                    tgt_emb = model.final_norm(tgt_emb)
                    logits = model.fc_out(tgt_emb[:, -1])

                    if step < 4:
                        logits[0, 2] = -1e9 # Prevent early <END>
                    logits[0, 3] = -1e9     # Suppress <UNK>

                    nxt_id = logits.argmax(dim=-1).item()
                    if nxt_id == 2: # <END>
                        break
                    generated.append(nxt_id)
                    gen_ids.append(nxt_id)

            # Filter out special tokens
            cand_tokens = [
                vocab.get_word(idx)
                for idx in gen_ids
                if idx not in [0, 1, 2, 3] # PAD, START, END, UNK
            ]

            hypotheses[img_name] = cand_tokens
            references[img_name] = raw_refs

    # 5. Compute Benchmark Metrics
    bleu_scores = compute_corpus_bleu(hypotheses, references, max_order=4)
    rouge_l = compute_corpus_rouge_l(hypotheses, references)

    print("\n" + "=" * 60)
    print("BENCHMARK EVALUATION RESULTS (TEST SPLIT: 1,091 IMAGES)")
    print("=" * 60)
    for metric, score in bleu_scores.items():
        print(f"  {metric:<10}: {score:.2f}")
    print(f"  ROUGE-L   : {rouge_l:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Image Captioning Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "adaptive", "vit_transformer"])
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of evaluation samples (0 for all)")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.model, max_samples=args.max_samples)
