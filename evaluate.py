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
from src.evaluation.metrics import compute_corpus_bleu, compute_corpus_rouge_l


def evaluate(checkpoint_path: Path, model_type: str = "baseline"):
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

    # 2. Load Test Dataset (1,091 images)
    test_dataset = Flickr8kDataset(split="test", vocab=vocab)
    print(f"Loaded Test Set: {len(test_dataset)} images")

    # 3. Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab_size = checkpoint.get("vocab_size", len(vocab))

    if model_type == "baseline":
        model = ShowAndTellModel(vocab_size=vocab_size).to(device)
    else:
        model = AdaptiveAttentionModel(vocab_size=vocab_size).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 4. Generate Predictions
    hypotheses = {}
    references = {}

    print("Generating candidate captions across test split...")
    with torch.no_grad():
        for i in range(len(test_dataset)):
            image_tensor, raw_refs, img_name = test_dataset[i]
            image_tensor = image_tensor.unsqueeze(0).to(device)

            # Generate token sequence
            if model_type == "baseline":
                gen_ids = model.generate(image_tensor, max_length=25)[0].cpu().tolist()
            else:
                # Attention model greedy decoding
                gen_ids = []
                # Fallback to decoder generation
                features = model.encoder.forward_spatial(image_tensor)
                b, c, h, w = features.shape
                spatial_flat = features.view(b, c, h * w).permute(0, 2, 1)
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
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "adaptive"])
    args = parser.parse_args()

    evaluate(args.checkpoint, args.model)
