"""
High-Speed ViT-B/16 Offline Feature Extractor for Flickr8k on RTX 2050.
Extracts all 8,091 images into 196 contextual patch embeddings (float16):
- Output: dataset/cached_features/vit_b16_patches.pt (8091 x 196 x 768)
"""

import sys
import time
from pathlib import Path
from typing import Dict, List
from PIL import Image
import torch
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import IMAGES_DIR, PROJECT_ROOT
from src.preprocessing.caption_loader import load_captions
from src.encoder.vit_encoder import VisionTransformerEncoder


def extract_vit_features():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = PROJECT_ROOT / "dataset" / "cached_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "vit_b16_patches.pt"

    print("=" * 60, flush=True)
    print(f"Starting ViT-B/16 Feature Extraction on {device}", flush=True)
    print("=" * 60, flush=True)

    captions_dict = load_captions()
    image_names = sorted(list(captions_dict.keys()))
    total = len(image_names)
    print(f"Extracting {total} images into ViT-B/16 patch tokens...", flush=True)

    encoder = VisionTransformerEncoder(fine_tune=False).to(device)
    encoder.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    batch_size = 64
    vit_features: Dict[str, torch.Tensor] = {}
    t0 = time.time()

    for idx in range(0, total, batch_size):
        chunk_names = image_names[idx : idx + batch_size]
        batch_imgs = []
        valid_chunk = []

        for name in chunk_names:
            img_p = IMAGES_DIR / name
            if not img_p.exists():
                continue
            with Image.open(img_p) as img:
                batch_imgs.append(preprocess(img.convert("RGB")))
                valid_chunk.append(name)

        if not batch_imgs:
            continue

        tensor_batch = torch.stack(batch_imgs).to(device)

        with torch.no_grad():
            patch_tokens, _ = encoder(tensor_batch) # (B, 196, 768)

        patch_cpu = patch_tokens.cpu().to(torch.float16)
        for i, name in enumerate(valid_chunk):
            vit_features[name] = patch_cpu[i]

        step = idx + len(chunk_names)
        if (idx // batch_size) % 5 == 0 or step >= total:
            elapsed = time.time() - t0
            rate = step / max(elapsed, 0.001)
            eta = (total - step) / max(rate, 0.001)
            print(f"[{step:04d}/{total}] Extracted ViT patches ({rate:.1f} imgs/s, ETA: {eta:.0f}s)", flush=True)

    print(f"Saving ViT-B/16 patches to {cache_path} ...", flush=True)
    torch.save(vit_features, cache_path)
    print(f"SUCCESS: ViT extraction completed in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    extract_vit_features()
