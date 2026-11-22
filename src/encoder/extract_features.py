"""
Optimized In-Memory Feature Extraction for Flickr8k on RTX 2050.
Extracts 8,091 images in batches of 64 into:
- dataset/cached_features/resnet50_global.pt (8091 x 2048 float32 tensor dict)
- dataset/cached_features/resnet50_spatial.pt (8091 x 2048 x 7 x 7 float16 tensor dict)
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from typing import Dict, List
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms

from configs.config import IMAGES_DIR, PROJECT_ROOT
from src.preprocessing.caption_loader import load_captions


def run_extraction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = PROJECT_ROOT / "dataset" / "cached_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    global_path = output_dir / "resnet50_global.pt"
    spatial_path = output_dir / "resnet50_spatial.pt"

    print("=" * 60, flush=True)
    print(f"Starting Offline Feature Extraction on {device}", flush=True)
    print("=" * 60, flush=True)

    captions_dict = load_captions()
    image_names = sorted(list(captions_dict.keys()))
    total = len(image_names)
    print(f"Found {total} images to extract.", flush=True)

    # Backbone
    backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    spatial_net = nn.Sequential(*list(backbone.children())[:-2]).to(device)
    avgpool = backbone.avgpool.to(device)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    batch_size = 64
    global_features: Dict[str, torch.Tensor] = {}
    spatial_features: Dict[str, torch.Tensor] = {}

    start_time = time.time()

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
            spatial = spatial_net(tensor_batch)             # (B, 2048, 7, 7)
            pooled = avgpool(spatial)                       # (B, 2048, 1, 1)
            glob = torch.flatten(pooled, 1)                 # (B, 2048)

        # Store in dicts as CPU tensors
        glob_cpu = glob.cpu().to(torch.float32)
        spatial_cpu = spatial.cpu().to(torch.float16)

        for i, name in enumerate(valid_chunk):
            global_features[name] = glob_cpu[i]
            spatial_features[name] = spatial_cpu[i]

        step = idx + len(chunk_names)
        if (idx // batch_size) % 5 == 0 or step >= total:
            elapsed = time.time() - start_time
            rate = step / max(elapsed, 0.001)
            eta = (total - step) / max(rate, 0.001)
            print(f"[{step:04d}/{total}] Extracted features ({rate:.1f} imgs/s, ETA: {eta:.0f}s)", flush=True)

    print(f"Saving global features to: {global_path} ...", flush=True)
    torch.save(global_features, global_path)

    print(f"Saving spatial features to: {spatial_path} ...", flush=True)
    torch.save(spatial_features, spatial_path)

    total_time = time.time() - start_time
    print("=" * 60, flush=True)
    print(f"SUCCESS: Feature extraction completed in {total_time:.1f}s!", flush=True)
    print(f"Saved: {len(global_features)} global vectors & {len(spatial_features)} spatial feature maps.", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    run_extraction()
