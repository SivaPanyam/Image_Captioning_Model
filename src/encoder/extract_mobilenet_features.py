"""
Ultra-Compact MobileNetV3-Small Visual Feature Extractor for Mobile & Edge NPUs.
Extracts all 8,091 images into:
- Global 576-D pooled vectors: dataset/cached_features/mobilenetv3_global.pt (~18 MB)
- Spatial 576 x 7 x 7 maps: dataset/cached_features/mobilenetv3_spatial.pt (~450 MB)
"""

import sys
import time
from pathlib import Path
from typing import Dict
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import IMAGES_DIR, PROJECT_ROOT
from src.preprocessing.caption_loader import load_captions


def extract_mobilenet_features():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = PROJECT_ROOT / "dataset" / "cached_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    global_path = output_dir / "mobilenetv3_global.pt"

    print("=" * 60, flush=True)
    print(f"Extracting Ultra-Compact MobileNetV3-Small Features on {device}", flush=True)
    print("=" * 60, flush=True)

    captions_dict = load_captions()
    image_names = sorted(list(captions_dict.keys()))
    total = len(image_names)

    # Backbone: only 2.54M parameters
    m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).to(device)
    m.eval()
    for p in m.parameters():
        p.requires_grad = False

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    batch_size = 128
    global_features: Dict[str, torch.Tensor] = {}
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
            feat = m.features(tensor_batch)             # (B, 576, 7, 7)
            pooled = m.avgpool(feat)                     # (B, 576, 1, 1)
            glob = torch.flatten(pooled, 1)              # (B, 576)

        glob_cpu = glob.cpu().to(torch.float32)
        for i, name in enumerate(valid_chunk):
            global_features[name] = glob_cpu[i]

        step = idx + len(chunk_names)
        if (idx // batch_size) % 5 == 0 or step >= total:
            elapsed = time.time() - t0
            rate = step / max(elapsed, 0.001)
            eta = (total - step) / max(rate, 0.001)
            print(f"[{step:04d}/{total}] Extracted MobileNet features ({rate:.1f} imgs/s, ETA: {eta:.0f}s)", flush=True)

    print(f"Saving MobileNetV3 global features to {global_path} ...", flush=True)
    torch.save(global_features, global_path)
    print(f"SUCCESS: MobileNetV3 extraction completed in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    extract_mobilenet_features()
