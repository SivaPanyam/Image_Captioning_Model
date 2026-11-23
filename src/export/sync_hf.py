"""
Robust Hugging Face Model Hub Sync Script.
Uploads model cards, vocabulary, and quantized checkpoints using chunked upload_file.
"""

from huggingface_hub import HfApi
from pathlib import Path
import os

token = os.environ.get("HF_TOKEN", "")
api = HfApi(token=token)
repo_id = "SivaPanyam/Neural-Image-Captioning-Model"

print("=" * 60)
print(f"Syncing Production Assets to: https://huggingface.co/{repo_id}")
print("=" * 60)

# 1. Upload Model Card
print("Uploading README.md ...")
api.upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="model",
)
print("  ✓ README.md synced")

# 2. Upload Vocabulary
print("Uploading dataset/vocab.json ...")
api.upload_file(
    path_or_fileobj="dataset/vocab.json",
    path_in_repo="dataset/vocab.json",
    repo_id=repo_id,
    repo_type="model",
)
print("  ✓ vocab.json synced")

# 3. Upload Ultra-Compact 6.08 MB Mobile NPU Model
print("Uploading checkpoints/mobile/mobilenet_captioner_int8.pth ...")
api.upload_file(
    path_or_fileobj="checkpoints/mobile/mobilenet_captioner_int8.pth",
    path_in_repo="checkpoints/mobile/mobilenet_captioner_int8.pth",
    repo_id=repo_id,
    repo_type="model",
)
print("  ✓ mobilenet_captioner_int8.pth synced (6.08 MB)")

# 4. Upload Gradio Space app.py
print("Uploading app/gradio_app.py as app.py ...")
api.upload_file(
    path_or_fileobj="app/gradio_app.py",
    path_in_repo="app.py",
    repo_id=repo_id,
    repo_type="model",
)
print("  ✓ app.py synced")

print("=" * 60)
print("ALL HUGGING FACE ASSETS SUCCESSFULLY SYNCHRONIZED!")
print("=" * 60)
