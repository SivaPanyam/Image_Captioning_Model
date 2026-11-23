# Neural Image Captioning: From CNN-LSTM to Vision Transformers with Adaptive Sentinel Gating

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.11.0%2Bcu128-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)

A complete, production-grade vision-language deep learning system built in PyTorch. This project evaluates the evolution of neural image captioning on the **Flickr8k benchmark**, progressing from classic CNN-LSTMs to **Dual-Stream Vision Transformers (ViT-B/16)** with **Adaptive Visual Sentinel Cross-Attention**.

---

## 📊 Benchmark Evaluation Matrix (Flickr8k Test Split: 1,091 Images)

Evaluated under multi-reference ground truth (5 references per image) with corpus-level n-gram precision and recall:

| Architecture | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | Parameters | Latency / Img |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline (Show & Tell: ResNet-50 + LSTM)** | 56.88 | 39.75 | 26.46 | 17.47 | 42.90 | ~25.5 M | ~12 ms |
| **2. Adaptive Spatial Attention (ResNet-50 + Sentinel Gate)** | 61.75 | 44.22 | 30.71 | 21.51 | 45.67 | ~28.2 M | ~18 ms |
| **3. Vision Transformer (ViT-B/16 + Cross-Attention Decoder)** | **61.52** | **44.69** | **31.84** | **`22.17`** | **45.62** | ~98.4 M | ~24 ms |
| **Absolute Performance Delta ($\Delta$)** | **+4.64%** | **+4.94%** | **+5.38%** | **`+4.70%`** | **+2.72%** | — | — |

---

## 🧠 Architectural Innovations

### 1. Adaptive Visual Sentinel Gating ($\beta_t$)
Standard attention decoders attend to image features unconditionally at every decoding step, forcing the model to look at pixels even when generating non-visual syntactic tokens (*"a"*, *"of"*, *"in"*). 

Our architecture implements a **Visual Sentinel Gate**:
$$\mathbf{s}_t = g(\mathbf{W}_x \mathbf{x}_t + \mathbf{W}_h \mathbf{h}_{t-1})$$
$$\beta_t = \sigma(\mathbf{w}_\beta^T [\mathbf{h}_t; \mathbf{s}_t] + b_\beta) \in [0, 1]$$
$$\hat{\mathbf{c}}_t = \beta_t \mathbf{c}_t + (1 - \beta_t) \mathbf{s}_t$$

When generating visual words (*"dog"*, *"pink"*, *"cannon"*), $\beta_t \to 1$, routing spatial patch attention. When generating functional syntax (*"the"*, *"with"*), $\beta_t \to 0$, protecting the visual representation from noise drift.

### 2. Vision Transformer Patch Representation (ViT-B/16)
Rather than flattening low-resolution CNN feature grids, input images ($224 \times 224$) are projected into $196$ non-overlapping $16 \times 16$ patch tokens ($D = 768$). Full self-attention across the patches enables global context and fine-grained texture discrimination (e.g., distinguishing dress textures from shirts and counting multiple interacting subjects).

### 3. Frequency-Aware Specificity Beam Search
Standard Maximum Likelihood Estimation (MLE) training causes mode collapse toward statistical heads (*"A group of people standing"*). We integrate an **Information-Theoretic Inverse Frequency Penalty**:
$$\mathbf{z}_t'(w) = \mathbf{z}_t(w) - \lambda \cdot \log(1 + f(w)) - \gamma \cdot \mathbb{I}[w \in w_{<t}]$$
This elevates rare, descriptive tokens (*"uniform"*, *"men"*, *"playing"*) into the active beam paths.

---

## 🚀 Quickstart & Inference

### 1. Installation
```bash
git clone https://github.com/SivaPanyam/Image_Captioning_Model.git
cd Image_Captioning_Model
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Single Image Caption Prediction (CLI)
```bash
# Vision Transformer (ViT-B/16):
python predict.py --image "sample.jpg" --checkpoint "checkpoints/vit_transformer/best_model.pth" --model vit_transformer

# Adaptive Attention Model with Specificity Boost:
python predict.py --image "sample.jpg" --checkpoint "checkpoints/adaptive/best_model.pth" --model adaptive --method beam --beam-width 4 --specificity-factor 0.4
```

### 3. Run Web Application (Streamlit or Gradio)
```bash
# Launch Streamlit:
streamlit run app/streamlit_app.py

# Launch Gradio (Hugging Face Spaces):
python app/gradio_app.py
```

---

## 🤗 Hugging Face Spaces Deployment Guide

To deploy this project as a live, shareable public web application on Hugging Face Spaces:

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Click **New Space** $\to$ Select **Gradio** SDK $\to$ Choose **Free CPU / GPU**.
3. Clone your Space repository locally:
   ```bash
   git clone https://huggingface.co/spaces/<YOUR_USERNAME>/<SPACE_NAME>
   ```
4. Copy the following files into your Space repo:
   ```text
   app/gradio_app.py      -> app.py
   requirements.txt       -> requirements.txt
   checkpoints/           -> checkpoints/
   dataset/vocab.json     -> dataset/vocab.json
   src/                   -> src/
   configs/               -> configs/
   ```
5. Commit and push:
   ```bash
   git add .
   git commit -m "feat: deploy neural image captioning to Hugging Face Spaces"
   git push
   ```
   Hugging Face will automatically build the environment and host your app at `https://huggingface.co/spaces/<YOUR_USERNAME>/<SPACE_NAME>`.

---

## 📁 Repository Structure

```text
Image_Captioning_Model/
├── app/
│   ├── streamlit_app.py          # Interactive Streamlit Web App
│   └── gradio_app.py             # Hugging Face Spaces Gradio App
├── checkpoints/
│   ├── baseline/best_model.pth   # Show and Tell CNN-LSTM weights
│   ├── adaptive/best_model.pth   # Adaptive Sentinel Attention weights
│   └── vit_transformer/          # ViT-B/16 + Cross-Attention weights
├── configs/
│   └── config.py                 # Central hyperparameter configuration
├── dataset/
│   ├── vocab.json                # 2,563-token isolated training vocabulary
│   └── cached_features/          # Pre-extracted visual tensor caches
├── models/
│   └── captioning_model.py       # End-to-end model definitions
├── src/
│   ├── encoder/
│   │   ├── resnet_encoder.py     # ResNet-50 visual backbone
│   │   ├── vit_encoder.py        # ViT-B/16 Vision Transformer backbone
│   │   └── extract_features.py   # High-speed offline tensor caching
│   ├── decoder/
│   │   ├── lstm_decoder.py       # Baseline LSTM decoder
│   │   ├── attention_decoder.py  # Adaptive Sentinel spatial decoder
│   │   └── vit_transformer_decoder.py # Multi-head cross-attention decoder
│   ├── datasets/
│   │   └── flickr8k_dataset.py   # PyTorch Dataset & dynamic batch collator
│   └── evaluation/
│       └── metrics.py            # Corpus BLEU-1..4 & ROUGE-L implementations
├── evaluate.py                   # Multi-architecture evaluation CLI
├── predict.py                    # Inference CLI with Specificity Beam Search
├── train.py                      # CNN-LSTM training pipeline
├── train_vit.py                  # ViT-Transformer training pipeline
├── requirements.txt              # Production dependency specifications
└── README.md                     # Research documentation
```

---

## 📜 Citation & License
Distributed under the MIT License. Built for reproducible computer vision & natural language processing research.
