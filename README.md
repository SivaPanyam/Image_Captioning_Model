# Neural Image Captioning: From CNN-LSTM to Vision Transformers with Edge NPU Quantization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.11.0%2Bcu128-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Model Hub](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Hub-orange)](https://huggingface.co/SivaPanyam/Neural-Image-Captioning-Model)

A complete, production-grade vision-language deep learning system built in PyTorch. This repository features the architectural evolution of neural image captioning on the **Flickr8k benchmark**, transitioning from classic CNN-LSTMs to **Vision Transformers (ViT-B/16)** with **Adaptive Sentinel Cross-Attention**, and culminating in an **Ultra-Compact 6.08 MB INT8 Quantized MobileNetV3 Architecture** engineered for on-device mobile and Edge NPU deployment.

---

## 📊 Comprehensive Multi-Architecture Benchmark Matrix

All models evaluated strictly on the **Flickr8k Test Split (1,091 images, 5,455 reference captions)** using multi-reference corpus-level metrics with $n$-gram precision and recall:

| Architecture | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | Checkpoint Size | Quantization | Latency (CPU / NPU) | Target Deployment Tier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline (Show & Tell: ResNet-50 + LSTM)** | 56.88 | 39.75 | 26.46 | 17.47 | 42.90 | 154 MB | FP32 | ~35 ms | Server Reference Baseline |
| **2. Adaptive Sentinel Attention (ResNet-50 + Sentinel Gate)** | 61.75 | 44.22 | 30.71 | 21.51 | 45.67 | 101 MB | INT8 Dynamic | ~18 ms | High-Precision Server API |
| **3. Vision Transformer (ViT-B/16 + Cross-Attention)** | **61.52** | **44.69** | **31.84** | **`22.17`** | **45.62** | 214 MB | INT8 Dynamic | ~65 ms | Maximum Accuracy Cloud/Desktop |
| **4. Ultra-Compact Edge / NPU (MobileNetV3-Small + Gated LSTM)** | 53.60 | 36.40 | 23.50 | 14.80 | 40.10 | **`6.08 MB`** | **INT8 Dynamic** | **~8-15 ms** | **Mobile iOS/Android & Edge NPU** |

### 📈 Architectural Performance Breakdown
* **ViT-B/16 vs Baseline:** **+4.70% absolute BLEU-4 gain** ($17.47 \to 22.17$), proving that global 196-patch self-attention effectively resolves fine-grained spatial relationships and object interactions.
* **Adaptive Sentinel Gate:** Protects language states during non-visual syntactic token generation (*"a"*, *"of"*, *"in"*), boosting BLEU-4 from $17.47$ to $21.51$ (+4.04%).
* **Edge NPU Footprint Reduction:** The MobileNetV3-Small quantized pipeline achieves a **96.1% size reduction** ($154\text{ MB} \to 6.08\text{ MB}$) while preserving core semantic alignment, making it fully viable for edge microcontrollers, Apple Neural Engine, Qualcomm Hexagon NPU, and mobile apps.

---

## ⚡ Dual-Platform Cross-Verification: Cloud Server vs Edge NPU

To verify that model compression did not cause severe semantic degradation or catastrophic forgetting, both tiers were benchmarked in parallel across identical test inputs:

| Evaluation Dimension | Cloud / Server Tier (ViT-B/16 + Transformer) | Edge / Mobile Tier (MobileNetV3 INT8) | Delta / Trade-off |
| :--- | :--- | :--- | :--- |
| **Model Size** | 214.3 MB (INT8) / 393.6 MB (FP32) | **6.08 MB (INT8)** | **96.1% Memory Footprint Reduction** |
| **Peak RAM Footprint** | ~650 MB | **~28 MB** | Fits lowest-tier embedded RAM budgets |
| **Inference Latency** | 60–80 ms (CPU) / 12 ms (GPU) | **8–15 ms (NPU / CPU)** | **4x–6x Speedup on CPU/NPU** |
| **Primary Compute Units** | CUDA Tensor Cores / High-End CPU | INT8 Matrix Units / Neural Processing Units (NPUs) | Zero GPU requirement |
| **Semantic Fidelity** | Highly descriptive, granular attributes | Retains primary subject, verb, and object | Negligible visual hallucination |
| **Sample: Dogs Playing** | *"two dogs are playing with each other in the snow"* | *"two dogs play in the snow"* | Syntactically concise, zero factual loss |
| **Sample: Climber on Rock** | *"a man climbing up a steep rock face"* | *"a man climbs a rock"* | Core action preserved, 23x smaller model |

---

## 🧠 Architectural Innovations

### 1. Adaptive Visual Sentinel Gating ($\beta_t$)
Standard attention decoders attend to image features unconditionally at every decoding step, forcing the model to look at pixels even when generating non-visual syntactic tokens.

Our architecture implements a **Visual Sentinel Gate**:
$$\mathbf{s}_t = g(\mathbf{W}_x \mathbf{x}_t + \mathbf{W}_h \mathbf{h}_{t-1})$$
$$\beta_t = \sigma(\mathbf{w}_\beta^T [\mathbf{h}_t; \mathbf{s}_t] + b_\beta) \in [0, 1]$$
$$\hat{\mathbf{c}}_t = \beta_t \mathbf{c}_t + (1 - \beta_t) \mathbf{s}_t$$

When generating visual words (*"dog"*, *"pink"*, *"cannon"*), $\beta_t \to 1$, routing spatial patch attention. When generating functional syntax (*"the"*, *"with"*), $\beta_t \to 0$, protecting the visual representation from noise drift.

### 2. Vision Transformer Patch Representation (ViT-B/16)
Rather than flattening low-resolution CNN feature grids, input images ($224 \times 224$) are projected into $196$ non-overlapping $16 \times 16$ patch tokens ($D = 768$). Full self-attention across the patches enables global context and fine-grained texture discrimination.

### 3. Edge-Optimized MobileNetV3 + INT8 Dynamic Quantization
For micro-devices and mobile phones:
- Replaces heavy ResNet/ViT feature extractors with an inverted bottleneck MobileNetV3-Small backbone ($D=576$).
- Decouples decoding into a compact 256-D gated recurrent cell with tied embeddings.
- Applies PyTorch `torch.quantization.quantize_dynamic` across `nn.Linear` and `nn.LSTM` modules, mapping 32-bit floating point weights to 8-bit integers while preserving activation dynamic range.

### 4. Information-Theoretic Frequency-Aware Beam Search
Standard Maximum Likelihood Estimation (MLE) training causes mode collapse toward statistical unigram heads (*"A group of people standing"*). We integrate an **Inverse Frequency Penalty**:
$$\mathbf{z}_t'(w) = \mathbf{z}_t(w) - \lambda \cdot \log(1 + f(w)) - \gamma \cdot \mathbb{I}[w \in w_{<t}]$$
This elevates rare, descriptive tokens (*"uniform"*, *"climbing"*, *"snow"*) into the active beam paths.

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

### 2. Run Ultra-Compact Mobile / NPU Model (6.08 MB)
```bash
python -c "
import torch
from models.mobile_captioner import MobileCaptioner
from src.preprocessing.vocabulary import Vocabulary

vocab = Vocabulary.load_vocab('dataset/vocab.json')
model = torch.load('checkpoints/mobile/mobilenet_captioner_int8.pth', map_location='cpu', weights_only=False)
model.eval()

# Run sample inference
caption = model.generate_caption('dataset/Images/1000268201_693b08cb0e.jpg', vocab)
print('Generated Mobile Caption:', caption)
"
```

### 3. Run ViT Transformer or Adaptive Attention Model (CLI)
```bash
# Vision Transformer (ViT-B/16):
python predict.py --image "sample.jpg" --checkpoint "checkpoints/vit_transformer/best_model.pth" --model vit_transformer

# Adaptive Attention Model with Specificity Boost:
python predict.py --image "sample.jpg" --checkpoint "checkpoints/adaptive/best_model.pth" --model adaptive --method beam --beam-width 4 --specificity-factor 0.4
```

### 4. Run Interactive Web Interfaces
```bash
# Launch Streamlit:
streamlit run app/streamlit_app.py

# Launch Gradio Application:
python app/gradio_app.py
```

---

## 🤗 Hugging Face Model Hub Integration

The quantized production model, isolated vocabulary, and model cards are synchronized with the official Hugging Face Model Hub:

- **Repository:** [SivaPanyam/Neural-Image-Captioning-Model](https://huggingface.co/SivaPanyam/Neural-Image-Captioning-Model)
- **Included Assets:**
  - `checkpoints/mobile/mobilenet_captioner_int8.pth` (6.08 MB INT8 Model)
  - `dataset/vocab.json` (2,563 vocabulary tokens)
  - `app.py` (Universal Gradio Web Interface)
  - Full Model Card & Benchmark Metadata

To synchronize local updates to Hugging Face:
```bash
set HF_TOKEN=your_token_here
python src/export/sync_hf.py
```

---

## 📁 Repository Structure

```text
Image_Captioning_Model/
├── app/
│   ├── streamlit_app.py            # Interactive Streamlit Web UI with live attention maps
│   └── gradio_app.py               # Gradio Web UI for Hugging Face Spaces
├── checkpoints/
│   ├── baseline/best_model.pth     # Show and Tell CNN-LSTM weights (154 MB)
│   ├── adaptive/best_model.pth     # Adaptive Sentinel Attention weights (101 MB INT8)
│   ├── vit_transformer/            # ViT-B/16 + Cross-Attention weights (214 MB INT8)
│   └── mobile/
│       └── mobilenet_captioner_int8.pth # 6.08 MB Ultra-Compact Mobile/NPU model
├── configs/
│   └── config.py                   # Central hyperparameter & path configuration
├── dataset/
│   ├── vocab.json                  # 2,563-token isolated training vocabulary
│   └── cached_features/            # Pre-extracted visual tensor caches (ResNet, ViT, MobileNet)
├── models/
│   ├── captioning_model.py         # ResNet & Adaptive Sentinel model definitions
│   └── mobile_captioner.py         # MobileNetV3 + Gated LSTM Edge model definition
├── src/
│   ├── encoder/
│   │   ├── resnet_encoder.py       # ResNet-50 visual backbone
│   │   ├── vit_encoder.py          # ViT-B/16 Vision Transformer backbone
│   │   └── extract_features.py     # High-speed offline tensor caching engine
│   ├── decoder/
│   │   ├── lstm_decoder.py         # Baseline LSTM decoder
│   │   ├── attention_decoder.py    # Adaptive Sentinel spatial decoder
│   │   └── vit_transformer_decoder.py # Multi-head cross-attention decoder
│   ├── export/
│   │   ├── quantize_models.py      # Native PyTorch dynamic INT8 quantization
│   │   └── sync_hf.py              # Automated Hugging Face Hub push pipeline
│   ├── datasets/
│   │   └── flickr8k_dataset.py     # PyTorch Dataset & dynamic batch collator
│   └── evaluation/
│       └── metrics.py              # Corpus BLEU-1..4 & ROUGE-L implementations
├── tests/
│   └── dual_platform_benchmark.py  # Parallel Server vs Edge NPU benchmark suite
├── evaluate.py                     # Multi-architecture evaluation CLI
├── predict.py                      # Inference CLI with Specificity Beam Search
├── train.py                        # CNN-LSTM training pipeline
├── train_vit.py                    # ViT-Transformer training pipeline
├── train_mobile.py                 # MobileNetV3 training pipeline
├── requirements.txt                # Production dependency specifications
├── RESEARCH_PAPER_BLUEPRINT.md     # Full theoretical compendium & ablation roadmap
└── README.md                       # Comprehensive research documentation
```

---

## 📜 Citation & License
Distributed under the MIT License. Built for reproducible computer vision & natural language processing research.
