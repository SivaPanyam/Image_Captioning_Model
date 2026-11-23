# Comprehensive Technical Documentation & Research Compendium
## Project: Adaptive Multi-Modal Image Captioning System
**Institution / Target:** Research Publication & University Defense  
**Author / Lead:** Siva Panyam  
**Target Benchmarks:** Flickr8k (Primary Verification) $\to$ MS-COCO (Scalability)  
**Hugging Face Hub:** [SivaPanyam/Neural-Image-Captioning-Model](https://huggingface.co/SivaPanyam/Neural-Image-Captioning-Model)

---

## 1. Executive Research Summary
This project designs and evaluates end-to-end deep learning architectures for automatic image description generation. Moving beyond legacy Show-and-Tell architectures, this initiative addresses key vision-language failure modes:
1. **Information Collapse in Global Average Pooling:** Loss of spatial coordinate grounding in vanilla CNNs.
2. **Exposure Bias in Teacher-Forcing Decoders:** Discrepancy between training with ground truth context and inference generation.
3. **Visual Drift & Mode Collapse:** Language models overpowering weak visual conditioning, defaulting to statistically frequent unigrams.
4. **Edge & NPU Deployment Bottlenecks:** Heavy parameter counts (> 100 MB) preventing real-time mobile and edge inference.

The research proposes:
- **Adaptive Visual Sentinel Cross-Attention** to dynamically route attention between image regions and language states.
- **Vision Transformer (ViT-B/16)** patch self-attention to capture complex inter-object visual semantics.
- **Ultra-Compact INT8 Dynamic Quantization (MobileNetV3)** to compress models to $6.08\text{ MB}$ for real-time mobile NPU execution without catastrophic semantic degradation.

---

## 2. Empirical Benchmark Matrix (Flickr8k Test Split: 1,091 Images)

Evaluated under multi-reference ground truth (5 references per image) with corpus-level $n$-gram precision and recall:

| Architecture | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | Checkpoint Size | Quantization | Latency | Target Deployment |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline (ResNet-50 + LSTM)** | 56.88 | 39.75 | 26.46 | 17.47 | 42.90 | 154 MB | FP32 | ~35 ms | Server Reference |
| **Adaptive Sentinel (ResNet-50)** | 61.75 | 44.22 | 30.71 | 21.51 | 45.67 | 101 MB | INT8 | ~18 ms | High-Precision API |
| **Vision Transformer (ViT-B/16)** | **61.52** | **44.69** | **31.84** | **`22.17`** | **45.62** | 214 MB | INT8 | ~65 ms | Maximum Accuracy Cloud |
| **Ultra-Compact MobileNetV3** | 53.60 | 36.40 | 23.50 | 14.80 | 40.10 | **`6.08 MB`** | **INT8** | **~8-15 ms** | **Edge / NPU Mobile** |

---

## 3. Dual-Platform Architecture & Latency Profile

| Dimension | Cloud / Desktop (ViT-B/16 + Transformer) | Edge / Mobile (MobileNetV3 INT8) | Delta / Trade-off |
|---|---|---|---|
| **Model Size** | 214.3 MB (INT8) / 393.6 MB (FP32) | **6.08 MB (INT8)** | **96.1% Memory Reduction** |
| **Peak RAM** | ~650 MB | **~28 MB** | Ultra-low memory footprint |
| **Inference Latency** | 60–80 ms (CPU) / 12 ms (GPU) | **8–15 ms (NPU / CPU)** | **4x–6x Speedup on CPU/NPU** |
| **Compute Engine** | CUDA Tensor Cores | INT8 Matrix / NPU Engines | Runs locally on phones |
| **Semantic Preservation** | High granularity (*"two dogs playing with each other in snow"*) | Core semantics (*"two dogs play in the snow"*) | Minimal semantic loss |

---

## 4. System Architecture & Component Inventory

### Stage 0–3: Environment & Data Partitions
* **Hardware Profile:** Intel Core i5-12450H (8 Cores), 16GB RAM, NVIDIA RTX 2050 Laptop GPU (4GB VRAM).
* **Dataset Partitioning (Seed = 42):**
  * Total images: $8,091$ | Total captions: $40,455$ ($5$ captions per image).
  * Training Split: $6,000$ images ($30,000$ captions).
  * Validation Split: $1,000$ images ($5,000$ captions).
  * Test Split: $1,091$ images ($5,455$ captions).
  * Mutual Exclusion: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.

### Stage 4: Preprocessing & Vocabulary Engine
* **Cleaning Pipeline (`src/preprocessing/caption_cleaner.py`):**
  * Case folding (lower), hyphen expansion (`-` $\to$ space), regex non-alpha stripping (`[^a-z\s]`), space normalization.
* **Special Tokens (`src/preprocessing/vocabulary.py`):**
  $$\langle\text{PAD}\rangle = 0, \quad \langle\text{START}\rangle = 1, \quad \langle\text{END}\rangle = 2, \quad \langle\text{UNK}\rangle = 3$$
* **Vocabulary Isolation:**
  * Vocabulary constructed strictly on training captions with $\text{min\_freq} = 5$.
  * Size: **$2,563$ unique tokens** saved at `dataset/vocab.json`.

### Stage 5: Visual Backbones
1. **ResNet-50 (`src/encoder/resnet_encoder.py`):** Global feature vector ($2048$) and $14 \times 14$ spatial grid map ($196 \times 2048$).
2. **Vision Transformer (`src/encoder/vit_encoder.py`):** Pretrained ViT-B/16 generating $196$ patch tokens ($D=768$).
3. **MobileNetV3-Small (`models/mobile_captioner.py`):** Inverted residual feature extractor ($D=576$) with INT8 dynamic quantization.

---

## 5. Theoretical Innovations

### Visual Sentinel Gating ($\beta_t$)
$$\mathbf{s}_t = g(\mathbf{W}_x \mathbf{x}_t + \mathbf{W}_h \mathbf{h}_{t-1})$$
$$\beta_t = \sigma(\mathbf{w}_\beta^T [\mathbf{h}_t; \mathbf{s}_t] + b_\beta) \in [0, 1]$$
$$\hat{\mathbf{c}}_t = \beta_t \mathbf{c}_t + (1 - \beta_t) \mathbf{s}_t$$

### Frequency-Aware Specificity Penalty
$$\mathbf{z}_t'(w) = \mathbf{z}_t(w) - \lambda \cdot \log(1 + f(w)) - \gamma \cdot \mathbb{I}[w \in w_{<t}]$$

---

## 6. Artifact & Deployment Inventory
- **GitHub Repository:** `https://github.com/SivaPanyam/Image_Captioning_Model.git`
- **Hugging Face Model Hub:** `https://huggingface.co/SivaPanyam/Neural-Image-Captioning-Model`
- **Quantized Mobile Model:** `checkpoints/mobile/mobilenet_captioner_int8.pth` ($6.08\text{ MB}$)
