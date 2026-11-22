# Comprehensive Technical Documentation & Research Compendium
## Project: Adaptive Multi-Modal Image Captioning System
**Institution / Target:** Research Publication & University Defense  
**Author / Lead:** Siva Panyam  
**Target Benchmarks:** Flickr8k (Primary Verification) $\to$ MS-COCO (Scalability)

---

## 1. Executive Research Summary
This project designs and evaluates end-to-end deep learning architectures for automatic image description generation. Moving beyond legacy Show-and-Tell architectures, this initiative addresses key vision-language failure modes:
1. **Information Collapse in Global Average Pooling:** Loss of spatial coordinate grounding.
2. **Exposure Bias in Teacher-Forcing Decoders:** Train/inference distribution discrepancy.
3. **Visual Drift & Object Hallucination:** Language models overpowering weak visual conditioning.

The research proposes **Adaptive Multi-Modal Semantic Routing (AMSR)** coupled with **Self-Critical Policy Gradient Fine-Tuning** to achieve state-of-the-art caption fidelity, high CIDEr/BLEU scores, and low hallucination (CHAIR).

---

## 2. System Architecture & Component Inventory

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
* **Sequence Utilities (`src/preprocessing/sequence_utils.py`):**
  * Teacher-forcing shifted pairs:
    $$\mathbf{x}_{\text{input}} = [\langle\text{START}\rangle, w_1, w_2, \dots, w_T], \quad \mathbf{y}_{\text{target}} = [w_1, w_2, \dots, w_T, \langle\text{END}\rangle]$$

### Stage 5: Visual Backbone (`src/encoder/resnet_encoder.py`)
* Pretrained ResNet-50 backbone with stripped linear classification head.
* Dual output capability:
  1. Global Representation: $\mathbf{v} \in \mathbb{R}^{B \times 2048}$
  2. Spatial Grid Map: $\mathbf{V}_{\text{spatial}} \in \mathbb{R}^{B \times 2048 \times 14 \times 14} \implies 196$ localized visual tokens.

---

## 3. Comparative Research Ablation Roadmap

| Model Variant | Visual Feature Extractor | Sequence Decoder | Objective Function | Key Theoretical Novelty | Target CIDEr |
|---|---|---|---|---|---|
| **Model 0 (Baseline)** | ResNet-50 Global ($2048$) | Vanilla LSTM | Cross-Entropy (MLE) | Baseline Benchmark (Show & Tell) | $\approx 45.0$ |
| **Model 1 (Spatial)** | ResNet-50 Grid ($196 \times 2048$) | Bahdanau Attention LSTM | Cross-Entropy (MLE) | Spatial visual alignment | $\approx 62.0$ |
| **Model 2 (Transformer)** | ResNet-50 Grid | Multi-Head Cross-Attention | Label-Smoothed CE | Self-attention parallelization | $\approx 74.0$ |
| **Model 3 (Proposed Novelty)** | Hybrid Spatial + Contrastive Semantics | Dual-Gated Visual Sentinel (AMSR) | MLE + SCST (RL Reward) | Gated cross-modal routing + metric optimization | **$> 90.0$** |

---

## 4. Subagent Research Engine
A dedicated research agent (`captioning-research-strategist`) is registered to continuously formulate hypotheses, design novel loss formulations, and review training dynamics against CVPR/ACL literature.
