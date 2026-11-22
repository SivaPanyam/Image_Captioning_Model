# Deep Learning Image Captioning System

An end-to-end Image Captioning research and engineering pipeline implemented in PyTorch using the Flickr8k dataset.

## System Architecture Pipeline
1. **Visual Encoder:** Pretrained ResNet50 (Global Average Pooling $\to$ 2048-dim feature representation).
2. **Text Decoder:** Word Embedding $\to$ Autoregressive LSTM with Teacher Forcing.
3. **Advanced Iterations (Planned):** Bahdanau/Luong Attention, Vision Transformers (ViT), and Cross-Modal Vision-Language Models.

## Project Structure
```text
Captioning_model/
├── configs/             # Centralized hyperparameter and path definitions
├── dataset/             # Flickr8k dataset, image splits (train, val, test)
├── src/
│   ├── preprocessing/   # Dataset verification, caption loading, tokenization, vocab
│   ├── utils/           # Path verification and helper utilities
│   ├── datasets/        # PyTorch Dataset and custom Collate DataLoader
│   ├── encoder/         # ResNet50 visual feature extractor
│   ├── decoder/         # LSTM language decoder
│   ├── training/        # Optimization, checkpointing, loss functions
│   ├── evaluation/      # BLEU, METEOR, CIDEr, ROUGE evaluation metrics
│   └── inference/       # Greedy search, beam search, temperature sampling
├── tests/               # Pytest suite for preprocessing, models, and data pipelines
├── requirements.txt     # Environment dependencies
├── train.py             # Model training entrypoint
└── predict.py           # Single-image caption generation entrypoint
```

## Current Status
- **Stage 0–3:** Environment setup, deep learning theory, and Flickr8k dataset partitioning verified (8,091 images, 40,455 captions; 6,000 train / 1,000 val / 1,091 test).
- **Stage 4:** Caption Preprocessing & Vocabulary Engine (In Progress).
