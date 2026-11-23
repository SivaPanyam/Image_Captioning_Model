"""
Production Gradio Web Application for Hugging Face Spaces Deployment.
Supports:
- Drag-and-drop Image Upload
- Architecture Selector: Baseline (Show & Tell), Adaptive Attention, Vision Transformer (ViT-B/16)
- Decoding Strategy: Greedy Search vs Beam Search (Width 1-8)
- Real-time Specificity Boost Slider
- Standalone Hugging Face Spaces compatibility
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
from PIL import Image
import torch

from configs.config import PROJECT_ROOT, IMAGES_DIR
from src.preprocessing.vocabulary import Vocabulary, START_IDX, END_IDX
from src.encoder.feature_extractor import get_image_transforms
from predict import load_model, beam_search_decode, greedy_decode
from src.decoder.vit_transformer_decoder import ViTTransformerCaptioningModel

# 1. Device and Vocabulary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_path = PROJECT_ROOT / "dataset" / "vocab.json"
vocab = Vocabulary.load(vocab_path)
vocab_size = len(vocab)
transform = get_image_transforms(is_training=False)

# 2. Lazy Model Cache
models_cache = {}


def get_model(model_name: str):
    if model_name in models_cache:
        return models_cache[model_name]

    if model_name == "Baseline (Show & Tell)":
        ckpt_path = PROJECT_ROOT / "checkpoints" / "baseline" / "best_model.pth"
        m = load_model(ckpt_path, "baseline", vocab_size, device)
    elif model_name == "Adaptive Attention (ResNet + Gate)":
        ckpt_path = PROJECT_ROOT / "checkpoints" / "adaptive" / "best_model.pth"
        m = load_model(ckpt_path, "adaptive", vocab_size, device)
    else: # Vision Transformer
        ckpt_path = PROJECT_ROOT / "checkpoints" / "vit_transformer" / "best_model.pth"
        m = ViTTransformerCaptioningModel(vocab_size=vocab_size).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()

    models_cache[model_name] = m
    return m


def generate_caption(
    image: Image.Image,
    model_choice: str,
    strategy: str,
    beam_width: int,
    specificity_factor: float,
    repetition_penalty: float,
    min_length: int,
    max_length: int,
):
    if image is None:
        return "Please upload or select an image."

    img_rgb = image.convert("RGB")
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)
    model = get_model(model_choice)

    with torch.no_grad():
        if model_choice == "Vision Transformer (ViT-B/16 + Cross-Attn)":
            # ViT Autoregressive decoding
            patches, _ = model.encoder(img_tensor)
            memory = model.visual_proj(patches)

            generated = [START_IDX]
            for step in range(max_length):
                tgt_in = torch.tensor([generated], dtype=torch.long, device=device)
                seq_len = tgt_in.size(1)
                tgt_mask = model.generate_square_subsequent_mask(seq_len, device)
                tgt_emb = model.embedding(tgt_in) * (model.d_model ** 0.5)
                tgt_emb = model.pos_encoder(tgt_emb)

                for layer in model.layers:
                    tgt_emb = layer(tgt_emb, memory, tgt_mask=tgt_mask)
                tgt_emb = model.final_norm(tgt_emb)
                logits = model.fc_out(tgt_emb[:, -1])

                # Repetition penalty
                if repetition_penalty > 1.0:
                    for prev_tok in set(generated[1:]):
                        logits[0, prev_tok] /= repetition_penalty

                if step < min_length:
                    logits[0, END_IDX] = -1e9
                logits[0, 3] = -1e9 # Suppress UNK

                nxt_id = logits.argmax(dim=-1).item()
                if nxt_id == END_IDX:
                    break
                generated.append(nxt_id)

            words = [vocab.get_word(i) for i in generated[1:]]
            return " ".join(words).capitalize() + "."
        else:
            model_key = "baseline" if "Baseline" in model_choice else "adaptive"
            if strategy == "Greedy Search":
                cap, _ = greedy_decode(model, img_tensor, vocab, max_length, model_key, device)
            else:
                cap, _ = beam_search_decode(
                    model=model,
                    image_tensor=img_tensor,
                    vocab=vocab,
                    beam_width=beam_width,
                    max_length=max_length,
                    specificity_factor=specificity_factor,
                    repetition_penalty=repetition_penalty,
                    min_length=min_length,
                    model_type=model_key,
                    device=device,
                )
            return cap.capitalize() + "."


# 3. Gradio Interface Construction
sample_dir = PROJECT_ROOT / "dataset" / "Flickr8k" / "Images" / "Images"
sample_paths = [str(p) for p in list(sample_dir.glob("*.jpg"))[:4]] if sample_dir.exists() else []

with gr.Blocks(title="Neural Image Captioning Lab", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📷 Neural Image Captioning Research Platform")
    gr.Markdown(
        "Compare 3 Deep Learning vision-language architectures: "
        "**CNN-LSTM Baseline**, **Adaptive Spatial Attention with Sentinel Gating**, and **Vision Transformer (ViT-B/16) + Cross-Attention**."
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Input Image")
            model_select = gr.Dropdown(
                choices=[
                    "Vision Transformer (ViT-B/16 + Cross-Attn)",
                    "Adaptive Attention (ResNet + Gate)",
                    "Baseline (Show & Tell)",
                ],
                value="Vision Transformer (ViT-B/16 + Cross-Attn)",
                label="Architecture",
            )
            with gr.Accordion("Advanced Decoding Options", open=False):
                strategy = gr.Radio(["Beam Search", "Greedy Search"], value="Beam Search", label="Strategy")
                beam_width = gr.Slider(1, 8, value=4, step=1, label="Beam Width")
                specificity = gr.Slider(0.0, 2.0, value=0.4, step=0.1, label="Specificity Boost")
                repetition_penalty = gr.Slider(1.0, 2.0, value=1.2, step=0.1, label="Repetition Penalty")
                min_len = gr.Slider(0, 15, value=6, step=1, label="Min Length")
                max_len = gr.Slider(10, 40, value=25, step=1, label="Max Tokens")

            btn = gr.Button("Generate Caption 🚀", variant="primary")

        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Generated Caption", lines=3, text_align="left")
            gr.Markdown(
                "### 📊 Benchmark Matrix (Flickr8k Test Set)\n"
                "| Architecture | BLEU-1 | BLEU-4 | ROUGE-L |\n"
                "| :--- | :---: | :---: | :---: |\n"
                "| **Baseline (CNN-LSTM)** | 56.88 | 17.47 | 42.90 |\n"
                "| **Adaptive Attention (LSTM)** | 61.75 | 21.51 | 45.67 |\n"
                "| **Vision Transformer (ViT-B/16)** | **61.52** | **22.17** | **45.62** |"
            )

    if sample_paths:
        gr.Examples(examples=sample_paths, inputs=input_img, label="Quick Test Samples")

    btn.click(
        fn=generate_caption,
        inputs=[input_img, model_select, strategy, beam_width, specificity, repetition_penalty, min_len, max_len],
        outputs=output_text,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
