"""
CLI utility to build and serialize the training vocabulary from Flickr8k.
Guarantees vocabulary is constructed strictly on training captions.
"""

from pathlib import Path
from configs.config import (
    CAPTIONS_FILE,
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
    MIN_WORD_FREQUENCY,
    PROJECT_ROOT,
)
from src.preprocessing.caption_loader import (
    load_captions,
    load_split_captions,
    load_split_image_names,
)
from src.preprocessing.caption_cleaner import SimpleTokenizer
from src.preprocessing.vocabulary import Vocabulary


def build_and_save_vocab():
    print("=" * 60)
    print("Building Training-Only Vocabulary")
    print("=" * 60)

    # 1. Load full captions mapping
    all_captions = load_captions(CAPTIONS_FILE)
    print(f"Loaded {len(all_captions)} unique images from {CAPTIONS_FILE.name}")

    # 2. Strict anti-leakage isolation: only train split
    train_images = load_split_image_names(TRAIN_FILE)
    val_images = load_split_image_names(VAL_FILE)
    test_images = load_split_image_names(TEST_FILE)

    assert len(train_images & val_images) == 0, "Data leakage detected: train & val overlap"
    assert len(train_images & test_images) == 0, "Data leakage detected: train & test overlap"
    assert len(val_images & test_images) == 0, "Data leakage detected: val & test overlap"
    print(f"Data leakage check passed: 0 overlap between splits.")

    train_captions_dict = load_split_captions(TRAIN_FILE, all_captions)
    print(f"Training split images: {len(train_captions_dict)}")

    # 3. Tokenize all training captions
    tokenizer = SimpleTokenizer()
    tokenized_train_captions = []
    total_caption_count = 0

    for img, captions in train_captions_dict.items():
        for cap in captions:
            tokens = tokenizer.tokenize(cap)
            tokenized_train_captions.append(tokens)
            total_caption_count += 1

    print(f"Tokenized {total_caption_count} training captions.")

    # 4. Build vocabulary
    vocab = Vocabulary()
    vocab.build_vocabulary(tokenized_train_captions, min_freq=MIN_WORD_FREQUENCY)

    print(f"Vocabulary Size (min_freq={MIN_WORD_FREQUENCY}): {len(vocab)}")
    print(f"Special Tokens: <PAD>={vocab.word2idx['<PAD>']}, <START>={vocab.word2idx['<START>']}, <END>={vocab.word2idx['<END>']}, <UNK>={vocab.word2idx['<UNK>']}")

    # 5. Save to disk
    vocab_output_path = PROJECT_ROOT / "dataset" / "vocab.json"
    vocab.save(vocab_output_path)
    print(f"Saved vocabulary to: {vocab_output_path}")

    # 6. Verify round-trip load
    loaded_vocab = Vocabulary.load(vocab_output_path)
    assert len(loaded_vocab) == len(vocab), "Serialized vocab length mismatch"
    print("Vocabulary round-trip serialization verified successfully.")
    print("=" * 60)


if __name__ == "__main__":
    build_and_save_vocab()
