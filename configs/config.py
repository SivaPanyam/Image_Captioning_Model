from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset
DATASET_DIR = PROJECT_ROOT / "dataset" / "Flickr8k"

IMAGES_DIR = DATASET_DIR / "Images" / "Images"

CAPTIONS_FILE = DATASET_DIR / "captions.txt"

TRAIN_FILE = DATASET_DIR / "train.txt"

VAL_FILE = DATASET_DIR / "val.txt"

TEST_FILE = DATASET_DIR / "test.txt"

# Image Parameters
IMAGE_SIZE = 224
CHANNELS = 3

# Training
BATCH_SIZE = 32
NUM_WORKERS = 4

# Vocabulary
MIN_WORD_FREQUENCY = 5
MAX_CAPTION_LENGTH = 30

# Model
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
FEATURE_DIM = 2048