from pathlib import Path
import random

from configs.config import CAPTIONS_FILE, TRAIN_FILE, VAL_FILE, TEST_FILE


# ======================================================
# Configuration
# ======================================================

TRAIN_SIZE = 6000
VAL_SIZE = 1000
RANDOM_SEED = 42


# ======================================================
# Load unique image names
# ======================================================

def load_image_names():

    image_names = []

    with open(CAPTIONS_FILE, "r", encoding="utf-8") as file:

        # Skip CSV header
        next(file)

        for line in file:

            image_name, _ = line.strip().split(",", 1)

            image_names.append(image_name)

    # Remove duplicates
    image_names = sorted(set(image_names))

    return image_names


# ======================================================
# Create train / validation / test splits
# ======================================================

def create_splits():

    image_names = load_image_names()

    total_images = len(image_names)

    print(f"Total images: {total_images}")

    if total_images != 8091:

        raise ValueError(
            f"Expected 8091 images, but found {total_images}"
        )

    random.seed(RANDOM_SEED)

    random.shuffle(image_names)

    train_images = image_names[:TRAIN_SIZE]

    val_images = image_names[
        TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE
    ]

    test_images = image_names[
        TRAIN_SIZE + VAL_SIZE:
    ]

    # Write files
    TRAIN_FILE.write_text(
        "\n".join(train_images),
        encoding="utf-8"
    )

    VAL_FILE.write_text(
        "\n".join(val_images),
        encoding="utf-8"
    )

    TEST_FILE.write_text(
        "\n".join(test_images),
        encoding="utf-8"
    )

    print()
    print("Splits created successfully.")
    print(f"Train      : {len(train_images)}")
    print(f"Validation : {len(val_images)}")
    print(f"Test       : {len(test_images)}")


if __name__ == "__main__":

    create_splits()