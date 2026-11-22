from configs.config import *

from src.utils.file_utils import verify_path


def verify_dataset():

    print("=" * 60)
    print("Flickr8k Dataset Verification")
    print("=" * 60)

    # --------------------------------------------------
    # Check paths
    # --------------------------------------------------

    verify_path(DATASET_DIR)
    verify_path(IMAGES_DIR)
    verify_path(CAPTIONS_FILE)
    verify_path(TRAIN_FILE)
    verify_path(VAL_FILE)
    verify_path(TEST_FILE)

    print("\nRequired paths found.")

    # --------------------------------------------------
    # Count images
    # --------------------------------------------------

    image_files = list(IMAGES_DIR.glob("*.jpg"))

    total_images = len(image_files)

    print(f"Images found      : {total_images}")

    if total_images != 8091:

        raise RuntimeError(
            f"Expected 8091 images, found {total_images}"
        )

    # --------------------------------------------------
    # Count captions
    # --------------------------------------------------

    with open(CAPTIONS_FILE, "r", encoding="utf-8") as file:

        caption_lines = file.readlines()

    total_captions = len(caption_lines) - 1

    print(f"Captions found    : {total_captions}")

    if total_captions != 40455:

        raise RuntimeError(
            f"Expected 40455 captions, found {total_captions}"
        )

    # --------------------------------------------------
    # Count splits
    # --------------------------------------------------

    train_count = len(
        TRAIN_FILE.read_text(encoding="utf-8")
        .splitlines()
    )

    val_count = len(
        VAL_FILE.read_text(encoding="utf-8")
        .splitlines()
    )

    test_count = len(
        TEST_FILE.read_text(encoding="utf-8")
        .splitlines()
    )

    print(f"Training images  : {train_count}")
    print(f"Validation images: {val_count}")
    print(f"Test images      : {test_count}")

    # --------------------------------------------------
    # Verify split count
    # --------------------------------------------------

    total_split_images = (
        train_count +
        val_count +
        test_count
    )

    if total_split_images != total_images:

        raise RuntimeError(
            "Train/validation/test split count "
            "does not equal total image count."
        )

    print()
    print("=" * 60)
    print("DATASET VERIFICATION PASSED")
    print("=" * 60)


if __name__ == "__main__":

    verify_dataset()