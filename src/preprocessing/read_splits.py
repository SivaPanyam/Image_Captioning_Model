from configs.config import *


def load_split(file_path):

    with open(file_path, "r") as file:

        images = file.read().splitlines()

    return images


if __name__ == "__main__":

    train = load_split(TRAIN_FILE)

    val = load_split(VAL_FILE)

    test = load_split(TEST_FILE)

    print("Training Images :", len(train))

    print("Validation Images :", len(val))

    print("Testing Images :", len(test))