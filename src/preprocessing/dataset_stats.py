from collections import Counter

from configs.config import *


image_counter = Counter()

caption_count = 0


with open(CAPTIONS_FILE, encoding="utf-8") as file:

    next(file)

    for line in file:

        image_name, caption = line.strip().split(",", 1)

        image_counter[image_name] += 1

        caption_count += 1


print()

print("Total Images :", len(image_counter))

print("Total Captions :", caption_count)

print("Average Captions/Image :", caption_count / len(image_counter))