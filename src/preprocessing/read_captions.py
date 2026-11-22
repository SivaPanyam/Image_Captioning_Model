from configs.config import *


with open(CAPTIONS_FILE, "r", encoding="utf-8") as file:

    for i, line in enumerate(file):

        print(line.strip())

        if i == 10:

            break