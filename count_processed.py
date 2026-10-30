"""
Counts how many source JPGs have been run through the pipeline so far,
under a given output directory. Each processed jpg produces one
.corrected.txt file, so counting those gives the jpg count.

Just set DIR_PATH below and run:

    python count_processed.py
"""

import os

DIR_PATH = "/data/modular-ocr-processing-pipeline/Official Gemini Output"


def main():
    jpg_count = 0

    for _, _, filenames in os.walk(DIR_PATH):
        for f in filenames:
            if f.endswith(".corrected.txt"):
                jpg_count += 1

    print(f"Directory: {DIR_PATH}")
    print(f"JPGs run so far: {jpg_count}")


if __name__ == "__main__":
    main()
