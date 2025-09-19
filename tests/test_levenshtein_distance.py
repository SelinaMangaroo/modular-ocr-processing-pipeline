import os
import logging
import pytest
import Levenshtein
from tests.test_utils import *

initialize_test_logger()

# This decorator tells pytest to run the test multiple times, once for each value of base_name.
@pytest.mark.parametrize("base_name", [
    name for name in os.listdir(output_dir)
    if os.path.isdir(os.path.join(output_dir, name))
])

def test_levenshtein_distance_change(base_name):
    """
    Compares raw Textract output vs corrected text using Levenshtein ratio.
    Ensures ChatGPT corrections are within reasonable change limits.
    """
    
    raw_path = os.path.join(output_dir, base_name, f"{base_name}.raw.txt")
    corrected_path = os.path.join(output_dir, base_name, f"{base_name}.corrected.txt")
    
    logging.info(f"Running Levenshtein test for: {base_name}")

    assert os.path.exists(raw_path), f"Missing raw file: {raw_path}"
    assert os.path.exists(corrected_path), f"Missing corrected file: {corrected_path}"

    with open(raw_path, 'r', encoding='utf-8') as rf:
        raw_text = rf.read()
    with open(corrected_path, 'r', encoding='utf-8') as cf:
        corrected_text = cf.read()
    
    # Normalize whitespace before comparing
    raw_text = normalize_whitespace(raw_text)
    corrected_text = normalize_whitespace(corrected_text)

    # Levenshtein ratio gives similarity score between 0.0 and 1.0
    # 1.0 means identical, 0.0 means completely different.
    similarity = Levenshtein.ratio(raw_text, corrected_text)
    difference_ratio = 1 - similarity  # How much was changed (as %)
    
    logging.info(f"[{base_name}] Similarity: {similarity:.2%} | Difference: {difference_ratio:.2%} | Allowed Diff: {levenshtein_max_diff_ratio:.2%}")

    if difference_ratio > levenshtein_max_diff_ratio:
        logging.warning(f"[{base_name}] FAILED: Difference {difference_ratio:.2%} exceeds allowed {levenshtein_max_diff_ratio:.2%}")

    # Assert that the difference is within the allowed tolerance. If the Levenshtein ratio is too low, the test fails.
    assert difference_ratio <= levenshtein_max_diff_ratio, (
        f"[{base_name}] Levenshtein difference too high: {difference_ratio:.2%} (Allowed: {levenshtein_max_diff_ratio:.2%})"
    )
