import os
import logging
import pytest
from tests.test_utils import *

initialize_test_logger()

# This decorator tells pytest to run the test multiple times, once for each value of base_name.
@pytest.mark.parametrize("base_name", [
    name for name in os.listdir(output_dir)
    if os.path.isdir(os.path.join(output_dir, name))
])

def test_word_count_consistency(base_name):
    """
    Test that the word count of the raw and corrected text files are consistent.
    The difference in word count should not exceed a specified tolerance.
    """
    
    # Paths to the raw and corrected text files
    raw_path = os.path.join(output_dir, base_name, f"{base_name}.raw.txt")
    corrected_path = os.path.join(output_dir, base_name, f"{base_name}.corrected.txt")
    
    logging.info(f"Running Word Count test for: {base_name}")
    
    # Check if the files exist
    assert os.path.exists(raw_path), f"Missing raw file: {raw_path}"
    assert os.path.exists(corrected_path), f"Missing corrected file: {corrected_path}"

    # Read the contents of the files
    with open(raw_path, 'r', encoding='utf-8') as rf:
        raw_text = rf.read()
    with open(corrected_path, 'r', encoding='utf-8') as cf:
        corrected_text = cf.read()

    # Normalize whitespace before counting words
    raw_words = len(normalize_whitespace(raw_text).split())
    corrected_words = len(normalize_whitespace(corrected_text).split())
    
    # Calculate the difference and allowed tolerance
    delta = abs(raw_words - corrected_words) #Absolute difference in word count.
    allowed = max(1, int(raw_words * wordcount_tolerance))  # Never allow 0-word diff
    
    logging.info(f"[{base_name}] Raw words: {raw_words} | Corrected words: {corrected_words} | Delta: {delta} | Allowed: {allowed}")

    if delta > allowed:
        logging.warning(f"[{base_name}] FAILED: Word count difference {delta} exceeds allowed {allowed}")

    # Assert that the difference is within the allowed tolerance. If the word count delta is too high, the test fails.
    assert delta <= allowed, (
        f"[{base_name}] Word count delta too high: raw={raw_words}, corrected={corrected_words}, allowed={allowed}"
    )
