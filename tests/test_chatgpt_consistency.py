import os
import pytest
from utils.chatgpt_utils import correct_text_with_chatgpt
from tests.test_utils import *

initialize_test_logger()

if target_file:
    test_cases = [target_file] if os.path.isdir(os.path.join(output_dir, target_file)) else []
else:
    test_cases = [
        name for name in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, name))
    ]
if target_file and not test_cases:
    logging.warning(f"[WARN] TEST_TARGET_FILE '{target_file}' not found in output directory.")

@pytest.mark.parametrize("base_name", test_cases)

def test_chatgpt_consistency(base_name):
    logging.info(f"[{base_name}] Running ChatGPT consistency test for {chatgpt_runs} runs")

    raw_path = os.path.join(output_dir, base_name, f"{base_name}.raw.txt")
    assert os.path.exists(raw_path), f"Missing raw file: {raw_path}"

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    corrected_versions = []

    for i in range(chatgpt_runs):
        corrected = correct_text_with_chatgpt(raw_text, base_name, output_dir, chat_gpt_client, model_name, save=False)
        corrected_versions.append(corrected)

    # Compare all pairs of results
    for i in range(len(corrected_versions)):
        for j in range(i + 1, len(corrected_versions)):
            a, b = corrected_versions[i], corrected_versions[j]
            logging.info(f"[{base_name}] Run {i+1} vs {j+1}")
            check_levenshtein_similarity(a, b, base_name)
            check_word_count_similarity(a, b, base_name)