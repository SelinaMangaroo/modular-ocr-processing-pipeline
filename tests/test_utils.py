import re
import logging
import os
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv
import Levenshtein

load_dotenv()

wordcount_tolerance = float(os.environ.get("TEST_WORD_COUNT_TOLERANCE", "0.05"))
levenshtein_max_diff_ratio = float(os.environ.get("TEST_LEVENSHTEIN_MAX_DIFF_RATIO", "0.10"))
output_dir = os.environ.get("OUTPUT_DIR")

chat_gpt_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

chatgpt_runs = int(os.environ.get("TEST_GPT_REPEAT_CORRECTIONS", 3))
target_file = os.environ.get("TEST_TARGET_FILE")

def initialize_test_logger(log_dir="test_logs"):
    """
    Initializes logging for test cases.
    Args:
        log_dir (str): Directory to save the log file. Default is "test_logs".
    """
    os.makedirs(log_dir, exist_ok=True)
    log_filename = datetime.now().strftime("test_%m-%d-%Y_%H-%M-%S.log")
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Console output (so you can still see in terminal)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(console_handler)

    # File output (persistent log file for test runs)
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%m-%d-%Y_%H-%M-%S"))
    logger.addHandler(file_handler)

    logger.info(f"Test logging initialized → {log_path}")

def normalize_whitespace(text):
    """
    Normalize whitespace in the text by:
    - Converting to lowercase.
    - Stripping leading/trailing spaces.
    - Replacing multiple spaces/tabs/newlines with a single space.
    """
    text = text.lower()
    return re.sub(r'\s+', ' ', text).strip()

def check_word_count_similarity(text_a, text_b, base_name):
    a_words = len(normalize_whitespace(text_a).split())
    b_words = len(normalize_whitespace(text_b).split())
    difference = abs(a_words - b_words)
    allowed = max(1, int(a_words * wordcount_tolerance))

    logging.info(f"[{base_name}] Word Count Check → A: {a_words} | B: {b_words} | Difference: {difference} | Allowed: {allowed}")

    assert difference <= allowed, (
        f"[{base_name}] Word count delta too high: {a_words} vs {b_words} (Allowed: {allowed})"
    )

def check_levenshtein_similarity(text_a, text_b, base_name):
    a_norm = normalize_whitespace(text_a)
    b_norm = normalize_whitespace(text_b)
    similarity = Levenshtein.ratio(a_norm, b_norm)
    diff = 1 - similarity

    logging.info(f"[{base_name}] Levenshtein Check → Similarity: {similarity:.2%}, Difference: {diff:.2%}, Allowed: {levenshtein_max_diff_ratio:.2%}")

    assert diff <= levenshtein_max_diff_ratio, (
        f"[{base_name}] Levenshtein difference too high: {diff:.2%} (Allowed: {levenshtein_max_diff_ratio:.2%})"
    )