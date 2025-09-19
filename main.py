import os
import logging
import time
import importlib
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.helpers import (
    initialize_logging,
    split_into_batches,
    clean_tmp_folder,
    log_runtime
)

# --- Load environment variables ---
load_dotenv()
bucket_name = os.environ.get("BUCKET_NAME")
region = os.environ.get("REGION")
max_threads = int(os.environ.get("MAX_THREADS", 4))
tmp_dir = os.environ.get("TMP_DIR")
input_dir = os.environ.get("INPUT_DIR")
output_dir = os.environ.get("OUTPUT_DIR")
batch_size = int(os.environ.get("BATCH_SIZE", 5))
image_magick_command = os.environ.get("IMAGE_MAGICK_COMMAND", "convert")
api_key = os.environ.get("OPENAI_API_KEY")
model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

ocr_provider_name = os.getenv("OCR_PROVIDER", "aws")
llm_provider_name = os.getenv("LLM_PROVIDER", "chatgpt")

# --- Choose API key + model dynamically ---
if llm_provider_name == "claude":
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    model_name = os.environ.get("CLAUDE_MODEL", "claude-3-7-sonnet-20250219")
elif llm_provider_name == "chatgpt":
    api_key = os.environ.get("OPENAI_API_KEY")
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
elif llm_provider_name == "llama":
    api_key = None  # Ollama runs locally
    model_name = os.environ.get("LLAMA_MODEL", "llama3.1:8b")
else:
    raise ValueError(f"Unknown LLM_PROVIDER: {llm_provider_name}")

# --- Initialize logging ---
initialize_logging()
logging.info("OCR processing pipeline started.")

start_time = time.time()

# --- import provider modules ---
ocr_module = importlib.import_module(f"providers.{ocr_provider_name}_provider")
llm_module = importlib.import_module(f"utils.{llm_provider_name}_utils")

if not input_dir or not os.path.isdir(input_dir):
    logging.error("INPUT_DIR is not set or does not exist.")
    exit(1)

# --- Collect files ---
files = [
    f for f in os.listdir(input_dir)
    if not f.startswith("._") and f.lower().endswith((".jpg", ".jpeg", ".pdf", ".png", ".tif", ".tiff"))
]
batches = list(split_into_batches(files, batch_size))

logging.info(f"Batch size: {batch_size} | Max Threads: {max_threads}")

# --- Process in batches ---
for batch_index, current_batch in enumerate(batches):
    logging.info(f"Processing batch {batch_index + 1} of {len(batches)}")

    jobs = {}

    # --- Prepare files in parallel ---
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [
            executor.submit(
                ocr_module.prepare_file,
                filename,
                tmp_dir,
                input_dir,
                output_dir,
                image_magick_command,
                bucket_name,
                region
            )
            for filename in current_batch
        ]
        for future in as_completed(futures):
            result = future.result()
            if result:
                base_name, job_info = result
                jobs[base_name] = job_info

    # --- Process results in parallel ---
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [
            executor.submit(
                ocr_module.process_file,
                base_name,
                job_info,
                llm_module,
                model_name,
                api_key
            )
            for base_name, job_info in jobs.items()
        ]
        for future in as_completed(futures):
            future.result()

    clean_tmp_folder(tmp_dir)

log_runtime(start_time)
