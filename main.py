import os
import logging
import time
import importlib
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.helpers import initialize_logging, split_into_batches, clean_tmp_folder, log_runtime

# --- Load environment variables ---
load_dotenv()

# Directories
tmp_dir = os.environ.get("TMP_DIR")
input_dir = os.environ.get("INPUT_DIR")
output_dir = os.environ.get("OUTPUT_DIR")

# General params
max_threads = int(os.environ.get("MAX_THREADS", 4))
batch_size = int(os.environ.get("BATCH_SIZE", 5))
image_magick_command = os.environ.get("IMAGE_MAGICK_COMMAND", "convert")

# Providers
ocr_provider = os.getenv("OCR_PROVIDER", "aws").lower()
llm_provider = os.getenv("LLM_PROVIDER", "chatgpt").lower()

# --- Choose API key + model dynamically ---
api_key, model_name = None, None
if llm_provider == "claude":
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    model_name = os.environ.get("CLAUDE_MODEL", "claude-3-7-sonnet-20250219")
elif llm_provider == "chatgpt":
    api_key = os.environ.get("OPENAI_API_KEY")
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
elif llm_provider == "llama":
    api_key = None  # Ollama runs locally
    model_name = os.environ.get("LLAMA_MODEL", "llama3.1:8b")

# --- Provider-specific options ---
service_options = {}
if ocr_provider == "aws":
    bucket_name = os.getenv("BUCKET_NAME")
    region = os.getenv("REGION")
    if not bucket_name or not region:
        logging.error("[AWS] BUCKET_NAME or REGION not set — this will cause errors")
    else:
        service_options["bucket_name"] = bucket_name
        service_options["region"] = region

# --- Setup logging ---
initialize_logging()
logging.info("OCR processing pipeline started.")

start_time = time.time()

# --- Import provider modules ---
try:
    ocr_module = importlib.import_module(f"ocr_providers.{ocr_provider}_provider")
    llm_module = importlib.import_module(f"llms.{llm_provider}")
except ModuleNotFoundError as e:
    logging.error(f"Provider import error: {e}")
    raise

# --- Collect files ---
if not input_dir or not os.path.isdir(input_dir):
    logging.error("INPUT_DIR is not set or does not exist.")
    files = []
else:
    files = [
        f for f in os.listdir(input_dir)
        if not f.startswith("._") and f.lower().endswith((".jpg", ".jpeg", ".pdf", ".png", ".tif", ".tiff"))
    ]

if not files:
    logging.warning("No input files found to process.")

batches = list(split_into_batches(files, batch_size))
logging.info(f"Batch size: {batch_size} | Max Threads: {max_threads} | Total files: {len(files)}")

# --- Process in batches ---
for batch_index, current_batch in enumerate(batches):
    logging.info(f"Processing batch {batch_index + 1} of {len(batches)}")

    jobs = {}

    # --- Prepare files in parallel ---
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {
            executor.submit(
                ocr_module.prepare_file, filename, tmp_dir, input_dir, output_dir, image_magick_command, **service_options,
            ): filename
            for filename in current_batch
        }
        for future in as_completed(futures):
            filename = futures[future]
            try:
                result = future.result()
                if result:
                    base_name, job_info = result
                    jobs[base_name] = job_info
            except Exception as e:
                logging.error(f"Prep failed for {filename}: {e}", exc_info=True)

    # --- Process results in parallel ---
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {
            executor.submit(
                ocr_module.process_file, base_name, job_info, llm_module, model_name, api_key,
            ): base_name
            for base_name, job_info in jobs.items()
        }
        for future in as_completed(futures):
            base_name = futures[future]
            try:
                future.result()
            except Exception as e:
                logging.error(f"Processing failed for {base_name}: {e}", exc_info=True)

    clean_tmp_folder(tmp_dir)

log_runtime(start_time)
