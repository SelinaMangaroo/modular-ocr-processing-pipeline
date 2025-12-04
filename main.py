import os
import logging
import time
from datetime import timedelta
import importlib
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.helpers import initialize_logging, split_into_batches, clean_tmp_folder

# --- Load environment ---
load_dotenv()
# tmp_dir = os.getenv("TMP_DIR")
input_dir = os.getenv("INPUT_DIR")
output_dir = os.getenv("OUTPUT_DIR")

# --- TMP DIR ---
base_tmp = os.getenv("TMP_DIR", "./tmp")
letterbook_name = (
    os.path.basename(os.path.dirname(os.path.normpath(input_dir)))
    if input_dir else "unknown_book"
)
tmp_dir = os.path.join(base_tmp, letterbook_name)
os.makedirs(tmp_dir, exist_ok=True)

max_threads = int(os.getenv("MAX_THREADS", 4))
batch_size = int(os.getenv("BATCH_SIZE", 5))
image_magick_command = os.getenv("IMAGE_MAGICK_COMMAND", "convert")

ocr_provider = os.getenv("OCR_PROVIDER", "aws").lower()
if ocr_provider == "azure":
    max_threads = 1
else:
    max_threads = int(os.getenv("MAX_THREADS", 4))
    
llm_provider = os.getenv("LLM_PROVIDER", "chatgpt").lower()

if llm_provider == "chatgpt":
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

elif llm_provider == "claude":
    api_key = os.getenv("ANTHROPIC_API_KEY")
    model_name = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

# --- AWS Setup ---
service_options = {
    "bucket_name": os.getenv("BUCKET_NAME"),
    "region": os.getenv("REGION"),
}

initialize_logging()
logging.info("OCR processing pipeline started.")
overall_start = time.time()

# --- Import providers ---
ocr_module = importlib.import_module(f"ocr_providers.{ocr_provider}_provider")
llm_module = importlib.import_module(f"llms.{llm_provider}")

# --- Collect files ---
if not input_dir or not os.path.isdir(input_dir):
    logging.error("INPUT_DIR not found.")
    files = []
else:
    files = [f for f in os.listdir(input_dir)
             if not f.startswith("._") and f.lower().endswith((".jpg", ".jpeg", ".pdf", ".png", ".tif", ".tiff"))]

if not files:
    logging.warning("No input files found.")
batches = list(split_into_batches(files, batch_size))
logging.info(f"Batch size={batch_size}, Max Threads={max_threads}, Total={len(files)}")

# --- Process batches ---
for batch_index, current_batch in enumerate(batches, 1):
    logging.info(f"=== Processing Batch {batch_index}/{len(batches)} ===")
    jobs = {}

    # Prepare files in parallel
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {
            executor.submit(
                ocr_module.prepare_file, filename, tmp_dir, input_dir, output_dir,
                image_magick_command, **service_options
            ): filename for filename in current_batch
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

    # Process each file (queue ChatGPT requests)
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {
            executor.submit(
                ocr_module.process_file, base_name, job_info,
                llm_module, model_name, api_key
            ): base_name for base_name, job_info in jobs.items()
        }
        for future in as_completed(futures):
            base_name = futures[future]
            try:
                future.result()
            except Exception as e:
                logging.error(f"Processing failed for {base_name}: {e}", exc_info=True)

    # Submit queued batch to ChatGPT once per batch
    logging.info(f"[LLM] Submitting {llm_provider.capitalize()} batch for batch {batch_index}")

    batch_output_1 = llm_module.submit_batch_and_get_output(llm_module.get_client(api_key))
    if batch_output_1:
        llm_module.hydrate_batch_output(batch_output_1, output_dir)

    # Queue entity/split requests from corrected text
    for base_name, job_info in jobs.items():
        corrected_path = os.path.join(job_info["doc_output_dir"], f"{base_name}.corrected.txt")
        if os.path.exists(corrected_path):
            with open(corrected_path, "r", encoding="utf-8") as f:
                corrected_text = f.read()
            llm_module.extract_and_explain_entities(corrected_text, base_name, job_info["doc_output_dir"],
                                                    llm_module.get_client(api_key), model_name)
            llm_module.extract_page_and_split_letters(corrected_path, llm_module.get_client(api_key), model_name)

    # Submit second batch (entities + splits)
    batch_output_2 = llm_module.submit_batch_and_get_output(llm_module.get_client(api_key))
    if batch_output_2:
        llm_module.hydrate_batch_output(batch_output_2, output_dir)

    clean_tmp_folder(tmp_dir)

# --- Timing ---
total_time = time.time() - overall_start
logging.info(f"Pipeline finished successfully in {timedelta(seconds=total_time)}.")