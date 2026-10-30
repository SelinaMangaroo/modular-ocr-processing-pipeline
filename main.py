import os
import logging
import time
from datetime import timedelta
import importlib
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from utils.helpers import initialize_logging, split_into_batches, clean_tmp_folder, audit_outputs, entities_ok

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
llm_provider = os.getenv("LLM_PROVIDER", "chatgpt").lower()

# Some LLM providers (Gemini, Claude v2) handle OCR + correction themselves
# via vision-capable calls, so there's no separate OCR provider module for
# them — OCR_PROVIDER is ignored whenever LLM_PROVIDER is one of these.
CONSOLIDATED_LLM_PROVIDERS = {"gemini", "claude2", "chatgpt2", "glm"}
use_consolidated_ocr = llm_provider in CONSOLIDATED_LLM_PROVIDERS

if use_consolidated_ocr:
    max_threads = int(os.getenv("MAX_THREADS", 4))
elif ocr_provider == "azure":
    max_threads = 1
else:
    max_threads = int(os.getenv("MAX_THREADS", 4))

if llm_provider == "chatgpt":
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

elif llm_provider == "chatgpt2":
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL", "gpt-5.6-terra")

elif llm_provider == "claude":
    api_key = os.getenv("ANTHROPIC_API_KEY")
    model_name = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

elif llm_provider == "claude2":
    api_key = os.getenv("ANTHROPIC_API_KEY")
    model_name = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")

elif llm_provider == "gemini":
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")

elif llm_provider == "glm":
    api_key = os.getenv("GLM_API_KEY")
    model_name = os.getenv("GLM_MODEL", "glm-5.3-flash")

# --- AWS Setup ---
service_options = {
    "bucket_name": os.getenv("BUCKET_NAME"),
    "region": os.getenv("REGION"),
}

initialize_logging()
logging.info("OCR processing pipeline started.")
overall_start = time.time()

# --- Import providers ---
if not use_consolidated_ocr:
    ocr_module = importlib.import_module(f"ocr_providers.{ocr_provider}_provider")
llm_module = importlib.import_module(f"llms.{llm_provider}")

# --- Collect files ---
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".pdf", ".png", ".tif", ".tiff")

if not input_dir or not os.path.exists(input_dir):
    logging.error("INPUT_DIR not found.")
    files = []
elif os.path.isfile(input_dir):
    # Single-file mode: INPUT_DIR points directly at one image/pdf. The rest
    # of the pipeline expects INPUT_DIR to be a folder (it joins INPUT_DIR +
    # filename), so re-point it at the containing directory and process just
    # that one file.
    single_file = os.path.basename(input_dir)
    input_dir = os.path.dirname(input_dir)
    if not single_file.startswith("._") and single_file.lower().endswith(SUPPORTED_EXTENSIONS):
        files = [single_file]
        logging.info(f"INPUT_DIR points to a single file: {single_file}")
    else:
        logging.error(f"INPUT_DIR points to an unsupported file: {single_file}")
        files = []
else:
    files = [f for f in os.listdir(input_dir)
             if not f.startswith("._") and f.lower().endswith(SUPPORTED_EXTENSIONS)]

if not files:
    logging.warning("No input files found.")
batches = list(split_into_batches(files, batch_size))
logging.info(f"Batch size={batch_size}, Max Threads={max_threads}, Total={len(files)}")

# --- Process batches ---
for batch_index, current_batch in enumerate(batches, 1):
    logging.info(f"=== Processing Batch {batch_index}/{len(batches)} ===")
    jobs = {}

    if use_consolidated_ocr:
        # This provider handles OCR + correction itself (Gemini or Claude v2):
        # prep every file in this pipeline batch first, then submit ONE batch
        # job covering all of them (rather than one API call per file), wait
        # for it, then do the same for entity extraction + letter splitting.
        client = llm_module.get_client(api_key)

        prepared_items = []
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = {
                executor.submit(
                    llm_module.prepare_and_upload, filename, tmp_dir, input_dir, output_dir,
                    image_magick_command, client
                ): filename for filename in current_batch
            }
            for future in as_completed(futures):
                filename = futures[future]
                try:
                    result = future.result()
                    if result:
                        prepared_items.append(result)
                except Exception as e:
                    logging.error(f"Prep failed for {filename}: {e}", exc_info=True)

        jobs = llm_module.run_ocr_batch(prepared_items, model_name, api_key)

        # On reruns (SKIP_OCR_IF_EXISTS=true) don't pay again for files whose
        # entities were already extracted successfully.
        if os.getenv("SKIP_OCR_IF_EXISTS", "false").lower() == "true":
            jobs_for_entities = {
                base: info for base, info in jobs.items()
                if not entities_ok(output_dir, base + ".jpg")
            }
        else:
            jobs_for_entities = jobs
        llm_module.run_entities_and_split_batch(jobs_for_entities, model_name, api_key)

    else:
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

        # Process each file (queue LLM correction requests)
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

        # Submit queued correction batch once per batch
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

# --- Completeness audit: every input file must have a usable .corrected.txt and .entities.json ---
audit_problems, audit_review = [], []
if output_dir:
    audit_problems, audit_review = audit_outputs(files, output_dir)
    for fname, kind, detail in audit_problems:
        logging.error(f"[AUDIT] {kind}: {fname} ({detail})")
    for fname, kind, detail in audit_review:
        logging.warning(f"[AUDIT] REVIEW {kind}: {fname} ({detail})")
    if audit_problems:
        logging.error(f"[AUDIT] INCOMPLETE: {len(files) - len({p[0] for p in audit_problems})}/{len(files)} files complete.")
    else:
        logging.info(f"[AUDIT] OK: all {len(files)} files have .corrected.txt and .entities.json.")

# --- Cost summary (consolidated providers only; no-op for chatgpt/claude legacy) ---
if hasattr(llm_module, "print_cost_summary"):
    llm_module.print_cost_summary()

if hasattr(llm_module, "get_run_totals") and output_dir:
    import json as _json
    from datetime import datetime as _datetime

    totals = llm_module.get_run_totals()
    summary = {
        "provider": llm_provider,
        "model": model_name,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "documents_processed": len(files),
        "total_tokens": totals["total_tokens"],
        "total_cost_usd": round(totals["total_cost"], 6),
        "cost_per_document_usd": round(totals["total_cost"] / len(files), 6) if files else None,
        "run_finished_at": _datetime.now().isoformat(timespec="seconds"),
        "run_duration_seconds": round(total_time, 1),
        "audit_complete": not audit_problems,
        "audit_problems": [list(p) for p in audit_problems],
        "audit_review": [list(p) for p in audit_review],
    }
    summary_path = os.path.join(output_dir, "run_summary.json")
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            _json.dump(summary, f, indent=2)
        logging.info(f"Wrote run summary: {summary_path}")
    except Exception as e:
        logging.warning(f"Could not write run summary to {summary_path}: {e}")

# Non-zero exit so run_batch_directories.py reports this folder as FAILED when anything is missing.
if audit_problems:
    sys.exit(1)