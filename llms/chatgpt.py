import os
import json
import time
import logging
import uuid
from openai import OpenAI

# === Dynamic batch directory setup ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

input_dir = os.getenv("INPUT_DIR") or ""
letterbook_name = (
    os.path.basename(os.path.dirname(os.path.normpath(input_dir)))
    if input_dir else "default_batch"
)

# Create a unique subdirectory for this letterbook's batch files
BATCH_DIR = os.path.join(BASE_DIR, "batchfiles", letterbook_name)
os.makedirs(BATCH_DIR, exist_ok=True)

# === CONFIG ===
PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts.json")
with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
    PROMPTS = json.load(f)

batch_requests = []  # global batch queue

# === HELPERS ===
def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

def get_unique_batch_filenames():
    """Generates unique filenames for batch input/output in ./batchfiles."""
    timestamp = time.strftime("%m%d%Y_%H%M%S")
    uid = uuid.uuid4().hex[:3]
    input_file = os.path.join(BATCH_DIR, f"chatgpt_input_{timestamp}_{uid}.jsonl")
    output_file = os.path.join(BATCH_DIR, f"chatgpt_output_{timestamp}_{uid}.jsonl")
    return input_file, output_file

def queue_batch_request(base_name, task, model, messages, temperature=0.0):
    """Add a request to the global batch queue."""
    custom_id = f"{base_name}::{task}"
    batch_requests.append({
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": model, "messages": messages, "temperature": temperature},
    })

# === CORE BATCH API ===
def submit_batch_and_get_output(client: OpenAI):
    """Submit queued requests, wait for completion, and download results."""
    if not batch_requests:
        logging.warning("[ChatGPT] No batch requests queued.")
        return None

    input_file, output_file = get_unique_batch_filenames()

    # Write batch input file
    with open(input_file, "w", encoding="utf-8") as f:
        for req in batch_requests:
            json.dump(req, f)
            f.write("\n")

    upload = client.files.create(file=open(input_file, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    logging.info(f"[ChatGPT] Submitted batch ID {batch.id} with {len(batch_requests)} requests.")
    batch_requests.clear()

    # Poll until finished
    while True:
        batch = client.batches.retrieve(batch.id)
        if batch.status == "completed":
            break
        elif batch.status in ("failed", "expired", "cancelled"):
            logging.error(f"[ChatGPT] Batch failed: {batch.status}")
            return None
        logging.info(f"[ChatGPT] Waiting... Status: {batch.status}")
        time.sleep(30)

    # Download output
    output_file_id = batch.output_file_id
    if not output_file_id:
        logging.error("[ChatGPT] Batch completed but no output file found.")
        return None

    content = client.files.content(output_file_id)
    with open(output_file, "wb") as f:
        f.write(content.read())

    logging.info(f"[ChatGPT] Batch completed and saved: {output_file}")
    return output_file

# === TASKS ===
def correct_text(raw_text, base_name, doc_output_dir, client, model_name):
    """Queue text correction for batch processing."""
    os.makedirs(doc_output_dir, exist_ok=True)
    logging.info(f"[ChatGPT] Queued correction for {base_name}")

    if not raw_text or not raw_text.strip():
        raw_text = "(No OCR text available.)"

    messages = [
        {"role": "system", "content": PROMPTS["correct_text"]},
        {"role": "user", "content": raw_text},
    ]
    queue_batch_request(base_name, "correct_text", model_name, messages)

    return None

def extract_and_explain_entities(corrected_text, base_name, doc_output_dir, client, model_name):
    """Queue entity extraction and explanation."""
    os.makedirs(doc_output_dir, exist_ok=True)
    logging.info(f"[ChatGPT] Queued entity extraction for {base_name}")

    if not corrected_text or not corrected_text.strip():
        corrected_text = "(No corrected text available.)"

    messages = [
        {"role": "system", "content": PROMPTS["extract_and_explain_entities"]},
        {"role": "user", "content": corrected_text},
    ]
    queue_batch_request(base_name, "extract_and_explain_entities", model_name, messages)

    return None

def extract_page_and_split_letters(corrected_text_path, client, model_name):
    """Queue letter-splitting."""
    base_name = os.path.basename(corrected_text_path).replace(".corrected.txt", "")
    logging.info(f"[ChatGPT] Queued letter splitting for {base_name}")

    try:
        with open(corrected_text_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except FileNotFoundError:
        text = "(No corrected text file found.)"

    messages = [
        {"role": "user", "content": PROMPTS["split_letters"] + f"\nText:\n{text}"},
    ]
    queue_batch_request(base_name, "split_letters", model_name, messages)

    return None

# === HYDRATION ===
def hydrate_batch_output(batch_output_file, output_dir="output"):
    """Read results from batch file and write them into local output files."""
    if not os.path.exists(batch_output_file):
        logging.error(f"[Hydration] Batch output not found: {batch_output_file}")
        return

    with open(batch_output_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            custom_id = record.get("custom_id", "")
            response = record.get("response", {})
            content = (
                response.get("body", {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if not custom_id or not content:
                continue

            base_name, task = custom_id.split("::", 1)
            doc_dir = os.path.join(output_dir, base_name)
            os.makedirs(doc_dir, exist_ok=True)

            if task == "correct_text":
                path = os.path.join(doc_dir, f"{base_name}.corrected.txt")
                with open(path, "w", encoding="utf-8") as out:
                    out.write(content)

            elif task == "extract_and_explain_entities":
                path = os.path.join(doc_dir, f"{base_name}.entities.json")
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    parsed = {}
                json.dump(parsed, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

            elif task == "split_letters":
                path = os.path.join(doc_dir, f"{base_name}.combined_output.json")
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    parsed = []
                json.dump({"letters": parsed}, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    logging.info(f"[Hydration] Completed hydration for {batch_output_file}")