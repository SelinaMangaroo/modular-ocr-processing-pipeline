import os
import json
import time
import logging
import uuid
from anthropic import Anthropic
from anthropic.types.messages.batch_create_params import Request

# === Dynamic batch directory setup ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

input_dir = os.getenv("INPUT_DIR") or ""
letterbook_name = os.path.basename(os.path.normpath(os.path.dirname(input_dir))) if input_dir else "default_batch"

# Create a unique subdirectory for this letterbook's batch files
BATCH_DIR = os.path.join(BASE_DIR, "batchfiles", letterbook_name)
os.makedirs(BATCH_DIR, exist_ok=True)

# === CONFIG ===
PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts.json")
with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
    PROMPTS = json.load(f)

batch_requests = []  # global queue

# === HELPERS ===
def get_client(api_key: str) -> Anthropic:
    return Anthropic(api_key=api_key)

def get_unique_batch_filenames():
    timestamp = time.strftime("%m%d%Y_%H%M%S")
    uid = uuid.uuid4().hex[:3]
    input_file = os.path.join(BATCH_DIR, f"claude_input_{timestamp}_{uid}.jsonl")
    output_file = os.path.join(BATCH_DIR, f"claude_output_{timestamp}_{uid}.jsonl")
    return input_file, output_file

# === BATCH MANAGEMENT ===
def queue_batch_request(base_name, task, model, messages, temperature=0.0, max_tokens=4096):
    """Queue a Claude batch request."""
    custom_id = f"{base_name}__{task}"

    batch_requests.append(
        Request(
            custom_id=custom_id,
            params={
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages,
            },
        )
    )

def submit_batch_and_get_output(client: Anthropic):
    global batch_requests
    if not batch_requests:
        logging.warning("[Claude] No batch requests queued.")
        return None

    _, output_file = get_unique_batch_filenames()

    batch = client.messages.batches.create(requests=batch_requests)
    logging.info(f"[Claude] Submitted batch job to Anthropic (ID: {batch.id})")

    while True:
        current = client.messages.batches.retrieve(batch.id)
        status = current.processing_status
        logging.info(f"[Claude] Batch status: {status}")

        if status == "ended":
            break
        elif status in ("failed", "expired", "canceled"):
            logging.error(f"[Claude] Batch failed or canceled: {status}")
            return None

        time.sleep(30)

    with open(output_file, "w", encoding="utf-8") as f:
        for result in client.messages.batches.results(batch.id):
            f.write(json.dumps(result.model_dump(), ensure_ascii=False) + "\n")

    logging.info(f"[Claude] Batch completed and results saved: {output_file}")
    batch_requests = []
    return output_file

# === MAIN TASKS ===
def correct_text(text, base_name, doc_output_dir, client, model_name, save=True):
    os.makedirs(doc_output_dir, exist_ok=True)
    logging.info(f"[Claude] Queued text correction for batch processing: {base_name}")

    if not text.strip():
        text = "(No OCR text available.)"

    messages = [{"role": "user", "content": f"{PROMPTS['correct_text']}\n{text}"}]
    queue_batch_request(base_name, "correct_text", model_name, messages)

    corrected_path = os.path.join(doc_output_dir, f"{base_name}.corrected.txt")
    with open(corrected_path, "w", encoding="utf-8") as f:
        f.write("")

    return None

def extract_and_explain_entities(text, base_name, doc_output_dir, client, model_name):
    os.makedirs(doc_output_dir, exist_ok=True)
    logging.info(f"[Claude] Queued entity extraction + explanation for batch processing: {base_name}")

    if not text.strip():
        text = "(No text available.)"

    messages = [{"role": "user", "content": f"{PROMPTS['extract_and_explain_entities']}\n{text}"}]
    queue_batch_request(base_name, "extract_and_explain_entities", model_name, messages)

    entity_path = os.path.join(doc_output_dir, f"{base_name}.entities.json")
    with open(entity_path, "w", encoding="utf-8") as f:
        json.dump({"People": {}, "Productions": {}, "Companies": {}, "Theaters": {}, "Dates": {}}, f, indent=2)

    return None

def extract_page_and_split_letters(corrected_text_path, client, model_name):
    os.makedirs(os.path.dirname(corrected_text_path), exist_ok=True)
    base_name = os.path.basename(corrected_text_path).replace(".corrected.txt", "")
    logging.info(f"[Claude] Queued letter-splitting request for batch processing: {base_name}")

    try:
        with open(corrected_text_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except FileNotFoundError:
        text = ""

    messages = [{"role": "user", "content": f"{PROMPTS['split_letters']}\n{text}"}]
    queue_batch_request(base_name, "split_letters", model_name, messages)

    combined_path = corrected_text_path.replace(".corrected.txt", ".combined_output.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write("[]")

    return None

# === HYDRATION ===
def hydrate_batch_output(batch_output_file=None, output_dir="output"):
    if not batch_output_file or not os.path.exists(batch_output_file):
        logging.error(f"[Hydration] Claude batch output not found: {batch_output_file}")
        return

    logging.info(f"[Hydration] Reading Claude batch output: {batch_output_file}")
    hydrated_count = 0

    with open(batch_output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            custom_id = record.get("custom_id", "")
            result = record.get("result", {})

            if not custom_id or result.get("type") != "succeeded":
                continue

            if "__" not in custom_id:
                continue
            base_name, task = custom_id.split("__", 1)

            message = result.get("message", {})
            text_content = "".join(
                block.get("text", "")
                for block in message.get("content", [])
                if block.get("type") == "text"
            ).strip()

            if text_content.startswith("```"):
                text_content = text_content.strip("`").replace("json", "", 1).strip()

            doc_dir = os.path.join(output_dir, base_name)
            os.makedirs(doc_dir, exist_ok=True)

            if task == "correct_text":
                path = os.path.join(doc_dir, f"{base_name}.corrected.txt")
                with open(path, "w", encoding="utf-8") as out:
                    out.write(text_content)
                logging.info(f"[Hydration] Filled corrected text for {base_name}")

            elif task == "extract_and_explain_entities":
                path = os.path.join(doc_dir, f"{base_name}.entities.json")
                try:
                    parsed = json.loads(text_content)
                except json.JSONDecodeError:
                    parsed = {}
                with open(path, "w", encoding="utf-8") as out:
                    json.dump(parsed, out, indent=2, ensure_ascii=False)
                logging.info(f"[Hydration] Filled entities + explanations for {base_name}")

            elif task == "split_letters":
                path = os.path.join(doc_dir, f"{base_name}.combined_output.json")
                try:
                    parsed_letters = json.loads(text_content)
                except json.JSONDecodeError:
                    parsed_letters = []
                with open(path, "w", encoding="utf-8") as out:
                    json.dump({"letters": parsed_letters}, out, indent=2, ensure_ascii=False)
                logging.info(f"[Hydration] Filled combined letters output for {base_name}")

            hydrated_count += 1

    logging.info(f"[Hydration] Completed Claude hydration for {hydrated_count} files.")
    return batch_requests
