import os
import json
import time
import base64
import logging
import threading
from anthropic import Anthropic
from anthropic.types.messages.batch_create_params import Request
from utils.helpers import get_file_paths, resize_image

# === CONFIG ===
PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts.json")
with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
    PROMPTS = json.load(f)

# Reuses the same OCR+correction prompt Gemini uses (key name is a holdover
# from when Gemini was the only consolidated provider — the instructions
# themselves are model-agnostic). Worth renaming to something neutral like
# "ocr_and_correct" in prompts.json if a third provider gets added later.
OCR_PROMPT_KEY = "gemini_ocr_and_correct"
RAW_MARKER = "=== RAW TRANSCRIPTION ==="
CORRECTED_MARKER = "=== SPELLING-CORRECTED ==="

POLL_INTERVAL_SECONDS = int(os.getenv("CLAUDE_BATCH_POLL_INTERVAL", "20"))

# === COST TRACKING ===
# Pricing per 1M tokens, standard (non-batch) rate, in USD. From Anthropic's
# official pricing table (confirmed current, no expiration noted for Sonnet 5's
# $2/$10 rate as of Aug 2026 — re-verify at https://anthropic.com/pricing if
# costs look off later). Batch API applies a flat 50% discount to input/output;
# cached input ("Cache Hits & Refreshes" in Anthropic's table) is NOT further
# discounted in batch mode.
CLAUDE_PRICING = {
    "claude-sonnet-5": {"input": 2.00, "output": 10.00, "cached_input": 0.20},
    "claude-opus-5": {"input": 5.00, "output": 25.00, "cached_input": 0.50},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00, "cached_input": 0.10},
}
DEFAULT_PRICING_KEY = "claude-sonnet-5"
BATCH_DISCOUNT = 0.5

_cost_lock = threading.Lock()
_run_total_cost = 0.0
_run_total_tokens = 0


def _get_pricing(model_name):
    pricing = CLAUDE_PRICING.get(model_name)
    if not pricing:
        logging.warning(
            f"[Claude] No pricing entry for model '{model_name}'; "
            f"using {DEFAULT_PRICING_KEY} rates as an estimate. Verify actual pricing."
        )
        pricing = CLAUDE_PRICING[DEFAULT_PRICING_KEY]
    return pricing


def _estimate_cost(usage, model_name):
    pricing = _get_pricing(model_name)
    billable_input = max(usage["input"] - usage["cached"], 0)

    discounted_cost = (billable_input * pricing["input"] + usage["output"] * pricing["output"]) * BATCH_DISCOUNT
    cached_cost = usage["cached"] * pricing["cached_input"]  # not discounted in batch mode

    return (discounted_cost + cached_cost) / 1_000_000


def _log_cost(base_name, task, model_name, usage):
    global _run_total_cost, _run_total_tokens
    cost = _estimate_cost(usage, model_name)

    with _cost_lock:
        _run_total_cost += cost
        _run_total_tokens += usage["total"]

    logging.info(
        f"[Claude] Cost for {base_name}::{task} — "
        f"{usage['total']} tokens (in={usage['input']}, out={usage['output']}, cached={usage['cached']}) "
        f"≈ ${cost:.4f} [batch rate, 50% off]"
    )


def print_cost_summary():
    logging.info(
        f"[Claude] Run totals — {_run_total_tokens} tokens, ≈ ${_run_total_cost:.4f} estimated (batch rate)."
    )

def get_run_totals():
    """Returns the running token/cost totals for this process's lifetime, for
    external tooling (e.g. a cross-provider comparison script) to read without
    scraping log files."""
    return {"total_tokens": _run_total_tokens, "total_cost": _run_total_cost}


# === HELPERS ===
def get_client(api_key: str) -> Anthropic:
    return Anthropic(api_key=api_key)


def _parse_ocr_response(response_text, base_name):
    if RAW_MARKER in response_text and CORRECTED_MARKER in response_text:
        raw_part = response_text.split(RAW_MARKER, 1)[1].split(CORRECTED_MARKER, 1)[0].strip()
        corrected_part = response_text.split(CORRECTED_MARKER, 1)[1].strip()
        return raw_part, corrected_part

    logging.warning(
        f"[Claude] Response for {base_name} did not match expected format; "
        "saving full response as both raw and corrected for manual review."
    )
    return response_text.strip(), response_text.strip()


def _extract_usage(usage):
    """Anthropic's usage object uses stable, well-documented field names —
    no guessing needed here the way Gemini's newer Interactions API required."""
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    cached_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
    return {
        "input": input_tokens,
        "output": output_tokens,
        "cached": cached_tokens,
        "total": input_tokens + output_tokens,
    }


def _build_text_request(custom_id, prompt_text, model_name, max_tokens=4096):
    return Request(
        custom_id=custom_id,
        params={
            "model": model_name,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt_text}],
        },
    )


def _build_ocr_request(custom_id, prompt_text, data_b64, media_type, block_type, model_name, max_tokens=4096):
    content = [
        {"type": "text", "text": prompt_text},
        {"type": block_type, "source": {"type": "base64", "media_type": media_type, "data": data_b64}},
    ]
    return Request(
        custom_id=custom_id,
        params={
            "model": model_name,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
        },
    )


# === BATCH JOB EXECUTION (shared by OCR+correction and entities+split) ===
def _run_batch_job(client, requests, display_name):
    """
    requests: list of anthropic Request objects.
    Returns: dict mapping custom_id -> {"text": str, "usage": dict} for
    succeeded requests. Failed individual requests are logged and simply
    absent from the returned dict.
    """
    if not requests:
        return {}

    batch = client.messages.batches.create(requests=requests)
    logging.info(f"[Claude] Submitted batch job '{display_name}' ({len(requests)} requests): {batch.id}")

    while True:
        batch = client.messages.batches.retrieve(batch.id)
        status = batch.processing_status
        logging.info(f"[Claude] Batch '{display_name}' status: {status}")
        if status == "ended":
            break
        elif status in ("failed", "expired", "canceled"):
            logging.error(f"[Claude] Batch '{display_name}' did not succeed: {status}")
            return {}
        time.sleep(POLL_INTERVAL_SECONDS)

    results = {}
    for item in client.messages.batches.results(batch.id):
        custom_id = item.custom_id
        if item.result.type != "succeeded":
            error_detail = getattr(item.result, "error", None)
            logging.error(f"[Claude] Batch request failed for {custom_id}: {item.result.type} — {error_detail}")
            continue
        message = item.result.message
        text = "".join(block.text for block in message.content if block.type == "text").strip()
        usage = _extract_usage(message.usage)
        results[custom_id] = {"text": text, "usage": usage}

    logging.info(f"[Claude] Batch '{display_name}' complete — {len(results)}/{len(requests)} succeeded.")
    return results


# === OCR + CORRECTION ===
def prepare_and_upload(filename, tmp_dir, input_dir, output_dir, image_magick_command, client=None):
    """
    Resolves paths, checks skip logic, resizes/prepares the image (or reads
    PDFs as-is), and base64-encodes it for inline submission. Claude's Batch
    API takes request bodies inline — no separate file-upload step needed the
    way Gemini's file-based batch requires. `client` is accepted for interface
    parity with gemini.py's prepare_and_upload but isn't used here.
    """
    paths = get_file_paths(filename, tmp_dir, input_dir, output_dir)
    base_name = paths["base_name"]
    doc_output_dir = paths["doc_output_dir"]
    os.makedirs(doc_output_dir, exist_ok=True)

    raw_path = os.path.join(doc_output_dir, f"{base_name}.raw.txt")
    corrected_path = os.path.join(doc_output_dir, f"{base_name}.corrected.txt")

    skip_ocr = os.getenv("SKIP_OCR_IF_EXISTS", "false").lower() == "true"
    if skip_ocr and os.path.exists(raw_path) and os.path.exists(corrected_path):
        logging.info(f"[Claude] Skipping OCR for {base_name} (raw + corrected already exist).")
        return {"base_name": base_name, "doc_output_dir": doc_output_dir, "skip": True}

    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        with open(paths["path_to_file"], "rb") as f:
            data_b64 = base64.b64encode(f.read()).decode("utf-8")
        media_type = "application/pdf"
        block_type = "document"
    else:
        resized_path = os.path.join(tmp_dir, f"{base_name}_resized.jpg")
        prepared_file = resize_image(paths["path_to_file"], resized_path, image_magick_command, filename)
        with open(prepared_file, "rb") as f:
            data_b64 = base64.b64encode(f.read()).decode("utf-8")
        media_type = "image/jpeg"
        block_type = "image"

    return {
        "base_name": base_name,
        "doc_output_dir": doc_output_dir,
        "skip": False,
        "data": data_b64,
        "media_type": media_type,
        "block_type": block_type,
    }


def run_ocr_batch(prepared_items, model_name, api_key):
    """
    Submits one Batch API job covering OCR + correction for every
    non-skipped file, waits for it to complete, and writes .raw.txt /
    .corrected.txt for each. Returns a jobs dict (base_name -> {doc_output_dir})
    for every file ready for the next stage.
    """
    client = Anthropic(api_key=api_key)
    claude_model = os.getenv("CLAUDE_MODEL", model_name or "claude-sonnet-5")

    jobs = {}
    batch_requests = []
    lookup = {}

    for item in prepared_items:
        if not item:
            continue
        base_name = item["base_name"]
        if item.get("skip"):
            jobs[base_name] = {"doc_output_dir": item["doc_output_dir"]}
            continue
        custom_id = f"ocr__{base_name}"
        req = _build_ocr_request(
            custom_id, PROMPTS[OCR_PROMPT_KEY], item["data"], item["media_type"], item["block_type"], claude_model
        )
        batch_requests.append(req)
        lookup[custom_id] = item

    if not batch_requests:
        return jobs

    results = _run_batch_job(client, batch_requests, "ocr-and-correct")

    for custom_id, item in lookup.items():
        base_name = item["base_name"]
        doc_output_dir = item["doc_output_dir"]
        result = results.get(custom_id)
        if not result:
            logging.error(f"[Claude] No OCR result for {base_name}; it will be skipped downstream.")
            continue

        raw_text, corrected_text = _parse_ocr_response(result["text"], base_name)
        with open(os.path.join(doc_output_dir, f"{base_name}.raw.txt"), "w", encoding="utf-8") as f:
            f.write(raw_text)
        with open(os.path.join(doc_output_dir, f"{base_name}.corrected.txt"), "w", encoding="utf-8") as f:
            f.write(corrected_text)

        _log_cost(base_name, "ocr_and_correct", claude_model, result["usage"])
        jobs[base_name] = {"doc_output_dir": doc_output_dir}

    return jobs


# === ENTITY EXTRACTION + LETTER SPLITTING ===
def run_entities_and_split_batch(jobs, model_name, api_key):
    """
    Submits one Batch API job covering both entity extraction and letter
    splitting (two requests per document) for every file in jobs.
    """
    client = Anthropic(api_key=api_key)
    claude_model = os.getenv("CLAUDE_MODEL", model_name or "claude-sonnet-5")

    batch_requests = []
    for base_name, job_info in jobs.items():
        doc_output_dir = job_info["doc_output_dir"]
        corrected_path = os.path.join(doc_output_dir, f"{base_name}.corrected.txt")
        if not os.path.exists(corrected_path):
            continue
        with open(corrected_path, "r", encoding="utf-8") as f:
            corrected_text = f.read()
        if not corrected_text.strip():
            corrected_text = "(No corrected text available.)"

        entities_prompt = f"{PROMPTS['extract_and_explain_entities']}\n\n{corrected_text}"
        split_prompt = f"{PROMPTS['split_letters']}\nText:\n{corrected_text}"

        batch_requests.append(_build_text_request(f"entities__{base_name}", entities_prompt, claude_model))
        batch_requests.append(_build_text_request(f"split__{base_name}", split_prompt, claude_model))

    if not batch_requests:
        return

    results = _run_batch_job(client, batch_requests, "entities-and-split")

    for custom_id, result in results.items():
        task, base_name = custom_id.split("__", 1)
        doc_output_dir = jobs[base_name]["doc_output_dir"]
        content = result["text"]
        if content.startswith("```"):
            content = content.strip("`").replace("json", "", 1).strip()

        if task == "entities":
            path = os.path.join(doc_output_dir, f"{base_name}.entities.json")
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = {}
            with open(path, "w", encoding="utf-8") as out:
                json.dump(parsed, out, indent=2, ensure_ascii=False)
            _log_cost(base_name, "extract_and_explain_entities", claude_model, result["usage"])

        elif task == "split":
            path = os.path.join(doc_output_dir, f"{base_name}.combined_output.json")
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = []
            with open(path, "w", encoding="utf-8") as out:
                json.dump({"letters": parsed}, out, indent=2, ensure_ascii=False)
            _log_cost(base_name, "split_letters", claude_model, result["usage"])