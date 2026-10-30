import os
import json
import time
import uuid
import logging
import threading
from google import genai
from google.genai import types
from utils.helpers import get_file_paths, resize_image

# === CONFIG ===
PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts.json")
with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
    PROMPTS = json.load(f)


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
BATCH_FILES_DIR = os.path.join(BASE_DIR, "batchfiles", "gemini")

# Every submitted batch job gets a small record written here the moment it's
# created, BEFORE polling starts. If the whole process dies for any reason
# (crash, killed SSH session, server reboot) while a job is still in flight,
# this record survives on disk — recover_pending_batches.py sweeps these up
# and finishes writing outputs for anything that completed while nobody was
# watching. One file per job (not a shared log) so concurrent main.py
# processes never collide writing to the same file.
PENDING_JOBS_DIR = os.path.join(BASE_DIR, "batchfiles", "gemini", "pending")
os.makedirs(PENDING_JOBS_DIR, exist_ok=True)


def _pending_job_path(job_name):
    safe_name = job_name.replace("/", "_")
    return os.path.join(PENDING_JOBS_DIR, f"{safe_name}.json")


def _record_pending_job(job_name, display_name, model_name, output_dir):
    record = {
        "job_name": job_name,
        "display_name": display_name,
        "model": model_name,
        "output_dir": output_dir,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(_pending_job_path(job_name), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)


def _clear_pending_job(job_name):
    path = _pending_job_path(job_name)
    if os.path.exists(path):
        os.remove(path)
os.makedirs(BATCH_FILES_DIR, exist_ok=True)

# How often to poll a running batch job, in seconds. Google's own examples
# use 10-30s; batch jobs can legitimately take minutes even for small test
# batches since they're queued, not instant.
POLL_INTERVAL_SECONDS = int(os.getenv("GEMINI_BATCH_POLL_INTERVAL", "20"))

COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
}

# === COST TRACKING ===
# Pricing per 1M tokens, in USD, at STANDARD (non-batch) rates. The Batch API
# applies a flat 50% discount to both input and output — applied below in
# _estimate_cost rather than duplicating a second pricing table. Verified
# against Google's published pricing as of Aug 2026; preview model pricing
# has shifted before and may shift again — check https://ai.google.dev/pricing
# if costs look off. "long_context_threshold" is the token count above which
# the higher input/output rates apply.
GEMINI_PRICING = {
    "gemini-3.1-pro-preview": {
        "input": 2.00,
        "output": 12.00,
        "cached_input": 0.20,
        "long_context_threshold": 200_000,
        "input_long": 4.00,
        "output_long": 18.00,
    },
}
DEFAULT_PRICING_KEY = "gemini-3.1-pro-preview"
BATCH_DISCOUNT = 0.5
# Note: explicit context caching also carries a separate storage cost
# ($4.50 per 1M tokens/hour as of Aug 2026) not modeled here, since this
# pipeline doesn't use explicit caching (no cached_content resource is
# created) — cached_content_token_count should normally be 0. If explicit
# caching gets added later for cost savings, this needs revisiting.

_cost_lock = threading.Lock()
_run_total_cost = 0.0
_run_total_tokens = 0


def _get_pricing(model_name):
    pricing = GEMINI_PRICING.get(model_name)
    if not pricing:
        logging.warning(
            f"[Gemini] No pricing entry for model '{model_name}'; "
            f"using {DEFAULT_PRICING_KEY} rates as an estimate. Verify actual pricing."
        )
        pricing = GEMINI_PRICING[DEFAULT_PRICING_KEY]
    return pricing


def _estimate_cost(usage, model_name):
    """Estimates cost in USD for one batch request. Thinking tokens are
    billed at the output rate. Cached input tokens get the discounted
    cached rate — and per Google's pricing page, cached-token pricing is
    the SAME in batch mode as standard mode (not further discounted), so
    the 50% batch discount below applies only to input/output, not cached."""
    pricing = _get_pricing(model_name)

    billable_input = max(usage["input"] - usage["cached"], 0)
    over_threshold = usage["input"] > pricing["long_context_threshold"]

    input_rate = pricing["input_long"] if over_threshold else pricing["input"]
    output_rate = pricing["output_long"] if over_threshold else pricing["output"]

    output_tokens_billed = usage["output"] + usage["thinking"]

    discounted_cost = (
        (billable_input * input_rate) + (output_tokens_billed * output_rate)
    ) * BATCH_DISCOUNT

    cached_cost = usage["cached"] * pricing["cached_input"]  # not discounted in batch mode

    return (discounted_cost + cached_cost) / 1_000_000


def _log_cost(base_name, task, model_name, usage):
    global _run_total_cost, _run_total_tokens
    cost = _estimate_cost(usage, model_name)

    with _cost_lock:
        _run_total_cost += cost
        _run_total_tokens += usage["total"]

    logging.info(
        f"[Gemini] Cost for {base_name}::{task} — "
        f"{usage['total']} tokens (in={usage['input']}, out={usage['output']}, "
        f"thinking={usage['thinking']}, cached={usage['cached']}) "
        f"≈ ${cost:.4f} [batch rate, 50% off]"
    )


def print_cost_summary():
    """Logs the running total for this pipeline execution. Safe to call
    even if no Gemini calls were made (totals will just be zero)."""
    logging.info(
        f"[Gemini] Run totals — {_run_total_tokens} tokens, ≈ ${_run_total_cost:.4f} estimated (batch rate)."
    )

def get_run_totals():
    """Returns the running token/cost totals for this process's lifetime, for
    external tooling (e.g. a cross-provider comparison script) to read without
    scraping log files."""
    return {"total_tokens": _run_total_tokens, "total_cost": _run_total_cost}


# === HELPERS ===
def get_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


# (Raw/corrected marker parsing removed — the prompt now returns only the
# corrected text directly, since raw.txt is no longer saved. If output ever
# looks malformed, check the prompt in prompts.json first.)


# Gemini 3.1 Pro defaults to thinking_level "high" if unspecified — the most
# expensive setting, and since thinking tokens bill at the OUTPUT rate, this
# is usually the single biggest cost driver. Thinking can't be fully disabled
# on Gemini 3 Pro models, but can be dropped to "low"/"minimal". Configurable
# per task since OCR (harder task) may need more reasoning than entity
# extraction/letter splitting (more mechanical tasks) — test both against
# output quality before trusting a lower level at scale.
OCR_THINKING_LEVEL = os.getenv("GEMINI_OCR_THINKING_LEVEL", "low")
TEXT_THINKING_LEVEL = os.getenv("GEMINI_TEXT_THINKING_LEVEL", "low")


def _build_text_request(prompt_text, thinking_level=TEXT_THINKING_LEVEL):
    request = {"contents": [{"role": "user", "parts": [{"text": prompt_text}]}]}
    if thinking_level:
        request["generation_config"] = {"thinking_config": {"thinking_level": thinking_level}}
    return request


def _build_ocr_request(prompt_text, file_uri, mime_type, thinking_level=OCR_THINKING_LEVEL):
    request = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt_text},
                {"file_data": {"file_uri": file_uri, "mime_type": mime_type}},
            ],
        }]
    }
    if thinking_level:
        request["generation_config"] = {"thinking_config": {"thinking_level": thinking_level}}
    return request


def _extract_text_from_response(response):
    try:
        parts = response["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError, TypeError):
        return ""


def _extract_usage_from_dict(u):
    """Batch results use the classic generateContent usageMetadata schema
    (camelCase, since these come back as raw JSON, not SDK objects)."""
    u = u or {}
    return {
        "input": u.get("promptTokenCount", 0) or 0,
        "cached": u.get("cachedContentTokenCount", 0) or 0,
        "thinking": u.get("thoughtsTokenCount", 0) or 0,
        "output": u.get("candidatesTokenCount", 0) or 0,
        "total": u.get("totalTokenCount", 0) or 0,
    }


# === BATCH JOB EXECUTION (shared by OCR+correction and entities+split) ===
def _run_batch_job(client, model_name, items, display_name, output_dir=None):
    """
    items: list of {"key": str, "request": dict} (GenerateContentRequest-shaped)
    Returns: dict mapping key -> {"text": str, "usage": dict} for succeeded
    requests. Failed individual requests are logged and simply absent from
    the returned dict — callers should treat a missing key as "this one failed".
    """
    if not items:
        return {}

    timestamp = time.strftime("%m%d%Y_%H%M%S")
    uid = uuid.uuid4().hex[:3]
    jsonl_path = os.path.join(BATCH_FILES_DIR, f"{display_name}_{timestamp}_{uid}.jsonl")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps({"key": item["key"], "request": item["request"]}, ensure_ascii=False) + "\n")

    uploaded = client.files.upload(
        file=jsonl_path,
        config=types.UploadFileConfig(display_name=display_name, mime_type="jsonl"),
    )

    batch_job = client.batches.create(
        model=model_name,
        src=uploaded.name,
        config={"display_name": display_name},
    )
    logging.info(f"[Gemini] Submitted batch job '{display_name}' ({len(items)} requests): {batch_job.name}")

    # Record BEFORE polling starts, so this job is recoverable even if this
    # whole process dies while waiting (crash, killed SSH session, reboot).
    # See recover_pending_batches.py.
    _record_pending_job(batch_job.name, display_name, model_name, output_dir)

    consecutive_poll_failures = 0
    while batch_job.state.name not in COMPLETED_STATES:
        time.sleep(POLL_INTERVAL_SECONDS)
        try:
            batch_job = client.batches.get(name=batch_job.name)
            consecutive_poll_failures = 0
        except Exception as e:
            consecutive_poll_failures += 1
            # No cap: a transient error costs only time, not money or data,
            # so we keep retrying for as long as this process stays alive.
            logging.warning(
                f"[Gemini] Transient error polling batch '{display_name}' "
                f"(consecutive failure #{consecutive_poll_failures}): {e}. "
                f"Job {batch_job.name} is recorded and will be retried."
            )
            continue
        logging.info(f"[Gemini] Batch '{display_name}' status: {batch_job.state.name}")

    if batch_job.state.name != "JOB_STATE_SUCCEEDED":
        logging.error(f"[Gemini] Batch '{display_name}' did not succeed: {batch_job.state.name}")
        if getattr(batch_job, "error", None):
            logging.error(f"[Gemini] Batch '{display_name}' error detail: {batch_job.error}")
        _clear_pending_job(batch_job.name)
        return {}

    results = {}
    result_file_name = getattr(batch_job.dest, "file_name", None) if batch_job.dest else None
    if not result_file_name:
        logging.error(f"[Gemini] Batch '{display_name}' succeeded but no result file found.")
        _clear_pending_job(batch_job.name)
        return {}

    # This one-time download can hit the same kind of transient server error
    # the polling loop can — retry indefinitely rather than losing a
    # successfully-completed (and already paid for) job to a momentary 503.
    download_attempt = 0
    content = None
    while content is None:
        try:
            content = client.files.download(file=result_file_name).decode("utf-8")
        except Exception as e:
            download_attempt += 1
            logging.warning(
                f"[Gemini] Transient error downloading results for batch '{display_name}' "
                f"(attempt #{download_attempt}): {e}. Job {batch_job.name} succeeded and is "
                f"recorded — will keep retrying the download."
            )
            time.sleep(POLL_INTERVAL_SECONDS)
    for line in content.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = record.get("key")
        if not key:
            continue
        if record.get("error"):
            logging.error(f"[Gemini] Batch request failed for {key}: {record['error']}")
            continue
        response = record.get("response")
        if not response:
            continue
        text = _extract_text_from_response(response)
        usage = _extract_usage_from_dict(response.get("usageMetadata"))
        results[key] = {"text": text, "usage": usage}

    logging.info(f"[Gemini] Batch '{display_name}' complete — {len(results)}/{len(items)} succeeded.")
    _clear_pending_job(batch_job.name)
    return results


# === OCR + CORRECTION ===
def prepare_and_upload(filename, tmp_dir, input_dir, output_dir, image_magick_command, client):
    """
    Resolves paths, checks skip logic, resizes/prepares the image (or passes
    PDFs through untouched), and uploads it to Gemini's File API. Does NOT
    call the model — that happens later, once, for the whole batch.
    """
    paths = get_file_paths(filename, tmp_dir, input_dir, output_dir)
    base_name = paths["base_name"]
    doc_output_dir = paths["doc_output_dir"]
    os.makedirs(doc_output_dir, exist_ok=True)

    corrected_path = os.path.join(doc_output_dir, f"{base_name}.corrected.txt")

    skip_ocr = os.getenv("SKIP_OCR_IF_EXISTS", "false").lower() == "true"
    if skip_ocr and os.path.exists(corrected_path):
        logging.info(f"[Gemini] Skipping OCR for {base_name} (corrected text already exists).")
        return {"base_name": base_name, "doc_output_dir": doc_output_dir, "skip": True}

    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        prepared_file = paths["path_to_file"]
        mime_type = "application/pdf"
    else:
        resized_path = os.path.join(tmp_dir, f"{base_name}_resized.jpg")
        prepared_file = resize_image(paths["path_to_file"], resized_path, image_magick_command, filename)
        mime_type = "image/jpeg"

    uploaded_file = client.files.upload(file=prepared_file)
    request = _build_ocr_request(PROMPTS["gemini_transcribe"], uploaded_file.uri, mime_type)

    return {
        "base_name": base_name,
        "doc_output_dir": doc_output_dir,
        "skip": False,
        "key": f"ocr::{base_name}",
        "request": request,
    }


def run_ocr_batch(prepared_items, model_name, api_key):
    """
    Submits one Batch API job covering OCR + correction for every non-skipped
    file in prepared_items, waits for it to complete, and writes
    .corrected.txt for each. (Raw, uncorrected OCR text is not saved — only
    the final corrected version.) Returns a jobs dict (base_name ->
    {doc_output_dir}) for every file that's ready for the next stage,
    whether it was skipped, newly processed, or (if it failed) simply absent.
    """
    client = genai.Client(api_key=api_key)
    gemini_model = os.getenv("GEMINI_MODEL", model_name or "gemini-3.1-pro-preview")

    jobs = {}
    batch_items = []
    lookup = {}

    for item in prepared_items:
        if not item:
            continue
        base_name = item["base_name"]
        if item.get("skip"):
            jobs[base_name] = {"doc_output_dir": item["doc_output_dir"]}
            continue
        batch_items.append({"key": item["key"], "request": item["request"]})
        lookup[item["key"]] = item

    if not batch_items:
        return jobs

    output_dir = os.path.dirname(next(iter(lookup.values()))["doc_output_dir"])
    results = _run_batch_job(client, gemini_model, batch_items, "ocr-and-correct", output_dir=output_dir)

    for key, item in lookup.items():
        base_name = item["base_name"]
        doc_output_dir = item["doc_output_dir"]
        result = results.get(key)
        if not result:
            logging.error(f"[Gemini] No OCR result for {base_name}; it will be skipped downstream.")
            continue

        corrected_text = result["text"].strip()
        with open(os.path.join(doc_output_dir, f"{base_name}.corrected.txt"), "w", encoding="utf-8") as f:
            f.write(corrected_text)

        _log_cost(base_name, "ocr_and_correct", gemini_model, result["usage"])
        jobs[base_name] = {"doc_output_dir": doc_output_dir}

    return jobs


# === ENTITY EXTRACTION + LETTER SPLITTING ===
def run_entities_and_split_batch(jobs, model_name, api_key):
    """
    Submits one Batch API job covering entity extraction for every file in
    jobs, writing .entities.json. Letter/section splitting was removed from
    the pipeline (no .combined_output.json is produced anymore) — the
    function name is kept as-is since main.py calls it generically across
    all consolidated providers.
    """
    client = genai.Client(api_key=api_key)
    gemini_model = os.getenv("GEMINI_MODEL", model_name or "gemini-3.1-pro-preview")

    batch_items = []
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
        batch_items.append({"key": f"entities::{base_name}", "request": _build_text_request(entities_prompt)})

    if not batch_items:
        return

    output_dir = os.path.dirname(next(iter(jobs.values()))["doc_output_dir"])
    results = _run_batch_job(client, gemini_model, batch_items, "entities", output_dir=output_dir)

    for key, result in results.items():
        task, base_name = key.split("::", 1)
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
            _log_cost(base_name, "extract_and_explain_entities", gemini_model, result["usage"])