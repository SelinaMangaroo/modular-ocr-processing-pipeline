import os
import json
import time
import base64
import logging
import uuid
import threading
from openai import OpenAI
from utils.helpers import get_file_paths, resize_image

# === CONFIG ===
PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts.json")
with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
    PROMPTS = json.load(f)

OCR_PROMPT_KEY = "gemini_ocr_and_correct"
RAW_MARKER = "=== RAW TRANSCRIPTION ==="
CORRECTED_MARKER = "=== SPELLING-CORRECTED ==="

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
BATCH_FILES_DIR = os.path.join(BASE_DIR, "batchfiles", "chatgpt_v2")
os.makedirs(BATCH_FILES_DIR, exist_ok=True)

POLL_INTERVAL_SECONDS = int(os.getenv("CHATGPT_BATCH_POLL_INTERVAL", "20"))

CHATGPT_PRICING = {
    "gpt-5.6-terra": {
        "short": {"input": 1.00, "cached_input": 0.10, "output": 6.00},
        "long": {"input": 2.00, "cached_input": 0.20, "output": 9.00},
        "long_context_threshold": 200_000,
    },
}
DEFAULT_PRICING_KEY = "gpt-5.6-terra"


def _get_pricing(model_name):
    pricing = CHATGPT_PRICING.get(model_name)
    if not pricing:
        logging.warning(
            f"[ChatGPT] No pricing entry for model '{model_name}'; "
            f"using {DEFAULT_PRICING_KEY} rates as an estimate. Verify actual pricing."
        )
        pricing = CHATGPT_PRICING[DEFAULT_PRICING_KEY]
    return pricing


_cost_lock = threading.Lock()
_run_total_cost = 0.0
_run_total_tokens = 0


def _estimate_cost(usage, model_name):
    pricing = _get_pricing(model_name)
    tier = "long" if usage["input"] > pricing["long_context_threshold"] else "short"
    rates = pricing[tier]

    billable_input = max(usage["input"] - usage["cached"], 0)

    cost = (
        billable_input * rates["input"]
        + usage["cached"] * rates["cached_input"]
        + usage["output"] * rates["output"]
    ) / 1_000_000

    return cost


def _log_cost(base_name, task, model_name, usage):
    global _run_total_cost, _run_total_tokens
    cost = _estimate_cost(usage, model_name)

    with _cost_lock:
        _run_total_cost += cost
        _run_total_tokens += usage["total"]

    logging.info(
        f"[ChatGPT] Cost for {base_name}::{task} — "
        f"{usage['total']} tokens (in={usage['input']}, out={usage['output']}, cached={usage['cached']}) "
        f"≈ ${cost:.4f} [batch rate]"
    )


def print_cost_summary():
    logging.info(
        f"[ChatGPT] Run totals — {_run_total_tokens} tokens, ≈ ${_run_total_cost:.4f} estimated (batch rate)."
    )


def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def _parse_ocr_response(response_text, base_name):
    if RAW_MARKER in response_text and CORRECTED_MARKER in response_text:
        raw_part = response_text.split(RAW_MARKER, 1)[1].split(CORRECTED_MARKER, 1)[0].strip()
        corrected_part = response_text.split(CORRECTED_MARKER, 1)[1].strip()
        return raw_part, corrected_part

    logging.warning(
        f"[ChatGPT] Response for {base_name} did not match expected format; "
        "saving full response as both raw and corrected for manual review."
    )
    return response_text.strip(), response_text.strip()


def _extract_usage(usage_dict):
    usage_dict = usage_dict or {}
    prompt_tokens = usage_dict.get("prompt_tokens", 0) or 0
    completion_tokens = usage_dict.get("completion_tokens", 0) or 0
    cached_tokens = (usage_dict.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    total_tokens = usage_dict.get("total_tokens", 0) or (prompt_tokens + completion_tokens)
    return {
        "input": prompt_tokens,
        "output": completion_tokens,
        "cached": cached_tokens,
        "total": total_tokens,
    }


def _build_text_body(prompt_text):
    return {"messages": [{"role": "user", "content": prompt_text}]}


def _build_ocr_body(prompt_text, data_b64, media_type, block_kind):
    if block_kind == "image":
        file_block = {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data_b64}"}}
    else:
        file_block = {
            "type": "file",
            "file": {"filename": "document.pdf", "file_data": f"data:{media_type};base64,{data_b64}"},
        }
    return {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                file_block,
            ],
        }]
    }


def _run_batch_job(client, model_name, items, display_name):
    if not items:
        return {}

    timestamp = time.strftime("%m%d%Y_%H%M%S")
    uid = uuid.uuid4().hex[:3]
    jsonl_path = os.path.join(BATCH_FILES_DIR, f"{display_name}_{timestamp}_{uid}.jsonl")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in items:
            body = dict(item["body"])
            body["model"] = model_name
            line = {
                "custom_id": item["custom_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    uploaded = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")

    for attempt in range(5):
        try:
            client.files.retrieve(uploaded.id)
            break
        except Exception as e:
            logging.warning(f"[ChatGPT] Waiting for uploaded file to propagate ({attempt + 1}/5)... {e}")
            time.sleep(2)
    else:
        logging.error(f"[ChatGPT] Uploaded file {uploaded.id} never became retrievable; aborting batch.")
        return {}

    logging.info("[ChatGPT] Waiting for file to propagate to batch validation service...")
    time.sleep(15)

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    logging.info(f"[ChatGPT] Submitted batch job '{display_name}' ({len(items)} requests): {batch.id}")

    while True:
        batch = client.batches.retrieve(batch.id)
        logging.info(f"[ChatGPT] Batch '{display_name}' status: {batch.status}")
        if batch.status == "completed":
            break
        elif batch.status in ("failed", "expired", "cancelled"):
            logging.error(f"[ChatGPT] Batch '{display_name}' did not succeed: {batch.status}")
            return {}
        time.sleep(POLL_INTERVAL_SECONDS)

    results = {}
    output_file_id = batch.output_file_id
    if not output_file_id:
        logging.error(f"[ChatGPT] Batch '{display_name}' completed but no output file found.")
        return {}

    content = client.files.content(output_file_id).text
    for line in content.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        custom_id = record.get("custom_id")
        if not custom_id:
            continue
        if record.get("error"):
            logging.error(f"[ChatGPT] Batch request failed for {custom_id}: {record['error']}")
            continue
        response = record.get("response", {})
        body = response.get("body", {})
        choices = body.get("choices", [{}])
        text = (choices[0].get("message", {}).get("content", "") or "").strip()
        usage = _extract_usage(body.get("usage"))
        results[custom_id] = {"text": text, "usage": usage}

    logging.info(f"[ChatGPT] Batch '{display_name}' complete — {len(results)}/{len(items)} succeeded.")
    return results


def prepare_and_upload(filename, tmp_dir, input_dir, output_dir, image_magick_command, client=None):
    paths = get_file_paths(filename, tmp_dir, input_dir, output_dir)
    base_name = paths["base_name"]
    doc_output_dir = paths["doc_output_dir"]
    os.makedirs(doc_output_dir, exist_ok=True)

    raw_path = os.path.join(doc_output_dir, f"{base_name}.raw.txt")
    corrected_path = os.path.join(doc_output_dir, f"{base_name}.corrected.txt")

    skip_ocr = os.getenv("SKIP_OCR_IF_EXISTS", "false").lower() == "true"
    if skip_ocr and os.path.exists(raw_path) and os.path.exists(corrected_path):
        logging.info(f"[ChatGPT] Skipping OCR for {base_name} (raw + corrected already exist).")
        return {"base_name": base_name, "doc_output_dir": doc_output_dir, "skip": True}

    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        with open(paths["path_to_file"], "rb") as f:
            data_b64 = base64.b64encode(f.read()).decode("utf-8")
        media_type = "application/pdf"
        block_kind = "file"
    else:
        resized_path = os.path.join(tmp_dir, f"{base_name}_resized.jpg")
        prepared_file = resize_image(paths["path_to_file"], resized_path, image_magick_command, filename)
        with open(prepared_file, "rb") as f:
            data_b64 = base64.b64encode(f.read()).decode("utf-8")
        media_type = "image/jpeg"
        block_kind = "image"

    return {
        "base_name": base_name,
        "doc_output_dir": doc_output_dir,
        "skip": False,
        "data": data_b64,
        "media_type": media_type,
        "block_kind": block_kind,
    }


def run_ocr_batch(prepared_items, model_name, api_key):
    client = OpenAI(api_key=api_key)
    gpt_model = os.getenv("OPENAI_MODEL", model_name or "gpt-5.6-terra")

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
        custom_id = f"ocr__{base_name}"
        body = _build_ocr_body(PROMPTS[OCR_PROMPT_KEY], item["data"], item["media_type"], item["block_kind"])
        batch_items.append({"custom_id": custom_id, "body": body})
        lookup[custom_id] = item

    if not batch_items:
        return jobs

    results = _run_batch_job(client, gpt_model, batch_items, "ocr-and-correct")

    for custom_id, item in lookup.items():
        base_name = item["base_name"]
        doc_output_dir = item["doc_output_dir"]
        result = results.get(custom_id)
        if not result:
            logging.error(f"[ChatGPT] No OCR result for {base_name}; it will be skipped downstream.")
            continue

        raw_text, corrected_text = _parse_ocr_response(result["text"], base_name)
        with open(os.path.join(doc_output_dir, f"{base_name}.raw.txt"), "w", encoding="utf-8") as f:
            f.write(raw_text)
        with open(os.path.join(doc_output_dir, f"{base_name}.corrected.txt"), "w", encoding="utf-8") as f:
            f.write(corrected_text)

        _log_cost(base_name, "ocr_and_correct", gpt_model, result["usage"])
        jobs[base_name] = {"doc_output_dir": doc_output_dir}

    return jobs


def run_entities_and_split_batch(jobs, model_name, api_key):
    client = OpenAI(api_key=api_key)
    gpt_model = os.getenv("OPENAI_MODEL", model_name or "gpt-5.6-terra")

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
        split_prompt = f"{PROMPTS['split_letters']}\nText:\n{corrected_text}"

        batch_items.append({"custom_id": f"entities__{base_name}", "body": _build_text_body(entities_prompt)})
        batch_items.append({"custom_id": f"split__{base_name}", "body": _build_text_body(split_prompt)})

    if not batch_items:
        return

    results = _run_batch_job(client, gpt_model, batch_items, "entities-and-split")

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
            _log_cost(base_name, "extract_and_explain_entities", gpt_model, result["usage"])

        elif task == "split":
            path = os.path.join(doc_output_dir, f"{base_name}.combined_output.json")
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = []
            with open(path, "w", encoding="utf-8") as out:
                json.dump({"letters": parsed}, out, indent=2, ensure_ascii=False)
            _log_cost(base_name, "split_letters", gpt_model, result["usage"])
