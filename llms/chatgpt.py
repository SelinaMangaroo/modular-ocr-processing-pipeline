# import os
# import logging
# import json
# from openai import OpenAI
# from schemas.llm_schemas import CorrectedText, EntitiesOutput, CombinedOutput, EntityExplanations

# PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts.json")
# with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
#     PROMPTS = json.load(f)

# def get_client(api_key):
#     return OpenAI(api_key=api_key)

# def correct_text(text, base_name, doc_output_dir, client, model_name, save=True) -> CorrectedText:                                                                                                                                                                                                                                    
#     logging.info(f"Correcting OCR text for: {base_name}")
    
#     response = client.chat.completions.create(
#         model=model_name,
#         messages=[
#             {"role": "system", "content": PROMPTS["correct_text"]},
#             {"role": "user", "content": text}
#         ],
#         temperature=0.0
#     )
    
#     corrected_text = response.choices[0].message.content.strip()

#     if save:
#         corrected_path = os.path.join(doc_output_dir, base_name + ".corrected.txt")
#         with open(corrected_path, "w", encoding="utf-8") as f:
#             f.write(corrected_text)
#         logging.info(f"Corrected text saved: {corrected_path}")

#     return CorrectedText(corrected_text=corrected_text)

# def extract_entities(text, base_name, doc_output_dir, client, model_name) -> EntitiesOutput:
#     logging.info(f"Extracting entities with ChatGPT for: {base_name}")
    
#     response = client.chat.completions.create(
#         model=model_name,
#         messages=[
#             {"role": "system", "content": PROMPTS["extract_entities"]},
#             {"role": "user", "content": text}
#         ],
#         temperature=0.2
#     )

#     result = response.choices[0].message.content.strip()

#     try:
#         parsed = json.loads(result)
#     except json.JSONDecodeError:
#         parsed = {"People": [], "Productions": [], "Companies": [], "Theaters": [], "Dates": []}

#     entities = EntitiesOutput(**parsed)

#     entity_path = os.path.join(doc_output_dir, base_name + ".entities.json")
#     with open(entity_path, "w", encoding="utf-8") as f:
#         f.write(entities.model_dump_json(indent=2))

#     logging.info(f"Entity extraction saved: {entity_path}")
#     return entities

# def explain_entities(entities, base_name, doc_output_dir, client, model_name) -> EntityExplanations:
#     logging.info(f"[ChatGPT] Explaining entities for {base_name}")
#     explanations = {"People": {}, "Productions": {}, "Companies": {}, "Theaters": {}}

#     for category in explanations.keys():
#         items = getattr(entities, category, [])
#         for item in items:

#             response = client.chat.completions.create(
#                 model=model_name, 
#                 messages=[
#                     {"role": "system", "content": PROMPTS["explain_entities"]},
#                     {"role": "user", "content": f"Category: {category}\nEntity: {item}"}
#                 ], temperature=0.2
#             )

#             explanation = response.choices[0].message.content.strip()
#             explanations[category][item] = explanation

#     entity_explanations = EntityExplanations(**explanations)

#     explain_path = os.path.join(doc_output_dir, base_name + ".entities_explained.json")
#     with open(explain_path, "w", encoding="utf-8") as f:
#         f.write(entity_explanations.model_dump_json(indent=2))

#     logging.info(f"[ChatGPT] Entity explanations saved: {explain_path}")
#     return entity_explanations

# def extract_page_and_split_letters(corrected_text_path, client, model_name) -> CombinedOutput:
#     try:
#         with open(corrected_text_path, "r", encoding="utf-8") as f:
#             lines = f.readlines()

#         if not lines:
#             return CombinedOutput(page_number=None, letters=[])

#         first_line = lines[0].strip()
#         text = "".join(lines)

#         try:
#             page_number = int(first_line.strip())
#         except ValueError:
#             page_number = None

#         response = client.chat.completions.create(
#             model=model_name,
#             messages=[{"role": "user", "content": PROMPTS["split_letters"] + f"\nText:\n{text}"}],
#             temperature=0
#         )

#         result = response.choices[0].message.content.strip()

#         try:
#             letters = json.loads(result)
#             if not isinstance(letters, list):
#                 letters = [text]
#         except json.JSONDecodeError:
#             logging.warning(f"Invalid JSON returned when splitting letters. Falling back to full text.")
#             letters = [text]

#         combined = CombinedOutput(page_number=page_number, letters=letters)

#         combined_path = corrected_text_path.replace(".corrected.txt", ".combined_output.json")
#         with open(combined_path, "w", encoding="utf-8") as f:
#             f.write(combined.model_dump_json(indent=2))

#         logging.info(f"Combined output saved: {combined_path}")
#         return combined
#     except Exception as e:
#         logging.error(f"Failed to extract page and split letters for {corrected_text_path}: {e}")
#         return CombinedOutput(page_number=None, letters=[])

import os
import json
import time
import logging
from openai import OpenAI
from schemas.llm_schemas import CorrectedText, EntitiesOutput, EntityExplanations, CombinedOutput

# === CONFIG ===
PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts.json")
with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
    PROMPTS = json.load(f)

BATCH_INPUT_FILE = "batch_input.jsonl"
BATCH_OUTPUT_FILE = "batch.output.jsonl"
_batch_requests = []

# === HELPERS ===
def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

def queue_batch_request(base_name, task, model, messages, temperature=0.0):
    """Add a request to the global batch queue."""
    custom_id = f"{base_name}::{task}"
    _batch_requests.append({
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": model, "messages": messages, "temperature": temperature},
    })


def submit_batch_and_get_output(client: OpenAI):
    """Submit queued batch requests, wait for completion, and download the result file."""
    if not _batch_requests:
        logging.warning("No batch requests queued.")
        return None

    with open(BATCH_INPUT_FILE, "w", encoding="utf-8") as f:
        for req in _batch_requests:
            json.dump(req, f)
            f.write("\n")

    upload = client.files.create(file=open(BATCH_INPUT_FILE, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    logging.info(f"Submitted batch: {batch.id}")

    while True:
        batch = client.batches.retrieve(batch.id)
        logging.info(f"Batch status: {batch.status}")
        if batch.status == "completed":
            break
        elif batch.status in ("failed", "expired", "cancelled"):
            logging.error(f"Batch failed: {batch.status}")
            return None
        time.sleep(30)

    try:
        output_file_id = batch.output_file_id
        if not output_file_id:
            logging.error("Batch completed but no output_file_id found.")
            return None

        output_file = client.files.retrieve_content(output_file_id)
        with open(BATCH_OUTPUT_FILE, "wb") as f:
            f.write(output_file.encode("utf-8"))
        logging.info(f"✅ Batch output downloaded: {BATCH_OUTPUT_FILE}")
        return BATCH_OUTPUT_FILE
    except Exception as e:
        logging.error(f"Failed to download batch output: {e}")
        return None

# === MAIN TASKS ===
def correct_text(text, base_name, doc_output_dir, client, model_name, save=True):
    """Queue text correction request."""
    logging.info(f"[ChatGPT] Correcting text for: {base_name}")
    os.makedirs(doc_output_dir, exist_ok=True)

    if not text or not text.strip():
        logging.warning(f"[ChatGPT] Empty OCR text for {base_name}, skipping correction.")
        text = "(No OCR text available.)"

    messages = [
        {"role": "system", "content": PROMPTS["correct_text"]},
        {"role": "user", "content": text},
    ]
    queue_batch_request(base_name, "correct_text", model_name, messages)

    corrected_path = os.path.join(doc_output_dir, f"{base_name}.corrected.txt")
    with open(corrected_path, "w", encoding="utf-8") as f:
        f.write("")  # placeholder
    return CorrectedText(corrected_text=text)

# def extract_entities(text, base_name, doc_output_dir, client, model_name):
#     """Queue entity extraction request."""
#     logging.info(f"[ChatGPT] Extracting entities for: {base_name}")
#     os.makedirs(doc_output_dir, exist_ok=True)

#     if not text or not text.strip():
#         logging.warning(f"[ChatGPT] Empty text for entity extraction: {base_name}")
#         text = "(No text available.)"

#     messages = [
#         {"role": "system", "content": PROMPTS["extract_entities"]},
#         {"role": "user", "content": text},
#     ]
#     queue_batch_request(base_name, "extract_entities", model_name, messages)

#     entity_path = os.path.join(doc_output_dir, f"{base_name}.entities.json")
#     with open(entity_path, "w", encoding="utf-8") as f:
#         json.dump({"People": [], "Productions": [], "Companies": [], "Theaters": [], "Dates": []}, f, indent=2)
#     return EntitiesOutput(People=[], Productions=[], Companies=[], Theaters=[], Dates=[])

def extract_entities(text, base_name, doc_output_dir, client, model_name):
    """Queue entity extraction + explanation request."""
    logging.info(f"[ChatGPT] Extracting + explaining entities for: {base_name}")
    os.makedirs(doc_output_dir, exist_ok=True)

    if not text or not text.strip():
        logging.warning(f"[ChatGPT] Empty text for entity extraction: {base_name}")
        text = "(No text available.)"

    messages = [
        {"role": "system", "content": PROMPTS["extract_and_explain_entities"]},
        {"role": "user", "content": text},
    ]
    queue_batch_request(base_name, "extract_entities", model_name, messages)

    # Save empty placeholder (merged with explanations now)
    entity_path = os.path.join(doc_output_dir, f"{base_name}.entities.json")
    with open(entity_path, "w", encoding="utf-8") as f:
        json.dump({
            "People": {},
            "Productions": {},
            "Companies": {},
            "Theaters": {},
            "Dates": {}
        }, f, indent=2)
    
    return EntityExplanations(
        People={}, Productions={}, Companies={}, Theaters={}, Dates={}
    )

def explain_entities(entities, base_name, doc_output_dir, client, model_name):
    """Queue a grouped explanation for all entities."""
    logging.info(f"[ChatGPT] Queuing grouped explanation for: {base_name}")
    os.makedirs(doc_output_dir, exist_ok=True)

    grouped_entities = entities.model_dump()

    # If there's nothing to explain, skip
    if all(len(v) == 0 for v in grouped_entities.values()):
        logging.warning(f"[ChatGPT] No entities to explain for {base_name}")
        return EntityExplanations(People={}, Productions={}, Companies={}, Theaters={}, Dates={})

    messages = [
        {"role": "system", "content": PROMPTS["explain_entities"]},
        {"role": "user", "content": json.dumps(grouped_entities, indent=2)},
    ]
    queue_batch_request(base_name, "explain_entities", model_name, messages)

    # Create placeholder file
    explain_path = os.path.join(doc_output_dir, f"{base_name}.entities_explained.json")
    with open(explain_path, "w", encoding="utf-8") as f:
        json.dump({}, f, indent=2)

    return EntityExplanations(People={}, Productions={}, Companies={}, Theaters={}, Dates={})

def extract_page_and_split_letters(corrected_text_path, client, model_name):
    """Queue letter splitting request."""
    logging.info(f"[ChatGPT] Splitting letters for: {corrected_text_path}")
    os.makedirs(os.path.dirname(corrected_text_path), exist_ok=True)

    try:
        with open(corrected_text_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except FileNotFoundError:
        text = ""

    if not text:
        logging.warning(f"[ChatGPT] No corrected text found in {corrected_text_path}")
        text = "(Empty document.)"

    messages = [
        {"role": "user", "content": PROMPTS["split_letters"] + f"\nText:\n{text}"},
    ]
    base_name = os.path.basename(corrected_text_path).replace(".corrected.txt", "")
    queue_batch_request(base_name, "split_letters", model_name, messages)

    combined_path = corrected_text_path.replace(".corrected.txt", ".combined_output.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write("[]")
    return CombinedOutput(page_number=None, letters=[])

# === HYDRATION ===
def hydrate_batch_output(output_dir="output", batch_output_file=BATCH_OUTPUT_FILE):
    """Hydrate all model outputs from the batch results into their respective files."""
    if not os.path.exists(batch_output_file):
        logging.error(f"Batch output file not found: {batch_output_file}")
        return

    logging.info(f"[Hydration] Reading {batch_output_file}")
    hydrated_count = 0
    explanations_map = {}

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

            parts = custom_id.split("::")
            base_name = parts[0]
            task = parts[1]

            doc_dir = os.path.join(output_dir, base_name)
            os.makedirs(doc_dir, exist_ok=True)

            if task == "correct_text":
                path = os.path.join(doc_dir, f"{base_name}.corrected.txt")
                with open(path, "w", encoding="utf-8") as out:
                    out.write(content)

            # elif task == "extract_entities":
            #     path = os.path.join(doc_dir, f"{base_name}.entities.json")
            #     try:
            #         parsed = json.loads(content)
            #     except json.JSONDecodeError:
            #         parsed = {"People": [], "Productions": [], "Companies": [], "Theaters": [], "Dates": []}
            #     with open(path, "w", encoding="utf-8") as out:
            #         json.dump(parsed, out, indent=2)
            
            elif task == "extract_entities":
                path = os.path.join(doc_dir, f"{base_name}.entities.json")
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    parsed = {"People": {}, "Productions": {}, "Companies": {}, "Theaters": {}, "Dates": {}}

                with open(path, "w", encoding="utf-8") as out:
                    json.dump(parsed, out, indent=2)

                logging.info(f"[Hydration] Wrote entities (w/ explanations) for {base_name}")

            elif task == "split_letters":
                path = os.path.join(doc_dir, f"{base_name}.combined_output.json")

                # Try parsing the returned content
                try:
                    parsed_letters = json.loads(content)
                except json.JSONDecodeError:
                    parsed_letters = []

                # Read page number from .corrected.txt file
                corrected_path = os.path.join(doc_dir, f"{base_name}.corrected.txt")
                page_number = None
                try:
                    with open(corrected_path, "r", encoding="utf-8") as f:
                        text = f.read()
                        import re
                        match = re.search(r"\b(\d{1,4})\b", text)
                        if match:
                            page_number = int(match.group(1))
                except Exception as e:
                    logging.warning(f"[Hydration] Could not extract page number for {base_name}: {e}")

                # Write the hydrated output
                with open(path, "w", encoding="utf-8") as out:
                    json.dump(
                        {
                            "page_number": page_number,
                            "letters": parsed_letters,
                        },
                        out,
                        indent=2,
                        ensure_ascii=False,
                    )

                logging.info(f"[Hydration] Wrote combined_output with page_number for {base_name}")

            elif task == "explain_entities":
                path = os.path.join(doc_dir, f"{base_name}.entities_explained.json")
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    parsed = {}
                with open(path, "w", encoding="utf-8") as out:
                    json.dump(parsed, out, indent=2)

            hydrated_count += 1
            logging.info(f"[Hydration] Wrote {task} for {base_name}")

    # === After loop, write .entities_explained.json for each doc ===
    for base_name, cat_map in explanations_map.items():
        doc_dir = os.path.join(output_dir, base_name)
        path = os.path.join(doc_dir, f"{base_name}.entities_explained.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cat_map, f, indent=2)
        logging.info(f"[Hydration] Wrote explanations: {path}")

    logging.info(f"[Hydration] Hydrated {hydrated_count} files total.")