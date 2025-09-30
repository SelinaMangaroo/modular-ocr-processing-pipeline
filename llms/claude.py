import os
import logging
import json
from anthropic import Anthropic
from schemas.llm_schemas import CorrectedText, EntitiesOutput, CombinedOutput, EntityExplanations

PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts.json")
with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
    PROMPTS = json.load(f)

def get_client(api_key=None):
    return Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

def correct_text(text, base_name, output_dir, client, model_name) -> CorrectedText:
    logging.info(f"[Claude] Correcting OCR text for: {base_name}")

    resp = client.messages.create(
        model=model_name, max_tokens=4096,
        messages=[{"role": "user", "content": PROMPTS["correct_text"] + f"\nText:\n{text}"}]
    )

    corrected_text = resp.content[0].text.strip()
    corrected_path = os.path.join(output_dir, base_name + ".corrected.txt")
    with open(corrected_path, "w", encoding="utf-8") as f:
        f.write(corrected_text)

    return CorrectedText(corrected_text=corrected_text)

def explain_entities(entities, base_name, output_dir, client, model_name) -> EntityExplanations:
    logging.info(f"[Claude] Explaining entities for {base_name}")
    explanations = {"People": {}, "Productions": {}, "Companies": {}, "Theaters": {}}

    for category in explanations.keys():
        items = getattr(entities, category, [])
        for item in items:

            resp = client.messages.create(
                model=model_name, max_tokens=4096,
                messages=[
                    {"role": "system", "content": PROMPTS["explain_entities"]},
                    {"role": "user", "content": f"Category: {category}\nEntity: {item}"}
                ]
            )

            explanation = resp.content[0].text.strip()
            explanations[category][item] = explanation

    entity_explanations = EntityExplanations(**explanations)

    explain_path = os.path.join(output_dir, base_name + ".entities_explained.json")
    with open(explain_path, "w", encoding="utf-8") as f:
        f.write(entity_explanations.model_dump_json(indent=2))

    logging.info(f"[Claude] Entity explanations saved: {explain_path}")
    return entity_explanations

def extract_page_and_split_letters(corrected_text_path, client, model_name) -> CombinedOutput:
    with open(corrected_text_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return CombinedOutput(page_number=None, letters=[])

    # first line as page number
    first_line = lines[0].strip()
    full_text = "".join(lines)
    try:
        page_number = int(first_line)
    except ValueError:
        page_number = None

    resp = client.messages.create(
        model=model_name,
        max_tokens=4096,
        messages=[{"role": "user", "content": PROMPTS["split_letters"] + f"\nText:\n{full_text}"}]
    )

    raw_result = resp.content[0].text.strip()

    try:
        letters = json.loads(raw_result)
        if not isinstance(letters, list):
            letters = [full_text]
    except json.JSONDecodeError:
        logging.warning("[Claude] Invalid JSON when splitting letters, falling back to full text.")
        letters = [full_text]

    combined = CombinedOutput(page_number=page_number, letters=letters)

    combined_path = corrected_text_path.replace(".corrected.txt", ".combined_output.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write(combined.model_dump_json(indent=2))

    logging.info(f"[Claude] Combined output saved: {combined_path}")
    return combined