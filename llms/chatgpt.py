import os
import logging
import json
from openai import OpenAI
from schemas.llm_schemas import CorrectedText, EntitiesOutput, CombinedOutput, EntityExplanations

PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts.json")
with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
    PROMPTS = json.load(f)

def get_client(api_key):
    return OpenAI(api_key=api_key)

def correct_text(text, base_name, doc_output_dir, client, model_name, save=True) -> CorrectedText:                                                                                                                                                                                                                                    
    logging.info(f"Correcting OCR text for: {base_name}")
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": PROMPTS["correct_text"]},
            {"role": "user", "content": text}
        ],
        temperature=0.0
    )
    
    corrected_text = response.choices[0].message.content.strip()

    if save:
        corrected_path = os.path.join(doc_output_dir, base_name + ".corrected.txt")
        with open(corrected_path, "w", encoding="utf-8") as f:
            f.write(corrected_text)
        logging.info(f"Corrected text saved: {corrected_path}")

    return CorrectedText(corrected_text=corrected_text)

def extract_entities(text, base_name, doc_output_dir, client, model_name) -> EntitiesOutput:
    logging.info(f"Extracting entities with ChatGPT for: {base_name}")
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": PROMPTS["extract_entities"]},
            {"role": "user", "content": text}
        ],
        temperature=0.2
    )

    result = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        parsed = {"People": [], "Productions": [], "Companies": [], "Theaters": [], "Dates": []}

    entities = EntitiesOutput(**parsed)

    entity_path = os.path.join(doc_output_dir, base_name + ".entities.json")
    with open(entity_path, "w", encoding="utf-8") as f:
        f.write(entities.model_dump_json(indent=2))

    logging.info(f"Entity extraction saved: {entity_path}")
    return entities

def explain_entities(entities, base_name, doc_output_dir, client, model_name) -> EntityExplanations:
    logging.info(f"[ChatGPT] Explaining entities for {base_name}")
    explanations = {"People": {}, "Productions": {}, "Companies": {}, "Theaters": {}}

    for category in explanations.keys():
        items = getattr(entities, category, [])
        for item in items:

            response = client.chat.completions.create(
                model=model_name, 
                messages=[
                    {"role": "system", "content": PROMPTS["explain_entities"]},
                    {"role": "user", "content": f"Category: {category}\nEntity: {item}"}
                ], temperature=0.2
            )

            explanation = response.choices[0].message.content.strip()
            explanations[category][item] = explanation

    entity_explanations = EntityExplanations(**explanations)

    explain_path = os.path.join(doc_output_dir, base_name + ".entities_explained.json")
    with open(explain_path, "w", encoding="utf-8") as f:
        f.write(entity_explanations.model_dump_json(indent=2))

    logging.info(f"[ChatGPT] Entity explanations saved: {explain_path}")
    return entity_explanations

def extract_page_and_split_letters(corrected_text_path, client, model_name) -> CombinedOutput:
    try:
        with open(corrected_text_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            return CombinedOutput(page_number=None, letters=[])

        first_line = lines[0].strip()
        text = "".join(lines)

        try:
            page_number = int(first_line.strip())
        except ValueError:
            page_number = None

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": PROMPTS["split_letters"] + f"\nText:\n{text}"}],
            temperature=0
        )

        result = response.choices[0].message.content.strip()

        try:
            letters = json.loads(result)
            if not isinstance(letters, list):
                letters = [text]
        except json.JSONDecodeError:
            logging.warning(f"Invalid JSON returned when splitting letters. Falling back to full text.")
            letters = [text]

        combined = CombinedOutput(page_number=page_number, letters=letters)

        combined_path = corrected_text_path.replace(".corrected.txt", ".combined_output.json")
        with open(combined_path, "w", encoding="utf-8") as f:
            f.write(combined.model_dump_json(indent=2))

        logging.info(f"Combined output saved: {combined_path}")
        return combined
    except Exception as e:
        logging.error(f"Failed to extract page and split letters for {corrected_text_path}: {e}")
        return CombinedOutput(page_number=None, letters=[])