import os
import logging
import json
import subprocess
from typing import Optional
from schemas.llm_schemas import CorrectedText, EntitiesOutput, CombinedOutput, EntityExplanations

PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts.json")
with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
    PROMPTS = json.load(f)

def get_client(api_key: Optional[str] = None):
    return None

def run_ollama(model: str, prompt: str) -> str:
    try:
        result = subprocess.run(
            ["ollama", "run", model], input=prompt.encode("utf-8"), capture_output=True, check=True,
        )
        return result.stdout.decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"[LLaMA] Ollama error: {e.stderr.decode()}")
        return ""

def correct_text(text: str, base_name: str, output_dir: str, model_name: Optional[str] = None, client: Optional[object] = None) -> CorrectedText:
    model = model_name or os.getenv("LLAMA_MODEL", "llama3.1:8b")

    corrected = run_ollama(model, PROMPTS["correct_text"] + f"\nText:\n{text}")

    corrected_path = os.path.join(output_dir, base_name + ".corrected.txt")
    with open(corrected_path, "w", encoding="utf-8") as f:
        f.write(corrected)

    logging.info(f"[LLaMA] Corrected text saved: {corrected_path}")
    return CorrectedText(corrected_text=corrected)

def extract_entities(text: str, base_name: str, output_dir: str, model_name: Optional[str] = None, client: Optional[object] = None) -> EntitiesOutput:
    model = model_name or os.getenv("LLAMA_MODEL", "llama3.1:8b")
    response = run_ollama(model, PROMPTS["extract_entities"] + f"\nText:\n{text}")

    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        parsed = {"People": [], "Productions": [], "Companies": [], "Theaters": [], "Dates": []}

    entities = EntitiesOutput(**parsed)

    path = os.path.join(output_dir, base_name + ".entities.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(entities.model_dump_json(indent=2))

    logging.info(f"[LLaMA] Entities saved: {path}")
    return entities

def explain_entities(entities, base_name, output_dir, model_name: Optional[str] = None, client: Optional[object] = None) -> EntityExplanations:
    logging.info(f"[LLaMA] Explaining entities for {base_name}")
    explanations = {"People": {}, "Productions": {}, "Companies": {}, "Theaters": {}}
    model = model_name or os.getenv("LLAMA_MODEL", "llama3.1:8b")

    for category in explanations.keys():
        items = getattr(entities, category, [])
        for item in items:
            response = run_ollama(model, PROMPTS["explain_entities"] + f"\nCategory: {category}\nEntity: {item}")
            explanations[category][item] = response.strip()

    entity_explanations = EntityExplanations(**explanations)

    explain_path = os.path.join(output_dir, base_name + ".entities_explained.json")
    with open(explain_path, "w", encoding="utf-8") as f:
        f.write(entity_explanations.model_dump_json(indent=2))

    logging.info(f"[LLaMA] Entity explanations saved: {explain_path}")
    return entity_explanations

def extract_page_and_split_letters(corrected_path: str,model_name: Optional[str] = None, client: Optional[object] = None) -> CombinedOutput:
    model = model_name or os.getenv("LLAMA_MODEL", "llama3.1:8b")

    with open(corrected_path, "r", encoding="utf-8") as f:
        text = f.read()

    # First line may be a page number
    lines = text.splitlines()
    try:
        page_number = int(lines[0].strip())
    except (ValueError, IndexError):
        page_number = None

    response = run_ollama(model, PROMPTS["split_letters"] + f"\nText:\n{text}")

    try:
        letters = json.loads(response)
        if not isinstance(letters, list):
            letters = [text]
    except json.JSONDecodeError:
        letters = [text]

    logging.info(f"[LLaMA] Split into {len(letters)} sections.")
    return CombinedOutput(page_number=page_number, letters=letters)