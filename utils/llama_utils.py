import subprocess
import os
import logging
import json
from typing import Optional
from schemas.llm_schemas import CorrectedText, EntitiesOutput, CombinedOutput

def get_client(api_key: Optional[str] = None):
    return None

def run_ollama(model: str, prompt: str) -> str:
    """Run an Ollama model locally with the given prompt and return output."""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            capture_output=True,
            check=True,
        )
        return result.stdout.decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"[LLaMA] Ollama error: {e.stderr.decode()}")
        return ""

def correct_text(text: str, base_name: str, output_dir: str, model_name: Optional[str] = None, client: Optional[object] = None) -> CorrectedText:
    """
    Correct OCR text for spelling, spacing, and punctuation.
    Saves corrected text to disk and returns a CorrectedText schema.
    """
    model = model_name or os.getenv("LLAMA_MODEL", "llama3.1:8b")

    prompt = (
        "You are a helpful assistant that only corrects spelling, OCR mistakes, and punctuation errors in text. "
        "Do not add or infer any additional content. Keep the original meaning intact. "
        "If the text already seems correct, leave it as is, and if you are unsure, leave it as is."
        f"{text}"
    )
    corrected = run_ollama(model, prompt) or text

    corrected_path = os.path.join(output_dir, base_name + ".corrected.txt")
    with open(corrected_path, "w", encoding="utf-8") as f:
        f.write(corrected)

    logging.info(f"[LLaMA] Corrected text saved: {corrected_path}")
    return CorrectedText(corrected_text=corrected)


def extract_entities(text: str, base_name: str, output_dir: str, model_name: Optional[str] = None, client: Optional[object] = None) -> EntitiesOutput:
    """
    Extract People, Productions, Companies, Theaters, and Dates.
    Saves structured JSON to disk and returns an EntitiesOutput schema.
    """
    model = model_name or os.getenv("LLAMA_MODEL", "llama3.1:8b")

    prompt = (
        "You are an assistant that extracts structured data from OCR-scanned historical letters. "
        "Return your answer as a **valid JSON object**, with the following keys: "
        "`People`, `Productions`, `Companies`, `Theaters`, and `Dates`. "
        "Each value should be a list of strings. If no items are found for a category, return an empty list. "
        "Do not include any explanation or formatting — only the JSON object."
        f"{text}"
    )
    response = run_ollama(model, prompt)

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

def extract_page_and_split_letters(corrected_path: str,model_name: Optional[str] = None, client: Optional[object] = None) -> CombinedOutput:
    """
    Split corrected text into one or more letters.
    Returns a CombinedOutput schema with page_number and letters.
    """
    model = model_name or os.getenv("LLAMA_MODEL", "llama3.1:8b")

    with open(corrected_path, "r", encoding="utf-8") as f:
        text = f.read()

    # First line may be a page number
    lines = text.splitlines()
    try:
        page_number = int(lines[0].strip())
    except (ValueError, IndexError):
        page_number = None

    prompt = (
            "The following is OCR-corrected text from scanned historical documents. "
            "Please detect if there are **multiple letters** present. Each letter typically starts with a recipient block "
            "(e.g. a name and address) followed by a greeting (e.g., 'Dear', 'Friend', 'Dear Sir:' or 'Gentlemen:'). "
            "It ends with a sign-off like 'Sincerely yours', 'Yours truly', or 'Yours sincerely'. "
            "Split the text into a **JSON array of full letters** — one string per letter. "
            "Return the full content of each letter, including greetings and sign-offs. "
            "If it’s just one letter, return a list with one string. "
            "IMPORTANT: Only return a JSON list — do NOT include any explanation or notes. "
            "Do not add any additional content, do not alter the text."
        f"{text}"
    )
    response = run_ollama(model, prompt)

    try:
        letters = json.loads(response)
        if not isinstance(letters, list):
            letters = [text]
    except json.JSONDecodeError:
        letters = [text]

    logging.info(f"[LLaMA] Split into {len(letters)} sections.")
    return CombinedOutput(page_number=page_number, letters=letters)