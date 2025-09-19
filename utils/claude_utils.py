import os
import logging
import json
from anthropic import Anthropic
from schemas.llm_schemas import CorrectedText, EntitiesOutput, CombinedOutput

def get_client(api_key=None):
    return Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

def correct_text(text, base_name, output_dir, client, model_name) -> CorrectedText:
    logging.info(f"[Claude] Correcting OCR text for: {base_name}")

    resp = client.messages.create(
        model=model_name,
        max_tokens=4096,
        messages=[
            {"role": "user", "content": (
                "You are a helpful assistant that only corrects spelling, OCR mistakes, and punctuation errors in text. "
                "Do not add or infer any additional content. Keep the original meaning intact. "
                "If the text already seems correct, leave it as is, and if you are unsure, leave it as is."
                f"{text}"
            )}
        ]
    )

    corrected_text = resp.content[0].text.strip()
    corrected_path = os.path.join(output_dir, base_name + ".corrected.txt")
    with open(corrected_path, "w", encoding="utf-8") as f:
        f.write(corrected_text)

    return CorrectedText(corrected_text=corrected_text)

def extract_entities(text, base_name, output_dir, client, model_name) -> EntitiesOutput:
    logging.info(f"[Claude] Extracting entities for: {base_name}")

    resp = client.messages.create(
        model=model_name,
        max_tokens=4096,
        messages=[
            {"role": "user", "content": (
                "You are an assistant that extracts structured data from OCR-scanned historical letters. "
                "Return your answer as a **valid JSON object**, with the following keys: "
                "`People`, `Productions`, `Companies`, `Theaters`, and `Dates`. "
                "Each value should be a list of strings. If no items are found for a category, return an empty list. "
                "Do not include any explanation or formatting — only the JSON object."
                f"{text}"
            )}
        ]
    )

    try:
        parsed = json.loads(resp.content[0].text.strip())
    except json.JSONDecodeError:
        parsed = {"People": [], "Productions": [], "Companies": [], "Theaters": [], "Dates": []}

    entities = EntitiesOutput(**parsed)
    path = os.path.join(output_dir, base_name + ".entities.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(entities.model_dump_json(indent=2))
    return entities

def extract_page_and_split_letters(corrected_text_path, client, model_name) -> CombinedOutput:
    """
    Uses Claude to detect multiple letters in OCR text.
    Returns CombinedOutput schema with page_number and letters.
    """
    with open(corrected_text_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return CombinedOutput(page_number=None, letters=[])

    # Try to parse first line as page number
    first_line = lines[0].strip()
    full_text = "".join(lines)
    try:
        page_number = int(first_line)
    except ValueError:
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
        f"{full_text}"
    )

    resp = client.messages.create(
        model=model_name,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
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
