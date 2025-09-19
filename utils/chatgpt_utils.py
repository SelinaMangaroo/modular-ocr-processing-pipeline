import os
import logging
import json
from openai import OpenAI
from schemas.llm_schemas import CorrectedText, EntitiesOutput, CombinedOutput

def get_client(api_key):
    """Return an initialized OpenAI client for ChatGPT."""
    return OpenAI(api_key=api_key)

def correct_text(text, base_name, doc_output_dir, client, model_name, save=True) -> CorrectedText:                                                                                                                                                                                                                                    
    """
    Sends OCR text to ChatGPT for basic correction, then saves it to a .corrected.txt file.
    """

    logging.info(f"Correcting OCR text for: {base_name}")
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that only corrects spelling, OCR mistakes, and punctuation errors in text. "
                    "Do not add or infer any additional content. Keep the original meaning intact. "
                    "If the text already seems correct, leave it as is, and if you are unsure, leave it as is."
                )
            },
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

    """
    Sends OCR text to ChatGPT and extracts named entities as JSON. 
    Falls back to saving raw output if JSON decoding fails.
    """
    logging.info(f"Extracting entities with ChatGPT for: {base_name}")
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an assistant that extracts structured data from OCR-scanned historical letters. "
                    "Return your answer as a **valid JSON object**, with the following keys: "
                    "`People`, `Productions`, `Companies`, `Theaters`, and `Dates`. "
                    "Each value should be a list of strings. If no items are found for a category, return an empty list. "
                    "Do not include any explanation or formatting — only the JSON object."
                )
            },
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

    # Save to file
    entity_path = os.path.join(doc_output_dir, base_name + ".entities.json")
    with open(entity_path, "w", encoding="utf-8") as f:
        f.write(entities.model_dump_json(indent=2))

    logging.info(f"Entity extraction saved: {entity_path}")
    return entities

def extract_page_and_split_letters(corrected_text_path, client, model_name) -> CombinedOutput:
    """
    Extracts the page number and splits multiple letters if present.
    """
    try:
        with open(corrected_text_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            return CombinedOutput(page_number=None, letters=[])

        first_line = lines[0].strip()
        full_text = "".join(lines)

        try:
            page_number = int(first_line.strip())
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
            f"\nText:\n{full_text}"
        )

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        result = response.choices[0].message.content.strip()

        try:
            letters = json.loads(result)
            if not isinstance(letters, list):
                letters = [full_text]
        except json.JSONDecodeError:
            logging.warning(f"Invalid JSON returned when splitting letters. Falling back to full text.")
            letters = [full_text]

        combined = CombinedOutput(page_number=page_number, letters=letters)

        combined_path = corrected_text_path.replace(".corrected.txt", ".combined_output.json")
        with open(combined_path, "w", encoding="utf-8") as f:
            f.write(combined.model_dump_json(indent=2))

        logging.info(f"Combined output saved: {combined_path}")
        return combined

    except Exception as e:
        logging.error(f"Failed to extract page and split letters for {corrected_text_path}: {e}")
        return CombinedOutput(page_number=None, letters=[])