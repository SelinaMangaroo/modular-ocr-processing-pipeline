import os
import logging
import json
import mimetypes
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from utils.helpers import get_file_paths, convert_to_pdf, resize_image

def prepare_file(filename, tmp_dir, input_dir, output_dir, image_magick_command, **kwargs):
    paths = get_file_paths(filename, tmp_dir, input_dir, output_dir)
    base_name = paths["base_name"]
    os.makedirs(paths["doc_output_dir"], exist_ok=True)

    ext = os.path.splitext(filename)[1].lower()

    if ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        resized_path = os.path.join(tmp_dir, f"{base_name}_resized{ext}")
        prepared_file = resize_image(paths["path_to_file"], resized_path, image_magick_command, filename)
        logging.info(f"[Azure] Resized image prepared: {prepared_file}")
    else:
        convert_to_pdf(paths["path_to_file"], paths["pdf_file"], image_magick_command, filename)
        prepared_file = paths["pdf_file"]
        logging.info(f"[Azure] PDF prepared: {prepared_file}")

    return base_name, {
        "doc_output_dir": paths["doc_output_dir"],
        "prepared_file": prepared_file,
    }


def process_file(base_name, job_info, llm_module, model_name, api_key):
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    if not endpoint or not key:
        raise ValueError("Azure endpoint/key not set")

    client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))

    prepared_file = job_info["prepared_file"]
    content_type, _ = mimetypes.guess_type(prepared_file)
    content_type = content_type or "application/octet-stream"

    logging.info(f"[Azure] Running OCR on {base_name} ({os.path.basename(prepared_file)})")

    # --- OCR ---
    with open(prepared_file, "rb") as f:
        poller = client.begin_analyze_document(
            model_id="prebuilt-read",
            body=f,
            content_type=content_type,
        )
    result = poller.result()

    # --- Collect OCR results ---
    ocr_lines, coords_data = [], []
    for page_num, page in enumerate(result.pages or [], start=1):
        for line in (page.lines or []):
            ocr_lines.append(line.content)
            coords_data.append({
                "page": page_num,
                "text": line.content,
                "boundingBox": line.polygon or []
            })

    if not ocr_lines:
        logging.warning(f"[Azure] No OCR text detected in {base_name}")

    raw_text = "\n".join(ocr_lines)

    # --- Save raw text ---
    raw_path = os.path.join(job_info["doc_output_dir"], base_name + ".raw.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw_text)
    logging.info(f"[Azure] Raw text saved: {raw_path}")

    # --- Save coordinates ---
    coords_path = os.path.join(job_info["doc_output_dir"], base_name + ".coords.json")
    with open(coords_path, "w", encoding="utf-8") as f:
        json.dump(coords_data, f, indent=2, ensure_ascii=False)
    logging.info(f"[Azure] Coordinates saved: {coords_path}")

    # --- LLM client ---
    client_llm = llm_module.get_client(api_key)

    # --- Correct text ---
    corrected_obj, corrected_path = None, os.path.join(job_info["doc_output_dir"], base_name + ".corrected.txt")
    try:
        corrected_obj = llm_module.correct_text(raw_text, base_name, job_info["doc_output_dir"], client_llm, model_name)
    except Exception as e:
        logging.warning(f"[Azure] Correction failed for {base_name}: {e}")

    text_for_entities = raw_text
    if corrected_obj:
        text_for_entities = corrected_obj.corrected_text
        with open(corrected_path, "w", encoding="utf-8") as f:
            f.write(text_for_entities)
        logging.info(f"[Azure] Corrected text saved: {corrected_path}")

    # --- Extract entities ---
    try:
        entities = llm_module.extract_entities(
            text_for_entities, base_name, job_info["doc_output_dir"], client_llm, model_name
        )
        if entities:
            logging.info(f"[Azure] Entities extracted for {base_name}")
    except Exception as e:
        logging.warning(f"[Azure] Entity extraction failed for {base_name}: {e}")
        entities = {}

    # --- Split into letters ---
    combined, combined_path = None, None
    try:
        combined = llm_module.extract_page_and_split_letters(corrected_path, client_llm, model_name)
        if combined:
            combined_path = os.path.join(job_info["doc_output_dir"], base_name + ".combined_output.json")
            with open(combined_path, "w", encoding="utf-8") as f:
                json.dump(
                    combined.model_dump() if hasattr(combined, "model_dump") else combined,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            logging.info(f"[Azure] Combined output saved: {combined_path}")
    except Exception as e:
        logging.warning(f"[Azure] Letter splitting failed for {base_name}: {e}")

    return {
        "status": "success",
        "base_name": base_name,
        "raw_path": raw_path,
        "corrected_path": corrected_path if corrected_obj else None,
        "entities": entities if entities else None,
        "combined_path": combined_path if combined else None,
    }
