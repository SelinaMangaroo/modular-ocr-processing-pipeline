import os
import logging
import json
import mimetypes
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from utils.helpers import get_file_paths, convert_to_pdf, resize_image

def prepare_file(filename, tmp_dir, input_dir, output_dir, image_magick_command, bucket_name=None, region=None):
    try:
        paths = get_file_paths(filename, tmp_dir, input_dir, output_dir)
        base_name = paths["base_name"]
        os.makedirs(paths["doc_output_dir"], exist_ok=True)

        ext = os.path.splitext(filename)[1].lower()

        # Azure has file size limits — resize images to be safe
        if ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
            resized_path = os.path.join(tmp_dir, f"{base_name}_resized{ext}")
            prepared_file = resize_image(paths["path_to_file"], resized_path, image_magick_command, filename)
        else:
            convert_to_pdf(paths["path_to_file"], paths["pdf_file"], image_magick_command, filename)
            prepared_file = paths["pdf_file"]

        return base_name, {
            "doc_output_dir": paths["doc_output_dir"],
            "prepared_file": prepared_file,
        }

    except Exception as e:
        logging.error(f"[Azure PREP ERROR] {filename}: {e}")
        return None

def process_file(base_name, job_info, llm_module, model_name, api_key):
    try:
        endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        if not endpoint or not key:
            raise RuntimeError("Azure endpoint/key not set")

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
        ocr_lines = []
        coords_data = []
        for page_num, page in enumerate(result.pages or [], start=1):
            for line in (page.lines or []):
                ocr_lines.append(line.content)
                coords_data.append({
                    "page": page_num,
                    "text": line.content,
                    "boundingBox": line.polygon or []
                })

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

        client_llm = llm_module.get_client(api_key)

        # Correct text
        corrected_obj = llm_module.correct_text(raw_text, base_name, job_info["doc_output_dir"], client_llm, model_name)
        corrected_path = os.path.join(job_info["doc_output_dir"], base_name + ".corrected.txt")

        text_for_entities = raw_text
        if corrected_obj:
            text_for_entities = corrected_obj.corrected_text
            with open(corrected_path, "w", encoding="utf-8") as f:
                f.write(text_for_entities)
            logging.info(f"[Azure] Corrected text saved: {corrected_path}")

        # Extract entities
        entities = llm_module.extract_entities(
            text_for_entities, base_name, job_info["doc_output_dir"], client_llm, model_name
        )
        if entities:
            logging.info(f"[Azure] Entities extracted for {base_name}")

        # Split into letters
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
        logging.error(f"[Azure PROCESS ERROR] {base_name}: {e}")
