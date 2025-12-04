import os
import logging
import json
import mimetypes
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from utils.helpers import get_file_paths, convert_to_pdf, resize_image

def prepare_file(filename, tmp_dir, input_dir, output_dir, image_magick_command, **kwargs):
    """
    Prepares document for OCR:
      - resize image or convert to PDF
    """
    paths = get_file_paths(filename, tmp_dir, input_dir, output_dir)
    base_name = paths["base_name"]

    os.makedirs(paths["doc_output_dir"], exist_ok=True)

    ext = os.path.splitext(filename)[1].lower()

    # Convert or resize depending on file type
    if ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        resized_path = os.path.join(tmp_dir, f"{base_name}_resized{ext}")
        prepared_file = resize_image(paths["path_to_file"], resized_path, image_magick_command, filename)
        logging.info(f"[Azure] Resized image prepared: {prepared_file}")
    else:
        prepared_file = paths["pdf_file"]
        convert_to_pdf(paths["path_to_file"], prepared_file, image_magick_command, filename)
        logging.info(f"[Azure] Converted to PDF: {prepared_file}")

    # === SKIP OCR LOGIC ===
    skip_ocr = os.getenv("SKIP_OCR_IF_EXISTS", "false").lower() == "true"
    raw_txt_path = os.path.join(paths["doc_output_dir"], f"{base_name}.raw.txt")

    if skip_ocr and os.path.exists(raw_txt_path):
        logging.info(f"[Azure] Skipping OCR for {base_name} (raw text already exists).")
        return base_name, {
            "doc_output_dir": paths["doc_output_dir"],
            "prepared_file": prepared_file,
            "skipped_ocr": True,
        }

    return base_name, {
        "doc_output_dir": paths["doc_output_dir"],
        "prepared_file": prepared_file,
        "skipped_ocr": False,
    }

def process_file(base_name, job_info, llm_module, model_name, api_key):
    """
    Runs Azure OCR to produce:
      - <base>.raw.txt
      - <base>.coords.json
    """
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    if not endpoint or not key:
        raise ValueError("Azure endpoint/key not set")

    client_ai = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))

    prepared_file = job_info["prepared_file"]
    doc_output_dir = job_info["doc_output_dir"]

    raw_path = os.path.join(doc_output_dir, f"{base_name}.raw.txt")
    coords_path = os.path.join(doc_output_dir, f"{base_name}.coords.json")

    # If OCR was skipped
    if job_info.get("skipped_ocr"):
        logging.info(f"[Azure] Using existing OCR for {base_name}")
        if not os.path.exists(coords_path):
            logging.error(f"[Azure] Missing coords for {base_name} despite SKIP_OCR_IF_EXISTS=True")
            return None
    else:
        # === Perform Azure OCR ===
        content_type, _ = mimetypes.guess_type(prepared_file)
        content_type = content_type or "application/octet-stream"

        logging.info(f"[Azure] Starting OCR for {base_name}")

        with open(prepared_file, "rb") as f:
            poller = client_ai.begin_analyze_document(
                model_id="prebuilt-read",
                body=f,
                content_type=content_type,
            )

        result = poller.result()

        # Extract text + bounding boxes
        ocr_lines = []
        coords_data = []

        for page_num, page in enumerate(result.pages or [], start=1):
            for line in page.lines or []:
                ocr_lines.append(line.content)
                coords_data.append({
                    "page": page_num,
                    "text": line.content,
                    "boundingBox": line.polygon or [],
                })

        # Save raw text
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write("\n".join(ocr_lines))

        # Save coordinates JSON
        with open(coords_path, "w", encoding="utf-8") as f:
            json.dump(coords_data, f, indent=2, ensure_ascii=False)

        logging.info(f"[Azure] OCR text saved: {raw_path}")

    # Safety check
    if not os.path.exists(raw_path):
        logging.error(f"[Azure] Missing raw text for {base_name}")
        return None

    # === Load raw text (fix) ===
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # === Queue LLM correction request (Batch 1) ===
    client_llm = llm_module.get_client(api_key)
    llm_module.correct_text(raw_text, base_name, doc_output_dir, client_llm, model_name)

    return base_name, raw_path