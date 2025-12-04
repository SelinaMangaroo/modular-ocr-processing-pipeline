import os
import logging
import boto3
from botocore.config import Config
from utils.helpers import get_file_paths, convert_to_pdf
from utils.aws_utils import (
    upload_file_to_s3,
    start_textract_job,
    wait_for_completion,
    extract_and_save_text_and_coords,
    delete_all_files_in_bucket,
)

# === PREPARE FILE ===
def prepare_file(filename, tmp_dir, input_dir, output_dir, image_magick_command, **kwargs):
    """Convert input images to PDF (if needed), upload to S3, and start Textract."""
    bucket_name = kwargs.get("bucket_name")
    region = kwargs.get("region")

    if not bucket_name or not region:
        raise ValueError("Missing bucket_name or region for AWS provider.")

    boto_config = Config(max_pool_connections=16)
    s3 = boto3.client("s3", region_name=region, config=boto_config)
    textract = boto3.client("textract", region_name=region, config=boto_config)

    paths = get_file_paths(filename, tmp_dir, input_dir, output_dir)
    base_name = paths["base_name"]
    os.makedirs(paths["doc_output_dir"], exist_ok=True)

    # Skip OCR if text already exists
    skip_ocr = os.getenv("SKIP_OCR_IF_EXISTS", "false").lower() == "true"
    raw_txt_path = os.path.join(paths["doc_output_dir"], f"{base_name}.raw.txt")
    coords_path = os.path.join(paths["doc_output_dir"], f"{base_name}.coords.json")
    if skip_ocr and os.path.exists(raw_txt_path):
        logging.info(f"[AWS] Skipping OCR for {base_name} (raw text exists).")
        if not os.path.exists(coords_path):
            logging.error(f"[AWS] Missing coords.json for {base_name} despite SKIP_OCR_IF_EXISTS=True")
            return None
        return base_name, {
            "skipped_ocr": True,
            "job_id": None,
            "doc_output_dir": paths["doc_output_dir"],
            "bucket_name": bucket_name,
            "region": region,
        }

    # Convert to PDF if needed
    ext = os.path.splitext(filename)[1].lower()
    pdf_path = paths["pdf_file"] if ext != ".pdf" else paths["path_to_file"]
    if ext != ".pdf":
        convert_to_pdf(paths["path_to_file"], paths["pdf_file"], image_magick_command, filename)
        logging.info(f"[AWS] Converted {filename} → {pdf_path}")

    # Upload to S3 + start Textract
    upload_file_to_s3(pdf_path, s3, bucket_name, paths["s3_pdf_key"])
    job_id = start_textract_job(paths["s3_pdf_key"], textract, bucket_name)
    logging.info(f"[AWS] Textract job started for {base_name} (Job ID: {job_id})")
    
    s3_prefix = os.path.dirname(paths["s3_pdf_key"])

    return base_name, {
        "skipped_ocr": False,
        "job_id": job_id,
        "doc_output_dir": paths["doc_output_dir"],
        "bucket_name": bucket_name,
        "region": region,
        "s3_prefix": s3_prefix,
    }

# === PROCESS FILE ===
def process_file(base_name, job_info, llm_module, model_name, api_key):
    """
    Processes one file:
      1. Runs Textract (or loads existing OCR)
      2. Queues ChatGPT correction and entity extraction requests
    Batch submission + hydration is handled in main.py.
    """
    boto_config = Config(max_pool_connections=16)
    textract = boto3.client("textract", region_name=job_info["region"], config=boto_config)
    s3 = boto3.client("s3", region_name=job_info["region"], config=boto_config)
    client = llm_module.get_client(api_key)

    base_output_dir = job_info["doc_output_dir"]

    # === OCR / Textract Phase ===
    if not job_info.get("skipped_ocr"):
        finished = wait_for_completion(job_info["job_id"], textract, max_retries=120, delay=5)
        if not finished:
            logging.error(f"[AWS] Textract timeout for {base_name}")
            return
        extract_and_save_text_and_coords(job_info["job_id"], base_name, base_output_dir, textract)
    else:
        logging.info(f"[AWS] Using existing OCR for {base_name}")

    raw_path = os.path.join(base_output_dir, f"{base_name}.raw.txt")
    if not os.path.exists(raw_path):
        logging.error(f"[AWS] Missing raw text for {base_name}")
        return

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # === Queue ChatGPT requests ===
    llm_module.correct_text(raw_text, base_name, base_output_dir, client, model_name)
    corrected_path = os.path.join(base_output_dir, f"{base_name}.corrected.txt")
        
    # === S3 Cleanup ===
    try:
        if not job_info.get("skipped_ocr"):
            prefix = job_info.get("s3_prefix")
            if prefix:
                delete_all_files_in_bucket(s3, job_info["bucket_name"], prefix=prefix)
                logging.info(f"[AWS] Cleaned up S3 prefix: {prefix}")
    except Exception as e:
        logging.warning(f"[AWS] Cleanup failed for {base_name}: {e}")

    # Entity extraction + split will be queued *after hydration* in main.py
    return base_name, corrected_path