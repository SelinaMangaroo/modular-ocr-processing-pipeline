import os
import logging
import json
import boto3
from botocore.config import Config
from utils.helpers import get_file_paths, convert_to_pdf
from utils.aws_utils import (upload_file_to_s3, start_textract_job, wait_for_completion, extract_and_save_text_and_coords, delete_all_files_in_bucket)

def prepare_file(filename, tmp_dir, input_dir, output_dir, image_magick_command, **kwargs):
    bucket_name = kwargs.get("bucket_name")
    region = kwargs.get("region")

    if not bucket_name or not region:
        raise ValueError("AWS provider requires both bucket_name and region")

    boto_config = Config(max_pool_connections=16)
    s3 = boto3.client("s3", region_name=region, config=boto_config)
    textract = boto3.client("textract", region_name=region, config=boto_config)

    paths = get_file_paths(filename, tmp_dir, input_dir, output_dir)
    base_name = paths["base_name"]
    os.makedirs(paths["doc_output_dir"], exist_ok=True)

    ext = os.path.splitext(filename)[1].lower()
    pdf_to_upload = paths["pdf_file"] if ext != ".pdf" else paths["path_to_file"]

    if ext != ".pdf":
        convert_to_pdf(paths["path_to_file"], paths["pdf_file"], image_magick_command, filename)
        logging.info(f"[AWS] Converted {filename} to PDF: {paths['pdf_file']}")

    # Upload to S3 + start Textract
    upload_file_to_s3(pdf_to_upload, s3, bucket_name, paths["s3_pdf_key"])
    logging.info(f"[AWS] Uploaded {filename} to S3 as {paths['s3_pdf_key']}")

    job_id = start_textract_job(paths["s3_pdf_key"], textract, bucket_name)
    logging.info(f"[AWS] Started Textract job {job_id} for {filename}")

    return base_name, {
        "job_id": job_id,
        "doc_output_dir": paths["doc_output_dir"],
        "s3_pdf_key": paths["s3_pdf_key"],
        "bucket_name": bucket_name,
        "region": region,
    }

def process_file(base_name, job_info, llm_module, model_name, api_key):
    boto_config = Config(max_pool_connections=16)
    textract = boto3.client("textract", region_name=job_info["region"], config=boto_config)
    s3 = boto3.client("s3", region_name=job_info["region"], config=boto_config)

    logging.info(f"[AWS] Waiting on Textract for {base_name}.pdf")

    # --- Wait for Textract job ---
    finished = wait_for_completion(job_info["job_id"], textract, max_retries=120, delay=5)
    if not finished:
        logging.error(f"[AWS] Textract did not finish for {base_name}")
        return {"status": "failed", "reason": "textract_timeout", "base_name": base_name}

    # --- Save raw + coords ---
    extract_and_save_text_and_coords(job_info["job_id"], base_name, job_info["doc_output_dir"], textract)
    raw_path = os.path.join(job_info["doc_output_dir"], base_name + ".raw.txt")

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # --- LLM client ---
    client = llm_module.get_client(api_key)

    # --- Correct text ---
    corrected_obj = None
    try:
        corrected_obj = llm_module.correct_text(raw_text, base_name, job_info["doc_output_dir"], client, model_name)
    except Exception as e:
        logging.warning(f"[AWS] Correction failed for {base_name}: {e}")

    corrected_path = os.path.join(job_info["doc_output_dir"], base_name + ".corrected.txt")
    text_for_entities = raw_text

    if corrected_obj:
        text_for_entities = corrected_obj.corrected_text
        with open(corrected_path, "w", encoding="utf-8") as f:
            f.write(text_for_entities)
        logging.info(f"[AWS] Corrected text saved: {corrected_path}")

    # --- Extract entities ---
    try:
        entities = llm_module.extract_entities(
            text_for_entities, base_name, job_info["doc_output_dir"], client, model_name
        )
        if entities:
            logging.info(f"[AWS] Entities extracted for {base_name}")
    except Exception as e:
        logging.warning(f"[AWS] Entity extraction failed for {base_name}: {e}")
        entities = {}

    # --- Split into letters ---
    combined = None
    try:
        combined = llm_module.extract_page_and_split_letters(corrected_path, client, model_name)
        if combined:
            combined_path = os.path.join(job_info["doc_output_dir"], base_name + ".combined_output.json")
            with open(combined_path, "w", encoding="utf-8") as f:
                json.dump(
                    combined.model_dump() if hasattr(combined, "model_dump") else combined,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            logging.info(f"[AWS] Combined output saved: {combined_path}")
    except Exception as e:
        logging.warning(f"[AWS] Letter splitting failed for {base_name}: {e}")

    # --- Cleanup bucket ---
    try:
        delete_all_files_in_bucket(s3, job_info["bucket_name"])
        logging.info(f"[AWS] Cleaned up S3 bucket: {job_info['bucket_name']}")
    except Exception as e:
        logging.warning(f"[AWS] Cleanup failed for {base_name}: {e}")

    return {
        "status": "success",
        "base_name": base_name,
        "raw_path": raw_path,
        "corrected_path": corrected_path if corrected_obj else None,
        "entities": entities if entities else None,
        "combined_path": combined_path if combined else None,
    }