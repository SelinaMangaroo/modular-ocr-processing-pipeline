import os
import logging
import json
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

def prepare_file(filename, tmp_dir, input_dir, output_dir, image_magick_command, bucket_name, region):
    """
    Convert file to PDF (if needed), upload to S3, and start Textract job.
    Returns job info dict for process_file().
    """
    try:
        boto_config = Config(max_pool_connections=16)
        s3 = boto3.client("s3", region_name=region, config=boto_config)
        textract = boto3.client("textract", region_name=region, config=boto_config)

        paths = get_file_paths(filename, tmp_dir, input_dir, output_dir)
        base_name = paths["base_name"]
        os.makedirs(paths["doc_output_dir"], exist_ok=True)

        # Ensure PDF
        ext = os.path.splitext(filename)[1].lower()
        pdf_to_upload = paths["pdf_file"] if ext != ".pdf" else paths["path_to_file"]

        if ext != ".pdf":
            convert_to_pdf(paths["path_to_file"], paths["pdf_file"], image_magick_command, filename)

        # Upload to S3 + start Textract
        upload_file_to_s3(pdf_to_upload, s3, bucket_name, paths["s3_pdf_key"])
        job_id = start_textract_job(paths["s3_pdf_key"], textract, bucket_name)

        return base_name, {
            "job_id": job_id,
            "doc_output_dir": paths["doc_output_dir"],
            "s3_pdf_key": paths["s3_pdf_key"],
            "bucket_name": bucket_name,
            "region": region,
        }

    except Exception as e:
        logging.error(f"[AWS PREP ERROR] {filename}: {e}")
        return None


def process_file(base_name, job_info, llm_module, model_name, api_key):
    """
    Run AWS Textract OCR, save raw text + coords, then send through LLM pipeline.
    Saves: .corrected.txt, .entities.json, .combined_output.json
    """
    try:
        # AWS clients
        boto_config = Config(max_pool_connections=16)
        textract = boto3.client("textract", region_name=job_info["region"], config=boto_config)
        s3 = boto3.client("s3", region_name=job_info["region"], config=boto_config)

        logging.info(f"[AWS] Waiting on Textract for: {base_name}.pdf")

        # Wait for job
        finished = wait_for_completion(job_info["job_id"], textract, max_retries=120, delay=5)
        if not finished:
            logging.error(f"[AWS] Textract did not finish for {base_name}")
            return

        # Save raw + coords
        extract_and_save_text_and_coords(job_info["job_id"], base_name, job_info["doc_output_dir"], textract)
        raw_path = os.path.join(job_info["doc_output_dir"], base_name + ".raw.txt")

        with open(raw_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # LLM client
        client = llm_module.get_client(api_key)

        # === Correct text ===
        corrected_obj = llm_module.correct_text(raw_text, base_name, job_info["doc_output_dir"], client, model_name)
        corrected_path = os.path.join(job_info["doc_output_dir"], base_name + ".corrected.txt")

        text_for_entities = raw_text
        if corrected_obj:
            text_for_entities = corrected_obj.corrected_text
            with open(corrected_path, "w", encoding="utf-8") as f:
                f.write(text_for_entities)
            logging.info(f"[AWS] Corrected text saved: {corrected_path}")

        # === Extract entities ===
        entities = llm_module.extract_entities(
            text_for_entities, base_name, job_info["doc_output_dir"], client, model_name
        )
        if entities:
            logging.info(f"[AWS] Entities extracted for {base_name}")

        # === Split into letters ===
        combined = llm_module.extract_page_and_split_letters(corrected_path, client, model_name)
        if combined:
            combined_path = os.path.join(job_info["doc_output_dir"], base_name + ".combined_output.json")
            with open(combined_path, "w", encoding="utf-8") as f:
                json.dump(
                    combined.model_dump() if hasattr(combined, "model_dump") else combined,
                    f, indent=2, ensure_ascii=False,
                )
            logging.info(f"[AWS] Combined output saved: {combined_path}")

        # Cleanup bucket
        delete_all_files_in_bucket(s3, job_info["bucket_name"])

    except Exception as e:
        logging.error(f"[AWS PROCESS ERROR] {base_name}: {e}")
