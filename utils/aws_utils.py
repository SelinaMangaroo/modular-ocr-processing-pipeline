import os
import time
import logging
import json
        
def upload_file_to_s3(file_path, s3, bucket_name, s3_key):
    """
    Uploads a file to an S3 bucket and verifies accessibility.
    Adds a short delay to ensure S3 object propagation before Textract reads it.
    """
    s3_key = s3_key.lstrip("/")

    try:
        s3.upload_file(file_path, bucket_name, s3_key)
        logging.info(f"[UPLOAD] {os.path.basename(file_path)} → s3://{bucket_name}/{s3_key}")

        # Verify upload succeeded
        for attempt in range(3):
            try:
                s3.head_object(Bucket=bucket_name, Key=s3_key)
                logging.info(f"[VERIFY] S3 object accessible after {attempt + 1} check(s).")
                break
            except Exception as e:
                logging.warning(f"[VERIFY] Waiting for S3 to finalize ({attempt + 1}/3)... {e}")
                time.sleep(2)
        else:
            raise RuntimeError(f"S3 object not accessible after upload: {s3_key}")

        # Safety delay before Textract access
        time.sleep(2)

    except Exception as e:
        logging.error(f"[UPLOAD ERROR] {file_path}: {e}")
        raise

def delete_all_files_in_bucket(s3, bucket_name, prefix=None):
    """
    Deletes only files in the specified S3 bucket (optionally limited to a prefix).
    """
    try:
        kwargs = {"Bucket": bucket_name}
        if prefix:
            kwargs["Prefix"] = prefix  # only delete this folder’s files

        response = s3.list_objects_v2(**kwargs)
        objects = response.get("Contents", [])
        if not objects:
            logging.info(f"No files to delete in S3 bucket (prefix={prefix or 'ALL'}).")
            return

        delete_keys = [{"Key": obj["Key"]} for obj in objects]
        result = s3.delete_objects(Bucket=bucket_name, Delete={"Objects": delete_keys})
        total = len(result.get("Deleted", []))
        logging.info(
            f"Deleted {total} files from S3 bucket '{bucket_name}' (prefix={prefix or 'ALL'})."
        )

    except Exception as e:
        logging.error(f"Error deleting files from S3 (prefix={prefix or 'ALL'}): {e}")


def extract_and_save_text_and_coords(job_id, base_name, doc_output_dir, textract):
    """
    Fetches Textract results and saves the text and word-level bounding box data.
    Args:
        job_id (str): Textract job ID.
        base_name (str): Base name for the output files.
        doc_output_dir (str): Directory to save the output files.
        textract (boto3.client): Boto3 Textract client.
    """
    lines = []
    word_info = []

    next_token = None
    while True:
        response = textract.get_document_text_detection(JobId=job_id, NextToken=next_token) if next_token else textract.get_document_text_detection(JobId=job_id)
        
        for block in response.get('Blocks', []):
            if block['BlockType'] == 'LINE':
                lines.append(block['Text'])
            elif block['BlockType'] == 'WORD':
                word_info.append({
                    "text": block['Text'],
                    "confidence": block['Confidence'],
                    "boundingBox": block['Geometry']['BoundingBox']
                })

        next_token = response.get('NextToken')
        if not next_token:
            break

    # Save plain text
    with open(os.path.join(doc_output_dir, f"{base_name}.raw.txt"), 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    # Save word-level bounding box data
    with open(os.path.join(doc_output_dir, f"{base_name}.coords.json"), 'w', encoding='utf-8') as jf:
        json.dump(word_info, jf, indent=2)

    logging.info(f"Saved text and coordinates for {base_name}")

def start_textract_job(s3_pdf_key, textract, bucket_name):
    """
    Starts a Textract job for the specified PDF file in S3.
    Includes retry logic for transient S3_OBJECT_INACCESSIBLE issues.
    """
    s3_pdf_key = s3_pdf_key.lstrip("/")

    for attempt in range(3):
        try:
            response = textract.start_document_text_detection(
                DocumentLocation={
                    "S3Object": {"Bucket": bucket_name, "Name": s3_pdf_key}
                }
            )
            logging.info(
                f"Started Textract job for {s3_pdf_key} (JobId: {response['JobId']})"
            )
            return response["JobId"]
        except Exception as e:
            msg = str(e)
            if "InvalidS3ObjectException" in msg or "S3_OBJECT_INACCESSIBLE" in msg:
                logging.warning(
                    f"[RETRY] Textract couldn't access {s3_pdf_key} ({attempt+1}/3). Retrying in 3s..."
                )
                time.sleep(3)
                continue
            logging.error(f"Failed to start Textract job for {s3_pdf_key}: {e}")
            raise

    raise RuntimeError(f"Textract could not access {s3_pdf_key} after 3 retries.")

def wait_for_completion(job_id, textract, max_retries, delay):
    """
    Waits for the Textract job to complete.
    Args:
        job_id (str): Textract job ID.
        textract (boto3.client): Boto3 Textract client.
        max_retries (int): Maximum number of retries before timing out.
        delay (int): Delay between retries in seconds.
    Returns:
        bool: True if the job succeeded, False if it failed or timed out.
    """
    for _ in range(max_retries):
        result = textract.get_document_text_detection(JobId=job_id)
        status = result['JobStatus']
        if status == 'SUCCEEDED':
            return True
        elif status == 'FAILED':
            logging.error(f"Textract job failed: {result.get('StatusMessage')}")
            return False
        time.sleep(delay)
    logging.error(f"Textract job {job_id} timed out.")
    return False
