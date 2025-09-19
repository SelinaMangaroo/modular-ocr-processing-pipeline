import os
import subprocess # Run ImageMagick CLI command
import logging # Logging setup
from datetime import datetime
import time
import shutil

def split_into_batches(items, batch_size):
    """Splits a list into batches of size `batch_size`."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def get_file_paths(filename, tmp_dir, input_dir, output_dir):
    """
    Generates file paths for the input filename, including:
    - Base name (without extension)
    - Path to the original file
    - Path to the .pdf file
    - S3 key for the .pdf file
    - Directory for the document output
    """
    base_name, _ = os.path.splitext(filename)
    return {
        "base_name": base_name,
        "path_to_file": os.path.join(input_dir, filename),
        "pdf_file": os.path.join(tmp_dir, f"{base_name}.pdf"),
        "s3_pdf_key": f"{base_name}.pdf",
        "doc_output_dir": os.path.join(output_dir, base_name),
    }

def convert_to_pdf(file_path, pdf_path, image_magick_command="convert", filename=""):
    """
    Converts an image file to PDF using ImageMagick.
    Args:
        file_path (str): Path to the input image file.
        pdf_path (str): Path to save the output PDF file.
        image_magick_command (str): Command to run ImageMagick. Default is "convert".
        filename (str): Name of the file being processed, for logging purposes.
    """
    try:
        logging.info(f"Converting {filename} to PDF...")
        subprocess.run([image_magick_command, file_path, pdf_path], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"ImageMagick failed to convert {file_path} to PDF: {e}")
        raise

def clean_tmp_folder(tmp_dir):
    """
    Cleans up the temporary directory by deleting all its contents and recreating it.
    Args:
        tmp_dir (str): Path to the temporary directory.
    """
    try:
        shutil.rmtree(tmp_dir)  # Delete entire temp folder and contents
        os.makedirs(tmp_dir, exist_ok=True)  # Recreate clean folder
        logging.info("Cleaned up temporary directory.")
    except Exception as e:
        logging.error(f"Failed to clean temporary directory: {e}")

def initialize_logging(log_dir="logs"):
    """
    Initializes logging to both a file and the console.
    Args:
        log_dir (str): Directory to save the log file. Default is "logs".
    Returns:
        str: Path to the log file.
    """
    os.makedirs(log_dir, exist_ok=True)

    log_filename = datetime.now().strftime("%m-%d-%Y_%H-%M-%S.log")
    log_path = os.path.join(log_dir, log_filename)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%m-%d-%Y_%H-%M-%S"
    )

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    # Root logger setup
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Avoid duplicate logs if reinitialized
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info("Logging initialized.")
    return log_path

def log_runtime(start_time):
    """
    Logs the total runtime of the script.
    Args:
        start_time (float): Start time of the script.
    """
    hrs, rem = divmod(time.time() - start_time, 3600)
    mins, secs = divmod(rem, 60)
    logging.info(f"Pipeline completed in {int(hrs)}h {int(mins)}m {int(secs)}s.")

import subprocess
import logging
import os

def resize_image(file_path, output_path, image_magick_command="convert", filename="", max_dim="2000x2000>"):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logging.info(f"Resizing {filename or os.path.basename(file_path)} to max {max_dim}...")

        cmd = [
            image_magick_command,
            file_path,
            "-resize", max_dim,
            "-strip",       # remove metadata
            "-quality", "85",  # for JPG/PNG compression
            output_path
        ]
        subprocess.run(cmd, check=True)

        logging.info(f"Resized image saved: {output_path}")
        return output_path

    except subprocess.CalledProcessError as e:
        logging.error(f"ImageMagick failed to resize {file_path}: {e}")
        raise
