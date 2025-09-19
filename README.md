# OCR Processing Pipeline

This project is a batch OCR (Optical Character Recognition) pipeline that uses **AWS Textract** and **OpenAI's ChatGPT** to process scanned .jpg images. It extracts raw text, corrects OCR errors, and structures key entities (e.g. people, companies, dates) from historical documents such as theater letters.

---

## Features

- Converts .jpg images to .pdf using ImageMagick
- Runs asynchronous AWS Textract OCR jobs
- Uses OpenAI's GPT to:
  - Correct OCR errors
  - Extract entities and other metadata
- S3 uploads & automatic cleanup after processing 
- Supports batching with configurable size
- Structured output: raw text, corrected text, entity JSON
- Logs all operations to timestamped log files
- Includes word count + Levenshtein distance validation via pytest
- Optimized with parallel processing (ThreadPoolExecutor)

---

## Project Structure

```
ocr-processing-pipeline/
├── utils/                        # Utility modules 
│   ├── aws_utils.py              # AWS Textract, S3, and OCR logic
│   ├── chatgpt_utils.py          # GPT-based text correction and entity extraction
│   ├── helpers.py                # PDF conversion, logging, etc.
├── tests/                        # Pytest test suite
│   ├── test_wordcount.py         # Test: word count differences
│   ├── test_levenshtein_distance.py  # Test: text delta magnitude
│   ├── test_chatgpt_consistency.py   # Test: GPT output stability
│   └── test_utils.py             # Shared test utilities
├── test_logs/                    # Runtime test logs
├── logs/                         # Runtime logs
├── output/                       # Output text, JSON, corrected results
├── venv/                         # Virtual environment (not tracked in version control)
├── main.py                       # Entry point for the pipeline
├── .env                          # Configuration variables     
├── .gitignore                    # Specifies files/directories to exclude from Git
├── README.md                     # Project documentation
├── requirements.txt              # List of dependencies
```

---

## Setup

1. Clone the repository:

```bash
git clone https://github.com/SelinaMangaroo/ocr-processing-pipeline.git
cd ocr-processing-pipeline
```

2. Create a virtual environment and install dependencies:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
**ImageMagick** must be installed system-wide, it's not a Python package.

```
brew install imagemagick # macOS
sudo apt install imagemagick  # Debian/Ubuntu
```

---

## AWS Setup (Textract + S3)

To use this OCR pipeline, you'll need to configure your AWS account with access to both **Amazon Textract** and **Amazon S3**.

Textract is a machine learning service from AWS, to extract printed text and data from scanned documents. The script interacts with Textract and S3 using **Boto3**, the official AWS SDK for Python.

### Create an IAM User

Go to the [AWS IAM Console](https://console.aws.amazon.com/iam/):

- Create a new IAM user or use an existing one.
- Attach the following permissions:
  - `AmazonTextractFullAccess`
  - `AmazonS3FullAccess`
- Generate **Access Key ID** and **Secret Access Key** to set as environment variables

Alternatively if you do not want to set the Access Key ID and Secret Access Key as environment variables, you can store the credentials locally on your machine at `~/.aws/credentials`

Because boto3 follows AWS's default credential provider chain, it automatically uses the credentials from ~/.aws/credentials when no aws_access_key_id or aws_secret_access_key are explicitly passed.

If you’re switching between .env and ~/.aws/credentials, the environment variables will take precedence if both are present.

In ``~/.aws/credentials``
```
[default]
aws_access_key_id = YOUR_ACCESS_KEY_ID
aws_secret_access_key = YOUR_SECRET_ACCESS_KEY
```

### Create an S3 Bucket

Go to the S3 Console:

Create a new bucket or use an existing one.

Ensure your IAM user has read/write access to it.

Textract works with PDF files stored in S3. The pipeline will automatically convert input images to PDFs before sending them to Textract.

---

## OpenAI API Setup (ChatGPT)

The ChatGPT API is integrated via the `openai` Python package to post-process text extracted by Textract.
This pipeline uses the OpenAI ChatGPT API to:
- Correct OCR errors (e.g. misspellings, broken words)
- Extract structured entities like names, dates, companies, and locations

### Get an API Key

- Create an account at [OpenAI](https://platform.openai.com/).
- Go to your [API Keys page](https://platform.openai.com/account/api-keys) and generate a new key.

### Set Environment Variables

In your `.env` file, add the following:

```env
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4o-mini  # or gpt-4, gpt-3.5-turbo, etc.
```

---

3. Set up your .env file with the following keys:

```
AWS_ACCESS_KEY_ID=EXAMPLEACCESSKEYID (optional)
AWS_SECRET_ACCESS_KEY=EXAMPLESECRETACCESSKEY (optional)
BUCKET_NAME=your_s3_bucket_name
REGION=us-east-1
TEXTRACT_MAX_RETRIES=120
TEXTRACT_DELAY=5
MAX_THREADS=8
BATCH_SIZE=10

OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini

TMP_DIR=./tmp
INPUT_DIR=./input
OUTPUT_DIR=./output

IMAGE_MAGICK_COMMAND=magick

TEST_LEVENSHTEIN_MAX_DIFF_RATIO=0.15
TEST_WORD_COUNT_TOLERANCE=0.15
TEST_GPT_REPEAT_CORRECTIONS=10
TEST_TARGET_FILE=EXAMPLE_FILE
```
### Environment Variable Explanations

| Variable                             | Description                                                                             |
| ------------------------------------ | --------------------------------------------------------------------------------------- |
| `AWS_ACCESS_KEY_ID` *(optional)*     | AWS IAM access key. Required only if not using `~/.aws/credentials`.                    |
| `AWS_SECRET_ACCESS_KEY` *(optional)* | AWS IAM secret access key. Only used if the above is set.                               |
| `BUCKET_NAME`                        | Name of the S3 bucket used to upload PDFs and receive Textract output.                  |
| `REGION`                             | AWS region where your S3 bucket and Textract are hosted (e.g., `us-east-1`).            |
| `TEXTRACT_MAX_RETRIES`               | Max retries while polling for Textract job completion.                                  |
| `TEXTRACT_DELAY`                     | Delay (in seconds) between retries while waiting for Textract to finish.                |
| `MAX_THREADS`                        | Number of threads to use for parallel processing. Improves speed for large batches.     |
| `BATCH_SIZE`                         | Number of documents to process per batch. Helps control memory and API usage.           |
| `OPENAI_API_KEY`                     | Your OpenAI API key for accessing GPT models.                                           |
| `OPENAI_MODEL`                       | GPT model to use (`gpt-4o-mini`, `gpt-4`, `gpt-3.5-turbo`, etc.).                       |
| `TMP_DIR`                            | Local temporary folder used for processing intermediate files.                          |
| `INPUT_DIR`                          | Local folder containing input images or PDFs for processing.                            |
| `OUTPUT_DIR`                         | Folder where the corrected text, entities and other output is stored.                   |
| `IMAGE_MAGICK_COMMAND`               | CLI command for ImageMagick (`magick` on macOS, `convert` on Linux).                    |
| `TEST_LEVENSHTEIN_MAX_DIFF_RATIO`    | Max allowed difference ratio (0–1) for Levenshtein test. Flags major ChatGPT edits.     |
| `TEST_WORD_COUNT_TOLERANCE`          | Allowed word count difference ratio (0–1) between raw and corrected text.               |
| `TEST_GPT_REPEAT_CORRECTIONS`        | Number of times to re-run GPT correction to test consistency across runs.               |
| `TEST_TARGET_FILE`                   | Run tests on a specific file. If not set, tests apply to all available files.           |


1. Run the pipeline:
```
python main.py
```

---

## Testing
Testing is done using `pytest`. 

The `tests/` directory contains word count consistency tests to verify corrections didn't significantly alter content.

Run all tests:
```
pytest
```

Run a specific test:
```
pytest tests/test_wordcount.py
```

### Test Suite Overview

| Test File                      | Description                                                       |
| ------------------------------ | ----------------------------------------------------------------- |
| `test_wordcount.py`            | Compares the word count between raw OCR text and GPT-corrected output. Fails if the difference exceeds the threshold set in `TEST_WORD_COUNT_TOLERANCE`. Helps detect overcorrection or content loss.           |
| `test_levenshtein_distance.py` | Calculates the Levenshtein distance ratio between raw and corrected text. Fails if the ratio exceeds `TEST_LEVENSHTEIN_MAX_DIFF_RATIO`. Useful for catching excessive edits.                              |
| `test_chatgpt_consistency.py`  | Runs GPT correction multiple times on the same input (`TEST_GPT_REPEAT_CORRECTIONS`) and compares results. Flags inconsistencies in word count or Levenshtein distance across runs. Validates output stability.       |

---

## Logs & Outputs

Logs: /logs/MM-DD-YYYY_HH-MM-SS.log
Test Logs: /test_logs/test_MM-DD-YYYY_HH-MM-SS.log

Outputs:

- .raw.txt (original Textract output)
- .corrected.txt (GPT-corrected text)
- .coords.json (bounding box info)
- .entities.json (structured entities)
- .combined_output.json (page numbers and split letters)