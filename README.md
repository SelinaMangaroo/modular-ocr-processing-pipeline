# Modular OCR Processing Pipeline

This project is a **modular, batch OCR (Optical Character Recognition) pipeline** that can use multiple providers for both OCR and LLM (Large Language Model) post-processing.  

It extracts raw text from scanned documents (e.g. historical theater letters), corrects OCR errors, and structures key entities like people, productions, companies, theaters, and dates.  

---

## Features

- **Multiple OCR providers**:  
  - AWS Textract (requires S3 + Textract setup)  
  - Azure Document Intelligence  
- **Multiple LLM providers** for post-processing:  
  - OpenAI ChatGPT  
  - Anthropic Claude  
  - Local LLaMA via Ollama  
- **Pre-processing**:  
  - Converts `.jpg/.png/.tiff` images to `.pdf` via ImageMagick  
- **Post-processing**:  
  - Corrects OCR errors  
  - Extracts structured entities (names, productions, dates, etc.)  
  - Splits corrected text into letters per page  
- **Batch support** with configurable size and multithreaded execution  
- **Structured outputs**:  
  - `.raw.txt` – raw OCR text  
  - `.corrected.txt` – LLM-corrected text  
  - `.coords.json` – bounding box coordinates (from OCR)  
  - `.entities.json` – structured metadata  
  - `.combined_output.json` – split letters with page references  
- **Logging**: all operations logged to timestamped log files  

---

## Project Structure

```
ocr-processing-pipeline/
├── logs/                         # Runtime logs
├── output/                       # Output text, JSON, corrected results
├── venv/                         # Virtual environment (not tracked in version control)
├── .env                          # Environment config
├── .gitignore
├── PROJECT_STRUCTURE.md
├── README.md
├── input
├── llms
│   ├── chatgpt.py                # OpenAI GPT integration
│   ├── claude.py                 # Anthropic Claude integration
│   └── llama.py                  # Local LLaMA (Ollama) integration
├── main.py                       # Entry point for the pipeline
├── ocr_providers                 # OCR backends
│   ├── aws_provider.py           # AWS Textract implementation
│   └── azure_provider.py         # Azure Document Intelligence implementation
├── prompts.json                  # Shared LLM prompts
├── requirements.txt              # Python dependencies
├── schemas
│   └── llm_schemas.py            # Pydantic schemas for structured outputs
└── utils
    ├── aws_utils.py              # AWS helpers
    └── helpers.py                # File prep, batching, logging

```

---

## Setup

1. Clone the repository:

```bash
git clone https://github.com/SelinaMangaroo/modular-ocr-processing-pipeline.git
cd modular-ocr-processing-pipeline
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
sudo apt install imagemagick  # Linux
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
# General
MAX_THREADS=8
BATCH_SIZE=10
TMP_DIR=./tmp
INPUT_DIR=./input
OUTPUT_DIR=./output
IMAGE_MAGICK_COMMAND=magick   # "convert" on Linux

# OCR provider: aws | azure
OCR_PROVIDER=aws

# LLM provider: chatgpt | claude | llama
LLM_PROVIDER=chatgpt

# AWS (only if OCR_PROVIDER=aws)
BUCKET_NAME=your-s3-bucket
REGION=us-east-1
AWS_ACCESS_KEY_ID=EXAMPLEACCESSKEYID (optional)
AWS_SECRET_ACCESS_KEY=EXAMPLESECRETACCESSKEY (optional)
TEXTRACT_MAX_RETRIES=120
TEXTRACT_DELAY=5

# OpenAI (only if LLM_PROVIDER=chatgpt)
OPENAI_API_KEY=your-openai-key
OPENAI_MODEL=gpt-4o-mini

# Anthropic (only if LLM_PROVIDER=claude)
ANTHROPIC_API_KEY=your-anthropic-key
CLAUDE_MODEL=claude-3-7-sonnet-20250219

# Ollama (only if LLM_PROVIDER=llama)
LLAMA_MODEL=llama3.1:8b

```

### Environment Variable Explanations

| General Variables                    | Description                                                                             |
| `MAX_THREADS`                        | Number of threads to use for parallel processing. Improves speed for large batches.     |
| `BATCH_SIZE`                         | Number of documents to process per batch. Helps control memory and API usage.           |
| `TMP_DIR`                            | Local temporary folder used for processing intermediate files.                          |
| `INPUT_DIR`                          | Local folder containing input images or PDFs for processing.                            |
| `OUTPUT_DIR`                         | Folder where the corrected text, entities and other output is stored.                   |
| `IMAGE_MAGICK_COMMAND`               | CLI command for ImageMagick (`magick` on macOS, `convert` on Linux).                    |
| `OCR_PROVIDER`                       | Provider for the OCR Service                                                            |
| `LLM_PROVIDER`                       | Provider for the LLM Service                                                            |

| AWS Variables                        | Description                                                                             |
| ------------------------------------ | --------------------------------------------------------------------------------------- |
| `AWS_ACCESS_KEY_ID` *(optional)*     | AWS IAM access key. Required only if not using `~/.aws/credentials`.                    |
| `AWS_SECRET_ACCESS_KEY` *(optional)* | AWS IAM secret access key. Only used if the above is set.                               |
| `BUCKET_NAME`                        | Name of the S3 bucket used to upload PDFs and receive Textract output.                  |
| `REGION`                             | AWS region where your S3 bucket and Textract are hosted (e.g., `us-east-1`).            |
| `TEXTRACT_MAX_RETRIES`               | Max retries while polling for Textract job completion.                                  |
| `TEXTRACT_DELAY`                     | Delay (in seconds) between retries while waiting for Textract to finish.                |

| Azure Variables                       | Description                                                                            |
| ------------------------------------- | -------------------------------------------------------------------------------------- |
| `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT`| Endpoint needed for the Read API Service.                                              |
| `AZURE_DOCUMENT_INTELLIGENCE_KEY`     | Key needed for the Read API Service.                                                   |

| OpenAI Variables                      | Description                                                                            |
| ------------------------------------- | -------------------------------------------------------------------------------------- |
| `OPENAI_API_KEY`                      | Your OpenAI API key for accessing GPT models.                                          |
| `OPENAI_MODEL`                        | GPT model to use (`gpt-4o-mini`, `gpt-4`, `gpt-3.5-turbo`, etc.).                      |

| Claude Variables                      | Description                                                                            |
| ------------------------------------- | -------------------------------------------------------------------------------------- |
| `ANTHROPIC_API_KEY`                   | Your Anthropic API key for accessing Claude models.                                    |
| `CLAUDE_MODEL`                        | Claude model to use.                                                                   |

| LLAMA Variables                       | Description                                                                            |
| ------------------------------------- | -------------------------------------------------------------------------------------- |
| `LLAMA_MODEL`                         | Llama model to use.                                                                    |


1. Run the pipeline:
```
python main.py
```

---


## Logs & Outputs

Logs: /logs/MM-DD-YYYY_HH-MM-SS.log
Outputs: stored in OUTPUT_DIR/<filename>/

Outputs:

- .raw.txt (original raw text output)
- .corrected.txt (llm-corrected text)
- .coords.json (bounding box info)
- .entities.json (structured entities)
- .combined_output.json (page numbers and split letters)