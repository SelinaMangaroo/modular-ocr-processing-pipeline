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

## How the Pipeline Works (Batch Flow)

This pipeline operates in three major phases, repeated for every batch of documents:

1. OCR Phase (AWS Textract or Azure Document Intelligence)

For each file in the batch:
- Images (.jpg, .png, .tif) → converted to PDF
- PDF → uploaded to S3 (AWS only)
- OCR is performed

The pipeline produces:
- `file.raw.txt` — raw extracted text
- `file.coords.json` — bounding box geometry

If `SKIP_OCR_IF_EXISTS=true` and both files exist, OCR is skipped.

2. LLM Phase (Batch 1: Text Correction)

The pipeline queues all correction tasks from the current batch into a single LLM batch request (ChatGPT or Claude):

- Fixes OCR mistakes
- Preserves original meaning
- No added or inferred content

Produces:
- `file.corrected.txt`

After the batch completes, results are hydrated into the output directory.

3. LLM Phase (Batch 2: Entity Extraction + Letter Splitting)

After corrected text is available:

- Extracts structured metadata (People, Productions, Dates, Companies, Theaters)
- Splits multi-letter pages into multiple logical letters

Produces:
- `file.entities.json`
- `file.combined_output.json`

All tasks for the batch are combined into one LLM batch submission and hydrated afterward.

---

## Project Structure

```
ocr-processing-pipeline/
├── logs/                         # Runtime logs
├── input/                        # Input directory with files to be run in pipeline
├── output/                       # Output text, JSON, corrected results
├── venv/                         # Virtual environment (not tracked in version control)
├── .env                          # Environment config
├── .gitignore
├── PROJECT_STRUCTURE.md
├── README.md
├── tmp/                          # tmp files created during the ocr process
├── batchfiles/                   # Batchfiles created from the llm providers
├── llms
│   ├── chatgpt.py                # OpenAI GPT integration
│   ├── claude.py                 # Anthropic Claude integration
├── main.py                       # Entry point for the pipeline
├── ocr_providers                 # OCR backends
│   ├── aws_provider.py           # AWS Textract implementation
│   └── azure_provider.py         # Azure Document Intelligence implementation
├── prompts.json                  # Shared LLM prompts
├── requirements.txt              # Python dependencies
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


## Azure Document Intelligence Setup

Azure’s Document Intelligence service can extract printed/handwritten text and layout information.

### Create a Resource

- Create an Azure account if you don't already have one and go to the Azure Portal
- Create a Document Intelligence resource.
- Note down your Endpoint URL and API Key.

### Set Environment Variables

In your .env file:

```env
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://<your-resource-name>.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-azure-api-key
```

The pipeline uses the prebuilt-read model by default.

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

## Anthropic Claude Setup

Claude is Anthropic’s LLM, accessed via the Anthropic Python SDK.

### Get an API Key

Create an account at Anthropic.
Go to your API Keys page and generate a key.

### Set Environment Variables

In your `.env` file, add the following:

```env
ANTHROPIC_API_KEY=your-anthropic-api-key
CLAUDE_MODEL=claude-3-7-sonnet-20250219
```

The CLAUDE_MODEL can be swapped with other Claude models.

---

## LLM Prompt Behaviors (What Each Prompt Does)
The pipeline uses three core prompts defined in `prompts.json`.

1. Text Correction Prompt (correct_text)

Purpose:
- Fix OCR artifacts
- Correct broken words, extra spacing, punctuation errors
- No new content is ever added
- If uncertain → leave text unchanged

Output becomes `file.corrected.txt`.

2. Entity Extraction + Explanation (extract_and_explain_entities)

Purpose:

  Identify and explain:
  - People
  - Productions
  - Companies
  - Theaters
  - Dates

  Provide accurate historical context, not just context from the letter
  - Attempt to include a reliable source
  - Output must be valid JSON only

Results saved in `file.entities.json`.

3. Letter Splitting Prompt (split_letters)

Purpose:

- Detect how many individual letters appear in the text

- Split based on:
  - Greetings (Dear Sir, Gentlemen, Friend, etc.)
  - Addresses / headings
  - Closings (Yours truly, Sincerely, etc.)

- Output must be a JSON array of strings, each one full letter

Saved in `file.combined_output.json`.

---


1. Set up your .env file with the following keys:

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

SKIP_OCR_IF_EXISTS=true

# LLM provider: chatgpt | claude
LLM_PROVIDER=chatgpt

# AWS (only if OCR_PROVIDER=aws)
BUCKET_NAME=your-s3-bucket
REGION=us-east-1
AWS_ACCESS_KEY_ID=EXAMPLEACCESSKEYID (optional)
AWS_SECRET_ACCESS_KEY=EXAMPLESECRETACCESSKEY (optional)
TEXTRACT_MAX_RETRIES=120
TEXTRACT_DELAY=5

# Azure Document Intelligence (only if OCR_PROVIDER=azure)
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://azure-ocr......com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-azure-key

# OpenAI (only if LLM_PROVIDER=chatgpt)
OPENAI_API_KEY=your-openai-key
OPENAI_MODEL=gpt-4o-mini

# Anthropic (only if LLM_PROVIDER=claude)
ANTHROPIC_API_KEY=your-anthropic-key
CLAUDE_MODEL=claude-3-7-sonnet-20250219

```

### Environment Variable Explanations

| General Variables                    | Description                                                                             |
| ------------------------------------ | --------------------------------------------------------------------------------------- |
| `MAX_THREADS`                        | Number of threads to use for parallel processing. Improves speed for large batches.     |
| `BATCH_SIZE`                         | Number of documents to process per batch. Helps control memory and API usage.           |
| `TMP_DIR`                            | Local temporary folder used for processing intermediate files.                          |
| `INPUT_DIR`                          | Local folder containing input images or PDFs for processing.                            |
| `OUTPUT_DIR`                         | Folder where the corrected text, entities and other output is stored.                   |
| `IMAGE_MAGICK_COMMAND`               | CLI command for ImageMagick (`magick` on macOS, `convert` on Linux).                    |
| `OCR_PROVIDER`                       | Provider for the OCR Service                                                            |
| `SKIP_OCR_IF_EXISTS`                 | Skip OCR step if the raw text and coordinate files already exist                        |
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


1. Run the pipeline:
```
python main.py
```

---


## Logs & Outputs

Logs: /logs/MM-DD-YYYY_HH-MM-SS.log
Outputs: stored in OUTPUT_DIR/<filename>/

Output Structure (What the Pipeline Generates):

Each processed file receives its own folder under OUTPUT_DIR, containing all OCR + LLM results.

```
output/
└── LetterBook_L-01_0175/
    ├── LetterBook_L-01_0175.raw.txt
    ├── LetterBook_L-01_0175.corrected.txt
    ├── LetterBook_L-01_0175.coords.json
    ├── LetterBook_L-01_0175.entities.json
    └── LetterBook_L-01_0175.combined_output.json
```

- .raw.txt (Raw OCR output, uncorrected)
- .corrected.txt (llm-corrected text)
- .coords.json (Word-level bounding boxes from OCR)
- .entities.json (Structured metadata extracted by LLM)
- .combined_output.json (All detected letters broken into segments)