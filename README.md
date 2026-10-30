# Modular OCR Processing Pipeline

A batch pipeline that transcribes scanned, handwritten and typed documents (originally the Shubert Brothers archive, roughly 60,000 pages) and extracts structured entities from them.

The production setup uses **Google Gemini** (`gemini-3.1-pro-preview`) through the **Batch API**, which is half the price of standard calls. Each image goes through two steps:

1. **Transcription**: Gemini reads the image and returns the text verbatim, preserving the author's spelling, punctuation and line breaks. Unreadable text is marked `[illegible]`.
2. **Entity extraction**: Gemini reads the transcription and returns structured JSON for people, productions, companies, theaters and dates.

There is no separate OCR service, no raw-text pass and no letter splitting.

---

## Output

Every image gets its own folder with exactly two files:

```
<output folder>/
└── LS_EuroTrip1951_B01_F02_009/
    ├── LS_EuroTrip1951_B01_F02_009.corrected.txt   # verbatim transcription
    └── LS_EuroTrip1951_B01_F02_009.entities.json   # structured entities (valid JSON)
```

Each output folder also gets a `run_summary.json` with provider, model, document count, token usage, cost, duration and the audit result.

---

## Project Structure

```
modular-ocr-processing-pipeline/
├── main.py                      # Processes ONE input folder (or one file) end to end
├── run_batch_directories.py     # Runs main.py across the whole archive, then audits the result
├── count_processed.py           # Quick count of .corrected.txt files produced so far
├── prompts.json                 # gemini_transcribe and extract_and_explain_entities prompts
├── requirements.txt
├── PROJECT_STRUCTURE.md
├── README.md
├── llms/
│   ├── gemini.py                # Production provider (Batch API)
│   ├── claude2.py               # Claude, same consolidated flow (used for cost comparison)
│   ├── chatgpt2.py              # ChatGPT, same consolidated flow (used for cost comparison)
│   ├── claude.py                # Legacy, used with an OCR provider
│   └── chatgpt.py               # Legacy, used with an OCR provider
├── ocr_providers/               # Legacy OCR backends
│   ├── aws_provider.py
│   └── azure_provider.py
└── utils/
    ├── helpers.py               # Logging, batching, image resize, completeness audit
    └── aws_utils.py             # Legacy AWS helpers
```

Created at runtime: `logs/`, `tmp/`, `batchfiles/` (batch request files and the pending-job ledger) and the output folders.

---

## Setup

```bash
git clone https://github.com/SelinaMangaroo/modular-ocr-processing-pipeline.git
cd modular-ocr-processing-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**ImageMagick** must be installed system-wide. It is not a Python package.

```bash
brew install imagemagick       # macOS
sudo apt install imagemagick   # Linux
```


### Gemini API key

Create a key in [Google AI Studio](https://aistudio.google.com/). If the key belongs to an organization's Google Cloud project, the Gemini API must be enabled on that project and allowed in the key's API restrictions, otherwise calls fail with a 403 `SERVICE_DISABLED` error.

---

## Configuration (`.env`)

```env
# General
MAX_THREADS=8                  # image resize/upload threads inside one folder
BATCH_SIZE=30                  # images per Gemini batch job
TMP_DIR=./tmp
IMAGE_MAGICK_COMMAND=convert   # "magick" on macOS, "convert" on Linux
SKIP_OCR_IF_EXISTS=true        # reuse existing transcriptions and valid entities on reruns

# Provider
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-key
GEMINI_MODEL=gemini-3.1-pro-preview
GEMINI_OCR_THINKING_LEVEL=low
GEMINI_TEXT_THINKING_LEVEL=low

# Single folder or single file (used when running main.py directly)
INPUT_DIR=/path/to/folder/or/image.jpg
OUTPUT_DIR=/path/to/output

# Whole-archive runs (used by run_batch_directories.py)
BATCH_SOURCE_ROOT=/thunderbolt/shubert_aug_2026/shubert
BATCH_OUTPUT_ROOT=/data/modular-ocr-processing-pipeline/Official Gemini Output
BATCH_CONCURRENCY=5
# BATCH_LIMIT=2
# BATCH_FORCE=false
# BATCH_DRY_RUN=false
```

| Variable | Description |
| --- | --- |
| `MAX_THREADS` | Threads used to resize and upload images within a folder. |
| `BATCH_SIZE` | Images per Gemini batch job. |
| `IMAGE_MAGICK_COMMAND` | ImageMagick command (`convert` on Linux, `magick` on macOS). |
| `SKIP_OCR_IF_EXISTS` | Skip transcription for images that already have a `.corrected.txt`, and skip entity extraction for images that already have a valid `.entities.json`. |
| `GEMINI_OCR_THINKING_LEVEL`, `GEMINI_TEXT_THINKING_LEVEL` | Gemini 3.1 Pro defaults to high thinking, which is billed as output tokens. `low` keeps cost down. |
| `INPUT_DIR`, `OUTPUT_DIR` | Used by `main.py`. `INPUT_DIR` can be a folder or a single image. |
| `BATCH_SOURCE_ROOT` | Top-level folder to scan. Must be an absolute path. The scan finds every folder named exactly `JPG` at any depth. |
| `BATCH_OUTPUT_ROOT` | Where the `output-*` folders are created. |
| `BATCH_CONCURRENCY` | How many folders run in parallel (default 5). |
| `BATCH_LIMIT`, `BATCH_FORCE`, `BATCH_DRY_RUN` | Environment versions of `--limit`, `--force` and `--dry-run`. |

`python-dotenv` never overrides variables already set in the shell. A stray exported variable (for example an empty `ANTHROPIC_API_KEY`) will shadow `.env`, so `unset` it if values look ignored. The batch driver passes `INPUT_DIR` and `OUTPUT_DIR` to each folder run itself, so those two do not need editing between folders.

---

## Running

### One folder or one image

```bash
python main.py        # uses INPUT_DIR / OUTPUT_DIR from .env
```

### The whole archive

For a JPG folder at `.../<CollectionName_BXX_FYY>/JPG`, the output goes to `<BATCH_OUTPUT_ROOT>/output-shubert-<CollectionName_BXX_FYY>`.

```bash
tmux new -s shubert
source .venv/bin/activate

python run_batch_directories.py --dry-run     # list what would run, spend nothing
python run_batch_directories.py --limit 2     # test on the first 2 folders and spot-check the text
python run_batch_directories.py               # full run
```

Detach from tmux with `Ctrl+B` then `D`, and reattach with `tmux attach -t shubert`. The run keeps going if your connection drops.

| Flag | Description |
| --- | --- |
| `--source-root`, `--output-root` | Override `BATCH_SOURCE_ROOT` / `BATCH_OUTPUT_ROOT`. |
| `--dry-run` | Show the plan without processing. |
| `--limit N` | Process only the first N folders needing work. |
| `--force` | Rerun folders even if they are already complete. |
| `--concurrency N` | Folders processed in parallel. |
| `--audit-only` | Process nothing. Compare all source images with outputs and report gaps. |

A folder counts as **done** only when every image in it has a valid `.corrected.txt` and `.entities.json`. Rerunning the driver always picks up exactly the folders with gaps and skips finished work.

---

## Making Sure Everything Was Processed

Completeness is checked automatically, not by inspecting folders by hand.

- **End of every folder run (`main.py`)**: each input image is checked. Anything missing is logged as `[AUDIT]` errors, the audit result is saved in `run_summary.json`, and the run exits with an error so the driver lists the folder as FAILED.
- **End of every driver run (`run_batch_directories.py`)**: all source images are compared against all outputs and the result is printed and logged: folders, source images, fully processed, and missing. Every problem is written to `logs/audit_<timestamp>.csv`, and the run ends with `AUDIT RESULT: PASS` or `FAIL`. The exit code is 1 on any failure.
- **On demand**: `python run_batch_directories.py --audit-only`.

Problem types in the report:

| Problem | Meaning |
| --- | --- |
| `MISSING_CORRECTED`, `MISSING_ENTITIES` | An output file is absent. |
| `INVALID_ENTITIES_JSON` | `.entities.json` is not valid JSON. |
| `EMPTY_ENTITIES` | `.entities.json` is `{}`, which the pipeline writes when Gemini's entity output could not be parsed. |
| `EMPTY_CORRECTED` | Empty transcription (for example a blank page). Listed for human review and does not fail the audit. |

To fix a FAIL, run `python run_batch_directories.py` again. Repeat until the audit passes. The audit proves outputs exist and are valid. It does not judge transcription accuracy, so keep spot-checking text quality. It only sees images inside folders named exactly `JPG`.

---

## Gemini Batch API: What to Expect

- A batch job can finish in minutes or take many hours. Jobs expire after 48 hours, and results stay available for about 6 weeks.
- There is no web interface for Gemini Developer API batches in Google Cloud. Check status from the logs, which print the job state while polling (`JOB_STATE_PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`, `CANCELLED`, `EXPIRED`), or through the API with `client.batches.get(name=...)` and `client.batches.list()`.
- Temporary 503 errors can occur on any call. Polling and downloading retry automatically.
- A batch can succeed overall while individual requests inside it fail. Those images end up missing outputs, and the audit catches them.
- Timing reference: 13 folders run concurrently took about 1h54m.

### If a run is interrupted

If a run is killed or a download fails, the images that were in flight are simply missing outputs. Run `python run_batch_directories.py --audit-only` to list the gaps, then run `python run_batch_directories.py` again. Only folders with gaps are reprocessed, and finished images are skipped. Any pages whose batch had already finished on Google's side but was not saved are submitted again and billed again, so run inside tmux to avoid interruptions.

---

## Cost

Batch pricing for `gemini-3.1-pro-preview` is $1.00 per million input tokens and $6.00 per million output tokens (up to 200K context), half the standard rate. Thinking tokens are billed as output, which is why thinking is set to `low`.

- Measured cost is about **$0.058 per page** for transcription plus entities.
- 60,000 pages comes to roughly **$4,000**.
- Per-document token counts and estimated cost are written to the log, and each folder's `run_summary.json` has totals.

Because rerunning is skipped for finished work (`SKIP_OCR_IF_EXISTS=true`), recovering gaps costs only the missing pages. Using `--force` together with `SKIP_OCR_IF_EXISTS=false` repays the full cost, so check the page count first.

---

## Prompts (`prompts.json`)

1. **`gemini_transcribe`**: transcribe the image exactly as written, keeping the author's spelling, punctuation, capitalization, wording and line breaks, including mistakes. Genuinely unreadable text becomes `[illegible]`. Output is the text only, with no headers, labels or commentary. Saved as `.corrected.txt`.
2. **`extract_and_explain_entities`**: identify and explain people, productions, companies, theaters and dates with accurate historical context, output as valid JSON only. Saved as `.entities.json`.

---

## Logs

| File | Contents |
| --- | --- |
| `logs/MM-DD-YYYY_HH-MM-SS_pid<PID>.log` | One log per `main.py` run (the process ID prevents collisions between concurrent folders). |
| `logs/batch_run_<timestamp>.txt` | Master log for a driver run: per-folder OK/FAILED and the audit summary. |
| `logs/audit_<timestamp>.csv` | Every problem found by the completeness audit. |

---

## Other Providers

`LLM_PROVIDER` accepts `gemini` (production), `claude2` and `chatgpt2`. The latter two use the same flow (vision model transcribes, then extracts entities, via each vendor's batch API) and were used to compare cost across providers. Set the matching `ANTHROPIC_API_KEY` / `CLAUDE_MODEL` or `OPENAI_API_KEY` / `OPENAI_MODEL` to use them.

Measured cost per page for the comparison: Gemini about $0.058, Claude about $0.027, ChatGPT about $0.029. Gemini is the chosen provider.

### Legacy flow (AWS Textract / Azure + ChatGPT / Claude)

`llms/chatgpt.py`, `llms/claude.py` and `ocr_providers/` are the original pipeline: OCR with AWS Textract or Azure Document Intelligence, then LLM correction and letter splitting. They remain in the repo . They produce `.raw.txt`, `.coords.json` and `.combined_output.json`, and they rely on the `correct_text` and `split_letters` prompts, which are no longer in `prompts.json` . Selecting these providers requires `OCR_PROVIDER=aws|azure`, `LLM_PROVIDER=chatgpt|claude`, and the AWS (`BUCKET_NAME`, `REGION`, credentials) or Azure (`AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT`, `AZURE_DOCUMENT_INTELLIGENCE_KEY`) settings.