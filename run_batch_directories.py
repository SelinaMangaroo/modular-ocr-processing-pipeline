"""
Scans a source root for every directory literally named "JPG" and runs the
pipeline (main.py) once per folder found, deriving a matching output folder
name automatically. INPUT_DIR/OUTPUT_DIR are passed as real environment
variables to each subprocess call — python-dotenv's load_dotenv() (used by
main.py) never overrides variables already set in the environment, so this
takes priority over .env without ever editing that file.

Output naming: for a JPG folder at .../<CollectionName_BXX_FYY>/JPG, the
output folder is <output_root>/output-shubert-<CollectionName_BXX_FYY> —
matching the convention already in use (e.g.
output-shubert-LS_Correspondences1908-1910_B03_F01).

Usage:
    # Preview what would run, without spending anything:
    python run_batch_directories.py --source-root /thunderbolt/shubert_aug_2026/shubert --output-root "/data/modular-ocr-processing-pipeline/Official Gemini Output" --dry-run

    # Run everything found:
    python run_batch_directories.py --source-root /thunderbolt/shubert_aug_2026/shubert --output-root "/data/modular-ocr-processing-pipeline/Official Gemini Output"

    # Test on just the first 2 folders before committing to the full run:
    python run_batch_directories.py --source-root /thunderbolt/shubert_aug_2026/shubert --output-root "/data/modular-ocr-processing-pipeline/Official Gemini Output" --limit 2

    # Compare every source image with its outputs, no processing (exit 1 if anything is missing):
    python run_batch_directories.py --audit-only

    # Re-run folders even if they are already complete:
    python run_batch_directories.py --source-root ... --output-root ... --force
"""

import os
import sys
import argparse
import subprocess
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import csv
from dotenv import load_dotenv
from utils.helpers import list_source_images, audit_outputs

load_dotenv()

PIPELINE_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".pdf")


def find_jpg_folders(source_root):
    """Walks source_root and yields every directory literally named 'JPG'."""
    for dirpath, dirnames, _ in os.walk(source_root):
        for d in dirnames:
            if d == "JPG":
                yield os.path.join(dirpath, d)


def has_images(folder):
    try:
        return any(f.lower().endswith(IMAGE_EXTENSIONS) for f in os.listdir(folder))
    except OSError:
        return False


def output_name_for(jpg_folder):
    """Derives the output folder name from the JPG folder's parent directory name."""
    parent_name = os.path.basename(os.path.dirname(jpg_folder))
    return f"output-shubert-{parent_name}"


def audit_folder(jpg_folder, output_dir):
    """Returns (n_images, problems, review) by checking real output files, not run_summary.json."""
    images = list_source_images(jpg_folder)
    problems, review = audit_outputs(images, output_dir)
    return len(images), problems, review


def already_done(jpg_folder, output_dir):
    """A folder is done only if every image has a valid .corrected.txt and .entities.json."""
    n, problems, _ = audit_folder(jpg_folder, output_dir)
    return n > 0 and not problems


def run_full_audit(jpg_folders, output_root, log):
    """
    Whole-archive check. Compares every source image with its outputs, writes
    logs/audit_<timestamp>.csv, prints totals. Returns True only if nothing is missing.
    """
    totals = dict(folders=0, folders_complete=0, images=0, complete=0)
    rows, review_rows = [], []
    expected_dirs = set()
    for jpg_folder in jpg_folders:
        output_dir = os.path.join(output_root, output_name_for(jpg_folder))
        expected_dirs.add(os.path.basename(output_dir))
        n, problems, review = audit_folder(jpg_folder, output_dir)
        if n == 0:
            continue
        totals["folders"] += 1
        totals["images"] += n
        bad_files = {p[0] for p in problems}
        totals["complete"] += n - len(bad_files)
        if not problems:
            totals["folders_complete"] += 1
        for fname, kind, detail in problems:
            rows.append((kind, jpg_folder, fname, detail))
        for fname, kind, detail in review:
            review_rows.append((kind, jpg_folder, fname, detail))

    # Output folders that no source folder maps to (usually a naming mismatch)
    unexpected = []
    if os.path.isdir(output_root):
        unexpected = sorted(d for d in os.listdir(output_root)
                            if d.startswith("output-shubert-") and d not in expected_dirs)

    report = os.path.join(PIPELINE_ROOT, "logs", f"audit_{datetime.now().strftime('%m%d%Y_%H%M%S')}.csv")
    os.makedirs(os.path.dirname(report), exist_ok=True)
    with open(report, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["problem", "source_folder", "image", "detail"])
        w.writerows(rows)
        w.writerows(review_rows)

    missing = totals["images"] - totals["complete"]
    log("=== COMPLETENESS AUDIT ===")
    log(f"JPG folders with images : {totals['folders']}  (complete: {totals['folders_complete']}, incomplete: {totals['folders'] - totals['folders_complete']})")
    log(f"Source images           : {totals['images']}")
    log(f"Fully processed         : {totals['complete']}  (valid .corrected.txt + .entities.json)")
    log(f"Missing / incomplete    : {missing}")
    if review_rows:
        log(f"Needs human review      : {len(review_rows)}  (e.g. empty transcription; see report)")
    if unexpected:
        log(f"Output folders with no matching source folder: {len(unexpected)} (e.g. {unexpected[0]})")
    log(f"Audit report            : {report}")
    ok = (missing == 0)
    log("AUDIT RESULT: PASS - every source image has both outputs." if ok
        else "AUDIT RESULT: FAIL - rerun `python run_batch_directories.py` to process the gaps (completed work is skipped).")
    return ok


def _env_bool(name, default=False):
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", default=None,
        help="Top-level directory to scan for JPG folders. Falls back to BATCH_SOURCE_ROOT in .env if omitted."
    )
    parser.add_argument(
        "--output-root", default=None,
        help="Where each folder's output-shubert-* subfolder goes. Falls back to BATCH_OUTPUT_ROOT in .env if omitted."
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=None,
        help="List what would run without actually running it. Falls back to BATCH_DRY_RUN in .env if omitted."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N folders found. Falls back to BATCH_LIMIT in .env if omitted."
    )
    parser.add_argument(
        "--force", action="store_true", default=None,
        help="Re-run folders even if every image already has valid outputs. Falls back to BATCH_FORCE in .env if omitted."
    )
    parser.add_argument(
        "--concurrency", type=int, default=None,
        help="How many folders to process in parallel (each waits on its own Gemini batch job). "
             "Falls back to BATCH_CONCURRENCY in .env, or 5 if neither is set. Running sequentially "
             "(--concurrency 1) can take many hours to days for large collections since each folder's "
             "batch job can queue for a while on Google's side."
    )
    parser.add_argument(
        "--audit-only", action="store_true",
        help="Don't run anything. Just compare every source image with its outputs and report gaps (exit 1 if any)."
    )
    args = parser.parse_args()

    # CLI flags win when given; otherwise fall back to .env; otherwise a sensible default.
    args.source_root = args.source_root or os.getenv("BATCH_SOURCE_ROOT")
    args.output_root = args.output_root or os.getenv("BATCH_OUTPUT_ROOT")
    args.dry_run = args.dry_run if args.dry_run is not None else _env_bool("BATCH_DRY_RUN", False)
    args.force = args.force if args.force is not None else _env_bool("BATCH_FORCE", False)
    if args.limit is None:
        env_limit = os.getenv("BATCH_LIMIT")
        args.limit = int(env_limit) if env_limit else None
    if args.concurrency is None:
        args.concurrency = int(os.getenv("BATCH_CONCURRENCY", "5"))

    if not args.source_root:
        parser.error("--source-root is required (or set BATCH_SOURCE_ROOT in .env)")
    if not args.output_root:
        parser.error("--output-root is required (or set BATCH_OUTPUT_ROOT in .env)")

    if not os.path.isdir(args.source_root):
        parser.error(
            f"Source root does not exist or is not a directory: {args.source_root} "
            "(relative paths resolve against the pipeline directory. Did you drop the leading '/'?)"
        )

    jpg_folders = sorted(find_jpg_folders(args.source_root))

    if not jpg_folders:
        print(f"No 'JPG' folders found under {args.source_root}")
        sys.exit(1)

    if args.audit_only:
        ok = run_full_audit(jpg_folders, args.output_root, print)
        sys.exit(0 if ok else 1)

    plan = []
    for jpg_folder in jpg_folders:
        output_dir = os.path.join(args.output_root, output_name_for(jpg_folder))
        if not has_images(jpg_folder):
            plan.append((jpg_folder, output_dir, "skip-empty"))
            continue
        if not args.force and already_done(jpg_folder, output_dir):
            plan.append((jpg_folder, output_dir, "skip-done"))
            continue
        plan.append((jpg_folder, output_dir, "run"))

    to_run = [p for p in plan if p[2] == "run"]
    if args.limit:
        skipped_by_limit = to_run[args.limit:]
        to_run = to_run[: args.limit]
    else:
        skipped_by_limit = []

    print(f"Found {len(jpg_folders)} JPG folder(s) total.")
    print(f"  To run:          {len(to_run)}")
    print(f"  Already done:    {sum(1 for p in plan if p[2] == 'skip-done')}")
    print(f"  Empty, skipped:  {sum(1 for p in plan if p[2] == 'skip-empty')}")
    if args.limit:
        print(f"  Held back by --limit: {len(skipped_by_limit)}")
    print()

    if args.dry_run:
        print("--- DRY RUN: would process ---")
        for jpg_folder, output_dir, status in plan:
            marker = "RUN " if status == "run" else status.upper()
            print(f"  [{marker}] {jpg_folder}  ->  {output_dir}")
        return

    master_log_path = os.path.join(
        PIPELINE_ROOT, "logs", f"batch_run_{datetime.now().strftime('%m%d%Y_%H%M%S')}.txt"
    )
    os.makedirs(os.path.dirname(master_log_path), exist_ok=True)

    def log(msg):
        line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line)
        with open(master_log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"Starting batch run: {len(to_run)} folder(s) to process, concurrency={args.concurrency}.")

    log_lock = threading.Lock()

    def process_one(item):
        jpg_folder, output_dir, _ = item
        os.makedirs(output_dir, exist_ok=True)

        env = os.environ.copy()
        env["INPUT_DIR"] = jpg_folder
        env["OUTPUT_DIR"] = output_dir

        start = time.time()
        result = subprocess.run(
            [sys.executable, "main.py"],
            cwd=PIPELINE_ROOT,
            env=env,
        )
        elapsed = time.time() - start
        return jpg_folder, output_dir, result.returncode, elapsed

    succeeded, failed = [], []
    completed_count = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(process_one, item): item for item in to_run}
        for future in as_completed(futures):
            jpg_folder, output_dir, returncode, elapsed = future.result()
            with log_lock:
                completed_count += 1
                if returncode == 0:
                    succeeded.append(jpg_folder)
                    log(f"[{completed_count}/{len(to_run)}] OK ({elapsed:.0f}s): {output_dir}")
                else:
                    failed.append(jpg_folder)
                    log(f"[{completed_count}/{len(to_run)}] FAILED (exit {returncode}, {elapsed:.0f}s): {jpg_folder}")

    log(f"Batch run complete. Succeeded: {len(succeeded)}, Failed: {len(failed)}")
    if failed:
        log("Failed folders:")
        for f in failed:
            log(f"  - {f}")

    audit_ok = run_full_audit(jpg_folders, args.output_root, log)
    print(f"\nFull log: {master_log_path}")
    if failed or not audit_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()