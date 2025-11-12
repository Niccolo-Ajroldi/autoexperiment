#!/usr/bin/env python3
import csv
import json
import subprocess
from pathlib import Path
from filelock import FileLock
from concurrent.futures import ThreadPoolExecutor, as_completed

R = "\033[0m"
G = "\033[92m"
Y = "\033[93m"
P = "\033[95m"

BASE_DIR = Path("/leonardo_work/AIFAC_L01_028/najroldi/oellm_scaling")
CSV_PATH = BASE_DIR / "jobs.csv"
CSV_LOCK_PATH = CSV_PATH.with_suffix(".csv.lock")


def read_csv():
    with FileLock(str(CSV_LOCK_PATH)):
        with CSV_PATH.open(newline="") as f:
            return list(csv.DictReader(f))


def write_csv(rows):
    with FileLock(str(CSV_LOCK_PATH)):
        with CSV_PATH.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def mark_synced(job_name: str):
    """Mark a job as synced only if it's finished."""
    rows = read_csv()
    for r in rows:
        if r["job_name"] == job_name and r["status"] == "finished":
            r["wandb_synced"] = "True"
    write_csv(rows)


def sync_run(run: Path, job_name: str):
    print(f"  {G}-> Syncing: {run.name}{R}")
    result = subprocess.run(
        "wandb sync --mark-synced --no-include-synced *.wandb",
        cwd=run,
        shell=True,
        capture_output=True,
        text=True,
    )

    # We mark a run as synced if it has a summary
    summary_path = run / "files" / "wandb-summary.json"
    run_has_summary = False
    if summary_path.is_file():
        try:
            with summary_path.open() as f:
                data = json.load(f)
            run_has_summary = len(data) > 0
        except Exception:
            run_has_summary = False

    if result.returncode == 0 and run_has_summary:
        print(f"     {P}Done:{R} {run.name}")
        mark_synced(job_name)
    else:
        print(f"     {Y}Failed:{R} {run.name}")
        if result.stderr:
            print(result.stderr.strip())


def process_job(row):
    wandb_dir = BASE_DIR / "ckpts" / row["job_name"] / "wandb" / "wandb"
    if not wandb_dir.is_dir():
        return False

    print(f"{P}Syncing:{R} {Y}{row['job_name']}{R}")
    runs = [r for r in wandb_dir.iterdir() if r.is_dir() and r.name.startswith("offline-run-")]
    if not runs:
        return False

    # SERIAL sync of runs
    for run in runs:
        sync_run(run, row["job_name"])
    return True


def sync_wandb(max_jobs: int = 8):
    print(f"\n=============================================")
    print(f"Scanning for wandb runs in {G}{BASE_DIR}{R}")
    print(f"Extracting runs from {G}{CSV_PATH}{R}")
    print(f"Using {G}{max_jobs}{R} threads")
    print(f"=============================================\n")

    rows = read_csv()
    to_sync = [r for r in rows if r["wandb_synced"] in ("False", "false", "0", "")]
    if not to_sync:
        print(f"All jobs already synced.")
        return

    with ThreadPoolExecutor(max_workers=max_jobs) as ex:
        futures = [ex.submit(process_job, r) for r in to_sync]
        for _ in as_completed(futures):
            pass

    print(f"\nAll wandb runs successfully processed.\n")


if __name__ == "__main__":
    sync_wandb()
