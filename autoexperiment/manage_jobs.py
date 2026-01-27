#!/usr/bin/env python3
import os
import pandas as pd
import re
import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

R = "\033[0m"
G = "\033[92m"
Y = "\033[93m"
P = "\033[95m"

BASE_DIR = Path("/leonardo_work/OELLM_prod2026/users/najroldi/exp")
LOG_DIR = BASE_DIR / "logs"
CKPT_DIR = BASE_DIR / "ckpts"
CSV_PATH = BASE_DIR / "jobs.csv"

FINISHED_PATTERNS = [r"after training is done", r"KeyError"]
TIMEOUT_PATTERNS = [r"DUE TO TIME LIMIT"]
ERROR_PATTERNS = [
    r"Traceback \(most recent call last\):",
    r"\b(Exception|Error)\b",
    r"CUDA (error|out of memory)",
    r"Segmentation fault",
    r"Exited with exit code 1",
    r"ChildFailedError",
    r"srun: error",
]

LOG_RE = re.compile(
    r"^(?P<job_name>.+)-(?P<job_id>\d+)-(?P<ts>\d{4}-\d{2}-\d{2}_[\d-]+)\.out$"
)


def has_pattern(text, patterns):
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def get_slurm_status(job_id):
    result = subprocess.run(
        ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
        capture_output=True,
        text=True,
    )
    status = result.stdout.strip()
    return status.lower()


def get_status(out_file, err_file, job_id):
    with open(out_file, errors="ignore") as f:
        out_text = f.read()
    with open(err_file, errors="ignore") as f:
        err_text = f.read()

    if has_pattern(out_text, FINISHED_PATTERNS):
        return "finished"
    if has_pattern(err_text, TIMEOUT_PATTERNS):
        return "timeout"
    if has_pattern(err_text, ERROR_PATTERNS):
        return "failed"
    slurm_status = get_slurm_status(job_id)
    if slurm_status in ["pending", "running", "completing", "cg"]:
        return slurm_status
    return "unknown"


def process_log(base):
    out_file = Path(str(base) + ".out")
    err_file = Path(str(base) + ".err")
    if not out_file.exists() or not err_file.exists():
        return None

    m = LOG_RE.match(out_file.name)
    if not m:
        return None

    job_name, job_id = m.group("job_name"), m.group("job_id")
    job_id = int(job_id)
    status = get_status(out_file, err_file, job_id)

    # Look for a wandb run in the ckpt directory.
    # We can match the slurm jobid by checking wandb-metadata.json!
    wandb_run = None
    wandb_dir = CKPT_DIR / job_name / "wandb" / "wandb"
    if wandb_dir.is_dir():
        for run in wandb_dir.iterdir():
            if not run.is_dir() or not run.name.startswith("offline-run-"):
                continue

            meta = run / "files" / "wandb-metadata.json"
            if not meta.is_file():
                continue

            try:
                with meta.open() as mf:
                    data = json.load(mf)
                if int(data['slurm']['job_id']) == job_id:
                    wandb_run = run.name
                    break
            except Exception as e:
                print(f"Error reading {meta}: {e}")

    return {
        "job_name": job_name,
        "job_id": job_id,
        "status": status,
        "wandb_run": wandb_run,
        "wandb_synced": False,
    }


def sync_row(row):
    if row.wandb_synced or not row.wandb_run:
        return row

    run_dir = CKPT_DIR / row.job_name / "wandb" / "wandb" / row.wandb_run
    run_id = row.wandb_run.split("-")[-1]
    wandb_file = run_dir / f"run-{run_id}.wandb"

    is_running = get_slurm_status(row.job_id) != ""

    try:
        subprocess.run(
            ["wandb", "sync", str(wandb_file)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        row.wandb_synced = not is_running
        print(f"{row.job_name:<40} | {run_id:<10} | synced | running={is_running}")
    except Exception as e:
        print(f"Error syncing wandb for job {row.job_name}-{row.job_id}: {e}")

    return row




def manage_logs(max_jobs=8):
    print(f"\n=============================================")
    print(f"Scanning logs from {G}{LOG_DIR}{R}.")
    print(f"Scanning ckpts from {G}{CKPT_DIR}{R}.")
    print(f"Updating {G}{CSV_PATH}{R}.")
    print(f"Using {G}{max_jobs}{R} threads")
    print(f"=============================================\n")

    # Load existing csv
    df = (
        pd.read_csv(CSV_PATH)
        if CSV_PATH.exists()
        else pd.DataFrame(columns=['job_name','job_id','status','wandb_run','wandb_synced'])
    )

    # Only update CSV for running jobs.
    skip = {
        row.job_id
        for _, row in df.iterrows()
        if row.status in ['finished', 'timeout', 'failed']
    }

    # Scan log directory, build list of paths to process.
    base_paths = []
    for f in os.listdir(LOG_DIR):
        if f.endswith(".err"):
            base = (LOG_DIR / f[:-4])
            match = LOG_RE.match(Path(f[:-4] + ".out").name)
            if match:
                job_id = int(match.group("job_id"))
                if job_id not in skip:
                    base_paths.append(base)

    # If no logs found, exit.
    if not base_paths:
        print(f"{Y}No logs found.{R}")
        return
    print(f"Found {len(base_paths)} logs to process.")

    # Process logs in parallel and build df.
    with ThreadPoolExecutor(max_workers=max_jobs) as ex:
        results = [
            f.result() for f in as_completed([ex.submit(process_log, base) for base in base_paths]) if f.result()
        ]
    df_new = pd.DataFrame(results, columns=['job_name', 'job_id', 'status', 'wandb_run','wandb_synced'])

    if df_new.duplicated(subset=['job_name','job_id'], keep=False).any():
        print(f"Warning: Found duplicate log entries.")
    if df_new.empty:
        print(f"No new logs found.")
        return

    # Update existing df with new info.
    if df.empty:
        df = df_new
    else:
        df_new.set_index(['job_name', 'job_id'], inplace=True)
        df.set_index(['job_name', 'job_id'], inplace=True)
        df.update(df_new)
        missing = df_new[~df_new.index.isin(df.index)]
        df = pd.concat([df, missing])
        df.reset_index(inplace=True)

    # Save
    df = df.sort_values(by=['job_name', 'job_id']) 
    df.to_csv(CSV_PATH, index=False)
    print(f"CSV rebuilt successfully:{G} {CSV_PATH}{R}")
    
    # Find runs to sync.
    rows_to_sync = df[~df.wandb_synced & df.wandb_run.notnull()]
    print(f"Found {len(rows_to_sync)} offline wandb runs to sync.")

    # Threaded wandb sync
    updated_rows = []
    try:
        with ThreadPoolExecutor(max_workers=max_jobs) as ex:
            futs = [ex.submit(sync_row, row.copy()) for _, row in rows_to_sync.iterrows()]
            for f in as_completed(futs):
                updated_rows.append(f.result())

    except KeyboardInterrupt:
        print(f"\nInterrupted. Saving partial sync state...")

    finally:
        if updated_rows:
            df_updates = pd.DataFrame(updated_rows)
            df_updates.set_index(["job_name","job_id"], inplace=True)
            df.set_index(["job_name","job_id"], inplace=True)
            df.update(df_updates)
            df.reset_index(inplace=True)

            df = df.sort_values(by=["job_name","job_id"])
            df.to_csv(CSV_PATH, index=False)
            print(f"Partial wandb sync saved to CSV.")
        else:
            print("No updates to save.")


if __name__ == "__main__":
    manage_logs()
