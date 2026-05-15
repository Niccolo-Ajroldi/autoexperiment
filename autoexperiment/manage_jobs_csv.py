#!/usr/bin/env python3
import sys
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

BASE_DIR = Path("/scratch/project_462000963/users/niccolo/oellm_scaling")
LOG_DIR = BASE_DIR / "logs"
CKPT_DIR = BASE_DIR / "ckpts"
CSV_PATH = BASE_DIR / "jobs.csv"

JOB_FILTER = "_xm0"
FINISHED_PATTERN = r"after training is done"
TIMEOUT_PATTERNS = [
    r"DUE TO TIME LIMIT"
]
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

COLUMNS = [
    'job_name', 'job_id', 'timestamp', 'status', 
    'wandb_run', 'wandb_synced',
    'err_path', 'out_path',
]

SQUEUE = {}


def get_squeue_map():
    result = subprocess.run(
        ["squeue", "-h", "-o", "%i %T"],
        capture_output=True,
        text=True,
    )
    out = {}
    for line in result.stdout.splitlines():
        jid, state = line.split()
        jid = re.split(r'[_+]', jid)[0]   # handle array and pack jobs
        out[int(jid)] = state.lower()
    return out


def get_slurm_status(job_id):
    return SQUEUE.get(job_id, "")


def has_pattern(text, patterns):
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def is_finished(text: str) -> bool:
    """Job finished when: (1) termination string is there, (2) saved ckpt."""
    m = re.search(FINISHED_PATTERN, text, re.IGNORECASE)
    if not m:
        return False
    return bool(re.search(r'successfully saved checkpoint', text[m.end():], re.IGNORECASE))


def get_status(out_file, err_file, job_id):
    slurm_status = get_slurm_status(job_id)
    if slurm_status in ["pending", "running", "completing", "cg"]:
        return slurm_status

    with open(out_file, errors="ignore") as f:
        out_text = f.read()
    with open(err_file, errors="ignore") as f:
        err_text = f.read()

    if is_finished(out_text):
        return "finished ✅"
    if has_pattern(err_text, TIMEOUT_PATTERNS):
        return "timeout ⏳"
    if has_pattern(err_text, ERROR_PATTERNS):
        return "failed ❌"

    return "unknown"


def process_log(base):
    out_file = Path(str(base) + ".out")
    err_file = Path(str(base) + ".err")
    if not out_file.exists() or not err_file.exists():
        return None

    m = LOG_RE.match(out_file.name)
    if not m:
        return None

    job_name = m.group("job_name")
    job_id = m.group("job_id")
    timestamp = m.group("ts")
    
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
        "timestamp": timestamp,
        "status": status,
        "wandb_run": wandb_run,
        "wandb_synced": False,
        "err_path": str(err_file),
        "out_path": str(out_file),
    }


# def sync_row(row):
#     if row.wandb_synced or not row.wandb_run:
#         return row

#     run_dir = CKPT_DIR / row.job_name / "wandb" / "wandb" / row.wandb_run
#     run_id = row.wandb_run.split("-")[-1]
#     wandb_file = run_dir / f"run-{run_id}.wandb"

#     is_running = get_slurm_status(row.job_id) != ""

#     try:
#         subprocess.run(
#             ["wandb", "sync", str(wandb_file)],
#             stdout=subprocess.DEVNULL,
#             stderr=subprocess.DEVNULL,
#         )
#         row.wandb_synced = not is_running
#         print(f"{row.job_name:<40} | {run_id:<10} | synced | running={is_running}")
#     except Exception as e:
#         print(f"Error syncing wandb for job {row.job_name}-{row.job_id}: {e}")

#     return row


def manage_logs(max_jobs=8, build_from_scratch=False):
    print(f"\n=============================================")
    print(f"Scanning logs from {G}{LOG_DIR}{R}.")
    print(f"Scanning ckpts from {G}{CKPT_DIR}{R}.")
    print(f"Writing to {G}{CSV_PATH}{R}.")
    print(f"Using {G}{max_jobs}{R} threads")
    print(f"=============================================\n")

    global SQUEUE
    SQUEUE = get_squeue_map()

    # Load existing csv
    if build_from_scratch or not CSV_PATH.exists():
        df = pd.DataFrame(columns=COLUMNS)
    else:
        df = pd.read_csv(CSV_PATH)

    # Only update CSV for running jobs.
    skip = set() if build_from_scratch else {
        row.job_id
        for _, row in df.iterrows()
        if row.status in ['finished ✅', 'timeout ⏳', 'failed ❌']
    }

    # Scan log directory, build list of paths to process.
    base_paths = []
    for f in os.listdir(LOG_DIR):
        if f.endswith(".err"):
            base = LOG_DIR / f[:-4]
            match = LOG_RE.match(Path(f[:-4] + ".out").name)
            if not match:
                continue

            job_name = match.group("job_name")
            job_id = int(match.group("job_id"))

            if JOB_FILTER and JOB_FILTER not in job_name:
                continue
            if job_id in skip:
                continue

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
    df_new = pd.DataFrame(results, columns=COLUMNS)

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
    df["_N"] = pd.to_numeric(df.job_name.str.extract(r"dense_([0-9.]+)[MB]", expand=False), errors='coerce')
    df["_bsz"] = pd.to_numeric(df.job_name.str.extract(r"_gbsz(\d+)", expand=False), errors='coerce').fillna(0).astype(int)
    
    # df["_N"] = df.job_name.str.extract(r"dense_([0-9.]+)[MB]").astype(float)
    df["_lr"] = df.job_name.str.extract(r"_lr([0-9.]+)").astype(float)
    # df["_bsz"] = df.job_name.str.extract(r"_gbsz(\d+)").astype(int)
    df["_decay"] = df.job_name.str.extract(r"_([0-9.]+)B$").astype(float).fillna(0)
    df.sort_values(
        by=["_N", "_bsz", "_lr", "_decay", "job_id"],
        ascending=[True, True, True, True, True],
        inplace=True
    )
    # df.drop(columns=["", "_decay"], inplace=True)
    df = df[['_N', '_bsz', '_lr', '_decay'] + COLUMNS]

    # add pending jobs with no logs
    existing_ids = set(df.job_id.astype(int))
    rows = []
    for job_id, state in SQUEUE.items():
        if state != "pending":
            continue
        if job_id in existing_ids:
            continue
        rows.append({
            "job_name": "<unknown>",
            "job_id": job_id,
            "timestamp": None,
            "status": "pending",
            "wandb_run": None,
            "wandb_synced": False,
            "err_path": None,
            "out_path": None,
            "_N": None,
            "_bsz": None,
            "_lr": None,
            "_decay": None,
        })
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

    df.to_csv(CSV_PATH, index=False)
    print(f"CSV rebuilt successfully:{G} {CSV_PATH}{R}")
    
    # Find runs to sync.
    rows_to_sync = df[~df.wandb_synced & df.wandb_run.notnull()]
    print(f"Found {len(rows_to_sync)} offline wandb runs to sync.")

    # # Threaded wandb sync
    # updated_rows = []
    # try:
    #     with ThreadPoolExecutor(max_workers=max_jobs) as ex:
    #         futs = [ex.submit(sync_row, row.copy()) for _, row in rows_to_sync.iterrows()]
    #         for f in as_completed(futs):
    #             updated_rows.append(f.result())

    # except KeyboardInterrupt:
    #     print(f"\nInterrupted. Saving partial sync state...")

    # finally:
    #     if updated_rows:
    #         df_updates = pd.DataFrame(updated_rows)
    #         df_updates.set_index(["job_name","job_id"], inplace=True)
    #         df.set_index(["job_name","job_id"], inplace=True)
    #         df.update(df_updates)
    #         df.reset_index(inplace=True)

    #         df["_lr"] = df.job_name.str.extract(r"_lr([0-9.]+)").astype(float)
    #         df["_bsz"] = df.job_name.str.extract(r"_gbsz(\d+)").astype(int)
    #         df["_decay"] = (
    #             df.job_name.str.extract(r"decay(\d+)", expand=False)
    #             .astype(float)
    #             .fillna(0)
    #         )
    #         df = df.sort_values(
    #             by=["_bsz", "_lr", "_decay", "job_id"],
    #             ascending=[True, True, True, True, True],
    #         )
    #         df.drop(columns=["_lr", "_bsz", "_decay"], inplace=True)

    #         df.to_csv(CSV_PATH, index=False)
    #         print(f"Partial wandb sync saved to CSV.")
    #     else:
    #         print("No updates to save.")


if __name__ == "__main__":
    manage_logs(build_from_scratch="--build_from_scratch" in sys.argv)
