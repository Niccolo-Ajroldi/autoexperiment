#!/usr/bin/env python3
"""
Scan logs directory, build a csv with job infos.
"""

import os
import pandas as pd
import re
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

SQUEUE = {}


def get_squeue_map():
    result = subprocess.run(
        ["squeue", "--me", "-h", "-o", "%i %T %j"],
        capture_output=True,
        text=True,
    )
    out = {}
    for line in result.stdout.splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) < 3:
            continue
        jid, state, name = parts
        jid = re.split(r'[_+]', jid)[0]
        out[int(jid)] = (state.lower(), name)
    return out


FIELD_SPECS = {
    # "N": (r"dense_([0-9.]+)([MB])", lambda m: float(m[0]) * {"M": 1e-3, "B": 1}[m[1]]),
    "N": (
        r"dense_([0-9]+(?:_[0-9]+)?)([MB])",
        lambda m: float(m[0].replace("_", ".")) * {"M": 1e-3, "B": 1}[m[1]],
    ),
    "lr": (r"_lr([0-9.]+)", lambda m: float(m[0])),
    "bsz": (r"_gbsz(\d+)", lambda m: int(m[0])),
    "decay": (r"_([0-9.]+)B$", lambda m: float(m[0])),
}

def extract_fields(name):
    out = {}
    for k, (pattern, fn) in FIELD_SPECS.items():
        m = re.search(pattern, name)
        if not m:
            out[k] = None
            continue
        try:
            out[k] = fn(m.groups())
        except:
            out[k] = None
    return out


def get_slurm_status(job_id):
    return SQUEUE.get(job_id, "")


def has_pattern(text, patterns):
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def is_finished(text: str) -> bool:
    """Job finished when: (1) termination string is there, (2) saved ckpt."""
    return bool(re.search(FINISHED_PATTERN, text, re.IGNORECASE))
    # m = re.search(FINISHED_PATTERN, text, re.IGNORECASE)
    # if not m:
    #     return False
    # return bool(re.search(r'successfully saved checkpoint', text[m.end():], re.IGNORECASE))


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

    ckpt_path = CKPT_DIR / job_name
    ckpt_path = str(ckpt_path) if ckpt_path.exists() else None

    job_id = int(job_id)
    status = get_status(out_file, err_file, job_id)

    row = {
        "job_name": job_name,
        "job_id": job_id,
        "timestamp": timestamp,
        "status": status,
        "err_path": str(err_file),
        "out_path": str(out_file),
        "ckpt_path": ckpt_path
    }
    row.update(extract_fields(job_name))

    return row


def main(max_jobs=8):

    print(f"Using {G}{max_jobs}{R} threads")
    print(f"Scanning logs from {G}{LOG_DIR}{R}.")
    print(f"Scanning ckpts from {G}{CKPT_DIR}{R}.")
    print(f"Writing csv to {G}{CSV_PATH}{R}.")

    global SQUEUE
    SQUEUE = get_squeue_map()

    # Scan log directory, build list of paths to process.
    base_paths = [
        LOG_DIR / f[:-4]
        for f in os.listdir(LOG_DIR)
        if f.endswith(".out") and (not JOB_FILTER or JOB_FILTER in f)
    ]

    # If no logs found, exit.
    if not base_paths:
        print(f"{Y}No logs found.{R}"); return
    print(f"Found {len(base_paths)} logs to process.")

    # Process logs in parallel and build df.
    with ThreadPoolExecutor(max_workers=max_jobs) as ex:
        results = [
            f.result() for f in as_completed([ex.submit(process_log, base) for base in base_paths]) if f.result()
        ]
    df = pd.DataFrame(results)

    if df.duplicated(subset=['job_name', 'job_id'], keep=False).any():
        print(f"Found duplicate log entries.")
    if df.empty:
        raise ValueError(f"No logs found.")

    # add pending jobs with no logs
    existing_ids = set(df.job_id.astype(int))
    rows = []
    for job_id, (state, name) in SQUEUE.items():
        if state != "pending":
            continue
        if JOB_FILTER and JOB_FILTER not in name:
            continue
        if job_id in existing_ids:
            continue
        rows.append({
            "job_name": name,
            "job_id": job_id,
            "timestamp": None,
            "status": "pending",
            "err_path": None,
            "out_path": None,
            "N": None,
            "bsz": None,
            "lr": None,
            "decay": None,
        })
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

    # Save
    df.sort_values(
        by=["N", "bsz", "lr", "decay", "job_id"],
        ascending=[True, True, True, True, True],
        inplace=True
    )
    cols = ["N", "bsz", "lr", "decay", "job_id", "timestamp", "status"] + [
        c for c in df.columns if c not in ["N", "bsz", "lr", "decay", "job_id", "timestamp", "status"]
    ]
    df = df[cols]
    df.to_csv(CSV_PATH, index=False)
    print(f"CSV rebuilt successfully:{G} {CSV_PATH}{R}")


if __name__ == "__main__":
    main()
