#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

R = "\033[0m"
G = "\033[92m"
Y = "\033[93m"
P = "\033[95m"

BASE_DIR = Path("/leonardo_work/AIFAC_L01_028/najroldi/oellm_scaling")
LOG_DIR = BASE_DIR / "logs"
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


def get_status(out_file, err_file):
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
    return "running"


def process_log(base):
    out_file = Path(str(base) + ".out")
    err_file = Path(str(base) + ".err")
    if not out_file.exists() or not err_file.exists():
        return None

    m = LOG_RE.match(out_file.name)
    if not m:
        return None

    job_name, job_id, ts = m.group("job_name"), m.group("job_id"), m.group("ts")
    status = get_status(out_file, err_file)

    # print(f"{Y}{job_name}{R} [{job_id}] {P}{ts}{R}: {G}{status}{R}")

    return {
        "job_name": job_name,
        "job_id": job_id,
        "status": status,
        "wandb_synced": None,
    }


def manage_logs(max_jobs=8):
    print(f"\n=============================================")
    print(f"Scanning logs from {G}{LOG_DIR}{R}")
    print(f"Rebuilding {G}{CSV_PATH}{R} from scratch")
    print(f"Using {G}{max_jobs}{R} threads")
    print(f"\n=============================================")

    # Load existing csv
    df_old = pd.read_csv(CSV_PATH) if CSV_PATH.exists() else pd.DataFrame(
        columns=["job_name", "job_id", "status", "wandb_synced"]
    )

    # Build list of names
    base_paths = [LOG_DIR / f[:-4] for f in os.listdir(LOG_DIR) if f.endswith(".err")]
    if not base_paths:
        print(f"{Y}No logs found.{R}")
        return

    # Build df from logs
    with ThreadPoolExecutor(max_workers=max_jobs) as ex:
        results = [
            f.result() for f in as_completed([ex.submit(process_log, base) for base in base_paths]) if f.result()
        ]
        
    df_new = (
        pd.DataFrame(results, columns=["job_id", "job_name", "status", "wandb_synced"])
        .dropna(subset=["job_id", "job_name"])
        .drop_duplicates(subset=["job_id", "job_name"], keep="last")
    )
    if df_new.empty:
        print(f"{Y}No valid logs found.{R}")
        return
    
    # Fill None values in 'wandb_synced' with the ones form df_old
    df_new["job_id"] = df_new["job_id"].astype(str)
    df_old["job_id"] = df_old["job_id"].astype(str)

    df_new = df_new.set_index(["job_name", "job_id"])
    df_old = df_old.set_index(["job_name", "job_id"])
    
    # import pdb
    # pdb.set_trace()
    
    df_new = df_new.combine_first(df_old)
    df_new.fillna(False, inplace=True)

    # Save
    df_new = df_new.reset_index()
    df_new = df_new.sort_values(by=["job_id", "job_name"])    
    df_new.to_csv(CSV_PATH, index=False)
    print(f"CSV rebuilt successfully:{G} {CSV_PATH}{R}")
    

if __name__ == "__main__":
    manage_logs()



# set index: ["job_id", "job_name"]
# update new one with the old on 'wandb_synced'
