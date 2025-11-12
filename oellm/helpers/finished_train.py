#!/usr/bin/env python3
"""
finished_train.py — Return 1 if training is finished, else 0.

Usage:
    finished_train.py LOG_DIR JOB_NAME

Finds latest log starting with job_name, looks for termination string.
"""

import sys, os, re

FINISHED_PATTERNS = [
    r'after training is done',
    r'KeyError', # mark as finished even if crash occurs after final save TODO: fix
]
TIMEOUT_PATTERNS = [
    r'DUE TO TIME LIMIT',
    r'(Bus error: nonexistent physical address)',
    r'AssertionError: OptimizerParamScheduler: class input value', # patch TODO: remove
    r'Communication connection failure', # patch, TODO: remove
]
ERROR_PATTERNS = [
    r'Traceback \(most recent call last\):',
    r'\b(Exception|Error)\b',
    r'CUDA (error|out of memory)',
    r'Segmentation fault',
    r'Exited with exit code 1',
    r'ChildFailedError',
    r'srun: error',
]


def has_error(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in ERROR_PATTERNS)


def has_timeout(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in TIMEOUT_PATTERNS)


def has_finished(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in FINISHED_PATTERNS)


def find_latest_logs(log_dir, job_name):
    """Return tuple (out_file, err_file) for the latest job."""
    pattern = re.compile(rf'{re.escape(job_name)}-(\d+)-([\d_-]+)\.out')
    candidates = []
    for f in os.listdir(log_dir):
        m = pattern.fullmatch(f)
        if m:
            ts = m.group(2)
            base = os.path.join(log_dir, f[:-4])  # drop .out
            candidates.append((ts, base))
    if not candidates:
        return None, None
    latest = max(candidates, key=lambda x: x[0])[1]
    return latest + ".out", latest + ".err"


def main():
    if len(sys.argv) < 3:
        print("Usage: finished_train.py LOG_DIR JOB_NAME", file=sys.stderr)
        print(0)
        return

    log_dir, job_name = sys.argv[1], sys.argv[2]
    if not os.path.isdir(log_dir):
        print(0)
        return

    out_file, err_file = find_latest_logs(log_dir, job_name)
    if not out_file or not os.path.exists(out_file):
        print(0)
        return
    if not err_file or not os.path.exists(err_file):
        print(0)
        return

    with open(out_file, errors="ignore") as f:
        out_text = f.read()

    with open(err_file, errors="ignore") as f:
        err_text = f.read()

    if has_finished(out_text):
        print(1)  # finished -> no reschedule
    elif has_timeout(err_text):
        print(0)  # timeout -> reschedule
    elif has_error(err_text):
        print(1)  # died -> no reschedule
    else:
        print(0)  # reschedule

if __name__ == "__main__":
    main()
