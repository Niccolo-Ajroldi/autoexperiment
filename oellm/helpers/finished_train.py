#!/usr/bin/env python3
"""
Finds latest log starting with JOB_NAME in LOG_DIR.
Checks err and out files to asses if job is finished, timed out, or errored.
We use it to determine whether to reschedule a training job.

Usage:
    finished_train.py LOG_DIR JOB_NAME

Returns:
    1 — training finished or irrecoverably failed (do not reschedule)
    0 — training not finished or timed out (reschedule)
"""


import sys, os, re


FINISHED_PATTERNS = [
    r'after training is done',
]
TIMEOUT_PATTERNS = [
    r'DUE TO TIME LIMIT',
    r'--- Logging error -—',
    re.escape("FileNotFoundError: [Errno 2] No such file or directory: '/users/niccolo/.aiter/jit/build/lock_module_aiter_enum'"),
    re.escape("ModuleNotFoundError: No module named 'module_rope_general_fwd'"),
    re.escape('slurmstepd: error: execve(): bash: No such file or directory'),
    # 'Disk quota exceeded'
]
ERROR_PATTERNS = [
    r'Traceback \(most recent call last\):',
    r'\b(Exception|Error)\b',
    r'CUDA (error|out of memory)',
    r'Segmentation fault',
    r'Exited with exit code 1',
    r'ChildFailedError',
    r'srun: error',
    r'Communication connection failure',
    r'AssertionError',
    r'(Bus error: nonexistent physical address)',
]

SUCCESS_LOG_PATTERN = re.compile(
    r'iteration\s+\d+\s*/\s*\d+',
    re.IGNORECASE
)

MIN_SUCCESS_LOGS = 3

# Out filename format:
# {JOB_NAME}-{SLURM_ID}-{YYYY-MM-DD_HH-MM-SS}.out/.err
OUT_REGEX = r'^{job}-(\d+)-(.+)\.out$'


def has_error(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in ERROR_PATTERNS)


def has_timeout(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in TIMEOUT_PATTERNS)


# def has_finished(text: str) -> bool:
#     return any(re.search(p, text, re.IGNORECASE) for p in FINISHED_PATTERNS)

def has_finished(text: str) -> bool:
    """Job finished when: (1) termination string is there, (2) saved ckpt."""
    m = re.search(r'after training is done', text, re.IGNORECASE)
    if not m:
        return False
    return bool(re.search(r'successfully saved checkpoint', text[m.end():], re.IGNORECASE))


def has_enough_success_logs(text: str, n=MIN_SUCCESS_LOGS) -> bool:
    return len(SUCCESS_LOG_PATTERN.findall(text)) >= n


def find_latest_logs(log_dir, job_name):
    """Return tuple (out_file, err_file) for the latest job."""
    pattern = re.compile(OUT_REGEX.format(job=re.escape(job_name)))
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
        if has_enough_success_logs(out_text):
            print(0)  # reschedule only if job was actually training
            return
        else:
            print(1)  # no reschedule
            return
        
    else:
        print(0)  # reschedule


if __name__ == "__main__":
    main()
