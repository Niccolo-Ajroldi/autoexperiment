#!/usr/bin/env python3
"""
Return 1 if any log for JOB_NAME in LOG_DIR contains:
- FINISHED_PATTERN
- "could not load the checkpoint"

Else 0.
"""

import sys, os, re


FINISHED_PATTERN = r'validation loss at iteration'
CKPT_ERR_PATTERN = r'could not load the checkpoint'


def any_log_has_pattern(log_dir, job_name):
    ok_re = re.compile(FINISHED_PATTERN, re.IGNORECASE)
    ckpt_re = re.compile(CKPT_ERR_PATTERN, re.IGNORECASE)
    out_regex = re.compile(rf'^{re.escape(job_name)}-(\d+)-(.+)\.out$')

    for f in os.listdir(log_dir):
        if not f.endswith(".out") or not out_regex.fullmatch(f):
            continue

        path = os.path.join(log_dir, f)
        try:
            with open(path, errors="ignore") as fh:
                content = fh.read()

                if ok_re.search(content) or ckpt_re.search(content):
                    return True

        except Exception:
            pass

    return False


def main():
    if len(sys.argv) < 3:
        print(0)
        return

    log_dir, job_name = sys.argv[1], sys.argv[2]

    if not os.path.isdir(log_dir):
        print(0)
        return

    print(1 if any_log_has_pattern(log_dir, job_name) else 0)


if __name__ == "__main__":
    main()
