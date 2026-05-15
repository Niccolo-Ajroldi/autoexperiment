"""
Loop over jobs in a checkpoint directory.
For each job,:
- read its logs, look for evaluation results;
- store results in a csv;
- write files for each model size with the unevaluated jobs.
"""

import os
import re
import argparse
import pandas as pd
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts_dir", type=Path, default=os.getenv("CKPTS_DIR"))
    p.add_argument("--logs_dir", type=Path, default=os.getenv("LOGS_DIR"))
    p.add_argument("--out_dir", type=Path, default=os.getenv("OUT_FILENAMES_DIR"))
    args = p.parse_args()
    if not all([args.ckpts_dir, args.logs_dir, args.out_dir]):
        raise ValueError("Provide CKPTS_DIR, LOGS_DIR, OUT_FILENAMES_DIR (args or env).")
    return args

args = parse_args()

CKPTS_DIR = args.ckpts_dir
LOGS_DIR = args.logs_dir
OUT_FILENAMES_DIR = args.out_dir
CSV_PATH = OUT_FILENAMES_DIR / "eval_status.csv"

MODELS = ["50M", "130M", "300M", "600M", "1B", "1_7B"]

pattern = re.compile(
    r"\[nid\d+:\d+\]: validation loss at iteration (?P<iteration>\d+) on validation set \|\s*"
    r"lm loss value: (?P<lm_loss_value>[\d.E+-]+) \|\s*"
    r"lm loss PPL: (?P<lm_loss_ppl>[\d.E+-]+) \|\s*"
    r"lm loss num of tokens value: (?P<num_tokens>[\d.E+-]+) \|\s*"
    r"lm loss per token sum value: (?P<per_token_sum>[\d.E+-]+) \|\s*"
    r"lm loss squared per token sum value: (?P<per_token_sq_sum>[\d.E+-]+) \|\s*"
    r"lm loss per token mean value: (?P<per_token_mean>[\d.E+-]+) \|\s*"
    r"lm loss per token std value: (?P<per_token_std>[\d.E+-]+) \|\s*"
    r"lm loss num of samples value: (?P<num_samples>[\d.E+-]+) \|\s*"
    r"lm loss per sample sum value: (?P<per_sample_sum>[\d.E+-]+) \|\s*"
    r"lm loss squared per sample sum value: (?P<per_sample_sq_sum>[\d.E+-]+) \|\s*"
    r"lm loss per sample mean value: (?P<per_sample_mean>[\d.E+-]+) \|\s*"
    r"lm loss per sample std value: (?P<per_sample_std>[\d.E+-]+)\s*\|"
)


def get_eval_results(out):
    """Extract evaluation metrics from logs."""
    with open(out) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                d = m.groupdict()
                d = {k: float(v) for k, v in d.items() if k != "iteration"}
                d["iteration"] = int(m.group("iteration"))
                return d
    return None


# Build a csv of names and evaluated status
rows = []
for ckpt in sorted(CKPTS_DIR.iterdir()):
    if not ckpt.is_dir():
        continue

    name = ckpt.name.replace("_latest_eval", "")
    name = ckpt.name.replace("_pre_decay_eval", "")
    outs = list(LOGS_DIR.glob(f"{name}*.out"))

    # if there are multiple logs, check that they all match
    first = None
    evaluated = False
    for out in outs:
        r = get_eval_results(out)
        if r is None:
            continue
        if first is None:
            first = r
        elif r != first:
            raise ValueError(f"Mismatch in logs for {name}")
        evaluated = True
    eval_dict = first or {}

    # Append row
    rows.append({
        "checkpoint": name,
        "evaluated": evaluated,
        **eval_dict
    })

# Make df
df = pd.DataFrame(rows)
df.to_csv(CSV_PATH, index=False)

# Extract checkpoints not evaluated, write them in separate txt files
pending = {m: [] for m in MODELS}
uneval = df[df["evaluated"] == False]
for name in uneval["checkpoint"]:
    for m in MODELS:
        if f"dense_{m}" in name:
            pending[m].append(name)
            break

# Write
OUT_FILENAMES_DIR.mkdir(parents=True, exist_ok=True)
for m, names in pending.items():
    path = OUT_FILENAMES_DIR / f"{m}.txt"
    with open(path, "w") as f:
        for n in names:
            f.write(n + "\n")

print("CSV:", CSV_PATH)
print("TXT files:", OUT_FILENAMES_DIR)
