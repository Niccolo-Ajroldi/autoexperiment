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
        raise ValueError("Provide CKPTS_DIR, LOGS_DIR, OUT_FILENAMES_DIR.")
    return args


args = parse_args()

CKPTS_DIR = args.ckpts_dir
LOGS_DIR = args.logs_dir
OUT_FILENAMES_DIR = args.out_dir
CSV_PATH = OUT_FILENAMES_DIR / "eval_status.csv"

MODELS = ["50M", "130M", "300M", "600M", "1B"]

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


def parse_log_name(path):
    base = path.stem
    parts = base.split("_")

    # step is the token before "stable"
    idx = parts.index("stable")
    step = int(parts[idx - 1])

    ckpt = "_".join(parts[:idx - 1])
    return ckpt, step


# -------- BUILD CSV (per ckpt, step) --------
rows = []

for ckpt in sorted(CKPTS_DIR.iterdir()):
    if not ckpt.is_dir():
        continue

    name = ckpt.name

    # find all steps from ckpt dir
    steps = []
    for d in ckpt.iterdir():
        if d.is_dir() and d.name.startswith("iter_"):
            try:
                steps.append(int(d.name.split("_")[1]))
            except:
                continue

    # find logs for this ckpt
    outs = list(LOGS_DIR.glob(f"{name}_*.out"))

    log_map = {}
    for out in outs:
        ckpt_name, step = parse_log_name(out)
        r = get_eval_results(out)
        if r is not None:
            log_map[step] = r

    # build rows per step
    for step in steps:
        r = log_map.get(step)

        rows.append({
            "checkpoint": name,
            "step": step,
            "evaluated": r is not None,
            **(r or {})
        })


df = pd.DataFrame(rows)
OUT_FILENAMES_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(CSV_PATH, index=False)


# -------- UNEVALUATED --------
pending = {m: [] for m in MODELS}

uneval = df[df["evaluated"] == False]

for _, row in uneval.iterrows():
    name = row["checkpoint"]
    step = int(row["step"])

    for m in MODELS:
        if f"dense_{m}" in name:
            pending[m].append(f"{name}_{step:07d}")
            break


# -------- WRITE --------
for m, pairs in pending.items():
    path = OUT_FILENAMES_DIR / f"{m}.txt"
    with open(path, "w") as f:
        for p in pairs:
            f.write(p + "\n")

print("CSV:", CSV_PATH)
print("TXT files:", OUT_FILENAMES_DIR)