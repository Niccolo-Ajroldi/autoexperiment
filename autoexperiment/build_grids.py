#!/usr/bin/env python3

"""
Reads a jobs log CSV, keeps the latest entry per (N, batch size, lr, decay),
and writes grid CSVs of job status indexed by batch size and learning rate,
organized by model size (N) and decay.

grid/
├── 50M/
│   ├── 6B.csv
│   └── 12B.csv
├── 130M/
│   └── 12B.csv
└── 300M/
    ├── 6B.csv
    └── 12B.csv

"""

import pandas as pd
from pathlib import Path


CSV_PATH = Path("/scratch/project_462000963/users/niccolo/oellm_scaling/jobs.csv")
OUT_DIR = Path("/scratch/project_462000963/users/niccolo/oellm_scaling/grids")
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["_N", "_bsz", "_lr", "_decay"])

# Types
df["_N"] = df["_N"].astype(float)
df["_bsz"] = df["_bsz"].astype(int)
df["_lr"] = df["_lr"].astype(float)
df["_decay"] = df["_decay"].astype(float)

# Last log only
df = df.sort_values("timestamp")
df = df.groupby(["_N", "_bsz", "_lr", "_decay"], as_index=False).last()

def fmt_N(N):
    return f"{int(N)}M" if N < 1000 else f"{int(N/1000)}B"

def fmt_decay(d):
    return f"{int(d)}B"

groups_N = list(df.groupby("_N"))
total_N = len(groups_N)

for i, (N, dfnN) in enumerate(groups_N, 1):
    print(f"[{i}/{total_N}] N={fmt_N(N)}")
    n_dir = OUT_DIR / fmt_N(N)
    n_dir.mkdir(exist_ok=True)

    for decay, dfn in dfnN.groupby("_decay"):
        grid = (
            dfn
            .pivot_table(
                index="_bsz",
                columns="_lr",
                values="status",
                aggfunc="last"
            )
            .sort_index()
            .sort_index(axis=1)
        )

        out = n_dir / f"{fmt_decay(decay)}.csv"
        grid.to_csv(out)
