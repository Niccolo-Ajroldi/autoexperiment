#!/usr/bin/env python3
import pandas as pd
import sys
import re
import math

SEQ_LEN = 2048  # must match training

if len(sys.argv) != 3:
    print("Usage: summarize_jobs.py jobs.csv summary.csv")
    sys.exit(1)

jobs_csv, out_csv = sys.argv[1], sys.argv[2]

df = pd.read_csv(jobs_csv)

df["BSZ"] = df.job_name.str.extract(r"_gbsz(\d+)").astype(int)
df["LR"]  = df.job_name.str.extract(r"_lr([0-9.e-]+)").astype(float)


def parse_phase(name, bsz):
    if "stable" in name:
        return "stable"

    m = re.search(r"_decay(\d+)", name)
    if not m:
        raise ValueError(f"Cannot parse phase from job_name: {name}")

    iters = int(m.group(1))
    tokens = iters * SEQ_LEN * bsz
    billions = int(round(tokens / 1e9))

    return f"decay_{billions}B"

df["PHASE"] = df.apply(lambda r: parse_phase(r.job_name, r.BSZ), axis=1)

summary = (
    df.groupby(["BSZ", "LR", "PHASE"])
      .status
      .apply(lambda s: "finished ✅" if (s == "finished ✅").any() else "not_finished")
      .reset_index(name="STATUS")
)


# stable first, then increasing billions
summary["_p"] = summary.PHASE.apply(
    lambda p: 0 if p == "stable" else int(p.split("_")[1][:-1])
)

summary = summary.sort_values(["BSZ", "LR", "_p"])
summary = summary[["BSZ", "LR", "PHASE", "STATUS"]]

summary.to_csv(out_csv, index=False)
