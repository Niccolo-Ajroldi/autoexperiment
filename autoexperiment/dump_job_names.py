#!/usr/bin/env python3
"""
Dump all autoexperiment job names for a YAML config.
"""

import sys
from omegaconf import OmegaConf
from clize import run as clize_run
from clize.parameters import multi

from autoexperiment.template import generate_job_defs


def dump_jobs(config, out, *, verbose=1, fix:('f', multi())):
    cfg = OmegaConf.load(config)

    if fix:
        for param in fix:
            key, value = param.split("=")
            cfg[key] = value

    jobdefs = generate_job_defs(cfg, verbose=verbose)

    with open(out, "w") as f:
        for jd in jobdefs:
            f.write(jd.name + "\n")

    print(f"Wrote {len(jobdefs)} job names to {out}")


def main():
    return clize_run(dump_jobs)


if __name__ == "__main__":
    sys.exit(main())
