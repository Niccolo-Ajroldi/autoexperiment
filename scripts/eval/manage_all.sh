#!/bin/bash

set -e

## Re-compile csv with logs
module load cray-python
python /scratch/project_462000963/users/niccolo/autoexperiment/autoexperiment/parse_logs.py

## Latest checkpoint eval
# Copy
bash /users/niccolo/code/lumio/copy_last_ckpt/copy_last_ckpt.sh
# Build list of checkpoints to eval
bash /scratch/project_462000963/users/niccolo/autoexperiment/scripts/eval/latest/eval_status.sh
# Launch Evals
autoexperiment build-and-run 

