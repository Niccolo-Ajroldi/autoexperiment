#!/bin/bash

set -e
module load cray-python

SCRATCH=/scratch/project_462000963/users/niccolo

OUT_FILENAMES_DIR=$SCRATCH/autoexperiment/scripts/eval/stable/filenames
CKPTS_DIR=$SCRATCH/more_ckpts/stable_at_budgets/ckpts
LOGS_DIR=$SCRATCH/more_ckpts/stable_at_budgets/logs

python $SCRATCH/autoexperiment/scripts/eval/stable/eval_status.py \
    --ckpts_dir=$CKPTS_DIR \
    --logs_dir=$LOGS_DIR \
    --out_dir=$OUT_FILENAMES_DIR
