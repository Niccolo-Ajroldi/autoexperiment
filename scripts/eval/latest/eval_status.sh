#!/bin/bash

set -e
module load cray-python

SCRATCH=/scratch/project_462000963/users/niccolo

OUT_FILENAMES_DIR=$SCRATCH/autoexperiment/scripts/eval/latest/filenames
CKPTS_DIR=$SCRATCH/latest_ckpts/ckpts
LOGS_DIR=$SCRATCH/latest_ckpts/logs/logs_eval

python $SCRATCH/autoexperiment/scripts/eval/eval_status.py \
    --ckpts_dir=$CKPTS_DIR \
    --logs_dir=$LOGS_DIR \
    --out_dir=$OUT_FILENAMES_DIR
