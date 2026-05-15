#!/bin/bash

set -e
module load cray-python

SCRATCH=/scratch/project_462000963/users/niccolo

OUT_FILENAMES_DIR=$SCRATCH/autoexperiment/scripts/eval/pre_decay/filenames
CKPTS_DIR=$SCRATCH/pre_decay_ckpts/ckpts
LOGS_DIR=$SCRATCH/pre_decay_ckpts/logs

python $SCRATCH/autoexperiment/scripts/eval/eval_status.py \
    --ckpts_dir=$CKPTS_DIR \
    --logs_dir=$LOGS_DIR \
    --out_dir=$OUT_FILENAMES_DIR
