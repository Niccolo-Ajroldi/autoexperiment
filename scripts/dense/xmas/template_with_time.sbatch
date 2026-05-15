#!/bin/bash

#SBATCH --time={time}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --gres=gpu:{gpus_per_node}
#SBATCH --mem=0
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --threads-per-core=1
#SBATCH --job-name={job_name}
#SBATCH --output={out_dir}/%x-%j.out
#SBATCH --error={out_dir}/%x-%j.err

# For quick tests on max 2 nodes: #SBATCH --qos=boost_qos_dbg
# For large node numbers (> 64 nodes): a dd #SBATCH --qos=boost_qos_bprod

# --------------------------------------------------------------------
# PATHS
#
# out_dir/
#     ├── %n-%j-%a-TIMESTAMP.err
#     └── %n-%j-%a-TIMESTAMP.out
#
# ckpt_dir/
#     └── %n/
#         ├── tensorboard/
#         ├── wandb/
#         ├── iter_0001000/
#         ├── iter_0002000/
#         └── ...
#
# MEGATRON_CACHE_BASE/
# └── USER/
#     ├── MEGATRON_CACHEDIR/
#     ├── APPTAINER_CACHEDIR/
#     └── APPTAINER_TMPDIR/
#
# This is a template, variables in brackets are meant to be replaced.
#
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# LOGS
# --------------------------------------------------------------------

# Bash "strict mode" (see http://redsymbol.net/articles/unofficial-bash-strict-mode/)
set -euo pipefail

# When slurm reschedules a job that ended on node failure, it will run
# with the same job ID, clobbering the original logs. Rename the logs
# and include timestamp to avoid this.
timestamp=$(date +%Y-%m-%d_%H-%M-%S)
logid="${SLURM_JOB_NAME}-${SLURM_JOBID}"
[[ -e "{out_dir}/$logid.out" ]] && mv -f "{out_dir}/$logid.out" "{out_dir}/$logid-$timestamp.out"
[[ -e "{out_dir}/$logid.err" ]] && mv -f "{out_dir}/$logid.err" "{out_dir}/$logid-$timestamp.err"

# --------------------------------------------------------------------
# CLUSTER SETUP
# --------------------------------------------------------------------

SETUP_SCRIPT={cluster_setup_script}
source $SETUP_SCRIPT

# --------------------------------------------------------------------
# UNIQUE AITER JIT DIR per job-array task to avoid cross-job build locks
# --------------------------------------------------------------------

export AITER_JIT_DIR="$LUMI_SCRATCH/users/$USER/aiter_jit/${SLURM_JOB_ID}"
mkdir -p "$AITER_JIT_DIR"

# --------------------------------------------------------------------
# MEGATRON ARGUMENTS
# --------------------------------------------------------------------

# Some directories depend on the JOB_NAME, so we set them here.
CHECKPOINT_PATH="{ckpt_dir}/$SLURM_JOB_NAME"
TENSORBOARD_DIR="${CHECKPOINT_PATH}/tensorboard"
WANDB_DIR="${CHECKPOINT_PATH}/wandb"
mkdir -p "{out_dir}" "{ckpt_dir}" "$CHECKPOINT_PATH" "$TENSORBOARD_DIR" "$WANDB_DIR"

# WANDB_ENTITY: Megatron does not support setting this via command line argument,
# we need to set an environment variable.
WANDB_ENTITY={wandb_entity}
WANDB__SERVICE_WAIT=300
WANDB_HTTP_TIMEOUT=300

# wandb exp name: we append the slurm job id to the given name, 
# to ensure uniqueness and ease of log tracking.
WANDB_EXP_NAME="{wandb_exp_name}_$SLURM_JOB_ID"

# Megatron arguments are injected by autoexperiment when the template is rendered.
ARGS={megatron_args}

# Append args depending on path
ARGS+=(
    --load "${CHECKPOINT_PATH}"
    --save "${CHECKPOINT_PATH}"
    --tensorboard-dir "${TENSORBOARD_DIR}"
    --wandb-save-dir "${WANDB_DIR}"
    --data-cache-path "${MEGATRON_CACHE}"
    --wandb-exp-name "${WANDB_EXP_NAME}"
)

# --------------------------------------------------------------------
# RUN
# --------------------------------------------------------------------

CMD="${MEGATRON_PATH}/pretrain_gpt.py ${ARGS[*]}"

echo '=============== CMD: ==============='
echo $CMD
echo '===================================='

cd $MEGATRON_PATH

######
# srun error handling:
# --wait=60:                        wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1:             terminate a step if any task exits with a non-zero exit code
# --network=disable_rdzv_get:       an alternative rendezvous protocol designed by HPE spesifically for AI workloads
######

srun \
    --exclusive \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --jobid=$SLURM_JOB_ID \
    --network=disable_rdzv_get \
    bash -c \
    "singularity exec \
    --pwd $MEGATRON_PATH \
    --env WANDB_ENTITY=$WANDB_ENTITY \
    --env WANDB__SERVICE_WAIT=$WANDB__SERVICE_WAIT \
    --env WANDB_HTTP_TIMEOUT=$WANDB_HTTP_TIMEOUT \
    --env AITER_JIT_DIR=$AITER_JIT_DIR \
    --bind $BIND_DIRS \
    $CONTAINER \
    python -u -m torch.distributed.run \
      --nnodes $SLURM_NNODES \
      --nproc-per-node $SLURM_GPUS_PER_NODE \
      --node_rank \$SLURM_PROCID \
      --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
      --rdzv_backend c10d \
      --max_restarts 0 \
      --tee 3 \
      --role $SLURMD_NODENAME: \
      $CMD"

echo "END $SLURM_JOBID: $(date)"

[[ -n "${AITER_JIT_DIR:-}" ]] && rm -rf "$AITER_JIT_DIR"
echo "REMOVED AITER_JIT_DIR at $AITER_JIT_DIR"

