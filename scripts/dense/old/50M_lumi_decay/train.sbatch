#!/bin/bash

#SBATCH --time=10:30:00
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
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

######################################################################
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
######################################################################

######################################################################
# PATHS and ENV VARS
######################################################################

# Bash "strict mode" (see http://redsymbol.net/articles/unofficial-bash-strict-mode/)
set -euo pipefail

# When slurm reschedules a job that ended on node failure, it will run
# with the same job ID, clobbering the original logs. Rename the logs
# and include timestamp to avoid this.
timestamp=$(date +%Y-%m-%d_%H-%M-%S)
logid="${SLURM_JOB_NAME}-${SLURM_JOBID}"
[[ -e "{out_dir}/$logid.out" ]] && mv -f "{out_dir}/$logid.out" "{out_dir}/$logid-$timestamp.out"
[[ -e "{out_dir}/$logid.err" ]] && mv -f "{out_dir}/$logid.err" "{out_dir}/$logid-$timestamp.err"

# SOURCE CLUSTER SETUP SCRIPT
SETUP_SCRIPT={cluster_setup_script}
source $SETUP_SCRIPT

# OUTPUT_DIR
CHECKPOINT_PATH="{ckpt_dir}/$SLURM_JOB_NAME"
TENSORBOARD_DIR="${CHECKPOINT_PATH}/tensorboard"
WANDB_DIR="${CHECKPOINT_PATH}/wandb"
mkdir -p "{out_dir}" "{ckpt_dir}" "$CHECKPOINT_PATH" "$TENSORBOARD_DIR" "$WANDB_DIR"

######################################################################
# MEGATRON ARGUMENTS
######################################################################

# Megatron arguments are injected when template is rendereed
ARGS={megatron_args}

# Append args depending on path
ARGS+=(
    --load "${CHECKPOINT_PATH}"
    --save "${CHECKPOINT_PATH}"
    --tensorboard-dir "${TENSORBOARD_DIR}"
    --wandb-save-dir "${WANDB_DIR}"
    --data-cache-path "${MEGATRON_CACHE}"
)

# Inject WANDB_ENTITY into the container, in order to separate personal runs from project runs.
WANDB_ENTITY=openeurollm-project

######################################################################
# RUN
######################################################################

# CMD="${MEGATRON_PATH}/pretrain_gpt.py ${ARGS[@]}"
CMD="${MEGATRON_PATH}/pretrain_gpt.py ${ARGS[*]}"

echo '=============== CMD: ==============='
echo $CMD
echo '===================================='

cd $MEGATRON_PATH

srun \
    --exclusive \
    --wait=60 \
    --jobid=$SLURM_JOB_ID \
    bash -c \
    "singularity exec \
    --pwd $MEGATRON_PATH \
    --bind $BIND_DIRS \
    --env WANDB_ENTITY=$WANDB_ENTITY \
    $CONTAINER \
    python -u -m torch.distributed.run \
        --nproc-per-node $SLURM_GPUS_PER_NODE \
        --nnodes $SLURM_NNODES \
        --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
        --rdzv_backend static \
        --max_restarts 0 \
        --tee 3 \
        --node_rank \$SLURM_PROCID \
        --role \$SLURMD_NODENAME: \
        $CMD"

echo "END $SLURM_JOBID: $(date)"
