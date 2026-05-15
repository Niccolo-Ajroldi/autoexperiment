#!/bin/bash

#SBATCH --time=11:59:00
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={gpus_per_node}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --gres=gpu:{gpus_per_node}
#SBATCH --cpus-per-task=7
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
# PATHS
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

# Some arguments are known at runtime only, as they depend on ckpt_dir.
ARGS="$ARGS \
--load $CHECKPOINT_PATH \
--save $CHECKPOINT_PATH \
--tensorboard-dir $TENSORBOARD_DIR \
--wandb-save-dir $WANDB_DIR"

######################################################################
# ENV VARS
######################################################################

export MACHINE_NAME=LUMI
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=20074

export NCCL_TUNER_PLUGIN=/users/rluukkon/rccl-tuner/librccl-tuner.so
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB

export HSA_ENABLE_SDMA=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0

export CC=gcc-12
export CXX=g++-12
export CUDA_DEVICE_MAX_CONNECTIONS=1
export USE_ROCM_AITER_ROPE_BACKEND=0

ml purge

MEGATRON_PATH=/scratch/project_462000963/users/niccolo/code/NVIDIA-Megatron-LM
BIND_DIRS="/boot/config-5.14.21-150500.55.49_13.0.56-cray_shasta_c,/pfs/,/scratch/,/flash/,/opt/cray/pe/lmod/"
CONTAINER=/scratch/project_462000963/users/rluukkon/container_cache/MegatronTrainingLumi_x86_64.sif

# Set wandb entity here to seprate personal runs from project runs.
WANDB_ENTITY=openeurollm-project

######################################################################
# RUN
######################################################################

CMD="${MEGATRON_PATH}/pretrain_gpt.py ${ARGS}"

echo '=============== CMD: ==============='
echo "$CMD"
echo '===================================='

cd "$MEGATRON_PATH"

srun \
    --exclusive \
    --wait=60 \
    --jobid=$SLURM_JOB_ID \
    --network=disable_rdzv_get \
    bash -c "
        export RANK=\$SLURM_PROCID;\
        export LOCAL_RANK=\$SLURM_LOCALID;\
        export WORLD_SIZE=\$SLURM_NTASKS;\

        export MACHINE_NAME=$MACHINE_NAME;\
        export MASTER_ADDR=$MASTER_ADDR;\
        export MASTER_PORT=$MASTER_PORT;\

        export NCCL_TUNER_PLUGIN=$NCCL_TUNER_PLUGIN;\
        export NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME;\
        export NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL;\

        export HSA_ENABLE_SDMA=$HSA_ENABLE_SDMA;\
        export HSA_FORCE_FINE_GRAIN_PCIE=$HSA_FORCE_FINE_GRAIN_PCIE;\
        export NVTE_DEBUG=$NVTE_DEBUG;\
        export NVTE_DEBUG_LEVEL=$NVTE_DEBUG_LEVEL;\

        export CC=$CC;\
        export CXX=$CXX;\
        export CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_DEVICE_MAX_CONNECTIONS;\
        export USE_ROCM_AITER_ROPE_BACKEND=$USE_ROCM_AITER_ROPE_BACKEND;\

        export PYTHONPATH=${PYTHONPATH:-};\
        export PROJECT_DIR=${PROJECT_DIR:-};\
        export RUN_DIR=${RUN_DIR:-};\

        export WANDB_ENTITY=$WANDB_ENTITY;\

        singularity exec \
            --pwd $MEGATRON_PATH \
            --bind $BIND_DIRS \
            $CONTAINER \
            python $MEGATRON_PATH/pretrain_gpt.py $ARGS
    "

echo "END $SLURM_JOBID: $(date)"
