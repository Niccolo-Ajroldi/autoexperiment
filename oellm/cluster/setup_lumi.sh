#!/bin/bash

# Container image
CONTAINER="/scratch/project_462000963/containers/MegatronTrainingLumi_x86_64.sif"

# Leonardo project directory
PROJECT_DIR="/scratch/project_462000963/"
PROJECT_FAST_DIR="/flash/project_462000963"

# Path to Megatron-LM repo
MEGATRON_PATH="/scratch/project_462000963/users/donutu/ROCm-Megatron-LM"

export MACHINE_NAME=LUMI
 
export CC=gcc-12
export CXX=g++-12


export SLURM_ACCOUNT="project_462000963"
export DATA_ACCOUNT="project_462000963"
export SLURM_PARTITION="dev-g"
export SLURM_PARTITION_DEBUG="dev-g"
export SLURM_QOS="normal"
export SLURM_QOS_DEBUG="normal"

export LUMI_SCRATCH="/project_462000963/"
export WORK="/scratch/$DATA_ACCOUNT/users/$USER"
export DATA_DIR="/scratch/project_462000963/preprocessed"
export OUTPUT_DIR="$WORK/output"
export APPTAINER_CACHEDIR="/scratch/project_462000963/containers"
export APPTAINER_TMPDIR="$WORK/.tmp"
export CONTAINER_CACHE_DIR="/scratch/project_462000963/containers"
export ARCH="$(uname -m)"

# MEGATRON CACHE
MEGATRON_CACHE_BASE=$PROJECT_FAST_DIR

# CACHE
MEGATRON_CACHE_FOLDER="${MEGATRON_CACHE_BASE}/users/${USER}"
export MEGATRON_CACHE="${MEGATRON_CACHE_FOLDER}/MEGATRON_CACHEDIR"
mkdir -p "$MEGATRON_CACHE_FOLDER"
mkdir -p "$MEGATRON_CACHE"

# Directories to map into the container
BIND_DIRS="${PROJECT_DIR},${PROJECT_FAST_DIR},${MEGATRON_PATH},${MEGATRON_CACHE_FOLDER}"
LUMI_FS_BINDS=
BIND_DIRS=$BIND_DIRS:/boot/config-5.14.21-150500.55.49_13.0.56-cray_shasta_c,/opt/cray,/var/spool/slurmd,/scratch/project_462000394/containers/for-turkunlp-team/tuner-2025-07-09/,/pfs,/scratch,/projappl,/project,/flash,/appl

######################################################################
# ENV VARS and SETTING
######################################################################

# DISTRIBUTED SETUP
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)  # master node hostname
#MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"  # IP numeric address
export MASTER_ADDR="$MASTER_ADDR"
export MASTER_PORT=20074
echo "MASTER_ADDR:MASTER_PORT set to: ${MASTER_ADDR}:${MASTER_PORT}"

# PERFORMANCE SETUP

#OMP THREADING
export OMP_NUM_THREADS=1                                #Virtual threads to accompany the HW threads set by slurm

#HSA=Heterogeneous System Architecture
#AMD provides HSA through ROCR https://rocm.docs.amd.com/projects/ROCR-Runtime/en/docs-6.2.4/index.html#hsa-runtime-api-and-runtime-for-rocm
export HSA_ENABLE_SDMA=0                                #https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-gpu-aware-mpi-readme/
                                                        #https://rocm.docs.amd.com/en/docs-6.2.0/conceptual/gpu-memory.html#system-direct-memory-access
                                                        #HSA_ENABLE_SDMA=0 -- Uses blit kernels for comms through the gpu -- more bandwith between gcd's
                                                        #With HSA_ENABLE_SDMA=1 capped at 50GB/S uni-directional bandwidth
                                                        #Enables overlapping communication with computation

export PYTHONWARNINGS=ignore

##TORCH COMPILE/TORCHINDUCTOR##
#export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1             #Disable all caching for Torchinductor
export TORCHINDUCTOR_MAX_AUTOTUNE=1                     #More rigorous autotuning done by Triton for GPU kernels, this might lead to more performant kernels at the cost of increased compilation time
export ROCM_PYTORCH_ARCH=gfx90a

##LIBFABRIC##
export FI_HMEM=rocr                                    #Ensure that libfabric uses rocr's hmem implementation. Not sure if this supported for rocm < 6.4
export FI_HMEM_ROCR_USE_DMABUF=1                       
export FI_MR_ROCR_CACHE_MONITOR_ENABLED=1              #Detecting ROCR device memory (FI_HMEM_ROCR) changes made between the device virtual addresses used by an application and the underlying device physical pages.

#CXI spesific libfabric variables
#These are recomended by HPE for AI workloads on rccl. See more here; https://support.hpe.com/hpesc/public/docDisplay?docId=dp00005991en_us&page=user/rccl.html&docLocale=en_US
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RX_MATCH_MODE=software                   #Address matching can be either hardware(cxi) or software
export FI_CXI_RDZV_PROTO=alt_read
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072
export NCCL_NCHANNELS_PER_PEER=16
export NCCL_MIN_CHANNELS=${NCCL_NCHANNELS_PER_PEER}

#NCCL/RCCL
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_DMABUF_ENABLE=1                             #Enable DMA buffers from the RCCL side
export NCCL_TUNER_PLUGIN=/scratch/project_462000394/containers/for-turkunlp-team/tuner-2025-07-09/librccl-tuner.so
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_DEBUG=0
export NVTE_DEBUG_LEVEL=0
##TRITON##
#export TRITON_ALWAYS_COMPILE=1                         #Always force Triton to compile even in the case of cache hit.

export TRITON_HOME=/tmp/triton_home