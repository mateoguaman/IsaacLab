#!/usr/bin/env bash

# Read environment variables for node and GPU counts, with defaults for single-GPU runs.
# This allows us to call the script like: `NODES=2 GPUS_PER_NODE=4 ./cluster_interface.sh job ...`
NODES=${NODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}

# Defaults for Hyak (Conservative defaults that fit most partitions)
ACCOUNT=${ACCOUNT:-weirdlab}
PARTITION=${PARTITION:-gpu-a40}
CPUS_PER_TASK=${CPUS_PER_TASK:-6}
MEM_PER_GPU=${MEM_PER_GPU:-60G}
CONSTRAINT=${CONSTRAINT:-"h200|l40|l40s|a40|a100"}
TIME=${TIME:-24:00:00}

# Calculate total tasks for SLURM
NTASKS=$((NODES * GPUS_PER_NODE))
echo "Requesting ${NODES} node(s) with ${GPUS_PER_NODE} GPU(s) per node."

# Derive SLURM log directory from the environment or isaaclab directory
SLURM_LOGS_DIR=${SLURM_LOGS_DIR:-"$(dirname "$1")/slurm_logs"}
mkdir -p "$SLURM_LOGS_DIR"

echo "----------------------------------------------------------------"
echo "Submitting Job to Hyak"
echo "----------------------------------------------------------------"
echo "Account:       ${ACCOUNT}"
echo "Partition:     ${PARTITION}"
echo "Nodes:         ${NODES}"
echo "GPUs per Node: ${GPUS_PER_NODE}"
echo "CPUs per Task: ${CPUS_PER_TASK}"
echo "Mem per GPU:   ${MEM_PER_GPU}"
if [ -n "${CONSTRAINT}" ]; then
    echo "Constraint:    ${CONSTRAINT}"
fi
echo "Time Limit:    ${TIME}"
echo "SLURM Logs:    ${SLURM_LOGS_DIR}"
echo "----------------------------------------------------------------"

# Enable requeue for all jobs - allows resumption after preemption or time limits
REQUEUE_FLAG="#SBATCH --requeue"

# Additional check for checkpoint partitions
if [[ "$PARTITION" == *"ckpt"* ]]; then
    echo "Detected checkpoint partition. Requeue enabled for preemption handling."
    if [[ "$ACCOUNT" != *"-ckpt" ]]; then
        echo "WARNING: Partition is '$PARTITION' but Account is '$ACCOUNT'. Checkpoint jobs usually require an account ending in '-ckpt'."
    fi
fi

# create job script with compute demands
cat <<'EOFSCRIPT' > job.sh
#!/bin/bash

# ------------------ Job Metadata ------------------
#SBATCH --job-name="isaaclab-dist-DATETIME_PLACEHOLDER"
#SBATCH --output=LOGS_PLACEHOLDER/%x-%j.out
#SBATCH --error=LOGS_PLACEHOLDER/%x-%j.err
#SBATCH --open-mode=append                        # Append to log files on requeue instead of overwriting

# ------------------ Resource Requests ------------------
#SBATCH --account=ACCOUNT_PLACEHOLDER
#SBATCH --partition=PARTITION_PLACEHOLDER
#SBATCH --nodes=NODES_PLACEHOLDER
#SBATCH --ntasks-per-node=1                       # one task per node (launch script handles distribution)
#SBATCH --gpus-per-node=GPUS_PLACEHOLDER
#SBATCH --cpus-per-task=CPUS_PLACEHOLDER
#SBATCH --mem=MEM_PLACEHOLDER
#SBATCH --time=TIME_PLACEHOLDER

# Signal handler: send USR1 30 seconds before time limit to trigger requeue
#SBATCH --signal=B:USR1@30

# Optional Constraint
CONSTRAINT_PLACEHOLDER

# Requeue flag
REQUEUE_PLACEHOLDER

# --- Requeue Handler for Time Limits and Preemption ---
requeue_handler() {
    echo "[$(date)] Caught signal: $1 - marking job $SLURM_JOB_ID for requeue"
    scontrol requeue $SLURM_JOB_ID
}

# Install signal handlers for both time limit (USR1) and preemption (TERM)
trap 'requeue_handler USR1' USR1
trap 'requeue_handler TERM' TERM

echo "[$(date)] Job $SLURM_JOB_ID starting on $(hostname)"
echo "[$(date)] Requeue handler installed - job will auto-resume if preempted or hits time limit"

# --- PyTorch Distributed Setup ---
# Find the IP address of the master node (the first node in the list)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Find a random free port for communication
export MASTER_PORT=$(($SLURM_JOB_ID % 55535 + 10000))

echo "Master Node: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"

# --- Execute the Job ---
srun --export=ALL bash "ISAACLAB_DIR_PLACEHOLDER/docker/cluster/run_singularity.sh" "ISAACLAB_DIR_PLACEHOLDER" "PROFILE_PLACEHOLDER" \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$SLURM_GPUS_ON_NODE" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --run_id="$SLURM_JOB_ID" \
    JOB_ARGS_PLACEHOLDER &

# Wait for srun but allow trap handlers to run
wait $!
EXIT_CODE=$?

echo "[$(date)] Job finished with exit code: $EXIT_CODE"
EOFSCRIPT

# Replace placeholders with actual values
# Using % as delimiter to avoid conflicts with | (in CONSTRAINT) and # (in SBATCH comments)
sed -i "s%DATETIME_PLACEHOLDER%$(date +"%Y-%m-%dT%H-%M")%g" job.sh
sed -i "s%LOGS_PLACEHOLDER%${SLURM_LOGS_DIR}%g" job.sh
sed -i "s%ACCOUNT_PLACEHOLDER%${ACCOUNT}%g" job.sh
sed -i "s%PARTITION_PLACEHOLDER%${PARTITION}%g" job.sh
sed -i "s%NODES_PLACEHOLDER%${NODES}%g" job.sh
sed -i "s%GPUS_PLACEHOLDER%${GPUS_PER_NODE}%g" job.sh
sed -i "s%CPUS_PLACEHOLDER%$((GPUS_PER_NODE * CPUS_PER_TASK))%g" job.sh
sed -i "s%MEM_PLACEHOLDER%$((GPUS_PER_NODE * $(echo ${MEM_PER_GPU} | tr -dc '0-9') ))G%g" job.sh
sed -i "s%TIME_PLACEHOLDER%${TIME}%g" job.sh
sed -i "s%ISAACLAB_DIR_PLACEHOLDER%$1%g" job.sh
sed -i "s%PROFILE_PLACEHOLDER%$2%g" job.sh
sed -i "s%JOB_ARGS_PLACEHOLDER%${*:3}%g" job.sh

# Handle optional constraint
if [ -n "${CONSTRAINT}" ]; then
    sed -i "s%CONSTRAINT_PLACEHOLDER%#SBATCH --constraint=\"${CONSTRAINT}\"%g" job.sh
else
    sed -i "s%CONSTRAINT_PLACEHOLDER%%g" job.sh
fi

# Handle requeue flag
sed -i "s%REQUEUE_PLACEHOLDER%${REQUEUE_FLAG}%g" job.sh

# Submit
sbatch --export=ALL < job.sh
rm job.sh
