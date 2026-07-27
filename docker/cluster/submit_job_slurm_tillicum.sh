#!/usr/bin/env bash

# Read environment variables for node and GPU counts, with defaults for single-GPU runs.
# This allows us to call the script like: `NODES=2 GPUS_PER_NODE=4 ./cluster_interface.sh job ...`
NODES=${NODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}

# Calculate total tasks for SLURM
NTASKS=$((NODES * GPUS_PER_NODE))
echo "Requesting ${NODES} node(s) with ${GPUS_PER_NODE} GPU(s) per node."

# Time limit (default 24 hours, use shorter for testing requeue, e.g., TIME=00:05:00)
TIME=${TIME:-24:00:00}

# Derive SLURM log directory from the environment or isaaclab directory
SLURM_LOGS_DIR=${SLURM_LOGS_DIR:-"$(dirname "$1")/slurm_logs"}
mkdir -p "$SLURM_LOGS_DIR"

echo "----------------------------------------------------------------"
echo "Submitting Job to Tillicum"
echo "----------------------------------------------------------------"
echo "Nodes:         ${NODES}"
echo "GPUs per Node: ${GPUS_PER_NODE}"
echo "Time Limit:    ${TIME}"
echo "SLURM Logs:    ${SLURM_LOGS_DIR}"
echo "----------------------------------------------------------------"

# create job script with compute demands
cat <<'EOFSCRIPT' > job.sh
#!/bin/bash

# ------------------ Job Metadata ------------------
#SBATCH --job-name="dist-training-DATETIME_PLACEHOLDER"
#SBATCH --output=LOGS_PLACEHOLDER/%x-%j.out
#SBATCH --error=LOGS_PLACEHOLDER/%x-%j.err
#SBATCH --open-mode=append                        # Append to log files on requeue instead of overwriting

# ------------------ Resource Requests ------------------
#SBATCH --nodes=NODES_PLACEHOLDER
#SBATCH --ntasks-per-node=1                       # one task per node (each task will use one GPU)
#SBATCH --gpus-per-node=GPUS_PLACEHOLDER          # one GPU per task (maps cleanly)
#SBATCH --cpus-per-task=CPUS_PLACEHOLDER          # 8 CPUs per GPU (cluster standard)
#SBATCH --mem-per-cpu=25G                         # 8 * 25G = 200G per GPU, matches cluster ratio
#SBATCH --qos=QOS_PLACEHOLDER                     # choose QOS: normal, debug, interactive, etc.
#SBATCH --time=TIME_PLACEHOLDER
#SBATCH --requeue                                 # enable requeue for time limit handling (works with --run_id)

# Signal handler: send USR1 30 seconds before time limit to trigger requeue
#SBATCH --signal=B:USR1@30

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
# Using % as delimiter to avoid conflicts with path characters
sed -i "s%DATETIME_PLACEHOLDER%$(date +"%Y-%m-%dT%H-%M")%g" job.sh
sed -i "s%LOGS_PLACEHOLDER%${SLURM_LOGS_DIR}%g" job.sh
sed -i "s%NODES_PLACEHOLDER%${NODES}%g" job.sh
sed -i "s%GPUS_PLACEHOLDER%${GPUS_PER_NODE}%g" job.sh
sed -i "s%CPUS_PLACEHOLDER%$((GPUS_PER_NODE * 8))%g" job.sh
sed -i "s%QOS_PLACEHOLDER%${QOS}%g" job.sh
sed -i "s%TIME_PLACEHOLDER%${TIME}%g" job.sh
sed -i "s%ISAACLAB_DIR_PLACEHOLDER%$1%g" job.sh
sed -i "s%PROFILE_PLACEHOLDER%$2%g" job.sh
sed -i "s%JOB_ARGS_PLACEHOLDER%${*:3}%g" job.sh

sbatch --export=ALL < job.sh
rm job.sh
