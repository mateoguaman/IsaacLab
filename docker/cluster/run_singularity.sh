#!/usr/bin/env bash

echo "(run_singularity.sh): Initial call on compute node."

# --- Argument Parsing ---
ISAACLAB_DIR_ARG="$1"
PROFILE_ARG="$2"
shift 2

# Use job-specific paths to prevent multiple jobs on the same node from interfering.
# When SLURM schedules multiple jobs to the same node, they share $TMPDIR; without
# unique paths, jobs delete each other's files causing squashfuse crashes.
TMPDIR=${TMPDIR:-/tmp}

# If the host's TMPDIR is on tmpfs (typical default for HPC compute nodes),
# redirect to a disk-backed filesystem. This is critical because NVRTC's C++
# frontend writes multi-MB JIT artifacts to /tmp inside the container, and
# Apptainer's --containall creates that /tmp as tmpfs backed by this same
# host TMPDIR. Once tmpfs fills, writes to mmap'd intermediate files SIGBUS
# deep inside warp.so (observed as the wp_cuda_graph_end_capture+0x212740
# crash we chased here).
tmp_fs="$(stat -f -c %T "$TMPDIR" 2>/dev/null)"
if [ "$tmp_fs" = "tmpfs" ]; then
    for alt in "${SLURM_TMPDIR:-}" "/gpfs/scrubbed/$USER" "/scratch/$USER" "$HOME/tmp"; do
        [ -z "$alt" ] && continue
        if mkdir -p "$alt" 2>/dev/null \
           && [ "$(stat -f -c %T "$alt" 2>/dev/null)" != "tmpfs" ]; then
            echo "[INFO] Host TMPDIR was tmpfs; redirecting to $alt (disk-backed)"
            TMPDIR="$alt"
            break
        fi
    done
fi

JOB_TMPDIR="$TMPDIR/job_$SLURM_JOB_ID"
mkdir -p "$JOB_TMPDIR"
# Dedicated directory bound into the container as /tmp — gives NVRTC and
# anything else in the container a writable location that is NOT tmpfs.
mkdir -p "$JOB_TMPDIR/container_tmp"
echo "[INFO] Using job-specific temp directory: $JOB_TMPDIR"

# Opt-in debug mode. Set CLUSTER_DEBUG=1 when submitting to enable: container
# version probe, CUDA_LAUNCH_BLOCKING, PYTHONFAULTHANDLER, core-dump rescue
# (local + coredumpctl + /var/lib/systemd/coredump). Off by default to keep
# normal jobs fast and quiet.
CLUSTER_DEBUG="${CLUSTER_DEBUG:-0}"
DEBUG_ENV_FLAGS=()
if [ "$CLUSTER_DEBUG" = "1" ]; then
    DEBUG_ENV_FLAGS+=(--env CUDA_LAUNCH_BLOCKING=1 --env PYTHONFAULTHANDLER=1 --env CLUSTER_DEBUG=1)
    ulimit -c unlimited
    echo "[DEBUG] CLUSTER_DEBUG=1 — serialised CUDA, faulthandler, core-dump rescue enabled"
fi

# Split incoming args into distributed-launcher flags (consumed by torch.distributed.run)
# and script-level flags (forwarded to the Python training script).
CLI_ARGS=()
DIST_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nnodes=*|--nproc_per_node=*|--rdzv_id=*|--rdzv_endpoint=*)
            DIST_ARGS+=("$1")
            shift
            ;;
        *)
            CLI_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "Parsed Distributed Args: ${DIST_ARGS[@]}"
echo "Parsed Script CLI Args: ${CLI_ARGS[@]}"

# Detect the selected logger from script arguments (used later to fail fast if wandb
# is requested but WANDB_API_KEY is missing on the compute node).
LOGGER_TYPE=""
for ((i=0; i<${#CLI_ARGS[@]}; i++)); do
    if [[ "${CLI_ARGS[$i]}" == "--logger" ]] && (( i + 1 < ${#CLI_ARGS[@]} )); then
        LOGGER_TYPE="${CLI_ARGS[$((i + 1))]}"
        break
    elif [[ "${CLI_ARGS[$i]}" == --logger=* ]]; then
        LOGGER_TYPE="${CLI_ARGS[$i]#--logger=}"
        break
    fi
done

#==
# Helper functions
#==

exit_with_error() {
    echo "[ERROR] $1" >&2
    exit 1
}

# Require a variable to be set, CR-free, and fully expanded. Mirrors the same check
# cluster_interface.sh runs locally, but repeated here as a defense-in-depth guard
# against malformed .env files reaching the compute node.
require_runtime_value() {
    local var_name="$1"
    local var_value="${!var_name:-}"
    if [ -z "$var_value" ]; then
        exit_with_error "Required variable '$var_name' is empty on compute node."
    fi
    if [[ "$var_value" == *$'\r'* ]]; then
        exit_with_error "Variable '$var_name' contains carriage return (CR). Check remote docker/cluster/.env.cluster formatting."
    fi
    if [[ "$var_value" == *'${'* ]] || [[ "$var_value" == *'$('* ]] || [[ "$var_value" == *'`'* ]]; then
        exit_with_error "Variable '$var_name' is unresolved on compute node (value: '$var_value')."
    fi
}

require_runtime_absolute_path() {
    local var_name="$1"
    require_runtime_value "$var_name"
    local var_value="${!var_name}"
    if [[ "$var_value" != /* ]]; then
        exit_with_error "Variable '$var_name' must be an absolute path, got '$var_value'."
    fi
}

validate_runtime_cluster_config() {
    require_runtime_absolute_path CLUSTER_ISAACLAB_DIR
    require_runtime_absolute_path CLUSTER_ISAAC_SIM_CACHE_DIR
    require_runtime_absolute_path CLUSTER_SIF_PATH
    require_runtime_absolute_path CLUSTER_MOUNT_DIR
    require_runtime_value CLUSTER_PYTHON_EXECUTABLE
    require_runtime_value REMOVE_CODE_COPY_AFTER_JOB
    require_runtime_value DOCKER_ISAACSIM_ROOT_PATH
    require_runtime_value DOCKER_USER_HOME

    case "$REMOVE_CODE_COPY_AFTER_JOB" in
        true|false)
            ;;
        *)
            exit_with_error "REMOVE_CODE_COPY_AFTER_JOB must be 'true' or 'false', got '$REMOVE_CODE_COPY_AFTER_JOB'."
            ;;
    esac
}

setup_directories() {
    # Create the persistent Isaac Sim cache directory layout on the shared
    # filesystem. These dirs are copied to the compute node at job start and
    # rsynced back on exit so kit/ov/shader caches persist across jobs.
    for dir in \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/kit" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/ov" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/pip" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/glcache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/computecache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/logs" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/data" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/documents"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            echo "Created directory: $dir"
        fi
    done
}


#==
# Main
#==


# get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# load variables to set the Isaac Lab path on the cluster.
# Guard against the compute node seeing a missing or Windows-CRLF-mangled env file
# before we source it (silent sourcing of CRLF-poisoned files produces hard-to-debug
# downstream failures).
CLUSTER_ENV_FILE="$SCRIPT_DIR/.env.cluster"
BASE_ENV_FILE="$SCRIPT_DIR/../.env.base"

if [ ! -f "$CLUSTER_ENV_FILE" ]; then
    exit_with_error "Missing $CLUSTER_ENV_FILE on compute node."
fi
if grep -q $'\r' "$CLUSTER_ENV_FILE"; then
    exit_with_error "$CLUSTER_ENV_FILE has CRLF line endings. Re-submit the job after fixing local env files."
fi
if [ ! -f "$BASE_ENV_FILE" ]; then
    exit_with_error "Missing $BASE_ENV_FILE on compute node."
fi
if grep -q $'\r' "$BASE_ENV_FILE"; then
    exit_with_error "$BASE_ENV_FILE has CRLF line endings."
fi

source "$CLUSTER_ENV_FILE"
source "$BASE_ENV_FILE"
validate_runtime_cluster_config

# make sure that all directories exist in the persistent cache directory
setup_directories

# warn if nested docker-isaac-sim directories exist (caused by older versions of
# this script that could recurse the cache into itself)
if find "$CLUSTER_ISAAC_SIM_CACHE_DIR" -mindepth 2 -type d -name docker-isaac-sim | grep -q .; then
    echo "[WARN] Detected nested 'docker-isaac-sim' directories under $CLUSTER_ISAAC_SIM_CACHE_DIR."
    echo "[WARN] Please remove them (e.g., 'rm -rf $CLUSTER_ISAAC_SIM_CACHE_DIR/docker-isaac-sim') to avoid cache recursion."
fi

# Clean up any leftover data from a prior invocation of this job (e.g., a
# SLURM requeue with the same SLURM_JOB_ID). Without this, a requeue could race
# with partial artifacts from the previous attempt.
rm -rf "$JOB_TMPDIR/docker-isaac-sim"
rm -rf "$JOB_TMPDIR/$PROFILE_ARG.sif"

# Copy all cache files to the job-specific temp directory (Isaac Sim needs
# a writable working copy; keeping the persistent cache read-only during the job).
cp -r "$CLUSTER_ISAAC_SIM_CACHE_DIR" "$JOB_TMPDIR"

# mkdir cache/warp so the bind source exists even if the persistent cache predates it.
mkdir -p "$JOB_TMPDIR/docker-isaac-sim/cache/warp"

# make sure logs directory exists (in the permanent isaaclab directory)
mkdir -p "$CLUSTER_ISAACLAB_DIR/logs"
touch "$CLUSTER_ISAACLAB_DIR/logs/.keep"

# copy the temporary isaaclab code snapshot to the compute node
cp -r "$ISAACLAB_DIR_ARG" "$JOB_TMPDIR"
# Get the directory name
dir_name=$(basename "$ISAACLAB_DIR_ARG")

# Ensure the logs bind destination exists in the code copy: the code rsync
# excludes logs/ (.dockerignore), so /workspace/isaaclab/logs is absent once the
# code is bind-mounted, and the persistent-logs bind (-B .../logs:/workspace/
# isaaclab/logs) fails with "destination doesn't exist". Create it here.
mkdir -p "$JOB_TMPDIR/$dir_name/logs"

# Extract container to the job-specific directory, with retries.
# Transient GPFS read failures are common enough on HPC filesystems that a
# single-shot tar extraction occasionally fails; retries make the script robust
# without hiding persistent failures.
MAX_RETRIES=3
RETRY_DELAY=5
for attempt in $(seq 1 $MAX_RETRIES); do
    echo "[INFO] Extracting container (attempt $attempt/$MAX_RETRIES)..."
    if tar -xf "$CLUSTER_SIF_PATH/$PROFILE_ARG.tar" -C "$JOB_TMPDIR" 2>&1; then
        # Verify the .sif file was extracted and is non-empty.
        if [ -f "$JOB_TMPDIR/$PROFILE_ARG.sif" ] && [ -s "$JOB_TMPDIR/$PROFILE_ARG.sif" ]; then
            echo "[INFO] Container extracted successfully."
            break
        else
            echo "[WARN] Container file missing or empty after extraction."
        fi
    else
        echo "[WARN] Tar extraction failed."
    fi

    if [ $attempt -lt $MAX_RETRIES ]; then
        echo "[INFO] Retrying in ${RETRY_DELAY}s..."
        sleep $RETRY_DELAY
        rm -f "$JOB_TMPDIR/$PROFILE_ARG.sif"  # clean up partial extraction
    else
        exit_with_error "Failed to extract container after $MAX_RETRIES attempts."
    fi
done

# Fail fast with a clear message if wandb logging was requested without a key.
# cluster_interface.sh checks this locally too, but the check here catches the
# case where the job is submitted from a different machine.
if [[ "$LOGGER_TYPE" == "wandb" ]] && [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[ERROR] --logger wandb was requested but WANDB_API_KEY is empty on the compute node." >&2
    echo "[ERROR] Ensure WANDB_API_KEY is exported before job submission so it reaches sbatch/srun." >&2
    exit 1
fi

# Forward selected env prefixes into the container.
# --containall strips the host environment, so we explicitly pass variables with
# --env. Default: forward anything starting with WANDB_. Users can override via
# CLUSTER_FORWARD_ENV_PREFIXES (comma-separated) if they need more.
FORWARD_ENV_PREFIXES_CSV="${CLUSTER_FORWARD_ENV_PREFIXES:-WANDB_}"
IFS=',' read -r -a FORWARD_ENV_PREFIXES <<< "$FORWARD_ENV_PREFIXES_CSV"

FORWARDED_ENV_FLAGS=()
FORWARDED_ENV_NAMES=()
for var in $(compgen -v); do
    for prefix in "${FORWARD_ENV_PREFIXES[@]}"; do
        [[ -z "$prefix" ]] && continue
        if [[ "$var" == "${prefix}"* ]]; then
            FORWARDED_ENV_FLAGS+=(--env "${var}=${!var}")
            FORWARDED_ENV_NAMES+=("${var}")
            break
        fi
    done
done

# Always forward SLURM_JOB_ID: write_run_manifest reads it to detect cluster
# runs and to load .manifest.bootstrap.json. --containall strips it otherwise,
# silently falling back to local-mode manifests with empty git_sha and
# train_args=[]. Hardcoded rather than added to FORWARD_ENV_PREFIXES so users
# don't have to opt into correct manifest provenance.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    FORWARDED_ENV_FLAGS+=(--env "SLURM_JOB_ID=${SLURM_JOB_ID}")
    FORWARDED_ENV_NAMES+=("SLURM_JOB_ID")
fi

# Pin Warp's PCH cache to the bound dir; otherwise --containall puts $HOME on
# a tiny tmpfs and NVRTC ENOSPCs on the multi-MB PCH.
FORWARDED_ENV_FLAGS+=(--env "WARP_CACHE_PATH=${DOCKER_USER_HOME}/.cache/warp")
FORWARDED_ENV_NAMES+=("WARP_CACHE_PATH")

# Runtime pip install for SIF-missing deps. Override via CLUSTER_RUNTIME_PIP_DEPS
# (space-separated); empty string disables. Installed into /tmp/runtime_pip.
CLUSTER_RUNTIME_PIP_DEPS=${CLUSTER_RUNTIME_PIP_DEPS-shapely mapbox-earcut manifold3d}
if [ -n "$CLUSTER_RUNTIME_PIP_DEPS" ]; then
    FORWARDED_ENV_FLAGS+=(--env "CLUSTER_RUNTIME_PIP_DEPS=$CLUSTER_RUNTIME_PIP_DEPS")
    FORWARDED_ENV_NAMES+=("CLUSTER_RUNTIME_PIP_DEPS")
    echo "[INFO] Runtime pip-install deps queued: $CLUSTER_RUNTIME_PIP_DEPS"
fi

if [ ${#FORWARDED_ENV_FLAGS[@]} -gt 0 ]; then
    echo "[INFO] Forwarding environment variables into container: ${FORWARDED_ENV_NAMES[*]}"
fi

if [ "$CLUSTER_DEBUG" = "1" ]; then
    echo "[DEBUG] Container runtime versions:"
    singularity exec --nv --containall "$JOB_TMPDIR/$PROFILE_ARG.sif" \
        /isaac-sim/python.sh -c "import warp, torch; print('[DEBUG]  warp=' + warp.__version__, ' torch=' + torch.__version__, ' cuda_runtime=' + str(torch.version.cuda))" \
        || echo "[DEBUG] version probe failed; continuing"
fi

# execute command in singularity container
# NOTE: ISAACLAB_PATH is normally set in `isaaclab.sh` but we directly call the
# isaac-sim python because we sync the entire Isaac Lab directory to the compute
# node and remove the symbolic link to isaac-sim.
#
# VK_LOADER_LAYERS_DISABLE prevents server GPUs (H200, L40, A100, ...) from
# crashing with VkResult: ERROR_DEVICE_LOST due to the laptop-oriented Optimus
# Vulkan layer that ships in Isaac Sim's container image.
#
# torch.distributed.run is invoked so --nnodes / --nproc_per_node / --rdzv_*
# from the submit script reach the launcher instead of the training script.
# The trailing --distributed tells AppLauncher to enable distributed mode.
singularity exec \
    -B "$JOB_TMPDIR/docker-isaac-sim/cache/kit":${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw \
    -B "$JOB_TMPDIR/docker-isaac-sim/cache/ov":${DOCKER_USER_HOME}/.cache/ov:rw \
    -B "$JOB_TMPDIR/docker-isaac-sim/cache/pip":${DOCKER_USER_HOME}/.cache/pip:rw \
    -B "$JOB_TMPDIR/docker-isaac-sim/cache/glcache":${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw \
    -B "$JOB_TMPDIR/docker-isaac-sim/cache/computecache":${DOCKER_USER_HOME}/.nv/ComputeCache:rw \
    -B "$JOB_TMPDIR/docker-isaac-sim/cache/warp":${DOCKER_USER_HOME}/.cache/warp:rw \
    -B "$JOB_TMPDIR/docker-isaac-sim/logs":${DOCKER_USER_HOME}/.nvidia-omniverse/logs:rw \
    -B "$JOB_TMPDIR/docker-isaac-sim/data":${DOCKER_USER_HOME}/.local/share/ov/data:rw \
    -B "$JOB_TMPDIR/docker-isaac-sim/documents":${DOCKER_USER_HOME}/Documents:rw \
    -B "$JOB_TMPDIR/$dir_name":/workspace/isaaclab:rw \
    -B "$CLUSTER_ISAACLAB_DIR/logs":/workspace/isaaclab/logs:rw \
    -B "$CLUSTER_MOUNT_DIR:$CLUSTER_MOUNT_DIR:rw" \
    -B "$JOB_TMPDIR/container_tmp":/tmp:rw \
    "${FORWARDED_ENV_FLAGS[@]}" \
    "${DEBUG_ENV_FLAGS[@]}" \
    --nv --containall "$JOB_TMPDIR/$PROFILE_ARG.sif" \
    bash -c '
        if [ -n "${CLUSTER_RUNTIME_PIP_DEPS:-}" ]; then
            mkdir -p /tmp/runtime_pip
            echo "[run_singularity] Installing runtime deps: $CLUSTER_RUNTIME_PIP_DEPS"
            for _pkg in $CLUSTER_RUNTIME_PIP_DEPS; do
                /isaac-sim/python.sh -m pip install --quiet --target /tmp/runtime_pip "$_pkg" \
                    || echo "[run_singularity] WARN: $_pkg install failed; continuing" >&2
            done
            export PYTHONPATH="/tmp/runtime_pip:${PYTHONPATH:-}"
        fi
        export VK_LOADER_LAYERS_DISABLE=VK_LAYER_NV_optimus
        export ISAACLAB_PATH=/workspace/isaaclab
        cd /workspace/isaaclab
        if [ "${CLUSTER_DEBUG:-0}" = "1" ]; then ulimit -c unlimited; fi
        exec /isaac-sim/python.sh -m torch.distributed.run --rdzv_backend=c10d "$@"
    ' _ "${DIST_ARGS[@]}" "$CLUSTER_PYTHON_EXECUTABLE" "${CLI_ARGS[@]}" --distributed
EXIT_CODE=$?

# On failure, rescue Kit's breakpad crash dumps before the job-tmp is wiped.
# They hold the native stack for crashes that never reach a Python traceback.
if [ "$EXIT_CODE" -ne 0 ]; then
    dump_dest="$CLUSTER_ISAACLAB_DIR/logs/crash_dumps/${SLURM_JOB_ID}"
    shopt -s nullglob globstar
    dumps=("$JOB_TMPDIR"/container_tmp/**/*.dmp "$JOB_TMPDIR"/docker-isaac-sim/**/*.dmp)
    if [ ${#dumps[@]} -gt 0 ]; then
        mkdir -p "$dump_dest"
        for dump_file in "${dumps[@]}"; do
            cp "$dump_file" "$dump_dest/" && echo "[INFO] Rescued crash dump: $(basename "$dump_file")"
        done
        echo "[INFO] Crash dumps saved to $dump_dest"
    else
        echo "[INFO] Job failed with code $EXIT_CODE; no breakpad dumps found."
    fi
    shopt -u nullglob globstar
fi

# Debug-only: core-dump rescue. Copies any core files the job produced into
# $CLUSTER_ISAACLAB_DIR/logs/cores/ before the job-tmp is wiped. Checks three
# possible locations: the job-local workspace (if core_pattern wrote there),
# the systemd-coredump journal (via coredumpctl), and the raw on-disk
# /var/lib/systemd/coredump/ directory (if the journal is group-restricted).
if [ "$CLUSTER_DEBUG" = "1" ]; then
    mkdir -p "$CLUSTER_ISAACLAB_DIR/logs/cores"
    shopt -s nullglob
    for core_file in "$JOB_TMPDIR/$dir_name"/core* "$JOB_TMPDIR/$dir_name"/*/core*; do
        dest="$CLUSTER_ISAACLAB_DIR/logs/cores/core.${SLURM_JOB_ID}.$(basename "$core_file")"
        mv "$core_file" "$dest"
        echo "[DEBUG] Rescued core dump to $dest"
    done
    shopt -u nullglob

    if command -v coredumpctl >/dev/null 2>&1; then
        echo "[DEBUG] systemd-coredump recent python3 crashes:"
        coredumpctl --no-pager list --since="10 min ago" COMM=python3 2>&1 | tail -10 || true
        cd_out="$CLUSTER_ISAACLAB_DIR/logs/cores/core.${SLURM_JOB_ID}.python3"
        cd_err="${cd_out}.extract.stderr"
        if coredumpctl --no-pager -1 dump COMM=python3 > "$cd_out" 2>"$cd_err"; then
            echo "[DEBUG] Extracted core via coredumpctl to $cd_out ($(stat -c '%s' "$cd_out" 2>/dev/null) bytes)"
        else
            echo "[DEBUG] coredumpctl dump failed; see $cd_err"
            rm -f "$cd_out"
        fi
    fi

    if [ -d /var/lib/systemd/coredump ]; then
        shopt -s nullglob
        for sc_core in /var/lib/systemd/coredump/core.python3.*; do
            if [ -r "$sc_core" ]; then
                dest="$CLUSTER_ISAACLAB_DIR/logs/cores/core.${SLURM_JOB_ID}.$(basename "$sc_core")"
                cp "$sc_core" "$dest" && echo "[DEBUG] Copied $sc_core -> $dest"
            fi
        done
        shopt -u nullglob
    fi
fi

# copy resulting cache files back to persistent storage
rsync -azPv "$JOB_TMPDIR/docker-isaac-sim" "$CLUSTER_ISAAC_SIM_CACHE_DIR/.."

# clean up job-specific temp directory to free node-local disk space
rm -rf "$JOB_TMPDIR"
echo "[INFO] Cleaned up job temp directory: $JOB_TMPDIR"

# if defined, remove the temporary isaaclab code snapshot pushed for this job
if $REMOVE_CODE_COPY_AFTER_JOB; then
    rm -rf "$ISAACLAB_DIR_ARG"
fi

echo "(run_singularity.sh): Return"

# Hand the training exit code back to srun so the submit script sees it.
# Defaults to 0 for the paths that return before the training launch.
exit "${EXIT_CODE:-0}"
