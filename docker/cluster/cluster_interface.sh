#!/usr/bin/env bash

#==
# Configurations
#==

# Exits if error occurs
set -e

# Set tab-spaces
tabs 4

# get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#==
# Functions
#==
# Function to display warnings in red
display_warning() {
    echo -e "\033[31mWARNING: $1\033[0m"
}

exit_with_error() {
    echo "[ERROR] $1" >&2
    exit 1
}

# Helper function to compare version numbers
version_gte() {
    # Returns 0 if the first version is greater than or equal to the second, otherwise 1
    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n 1)" == "$2" ]
}

# Function to check docker versions
check_docker_version() {
    # check if docker is installed
    if ! command -v docker &> /dev/null; then
        echo "[Error] Docker is not installed! Please check the 'Docker Guide' for instruction." >&2;
        exit 1
    fi
    # Retrieve Docker version
    docker_version=$(docker --version | awk '{ print $3 }')
    apptainer_version=$(apptainer --version | awk '{ print $3 }')

    # Check if Docker version is exactly 24.0.7 or Apptainer version is exactly 1.2.5
    if [ "$docker_version" = "24.0.7" ] && [ "$apptainer_version" = "1.2.5" ]; then
        echo "[INFO]: Docker version ${docker_version} and Apptainer version ${apptainer_version} are tested and compatible."

    # Check if Docker version is >= 27.0.0 and Apptainer version is >= 1.3.4
    elif version_gte "$docker_version" "27.0.0" && version_gte "$apptainer_version" "1.3.4"; then
        echo "[INFO]: Docker version ${docker_version} and Apptainer version ${apptainer_version} are tested and compatible."

    # Else, display a warning for non-tested versions
    else
        display_warning "Docker version ${docker_version} and Apptainer version ${apptainer_version} are non-tested versions. There could be issues, please try to update them. More info: https://isaac-sim.github.io/IsaacLab/source/deployment/cluster.html"
    fi
}

# Checks if a docker image exists, otherwise prints warning and exists
check_image_exists() {
    image_name="$1"
    if ! docker image inspect $image_name &> /dev/null; then
        echo "[Error] The '$image_name' image does not exist!" >&2;
        echo "[Error] You might be able to build it with /IsaacLab/docker/container.py." >&2;
        exit 1
    fi
}

# Check if the singularity image exists on the remote host, otherwise print warning and exit
check_singularity_image_exists() {
    local image_name="$1"
    local remote_tar_path="$CLUSTER_SIF_PATH/$image_name.tar"
    if ! ssh "$CLUSTER_LOGIN" "test -f $(printf '%q' "$remote_tar_path")"; then
        echo "[Error] The '$image_name' image does not exist on the remote host $CLUSTER_LOGIN!" >&2;
        exit 1
    fi
}

# Fail on env files that were edited on Windows and silently break shell sourcing.
validate_env_file() {
    local env_file="$1"
    local env_label="$2"
    if grep -q $'\r' "$env_file"; then
        exit_with_error "$env_label contains Windows CRLF line endings: $env_file. Convert it with: sed -i 's/\\r\$//' $env_file"
    fi
}

# Require a variable to be set, free of CRs, and fully expanded (no leftover ${...}, $(...), \`...\`).
require_resolved_value() {
    local var_name="$1"
    local var_value="${!var_name:-}"
    if [ -z "$var_value" ]; then
        exit_with_error "Required variable '$var_name' is empty after loading cluster config."
    fi
    if [[ "$var_value" == *$'\r'* ]]; then
        exit_with_error "Variable '$var_name' contains a carriage return (CR). Check line endings in env files."
    fi
    if [[ "$var_value" == *'${'* ]] || [[ "$var_value" == *'$('* ]] || [[ "$var_value" == *'`'* ]]; then
        exit_with_error "Variable '$var_name' is not fully resolved (value: '$var_value'). Check docker/cluster/.env.user and docker/cluster/.env.$CLUSTER_NAME."
    fi
}

# Like require_resolved_value, plus the path must be absolute.
require_absolute_path() {
    local var_name="$1"
    require_resolved_value "$var_name"
    local var_value="${!var_name}"
    if [[ "$var_value" != /* ]]; then
        exit_with_error "Variable '$var_name' must be an absolute path, got '$var_value'."
    fi
}

# Validate the combined .env.user + .env.<cluster> state before doing anything remote.
validate_cluster_config() {
    require_resolved_value CLUSTER_JOB_SCHEDULER
    require_resolved_value CLUSTER_LOGIN
    require_absolute_path CLUSTER_ISAACLAB_DIR
    require_absolute_path CLUSTER_ISAAC_SIM_CACHE_DIR
    require_absolute_path CLUSTER_SIF_PATH
    require_absolute_path CLUSTER_MOUNT_DIR
    require_resolved_value CLUSTER_PYTHON_EXECUTABLE
    require_resolved_value REMOVE_CODE_COPY_AFTER_JOB

    case "$CLUSTER_JOB_SCHEDULER" in
        SLURM|PBS)
            ;;
        *)
            exit_with_error "Unsupported CLUSTER_JOB_SCHEDULER '$CLUSTER_JOB_SCHEDULER'. Expected SLURM or PBS."
            ;;
    esac

    if [[ "$CLUSTER_LOGIN" != *@* ]]; then
        exit_with_error "CLUSTER_LOGIN '$CLUSTER_LOGIN' is invalid. Expected user@host."
    fi

    case "$REMOVE_CODE_COPY_AFTER_JOB" in
        true|false)
            ;;
        *)
            exit_with_error "REMOVE_CODE_COPY_AFTER_JOB must be 'true' or 'false', got '$REMOVE_CODE_COPY_AFTER_JOB'."
            ;;
    esac
}

# Build a shell-quoted .env.cluster payload that run_singularity.sh will source on
# the compute node. IMPORTANT: CLUSTER_ISAACLAB_DIR points to the PERSISTENT (non-
# timestamped) directory so the container mounts stable logs. The timestamped
# code-snapshot dir is passed separately as an arg to the submit script.
build_cluster_env_payload() {
    printf 'CLUSTER_JOB_SCHEDULER=%q\n' "$CLUSTER_JOB_SCHEDULER"
    printf 'CLUSTER_LOGIN=%q\n' "$CLUSTER_LOGIN"
    printf 'CLUSTER_ISAACLAB_DIR=%q\n' "$CLUSTER_ISAACLAB_DIR_PERSISTENT"
    printf 'CLUSTER_ISAAC_SIM_CACHE_DIR=%q\n' "$CLUSTER_ISAAC_SIM_CACHE_DIR"
    printf 'CLUSTER_SIF_PATH=%q\n' "$CLUSTER_SIF_PATH"
    printf 'CLUSTER_MOUNT_DIR=%q\n' "$CLUSTER_MOUNT_DIR"
    printf 'REMOVE_CODE_COPY_AFTER_JOB=%q\n' "$REMOVE_CODE_COPY_AFTER_JOB"
    printf 'CLUSTER_PYTHON_EXECUTABLE=%q\n' "$CLUSTER_PYTHON_EXECUTABLE"
    if [ -n "${CLUSTER_RUNTIME_PIP_DEPS+x}" ]; then
        printf 'CLUSTER_RUNTIME_PIP_DEPS=%q\n' "$CLUSTER_RUNTIME_PIP_DEPS"
    fi
}

# Write a manifest bootstrap into the cluster snapshot dir.
#
# Captures locally-knowable metadata (git SHA, train_args, profile, resources)
# to ${CLUSTER_ISAACLAB_DIR}/.manifest.bootstrap.json on the cluster, where
# ${CLUSTER_ISAACLAB_DIR} is the timestamped snapshot dir for this submission.
# train.py on the compute node reads this file once SLURM_JOB_ID and the
# experiment slug are known, then writes the final manifest.json next to the
# job's artifacts. See EVAL_WORKFLOW.md §3.1.
write_manifest_bootstrap() {
    local repo_root="$SCRIPT_DIR/../.."
    local sha branch dirty submitted_at submitted_by remote_path bootstrap_payload

    sha=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || echo "")
    branch=$(git -C "$repo_root" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
    if [ -z "$(git -C "$repo_root" status --porcelain 2>/dev/null)" ]; then
        dirty=false
    else
        dirty=true
    fi
    submitted_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    submitted_by="${USER}@$(hostname)"

    bootstrap_payload=$(python3 "$SCRIPT_DIR/build_manifest_bootstrap.py" \
        "$sha" "$branch" "$dirty" "$submitted_at" "$submitted_by" \
        "$CLUSTER_NAME" "$profile" "${NODES:-1}" "${GPUS_PER_NODE:-1}" \
        "$CLUSTER_ISAACLAB_DIR" "$CLUSTER_ISAACLAB_DIR_PERSISTENT" \
        -- "$@")

    remote_path="$CLUSTER_ISAACLAB_DIR/.manifest.bootstrap.json"
    echo "[INFO] Writing manifest bootstrap to $CLUSTER_LOGIN:$remote_path"
    ssh "$CLUSTER_LOGIN" "cat > $(printf '%q' "$remote_path")" <<< "$bootstrap_payload"
}

submit_job() {
    local -a job_script_args=("$@")
    local job_script_file
    local cluster_env_payload
    local remote_cluster_env_path
    local -a remote_env_assignments=()
    local var

    echo "[INFO] Arguments passed to job script ${job_script_args[*]}"

    case $CLUSTER_JOB_SCHEDULER in
        "SLURM")
            job_script_file=submit_job_slurm_${CLUSTER_NAME}.sh
            ;;
        "PBS")
            job_script_file=submit_job_pbs.sh
            ;;
        *)
            echo "[ERROR] Unsupported job scheduler specified: '$CLUSTER_JOB_SCHEDULER'. Supported options are: ['SLURM', 'PBS']"
            exit 1
            ;;
    esac

    # Write the resolved cluster config into the timestamped code copy so
    # run_singularity.sh on the compute node has everything it needs.
    cluster_env_payload="$(build_cluster_env_payload)"
    remote_cluster_env_path="$CLUSTER_ISAACLAB_DIR/docker/cluster/.env.cluster"
    ssh "$CLUSTER_LOGIN" "cat > $(printf '%q' "$remote_cluster_env_path")" <<< "$cluster_env_payload"

    # Forward WANDB_* env vars while preserving exact values.
    while IFS= read -r var; do
        remote_env_assignments+=("${var}=${!var}")
    done < <(compgen -v WANDB_)

    # Forward CLUSTER_DEBUG if set locally so run_singularity.sh picks up debug
    # instrumentation (version probe, core-dump rescue, CUDA_LAUNCH_BLOCKING,
    # PYTHONFAULTHANDLER) on the compute node too.
    if [ -n "${CLUSTER_DEBUG:-}" ]; then
        remote_env_assignments+=("CLUSTER_DEBUG=${CLUSTER_DEBUG}")
    fi

    # SSH joins its argument list with spaces and does NOT shell-escape individual args.
    # Empty strings (e.g. unset $PARTITION, $NODES) silently collapse, shifting all
    # positional indices on the remote side and corrupting argument parsing.
    # printf '%q' produces shell-safe representations (empty string → '') that survive
    # the round-trip through SSH and are correctly parsed by the remote bash.
    # shellcheck disable=SC2029
    ssh "$CLUSTER_LOGIN" "bash -s $(printf '%q ' \
        -- \
        "$CLUSTER_ISAACLAB_DIR" \
        "$CLUSTER_ISAACLAB_DIR/docker/cluster/$job_script_file" \
        "isaac-lab-$profile" \
        "$NODES" \
        "$GPUS_PER_NODE" \
        "$PARTITION" \
        "$ACCOUNT" \
        "$TIME" \
        "$CPUS_PER_TASK" \
        "$MEM_PER_GPU" \
        "$CONSTRAINT" \
        "$QOS" \
        "$SLURM_LOGS_DIR" \
        "${remote_env_assignments[@]}" \
        -- \
        "${job_script_args[@]}")" <<'REMOTE_SUBMIT'
remote_isaaclab_dir="$1"
remote_job_script="$2"
remote_profile="$3"
remote_nodes="$4"
remote_gpus_per_node="$5"
remote_partition="$6"
remote_account="$7"
remote_time="$8"
remote_cpus_per_task="$9"
remote_mem_per_gpu="${10}"
remote_constraint="${11}"
remote_qos="${12}"
remote_slurm_logs_dir="${13}"
shift 13

while [ $# -gt 0 ] && [ "$1" != "--" ]; do
    export "$1"
    shift
done

if [ $# -eq 0 ]; then
    echo "[ERROR] Failed to parse forwarded job arguments on remote host." >&2
    exit 1
fi
shift

cd "$remote_isaaclab_dir"
NODES="$remote_nodes" GPUS_PER_NODE="$remote_gpus_per_node" \
PARTITION="$remote_partition" ACCOUNT="$remote_account" TIME="$remote_time" \
CPUS_PER_TASK="$remote_cpus_per_task" MEM_PER_GPU="$remote_mem_per_gpu" CONSTRAINT="$remote_constraint" \
QOS="$remote_qos" SLURM_LOGS_DIR="$remote_slurm_logs_dir" \
bash "$remote_job_script" "$remote_isaaclab_dir" "$remote_profile" "$@"
REMOTE_SUBMIT
}

# Source user-specific configuration from .env.user (generated by setup.sh).
source_user_config() {
    local user_env_file="$SCRIPT_DIR/.env.user"
    if [ -f "$user_env_file" ]; then
        validate_env_file "$user_env_file" ".env.user"
        if grep -q '<your-' "$user_env_file"; then
            exit_with_error "docker/cluster/.env.user still contains template placeholders (<your-...>). Please fill in your real values."
        fi
        source "$user_env_file"
    else
        echo "[ERROR] User configuration file not found at $SCRIPT_DIR/.env.user" >&2
        echo "[ERROR] Generate it one of two ways:" >&2
        echo "        cluster_setup                  # if using the cluster_helpers.sh workflow" >&2
        echo "        ./docker/cluster/setup.sh      # if working directly on a branch with the bundle" >&2
        exit 1
    fi
}

# Source cluster-specific configuration (.env.hyak, .env.tillicum, ...).
# Relies on variables set by source_user_config above.
source_cluster_config() {
    local cluster_env_file="$SCRIPT_DIR/.env.$CLUSTER_NAME"
    if [ -f "$cluster_env_file" ]; then
        validate_env_file "$cluster_env_file" ".env.$CLUSTER_NAME"
        source "$cluster_env_file"
    else
        echo "[ERROR] Environment file for cluster '$CLUSTER_NAME' not found at $cluster_env_file" >&2
        exit 1
    fi
}

#==
# Main
#==

#!/bin/bash

help() {
    echo -e "\nusage: $(basename "$0") [-h] [--cluster <name>] <command> [<profile>] [<job_args>...] -- Utility for interfacing between IsaacLab and compute clusters."
    echo -e "\noptions:"
    echo -e "  -h              Display this help message."
    echo -e "  --cluster       Specify the cluster configuration to use (e.g. tillicum, hyak, delta). Defaults to 'tillicum'."
    echo -e "\ncommands:"
    echo -e "  push [<profile>]              Push the docker image to the cluster."
    echo -e "  job [<profile>] [<job_args>]  Submit a job to the cluster."
    echo -e "  validate [<profile>]          Validate local cluster config. With profile, also check that the remote image exists."
    echo -e "\nwhere:"
    echo -e "  <profile>  is the optional container profile specification. Defaults to 'base'."
    echo -e "  <job_args> are optional arguments specific to the job command."
    echo -e "\n" >&2
}

# Parse options: --cluster <name> selects which .env.<cluster> file to load later.
# Other flags are buffered into CLI_ARGS and restored as positional args.
CLUSTER_NAME="tillicum"
CLI_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cluster)
            CLUSTER_NAME="$2"
            shift 2
            ;;
        -h|--help)
            help
            exit 0
            ;;
        *)
            CLI_ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore positional arguments
set -- "${CLI_ARGS[@]}"

# Check for command
if [ $# -lt 1 ]; then
    echo "Error: Command is required." >&2
    help
    exit 1
fi

command=$1
shift
profile="base"

echo "[INFO] Using cluster configuration: $CLUSTER_NAME"

case $command in
    push)
        if [ $# -gt 1 ]; then
            echo "Error: Too many arguments for push command." >&2
            help
            exit 1
        fi
        [ $# -eq 1 ] && profile=$1
        echo "Executing push command"
        [ -n "$profile" ] && echo "Using profile: $profile"
        if ! command -v apptainer &> /dev/null; then
            echo "[INFO] Exiting because apptainer was not installed"
            echo "[INFO] You may follow the installation procedure from here: https://apptainer.org/docs/admin/main/installation.html#install-ubuntu-packages"
            exit
        fi
        # Check if Docker image exists
        check_image_exists isaac-lab-$profile:latest
        # Check docker and apptainer version
        check_docker_version
        # load user + cluster-specific configuration, then validate
        source_user_config
        source_cluster_config
        validate_cluster_config
        # make sure exports directory exists
        mkdir -p /$SCRIPT_DIR/exports
        # clear old exports for selected profile
        rm -rf /$SCRIPT_DIR/exports/isaac-lab-$profile*
        # create singularity image
        # NOTE: we create the singularity image as non-root user to allow for more flexibility. If this causes
        # issues, remove the --fakeroot flag and open an issue on the IsaacLab repository.
        cd /$SCRIPT_DIR/exports
        APPTAINER_NOHTTPS=1 apptainer build --fakeroot isaac-lab-$profile.sif docker-daemon://isaac-lab-$profile:latest
        # tar image (faster to send single file as opposed to directory with many files)
        tar -cvf /$SCRIPT_DIR/exports/isaac-lab-$profile.tar isaac-lab-$profile.sif
        # make sure target directories exist (SIF landing spot, persistent logs dir, cache dir)
        ssh $CLUSTER_LOGIN "mkdir -p $CLUSTER_SIF_PATH $CLUSTER_ISAACLAB_DIR $CLUSTER_ISAAC_SIM_CACHE_DIR"
        # send image to cluster
        scp $SCRIPT_DIR/exports/isaac-lab-$profile.tar $CLUSTER_LOGIN:$CLUSTER_SIF_PATH/isaac-lab-$profile.tar
        ;;
    job)
        if [ $# -ge 1 ]; then
            passed_profile=$1
            if [ -f "$SCRIPT_DIR/../.env.$passed_profile" ]; then
                profile=$passed_profile
                shift
            fi
        fi
        job_args="$@"
        echo "[INFO] Executing job command"
        [ -n "$profile" ] && echo -e "\tUsing profile: $profile"
        [ -n "$job_args" ] && echo -e "\tJob arguments: $job_args"
        # load user + cluster-specific configuration, then validate
        source_user_config
        source_cluster_config
        validate_cluster_config

        # Fail fast locally if wandb logging is requested without WANDB_API_KEY.
        # WANDB_* vars are forwarded from the local machine to the cluster via SSH,
        # so they MUST be set locally (e.g. in ~/.bashrc). Setting them only on the
        # cluster is NOT sufficient.
        prev_arg=""
        for arg in "$@"; do
            if [[ "$arg" == "wandb" && "$prev_arg" == "--logger" ]] || [[ "$arg" == --logger=wandb ]]; then
                if [[ -z "${WANDB_API_KEY:-}" ]]; then
                    exit_with_error "--logger wandb was requested but WANDB_API_KEY is not set on the local machine.
    WANDB_* variables must be exported in your local shell (e.g. ~/.bashrc) so they
    can be forwarded to the cluster. See the W&B section in the deployment docs."
                fi
                break
            fi
            prev_arg="$arg"
        done

        # Get current date and time
        current_datetime=$(date +"%Y%m%d_%H%M%S")
        # Save the persistent (non-timestamped) base directory for logs/checkpoints.
        # run_singularity.sh uses this to mount the permanent logs directory, while
        # the timestamped copy below is only used for the code snapshot of this job.
        CLUSTER_ISAACLAB_DIR_PERSISTENT="${CLUSTER_ISAACLAB_DIR}"
        # Append current date and time to create a unique code-snapshot directory
        CLUSTER_ISAACLAB_DIR="${CLUSTER_ISAACLAB_DIR}_${current_datetime}"
        # Check if singularity image exists on the remote host
        check_singularity_image_exists isaac-lab-$profile
        # make sure target directories exist (code snapshot, persistent logs, cache, SIF path)
        ssh $CLUSTER_LOGIN "mkdir -p $CLUSTER_ISAACLAB_DIR $CLUSTER_ISAACLAB_DIR_PERSISTENT $CLUSTER_ISAAC_SIM_CACHE_DIR $CLUSTER_SIF_PATH"
        # Sync Isaac Lab code
        echo "[INFO] Syncing Isaac Lab code..."
        rsync -rh  --exclude="*.git*" --filter=':- .dockerignore'  /$SCRIPT_DIR/../.. $CLUSTER_LOGIN:$CLUSTER_ISAACLAB_DIR
        # write the manifest bootstrap; train.py finalizes manifest.json on the compute node
        write_manifest_bootstrap "$@"
        # execute job script
        echo "[INFO] Executing job script..."
        # pass through the remaining positional args; "$@" preserves quoting
        submit_job "$@"
        ;;
    validate)
        if [ $# -gt 1 ]; then
            echo "Error: Too many arguments for validate command." >&2
            help
            exit 1
        fi
        [ $# -eq 1 ] && profile=$1

        echo "[INFO] Executing validate command"

        source_user_config
        source_cluster_config
        validate_cluster_config

        echo "[INFO] Local cluster configuration is valid."
        echo -e "\tCluster login:       $CLUSTER_LOGIN"
        echo -e "\tCluster IsaacLab dir: $CLUSTER_ISAACLAB_DIR"
        echo -e "\tCluster cache dir:    $CLUSTER_ISAAC_SIM_CACHE_DIR"
        echo -e "\tCluster SIF path:     $CLUSTER_SIF_PATH"
        echo -e "\tCluster mount dir:    $CLUSTER_MOUNT_DIR"

        if [ $# -eq 1 ]; then
            echo "[INFO] Checking remote image for profile '$profile'..."
            check_singularity_image_exists "isaac-lab-$profile"
            echo "[INFO] Remote image check passed for profile '$profile'."
        fi
        ;;
    *)
        echo "Error: Invalid command: $command" >&2
        help
        exit 1
        ;;
esac
