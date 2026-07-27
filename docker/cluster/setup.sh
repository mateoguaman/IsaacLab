#!/usr/bin/env bash

# IsaacLab UW Cluster Interactive Setup Script
# This script helps users configure their .env.user file.

set -e

# Get the absolute path to the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
USER_ENV_FILE="$SCRIPT_DIR/.env.user"
TEMPLATE_FILE="$SCRIPT_DIR/.env.user.template"

if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "[ERROR] Template file not found: $TEMPLATE_FILE" >&2
    exit 1
fi

# Helper to read with a default value
read_with_default() {
    local label="$1"
    local default="$2"
    local var_name="$3"
    local val

    if [ -n "$default" ]; then
        read -p "$label [$default]: " val
    else
        read -p "$label: " val
    fi

    printf -v "$var_name" "%s" "${val:-$default}"
}

echo "----------------------------------------------------------------"
echo "        IsaacLab UW Cluster Setup Wizard"
echo "----------------------------------------------------------------"

# Load defaults: prefer existing .env.user (smart re-run), fall back to template.
# set +e while sourcing so a malformed file doesn't exit the wizard.
if [ -f "$USER_ENV_FILE" ]; then
    echo "[INFO] Loading existing configuration from .env.user"
    source_file="$USER_ENV_FILE"
else
    source_file="$TEMPLATE_FILE"
fi
set +e
# shellcheck source=/dev/null
source "$source_file"
set -e

# 1. Gather User Information
echo "This wizard will help you set up your cluster environment variables."
echo ""

# Clear placeholder values so they don't show up as suggested defaults
current_netid="${CLUSTER_USER:-}"
[[ "$current_netid" == "<your-uw-netid>" ]] && current_netid=""

current_slurm_account="${HYAK_SLURM_ACCOUNT:-}"
[[ "$current_slurm_account" == "<your-lab-group>" ]] && current_slurm_account=""

current_storage_group="${HYAK_STORAGE_GROUP:-}"
[[ "$current_storage_group" == "<your-storage-group>" ]] && current_storage_group=""

current_ncsa_username="${NCSA_USERNAME:-}"
[[ "$current_ncsa_username" == "<your-ncsa-username>" ]] && current_ncsa_username=""

current_delta_slurm_account="${DELTA_SLURM_ACCOUNT:-}"
[[ "$current_delta_slurm_account" == "<your-delta-slurm-account>" ]] && current_delta_slurm_account=""

current_deltaai_slurm_account="${DELTAAI_SLURM_ACCOUNT:-}"
[[ "$current_deltaai_slurm_account" == "<your-deltaai-slurm-account>" ]] && current_deltaai_slurm_account=""

current_ncsa_project_code="${NCSA_PROJECT_CODE:-}"
[[ "$current_ncsa_project_code" == "<your-ncsa-project-code>" ]] && current_ncsa_project_code=""

read_with_default "UW NetID" "$current_netid" netid
if [ -z "$netid" ]; then echo "[ERROR] NetID cannot be empty."; exit 1; fi

read_with_default "Hyak SLURM Account" "$current_slurm_account" hyak_slurm_account
if [ -z "$hyak_slurm_account" ]; then echo "[ERROR] SLURM Account cannot be empty."; exit 1; fi

echo ""
echo "(Press ENTER to accept the [default] value shown in brackets.)"
echo ""

# Default storage group to the SLURM account (press ENTER to auto-populate).
default_storage="${current_storage_group:-$hyak_slurm_account}"
read_with_default "Hyak Storage Group (/gscratch/<group>)" "$default_storage" hyak_storage_group

echo ""
echo "--- NCSA (Delta + DeltaAI) — leave NCSA Username blank to skip both ---"
read_with_default "NCSA Username" "$current_ncsa_username" ncsa_username
if [ -n "$ncsa_username" ]; then
    read_with_default "Delta SLURM Account" "$current_delta_slurm_account" delta_slurm_account
    if [ -z "$delta_slurm_account" ]; then echo "[ERROR] Delta SLURM Account cannot be empty when NCSA Username is set."; exit 1; fi
    # Default project code to the prefix of the SLURM account (e.g. bhiq-delta-gpu → bhiq).
    default_project_code="${current_ncsa_project_code:-${delta_slurm_account%%-*}}"
    read_with_default "NCSA Project Code" "$default_project_code" ncsa_project_code
    if [ -z "$ncsa_project_code" ]; then echo "[ERROR] NCSA Project Code cannot be empty when NCSA Username is set."; exit 1; fi
    # DeltaAI account is a separate ACCESS allocation; default derives from the
    # NCSA project code (e.g. bhiq → bhiq-dtai-gh). Leave blank to skip DeltaAI.
    default_deltaai_account="${current_deltaai_slurm_account:-${ncsa_project_code}-dtai-gh}"
    read_with_default "DeltaAI SLURM Account (blank to skip DeltaAI)" "$default_deltaai_account" deltaai_slurm_account
else
    delta_slurm_account=""
    ncsa_project_code=""
    deltaai_slurm_account=""
fi

# Seed default SLURM values from whatever we just sourced (template or existing .env.user)
hyak_partition="$HYAK_PARTITION"
hyak_cpus="$HYAK_CPUS_PER_TASK"
hyak_mem="$HYAK_MEM_PER_GPU"
hyak_constraint="$HYAK_CONSTRAINT"
hyak_time="$HYAK_TIME"
tillicum_time="$TILLICUM_TIME"
tillicum_qos="$TILLICUM_QOS"
delta_partition="$DELTA_PARTITION"
delta_cpus="$DELTA_CPUS_PER_TASK"
delta_mem_per_gpu="$DELTA_MEM_PER_GPU"
delta_time="$DELTA_TIME"
deltaai_partition="$DELTAAI_PARTITION"
deltaai_cpus="$DELTAAI_CPUS_PER_TASK"
deltaai_mem_per_gpu="$DELTAAI_MEM_PER_GPU"
deltaai_time="$DELTAAI_TIME"

# Strip any quotes around the constraint for display in the prompt
hyak_constraint="${hyak_constraint%\"}"
hyak_constraint="${hyak_constraint#\"}"

echo ""
read -p "Do you want to customize SLURM defaults (partitions, limits)? (y/N) [n]: " customize
if [[ "$customize" =~ ^[Yy]$ ]]; then
    echo ""
    echo "--- Hyak Customization ---"
    read_with_default "  Hyak Partition" "$hyak_partition" hyak_partition
    read_with_default "  Hyak CPUs per Task" "$hyak_cpus" hyak_cpus
    read_with_default "  Hyak Memory per GPU" "$hyak_mem" hyak_mem
    read_with_default "  Hyak Constraint" "$hyak_constraint" hyak_constraint
    read_with_default "  Hyak Time Limit" "$hyak_time" hyak_time

    echo ""
    echo "--- Tillicum Customization ---"
    read_with_default "  Tillicum Time Limit" "$tillicum_time" tillicum_time
    read_with_default "  Tillicum QOS" "$tillicum_qos" tillicum_qos

    if [ -n "$ncsa_username" ]; then
        echo ""
        echo "--- Delta Customization ---"
        read_with_default "  Delta Partition" "$delta_partition" delta_partition
        read_with_default "  Delta CPUs per Task" "$delta_cpus" delta_cpus
        read_with_default "  Delta Memory per GPU" "$delta_mem_per_gpu" delta_mem_per_gpu
        read_with_default "  Delta Time Limit" "$delta_time" delta_time

        if [ -n "$deltaai_slurm_account" ]; then
            echo ""
            echo "--- DeltaAI Customization ---"
            read_with_default "  DeltaAI Partition" "$deltaai_partition" deltaai_partition
            read_with_default "  DeltaAI CPUs per Task" "$deltaai_cpus" deltaai_cpus
            read_with_default "  DeltaAI Memory per GPU" "$deltaai_mem_per_gpu" deltaai_mem_per_gpu
            read_with_default "  DeltaAI Time Limit" "$deltaai_time" deltaai_time
        fi
    fi
fi

# Normalize quoting on constraint: strip any quotes the user typed, then always re-add.
hyak_constraint="${hyak_constraint%\"}"
hyak_constraint="${hyak_constraint#\"}"
hyak_constraint_quoted="\"$hyak_constraint\""

# 2. Create .env.user from template, then patch values into place.
#    Each sed replaces ONLY the value portion (up to the first whitespace or
#    start of comment), so the template's column alignment and "# Default: ..."
#    comments are preserved.
echo ""
echo "[INFO] Generating .env.user..."
cp "$TEMPLATE_FILE" "$USER_ENV_FILE"

sed -i "s%<your-uw-netid>%${netid}%g" "$USER_ENV_FILE"
sed -i "s%<your-lab-group>%${hyak_slurm_account}%g" "$USER_ENV_FILE"
sed -i "s%<your-storage-group>%${hyak_storage_group}%g" "$USER_ENV_FILE"
sed -i "s%<your-ncsa-username>%${ncsa_username}%g" "$USER_ENV_FILE"
sed -i "s%<your-delta-slurm-account>%${delta_slurm_account}%g" "$USER_ENV_FILE"
sed -i "s%<your-deltaai-slurm-account>%${deltaai_slurm_account}%g" "$USER_ENV_FILE"
sed -i "s%<your-ncsa-project-code>%${ncsa_project_code}%g" "$USER_ENV_FILE"

sed -i -E "s%^(HYAK_PARTITION=)[^#[:space:]]+%\1${hyak_partition}%"       "$USER_ENV_FILE"
sed -i -E "s%^(HYAK_CPUS_PER_TASK=)[^#[:space:]]+%\1${hyak_cpus}%"         "$USER_ENV_FILE"
sed -i -E "s%^(HYAK_MEM_PER_GPU=)[^#[:space:]]+%\1${hyak_mem}%"            "$USER_ENV_FILE"
sed -i -E "s%^(HYAK_CONSTRAINT=)[^#[:space:]]+%\1${hyak_constraint_quoted}%" "$USER_ENV_FILE"
sed -i -E "s%^(HYAK_TIME=)[^#[:space:]]+%\1${hyak_time}%"                  "$USER_ENV_FILE"
sed -i -E "s%^(TILLICUM_TIME=)[^#[:space:]]+%\1${tillicum_time}%"          "$USER_ENV_FILE"
sed -i -E "s%^(TILLICUM_QOS=)[^#[:space:]]+%\1${tillicum_qos}%"            "$USER_ENV_FILE"
sed -i -E "s%^(DELTA_PARTITION=)[^#[:space:]]+%\1${delta_partition}%"      "$USER_ENV_FILE"
sed -i -E "s%^(DELTA_CPUS_PER_TASK=)[^#[:space:]]+%\1${delta_cpus}%"       "$USER_ENV_FILE"
sed -i -E "s%^(DELTA_MEM_PER_GPU=)[^#[:space:]]+%\1${delta_mem_per_gpu}%"  "$USER_ENV_FILE"
sed -i -E "s%^(DELTA_TIME=)[^#[:space:]]+%\1${delta_time}%"                "$USER_ENV_FILE"
sed -i -E "s%^(DELTAAI_PARTITION=)[^#[:space:]]+%\1${deltaai_partition}%"  "$USER_ENV_FILE"
sed -i -E "s%^(DELTAAI_CPUS_PER_TASK=)[^#[:space:]]+%\1${deltaai_cpus}%"   "$USER_ENV_FILE"
sed -i -E "s%^(DELTAAI_MEM_PER_GPU=)[^#[:space:]]+%\1${deltaai_mem_per_gpu}%" "$USER_ENV_FILE"
sed -i -E "s%^(DELTAAI_TIME=)[^#[:space:]]+%\1${deltaai_time}%"            "$USER_ENV_FILE"

# Source the generated file to resolve paths for the summary
source "$USER_ENV_FILE"

# Protect .env.user from accidental `git add` on branches that don't have the
# feature/uw-cluster .gitignore entry. .git/info/exclude is per-clone, never committed,
# and survives the squash-merge/reset cycle. Idempotent.
exclude_file="$(git -C "$SCRIPT_DIR" rev-parse --git-path info/exclude 2>/dev/null)"
if [ -n "$exclude_file" ] && [ -f "$exclude_file" ]; then
    if ! grep -qxF "docker/cluster/.env.user" "$exclude_file"; then
        printf '\n# cluster_setup: prevent .env.user from being committed\ndocker/cluster/.env.user\n' >> "$exclude_file"
        echo "[INFO] Added docker/cluster/.env.user to $exclude_file"
    fi
fi

echo "----------------------------------------------------------------"
echo "[SUCCESS] Configuration complete!"
echo ""
echo "Summary of your configuration in .env.user:"
echo "  NetID:            $CLUSTER_USER"
echo "  Hyak SLURM:       $HYAK_SLURM_ACCOUNT"
echo "  Hyak Storage:     $HYAK_STORAGE_GROUP"
echo ""
echo "  Hyak Paths:"
echo "    Main Dir:       $HYAK_DIR"
echo "    Cache Dir:      $HYAK_CACHE_DIR"
echo "    SIF Path:       $HYAK_SIF_PATH"
echo ""
echo "  Tillicum Paths:"
echo "    Main Dir:       $TILLICUM_DIR"
echo "    Cache Dir:      $TILLICUM_CACHE_DIR"
echo "    SIF Path:       $TILLICUM_SIF_PATH"
echo ""
echo "  Hyak SLURM Defaults:"
echo "    Partition:      $HYAK_PARTITION"
echo "    CPUs:           $HYAK_CPUS_PER_TASK"
echo "    Memory:         $HYAK_MEM_PER_GPU"
echo "    Constraint:     $HYAK_CONSTRAINT"
echo "    Time Limit:     $HYAK_TIME"
echo ""
echo "  Tillicum SLURM Defaults:"
echo "    Time Limit:     $TILLICUM_TIME"
echo "    QOS:            $TILLICUM_QOS"
echo ""
if [ -n "$NCSA_USERNAME" ]; then
    echo "  Delta:"
    echo "    NCSA Username:  $NCSA_USERNAME"
    echo "    SLURM Account:  $DELTA_SLURM_ACCOUNT"
    echo "    Project Code:   $NCSA_PROJECT_CODE"
    echo ""
    echo "  Delta Paths:"
    echo "    Main Dir:       $DELTA_DIR"
    echo "    Cache Dir:      $DELTA_CACHE_DIR"
    echo "    SIF Path:       $DELTA_SIF_PATH"
    echo ""
    echo "  Delta SLURM Defaults:"
    echo "    Partition:      $DELTA_PARTITION"
    echo "    CPUs:           $DELTA_CPUS_PER_TASK"
    echo "    Memory per GPU: $DELTA_MEM_PER_GPU"
    echo "    Time Limit:     $DELTA_TIME"
    echo ""
    if [ -n "$DELTAAI_SLURM_ACCOUNT" ]; then
        echo "  DeltaAI:"
        echo "    NCSA Username:  $NCSA_USERNAME"
        echo "    SLURM Account:  $DELTAAI_SLURM_ACCOUNT"
        echo "    Project Code:   $NCSA_PROJECT_CODE"
        echo ""
        echo "  DeltaAI Paths:"
        echo "    Main Dir:       $DELTAAI_DIR"
        echo "    Cache Dir:      $DELTAAI_CACHE_DIR"
        echo "    SIF Path:       $DELTAAI_SIF_PATH"
        echo ""
        echo "  DeltaAI SLURM Defaults:"
        echo "    Partition:      $DELTAAI_PARTITION"
        echo "    CPUs:           $DELTAAI_CPUS_PER_TASK"
        echo "    Memory per GPU: $DELTAAI_MEM_PER_GPU"
        echo "    Time Limit:     $DELTAAI_TIME"
        echo ""
    else
        echo "  DeltaAI:          [skipped — DeltaAI SLURM Account not set]"
        echo ""
    fi
else
    echo "  Delta:            [skipped — NCSA Username not set]"
    echo "  DeltaAI:          [skipped — NCSA Username not set]"
    echo ""
fi
echo "The required directories on the cluster will be created automatically"
echo "the first time you run a 'push' or 'job' command."
echo "----------------------------------------------------------------"
