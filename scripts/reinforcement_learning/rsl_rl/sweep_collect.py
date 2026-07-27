#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect training logs from cluster after a sweep and optionally launch tensorboard."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


def _get_cluster_info(cluster: str) -> tuple[str, str]:
    """Return (CLUSTER_LOGIN, CLUSTER_ISAACLAB_DIR) for the given cluster name.

    Sources .env.user and .env.<cluster> in a bash subprocess so shell variable
    references like ``CLUSTER_LOGIN=${TILLICUM_LOGIN}`` are expanded before we
    capture the value. Returns them NUL-separated for safety against any
    special characters in paths.
    """
    cluster_dir = Path("docker/cluster")
    user_env = cluster_dir / ".env.user"
    cluster_env = cluster_dir / f".env.{cluster}"

    if not cluster_env.exists():
        available = [
            p.name.split(".", 2)[-1]
            for p in cluster_dir.glob(".env.*")
            if p.name not in (".env.cluster", ".env.base", ".env.user", ".env.user.template")
        ]
        raise FileNotFoundError(
            f"Cluster env file not found: {cluster_env}\nAvailable clusters: {', '.join(available)}"
        )
    if not user_env.exists():
        raise FileNotFoundError(f"User env file not found: {user_env}. Run `cluster_setup` first.")

    script = f"""
set -e
source '{user_env}'
source '{cluster_env}'
printf '%s\\0%s\\0' "$CLUSTER_LOGIN" "$CLUSTER_ISAACLAB_DIR"
"""
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
    parts = result.stdout.split("\x00")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Missing CLUSTER_LOGIN or CLUSTER_ISAACLAB_DIR after sourcing {user_env} and {cluster_env}.")
    return parts[0], parts[1]


def _list_latest_policy_paths(login: str, remote_experiment_path: str) -> list[str]:
    """Return latest model_<n>.pt path (relative) for each remote directory."""
    remote_script = r"""
import os
import re
import sys

root = sys.argv[1]
pattern = re.compile(r"^model_(\d+)\.pt$")
latest = []

for dirpath, _, filenames in os.walk(root):
    best_name = None
    best_step = -1
    for name in filenames:
        match = pattern.match(name)
        if not match:
            continue
        step = int(match.group(1))
        if step > best_step:
            best_step = step
            best_name = name
    if best_name:
        full_path = os.path.join(dirpath, best_name)
        rel_path = os.path.relpath(full_path, root)
        latest.append(rel_path)

for rel_path in sorted(latest):
    print(rel_path)
"""
    cmd = ["ssh", login, "python3", "-", remote_experiment_path]
    result = subprocess.run(cmd, input=remote_script, text=True, capture_output=True, check=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Collect training logs from cluster after a sweep.")
    parser.add_argument(
        "--sweep",
        required=True,
        help="Sweep ID(s) to collect (comma-separated timestamps, e.g. 20260212_111948).",
    )
    parser.add_argument(
        "--cluster",
        default=None,
        help="Override the cluster (defaults to what's in the sweep manifest).",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Launch tensorboard after collecting logs.",
    )
    parser.add_argument(
        "--get_policies",
        choices=["none", "last", "all"],
        default="none",
        help="Which policy checkpoints (.pt) to download: none (default), last (latest per run), all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be rsynced without actually doing it.",
    )
    args = parser.parse_args(argv)

    sweep_ids = [s.strip() for s in args.sweep.split(",")]
    manifests_dir = Path("logs") / "sweeps"
    local_log_base = Path("logs") / "rsl_rl"

    # Each sweep has a unique experiment_name recorded in its manifest.
    # We just rsync the entire experiment directory for each sweep.
    # Key: (cluster, experiment_name) -> deduplicated
    targets: dict[tuple[str, str], str] = {}  # (cluster, experiment) -> sweep_id

    for sweep_id in sweep_ids:
        manifest_path = manifests_dir / f"{sweep_id}.json"
        if not manifest_path.exists():
            print(f"[ERROR] Sweep manifest not found: {manifest_path}")
            return 1

        with manifest_path.open() as f:
            manifest = json.load(f)

        cluster = args.cluster or manifest.get("cluster", "tillicum")
        experiment = manifest.get("experiment_name")
        if not experiment:
            print(
                f"[ERROR] Sweep {sweep_id} has no experiment_name in its manifest.\n"
                f"        This sweep was created before experiment_name tracking was added.\n"
                f"        Please re-run the sweep with the latest sweep.py."
            )
            return 1

        num_jobs = manifest.get("num_jobs", 0)
        print(f"[INFO] Sweep {sweep_id}: cluster={cluster}, experiment={experiment}, jobs={num_jobs}")
        targets[(cluster, experiment)] = sweep_id

    if not targets:
        print("[WARN] Nothing to collect.")
        return 0

    # Rsync each experiment directory
    synced_dirs: list[Path] = []

    for (cluster, experiment), sweep_id in targets.items():
        login, isaaclab_dir = _get_cluster_info(cluster)
        remote_path = f"{isaaclab_dir}/logs/rsl_rl/{experiment}"
        local_dir = local_log_base / experiment
        local_dir.mkdir(parents=True, exist_ok=True)

        rsync_cmd = [
            "rsync",
            "-avz",
            "--progress",
            f"{login}:{remote_path}/",
            str(local_dir) + "/",
        ]
        if args.get_policies in ("none", "last"):
            rsync_cmd.extend(["--exclude", "*.pt"])
        print(f"\n[RSYNC] {' '.join(rsync_cmd)}")
        if not args.dry_run:
            subprocess.run(rsync_cmd, check=True)

            if args.get_policies == "last":
                latest_policies = _list_latest_policy_paths(login, remote_path)
                if latest_policies:
                    print(f"[INFO] Found {len(latest_policies)} latest policy file(s) for {experiment}")
                else:
                    print(f"[INFO] No policy files found for {experiment}")
                for rel_policy_path in latest_policies:
                    destination_dir = local_dir / Path(rel_policy_path).parent
                    destination_dir.mkdir(parents=True, exist_ok=True)
                    policy_rsync_cmd = [
                        "rsync",
                        "-avz",
                        "--progress",
                        f"{login}:{remote_path}/{rel_policy_path}",
                        str(destination_dir) + "/",
                    ]
                    print(f"[RSYNC] {' '.join(policy_rsync_cmd)}")
                    subprocess.run(policy_rsync_cmd, check=True)
        synced_dirs.append(local_dir)

    if args.dry_run:
        print("\n[DRY RUN] No files were transferred.")
    else:
        print(f"\n[INFO] Collected {len(synced_dirs)} experiment(s) to {local_log_base}/")

    # Launch tensorboard if requested
    if args.tensorboard and not args.dry_run and synced_dirs:
        logdir = ",".join(str(d) for d in synced_dirs)
        print(f"\n[INFO] Launching tensorboard --logdir {logdir}")
        subprocess.run(["tensorboard", "--logdir", logdir])

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


"""
Usage examples
--------------
Collect logs for a specific sweep:

    python scripts/reinforcement_learning/rsl_rl/sweep_collect.py \\
        --sweep 20260212_111948

Collect multiple sweeps and launch tensorboard:

    python scripts/reinforcement_learning/rsl_rl/sweep_collect.py \\
        --sweep 20260210_172221,20260212_111948 \\
        --tensorboard

Preview what would be rsynced:

    python scripts/reinforcement_learning/rsl_rl/sweep_collect.py \\
        --sweep 20260212_111948 \\
        --dry-run

Notes:
- Each sweep has a unique experiment_name (set automatically by sweep.py) which maps
  directly to a directory on the cluster: logs/rsl_rl/{experiment_name}/
- Sweep manifests are read from logs/sweeps/{sweep_id}.json
- Cluster connection info is read from docker/cluster/.env.{cluster}
- The script uses SSH and rsync, so your SSH keys must be set up for the cluster
- Only works with sweeps launched by the updated sweep.py that records experiment_name
"""
