#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility to launch hyper-parameter sweeps via the cluster interface."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path


def _parse_cluster_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Submit multiple cluster jobs by expanding comma-separated CLI values."
    )
    parser.add_argument("--cluster", default="tillicum", help="Cluster to target (tillicum or hyak).")
    parser.add_argument(
        "--profile",
        default=None,
        help="Optional container profile to pass to cluster_interface.sh (defaults to base).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the expanded commands without submitting them.",
    )
    parser.add_argument(
        "--submission-delay",
        type=float,
        default=3.0,
        help="Seconds to wait between job submissions to reduce GPFS contention (default: 3.0).",
    )
    return parser.parse_known_args(argv)


# Shell environment variables: VAR=value (no dots in key)
ENV_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")

# Config overrides for training script: dotted.path=value (must have at least one dot)
CONFIG_OVERRIDE_PATTERN = re.compile(
    r"^[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+=[^=].*$"
    r"|^[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+=$"
)


def _extract_env_and_config(tokens: Sequence[str]) -> tuple[dict[str, str], list[tuple[str, str]], list[str]]:
    """Separate VAR=value env tokens and dotted.config=value overrides from flags."""
    env_vars: dict[str, str] = {}
    config_overrides: list[tuple[str, str]] = []
    remaining: list[str] = []
    for token in tokens:
        if token.startswith("--"):
            remaining.append(token)
        elif token.startswith("presets="):
            # Hydra's only non-dotted config key. Route to config_overrides so
            # train.py sees it; otherwise ENV_PATTERN would capture it as a
            # shell var and it would never reach the training script.
            key, value = token.split("=", 1)
            config_overrides.append((key, value))
        elif ENV_PATTERN.match(token):
            key, value = token.split("=", 1)
            env_vars[key] = value
        elif CONFIG_OVERRIDE_PATTERN.match(token):
            key, value = token.split("=", 1)
            config_overrides.append((key, value))
        else:
            remaining.append(token)
    return env_vars, config_overrides, remaining


def _tokenize_flags(tokens: Sequence[str]) -> list[tuple[str, str | None]]:
    """Convert CLI tokens into ordered (flag, value) pairs.

    Boolean flags (no value) are supported. A token is consumed as a value only if
    it doesn't start with '-' (to avoid consuming negative numbers as values for
    boolean flags that precede config overrides).
    """
    entries: list[tuple[str, str | None]] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected positional argument: {token}")
        if "=" in token:
            flag, value = token.split("=", 1)
            entries.append((flag, value))
        else:
            # look ahead for value - only consume if it doesn't look like a flag
            # and doesn't start with '-' (could be negative number for next config override)
            value = None
            if i + 1 < len(tokens):
                next_token = tokens[i + 1]
                if not next_token.startswith("-"):
                    value = next_token
                    i += 1
            entries.append((token, value))
        i += 1
    return entries


def _split_preset_values(value: str) -> list[str]:
    """Split a ``presets=`` value into one or more preset combos.

    Comma keeps its Hydra meaning of "list of presets in one combo", so
    ``presets=a,b`` is a single combo (one job). To sweep across combos,
    use ``/`` as the alternatives separator: ``presets=a,b/c,d`` yields two
    jobs with values ``a,b`` and ``c,d``. ``/`` is shell-safe so no quoting
    is needed.
    """
    return [part.strip() for part in value.split("/") if part.strip()]


def _expand_combinations(entries: Sequence[tuple[str, str | None]]) -> list[list[tuple[str, str | None]]]:
    """Return the cartesian product over comma-separated values."""
    option_lists: list[list[tuple[str, str | None]]] = []
    for flag, value in entries:
        if value is None:
            option_lists.append([(flag, None)])
            continue
        if flag == "presets":
            values = _split_preset_values(value)
        else:
            values = [part.strip() for part in value.split(",") if part.strip()]
        if not values:
            raise ValueError(f"No value provided for flag {flag}.")
        option_lists.append([(flag, val) for val in values])
    combinations = list(itertools.product(*option_lists))
    return [list(combo) for combo in combinations]


def _short_key(key: str) -> str:
    """Shorten a parameter key for use in auto-generated run names.

    ``--seed`` becomes ``seed``, ``agent.algorithm.cfm_loss_mode`` becomes
    ``cfm_loss_mode`` (last dotted segment).
    """
    key = key.lstrip("-")
    if "." in key:
        key = key.rsplit(".", 1)[-1]
    return key


def _generate_run_names(
    combinations: list[list[tuple[str, str | None]]],
) -> list[str]:
    """Build a descriptive run name for each combination from the parameters that vary.

    Only parameters with more than one unique value across the sweep are included.
    Returns one name string per combination.
    """
    if not combinations:
        return []

    num_params = len(combinations[0])
    # Identify indices of parameters that vary across combinations
    varying_indices: list[int] = []
    for i in range(num_params):
        unique_values = {combo[i][1] for combo in combinations}
        if len(unique_values) > 1:
            varying_indices.append(i)

    names: list[str] = []
    for combo in combinations:
        parts: list[str] = []
        for i in varying_indices:
            key, value = combo[i]
            short = _short_key(key)
            parts.append(f"{short}-{value}")
        names.append("_".join(parts) if parts else "")
    return names


def _has_flag(entries: Sequence[tuple[str, str | None]], flag: str) -> str | None:
    """Return the value for *flag* if present in entries, else ``None``."""
    for key, value in entries:
        if key == flag:
            return value
    return None


def _build_command(cluster: str, profile: str | None, combo: Sequence[tuple[str, str | None]]) -> list[str]:
    cmd = ["./docker/cluster/cluster_interface.sh", "--cluster", cluster, "job"]
    if profile:
        cmd.append(profile)
    for key, value in combo:
        if key.startswith("--"):
            # Regular flag
            cmd.append(key)
            if value is not None:
                cmd.append(value)
        else:
            # Config override (dotted path) - pass as key=value
            cmd.append(f"{key}={value}")
    return cmd


def main(argv: Sequence[str]) -> int:
    args, remainder = _parse_cluster_args(argv)
    if not remainder:
        raise ValueError("No training arguments provided. Supply flags such as --task <TaskName>.")

    env_assignments, config_overrides, flag_tokens = _extract_env_and_config(remainder)
    flag_entries = _tokenize_flags(flag_tokens)
    # Combine flag entries with config overrides for expansion
    all_entries = flag_entries + config_overrides

    # cluster_sweep launches new sweeps and auto-uniquifies --experiment_name to
    # keep the sweep manifest in 1:1 correspondence with the log dir on the
    # cluster. Re-adds (--run_id pointing at an existing run) need the *original*
    # experiment_name preserved verbatim so train.py's auto-resume finds the
    # checkpoints. That's a different tool — refuse early with guidance.
    if _has_flag(all_entries, "--run_id") is not None:
        raise SystemExit(
            "error: --run_id is not valid with cluster_sweep (each sweep job gets a fresh SLURM id).\n"
            "To re-add or resume a failed run, use cluster_submit instead, which preserves\n"
            "--experiment_name verbatim and forwards --run_id to enable train.py auto-resume:\n"
            "    cluster_submit --run_id <id> --experiment_name <orig> --run_name <orig> ..."
        )

    combinations = _expand_combinations(all_entries)
    total_jobs = len(combinations)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build a unique experiment_name so sweep_collect.py can find the logs.
    # Format: "{user_prefix}_{sweep_id}" or just "{sweep_id}".
    # This overrides whatever the agent config sets, ensuring a 1:1 mapping
    # between the sweep manifest and the log directory on the cluster.
    user_experiment = _has_flag(all_entries, "--experiment_name")
    if user_experiment:
        experiment_name = f"{user_experiment}_{timestamp}"
        # Replace the user-provided value with the unique one in all combinations
        for combo in combinations:
            for j, (key, val) in enumerate(combo):
                if key == "--experiment_name":
                    combo[j] = ("--experiment_name", experiment_name)
                    break
    else:
        experiment_name = timestamp
        # Inject --experiment_name into every combination
        for combo in combinations:
            combo.append(("--experiment_name", experiment_name))
    print(f"[INFO] Experiment name: {experiment_name}")

    # Auto-generate --run_name for each job based on parameters that vary,
    # unless the user already provided --run_name in the CLI args.
    user_provided_run_name = _has_flag(all_entries, "--run_name") is not None
    run_names: list[str] = []
    if not user_provided_run_name and total_jobs > 1:
        run_names = _generate_run_names(combinations)
        if run_names and any(run_names):
            print(f"[INFO] Auto-generating --run_name for {total_jobs} jobs")
            for i, combo in enumerate(combinations):
                if run_names[i]:
                    combo.append(("--run_name", run_names[i]))

    # When using wandb, auto-set the run group so all sweep jobs are grouped together
    logger_value = _has_flag(all_entries, "--logger")
    if logger_value == "wandb":
        group_name = f"sweep_{timestamp}"
        env_assignments["WANDB_RUN_GROUP"] = group_name
        print(f"[INFO] Setting WANDB_RUN_GROUP={group_name}")

    manifest_dir = Path("logs") / "sweeps"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{timestamp}.json"

    print(f"[INFO] Sweep ID: {timestamp}")
    print(f"[INFO] Cluster: {args.cluster}")
    if args.profile:
        print(f"[INFO] Profile: {args.profile}")
    print(f"[INFO] Jobs to submit: {total_jobs}")
    if total_jobs > 1 and args.submission_delay > 0:
        print(f"[INFO] Submission delay: {args.submission_delay}s between jobs")

    job_records = []
    for idx, combo in enumerate(combinations, start=1):
        # Add delay between submissions (not before the first job)
        if idx > 1 and not args.dry_run and args.submission_delay > 0:
            print(f"[INFO] Waiting {args.submission_delay}s before next submission...")
            time.sleep(args.submission_delay)

        cmd_parts = _build_command(args.cluster, args.profile, combo)
        quoted_cmd = " ".join(shlex.quote(part) for part in cmd_parts)
        if env_assignments:
            prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_assignments.items())
            cmd_str = f"{prefix} {quoted_cmd}"
        else:
            cmd_str = quoted_cmd
        print(f"[{idx}/{total_jobs}] {cmd_str}")
        if not args.dry_run:
            run_env = os.environ.copy()
            run_env.update(env_assignments)
            subprocess.run(cmd_parts, check=True, env=run_env)
        job_records.append(
            {
                "index": idx,
                "command": cmd_str,
                "arguments": {flag: value for flag, value in combo},
            }
        )

    manifest = {
        "sweep_id": timestamp,
        "created_at": datetime.now().isoformat(),
        "cluster": args.cluster,
        "profile": args.profile,
        "experiment_name": experiment_name,
        "dry_run": args.dry_run,
        "submission_delay": args.submission_delay,
        "cluster_env": env_assignments,
        "num_jobs": total_jobs,
        "jobs": job_records,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] Sweep manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""
Usage examples
--------------
Single run (no sweep):

    python scripts/reinforcement_learning/rsl_rl/sweep.py \\
        --cluster hyak \\
        --task Isaac-Cartpole-v0 \\
        --num_envs 64

Sweep over flag values (comma-separated):

    python scripts/reinforcement_learning/rsl_rl/sweep.py \\
        --cluster hyak \\
        GPUS_PER_NODE=4 ACCOUNT=robotics \\
        --dry-run \\
        --task Isaac-Cartpole-v0 \\
        --num_envs 32,64 \\
        --seed 1,2

Sweep over config overrides (dotted paths):

    python scripts/reinforcement_learning/rsl_rl/sweep.py \\
        --task Isaac-Velocity-Rough-Anymal-C-v0 \\
        --num_envs 4096 \\
        --headless \\
        agent.max_iterations=500,1000,2000 \\
        --dry-run

Disable submission delay for faster testing:

    python scripts/reinforcement_learning/rsl_rl/sweep.py \\
        --submission-delay 0 \\
        --task Isaac-Cartpole-v0 \\
        --num_envs 32,64

Wandb logging with automatic run grouping:

    python scripts/reinforcement_learning/rsl_rl/sweep.py \\
        --cluster hyak \\
        --task Isaac-Cartpole-v0 \\
        --logger wandb \\
        --log_project_name cartpole-sweep \\
        --seed 1,2,3 \\
        env.scene.num_envs=2048,4096

Hydra presets (single combo, comma keeps Hydra list semantics):

    python scripts/reinforcement_learning/rsl_rl/sweep.py \\
        --task Isaac-Cartpole-v0 \\
        --seed 1,2,3 \\
        presets=physx,newton           # one combo applied to every job

Sweep across preset alternatives (slash separates combos, no quoting):

    python scripts/reinforcement_learning/rsl_rl/sweep.py \\
        --task Isaac-Cartpole-v0 \\
        --seed 1,2 \\
        presets=physx/physx,newton     # 2 alternatives x 2 seeds = 4 jobs

Notes:
- Boolean flags (like --headless) work correctly regardless of position
- Config overrides use dotted.path=value syntax and can be placed anywhere
- Shell env vars use VAR=value syntax (no dots in key)
- All comma-separated values create a cartesian product of jobs
- presets=... is the one exception: comma keeps Hydra "list of presets in one
  combo" semantics. Use ``/`` to sweep across combos (e.g. ``presets=a,b/c,d``)
- Default 3s delay between submissions reduces GPFS contention on cluster
- --experiment_name is auto-set to a unique ID (sweep timestamp) so sweep_collect.py can find logs
  Optionally prefix with --experiment_name <prefix> for readability (becomes <prefix>_<sweep_id>)
- When --logger wandb is used, WANDB_RUN_GROUP is auto-set to group all sweep runs
- Auto-generated --run_name based on varying parameters (override with explicit --run_name)
"""
