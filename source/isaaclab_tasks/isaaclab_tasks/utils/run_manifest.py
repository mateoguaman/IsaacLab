# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run manifest writer used by training entry points.

A run manifest is a small JSON file co-located with the training artifacts that
records what was trained, by whom, on what code, and where. It powers the
local ``play <jobid>`` workflow described in ``EVAL_WORKFLOW.md`` §3.1.

The presence of ``SLURM_JOB_ID`` discriminates between cluster and local runs:

* Cluster: merge ``$ISAACLAB_PATH/.manifest.bootstrap.json`` (written at submit
  time by ``docker/cluster/cluster_interface.sh``) with run-time fields.
* Local: collect git + ``user@host`` directly.
"""

from __future__ import annotations

import getpass
import importlib.metadata as metadata
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone

import rsl_rl

__all__ = ["write_run_manifest"]


def _collect_rsl_rl_metadata() -> dict:
    """Identify which rsl_rl source the run used (PyPI vs editable fork)."""
    repo_dir = _rsl_rl_editable_repo_dir()
    if repo_dir is None:
        return {"rsl_rl_source": "pypi", "rsl_rl_version": metadata.version("rsl-rl-lib")}
    info: dict = {"rsl_rl_source": "editable", "rsl_rl_version": metadata.version("rsl-rl-lib")}
    info.update(_git_introspection(repo_dir))
    return info


def _rsl_rl_editable_repo_dir() -> str | None:
    """Return the editable rsl_rl repo root, or ``None`` if installed from PyPI.

    Detects editable installs by location: PyPI/wheel installs land inside a
    ``site-packages`` directory, editable installs do not. This avoids relying
    on a ``.git`` directory next to the source, which is rsync-excluded on the
    cluster.
    """
    pkg_file = getattr(rsl_rl, "__file__", None)
    if pkg_file is not None:
        pkg_dir = os.path.dirname(pkg_file)
    else:
        # Namespace-package layout (uv editable install): derive from a submodule.
        try:
            import rsl_rl.utils as _probe
        except ImportError:
            return None
        pkg_dir = os.path.dirname(os.path.dirname(_probe.__file__))
    parent = os.path.dirname(pkg_dir)
    if "site-packages" in os.path.normpath(parent).split(os.sep):
        return None
    return parent


def _git_introspection(repo_dir: str) -> dict:
    """Capture remote URL, branch, SHA, and dirty status for an rsl_rl editable repo."""

    def _git(*args: str) -> str:
        try:
            return subprocess.check_output(["git", "-C", repo_dir, *args], stderr=subprocess.DEVNULL).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    info: dict = {}
    if remote := _git("config", "--get", "remote.origin.url"):
        info["rsl_rl_git_remote"] = remote
    if (branch := _git("rev-parse", "--abbrev-ref", "HEAD")) and branch != "HEAD":
        info["rsl_rl_git_branch"] = branch
    if sha := _git("rev-parse", "HEAD"):
        info["rsl_rl_git_sha"] = sha
    info["rsl_rl_git_dirty"] = bool(_git("status", "--porcelain"))
    return info


def _collect_local_metadata(isaaclab_path: str, train_args: list[str]) -> dict:
    """Return git + ``user@host`` metadata for a local (non-SLURM) run."""

    def _git(*args: str) -> str:
        try:
            return (
                subprocess.check_output(["git", "-C", isaaclab_path, *args], stderr=subprocess.DEVNULL).decode().strip()
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "submitted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "submitted_by": f"{getpass.getuser()}@{socket.gethostname()}",
        "train_args": train_args,
    }


def write_run_manifest(
    log_dir: str,
    task: str,
    experiment: str,
    *,
    train_args: list[str] | None = None,
) -> None:
    """Write ``<log_dir>/manifest.json`` describing this training run.

    Args:
        log_dir: Absolute path to the run's log directory (where checkpoints land).
        task: Gymnasium task id, e.g. ``"Isaac-Velocity-Position-Anymal-C-v0"``.
        experiment: ``agent_cfg.experiment_name`` — the rsl_rl experiment slug
            that determines the parent directory of ``log_dir``.
        train_args: Original training CLI args (``sys.argv[1:]`` captured before
            any in-process rewrite). Defaults to ``sys.argv[1:]``; ``train.py``
            passes a pre-Hydra snapshot because it clobbers ``sys.argv`` for
            Hydra integration before this function runs. Ignored on the SLURM
            path — the cluster bootstrap manifest already carries train_args
            from submit time.

    No-op when ``ISAACLAB_PATH`` is unset (i.e. running raw ``python train.py``
    without ``isaaclab.sh``). Callers should rank-0-gate this themselves.
    """
    isaaclab_path = os.environ.get("ISAACLAB_PATH")
    if not isaaclab_path:
        return
    if train_args is None:
        train_args = sys.argv[1:]

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        bootstrap_path = os.path.join(isaaclab_path, ".manifest.bootstrap.json")
        if os.path.exists(bootstrap_path):
            with open(bootstrap_path) as f:
                manifest = json.load(f)
            artifacts_dir = os.path.join(
                manifest["persistent_dir"], "logs", "rsl_rl", experiment, os.path.basename(log_dir)
            )
        else:
            print(
                f"[WARN] SLURM_JOB_ID={slurm_job_id} but {bootstrap_path} is missing;"
                " manifest will lack cluster context. Check docker/cluster/cluster_interface.sh."
            )
            manifest = _collect_local_metadata(isaaclab_path, train_args)
            artifacts_dir = log_dir
        jobid = slurm_job_id
    else:
        manifest = _collect_local_metadata(isaaclab_path, train_args)
        jobid = os.path.basename(log_dir)
        artifacts_dir = log_dir

    manifest.update(
        {
            "jobid": jobid,
            "task": task,
            "experiment": experiment,
            "artifacts_dir": artifacts_dir,
        }
    )
    manifest.update(_collect_rsl_rl_metadata())

    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, "manifest.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"[INFO] Wrote run manifest to {out_path}")
