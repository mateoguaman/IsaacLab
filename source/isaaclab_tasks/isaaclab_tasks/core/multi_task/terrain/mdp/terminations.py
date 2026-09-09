# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination predicates for the position locomotion MDP."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

    from ...mdp.commands.state_command.state_command import StateCommand


def success_terminate(env: ManagerBasedRLEnv, command_name: str = "goal_point") -> torch.Tensor:
    """Episode-success termination: fires when the goal-tracking command reports done."""
    command_term: StateCommand = env.command_manager.get_term(command_name)
    return command_term.get_task_done()


def command_task_done(env: ManagerBasedRLEnv, command_name: str = "goal_point") -> torch.Tensor:
    """Expose :attr:`MultiTaskCommand.task_done` as a :class:`DoneTerm` predicate.

    Fires when the command term reports success — all active-instant subtasks for the
    env's assigned task have been achieved. Bind with ``time_out=False`` so rsl_rl
    does not bootstrap on top of the terminal multiplicative reward.
    """
    return env.command_manager.get_term(command_name).task_done


def time_out_reach_truncate(env: ManagerBasedRLEnv, command_name: str = "goal_point") -> torch.Tensor:
    """Timeout predicate for envs whose current task contains ≥1 instant subtask.

    Fires when ``episode_length_buf >= max_episode_length`` AND the env's task
    has an instant subtask (pure-reach or mixed). Bind with ``time_out=True``
    so rsl_rl treats this as a truncation and bootstraps ``γ·V(s_T)`` onto
    the last reward — the reach was incomplete only because the artificial
    episode cap ran out, and value should propagate through partial progress.

    Paired with :func:`time_out_track_terminate` (which covers pure-tracking
    envs with ``time_out=False``). Together they replace the single
    ``mdp.time_out`` DoneTerm.

    Reach/mixed envs always use the env's global ``max_episode_length`` — the
    adaptive curriculum only affects pure-tracking envs.
    """
    cmd = env.command_manager.get_term(command_name)
    timeout = env.episode_length_buf >= env.max_episode_length
    return timeout & cmd.spec.task_has_instant[cmd.task_samples]


def time_out_track_terminate(env: ManagerBasedRLEnv, command_name: str = "goal_point") -> torch.Tensor:
    """Timeout predicate for envs whose current task has NO instant subtask.

    Fires when ``episode_length_buf >= effective_max_episode_length`` AND the
    env's task is pure-tracking. Bind with ``time_out=False`` so rsl_rl treats
    this as a real termination — the episode cap is the task's natural
    endpoint, and the composer's ``G = transit_mean`` is the complete
    episodic return. A bootstrap would double-count.

    The effective cap is per-env — under the adaptive episode-length
    curriculum (see :attr:`MultiTaskCfg.tracking_adaptive_err_threshold`) it
    shortens when tracking error is high and lengthens when error is low.
    When the curriculum is disabled the cap is just ``max_episode_length``.
    """
    cmd = env.command_manager.get_term(command_name)
    timeout = env.episode_length_buf >= cmd.effective_max_episode_length
    return timeout & ~cmd.spec.task_has_instant[cmd.task_samples]


def root_height_above_terrain_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """Terminate when the root sits closer than ``minimum_height`` [m] to the terrain below it.

    The terrains span tens of metres of elevation, so an absolute world-z threshold cannot
    separate a fallen robot from one standing on a low tile. Local ground level comes from the
    height scanner's ray hit nearest the root in XY — the patch spans metres and would
    otherwise average in a step ahead or a pit rim beside the robot. Rays that miss (gaps,
    pits, bottomless tiles) return non-finite hits and are excluded, so a robot at the edge of
    a hole is measured against ground it can actually see. Envs whose rays all miss never fire.

    Args:
        env: The environment instance.
        minimum_height: Root clearance above the local terrain below which the episode ends [m].
        asset_cfg: The articulation whose root height is checked.
        sensor_cfg: The ray-caster sensor supplying local terrain height.

    Returns:
        Boolean tensor, shape ``[num_envs]``, true where the root is too low.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_cfg.name]

    root_pos = asset.data.root_pos_w.torch
    hits = sensor.data.ray_hits_w.torch
    finite = torch.isfinite(hits).all(dim=-1)

    # Nearest hit in XY; missed rays are pushed to +inf so they lose the argmin.
    xy_dist = torch.linalg.norm(hits[..., :2] - root_pos[:, None, :2], dim=-1)
    xy_dist = torch.where(finite, xy_dist, torch.inf)
    nearest = xy_dist.argmin(dim=1)
    ground_z = hits[torch.arange(hits.shape[0], device=hits.device), nearest, 2]

    # No hit anywhere: fall back to the root height so the clearance is zero-free and large.
    ground_z = torch.where(finite.any(dim=1), ground_z, root_pos[:, 2])
    return (root_pos[:, 2] - ground_z) < minimum_height
