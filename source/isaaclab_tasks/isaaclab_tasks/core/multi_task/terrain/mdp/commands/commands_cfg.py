# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locomotion building blocks for the shared :class:`~...mdp.commands.StateCommand`.

The env wires up a :class:`~...mdp.commands.StateCommandCfg` directly; this module
only supplies the locomotion ``task_table`` cfg, the base/foot payload cfgs (which
own their debug visualizers), and the command-variant cfgs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.markers import BLUE_ARROW_X_MARKER_CFG, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from ....mdp.commands.state_command.state_command_cfg import StateCommandCfg
from .state_command_payloads import CommandPayloadBaseFootState, CommandPayloadBaseState
from .task_table_builder import build_relative_state_task_table

if TYPE_CHECKING:
    from ...retarget.cfg import RetargetPipelineCfg


# Default debug-vis marker cfgs shared by the base/foot payloads (configclass
# deep-copies these per instance).
_GOAL_VISUALIZER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/goal_state",
    markers={
        "vel_arrow": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(0.5, 0.5, 0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        "cuboid": sim_utils.CuboidCfg(
            size=(0.25, 0.25, 0.25),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "pose_arrow": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(0.5, 0.5, 0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "sphere": sim_utils.SphereCfg(
            radius=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.0)),
        ),
    },
)

_CURRENT_VEL_VISUALIZER_CFG = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_current")
_CURRENT_VEL_VISUALIZER_CFG.markers["arrow"].scale = (0.5, 0.5, 0.5)


@configclass
class TaskTableCfg(StateCommandCfg.TaskTableCfg):
    """Task-table builder configuration for the locomotion command."""

    class_type: Callable = build_relative_state_task_table
    """Callable that builds the task table."""

    pipeline_cfg: RetargetPipelineCfg = MISSING  # type: ignore[assignment]
    """Retarget pipeline configuration for generating IK-solved spawn states."""

    pool_spacing: float = 1.0
    """Target spacing between final IK-solved terrain states [m].

    The table builder derives pool size from sampling area, so sample density
    scales with the full terrain grid or :attr:`pool_sampling_size`.
    """

    pool_spacing_area_divisor: float = 3.0
    """Area divisor used to derive spacing-mode pool size."""

    pool_sampling_size: tuple[float, float] | None = None
    """Optional centered terrain-state sampling window size ``(x, y)`` [m].

    ``None`` samples over the full terrain grid. When set, the table builder
    still bins states against the full terrain grid, but the IK retarget
    sampler is clipped to this centered window before pool sizing and sampling.
    """

    exclude_self_pairs: bool = True
    """Whether to drop ``(spawn == target)`` pairs from the per-cell Cartesian product.

    When ``True``, the per-cell ``n × n`` pair grid is reduced to ``n × (n - 1)``
    by removing the diagonal. Cells with fewer than two valid states then
    contribute no pairs.
    """

    max_spawns_per_cell: int = 0
    """Optional cap on per-cell spawn states for the spawn × target pairing.

    ``0`` keeps every IK-solved state in the cell as a possible spawn.
    A positive integer ``N`` first picks ``min(N, n_c)`` spawn states via
    :func:`~isaaclab_tasks.core.multi_task.utils.grid_downsample.grid_bucket_downsample`.
    Target states are then selected from the remaining non-spawn states
    when any remain, falling back to the full cell state pool otherwise.
    """

    num_targets_per_cell: int = 0
    """Optional cap on per-cell target states for the spawn × target pairing.

    ``0`` keeps the full per-cell Cartesian product. A positive ``N`` first
    picks ``min(N, n_c)`` targets per cell, then pairs every spawn with each
    picked target.
    """


@configclass
class BaseStatePayloadCfg(StateCommandCfg.PayloadCfg):
    """Payload that commands base state only."""

    class_type: type = CommandPayloadBaseState
    """Payload worker class."""

    pos_std: float = 0.5
    """Default base-position success threshold [m]. Per-task overridable via command rows."""

    rot_std: float = 0.5
    """Default base-orientation success threshold [rad]. Per-task overridable via command rows."""

    lin_vel_std: float = 0.5
    """Default linear-velocity success threshold [m/s]. Per-task overridable via command rows."""

    ang_vel_std: float = 0.5
    """Default angular-velocity success threshold [rad/s]. Per-task overridable via command rows."""

    normalize_command_obs: bool = False
    """Whether to divide command channels by the per-task success threshold."""

    success_joint_dev_eps: float = 0.0
    """Extra success gate: every joint named by :attr:`success_joint_dev_joint_names` must sit
    within this angle of its default position [rad]. Applied on top of the pose thresholds, so
    a robot that reaches the goal in a contorted pose does not count as successful. Defaults to
    0.0, which disables the gate."""

    success_joint_dev_include: str = ""
    """Comma-separated substrings selecting the joints the gate applies to, e.g.
    ``"waist,shoulder,elbow,wrist"``. Empty means every joint.

    Deliberately not a regex list: brackets and quotes do not survive being spliced into the
    SLURM job script, which silently degrades a list into its individual characters. Leg
    joints also deviate by design — the retarget IK rotates hips to place feet on stones and
    treads — so gating on them forbids terrain adaptation rather than bad posture."""

    track_body_name: str = ""
    """Body whose pose is driven to the goal, instead of the articulation root. Empty string
    keeps the root. The body must survive fixed-joint merging during URDF conversion."""

    track_body_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Offset from :attr:`track_body_name`'s origin to the tracked point, in that body's
    frame [m]. Lets a frame that was merged away be addressed via its surviving parent."""

    goal_visualizer_cfg: VisualizationMarkersCfg = _GOAL_VISUALIZER_CFG
    """Debug marker for the goal state (pos/pose/vel)."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = _CURRENT_VEL_VISUALIZER_CFG
    """Debug marker for the current base velocity."""


@configclass
class BaseFootStatePayloadCfg(StateCommandCfg.PayloadCfg):
    """Payload that commands base state and terrain target foot positions."""

    class_type: type = CommandPayloadBaseFootState
    """Payload worker class."""

    pos_std: float = 0.5
    """Default base-position success threshold [m]. Per-task overridable via command rows."""

    rot_std: float = 0.5
    """Default base-orientation success threshold [rad]. Per-task overridable via command rows."""

    lin_vel_std: float = 0.5
    """Default linear-velocity success threshold [m/s]. Per-task overridable via command rows."""

    ang_vel_std: float = 0.5
    """Default angular-velocity success threshold [rad/s]. Per-task overridable via command rows."""

    foot_pos_std: float = 0.1
    """Default per-foot target-error threshold [m]."""

    normalize_command_obs: bool = False
    """Whether to divide command channels by the per-task success threshold."""

    goal_visualizer_cfg: VisualizationMarkersCfg = _GOAL_VISUALIZER_CFG
    """Debug marker for the goal state (pos/pose/vel/foot)."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = _CURRENT_VEL_VISUALIZER_CFG
    """Debug marker for the current base velocity."""


@configclass
class Commands:
    pos_x: tuple[float, float] | None = None
    """Range for the x position (in m)."""

    pos_y: tuple[float, float] | None = None
    """Range for the y position (in m)."""

    pos_z: tuple[float, float] | None = None
    """Range for the y position (in m)."""

    roll: tuple[float, float] | None = None
    """Range for the base roll orientation (in radian)."""

    pitch: tuple[float, float] | None = None
    """Range for the base pitch orientation (in radian)."""

    yaw: tuple[float, float] | None = None
    """Range for the base yaw orientation (in radian)."""

    lin_vel_x: tuple[float, float] | None = None
    """Range for the linear-x velocity command (in m/s)."""

    lin_vel_y: tuple[float, float] | None = None
    """Range for the linear-y velocity command (in m/s)."""

    lin_vel_z: tuple[float, float] | None = None
    """Range for the linear-z velocity command (in m/s)."""

    ang_vel_x: tuple[float, float] | None = None
    """Range for the angular-x velocity command (in rad/s)."""

    ang_vel_y: tuple[float, float] | None = None
    """Range for the angular-y velocity command (in rad/s)."""

    ang_vel_z: tuple[float, float] | None = None
    """Range for the angular-z velocity command (in rad/s)."""

    duration: tuple[float, float] = (1.0, 1.0)
    """time required to be considered as success."""

    pos_std: float | None = None
    """Per-task override for the base-position success threshold [m]. ``None``
    falls back to the active payload's default."""

    rot_std: float | None = None
    """Per-task override for the base-orientation success threshold [rad]."""

    lin_vel_std: float | None = None
    """Per-task override for the linear-velocity success threshold [m/s]."""

    ang_vel_std: float | None = None
    """Per-task override for the angular-velocity success threshold [rad/s]."""


@configclass
class PositionCommands(Commands):
    """Uniform distribution ranges for the position commands."""

    pos_x: tuple[float, float] | None = MISSING
    """Range for the x position (in m)."""

    pos_y: tuple[float, float] | None = MISSING
    """Range for the y position (in m)."""

    pos_z: tuple[float, float] | None = MISSING
    """Range for the y position (in m)."""


@configclass
class PoseCommands(Commands):
    pos_x: tuple[float, float] | None = MISSING
    """Range for the x position (in m)."""

    pos_y: tuple[float, float] | None = MISSING
    """Range for the y position (in m)."""

    pos_z: tuple[float, float] | None = MISSING
    """Range for the y position (in m)."""

    roll: tuple[float, float] | None = MISSING
    """Range for the base roll orientation (in radian)."""

    pitch: tuple[float, float] | None = MISSING
    """Range for the base pitch orientation (in radian)."""

    yaw: tuple[float, float] | None = MISSING
    """Range for the base yaw orientation (in radian)."""


@configclass
class VelocityCommands(Commands):
    lin_vel_x: tuple[float, float] | None = MISSING
    """Range for the linear-x velocity command (in m/s)."""

    lin_vel_y: tuple[float, float] | None = MISSING
    """Range for the linear-y velocity command (in m/s)."""

    lin_vel_z: tuple[float, float] | None = MISSING
    """Range for the linear-z velocity command (in m/s)."""

    ang_vel_x: tuple[float, float] | None = MISSING
    """Range for the angular-x velocity command (in rad/s)."""

    ang_vel_y: tuple[float, float] | None = MISSING
    """Range for the angular-y velocity command (in rad/s)."""

    ang_vel_z: tuple[float, float] | None = MISSING
    """Range for the angular-z velocity command (in rad/s)."""


@configclass
class TerrainCommands:
    """Terrain-state command backed by sampled spawn/target states."""

    target_key: str = "target"
    """State-buffer key used for command target states."""

    match_base_pos: bool = True
    """Whether to command the target state's base position [m]."""

    match_base_rot: bool = False
    """Whether to command the target state's base orientation [rad]."""

    duration: tuple[float, float] = (1.0, 1.0)
    """Time required to be considered successful [s]."""

    pos_std: float | None = None
    """Per-task override for the base-position success threshold [m]. ``None``
    falls back to the active payload's default."""

    rot_std: float | None = None
    """Per-task override for the base-orientation success threshold [rad]."""

    lin_vel_std: float | None = None
    """Per-task override for the linear-velocity success threshold [m/s]."""

    ang_vel_std: float | None = None
    """Per-task override for the angular-velocity success threshold [rad/s]."""
