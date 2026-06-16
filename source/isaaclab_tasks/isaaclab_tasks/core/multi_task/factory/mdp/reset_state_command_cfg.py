# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkersCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from ...mdp.commands.state_command.state_command_cfg import StateCommandCfg
from ...utils.symmetry import AssetSymmetryCfg
from ..retarget.cfg import FactoryIKPipelineCfg

if TYPE_CHECKING:
    # type-only: the warp-heavy payload class parameterizes its ``class_type`` for
    # IDE autocomplete WITHOUT pulling the implementation at config import time --
    # the resolvable string is the runtime value, resolved lazily at instantiation.
    from .reset_state_command_payloads import FactoryAssemblyPayload


@configclass
class FactoryAssemblyAssetCommandCfg:
    """Command defined by the relative state between fixed and held assets."""

    orientation_threshold: float = 0.025
    """Allowed roll/pitch alignment error for success [rad]."""

    position_threshold: float = 0.0025
    """Allowed relative asset position error for success [m]."""

    duration: tuple[float, float] = (0.0, 1.0)
    """Required hold duration range for success [s]."""


@configclass
class FactoryResetStateTableCfg(StateCommandCfg.TaskTableCfg):
    """Reset-state table generation settings + builder selection."""

    class_type: Callable | str = "{DIR}.reset_state_task_table:build_factory_reset_state_task_table"
    """Builder that fills the :class:`~.reset_state_task_table.FactoryResetStateTaskTable`
    (resolvable string; the command calls ``class_type(cfg, env)``)."""

    pipeline_cfg: FactoryIKPipelineCfg = MISSING
    """Offline Newton-IK pipeline filling the table (see
    :class:`~..retarget.FactoryIKPipeline`). Tags are pipeline data; rows pass
    the settle gate below before being stored."""

    rows_per_board: int | PresetCfg = 8
    """Average reset-state rows kept PER board configuration; the total table size
    is derived as ``rows_per_board x pipeline_cfg.board.num_boards`` (the
    locomotion ``pool_spacing`` idea: declare density, the size emerges from the
    world library)."""

    targets_per_board: int = 4
    """Goal states selected PER board configuration, as a spatially-spread SUBSET
    of that board's stored rows (locomotion's ``num_targets_per_cell``: targets
    are existing states, so this must be <= :attr:`rows_per_board`). Every spawn
    in a board is paired with its board's full goal set."""

    state_table_fps_features: Callable | None = None
    """Feature extractor for state-table compaction and sampler layout."""

    settle_steps: int = 0
    """Physics settle steps before harvesting. Default ``0`` stores rows geometrically: a
    settle filter cannot be reproduced by the no-sim retarget/preview path, so it would make
    the stored table diverge from -- and not be validatable by -- that path. The geometric
    criteria (reachability, collision, aperture) are the acceptance test, and ``nut_bounds``
    below is the only other filter (geometric, reproducible offline). Set ``> 0`` to opt into
    a physics settle, accepting it is not offline-validatable."""

    settle_max_drift: float = 0.005
    """Reject a settled row whose held-asset position drifted more than this [m]."""

    finger_squeeze: float = 0.001
    """Grasped rows close the fingers this much past contact [m] so the position
    drive holds the asset with real clamp force during settle and after reset."""

    nut_bounds: dict[str, tuple[float, float]] | None = None
    """Optional per-axis ``(min, max)`` env-local bounds on the held-asset (nut)
    root position. Rows whose nut spawns outside are rejected at build so they
    cannot trigger the ``oob`` termination on the first step (a dead-on-arrival
    state that pollutes curriculum success). Keep in sync with the task's
    :func:`~...mdp.terminations.out_of_bound` ``in_bound_range``. ``None`` disables
    the filter (no change to the stored table)."""

    stash_viz_geometry: bool = False
    """Precompute the success-grid silhouettes at table build (see
    :mod:`~..viz.geometry`) and stash them on the table for the curriculum image
    logger. Off by default so the stored table is unchanged; turn on alongside the
    :func:`~..viz.sampler_images.log_factory_board_grid` ``sampler_visual_logger``."""


@configclass
class FactoryAssemblyPayloadCfg(StateCommandCfg.PayloadCfg):
    """Configuration for :class:`FactoryAssemblyPayload`."""

    class_type: type[FactoryAssemblyPayload] | str = "{DIR}.reset_state_command_payloads:FactoryAssemblyPayload"

    reset_assets: list[str] = MISSING
    """Scene asset names included in each stored reset-state row."""

    held_asset_cfg: SceneEntityCfg = MISSING
    """Held asset used for assembly progress computation."""

    fixed_asset_cfg: SceneEntityCfg = MISSING
    """Fixed asset used for assembly progress computation."""

    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    """Robot whose base frame expresses the command's position delta."""

    symmetry: AssetSymmetryCfg = AssetSymmetryCfg()
    """Held-asset symmetry, consumed by the :class:`~...utils.symmetry.Symmetry`.
    Default is no symmetry (identity). Use :class:`~...utils.symmetry.AssetSymmetryCfg`
    with explicit :class:`~...utils.symmetry.AxisSymmetryCfg` or
    :class:`~...utils.symmetry.SemanticSymmetryCfg` elements. Set per-variant via
    :class:`~..factory_presets.HeldAssetSymmetryCfg`."""

    held_asset_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/factory_reset_state/held_asset_target_frame"
    )
    """Debug marker for the held asset's target root-frame pose."""

    held_asset_visualizer_cfg.markers["frame"].scale = (0.025, 0.025, 0.025)
