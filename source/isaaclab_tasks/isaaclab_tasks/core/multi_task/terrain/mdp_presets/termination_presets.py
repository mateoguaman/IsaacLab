# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from ...mdp.terminations import BaseTerminationsCfg, joint_reaction_overload
from .. import mdp


@configclass
class PositionTerminationsCfg(BaseTerminationsCfg):
    """Locomotion termination cfg.

    Inherits ``time_out`` and ``abnormal`` from
    :class:`~isaaclab_tasks.core.multi_task.mdp.terminations.BaseTerminationsCfg`.
    Adds:

    - ``oob`` — fires when the robot's env-relative root drops below 20 m
      (replaces the older ``mdp.root_height_below_minimum``-based ``drop``,
      which compared against absolute world z and broke for terrains with
      non-zero spawn heights).
    - ``base_contact`` — fires on impact contact (force above 3× bodyweight)
      against any body. Static loading on any body (foot stance, kneeling,
      leaning) is bounded above by ~1× total BW by static equilibrium, so
      3× cleanly gates impact contacts regardless of which body is involved
      — knees, base, and feet alike. Per-robot configs can narrow
      ``sensor_cfg.body_names`` to exclude small appendages (e.g. fingers /
      toes / tail) where contact at any force is expected during normal
      motion. Reference: CaT (Chane-Sane et al., IROS 2024) uses ~2× total
      BW under stochastic termination + compliant impedance control on a
      light quadruped; 3× is the deterministic-termination, stiffer-actuator
      equivalent.
    - ``joint_reaction`` — fires when a joint's measured reaction force exceeds
      6× its effort limit (via the ``joint_wrench`` sensor); mechanical-overload
      guard carried over from the legacy position stack.
    - ``fallen`` — fires when the root sits under 0.1 m above the terrain below it.
      ``base_contact`` only catches impacts above 3× bodyweight, so a robot that
      collapses gently and lies still stays under that bar and runs out the whole
      episode. Height is measured against the height scanner rather than world z,
      because the terrains span tens of metres of elevation.
    - ``success`` — episode-success termination from the goal-tracking command.
    """

    oob = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "in_bound_range": {"x": (-1e6, 1e6), "y": (-1e6, 1e6), "z": (-20.0, 1e6)},
        },
    )

    base_contact = DoneTerm(
        func=mdp.illegal_contact_ratio,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"),
            "threshold_ratio": 3.0,
        },
    )

    joint_reaction = DoneTerm(
        func=joint_reaction_overload,
        params={
            "sensor_cfg": SceneEntityCfg("joint_wrench"),
            "force_ratio": 6.0,
        },
    )

    # time_out is left False: a fall is a real failure, so the value function should not
    # bootstrap off it. No paired reward term either — the cost is the forgone success.
    fallen = DoneTerm(
        func=mdp.root_height_above_terrain_below_minimum,
        params={
            "minimum_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )

    success = DoneTerm(func=mdp.success_terminate)


@configclass
class TerminationsCfg(PresetCfg):
    position = PositionTerminationsCfg()
    default = position
