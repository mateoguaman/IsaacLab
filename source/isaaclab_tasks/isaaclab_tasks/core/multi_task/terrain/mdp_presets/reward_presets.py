# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from .. import mdp
from .robots.robot_presets import NonFootContactBodyNamesCfg


@configclass
class PositionRewardsCfg:
    # task rewards
    success = RewTerm(func=mdp.command_success, weight=5.0)

    mech_work = RewTerm(func=mdp.mechanical_power, weight=-0.000025)

    # Penalize contact on anything that is not a foot. The body pattern is
    # per-robot: matching on ``FOOT`` names every body on a robot whose feet
    # are named otherwise, which turns this into a penalty for standing.
    undesired_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.01,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=NonFootContactBodyNamesCfg()),  # type: ignore[arg-type]
            "threshold": 1.0,
        },
    )


@configclass
class EmptyRewardsCfg:
    """No rewards. Used by self-supervised algorithms (e.g. CRL) where
    learning is driven by a contrastive loss, not a reward signal."""

    pass


@configclass
class RewardsCfg(PresetCfg):
    position = PositionRewardsCfg()
    crl = EmptyRewardsCfg()
    default = position
