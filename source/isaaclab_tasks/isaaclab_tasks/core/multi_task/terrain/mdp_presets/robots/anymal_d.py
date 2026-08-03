# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Anymal-D robot preset with PACE actuators. Activate with ``presets=anymal_d_pace``."""

from __future__ import annotations

__all__: list[str] = []

from isaaclab.assets import ArticulationCfg

import isaaclab_assets.robots.anymal as anymal

from ...retarget.criteria import BaseZError, FootPositionError, JointMargin
from .robot_presets import (
    AsyncFootPairsCfg,
    BaseBodyNameCfg,
    ExperimentNameCfg,
    FootBodyNamesCfg,
    HeightScannerPrimPathCfg,
    NonFootContactBodyNamesCfg,
    RetargetJointRegularizeTargetsCfg,
    RetargetLateralHipJointPatternCfg,
    RobotArticulationCfg,
    SyncFootPairsCfg,
)

# ANYmal-D with per-joint PACE actuators. PACE is an analytic Lab-side actuator (encoder bias +
# torque delay), so this preset targets the default/PhysX backend; ``anymal_d_pace,newton_mjwarp``
# is unsupported (Newton would treat the config as a plain DC motor and drop the bias/delay).
# The USD streams from the public ``ISAACLAB_NUCLEUS_DIR`` bucket baked into ``ANYMAL_D_PACE_CFG``.
_ANYMAL_D_PACE_CFG: ArticulationCfg = anymal.ANYMAL_D_PACE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

RobotArticulationCfg.anymal_d_pace = _ANYMAL_D_PACE_CFG
HeightScannerPrimPathCfg.anymal_d_pace = "{ENV_REGEX_NS}/Robot/base"
BaseBodyNameCfg.anymal_d_pace = "base"
NonFootContactBodyNamesCfg.anymal_d_pace = "^(?!.*FOOT).*$"
FootBodyNamesCfg.anymal_d_pace = ".*FOOT.*"
AsyncFootPairsCfg.anymal_d_pace = (
    ("LF_FOOT", "RF_FOOT"),
    ("RH_FOOT", "LH_FOOT"),
    ("LF_FOOT", "LH_FOOT"),
    ("RF_FOOT", "RH_FOOT"),
)
SyncFootPairsCfg.anymal_d_pace = (("LF_FOOT", "RH_FOOT"), ("RF_FOOT", "LH_FOOT"))
ExperimentNameCfg.anymal_d_pace = "anymal_d_pace_position_command"


# ---------------------------------------------------------------------------
# Retarget validation criteria for ANYmal-D (shares ANYmal-C joint naming)
# ---------------------------------------------------------------------------

ANYMAL_D_LATERAL_HIP_PATTERN = ".*HAA"
"""Regex matching ANYmal-D lateral hip joint names."""

RetargetLateralHipJointPatternCfg.anymal_d_pace = ANYMAL_D_LATERAL_HIP_PATTERN
# Pull lateral hips toward 0 and knees toward their init-pose flexion (front tuck forward,
# hind tuck back); hip flexion/extension stays free for IK.
RetargetJointRegularizeTargetsCfg.anymal_d_pace = {
    ANYMAL_D_LATERAL_HIP_PATTERN: 0.0,
    ".*F_KFE": -0.8,
    ".*H_KFE": 0.8,
}

# Re-export generic criteria used by existing terrain tuning scripts.
__all__ += ["FootPositionError", "JointMargin", "BaseZError"]
