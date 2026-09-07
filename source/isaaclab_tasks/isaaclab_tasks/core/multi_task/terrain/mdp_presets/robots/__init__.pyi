# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    "AsyncFootPairsCfg",
    "BaseBodyNameCfg",
    "ExperimentNameCfg",
    "FootBodyNamesCfg",
    "HeightScannerPrimPathCfg",
    "NonFootContactBodyNamesCfg",
    "RetargetJointRegularizeTargetsCfg",
    "RetargetFootRotationWeightCfg",
    "RetargetFootContactWeightCfg",
    "RetargetMorphPatchOversampleCfg",
    "RetargetTerrainCollisionMarginCfg",
    "RetargetLateralHipJointPatternCfg",
    "RetargetMinContactsCfg",
    "RetargetSnapDistanceCfg",
    "RobotArticulationCfg",
    "SyncFootPairsCfg",
]

from .robot_presets import (
    AsyncFootPairsCfg,
    BaseBodyNameCfg,
    ExperimentNameCfg,
    FootBodyNamesCfg,
    HeightScannerPrimPathCfg,
    NonFootContactBodyNamesCfg,
    RetargetJointRegularizeTargetsCfg,
    RetargetFootRotationWeightCfg,
    RetargetFootContactWeightCfg,
    RetargetMorphPatchOversampleCfg,
    RetargetLateralHipJointPatternCfg,
    RetargetMinContactsCfg,
    RetargetSnapDistanceCfg,
    RetargetTerrainCollisionMarginCfg,
    RobotArticulationCfg,
    SyncFootPairsCfg,
)

# Wildcard imports force eager module load so each robot's class-attribute
# registrations execute.  The ``__all__ = []`` inside each robot module keeps
# these from re-exporting any names.
from .anymal_c import *
from .anymal_d import *
from .b2 import *
from .g1 import *
from .go2 import *
from .h1 import *
from .mewtwo import *
from .spot import *
