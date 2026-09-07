# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree G1 (29 DoF) humanoid preset. Activate with ``presets=g1``.

The articulation follows the BeyondMimic / ``whole_body_tracking`` G1 config
(github.com/HybridRobotics/whole_body_tracking), which derives every PD gain
from the physical motor armature and one target closed-loop response rather
than hand-tuning per joint. See :data:`NATURAL_FREQUENCY_HZ`.

The asset is the URDF that project distributes, which carries exactly the 29
revolute joints of the hands-free G1. Fetch it before selecting this preset::

    ./isaaclab.sh -p scripts/fetch_robot_descriptions.py

Nucleus alternatives do not fit: ``Isaac/Robots/Unitree/G1/g1.usd`` is the
43-DoF Dex-3-hand variant, and ``IsaacLab/Robots/Unitree/G1/g1_minimal.usd``
uses the older 23-DoF naming (``torso_joint``, ``*_elbow_pitch_joint``).
"""

from __future__ import annotations

__all__: list[str] = []

import math
from pathlib import Path

from isaaclab_physx.sim.schemas import PhysxArticulationRootPropertiesCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

from .robot_presets import (
    AsyncFootPairsCfg,
    BaseBodyNameCfg,
    ExperimentNameCfg,
    FootBodyNamesCfg,
    HeightScannerPrimPathCfg,
    NonFootContactBodyNamesCfg,
    RetargetJointRegularizeTargetsCfg,
    RetargetFootContactWeightCfg,
    RetargetFootRotationWeightCfg,
    RetargetLateralHipJointPatternCfg,
    RetargetMinContactsCfg,
    RetargetSnapDistanceCfg,
    RetargetMorphPatchOversampleCfg,
    RetargetTerrainCollisionMarginCfg,
    RobotArticulationCfg,
    SyncFootPairsCfg,
)

G1_URDF_PATH = str(Path(ISAACLAB_ASSETS_DATA_DIR) / "unitree_description" / "urdf" / "g1" / "main.urdf")
"""Hands-free 29-DoF G1 description, populated by ``scripts/fetch_robot_descriptions.py``."""


# ---------------------------------------------------------------------------
# Actuator gains
# ---------------------------------------------------------------------------
# Each joint's PD gains come from its motor's rotor armature and a single
# target second-order response, so a joint is described by which motor drives
# it rather than by two tuned numbers:
#
#     stiffness = armature * omega_n^2
#     damping   = 2 * zeta * armature * omega_n
#
# Unitree ships four motor classes across the G1; ``7520_14`` and ``7520_22``
# are the two gear ratios of the 7520. Ankle and waist-roll/pitch joints are
# each driven by a pair of 5020s through a differential, so they take twice the
# single-motor armature and gains.

NATURAL_FREQUENCY_HZ = 10.0
"""Target closed-loop natural frequency of every joint [Hz]."""

DAMPING_RATIO = 2.0
"""Target closed-loop damping ratio [unitless]. Above 1 keeps the joint overdamped."""

_OMEGA = 2.0 * math.pi * NATURAL_FREQUENCY_HZ

ARMATURE_5020 = 0.003609725
"""Rotor armature of the 5020 motor [kg*m^2]."""
ARMATURE_7520_14 = 0.010177520
"""Rotor armature of the 7520 motor at the 14:1 ratio [kg*m^2]."""
ARMATURE_7520_22 = 0.025101925
"""Rotor armature of the 7520 motor at the 22:1 ratio [kg*m^2]."""
ARMATURE_4010 = 0.00425
"""Rotor armature of the 4010 motor [kg*m^2]."""


def _stiffness(armature: float) -> float:
    """Return the PD stiffness [N*m/rad] realizing :data:`NATURAL_FREQUENCY_HZ` for ``armature``."""
    return armature * _OMEGA**2


def _damping(armature: float) -> float:
    """Return the PD damping [N*m*s/rad] realizing :data:`DAMPING_RATIO` for ``armature``."""
    return 2.0 * DAMPING_RATIO * armature * _OMEGA


_G1_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=G1_URDF_PATH,
        fix_base=False,
        # The URDF models each shin and forearm as a cylinder; capsules give the
        # same silhouette without the flat end caps that snag on terrain edges.
        replace_cylinders_with_capsules=True,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=PhysxArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        # Zero the gains baked into the URDF drives; the actuator terms below own them.
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.76),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_yaw_joint": _stiffness(ARMATURE_7520_14),
                ".*_hip_roll_joint": _stiffness(ARMATURE_7520_22),
                ".*_hip_pitch_joint": _stiffness(ARMATURE_7520_14),
                ".*_knee_joint": _stiffness(ARMATURE_7520_22),
            },
            damping={
                ".*_hip_yaw_joint": _damping(ARMATURE_7520_14),
                ".*_hip_roll_joint": _damping(ARMATURE_7520_22),
                ".*_hip_pitch_joint": _damping(ARMATURE_7520_14),
                ".*_knee_joint": _damping(ARMATURE_7520_22),
            },
            armature={
                ".*_hip_yaw_joint": ARMATURE_7520_14,
                ".*_hip_roll_joint": ARMATURE_7520_22,
                ".*_hip_pitch_joint": ARMATURE_7520_14,
                ".*_knee_joint": ARMATURE_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=2.0 * _stiffness(ARMATURE_5020),
            damping=2.0 * _damping(ARMATURE_5020),
            armature=2.0 * ARMATURE_5020,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
            stiffness=_stiffness(ARMATURE_7520_14),
            damping=_damping(ARMATURE_7520_14),
            armature=ARMATURE_7520_14,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=2.0 * _stiffness(ARMATURE_5020),
            damping=2.0 * _damping(ARMATURE_5020),
            armature=2.0 * ARMATURE_5020,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": _stiffness(ARMATURE_5020),
                ".*_shoulder_roll_joint": _stiffness(ARMATURE_5020),
                ".*_shoulder_yaw_joint": _stiffness(ARMATURE_5020),
                ".*_elbow_joint": _stiffness(ARMATURE_5020),
                ".*_wrist_roll_joint": _stiffness(ARMATURE_5020),
                ".*_wrist_pitch_joint": _stiffness(ARMATURE_4010),
                ".*_wrist_yaw_joint": _stiffness(ARMATURE_4010),
            },
            damping={
                ".*_shoulder_pitch_joint": _damping(ARMATURE_5020),
                ".*_shoulder_roll_joint": _damping(ARMATURE_5020),
                ".*_shoulder_yaw_joint": _damping(ARMATURE_5020),
                ".*_elbow_joint": _damping(ARMATURE_5020),
                ".*_wrist_roll_joint": _damping(ARMATURE_5020),
                ".*_wrist_pitch_joint": _damping(ARMATURE_4010),
                ".*_wrist_yaw_joint": _damping(ARMATURE_4010),
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_5020,
                ".*_shoulder_roll_joint": ARMATURE_5020,
                ".*_shoulder_yaw_joint": ARMATURE_5020,
                ".*_elbow_joint": ARMATURE_5020,
                ".*_wrist_roll_joint": ARMATURE_5020,
                ".*_wrist_pitch_joint": ARMATURE_4010,
                ".*_wrist_yaw_joint": ARMATURE_4010,
            },
        ),
    },
)


RobotArticulationCfg.g1 = _G1_CFG
HeightScannerPrimPathCfg.g1 = "{ENV_REGEX_NS}/Robot/pelvis"
# The pelvis is the articulation root and sits near the CoM, so it is what the
# stability criteria use as their CoM proxy. ``torso_link`` moves with the waist
# joints and would drift from the CoM whenever the torso leans.
BaseBodyNameCfg.g1 = "pelvis"
FootBodyNamesCfg.g1 = ".*_ankle_roll_link"
NonFootContactBodyNamesCfg.g1 = "^(?!.*ankle_roll_link).*$"
AsyncFootPairsCfg.g1 = (("left_ankle_roll_link", "right_ankle_roll_link"),)
SyncFootPairsCfg.g1 = ()
ExperimentNameCfg.g1 = "g1_position_command"


# ---------------------------------------------------------------------------
# Retarget validation criteria for Unitree G1
# ---------------------------------------------------------------------------

G1_LATERAL_HIP_PATTERN = ".*_hip_roll_joint"
"""Regex matching G1 lateral hip joint names."""

RetargetLateralHipJointPatternCfg.g1 = G1_LATERAL_HIP_PATTERN

# Standing, the shin reaches to 19 mm above the sole plane and the ankle-pitch
# link to 28 mm, so the quadruped default of 50 mm would push both away from the
# terrain and carry the foot up with them. 10 mm clears the lowest of them.
RetargetTerrainCollisionMarginCfg.g1 = 0.01

# Two feet draw half the patches a quadruped does from the same per-slot
# budget, leaving each foot a coarse grid to snap onto. Restoring the density
# lifts the share of placements that get both soles onto the ground from 86%
# to 97%.
RetargetMorphPatchOversampleCfg.g1 = 16.0

# G1's sole is a 21 x 7 cm plate, so it only rests on a surface it is aligned
# with; position alone would let the IK pivot it into the ground. At this
# weight the sole sits within a few tenths of a millimetre of the surface it
# stands on, sloped or otherwise; weaker leaves a couple of degrees of tilt,
# which buries a corner several millimetres deep.
RetargetFootRotationWeightCfg.g1 = 4.0

# Two feet defend their contacts with half a quadruped's total pull, and the
# sole-orientation objective competes for the same joints, so the contact
# targets need a firmer hold. This brings foot placement error down from
# ~100 mm to ~3 mm without unseating the soles.
RetargetFootContactWeightCfg.g1 = 8.0

# Requiring both feet to snap left the sampler unable to report a foot as
# airborne, so on terrain where no patch fits a 21 cm sole it claimed a
# contact anyway. Allowing one lets it say so, and the support region built
# from sole outlines handles single support correctly.
RetargetMinContactsCfg.g1 = 1

# A fifth of a metre is most of G1's 24 cm stance width, so feet were being
# snapped to patches their legs could not reach and still flagged as
# contacts. Kept above the patch spacing the denser pool provides.
RetargetSnapDistanceCfg.g1 = 0.08

# Hold the upper body near its default stance and keep the legs from splaying,
# leaving hip pitch, knee, and ankle pitch free so IK can shape the stance to
# the terrain. Values match the articulation's default pose so the regularizer
# and the spawn pose agree.
RetargetJointRegularizeTargetsCfg.g1 = {
    G1_LATERAL_HIP_PATTERN: 0.0,
    ".*_hip_yaw_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    ".*_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    ".*_shoulder_yaw_joint": 0.0,
    ".*_elbow_joint": 0.6,
    ".*_wrist_roll_joint": 0.0,
    ".*_wrist_pitch_joint": 0.0,
    ".*_wrist_yaw_joint": 0.0,
}
