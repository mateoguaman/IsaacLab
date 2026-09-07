# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-axis preset *classes* for the position locomotion task.

Every :class:`PresetCfg` declared here represents one robot-specific scalar
that the base environment (and rsl_rl agent) references inline instead of
hard-coding.  Per-robot modules in this package (e.g. :mod:`anymal_c`,
:mod:`go2`) populate their own field on these classes at import time, and the
preset resolver then substitutes the picked field at every consumption site.

To register a new robot, drop a new module alongside this one and set class
attributes such as::

    from .robot_presets import BaseBodyNameCfg, RobotArticulationCfg, ...

    RobotArticulationCfg.<robot> = <ROBOT>_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    BaseBodyNameCfg.<robot> = "..."
    ...

Only add fields; **never reassign** ``default`` -- it is the fallback used
when no robot preset is selected.  Package-level defaults preserve the
strings previously hard-coded in the base env / reward / termination configs
so the task still loads without a robot picked (except for
:class:`RobotArticulationCfg.default`, which is :data:`MISSING` and will fail
loudly if resolution happens with no robot selected).
"""

from dataclasses import MISSING, field

from isaaclab.assets import ArticulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg


@configclass
class RobotArticulationCfg(PresetCfg):
    """Full :class:`ArticulationCfg` for the robot asset at ``scene.robot``.

    Required -- no sensible default exists, so resolving without picking a
    robot preset leaves ``scene.robot`` as :data:`MISSING`.
    """

    default: ArticulationCfg = MISSING  # type: ignore[assignment]


@configclass
class HeightScannerPrimPathCfg(PresetCfg):
    """USD prim path anchoring the height-scanner ray caster."""

    default: str = "{ENV_REGEX_NS}/Robot/base"


@configclass
class BaseBodyNameCfg(PresetCfg):
    """Robot base / torso body name.

    Used by events that target the base mass (``add_base_mass``) and by the
    viewer (``viewer.body_name``).
    """

    default: str = "base"


@configclass
class NonFootContactBodyNamesCfg(PresetCfg):
    """Body-name regex for non-foot bodies (everything except the feet).

    Drives the ``base_contact`` termination's ``sensor_cfg.body_names``.
    The intent is to detect impact on any non-foot body (knees, body,
    head, etc.), gated by an impact-force threshold so routine soft
    contact (kneeling, climbing, leaning) is allowed while shock impacts
    terminate the episode.
    """

    default: str = "^(?!.*foot).*$"


@configclass
class FootBodyNamesCfg(PresetCfg):
    """Body-name regex matching the robot's feet.

    Drives reward terms that index foot bodies and the retarget pipeline's
    terrain-contact bodies.
    """

    default: str = ".*FOOT.*"


@configclass
class AsyncFootPairsCfg(PresetCfg):
    """Foot pairs whose air/contact times should differ (gait diagnostic)."""

    default: tuple[tuple[str, str], ...] = ()


@configclass
class SyncFootPairsCfg(PresetCfg):
    """Foot pairs whose air/contact times should match (gait diagnostic)."""

    default: tuple[tuple[str, str], ...] = ()


@configclass
class ExperimentNameCfg(PresetCfg):
    """rsl_rl experiment name, picked up by ``PositionLocomotionPPORunnerCfg``."""

    default: str = "position_command"


@configclass
class RetargetLateralHipJointPatternCfg(PresetCfg):
    """Regex matching lateral hip joints for retarget validation.

    Consumed by :attr:`RetargetPipelineCfg.lateral_hip_joint_pattern`.
    ``None`` disables lateral-hip angle validation -- appropriate for
    robots with no lateral hip joints or where over-splay is not a concern.
    """

    default: str | None = None


@configclass
class RetargetTerrainCollisionMarginCfg(PresetCfg):
    """Clearance the retarget IK keeps between non-foot bodies and the terrain [m].

    Consumed by :attr:`IKObjectiveTerrainCollisionCfg.margin`. The usable value
    is bounded by the robot's own geometry: any non-foot body that already sits
    closer than this to the ground when standing gets pushed away from it, which
    lifts the attached foot and fights the contact targets. Shins and ankle
    links on a humanoid sit far lower than a quadruped's, so the ceiling is much
    tighter there.

    ``scripts/audit_collision_geometry.py`` reports each body's standing
    clearance and flags the ones a given margin would disturb.
    """

    default: float = 0.05


@configclass
class RetargetFootContactWeightCfg(PresetCfg):
    """Weight holding each foot on its contact target [unitless].

    Consumed by :attr:`RetargetPipelineCfg.foot_contact_weight`. Applied per
    foot, so the pull defending the stance totals this times the foot count
    while everything it competes against is independent of that count. A
    two-footed robot therefore needs roughly twice a quadruped's value to
    hold its contacts equally well.
    """

    default: float = 1.0


@configclass
class RetargetFootRotationWeightCfg(PresetCfg):
    """Weight holding each contact foot flat on its patch [unitless].

    Consumed by :attr:`RetargetPipelineCfg.foot_rotation_weight`. A sole with
    real extent has to match the surface under it or its corners drive into
    the terrain, so a flat-footed robot needs this; a point foot has no sole
    to keep flat and leaves it at zero.
    """

    default: float = 0.0


@configclass
class RetargetSnapDistanceCfg(PresetCfg):
    """How far a foot may move to reach a contact patch [m].

    Consumed by :attr:`SamplerCfg.terrain_snap_distance`. Beyond this a
    foot is reported airborne instead of snapped. The useful value scales
    with the robot: a fifth of a metre is a small fraction of a quadruped's
    stance but most of a humanoid's, where it lets feet claim patches the
    legs cannot reach. It also cannot go below the patch spacing, or feet
    find nothing to stand on -- see :class:`RetargetMorphPatchOversampleCfg`.
    """

    default: float = 0.2


@configclass
class RetargetMinContactsCfg(PresetCfg):
    """Contact slots a sampled stance must fill to be accepted [unitless].

    Consumed by :attr:`SamplerCfg.min_contacts`. Reaching the foot count
    makes every foot snap to a patch, which for two feet means the sampler
    can never report one as airborne even where no patch fits it. Allowing
    fewer lets it say so, and single-support stances are valid support
    regions in their own right once the sole outline is modelled.
    """

    default: int = 3


@configclass
class RetargetMorphPatchOversampleCfg(PresetCfg):
    """Terrain contact patches sampled per foot slot per placement [unitless].

    Consumed by :attr:`SamplerSizingCfg.morph_patch_oversample`. Because the
    budget is per foot slot, a two-footed robot ends up with half the patches
    over the same ground as a quadruped, leaving each foot a coarser grid to
    snap onto. Raising it restores the spatial density, which is what decides
    how far a foot has to move from its sampled stance to reach a patch.
    """

    default: float = 4.0


@configclass
class RetargetJointRegularizeTargetsCfg(PresetCfg):
    """Per-robot joint-name regex -> target-angle dict for retarget IK.

    Consumed by :attr:`RetargetPipelineCfg.joint_regularize_targets`.
    Each entry pulls its matched DOFs toward the listed angle during
    IK; unmatched DOFs are left free. Empty dict disables the
    regularizer entirely -- appropriate for robots where no joint-
    space prior is needed.
    """

    default: dict[str, float] = field(default_factory=dict)
