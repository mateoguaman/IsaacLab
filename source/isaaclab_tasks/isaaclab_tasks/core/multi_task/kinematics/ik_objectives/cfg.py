# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration dataclasses for custom Newton IK objectives.

Each :class:`IKObjectiveBaseCfg` subclass declares the static parameters
of an IK objective and sets :attr:`class_type` to the objective
implementation. The pipeline instantiates objectives via
``cfg.class_type(cfg, pipeline, wp_mesh)``; the objective's ``__init__``
pulls any runtime state it needs (kinematics, foot indices, sampler
state) from ``pipeline``.
"""

from __future__ import annotations

from dataclasses import MISSING, field

from isaaclab.utils.configclass import configclass


@configclass
class IKObjectiveBaseCfg:
    """Base configuration for a retarget IK objective.

    Subclasses set :attr:`class_type` to the objective implementation
    (resolvable ``"{DIR}.module:ClassName"`` string). Called as
    ``class_type(cfg, pipeline, wp_mesh)``.
    """

    class_type: type | str = MISSING  # type: ignore[assignment]
    """Objective implementation class."""


@configclass
class IKObjectiveTerrainCollisionCfg(IKObjectiveBaseCfg):
    """Config for :class:`IKObjectiveTerrainCollision`.

    The objective probes robot bodies against the terrain mesh and
    penalizes penetration. Foot bodies are excluded from probing since
    they are expected to contact the terrain.
    """

    class_type: type | str = "{DIR}.terrain_collision:IKObjectiveTerrainCollision"

    weight: float = 3.0
    """Residual weight [unitless]."""

    margin: float = 0.05
    """Softplus temperature [m]. Larger values soften the penalty's knee."""

    n_samples: int = 4
    """Surface probe points per body."""


@configclass
class IKObjectiveStabilityMarginCfg(IKObjectiveBaseCfg):
    """Config for :class:`IKObjectiveStabilityMargin`.

    The support region is the convex hull of the sole outlines of every foot
    in contact, derived from the robot's own collision geometry, so the
    objective applies unchanged at any contact count.
    """

    class_type: type | str = "{DIR}.stability_margin:IKObjectiveStabilityMargin"

    weight: float = 1.0
    """Residual weight [unitless]."""

    contact_height_tol: float = 0.001
    """How far above its lowest point a foot sample still counts as sole [m].

    Forwarded to
    :meth:`~isaaclab_tasks.core.multi_task.kinematics.NewtonKinematics.foot_contact_hulls`.
    Keep it in step with the matching criterion so the solver optimises the
    same support region the acceptance check enforces.
    """

    num_directions: int = 16
    """Probe directions used for the support-function test [unitless].

    The region is bounded by the sampled directions, so a low count admits
    slightly more than the true hull: the overshoot is ``1 / cos(pi / n) - 1``
    of the inradius, about 2% at the default.
    """


@configclass
class IKObjectiveGravityTorqueCfg(IKObjectiveBaseCfg):
    """Config for :class:`IKObjectiveGravityTorque`.

    Penalizes the per-revolute static gravity-compensation torque
    :math:`\\tau_j = \\hat{a}_j \\cdot (r_j \\times m_j\\, g)` where
    :math:`\\hat{a}_j` is the joint axis, :math:`r_j` the vector from
    the joint to the subtree COM, and :math:`m_j` the subtree mass.
    The zero-torque configuration is the subtree COM directly below the
    joint axis (a "hanging" pose), so this term pulls free DOFs --
    e.g. a raised foot's leg in an nc<4 stance -- toward a
    gravitationally natural posture without fighting the constrained
    DOFs required to hit the contact targets.

    Weight should be small enough that the foot-contact, base-pose, and
    joint-regularize objectives dominate: empirically ``0.01``--``0.05``
    gives natural hanging/folded postures for unconstrained limbs
    without biasing the stance legs away from their contact solutions.
    """

    class_type: type | str = "{DIR}.gravity_torque:IKObjectiveGravityTorque"

    weight: float = 0.02
    """Residual weight [unitless] applied uniformly across revolute joints."""


@configclass
class IKObjectiveJointDefaultCfg(IKObjectiveBaseCfg):
    """Config for :class:`IKObjectiveJointDefault`.

    Penalizes deviation of every (non-root) joint DOF from its angle in
    :attr:`~isaaclab_tasks.core.multi_task.kinematics.NewtonKinematics.default_joint_q`,
    i.e. the asset's default joint configuration captured at kinematics
    init. Differs from :class:`IKObjectiveJointRegularizeCfg` in two ways:

    * **Coverage:** every (non-root) DOF is regularized; there is no
      regex selection.
    * **Targets:** targets are fixed to the per-DOF entries of
      ``default_joint_q``, so each joint is pulled toward the robot's
      nominal pose rather than a user-supplied angle.

    Use a small weight (typical ``0.02``--``0.1``) so the foot-contact,
    base-pose, and stability objectives still dominate the solve and the
    objective only nudges the IK toward the robot's nominal pose where
    the contact constraints leave slack.
    """

    class_type: type | str = "{DIR}.joint_default:IKObjectiveJointDefault"

    weight: float = 0.05
    """Uniform residual weight [unitless] applied to every (non-root) DOF."""

    skip_root: bool = True
    """Exclude the free-root joint's 6 DOFs from regularization."""


@configclass
class IKObjectiveJointRegularizeCfg(IKObjectiveBaseCfg):
    """Config for :class:`IKObjectiveJointRegularize`.

    Each entry in :attr:`joint_targets` maps a joint-name regex to the
    target angle [rad] its matched DOFs are pulled toward. DOFs not
    matched by any pattern are left free. If multiple patterns match the
    same DOF, the **last** matching entry's target wins (Python dict
    insertion order). Patterns that match zero joints on the current
    robot are silently skipped -- useful for multi-robot presets where
    each robot uses a different joint-naming convention.

    An empty :attr:`joint_targets` falls back to
    :attr:`RetargetPipelineCfg.joint_regularize_targets` (typically
    resolved per robot preset). If both are empty the objective raises
    at build time -- omit it from ``extra_objectives`` to disable
    regularization entirely.

    Example::

        IKObjectiveJointRegularizeCfg(
            joint_targets={
                ".*HAA": 0.0,  # ANYmal-C HAA -> 0 rad
                ".*hip_joint": 0.0,  # go2/b2 HAA   -> 0 rad
                ".*hip_x": 0.0,  # spot HAA     -> 0 rad
            },
            weight=3.0,
        )
    """

    class_type: type | str = "{DIR}.joint_regularize:IKObjectiveJointRegularize"

    joint_targets: dict[str, float] = field(default_factory=dict)
    """Mapping of joint-name regex -> target angle [rad]. Empty dict falls back to the pipeline cfg."""

    weight: float = 1.0
    """Uniform residual weight [unitless] applied to every matched DOF."""
