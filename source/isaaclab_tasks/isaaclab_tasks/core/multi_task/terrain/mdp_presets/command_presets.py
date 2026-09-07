# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command presets selectable via ``env.commands.goal_point.commands=<name>``."""

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from ...kinematics import NewtonKinematicsCfg
from ...kinematics.ik_objectives.cfg import (
    IKObjectiveGravityTorqueCfg,
    IKObjectiveJointDefaultCfg,
    IKObjectiveStabilityMarginCfg,
    IKObjectiveTerrainCollisionCfg,
)
from .. import mdp
from ..retarget import RetargetPipelineCfg
from ..retarget.cfg import PatchSamplingCfg, SamplerCfg, SamplerSizingCfg
from ..retarget.criteria_cfg import (
    CollisionCheckCfg,
    FootPositionErrorCfg,
    JointWithinLimitCfg,
    LateralHipLimitCfg,
    SolverCostOutlierCfg,
    SupportPolygonStabilityCfg,
)
from ..retarget.feature_extractors import (
    XYZYawFeatures,
)
from .robots.robot_presets import (
    FootBodyNamesCfg,
    RetargetFootContactWeightCfg,
    RetargetFootRotationWeightCfg,
    RetargetJointRegularizeTargetsCfg,
    RetargetLateralHipJointPatternCfg,
    RetargetMinContactsCfg,
    RetargetMorphPatchOversampleCfg,
    RetargetSnapDistanceCfg,
    RetargetTerrainCollisionMarginCfg,
)


@configclass
class CommandsPresetCfg(PresetCfg):
    """Named command configurations for the position locomotion task."""

    all_commands = {
        "lin_vel_cmd": mdp.VelocityCommands(
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=None,
            lin_vel_x=(-2.0, 2.0),
            lin_vel_y=(-2.0, 2.0),
            lin_vel_z=None,
            ang_vel_x=None,
            ang_vel_y=None,
            ang_vel_z=(-0.2, 0.2),
            duration=(0.05, 4.0),
        ),
        "ang_vel_cmd": mdp.VelocityCommands(
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=None,
            lin_vel_x=(-0.0, 0.0),
            lin_vel_y=(-0.0, 0.0),
            lin_vel_z=None,
            ang_vel_x=None,
            ang_vel_y=None,
            ang_vel_z=(-2.0, 2.0),
            duration=(0.05, 4.0),
        ),
        "terrain_position_cmd": mdp.TerrainCommands(
            match_base_pos=True,
            match_base_rot=False,
            duration=(0.05, 2.0),
        ),
        "terrain_pose_cmd": mdp.TerrainCommands(
            match_base_pos=True,
            match_base_rot=True,
            duration=(0.05, 2.0),
        ),
        "terrain_stand_up_cmd": mdp.TerrainCommands(
            match_base_pos=False,
            match_base_rot=True,
            duration=(0.05, 4.0),
        ),
        "position_cmd": mdp.PositionCommands(
            pos_x=(-3.0, 3.0),
            pos_y=(-3.0, 3.0),
            pos_z=None,
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=None,
            duration=(0.05, 2.0),
        ),
        "pose_cmd": mdp.PoseCommands(
            pos_x=(-3.0, 3.0),
            pos_y=(-3.0, 3.0),
            pos_z=None,
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=None,
            duration=(0.05, 2.0),
        ),
    }
    terrain = {
        "terrain_pose_cmd": mdp.TerrainCommands(
            match_base_pos=True,
            match_base_rot=True,
            duration=(0.05, 2.0),
        ),
        "terrain_position_cmd": mdp.TerrainCommands(
            match_base_pos=True,
            match_base_rot=False,
            duration=(0.05, 2.0),
        ),
    }
    terrain_pos = {
        "terrain_position_cmd": mdp.TerrainCommands(
            match_base_pos=True,
            match_base_rot=False,
            duration=(0.05, 1.0),
        ),
    }
    terrain_pose = {
        "terrain_pose_cmd": mdp.TerrainCommands(
            match_base_pos=True,
            match_base_rot=True,
            duration=(0.05, 1.0),
        ),
    }
    pose = {
        "pose_cmd": mdp.PoseCommands(
            pos_x=(-3.0, 3.0),
            pos_y=(-3.0, 3.0),
            pos_z=None,
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=(-3.14, 3.14),
            duration=(0.05, 2.0),
        ),
    }
    pos = {
        "position_cmd": mdp.PositionCommands(
            pos_x=(-3.0, 3.0),
            pos_y=(-3.0, 3.0),
            pos_z=None,
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=None,
            duration=(0.05, 2.0),
        ),
    }
    vel = {
        "lin_vel_cmd": mdp.VelocityCommands(
            lin_vel_x=(-2.0, 2.0),
            lin_vel_y=(-2.0, 2.0),
            lin_vel_z=None,
            ang_vel_x=None,
            ang_vel_y=None,
            ang_vel_z=(-0.2, 0.2),
            duration=(0.05, 2.0),
        ),
    }
    default = terrain


@configclass
class CommandPayloadPresetCfg(PresetCfg):
    """Named payload configurations for the position locomotion command."""

    base = mdp.BaseStatePayloadCfg(
        pos_std=0.4,
        rot_std=0.5,
        lin_vel_std=0.2,
        ang_vel_std=0.2,
    )
    base_foot = mdp.BaseFootStatePayloadCfg(
        pos_std=0.4,
        rot_std=0.5,
        lin_vel_std=0.2,
        ang_vel_std=0.2,
        foot_pos_std=0.25,
    )
    default = base


@configclass
class CommandsCfg:
    "Command specifications for the MDP."

    goal_point = mdp.StateCommandCfg(
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        randomize_command_indices=False,
        states_relative=True,
        commands=CommandsPresetCfg(),  # type: ignore
        payload=CommandPayloadPresetCfg(),  # type: ignore
        task_table=mdp.TaskTableCfg(
            pool_spacing=0.5,
            max_spawns_per_cell=20,
            num_targets_per_cell=20,
            pipeline_cfg=RetargetPipelineCfg(
                asset_cfg=SceneEntityCfg("robot"),
                kin=NewtonKinematicsCfg(usd_path=""),
                sampler=SamplerCfg(
                    patch=PatchSamplingCfg(  # this samples foot patch
                        contact_radius=0.04, max_height_diff=0.03, horizontal_scale=0.01, oversample_ratio=5.0
                    ),
                    sizing=SamplerSizingCfg(
                        fps_features=XYZYawFeatures(yaw_scale=0.1),
                        criteria_yield=0.10,
                        morph_patch_oversample=RetargetMorphPatchOversampleCfg(),  # type: ignore[arg-type]
                    ),
                    min_contacts=RetargetMinContactsCfg(),  # type: ignore[arg-type]
                    terrain_snap_distance=RetargetSnapDistanceCfg(),  # type: ignore[arg-type]
                    outward_snap_penalty=1.0,
                ),
                foot_body_names=FootBodyNamesCfg(),  # type: ignore[arg-type]
                lateral_hip_joint_pattern=RetargetLateralHipJointPatternCfg(),  # type: ignore[arg-type]
                base_pos_weight=0.05,
                base_rot_weight=0.5,
                foot_contact_weight=RetargetFootContactWeightCfg(),  # type: ignore[arg-type]
                foot_rotation_weight=RetargetFootRotationWeightCfg(),  # type: ignore[arg-type]
                joint_regularize_targets=RetargetJointRegularizeTargetsCfg(),  # type: ignore[arg-type]
                extra_objectives=[
                    IKObjectiveTerrainCollisionCfg(
                        weight=2.0,
                        margin=RetargetTerrainCollisionMarginCfg(),  # type: ignore[arg-type]
                        n_samples=4,
                    ),
                    IKObjectiveStabilityMarginCfg(weight=1.0),
                    IKObjectiveGravityTorqueCfg(weight=0.02),
                    IKObjectiveJointDefaultCfg(weight=0.5),
                ],
                criteria=[
                    CollisionCheckCfg(n_samples=16, max_pen=0.02),
                    JointWithinLimitCfg(limit_ratio=0.9),
                    LateralHipLimitCfg(max_angle=1.05),
                    SupportPolygonStabilityCfg(),
                    FootPositionErrorCfg(max_err=0.4, aggregate="sum"),
                    SolverCostOutlierCfg(threshold_multiplier=3.0),
                ],
                ik_iterations=200,
                ik_convergence_threshold=0.01,
            ),
        ),
    )
