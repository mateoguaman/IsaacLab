# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config-only resolution checks for the curriculum + robot preset tokens added for the campaign.

These boot no sim app: they instantiate the env / runner cfgs and resolve the preset tokens, then
assert each token selects the intended term / actuator. Runtime behavior is validated separately by
the local GPU training smokes.
"""

from __future__ import annotations


def _resolve_env(token: str):
    from isaaclab_tasks.core.multi_task.position_env_cfg import LocomotionPositionCommandEnvCfg
    from isaaclab_tasks.utils import resolve_presets

    cfg = LocomotionPositionCommandEnvCfg()
    resolve_presets(cfg, frozenset({token}))
    return cfg


def test_uniform_selects_uniform_strategy():
    from isaaclab_tasks.core.multi_task.curriculum import UniformSamplingStrategyCfg

    cfg = _resolve_env("uniform")
    strategies = cfg.curriculum.terrain_levels.params["sampling"].strategies
    assert any(isinstance(s, UniformSamplingStrategyCfg) for s in strategies)


def test_linear_selects_linear_term():
    from isaaclab_tasks.core.multi_task.mdp.curriculums import linear_terrain_levels

    cfg = _resolve_env("linear")
    assert cfg.curriculum.terrain_levels.func is linear_terrain_levels


def test_plr_selects_level_sampler_term_and_runner():
    from isaaclab_tasks.core.multi_task.curriculum import LevelSamplerCfg
    from isaaclab_tasks.core.multi_task.mdp.curriculums import level_sampler_curriculum
    from isaaclab_tasks.core.multi_task.terrain.config.rsl_rl_cfg import PositionRunnerCfg
    from isaaclab_tasks.utils import resolve_presets

    env_cfg = _resolve_env("plr")
    assert env_cfg.curriculum.terrain_levels.func is level_sampler_curriculum
    assert isinstance(env_cfg.curriculum.terrain_levels.params["sampling"], LevelSamplerCfg)

    runner_cfg = PositionRunnerCfg()
    resolved = resolve_presets(runner_cfg, frozenset({"plr"}))
    runner_cfg = resolved if resolved is not None else runner_cfg
    assert runner_cfg.class_type.endswith("OnPolicyRunnerWithLevelSampler")


def test_anymal_d_pace_uses_pace_actuators():
    from isaaclab.actuators import PaceDCMotorCfg

    # Importing the robot preset package registers the token via class-attribute side effects.
    from isaaclab_tasks.core.multi_task.terrain.mdp_presets.robots import RobotArticulationCfg

    robot_cfg = RobotArticulationCfg.anymal_d_pace
    assert robot_cfg.actuators
    assert all(isinstance(act, PaceDCMotorCfg) for act in robot_cfg.actuators.values())
