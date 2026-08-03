# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.multi_task.curriculum import (
    BetaSamplingStrategyCfg,
    EMAAggregationCfg,
    FrontierSamplingStrategyCfg,
    GAEMagnitudeScoringCfg,
    LevelSamplerCfg,
    ProportionateProposalCfg,
    RankPrioritizationCfg,
    SamplerCfg,
    StalenessCfg,
    StateLayoutCfg,
    SuccessMonitorCfg,
    UniformSamplingStrategyCfg,
)
from isaaclab_tasks.utils import PresetCfg, preset

from .. import mdp
from ..viz.sampler_images import log_spawn_goal_sampler_images


@configclass
class PositionCurriculumSamplerCfg:
    terrain_levels = CurrTerm(
        func=mdp.success_rate_sampler,
        params={
            "success_rates_bind": "env.command_manager.get_term('goal_point').success_rates",
            "sample_indices_bind": "env.command_manager.get_term('goal_point').cmd_indices",
            "layout": StateLayoutCfg(
                coords_bind="env.command_manager.get_term('goal_point').table.spawn_states[:, :2]",
                spawn_index_bind="env.command_manager.get_term('goal_point').table.spawn_index",
                target_index_bind="env.command_manager.get_term('goal_point').table.target_index",
                task_partition_bind="env.command_manager.get_term('goal_point').table.task_partition",
            ),
            "sampling": preset(
                default=SamplerCfg(
                    strategies=[
                        BetaSamplingStrategyCfg(target=0.66, kappa=2.5, weight=1.0, success_rate_bind="success_rates")
                    ],
                    eps=1e-4,
                ),
                uniform=SamplerCfg(strategies=[UniformSamplingStrategyCfg(weight=1.0)], eps=0.0),
                beta66=SamplerCfg(
                    strategies=[
                        BetaSamplingStrategyCfg(target=0.66, kappa=2.5, weight=1.0, success_rate_bind="success_rates")
                    ],
                    eps=1e-4,
                ),
                frontier=SamplerCfg(
                    strategies=[
                        BetaSamplingStrategyCfg(target=0.66, kappa=2.5, weight=1.0, success_rate_bind="success_rates"),
                        FrontierSamplingStrategyCfg(
                            k=8, dilation_steps=2, weight=0.5, success_rate_bind="success_rates"
                        ),
                    ],
                    eps=1e-4,
                ),
            ),
            "success_monitor_cfg": SuccessMonitorCfg(monitored_history_len=100),
            "success_bind": "env.termination_manager.get_term('success')",
            "sampler_visual_logger": log_spawn_goal_sampler_images,
            "sampler_visual_log_period": 1000,
        },
    )


@configclass
class CRLSamplerCfg:
    """Terrain curriculum without reward-dependent terms."""

    terrain_levels = CurrTerm(
        func=mdp.success_rate_sampler,
        params={
            "success_rates_bind": "env.command_manager.get_term('goal_point').success_rates",
            "sample_indices_bind": "env.command_manager.get_term('goal_point').cmd_indices",
            "layout": StateLayoutCfg(
                coords_bind="env.command_manager.get_term('goal_point').table.spawn_states[:, :2]",
                spawn_index_bind="env.command_manager.get_term('goal_point').table.spawn_index",
                target_index_bind="env.command_manager.get_term('goal_point').table.target_index",
                task_partition_bind="env.command_manager.get_term('goal_point').table.task_partition",
            ),
            "sampling": SamplerCfg(
                strategies=[
                    BetaSamplingStrategyCfg(target=0.66, kappa=1.0, weight=1.0, success_rate_bind="success_rates")
                ],
                eps=1e-8,
            ),
            "success_monitor_cfg": SuccessMonitorCfg(monitored_history_len=100),
            "success_bind": "env.termination_manager.get_term('success')",
            "sampler_visual_logger": log_spawn_goal_sampler_images,
            "sampler_visual_log_period": 1000,
        },
    )


@configclass
class LinearCurriculumCfg:
    """Monotonic per-env terrain-level ramp baseline (``presets=linear``; quadruped-scoped)."""

    terrain_levels = CurrTerm(
        func=mdp.linear_terrain_levels,
        params={
            "success_rates_bind": "env.command_manager.get_term('goal_point').success_rates",
            "sample_indices_bind": "env.command_manager.get_term('goal_point').cmd_indices",
            "success_bind": "env.termination_manager.get_term('success')",
            "goal_term_name": "goal_point",
            "success_monitor_cfg": SuccessMonitorCfg(monitored_history_len=100),
            "demotion_fraction": 0.5,
        },
    )


@configclass
class PLRCurriculumCfg:
    """Prioritized Level Replay curriculum (``presets=plr``).

    Selects the level-sampler curriculum term on the env side. The matching ``plr`` agent preset
    repoints the runner ``class_type`` to :class:`OnPolicyRunnerWithLevelSampler`, which feeds the
    per-iteration rollout regret this sampler prioritizes on; without it the sampler never receives
    scores.
    """

    terrain_levels = CurrTerm(
        func=mdp.level_sampler_curriculum,
        params={
            "success_rates_bind": "env.command_manager.get_term('goal_point').success_rates",
            "sample_indices_bind": "env.command_manager.get_term('goal_point').cmd_indices",
            "success_bind": "env.termination_manager.get_term('success')",
            "goal_term_name": "goal_point",
            "success_monitor_cfg": SuccessMonitorCfg(monitored_history_len=100),
            "sampling": LevelSamplerCfg(
                scoring=GAEMagnitudeScoringCfg(use_unnormalized=True),
                aggregation=EMAAggregationCfg(alpha=1.0),
                prioritization=RankPrioritizationCfg(),
                staleness=StalenessCfg(),
                proposal=ProportionateProposalCfg(warmup_threshold=0.0),
            ),
        },
    )


@configclass
class CurriculumPresetCfg(PresetCfg):
    position = PositionCurriculumSamplerCfg()
    crl = CRLSamplerCfg()
    linear = LinearCurriculumCfg()
    plr = PLRCurriculumCfg()
    default = position
