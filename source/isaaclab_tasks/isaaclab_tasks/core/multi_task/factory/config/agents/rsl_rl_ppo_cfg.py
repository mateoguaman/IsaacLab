# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from isaaclab_tasks.core.multi_task.rl.rsl_rl import RslRlResidualMLPEncoderModelCfg
from isaaclab_tasks.utils import PresetCfg, preset

# Shared PPO hyper-parameters reused by both the plain-PPO and value-shift variants.
_FACTORY_PPO_KWARGS: dict[str, Any] = dict(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=6e-3,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1.0e-4,
    schedule="adaptive",
    gamma=0.995,
    lam=0.92,
    desired_kl=0.01,
    max_grad_norm=1.0,
)


@configclass
class FactoryActorPresetCfg(PresetCfg):
    """Actor presets selectable via ``agent.actor=<name>``."""

    actor_critic = RslRlMLPModelCfg(
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
        obs_normalization=True,
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
    )
    simba = RslRlResidualMLPEncoderModelCfg(
        hidden_dim=256,
        num_blocks=2,
        expand=4,
        activation="swish",
        norm=True,
        obs_normalization=True,
        encoder_normalization=True,
        stochastic=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
        encoder_cfg={},
    )
    simba_big = RslRlResidualMLPEncoderModelCfg(
        hidden_dim=512,
        num_blocks=2,
        expand=4,
        activation="swish",
        norm=True,
        obs_normalization=True,
        encoder_normalization=True,
        stochastic=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
        encoder_cfg={},
    )
    default = actor_critic


@configclass
class FactoryCriticPresetCfg(PresetCfg):
    """Critic presets selectable via ``agent.critic=<name>``."""

    actor_critic = RslRlMLPModelCfg(
        obs_normalization=True,
        hidden_dims=[512, 256, 128, 64],
        activation="elu",
    )
    simba = RslRlResidualMLPEncoderModelCfg(
        hidden_dim=256,
        num_blocks=2,
        expand=4,
        activation="swish",
        norm=True,
        obs_normalization=True,
        encoder_normalization=True,
        stochastic=False,
        encoder_cfg={},
    )
    simba_big = RslRlResidualMLPEncoderModelCfg(
        hidden_dim=1024,
        num_blocks=4,
        expand=4,
        activation="swish",
        norm=True,
        obs_normalization=True,
        encoder_normalization=True,
        stochastic=False,
        encoder_cfg={},
    )
    default = actor_critic


@configclass
class ValueShiftAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO algorithm cfg + bind expressions for the value-shift curriculum.

    Bind expressions are popped off this cfg by
    :meth:`ValueShiftPPO.construct_algorithm` (so they never reach
    ``PPO.__init__``) and ``eval``-ed against ``{env, alg, setattr}``. They wire
    :class:`ValueShiftPPO`'s three buffers to the matching
    :class:`ValueShiftSamplingStrategy` living on the ``reset_sampler``
    curriculum term. Only meaningful when that curriculum sampler preset
    includes value-shift scoring (``value_shift`` / ``beta_value_shift``).
    """

    class_name: str = "isaaclab_tasks.core.multi_task.rl.rsl_rl.algorithms:ValueShiftPPO"
    # ``env`` here is the ``RslRlVecEnvWrapper`` from the runner -- managers
    # live on the underlying ``ManagerBasedRLEnv`` accessed via ``.unwrapped``.
    bind_observation_exp: str = (
        "setattr(alg, '_obs_cache',"
        " env.unwrapped.curriculum_manager.get_term('reset_sampler')"
        "._sampler._impl.value_shift_strategy.observation_cache)"
    )
    bind_current_value_exp: str = (
        "setattr(alg, '_cur_buf',"
        " env.unwrapped.curriculum_manager.get_term('reset_sampler')"
        "._sampler._impl.value_shift_strategy.cur_val)"
    )
    bind_value_diff_exp: str = (
        "setattr(alg, '_diff_buf',"
        " env.unwrapped.curriculum_manager.get_term('reset_sampler')"
        "._sampler._impl.value_shift_strategy.diff_val)"
    )


@configclass
class SuccessorLatentAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO algorithm cfg + successor-latent value loss coefficient."""

    class_name: str = "isaaclab_tasks.core.multi_task.rl.rsl_rl.algorithms:SuccessorLatentPPO"
    successor_loss_coef: float = 0.01
    """Coefficient for the vector-valued successor-latent loss."""


@configclass
class PpoAlgorithmCfg(PresetCfg):
    """Algorithm preset selectable via ``agent.algorithm=<name>``."""

    actor_critic = RslRlPpoAlgorithmCfg(class_name="PPO", **_FACTORY_PPO_KWARGS)
    default = actor_critic
    value_shift = ValueShiftAlgorithmCfg(**_FACTORY_PPO_KWARGS)
    beta_value_shift = value_shift
    successor_latent = SuccessorLatentAlgorithmCfg(**_FACTORY_PPO_KWARGS)


@configclass
class FactoryPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 15000
    save_interval = 200
    experiment_name = "factory"
    obs_groups = preset(
        default={"actor": ["policy"], "critic": ["policy"]},
        actor_critic={"actor": ["policy"], "critic": ["policy"]},
        simba={"actor": ["policy"], "critic": ["policy"]},
        simba_big={"actor": ["policy"], "critic": ["policy"]},
    )  # type: ignore
    actor = FactoryActorPresetCfg()  # type: ignore
    critic = FactoryCriticPresetCfg()  # type: ignore
    algorithm = PpoAlgorithmCfg()  # type: ignore
