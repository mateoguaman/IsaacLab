# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""rsl_rl runner configurations + actor / critic preset dispatchers.

This module assembles the position-locomotion runner cfgs. Model cfg instances
themselves (``COMMANDER_ACTOR``, ``SIMBA_CRITIC``, etc.) live in
:mod:`rsl_rl_model_cfg` so model definitions stay independent of preset /
runner composition. The preset classes here just wire those instances into
``agent.actor=<name>`` / ``agent.critic=<name>`` selectable alternatives.
"""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlRndCfg

from isaaclab_tasks.utils import PresetCfg

from ...rl.rsl_rl import (
    RslRlCrlAlgorithmCfg,
    RslRlHerCfg,
    RslRlOffPolicyRunnerCfg,
    RslRlResidualMLPCfg,
)
from ..mdp_presets import ExperimentNameCfg
from .rsl_rl_model_cfg import (
    CNN_ACTOR_CFG,
    CNN_CRITIC_CFG,
    COMMANDER_ACTOR,
    ENCODER_ACTOR,
    FLAT_ACTOR,
    FLAT_CRITIC,
    LSTM_ACTOR,
    LSTM_CRITIC,
    MLP_ENCODER_CRITIC,
    SIMBA_ACTOR,
    SIMBA_BIG_CRITIC,
    SIMBA_CRITIC,
    SIMBA_RNN_ACTOR,
    SIMBA_RNN_CRITIC,
    TASK_EASING_ACTOR,
)


@configclass
class PositionActorPresetCfg(PresetCfg):
    """Actor presets selectable via ``agent.actor=<name>``. Instances live in
    :mod:`rsl_rl_model_cfg`; this preset just maps names → model cfgs."""

    commander = COMMANDER_ACTOR
    task_easing = TASK_EASING_ACTOR
    lstm = LSTM_ACTOR
    flat = FLAT_ACTOR
    cnn = CNN_ACTOR_CFG
    encoder = ENCODER_ACTOR
    simba = SIMBA_ACTOR
    simba_big = SIMBA_ACTOR
    simba_rnn = SIMBA_RNN_ACTOR
    default = encoder


@configclass
class PositionCriticPresetCfg(PresetCfg):
    """Critic presets selectable via ``agent.critic=<name>``. Instances live in
    :mod:`rsl_rl_model_cfg`; this preset just maps names → model cfgs."""

    flat = FLAT_CRITIC
    cnn = CNN_CRITIC_CFG
    lstm = LSTM_CRITIC
    mlp_encoder = MLP_ENCODER_CRITIC
    simba = SIMBA_CRITIC
    simba_big = SIMBA_BIG_CRITIC
    simba_rnn = SIMBA_RNN_CRITIC
    default = mlp_encoder


POSITION_PPO_ALGORITHM_CFG = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.005,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1.0e-4,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
    share_cnn_encoders=True,
)


POSITION_RND_CFG = RslRlRndCfg(
    weight=0.01,
    reward_normalization=True,
    state_normalization=True,
    learning_rate=1.0e-3,
    num_outputs=1,
    predictor_hidden_dims=[-1],
    target_hidden_dims=[-1],
)


@configclass
class PositionLocomotionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 20000
    save_interval = 200
    resume = False
    experiment_name: str = ExperimentNameCfg()  # type: ignore
    obs_groups = {"actor": ["policy", "task", "height_scan"], "critic": ["policy", "task", "height_scan"]}
    actor = PositionActorPresetCfg()  # type: ignore
    critic = PositionCriticPresetCfg()  # type: ignore
    algorithm: RslRlPpoAlgorithmCfg = POSITION_PPO_ALGORITHM_CFG


@configclass
class PositionLocomotionRndPPORunnerCfg(PositionLocomotionPPORunnerCfg):
    obs_groups = {
        "actor": ["policy", "task", "height_scan"],
        "critic": ["policy", "task", "height_scan"],
        "rnd_state": ["policy", "task"],
    }
    algorithm: RslRlPpoAlgorithmCfg = POSITION_PPO_ALGORITHM_CFG.replace(rnd_cfg=POSITION_RND_CFG)


@configclass
class PositionLocomotionCRLRunnerCfg(RslRlOffPolicyRunnerCfg):
    """OffPolicyRunner configuration for CRL on the position locomotion task."""

    num_steps_per_env = 62
    max_iterations = 5000
    save_interval = 500
    resume = False
    experiment_name = "position_crl"
    obs_groups = {
        "actor": ["current_state", "height_scan", "policy", "target_state"],
        "critic": ["current_state", "height_scan", "policy", "target_state"],
    }
    # Per-group encoder so the high-dim ``height_scan`` (``(B, 1, 76, 126)``,
    # 9576 raw values) is compressed before the residual MLP. The same encoder
    # cfg is forwarded by ``construct_algorithm`` to a single shared
    # ``_SharedStateEncoder`` instance that is wired into both actor and critic
    # (see :paramref:`RslRlCrlAlgorithmCfg.share_encoders`). Other groups
    # (``current_state``, ``target_state``, ``policy``) are low-dim and pass
    # through without an encoder.
    _CRL_ENCODER_CFG = {
        "height_scan": {"hidden_dims": [256, 256], "output_dim": 256, "activation": "elu"},
    }
    actor: RslRlResidualMLPCfg = RslRlResidualMLPCfg(
        hidden_dim=256,
        depth=8,
        num_layers_per_block=4,
        expand=1,
        activation="swish",
        encoder_cfg=_CRL_ENCODER_CFG,
    )
    critic: RslRlResidualMLPCfg = RslRlResidualMLPCfg(
        hidden_dim=256,
        depth=8,
        num_layers_per_block=4,
        expand=1,
        activation="swish",
        repr_dim=64,
        encoder_cfg=_CRL_ENCODER_CFG,
    )
    algorithm: RslRlCrlAlgorithmCfg = RslRlCrlAlgorithmCfg(
        actor_lr=8e-4,
        critic_lr=8e-4,
        alpha_lr=3e-4,
        max_replay_size=1500,
        min_replay_size=500,
        replay_ratio=0.04,
        num_sgd_steps=400,
        logsumexp_penalty_coeff=0.1,
        entropy_param=0.5,
        share_encoders=True,
        her_cfg=RslRlHerCfg(gamma=0.99),
    )


@configclass
class PositionRunnerCfg(PresetCfg):
    """Runner presets: ``presets=crl`` selects CRL, ``presets=plr`` selects the PLR runner, default is PPO."""

    position = PositionLocomotionPPORunnerCfg()
    rnd = PositionLocomotionRndPPORunnerCfg()
    crl = PositionLocomotionCRLRunnerCfg()
    # PLR: the standard PPO runner with a level-sampler rollout-push extension. Pairs with the
    # env-side ``plr`` curriculum preset (which selects the level_sampler_curriculum term).
    plr = PositionLocomotionPPORunnerCfg().replace(
        class_type="isaaclab_tasks.core.multi_task.rl.rsl_rl.runners:OnPolicyRunnerWithLevelSampler"
    )
    default = position


@configclass
class MultiTaskLocomotionPPORunnerCfg(PositionLocomotionPPORunnerCfg):
    """PPO runner for the multi-task env.

    The multi-task env publishes two obs groups (``policy`` proprioception and
    ``task`` goal deltas) but no ``height_scan``. :attr:`obs_groups` drops that
    key, and :attr:`actor` / :attr:`critic` use the plain-MLP model cfgs —
    ``FLAT_ACTOR`` / ``FLAT_CRITIC`` from :mod:`rsl_rl_model_cfg` — so nothing
    tries to bind a ``height_scan`` encoder that doesn't exist here.
    """

    obs_groups = {"actor": ["policy", "task"], "critic": ["policy", "task"]}
    actor = FLAT_ACTOR  # type: ignore[assignment]
    critic = FLAT_CRITIC  # type: ignore[assignment]
    algorithm: RslRlPpoAlgorithmCfg = POSITION_PPO_ALGORITHM_CFG.replace(share_cnn_encoders=False)
