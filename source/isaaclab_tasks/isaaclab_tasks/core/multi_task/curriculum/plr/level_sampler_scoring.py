# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-functional per-env score computation for the unified :class:`LevelSampler`.

Each scoring function here corresponds to one row of PLR Table 1
:cite:`Jiang2021PLR`. The dispatcher :func:`compute_per_env_score` selects the
right one based on the :class:`LevelSamplerCfg.scoring` field and returns a
1-D tensor of length ``num_envs`` -- one score per env per training iteration.

The scoring functions read from a generic :class:`RolloutData` namespace
rather than rsl_rl's storage directly, so unit tests can construct synthetic
rollouts without instantiating a runner. The runner extension's job is to
adapt its storage into a :class:`RolloutData` instance before calling the
dispatcher.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .level_sampler_cfg import (
    GAEMagnitudeScoringCfg,
    GAESignedScoringCfg,
    ScoringCfg,
    TD1MagnitudeScoringCfg,
)


@dataclass
class RolloutData:
    """Adapter exposing the rollout fields each scoring function may need.

    Each field is optional; the dispatcher checks that the requested scoring
    has its inputs and raises ``ValueError`` otherwise. All tensors are
    expected to be shape ``[T, N]`` or ``[T, N, 1]`` (the latter is squeezed
    on the last dim before reduction).

    Attributes:
        advantages: Per-step GAE advantages, optionally standardized over the
            rollout. Shape ``[T, N, 1]`` or ``[T, N]``. Required by GAE-based
            scorings when ``use_unnormalized=False``.
        returns: Per-step returns ``\\sum_{k>=t} gamma^(k-t) r_k`` (with
            bootstrap). Required for GAE-based scorings with
            ``use_unnormalized=True`` (then ``GAE = returns - values``) and for
            1-step TD scoring's value baseline.
        values: Per-step value-function predictions :math:`V(s_t)`. Required
            for raw-GAE and 1-step TD scorings.
        rewards: Per-step rewards :math:`r_t`. Required for 1-step TD scoring.
        gamma: Discount factor for any bootstrap-based scoring (1-step TD).
            The runner extension sources it from the PPO algorithm so it
            stays in sync without an explicit cfg knob.
    """

    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None
    values: torch.Tensor | None = None
    rewards: torch.Tensor | None = None
    gamma: float | None = None


def _squeeze_last(t: torch.Tensor) -> torch.Tensor:
    """Drop a trailing length-1 dimension if present.

    rsl_rl's storage uses ``[T, N, 1]`` for scalar-per-step quantities;
    test inputs may use ``[T, N]``. Normalize to ``[T, N]`` before reducing.
    """
    if t.dim() >= 1 and t.shape[-1] == 1:
        return t.squeeze(-1)
    return t


# ---------------------------------------------------------------------------
# Per-scoring implementations
# ---------------------------------------------------------------------------


def _gae_per_step(rollouts: RolloutData, use_unnormalized: bool) -> torch.Tensor:
    """Return per-step GAE advantages, shape ``[T, N]``.

    When ``use_unnormalized=True``, computes ``returns - values``; otherwise
    returns ``advantages`` directly (which the algorithm has already
    standardized over the rollout).
    """
    if use_unnormalized:
        if rollouts.returns is None or rollouts.values is None:
            raise ValueError(
                "GAE scoring with use_unnormalized=True requires both `returns` and `values` in RolloutData."
            )
        gae = rollouts.returns - rollouts.values
    else:
        if rollouts.advantages is None:
            raise ValueError("GAE scoring with use_unnormalized=False requires `advantages` in RolloutData.")
        gae = rollouts.advantages
    return _squeeze_last(gae)


def _score_gae_magnitude(rollouts: RolloutData, cfg: GAEMagnitudeScoringCfg) -> torch.Tensor:
    """``(1/T) sum_t |GAE_t|`` per env."""
    gae = _gae_per_step(rollouts, cfg.use_unnormalized)
    return gae.abs().mean(dim=0)


def _score_gae_signed(rollouts: RolloutData, cfg: GAESignedScoringCfg) -> torch.Tensor:
    """``(1/T) sum_t GAE_t`` per env, optionally clamped to one sign."""
    gae = _gae_per_step(rollouts, cfg.use_unnormalized)
    if cfg.sign == "positive":
        gae = gae.clamp_min(0.0)
    elif cfg.sign == "negative":
        gae = (-gae).clamp_min(0.0)
    return gae.mean(dim=0)


def _score_td1_magnitude(rollouts: RolloutData, cfg: TD1MagnitudeScoringCfg) -> torch.Tensor:
    """``(1/(T-1)) sum_t |r_t + gamma V(s_{t+1}) - V(s_t)|`` per env.

    Uses ``T - 1`` denominator because the last timestep has no
    ``V(s_{t+1})`` available within the rollout slice. Time-varying done
    masks are not applied here -- the runner extension should zero out
    cross-episode bootstraps if it cares about correctness across resets.
    """
    if rollouts.rewards is None or rollouts.values is None:
        raise ValueError("1-step TD scoring requires both `rewards` and `values` in RolloutData.")
    if rollouts.gamma is None:
        raise ValueError("1-step TD scoring requires `gamma` in RolloutData.")
    r = _squeeze_last(rollouts.rewards)  # [T, N]
    v = _squeeze_last(rollouts.values)  # [T, N]
    if r.shape[0] < 2:
        raise ValueError(f"1-step TD scoring needs T >= 2, got T={r.shape[0]}.")
    delta = r[:-1] + float(rollouts.gamma) * v[1:] - v[:-1]  # [T-1, N]
    return delta.abs().mean(dim=0)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def compute_per_env_score(rollouts: RolloutData, cfg: ScoringCfg) -> torch.Tensor:
    """Dispatch ``cfg`` to the appropriate scoring function.

    Args:
        rollouts: Rollout buffer adapter holding the per-step quantities each
            scoring function may need.
        cfg: One of the :class:`ScoringCfg` variants. Determines which
            scoring function is invoked.

    Returns:
        Per-env score tensor of shape ``[N]``, dtype ``float32`` on the same
        device as the rollout tensors.

    Raises:
        ValueError: If ``rollouts`` is missing a required field for the
            selected scoring.
        TypeError: If ``cfg`` is not a recognized :class:`ScoringCfg` variant.
    """
    if isinstance(cfg, GAEMagnitudeScoringCfg):
        return _score_gae_magnitude(rollouts, cfg)
    if isinstance(cfg, GAESignedScoringCfg):
        return _score_gae_signed(rollouts, cfg)
    if isinstance(cfg, TD1MagnitudeScoringCfg):
        return _score_td1_magnitude(rollouts, cfg)
    raise TypeError(f"Unsupported scoring cfg type: {type(cfg).__name__}")
