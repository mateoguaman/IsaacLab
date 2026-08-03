# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Composed config families for :class:`LevelSampler`.

Each family is a typed union; the sampler dispatches by ``isinstance`` over
each axis. This decouples *what to score*, *how to aggregate over time*, *how
to map scores to a priority distribution*, *whether to mix in staleness*, and
*how to mix replay vs. unseen-level proposal*.
"""

from __future__ import annotations

from typing import Literal

from isaaclab.utils.configclass import configclass

# ---------------------------------------------------------------------------
# ScoringCfg -- what per-env signal to compute from a rollout
# ---------------------------------------------------------------------------


@configclass
class GAEMagnitudeScoringCfg:
    """Per-step GAE-magnitude (a.k.a. L1 value loss) score :cite:`Jiang2021PLR`.

    Computes :math:`(1/T) \\sum_t |GAE_t|` per env, where ``GAE_t`` is the
    Generalized Advantage Estimate at timestep ``t``. This is the default
    scoring function in the PLR paper (Table 1, "GAE magnitude"); it equals
    the L1 value-prediction loss when the policy uses GAE for its targets.

    Attributes:
        use_unnormalized: If ``True``, read ``returns - values`` (raw GAE).
            If ``False``, read ``storage.advantages`` which is per-rollout
            standardized to zero mean and unit variance. The PLR paper uses
            raw GAE; standardization compresses cross-cell magnitude
            differences.
    """

    use_unnormalized: bool = True


@configclass
class GAESignedScoringCfg:
    """Per-step signed GAE score (PLR Table 1, "GAE").

    Computes :math:`(1/T) \\sum_t GAE_t` -- positive when returns exceed
    expectations on average over the rollout, negative when they fall short.
    Useful for asymmetric prioritization where positive vs. negative surprise
    should be weighted differently.

    Attributes:
        use_unnormalized: As :class:`GAEMagnitudeScoringCfg`.
        sign: ``"both"`` keeps the signed mean (can be negative). ``"positive"``
            takes ``mean(max(GAE, 0))`` (pleasant-surprise). ``"negative"`` takes
            ``mean(max(-GAE, 0))`` (unpleasant-surprise). Negative scores are
            still valid signals when fed into a sign-aware prioritization, but
            most prioritization kernels expect nonneg inputs -- pick a sign-mode
            that matches the downstream :class:`PrioritizationCfg`.
    """

    use_unnormalized: bool = True
    sign: Literal["both", "positive", "negative"] = "both"


@configclass
class TD1MagnitudeScoringCfg:
    """1-step TD-error magnitude (PLR Table 1, "1-step TD error").

    Computes :math:`(1/T) \\sum_t |\\delta_t|` where
    :math:`\\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)`. The discount
    :math:`\\gamma` is sourced from :attr:`RolloutData.gamma`, which the
    runner extension populates from the PPO algorithm so the two stay in
    sync without an explicit cfg knob.
    """

    pass


ScoringCfg = GAEMagnitudeScoringCfg | GAESignedScoringCfg | TD1MagnitudeScoringCfg
"""Union of supported scoring functions. Add new variants to both this union
and :func:`compute_per_env_score` in ``level_sampler_scoring.py``."""


# ---------------------------------------------------------------------------
# AggregationCfg -- how to combine new scores into the per-cell estimate
# ---------------------------------------------------------------------------


@configclass
class RollingMeanAggregationCfg:
    """Per-cell rolling mean over the most recent ``history_len`` writes.

    Each push appends one entry per env to the cell's circular buffer; the
    cell's score is the mean of all valid entries (size capped at
    ``history_len``).

    Attributes:
        history_len: Maximum number of recent writes to retain per cell.
            Larger values smooth more aggressively; smaller values track
            recent changes more quickly. Must be >= 1.
    """

    history_len: int = 100


@configclass
class EMAAggregationCfg:
    """Per-iter-mean then exponential moving average.

    Each push first averages scores within each visited cell (so multiple envs
    landing in the same cell contribute one iter-mean), then blends the cell's
    running estimate toward that iter-mean: ``score_i <- (1 - alpha) * old +
    alpha * iter_mean_i``. With ``alpha=1.0`` the cell's score becomes exactly
    this iter's mean for that cell.

    Attributes:
        alpha: Smoothing factor in :math:`[0, 1]`. ``1.0`` discards history;
            smaller values give stronger exponential decay.
    """

    alpha: float = 1.0


AggregationCfg = RollingMeanAggregationCfg | EMAAggregationCfg


# ---------------------------------------------------------------------------
# PrioritizationCfg -- map per-cell scores to a priority distribution P_S
# ---------------------------------------------------------------------------


@configclass
class ProportionalPrioritizationCfg:
    """PER-style proportional priority: :math:`P(i) \\propto (S_i + \\epsilon)^\\alpha`.

    Sampled probability via :math:`\\text{softmax}(\\log P / \\tau)`; with
    :math:`\\tau = 1` this reduces to plain proportional. Use :math:`\\epsilon
    = 10^{-3}` for the PER-style meaningful floor or :math:`10^{-8}` for a
    purely numerical guard.

    Attributes:
        alpha: Sharpness exponent. ``alpha=0`` is uniform, ``alpha=1`` is plain
            proportional, larger values approach greedy.
        eps: Priority floor. PER recommends ~``1e-3`` so unseen / zero-score
            cells retain non-zero probability; ``1e-8`` is effectively a
            numerical-only floor.
        temperature: Softmax temperature. Combined exponent on raw scores is
            ``alpha / temperature`` -- the two knobs partially overlap.
    """

    alpha: float = 1.0
    eps: float = 1e-3
    temperature: float = 1.0


@configclass
class RankPrioritizationCfg:
    """PLR-recommended rank-based priority: :math:`P(i) \\propto 1 / \\text{rank}(S_i)^{1/\\beta}`.

    Robust to outliers because only the relative ordering matters. With
    :math:`\\beta = 1` this is the harmonic ranking; smaller :math:`\\beta`
    sharpens, larger flattens.

    Attributes:
        temperature: Inverse exponent ``beta``. Smaller concentrates on the
            top cell. ``1.0`` is harmonic 1/rank; PLR Procgen used ``0.1``.
    """

    temperature: float = 1.0


PrioritizationCfg = ProportionalPrioritizationCfg | RankPrioritizationCfg


# ---------------------------------------------------------------------------
# StalenessCfg -- mix the score distribution with a staleness distribution
# ---------------------------------------------------------------------------


@configclass
class StalenessCfg:
    """PLR staleness mixing :math:`P_{\\text{replay}} = (1 - \\rho) P_S + \\rho P_C`.

    Where :math:`P_C(i) \\propto (c - C_i)` and :math:`c` is the global visit
    count, :math:`C_i` the visit count when cell ``i`` was last sampled.
    Without staleness mixing, cells whose rolling-mean score settles low
    never get re-sampled; PLR's ablations show both pure scores
    (:math:`\\rho=0`) and pure staleness (:math:`\\rho=1`) are worse than
    uniform -- the mix is what works.

    Attributes:
        staleness_coefficient: Mixing weight :math:`\\rho \\in [0, 1]`. ``0``
            disables staleness; ``1`` ignores scores. PLR Procgen used
            ``0.1``; MiniGrid ``0.3``.
        transform: Prioritization applied to per-cell staleness ``c - C_i``
            before normalization. Defaults to plain proportional.
    """

    staleness_coefficient: float = 0.3
    transform: PrioritizationCfg = ProportionalPrioritizationCfg(alpha=1.0, eps=0.0, temperature=1.0)


# ---------------------------------------------------------------------------
# ProposalCfg -- mix replay (seen cells) with new (unseen cells) proposals
# ---------------------------------------------------------------------------


@configclass
class AlwaysReplayProposalCfg:
    """Always sample from the replay distribution; never explicitly draw unseen.

    Unseen cells only receive probability through the priority floor (or,
    with rank-based priority, through their eventual placement in the
    ordering once any score is recorded).
    """

    pass


@configclass
class ProportionateProposalCfg:
    """PLR proportionate proposal: :math:`P(\\text{replay}) = \\text{proportion seen}`
    once warmup cleared.

    Replay probability anneals naturally from 0 to 1 as more cells become
    seen. Matches the PLR paper's text description ("we opt to naturally
    anneal :math:`P_D(d=1)` as
    :math:`|\\Lambda_{\\text{seen}}|/|\\Lambda_{\\text{train}}|`").

    Attributes:
        warmup_threshold: Fraction of cells that must have been visited
            before replay begins. ``0.0`` skips warmup.
    """

    warmup_threshold: float = 0.0


ProposalCfg = AlwaysReplayProposalCfg | ProportionateProposalCfg


# ---------------------------------------------------------------------------
# LevelSamplerCfg -- top-level composer
# ---------------------------------------------------------------------------


@configclass
class LevelSamplerCfg:
    """Composed PLR-style sampler config.

    Attributes:
        scoring: Per-env score to compute from each rollout.
        aggregation: How to combine successive per-env scores into the
            per-cell estimate.
        prioritization: Map per-cell score to a priority distribution
            :math:`P_S`.
        staleness: Optional staleness mixing. ``None`` disables.
        proposal: How to mix replay (seen) and new (unseen) cells.
    """

    scoring: ScoringCfg = GAEMagnitudeScoringCfg(use_unnormalized=False)
    aggregation: AggregationCfg = RollingMeanAggregationCfg(history_len=100)
    prioritization: PrioritizationCfg = ProportionalPrioritizationCfg(alpha=1.0, eps=1e-8, temperature=1.0)
    staleness: StalenessCfg | None = None
    proposal: ProposalCfg = AlwaysReplayProposalCfg()
