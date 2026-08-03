# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the unified :class:`LevelSampler`.

Exercises each axis of :class:`LevelSamplerCfg` in isolation (scoring,
aggregation, prioritization, staleness, proposal) plus end-to-end
``push -> sample`` smoke tests. Pure-pytest, no Isaac dependency.
"""

from __future__ import annotations

import pytest
import torch

from isaaclab_tasks.core.multi_task.curriculum.plr.level_sampler import LevelSampler
from isaaclab_tasks.core.multi_task.curriculum.plr.level_sampler_cfg import (
    AlwaysReplayProposalCfg,
    EMAAggregationCfg,
    GAEMagnitudeScoringCfg,
    GAESignedScoringCfg,
    LevelSamplerCfg,
    ProportionalPrioritizationCfg,
    ProportionateProposalCfg,
    RankPrioritizationCfg,
    RollingMeanAggregationCfg,
    StalenessCfg,
    TD1MagnitudeScoringCfg,
)
from isaaclab_tasks.core.multi_task.curriculum.plr.level_sampler_scoring import (
    RolloutData,
    compute_per_env_score,
)

# ---------------------------------------------------------------------------
# Default-cfg construction smoke
# ---------------------------------------------------------------------------


def test_default_cfg_constructs():
    """The default :class:`LevelSamplerCfg` should reproduce the legacy ``advantage`` preset shape."""
    cfg = LevelSamplerCfg()
    assert isinstance(cfg.scoring, GAEMagnitudeScoringCfg)
    assert cfg.scoring.use_unnormalized is False  # matches default `advantage` preset
    assert isinstance(cfg.aggregation, RollingMeanAggregationCfg)
    assert cfg.aggregation.history_len == 100
    assert isinstance(cfg.prioritization, ProportionalPrioritizationCfg)
    assert cfg.prioritization.alpha == 1.0
    assert cfg.prioritization.eps == 1e-8
    assert cfg.prioritization.temperature == 1.0
    assert cfg.staleness is None
    assert isinstance(cfg.proposal, AlwaysReplayProposalCfg)


# ---------------------------------------------------------------------------
# compute_per_env_score (one block per ScoringCfg)
# ---------------------------------------------------------------------------


def _fake_rollout(T: int = 4, N: int = 3, seed: int = 0) -> RolloutData:
    """Synthetic rollout data with deterministic, hand-checkable values."""
    g = torch.Generator().manual_seed(seed)
    advantages = torch.randn(T, N, 1, generator=g)
    values = torch.randn(T, N, 1, generator=g)
    returns = values + advantages  # so that returns - values = advantages exactly
    rewards = torch.randn(T, N, 1, generator=g)
    return RolloutData(
        advantages=advantages,
        returns=returns,
        values=values,
        rewards=rewards,
    )


def test_score_gae_magnitude_unnormalized_uses_returns_minus_values():
    rollouts = _fake_rollout()
    cfg = GAEMagnitudeScoringCfg(use_unnormalized=True)
    score = compute_per_env_score(rollouts, cfg)
    expected = (rollouts.returns - rollouts.values).abs().squeeze(-1).mean(dim=0)
    assert torch.allclose(score, expected, atol=1e-6)


def test_score_gae_magnitude_normalized_uses_advantages():
    rollouts = _fake_rollout()
    cfg = GAEMagnitudeScoringCfg(use_unnormalized=False)
    score = compute_per_env_score(rollouts, cfg)
    expected = rollouts.advantages.abs().squeeze(-1).mean(dim=0)
    assert torch.allclose(score, expected, atol=1e-6)


def test_score_gae_signed_both_returns_signed_mean():
    rollouts = _fake_rollout()
    cfg = GAESignedScoringCfg(use_unnormalized=False, sign="both")
    score = compute_per_env_score(rollouts, cfg)
    expected = rollouts.advantages.squeeze(-1).mean(dim=0)
    assert torch.allclose(score, expected, atol=1e-6)


def test_score_gae_signed_positive_clamps_negatives_to_zero():
    rollouts = _fake_rollout()
    cfg = GAESignedScoringCfg(use_unnormalized=False, sign="positive")
    score = compute_per_env_score(rollouts, cfg)
    expected = rollouts.advantages.squeeze(-1).clamp_min(0.0).mean(dim=0)
    assert torch.allclose(score, expected, atol=1e-6)
    assert (score >= 0).all()


def test_score_gae_signed_negative_clamps_positives_to_zero():
    rollouts = _fake_rollout()
    cfg = GAESignedScoringCfg(use_unnormalized=False, sign="negative")
    score = compute_per_env_score(rollouts, cfg)
    expected = (-rollouts.advantages.squeeze(-1)).clamp_min(0.0).mean(dim=0)
    assert torch.allclose(score, expected, atol=1e-6)
    assert (score >= 0).all()


def test_score_td1_magnitude_matches_hand_computed():
    """Construct a tiny rollout with known r, v and verify ``|r + γV(s')-V(s)|``."""
    rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # [T=3, N=2]
    values = torch.tensor([[0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
    rollouts = RolloutData(rewards=rewards.unsqueeze(-1), values=values.unsqueeze(-1), gamma=0.5)
    cfg = TD1MagnitudeScoringCfg()

    score = compute_per_env_score(rollouts, cfg)
    # Per env, sum |delta_t| / (T-1):
    # env 0: t=0: |1 + 0.5*1 - 0|=1.5; t=1: |3 + 0.5*2 - 1|=3.0; mean = 2.25
    # env 1: t=0: |2 + 0.5*1 - 1|=1.5; t=1: |4 + 0.5*0 - 1|=3.0; mean = 2.25
    expected = torch.tensor([2.25, 2.25])
    assert torch.allclose(score, expected, atol=1e-6)


def test_score_td1_requires_t_at_least_two():
    rewards = torch.tensor([[1.0, 2.0]]).unsqueeze(-1)  # [1, 2, 1]
    values = torch.tensor([[0.0, 1.0]]).unsqueeze(-1)
    rollouts = RolloutData(rewards=rewards, values=values, gamma=0.99)
    cfg = TD1MagnitudeScoringCfg()
    with pytest.raises(ValueError, match=r"T >= 2"):
        compute_per_env_score(rollouts, cfg)


def test_score_td1_requires_gamma_in_rollout_data():
    """``gamma`` must come from RolloutData (the runner-supplied alg context)."""
    rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).unsqueeze(-1)
    values = torch.tensor([[0.0, 1.0], [1.0, 1.0]]).unsqueeze(-1)
    rollouts = RolloutData(rewards=rewards, values=values, gamma=None)
    cfg = TD1MagnitudeScoringCfg()
    with pytest.raises(ValueError, match=r"gamma"):
        compute_per_env_score(rollouts, cfg)


def test_compute_per_env_score_raises_on_unknown_cfg_type():
    with pytest.raises(TypeError, match="Unsupported scoring cfg type"):
        compute_per_env_score(RolloutData(), object())  # type: ignore[arg-type]


def test_compute_per_env_score_raises_on_missing_fields():
    cfg = GAEMagnitudeScoringCfg(use_unnormalized=True)
    with pytest.raises(ValueError, match="returns.*values"):
        compute_per_env_score(RolloutData(), cfg)


# ---------------------------------------------------------------------------
# Aggregation: RollingMean, EMA, LastValue
# ---------------------------------------------------------------------------


def _make_sampler(
    *,
    aggregation=None,
    prioritization=None,
    staleness=None,
    proposal=None,
    num_cells: int = 4,
) -> LevelSampler:
    cfg = LevelSamplerCfg(
        scoring=GAEMagnitudeScoringCfg(),  # not exercised in pure-aggregation tests
        aggregation=aggregation or RollingMeanAggregationCfg(history_len=8),
        prioritization=prioritization or ProportionalPrioritizationCfg(alpha=1.0, eps=1e-8, temperature=1.0),
        staleness=staleness,
        proposal=proposal or AlwaysReplayProposalCfg(),
    )
    return LevelSampler(cfg, num_cells=num_cells, device="cpu")


def test_rolling_mean_aggregation_appends_per_cell():
    sampler = _make_sampler(aggregation=RollingMeanAggregationCfg(history_len=8))
    sampler.push_per_env_scores(torch.tensor([0.5, 0.7, 0.3]), torch.tensor([0, 1, 0], dtype=torch.int64))
    # Cell 0 gets two writes (0.5, 0.3) -> mean 0.4. Cell 1 gets one (0.7).
    assert sampler.scores[0].item() == pytest.approx(0.4, abs=1e-6)
    assert sampler.scores[1].item() == pytest.approx(0.7, abs=1e-6)
    assert sampler.scores[2].item() == 0.0
    assert sampler.scores[3].item() == 0.0


def test_rolling_mean_aggregation_overflow_keeps_latest():
    sampler = _make_sampler(aggregation=RollingMeanAggregationCfg(history_len=4))
    # 6 writes to cell 0; only the last 4 should be kept.
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    cells = torch.zeros(6, dtype=torch.int64)
    sampler.push_per_env_scores(values, cells)
    # Last 4 values are [3, 4, 5, 6] -> mean 4.5.
    assert sampler.scores[0].item() == pytest.approx(4.5, abs=1e-6)


def test_ema_aggregation_full_replacement_at_alpha_one():
    sampler = _make_sampler(aggregation=EMAAggregationCfg(alpha=1.0))
    sampler.push_per_env_scores(torch.tensor([0.5]), torch.tensor([0], dtype=torch.int64))
    sampler.push_per_env_scores(torch.tensor([0.9]), torch.tensor([0], dtype=torch.int64))
    assert sampler.scores[0].item() == pytest.approx(0.9, abs=1e-6)


def test_ema_aggregation_blends_with_alpha_half():
    sampler = _make_sampler(aggregation=EMAAggregationCfg(alpha=0.5))
    sampler.push_per_env_scores(torch.tensor([1.0]), torch.tensor([0], dtype=torch.int64))
    # First push: score becomes (1-0.5)*0 + 0.5*1.0 = 0.5
    sampler.push_per_env_scores(torch.tensor([0.0]), torch.tensor([0], dtype=torch.int64))
    # Second push: score becomes (1-0.5)*0.5 + 0.5*0.0 = 0.25
    assert sampler.scores[0].item() == pytest.approx(0.25, abs=1e-6)


def test_ema_aggregates_multiple_envs_in_one_iter_via_mean():
    """Per-iter-mean semantics: many envs in one cell contribute their *mean*, not last-write."""
    sampler = _make_sampler(aggregation=EMAAggregationCfg(alpha=1.0))
    sampler.push_per_env_scores(torch.tensor([0.2, 0.4, 0.6]), torch.tensor([0, 0, 0], dtype=torch.int64))
    # alpha=1.0 + per-iter-mean => cell.score = mean([0.2, 0.4, 0.6]) = 0.4
    assert sampler.scores[0].item() == pytest.approx(0.4, abs=1e-6)


def test_ema_per_cell_iter_mean_with_mixed_cells():
    """Per-iter mean is computed per-cell -- envs in different cells don't get mixed."""
    sampler = _make_sampler(aggregation=EMAAggregationCfg(alpha=1.0), num_cells=4)
    sampler.push_per_env_scores(
        torch.tensor([1.0, 0.0, 0.5, 0.7]),
        torch.tensor([0, 0, 1, 2], dtype=torch.int64),
    )
    # cell 0: mean([1.0, 0.0]) = 0.5; cell 1: mean([0.5]) = 0.5; cell 2: mean([0.7]) = 0.7
    assert sampler.scores[0].item() == pytest.approx(0.5, abs=1e-6)
    assert sampler.scores[1].item() == pytest.approx(0.5, abs=1e-6)
    assert sampler.scores[2].item() == pytest.approx(0.7, abs=1e-6)
    assert sampler.scores[3].item() == 0.0  # cell 3 was unvisited


def test_ema_alpha_half_blends_iter_mean_with_old():
    """alpha < 1: blend the per-cell iter-mean with the old per-cell estimate."""
    sampler = _make_sampler(aggregation=EMAAggregationCfg(alpha=0.5))
    sampler.push_per_env_scores(torch.tensor([1.0, 1.0]), torch.tensor([0, 0], dtype=torch.int64))
    # iter 1 mean = 1.0; new score = 0.5 * 0 + 0.5 * 1.0 = 0.5
    assert sampler.scores[0].item() == pytest.approx(0.5, abs=1e-6)
    sampler.push_per_env_scores(torch.tensor([0.0, 0.0]), torch.tensor([0, 0], dtype=torch.int64))
    # iter 2 mean = 0.0; new score = 0.5 * 0.5 + 0.5 * 0.0 = 0.25
    assert sampler.scores[0].item() == pytest.approx(0.25, abs=1e-6)


def test_push_marks_visited_cells_as_seen():
    sampler = _make_sampler()
    assert sampler.unseen_mask.all().item()
    sampler.push_per_env_scores(torch.tensor([0.1, 0.2]), torch.tensor([1, 3], dtype=torch.int64))
    assert sampler.unseen_mask[0].item() is True
    assert sampler.unseen_mask[1].item() is False
    assert sampler.unseen_mask[2].item() is True
    assert sampler.unseen_mask[3].item() is False


def test_push_length_mismatch_raises():
    sampler = _make_sampler()
    with pytest.raises(ValueError, match="Length mismatch"):
        sampler.push_per_env_scores(torch.tensor([0.1, 0.2]), torch.tensor([0], dtype=torch.int64))


# ---------------------------------------------------------------------------
# Prioritization
# ---------------------------------------------------------------------------


def test_proportional_priority_with_eps_one_alpha_one_temp_one_is_identity():
    """``Proportional(α=1, ε=0, τ=1)`` should give probs ∝ score (canonical PER)."""
    sampler = _make_sampler(
        prioritization=ProportionalPrioritizationCfg(alpha=1.0, eps=0.0, temperature=1.0),
    )
    sampler.scores = torch.tensor([0.1, 0.4, 0.7, 0.2])
    sampler.unseen_mask.fill_(False)  # mark all seen so unseen-floor doesn't kick in via proposal

    p_s = sampler._compute_score_distribution()
    expected = sampler.scores / sampler.scores.sum()
    assert torch.allclose(p_s, expected, atol=1e-5)


def test_proportional_priority_alpha_zero_uniform():
    sampler = _make_sampler(
        prioritization=ProportionalPrioritizationCfg(alpha=0.0, eps=1e-8, temperature=1.0),
    )
    sampler.scores = torch.tensor([0.1, 0.5, 1.0, 0.25])

    p_s = sampler._compute_score_distribution()
    assert torch.allclose(p_s, torch.full((4,), 0.25), atol=1e-6)


def test_proportional_priority_eps_floor_lifts_zero_score_cells():
    """With ε=1e-3, an unvisited (score=0) cell still gets meaningful mass."""
    sampler = _make_sampler(
        prioritization=ProportionalPrioritizationCfg(alpha=1.0, eps=1e-3, temperature=1.0),
    )
    sampler.scores = torch.tensor([0.0, 1.0, 1.0, 1.0])

    p_s = sampler._compute_score_distribution()
    # cell 0 has weight (0 + 1e-3)^1 = 1e-3; others have weight ~1.001 each.
    # Ratio: cell 0 / cell 1 ≈ 1e-3 -- non-trivial floor, much larger than 1e-8.
    assert p_s[0].item() > 1e-4  # vs ~1e-9 with the legacy 1e-8 eps
    assert p_s[0].item() < 1e-2


def test_rank_priority_temperature_one_yields_harmonic():
    """``Rank(τ=1)`` => weight = 1/rank; probs are harmonic-distribution."""
    sampler = _make_sampler(prioritization=RankPrioritizationCfg(temperature=1.0))
    sampler.scores = torch.tensor([0.1, 0.4, 0.7, 0.2])

    p_s = sampler._compute_score_distribution()
    # Sorted descending: cell 2 (0.7), cell 1 (0.4), cell 3 (0.2), cell 0 (0.1).
    # Ranks: 2→1, 1→2, 3→3, 0→4. Weights: [1/4, 1/2, 1, 1/3].
    expected = torch.tensor([1 / 4, 1 / 2, 1.0, 1 / 3])
    expected = expected / expected.sum()
    assert torch.allclose(p_s, expected, atol=1e-6)


def test_rank_priority_low_temperature_concentrates_on_top():
    sampler = _make_sampler(prioritization=RankPrioritizationCfg(temperature=0.1))
    sampler.scores = torch.tensor([0.1, 0.4, 0.7, 0.2])

    p_s = sampler._compute_score_distribution()
    assert p_s.argmax().item() == 2  # highest-score cell
    assert p_s[2].item() > 0.9  # nearly all the mass


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------


def test_staleness_distribution_proportional_to_age():
    """``P_C ∝ (c - C_i)``; cell visited longer ago should have higher mass."""
    sampler = _make_sampler(
        staleness=StalenessCfg(
            staleness_coefficient=1.0,  # ignore P_S entirely so we can read P_C off the result
            transform=ProportionalPrioritizationCfg(alpha=1.0, eps=0.0, temperature=1.0),
        ),
    )
    # Visit cell 0 once, then cell 1 three times -- cell 0 is older.
    sampler.push_per_env_scores(torch.tensor([1.0]), torch.tensor([0], dtype=torch.int64))
    for _ in range(3):
        sampler.push_per_env_scores(torch.tensor([1.0]), torch.tensor([1], dtype=torch.int64))
    # Now global_visit_count = 4. last_visit_count: cell 0 -> 1, cell 1 -> 4, cells 2/3 -> 0.
    # Staleness: cell 0 -> 3, cell 1 -> 0, cells 2/3 -> 4.
    p_c = sampler._compute_staleness_distribution()
    expected_raw = torch.tensor([3.0, 0.0, 4.0, 4.0])
    expected = expected_raw / expected_raw.sum()
    assert torch.allclose(p_c, expected, atol=1e-5)


def test_staleness_mixing_blends_p_s_and_p_c():
    sampler = _make_sampler(
        staleness=StalenessCfg(
            staleness_coefficient=0.3,
            transform=ProportionalPrioritizationCfg(alpha=1.0, eps=0.0, temperature=1.0),
        ),
    )
    # Set up: visit cells with known ages, write distinct scores.
    sampler.push_per_env_scores(torch.tensor([1.0, 1.0, 1.0, 1.0]), torch.arange(4, dtype=torch.int64))
    # Now overwrite scores to known values for the test.
    sampler.scores = torch.tensor([0.1, 0.4, 0.7, 0.2])

    expected_p_s = sampler._compute_score_distribution()
    expected_p_c = sampler._compute_staleness_distribution()
    expected_replay = (1 - 0.3) * expected_p_s + 0.3 * expected_p_c

    p_replay = sampler._compute_replay_distribution()
    assert torch.allclose(p_replay, expected_replay, atol=1e-6)


def test_staleness_disabled_returns_p_s():
    sampler = _make_sampler(staleness=None)
    sampler.scores = torch.tensor([0.1, 0.4, 0.7, 0.2])
    sampler.unseen_mask.fill_(False)

    p_replay = sampler._compute_replay_distribution()
    p_s = sampler._compute_score_distribution()
    assert torch.allclose(p_replay, p_s, atol=1e-6)


# ---------------------------------------------------------------------------
# Proposal: replay vs new mixing
# ---------------------------------------------------------------------------


def test_always_replay_proposal_share_is_one():
    sampler = _make_sampler(proposal=AlwaysReplayProposalCfg())
    # Mark some cells seen to exercise the unseen-fallback condition.
    sampler.unseen_mask[:2] = False
    assert sampler._compute_replay_share() == 1.0


def test_proportionate_proposal_share_equals_proportion_seen():
    sampler = _make_sampler(proposal=ProportionateProposalCfg(warmup_threshold=0.0), num_cells=10)
    sampler.unseen_mask[:7] = False  # 0.7 seen
    share = sampler._compute_replay_share()
    assert share == pytest.approx(0.7, abs=1e-6)


def test_proportionate_proposal_warmup_blocks_replay():
    sampler = _make_sampler(proposal=ProportionateProposalCfg(warmup_threshold=0.5), num_cells=10)
    sampler.unseen_mask[:3] = False  # 0.3 seen, below threshold
    assert sampler._compute_replay_share() == 0.0


def test_unseen_distribution_is_uniform_over_unseen():
    sampler = _make_sampler(num_cells=5)
    sampler.unseen_mask[1] = False
    sampler.unseen_mask[3] = False
    p_unseen = sampler._compute_unseen_distribution()
    expected = torch.tensor([1 / 3, 0.0, 1 / 3, 0.0, 1 / 3])
    assert torch.allclose(p_unseen, expected, atol=1e-6)


def test_unseen_distribution_zeros_when_all_seen():
    sampler = _make_sampler(num_cells=4)
    sampler.unseen_mask.fill_(False)
    p_unseen = sampler._compute_unseen_distribution()
    assert torch.all(p_unseen == 0)


# ---------------------------------------------------------------------------
# End-to-end sampling smoke
# ---------------------------------------------------------------------------


def test_sample_default_cfg_produces_valid_distribution():
    """Default cfg (matches legacy ``advantage`` preset) should produce valid probs."""
    sampler = _make_sampler()  # default = ProportionalPrioritizationCfg, AlwaysReplay
    sampler.push_per_env_scores(torch.tensor([0.1, 0.5, 0.7, 0.3]), torch.arange(4, dtype=torch.int64))

    choices, probs, info = sampler.sample(num=100)

    assert choices.shape == (100,)
    assert probs.shape == (4,)
    assert torch.all(choices >= 0) and torch.all(choices < 4)
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.all(probs >= 0)
    assert "P_S" in info
    assert "P_replay" in info
    assert "replay_share" in info


def test_sample_with_proportionate_proposal_includes_unseen_mass():
    """A non-trivial proposal should give unseen cells nonzero probability."""
    sampler = _make_sampler(proposal=ProportionateProposalCfg(warmup_threshold=0.0), num_cells=10)
    # Visit only 5 of 10 cells.
    sampler.push_per_env_scores(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]), torch.arange(5, dtype=torch.int64))

    _, probs, _ = sampler.sample(num=10)

    # Unseen cells (5..9) should each carry at least the per-unseen share of the (1 - proportion_seen)
    # mass: that's 0.5 / 5 = 0.1 each.
    unseen_probs = probs[5:]
    assert (unseen_probs > 0.05).all()


def test_sample_choice_frequency_matches_probs():
    """Empirical sample frequency converges to the analytic ``probs`` (binomial check)."""
    sampler = _make_sampler()
    sampler.push_per_env_scores(torch.tensor([0.1, 0.4, 0.4, 0.1]), torch.arange(4, dtype=torch.int64))

    torch.manual_seed(0)
    n = 100_000
    choices, probs, _ = sampler.sample(num=n)
    counts = torch.bincount(choices, minlength=4).float() / n
    assert torch.allclose(counts, probs, atol=5e-3)


# ---------------------------------------------------------------------------
# State serialization
# ---------------------------------------------------------------------------


def test_state_dict_round_trip_rolling_mean():
    src = _make_sampler(aggregation=RollingMeanAggregationCfg(history_len=8))
    src.push_per_env_scores(torch.tensor([0.1, 0.5, 0.7]), torch.tensor([0, 1, 0], dtype=torch.int64))
    snapshot = src.state_dict()

    dst = _make_sampler(aggregation=RollingMeanAggregationCfg(history_len=8))
    dst.load_state_dict(snapshot)

    assert torch.equal(src.scores, dst.scores)
    assert torch.equal(src.unseen_mask, dst.unseen_mask)
    assert torch.equal(src._last_visit_count, dst._last_visit_count)
    assert src._global_visit_count == dst._global_visit_count
    assert torch.equal(src._buffer, dst._buffer)
    assert torch.equal(src._pointer, dst._pointer)
    assert torch.equal(src._size, dst._size)


def test_state_dict_round_trip_ema():
    src = _make_sampler(aggregation=EMAAggregationCfg(alpha=0.5))
    src.push_per_env_scores(torch.tensor([0.5]), torch.tensor([0], dtype=torch.int64))
    snapshot = src.state_dict()

    dst = _make_sampler(aggregation=EMAAggregationCfg(alpha=0.5))
    dst.load_state_dict(snapshot)

    assert torch.equal(src.scores, dst.scores)
    assert "buffer" not in snapshot  # EMA path doesn't store rolling state


def test_reset_state_zeros_everything():
    sampler = _make_sampler()
    sampler.push_per_env_scores(torch.tensor([1.0, 2.0]), torch.tensor([0, 1], dtype=torch.int64))
    sampler.reset_state()

    assert sampler.scores.sum().item() == 0
    assert sampler.unseen_mask.all().item()
    assert sampler._last_visit_count.sum().item() == 0
    assert sampler._global_visit_count == 0
    assert sampler._buffer.sum().item() == 0
    assert sampler._pointer.sum().item() == 0
    assert sampler._size.sum().item() == 0


# ---------------------------------------------------------------------------
# Lifecycle hooks
# ---------------------------------------------------------------------------


def test_after_policy_update_is_noop():
    sampler = _make_sampler()
    sampler.push_per_env_scores(torch.tensor([1.0]), torch.tensor([0], dtype=torch.int64))
    score_before = sampler.scores.clone()
    sampler.after_policy_update()
    assert torch.equal(sampler.scores, score_before)


# ---------------------------------------------------------------------------
# Routing diagnostics
# ---------------------------------------------------------------------------


def test_routing_diagnostics_returns_expected_keys_without_staleness():
    sampler = _make_sampler(staleness=None)
    sampler.push_per_env_scores(torch.tensor([0.5, 0.5, 0.5, 0.5]), torch.arange(4, dtype=torch.int64))

    diag = sampler.routing_diagnostics()

    expected_keys = {
        "score_mean",
        "score_max",
        "score_min",
        "score_std",
        "score_dynamic_range",
        "unseen_frac",
        "proportion_seen",
        "score_dist_entropy",
        "score_top1_prob",
        "replay_dist_entropy",
        "unseen_dist_entropy",
        "mass_on_unseen",
        "replay_share",
    }
    assert set(diag.keys()) == expected_keys
    # Staleness-only keys absent
    assert "staleness_dist_entropy" not in diag
    assert "effective_rho" not in diag
    assert "staleness_cv" not in diag


def test_routing_diagnostics_returns_staleness_keys_when_enabled():
    sampler = _make_sampler(staleness=StalenessCfg(staleness_coefficient=0.2))
    sampler.push_per_env_scores(torch.tensor([1.0, 1.0]), torch.tensor([0, 1], dtype=torch.int64))

    diag = sampler.routing_diagnostics()

    for key in ("staleness_dist_entropy", "effective_rho", "staleness_cv"):
        assert key in diag
    assert diag["effective_rho"] == pytest.approx(0.2)


def test_routing_diagnostics_score_dist_entropy_at_collapse():
    """Single-cell collapse yields entropy ≈ 0; uniform yields ≈ log(N)."""
    sampler = _make_sampler(
        prioritization=ProportionalPrioritizationCfg(alpha=20.0, eps=1e-8, temperature=1.0),
        num_cells=4,
    )
    # Concentrate all mass on cell 2 via large alpha.
    sampler.scores = torch.tensor([0.1, 0.2, 0.9, 0.3])
    sampler.unseen_mask.fill_(False)

    diag = sampler.routing_diagnostics()
    # alpha=20 + uneven scores: P_S concentrates on cell 2 -> entropy near 0.
    assert diag["score_dist_entropy"] < 0.1


def test_routing_diagnostics_score_dist_entropy_at_uniform():
    """Equal scores produce uniform P_S, entropy ≈ log(num_cells)."""
    import math as _math

    sampler = _make_sampler(num_cells=4)
    sampler.scores = torch.tensor([0.5, 0.5, 0.5, 0.5])
    sampler.unseen_mask.fill_(False)

    diag = sampler.routing_diagnostics()
    assert diag["score_dist_entropy"] == pytest.approx(_math.log(4), abs=1e-4)


def test_routing_diagnostics_mass_on_unseen_under_proportionate_proposal():
    """``ProportionateProposalCfg`` routes ``1 - proportion_seen`` of mass to unseen cells."""
    sampler = _make_sampler(proposal=ProportionateProposalCfg(warmup_threshold=0.0), num_cells=10)
    # Mark 4 of 10 cells as seen.
    sampler.unseen_mask[:4] = False

    diag = sampler.routing_diagnostics()
    # mass_on_unseen = (1 - replay_share) * sum(P_unseen) = (1 - 0.4) * 1.0 = 0.6
    assert diag["mass_on_unseen"] == pytest.approx(0.6, abs=1e-6)
    assert diag["replay_share"] == pytest.approx(0.4, abs=1e-6)


def test_routing_diagnostics_mass_on_unseen_zero_under_always_replay():
    """``AlwaysReplayProposalCfg`` puts all draws on the replay distribution."""
    sampler = _make_sampler(proposal=AlwaysReplayProposalCfg(), num_cells=5)
    sampler.unseen_mask[:2] = False

    diag = sampler.routing_diagnostics()
    assert diag["mass_on_unseen"] == 0.0
    assert diag["replay_share"] == 1.0


def test_routing_diagnostics_staleness_cv_high_when_uneven_visit_history():
    """Cells visited at very different recency should produce high staleness CV."""
    sampler = _make_sampler(staleness=StalenessCfg(staleness_coefficient=0.5), num_cells=5)
    # Visit cell 0 once early, then cell 1 many times -- cell 0 is much older.
    sampler.push_per_env_scores(torch.tensor([1.0]), torch.tensor([0], dtype=torch.int64))
    for _ in range(20):
        sampler.push_per_env_scores(torch.tensor([1.0]), torch.tensor([1], dtype=torch.int64))

    diag = sampler.routing_diagnostics()
    # Cell 0 was visited at iter 1, cell 1 at iter 21. Cells 2,3,4 never visited
    # (last_visit_count=0, staleness=21). High CV expected.
    assert diag["staleness_cv"] > 0.1


def test_routing_diagnostics_unseen_dist_entropy_matches_log_num_unseen():
    """``unseen_dist_entropy`` equals ``log(num_unseen_cells)`` when any unseen exist."""
    import math as _math

    sampler = _make_sampler(num_cells=10)
    sampler.unseen_mask[:7] = False  # 3 unseen

    diag = sampler.routing_diagnostics()
    assert diag["unseen_dist_entropy"] == pytest.approx(_math.log(3), abs=1e-4)


def test_routing_diagnostics_unseen_dist_entropy_zero_when_all_seen():
    sampler = _make_sampler(num_cells=4)
    sampler.unseen_mask.fill_(False)

    diag = sampler.routing_diagnostics()
    assert diag["unseen_dist_entropy"] == 0.0


def test_routing_diagnostics_score_dynamic_range_only_uses_seen_cells():
    """``score_dynamic_range`` is computed over seen cells only, not zero-score unseen."""
    sampler = _make_sampler(num_cells=5)
    # Visit 3 cells with known scores; leave 2 unvisited.
    sampler.push_per_env_scores(
        torch.tensor([0.1, 1.0, 5.0]),
        torch.tensor([0, 1, 2], dtype=torch.int64),
    )

    diag = sampler.routing_diagnostics()
    # max=5.0, min(seen)=0.1 -> ratio 50.
    assert diag["score_dynamic_range"] == pytest.approx(50.0, abs=1e-4)


def test_routing_diagnostics_score_dynamic_range_zero_when_no_seen_cells():
    sampler = _make_sampler(num_cells=4)
    diag = sampler.routing_diagnostics()
    assert diag["score_dynamic_range"] == 0.0


def test_routing_diagnostics_score_top1_prob_at_collapse():
    """Score collapse: a single cell holds nearly all P_S mass; top1_prob ≈ 1."""
    sampler = _make_sampler(
        prioritization=ProportionalPrioritizationCfg(alpha=20.0, eps=1e-8, temperature=1.0),
        num_cells=4,
    )
    sampler.scores = torch.tensor([0.1, 0.2, 0.9, 0.3])
    sampler.unseen_mask.fill_(False)

    diag = sampler.routing_diagnostics()
    assert diag["score_top1_prob"] > 0.99


def test_routing_diagnostics_score_top1_prob_at_uniform():
    sampler = _make_sampler(num_cells=4)
    sampler.scores = torch.tensor([0.5, 0.5, 0.5, 0.5])
    sampler.unseen_mask.fill_(False)

    diag = sampler.routing_diagnostics()
    # Uniform P_S over 4 cells => top1 = 0.25
    assert diag["score_top1_prob"] == pytest.approx(0.25, abs=1e-4)
