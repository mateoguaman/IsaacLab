# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PLR-style level sampler over a fixed cell set.

Composed of typed cfg axes (see :mod:`level_sampler_cfg`); each axis is
dispatched independently inside this class.

Pipeline: aggregation (rolling mean / EMA) -> prioritization (proportional /
rank) -> optional staleness mixing (:math:`(1 - \\rho) P_S + \\rho P_C`) ->
proposal mix between the replay distribution and uniform-over-unseen cells.
"""

from __future__ import annotations

import torch

from .level_sampler_cfg import (
    AlwaysReplayProposalCfg,
    EMAAggregationCfg,
    LevelSamplerCfg,
    PrioritizationCfg,
    ProportionalPrioritizationCfg,
    ProportionateProposalCfg,
    RankPrioritizationCfg,
    RollingMeanAggregationCfg,
)


def _shannon_entropy(p: torch.Tensor) -> float:
    """Shannon entropy in nats, with a numerical clamp inside ``log``."""
    return float(-(p * p.clamp_min(1e-12).log()).sum().item())


class LevelSampler:
    """PLR-style sampler over a fixed set of cells.

    Args:
        cfg: Composed sampler configuration.
        num_cells: Number of discrete cells (e.g., (level, type, spawn,
            target) combinations).
        device: Torch device for all internal buffers.
    """

    def __init__(self, cfg: LevelSamplerCfg, num_cells: int, device: str | torch.device):
        self.cfg = cfg
        self.num_cells = int(num_cells)
        self.device = torch.device(device)

        # Per-cell score estimate (output of aggregation stage).
        self.scores = torch.zeros(self.num_cells, device=self.device, dtype=torch.float32)
        # Cells never visited get this floor mask cleared.
        self.unseen_mask = torch.ones(self.num_cells, device=self.device, dtype=torch.bool)

        # Aggregation-specific state.
        agg = cfg.aggregation
        if isinstance(agg, RollingMeanAggregationCfg):
            self._buffer = torch.zeros((self.num_cells, agg.history_len), device=self.device, dtype=torch.float32)
            self._pointer = torch.zeros(self.num_cells, device=self.device, dtype=torch.int64)
            self._size = torch.zeros(self.num_cells, device=self.device, dtype=torch.int64)
            self._history_len = int(agg.history_len)
        # EMA needs no extra state beyond ``self.scores``.

        # Staleness state (allocated unconditionally so state_dict is shape-stable).
        self._global_visit_count = 0
        self._last_visit_count = torch.zeros(self.num_cells, device=self.device, dtype=torch.int64)

    # ------------------------------------------------------------------
    # Aggregation: fold per-env scores into per-cell estimates
    # ------------------------------------------------------------------

    def push_per_env_scores(self, per_env_scores: torch.Tensor, cell_ids: torch.Tensor) -> None:
        """Aggregate a batch of per-env scores attributed to specific cells.

        Args:
            per_env_scores: Per-env score tensor of shape ``[N]``, on
                :attr:`device`. Output of
                :func:`level_sampler_scoring.compute_per_env_score`.
            cell_ids: Cell index each env's score is attributed to, shape
                ``[N]``, dtype int64. Each env contributes one entry to its
                cell's aggregator.
        """
        if per_env_scores.numel() != cell_ids.numel():
            raise ValueError(
                f"Length mismatch: per_env_scores has {per_env_scores.numel()} entries but cell_ids"
                f" has {cell_ids.numel()}."
            )
        per_env_scores = per_env_scores.to(device=self.device, dtype=torch.float32)
        cell_ids = cell_ids.to(device=self.device, dtype=torch.int64)

        agg = self.cfg.aggregation
        if isinstance(agg, RollingMeanAggregationCfg):
            self._update_rolling_mean(per_env_scores, cell_ids)
        elif isinstance(agg, EMAAggregationCfg):
            self._update_ema(per_env_scores, cell_ids, agg)
        else:
            raise TypeError(f"Unsupported aggregation cfg type: {type(agg).__name__}")

        # Mark visited cells as seen and update staleness clock.
        unique_cells = torch.unique(cell_ids)
        self.unseen_mask[unique_cells] = False
        self._global_visit_count += 1
        self._last_visit_count[unique_cells] = self._global_visit_count

    def _update_rolling_mean(self, scores: torch.Tensor, cell_ids: torch.Tensor) -> None:
        """Append each (cell, score) entry to the cell's circular buffer.

        Multiple entries to the same cell within one call are appended in
        order, clamped to ``history_len`` if they overflow.
        """
        unique_cells, inv, counts = torch.unique(cell_ids, return_inverse=True, return_counts=True)
        counts_clamped = counts.clamp(max=self._history_len).to(torch.int64)

        ptrs = self._pointer[unique_cells]
        # Sort scores so each unique cell's entries are contiguous; then for cells whose count
        # exceeds ``history_len``, keep only the most recent ``history_len`` entries.
        ordered = scores[torch.argsort(inv)]
        splits = torch.split(ordered, counts.tolist())
        clamped_values = torch.cat([grp[-n:] for grp, n in zip(splits, counts_clamped.tolist())])

        cell_indices = torch.repeat_interleave(unique_cells, counts_clamped)
        buf_indices = torch.cat(
            [
                torch.arange(start, start + n, dtype=torch.int64, device=self.device) % self._history_len
                for start, n in zip(ptrs.tolist(), counts_clamped.tolist())
            ]
        )
        self._buffer.index_put_((cell_indices, buf_indices), clamped_values)

        self._pointer.index_add_(0, unique_cells, counts_clamped)
        self._pointer = self._pointer % self._history_len
        self._size.index_add_(0, unique_cells, counts_clamped)
        self._size = self._size.clamp(max=self._history_len)
        self.scores = self._buffer.sum(dim=1) / self._size.clamp(min=1).float()

    def _update_ema(
        self,
        scores: torch.Tensor,
        cell_ids: torch.Tensor,
        agg: EMAAggregationCfg,
    ) -> None:
        """Per-iter-mean then EMA: ``score_i <- (1 - alpha) * old + alpha * mean(this_iter_for_i)``.

        Multiple envs landing in the same cell within one push are first averaged
        into a single per-cell value for this iter; the cell's running estimate is
        then blended toward that iter-mean. With ``alpha=1.0`` the cell's score
        becomes exactly this iter's mean for that cell -- the PLR-faithful
        "score = most recent visit" behavior.
        """
        alpha = float(agg.alpha)
        unique_cells, inverse = torch.unique(cell_ids, return_inverse=True)
        per_cell_sum = torch.zeros(unique_cells.numel(), device=self.device, dtype=torch.float32)
        per_cell_count = torch.zeros(unique_cells.numel(), device=self.device, dtype=torch.float32)
        per_cell_sum.scatter_add_(0, inverse, scores)
        per_cell_count.scatter_add_(0, inverse, torch.ones_like(scores))
        iter_mean = per_cell_sum / per_cell_count
        self.scores[unique_cells] = (1.0 - alpha) * self.scores[unique_cells] + alpha * iter_mean

    # ------------------------------------------------------------------
    # Sampling: compute P_replay = (1-rho) P_S + rho P_C, then mix proposal
    # ------------------------------------------------------------------

    def sample(self, num: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Draw ``num`` cell indices according to the current sampler state.

        Args:
            num: Number of draws to return.

        Returns:
            A 3-tuple ``(choices, probs, info)`` where:
              * ``choices`` is an int64 tensor of length ``num`` -- the sampled
                cell indices.
              * ``probs`` is a float32 tensor of length :attr:`num_cells` -- the
                effective per-cell probability used to draw, *averaged over the
                proposal mix*. Note: with a proposal that mixes replay and new,
                the effective probability for an unseen cell includes its share
                of the new-mass.
              * ``info`` carries diagnostic tensors keyed by name (``P_S``,
                ``P_C``, ``replay_share``, ``proportion_seen``, etc.) for
                downstream logging.
        """
        replay_probs = self._compute_replay_distribution()  # P_S (mixed with P_C if staleness)
        unseen_probs = self._compute_unseen_distribution()  # uniform over unseen, zeros elsewhere
        replay_share = self._compute_replay_share()  # scalar in [0, 1]

        effective_probs = replay_share * replay_probs + (1.0 - replay_share) * unseen_probs
        effective_probs = self._safe_normalize(effective_probs)

        choices = torch.multinomial(effective_probs, num, replacement=True).to(torch.int64)

        info = {
            "P_S": self._compute_score_distribution(),
            "P_replay": replay_probs,
            "P_unseen": unseen_probs,
            "replay_share": torch.tensor(replay_share, device=self.device),
            "proportion_seen": torch.tensor(
                float((~self.unseen_mask).sum()) / float(self.num_cells), device=self.device
            ),
        }
        if self.cfg.staleness is not None:
            info["P_C"] = self._compute_staleness_distribution()
        return choices, effective_probs, info

    def _compute_replay_distribution(self) -> torch.Tensor:
        """Score-prioritized distribution :math:`P_S`, optionally mixed with :math:`P_C`."""
        p_s = self._compute_score_distribution()
        if self.cfg.staleness is None:
            return p_s
        p_c = self._compute_staleness_distribution()
        rho = float(self.cfg.staleness.staleness_coefficient)
        return (1.0 - rho) * p_s + rho * p_c

    def _compute_score_distribution(self) -> torch.Tensor:
        return self._apply_prioritization(self.scores, self.cfg.prioritization)

    def _compute_staleness_distribution(self) -> torch.Tensor:
        # Staleness is age in visit-count units. Cells never visited have last_visit_count=0,
        # so their staleness equals the global counter -- i.e., they are maximally stale,
        # which is a natural unseen-floor for the staleness path.
        staleness = (self._global_visit_count - self._last_visit_count).clamp_min(0).float()
        return self._apply_prioritization(staleness, self.cfg.staleness.transform)

    def _compute_unseen_distribution(self) -> torch.Tensor:
        """Uniform distribution over unseen cells; zeros if none unseen."""
        unseen_count = int(self.unseen_mask.sum())
        if unseen_count == 0:
            return torch.zeros(self.num_cells, device=self.device, dtype=torch.float32)
        out = torch.zeros(self.num_cells, device=self.device, dtype=torch.float32)
        out[self.unseen_mask] = 1.0 / float(unseen_count)
        return out

    def _compute_replay_share(self) -> float:
        """Fraction of draws to take from the replay distribution.

        Concretely: ``1 - P(new)``. :class:`AlwaysReplayProposalCfg` returns 1
        unconditionally. :class:`ProportionateProposalCfg` anneals up with
        ``proportion_seen`` after the warmup threshold is cleared (returns 0
        during warmup).
        """
        proposal = self.cfg.proposal
        proportion_seen = float((~self.unseen_mask).sum()) / float(self.num_cells)
        if isinstance(proposal, AlwaysReplayProposalCfg):
            return 1.0
        if isinstance(proposal, ProportionateProposalCfg):
            if proportion_seen < proposal.warmup_threshold:
                return 0.0
            return proportion_seen
        raise TypeError(f"Unsupported proposal cfg type: {type(proposal).__name__}")

    def _safe_normalize(self, weights: torch.Tensor) -> torch.Tensor:
        """Normalize weights to a valid probability vector; fall back to uniform on degenerate input."""
        total = weights.sum()
        if not torch.isfinite(total) or total.item() < 1e-12:
            return torch.full_like(weights, 1.0 / float(self.num_cells))
        return weights / total

    # ------------------------------------------------------------------
    # Prioritization: shared dispatcher over PrioritizationCfg variants
    # ------------------------------------------------------------------

    def _apply_prioritization(self, raw_scores: torch.Tensor, prio: PrioritizationCfg) -> torch.Tensor:
        """Map a per-cell raw score vector into a normalized priority distribution."""
        if isinstance(prio, ProportionalPrioritizationCfg):
            return self._proportional_priority(raw_scores, prio)
        if isinstance(prio, RankPrioritizationCfg):
            return self._rank_priority(raw_scores, prio)
        raise TypeError(f"Unsupported prioritization cfg type: {type(prio).__name__}")

    def _proportional_priority(self, raw_scores: torch.Tensor, prio: ProportionalPrioritizationCfg) -> torch.Tensor:
        # ``prio.eps`` doubles as the priority floor and the inner-log guard so a single
        # knob covers both. When eps == 0 we still need a tiny constant for log(0).
        floor = float(prio.eps) if prio.eps > 0 else 1e-30
        weights = (raw_scores.clamp_min(0.0) + float(prio.eps)).pow(float(prio.alpha)).clamp_min(floor)
        logits = torch.log(weights + floor)
        return torch.softmax(logits / float(prio.temperature), dim=0)

    def _rank_priority(self, raw_scores: torch.Tensor, prio: RankPrioritizationCfg) -> torch.Tensor:
        # rank=1 for the largest score; ties broken by torch.argsort stability.
        n = raw_scores.numel()
        order = torch.argsort(raw_scores, descending=True)
        ranks = torch.empty(n, device=self.device, dtype=torch.float32)
        ranks[order] = torch.arange(1, n + 1, device=self.device, dtype=torch.float32)
        weights = ranks.pow(-1.0 / float(prio.temperature))
        return weights / weights.sum()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def routing_diagnostics(self) -> dict[str, float]:
        """Cheap per-iter sampler-state metrics for the curriculum's ``Routing/*`` log.

        Returned keys (no prefix; the caller adds one):

        * ``score_mean`` / ``score_max`` / ``score_min`` / ``score_std``: per-cell
          aggregated score summary.
        * ``score_dynamic_range``: ``max / min`` over **seen** cells (clamped).
          Catches "all scores washed to near-equal" failures where entropy
          looks fine but the prio is operating on noise.
        * ``unseen_frac``: fraction of cells that have never been visited.
        * ``proportion_seen``: ``1 - unseen_frac``, kept as a separate key for
          plotting convenience.
        * ``score_dist_entropy``: Shannon entropy of :math:`P_S` (in nats).
          ``0`` = single-cell collapse, ``log(num_cells)`` = uniform.
        * ``score_top1_prob``: maximum probability mass on any single cell
          under :math:`P_S`. Complements entropy: high when collapsed, low
          when uniform.
        * ``replay_dist_entropy``: Shannon entropy of the replay distribution
          (= :math:`P_S` mixed with :math:`\\rho P_C` if staleness is on).
        * ``unseen_dist_entropy``: Shannon entropy of the uniform-over-unseen
          distribution. Equals ``log(num_unseen)`` when any unseen exist; else 0.
        * ``mass_on_unseen``: total sampling mass routed through the
          unseen-cell branch of the proposal: ``(1 - replay_share) * sum(P_unseen)``.
        * ``replay_share``: scalar :math:`P(\\text{replay})` from the proposal.

        With staleness enabled, additionally:

        * ``staleness_dist_entropy``: Shannon entropy of :math:`P_C`.
        * ``effective_rho``: configured staleness coefficient (echoes back so
          sweep configs surface in the log).
        * ``staleness_cv``: coefficient of variation
          (:math:`\\sigma / \\mu`) of per-cell staleness across cells.
        """
        p_s = self._compute_score_distribution()
        p_replay = self._compute_replay_distribution()
        p_unseen = self._compute_unseen_distribution()
        replay_share = self._compute_replay_share()
        proportion_seen = float((~self.unseen_mask).sum()) / float(self.num_cells)

        # Dynamic range over seen cells: catches "all scores washed together" failures
        # where entropy looks fine but the actual ratio max/min is near 1.
        seen_scores = self.scores[~self.unseen_mask]
        if seen_scores.numel() > 0:
            score_dynamic_range = float((seen_scores.max() / seen_scores.abs().min().clamp_min(1e-8)).item())
        else:
            score_dynamic_range = 0.0

        out: dict[str, float] = {
            "score_mean": float(self.scores.mean().item()),
            "score_max": float(self.scores.max().item()),
            "score_min": float(self.scores.min().item()),
            "score_std": float(self.scores.std().item()) if self.scores.numel() > 1 else 0.0,
            "score_dynamic_range": score_dynamic_range,
            "unseen_frac": float(self.unseen_mask.float().mean().item()),
            "proportion_seen": proportion_seen,
            "score_dist_entropy": _shannon_entropy(p_s),
            "score_top1_prob": float(p_s.max().item()),
            "replay_dist_entropy": _shannon_entropy(p_replay),
            "unseen_dist_entropy": _shannon_entropy(p_unseen),
            "mass_on_unseen": float((1.0 - replay_share) * p_unseen.sum().item()),
            "replay_share": float(replay_share),
        }

        if self.cfg.staleness is not None:
            p_c = self._compute_staleness_distribution()
            out["staleness_dist_entropy"] = _shannon_entropy(p_c)
            out["effective_rho"] = float(self.cfg.staleness.staleness_coefficient)
            staleness = (self._global_visit_count - self._last_visit_count).float()
            mean = staleness.mean()
            if mean.item() > 0:
                out["staleness_cv"] = float((staleness.std() / mean).item())
            else:
                out["staleness_cv"] = 0.0

        return out

    # ------------------------------------------------------------------
    # State serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Snapshot all serializable state to a dict on CPU."""
        state: dict = {
            "scores": self.scores.detach().cpu().clone(),
            "unseen_mask": self.unseen_mask.detach().cpu().clone(),
            "last_visit_count": self._last_visit_count.detach().cpu().clone(),
            "global_visit_count": int(self._global_visit_count),
        }
        if isinstance(self.cfg.aggregation, RollingMeanAggregationCfg):
            state["buffer"] = self._buffer.detach().cpu().clone()
            state["pointer"] = self._pointer.detach().cpu().clone()
            state["size"] = self._size.detach().cpu().clone()
        return state

    def load_state_dict(self, state: dict) -> None:
        """Restore from a snapshot produced by :meth:`state_dict`. Mutates buffers in place."""
        self.scores.copy_(state["scores"].to(self.scores.device))
        self.unseen_mask.copy_(state["unseen_mask"].to(self.unseen_mask.device))
        self._last_visit_count.copy_(state["last_visit_count"].to(self._last_visit_count.device))
        self._global_visit_count = int(state["global_visit_count"])
        if isinstance(self.cfg.aggregation, RollingMeanAggregationCfg):
            self._buffer.copy_(state["buffer"].to(self._buffer.device))
            self._pointer.copy_(state["pointer"].to(self._pointer.device))
            self._size.copy_(state["size"].to(self._size.device))

    def reset_state(self) -> None:
        """Zero all per-cell state. Useful between eval episodes."""
        self.scores.zero_()
        self.unseen_mask.fill_(True)
        self._last_visit_count.zero_()
        self._global_visit_count = 0
        if isinstance(self.cfg.aggregation, RollingMeanAggregationCfg):
            self._buffer.zero_()
            self._pointer.zero_()
            self._size.zero_()

    # ------------------------------------------------------------------
    # Lifecycle hook -- runner extension calls this after each policy update
    # ------------------------------------------------------------------

    def after_policy_update(self) -> None:
        """Hook called after each PPO update completes. Currently a no-op."""
        return
