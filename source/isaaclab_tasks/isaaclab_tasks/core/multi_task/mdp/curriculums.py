# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase

from isaaclab_tasks.core.multi_task.curriculum import (
    LevelSampler,
    RolloutData,
    SamplerCfg,
    StateLayoutCfg,
    SuccessMonitorCfg,
    compute_per_env_score,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class success_rate_sampler(ManagerTermBase):
    """Update item success rates and sample the next item indices."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._log_counter = 0

        self.success_rates: torch.Tensor = eval(cfg.params["success_rates_bind"])  # noqa: S307
        self.sample_indices: torch.Tensor = eval(cfg.params["sample_indices_bind"])  # noqa: S307
        self.success: torch.Tensor = eval(cfg.params["success_bind"])  # noqa: S307

        monitor_cfg: SuccessMonitorCfg = cfg.params["success_monitor_cfg"]
        monitor_cfg.num_monitored_data = int(self.success_rates.numel())
        monitor_cfg.device = env.device
        monitor_cfg.max_updates = env.num_envs if monitor_cfg.max_updates is None else monitor_cfg.max_updates
        self.success_monitor = monitor_cfg.class_type(monitor_cfg, self.success_rates)

        self._sampling_cfg: SamplerCfg = cfg.params["sampling"]
        if self._sampling_cfg.max_samples is None:
            self._sampling_cfg.max_samples = env.num_envs
        layout_cfg: StateLayoutCfg = cfg.params["layout"]
        self._sampler = self._sampling_cfg.class_type(
            self._sampling_cfg, layout_cfg.build(env), env=env, success_rates=self.success_rates
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        success_rates_bind: str,
        sample_indices_bind: str,
        success_bind: str,
        layout: StateLayoutCfg,
        sampling: SamplerCfg,
        success_monitor_cfg: SuccessMonitorCfg,
        sampler_visual_logger: Callable[..., None] | None = None,
        sampler_visual_log_period: int = 1000,
    ):
        if env_ids.numel() == 0:
            return {"success": self.success_rates.mean()}

        prev_idx = self.sample_indices[env_ids]
        self.success_monitor.success_update(prev_idx, self.success[env_ids])

        num_samples = self._sampling_cfg.max_samples if self._sampling_cfg.warp else len(env_ids)
        probs, choices = self._sampler.probabilities_and_sample(num_samples)
        self.sample_indices[env_ids] = choices[: len(env_ids)]

        if sampler_visual_logger is not None:
            # The RL wrapper resets the env once (running this term) before the runner starts
            # wandb, so counter 0 lands while ``wandb.run`` is None and that upload is dropped.
            # Hold the schedule until the logging backend is live, then emit on the first
            # post-init call (counter 0) and every period after -- otherwise the first image
            # waits a full period.
            try:
                import wandb  # noqa: PLC0415

                backend_live = wandb.run is not None
            except ImportError:
                backend_live = False
            if backend_live:
                if self._log_counter % sampler_visual_log_period == 0:
                    sampler_visual_logger(env, self._sampler, self.success_rates, probs)
                self._log_counter += 1

        return {"success": self.success_rates.mean()}

    def state_dict(self) -> dict:
        """Return the monitor + sampler state so a resume continues the learned curriculum.

        The bound aggregate ``success_rates`` and the per-env ``sample_indices`` are not
        persisted here: ``success_rates`` is the same tensor the command term owns and
        round-trips, and per-env sample indices reset cleanly on resume (Design B).
        """
        return {
            "success_monitor": self.success_monitor.state_dict(),
            "sampler": self._sampler.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore the monitor + sampler state produced by :meth:`state_dict`."""
        if "success_monitor" in state:
            self.success_monitor.load_state_dict(state["success_monitor"])
        if "sampler" in state:
            self._sampler.load_state_dict(state["sampler"])


class linear_terrain_levels(ManagerTermBase):
    """Monotonic per-env terrain-level curriculum (quadruped-scoped baseline).

    Each env holds a difficulty level encoded in its current task's terrain tile
    (``tile_index = row * num_cols + col``, where ``row`` is the difficulty level). After each
    episode the env is promoted (``+1`` level) on success or demoted (``-1``) when it closed only
    a small fraction of its goal distance, staying within the same terrain column, then re-samples
    a task uniformly from the target ``(level, col)`` tile.

    Promotion rule::

        move_up = success_term[env_ids]

    Demotion rule::

        move_down = (closed_fraction < demotion_fraction) & ~move_up

    where ``closed_fraction = 1 - remaining / total`` uses the base-frame goal command as the
    remaining horizontal distance and the task table's spawn→target distance as the total.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.success_rates: torch.Tensor = eval(cfg.params["success_rates_bind"])  # noqa: S307
        self.sample_indices: torch.Tensor = eval(cfg.params["sample_indices_bind"])  # noqa: S307
        self.success: torch.Tensor = eval(cfg.params["success_bind"])  # noqa: S307
        self._goal_term = env.command_manager.get_term(cfg.params["goal_term_name"])
        self._table = self._goal_term.table

        gen = env.scene.terrain.cfg.terrain_generator
        self.num_rows = int(gen.num_rows)
        self.num_cols = int(gen.num_cols)

        monitor_cfg: SuccessMonitorCfg = cfg.params["success_monitor_cfg"]
        monitor_cfg.num_monitored_data = int(self.success_rates.numel())
        monitor_cfg.device = env.device
        monitor_cfg.max_updates = env.num_envs if monitor_cfg.max_updates is None else monitor_cfg.max_updates
        self.success_monitor = monitor_cfg.class_type(monitor_cfg, self.success_rates)

        # CSR layout mapping each terrain tile -> the task rows whose spawn lies on it, so a
        # promoted/demoted env can draw a fresh task on its new (level, col) tile.
        tile_index = self._table.tile_index.to(torch.long)
        n_tiles = self.num_rows * self.num_cols
        self._tile_task_values = torch.argsort(tile_index)
        counts = torch.bincount(tile_index, minlength=n_tiles)
        self._tile_task_offsets = torch.zeros(n_tiles + 1, device=env.device, dtype=torch.long)
        self._tile_task_offsets[1:] = counts.cumsum(0)

        # Per-env difficulty tile, lazily initialized from each env's first task and persisted so
        # the curriculum position continues across a resume.
        self._current_tile = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    def _sample_task_on_tiles(self, tiles: torch.Tensor) -> torch.Tensor:
        """Sample one task row uniformly per tile; returns ``-1`` for tiles with no tasks."""
        starts = self._tile_task_offsets[tiles]
        counts = self._tile_task_offsets[tiles + 1] - starts
        nonempty = counts > 0
        rand = (torch.rand(tiles.shape[0], device=tiles.device) * counts.clamp_min(1).float()).long()
        gather_idx = (starts + rand).clamp_(max=self._tile_task_values.numel() - 1)
        sampled = self._tile_task_values[gather_idx]
        return torch.where(nonempty, sampled, torch.full_like(sampled, -1))

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        success_rates_bind: str,
        sample_indices_bind: str,
        success_bind: str,
        goal_term_name: str,
        success_monitor_cfg: SuccessMonitorCfg,
        demotion_fraction: float = 0.5,
    ):
        if env_ids.numel() == 0:
            return {"success": self.success_rates.mean()}

        prev_idx = self.sample_indices[env_ids]
        success = self.success[env_ids]
        self.success_monitor.success_update(prev_idx, success)

        # Lazily seed the per-env difficulty tile from the env's current task.
        uninit = self._current_tile[env_ids] < 0
        if uninit.any():
            self._current_tile[env_ids[uninit]] = self._table.tile_index[prev_idx[uninit]].to(torch.long)
        current_tile = self._current_tile[env_ids]
        level = current_tile // self.num_cols
        col = current_tile % self.num_cols

        # closed_fraction: how much of the goal distance the env closed this episode.
        remaining = env.command_manager.get_command(goal_term_name)[env_ids, :2].norm(dim=1)
        spawn_states, target_states = self._table.gather(prev_idx)
        total = (target_states[:, :2] - spawn_states[:, :2]).norm(dim=1).clamp_min(1e-6)
        closed_fraction = 1.0 - remaining / total

        move_up = success
        move_down = (closed_fraction < demotion_fraction) & ~move_up
        new_level = torch.clamp(level + move_up.long() - move_down.long(), 0, self.num_rows - 1)
        new_tile = new_level * self.num_cols + col

        new_idx = self._sample_task_on_tiles(new_tile)
        empty = new_idx < 0  # keep the current task/tile where the target tile has no tasks
        self._current_tile[env_ids] = torch.where(empty, current_tile, new_tile)
        self.sample_indices[env_ids] = torch.where(empty, prev_idx, new_idx).to(self.sample_indices.dtype)

        return {"success": self.success_rates.mean()}

    def state_dict(self) -> dict:
        """Return the monitor buffers and per-env difficulty tile for a resume."""
        return {"success_monitor": self.success_monitor.state_dict(), "current_tile": self._current_tile.clone()}

    def load_state_dict(self, state: dict) -> None:
        """Restore the monitor buffers and per-env difficulty tile in place."""
        if "success_monitor" in state:
            self.success_monitor.load_state_dict(state["success_monitor"])
        if "current_tile" in state:
            self._current_tile.copy_(state["current_tile"])


class level_sampler_curriculum(ManagerTermBase):
    """Prioritized Level Replay (PLR) curriculum term.

    Hosts a :class:`~isaaclab_tasks.core.multi_task.curriculum.LevelSampler` that scores each
    terrain task by the rollout regret (GAE magnitude by default) the policy experiences on it,
    then samples the next tasks from a prioritization + staleness + replay-vs-unseen mixture over
    those scores. The per-env regret is pushed once per training iteration by
    :class:`OnPolicyRunnerWithLevelSampler`, which duck-types this term via its
    :attr:`level_sampler` attribute and :meth:`update_with_rollouts`.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.success_rates: torch.Tensor = eval(cfg.params["success_rates_bind"])  # noqa: S307
        self.sample_indices: torch.Tensor = eval(cfg.params["sample_indices_bind"])  # noqa: S307
        self.success: torch.Tensor = eval(cfg.params["success_bind"])  # noqa: S307
        self._goal_term = env.command_manager.get_term(cfg.params["goal_term_name"])
        self._table = self._goal_term.table

        monitor_cfg: SuccessMonitorCfg = cfg.params["success_monitor_cfg"]
        monitor_cfg.num_monitored_data = int(self.success_rates.numel())
        monitor_cfg.device = env.device
        monitor_cfg.max_updates = env.num_envs if monitor_cfg.max_updates is None else monitor_cfg.max_updates
        self.success_monitor = monitor_cfg.class_type(monitor_cfg, self.success_rates)

        self.level_sampler = LevelSampler(
            cfg=cfg.params["sampling"], num_cells=int(self._table.num_tasks), device=env.device
        )

    def update_with_rollouts(self, rollouts: RolloutData) -> None:
        """Push per-env rollout-regret scores to the level sampler (once per training iter).

        Called by :class:`OnPolicyRunnerWithLevelSampler` after ``alg.compute_returns`` so the
        rollout advantages/returns/values are populated and the policy has not yet stepped.
        """
        per_env_score = compute_per_env_score(rollouts, self.level_sampler.cfg.scoring)
        env_ids = torch.arange(per_env_score.shape[0], device=per_env_score.device, dtype=torch.int64)
        cell_ids = self.sample_indices[env_ids].to(torch.int64)
        self.level_sampler.push_per_env_scores(per_env_score, cell_ids)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        success_rates_bind: str,
        sample_indices_bind: str,
        success_bind: str,
        goal_term_name: str,
        sampling,
        success_monitor_cfg: SuccessMonitorCfg,
    ):
        if env_ids.numel() == 0:
            return {"success": self.success_rates.mean()}
        prev_idx = self.sample_indices[env_ids]
        self.success_monitor.success_update(prev_idx, self.success[env_ids])
        choices, _probs, _info = self.level_sampler.sample(num=env_ids.numel())
        self.sample_indices[env_ids] = choices.to(self.sample_indices.dtype)
        return {"success": self.success_rates.mean()}

    def state_dict(self) -> dict:
        """Return the monitor buffers and the level sampler's replay state for a resume."""
        return {
            "success_monitor": self.success_monitor.state_dict(),
            "level_sampler": self.level_sampler.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore the monitor buffers and level sampler replay state."""
        if "success_monitor" in state:
            self.success_monitor.load_state_dict(state["success_monitor"])
        if "level_sampler" in state:
            self.level_sampler.load_state_dict(state["level_sampler"])
