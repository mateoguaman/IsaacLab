# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase

from isaaclab_tasks.core.multi_task.curriculum import SamplerCfg, StateLayoutCfg, SuccessMonitorCfg

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
