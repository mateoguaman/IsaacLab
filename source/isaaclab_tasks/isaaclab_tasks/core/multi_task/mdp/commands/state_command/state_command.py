# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Domain-agnostic state command: track selected task rows, delegate semantics.

The command owns only per-env lifecycle (the selected table row per env, the
policy-facing command observation, the error tensor, and the curriculum
success-rate buffer). Everything domain-specific lives behind two collaborators:

- a **task table** (built from ``cfg.task_table``) exposing ``num_tasks`` and
  ``gather(task_rows) -> (spawn_states, target_states)``;
- a **payload** (built from ``cfg.payload``) that owns its own buffers and
  implements target writing, observation/error computation, success/hold
  semantics, threshold reporting, metric aggregation, and debug drawing.

Factory assembly and legged locomotion supply different tables and payloads but
share this shell.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm

from isaaclab_tasks.core.multi_task.curriculum import set_reset_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .state_command_cfg import StateCommandCfg


class StateCommand(CommandTerm):
    """Track the selected task row per env and delegate command-state semantics."""

    cfg: StateCommandCfg

    def __init__(self, cfg: StateCommandCfg, env: ManagerBasedEnv):
        # table first, then the payload bound to that table (one construction
        # order for every domain). The factory table reads ``cfg.payload`` for
        # its reset-asset set, so it needs no constructed payload. Both are built
        # BEFORE ``super().__init__`` because the base initializer invokes
        # ``set_debug_vis`` -> ``_set_debug_vis_impl``, which delegates to the
        # payload. The builders take ``env``/``cfg`` directly, so no ``self``
        # state from the base is needed yet.
        self.table = cfg.task_table.class_type(cfg, env)
        self._payload = cfg.payload.class_type(cfg, env, self.table)

        super().__init__(cfg, env)

        self.states_relative: bool = cfg.states_relative
        self.randomize_command_indices: bool = cfg.randomize_command_indices

        # per-env selected table row; the curriculum may bind and overwrite it
        self.cmd_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # per-task curriculum success rate (written by the curriculum's monitor)
        self.success_rates = torch.zeros(self.table.num_tasks, device=self.device, dtype=torch.float32)

        self._command = torch.zeros(self.num_envs, self._payload.command_dim, device=self.device)
        self._err = torch.empty(self.num_envs, self._payload.error_dim, device=self.device)
        for name in self._payload.error_names:
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

        self._resample_command(torch.arange(self.num_envs, device=self.device))
        self._update_command()

    def __str__(self) -> str:
        msg = f"{type(self).__name__}:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return the per-task curriculum aggregate so a resume keeps the learned distribution.

        Only :attr:`success_rates` (the per-task success aggregate, shared with the
        curriculum's success monitor) is persisted. The per-env :attr:`cmd_indices` are
        intentionally excluded: under Design B envs restart cleanly and resample their goal on
        the first reset.
        """
        return {"success_rates": self.success_rates.clone()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore the per-task success aggregate in place (shared alias with the monitor)."""
        if "success_rates" in state:
            self.success_rates.copy_(state["success_rates"])

    @property
    def payload(self):
        """The active domain payload (its public methods carry domain semantics)."""
        return self._payload

    @property
    def command(self) -> torch.Tensor:
        """Policy-facing command observation written by the active payload."""
        return self._command

    @property
    def error(self) -> torch.Tensor:
        """Per-env, per-group command error written by the active payload."""
        return self._err

    @property
    def command_std(self) -> torch.Tensor:
        """Per-env success thresholds for the currently-bound rows ``[N, error_dim]``."""
        return self._payload.command_std()

    def get_task_done(self) -> torch.Tensor:
        """Per-env task-success flag (payload-defined)."""
        return self._payload.get_task_done()

    def get_task_reward(self) -> torch.Tensor:
        """Per-env task-success reward (payload-defined)."""
        return self._payload.get_task_reward()

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.long()
        if self.randomize_command_indices:
            self.cmd_indices[env_ids] = torch.randint(0, self.table.num_tasks, (env_ids.numel(),), device=self.device)

        task_rows = self.cmd_indices[env_ids]
        spawn_states, target_states = self.table.gather(task_rows)
        # explicit frame: env-local states are lifted to world by ``env_origins``
        # (per-asset for the spawn write, per-payload for the target)
        target_origin = self._env.scene.env_origins[env_ids] if self.states_relative else None
        self._payload.resample(env_ids, task_rows, target_states, target_origin)
        set_reset_state(self._env, spawn_states, env_ids, self._payload.reset_assets, is_relative=self.states_relative)
        self._payload.update(0.0, self._command, self._err)

    def _update_command(self) -> None:
        self._payload.update(self._env.step_dt, self._command, self._err)

    def _update_metrics(self) -> None:
        for group_idx, name in enumerate(self._payload.error_names):
            self.metrics[name] = self._err[:, group_idx]
        self._payload.log_metrics(self._env, self.success_rates)

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        self._payload.set_debug_vis(debug_vis)

    def _debug_vis_callback(self, event) -> None:
        self._payload.debug_visualize(self._env)
