# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime sampling strategies.

Each strategy binds its own input signals via configclass ``*_bind`` fields,
resolved at construction time against the ``bind_ns`` kwargs forwarded from
:class:`Sampler`. Strategies that need no inputs simply discard their args.

:meth:`SamplingStrategy.score` writes a per-item non-negative score tensor into
the pre-allocated ``out`` buffer; the sampler sums weighted scores and
normalizes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch
from tensordict import TensorDict

from ..state_layout import StateLayout

if TYPE_CHECKING:
    from .sampling_strategies_cfg import (
        BetaSamplingStrategyCfg,
        FrontierSamplingStrategyCfg,
        UniformSamplingStrategyCfg,
        ValueShiftSamplingStrategyCfg,
    )


class SamplingStrategy(Protocol):
    """Per-item scorer for sampler probability construction.

    Strategies declare their runtime inputs as ``*_bind`` cfg fields (string
    expressions); the sampler resolves them against a shared ``bind_ns`` dict
    at strategy ctor time. :meth:`score` then writes non-negative scores into
    ``out`` using the bound references.
    """

    name: str
    """Short identifier used in diagnostic / log keys."""

    def score(self, out: torch.Tensor) -> None:
        """Write ``[num_items]`` non-negative unnormalized scores into ``out``."""
        ...


class BetaSamplingStrategy:
    """Per-item Beta-kernel score peaked at a target success rate."""

    name = "beta"

    def __init__(self, cfg: BetaSamplingStrategyCfg, layout: StateLayout, **bind_ns) -> None:
        del layout
        target = max(0.0, min(1.0, float(cfg.target)))
        kappa = max(0.0, float(cfg.kappa))
        self._a = 1.0 + kappa * target
        self._b = 1.0 + kappa * (1.0 - target)
        self._rates: torch.Tensor = eval(cfg.success_rate_bind, bind_ns)  # noqa: S307

    def score(self, out: torch.Tensor) -> None:
        out.copy_(self._rates)
        out.pow_(self._a - 1.0)
        out.mul_((1.0 - self._rates).pow(self._b - 1.0))


class FrontierSamplingStrategy:
    """Per-task frontier score from a kNN graph over the task feature space."""

    name = "frontier"

    def __init__(self, cfg: FrontierSamplingStrategyCfg, layout: StateLayout, **bind_ns) -> None:
        self._dilation_steps = max(1, int(cfg.dilation_steps))
        self._rates: torch.Tensor = eval(cfg.success_rate_bind, bind_ns)  # noqa: S307
        spawn_feat = layout.coords[layout.spawn_index]
        if layout.target_index is None:
            task_features = spawn_feat
        else:
            target_feat = layout.coords[layout.target_index]
            task_features = torch.cat([spawn_feat, target_feat], dim=-1)

        from scipy.spatial import cKDTree

        k = int(cfg.k)
        if k < 1:
            raise ValueError(f"k must be >= 1; got {k}.")
        n = int(task_features.shape[0])
        device = task_features.device
        self_idx = torch.arange(n, device=device, dtype=torch.long).unsqueeze(-1)
        self._knn = self_idx.expand(n, k).clone()

        partition_keys = [None] if layout.task_partition is None else torch.unique(layout.task_partition).tolist()
        for partition_key in partition_keys:
            if partition_key is None:
                member_idx = torch.arange(n, device=device, dtype=torch.long)
                features = task_features
            else:
                member_idx = (layout.task_partition == partition_key).nonzero(as_tuple=False).squeeze(-1)
                features = task_features[member_idx]

            n_member = int(features.shape[0])
            if n_member <= 1:
                continue

            k_eff = min(k, n_member - 1)
            features_np = features.detach().cpu().numpy()
            _, idx = cKDTree(features_np).query(features_np, k=k_eff + 1)
            local_knn = torch.as_tensor(idx[:, 1 : k_eff + 1], device=device, dtype=torch.long)
            if k_eff < k:
                pad = torch.arange(n_member, device=device, dtype=torch.long).unsqueeze(-1).expand(n_member, k - k_eff)
                local_knn = torch.cat([local_knn, pad], dim=1)
            self._knn[member_idx] = member_idx[local_knn]

    def score(self, out: torch.Tensor) -> None:
        rates = self._rates
        s_dil = rates
        for _ in range(self._dilation_steps):
            neighbor_max = s_dil[self._knn].amax(dim=-1)
            s_dil = torch.maximum(s_dil, neighbor_max)
        out.copy_(s_dil)
        out.sub_(rates).clamp_min_(0.0)
        out.mul_(1.0 - rates)


class UniformSamplingStrategy:
    """Constant 1.0 per item -- the trivial baseline / floor."""

    name = "uniform"

    def __init__(self, cfg: UniformSamplingStrategyCfg, layout: StateLayout, **bind_ns) -> None:
        del cfg, layout, bind_ns

    def score(self, out: torch.Tensor) -> None:
        out.fill_(1.0)


class ValueShiftSamplingStrategy:
    """Per-state critic value-shift score.

    Maintains a fixed observation cache (one entry per discretized command
    state) built once at ``__init__``. An external RL algorithm (e.g.
    :class:`ValueShiftPPO`) evaluates the critic on this cache every update
    and writes the per-state ``|V_new - V_prev|`` magnitude into
    :attr:`diff_val`. The strategy's :meth:`score` simply copies that signal
    into the sampler's output buffer.

    The cache fill loop drives state transitions by writing
    :attr:`cmd_indices` and calling :attr:`resample_command_fn`; between each
    batch it runs ``env.sim.forward()`` + ``env.scene.update(dt=0.0)`` so
    sensor inputs (e.g. height_scan via ``FastTerrainScanner`` reading
    ``body_pos_w``) reflect the freshly written pose before
    :meth:`get_critic_obs_fn` is queried.
    """

    name = "value_shift"

    def __init__(self, cfg: ValueShiftSamplingStrategyCfg, layout: StateLayout, **bind_ns) -> None:
        del layout
        env = bind_ns["env"]
        self._sim = env.sim
        self._scene = env.scene
        self.state_buffer: torch.Tensor = eval(cfg.state_buffer_bind, bind_ns)  # noqa: S307
        self.cmd_indices: torch.Tensor = eval(cfg.cmd_indices_bind, bind_ns)  # noqa: S307
        self.resample_command_fn = eval(cfg.resample_command_fn_bind, bind_ns)  # noqa: S307
        self.get_critic_obs_fn = eval(cfg.get_critic_obs_fn_bind, bind_ns)  # noqa: S307

        assert isinstance(self.state_buffer, torch.Tensor) and self.state_buffer.shape[0] > 0, (
            "ValueShift state_buffer must be a non-empty Tensor."
        )
        assert self.cmd_indices.dtype == torch.long, (
            f"ValueShift cmd_indices must be torch.long; got {self.cmd_indices.dtype}."
        )
        assert callable(self.resample_command_fn) and callable(self.get_critic_obs_fn)

        self.observation_cache: TensorDict = self._create_observation_cache()
        n = self.observation_cache.batch_size[0]
        # Allocate cur/diff on the OBS device (matches what critic reads),
        # not state_buffer.device — they can differ in multi-GPU setups.
        device = next(iter(self.observation_cache.values())).device
        self.cur_val = torch.zeros(n, device=device)
        self.diff_val = torch.zeros(n, device=device)

    def _create_observation_cache(self) -> TensorDict:
        """Sweep through every state in :attr:`state_buffer` and cache obs."""
        n = int(self.state_buffer.shape[0])
        # Probe once for shapes/dtypes; flush kinematics first so the probe
        # reflects current (post-init) env state rather than uninitialized buffers.
        self._sim.forward()
        self._scene.update(dt=0.0)
        probe: dict[str, torch.Tensor] = self.get_critic_obs_fn()
        num_envs = next(iter(probe.values())).shape[0]
        cache_dict: dict[str, torch.Tensor] = {
            group: torch.zeros(
                (n, *tensor.shape[1:]),
                device=tensor.device,
                dtype=tensor.dtype,
            )
            for group, tensor in probe.items()
        }
        device = self.cmd_indices.device
        count = 0
        while count < n:
            batch = min(num_envs, n - count)
            env_ids = torch.arange(batch, device=device, dtype=torch.long)
            state_ids = torch.arange(count, count + batch, device=device, dtype=torch.long)
            self.cmd_indices[env_ids] = state_ids
            self.resample_command_fn(env_ids)
            # Refresh kinematics + asset data buffers so sensors (e.g. height_scan
            # via FastTerrainScanner reading body_pos_w) see the freshly written pose.
            self._sim.forward()
            self._scene.update(dt=0.0)
            obs: dict[str, torch.Tensor] = self.get_critic_obs_fn()
            for group in cache_dict:
                cache_dict[group][state_ids] = obs[group][:batch].detach()
            count += batch
        return TensorDict(cache_dict, batch_size=[n])

    def score(self, out: torch.Tensor) -> None:
        out.copy_(self.diff_val)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Snapshot the critic value-shift buffers so a resume keeps the learned signal.

        Restoring :attr:`cur_val` (the previous-update critic values) means the first
        post-resume :class:`ValueShiftPPO` update computes a real ``|V_new - V_prev|`` instead
        of a spurious ``|V_new|`` spike; :attr:`diff_val` keeps the current per-state score so
        the first resumed rollout samples from the learned distribution.
        """
        return {"cur_val": self.cur_val.clone(), "diff_val": self.diff_val.clone()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore the value-shift buffers in place (preserves the ValueShiftPPO aliases)."""
        self.cur_val.copy_(state["cur_val"])
        self.diff_val.copy_(state["diff_val"])
