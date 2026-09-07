# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pipeline orchestrator for geometry-constrained articulation retargeting."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING

import newton.ik as ik
import numpy as np
import torch
import warp as wp

from ...kinematics import NewtonKinematics
from .buffer import RetargetBuffer
from .cfg import RetargetPipelineCfg
from .sampler_base import SamplerBase, SamplerOutput, SamplerSizing, compute_sampler_sizing

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Re-exports so existing ``from .pipeline import ...`` call sites keep
# resolving the same names.
__all__ = [
    "CriterionFn",
    "RetargetPipeline",
    "SamplerBase",
    "SamplerOutput",
    "SamplerSizing",
    "compute_sampler_sizing",
    "resolve_description_path",
    "resolve_foot_body_names",
]

CriterionFn = Callable[[RetargetBuffer, int], torch.Tensor]
"""Signature for a validation criterion.

Args:
    buffer: The retarget buffer with IK results populated.
    n_active: Number of geometry-valid candidates (first ``n_active`` rows).

Returns:
    Boolean tensor of shape ``[n_active]`` -- ``True`` = passes this criterion.
"""


_TIMING_LABELS: dict[str, str] = {
    # Pipeline-level phases (keys set by :meth:`RetargetPipeline._time`).
    "sampler": "sampler (polygon candidate gen)",
    "ik_build": "IK build (solver + objectives)",
    "ik_solve": "IK solve (batched)",
    "fk_eval": "FK eval (post-solve)",
    "criteria": "criteria (acceptance checks)",
    # Sub-phases of :class:`Sampler`. Keys unmatched here
    # fall back to their last dotted segment, so custom samplers still
    # render readably even without an entry.
    "sampler.morph": "morph-patch pool",
    "sampler.morph.setup": "allocate warp arrays",
    "sampler.morph.rasterize": "rasterize terrain",
    "sampler.morph.validity": "patch flatness check",
    "sampler.morph.candidates": "foot-reach candidate patches",
    "sampler.morph.fps": "FPS-thin morph pool",
    "sampler.neighbors": "assemble foot neighborhoods",
    "sampler.polygon_build": "stitch feet → polygons",
    "sampler.sampler_fps": "FPS-thin polygon pool",
    "sampler.prepare_ik": "build IK seeds",
}
"""Human-readable labels for timing keys in :attr:`RetargetPipeline._timings`."""


def _validate_results(
    buffer: RetargetBuffer,
    criteria: dict[str, CriterionFn],
) -> tuple[dict[str, int], torch.Tensor]:
    """Run user-defined acceptance criteria with per-candidate waterfall.

    Each criterion is a callable ``(buffer, N) -> bool[N]``. Criteria
    are evaluated in insertion order; each candidate is attributed to
    its first failing criterion.

    Side-effect free — the caller is responsible for updating
    ``buffer._ik_valid`` / ``num_ik_valid`` / ``num_final_valid`` from
    the returned mask.

    Args:
        buffer: Retarget buffer with ``joint_q_result`` populated and
            FK already evaluated.
        criteria: Ordered mapping from criterion name to callable.

    Returns:
        ``(reject, cum_pass)`` where ``reject`` maps criterion name to
        the number of rejections plus an ``"ok"`` key for survivors,
        and ``cum_pass`` is the per-candidate ``bool[N]`` mask of
        all-criteria-passing rows.
    """
    N = buffer.num_geometry_valid
    device = buffer.device
    if N == 0:
        return {}, torch.zeros(0, device=device, dtype=torch.bool)

    masks: dict[str, torch.Tensor] = {}
    for name, fn in criteria.items():
        masks[name] = fn(buffer, N)

    cum_pass = torch.ones(N, device=device, dtype=torch.bool)
    reject: dict[str, int] = {}
    for name, mask in masks.items():
        failed_here = cum_pass & ~mask
        reject[name] = int(failed_here.sum())
        cum_pass = cum_pass & mask
    reject["ok"] = int(cum_pass.sum())

    return reject, cum_pass


def resolve_foot_body_names(foot_body_names: Sequence[str] | str, body_names: Sequence[str]) -> list[str]:
    """Resolve a foot-body spec against available body names.

    Args:
        foot_body_names: Either exact body names or a regex matched with
            :func:`re.fullmatch`.
        body_names: Available body names.

    Returns:
        Exact body names in model order.

    Raises:
        ValueError: If the spec resolves to no body names or references
            missing exact names.
    """
    if isinstance(foot_body_names, str):
        regex = re.compile(foot_body_names)
        resolved = [name for name in body_names if regex.fullmatch(name)]
        if not resolved:
            raise ValueError(f"Foot body regex '{foot_body_names}' matched no bodies. Available: {list(body_names)}")
        return resolved

    missing = [name for name in foot_body_names if name not in body_names]
    if missing:
        raise ValueError(f"Foot body names not found: missing={missing}, available={list(body_names)}")
    return list(foot_body_names)


def resolve_description_path(spawn_cfg) -> str:
    """Return the robot description file a spawn config points at.

    :class:`~isaaclab.sim.UsdFileCfg` names the field ``usd_path`` while
    :class:`~isaaclab.sim.UrdfFileCfg` names it ``asset_path``. Both are
    accepted by :class:`~isaaclab_tasks.core.multi_task.kinematics.NewtonKinematics`,
    so callers only need the path.

    Args:
        spawn_cfg: Spawn configuration of the robot articulation.

    Returns:
        Path or URL of the description file.

    Raises:
        TypeError: If the spawn config exposes neither field.
    """
    for field in ("usd_path", "asset_path"):
        path = getattr(spawn_cfg, field, None)
        if path:
            return str(path)
    raise TypeError(
        f"{type(spawn_cfg).__name__} exposes no 'usd_path' or 'asset_path'; cannot locate the robot description."
    )


class RetargetPipeline:
    """Orchestrates the staged retargeting pipeline.

    Constructed from a single :class:`RetargetPipelineCfg` that nests
    the kinematics, sampler, and foot specification.  The pipeline
    builds the :class:`NewtonKinematics`, sampler, and standard IK
    objectives internally.

    Args:
        cfg: Full pipeline configuration.
        env: Environment used to resolve :attr:`RetargetPipelineCfg.asset_cfg`.
        device: Warp device string used when resolving :attr:`RetargetPipelineCfg.asset_cfg`.
    """

    def __init__(self, cfg: RetargetPipelineCfg, env: ManagerBasedEnv | None = None, device: str | None = None):
        if cfg.asset_cfg is not None:
            if env is None:
                raise ValueError("RetargetPipelineCfg.asset_cfg requires RetargetPipeline(..., env=env).")
            from isaaclab.utils.assets import check_file_path, retrieve_file_path

            articulation_cfg = env.scene[cfg.asset_cfg.name].cfg
            usd_path = resolve_description_path(articulation_cfg.spawn)
            status = check_file_path(usd_path)
            if status == 0:
                raise FileNotFoundError(f"Robot description not found: {usd_path}")
            if status == 2:
                usd_path = retrieve_file_path(usd_path, force_download=False)

            cfg = cfg.replace(kin=cfg.kin.copy())
            cfg.kin.usd_path = usd_path
            cfg.kin.default_pos = (0.0, 0.0, articulation_cfg.init_state.pos[2])
            cfg.kin.default_joint_pos = articulation_cfg.init_state.joint_pos
            cfg.kin.device = device if device is not None else cfg.kin.device

        self.cfg = cfg

        self.kin = NewtonKinematics(cfg.kin)
        self.foot_body_names = resolve_foot_body_names(cfg.foot_body_names, self.kin.body_names)
        self.foot_body_ids = self.kin.find_body_indices(self.foot_body_names)

        self.sampler = cfg.sampler.class_type(cfg.sampler, self.kin, self.foot_body_ids)

        # Buffer is allocated lazily on the first ``run()`` call -- sized
        # from the sampler's yield-rate sizing for that call's ``n_desired``.
        # Subsequent calls reuse the buffer if it still fits; otherwise it
        # is reallocated to the larger sizing.
        self.buffer: RetargetBuffer | None = None
        self._fk_joint_qd: wp.array | None = None
        self._fk_body_qd: wp.array | None = None
        self._timings: dict[str, float] = {}
        self._reject_geo: dict[str, int] = {}
        self._reject_val: dict[str, int] = {}
        self._ik_iterations_used: int = 0
        self._solver_costs: torch.Tensor | None = None
        self._n_diverged: int = 0
        self._n_ik_problems: int = 0
        """IK problem count from the last run — one problem per accepted
        polygon."""
        self._n_desired: int = 0
        """Target placement count from the last :meth:`run`. Used by
        :attr:`rejection_summary` to explain the sizing cascade."""
        self._sizing: SamplerSizing | None = None
        """Back-cascaded sampler sizing from the last :meth:`run`. Used
        by :attr:`rejection_summary` to report the polygon budget."""
        self._run_time: float = 0.0
        """Total wall time of the last :meth:`run` call (CUDA-synced at
        entry/exit). Used by :attr:`rejection_summary` to attribute the
        gap between tracked phases and overall run time."""
        self._sampler_diagnostics: dict[str, object] = {}
        """Per-call diagnostics dict from the last sampler invocation.

        Opaque, sampler-specific. Consumed by offline metrics tools; not
        part of the public pipeline API.
        """
        self._chunk_profile: list[dict] = []
        """Per-chunk timing + memory snapshots from the last :meth:`run`
        call's IK loop. Populated only when CUDA is available; surfaced
        via :attr:`chunk_profile_summary` for orchestration analysis.
        """
        self._chunk_profile_meta: dict[str, object] = {}
        """Run-level context (N, chunk_size, max_iters) that pairs with
        :attr:`_chunk_profile` for summary rendering.
        """

    def _ensure_buffer(self, n_desired: int) -> None:
        """Allocate (or grow) the retarget buffer for a given ``n_desired``.

        Sized at the *post-FPS* workload (``sizing.ik_capacity``), not the
        pre-FPS polygon pool (``sizing.max_neighborhoods``). The polygon-FPS
        oversample lives in cheap per-polygon tensors local to the sampler
        and never needs the buffer's per-body slots.
        """
        needed = self.sampler.sizing(n_desired).ik_capacity
        if self.buffer is not None and self.buffer.max_candidates >= needed:
            return
        self.buffer = RetargetBuffer(
            max_candidates=needed,
            joint_coord_count=self.kin.model.joint_coord_count,
            num_bodies=self.kin.model.body_count,
            num_contacts=len(self.foot_body_ids),
            device=self.kin.device,
        )

    @contextmanager
    def _time(self, name: str):
        """Record wall time for a pipeline phase, with CUDA sync on enter/exit."""
        if self.kin.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.kin.device.startswith("cuda"):
                torch.cuda.synchronize()
            self._timings[name] = self._timings.get(name, 0.0) + (time.perf_counter() - t0)

    def _snapshot_memory(self) -> dict[str, float]:
        """Snapshot CUDA + torch memory state in MiB.

        ``cuda_free`` reflects the driver-level free pool, which counts
        Warp's allocations + any other process on the device -- so the
        delta across chunks captures Warp pool churn that ``torch_alloc``
        misses entirely.
        """
        if not self.kin.device.startswith("cuda"):
            return {}
        mib = 1024.0 * 1024.0
        free_b, total_b = torch.cuda.mem_get_info()
        return {
            "torch_alloc_mib": torch.cuda.memory_allocated() / mib,
            "torch_reserved_mib": torch.cuda.memory_reserved() / mib,
            "torch_peak_alloc_mib": torch.cuda.max_memory_allocated() / mib,
            "cuda_free_mib": free_b / mib,
            "cuda_total_mib": total_b / mib,
        }

    def _build_objectives(
        self,
        N: int,
        wp_mesh: wp.Mesh,
    ) -> tuple[list, list[ik.IKObjectivePosition], list, ik.IKObjectivePosition, ik.IKObjectiveRotation]:
        """Build standard + extra IK objectives for ``N`` problems."""
        device = self.kin.device

        contact_objs = [
            ik.IKObjectivePosition(
                link_index=fid,
                link_offset=wp.vec3(0, 0, 0),
                target_positions=wp.zeros(N, dtype=wp.vec3, device=device),
                weight=self.cfg.foot_contact_weight,
            )
            for fid in self.foot_body_ids
        ]
        base_pos_obj = ik.IKObjectivePosition(
            link_index=0,
            link_offset=wp.vec3(0, 0, 0),
            target_positions=wp.zeros(N, dtype=wp.vec3, device=device),
            weight=self.cfg.base_pos_weight,
        )
        base_rot_obj = ik.IKObjectiveRotation(
            link_index=0,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.zeros(N, dtype=wp.vec4, device=device),
            weight=self.cfg.base_rot_weight,
        )
        jl_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.kin.model.joint_limit_lower,
            joint_limit_upper=self.kin.model.joint_limit_upper,
            weight=10.0,
        )

        # Sole orientation. Only worth building when a sole has enough extent
        # for its tilt to matter; a point foot leaves this at zero weight.
        contact_rot_objs: list = []
        if self.cfg.foot_rotation_weight > 0.0:
            contact_rot_objs = [
                ik.IKObjectiveRotation(
                    link_index=fid,
                    link_offset_rotation=wp.quat_identity(),
                    target_rotations=wp.zeros(N, dtype=wp.vec4, device=device),
                    weight=self.cfg.foot_rotation_weight,
                )
                for fid in self.foot_body_ids
            ]

        all_objs = [*contact_objs, *contact_rot_objs, base_pos_obj, base_rot_obj, jl_obj]

        for obj_cfg in self.cfg.extra_objectives:
            all_objs.append(obj_cfg.class_type(obj_cfg, self, wp_mesh))

        return all_objs, contact_objs, contact_rot_objs, base_pos_obj, base_rot_obj

    def _build_criteria(self, wp_mesh: wp.Mesh) -> dict[str, CriterionFn]:
        """Build the acceptance-criteria dict from ``cfg.criteria``.

        Called once per ``run()`` so criteria can close over the current
        ``wp_mesh`` and the pipeline's per-run state (e.g. solver costs).
        Preserves list order in the resulting dict so rejection buckets
        attribute failures to the first criterion a candidate violates.
        """
        return {crit_cfg.name: crit_cfg.class_type(crit_cfg, self, wp_mesh) for crit_cfg in self.cfg.criteria}

    def run(
        self,
        wp_mesh: wp.Mesh,
        origin: np.ndarray,
        n_desired: int,
        criteria: dict[str, CriterionFn] | None = None,
    ) -> RetargetBuffer:
        """Run the full pipeline.

        Args:
            wp_mesh: Terrain warp mesh.
            origin: Terrain origin ``[3]``.
            n_desired: Number of valid results desired.
            criteria: Ordered dict of ``{name: fn(buffer, N) -> bool[N]}``
                acceptance criteria. ``None`` (default) builds criteria
                from :attr:`RetargetPipelineCfg.criteria`; pass an empty
                dict ``{}`` to skip acceptance checks entirely.

        Returns:
            The buffer with ``num_selected`` results indexed by ``selected``.
        """
        if criteria is None:
            criteria = self._build_criteria(wp_mesh)
        self._ensure_buffer(n_desired)
        self.buffer.reset()
        self._timings.clear()
        self._reject_val = {}
        self._ik_iterations_used = 0
        self._n_ik_problems = 0
        self._n_desired = n_desired
        self._sizing = self.sampler.sizing(n_desired)
        self._chunk_profile = []
        self._chunk_profile_meta = {}
        if self.kin.device.startswith("cuda"):
            torch.cuda.synchronize()
        _run_t0 = time.perf_counter()

        with self._time("sampler"):
            sampler_out = self.sampler(
                wp_mesh,
                origin,
                self.buffer,
                n_desired,
            )
            n_written = sampler_out.num_written
            self._reject_geo = sampler_out.reject_stats
            self._sampler_diagnostics = sampler_out.diagnostics

            # Per-slot contact flags: copy into buffer when the sampler
            # supplies them. ``None`` means "accept buffer defaults"
            # (all hard contacts), which ``buffer.reset`` already restored.
            if sampler_out.is_contact is not None and n_written > 0:
                nc = self.buffer.num_contacts
                flat_ic = sampler_out.is_contact.reshape(-1)
                self.buffer.is_contact_t[: n_written * nc] = flat_ic[: n_written * nc]

        for k, v in self.sampler.sub_timings.items():
            # Dots in the key indicate nested phases; indent proportionally.
            depth = k.count(".")
            self._timings[f"{'  ' * (depth + 1)}sampler.{k}"] = v

        if n_written == 0:
            if self.kin.device.startswith("cuda"):
                torch.cuda.synchronize()
            self._run_time = time.perf_counter() - _run_t0
            return self.buffer

        N = self.buffer.num_geometry_valid
        self._n_ik_problems = N
        with self._time("ik_build"):
            # Build objectives at chunk_size so the solver's per-row
            # ``(n_residuals, n_dofs)`` Jacobian + auxiliaries fit in
            # available GPU memory. A throwaway N=1 build first lets us
            # count residuals before sizing the real solver.
            sniff_objs = self._build_objectives(1, wp_mesh)[0]
            chunk_size = self._compute_ik_chunk_size(N, sniff_objs)
            del sniff_objs

            all_objs, contact_objs, contact_rot_objs, base_pos_obj, base_rot_obj = self._build_objectives(
                chunk_size, wp_mesh
            )
            has_autodiff = any(not obj.supports_analytic() for obj in all_objs)
            jac_mode = ik.IKJacobianType.MIXED if has_autodiff else ik.IKJacobianType.ANALYTIC
            solver = self.kin.create_ik_solver(all_objs, chunk_size, jacobian_mode=jac_mode)
            n_chunks = (N + chunk_size - 1) // chunk_size

        with self._time("ik_solve"):
            max_iters = self.cfg.ik_iterations
            threshold = self.cfg.ik_convergence_threshold
            batch_size = max(1, min(3, max_iters))
            iters_used = 0
            all_costs_t = torch.empty(N, device=self.buffer.device, dtype=torch.float32)
            total_diverged = 0

            is_cuda = self.kin.device.startswith("cuda")
            self._chunk_profile_meta = {
                "N": N,
                "chunk_size": chunk_size,
                "n_chunks": n_chunks,
                "max_iters": max_iters,
                "batch_size": batch_size,
                "threshold": threshold,
            }

            def _sync_now() -> None:
                if is_cuda:
                    torch.cuda.synchronize()

            for chunk_idx in range(n_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, N)
                c_size = end - start

                if is_cuda:
                    torch.cuda.reset_peak_memory_stats()
                _sync_now()
                profile: dict = {
                    "chunk_idx": chunk_idx,
                    "c_size": c_size,
                    "rebuild": False,
                    "rebuild_ms": 0.0,
                    "iter_solve_ms": [],
                    "iter_cost_rb_ms": [],
                    "iter_costs": [],
                }
                profile["mem_before"] = self._snapshot_memory()
                chunk_t0 = time.perf_counter()

                # Tail chunk smaller than the solver's pre-built batch:
                # rebuild objectives + solver at ``c_size`` (one-time JIT
                # cost; warp's kernel cache amortizes the same shapes).
                if c_size < chunk_size:
                    rebuild_t0 = time.perf_counter()
                    del solver
                    all_objs, contact_objs, contact_rot_objs, base_pos_obj, base_rot_obj = self._build_objectives(
                        c_size, wp_mesh
                    )
                    solver = self.kin.create_ik_solver(all_objs, c_size, jacobian_mode=jac_mode)
                    _sync_now()
                    profile["rebuild"] = True
                    profile["rebuild_ms"] = (time.perf_counter() - rebuild_t0) * 1000.0

                scatter_t0 = time.perf_counter()
                if contact_objs:
                    self.buffer.scatter_contact_targets(contact_objs, c_size, src_offset=start)
                self.buffer.scatter_contact_rotations(contact_rot_objs, c_size, src_offset=start)
                wp.copy(
                    base_pos_obj.target_positions,
                    self.buffer.base_target_pos,
                    src_offset=start,
                    count=c_size,
                )
                wp.copy(
                    base_rot_obj.target_rotations,
                    self.buffer.base_target_rot,
                    src_offset=start,
                    count=c_size,
                )
                jq_in = wp.from_torch(self.buffer.joint_q_init_t[start:end].contiguous())
                jq_out = wp.from_torch(self.buffer.joint_q_result_t[start:end].contiguous())
                _sync_now()
                profile["scatter_ms"] = (time.perf_counter() - scatter_t0) * 1000.0

                prev_cost = float("inf")
                chunk_iters = 0
                n_diverged = 0
                for _ in range(0, max_iters, batch_size):
                    iters = min(batch_size, max_iters - chunk_iters)
                    solve_t0 = time.perf_counter()
                    solver.step(jq_in, jq_out, iterations=iters)
                    _sync_now()
                    profile["iter_solve_ms"].append((time.perf_counter() - solve_t0) * 1000.0)
                    chunk_iters += iters
                    rb_t0 = time.perf_counter()
                    # A single diverged problem must not decide the whole
                    # chunk's stopping rule: averaging in its non-finite cost
                    # makes every later comparison NaN, which never satisfies
                    # the threshold and burns the full iteration budget on
                    # problems that already converged. Reduce over the finite
                    # ones and carry the diverged count instead; the
                    # solver-cost criterion drops those rows afterwards.
                    costs = wp.to_torch(solver.costs)[:c_size]
                    finite = torch.isfinite(costs)
                    n_diverged = int((~finite).sum())
                    profile["iter_cost_rb_ms"].append((time.perf_counter() - rb_t0) * 1000.0)
                    if n_diverged == c_size:
                        cur_cost = float("inf")
                        profile["iter_costs"].append(cur_cost)
                        break
                    cur_cost = float(costs[finite].mean())
                    profile["iter_costs"].append(cur_cost)
                    if abs(prev_cost - cur_cost) < threshold:
                        break
                    prev_cost = cur_cost
                    jq_in = jq_out
                profile["n_diverged"] = n_diverged
                total_diverged += n_diverged

                wb_t0 = time.perf_counter()
                iters_used = max(iters_used, chunk_iters)
                self.buffer.joint_q_result_t[start:end] = wp.to_torch(jq_out)
                all_costs_t[start:end] = wp.to_torch(solver.costs)[:c_size]
                _sync_now()
                profile["writeback_ms"] = (time.perf_counter() - wb_t0) * 1000.0

                profile["iters_used"] = chunk_iters
                profile["wall_ms"] = (time.perf_counter() - chunk_t0) * 1000.0
                profile["mem_after"] = self._snapshot_memory()
                self._chunk_profile.append(profile)

            self._solver_costs = all_costs_t
            self._n_diverged = total_diverged
            if total_diverged:
                print(
                    f"  IK diverged on {total_diverged} of {N} problems (non-finite cost);"
                    " the solver-cost criterion drops them.",
                    flush=True,
                )
            total_iters = iters_used

        with self._time("fk_eval"):
            self._eval_fk_batched(N)
        self._ik_iterations_used = total_iters

        # Apply criteria to the candidate set. Each candidate is one IK
        # problem (samplers always emit explicit slot assignment, so there
        # is no sibling/group structure to collapse).
        with self._time("criteria"):
            if criteria:
                self._reject_val, cum_pass = _validate_results(self.buffer, criteria)
            else:
                cum_pass = torch.ones(N, device=self.buffer.device, dtype=torch.bool)
                self._reject_val = {"ok": N}

        self.buffer._ik_valid[:N] = cum_pass
        self.buffer.num_ik_valid = int(cum_pass.sum())
        self.buffer.num_final_valid = self.buffer.num_ik_valid

        # The pipeline emits every post-criteria candidate; thinning is
        # the caller's responsibility. One-shot scripts call
        # :func:`~.feature_extractors.apply_final_fps`; the streaming
        # factory accumulator routes through
        # :class:`~isaaclab_tasks.core.multi_task.curriculum.StateBuffer`.
        valid_t = self.buffer._ik_valid[: self.buffer.num_written]
        valid_indices = valid_t.nonzero(as_tuple=False).squeeze(-1)
        n_valid = valid_indices.shape[0]
        if n_valid > 0:
            self.buffer._selected[:n_valid] = valid_indices.to(torch.int32)
        self.buffer.num_selected = n_valid
        if self.kin.device.startswith("cuda"):
            torch.cuda.synchronize()
        self._run_time = time.perf_counter() - _run_t0
        return self.buffer

    def _compute_ik_chunk_size(self, N: int, all_objs: list) -> int:
        """Return the IK batch size to process per kernel launch.

        The Levenberg-Marquardt solver in :class:`newton.ik.IKSolver`
        allocates ``(N, n_residuals, n_dofs) × float32`` for the Jacobian
        plus several smaller ``N``-rate scratches. At high ``n_desired``
        this dominates GPU memory; chunking keeps the peak bounded by
        ``chunk_size`` instead of ``N`` without affecting results
        (per-row IK is independent).

        ``cfg.ik_chunk_size > 0`` pins the chunk manually; ``0`` (default)
        auto-derives it from ``torch.cuda.mem_get_info`` -- 80% of the
        currently free GPU bytes divided by the per-row solver
        footprint, then clamped to ``N``.

        Args:
            N: Total IK problems to solve.
            all_objs: Constructed objective list -- consulted for
                ``residual_dim()`` so the per-row footprint reflects the
                actual objective set rather than a worst-case estimate.

        Returns:
            Chunk size in IK problems (``>= 1``, ``<= N``).
        """
        cfg_size = int(getattr(self.cfg, "ik_chunk_size", 0))
        if cfg_size > 0:
            return min(cfg_size, N)

        if not self.kin.device.startswith("cuda"):
            return N

        free_bytes, _ = torch.cuda.mem_get_info(torch.device(self.kin.device))
        budget = int(0.8 * free_bytes)

        n_residuals = sum(o.residual_dim() for o in all_objs)
        n_dofs = int(self.kin.model.joint_dof_count)
        n_coords = int(self.kin.model.joint_coord_count)
        body_count = int(self.kin.model.body_count)
        joint_count = int(self.kin.model.joint_count)

        # Per-row footprint of ``IKOptimizerLM._alloc_solver_buffers``
        # plus the small per-row state held alongside it. The Jacobian
        # term ``n_residuals * n_dofs * 4`` dominates at typical sizes
        # but the others are non-trivial and worth budgeting.
        per_row = (
            n_dofs * 4  # qd_zero
            + body_count * (28 + 24)  # body_q (transformf) + body_qd (spatial_vectorf)
            + 3 * n_residuals * 4  # residuals + residuals_proposed + residuals_3d
            + n_residuals * n_dofs * 4  # jacobian
            + n_dofs * 4  # dq_dof
            + n_coords * 4  # joint_q_proposed
            + 5 * 4  # costs / costs_proposed / lambda / accept / pred_reduction
            + 4  # problem_idx_identity (int32)
            + joint_count * 28  # X_local
            + n_dofs * 24  # joint_S_s
        )

        chunk_size = max(1, budget // max(1, per_row))
        return min(chunk_size, N)

    def _eval_fk_batched(self, N: int) -> None:
        """Run batched FK on solved joint coordinates and store ``body_q`` in the buffer.

        Allocates (or reallocates, when the buffer has grown) ``joint_qd`` and
        ``body_qd`` scratches at the buffer's ``max_candidates`` -- which is
        the IK-stage capacity (post-FPS workload), *not* the sampler's pre-FPS
        polygon pool. Reuses the buffer's ``body_q`` slab as the zero-copy
        output. Delegates the actual kernel launch to
        :meth:`NewtonKinematics.eval_fk_batched`.

        Args:
            N: Number of candidates to evaluate (first ``N`` rows of the buffer).
        """
        max_cand = self.buffer.max_candidates
        nb = self.kin.model.body_count

        if self._fk_joint_qd is None or self._fk_joint_qd.shape[0] < max_cand:
            self._fk_joint_qd = wp.zeros(
                (max_cand, self.kin.model.joint_dof_count), dtype=wp.float32, device=self.kin.device
            )
            self._fk_body_qd = wp.zeros((max_cand, nb), dtype=wp.spatial_vectorf, device=self.kin.device)

        # Zero-copy view into the pre-allocated body_q slab as [max_cand, nb] transformf.
        bq_wp = wp.from_torch(self.buffer.body_q_t.view(max_cand, nb, 7), dtype=wp.transformf)

        self.kin.eval_fk_batched(
            self.buffer.joint_q_result[:N],
            self._fk_joint_qd[:N],
            bq_wp[:N],
            self._fk_body_qd[:N],
        )

    @property
    def rejection_summary(self) -> str:
        """Human-readable summary of the last pipeline run.

        Each stage reports its *unit* (polygons, IK candidates, or
        placements) so the funnel from one count to the next is
        unambiguous.

        Stages:

        1. **Sampler** — morph-patch pool → per-foot reachability
           sampling (annulus ∩ sector) → polygon-FPS. Unit: polygons.
        2. **IK solve** — per-polygon IK with the contact + base
           pose + joint-limit objectives.
        3. **Criteria** — acceptance predicates applied per-candidate.
        4. **Final FPS** — farthest-point downsample to ``n_desired``.
        """
        buf = self.buffer
        reject_geo = self._reject_geo
        reject_val = self._reject_val
        n_polys_valid = self._n_ik_problems
        n_polys_rejected = sum(reject_geo.values())
        n_polys_attempted = n_polys_valid + n_polys_rejected

        def _render_table(headers: list[str], aligns: list[str], sections: list[list[list[str]]]) -> list[str]:
            """Render a Unicode box table with section separators."""
            all_rows = [headers] + [r for sec in sections for r in sec]
            cols = len(headers)
            widths = [max(len(row[i]) for row in all_rows) for i in range(cols)]

            def border(L: str, M: str, R: str) -> str:
                return L + M.join("─" * (w + 2) for w in widths) + R

            def fmt_row(cells: list[str]) -> str:
                parts = []
                for i, c in enumerate(cells):
                    if aligns[i] == "r":
                        parts.append(" " + c.rjust(widths[i]) + " ")
                    else:
                        parts.append(" " + c.ljust(widths[i]) + " ")
                return "│" + "│".join(parts) + "│"

            out = [border("┌", "┬", "┐"), fmt_row(headers), border("├", "┼", "┤")]
            for i, sec in enumerate(sections):
                if i > 0:
                    out.append(border("├", "┼", "┤"))
                for row in sec:
                    out.append(fmt_row(row))
            out.append(border("└", "┴", "┘"))
            return out

        def fmt(n: int) -> str:
            return f"{n:,}"

        header = "Retarget pipeline"
        lines = [header]
        if self.sampler.init_info:
            lines.append(f"  init: {self.sampler.init_info}")

        # Sizing explainer: back-cascade from ``n_desired`` yields the
        # polygon budget ``max_neighborhoods``.
        polygon_budget = self._sizing.max_neighborhoods if self._sizing is not None else 0
        if self._n_desired > 0 and polygon_budget > 0:
            over = polygon_budget / self._n_desired
            line = (
                f"  sizing: target={self._n_desired} → budget={fmt(polygon_budget)} polygons"
                f" ({over:.0f}× oversample for FPS diversity)"
            )
            lines.append(line)
            # Cascade breakdown is cfg-specific; emit it if the sampler
            # carries a ``SamplerSizingCfg`` at ``cfg.sizing``.
            sz_cfg = getattr(getattr(self.sampler, "cfg", None), "sizing", None)
            if sz_cfg is not None and all(
                hasattr(sz_cfg, k)
                for k in (
                    "final_fps_oversample",
                    "criteria_yield",
                    "polygon_fps_oversample",
                    "polygon_assembly_yield",
                )
            ):
                lines.append(
                    f"    cascade: {self._n_desired}"
                    f" ×{sz_cfg.final_fps_oversample:g}(final FPS)"
                    f" ÷{sz_cfg.criteria_yield:g}(criteria yield)"
                    f" ×{sz_cfg.polygon_fps_oversample:g}(polygon FPS)"
                    f" ÷{sz_cfg.polygon_assembly_yield:g}(geometry yield)"
                    f" = {fmt(polygon_budget)}"
                )
        lines.append("")

        # --- Main pipeline table: funnel with units per stage ---
        sections: list[list[list[str]]] = []

        # Sampler (polygon funnel).  Rejection rows render as "−N" so the
        # reader sees they subtract from ``attempted`` down to ``valid``.
        # The ``valid`` row appends the geometric yield rate.
        attempted_cell = (
            f"{fmt(n_polys_attempted)} / {fmt(polygon_budget)} budget" if polygon_budget > 0 else fmt(n_polys_attempted)
        )
        sampler_rows = [["Sampler", "polygons", attempted_cell]]
        for reason, count in reject_geo.items():
            sampler_rows.append([f"  ─ {reason}", "", f"−{fmt(count)}"])
        geo_yield = 100.0 * n_polys_valid / n_polys_attempted if n_polys_attempted > 0 else 0.0
        sampler_rows.append(["  valid", "polygons", f"{fmt(n_polys_valid)} ({geo_yield:.1f}%)"])
        sections.append(sampler_rows)

        # IK solve.
        sections.append([[f"IK solve ({self._ik_iterations_used} iters)", "candidates", fmt(self._n_ik_problems)]])

        # Criteria waterfall (per-candidate).
        if reject_val and any(k != "ok" for k in reject_val):
            crit_total_in = self._n_ik_problems
            crit_rows = [["Criteria (per-candidate)", "candidates", fmt(crit_total_in)]]
            for name, count in reject_val.items():
                if name == "ok":
                    continue
                crit_rows.append([f"  ─ {name}", "", f"−{fmt(count)}"])
            ok = reject_val.get("ok", 0)
            pct = 100.0 * ok / crit_total_in if crit_total_in > 0 else 0.0
            crit_rows.append(["  survivors", "candidates", f"{fmt(ok)} ({pct:.1f}%)"])
            sections.append(crit_rows)
        else:
            sections.append([["Criteria (none applied)", "—", fmt(buf.num_ik_valid)]])

        # Final FPS downsample to target.
        sections.append([["Final FPS", "placements", f"{fmt(buf.num_ik_valid)} → {fmt(buf.num_selected)}"]])

        lines.extend(_render_table(["Stage", "Unit", "Count"], ["l", "l", "r"], sections))

        # --- Timings table ---
        if self._timings:
            # Sub-entries stored with a leading-space prefix so iteration
            # order keeps them grouped under their parent; exclude those
            # from the total to avoid double-counting.
            tracked = sum(dt for name, dt in self._timings.items() if not name.startswith(" "))
            run_total = self._run_time if self._run_time > 0.0 else tracked
            # Denominator for the percent column: prefer the full run time
            # so rows sum to ~100% and the untracked slice is visible.
            denom = run_total if run_total > 0 else 1.0
            timing_rows: list[list[str]] = []
            for name, dt in self._timings.items():
                stripped = name.lstrip(" ")
                leading = name[: len(name) - len(stripped)]
                # Prefer a human-readable label for this phase; otherwise
                # fall back to the last dotted segment (the indentation
                # already conveys the hierarchy, so prefixes are redundant).
                display = _TIMING_LABELS.get(stripped, stripped.rsplit(".", 1)[-1])
                label = leading + display
                timing_rows.append([label, f"{dt:.3f}s", f"{100.0 * dt / denom:.1f}%"])
            untracked = run_total - tracked
            if untracked > 0.0005:
                timing_rows.append(
                    [
                        "untracked (Python glue / buffer reset / CUDA sync gaps)",
                        f"{untracked:.3f}s",
                        f"{100.0 * untracked / denom:.1f}%",
                    ]
                )
            lines.append("")
            lines.append(
                f"Timings  (run {run_total:.3f}s = {tracked:.3f}s tracked"
                f" + {max(untracked, 0.0):.3f}s untracked; excludes pipeline constructor,"
                " reachability precompute, and first-call JIT/CUDA-graph compile)"
            )
            lines.extend(_render_table(["Phase", "Time", "%"], ["l", "r", "r"], [timing_rows]))

        return "\n".join(lines)

    @property
    def chunk_profile_summary(self) -> str:
        """Per-chunk timing + memory breakdown of the last :meth:`run` call's
        IK solve loop. Empty string when the loop hasn't run or CUDA was
        unavailable. Designed for studying chunk-size headroom and tail-chunk
        rebuild cost, not for routine logging.
        """
        if not self._chunk_profile:
            return ""

        meta = self._chunk_profile_meta
        N = int(meta.get("N", 0))
        chunk_size = int(meta.get("chunk_size", 0))
        n_chunks = int(meta.get("n_chunks", 0))
        max_iters = int(meta.get("max_iters", 0))
        batch_size = int(meta.get("batch_size", 0))

        def fmt_ms(x: float | None) -> str:
            return "—" if x is None or x <= 0.0 else f"{x:.1f}"

        def fmt_mib(x: float | None) -> str:
            return "—" if x is None else f"{x:,.0f}"

        def fmt_conv(costs: list[float]) -> str:
            if not costs:
                return "—"
            if len(costs) == 1:
                return f"{costs[0]:.2e}"
            return f"{costs[0]:.2e}→{costs[-1]:.2e}"

        headers = [
            "chunk",
            "c_size",
            "wall ms",
            "build ms",
            "setup ms",
            "solve ms",
            "rb+wb ms",
            "iters",
            "conv (mean cost)",
            "cuda free MiB (before→after)",
            "peak alloc MiB",
        ]
        rows: list[list[str]] = []
        for p in self._chunk_profile:
            iters_used = int(p.get("iters_used", 0))
            solve_total = sum(p.get("iter_solve_ms", []))
            rb_total = sum(p.get("iter_cost_rb_ms", [])) + float(p.get("writeback_ms", 0.0))
            mem_b = p.get("mem_before") or {}
            mem_a = p.get("mem_after") or {}
            free_before = mem_b.get("cuda_free_mib")
            free_after = mem_a.get("cuda_free_mib")
            free_str = f"{fmt_mib(free_before)} → {fmt_mib(free_after)}"
            peak = mem_a.get("torch_peak_alloc_mib")
            rows.append(
                [
                    str(p["chunk_idx"]),
                    f"{p['c_size']:,}",
                    fmt_ms(p.get("wall_ms")),
                    fmt_ms(p.get("rebuild_ms")) if p.get("rebuild") else "—",
                    fmt_ms(p.get("scatter_ms")),
                    fmt_ms(solve_total),
                    fmt_ms(rb_total),
                    f"{iters_used}/{max_iters}",
                    fmt_conv(p.get("iter_costs", [])),
                    free_str,
                    fmt_mib(peak),
                ]
            )

        widths = [max(len(headers[i]), max(len(r[i]) for r in rows)) for i in range(len(headers))]

        def border(L: str, M: str, R: str) -> str:
            return L + M.join("─" * (w + 2) for w in widths) + R

        def fmt_row(cells: list[str]) -> str:
            parts = []
            for i, c in enumerate(cells):
                # Right-align everything except the first two columns.
                align_right = i >= 2
                cell = c.rjust(widths[i]) if align_right else c.ljust(widths[i])
                parts.append(" " + cell + " ")
            return "│" + "│".join(parts) + "│"

        lines = [
            f"Chunk profile  (N={N:,} IK problems, {n_chunks} chunks @ chunk_size={chunk_size:,};"
            f" max_iters={max_iters}, LM batch={batch_size})",
            border("┌", "┬", "┐"),
            fmt_row(headers),
            border("├", "┼", "┤"),
        ]
        for row in rows:
            lines.append(fmt_row(row))
        lines.append(border("└", "┴", "┘"))

        # Headroom hint: minimum cuda_free across chunks vs the budget that
        # ``_compute_ik_chunk_size`` saw at sniff time. If min_free is much
        # higher than the 20% safety margin, chunk_size was undersized.
        free_after_vals = [(p.get("mem_after") or {}).get("cuda_free_mib") for p in self._chunk_profile]
        free_after_vals = [v for v in free_after_vals if v is not None]
        if free_after_vals:
            min_free = min(free_after_vals)
            max_free = max(free_after_vals)
            lines.append(
                f"  cuda_free over chunks: min={min_free:,.0f} MiB, max={max_free:,.0f} MiB"
                f"  (delta {max_free - min_free:,.0f} MiB)"
            )
        return "\n".join(lines)
