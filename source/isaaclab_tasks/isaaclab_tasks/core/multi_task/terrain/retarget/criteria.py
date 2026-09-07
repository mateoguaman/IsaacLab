# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-agnostic retarget validation criteria.

These criteria work with any articulated robot without requiring
robot-specific constants. For robot-specific criteria (e.g. lateral hip
joint limits), see the per-robot preset modules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp


def _quat_to_matrix_xyzw(quat: torch.Tensor) -> torch.Tensor:
    """Rotation matrices for ``(x, y, z, w)`` quaternions.

    Args:
        quat: ``[..., 4]`` quaternions in Newton's xyzw order.

    Returns:
        ``[..., 3, 3]`` rotation matrices.
    """
    quat = torch.nn.functional.normalize(quat, dim=-1)
    x, y, z, w = quat.unbind(-1)
    return torch.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*quat.shape[:-1], 3, 3)


if TYPE_CHECKING:
    from ...kinematics import NewtonKinematics
    from .buffer import RetargetBuffer
    from .criteria_cfg import (
        CollisionCheckCfg,
        FootPositionErrorCfg,
        JointWithinLimitCfg,
        LateralHipLimitCfg,
        SolverCostOutlierCfg,
        SupportPolygonStabilityCfg,
    )
    from .pipeline import RetargetPipeline


class FootPositionError:
    """Criterion: FK foot positions must match contact targets within tolerance [m].

    Reads ``body_q`` directly from the buffer (populated by the
    pipeline after IK) -- no separate FK pass needed. The ``aggregate``
    knob selects how per-foot errors are combined: ``"max"`` bounds the
    worst foot, ``"sum"`` bounds the total drift across the polygon.

    Args:
        cfg: :class:`~.criteria_cfg.FootPositionErrorCfg` with ``max_err``
            and ``aggregate`` fields.
        pipeline: Live :class:`RetargetPipeline` — read for
            ``kin.model.body_count`` and ``foot_body_ids``.
        wp_mesh: Unused (kept for uniform construction signature).
    """

    def __init__(self, cfg: FootPositionErrorCfg, pipeline: RetargetPipeline, wp_mesh: object = None) -> None:
        self.num_bodies = pipeline.kin.model.body_count
        self.foot_ids = list(pipeline.foot_body_ids)
        self.max_err = cfg.max_err
        self.aggregate = cfg.aggregate

    def __call__(self, buffer: RetargetBuffer, N: int) -> torch.Tensor:
        nc = len(self.foot_ids)
        body_q = buffer.body_q_t[: N * self.num_bodies].view(N, self.num_bodies, 7)
        ct = buffer.contact_targets_t[: N * nc].view(N, nc, 3)
        idx = torch.tensor(self.foot_ids, device=buffer.device, dtype=torch.long)
        per_foot = (body_q[:, idx, :3] - ct).norm(dim=-1)  # [N, nc]
        # Mask air slots (``is_contact == False``) out of the foot-error
        # reduction -- their target is a kinematic reference, not a
        # ground-truth ground contact, so foot drift there isn't a
        # physical-stability violation. Sum-aggregate zeros masked slots;
        # max-aggregate pushes them to ``-inf`` so they never dominate.
        is_contact = buffer.is_contact_t[: N * nc].view(N, nc)
        if self.aggregate == "sum":
            per_foot = torch.where(is_contact, per_foot, torch.zeros_like(per_foot))
            err = per_foot.sum(dim=-1)
        elif self.aggregate == "max":
            neg_inf = torch.full_like(per_foot, float("-inf"))
            masked = torch.where(is_contact, per_foot, neg_inf)
            err = masked.max(dim=-1).values
            # If no slot is in contact, ``err = -inf`` trivially passes;
            # that's the right behavior (no ground contact → no foot-error
            # criterion to enforce).
        else:
            raise ValueError(f"FootPositionError.aggregate must be 'max' or 'sum', got {self.aggregate!r}")
        return err <= self.max_err


@dataclass
class JointMargin:
    """Criterion: revolute joints must stay within ``margin`` of their limits.

    Args:
        kin: :class:`NewtonKinematics` instance.
        margin: Fraction of joint range to keep as safety margin.
    """

    kin: NewtonKinematics
    margin: float = 0.1

    def __call__(self, buffer: RetargetBuffer, N: int) -> torch.Tensor:
        jl = wp.to_torch(self.kin.model.joint_limit_lower)
        ju = wp.to_torch(self.kin.model.joint_limit_upper)
        lo, hi = jl[6:], ju[6:]
        n_rev = lo.shape[0]
        jq = buffer.joint_q_result_t[:N, 7 : 7 + n_rev]
        safe_lo = lo + self.margin * (hi - lo)
        safe_hi = hi - self.margin * (hi - lo)
        violation = ((safe_lo - jq).clamp(min=0) + (jq - safe_hi).clamp(min=0)).max(dim=-1).values
        return violation <= 0


class LateralHipLimit:
    """Criterion: lateral hip joint angles must not exceed ``max_angle`` [rad].

    Resolves DOF indices at construction time from a joint name regex
    via :meth:`NewtonKinematics.find_joint_dof_indices`. The regex comes
    from ``cfg.joint_pattern`` if set, else from the pipeline's
    ``lateral_hip_joint_pattern`` (resolved per robot preset).

    Args:
        cfg: :class:`~.criteria_cfg.LateralHipLimitCfg` with
            ``joint_pattern`` and ``max_angle`` fields.
        pipeline: Live :class:`RetargetPipeline` — used for joint-name
            resolution and the fallback regex.
        wp_mesh: Unused (kept for uniform construction signature).
    """

    def __init__(self, cfg: LateralHipLimitCfg, pipeline: RetargetPipeline, wp_mesh: object = None) -> None:
        pattern = cfg.joint_pattern
        if pattern is None:
            pattern = pipeline.cfg.lateral_hip_joint_pattern
        if pattern is None:
            raise ValueError(
                "LateralHipLimit requires a joint_pattern. Either set LateralHipLimitCfg.joint_pattern or "
                "RetargetPipelineCfg.lateral_hip_joint_pattern (typically resolved per robot preset)."
            )
        self.max_angle = cfg.max_angle
        self._joint_indices = pipeline.kin.find_joint_dof_indices(pattern)

    def __call__(self, buffer: RetargetBuffer, N: int) -> torch.Tensor:
        if not self._joint_indices:
            return torch.ones(N, device=buffer.device, dtype=torch.bool)
        n_rev = buffer.joint_q_result_t.shape[1] - 7
        joint_idx = torch.tensor(
            [i for i in self._joint_indices if i < n_rev],
            device=buffer.device,
            dtype=torch.long,
        )
        if joint_idx.numel() == 0:
            return torch.ones(N, device=buffer.device, dtype=torch.bool)
        jq_rev = buffer.joint_q_result_t[:N, 7:]
        return jq_rev[:, joint_idx].abs().max(dim=-1).values <= self.max_angle


class JointWithinLimit:
    """Criterion: all non-root joints must stay inside a scaled retarget joint interval.

    The checked interval is the same effective interval used by the FK
    sampler: Newton joint limits intersected with
    ``default_joint_q ± sampler.fk_joint_range`` and any
    ``sampler.fk_joint_range_overrides``. The final interval is then
    shrunk around its center by ``limit_ratio``.

    Args:
        cfg: :class:`~.criteria_cfg.JointWithinLimitCfg` with
            ``limit_ratio``.
        pipeline: Live :class:`RetargetPipeline` — read for Newton joint
            limits and sampler FK joint ranges.
        wp_mesh: Unused (kept for uniform construction signature).
    """

    def __init__(self, cfg: JointWithinLimitCfg, pipeline: RetargetPipeline, wp_mesh: object = None) -> None:
        if not 0.0 < cfg.limit_ratio <= 1.0:
            raise ValueError(f"JointWithinLimit.limit_ratio must be in (0, 1], got {cfg.limit_ratio}.")
        jl = wp.to_torch(pipeline.kin.model.joint_limit_lower)
        ju = wp.to_torch(pipeline.kin.model.joint_limit_upper)
        default_q = torch.from_numpy(pipeline.kin.default_joint_q).float().to(pipeline.kin.device)
        rev_default = default_q[7:]
        n_rev = rev_default.shape[0]
        joint_range = torch.full((n_rev,), float(pipeline.cfg.sampler.fk_joint_range), device=pipeline.kin.device)
        for pattern, clamp in pipeline.cfg.sampler.fk_joint_range_overrides.items():
            for joint_index in pipeline.kin.find_joint_dof_indices(pattern):
                if joint_index < n_rev:
                    joint_range[joint_index] = float(clamp)
        lo = torch.maximum(jl[6 : 6 + n_rev], rev_default - joint_range)
        hi = torch.minimum(ju[6 : 6 + n_rev], rev_default + joint_range)
        half_margin = 0.5 * (1.0 - cfg.limit_ratio)
        span = hi - lo
        self._safe_lo = lo + half_margin * span
        self._safe_hi = hi - half_margin * span

    def __call__(self, buffer: RetargetBuffer, N: int) -> torch.Tensor:
        n_rev = self._safe_lo.shape[0]
        jq = buffer.joint_q_result_t[:N, 7 : 7 + n_rev]
        if jq.shape[1] != n_rev:
            raise RuntimeError(f"JointWithinLimit expected {n_rev} non-root joint coordinates, got {jq.shape[1]}.")
        return ((jq >= self._safe_lo) & (jq <= self._safe_hi)).all(dim=-1)


@dataclass
class BaseZError:
    """Criterion: base z must not deviate more than ``max_err`` from target [m].

    Args:
        max_err: Maximum absolute base z deviation [m].
    """

    max_err: float = 0.3

    def __call__(self, buffer: RetargetBuffer, N: int) -> torch.Tensor:
        base_z = buffer.joint_q_result_t[:N, 2]
        target_z = buffer.base_target_pos_t[:N, 2]
        return (base_z - target_z).abs() <= self.max_err


class SupportPolygonStability:
    """Criterion: base (CoM proxy) must project inside the support region.

    Physically-grounded quasi-static stability check: under constant
    gravity, the CoM's vertical projection must lie within the support
    region in world-frame XY. Uses the base-link origin as a cheap CoM
    proxy -- for quadrupeds the base is near the CoM, so the
    approximation is tight.

    The support region depends on the number of contacts ``nc``:

    * ``nc == 1``: support collapses to a point, which is a measure-zero
      statically stable region -- always rejected.
    * ``nc == 2``: support collapses to the segment between the two
      contacts. The base must project onto the closed segment with
      axial parameter :math:`t \\in [0, 1]`, and perpendicular distance
      bounded by ``segment_tol_frac × segment_length``. Scaling the
      lateral margin with the segment length (rather than an absolute
      threshold) captures the finite foot-contact-footprint regularization
      of the otherwise measure-zero balance condition.
    * ``nc >= 3``: the support region is the convex hull of the contacts.
      Sort CCW around the centroid, then for each edge check the sign of
      the 2D cross product ``(v_{i+1} - v_i) × (p - v_i)``. All non-
      negative iff ``p`` is inside the hull.

    Args:
        cfg: :class:`~.criteria_cfg.SupportPolygonStabilityCfg`.
        pipeline: Live :class:`RetargetPipeline` — read for ``kin`` (to derive
            each foot's sole outline) and ``foot_body_ids``. When ``None``,
            feet are treated as point contacts.
        wp_mesh: Unused (kept for uniform construction signature).
    """

    def __init__(
        self,
        cfg: SupportPolygonStabilityCfg | None = None,
        pipeline: RetargetPipeline | None = None,
        wp_mesh: object = None,
    ) -> None:
        from .criteria_cfg import SupportPolygonStabilityCfg as _Cfg

        cfg = _Cfg() if cfg is None else cfg
        self.margin = cfg.margin
        self.num_directions = cfg.num_directions
        self.num_bodies: int | None = None
        self.foot_ids: list[int] | None = None
        # Sole outline per foot in the foot's own xy frame. Without a pipeline
        # the robot's geometry is unknown, so each foot is a single vertex at
        # its own origin -- a point contact, which the support test below
        # handles as the degenerate case of a hull.
        self._hull_local_np = np.zeros((1, 1, 2), dtype=np.float32)
        if pipeline is not None:
            self.num_bodies = pipeline.kin.model.body_count
            self.foot_ids = list(pipeline.foot_body_ids)
            self._hull_local_np = pipeline.kin.foot_contact_hulls(self.foot_ids, height_tol=cfg.contact_height_tol)[
                "vertices"
            ]
        self._hull_local: torch.Tensor | None = None
        self._directions: torch.Tensor | None = None

    def _cached(self, device: torch.device, nc: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the per-foot sole hulls and probe directions on ``device``."""
        if self._hull_local is None or self._hull_local.device != device:
            hull = torch.as_tensor(self._hull_local_np, dtype=torch.float32, device=device)
            if hull.shape[0] == 1 and nc > 1:
                hull = hull.expand(nc, -1, -1)
            self._hull_local = hull.contiguous()
            angles = torch.arange(self.num_directions, dtype=torch.float32, device=device)
            angles *= 2.0 * math.pi / self.num_directions
            self._directions = torch.stack([angles.cos(), angles.sin()], dim=-1)
        return self._hull_local, self._directions  # type: ignore[return-value]

    def __call__(self, buffer: RetargetBuffer, N: int) -> torch.Tensor:
        nc = buffer.num_contacts
        device = buffer.device
        if N == 0:
            return torch.zeros(0, device=device, dtype=torch.bool)

        hull_local, directions = self._cached(torch.device(device), nc)
        num_vertices = hull_local.shape[1]

        contact_xy = buffer.contact_targets_t[: N * nc].view(N, nc, 3)[..., :2]
        is_contact = buffer.is_contact_t[: N * nc].view(N, nc)
        com_xy = buffer.joint_q_result_t[:N, 0:2]

        # Place each foot's sole outline in the world. Only the horizontal
        # projection bears load, so the foot's rotation is applied and the xy
        # block taken -- a tilted foot correctly presents a foreshortened
        # outline rather than its full footprint.
        if self.foot_ids is not None and self.num_bodies is not None:
            body_q = buffer.body_q_t[: N * self.num_bodies].view(N, self.num_bodies, 7)
            foot_index = torch.as_tensor(self.foot_ids, device=device, dtype=torch.long)
            rotation_xy = _quat_to_matrix_xyzw(body_q[:, foot_index, 3:7])[..., :2, :2]
            offsets = torch.einsum("nfij,fvj->nfvi", rotation_xy, hull_local)
        else:
            offsets = hull_local.unsqueeze(0).expand(N, -1, -1, -1)
        vertices = (contact_xy.unsqueeze(2) + offsets).reshape(N, nc * num_vertices, 2)

        # Support-function test. For every probe direction, compare how far the
        # support region reaches against how far the CoM sits. The smallest gap
        # is the signed distance from the CoM to the region's boundary, so it
        # doubles as the stability margin and needs no hull ordering -- which
        # matters because most sole vertices lie strictly inside the region.
        active = is_contact.unsqueeze(-1).expand(N, nc, num_vertices).reshape(N, nc * num_vertices)
        projection = vertices @ directions.T
        projection = torch.where(active.unsqueeze(-1), projection, torch.full_like(projection, -torch.inf))
        support = projection.amax(dim=1)
        return (support - com_xy @ directions.T).amin(dim=-1) >= self.margin


class SolverCostOutlier:
    """Criterion: reject candidates whose final solver cost is an outlier.

    Residual IK divergence -- after all geometric constraints pass, the
    solver may still settle on a poor local minimum. Reject candidates
    whose cost exceeds ``threshold_multiplier * median(costs)``.

    Args:
        cfg: :class:`~.criteria_cfg.SolverCostOutlierCfg` with
            ``threshold_multiplier``.
        pipeline: Live :class:`RetargetPipeline` — read for
            ``_solver_costs``.
        wp_mesh: Unused (kept for uniform construction signature).
    """

    def __init__(self, cfg: SolverCostOutlierCfg, pipeline: RetargetPipeline, wp_mesh: object = None) -> None:
        self.pipeline = pipeline
        self.threshold_multiplier = cfg.threshold_multiplier

    def __call__(self, buffer: RetargetBuffer, N: int) -> torch.Tensor:
        costs = self.pipeline._solver_costs[:N]
        median = costs.median()
        return costs < median * self.threshold_multiplier


class CollisionCheck:
    """Criterion: reject candidates where body probes penetrate terrain.

    Reads ``body_q`` from the buffer (populated by the pipeline after
    IK), transforms collision probe offsets into world space via a Warp
    kernel, and queries the terrain mesh for penetration. No separate
    FK pass -- uses the body transforms already stored in the buffer.

    Args:
        cfg: :class:`~.criteria_cfg.CollisionCheckCfg` with ``n_samples``
            and ``max_pen``.
        pipeline: Live :class:`RetargetPipeline` — read for
            ``kin.builder`` (probe generation) and ``foot_body_ids``
            (exclusion).
        wp_mesh: Terrain Warp mesh to query for penetration depth.
    """

    def __init__(self, cfg: CollisionCheckCfg, pipeline: RetargetPipeline, wp_mesh: object) -> None:
        from ...kinematics.ik_objectives.terrain_collision import _build_collision_probes  # noqa: PLC0415

        self.kin = pipeline.kin
        self.wp_mesh = wp_mesh
        self.max_pen = cfg.max_pen
        bodies, offsets, slots = _build_collision_probes(self.kin.builder, pipeline.foot_body_ids, cfg.n_samples)
        self._n_probes = len(bodies)
        self._n_feet = len(pipeline.foot_body_ids)
        self._probe_body_np = np.array(bodies, dtype=np.int32)
        self._probe_offset_np = np.array(offsets, dtype=np.float32)
        self._probe_foot_slot_np = np.array(slots, dtype=np.int32)
        self._wp_bufs: dict[str, tuple[wp.array, wp.array, wp.array]] = {}

    def _get_wp_bufs(self, device: str) -> tuple[wp.array, wp.array, wp.array]:
        """Lazily allocate device buffers for probe data."""
        if device not in self._wp_bufs:
            self._wp_bufs[device] = (
                wp.array(self._probe_body_np, dtype=wp.int32, device=device),
                wp.from_numpy(self._probe_offset_np, dtype=wp.vec3, device=device),
                wp.array(self._probe_foot_slot_np, dtype=wp.int32, device=device),
            )
        return self._wp_bufs[device]

    def __call__(self, buffer: RetargetBuffer, N: int) -> torch.Tensor:
        nb = self.kin.model.body_count
        body_q_wp = wp.from_torch(
            buffer.body_q_t[: N * nb].contiguous(),
            dtype=wp.transformf,
        )
        probe_body, probe_offset, probe_foot_slot = self._get_wp_bufs(buffer.device)
        # Slot-ordered per-problem contact flag; contact feet shouldn't be
        # flagged for "penetrating" terrain since they're meant to touch.
        is_contact_u8 = buffer.is_contact_t[: N * self._n_feet].view(N, self._n_feet).to(torch.uint8).contiguous()
        is_contact_wp = wp.from_torch(is_contact_u8, dtype=wp.uint8)

        pen_out = torch.zeros(N * self._n_probes, device=buffer.device, dtype=torch.float32)
        wp_pen = wp.from_torch(pen_out)
        wp.launch(
            _collision_check_kernel,
            dim=[N, self._n_probes],
            inputs=[
                body_q_wp,
                nb,
                self.wp_mesh.id,
                probe_body,
                probe_offset,
                probe_foot_slot,
                is_contact_wp,
                2.0,
                self._n_probes,
            ],
            outputs=[wp_pen],
            device=buffer.device,
        )
        wp.synchronize()
        return pen_out.view(N, self._n_probes).max(dim=-1).values <= self.max_pen


@wp.kernel
def _collision_check_kernel(
    body_q: wp.array1d(dtype=wp.transformf),
    n_bodies: int,
    mesh_id: wp.uint64,
    probe_body: wp.array1d(dtype=wp.int32),
    probe_offset: wp.array1d(dtype=wp.vec3),
    probe_foot_slot: wp.array1d(dtype=wp.int32),
    is_contact: wp.array2d(dtype=wp.uint8),
    max_dist: float,
    n_probes: int,
    penetration: wp.array1d(dtype=wp.float32),
):
    """Penetration depth for each (candidate, probe) pair.

    Combines two signals to handle both open and closed meshes:

    - ``-sign * dist``: correct for watertight meshes where
      ``mesh_query_point`` can determine inside/outside.
    - ``surface_z - probe_z``: correct for open heightfield terrains
      where ``sign`` is always +1.

    The max of both is used so penetration is detected either way.

    Probes on a foot body (``probe_foot_slot >= 0``) are zeroed when that
    foot is in contact: a contact foot is meant to touch terrain, so the
    criterion should not flag it as penetrating.
    """
    row, probe_idx = wp.tid()
    out_idx = row * n_probes + probe_idx
    slot = probe_foot_slot[probe_idx]
    if slot >= 0 and is_contact[row, slot] != wp.uint8(0):
        penetration[out_idx] = 0.0
        return
    tf = body_q[row * n_bodies + probe_body[probe_idx]]
    world_pos = wp.transform_point(tf, probe_offset[probe_idx])
    query = wp.mesh_query_point(mesh_id, world_pos, max_dist)
    if query.result:
        surface_pt = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        dist = wp.length(world_pos - surface_pt)
        sign_pen = -query.sign * dist
        z_pen = surface_pt[2] - world_pos[2]
        penetration[out_idx] = wp.max(sign_pen, z_pen)
    else:
        penetration[out_idx] = 0.0
