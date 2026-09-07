# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Template-projection contact sampling strategy for the retarget pipeline.

Build-time: random-joint FK (with joints clamped to ``default ± fk_joint_range``
to avoid wrap-around leg configurations) produces a library of canonical
foot-polygon shapes; samples whose ``nc`` feet don't form a convex polygon
(self-intersecting CCW ordering or one foot inside the triangle of the
others) are dropped. Per-foot nominal angles and the standing height are
derived from the same FK batch.

Query-time: each candidate picks a random morph patch (center) + random
yaw + random template, un-canonicalizes the template at that pose, and
assigns each foot to a morph patch via a constrained linear-sum-assignment
over ``nc^nc`` combinations. The LSA minimises total distance subject to
(1) distinct patches per contact foot, (2) the snapped stance preserving
the template's slot→CCW-rank permutation (no crossed feet), and (3) the
stance walked in angular order being strictly convex (no foot inside the
triangle of the others). Feet within ``terrain_snap_distance`` of their
assigned patch become **contact** targets (patch + ``foot_ground_offset``);
feet farther away become **air** targets (projected template xy, z clamped
above the local terrain surface).
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
import torch
import warp as wp

from ...kinematics import NewtonKinematics
from ...utils.grid_downsample import grid_bucket_downsample
from ..terrains.patch_sampling.cfg import CircleFootprintCfg, MorphologicalPatchSamplingCfg, RectFootprintCfg
from ..terrains.patch_sampling.morph import MORPH_TIMINGS, MORPH_YAW_BINS
from .buffer import RetargetBuffer
from .canonical_shape import canonicalize_shape
from .cfg import PatchSamplingCfg, SamplerCfg, SamplerSizingCfg
from .fused_sampler_kernel import run_fused_sampler
from .sampler_base import SamplerBase, SamplerOutput, SamplerSizing, compute_sampler_sizing


def _prepare_ik_batched(
    v_contact: torch.Tensor,
    v_base: torch.Tensor,
    v_yaw: torch.Tensor,
    jq_rev_seed: torch.Tensor,
    buffer: RetargetBuffer,
) -> None:
    """Plane-fit foot polygon → base pitch/roll → per-problem IK seed.

    Writes ``contact_targets``, ``joint_q_init``, ``base_target_pos`` and
    ``base_target_rot`` into ``buffer`` for the first ``v_contact.shape[0]``
    rows.

    Plane-fit is a closed-form least-squares solve on the foot polygon's
    mean-centered ``(x, y, z)`` covariance. The 2×2 system can be
    ill-conditioned when the foot xy layout is near-collinear (nc=2 biped
    or degenerate polygon), so we only take the fit when
    ``det > 1e-6 * xx * yy`` — a relative rank check that matches the
    degeneracy guard in :func:`~.canonical_shape.canonicalize_shape` so
    shape-match and IK seed agree.

    The base-target quaternion uses **half** of the plane-fit roll
    (``roll * 0.25`` after half-angle), keeping the base more upright on
    sloped terrain; full roll pulled the base near joint limits and
    spiked foot-error.

    Args:
        v_contact: Per-placement foot target positions [m],
            shape ``[n_ik, nc, 3]``.
        v_base: Per-placement base target position [m],
            shape ``[n_ik, 3]``.
        v_yaw: Per-placement base target yaw [rad], shape ``[n_ik]``.
        jq_rev_seed: Per-placement revolute-joint IK seed [rad],
            shape ``[n_ik, n_rev]``. Populated from the matched FK
            template's joint configuration so IK starts in the same
            basin of attraction as the target stance, avoiding the
            crossed-leg local minimum that a uniform default-stance
            seed is vulnerable to on twisted poses.
        buffer: Retarget buffer to scatter the prepared IK problem into.
    """
    n_ik, nc, _ = v_contact.shape
    n_rev = jq_rev_seed.shape[1]
    jc = 7 + n_rev

    centroid = v_contact.mean(dim=-2, keepdim=True)  # [n_ik, 1, 3]
    delta = v_contact - centroid  # [n_ik, nc, 3]
    dx = delta[..., 0]
    dy = delta[..., 1]
    dz = delta[..., 2]

    xx = (dx * dx).sum(dim=-1)
    yy = (dy * dy).sum(dim=-1)
    xym = (dx * dy).sum(dim=-1)
    xz = (dx * dz).sum(dim=-1)
    yz = (dy * dz).sum(dim=-1)

    det = xx * yy - xym * xym
    plane_rank_ok = det > 1.0e-6 * xx * yy
    det_safe = det.clamp_min(1.0e-12)
    a_raw = (yy * xz - xym * yz) / det_safe
    b_raw = (xx * yz - xym * xz) / det_safe

    # Gate on both rank and z range: flat or degenerate → zero tilt.
    z_range = dz.amax(dim=-1) - dz.amin(dim=-1)
    use_fit = plane_rank_ok & (z_range >= 1.0e-4)
    zero = torch.zeros_like(a_raw)
    a = torch.where(use_fit, a_raw, zero)
    b = torch.where(use_fit, b_raw, zero)

    # Rotate plane slopes ``(a, b) = (dz/dx, dz/dy)_world`` into base frame
    # by ``-yaw``; atan then gives body-frame pitch/roll.
    cy = torch.cos(v_yaw)
    sy = torch.sin(v_yaw)
    pitch = -torch.atan(a * cy + b * sy)
    roll = torch.atan(-a * sy + b * cy)

    # Two quaternions from (yaw, pitch, roll): joint_q_init uses full roll
    # (matches the foot-plane exactly), base_target_rot uses half roll.
    def _euler_to_quat(y: torch.Tensor, p: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        cy2 = torch.cos(y * 0.5)
        sy2 = torch.sin(y * 0.5)
        cp2 = torch.cos(p * 0.5)
        sp2 = torch.sin(p * 0.5)
        cr2 = torch.cos(r * 0.5)
        sr2 = torch.sin(r * 0.5)
        qw = cy2 * cr2 * cp2 + sy2 * sr2 * sp2
        qx = cy2 * sr2 * cp2 - sy2 * cr2 * sp2
        qy = cy2 * cr2 * sp2 + sy2 * sr2 * cp2
        qz = sy2 * cr2 * cp2 - cy2 * sr2 * sp2
        return torch.stack([qx, qy, qz, qw], dim=-1)

    quat_full = _euler_to_quat(v_yaw, pitch, roll)
    quat_half = _euler_to_quat(v_yaw, pitch, roll * 0.5)

    buffer.contact_targets_t[: n_ik * nc] = v_contact.reshape(-1, 3)
    buffer.base_target_pos_t[:n_ik] = v_base
    buffer.base_target_rot_t[:n_ik] = quat_half
    buffer.joint_q_init_t[:n_ik, :3] = v_base
    buffer.joint_q_init_t[:n_ik, 3:7] = quat_full
    buffer.joint_q_init_t[:n_ik, 7:jc] = jq_rev_seed


class Sampler(SamplerBase):
    """Template-projection contact sampler.

    Builds a library of canonical foot-polygon shapes from random-joint
    FK samples (joints clamped to ``default ± fk_joint_range`` to avoid
    wrap-around leg configurations; non-convex-hull samples dropped). At
    query time, each candidate projects a random template at a random
    ``(center, yaw)`` and runs a constrained bipartite assignment over
    ``nc^nc`` combinations to snap feet to morph patches without
    flipping the slot→CCW-rank permutation or collapsing the hull.

    Args:
        cfg: Sampling configuration.
        kin: Newton kinematics model (provides default stance + joint limits).
        foot_body_ids: Newton body indices for the feet.
    """

    def __init__(self, cfg: SamplerCfg, kin: NewtonKinematics, foot_body_ids: list[int]):
        super().__init__(cfg, kin, foot_body_ids)

        # ``foot_ground_offset`` is the foot-body-to-sole z offset, derived
        # from the foot's actual collision geometry (sphere/capsule/box/mesh).
        geom = kin.foot_geometry(foot_body_ids)
        self.foot_ground_offset: float = float(geom["foot_ground_offset"])
        self.default_joint_q = kin.default_joint_q

        # Per-foot nominal angle, canonical-shape library, and standing
        # height are derived from the FK distribution inside
        # :meth:`_compute_foot_reachability`. The derivation is invariant
        # to the URDF default base pose (translation + yaw): base held
        # fixed during FK sampling cancels out when every quantity is
        # measured in the polygon-centroid frame.
        self._compute_foot_reachability()

    def _patch_footprint(self):
        """Footprint the flatness filter should test terrain against.

        A sole with real extent only rests flat where the surface is flat over
        that whole extent, and for a non-square sole which headings fit depends
        on the terrain, so the foot's own outline is used when it is elongated
        enough for orientation to matter. A near-circular foot is reported as a
        disc, which keeps the filter yaw-invariant and cheap.
        """
        hull = self.kin.foot_contact_hulls(self.foot_body_ids)
        extent = hull["vertices"].max(axis=1) - hull["vertices"].min(axis=1)  # [nc, 2]
        length = float(extent[:, 0].max())
        width = float(extent[:, 1].max())
        contact_radius = float(self.cfg.patch.contact_radius)
        if max(length, width) < 2.0 * contact_radius or max(length, width) <= 0.0:
            return CircleFootprintCfg(radius=contact_radius)
        longer, shorter = max(length, width), min(length, width)
        if longer < 1.5 * shorter:
            # Roughly square: a disc inscribed in it says the same thing about
            # flatness without paying for eight rotated masks.
            return CircleFootprintCfg(radius=0.5 * shorter)
        return RectFootprintCfg(length=longer, width=shorter)

    def _compute_foot_reachability(self, seed: int = 0) -> None:
        """Build the canonical FK shape library and derived scalars.

        Samples ``cfg.fk_num_samples`` random revolute joint configurations
        (base held at default pose; per-joint range clamped to
        ``default ± cfg.fk_joint_range`` intersected with URDF limits to
        avoid wrap-around jq whose FK foot positions look plausible but
        whose underlying configuration puts legs on the wrong side of
        the chassis), runs batched FK, canonicalises each polygon
        (centroid-center + plane-fit pitch/roll + per-foot de-nominal),
        drops non-convex-hull samples, and stride-thins the remainder
        to ``cfg.fk_num_retained`` shape samples.

        Stored attributes (all tensors on ``kin.device``):

        * ``_nominal_angle_t`` — per-foot circular-mean angle around
          the polygon centroid [rad], shape ``[nc]``. Defines the
          per-slot "de-nominal" rotation inside
          :func:`~.canonical_shape.canonicalize_shape`.
        * ``_fk_shape_samples`` — canonical foot-polygon shapes,
          ``float32 [n_retained, nc, 3]``.
        * ``_fk_joint_q_rev`` — revolute-joint configurations that
          generated each retained shape, ``float32 [n_retained, n_rev]``.
          Used at query time to seed IK from the matched template's
          own joint configuration, avoiding crossed-leg local minima
          that a uniform default-stance seed is vulnerable to.
        * ``standing_height`` — p95 of ``base_z - min_foot_z`` across
          the FK batch [m]. Drives the IK-seed base z lift.
        """
        kin = self.kin
        device = kin.device
        n_samples = int(self.cfg.fk_num_samples)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        jl_lo = wp.to_torch(kin.model.joint_limit_lower)  # type: ignore[arg-type]
        jl_hi = wp.to_torch(kin.model.joint_limit_upper)  # type: ignore[arg-type]
        # Clamp to ``default ± fk_joint_range`` (with per-pattern overrides)
        # so continuous joints (URDF limits ``±1e10`` after USD conversion)
        # don't produce wrap-around jq where e.g. a lateral hip is swung 90° and the
        # leg routes across the chassis. The foot positions of such samples
        # look fine as a convex quad, but the underlying jq makes IK land
        # on a physically crossed-leg pose.
        import re

        default_q = torch.from_numpy(kin.default_joint_q).float().to(device)
        rev_default = default_q[7:]
        n_rev = rev_default.shape[0]
        joint_range = torch.full((n_rev,), float(self.cfg.fk_joint_range), device=device)
        overrides = dict(self.cfg.fk_joint_range_overrides)
        if overrides:
            # ``joint_names[0]`` is the floating-base root; revolute names
            # follow in 1-based slot order matching jl_lo[6:] / jl_hi[6:].
            rev_names = kin.joint_names[1 : 1 + n_rev]
            for pattern, clamp in overrides.items():
                regex = re.compile(pattern)
                for i, name in enumerate(rev_names):
                    if regex.fullmatch(name):
                        joint_range[i] = float(clamp)
        rev_lo = torch.maximum(jl_lo[6:], rev_default - joint_range)
        rev_hi = torch.minimum(jl_hi[6:], rev_default + joint_range)

        # Uniform random joint sampling across the clamped physical range.
        gen = torch.Generator(device=device).manual_seed(seed)
        jq = default_q.unsqueeze(0).expand(n_samples, -1).contiguous()
        rand_u = torch.rand(n_samples, n_rev, device=device, generator=gen)
        jq[:, 7 : 7 + n_rev] = rand_u * (rev_hi - rev_lo) + rev_lo

        body_q_wp, _ = kin.eval_fk_batched(wp.from_torch(jq))
        body_q_t = wp.to_torch(body_q_wp).view(n_samples, -1, 7)  # type: ignore[arg-type]
        foot_ids_t = torch.tensor(self.foot_body_ids, device=device, dtype=torch.long)
        foot_xy = body_q_t[:, foot_ids_t, :2]  # [n_samples, nc, 2]

        # Per-foot nominal angle: circular mean of the foot's angle around
        # the per-sample polygon centroid. Robust to ±π wraparound and
        # independent of URDF-declared foot offsets — each foot's random-
        # joint-induced xy distribution is unimodal around its hip-outward
        # direction, so the circular mean recovers that direction.
        centroid_xy = foot_xy.mean(dim=1, keepdim=True)  # [n_samples, 1, 2]
        rel = foot_xy - centroid_xy
        theta = torch.atan2(rel[..., 1], rel[..., 0])
        nominal = torch.atan2(torch.sin(theta).mean(dim=0), torch.cos(theta).mean(dim=0))  # [nc]
        self._nominal_angle_t = nominal.float().contiguous()

        # Standing-height IK seed prior: p95 of ``base_z - min_foot_z``
        # across the FK batch. Using base held fixed at default and
        # quantile-aggregating over random joints isolates the joint-
        # induced variation from the URDF default base z.
        foot_ids_t_world = torch.tensor(self.foot_body_ids, device=device, dtype=torch.long)
        foot_z_min = body_q_t[:, foot_ids_t_world, 2].amin(dim=-1)  # [n_samples]
        base_z = body_q_t[:, 0, 2]  # [n_samples]
        self.standing_height = float(torch.quantile(base_z - foot_z_min, 0.95).item())

        # Per-foot canonical shape. Canonicalize each FK-produced polygon
        # the same way query time will (centroid-center + plane-fit +
        # per-foot de-nominal); NN in this space is a pure shape match.
        foot_xyz = body_q_t[:, foot_ids_t, :3]  # [n_samples, nc, 3]

        # Drop samples whose ``nc`` feet don't form a convex polygon in xy
        # (self-intersecting CCW-around-centroid walk, or one foot inside
        # the triangle of the others) OR whose per-slot CCW ordering
        # disagrees with the robot's nominal ordering — the latter catches
        # fully-mirrored stances (e.g., all front legs crossed + all back
        # legs crossed) that keep the hull convex but route every leg
        # across the chassis. Mirrored templates would seed IK into a
        # legs-across-body basin even though ``tgt_perm == tpl_perm`` at
        # query time.
        nc = len(self.foot_body_ids)
        xy = foot_xyz[..., :2]
        centroid = xy.mean(dim=1, keepdim=True)
        rel_xy = xy - centroid
        angles = torch.atan2(rel_xy[..., 1], rel_xy[..., 0])
        order_ccw = angles.argsort(dim=-1)
        if nc >= 3:
            sorted_xy = torch.gather(xy, 1, order_ccw.unsqueeze(-1).expand(-1, -1, 2))
            edges = sorted_xy.roll(-1, dims=1) - sorted_xy
            next_edges = edges.roll(-1, dims=1)
            cross = edges[..., 0] * next_edges[..., 1] - edges[..., 1] * next_edges[..., 0]
            hull_convex = (cross > 0).all(dim=-1) | (cross < 0).all(dim=-1)  # [n_samples]
        else:
            # Fewer than three feet enclose no area, so every failure the
            # convexity walk detects -- a self-intersecting boundary, or a foot
            # sitting inside the others -- cannot arise. Its cross products are
            # identically zero here, which would reject every sample; the
            # ordering check below is what catches the crossed-leg stances that
            # actually matter for a biped.
            hull_convex = torch.ones(xy.shape[0], dtype=torch.bool, device=xy.device)

        # Nominal slot→CCW-rank permutation (the robot's "correct" layout).
        nominal_perm = self._nominal_angle_t.argsort()  # [nc]
        perm_correct = (order_ccw == nominal_perm.unsqueeze(0)).all(dim=-1)  # [n_samples]

        hull_valid = hull_convex & perm_correct
        n_hull_valid = int(hull_valid.sum().item())
        if n_hull_valid < n_samples:
            foot_xyz = foot_xyz[hull_valid]
            jq_rev = jq[hull_valid, 7 : 7 + n_rev]
        else:
            jq_rev = jq[:, 7 : 7 + n_rev]

        canon_all = canonicalize_shape(foot_xyz, self._nominal_angle_t)
        n_retained = int(min(self.cfg.fk_num_retained, canon_all.shape[0]))
        if n_retained < canon_all.shape[0]:
            # FPS-thin on the flattened canonical shape so each retained
            # template represents a geometrically distinct stance. Stride
            # subsampling would just give another uniform draw from the
            # joint-space distribution — FPS ensures random ``tpl_idx``
            # at query time evenly covers the FK shape manifold.
            flat = canon_all.reshape(canon_all.shape[0], -1)
            keep_idx = grid_bucket_downsample(flat, n_retained)
            self._fk_shape_samples = canon_all[keep_idx].contiguous()
            self._fk_joint_q_rev = jq_rev[keep_idx].contiguous()
        else:
            self._fk_shape_samples = canon_all.contiguous()
            self._fk_joint_q_rev = jq_rev.contiguous()

        # Cache the maximum per-foot distance from template centroid:
        # this bounds how far a projected foot can sit from the candidate
        # centre, and (plus snap_distance with a margin) becomes the
        # ``spatial_topk_xy`` query radius. Using the FK-derived bound
        # keeps the hash-grid search tight for small robots and elastic
        # for larger ones without hard-coded heuristics.
        self._fk_max_foot_reach = float(self._fk_shape_samples[..., :2].norm(dim=-1).max().item())

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        self.init_info = (
            f"reachability: {n_samples} FK samples in {dt:.3f}s"
            f" ({n_samples - n_hull_valid} dropped for non-convex hull,"
            f" retained {self._fk_shape_samples.shape[0]} shape samples)"
        )

    def sizing(self, n_desired: int) -> SamplerSizing:
        """Back-derive stage sizes for a given final-robot target.

        Uses yield-rate knobs from :class:`SamplerSizingCfg` so a single
        ``n_desired`` determines every internal stage size.
        """
        sz: SamplerSizingCfg = self.cfg.sizing
        return compute_sampler_sizing(
            n_desired,
            final_fps_oversample=sz.final_fps_oversample,
            criteria_yield=sz.criteria_yield,
            polygon_fps_oversample=sz.polygon_fps_oversample,
            polygon_assembly_yield=sz.polygon_assembly_yield,
            morph_patch_oversample=sz.morph_patch_oversample,
            patches_per_polygon=len(self.foot_body_ids),
        )

    @contextmanager
    def _time(self, name: str):
        """Record wall time for a sampler sub-phase, with CUDA sync on enter/exit."""
        dev = self.kin.device
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            self.sub_timings[name] = self.sub_timings.get(name, 0.0) + (time.perf_counter() - t0)

    def __call__(
        self,
        wp_mesh: wp.Mesh,
        origin: np.ndarray,
        buffer: RetargetBuffer,
        n_desired: int,
    ) -> SamplerOutput:
        """Template-projection query path.

        For each candidate: pick a random morph patch as center, a random
        yaw, and a random template; un-canonicalize the template at that
        pose; solve a constrained LSA that snaps feet to morph patches
        without flipping slot→CCW-rank or collapsing the hull. Feet within
        ``terrain_snap_distance`` of their assigned patch become contact
        targets; feet farther away become air targets (z clamped above
        local terrain).
        """
        self.sub_timings.clear()
        patch: PatchSamplingCfg = self.cfg.patch
        nc = buffer.num_contacts
        max_n = buffer.max_candidates

        sizing = self.sizing(n_desired)
        num_patches = max(200, sizing.n_morph_patches)

        with self._time("morph"):
            fc_cfg = MorphologicalPatchSamplingCfg(
                num_patches=num_patches,
                footprint=self._patch_footprint(),
                max_height_diff=patch.max_height_diff,
                horizontal_scale=patch.horizontal_scale,
                oversample_ratio=patch.oversample_ratio,
                x_range=patch.x_range if patch.x_range is not None else (-1e6, 1e6),
                y_range=patch.y_range if patch.y_range is not None else (-1e6, 1e6),
            )
            fp = fc_cfg.func(wp_mesh, origin, fc_cfg)
            dev_str = fp.device
            origin_t = torch.tensor(origin, dtype=torch.float, device=dev_str)
            fp[:, :3] += origin_t
            patch_pts = fp[:, :3].contiguous()  # [N_p, 3]
            # Orientation the surface demands of anything resting here, plus
            # which headings the footprint fits. Both ride along to the fused
            # kernel so a foot is matched to a patch it can actually lie on.
            patch_quats = fp[:, 3:7].contiguous()  # [N_p, 4] xyzw
            patch_admissible = fp[:, 7].to(torch.int32).contiguous()  # [N_p]
        for _sub_name, _sub_dt in MORPH_TIMINGS.items():
            key = f"morph.{_sub_name}"
            self.sub_timings[key] = self.sub_timings.get(key, 0.0) + _sub_dt

        # ``K`` (polygon pool) lives in cheap per-polygon scratches local
        # to this sampler — independent of the buffer's per-body slots.
        # ``target_n`` (post-FPS, what writes into the buffer) is the only
        # value clamped to ``max_n``.
        K = sizing.max_neighborhoods
        target_n = min(n_desired * sizing.oversample_candidates, max_n)

        torch.manual_seed(42)
        if torch.device(dev_str).type == "cuda":
            torch.cuda.manual_seed_all(42)

        with self._time("project"):
            # Sampler config -- pulled from cfg once and forwarded to the
            # fused kernel below. ``force_all_snap`` (every foot must
            # contact a patch) is set when ``min_contacts>=nc`` and
            # widens the search radius to ``query_radius``; otherwise
            # contacts are gated at the tighter ``terrain_snap_distance``.
            radius = float(self.cfg.terrain_snap_distance)
            outward_pen = float(self.cfg.outward_snap_penalty)
            raw_mc = int(self.cfg.min_contacts)
            min_contacts = nc if (raw_mc < 0 or raw_mc > nc) else max(1, raw_mc)
            force_all_snap = min_contacts >= nc
            query_radius = self._fk_max_foot_reach + 4.0 * radius

            # Single Warp kernel: per candidate slot, sample
            # ``(center, yaw, tpl_idx)``, un-canonicalize the FK template,
            # query the morph-patch hashgrid for the top-nc nearest
            # patches per foot, then iterate the foot-rank permutations
            # with cost / distinctness / force-contact / winding /
            # convex-hull penalties and emit the best survivor's
            # outputs. All per-thread intermediates live in registers,
            # so the only K-sized memory the sampler needs is the
            # output tensors ``run_fused_sampler`` allocates internally.
            if torch.device(dev_str).type == "cuda":
                torch.cuda.synchronize()
            fused_t0 = time.perf_counter()
            outputs = run_fused_sampler(
                seed=42,
                K=K,
                patch_pts=patch_pts,
                patch_quats=patch_quats,
                patch_admissible=patch_admissible,
                num_yaw_bins=MORPH_YAW_BINS,
                fk_shape_samples=self._fk_shape_samples,
                nominal_angles=self._nominal_angle_t,
                radius=radius,
                query_radius=query_radius,
                outward_pen=outward_pen,
                force_all_snap=force_all_snap,
                foot_ground_offset=self.foot_ground_offset,
            )
            if torch.device(dev_str).type == "cuda":
                torch.cuda.synchronize()
            fused_dt = time.perf_counter() - fused_t0
            self.sub_timings["project.fused_kernel"] = self.sub_timings.get("project.fused_kernel", 0.0) + fused_dt
            throughput = K / max(fused_dt, 1.0e-12) / 1.0e6
            print(
                f"[contact_sampling] fused single-kernel sampler over K={K} placements "
                f"in {fused_dt:.3f} s ({throughput:.2f} M placements/s)",
                flush=True,
            )
            yaws = outputs["yaws"]
            tpl_idx = outputs["tpl_idx"]
            is_contact_full = outputs["is_contact_full"]
            contact_ik = outputs["contact_ik"]
            contact_rot = outputs["contact_rot"]
            n_found = outputs["n_found"]
            no_convex = outputs["no_convex"]

            out_of_reach = n_found < min_contacts
            valid = ~out_of_reach & ~no_convex
            all_valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)
            n_all_valid = all_valid_idx.shape[0]
            n_valid = min(n_all_valid, target_n, max_n)

        with self._time("sampler_fps"):
            if n_all_valid > n_valid:
                # Compute the centroid mean over full ``contact_ik`` first
                # then index, instead of ``contact_ik[idx].mean(...)``. The
                # latter allocates a transient ``[n_valid, nc, 3]`` (~1.4 GiB
                # at K=60M with 50% valid) before reducing to ``[n_valid, 3]``;
                # the former allocates only ``[K, 3]`` once (~720 MiB at
                # K=60M) and indexes that to a smaller ``[n_valid, 3]``.
                centroid_xyz_full = contact_ik.mean(dim=-2)
                centroid_xyz = centroid_xyz_full[all_valid_idx]
                del centroid_xyz_full
                local_idx = grid_bucket_downsample(centroid_xyz, n_valid)
                valid_idx = all_valid_idx[local_idx]
            else:
                valid_idx = all_valid_idx

        reject = {
            "out_of_reach": int(out_of_reach.sum()),
            "non_convex_stance": int((no_convex & ~out_of_reach).sum()),
        }
        diagnostics: dict[str, object] = {
            "n_contact_histogram": torch.bincount(n_found[valid_idx], minlength=nc + 1).detach().clone(),
        }

        if n_valid == 0:
            buffer.num_written = 0
            buffer.num_geometry_valid = 0
            return SamplerOutput(num_written=0, reject_stats=reject, diagnostics=diagnostics)

        with self._time("prepare_ik"):
            v_contact = contact_ik[valid_idx].contiguous()  # [n_valid, nc, 3]
            v_contact_rot = contact_rot[valid_idx].contiguous()  # [n_valid, nc, 4]
            is_contact_ik = is_contact_full[valid_idx].contiguous()
            centroid_sel = v_contact.mean(dim=-2)
            v_base = torch.stack(
                [centroid_sel[..., 0], centroid_sel[..., 1], centroid_sel[..., 2] + self.standing_height],
                dim=-1,
            )
            v_yaw = yaws[valid_idx].contiguous()
            # Seed IK from the matched template's own joint configuration
            # so the solver starts in the correct basin of attraction —
            # crucial for avoiding crossed-leg local minima on twisted
            # poses where default-stance seeds have to route legs through
            # each other.
            jq_rev_seed = self._fk_joint_q_rev[tpl_idx[valid_idx]].contiguous()

            n_ik = v_contact.shape[0]
            _prepare_ik_batched(v_contact, v_base, v_yaw, jq_rev_seed, buffer)
            # Orientation each foot must adopt to lie on its patch, kept
            # beside the position target for the rotation objective.
            buffer.contact_target_rot_t[: n_ik * nc] = v_contact_rot.reshape(-1, 4)
            buffer._geom_valid[:n_ik] = True
            buffer.num_written = n_ik
            buffer.num_geometry_valid = n_ik

        return SamplerOutput(
            num_written=n_ik,
            reject_stats=reject,
            is_contact=is_contact_ik,
            diagnostics=diagnostics,
        )
