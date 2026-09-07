# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fused single-kernel polygon sampler for the contact-sampling stage.

This module contains the **end-to-end** Warp kernel that, in a single
``wp.launch`` over ``K`` candidate slots, does:

1. Per-thread RNG init (deterministic from ``seed`` + thread id).
2. Sample one ``(center_idx, yaw, tpl_idx)`` triple per slot.
3. Un-canonicalize the chosen FK template into world-frame foot xyz
   under the sampled yaw + center.
4. Query the prebuilt morph-patch ``HashGrid`` for the top-``nc``
   nearest patches per foot.
5. Precompute per-``(foot, rank)`` cost / target_xy / contact-bit so
   the permutation loop becomes a sum + small validity checks rather
   than a square-root-heavy reconstruction per combo.
6. Iterate the foot-rank permutations (cost / distinctness /
   force-must-contact / winding / convex-hull) and pick the best
   survivor.
7. Emit per-slot outputs (``is_contact``, ``contact_ik``, ``n_found``,
   ``no_convex``, ``yaws``, ``tpl_idx``).

Because every per-slot intermediate lives in registers, this fused
kernel **eliminates the chunking infrastructure** the previous
implementation needed: there is no ``[Kc, C, nc, ...]`` family of
tensors to chunk around, no per-chunk budget, no ``empty_cache``
dance. The only allocations are the K-sized output tensors that the
downstream FPS + IK stages consume, and they're the smallest dtypes
that fit each value.

**Scope**: two and four feet, each with its own kernel. They share the
sampling, template projection, patch query, and per-rank precompute, and
differ only in the stance validity test -- four feet must form a convex
quad in the template's winding order, whereas two need only keep their
left-right ordering, having no area to be convex about. Other foot counts
raise :class:`NotImplementedError` from the launcher.
"""

from __future__ import annotations

import math

import torch
import warp as wp

from .spatial_topk import build_spatial_grid_xy

# Compile-time constants -- kernel inner loops unroll cleanly at nc=4.
NC: wp.constant = wp.constant(4)
N_COMBOS: wp.constant = wp.constant(24)  # 4! permutations -- distinctness penalty makes the rest waste.
TWO_PI: wp.constant = wp.constant(float(2.0 * math.pi))


# ----------------------------------------------------------------- structs


@wp.struct
class _FootTopK:
    """Top-``nc`` patch hits for one foot, in ascending-distance order.

    Empty slots come back as ``idx=-1``, ``dist=+inf``. The launcher
    clamps the indices non-negative downstream so the kernel's later
    ``patch_pts[idx]`` reads never go out of bounds; the ``+inf``
    distance keeps the contact mask correctly false for those slots.
    """

    idx: wp.vec4i
    dist: wp.vec4


@wp.struct
class _FootRanks:
    """Per-(foot, rank) precomputed values, packed by rank into vec4s.

    Computed once per thread up-front so the permutation loop becomes
    a 4-lookup sum + validity check rather than a per-combo
    reconstruction with its own sqrt and branchy gather.

    For each of the 4 ranks of a single foot:

    * ``contact[r]``: 1 if the foot's r-th-nearest patch is within
      ``effective_radius`` (a true contact target), else 0 (an
      "air" target -- foot stays at its template-projected xy).
    * ``target_x[r]`` / ``target_y[r]``: the xy the foot will end up
      at if rank ``r`` is chosen (patch xy if contact, projected xy
      if air).
    * ``cost[r]``: per-foot contribution to the LSA cost (``chosen_d
      + outward_pen * outward`` if contact, ``radius`` if air). The
      total cost of a permutation is the sum of four such per-foot
      values, one per foot's chosen rank.
    """

    contact: wp.vec4i
    target_x: wp.vec4
    target_y: wp.vec4
    cost: wp.vec4


# ---------------------------------------------------------------- helpers


@wp.func
def _topk4_query(
    grid_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    qx: wp.float32,
    qy: wp.float32,
    radius: wp.float32,
) -> _FootTopK:
    """Query ``grid_id`` for the 4 nearest patches around ``(qx, qy)``."""
    out = _FootTopK()
    out.idx = wp.vec4i(-1, -1, -1, -1)
    out.dist = wp.vec4(wp.inf, wp.inf, wp.inf, wp.inf)

    nb_iter = wp.hash_grid_query(grid_id, wp.vec3(qx, qy, 0.0), radius)
    nb = int(0)
    while wp.hash_grid_query_next(nb_iter, nb):
        p = points[nb]
        dx = p[0] - qx
        dy = p[1] - qy
        d = wp.sqrt(dx * dx + dy * dy)
        if d <= radius:
            # Find slot with the current max distance.
            max_d = out.dist[0]
            max_slot = int(0)
            for j in range(1, 4):
                if out.dist[j] > max_d:
                    max_d = out.dist[j]
                    max_slot = j
            if d < max_d:
                out.dist[max_slot] = d
                out.idx[max_slot] = nb

    # 4-element sorting network (5 compare-swaps) -- ascending by distance.
    if out.dist[0] > out.dist[1]:
        t = out.dist[0]
        out.dist[0] = out.dist[1]
        out.dist[1] = t
        ti = out.idx[0]
        out.idx[0] = out.idx[1]
        out.idx[1] = ti
    if out.dist[2] > out.dist[3]:
        t = out.dist[2]
        out.dist[2] = out.dist[3]
        out.dist[3] = t
        ti = out.idx[2]
        out.idx[2] = out.idx[3]
        out.idx[3] = ti
    if out.dist[0] > out.dist[2]:
        t = out.dist[0]
        out.dist[0] = out.dist[2]
        out.dist[2] = t
        ti = out.idx[0]
        out.idx[0] = out.idx[2]
        out.idx[2] = ti
    if out.dist[1] > out.dist[3]:
        t = out.dist[1]
        out.dist[1] = out.dist[3]
        out.dist[3] = t
        ti = out.idx[1]
        out.idx[1] = out.idx[3]
        out.idx[3] = ti
    if out.dist[1] > out.dist[2]:
        t = out.dist[1]
        out.dist[1] = out.dist[2]
        out.dist[2] = t
        ti = out.idx[1]
        out.idx[1] = out.idx[2]
        out.idx[2] = ti

    # Sentinel-safe: clamp negative indices to 0 so downstream
    # ``patch_*[idx]`` reads can't OOB. The matching +inf distance
    # keeps the contact mask correctly false.
    out.idx = wp.max(out.idx, wp.vec4i(0, 0, 0, 0))
    return out


@wp.func
def _argsort4(a: wp.float32, b: wp.float32, c: wp.float32, d: wp.float32) -> wp.vec4i:
    """Sorting-network argsort of 4 floats (5 compare-swaps), ascending."""
    v0 = a
    v1 = b
    v2 = c
    v3 = d
    i0 = int(0)
    i1 = int(1)
    i2 = int(2)
    i3 = int(3)
    if v0 > v1:
        tv = v0
        v0 = v1
        v1 = tv
        ti = i0
        i0 = i1
        i1 = ti
    if v2 > v3:
        tv = v2
        v2 = v3
        v3 = tv
        ti = i2
        i2 = i3
        i3 = ti
    if v0 > v2:
        tv = v0
        v0 = v2
        v2 = tv
        ti = i0
        i0 = i2
        i2 = ti
    if v1 > v3:
        tv = v1
        v1 = v3
        v3 = tv
        ti = i1
        i1 = i3
        i3 = ti
    if v1 > v2:
        tv = v1
        v1 = v2
        v2 = tv
        ti = i1
        i1 = i2
        i2 = ti
    return wp.vec4i(i0, i1, i2, i3)


@wp.func
def _has_dup4(a: wp.int32, b: wp.int32, c: wp.int32, d: wp.int32) -> int:
    """1 if any two of the four ints are equal, else 0."""
    if a == b:
        return 1
    if a == c:
        return 1
    if a == d:
        return 1
    if b == c:
        return 1
    if b == d:
        return 1
    if c == d:
        return 1
    return 0


@wp.func
def _foot_ranks(
    top: _FootTopK,
    proj_x: wp.float32,
    proj_y: wp.float32,
    tpl_centroid_x: wp.float32,
    tpl_centroid_y: wp.float32,
    tpl_r: wp.float32,
    patch_xy: wp.array(dtype=wp.vec3),
    radius: wp.float32,
    effective_radius: wp.float32,
    outward_pen: wp.float32,
) -> _FootRanks:
    """Precompute per-rank cost / target / contact for one foot.

    Each foot has 4 candidate ranks (its 4 nearest patches in distance
    order). Whether a permutation picks rank ``r`` for this foot is
    decided by the combo loop downstream; everything that depends only
    on ``(foot, rank)`` -- and not on the combo as a whole -- is
    computed once here so the combo loop becomes a sum + validity
    check.
    """
    out = _FootRanks()
    out.contact = wp.vec4i(0, 0, 0, 0)
    out.target_x = wp.vec4(0.0, 0.0, 0.0, 0.0)
    out.target_y = wp.vec4(0.0, 0.0, 0.0, 0.0)
    out.cost = wp.vec4(0.0, 0.0, 0.0, 0.0)

    for r in range(4):
        chi = top.idx[r]
        chd = top.dist[r]
        if chd < effective_radius:
            out.contact[r] = 1
            px = patch_xy[chi]
            out.target_x[r] = px[0]
            out.target_y[r] = px[1]
            patch_r = wp.sqrt(
                (px[0] - tpl_centroid_x) * (px[0] - tpl_centroid_x)
                + (px[1] - tpl_centroid_y) * (px[1] - tpl_centroid_y)
            )
            outward = wp.max(patch_r - tpl_r, 0.0)
            out.cost[r] = chd + outward_pen * outward
        else:
            out.contact[r] = 0
            out.target_x[r] = proj_x
            out.target_y[r] = proj_y
            out.cost[r] = radius

    return out


@wp.func
def _yaw_admits(bits: wp.int32, yaw: wp.float32, num_yaw: int) -> int:
    """1 if the patch's footprint fits a heading of ``yaw``, else 0.

    The flatness search records one bit per tested heading over a half turn,
    a rectangle being symmetric under 180 degrees. A heading is accepted when
    the nearest tested bin is set, so a foot is only matched to a patch its
    sole actually fits on.
    """
    if bits == -1:
        return 1
    span = wp.pi / float(num_yaw)
    folded = wp.mod(yaw, wp.pi)
    if folded < 0.0:
        folded = folded + wp.pi
    bin_index = int(wp.round(folded / span)) % num_yaw
    if (bits >> bin_index) & 1 == 1:
        return 1
    return 0


@wp.func
def _yaw_admits_all4(
    bits: wp.array(dtype=wp.int32),
    e0: wp.int32,
    c0: wp.int32,
    e1: wp.int32,
    c1: wp.int32,
    e2: wp.int32,
    c2: wp.int32,
    e3: wp.int32,
    c3: wp.int32,
    yaw: wp.float32,
    num_yaw: int,
) -> int:
    """1 when every contact foot's patch admits the stance heading."""
    if c0 == 1 and _yaw_admits(bits[e0], yaw, num_yaw) == 0:
        return 0
    if c1 == 1 and _yaw_admits(bits[e1], yaw, num_yaw) == 0:
        return 0
    if c2 == 1 and _yaw_admits(bits[e2], yaw, num_yaw) == 0:
        return 0
    if c3 == 1 and _yaw_admits(bits[e3], yaw, num_yaw) == 0:
        return 0
    return 1


# ----------------------------------------------------------------- kernel


@wp.kernel
def _fused_sampler_kernel(
    # ---- scalars / RNG ----
    seed: int,
    K: int,
    n_pts: int,
    n_tpl: int,
    radius: wp.float32,
    effective_radius: wp.float32,
    query_radius: wp.float32,
    outward_pen: wp.float32,
    force_all_snap: int,
    foot_ground_offset: wp.float32,
    # ---- precomputed tables ----
    patch_pts: wp.array(dtype=wp.vec3),  # [N_p] real xyz
    patch_pts_z0: wp.array(dtype=wp.vec3),  # [N_p] same xyz with z=0 (hashgrid points)
    patch_grid_id: wp.uint64,  # built over patch_pts_z0
    patch_quats: wp.array(dtype=wp.quat),
    patch_admissible: wp.array(dtype=wp.int32),
    num_yaw_bins: int,
    fk_shape_samples: wp.array2d(dtype=wp.vec3),  # [n_tpl, NC] canonical foot xyz
    cos_n: wp.vec4,  # per-foot cos(nominal angle)
    sin_n: wp.vec4,  # per-foot sin(nominal angle)
    combo: wp.array2d(dtype=wp.int32),  # [N_COMBOS, NC] permutations
    # ---- outputs (allocated by the launcher) ----
    yaws_out: wp.array(dtype=wp.float32),  # [K]
    tpl_idx_out: wp.array(dtype=wp.int32),  # [K]
    is_contact_out: wp.array2d(dtype=wp.uint8),  # [K, NC]
    contact_ik_out: wp.array2d(dtype=wp.vec3),  # [K, NC]
    contact_rot_out: wp.array2d(dtype=wp.quat),  # [K, NC]
    n_found_out: wp.array(dtype=wp.uint8),  # [K]
    no_convex_out: wp.array(dtype=wp.uint8),  # [K]
):
    """One thread per candidate slot. Does the full sampler work end-to-end."""
    k = wp.tid()
    if k >= K:
        return

    # ----- 1. Per-thread RNG -----
    state = wp.rand_init(seed, k)
    center_idx = wp.randi(state, 0, n_pts)
    yaw = wp.randf(state) * TWO_PI
    tpl_idx_val = wp.randi(state, 0, n_tpl)
    yaws_out[k] = yaw
    tpl_idx_out[k] = tpl_idx_val

    center = patch_pts[center_idx]
    cos_y = wp.cos(yaw)
    sin_y = wp.sin(yaw)

    # ----- 2. Un-canonicalize template into world-frame foot positions -----
    tc0 = fk_shape_samples[tpl_idx_val, 0]
    tc1 = fk_shape_samples[tpl_idx_val, 1]
    tc2 = fk_shape_samples[tpl_idx_val, 2]
    tc3 = fk_shape_samples[tpl_idx_val, 3]

    # mid = R(nominal_angle) @ tpl_canon, world = R(yaw) @ mid + center.
    mx0 = cos_n[0] * tc0[0] - sin_n[0] * tc0[1]
    my0 = sin_n[0] * tc0[0] + cos_n[0] * tc0[1]
    mx1 = cos_n[1] * tc1[0] - sin_n[1] * tc1[1]
    my1 = sin_n[1] * tc1[0] + cos_n[1] * tc1[1]
    mx2 = cos_n[2] * tc2[0] - sin_n[2] * tc2[1]
    my2 = sin_n[2] * tc2[0] + cos_n[2] * tc2[1]
    mx3 = cos_n[3] * tc3[0] - sin_n[3] * tc3[1]
    my3 = sin_n[3] * tc3[0] + cos_n[3] * tc3[1]

    wx0 = cos_y * mx0 - sin_y * my0 + center[0]
    wy0 = sin_y * mx0 + cos_y * my0 + center[1]
    wz0 = tc0[2] + center[2]
    wx1 = cos_y * mx1 - sin_y * my1 + center[0]
    wy1 = sin_y * mx1 + cos_y * my1 + center[1]
    wz1 = tc1[2] + center[2]
    wx2 = cos_y * mx2 - sin_y * my2 + center[0]
    wy2 = sin_y * mx2 + cos_y * my2 + center[1]
    wz2 = tc2[2] + center[2]
    wx3 = cos_y * mx3 - sin_y * my3 + center[0]
    wy3 = sin_y * mx3 + cos_y * my3 + center[1]
    wz3 = tc3[2] + center[2]

    # Template centroid + per-foot template radii (one-time per thread).
    tcx = (wx0 + wx1 + wx2 + wx3) * 0.25
    tcy = (wy0 + wy1 + wy2 + wy3) * 0.25
    tpl_r0 = wp.sqrt((wx0 - tcx) * (wx0 - tcx) + (wy0 - tcy) * (wy0 - tcy))
    tpl_r1 = wp.sqrt((wx1 - tcx) * (wx1 - tcx) + (wy1 - tcy) * (wy1 - tcy))
    tpl_r2 = wp.sqrt((wx2 - tcx) * (wx2 - tcx) + (wy2 - tcy) * (wy2 - tcy))
    tpl_r3 = wp.sqrt((wx3 - tcx) * (wx3 - tcx) + (wy3 - tcy) * (wy3 - tcy))

    # Template winding order (one argsort per thread, not per combo).
    tpl_perm = _argsort4(
        wp.atan2(wy0 - tcy, wx0 - tcx),
        wp.atan2(wy1 - tcy, wx1 - tcx),
        wp.atan2(wy2 - tcy, wx2 - tcx),
        wp.atan2(wy3 - tcy, wx3 - tcx),
    )

    # ----- 3. Top-NC hashgrid query per foot -----
    top0 = _topk4_query(patch_grid_id, patch_pts_z0, wx0, wy0, query_radius)
    top1 = _topk4_query(patch_grid_id, patch_pts_z0, wx1, wy1, query_radius)
    top2 = _topk4_query(patch_grid_id, patch_pts_z0, wx2, wy2, query_radius)
    top3 = _topk4_query(patch_grid_id, patch_pts_z0, wx3, wy3, query_radius)

    # ----- 4. Per-(foot, rank) precompute -----
    # Each foot's 4 ranks contribute (cost, target_xy, contact-bit) that
    # only depend on this foot's data, not on which permutation we're
    # evaluating. Computing them once trims the combo loop from
    # ~50 ops/iter (with sqrt per combo) to ~30 ops/iter (just lookup +
    # sum + small validity).
    fr0 = _foot_ranks(top0, wx0, wy0, tcx, tcy, tpl_r0, patch_pts_z0, radius, effective_radius, outward_pen)
    fr1 = _foot_ranks(top1, wx1, wy1, tcx, tcy, tpl_r1, patch_pts_z0, radius, effective_radius, outward_pen)
    fr2 = _foot_ranks(top2, wx2, wy2, tcx, tcy, tpl_r2, patch_pts_z0, radius, effective_radius, outward_pen)
    fr3 = _foot_ranks(top3, wx3, wy3, tcx, tcy, tpl_r3, patch_pts_z0, radius, effective_radius, outward_pen)

    # Per-foot "is there at least one in-radius patch" (force-contact gate).
    has_opt0 = int(0)
    if top0.dist[0] < radius:
        has_opt0 = 1
    has_opt1 = int(0)
    if top1.dist[0] < radius:
        has_opt1 = 1
    has_opt2 = int(0)
    if top2.dist[0] < radius:
        has_opt2 = 1
    has_opt3 = int(0)
    if top3.dist[0] < radius:
        has_opt3 = 1

    # ----- 5. LSA permutation loop -----
    best_cost = float(1.0e30)
    best_c = int(-1)

    for c in range(N_COMBOS):
        r0 = combo[c, 0]
        r1 = combo[c, 1]
        r2 = combo[c, 2]
        r3 = combo[c, 3]

        contact0 = fr0.contact[r0]
        contact1 = fr1.contact[r1]
        contact2 = fr2.contact[r2]
        contact3 = fr3.contact[r3]

        # Force-must-contact gate.
        if force_all_snap == 0:
            if has_opt0 == 1 and contact0 == 0:
                continue
            if has_opt1 == 1 and contact1 == 0:
                continue
            if has_opt2 == 1 and contact2 == 0:
                continue
            if has_opt3 == 1 and contact3 == 0:
                continue

        # Distinctness: contact feet must pick distinct patches; air feet
        # use a per-foot negative sentinel so they never collide.
        e0 = -1
        if contact0 == 1:
            e0 = top0.idx[r0]
        e1 = -2
        if contact1 == 1:
            e1 = top1.idx[r1]
        e2 = -3
        if contact2 == 1:
            e2 = top2.idx[r2]
        e3 = -4
        if contact3 == 1:
            e3 = top3.idx[r3]
        if _has_dup4(e0, e1, e2, e3) == 1:
            continue

        # Orientation: skip patches whose footprint does not fit this stance's
        # heading. Discs set every bit, so a point-footed robot is unaffected.
        admits = _yaw_admits_all4(
            patch_admissible,
            e0,
            contact0,
            e1,
            contact1,
            e2,
            contact2,
            e3,
            contact3,
            yaw,
            num_yaw_bins,
        )
        if admits == 0:
            continue

        # Cost: precomputed per-(foot, rank) sum.
        cost = fr0.cost[r0] + fr1.cost[r1] + fr2.cost[r2] + fr3.cost[r3]
        if cost >= best_cost:
            continue

        # Hull validity: target winding matches template + strict convex.
        tx0 = fr0.target_x[r0]
        ty0 = fr0.target_y[r0]
        tx1 = fr1.target_x[r1]
        ty1 = fr1.target_y[r1]
        tx2 = fr2.target_x[r2]
        ty2 = fr2.target_y[r2]
        tx3 = fr3.target_x[r3]
        ty3 = fr3.target_y[r3]

        ttx = (tx0 + tx1 + tx2 + tx3) * 0.25
        tty = (ty0 + ty1 + ty2 + ty3) * 0.25
        tgt_perm = _argsort4(
            wp.atan2(ty0 - tty, tx0 - ttx),
            wp.atan2(ty1 - tty, tx1 - ttx),
            wp.atan2(ty2 - tty, tx2 - ttx),
            wp.atan2(ty3 - tty, tx3 - ttx),
        )
        if (
            tgt_perm[0] != tpl_perm[0]
            or tgt_perm[1] != tpl_perm[1]
            or tgt_perm[2] != tpl_perm[2]
            or tgt_perm[3] != tpl_perm[3]
        ):
            continue

        # Vec4-indexed permutation gather (replaces 16 if-branches).
        tx_v = wp.vec4(tx0, tx1, tx2, tx3)
        ty_v = wp.vec4(ty0, ty1, ty2, ty3)
        sx0 = tx_v[tgt_perm[0]]
        sy0 = ty_v[tgt_perm[0]]
        sx1 = tx_v[tgt_perm[1]]
        sy1 = ty_v[tgt_perm[1]]
        sx2 = tx_v[tgt_perm[2]]
        sy2 = ty_v[tgt_perm[2]]
        sx3 = tx_v[tgt_perm[3]]
        sy3 = ty_v[tgt_perm[3]]

        # Edges + cyclic cross products.
        ex0 = sx1 - sx0
        ey0 = sy1 - sy0
        ex1 = sx2 - sx1
        ey1 = sy2 - sy1
        ex2 = sx3 - sx2
        ey2 = sy3 - sy2
        ex3 = sx0 - sx3
        ey3 = sy0 - sy3
        cr0 = ex0 * ey1 - ey0 * ex1
        cr1 = ex1 * ey2 - ey1 * ex2
        cr2 = ex2 * ey3 - ey2 * ex3
        cr3 = ex3 * ey0 - ey3 * ex0

        all_pos = int(0)
        if cr0 > 0.0 and cr1 > 0.0 and cr2 > 0.0 and cr3 > 0.0:
            all_pos = 1
        all_neg = int(0)
        if cr0 < 0.0 and cr1 < 0.0 and cr2 < 0.0 and cr3 < 0.0:
            all_neg = 1
        if all_pos == 0 and all_neg == 0:
            continue

        best_cost = cost
        best_c = c

    # ----- 6. Emit outputs from best_c -----
    if best_c < 0:
        no_convex_out[k] = wp.uint8(1)
        n_found_out[k] = wp.uint8(0)
        is_contact_out[k, 0] = wp.uint8(0)
        is_contact_out[k, 1] = wp.uint8(0)
        is_contact_out[k, 2] = wp.uint8(0)
        is_contact_out[k, 3] = wp.uint8(0)
        contact_ik_out[k, 0] = wp.vec3(0.0, 0.0, 0.0)
        contact_ik_out[k, 1] = wp.vec3(0.0, 0.0, 0.0)
        contact_ik_out[k, 2] = wp.vec3(0.0, 0.0, 0.0)
        contact_ik_out[k, 3] = wp.vec3(0.0, 0.0, 0.0)
        contact_rot_out[k, 0] = wp.quat_identity()
        contact_rot_out[k, 1] = wp.quat_identity()
        contact_rot_out[k, 2] = wp.quat_identity()
        contact_rot_out[k, 3] = wp.quat_identity()
        return

    no_convex_out[k] = wp.uint8(0)

    # Per-foot air-floor: each foot's nearest patch z + ground offset.
    af0 = patch_pts[top0.idx[0]][2] + foot_ground_offset
    af1 = patch_pts[top1.idx[0]][2] + foot_ground_offset
    af2 = patch_pts[top2.idx[0]][2] + foot_ground_offset
    af3 = patch_pts[top3.idx[0]][2] + foot_ground_offset

    n_count = int(0)

    # Foot 0
    rb = combo[best_c, 0]
    if fr0.contact[rb] == 1:
        pi = top0.idx[rb]
        pp = patch_pts[pi]
        contact_ik_out[k, 0] = wp.vec3(pp[0], pp[1], pp[2] + foot_ground_offset)
        contact_rot_out[k, 0] = patch_quats[pi]
        is_contact_out[k, 0] = wp.uint8(1)
        n_count += 1
    else:
        contact_ik_out[k, 0] = wp.vec3(wx0, wy0, wp.max(wz0, af0))
        contact_rot_out[k, 0] = wp.quat_identity()
        is_contact_out[k, 0] = wp.uint8(0)
    # Foot 1
    rb = combo[best_c, 1]
    if fr1.contact[rb] == 1:
        pi = top1.idx[rb]
        pp = patch_pts[pi]
        contact_ik_out[k, 1] = wp.vec3(pp[0], pp[1], pp[2] + foot_ground_offset)
        contact_rot_out[k, 1] = patch_quats[pi]
        is_contact_out[k, 1] = wp.uint8(1)
        n_count += 1
    else:
        contact_ik_out[k, 1] = wp.vec3(wx1, wy1, wp.max(wz1, af1))
        contact_rot_out[k, 1] = wp.quat_identity()
        is_contact_out[k, 1] = wp.uint8(0)
    # Foot 2
    rb = combo[best_c, 2]
    if fr2.contact[rb] == 1:
        pi = top2.idx[rb]
        pp = patch_pts[pi]
        contact_ik_out[k, 2] = wp.vec3(pp[0], pp[1], pp[2] + foot_ground_offset)
        contact_rot_out[k, 2] = patch_quats[pi]
        is_contact_out[k, 2] = wp.uint8(1)
        n_count += 1
    else:
        contact_ik_out[k, 2] = wp.vec3(wx2, wy2, wp.max(wz2, af2))
        contact_rot_out[k, 2] = wp.quat_identity()
        is_contact_out[k, 2] = wp.uint8(0)
    # Foot 3
    rb = combo[best_c, 3]
    if fr3.contact[rb] == 1:
        pi = top3.idx[rb]
        pp = patch_pts[pi]
        contact_ik_out[k, 3] = wp.vec3(pp[0], pp[1], pp[2] + foot_ground_offset)
        contact_rot_out[k, 3] = patch_quats[pi]
        is_contact_out[k, 3] = wp.uint8(1)
        n_count += 1
    else:
        contact_ik_out[k, 3] = wp.vec3(wx3, wy3, wp.max(wz3, af3))
        contact_rot_out[k, 3] = wp.quat_identity()
        is_contact_out[k, 3] = wp.uint8(0)

    n_found_out[k] = wp.uint8(n_count)


@wp.kernel
def _fused_sampler_kernel_nc2(
    # ---- scalars / RNG ----
    seed: int,
    K: int,
    n_pts: int,
    n_tpl: int,
    radius: wp.float32,
    effective_radius: wp.float32,
    query_radius: wp.float32,
    outward_pen: wp.float32,
    force_all_snap: int,
    foot_ground_offset: wp.float32,
    n_combos: int,
    # ---- precomputed tables ----
    patch_pts: wp.array(dtype=wp.vec3),
    patch_pts_z0: wp.array(dtype=wp.vec3),
    patch_grid_id: wp.uint64,
    patch_quats: wp.array(dtype=wp.quat),
    patch_admissible: wp.array(dtype=wp.int32),
    num_yaw_bins: int,
    fk_shape_samples: wp.array2d(dtype=wp.vec3),
    cos_n: wp.vec2,
    sin_n: wp.vec2,
    combo: wp.array2d(dtype=wp.int32),
    # ---- outputs (allocated by the launcher) ----
    yaws_out: wp.array(dtype=wp.float32),
    tpl_idx_out: wp.array(dtype=wp.int32),
    is_contact_out: wp.array2d(dtype=wp.uint8),
    contact_ik_out: wp.array2d(dtype=wp.vec3),
    contact_rot_out: wp.array2d(dtype=wp.quat),
    n_found_out: wp.array(dtype=wp.uint8),
    no_convex_out: wp.array(dtype=wp.uint8),
):
    """Two-foot counterpart of :func:`_fused_sampler_kernel`.

    Same pipeline -- sample a placement, project the template, query patches,
    score assignments -- with the stance validity test changed to suit a pair
    of feet. Two contacts enclose no area, so the four-point convex-hull and
    winding checks have nothing to measure; what they exist to prevent is a
    stance whose legs cross the body, and for a pair that is exactly the feet
    swapping sides. The check below is therefore that the placed stance axis
    still points the same way as the template's.

    ``no_convex_out`` keeps the name it has in the quadruped kernel; here it
    flags a slot where no assignment satisfied the contact, distinctness, and
    side-ordering rules.
    """
    k = wp.tid()
    if k >= K:
        return

    # ----- 1. Per-thread RNG -----
    state = wp.rand_init(seed, k)
    center_idx = wp.randi(state, 0, n_pts)
    yaw = wp.randf(state) * TWO_PI
    tpl_idx_val = wp.randi(state, 0, n_tpl)
    yaws_out[k] = yaw
    tpl_idx_out[k] = tpl_idx_val

    center = patch_pts[center_idx]
    cos_y = wp.cos(yaw)
    sin_y = wp.sin(yaw)

    # ----- 2. Un-canonicalize template into world-frame foot positions -----
    tc0 = fk_shape_samples[tpl_idx_val, 0]
    tc1 = fk_shape_samples[tpl_idx_val, 1]

    mx0 = cos_n[0] * tc0[0] - sin_n[0] * tc0[1]
    my0 = sin_n[0] * tc0[0] + cos_n[0] * tc0[1]
    mx1 = cos_n[1] * tc1[0] - sin_n[1] * tc1[1]
    my1 = sin_n[1] * tc1[0] + cos_n[1] * tc1[1]

    wx0 = cos_y * mx0 - sin_y * my0 + center[0]
    wy0 = sin_y * mx0 + cos_y * my0 + center[1]
    wz0 = tc0[2] + center[2]
    wx1 = cos_y * mx1 - sin_y * my1 + center[0]
    wy1 = sin_y * mx1 + cos_y * my1 + center[1]
    wz1 = tc1[2] + center[2]

    tcx = (wx0 + wx1) * 0.5
    tcy = (wy0 + wy1) * 0.5
    tpl_r0 = wp.sqrt((wx0 - tcx) * (wx0 - tcx) + (wy0 - tcy) * (wy0 - tcy))
    tpl_r1 = wp.sqrt((wx1 - tcx) * (wx1 - tcx) + (wy1 - tcy) * (wy1 - tcy))

    # Template stance axis, foot 0 -> foot 1. Comparing against this by dot
    # product avoids the wrap-around that an angle comparison hits when the
    # stance happens to straddle +/-pi.
    tpl_ax = wx1 - wx0
    tpl_ay = wy1 - wy0

    # ----- 3. Top-k hashgrid query per foot -----
    top0 = _topk4_query(patch_grid_id, patch_pts_z0, wx0, wy0, query_radius)
    top1 = _topk4_query(patch_grid_id, patch_pts_z0, wx1, wy1, query_radius)

    # ----- 4. Per-(foot, rank) precompute -----
    fr0 = _foot_ranks(top0, wx0, wy0, tcx, tcy, tpl_r0, patch_pts_z0, radius, effective_radius, outward_pen)
    fr1 = _foot_ranks(top1, wx1, wy1, tcx, tcy, tpl_r1, patch_pts_z0, radius, effective_radius, outward_pen)

    has_opt0 = int(0)
    if top0.dist[0] < radius:
        has_opt0 = 1
    has_opt1 = int(0)
    if top1.dist[0] < radius:
        has_opt1 = 1

    # ----- 5. Assignment loop -----
    best_cost = float(1.0e30)
    best_c = int(-1)

    for c in range(n_combos):
        r0 = combo[c, 0]
        r1 = combo[c, 1]

        contact0 = fr0.contact[r0]
        contact1 = fr1.contact[r1]

        if force_all_snap == 0:
            if has_opt0 == 1 and contact0 == 0:
                continue
            if has_opt1 == 1 and contact1 == 0:
                continue

        # Distinctness: contact feet must pick distinct patches; air feet use
        # a per-foot negative sentinel so they never collide.
        e0 = -1
        if contact0 == 1:
            e0 = top0.idx[r0]
        e1 = -2
        if contact1 == 1:
            e1 = top1.idx[r1]
        if e0 == e1:
            continue

        # Orientation: a patch found flat only along certain headings cannot
        # take a sole pointing elsewhere. The stance yaw sets each foot's
        # heading, so reject the pairing rather than tilt the foot off the
        # surface later.
        if contact0 == 1:
            if _yaw_admits(patch_admissible[e0], yaw, num_yaw_bins) == 0:
                continue
        if contact1 == 1:
            if _yaw_admits(patch_admissible[e1], yaw, num_yaw_bins) == 0:
                continue

        cost = fr0.cost[r0] + fr1.cost[r1]
        if cost >= best_cost:
            continue

        # Side ordering: the placed stance axis must not have flipped, which
        # would put the left foot where the right belongs.
        ax = fr1.target_x[r1] - fr0.target_x[r0]
        ay = fr1.target_y[r1] - fr0.target_y[r0]
        if ax * tpl_ax + ay * tpl_ay <= 0.0:
            continue

        best_cost = cost
        best_c = c

    # ----- 6. Emit outputs from best_c -----
    if best_c < 0:
        no_convex_out[k] = wp.uint8(1)
        n_found_out[k] = wp.uint8(0)
        is_contact_out[k, 0] = wp.uint8(0)
        is_contact_out[k, 1] = wp.uint8(0)
        contact_ik_out[k, 0] = wp.vec3(0.0, 0.0, 0.0)
        contact_ik_out[k, 1] = wp.vec3(0.0, 0.0, 0.0)
        contact_rot_out[k, 0] = wp.quat_identity()
        contact_rot_out[k, 1] = wp.quat_identity()
        return

    no_convex_out[k] = wp.uint8(0)

    af0 = patch_pts[top0.idx[0]][2] + foot_ground_offset
    af1 = patch_pts[top1.idx[0]][2] + foot_ground_offset

    n_count = int(0)

    rb = combo[best_c, 0]
    if fr0.contact[rb] == 1:
        pi = top0.idx[rb]
        pp = patch_pts[pi]
        contact_ik_out[k, 0] = wp.vec3(pp[0], pp[1], pp[2] + foot_ground_offset)
        contact_rot_out[k, 0] = patch_quats[pi]
        is_contact_out[k, 0] = wp.uint8(1)
        n_count += 1
    else:
        contact_ik_out[k, 0] = wp.vec3(wx0, wy0, wp.max(wz0, af0))
        contact_rot_out[k, 0] = wp.quat_identity()
        is_contact_out[k, 0] = wp.uint8(0)

    rb = combo[best_c, 1]
    if fr1.contact[rb] == 1:
        pi = top1.idx[rb]
        pp = patch_pts[pi]
        contact_ik_out[k, 1] = wp.vec3(pp[0], pp[1], pp[2] + foot_ground_offset)
        contact_rot_out[k, 1] = patch_quats[pi]
        is_contact_out[k, 1] = wp.uint8(1)
        n_count += 1
    else:
        contact_ik_out[k, 1] = wp.vec3(wx1, wy1, wp.max(wz1, af1))
        contact_rot_out[k, 1] = wp.quat_identity()
        is_contact_out[k, 1] = wp.uint8(0)

    n_found_out[k] = wp.uint8(n_count)


# --------------------------------------------------------------- launcher


def run_fused_sampler(
    *,
    seed: int,
    K: int,
    patch_pts: torch.Tensor,  # [N, 3] float -- real xyz
    patch_quats: torch.Tensor,  # [N, 4] float xyzw -- surface pose at each patch
    patch_admissible: torch.Tensor,  # [N] int32 -- bitmask of headings the footprint fits
    num_yaw_bins: int,
    fk_shape_samples: torch.Tensor,  # [n_tpl, NC, 3] float
    nominal_angles: torch.Tensor,  # [NC] float
    radius: float,
    query_radius: float,
    outward_pen: float,
    force_all_snap: bool,
    foot_ground_offset: float,
    block_dim: int = 128,
) -> dict[str, torch.Tensor]:
    """Single-launch end-to-end sampler.

    Allocates the K-sized output tensors (smallest dtype that fits each
    value), builds the morph-patch hashgrid, and runs the fused kernel
    once over ``K`` candidate slots. Returns a dict of K-sized outputs.

    Dispatches on foot count: four feet run the quadruped kernel, two run
    :func:`_fused_sampler_kernel_nc2`. CUDA tensors required.
    """
    nc = int(fk_shape_samples.shape[1])
    if nc not in (2, 4):
        raise NotImplementedError(f"Fused sampler kernel supports nc=2 and nc=4; got nc={nc}.")

    device = patch_pts.device
    if not patch_pts.is_cuda:
        raise ValueError("Fused sampler requires CUDA tensors.")

    # Patch hashgrid (z=0 projection so radius is xy-only).
    patch_xy = patch_pts[:, :2].contiguous()
    patch_grid = build_spatial_grid_xy(patch_xy, radius=query_radius)

    import itertools

    if nc == 4:
        # 24 permutations (the all-distinct subset; non-distinct combos all
        # get rejected by the kernel's distinctness check).
        combo = torch.tensor(list(itertools.permutations(range(nc))), device=device, dtype=torch.int32)
    else:
        # With only two feet, permutations of the rank slots would leave just
        # two assignments to choose from. Every rank pair is cheap enough to
        # score outright, which gives the matching real freedom -- the
        # distinctness check still rules out both feet taking one patch.
        combo = torch.tensor(list(itertools.product(range(4), repeat=nc)), device=device, dtype=torch.int32)

    # Per-foot nominal-angle constants packed into a warp vector.
    cos_n_t = torch.cos(nominal_angles).to(torch.float32).cpu().tolist()
    sin_n_t = torch.sin(nominal_angles).to(torch.float32).cpu().tolist()
    if nc == 4:
        cos_n_v = wp.vec4(cos_n_t[0], cos_n_t[1], cos_n_t[2], cos_n_t[3])
        sin_n_v = wp.vec4(sin_n_t[0], sin_n_t[1], sin_n_t[2], sin_n_t[3])
    else:
        cos_n_v = wp.vec2(cos_n_t[0], cos_n_t[1])
        sin_n_v = wp.vec2(sin_n_t[0], sin_n_t[1])

    # K-sized outputs in their tightest dtypes:
    # * yaws: float32 (precision matters for the IK seed).
    # * tpl_idx: int32 (n_tpl easily fits in int32; rsl_rl downstream
    #   wants int64 so we convert at the end).
    # * is_contact / n_found / no_convex: uint8 -- bool / [0..4] /
    #   bool respectively. Saves ~1 GiB at K=60M vs the int32 we
    #   used to allocate.
    yaws = torch.empty((K,), dtype=torch.float32, device=device)
    tpl_idx = torch.empty((K,), dtype=torch.int32, device=device)
    is_contact = torch.empty((K, nc), dtype=torch.uint8, device=device)
    contact_ik = torch.empty((K, nc, 3), dtype=torch.float32, device=device)
    contact_rot = torch.empty((K, nc, 4), dtype=torch.float32, device=device)
    n_found = torch.empty((K,), dtype=torch.uint8, device=device)
    no_convex = torch.empty((K,), dtype=torch.uint8, device=device)

    scalar_inputs = [
        int(seed),
        int(K),
        int(patch_pts.shape[0]),
        int(fk_shape_samples.shape[0]),
        float(radius),
        float(query_radius if force_all_snap else radius),
        float(query_radius),
        float(outward_pen),
        int(1 if force_all_snap else 0),
        float(foot_ground_offset),
    ]
    # The two-foot kernel scores every rank pair, so its combo count varies
    # with the table rather than being a compile-time constant.
    if nc == 2:
        scalar_inputs.append(int(combo.shape[0]))

    wp.launch(
        _fused_sampler_kernel if nc == 4 else _fused_sampler_kernel_nc2,
        dim=K,
        block_dim=int(block_dim),
        inputs=[
            *scalar_inputs,
            wp.from_torch(patch_pts.contiguous(), dtype=wp.vec3),
            patch_grid._pts_wp,  # the z=0 grid points (shared with the grid)
            patch_grid.grid.id,
            wp.from_torch(patch_quats.contiguous(), dtype=wp.quat),
            wp.from_torch(patch_admissible.contiguous(), dtype=wp.int32),
            int(num_yaw_bins),
            wp.from_torch(fk_shape_samples.contiguous(), dtype=wp.vec3),
            cos_n_v,
            sin_n_v,
            wp.from_torch(combo, dtype=wp.int32),
        ],
        outputs=[
            wp.from_torch(yaws, dtype=wp.float32),
            wp.from_torch(tpl_idx, dtype=wp.int32),
            wp.from_torch(is_contact, dtype=wp.uint8),
            wp.from_torch(contact_ik, dtype=wp.vec3),
            wp.from_torch(contact_rot, dtype=wp.quat),
            wp.from_torch(n_found, dtype=wp.uint8),
            wp.from_torch(no_convex, dtype=wp.uint8),
        ],
        device=str(device),
    )

    # Tighter dtypes inside the kernel; expand back to what the caller
    # expects (rsl_rl downstream uses int64 / bool tensors).
    return {
        "yaws": yaws,
        "tpl_idx": tpl_idx.to(torch.int64),
        "is_contact_full": is_contact.bool(),
        "contact_ik": contact_ik,
        "contact_rot": contact_rot,
        "n_found": n_found.to(torch.int64),
        "no_convex": no_convex.bool(),
    }


__all__ = ["run_fused_sampler"]
