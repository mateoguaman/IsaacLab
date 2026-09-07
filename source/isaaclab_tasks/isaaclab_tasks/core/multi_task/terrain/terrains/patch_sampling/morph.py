# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Morphological (deterministic, GPU-batched) patch sampling.

Rasterizes a mesh into a 2D heightmap, runs a max-min morphological
filter with the robot footprint as kernel, then samples ``num_patches``
cells from the valid region. Replaces the earlier rejection-sampling
path for large, dense terrains.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager

import numpy as np
import torch
import warp as wp

from ....utils.grid_downsample import grid_bucket_downsample
from . import cfg as patch_cfg
from .kernels import morph_validity_kernel, rasterize_grid_kernel

MORPH_YAW_BINS = 8
"""Headings tested per cell for a rectangular footprint, spanning a half turn.

A rectangle is symmetric under 180 degrees, so ``[0, pi)`` covers every
distinct orientation. Consumers matching a foot's heading against a patch must
use the same count to read the admissibility bits.
"""

MORPH_TIMINGS: dict[str, float] = {}
"""Cumulative wall-time per sub-phase of :func:`find_flat_patches_morphological`.

Populated only while :func:`_morph_time` is active. Cleared at the start of
each call to :func:`find_flat_patches_morphological`. Callers can read this
dict after invocation to report a breakdown (see RetargetPipeline).
"""


@contextmanager
def _morph_time(name: str, device):
    """Record wall time for a morphological-sampling sub-phase with CUDA sync.

    ``device`` may be a :class:`torch.device` or a device-string such as
    ``"cuda:0"`` (what :func:`warp.device_to_torch` returns on Warp meshes).
    """
    dev_str = device.type if isinstance(device, torch.device) else str(device)
    is_cuda = dev_str.startswith("cuda")
    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if is_cuda:
            torch.cuda.synchronize()
        MORPH_TIMINGS[name] = MORPH_TIMINGS.get(name, 0.0) + (time.perf_counter() - t0)


def _resolve_footprint(cfg):
    """Ensure *cfg* is a footprint configclass, not a plain dict.

    The monkey-patching in ``terrain_cfg.py`` round-trips configs through
    ``to_dict()`` / ``cfg_class(**dict)``, which turns nested configclasses
    into plain dicts.  This helper reconstitutes the correct type.
    """
    if isinstance(cfg, (patch_cfg.CircleFootprintCfg, patch_cfg.RectFootprintCfg)):
        return cfg
    if isinstance(cfg, dict):
        if "length" in cfg and "width" in cfg:
            return patch_cfg.RectFootprintCfg(length=cfg["length"], width=cfg["width"])
        return patch_cfg.CircleFootprintCfg(radius=cfg["radius"])
    raise TypeError(f"Unknown footprint type: {type(cfg)}")


def _build_footprint_mask(
    cfg: patch_cfg.CircleFootprintCfg | patch_cfg.RectFootprintCfg, scale: float, device: torch.device
) -> torch.Tensor:
    """Convert a footprint config into a 2D boolean kernel mask.

    Args:
        cfg: Footprint configuration (or a dict that will be auto-resolved).
        scale: Grid cell size [m] (horizontal_scale).
        device: Torch device for the output tensor.

    Returns:
        Boolean tensor of shape ``[K, K]`` where K is always odd.
    """
    cfg = _resolve_footprint(cfg)

    if isinstance(cfg, patch_cfg.CircleFootprintCfg):
        r_cells = math.ceil(cfg.radius / scale)
        k = 2 * r_cells + 1
        y, x = torch.meshgrid(
            torch.arange(k, device=device) - r_cells,
            torch.arange(k, device=device) - r_cells,
            indexing="ij",
        )
        mask = (x.float() * scale) ** 2 + (y.float() * scale) ** 2 <= cfg.radius**2
    elif isinstance(cfg, patch_cfg.RectFootprintCfg):
        hl = math.ceil(cfg.length / (2.0 * scale))  # half-length along +x (forward)
        hw = math.ceil(cfg.width / (2.0 * scale))  # half-width along +y (lateral)
        k = 2 * max(hl, hw) + 1
        y, x = torch.meshgrid(
            torch.arange(k, device=device) - k // 2,
            torch.arange(k, device=device) - k // 2,
            indexing="ij",
        )
        mask = (x.float().abs() * scale <= cfg.length / 2.0) & (y.float().abs() * scale <= cfg.width / 2.0)
    else:
        raise TypeError(f"Unknown footprint type: {type(cfg)}")

    if not mask.any():
        mask[k // 2, k // 2] = True
    return mask


def _build_rotated_rect_masks(
    footprint, scale: float, yaw_angles: torch.Tensor, device: torch.device
) -> list[torch.Tensor]:
    """Build rotated rectangular footprint masks for each yaw angle.

    Returns a list of ``[K, K]`` boolean masks, one per yaw.
    """
    hl = footprint.length / 2.0  # half-length along +x (forward)
    hw = footprint.width / 2.0  # half-width along +y (lateral)
    r_max = math.sqrt(hl**2 + hw**2)
    r_cells = math.ceil(r_max / scale)
    k = 2 * r_cells + 1
    y, x = torch.meshgrid(
        torch.arange(k, device=device, dtype=torch.float32) - r_cells,
        torch.arange(k, device=device, dtype=torch.float32) - r_cells,
        indexing="ij",
    )
    wx = x * scale
    wy = y * scale

    masks = []
    for yaw in yaw_angles:
        c, s = float(yaw.cos()), float(yaw.sin())
        lx = wx * c + wy * s  # local x (forward)
        ly = -wx * s + wy * c  # local y (lateral)
        masks.append((lx.abs() <= hl) & (ly.abs() <= hw))
    return masks


def _yaw_to_quat_xyzw(yaw: torch.Tensor) -> torch.Tensor:
    """Convert yaw angles [rad] to quaternions in ``(x, y, z, w)`` convention.

    Args:
        yaw: Tensor of yaw angles, any shape.

    Returns:
        Quaternion tensor with shape ``(*yaw.shape, 4)``.
    """
    half = yaw * 0.5
    zeros = torch.zeros_like(half)
    return torch.stack([zeros, zeros, half.sin(), half.cos()], dim=-1)


def _fit_surface_planes_at(
    heightmap: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Least-squares plane fit of the surface under ``mask`` at chosen cells.

    A footprint resting on sloped or uneven ground must be tilted to match it,
    so each cell needs the plane its footprint actually spans rather than the
    single height sampled at its centre. Footprint masks are symmetric about
    their centre, which zeroes the cross terms of the normal equations and
    leaves each plane coefficient as a weighted sum over the window.

    Evaluated only at the requested cells: a terrain heightmap runs to hundreds
    of millions of cells, while the cells that become patches number in the
    thousands.

    Args:
        heightmap: ``[H, W]`` surface heights [m]; non-finite entries are
            treated as the window's own mean, which only arises at cells the
            validity filter already rejects.
        rows: ``[N]`` row indices of the cells to fit.
        cols: ``[N]`` column indices of the cells to fit.
        mask: ``[K, K]`` boolean footprint kernel.
        scale: Grid spacing [m].

    Returns:
        Surface normals ``[N, 3]`` (unit, +z up) and the plane's fitted height
        at each requested cell centre ``[N]`` [m].
    """
    device = heightmap.device
    weight = mask.to(torch.float32)
    k = weight.shape[0]
    radius = (k - 1) // 2
    offsets = (torch.arange(k, device=device, dtype=torch.float32) - radius) * scale
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")

    count = weight.sum().clamp_min(1.0)
    second_x = (weight * dx * dx).sum().clamp_min(1.0e-12)
    second_y = (weight * dy * dy).sum().clamp_min(1.0e-12)
    weight_x = weight * dx
    weight_y = weight * dy
    # Index arithmetic clamped to the map rather than a padded copy: a
    # production heightmap runs to hundreds of millions of cells, so padding it
    # to gather a few thousand windows costs more than the windows do. Clamping
    # matches replicate padding, and the validity filter has already excluded
    # cells whose footprint would leave the map.
    span = torch.arange(k, device=device) - radius
    height, width = heightmap.shape

    fitted_height = torch.empty(rows.shape[0], device=device)
    slope_x = torch.empty(rows.shape[0], device=device)
    slope_y = torch.empty(rows.shape[0], device=device)
    # Bound the gather so peak memory follows the chunk, not the patch count.
    chunk = max(1, 2_000_000 // max(1, k * k))
    for start in range(0, rows.shape[0], chunk):
        stop = min(start + chunk, rows.shape[0])
        row_idx = (rows[start:stop].view(-1, 1, 1) + span.view(1, -1, 1)).clamp_(0, height - 1)
        col_idx = (cols[start:stop].view(-1, 1, 1) + span.view(1, 1, -1)).clamp_(0, width - 1)
        windows = torch.nan_to_num(heightmap[row_idx, col_idx], nan=0.0, posinf=0.0, neginf=0.0)
        fitted_height[start:stop] = (windows * weight).sum(dim=(-2, -1)) / count
        slope_x[start:stop] = (windows * weight_x).sum(dim=(-2, -1)) / second_x
        slope_y[start:stop] = (windows * weight_y).sum(dim=(-2, -1)) / second_y

    normal = torch.stack([-slope_x, -slope_y, torch.ones_like(slope_x)], dim=-1)
    return torch.nn.functional.normalize(normal, dim=-1), fitted_height


def _surface_frame_to_quat_xyzw(normal: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Orientation whose +z follows ``normal`` and whose +x follows ``yaw``.

    The requested heading is projected onto the surface plane, so the returned
    frame both lies on the surface and points the footprint's long axis the way
    the flatness search found it fits.

    Args:
        normal: ``[..., 3]`` unit surface normals.
        yaw: ``[...]`` headings about world +z [rad].

    Returns:
        ``[..., 4]`` quaternions in ``(x, y, z, w)`` convention.
    """
    heading = torch.stack([yaw.cos(), yaw.sin(), torch.zeros_like(yaw)], dim=-1)
    forward = heading - normal * (heading * normal).sum(dim=-1, keepdim=True)
    # A heading parallel to the normal leaves nothing to project; any in-plane
    # axis is then equally valid, so fall back to world +y crossed with it.
    degenerate = forward.norm(dim=-1, keepdim=True) < 1.0e-6
    fallback = torch.cross(normal, torch.tensor([0.0, 1.0, 0.0], device=normal.device).expand_as(normal), dim=-1)
    forward = torch.where(degenerate, fallback, forward)
    x_axis = torch.nn.functional.normalize(forward, dim=-1)
    y_axis = torch.cross(normal, x_axis, dim=-1)

    # Shepperd's method: build from whichever diagonal term is largest so the
    # square root never divides by something near zero.
    m = torch.stack([x_axis, y_axis, normal], dim=-2).transpose(-1, -2)
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    quat = torch.zeros((*normal.shape[:-1], 4), device=normal.device, dtype=normal.dtype)

    w_big = trace > 0.0
    s = torch.sqrt((trace + 1.0).clamp_min(1.0e-12)) * 2.0
    quat[..., 3] = torch.where(w_big, 0.25 * s, quat[..., 3])
    quat[..., 0] = torch.where(w_big, (m[..., 2, 1] - m[..., 1, 2]) / s, quat[..., 0])
    quat[..., 1] = torch.where(w_big, (m[..., 0, 2] - m[..., 2, 0]) / s, quat[..., 1])
    quat[..., 2] = torch.where(w_big, (m[..., 1, 0] - m[..., 0, 1]) / s, quat[..., 2])

    # Surface normals stay within a hemisphere of +z here, so the trace is
    # positive except for degenerate fits; identity is the right answer there.
    quat[..., 3] = torch.where(w_big, quat[..., 3], torch.ones_like(quat[..., 3]))
    return torch.nn.functional.normalize(quat, dim=-1)


def _rasterize_mesh(
    wp_mesh: wp.Mesh, x_range: tuple[float, float], y_range: tuple[float, float], scale: float, device: torch.device
) -> tuple[torch.Tensor, float, float]:
    """Rasterize a warp mesh to a 2D heightmap via one grid-shaped Warp launch.

    Args:
        wp_mesh: The warp mesh.
        x_range: World-space x bounds ``(min, max)`` [m].
        y_range: World-space y bounds ``(min, max)`` [m].
        scale: Grid cell size [m].
        device: Torch device.

    Returns:
        Tuple of ``(heightmap, x_min, y_min)`` where heightmap is ``[H, W]``
        with ``inf`` at missed cells, x_min/y_min are the world-space
        coordinates of cell ``[0, 0]``.
    """
    nx = max(int((x_range[1] - x_range[0]) / scale), 1)
    ny = max(int((y_range[1] - y_range[0]) / scale), 1)

    heightmap = torch.full((nx, ny), float("inf"), dtype=torch.float32, device=device)

    wp.launch(
        rasterize_grid_kernel,
        dim=(nx, ny),
        inputs=[
            wp_mesh.id,
            float(x_range[0]),
            float(y_range[0]),
            float(scale),
            100.0,
            1.0e6,
        ],
        outputs=[wp.from_torch(heightmap, dtype=wp.float32)],
        device=wp_mesh.device,
    )
    return heightmap, x_range[0], y_range[0]


def _morphological_validity(
    heightmap: torch.Tensor, mask: torch.Tensor, max_height_diff: float, z_range: tuple[float, float]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Max-min-over-footprint validity plus the height range it measured.

    Args:
        heightmap: ``[H, W]`` heightmap tensor (``inf`` for missed rays).
        mask: ``[K, K]`` boolean footprint kernel.
        max_height_diff: Maximum allowed height range within the footprint [m].
        z_range: Allowed absolute height band [m].

    Returns:
        ``(valid, height_range)`` per cell.
    """
    device = heightmap.device
    H, W = heightmap.shape
    pad = mask.shape[0] // 2
    hm_c = heightmap.contiguous()
    mask_u8 = mask.to(torch.uint8).contiguous()

    out_valid = torch.empty((H, W), dtype=torch.uint8, device=device)
    out_h_range = torch.empty((H, W), dtype=torch.float32, device=device)

    wp.launch(
        morph_validity_kernel,
        dim=(H, W),
        inputs=[
            wp.from_torch(hm_c, dtype=wp.float32),
            wp.from_torch(mask_u8, dtype=wp.uint8),
            float(max_height_diff),
            float(z_range[0]),
            float(z_range[1]),
            int(pad),
        ],
        outputs=[
            wp.from_torch(out_valid, dtype=wp.uint8),
            wp.from_torch(out_h_range, dtype=wp.float32),
        ],
        device=str(device),
    )
    return out_valid.to(torch.bool), out_h_range


def find_flat_patches_morphological(
    wp_mesh: wp.Mesh,
    origin: np.ndarray | torch.Tensor | tuple[float, float, float],
    cfg: patch_cfg.MorphologicalPatchSamplingCfg,
) -> torch.Tensor:
    """Find flat patches using deterministic morphological heightmap filtering.

    Instead of rejection sampling, this function:

    1. Rasterizes the mesh to a 2D heightmap with one batched ray-cast.
    2. Computes a validity mask via morphological max-min filtering using the
       configured robot footprint kernel.
    3. For rectangular footprints, tests multiple yaw angles and records both
       which yaw produces the smallest height range at each cell and the full
       set of yaws that fit there.
    4. Least-squares fits the surface plane under the footprint, so each patch
       carries the orientation something resting on it must adopt.
    5. Samples ``num_patches`` from the valid region, optionally with
       farthest-point refinement for spatial coverage.

    Args:
        wp_mesh: The warp mesh to find patches on.
        origin: Sub-terrain origin in the mesh frame.
        cfg: Morphological sampling configuration.

    Returns:
        Tensor of shape ``(num_patches, 8)`` --
        ``[x, y, z, qx, qy, qz, qw, admissible_yaw_bits]`` in the mesh frame
        with origin subtracted. The quaternion is absolute: its ``+z`` is the
        fitted surface normal and its ``+x`` the best-fitting heading. The last
        column packs one bit per tested yaw (all bits set for a disc, which
        fits every heading).
    """
    MORPH_TIMINGS.clear()
    device = wp.device_to_torch(wp_mesh.device)
    footprint = _resolve_footprint(cfg.footprint)

    with _morph_time("setup", device):
        if isinstance(origin, np.ndarray):
            origin_t = torch.from_numpy(origin).float().to(device)
        elif isinstance(origin, torch.Tensor):
            origin_t = origin.float().to(device)
        else:
            origin_t = torch.tensor(origin, dtype=torch.float32, device=device)

        # Compute mesh XY bounds on GPU once, then pull four scalars -- avoids
        # the per-invocation ``wp_mesh.points.numpy()`` transfer of the whole
        # vertex buffer.
        verts_xy = wp.to_torch(wp_mesh.points)[:, :2]
        bounds_min = verts_xy.amin(dim=0)
        bounds_max = verts_xy.amax(dim=0)
        mesh_xmin = float(bounds_min[0])
        mesh_xmax = float(bounds_max[0])
        mesh_ymin = float(bounds_min[1])
        mesh_ymax = float(bounds_max[1])

        ox, oy, oz = origin_t[0].item(), origin_t[1].item(), origin_t[2].item()
        x_range = (max(cfg.x_range[0] + ox, mesh_xmin), min(cfg.x_range[1] + ox, mesh_xmax))
        y_range = (max(cfg.y_range[0] + oy, mesh_ymin), min(cfg.y_range[1] + oy, mesh_ymax))
        z_range = (cfg.z_range[0] + oz, cfg.z_range[1] + oz)

        scale = cfg.horizontal_scale

    with _morph_time("rasterize", device):
        heightmap, hm_x0, hm_y0 = _rasterize_mesh(wp_mesh, x_range, y_range, scale, device)
        H, W = heightmap.shape

    is_rect = isinstance(footprint, patch_cfg.RectFootprintCfg)

    with _morph_time("validity", device):
        if is_rect:
            # test 8 discrete yaw angles in [0, pi) — rectangle has 180-deg symmetry
            num_yaw = MORPH_YAW_BINS
            yaw_angles = torch.linspace(0, math.pi, num_yaw + 1, device=device)[:num_yaw]
            rotated_masks = _build_rotated_rect_masks(footprint, scale, yaw_angles, device)

            best_range = torch.full((H, W), float("inf"), device=device)
            best_yaw_idx = torch.zeros((H, W), dtype=torch.long, device=device)
            combined_valid = torch.zeros((H, W), dtype=torch.bool, device=device)
            # One bit per tested yaw: a rectangular footprint fits some
            # headings and not others, and downstream matching needs the whole
            # admissible set rather than only the single best heading.
            admissible_yaw = torch.zeros((H, W), dtype=torch.int32, device=device)

            for yi, mask in enumerate(rotated_masks):
                valid_yi, h_range = _morphological_validity(heightmap, mask, cfg.max_height_diff, z_range)
                admissible_yaw |= valid_yi.to(torch.int32) << yi
                improved = valid_yi & (h_range < best_range)
                best_range[improved] = h_range[improved]
                best_yaw_idx[improved] = yi
                combined_valid |= valid_yi

            valid = combined_valid
            yaw_map = yaw_angles[best_yaw_idx]
            fit_masks = rotated_masks
        else:
            footprint_mask = _build_footprint_mask(footprint, scale, device)
            valid, _ = _morphological_validity(heightmap, footprint_mask, cfg.max_height_diff, z_range)
            yaw_map = torch.zeros((H, W), device=device)
            # A disc fits every heading, so every tested yaw is admissible.
            admissible_yaw = torch.full((H, W), -1, dtype=torch.int32, device=device)
            best_yaw_idx = torch.zeros((H, W), dtype=torch.long, device=device)
            fit_masks = [footprint_mask]

        valid_coords = valid.nonzero(as_tuple=False)  # [K, 2]
        num_valid = valid_coords.shape[0]

    if num_valid < cfg.num_patches:
        total_cells = H * W
        valid_frac = num_valid / total_cells if total_cells > 0 else 0.0
        raise RuntimeError(
            f"Morphological patch sampling found only {num_valid} valid cells but"
            f" {cfg.num_patches} patches requested."
            f"\n\tGrid size: {H}x{W} ({total_cells} cells)"
            f"\n\tValid fraction: {valid_frac:.4f}"
            f"\n\tmax_height_diff: {cfg.max_height_diff}"
            f"\n\tfootprint: {footprint}"
            f"\n\tx_range: {x_range}, y_range: {y_range}, z_range: {z_range}"
            f"\n\tHint: lower horizontal_scale or relax max_height_diff to grow num_valid."
        )

    with _morph_time("candidates", device):
        n_candidates = min(int(cfg.num_patches * cfg.oversample_ratio), num_valid)
        if n_candidates * 8 < num_valid:
            # A production heightmap has valid cells in the hundreds of
            # millions while candidates number in the thousands, so permuting
            # the whole set to take a prefix costs gigabytes for a result that
            # is thrown away. Draw the indices directly instead; the repeats
            # this admits are harmless because the FPS thinning downstream
            # selects on position and collapses them.
            picks = torch.randint(0, num_valid, (n_candidates,), device=device)
        else:
            picks = torch.randperm(num_valid, device=device)[:n_candidates]
        candidates_rc = valid_coords[picks]

        rows, cols = candidates_rc[:, 0], candidates_rc[:, 1]
        cand_x = hm_x0 + (rows.float() + 0.5) * scale
        cand_y = hm_y0 + (cols.float() + 0.5) * scale
        # The fitted plane's height at the cell centre, rather than the raw
        # sample, is what a footprint spanning the cell actually rests on.
        cand_yaw = yaw_map[rows, cols]
        cand_yaw_idx = best_yaw_idx[rows, cols]
        cand_admissible = admissible_yaw[rows, cols].to(torch.float32)
        cand_xy = torch.stack([cand_x, cand_y], dim=-1)

    with _morph_time("fps", device):
        if cfg.oversample_ratio > 1.0 and n_candidates > cfg.num_patches:
            sel_idx = grid_bucket_downsample(cand_xy, cfg.num_patches)
        else:
            sel_idx = torch.arange(min(cfg.num_patches, n_candidates), device=device)
        sel_rows, sel_cols = rows[sel_idx], cols[sel_idx]
        sel_xy = cand_xy[sel_idx]
        sel_yaw = cand_yaw[sel_idx]
        sel_yaw_idx = cand_yaw_idx[sel_idx]
        admissible = cand_admissible[sel_idx]

    with _morph_time("plane_fit", device):
        # Fit the surface under each kept patch using the footprint orientation
        # that was found to fit there, so the pose reflects the ground the
        # footprint actually spans.
        fitted_z = torch.zeros(sel_rows.shape[0], device=device)
        normal = torch.zeros((sel_rows.shape[0], 3), device=device)
        normal[:, 2] = 1.0
        for yi, mask in enumerate(fit_masks):
            group = (sel_yaw_idx == yi).nonzero(as_tuple=False).squeeze(-1)
            if group.numel() == 0:
                continue
            normal_g, fitted_g = _fit_surface_planes_at(heightmap, sel_rows[group], sel_cols[group], mask, scale)
            normal[group] = normal_g
            fitted_z[group] = fitted_g

        quat = _surface_frame_to_quat_xyzw(normal, sel_yaw)
        pos = torch.cat([sel_xy, fitted_z.unsqueeze(-1)], dim=-1)
        result = torch.cat([pos - origin_t, quat, admissible.unsqueeze(-1)], dim=-1)
    return result
