# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU rasterizer for the ASCII visualizer, written in NVIDIA Warp.

Single fused kernel does instance-expand + setup + raster inline, with
per-triangle state kept in registers. The only per-triangle global-memory
array is ``lit_colors_buf`` (shaded color, read by the resolve pass).

Pipeline per ``draw_batch`` call:

1. *fused_expand_raster_kernel* (dim=I*T) — per output triangle: unpack
   (inst, tri), look up mesh verts, apply scale → rotate → translate,
   compute flat Lambertian shading, MVP → NDC → screen, near-plane cull,
   winding fix, bbox clip, and pixel-bbox raster with packed (depth, tri_id)
   uint64 atomic_min into the depth buffer.
2. *resolve_kernel* (dim=H*W) — per pixel, read winning tri_id from the
   depth buffer and write shaded color (or background) to RGBA8.

Then ``rgba()`` runs resolve once and returns ``rgba.numpy()`` — a single
GPU→CPU copy (~4 MB at 1264x784) per frame.

The fused design keeps scratch memory proportional to the triangle count
only via ``lit_colors_buf`` (one vec3 per triangle). Earlier separate
kernels needed 6 other scratch arrays (world_tris, screen, bbox,
inv_area, valid, base_colors) totalling ~100 bytes/tri; those are gone.
"""

from __future__ import annotations

import logging

import numpy as np
import warp as wp

logger = logging.getLogger(__name__)


# --- Sentinel value for an unpainted pixel. uint64 MAX beats any packed
# (depth, tri_id) so the resolve pass can detect "no hit".
_DEPTH_SENTINEL = 0xFFFFFFFFFFFFFFFF


# --------------------------------------------------------------------- kernels


@wp.kernel
def _clear_depth_kernel(depth_buf: wp.array(dtype=wp.uint64), sentinel: wp.uint64):
    pid = wp.tid()
    depth_buf[pid] = sentinel


@wp.kernel
def _fused_expand_raster_kernel(
    mesh_verts: wp.array(dtype=wp.vec3),          # (V,)
    mesh_idx: wp.array(dtype=wp.int32, ndim=2),   # (T, 3)
    xforms: wp.array(dtype=wp.transform),         # (I,) pos + quat xyzw, Newton-native
    scales: wp.array(dtype=wp.vec3),              # (I,) instance scale
    inst_colors: wp.array(dtype=wp.vec3),         # (I,) RGB in [0,1]
    n_tris_per_inst: int,
    mvp: wp.mat44,
    light_dir_neg: wp.vec3,
    ambient: float,
    width: int,
    height: int,
    # outputs:
    depth_buf: wp.array(dtype=wp.uint64),         # (H*W,) packed (depth, tri_id)
    lit_colors_buf: wp.array(dtype=wp.vec3),      # (I*T,) RGB in [0,255]
):
    tid = wp.tid()
    inst = tid / n_tris_per_inst
    tri = tid - inst * n_tris_per_inst

    # --- instance expand: mesh-local → world ---
    xf = xforms[inst]
    p = wp.transform_get_translation(xf)
    q = wp.transform_get_rotation(xf)
    s = scales[inst]

    i0 = mesh_idx[tri, 0]
    i1 = mesh_idx[tri, 1]
    i2 = mesh_idx[tri, 2]

    ml0 = mesh_verts[i0]
    ml1 = mesh_verts[i1]
    ml2 = mesh_verts[i2]

    ms0 = wp.vec3(ml0[0] * s[0], ml0[1] * s[1], ml0[2] * s[2])
    ms1 = wp.vec3(ml1[0] * s[0], ml1[1] * s[1], ml1[2] * s[2])
    ms2 = wp.vec3(ml2[0] * s[0], ml2[1] * s[1], ml2[2] * s[2])

    v0 = wp.quat_rotate(q, ms0) + p
    v1 = wp.quat_rotate(q, ms1) + p
    v2 = wp.quat_rotate(q, ms2) + p

    # --- flat Lambertian + ambient shading, baked per triangle ---
    e1 = v1 - v0
    e2 = v2 - v0
    n = wp.cross(e1, e2)
    n_len = wp.length(n)
    if n_len > 1.0e-9:
        n = n / n_len
    else:
        n = wp.vec3(0.0, 0.0, 1.0)
    lambert = wp.max(wp.dot(n, light_dir_neg), 0.0)
    shade = ambient + (1.0 - ambient) * lambert
    base = inst_colors[inst]
    lit_colors_buf[tid] = wp.vec3(
        base[0] * shade * 255.0,
        base[1] * shade * 255.0,
        base[2] * shade * 255.0,
    )

    # --- MVP: world → clip ---
    c0 = mvp @ wp.vec4(v0[0], v0[1], v0[2], 1.0)
    c1 = mvp @ wp.vec4(v1[0], v1[1], v1[2], 1.0)
    c2 = mvp @ wp.vec4(v2[0], v2[1], v2[2], 1.0)

    # --- near-plane cull (any vertex behind near plane kills the tri) ---
    if c0[3] < 1.0e-4 or c1[3] < 1.0e-4 or c2[3] < 1.0e-4:
        return

    # --- perspective divide → NDC ---
    iw0 = 1.0 / c0[3]
    iw1 = 1.0 / c1[3]
    iw2 = 1.0 / c2[3]
    nz0 = c0[2] * iw0
    nz1 = c1[2] * iw1
    nz2 = c2[2] * iw2

    # --- NDC → screen (integer pixel coords) ---
    w_f = float(width - 1)
    h_f = float(height - 1)
    sx0 = (c0[0] * iw0 + 1.0) * 0.5 * w_f
    sy0 = (1.0 - c0[1] * iw0) * 0.5 * h_f
    sx1 = (c1[0] * iw1 + 1.0) * 0.5 * w_f
    sy1 = (1.0 - c1[1] * iw1) * 0.5 * h_f
    sx2 = (c2[0] * iw2 + 1.0) * 0.5 * w_f
    sy2 = (1.0 - c2[1] * iw2) * 0.5 * h_f

    # --- signed area; degenerate triangles skipped, winding handled via
    # signed barycentrics (no branch divergence from flipping verts) ---
    area = (sx1 - sx0) * (sy2 - sy0) - (sy1 - sy0) * (sx2 - sx0)
    if wp.abs(area) < 1.0e-6:
        return
    inv_a = 1.0 / area  # signed

    # --- screen-space bbox, clamped to viewport ---
    min_x = wp.min(wp.min(sx0, sx1), sx2)
    max_x = wp.max(wp.max(sx0, sx1), sx2)
    min_y = wp.min(wp.min(sy0, sy1), sy2)
    max_y = wp.max(wp.max(sy0, sy1), sy2)
    x0 = int(wp.max(wp.floor(min_x), 0.0))
    x1 = int(wp.min(wp.ceil(max_x), float(width - 1)))
    y0 = int(wp.max(wp.floor(min_y), 0.0))
    y1 = int(wp.min(wp.ceil(max_y), float(height - 1)))
    if x1 < x0 or y1 < y0:
        return

    # --- raster: walk bbox; `b = w * inv_a` folds winding — barycentrics
    # are positive inside the triangle regardless of CW/CCW vertex order. ---
    for py in range(y0, y1 + 1):
        fy = float(py)
        for px in range(x0, x1 + 1):
            fx = float(px)
            w0 = (sx1 - fx) * (sy2 - fy) - (sy1 - fy) * (sx2 - fx)
            w1 = (sx2 - fx) * (sy0 - fy) - (sy2 - fy) * (sx0 - fx)
            w2 = (sx0 - fx) * (sy1 - fy) - (sy0 - fy) * (sx1 - fx)
            b0 = w0 * inv_a
            b1 = w1 * inv_a
            b2 = w2 * inv_a
            if b0 < 0.0 or b1 < 0.0 or b2 < 0.0:
                continue
            depth = b0 * nz0 + b1 * nz1 + b2 * nz2
            depth01 = wp.clamp((depth + 1.0) * 0.5, 0.0, 1.0)
            d_u32 = wp.uint32(depth01 * 4294967295.0)
            packed = (wp.uint64(d_u32) << wp.uint64(32)) | wp.uint64(tid)
            pix = py * width + px
            wp.atomic_min(depth_buf, pix, packed)


@wp.kernel
def _resolve_kernel(
    depth_buf: wp.array(dtype=wp.uint64),
    lit_colors: wp.array(dtype=wp.vec3),
    bg_r: wp.uint8,
    bg_g: wp.uint8,
    bg_b: wp.uint8,
    width: int,
    rgba: wp.array(dtype=wp.uint8, ndim=3),  # (H, W, 4)
):
    pid = wp.tid()
    py = pid // width
    px = pid % width

    packed = depth_buf[pid]
    sentinel = wp.uint64(0xFFFFFFFFFFFFFFFF)

    if packed == sentinel:
        rgba[py, px, 0] = bg_r
        rgba[py, px, 1] = bg_g
        rgba[py, px, 2] = bg_b
    else:
        tri_id = wp.int32(packed & wp.uint64(0xFFFFFFFF))
        c = lit_colors[tri_id]
        rgba[py, px, 0] = wp.uint8(wp.clamp(c[0], 0.0, 255.0))
        rgba[py, px, 1] = wp.uint8(wp.clamp(c[1], 0.0, 255.0))
        rgba[py, px, 2] = wp.uint8(wp.clamp(c[2], 0.0, 255.0))
    rgba[py, px, 3] = wp.uint8(255)


# --------------------------------------------------------------------- class


class WarpRasterizer:
    """Warp-based GPU rasterizer driven by (mesh, instances) pairs."""

    def __init__(
        self,
        width: int,
        height: int,
        bg_color: tuple[int, int, int] = (8, 10, 14),
        ambient: float = 0.25,
        light_dir: tuple[float, float, float] = (-0.4, -0.6, -0.7),
        device: str = "cuda:0",
    ):
        wp.init()
        self.width = int(width)
        self.height = int(height)
        self._device = device
        self.ambient = float(ambient)

        bg = np.clip(np.asarray(bg_color, dtype=np.int32), 0, 255).astype(np.uint8)
        self._bg_r = wp.uint8(int(bg[0]))
        self._bg_g = wp.uint8(int(bg[1]))
        self._bg_b = wp.uint8(int(bg[2]))

        light = np.asarray(light_dir, dtype=np.float32)
        norm = float(np.linalg.norm(light))
        if norm > 1.0e-9:
            light = light / norm
        else:
            light = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        # Lambert uses n · (-light_dir), so cache the negated direction.
        neg = (-light).astype(np.float32)
        self._light_dir_neg = wp.vec3(float(neg[0]), float(neg[1]), float(neg[2]))

        self._depth_sentinel = wp.uint64(_DEPTH_SENTINEL)

        n_pix = self.width * self.height
        self._depth_buf = wp.zeros(n_pix, dtype=wp.uint64, device=device)
        self._rgba = wp.zeros((self.height, self.width, 4), dtype=wp.uint8, device=device)

        # Scratch: only the per-triangle shaded color is needed across kernels
        # (resolve kernel reads it by winning tri_id). Grown on demand.
        self._scratch_cap = 0
        self._lit_colors: wp.array | None = None
        self._ensure_scratch(1024)

    def _ensure_scratch(self, n_tris: int) -> None:
        if n_tris <= self._scratch_cap:
            return
        cap = max(n_tris, self._scratch_cap * 2, 1024)
        self._lit_colors = wp.zeros(cap, dtype=wp.vec3, device=self._device)
        self._scratch_cap = cap

    def clear(self) -> None:
        wp.launch(
            _clear_depth_kernel,
            dim=self.width * self.height,
            inputs=[self._depth_buf, self._depth_sentinel],
            device=self._device,
        )

    def upload_mesh(
        self, verts: np.ndarray, indices: np.ndarray
    ) -> tuple[wp.array, wp.array]:
        """Upload static mesh data to this rasterizer's device.

        Returns (verts_wp, idx_wp) that can be passed to :meth:`draw_batch`.
        Call once per mesh (at ``log_mesh`` time) and cache; the returned
        arrays are retained on GPU and reused each frame."""
        verts_f32 = np.ascontiguousarray(verts, dtype=np.float32).reshape(-1, 3)
        idx_i32 = np.ascontiguousarray(indices, dtype=np.int32).reshape(-1, 3)
        verts_wp = wp.array(verts_f32, dtype=wp.vec3, device=self._device)
        idx_wp = wp.array(idx_i32, dtype=wp.int32, device=self._device)
        return verts_wp, idx_wp

    def draw_batch(
        self,
        mesh_verts_wp: wp.array,
        mesh_idx_wp: wp.array,
        xforms,                                      # wp.array or np.ndarray (I, 7)
        scales,                                      # wp.array or np.ndarray (I, 3) or None
        colors,                                      # wp.array or np.ndarray (I, 3) or None
        default_color: np.ndarray,                   # (3,) fallback
        view: np.ndarray,
        proj: np.ndarray,
    ) -> None:
        """Fused expand + raster for one (mesh, instances) pair on GPU.

        ``xforms`` / ``scales`` / ``colors`` may be warp arrays (zero-copy,
        preferred path from Newton) or numpy arrays (upload fallback, used
        by standalone tests). Passing a ``wp.array(dtype=wp.transform)``
        from Newton state avoids an entire GPU→CPU→GPU roundtrip."""
        n_tris_per_inst = int(mesh_idx_wp.shape[0])
        n_inst = int(xforms.shape[0])
        n = n_tris_per_inst * n_inst
        if n == 0:
            return
        self._ensure_scratch(n)

        xforms_wp = self._as_wp_transform(xforms, n_inst)
        scales_wp = self._as_wp_vec3(scales, n_inst, fill=(1.0, 1.0, 1.0))
        if colors is not None:
            inst_colors_wp = self._as_wp_vec3(colors, n_inst)
        else:
            dc = tuple(float(x) for x in np.asarray(default_color, dtype=np.float32).reshape(3))
            inst_colors_wp = self._as_wp_vec3(None, n_inst, fill=dc)

        mvp_np = (proj @ view).astype(np.float32)
        mvp_wp = wp.mat44(mvp_np.flatten().tolist())

        wp.launch(
            _fused_expand_raster_kernel,
            dim=n,
            inputs=[
                mesh_verts_wp,
                mesh_idx_wp,
                xforms_wp,
                scales_wp,
                inst_colors_wp,
                n_tris_per_inst,
                mvp_wp,
                self._light_dir_neg,
                float(self.ambient),
                int(self.width),
                int(self.height),
                self._depth_buf,
                self._lit_colors,
            ],
            device=self._device,
        )

    def _as_wp_transform(self, arr, n_inst: int) -> wp.array:
        """Normalize instance transforms to ``wp.array(dtype=wp.transform)``.

        Zero-copy when ``arr`` is already a warp transform array. Otherwise
        converts through numpy (I, 7) float32 and uploads."""
        if isinstance(arr, wp.array) and arr.dtype == wp.transform:
            return arr
        if isinstance(arr, wp.array):
            arr = arr.numpy()
        arr_np = np.ascontiguousarray(arr, dtype=np.float32).reshape(n_inst, 7)
        return wp.array(arr_np, dtype=wp.transform, device=self._device)

    def _as_wp_vec3(
        self,
        arr,
        n_inst: int,
        fill: tuple[float, float, float] | None = None,
    ) -> wp.array:
        """Normalize a per-instance vec3 input (scales/colors) to wp.vec3 array.

        Zero-copy when ``arr`` is already a warp vec3 array. ``fill`` provides
        a default value when ``arr`` is None."""
        if isinstance(arr, wp.array) and arr.dtype == wp.vec3:
            return arr
        if arr is None:
            if fill is None:
                fill = (0.0, 0.0, 0.0)
            arr_np = np.tile(
                np.asarray(fill, dtype=np.float32).reshape(1, 3), (n_inst, 1)
            )
        else:
            if isinstance(arr, wp.array):
                arr = arr.numpy()
            arr_np = np.ascontiguousarray(arr, dtype=np.float32).reshape(n_inst, 3)
        return wp.array(arr_np, dtype=wp.vec3, device=self._device)

    def rgba(self) -> np.ndarray:
        wp.launch(
            _resolve_kernel,
            dim=self.width * self.height,
            inputs=[
                self._depth_buf,
                self._lit_colors,
                self._bg_r,
                self._bg_g,
                self._bg_b,
                int(self.width),
                self._rgba,
            ],
            device=self._device,
        )
        return self._rgba.numpy()
