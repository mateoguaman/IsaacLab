# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton ``ViewerBase`` subclass that rasterizes the scene to RGBA with a
numpy software renderer, then pipes each frame to ``ascii-cli --raw-in`` for
colored ASCII terminal output.

Design: ``log_mesh`` registers a mesh prototype; ``log_instances`` stores the
current per-instance transforms; ``end_frame`` rasterizes everything and writes
the RGBA bytes to the ascii-cli subprocess stdin.
"""

from __future__ import annotations

import atexit
import logging
import math
import shutil
import subprocess
from typing import Any

import numpy as np

import newton
from newton._src.viewer.viewer import ViewerBase

logger = logging.getLogger(__name__)


def _to_numpy(arr: Any) -> np.ndarray | None:
    """Best-effort conversion of a Warp array / torch tensor / ndarray to numpy."""
    if arr is None:
        return None
    if isinstance(arr, np.ndarray):
        return arr
    numpy_fn = getattr(arr, "numpy", None)
    if callable(numpy_fn):
        try:
            return numpy_fn()
        except Exception:
            pass
    return np.asarray(arr)


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v`` (..., 3) by unit quaternions ``q`` (..., 4) in xyzw order."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    # v' = v + 2 * q_xyz × (q_xyz × v + w v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return np.stack([rx, ry, rz], axis=-1)


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Right-handed view matrix (world → camera) with -Z forward, +Y up."""
    f = target - eye
    f /= np.linalg.norm(f) + 1e-12
    s = np.cross(f, up)
    s /= np.linalg.norm(s) + 1e-12
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def _perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Standard OpenGL-style perspective matrix (clip Z in [-1, 1])."""
    f = 1.0 / math.tan(math.radians(fov_y_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


class NumpyRasterizer:
    """Small software rasterizer: z-buffered flat-shaded triangles in numpy."""

    def __init__(
        self,
        width: int,
        height: int,
        bg_color: tuple[int, int, int] = (8, 10, 14),
        ambient: float = 0.25,
        light_dir: tuple[float, float, float] = (-0.4, -0.6, -0.7),
    ):
        self.width = int(width)
        self.height = int(height)
        self.bg_color = np.array(bg_color, dtype=np.float32)
        self.ambient = float(ambient)
        light = np.array(light_dir, dtype=np.float32)
        norm = np.linalg.norm(light)
        self.light_dir = light / norm if norm > 1e-9 else np.array([0, 0, -1], dtype=np.float32)
        self._color_buf: np.ndarray | None = None
        self._depth_buf: np.ndarray | None = None

    def clear(self) -> None:
        self._color_buf = np.broadcast_to(self.bg_color, (self.height, self.width, 3)).copy()
        self._depth_buf = np.full((self.height, self.width), np.inf, dtype=np.float32)

    def rgba(self) -> np.ndarray:
        assert self._color_buf is not None
        rgba = np.empty((self.height, self.width, 4), dtype=np.uint8)
        rgba[..., :3] = np.clip(self._color_buf, 0, 255).astype(np.uint8)
        rgba[..., 3] = 255
        return rgba

    def draw_triangles(
        self,
        world_tris: np.ndarray,
        base_colors: np.ndarray,
        view: np.ndarray,
        proj: np.ndarray,
    ) -> None:
        """Rasterize a batch of world-space triangles.

        Args:
            world_tris: (N, 3, 3) vertex positions in world space.
            base_colors: (N, 3) RGB in [0, 1].
            view: 4x4 view matrix.
            proj: 4x4 projection matrix.
        """
        if world_tris.size == 0:
            return
        assert self._color_buf is not None and self._depth_buf is not None

        n_tris = world_tris.shape[0]
        # World-space face normals for lighting.
        edge1 = world_tris[:, 1] - world_tris[:, 0]
        edge2 = world_tris[:, 2] - world_tris[:, 0]
        normals = np.cross(edge1, edge2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.where(norms > 1e-9, normals / np.maximum(norms, 1e-9), np.array([0, 0, 1.0]))

        # Flat shading: Lambertian with -light_dir as the incoming direction.
        lambert = np.clip(normals @ (-self.light_dir), 0.0, 1.0)  # (N,)
        shade = self.ambient + (1.0 - self.ambient) * lambert  # (N,)
        lit = base_colors * shade[:, None] * 255.0  # (N, 3) in [0, 255]

        # World → clip.
        verts_h = np.concatenate([world_tris.reshape(-1, 3), np.ones((n_tris * 3, 1), dtype=np.float32)], axis=1)
        clip = verts_h @ (proj @ view).T  # (3N, 4)
        w = clip[:, 3:4]
        # Skip triangles with any vertex behind the near plane (simple cull).
        w3 = w.reshape(n_tris, 3)
        front = np.all(w3 > 1e-4, axis=1)
        if not np.any(front):
            return
        ndc = clip[:, :3] / np.maximum(np.abs(w), 1e-9) * np.sign(w)
        ndc = ndc.reshape(n_tris, 3, 3)  # (N, 3 verts, xyz)

        # NDC → screen (pixel center at integer coord).
        sx = (ndc[..., 0] + 1.0) * 0.5 * (self.width - 1)
        sy = (1.0 - ndc[..., 1]) * 0.5 * (self.height - 1)
        sz = ndc[..., 2]  # depth in [-1, 1]

        for i in np.flatnonzero(front):
            self._raster_one(sx[i], sy[i], sz[i], lit[i])

    def _raster_one(self, sx: np.ndarray, sy: np.ndarray, sz: np.ndarray, color: np.ndarray) -> None:
        # Screen-space bbox.
        x_min = max(int(math.floor(sx.min())), 0)
        x_max = min(int(math.ceil(sx.max())), self.width - 1)
        y_min = max(int(math.floor(sy.min())), 0)
        y_max = min(int(math.ceil(sy.max())), self.height - 1)
        if x_min > x_max or y_min > y_max:
            return

        (x0, y0), (x1, y1), (x2, y2) = zip(sx, sy)
        area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        if abs(area) < 1e-6:
            return
        # Consistent winding: fold so area > 0 and remember flip for culling.
        if area < 0:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            sz = sz[[0, 2, 1]]
            area = -area

        xs = np.arange(x_min, x_max + 1, dtype=np.float32)
        ys = np.arange(y_min, y_max + 1, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)

        w0 = (x1 - gx) * (y2 - gy) - (y1 - gy) * (x2 - gx)
        w1 = (x2 - gx) * (y0 - gy) - (y2 - gy) * (x0 - gx)
        w2 = (x0 - gx) * (y1 - gy) - (y0 - gy) * (x1 - gx)
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            return

        inv_area = 1.0 / area
        bary0 = w0 * inv_area
        bary1 = w1 * inv_area
        bary2 = w2 * inv_area
        depth = bary0 * sz[0] + bary1 * sz[1] + bary2 * sz[2]

        sub_depth = self._depth_buf[y_min : y_max + 1, x_min : x_max + 1]
        sub_color = self._color_buf[y_min : y_max + 1, x_min : x_max + 1]
        mask = inside & (depth < sub_depth)
        if not mask.any():
            return
        sub_depth[mask] = depth[mask]
        sub_color[mask] = color


class ViewerAscii(ViewerBase):
    """Newton viewer that rasterizes to RGBA and pipes to ``ascii-cli --raw-in``."""

    def __init__(
        self,
        ascii_cli_path: str = "ascii-cli",
        render_width: int = 320,
        render_height: int = 200,
        sample_res: int = 8,
        color_mode: str = "auto",
        bg: str = "dark",
        fov_deg: float = 45.0,
        near: float = 0.05,
        far: float = 100.0,
        light_direction: tuple[float, float, float] = (-0.4, -0.6, -0.7),
        ambient: float = 0.25,
        default_color: tuple[float, float, float] = (0.72, 0.74, 0.78),
        background_color: tuple[int, int, int] = (8, 10, 14),
        camera_position: tuple[float, float, float] = (8.0, 8.0, 3.0),
        camera_target: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        super().__init__()
        self._cli_path = self._resolve_cli(ascii_cli_path)
        self._render_w = int(render_width)
        self._render_h = int(render_height)
        self._sample_res = int(sample_res)
        self._color_mode = color_mode
        self._bg = bg
        self._fov = float(fov_deg)
        self._near = float(near)
        self._far = float(far)
        self._default_color = np.array(default_color, dtype=np.float32)
        self._camera_pos = np.array(camera_position, dtype=np.float32)
        self._camera_target = np.array(camera_target, dtype=np.float32)
        # Z-up matches Newton/Isaac Lab convention; _look_at handles orthogonalization.
        self._up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        self._rasterizer = NumpyRasterizer(
            width=self._render_w,
            height=self._render_h,
            bg_color=background_color,
            ambient=ambient,
            light_dir=light_direction,
        )

        self._meshes: dict[str, dict[str, np.ndarray]] = {}
        self._instance_batches: dict[str, dict[str, Any]] = {}

        self._proc: subprocess.Popen | None = None
        self._closed = False
        self._start_subprocess()
        atexit.register(self.close)

    @staticmethod
    def _resolve_cli(path: str) -> str:
        """Resolve ``ascii-cli`` path: absolute path, bare name on PATH, or the
        known Node script name."""
        # Absolute/explicit path.
        if "/" in path or path.endswith(".mjs"):
            return path
        found = shutil.which(path)
        if found:
            return found
        # Some users install by filename; try the mjs variant too.
        for candidate in (f"{path}.mjs", "ascii-cli.mjs"):
            found = shutil.which(candidate)
            if found:
                return found
        raise FileNotFoundError(
            f"Could not locate ascii-cli executable '{path}'. Pass an absolute path via "
            "AsciiVisualizerCfg.ascii_cli_path (e.g. /path/to/ascii_renderer/bin/ascii-cli.mjs)."
        )

    def _start_subprocess(self) -> None:
        """Spawn ``ascii-cli --raw-in WxH`` with stdout inherited (ANSI lands in
        the parent terminal). stdin is where we feed RGBA bytes."""
        cmd = [
            self._cli_path,
            "--raw-in",
            f"{self._render_w}x{self._render_h}",
            "--sample-res",
            str(self._sample_res),
            "--mode",
            self._color_mode,
            "--bg",
            self._bg,
        ]
        # If the path ends in .mjs and is not executable, prefix with node.
        if self._cli_path.endswith(".mjs"):
            import os

            if not os.access(self._cli_path, os.X_OK):
                cmd = ["node", *cmd]
        logger.info("[ViewerAscii] spawning %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            bufsize=0,  # unbuffered; we flush per frame
        )

    # --- ViewerBase abstract overrides ---

    def log_mesh(
        self,
        name: str,
        points,
        indices,
        normals=None,
        uvs=None,
        texture=None,
        hidden: bool = False,
        backface_culling: bool = True,
    ) -> None:
        verts = _to_numpy(points)
        idx = _to_numpy(indices)
        if verts is None or idx is None:
            return
        verts = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
        idx = np.asarray(idx, dtype=np.int32).reshape(-1, 3)
        self._meshes[name] = {"vertices": verts, "indices": idx}

    def log_instances(
        self,
        name: str,
        mesh: str,
        xforms,
        scales,
        colors,
        materials,
        hidden: bool = False,
    ) -> None:
        if hidden or xforms is None:
            self._instance_batches.pop(name, None)
            return
        xforms_np = _to_numpy(xforms)
        if xforms_np is None:
            self._instance_batches.pop(name, None)
            return
        # Each transform is (pos.x, pos.y, pos.z, q.x, q.y, q.z, q.w).
        xforms_np = np.asarray(xforms_np, dtype=np.float32).reshape(-1, 7)
        scales_np = _to_numpy(scales)
        if scales_np is not None:
            scales_np = np.asarray(scales_np, dtype=np.float32).reshape(-1, 3)
        colors_np = _to_numpy(colors)
        if colors_np is not None:
            colors_np = np.asarray(colors_np, dtype=np.float32).reshape(-1, 3)
        self._instance_batches[name] = {
            "mesh": mesh,
            "xforms": xforms_np,
            "scales": scales_np,
            "colors": colors_np,
        }

    def end_frame(self) -> None:
        if self._closed:
            return
        self._rasterizer.clear()

        view = _look_at(self._camera_pos, self._camera_target, self._up)
        aspect = self._render_w / max(1, self._render_h)
        proj = _perspective(self._fov, aspect, self._near, self._far)

        for batch in self._instance_batches.values():
            mesh = self._meshes.get(batch["mesh"])
            if mesh is None:
                continue
            xforms = batch["xforms"]
            if xforms.shape[0] == 0:
                continue
            world_tris, tri_colors = self._world_triangles(mesh, batch)
            if world_tris.size == 0:
                continue
            self._rasterizer.draw_triangles(world_tris, tri_colors, view, proj)

        rgba = self._rasterizer.rgba()
        self._write_frame(rgba)

    def _world_triangles(
        self, mesh: dict[str, np.ndarray], batch: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expand a (mesh, instances) pair into (N_tris, 3, 3) world vertices +
        (N_tris, 3) RGB base colors."""
        verts = mesh["vertices"]  # (V, 3)
        idx = mesh["indices"]  # (T, 3)
        xforms = batch["xforms"]  # (I, 7)
        scales = batch["scales"]  # (I, 3) or None
        colors = batch["colors"]  # (I, 3) or None

        n_inst = xforms.shape[0]
        # Scale → rotate → translate for each instance.
        tris = []
        col_out = []
        for i in range(n_inst):
            s = scales[i] if scales is not None else np.array([1.0, 1.0, 1.0], dtype=np.float32)
            t = xforms[i, 0:3]
            q = xforms[i, 3:7]
            scaled = verts * s[None, :]
            rotated = _quat_rotate(np.broadcast_to(q, scaled.shape[:-1] + (4,)), scaled)
            world = rotated + t[None, :]
            tris.append(world[idx])
            if colors is not None:
                c = colors[i]
            else:
                c = self._default_color
            col_out.append(np.broadcast_to(c, (idx.shape[0], 3)))
        return np.concatenate(tris, axis=0), np.concatenate(col_out, axis=0)

    def _write_frame(self, rgba: np.ndarray) -> None:
        if self._proc is None or self._proc.stdin is None:
            return
        try:
            self._proc.stdin.write(rgba.tobytes())
            self._proc.stdin.flush()
        except BrokenPipeError:
            logger.warning("[ViewerAscii] ascii-cli pipe closed; marking viewer as stopped.")
            self._closed = True

    # --- other overrides (mostly no-ops for MVP) ---

    def is_running(self) -> bool:
        if self._closed:
            return False
        if self._proc is None:
            return False
        return self._proc.poll() is None

    def log_lines(self, name, starts, ends, colors, width: float = 0.01, hidden: bool = False) -> None:
        pass

    def log_points(self, name, points, radii=None, colors=None, hidden: bool = False) -> None:
        pass

    def log_array(self, name: str, array) -> None:
        pass

    def log_scalar(self, name: str, value) -> None:
        pass

    def apply_forces(self, state: "newton.State") -> None:
        pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._proc is not None:
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
            self._proc = None

    # --- camera helpers ---

    def set_camera(self, pos, pitch: float, yaw: float) -> None:
        """Minimal set_camera implementation (ignores pitch/yaw; we look at a
        fixed target). Newton-side callers generally use this with converted
        yaw/pitch; for our use case explicit pos/target is simpler and we keep
        the target from construction or caller override."""
        self._camera_pos = np.asarray(pos, dtype=np.float32).reshape(3)

    def set_camera_view(
        self,
        eye: tuple[float, float, float],
        target: tuple[float, float, float],
    ) -> None:
        """Non-standard helper used by AsciiVisualizer to drive the camera."""
        self._camera_pos = np.asarray(eye, dtype=np.float32)
        self._camera_target = np.asarray(target, dtype=np.float32)
