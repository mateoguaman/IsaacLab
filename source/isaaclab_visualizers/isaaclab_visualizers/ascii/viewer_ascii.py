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
import errno
import logging
import math
import os
import shutil
import socket
import subprocess
from typing import Any

import numpy as np

import newton
from newton._src.viewer.viewer import ViewerBase

logger = logging.getLogger(__name__)

# Linux F_SETPIPE_SZ fcntl op. Not exposed via fcntl module names on all
# platforms; hard-coded so we don't force a Linux-only import.
_F_SETPIPE_SZ = 1031


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


class _Transport:
    """Output transport for RGBA frames. Subclasses own one channel (inline
    subprocess, named FIFO, or TCP client) and implement write/close."""

    def start(self, render_w: int, render_h: int) -> None:
        raise NotImplementedError

    def write(self, rgba: bytes) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def is_alive(self) -> bool:
        """Whether the transport is still functional (used by is_running)."""
        return True


def _resolve_ascii_cli(path: str) -> str:
    """Locate ``ascii-cli`` via explicit path, PATH, or the ``.mjs`` variant."""
    if "/" in path or path.endswith(".mjs"):
        return path
    found = shutil.which(path)
    if found:
        return found
    for candidate in (f"{path}.mjs", "ascii-cli.mjs"):
        found = shutil.which(candidate)
        if found:
            return found
    raise FileNotFoundError(
        f"Could not locate ascii-cli executable '{path}'. Pass an absolute path via "
        "AsciiVisualizerCfg.ascii_cli_path (e.g. /path/to/ascii_renderer/bin/ascii-cli.mjs)."
    )


def _ascii_cli_consumer_cmd(render_w: int, render_h: int) -> str:
    """Best-effort consumer command string for log messages."""
    cli = shutil.which("ascii-cli") or shutil.which("ascii-cli.mjs") or "/path/to/ascii-cli.mjs"
    return f"{cli} --raw-in {render_w}x{render_h}"


class _InlineTransport(_Transport):
    """Spawn ascii-cli as a child process and feed its stdin. Subprocess stdout
    is inherited from the parent Python process."""

    def __init__(self, ascii_cli_path: str, sample_res: int, color_mode: str, bg: str):
        self._cli_path = _resolve_ascii_cli(ascii_cli_path)
        self._sample_res = int(sample_res)
        self._color_mode = color_mode
        self._bg = bg
        self._proc: subprocess.Popen | None = None

    def start(self, render_w: int, render_h: int) -> None:
        cmd = [
            self._cli_path,
            "--raw-in", f"{render_w}x{render_h}",
            "--sample-res", str(self._sample_res),
            "--mode", self._color_mode,
            "--bg", self._bg,
        ]
        if self._cli_path.endswith(".mjs") and not os.access(self._cli_path, os.X_OK):
            cmd = ["node", *cmd]
        logger.info("[ViewerAscii] inline transport: spawning %s", " ".join(cmd))
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=0)

    def write(self, rgba: bytes) -> None:
        if self._proc is None or self._proc.stdin is None:
            return
        try:
            self._proc.stdin.write(rgba)
            self._proc.stdin.flush()
        except BrokenPipeError:
            logger.warning("[ViewerAscii] ascii-cli pipe closed; stopping.")
            self._proc = None

    def close(self) -> None:
        if self._proc is None:
            return
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

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


class _FifoTransport(_Transport):
    """Write RGBA frames to a named pipe. Consumer attaches via ``cat <fifo> | ascii-cli``."""

    def __init__(self, fifo_path: str | None):
        self._path = fifo_path
        self._fd: int | None = None
        self._frame_size = 0
        self._open_failed_logged = False

    def start(self, render_w: int, render_h: int) -> None:
        if self._path is None:
            self._path = f"/tmp/isaac-ascii-{os.getpid()}.fifo"
        # Recreate atomically so a crash'd previous run doesn't collide.
        if os.path.exists(self._path):
            try:
                os.unlink(self._path)
            except OSError as exc:
                logger.warning("[ViewerAscii] could not remove stale FIFO %s: %s", self._path, exc)
        os.mkfifo(self._path, 0o600)
        self._frame_size = render_w * render_h * 4
        logger.info(
            "[ViewerAscii] fifo transport ready at %s. In another terminal, run:\n  cat %s | %s",
            self._path, self._path, _ascii_cli_consumer_cmd(render_w, render_h),
        )

    def _ensure_fd(self) -> bool:
        if self._fd is not None:
            return True
        if self._path is None:
            return False
        try:
            self._fd = os.open(self._path, os.O_WRONLY | os.O_NONBLOCK)
        except OSError as exc:
            if exc.errno == errno.ENXIO:
                # No reader attached yet. Expected during startup / between sessions.
                if not self._open_failed_logged:
                    logger.debug("[ViewerAscii] FIFO has no reader yet; dropping frames.")
                    self._open_failed_logged = True
                return False
            raise
        # Enlarge pipe buffer so full-frame writes are atomic and don't partial-write.
        try:
            import fcntl
            fcntl.fcntl(self._fd, _F_SETPIPE_SZ, max(65536, self._frame_size * 2))
        except (OSError, ImportError):
            pass  # not fatal; just may lead to partial writes under load
        # Switch to blocking writes so individual frame writes are atomic.
        os.set_blocking(self._fd, True)
        self._open_failed_logged = False
        logger.info("[ViewerAscii] FIFO reader attached; streaming frames.")
        return True

    def write(self, rgba: bytes) -> None:
        if not self._ensure_fd():
            return
        try:
            n = os.write(self._fd, rgba)
            if n != len(rgba):
                logger.warning("[ViewerAscii] fifo partial write %d/%d (frame dropped)", n, len(rgba))
        except BrokenPipeError:
            # Consumer disconnected.
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def close(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        if self._path and os.path.exists(self._path):
            try:
                os.unlink(self._path)
            except OSError:
                pass

    def is_alive(self) -> bool:
        return True  # a fifo with no reader is still live; we just drop frames


class _TcpTransport(_Transport):
    """Listen on TCP; stream RGBA to the most recently connected client."""

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = int(port)
        self._listener: socket.socket | None = None
        self._client: socket.socket | None = None

    def start(self, render_w: int, render_h: int) -> None:
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((self._host, self._port))
        self._listener.listen(1)
        self._listener.setblocking(False)
        actual_port = self._listener.getsockname()[1]
        self._port = actual_port
        logger.info(
            "[ViewerAscii] tcp transport listening on %s:%d. In another terminal, run:\n  nc %s %d | %s",
            self._host, actual_port, self._host, actual_port, _ascii_cli_consumer_cmd(render_w, render_h),
        )

    def _poll_new_client(self) -> None:
        if self._listener is None:
            return
        try:
            conn, addr = self._listener.accept()
        except BlockingIOError:
            return
        logger.info("[ViewerAscii] TCP client connected from %s:%d", *addr)
        if self._client is not None:
            try:
                self._client.close()
            except OSError:
                pass
        conn.setblocking(True)
        try:
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        except OSError:
            pass
        self._client = conn

    def write(self, rgba: bytes) -> None:
        self._poll_new_client()
        if self._client is None:
            return
        try:
            self._client.sendall(rgba)
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            logger.info("[ViewerAscii] TCP client disconnected: %s", exc)
            try:
                self._client.close()
            except OSError:
                pass
            self._client = None

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except OSError:
                pass
            self._client = None
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
            self._listener = None

    def is_alive(self) -> bool:
        return self._listener is not None


class ViewerAscii(ViewerBase):
    """Newton viewer that rasterizes to RGBA and pipes to ``ascii-cli --raw-in``."""

    def __init__(
        self,
        output_mode: str = "fifo",
        ascii_cli_path: str = "ascii-cli",
        fifo_path: str | None = None,
        tcp_host: str = "127.0.0.1",
        tcp_port: int = 5555,
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
        self._render_w = int(render_width)
        self._render_h = int(render_height)
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

        self._transport = self._make_transport(
            output_mode, ascii_cli_path, sample_res, color_mode, bg, fifo_path, tcp_host, tcp_port
        )
        self._closed = False
        self._transport.start(self._render_w, self._render_h)
        atexit.register(self.close)

    @staticmethod
    def _make_transport(
        output_mode: str,
        ascii_cli_path: str,
        sample_res: int,
        color_mode: str,
        bg: str,
        fifo_path: str | None,
        tcp_host: str,
        tcp_port: int,
    ) -> _Transport:
        if output_mode == "inline":
            return _InlineTransport(ascii_cli_path, sample_res, color_mode, bg)
        if output_mode == "fifo":
            return _FifoTransport(fifo_path)
        if output_mode == "tcp":
            return _TcpTransport(tcp_host, tcp_port)
        raise ValueError(
            f"Unknown output_mode '{output_mode}'. Valid: 'inline', 'fifo', 'tcp'."
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
        if self._closed:
            return
        self._transport.write(rgba.tobytes())

    # --- other overrides (mostly no-ops for MVP) ---

    def is_running(self) -> bool:
        if self._closed:
            return False
        return self._transport.is_alive()

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
        try:
            self._transport.close()
        except Exception as exc:
            logger.warning("[ViewerAscii] transport close error: %s", exc)

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
