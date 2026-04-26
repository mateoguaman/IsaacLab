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
import time
from contextlib import contextmanager
from typing import Any

import numpy as np
import warp as wp

import newton
from newton._src.viewer.viewer import ViewerBase

from .warp_rasterizer import WarpRasterizer

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


class _AsciiPerfMeter:
    """Lightweight stage timer gated by ``ISAACLAB_ASCII_PROFILE`` env var.

    Accumulates per-stage wall time across frames and periodically logs the
    mean per-frame cost of each stage plus an overall frame rate. When
    disabled the ``stage()`` context manager is a no-op so steady-state
    overhead is a single env-var check per call."""

    def __init__(self, period: int = 60):
        env_period = os.environ.get("ISAACLAB_ASCII_PROFILE_PERIOD", "").strip()
        if env_period:
            try:
                period = max(1, int(env_period))
            except ValueError:
                pass
        self._period = max(1, int(period))
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._extras: dict[str, list[float]] = {}
        self._n_frames = 0
        self._last_wall: float | None = None
        flag = os.environ.get("ISAACLAB_ASCII_PROFILE", "").strip().lower()
        self._enabled = flag in ("1", "true", "yes", "on")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @contextmanager
    def stage(self, name: str):
        if not self._enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self._sums[name] = self._sums.get(name, 0.0) + dt
            self._counts[name] = self._counts.get(name, 0) + 1

    def record(self, name: str, value: float) -> None:
        if not self._enabled:
            return
        self._extras.setdefault(name, []).append(float(value))

    def tick(self) -> None:
        if not self._enabled:
            return
        self._n_frames += 1
        if self._n_frames % self._period:
            return
        now = time.perf_counter()
        parts: list[str] = []
        if self._last_wall is not None:
            fps = self._period / max(now - self._last_wall, 1e-9)
            parts.append(f"fps={fps:.1f}")
        self._last_wall = now
        for name in sorted(self._sums):
            mean_ms = self._sums[name] / max(self._counts[name], 1) * 1000.0
            parts.append(f"{name}={mean_ms:.2f}ms")
        for name in sorted(self._extras):
            vals = self._extras[name]
            if not vals:
                continue
            parts.append(f"{name}={sum(vals) / len(vals):.1f}")
        logger.info("[ViewerAscii perf] frame=%d %s", self._n_frames, " ".join(parts))
        self._sums.clear()
        self._counts.clear()
        self._extras.clear()


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


# Cell aspect ratio used by AsciiRenderer.computeGrid — cellH = sample_res * 4/3.
# Duplicated here so the producer can report the expected terminal grid without
# importing the JS renderer. Kept in sync manually with alphabet.metadata.
_CELL_ASPECT = 4.0 / 3.0

# Conventional install location advertised in the connection block. Users with
# a different layout can adjust the printed path by hand; no need to make this
# configurable until it becomes a real friction point.
_DEFAULT_ADVERTISED_ASCII_CLI = "~/projects/ascii_renderer/bin/ascii-cli.mjs"


def _expected_grid(render_w: int, render_h: int, sample_res: int) -> tuple[int, int]:
    """Cols × rows the viewer terminal must be at least as large as."""
    cols = max(1, render_w // sample_res)
    rows = max(1, int(render_h // (sample_res * _CELL_ASPECT)))
    return cols, rows


def _format_viewer_ready_block(
    host: str, port: int, render_w: int, render_h: int,
    sample_res: int, color_mode: str, bg: str,
) -> str:
    """Build the multi-line 'ASCII VIEWER READY' block advertised on TCP start.

    Topology-agnostic: prints only what the producer knows about itself. The
    user supplies their own SSH target (alias, -J jumphost, etc.) via their
    ~/.ssh/config — the command template just shows the default `ssh <host>`."""
    cols, rows = _expected_grid(render_w, render_h, sample_res)
    cli = _DEFAULT_ADVERTISED_ASCII_CLI
    raw = f"{render_w}x{render_h}"
    mode = color_mode if color_mode != "auto" else "truecolor"
    long_cmd = (
        f"  ssh {host} \"nc localhost {port}\" \\\n"
        f"    | {cli} \\\n"
        f"      --raw-in {raw} --sample-res {sample_res} --mode {mode} --bg {bg}"
    )
    short_cmd = f"  ascii-connect {host} {port} {raw} @{sample_res}"
    return (
        "\n================ ASCII VIEWER READY ================\n"
        f"host    : {host}\n"
        f"port    : {port}\n"
        f"frame   : {raw} @ sample-res={sample_res}\n"
        f"grid    : >={cols}x{rows} cells  (viewer terminal must be at least this big)\n"
        "\n"
        "On any machine from which you can SSH here, open a terminal, shrink the\n"
        "font (Cmd-minus / Ctrl-minus in most terminals), and run:\n"
        "\n"
        f"{long_cmd}\n"
        "\n"
        f"If `ssh {host}` doesn't work directly, adjust the SSH target the way\n"
        "you'd normally reach this host (aliases, `-J loginnode`, keys, etc.) —\n"
        "only the SSH target changes; everything else stays the same.\n"
        "\n"
        "Shorter form if you have ascii-connect on PATH:\n"
        f"{short_cmd}\n"
        "====================================================\n"
    )


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

    def __init__(self, host: str, port: int, sample_res: int, color_mode: str, bg: str):
        self._host = host
        self._port = int(port)
        self._sample_res = int(sample_res)
        self._color_mode = color_mode
        self._bg = bg
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
        # socket.gethostname() gives the advertisable name: a workstation's own
        # name on a LAN, or a slurm compute node's internal name (e.g. g3060)
        # that resolves via the user's ~/.ssh/config ProxyJump stanza. Falls
        # back to the bind address if resolution fails.
        advertised_host = socket.gethostname() or self._host
        logger.info(
            "%s",
            _format_viewer_ready_block(
                advertised_host, actual_port, render_w, render_h,
                self._sample_res, self._color_mode, self._bg,
            ),
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

        self._rasterizer = WarpRasterizer(
            width=self._render_w,
            height=self._render_h,
            bg_color=background_color,
            ambient=ambient,
            light_dir=light_direction,
        )

        self._meshes: dict[str, dict[str, np.ndarray]] = {}
        self._instance_batches: dict[str, dict[str, Any]] = {}

        self._perf = _AsciiPerfMeter(period=60)
        if self._perf.enabled:
            logger.info(
                "[ViewerAscii] profiler enabled (ISAACLAB_ASCII_PROFILE); logging every 60 frames"
            )

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
            return _TcpTransport(tcp_host, tcp_port, sample_res, color_mode, bg)
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
        verts_wp, idx_wp = self._rasterizer.upload_mesh(verts, idx)
        self._meshes[name] = {
            "vertices": verts,
            "indices": idx,
            "verts_wp": verts_wp,
            "idx_wp": idx_wp,
        }

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

        # Keep warp arrays as-is so draw_batch can consume them on GPU
        # without a GPU→CPU→GPU roundtrip. Only numpy-like inputs are
        # materialized to numpy here.
        def _keep_or_numpy(arr, row_len: int):
            if arr is None:
                return None
            if isinstance(arr, wp.array):
                return arr
            return np.asarray(_to_numpy(arr), dtype=np.float32).reshape(-1, row_len)

        xforms_stored = _keep_or_numpy(xforms, 7)
        if xforms_stored is None:
            self._instance_batches.pop(name, None)
            return
        scales_stored = _keep_or_numpy(scales, 3)
        colors_stored = _keep_or_numpy(colors, 3)
        self._instance_batches[name] = {
            "mesh": mesh,
            "xforms": xforms_stored,
            "scales": scales_stored,
            "colors": colors_stored,
        }

    def end_frame(self) -> None:
        if self._closed:
            return
        with self._perf.stage("1_clear"):
            self._rasterizer.clear()

        with self._perf.stage("2_matrices"):
            view = _look_at(self._camera_pos, self._camera_target, self._up)
            aspect = self._render_w / max(1, self._render_h)
            proj = _perspective(self._fov, aspect, self._near, self._far)

        n_inst_total = 0
        n_tris_total = 0
        with self._perf.stage("3_expand_and_draw"):
            for batch in self._instance_batches.values():
                mesh = self._meshes.get(batch["mesh"])
                if mesh is None:
                    continue
                xforms = batch["xforms"]
                if xforms.shape[0] == 0:
                    continue
                n_inst = int(xforms.shape[0])
                n_tris_per_inst = int(mesh["indices"].shape[0])
                n_inst_total += n_inst
                n_tris_total += n_inst * n_tris_per_inst
                self._rasterizer.draw_batch(
                    mesh["verts_wp"],
                    mesh["idx_wp"],
                    xforms,
                    batch["scales"],
                    batch["colors"],
                    self._default_color,
                    view,
                    proj,
                )

        self._perf.record("n_instances", n_inst_total)
        self._perf.record("n_triangles", n_tris_total)

        with self._perf.stage("4_rgba_pack"):
            rgba = self._rasterizer.rgba()

        with self._perf.stage("5_transport_write"):
            self._write_frame(rgba)

        self._perf.tick()

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
