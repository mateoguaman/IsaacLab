# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ASCII visualizer."""

from __future__ import annotations

from typing import Literal

from isaaclab.utils import configclass
from isaaclab.visualizers.visualizer_cfg import VisualizerCfg


@configclass
class AsciiVisualizerCfg(VisualizerCfg):
    """Configuration for ASCII visualizer (terminal shape-vector rendering).

    Renders the scene to RGBA frames with a numpy software rasterizer, then pipes
    the frames through the ``ascii-cli`` Node.js CLI which prints colored ASCII
    to stdout.
    """

    visualizer_type: str = "ascii"
    """Type identifier for ASCII visualizer."""

    output_mode: Literal["inline", "fifo", "tcp"] = "fifo"
    """Where raw RGBA frames are sent.

    * ``inline`` — spawn ``ascii-cli`` as a child process and feed its stdin. The
      subprocess inherits the parent's stdout, so ANSI output lands in the
      controlling terminal. Best for standalone demos. In a training run it
      interleaves with stdout logs (``print``, ``logger``, metric tables) and
      in alt-screen mode obliterates them — do not use for IsaacLab training.
    * ``fifo`` — create a named pipe on the filesystem; emit raw RGBA to it.
      The parent terminal is untouched. A second terminal consumes frames via
      ``cat <fifo_path> | ascii-cli --raw-in WxH`` (command is logged at init).
    * ``tcp`` — bind a TCP listener on ``(tcp_host, tcp_port)``; emit raw RGBA
      to whichever client is connected. Works across SSH / cluster boundaries.
      Consumer: ``nc <host> <port> | ascii-cli --raw-in WxH``.
    """

    ascii_cli_path: str = "ascii-cli"
    """Path to the ``ascii-cli`` executable. Used only in ``output_mode='inline'``.

    Defaults to ``ascii-cli`` on PATH. Set to an absolute path when the CLI is
    installed in a checkout (e.g. ``/path/to/ascii_renderer/bin/ascii-cli.mjs``).
    """

    fifo_path: str | None = None
    """Filesystem path for ``output_mode='fifo'``. ``None`` auto-generates
    ``/tmp/isaac-ascii-<pid>.fifo``."""

    tcp_host: str = "127.0.0.1"
    """Bind address for ``output_mode='tcp'``. Use ``0.0.0.0`` to accept
    connections from other hosts (careful: frames are unencrypted)."""

    tcp_port: int = 5555
    """TCP port for ``output_mode='tcp'``. Pass ``0`` to let the OS assign an
    ephemeral free port (the chosen port is logged at init)."""

    render_width: int = 320
    """Offscreen RGBA render width in pixels. Gets resampled by ascii-cli into
    terminal cells at ``render_width / sample_res`` columns."""

    render_height: int = 200
    """Offscreen RGBA render height in pixels."""

    sample_res: int = 8
    """Source pixels per terminal cell passed to ascii-cli (``--sample-res``)."""

    color_mode: Literal["truecolor", "256", "mono", "auto"] = "auto"
    """Color mode passed to ascii-cli (``--mode``). ``auto`` detects from env."""

    bg: Literal["dark", "light"] = "dark"
    """Terminal background hint (``--bg``). ``light`` flips the char mapping and
    clamps bright RGB so output stays visible on cream terminals."""

    fov_deg: float = 45.0
    """Vertical field of view of the rasterizer camera, in degrees."""

    near: float = 0.05
    """Near clip distance."""

    far: float = 100.0
    """Far clip distance."""

    light_direction: tuple[float, float, float] = (-0.4, -0.6, -0.7)
    """World-space direction the single directional light travels. Normalized
    before use. Default points down and slightly forward for typical robot
    scenes with Z-up."""

    ambient: float = 0.25
    """Ambient light strength added to lit color. Range ``[0, 1]``."""

    default_color: tuple[float, float, float] = (0.72, 0.74, 0.78)
    """Fallback base color applied to shapes with no per-instance color."""

    background_color: tuple[int, int, int] = (8, 10, 14)
    """Clear color (RGB 0-255) for pixels with no geometry. Slightly brighter
    than pure black so ASCII density stays readable on very dark rendered
    objects."""
