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

    ascii_cli_path: str = "ascii-cli"
    """Path (or command on PATH) to the ascii-cli Node.js executable.

    Defaults to ``ascii-cli`` on PATH. Set to an absolute path when the CLI is
    installed in a checkout (e.g. ``/path/to/ascii_renderer/bin/ascii-cli.mjs``).
    """

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
