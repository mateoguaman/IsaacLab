# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ASCII visualizer backend.

Renders the Newton scene to colored ASCII in a terminal via the ascii-cli
Node.js CLI (https://github.com/mateoguaman/ascii_renderer). A numpy software
rasterizer turns the scene into RGBA frames; those are piped to ``ascii-cli
--raw-in`` which prints the frame to stdout.

This package keeps imports lazy so configuration-only imports avoid pulling the
full runtime backend before it is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .ascii_visualizer_cfg import AsciiVisualizerCfg

if TYPE_CHECKING:
    from .ascii_visualizer import AsciiVisualizer

__all__ = ["AsciiVisualizer", "AsciiVisualizerCfg"]


def __getattr__(name: str):
    if name == "AsciiVisualizer":
        from .ascii_visualizer import AsciiVisualizer

        return AsciiVisualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
