#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test for the ASCII visualizer without launching Kit/Isaac Sim.

Constructs synthetic mesh/instance data and drives ``ViewerAscii`` directly,
bypassing ``AsciiVisualizer``'s ``BaseSceneDataProvider`` requirement. This
exercises the numpy rasterizer and the ``ascii-cli --raw-in`` subprocess
pipe — the two pieces most likely to have bugs.

Usage:
    ./isaaclab.sh -p scripts/demos/ascii_visualizer_smoke.py \\
        --ascii-cli /path/to/ascii_renderer/bin/ascii-cli.mjs
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

from isaaclab_visualizers.ascii.viewer_ascii import ViewerAscii


def make_cube(size: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    s = 0.5 * size
    verts = np.array(
        [
            [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
            [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [2, 3, 7], [2, 7, 6],
            [1, 2, 6], [1, 6, 5],
            [0, 4, 7], [0, 7, 3],
        ],
        dtype=np.int32,
    )
    return verts, faces


def make_plane(size: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    s = 0.5 * size
    verts = np.array(
        [[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return verts, faces


def quat_z(angle: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(angle * 0.5), math.cos(angle * 0.5)], dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ascii-cli",
        default="ascii-cli",
        help="Path or command name of ascii-cli (the Node.js CLI). Used only with --output-mode=inline.",
    )
    ap.add_argument(
        "--output-mode",
        default="inline",
        choices=("inline", "fifo", "tcp"),
        help="Transport for RGBA frames. 'inline' spawns ascii-cli locally (default for this demo).",
    )
    ap.add_argument("--fifo-path", default=None, help="Override auto-generated FIFO path.")
    ap.add_argument("--tcp-host", default="127.0.0.1")
    ap.add_argument("--tcp-port", type=int, default=5555)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--render-size", default="320x200", help="Offscreen RGBA size (WxH).")
    ap.add_argument("--sample-res", type=int, default=8, help="Pixels per terminal cell.")
    ap.add_argument("--color-mode", default="auto", choices=("auto", "truecolor", "256", "mono"))
    ap.add_argument("--bg", default="dark", choices=("dark", "light"))
    args = ap.parse_args()

    render_w, render_h = map(int, args.render_size.split("x"))

    viewer = ViewerAscii(
        output_mode=args.output_mode,
        ascii_cli_path=args.ascii_cli,
        fifo_path=args.fifo_path,
        tcp_host=args.tcp_host,
        tcp_port=args.tcp_port,
        render_width=render_w,
        render_height=render_h,
        sample_res=args.sample_res,
        color_mode=args.color_mode,
        bg=args.bg,
        camera_position=(4.5, 4.5, 3.0),
        camera_target=(0.0, 0.0, 0.4),
    )

    cube_verts, cube_faces = make_cube(1.0)
    plane_verts, plane_faces = make_plane(8.0)
    viewer.log_mesh("cube", cube_verts, cube_faces)
    viewer.log_mesh("plane", plane_verts, plane_faces)

    plane_xform = np.array([[0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)
    plane_scale = np.array([[1, 1, 1]], dtype=np.float32)
    plane_color = np.array([[0.22, 0.26, 0.32]], dtype=np.float32)

    dt = 1.0 / args.fps
    try:
        for frame in range(args.frames):
            if not viewer.is_running():
                break
            t = frame * dt
            q = quat_z(t * 0.8)
            cube_xform = np.array([[0, 0, 0.8, q[0], q[1], q[2], q[3]]], dtype=np.float32)
            cube_scale = np.array([[1, 1, 1]], dtype=np.float32)
            cube_color = np.array([[0.92, 0.58, 0.32]], dtype=np.float32)

            viewer.begin_frame(t)
            viewer.log_instances("plane-inst", "plane", plane_xform, plane_scale, plane_color, None)
            viewer.log_instances("cube-inst", "cube", cube_xform, cube_scale, cube_color, None)
            viewer.end_frame()
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
