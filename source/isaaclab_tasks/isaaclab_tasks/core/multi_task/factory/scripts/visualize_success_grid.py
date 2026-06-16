# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Preview the factory curriculum images with synthetic spread success / sampling values, no sim.

A fast eyeball for the curriculum image logger (:mod:`..viz.sampler_images`):
builds the board library + reset states via :class:`FactoryIKPipeline` (the same
Kit-free path as ``visualize_factory_retarget.py``), precomputes the static
top-down silhouettes, then renders both figures with random metrics -- the
per-board success + sampling-probability grid and the spawn->target tag matrix --
and writes them to PNGs. No training run required.

Run:
  S=source/isaaclab_tasks/isaaclab_tasks/core/multi_task/factory/scripts/visualize_success_grid.py
  ./isaaclab.sh -p $S presets=franka,nut_thread_m16
  # board count / rows-per-board / placement / FPS weights come from the resolved cfg;
  # change them via preset or hydra, e.g.
  #   presets=franka,nut_thread_m16 commands.reset_state.task_table.pipeline_cfg.board.num_boards=32
  # -> /tmp/factory_success_grid.png (grid) and _tags.png (tag matrix)
"""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

import torch

from isaaclab_tasks.core.multi_task.factory.retarget import FactoryIKPipeline
from isaaclab_tasks.core.multi_task.factory.retarget.cfg import resolve_from_task
from isaaclab_tasks.core.multi_task.factory.viz.geometry import build_success_grid_geometry
from isaaclab_tasks.core.multi_task.factory.viz.sampler_images import (
    _PROB_CMAP,
    _SUCCESS_COLORS,
    render_board_grids,
    render_tag_matrices,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Preview the factory success grid with random values (no sim).")
    # No cfg-shaping flags. Board count, rows/board, placement/grasp counts, and row-FPS
    # weights all come from the resolved production cfg. Override them via preset or hydra
    # dotted syntax (e.g. ``commands.reset_state.task_table.pipeline_cfg.board.num_boards=32``),
    # not custom argparse flags. Only rendering/output knobs live below.
    ap.add_argument("--max_states_per_board", type=int, default=None, help="Cap states drawn per cell (default: all).")
    ap.add_argument("--k", type=int, default=16, help="Silhouette polygon resolution (support directions).")
    ap.add_argument("--link_mode", choices=["outline", "fill", "off"], default="outline", help="Robot link rendering.")
    ap.add_argument("--link_alpha", type=float, default=0.3, help="Robot link silhouette opacity.")
    ap.add_argument("--nut_scale", type=float, default=2.5, help="Enlarge the nut marker by this factor.")
    ap.add_argument(
        "--draw_bound", action=argparse.BooleanOptionalAction, default=True, help="Overlay the oob in-bound box."
    )
    ap.add_argument(
        "--apply_bounds", action="store_true", help="Drop OOB states before rendering (preview the rejection fix)."
    )
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the random success/probability colors.")
    ap.add_argument("--out", type=str, default="/tmp/factory_success_grid", help="Output PNG path prefix.")

    from isaaclab_tasks.utils import setup_preset_cli

    args, remaining = setup_preset_cli(ap)
    if not any(tok.startswith("presets=") for tok in remaining):
        remaining = ["presets=franka,nut_thread_m16"] + remaining
        print("[viz] no presets= given, defaulting to presets=franka,nut_thread_m16")
    sys.argv = [sys.argv[0]] + remaining

    table_cfg = resolve_from_task()
    cfg = table_cfg.pipeline_cfg

    pipeline = FactoryIKPipeline(cfg)
    result = pipeline.build_balanced_table(table_cfg.rows_per_board * cfg.board.num_boards)
    print(f"\n{pipeline.rejection_summary}")
    n_states = int(result.joint_q.shape[0])
    print(f"[viz] {n_states} reset states across {cfg.board.num_boards} board configurations")

    # --- diagnostics the grid surfaces: OOB-at-start + nut-placement diversity ---
    nut_xyz = result.nut_pose[:, :3]
    board_idx = result.board_index
    # factory oob box (env-local), from FactoryTerminationsCfg success_terminate
    lo = torch.tensor([-0.0, -0.675, -0.05], device=nut_xyz.device)
    hi = torch.tensor([1.0, 0.675, 1.0], device=nut_xyz.device)
    oob = ((nut_xyz < lo) | (nut_xyz > hi)).any(dim=1)
    print(f"[diag] nut OOB at start vs in_bound_range x(0,1) y(-.675,.675) z(-.05,1): {int(oob.sum())}/{n_states}")
    for ai, axn in enumerate("xyz"):
        col = nut_xyz[:, ai]
        print(
            f"[diag]   nut {axn}: seen [{float(col.min()):.3f}, {float(col.max()):.3f}]  "
            f"bounds [{float(lo[ai]):.3f}, {float(hi[ai]):.3f}]  "
            f"{int((col < lo[ai]).sum())} below / {int((col > hi[ai]).sum())} above"
        )
    uniq = [
        int(torch.unique((nut_xyz[board_idx == b][:, :2] / 0.005).round().long(), dim=0).shape[0])
        for b in torch.unique(board_idx)
    ]
    ut = torch.tensor(uniq, dtype=torch.float32)
    print(
        f"[diag] distinct nut placements/board (5mm): min {int(ut.min())} median {int(ut.median())} max {int(ut.max())}"
        f"  |  rows/board ~ {n_states // len(uniq)} (rows = grasps x approaches x tags per placement)"
    )

    keep = ~oob if args.apply_bounds else torch.ones_like(oob)
    if args.apply_bounds:
        print(f"[diag] --apply_bounds: dropped {int(oob.sum())} OOB states -> {int(keep.sum())} kept")
    geom = build_success_grid_geometry(
        pipeline.model,
        result.joint_q[keep],
        result.nut_pose[keep],
        result.bolt_pose[keep],
        result.board_pose[keep],
        result.board_index[keep],
        k=args.k,
    )

    device = result.joint_q.device
    n_kept = int(geom["viz_link_polys"].shape[0])
    bound_xy = ((float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))) if args.draw_bound else None

    # pair states within each board (spawn x target) -- the real "problem" set, shared
    # by both figures so the grid's problem count + per-state aggregation and the tag
    # matrix are all faithful to training
    board_of = result.board_index[keep]
    sid = torch.arange(n_kept, device=device)
    spawn_chunks, target_chunks = [], []
    for b in torch.unique(board_of):
        bk = sid[board_of == b]
        spawn_chunks.append(bk.repeat_interleave(bk.numel()))
        target_chunks.append(bk.repeat(bk.numel()))
    spawn_index = torch.cat(spawn_chunks)
    target_index = torch.cat(target_chunks)
    n_slots = int(spawn_index.shape[0])

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    # Synthetic SPREAD (per-board gradient + jitter), not uniform random: per-cell averaging
    # collapses uniform random to ~0.5 (the colormap's pale middle), so nothing reads. A
    # gradient makes the full colormap range visible across the grid. Real success / sampling
    # mass come from training; this is only a layout + color-legibility preview.
    boards = torch.unique(board_of)
    rank = torch.searchsorted(boards, board_of).float() / max(1, boards.numel() - 1)  # [n_kept] in 0..1
    jit = ((torch.rand(2, board_of.shape[0], generator=gen) - 0.5) * 0.25).to(device)
    success = (rank + jit[0]).clamp(0.0, 1.0)[spawn_index]  # red -> green across boards
    prob = (1.0 - rank + jit[1]).clamp(0.0, 1.0)[spawn_index]  # opposite gradient, distinct panel
    prob = prob / prob.sum()

    grid_table = SimpleNamespace(spawn_index=spawn_index, target_index=target_index, **geom)
    tag_table = SimpleNamespace(
        state_tag_indices=result.tag[keep],
        state_tag_names=list(result.tag_names),
        spawn_index=spawn_index,
        target_index=target_index,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = render_board_grids(
        grid_table,
        [
            ("success rate", success, _SUCCESS_COLORS, 0.0, 1.0),
            ("sampling probability", prob, _PROB_CMAP, None, None),
        ],
        step=None,
        max_states_per_board=args.max_states_per_board,
        link_mode=args.link_mode,
        link_alpha=args.link_alpha,
        nut_scale=args.nut_scale,
        bound_xy=bound_xy,
        dpi=180,  # standalone inspection render: sharper than the periodic wandb log (dpi 70)
        cell_fill_alpha=0.55,  # tint each cell by its metric so the color reads at a glance
    )
    path = f"{args.out}.png"
    plt.imsave(path, image)
    print(f"[viz] wrote success+probability grid -> {path}  ({image.shape[1]}x{image.shape[0]}, {n_slots} problems)")

    matrix = render_tag_matrices(
        tag_table,
        [
            ("success rate", success, _SUCCESS_COLORS, "mean", 0.0, 1.0),
            ("sampling mass", prob, _PROB_CMAP, "sum", None, None),
        ],
    )
    mpath = f"{args.out}_tags.png"
    plt.imsave(mpath, matrix)
    print(f"[viz] wrote tag matrix -> {mpath}  ({matrix.shape[1]}x{matrix.shape[0]})")


if __name__ == "__main__":
    main()
