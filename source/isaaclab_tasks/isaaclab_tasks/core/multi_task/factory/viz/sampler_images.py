# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory success-grid image logger (the locomotion sampler-image analog).

One cell per board configuration, a fixed top-down view shared across all cells.
Each cell anchors on the board outline + bolt (light gray, recessed), then
overlays the board's stored reset states: the robot link silhouettes (thin
colored outlines, context) and the held nut drawn last -- filled, edged, and
slightly enlarged so the metric signal pops. The cell frame is tinted by the
board's mean. Several metrics (success rate, sampling probability) render as
side-by-side grids in one figure, each with a colorbar legend; the header reports
the board/state/slot totals.

For speed, each grid panel is drawn in a SINGLE axes -- every cell's polygons are
translated into a grid layout and batched into a few ``PolyCollection``s rather
than one matplotlib subplot per board. The geometry is static (precomputed at
table build, see :mod:`.geometry`); this module only recolors and draws, so it is
cheap to call periodically from the curriculum's ``sampler_visual_logger`` hook.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from ..mdp.reset_state_task_table import FactoryResetStateTaskTable

_GRID_TAG = "Sampler/factory_board_grid"
_MATRIX_TAG = "Sampler/factory_tag_matrix"
# structural backdrop (board plate, bolt, bound box): very light gray so it
# recedes and only the metric-colored nuts/links read as signal
_BOARD_EDGE = (0.82, 0.82, 0.82, 0.55)
_BOLT_FACE = (0.80, 0.80, 0.80, 0.45)
_BOLT_EDGE = (0.66, 0.66, 0.66, 0.7)
_NUT_EDGE = (0.2, 0.2, 0.2, 0.4)
_BOUND_EDGE = (0.72, 0.72, 0.72, 0.55)
# muted red -> sand -> sage (softer than RdYlGn for the success metric)
_SUCCESS_COLORS = [(0.78, 0.42, 0.38), (0.92, 0.86, 0.60), (0.45, 0.65, 0.45)]
_PROB_CMAP = "cividis"  # muted sequential


def wandb_log_image(tag: str, image: object) -> None:
    """Log an image to the active wandb run, if any (no-op otherwise)."""
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    wandb.log({tag: wandb.Image(image)}, step=int(getattr(wandb.run, "step", 0)))


def _resolve_cmap(spec):
    """Resolve a colormap from a name, a Colormap, or a list of anchor colors."""
    import matplotlib
    from matplotlib.colors import Colormap, LinearSegmentedColormap

    if isinstance(spec, Colormap):
        return spec
    if isinstance(spec, str):
        return matplotlib.colormaps[spec]
    return LinearSegmentedColormap.from_list("muted", list(spec))


def _per_state_value(
    values: torch.Tensor, spawn_index: torch.Tensor, target_index: torch.Tensor, n_states: int
) -> np.ndarray:
    """Per-state metric = mean of the per-slot value over slots touching the state (spawn or target)."""
    sums = torch.zeros(n_states, device=values.device)
    counts = torch.zeros(n_states, device=values.device)
    for idx in (spawn_index, target_index):
        sums.scatter_add_(0, idx, values)
        counts.scatter_add_(0, idx, torch.ones_like(values))
    return (sums / counts.clamp_min(1.0)).detach().cpu().numpy()


def _fig_to_rgb(fig) -> np.ndarray:
    """Rasterize a matplotlib figure to an ``[H, W, 3]`` uint8 array, then close it."""
    import matplotlib.pyplot as plt

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return img


def _cell_rank(cell_of: np.ndarray, n_states: int) -> np.ndarray:
    """Per-state 0-based rank within its cell (stable order), for capping states per cell."""
    order = np.argsort(cell_of, kind="stable")
    sorted_cells = cell_of[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_cells)) + 1]
    counts = np.diff(np.r_[starts, n_states])
    within = np.arange(n_states) - np.repeat(starts, counts)
    rank = np.empty(n_states, dtype=np.int64)
    rank[order] = within
    return rank


def render_board_grids(
    table: FactoryResetStateTaskTable,
    metrics: Sequence[tuple[str, torch.Tensor, object, float | None, float | None]],
    *,
    step: int | None = None,
    max_states_per_board: int | None = None,
    link_mode: str = "outline",
    link_alpha: float = 0.3,
    nut_scale: float = 2.5,
    bound_xy: tuple[tuple[float, float], tuple[float, float]] | None = None,
    dpi: int = 90,
    cell_fill_alpha: float = 0.0,
) -> np.ndarray | None:
    """Render one or more per-board metric grids side by side, or ``None`` if no geometry is stashed.

    Args:
        table: The factory reset-state table (must carry the ``viz_*`` stash).
        metrics: ``(label, per_slot_values, cmap, vmin, vmax)`` per side-by-side grid; ``cmap`` is a
            name / Colormap / anchor-color list, and ``vmin``/``vmax`` are ``None`` for auto range.
        step: Optional step counter for the title.
        max_states_per_board: Cap states drawn per cell (default: all).
        link_mode: Robot link rendering: ``"outline"`` (default), ``"fill"``, or ``"off"``.
        link_alpha: Opacity for the robot link silhouettes.
        nut_scale: Enlarge the nut polygon by this factor so the signal is visible.
        bound_xy: Optional ``((xmin, xmax), (ymin, ymax))`` box drawn per cell (the ``oob`` AABB).
        dpi: Figure dpi (lower is faster to rasterize).
    """
    link = getattr(table, "viz_link_polys", None)
    if link is None:
        return None
    nut, board, bolt = table.viz_nut_polys, table.viz_board_polys, table.viz_bolt_polys
    cell_of = np.asarray(table.viz_cell_of_state)
    n_boards = int(table.viz_n_boards)
    n_bodies, k = link.shape[1], link.shape[2]
    n_states = link.shape[0]
    n_slots = int(table.spawn_index.shape[0])

    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import Normalize

    # tight shared extent across cells (the geometry cluster; a drawn bound box is clipped to it)
    allp = np.concatenate([link.reshape(-1, 2), nut.reshape(-1, 2), board.reshape(-1, 2), bolt.reshape(-1, 2)], axis=0)
    lo, hi = allp.min(0), allp.max(0)
    pad = 0.04 * (hi - lo + 1e-6)
    x0, x1 = float(lo[0] - pad[0]), float(hi[0] + pad[0])
    y0, y1 = float(lo[1] - pad[1]), float(hi[1] + pad[1])
    span_x, span_y = x1 - x0, y1 - y0
    gap = 0.05 * max(span_x, span_y)
    pitch_x, pitch_y = span_x + gap, span_y + gap

    ncols = int(math.ceil(math.sqrt(n_boards)))
    nrows = int(math.ceil(n_boards / ncols))
    col_c = np.arange(n_boards) % ncols
    row_c = np.arange(n_boards) // ncols
    # per-cell translation: place each cell's data coords into its grid box (row 0 at top)
    off = np.stack([col_c * pitch_x - x0, (nrows - 1 - row_c) * pitch_y - y0], axis=1).astype(np.float32)

    keep = _cell_rank(cell_of, n_states) < max_states_per_board if max_states_per_board else np.ones(n_states, bool)
    state_off = off[cell_of[keep]]  # [Nk, 2]
    link_t = (link[keep] + state_off[:, None, None, :]).reshape(-1, k, 2)  # [Nk*nb, k, 2]
    ncen = nut[keep].mean(axis=1, keepdims=True)
    nut_t = ncen + nut_scale * (nut[keep] - ncen) + state_off[:, None, :]  # [Nk, k, 2]
    board_t = board + off[:, None, :]
    bolt_t = bolt + off[:, None, :]
    rect_corners = np.array([[0, 0], [span_x, 0], [span_x, span_y], [0, span_y]], np.float32)
    rects = np.stack([col_c * pitch_x, (nrows - 1 - row_c) * pitch_y], axis=1)[:, None, :] + rect_corners[None]
    if bound_xy is not None:
        (bx0, bx1), (by0, by1) = bound_xy
        box_corners = np.array([[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]], np.float32)
        box_t = box_corners[None] + off[:, None, :]
    xlim = (-0.3 * gap, ncols * pitch_x - gap + 0.3 * gap)
    ylim = (-0.3 * gap, nrows * pitch_y - gap + 0.3 * gap)
    states_by_cell = [np.where(cell_of == c)[0] for c in range(n_boards)]

    n_metrics = len(metrics)
    panel_w = ncols * 1.1 * (pitch_x / pitch_y)
    fig = plt.figure(figsize=(n_metrics * panel_w + 0.2 * n_metrics, nrows * 1.1 + 0.7), dpi=dpi)
    subfigs = fig.subfigures(1, n_metrics, squeeze=False, wspace=0.02)[0]
    for (label, values, cmap_spec, vmin, vmax), sub in zip(metrics, subfigs):
        cmap = _resolve_cmap(cmap_spec)
        rate = _per_state_value(values, table.spawn_index, table.target_index, n_states)
        vlo = float(rate.min()) if vmin is None else float(vmin)
        vhi = float(rate.max()) if vmax is None else float(vmax)
        if vhi <= vlo:
            vhi = vlo + 1e-6
        norm = Normalize(vlo, vhi)
        colors = cmap(norm(rate))[keep]

        ax = sub.add_axes((0.01, 0.10, 0.98, 0.86))
        cell_mean = np.array([rate[s].mean() if s.size else vlo for s in states_by_cell])
        cell_colors = cmap(norm(cell_mean))
        # translucent per-cell metric fill drawn first (underneath the detail) so the color
        # reads at a glance instead of only on the thin border + small nut markers
        if cell_fill_alpha > 0.0:
            cell_face = cell_colors.copy()
            cell_face[:, 3] = cell_fill_alpha
            ax.add_collection(PolyCollection(rects, facecolors=cell_face, edgecolors="none"))
        ax.add_collection(PolyCollection(board_t, facecolors="none", edgecolors=[_BOARD_EDGE], linewidths=0.5))
        ax.add_collection(PolyCollection(bolt_t, facecolors=[_BOLT_FACE], edgecolors=[_BOLT_EDGE], linewidths=0.4))
        if bound_xy is not None:
            ax.add_collection(
                PolyCollection(box_t, facecolors="none", edgecolors=[_BOUND_EDGE], linewidths=0.5, linestyles="--")
            )
        if link_mode != "off":
            lcols = np.repeat(colors, n_bodies, axis=0).copy()
            lcols[:, 3] = link_alpha
            if link_mode == "outline":
                ax.add_collection(PolyCollection(link_t, facecolors="none", edgecolors=lcols, linewidths=0.3))
            else:
                ax.add_collection(PolyCollection(link_t, facecolors=lcols, edgecolors="none"))
        ncols_rgba = colors.copy()
        ncols_rgba[:, 3] = 0.95
        ax.add_collection(PolyCollection(nut_t, facecolors=ncols_rgba, edgecolors=[_NUT_EDGE], linewidths=0.3))
        ax.add_collection(PolyCollection(rects, facecolors="none", edgecolors=cell_colors, linewidths=1.1))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.axis("off")

        cax = sub.add_axes((0.18, 0.05, 0.64, 0.025))
        cbar = sub.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(f"{label}  (mean {float(rate.mean()):.3g})", fontsize=9)

    title = f"Factory board grid  —  {n_boards} boards · {n_states} reset states · {n_slots} problems (spawn x target)"
    if step is not None:
        title += f"  (step {step})"
    fig.suptitle(title, fontsize=12)
    return _fig_to_rgb(fig)


def render_tag_matrices(
    table: FactoryResetStateTaskTable,
    metrics: Sequence[tuple[str, torch.Tensor, object, str, float | None, float | None]],
    *,
    step: int | None = None,
    annotate: bool = True,
) -> np.ndarray | None:
    """Render spawn-tag x target-tag matrices, or ``None`` if the table has no tags.

    Each cell (spawn tag i -> target tag j) aggregates a per-slot metric over the
    task slots with that spawn/target tag pair: success rate (which transitions
    are hardest/easiest) and sampling mass (where the curriculum spends effort).
    Tags that never appear (as spawn or target) are dropped; remaining empty cells
    are blanked.

    Args:
        table: The factory reset-state table (needs ``state_tag_indices`` /
            ``state_tag_names`` / ``spawn_index`` / ``target_index``).
        metrics: ``(label, per_slot_values, cmap, reduce, vmin, vmax)`` per panel, where
            ``reduce`` is ``"mean"`` (success) or ``"sum"`` (mass), and ``vmin``/``vmax``
            are ``None`` for auto range.
        step: Optional step counter for the title.
        annotate: Write each cell's value (text color auto-contrasted).
    """
    tag_names_all = list(table.state_tag_names)
    if not tag_names_all:
        return None
    n_all = len(tag_names_all)
    tag_idx = table.state_tag_indices.long()
    spawn_tag = tag_idx[table.spawn_index]
    target_tag = tag_idx[table.target_index]
    flat = spawn_tag * n_all + target_tag
    # drop tags that never appear as a spawn or a target (enumerated-but-unused
    # taxonomy entries, e.g. reaching_in_air) so the matrix has no dead row/column
    present = sorted(torch.unique(torch.cat([spawn_tag, target_tag])).cpu().tolist())
    if not present:
        return None
    sel = np.ix_(present, present)
    tag_names = [tag_names_all[t] for t in present]
    n_tags = len(present)

    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    counts = torch.zeros(n_all * n_all, device=flat.device)
    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
    empty = (counts.reshape(n_all, n_all).cpu().numpy() == 0)[sel]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(
        1, n_metrics, figsize=(n_metrics * (0.42 * n_tags + 2.4), 0.42 * n_tags + 2.8), dpi=120, squeeze=False
    )
    for (label, values, cmap_spec, reduce, vmin, vmax), ax in zip(metrics, axes[0]):
        cmap = _resolve_cmap(cmap_spec).copy()
        cmap.set_bad((0.92, 0.92, 0.92, 1.0))
        sums = torch.zeros(n_all * n_all, device=values.device)
        sums.scatter_add_(0, flat, values)
        if reduce == "mean":
            cnt = torch.zeros_like(sums)
            cnt.scatter_add_(0, flat, torch.ones_like(values))
            mat = (sums / cnt.clamp_min(1.0)).reshape(n_all, n_all).cpu().numpy()[sel]
        else:
            mat = sums.reshape(n_all, n_all).cpu().numpy()[sel]
        masked = np.ma.array(mat, mask=empty)
        vlo = float(masked.min()) if vmin is None else float(vmin)
        vhi = float(masked.max()) if vmax is None else float(vmax)
        if vhi <= vlo:
            vhi = vlo + 1e-6
        norm = Normalize(vlo, vhi)
        ax.imshow(masked, cmap=cmap, norm=norm, aspect="equal")
        ax.set_xticks(range(n_tags))
        ax.set_xticklabels(tag_names, rotation=55, ha="right", fontsize=7)
        ax.set_yticks(range(n_tags))
        ax.set_yticklabels(tag_names, fontsize=7)
        ax.set_xlabel("target tag", fontsize=9)
        ax.set_ylabel("spawn tag", fontsize=9)
        ax.set_title(f"{label}  (mean {float(masked.mean()):.3g})", fontsize=10)
        fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.03)
        if annotate:
            for i in range(n_tags):
                for j in range(n_tags):
                    if empty[i, j]:
                        continue
                    r, g, b, _ = cmap(norm(mat[i, j]))
                    tc = "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.55 else "white"
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5, color=tc)

    title = "Factory spawn->target tag matrix"
    if step is not None:
        title += f"  (step {step})"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _fig_to_rgb(fig)


def render_board_grid(
    table: FactoryResetStateTaskTable,
    success_rates: torch.Tensor,
    *,
    step: int | None = None,
    max_states_per_board: int | None = None,
    link_mode: str = "outline",
    link_alpha: float = 0.3,
    nut_scale: float = 2.5,
    bound_xy: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> np.ndarray | None:
    """Render a single success grid (thin wrapper over :func:`render_board_grids`)."""
    return render_board_grids(
        table,
        [("success rate", success_rates, _SUCCESS_COLORS, 0.0, 1.0)],
        step=step,
        max_states_per_board=max_states_per_board,
        link_mode=link_mode,
        link_alpha=link_alpha,
        nut_scale=nut_scale,
        bound_xy=bound_xy,
    )


def log_factory_board_grid(env, sampler, success_rates: torch.Tensor, probs: torch.Tensor) -> None:
    """``sampler_visual_logger`` hook: log the per-board grid + the spawn->target tag matrix (extras + wandb)."""
    table = env.command_manager.get_term("reset_state").table
    step = int(getattr(env, "common_step_counter", 0))
    images = {
        _GRID_TAG: render_board_grids(
            table,
            [
                ("success rate", success_rates, _SUCCESS_COLORS, 0.0, 1.0),
                ("sampling probability", probs, _PROB_CMAP, None, None),
            ],
            step=step,
            # rasterization-bound, so keep the periodic log quick by drawing fewer states
            # (max_states_per_board=8). A translucent per-cell fill makes the success /
            # sampling color readable at a glance instead of only the thin border + markers;
            # the standalone script renders all states at higher dpi for detailed inspection.
            max_states_per_board=8,
            dpi=110,
            cell_fill_alpha=0.55,
        ),
        _MATRIX_TAG: render_tag_matrices(
            table,
            [
                ("success rate", success_rates, _SUCCESS_COLORS, "mean", 0.0, 1.0),
                ("sampling mass", probs, _PROB_CMAP, "sum", None, None),
            ],
            step=step,
        ),
    }
    for tag, image in images.items():
        if image is None:
            continue
        env.extras.setdefault("log_images", {})[tag] = image
        wandb_log_image(tag, image)
