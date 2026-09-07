# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain-conforming spawn validation driven by the production env cfg.

Constructs :class:`LocomotionPositionCommandEnvCfg`, resolves presets the
same way the train script does, and reuses *only* two things from the
resolved env cfg:

* :attr:`env.scene.robot` — articulation cfg (USD path, init state, default
  joint pose) that drives the IK kinematics.
* :attr:`env.commands.goal_point.task_table` — the same retarget pipeline + sampler
  + criteria the training task uses, so this tool reflects production
  behaviour exactly. ``goal_point.task_table.pool_spacing`` also doubles as the
  default density.

The terrain mesh is built from ``env.scene.terrain.terrain_generator`` —
whichever sub-terrains the selected presets resolved to.

Usage::

    SCRIPT=source/isaaclab_tasks/isaaclab_tasks/manager_based/multi_task/terrain/utils/tools/validate_spawn_points.py

    # ANYmal-C on the default terrain mix.
    ./isaaclab.sh -p $SCRIPT presets=anymal_c

    # Stepping-stone-only terrain with ANYmal-C.
    ./isaaclab.sh -p $SCRIPT presets=stepping_stone,anymal_c

    # Eval mix with Go2.
    ./isaaclab.sh -p $SCRIPT presets=eval,go2

    # Compose with the FPS-feature presets too.
    ./isaaclab.sh -p $SCRIPT presets=anymal_c,xyzyaw

    # Override placement density (otherwise uses goal_point.task_table.pool_spacing).
    ./isaaclab.sh -p $SCRIPT presets=anymal_c --max_robots 200
    ./isaaclab.sh -p $SCRIPT presets=anymal_c --spacing 0.4

    # Headless diagnostic — no viser.
    ./isaaclab.sh -p $SCRIPT presets=anymal_c --no_viewer
"""

from __future__ import annotations

import sys

sys.path[:] = [p for p in sys.path if "pip_prebundle" not in p and "pip_archive" not in p]

import argparse
import builtins
import importlib
import socket
import time

import numpy as np
import torch
import warp as wp

from isaaclab.utils.warp import convert_to_warp_mesh

VISER_PORT = 8765


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def _set_registration_guard() -> None:
    """Stop ``isaaclab_tasks/__init__`` from eagerly importing every task package.

    A full ``import isaaclab_tasks`` pulls in Isaac-Sim-coupled task modules at
    import time. Setting this guard (stashed on ``builtins`` so it survives the
    sys.modules churn in ``AppLauncher``) lets us import only the position
    config packages we actually need.
    """
    builtins._isaaclab_tasks_registered = True  # type: ignore[attr-defined]


def _register_position_tasks() -> None:
    """Register the position-family gym task ids without booting Isaac Sim.

    Each config package calls ``gym.register`` with lazy *string* entry points,
    so the env cfg module is only imported when :func:`resolve_task_config`
    resolves it. This keeps ``--task`` working for both the preset-driven
    ``Isaac-Position-v0`` and the per-robot ``Isaac-Position-<Robot>-v0`` tasks
    without the Isaac-Sim-coupled side effects of a full ``import isaaclab_tasks``.
    """
    _set_registration_guard()
    candidates = ("isaaclab_tasks.core.multi_task.terrain.config",)
    for module_name in candidates:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - dev tool: surface and continue
            print(f"[validate_spawn_points] WARN: skipped registering '{module_name}': {exc}")


def _eager_load_presets() -> None:
    """Force-load preset packages so PresetCfg class attributes are registered.

    Per-robot modules (``anymal_c.py`` etc.) and per-terrain entries are
    populated at import time. We import a handful of leaf modules eagerly so
    that ``resolve_presets`` finds ``RobotArticulationCfg.<robot>`` and
    friends as class attributes.
    """
    importlib.import_module("isaaclab_tasks.core.multi_task.terrain.terrains")
    importlib.import_module("isaaclab_tasks.core.multi_task.terrain.mdp_presets.robots")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_robot_usd(robot_cfg) -> str:
    """Return the robot's description path (USD or URDF), downloading from Nucleus if needed."""
    from isaaclab.utils.assets import check_file_path, retrieve_file_path

    from isaaclab_tasks.core.multi_task.terrain.retarget import resolve_description_path

    usd_path = resolve_description_path(robot_cfg.spawn)
    status = check_file_path(usd_path)
    if status == 0:
        raise FileNotFoundError(
            f"Robot description not found: {usd_path}\n"
            "If this is a fetched bundle, run: ./isaaclab.sh -p scripts/fetch_robot_descriptions.py"
        )
    if status == 2:
        usd_path = retrieve_file_path(usd_path, force_download=False)
    return usd_path


def _patch_kin_with_robot(pipeline_cfg, robot_cfg, robot_usd: str, device: str):
    """Return ``pipeline_cfg`` with ``kin`` populated from the robot articulation cfg."""
    new_kin = pipeline_cfg.kin.copy()
    new_kin.usd_path = robot_usd
    new_kin.device = device
    new_kin.default_pos = (0.0, 0.0, robot_cfg.init_state.pos[2])
    new_kin.default_joint_pos = robot_cfg.init_state.joint_pos
    return pipeline_cfg.replace(asset_cfg=None, kin=new_kin)


def _patch_sampler_bounds(
    pipeline_cfg,
    sampler_x_range: tuple[float, float],
    sampler_y_range: tuple[float, float],
):
    """Clip the morph-patch sampler's ``x_range``/``y_range`` to the active tile region.

    The default pipeline cfg leaves ``x_range`` / ``y_range`` at ``±1e6`` so
    the patch sampler scans the *full* mesh — including the 20 m flat border
    around the tile grid. That floods the morph-patch pool with border
    patches and makes polygon centers in the inner active region fail
    foot-reachability against them. Production avoids this via
    :func:`task_table_builder._pipeline_with_inner_sampling_bounds`; we
    mirror that here so the validate tool reflects training behaviour.
    """
    sampler_cfg = pipeline_cfg.sampler
    patch_cfg = getattr(sampler_cfg, "patch", None)
    if patch_cfg is None:
        return pipeline_cfg
    return pipeline_cfg.replace(
        sampler=sampler_cfg.replace(patch=patch_cfg.replace(x_range=sampler_x_range, y_range=sampler_y_range))
    )


def _terrain_grid_extents(terrain_gen_cfg) -> tuple[tuple[float, float], tuple[float, float]]:
    """Inner (non-border) sampling extents in the mesh frame."""
    inner_x = terrain_gen_cfg.num_rows * terrain_gen_cfg.size[0] / 2.0
    inner_y = terrain_gen_cfg.num_cols * terrain_gen_cfg.size[1] / 2.0
    return (-inner_x, inner_x), (-inner_y, inner_y)


def _feature_extra_dim_and_vol(extractor) -> tuple[int, float]:
    """Return ``(extra_dim, extra_vol_factor)`` from the extractor's a-priori claim.

    ``extra_dim`` is the number of feature axes beyond xyz (z treated as
    ignored — terrain xy is a 2-D manifold). ``extra_vol_factor`` is the
    bbox volume those extra axes contribute, in the same units the extractor
    produces. Falls back to ``(0, 1.0)`` for raw callables or extractors
    that don't expose :meth:`feature_volume_contribution`.
    """
    if extractor is None:
        return 0, 1.0
    fn = getattr(extractor, "feature_volume_contribution", None)
    if fn is None:
        return 0, 1.0
    extra_dim, extra_vol = fn()
    return int(extra_dim), float(extra_vol)


def _derive_n_desired(
    args: argparse.Namespace,
    pool_spacing_default: float,
    sampler_x_range: tuple[float, float],
    sampler_y_range: tuple[float, float],
    extractor,
) -> int:
    """Resolve the budget ``n_desired`` from ``--max_robots`` / ``--spacing`` / cfg default.

    Generalises the original ``area / (3 × spacing²)`` heuristic to richer
    feature spaces: when the final-FPS feature extractor adds extra axes
    (yaw, rotation, joints), the budget grows so the upstream sampler
    produces enough survivors to actually fill the larger metric volume.
    The ``× 1/3`` factor still accounts for the empirical ~33% criteria
    yield.
    """
    if args.max_robots is not None:
        return args.max_robots
    spacing = args.spacing if args.spacing is not None else pool_spacing_default
    area = max((sampler_x_range[1] - sampler_x_range[0]) * (sampler_y_range[1] - sampler_y_range[0]), 0.0)
    extra_dim, extra_vol = _feature_extra_dim_and_vol(extractor)
    vol = area * extra_vol
    d_eff = 2 + extra_dim
    return max(1, int(vol / (3.0 * spacing**d_eff)))


def _gravity_weight_from_cfg(pipeline_cfg) -> float | None:
    """Return the configured gravity-torque objective weight, if present."""
    for objective_cfg in pipeline_cfg.extra_objectives:
        if type(objective_cfg).__name__ == "IKObjectiveGravityTorqueCfg":
            return objective_cfg.weight
    return None


def _draw_active_support_overlay(viewer, buf, feet_all, passed_idx, nc, device) -> None:
    """Draw a thick polygon of only the contact feet per selected placement.

    Distinct from the green reach polygon (which shows all ``nc`` sampled
    positions). The overlay traces the actual support region used by the
    stability criterion: a triangle for tripod placements, a segment for
    biped, a quad for 4-contact.
    """
    is_contact_np = buf.is_contact_t.cpu().numpy().reshape(-1, nc)
    s_list: list[np.ndarray] = []
    e_list: list[np.ndarray] = []
    n_quad = n_tri = n_bi = 0
    for idx in passed_idx:
        feet = feet_all[idx]
        mask = is_contact_np[idx].astype(bool)
        active = feet[mask].copy()
        if len(active) < 2:
            continue
        active[..., 2] += 0.03
        if len(active) == 2:
            s_list.append(active[0:1])
            e_list.append(active[1:2])
            n_bi += 1
            continue
        centroid = active[..., :2].mean(axis=0, keepdims=True)
        delta = active[..., :2] - centroid
        order = np.argsort(np.arctan2(delta[..., 1], delta[..., 0]))
        ordered = active[order]
        s_list.append(ordered)
        e_list.append(np.roll(ordered, -1, axis=0))
        if len(active) == 3:
            n_tri += 1
        else:
            n_quad += 1
    if not s_list:
        return
    s_arr = np.concatenate(s_list, axis=0)
    e_arr = np.concatenate(e_list, axis=0)
    viewer.log_lines(
        "polygons_support_active",
        wp.array(s_arr.tolist(), dtype=wp.vec3, device=device),
        wp.array(e_arr.tolist(), dtype=wp.vec3, device=device),
        colors=(1.0, 0.95, 0.2),
        width=0.012,
    )
    print(f"  Active-contact support polygons (yellow, thick): quad={n_quad} tri={n_tri} biped={n_bi}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import newton
    from newton.viewer import ViewerViser

    parser = argparse.ArgumentParser(
        description="Validate spawn points using the production env cfg (preset-driven).",
    )
    parser.add_argument("--task", type=str, default="Isaac-Position-v0", help="Gym task id.")
    parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry-point key.")
    density_group = parser.add_mutually_exclusive_group()
    density_group.add_argument(
        "--max_robots",
        type=int,
        default=None,
        help="Number of final placements. Overrides goal_point.task_table.pool_spacing.",
    )
    density_group.add_argument(
        "--spacing",
        type=float,
        default=None,
        help="Target placement spacing [m]. Defaults to goal_point.task_table.pool_spacing from the resolved cfg.",
    )
    parser.add_argument(
        "--no_viewer",
        action="store_true",
        help="Skip the viser viewer; print diagnostics and exit.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=VISER_PORT,
        help=f"Port for the viser web server (default {VISER_PORT}).",
    )
    parser.add_argument(
        "--polygon_limit",
        type=int,
        default=3000,
        help=(
            "Maximum support polygons drawn per outcome group. Each polygon is a handful of line"
            " segments, so a full terrain's worth swamps the browser; an evenly-spread subset shows"
            " the same distribution. 0 draws all."
        ),
    )
    parser.add_argument(
        "--show_collision",
        action="store_true",
        help=(
            "Draw each robot's collision geometry instead of its visual meshes, which is what the IK"
            " criteria and the terrain-collision objective actually reason about."
        ),
    )
    parser.add_argument(
        "--vis_limit",
        type=int,
        default=200,
        help=(
            "Maximum articulated robots drawn in the viewer. Each one costs a USD load, so drawing every"
            " placement is slow. Foot points and support polygons are always drawn in full. 0 draws all."
        ),
    )
    # Set the registration guard BEFORE importing any ``isaaclab_tasks`` submodule, so the
    # package __init__ does not eagerly import every (Isaac-Sim-coupled) task package.
    _set_registration_guard()

    from isaaclab_tasks.utils import setup_preset_cli

    args, remaining = setup_preset_cli(parser)
    # Hand the leftover preset / Hydra-override tokens to ``resolve_task_config`` through
    # ``sys.argv``, mirroring scripts/reinforcement_learning/rsl_rl/train.py.
    sys.argv = [sys.argv[0]] + remaining
    selected = {
        name
        for token in sys.argv[1:]
        if token.startswith("presets=")
        for name in token.split("=", 1)[1].split(",")
        if name
    }

    # Register the position task ids (lazy string entry points) and populate preset
    # tables, then resolve the requested ``--task`` env cfg WITHOUT booting Isaac Sim.
    _register_position_tasks()
    _eager_load_presets()

    from isaaclab_tasks.core.multi_task.terrain.retarget import RetargetPipeline, apply_final_fps
    from isaaclab_tasks.utils.hydra import resolve_task_config

    env_cfg, _ = resolve_task_config(args.task, args.agent)

    device = "cuda:0"

    # --- Robot ---
    robot_cfg = env_cfg.scene.robot
    if robot_cfg is None or not hasattr(robot_cfg, "spawn"):
        raise ValueError(
            "env.scene.robot was not resolved — did you forget a robot preset? "
            "Try: presets=anymal_c (or go2 / spot / h1 / b2 / mewtwo)."
        )
    robot_usd = _resolve_robot_usd(robot_cfg)

    # --- Terrain ---
    terrain_gen_cfg = env_cfg.scene.terrain.terrain_generator
    sub_terrain_names = list(terrain_gen_cfg.sub_terrains.keys())

    from isaaclab.terrains.terrain_generator import TerrainGenerator

    gen = TerrainGenerator(cfg=terrain_gen_cfg, device=device)
    mesh = gen.terrain_mesh
    origin = np.zeros(3, dtype=np.float32)
    sampler_x_range, sampler_y_range = _terrain_grid_extents(terrain_gen_cfg)
    wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)

    # --- Pipeline (from env's commands.goal_point) ---
    goal_cfg = env_cfg.commands.goal_point
    table_cfg = goal_cfg.task_table
    pipeline_cfg = _patch_kin_with_robot(table_cfg.pipeline_cfg, robot_cfg, robot_usd, device)
    pipeline_cfg = _patch_sampler_bounds(pipeline_cfg, sampler_x_range, sampler_y_range)

    # --- Density (feature-aware) ---
    spacing = args.spacing if args.spacing is not None else table_cfg.pool_spacing
    extractor = pipeline_cfg.sampler.sizing.fps_features
    extra_dim, extra_vol = _feature_extra_dim_and_vol(extractor)
    n_desired = _derive_n_desired(args, table_cfg.pool_spacing, sampler_x_range, sampler_y_range, extractor)

    # Spacing-driven post-IK count: pipeline derives ``k_target`` from the actual
    # feature-space bbox of survivors at this spacing, so adding richer features
    # (yaw, rotation, joints) automatically scales the final placement count up
    # rather than capping at a pre-computed ``n_desired``.
    pipeline_cfg = pipeline_cfg.replace(
        sampler=pipeline_cfg.sampler.replace(
            sizing=pipeline_cfg.sampler.sizing.replace(fps_spacing=spacing),
        )
    )

    print(f"Presets : {sorted(selected) or '(none — using defaults)'}")
    print(f"Robot   : usd={robot_usd}")
    print(
        f"Terrain : {len(sub_terrain_names)} sub-terrain(s) × {terrain_gen_cfg.num_rows}×{terrain_gen_cfg.num_cols}"
        f" tiles → {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces"
    )
    print(f"  sub_terrains: {sub_terrain_names}")
    print(f"  sampler ranges: x={sampler_x_range}, y={sampler_y_range}")
    print(
        f"Pipeline: min_contacts={pipeline_cfg.sampler.min_contacts}"
        f" terrain_snap_distance={pipeline_cfg.sampler.terrain_snap_distance}"
        f" pool_spacing={table_cfg.pool_spacing}"
    )
    print(
        f"IK wts  : base_rot={pipeline_cfg.base_rot_weight}"
        f" base_pos={pipeline_cfg.base_pos_weight}"
        f" gravity={_gravity_weight_from_cfg(pipeline_cfg)}"
    )
    extractor_name = type(extractor).__name__ if extractor is not None else "xyz (default)"
    print(
        f"Features: extractor={extractor_name} extra_dim={extra_dim} extra_vol={extra_vol:.3f}"
        f" → spacing={spacing:.3f} drives both pre-IK budget and post-IK FPS"
    )
    print(f"Density : n_desired (budget)={n_desired} (max_robots={args.max_robots} spacing={args.spacing})")

    print("\n--- Running retarget pipeline ---")
    t0 = time.time()
    pipeline = RetargetPipeline(pipeline_cfg)
    kin = pipeline.kin
    foot_ids = pipeline.foot_body_ids
    foot_names = pipeline.foot_body_names

    geom = kin.foot_geometry(foot_ids)
    foot_offsets = geom["foot_offsets"]
    standing_height = geom["standing_height"]
    print(f"  Bodies={kin.model.body_count} Joints={kin.model.joint_count}")
    print(f"  Feet: {foot_names} -> ids {foot_ids}")
    foot_spread = np.linalg.norm(foot_offsets[:, :2].max(axis=0) - foot_offsets[:, :2].min(axis=0))
    print(f"  standing_height={standing_height:.3f}m foot_spread={foot_spread:.3f}m")

    buf = pipeline.run(wp_mesh, origin, n_desired=n_desired)
    sizing = pipeline_cfg.sampler.sizing
    apply_final_fps(
        buf,
        n_desired=n_desired,
        extractor=getattr(sizing, "fps_features", None),
        spacing=getattr(sizing, "fps_spacing", None),
    )
    t_total = time.time() - t0

    print(pipeline.rejection_summary)
    chunk_profile = pipeline.chunk_profile_summary
    if chunk_profile:
        print()
        print(chunk_profile)
    print(f"  Wall time: {t_total:.2f}s")

    if buf.num_selected == 0:
        print("No valid candidates. Exiting.")
        sys.exit(1)

    sel_idx = buf._selected[: buf.num_selected].to(torch.long)
    nc = len(foot_ids)
    is_c_sel = buf.is_contact_t.view(-1, nc)[sel_idx]
    n_active_sel = is_c_sel.to(torch.int32).sum(dim=-1).cpu().tolist()
    nc_hist: dict[int, int] = {}
    for k in n_active_sel:
        nc_hist[int(k)] = nc_hist.get(int(k), 0) + 1
    print(f"  Selected placement contact-count histogram: {dict(sorted(nc_hist.items()))}")

    if args.no_viewer:
        print("\n--no_viewer: exiting without visualization.")
        return

    # --- Visualization ---
    print("\n--- Visualization ---")
    sel = buf._selected[: buf.num_selected].cpu().numpy()
    jq_results = buf.joint_q_result_t.cpu().numpy()
    ct_np = buf.contact_targets_t.cpu().numpy()
    cpc = kin.model.joint_coord_count

    # Drawing an articulated robot costs one USD load each, so cap how many are
    # rendered. Foot points and support polygons below stay complete -- they are
    # the actual distribution and cost nothing to draw.
    vis_sel = sel
    if args.vis_limit > 0 and len(sel) > args.vis_limit:
        vis_sel = sel[np.linspace(0, len(sel) - 1, args.vis_limit).astype(np.int64)]
        print(f"  Drawing {len(vis_sel)} of {len(sel)} placements (--vis_limit); polygons show all.")

    solved_qs = [jq_results[idx] for idx in vis_sel]
    selected_feet = [ct_np[idx * nc : (idx + 1) * nc] for idx in sel]

    vis_builder = newton.ModelBuilder()
    vis_builder.add_shape_mesh(
        body=-1,
        mesh=newton.Mesh(
            vertices=mesh.vertices.flatten().tolist(),
            indices=mesh.faces.flatten().tolist(),
        ),
        scale=wp.vec3(1, 1, 1),
    )

    if solved_qs:
        from isaaclab_tasks.core.multi_task.kinematics import add_robot_description

        template = newton.ModelBuilder()
        add_robot_description(template, robot_usd, collapse_fixed_joints=False, show_colliders=args.show_collision)
        for _ in solved_qs:
            vis_builder.add_world(template)

    vis_model = vis_builder.finalize(device=device)
    if solved_qs:
        vis_jq = vis_model.joint_q.numpy().copy()
        for i, sq in enumerate(solved_qs):
            vis_jq[i * cpc : (i + 1) * cpc] = sq
        vis_model.joint_q = wp.array(vis_jq, dtype=float, device=device)

    vis_state = vis_model.state()
    newton.eval_fk(
        vis_model,
        vis_model.joint_q,
        wp.zeros(vis_model.joint_dof_count, dtype=float, device=device),
        vis_state,
    )

    viewer = ViewerViser(port=args.port)
    viewer.set_model(vis_model)
    viewer.set_world_offsets((0.0, 0.0, 0.0))

    from isaaclab_tasks.core.multi_task.terrain.terrains.patch_sampling.cfg import (
        CircleFootprintCfg,
        MorphologicalPatchSamplingCfg,
    )

    # Conservative candidate density (~10 patches/m^2) so even low-valid-fraction
    # terrains (e.g. FLOATING_ISLAND at ~26% valid cells) comfortably satisfy
    # the request and no fallback is needed.
    vis_area = (sampler_x_range[1] - sampler_x_range[0]) * (sampler_y_range[1] - sampler_y_range[0])
    vis_num_patches = max(100, min(10000, int(vis_area * 10)))

    fc_cfg = MorphologicalPatchSamplingCfg(
        num_patches=vis_num_patches,
        footprint=CircleFootprintCfg(radius=0.04),
        max_height_diff=0.03,
        horizontal_scale=0.03,
        oversample_ratio=3.0,
        x_range=sampler_x_range,
        y_range=sampler_y_range,
    )

    fp = fc_cfg.func(wp_mesh, origin, fc_cfg)
    origin_t = torch.tensor(origin, dtype=torch.float, device=fp.device)
    fp[:, :3] += origin_t
    foot_pts = fp[:, :3].cpu().numpy()
    fl = foot_pts.copy()
    fl[:, 2] += 0.02
    viewer.log_points(
        "foot_candidates",
        wp.array(fl.tolist(), dtype=wp.vec3, device=device),
        radii=0.01,
        colors=(0.15, 0.5, 0.15),
    )

    if selected_feet:
        sf = np.concatenate(selected_feet)
        sf_l = sf.copy()
        if sf_l.ndim == 2:
            sf_l[:, 2] += 0.02
        viewer.log_points(
            "selected_feet",
            wp.array(sf_l.tolist(), dtype=wp.vec3, device=device),
            radii=0.03,
            colors=(0, 1, 1),
        )

    # Support-polygon edges, colored by pipeline outcome (sampler out_of_reach
    # rejections never reach this stage, so they aren't drawn):
    #   red    -- passed sampler, failed IK criteria (collision / lateral_hip_limit / stability / cost)
    #   orange -- passed criteria but dropped by final FPS downsample
    #   green  -- selected (kept)
    n_written = buf.num_written
    if n_written > 0:
        geom_valid_np = buf._geom_valid[:n_written].cpu().numpy()
        ik_valid_np = buf._ik_valid[:n_written].cpu().numpy()
        selected_mask = np.zeros(n_written, dtype=bool)
        selected_mask[sel] = True
        rejected_idx = np.nonzero(geom_valid_np & ~ik_valid_np)[0]
        filtered_idx = np.nonzero(ik_valid_np & ~selected_mask)[0]
        passed_idx = sel

        def _thin(indices: np.ndarray) -> np.ndarray:
            """Evenly spread subset of ``indices``, honouring ``--polygon_limit``."""
            if args.polygon_limit <= 0 or indices.size <= args.polygon_limit:
                return indices
            return indices[np.linspace(0, indices.size - 1, args.polygon_limit).astype(np.int64)]

        group_totals = {
            "criteria-rejected": rejected_idx.size,
            "bucket-filtered": filtered_idx.size,
            "selected": passed_idx.size,
        }
        rejected_idx = _thin(rejected_idx)
        filtered_idx = _thin(filtered_idx)
        passed_idx = _thin(passed_idx)

        feet_all = ct_np.reshape(-1, nc, 3)  # [max_candidates, nc, 3]

        def _polygon_edges(indices: np.ndarray, lift_z: float = 0.015):
            """Order each polygon's feet by angle around its centroid and emit edges."""
            if indices.size == 0:
                return None, None
            feet = feet_all[indices].copy()  # [M, nc, 3]
            feet[..., 2] += lift_z
            centroid_xy = feet[:, :, :2].mean(axis=1, keepdims=True)
            delta = feet[:, :, :2] - centroid_xy
            angles = np.arctan2(delta[..., 1], delta[..., 0])
            order = np.argsort(angles, axis=1)
            ordered = np.take_along_axis(feet, order[:, :, None].repeat(3, axis=2), axis=1)
            starts = ordered.reshape(-1, 3)
            ends = np.roll(ordered, -1, axis=1).reshape(-1, 3)
            return starts, ends

        for indices, color, name, lift in [
            (rejected_idx, (1.0, 0.15, 0.15), "polygons_rejected", 0.01),
            (filtered_idx, (1.0, 0.55, 0.0), "polygons_bucket_filtered", 0.015),
            (passed_idx, (0.1, 0.9, 0.2), "polygons_selected", 0.02),
        ]:
            starts, ends = _polygon_edges(indices, lift_z=lift)
            if starts is None:
                continue
            viewer.log_lines(
                name,
                wp.array(starts.tolist(), dtype=wp.vec3, device=device),
                wp.array(ends.tolist(), dtype=wp.vec3, device=device),
                colors=color,
                width=0.004,
            )
        drawn = {
            "criteria-rejected": len(rejected_idx),
            "bucket-filtered": len(filtered_idx),
            "selected": len(passed_idx),
        }
        summary = ", ".join(
            f"{drawn[k]} of {group_totals[k]} {k}" if drawn[k] != group_totals[k] else f"{group_totals[k]} {k}"
            for k in group_totals
        )
        print(f"  Support polygons: {summary}")

        # Active-support polygon overlay: only the *contact* feet per
        # selected placement. For 4-contact this overlaps green; for
        # tripod it's a triangle inside the quad; for biped it's a
        # segment. This is the polygon SupportPolygonStability tests
        # CoM against.
        _draw_active_support_overlay(viewer, buf, feet_all, passed_idx, nc, device)

    # Visualize collision probe points on ALL robots
    from isaaclab_tasks.core.multi_task.kinematics import _build_collision_probes

    probe_bodies, probe_offsets, _probe_slots = _build_collision_probes(kin.builder, foot_ids, n_samples=16)
    if solved_qs:
        from isaaclab.utils.math import quat_apply as _qa

        all_probe_world = []
        for sq in solved_qs:
            fk_jq = wp.array(sq, dtype=float, device=device)
            fk_state = kin.eval_fk(fk_jq)
            bq_np = fk_state.body_q.numpy()
            for i in range(len(probe_bodies)):
                bid = probe_bodies[i]
                off = np.array(probe_offsets[i], dtype=np.float32)
                bq = bq_np[bid]
                pos = bq[:3]
                quat = bq[3:7]
                q_t = torch.tensor(quat, dtype=torch.float32).unsqueeze(0)
                o_t = torch.tensor(off, dtype=torch.float32).unsqueeze(0)
                rotated = _qa(q_t, o_t).squeeze(0).numpy()
                all_probe_world.append((pos + rotated).tolist())
        viewer.log_points(
            "collision_probes",
            wp.array(all_probe_world, dtype=wp.vec3, device=device),
            radii=0.005,
            colors=(1.0, 0.2, 0.2),
        )
        print(f"  Collision probes: {len(all_probe_world)} points on {len(solved_qs)} robots (red)")

    viewer.begin_frame(0.0)
    viewer.log_state(vis_state)
    viewer.end_frame()

    hn = socket.gethostname()
    print(f"\n  http://localhost:{args.port}")
    print(f"  http://{hn}:{args.port}")
    print("  Dots: Green=candidates, Cyan=selected feet, Red=collision probes")
    print("  Polygons: Red=criteria-rejected, Orange=bucket-filtered, Green=selected")
    print(f"\n  {len(solved_qs)} robots placed.")
    print("\nPress Ctrl+C to stop.\n")

    try:
        while viewer.is_running():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        viewer.close()


if __name__ == "__main__":
    wp.init()
    main()
