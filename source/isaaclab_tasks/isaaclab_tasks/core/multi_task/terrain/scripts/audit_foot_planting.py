# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Audit how well the retargeted spawn poses plant their feet on the terrain.

Runs the same pipeline as ``validate_spawn_points.py``, then for every selected
placement raycasts straight down from each sole-hull vertex to the terrain mesh
and reports the resulting clearance. A planted foot has its lowest sole point
at the surface; a foot hanging in the air shows up as positive clearance.

Run::

    ./isaaclab.sh -p audit_foot_planting.py presets=g1,flat
    ./isaaclab.sh -p audit_foot_planting.py presets=anymal_c
"""

from __future__ import annotations

import sys

sys.path[:] = [p for p in sys.path if "pip_prebundle" not in p and "pip_archive" not in p]

import argparse

import numpy as np
import torch
import warp as wp

from isaaclab.utils.warp import convert_to_warp_mesh


@wp.kernel
def _terrain_height_under(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    ray_start_above: float,
    max_dist: float,
    out_height: wp.array(dtype=wp.float32),
    out_hit: wp.array(dtype=wp.uint8),
):
    """Terrain surface height directly below each query point [m]."""
    i = wp.tid()
    p = points[i]
    origin = wp.vec3(p[0], p[1], p[2] + ray_start_above)
    direction = wp.vec3(0.0, 0.0, -1.0)
    query = wp.mesh_query_ray(mesh_id, origin, direction, max_dist)
    if query.result:
        out_height[i] = origin[2] - query.t
        out_hit[i] = wp.uint8(1)
    else:
        out_height[i] = -1.0e9
        out_hit[i] = wp.uint8(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planted_tol", type=float, default=0.02, help="Clearance below which a foot counts as planted [m].")
    parser.add_argument("--dump", type=int, default=8, help="Worst-offender placements to print.")
    parser.add_argument(
        "--min_support",
        type=float,
        default=0.25,
        help="Fraction of a sole that must rest on the surface for the foot to count as planted.",
    )
    parser.add_argument("--max_robots", type=int, default=None, help="Placement budget override.")
    parser.add_argument("--spacing", type=float, default=None, help="Placement spacing override [m].")

    from isaaclab_tasks.utils import setup_preset_cli

    import isaaclab_tasks.core.multi_task.terrain.scripts.validate_spawn_points as vsp

    vsp._set_registration_guard()
    args, remaining = setup_preset_cli(parser)
    sys.argv = [sys.argv[0]] + remaining

    vsp._register_position_tasks()
    vsp._eager_load_presets()

    from isaaclab.terrains.terrain_generator import TerrainGenerator

    from isaaclab_tasks.core.multi_task.terrain.retarget import RetargetPipeline, apply_final_fps
    from isaaclab_tasks.utils.hydra import resolve_task_config

    env_cfg, _ = resolve_task_config("Isaac-Position-v0", "rsl_rl_cfg_entry_point")
    device = "cuda:0"

    robot_cfg = env_cfg.scene.robot
    robot_usd = vsp._resolve_robot_usd(robot_cfg)
    terrain_gen_cfg = env_cfg.scene.terrain.terrain_generator
    gen = TerrainGenerator(cfg=terrain_gen_cfg, device=device)
    mesh = gen.terrain_mesh
    x_range, y_range = vsp._terrain_grid_extents(terrain_gen_cfg)
    wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)

    table_cfg = env_cfg.commands.goal_point.task_table
    pipeline_cfg = vsp._patch_kin_with_robot(table_cfg.pipeline_cfg, robot_cfg, robot_usd, device)
    pipeline_cfg = vsp._patch_sampler_bounds(pipeline_cfg, x_range, y_range)
    extractor = pipeline_cfg.sampler.sizing.fps_features
    n_desired = vsp._derive_n_desired(args, table_cfg.pool_spacing, x_range, y_range, extractor)
    pipeline_cfg = pipeline_cfg.replace(
        sampler=pipeline_cfg.sampler.replace(sizing=pipeline_cfg.sampler.sizing.replace(fps_spacing=table_cfg.pool_spacing))
    )

    pipeline = RetargetPipeline(pipeline_cfg)
    kin = pipeline.kin
    foot_ids = pipeline.foot_body_ids
    foot_names = pipeline.foot_body_names
    nc = len(foot_ids)

    buffer = pipeline.run(wp_mesh, np.zeros(3, dtype=np.float32), n_desired=n_desired)
    apply_final_fps(
        buffer,
        n_desired=n_desired,
        extractor=extractor,
        spacing=pipeline_cfg.sampler.sizing.fps_spacing,
    )
    n_sel = int(buffer.num_selected)
    if n_sel == 0:
        print("No placements selected; nothing to audit.")
        return
    sel = buffer._selected[:n_sel].long()

    hulls = kin.foot_contact_hulls(foot_ids)
    hull_xy = torch.as_tensor(hulls["vertices"], device=device)  # [nc, V, 2]
    sole_z = torch.as_tensor(hulls["sole_z"], device=device)  # [nc]
    n_vert = hull_xy.shape[1]

    n_bodies = kin.model.body_count
    body_q = buffer.body_q_t[: buffer.num_written * n_bodies].view(-1, n_bodies, 7)[sel]
    foot_index = torch.as_tensor(foot_ids, device=device, dtype=torch.long)
    foot_pos = body_q[:, foot_index, :3]  # [N, nc, 3]
    foot_quat = body_q[:, foot_index, 3:7]  # [N, nc, 4] xyzw

    from isaaclab_tasks.core.multi_task.terrain.retarget.criteria import _quat_to_matrix_xyzw

    rotation = _quat_to_matrix_xyzw(foot_quat)  # [N, nc, 3, 3]
    local = torch.cat([hull_xy, sole_z.view(nc, 1, 1).expand(nc, n_vert, 1)], dim=-1)  # [nc, V, 3]
    world = foot_pos.unsqueeze(2) + torch.einsum("nfij,fvj->nfvi", rotation, local)  # [N, nc, V, 3]

    flat = world.reshape(-1, 3).contiguous()
    height = torch.empty(flat.shape[0], dtype=torch.float32, device=device)
    hit = torch.empty(flat.shape[0], dtype=torch.uint8, device=device)
    wp.launch(
        _terrain_height_under,
        dim=flat.shape[0],
        inputs=[wp_mesh.id, wp.from_torch(flat, dtype=wp.vec3), 5.0, 60.0],
        outputs=[wp.from_torch(height, dtype=wp.float32), wp.from_torch(hit, dtype=wp.uint8)],
        device=device,
    )
    clearance = (flat[:, 2] - height).view(-1, nc, n_vert)
    missed = (hit == 0).view(-1, nc, n_vert)

    # A foot is planted when enough of its sole rests on the surface and none
    # of it is driven through. Reducing with a plain minimum instead would call
    # a correctly perched foot "sunk": a sole vertex overhanging a void can
    # raycast onto higher ground beside it and report negative clearance, so a
    # partial foothold has to be judged on the part that bears load.
    resting = (clearance.abs() <= args.planted_tol) & ~missed
    penetrating = (clearance < -args.planted_tol) & ~missed
    support_fraction = resting.to(torch.float32).mean(dim=-1)  # [N, nc]
    # Clearance of the bearing part, for how well that part sits.
    bearing = torch.where(resting, clearance, torch.full_like(clearance, float("nan")))
    foot_clearance = bearing.nanmean(dim=-1)  # [N, nc]

    is_contact = buffer.is_contact_t[: buffer.num_written * nc].view(-1, nc)[sel]

    planted = (support_fraction >= args.min_support) & ~penetrating.any(dim=-1)
    n_planted = planted.sum(dim=-1)

    print()
    print(
        f"=== Foot planting audit: {n_sel} placements, {nc} feet,"
        f" tol +/-{args.planted_tol * 1000:.0f} mm, min support {args.min_support:.0%} ==="
    )
    print(f"  feet: {foot_names}")
    print()
    print("  placements by number of planted feet")
    for k in range(nc + 1):
        count = int((n_planted == k).sum())
        print(f"    {k}/{nc} planted : {count:8d}  ({100.0 * count / n_sel:5.1f}%)")
    print()
    print("  sole support fraction, over placements where the foot is flagged contact")
    for f in range(nc):
        values = support_fraction[is_contact[:, f], f]
        if values.numel():
            q = torch.quantile(values, torch.tensor([0.1, 0.5, 0.9], device=device)).tolist()
            print(
                f"    {foot_names[f]:24s} p10/50/90 = {q[0]:5.2f} {q[1]:5.2f} {q[2]:5.2f}"
                f"   fully supported: {float((values >= 0.99).float().mean()):5.1%}"
            )
    print()
    print("  bearing-surface clearance [mm], over placements where the foot is flagged contact")
    for f in range(nc):
        contact_mask = is_contact[:, f]
        values = foot_clearance[contact_mask, f]
        values = values[~torch.isnan(values)]
        if values.numel() == 0:
            print(f"    {foot_names[f]:24s} (no contact-flagged placements)")
            continue
        quantiles = torch.tensor([0.5, 0.9, 0.99], device=device)
        p50, p90, p99 = torch.quantile(values, quantiles).tolist()
        print(
            f"    {foot_names[f]:24s} n={values.numel():7d}  median={p50 * 1000:7.1f}"
            f"  p90={p90 * 1000:8.1f}  p99={p99 * 1000:8.1f}  max={values.max().item() * 1000:9.1f}"
        )
    print()
    print("  contact flag vs actual planting")
    flagged = int(is_contact.sum())
    flagged_planted = int((is_contact & planted).sum())
    print(f"    feet flagged contact       : {flagged:8d}")
    print(f"    of those actually planted  : {flagged_planted:8d}  ({100.0 * flagged_planted / max(flagged, 1):5.1f}%)")
    airborne = is_contact & ~planted & ~torch.isnan(foot_clearance)
    print(f"    of those hanging in air    : {int(airborne.sum()):8d}")
    below = is_contact & (foot_clearance < -args.planted_tol)
    print(f"    of those sunk below surface: {int(below.sum()):8d}")

    # Split the error: is the contact target itself placed correctly, and does
    # the IK actually reach it?
    targets = buffer.contact_targets_t[: buffer.num_written * nc].view(-1, nc, 3)[sel]
    target_flat = targets.reshape(-1, 3).contiguous()
    t_height = torch.empty(target_flat.shape[0], dtype=torch.float32, device=device)
    t_hit = torch.empty(target_flat.shape[0], dtype=torch.uint8, device=device)
    wp.launch(
        _terrain_height_under,
        dim=target_flat.shape[0],
        inputs=[wp_mesh.id, wp.from_torch(target_flat, dtype=wp.vec3), 5.0, 60.0],
        outputs=[wp.from_torch(t_height, dtype=wp.float32), wp.from_torch(t_hit, dtype=wp.uint8)],
        device=device,
    )
    ground_offset = float(kin.foot_geometry(foot_ids)["foot_ground_offset"])
    # Where the sole would sit if the foot body origin landed exactly on target.
    target_sole_clearance = (target_flat[:, 2] - ground_offset - t_height).view(-1, nc)
    ik_residual = (foot_pos - targets).norm(dim=-1)
    ik_residual_z = foot_pos[..., 2] - targets[..., 2]

    # The base is only weakly pinned, so an unreachable foot target can be
    # absorbed by the whole robot sliding instead of the leg extending. Split
    # the foot error into horizontal and vertical, and report the base drift
    # alongside it, so the two are distinguishable.
    base_pos = buffer.joint_q_result_t[sel, 0:3]
    base_target = buffer.base_target_pos_t[sel]
    base_drift = (base_pos - base_target).norm(dim=-1)
    base_drift_xy = (base_pos[:, :2] - base_target[:, :2]).norm(dim=-1)
    residual_xy = (foot_pos[..., :2] - targets[..., :2]).norm(dim=-1)
    print()
    print(f"  base drift from its sampler target: median={base_drift.median().item() * 1000:7.1f} mm"
          f"  (xy only {base_drift_xy.median().item() * 1000:7.1f} mm)")

    print()
    print(f"  error split (foot_ground_offset = {ground_offset * 1000:.1f} mm)")
    quantile_levels = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=device)
    for f in range(nc):
        spread = torch.quantile(residual_xy[:, f], quantile_levels).mul(1000.0).tolist()
        print(
            f"    {foot_names[f]:24s} foot residual xy p5/25/50/75/95 ="
            + " ".join(f"{v:7.1f}" for v in spread)
            + f" mm   z={ik_residual_z[:, f].median().item() * 1000:+6.1f} mm"
        )
    for f in range(nc):
        tsc = target_sole_clearance[:, f]
        tsc = tsc[t_hit.view(-1, nc)[:, f] == 1]
        print(
            f"    {foot_names[f]:24s} target sole vs terrain: median={tsc.median().item() * 1000:7.1f} mm"
            f"   |  IK residual: median={ik_residual[:, f].median().item() * 1000:6.1f} mm"
            f"  (z only {ik_residual_z[:, f].median().item() * 1000:+7.1f} mm)"
        )

    worst = foot_clearance.nan_to_num(-1e9).amin(dim=-1)
    highest = foot_clearance.nan_to_num(-1e9).amax(dim=-1)
    order = torch.argsort(highest, descending=True)[: args.dump]
    print()
    print(f"  worst {args.dump} placements by highest foot [buffer row, clearance mm, contact flags]")
    for row in order.tolist():
        cl = [f"{v * 1000:8.1f}" for v in foot_clearance[row].tolist()]
        fl = [int(v) for v in is_contact[row].tolist()]
        base_z = float(buffer.joint_q_result_t[sel[row], 2])
        print(f"    row={int(sel[row]):7d}  base_z={base_z:6.3f}  clearance={cl}  contact={fl}")

    print()
    print(f"  summary: {int((n_planted == nc).sum()) / n_sel * 100:.1f}% of placements have every foot planted;"
          f" {int((n_planted == 0).sum()) / n_sel * 100:.1f}% have none")
    _ = worst


if __name__ == "__main__":
    wp.init()
    main()
