# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Audit a robot's collision geometry as the retarget pipeline sees it.

Lists every body carrying collision shapes, how low each reaches at the default
stance, and whether the terrain-collision IK objective treats it as a foot. A
non-foot body sitting at ground level is pushed away from the terrain by that
objective, which fights the foot-contact targets.

Run::

    ./isaaclab.sh -p audit_collision_geometry.py presets=g1
"""

from __future__ import annotations

import sys

sys.path[:] = [p for p in sys.path if "pip_prebundle" not in p and "pip_archive" not in p]

import argparse

import warp as wp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--margin", type=float, default=0.05, help="Terrain-collision objective margin [m].")
    parser.add_argument("--probe_samples", type=int, default=4, help="Probes per body, matching the objective cfg.")

    import isaaclab_tasks.core.multi_task.terrain.scripts.validate_spawn_points as vsp
    from isaaclab_tasks.utils import setup_preset_cli

    vsp._set_registration_guard()
    args, remaining = setup_preset_cli(parser)
    sys.argv = [sys.argv[0]] + remaining
    vsp._register_position_tasks()
    vsp._eager_load_presets()

    from newton import GeoType

    from isaaclab_tasks.core.multi_task.kinematics import _build_collision_probes
    from isaaclab_tasks.core.multi_task.terrain.retarget import RetargetPipeline
    from isaaclab_tasks.utils.hydra import resolve_task_config

    env_cfg, _ = resolve_task_config("Isaac-Position-v0", "rsl_rl_cfg_entry_point")
    robot_cfg = env_cfg.scene.robot
    robot_usd = vsp._resolve_robot_usd(robot_cfg)
    table_cfg = env_cfg.commands.goal_point.task_table
    pipeline_cfg = vsp._patch_kin_with_robot(table_cfg.pipeline_cfg, robot_cfg, robot_usd, "cuda:0")

    pipeline = RetargetPipeline(pipeline_cfg)
    kin = pipeline.kin
    builder = kin.builder
    foot_ids = set(int(f) for f in pipeline.foot_body_ids)

    geometry = kin.foot_geometry(pipeline.foot_body_ids)
    ground_offset = float(geometry["foot_ground_offset"])
    body_q = kin.default_body_q
    # Default stance puts the soles on z = 0; shift so heights read as
    # "above the ground the feet are standing on".
    sole_world_z = min(float(body_q[f][2]) for f in foot_ids) - ground_offset

    shapes_by_body: dict[int, list[int]] = {}
    for si in range(len(builder.shape_body)):
        shapes_by_body.setdefault(int(builder.shape_body[si]), []).append(si)

    print()
    print(f"=== Collision geometry, {len(kin.body_names)} bodies, {len(builder.shape_body)} shapes ===")
    print(f"  feet: {pipeline.foot_body_names}  foot_ground_offset={ground_offset * 1000:.1f} mm")
    print(f"  terrain-collision objective margin: {args.margin * 1000:.0f} mm")
    print()
    print(f"  {'body':32s} {'shapes':>6s} {'lowest z':>10s} {'role':>10s}  note")

    offenders = []
    for body_id in sorted(shapes_by_body):
        name = kin.body_names[body_id]
        transform = body_q[body_id]
        lowest = None
        kinds = []
        for si in shapes_by_body[body_id]:
            kinds.append(GeoType(int(builder.shape_type[si])).name[:4])
            local_z = kin._shape_local_z_min(
                int(builder.shape_type[si]),
                builder.shape_scale[si],
                builder.shape_transform[si],
                builder.shape_source[si],
            )
            if local_z is None:
                continue
            world_z = float(transform[2]) + local_z - sole_world_z
            lowest = world_z if lowest is None else min(lowest, world_z)
        if lowest is None:
            continue
        is_foot = body_id in foot_ids
        role = "FOOT" if is_foot else "-"
        note = ""
        if not is_foot and lowest < args.margin:
            note = f"<-- within margin, pushed up {(args.margin - lowest) * 1000:.0f} mm"
            offenders.append((name, lowest))
        print(f"  {name:32s} {len(shapes_by_body[body_id]):6d} {lowest * 1000:9.1f} {role:>10s}  {note}")

    probe_bodies, probe_offsets, probe_slots = _build_collision_probes(
        builder, list(pipeline.foot_body_ids), args.probe_samples
    )
    n_gated = sum(1 for s in probe_slots if s >= 0)
    print()
    print(
        f"  collision probes: {len(probe_bodies)} total, {n_gated} on feet (gated by contact), {len(probe_bodies) - n_gated} always active"
    )

    print()
    if offenders:
        print(
            f"  {len(offenders)} non-foot bodies sit within the {args.margin * 1000:.0f} mm margin at the default stance:"
        )
        for name, lowest in sorted(offenders, key=lambda kv: kv[1]):
            print(f"    {name:32s} lowest {lowest * 1000:7.1f} mm above the sole plane")
        print()
        print("  The terrain-collision objective pushes each of these clear of the ground,")
        print("  which lifts the foot it is attached to and fights the contact target.")
    else:
        print("  No non-foot body sits within the margin; terrain collision is not lifting the feet.")


if __name__ == "__main__":
    wp.init()
    main()
