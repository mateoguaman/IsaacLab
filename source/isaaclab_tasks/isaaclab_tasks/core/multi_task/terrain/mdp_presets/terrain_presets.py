# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-terrain presets selectable via ``env.scene.terrain.terrain_generator.sub_terrains=<name>``."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from .. import terrains


@configclass
class SubTerrainPresetCfg(PresetCfg):
    """Named sub-terrain configurations for the position locomotion task."""

    terrain_curriculum = {
        "gap": terrains.GAP_CURRICULUM,
        "pit": terrains.PIT_CURRICULUM,
        "extreme_stair": terrains.EXTREME_STAIR_CURRICULUM,
        "slope_inv": terrains.SLOPE_INV_CURRICULUM,
        "stepping_stone": terrains.STEPPING_STONE_CURRICULUM,
        "radiating_beam": terrains.RADIATING_BEAM_CURRICULUM,
        "contour": terrains.CONTOUR_CURRICULUM,
        "climbing_box": terrains.CLIMBING_BOX_CURRICULUM,
        "floating_island": terrains.FLOATING_ISLAND_CURRICULUM,
        "maze": terrains.MAZE_CURRICULUM,
        "random_jump_box": terrains.RANDOM_JUMP_BOX_CURRICULUM,
        "random_parallel_box": terrains.RANDOM_PARALLEL_BOX_CURRICULUM,
        "balancing_beam": terrains.BALANCING_BEAM_CURRICULUM,
    }
    basic = {
        "gap": terrains.GAP_CURRICULUM,
        "pit": terrains.PIT_CURRICULUM,
        "extreme_stair": terrains.EXTREME_STAIR_CURRICULUM,
        "slope_inv": terrains.SLOPE_INV_CURRICULUM,
        "stepping_stone": terrains.STEPPING_STONE_CURRICULUM,
    }
    """Five-terrain subset of :attr:`terrain_curriculum`, for runs that should master fewer
    terrain types before facing the full set."""

    gap = {"gap": terrains.GAP}
    pit = {"pit": terrains.PIT}
    extreme_stair = {"extreme_stair": terrains.EXTREME_STAIR}
    slope_inv = {"slope_inv": terrains.SLOPE_INV}
    square_pillar_obstacle = {"square_pillar_obstacle": terrains.SQUARE_PILLAR_OBSTACLE}
    stepping_stone = {"stepping_stone": terrains.STEPPING_STONE}
    stepping_stone_curriculum = {"stepping_stone": terrains.STEPPING_STONE_CURRICULUM}
    radiating_beam = {"radiating_beam": terrains.RADIATING_BEAM}
    flat = {"flat": terrains.FLAT}
    default = terrain_curriculum
