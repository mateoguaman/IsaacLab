# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ASCII visualizer wrapping :class:`ViewerAscii`.

Mirrors the structure of the Viser/Rerun backends: a thin ``BaseVisualizer``
subclass that owns the Newton viewer instance, binds the scene data provider,
and drives ``begin_frame/log_state/end_frame`` each simulation step.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from isaaclab.visualizers.base_visualizer import BaseVisualizer

from .ascii_visualizer_cfg import AsciiVisualizerCfg
from .viewer_ascii import ViewerAscii

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from isaaclab.physics import BaseSceneDataProvider


class AsciiVisualizer(BaseVisualizer):
    """Terminal ASCII visualizer for Isaac Lab."""

    def __init__(self, cfg: AsciiVisualizerCfg):
        super().__init__(cfg)
        self.cfg: AsciiVisualizerCfg = cfg
        self._viewer: ViewerAscii | None = None
        self._model: Any | None = None
        self._state = None
        self._sim_time = 0.0
        self._last_camera_pose: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None

    def initialize(self, scene_data_provider: BaseSceneDataProvider) -> None:
        if self._is_initialized:
            logger.debug("[AsciiVisualizer] initialize() called while already initialized.")
            return
        if scene_data_provider is None:
            raise RuntimeError("ASCII visualizer requires a scene_data_provider.")

        self._scene_data_provider = scene_data_provider
        metadata = scene_data_provider.get_metadata()
        self._env_ids = self._compute_visualized_env_ids()
        if self._env_ids:
            get_filtered_model = getattr(scene_data_provider, "get_newton_model_for_env_ids", None)
            self._model = (
                get_filtered_model(self._env_ids)
                if callable(get_filtered_model)
                else scene_data_provider.get_newton_model()
            )
        else:
            self._model = scene_data_provider.get_newton_model()
        self._state = scene_data_provider.get_newton_state(self._env_ids)

        camera_pos, camera_target = self._resolve_initial_camera_pose()
        self._viewer = ViewerAscii(
            ascii_cli_path=self.cfg.ascii_cli_path,
            render_width=self.cfg.render_width,
            render_height=self.cfg.render_height,
            sample_res=self.cfg.sample_res,
            color_mode=self.cfg.color_mode,
            bg=self.cfg.bg,
            fov_deg=self.cfg.fov_deg,
            near=self.cfg.near,
            far=self.cfg.far,
            light_direction=self.cfg.light_direction,
            ambient=self.cfg.ambient,
            default_color=self.cfg.default_color,
            background_color=self.cfg.background_color,
            camera_position=camera_pos,
            camera_target=camera_target,
        )
        self._viewer.set_model(self._model)
        self._viewer.set_world_offsets((0.0, 0.0, 0.0))
        self._last_camera_pose = (camera_pos, camera_target)

        num_visualized_envs = len(self._env_ids) if self._env_ids is not None else int(metadata.get("num_envs", 0))
        self._log_initialization_table(
            logger=logger,
            title="AsciiVisualizer Configuration",
            rows=[
                ("ascii_cli_path", self.cfg.ascii_cli_path),
                ("render_size", f"{self.cfg.render_width}x{self.cfg.render_height}"),
                ("sample_res", self.cfg.sample_res),
                ("color_mode", self.cfg.color_mode),
                ("bg", self.cfg.bg),
                ("camera_position", camera_pos),
                ("camera_target", camera_target),
                ("num_visualized_envs", num_visualized_envs),
            ],
        )
        self._is_initialized = True

    def step(self, dt: float) -> None:
        if not self._is_initialized or self._is_closed or self._viewer is None:
            return
        if self._scene_data_provider is None:
            return

        if self.cfg.camera_source == "usd_path":
            self._update_camera_from_usd_path()

        self._state = self._scene_data_provider.get_newton_state(self._env_ids)
        self._sim_time += dt
        self._viewer.begin_frame(self._sim_time)
        if self._state is not None:
            self._viewer.log_state(self._state)
        self._viewer.end_frame()

    def close(self) -> None:
        if not self._is_initialized or self._is_closed:
            return
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception as exc:
                logger.warning("[AsciiVisualizer] Error during close: %s", exc)
            self._viewer = None
        self._is_closed = True

    def is_running(self) -> bool:
        if not self._is_initialized or self._is_closed:
            return False
        if self._viewer is None:
            return False
        return self._viewer.is_running()

    def supports_markers(self) -> bool:
        return False

    def supports_live_plots(self) -> bool:
        return False

    # --- camera helpers ---

    def _resolve_initial_camera_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        if self.cfg.camera_source == "usd_path":
            pose = self._resolve_camera_pose_from_usd_path(self.cfg.camera_usd_path)
            if pose is not None:
                return pose
            logger.warning(
                "[AsciiVisualizer] camera_usd_path '%s' not found; using configured camera.",
                self.cfg.camera_usd_path,
            )
        return self.cfg.camera_position, self.cfg.camera_target

    def _update_camera_from_usd_path(self) -> None:
        pose = self._resolve_camera_pose_from_usd_path(self.cfg.camera_usd_path)
        if pose is None or self._viewer is None:
            return
        if self._last_camera_pose == pose:
            return
        self._viewer.set_camera_view(pose[0], pose[1])
        self._last_camera_pose = pose

    def set_camera_view(self, eye: tuple, target: tuple) -> None:
        if self._viewer is not None:
            self._viewer.set_camera_view(tuple(eye), tuple(target))
            self._last_camera_pose = (tuple(eye), tuple(target))
