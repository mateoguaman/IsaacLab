# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Semantic workers for the unified :class:`~...mdp.commands.StateCommand`.

A payload owns its own lifecycle buffers and interprets gathered task rows: it
writes the per-env target, computes the policy observation + error, advances the
hold timer / success, reports thresholds, aggregates curriculum metrics, and
draws debug markers. The command shell only selects task rows and routes origin
framing.

The two payloads here share :class:`CommandPayloadBase` (the ``cmd_buf`` lifecycle
+ success/threshold/metric/debug machinery) and differ only in target writing,
delta/error computation, and the per-env state views.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.utils.math import (
    axis_angle_from_quat,
    euler_xyz_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from ....mdp.commands.state_command.state_command_cfg import StateCommandCfg
    from .task_table_builder import RelativeStateTaskTable


class CommandPayloadBase:
    """Shared ``cmd_buf`` lifecycle for relative-state payloads.

    Owns the per-env command buffer ``cmd_buf[N, 3, state_dim]`` (slot 0 target,
    1 delta, 2 current), the active-channel mask, the per-env command-type id,
    success/threshold readout, curriculum metric aggregation, and debug
    visualization. Subclasses set the dimensions + ``reward_scales`` and override
    target writing (:meth:`resample`), delta/error (:meth:`update`), the per-env
    state views, and :meth:`debug_visualize`.
    """

    def __init__(self, cfg: StateCommandCfg, env: ManagerBasedEnv, table: RelativeStateTaskTable):
        self._cfg = cfg
        self.table = table
        self.device = env.device
        self.num_envs = env.num_envs
        self._command_names = list(cfg.commands.keys())

    def _alloc_lifecycle(self) -> None:
        """Allocate the shared buffers once dimensions are known (subclass calls this)."""
        self.cmd_buf = torch.zeros(self.num_envs, 3, self.state_dim, device=self.device).contiguous()
        self.cmd_buf[:, 1] = 1.0
        self.cmd_mask = torch.zeros(self.num_envs, self.mask_dim, device=self.device, dtype=torch.bool)
        self.cmd_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._success_per_cmd = torch.zeros(self.table.kind.shape[0], device=self.device)

    def _store_task_selection(self, env_ids: torch.Tensor, task_rows: torch.Tensor) -> None:
        """Record the per-env command-type id (CSR bucket) and active-channel mask."""
        self.cmd_ids[env_ids] = torch.bucketize(task_rows, self.table.offsets[1:-1], right=True)
        self.cmd_mask.index_copy_(0, env_ids, self.table.task_mask[task_rows])

    @property
    def target_state(self) -> torch.Tensor:
        """Per-env target state rows ``cmd_buf[:, 0]``."""
        return self.cmd_buf[:, 0]

    def command_std(self) -> torch.Tensor:
        """Per-env success thresholds for the currently-bound command ids."""
        return self.reward_scales[self.cmd_ids]

    def success(self, error: torch.Tensor, cmd_ids: torch.Tensor) -> torch.Tensor:
        """Return per-env success from command-owned error."""
        reached = torch.all(error < self.reward_scales[cmd_ids], dim=1)
        eps = getattr(self, "success_joint_dev_eps", 0.0)
        if eps > 0.0:
            ids = self.success_joint_dev_ids
            joint_pos = self.robot.data.joint_pos.torch[:, ids]
            default = self.robot.data.default_joint_pos.torch[:, ids]
            reached = reached & ((joint_pos - default).abs().amax(dim=1) < eps)
        return reached

    def get_task_done(self) -> torch.Tensor:
        """Per-env done: the hold timer has fully drained."""
        return self.cmd_buf[:, 1, self.time_idx] <= 0.0

    def get_task_reward(self) -> torch.Tensor:
        """Per-env terminal reward: 1 when done, else 0."""
        return (self.cmd_buf[:, 1, self.time_idx] <= 0.0).float()

    def log_metrics(self, env: ManagerBasedEnv, success_rates: torch.Tensor) -> None:
        """Aggregate the per-task success rate into per-command-type log entries."""
        log = env.extras.setdefault("log", {})
        for cmd_id, name in enumerate(self._command_names):
            start = int(self.table.offsets[cmd_id].item())
            end = int(self.table.offsets[cmd_id + 1].item())
            self._success_per_cmd[cmd_id] = success_rates[start:end].mean() if end > start else 0.0
            log["Metrics/goal_point/success_rate_" + name] = self._success_per_cmd[cmd_id].item()

    def set_debug_vis(self, debug_vis: bool) -> None:
        """Create (lazily) and toggle the goal + current-velocity visualizers."""
        if debug_vis:
            from isaaclab.markers import VisualizationMarkers

            if not hasattr(self, "goal_visualizer"):
                self.goal_visualizer = VisualizationMarkers(self._cfg.payload.goal_visualizer_cfg)
            if not hasattr(self, "current_vel_visualizer"):
                self.current_vel_visualizer = VisualizationMarkers(self._cfg.payload.current_vel_visualizer_cfg)
            self.goal_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_visualizer"):
                self.goal_visualizer.set_visibility(False)
            if hasattr(self, "current_vel_visualizer"):
                self.current_vel_visualizer.set_visibility(False)

    def _draw_current_velocity(self) -> None:
        """Draw the current base-velocity arrow (shared across payloads)."""
        base_pos_w = wp.to_torch(self.robot.data.root_pos_w).clone()
        base_pos_w[:, 2] += 0.5
        xy_velocity = wp.to_torch(self.robot.data.root_lin_vel_b)[:, :2]
        scale = torch.tensor(self.current_vel_visualizer.cfg.markers["arrow"].scale, device=self.device).repeat(
            xy_velocity.shape[0], 1
        )
        scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        quat = quat_mul(wp.to_torch(self.robot.data.root_quat_w), quat_from_euler_xyz(zeros, zeros, heading_angle))
        self.current_vel_visualizer.visualize(base_pos_w, quat, scale)

    def _draw_base_goal_markers(self, env: ManagerBasedEnv) -> tuple[list, list, list, list]:
        """Build the shared pos/pose/vel goal marker batches (foot markers added by subclass)."""
        identity_quat = torch.zeros(self.num_envs, 4, device=self.device)
        identity_quat[:, 3] = 1.0
        kinds = self.table.kind[self.cmd_ids]
        pos_task_ids = env.scene._ALL_INDICES[kinds == 0]
        pose_task_ids = env.scene._ALL_INDICES[kinds == 1]
        vel_task_ids = env.scene._ALL_INDICES[kinds == 2]

        translations, orientations, scales, marker_indices = [], [], [], []
        if len(pos_task_ids) > 0:
            goal_pos = self.cmd_buf[pos_task_ids, 0, :3].clone()
            translations.append(goal_pos)
            orientations.append(identity_quat[: len(pos_task_ids)])
            scales.append(torch.tensor((1.0, 1.0, 1.0), device=self.device).repeat(len(pos_task_ids), 1))
            marker_indices.append(torch.full((len(pos_task_ids),), 1, device=self.device, dtype=torch.long))
        if len(pose_task_ids) > 0:
            goal_pos = self.cmd_buf[pose_task_ids, 0, :3].clone()
            translations.append(goal_pos)
            euler = self.cmd_buf[pose_task_ids, 0, 3:6]
            orientations.append(quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2]))
            scales.append(torch.tensor((1.0, 1.0, 1.0), device=self.device).repeat(len(pose_task_ids), 1))
            marker_indices.append(torch.full((len(pose_task_ids),), 2, device=self.device, dtype=torch.long))
        if len(vel_task_ids) > 0:
            base_pos_w = wp.to_torch(self.robot.data.root_pos_w)[vel_task_ids].clone()
            base_pos_w[:, 2] += 0.5
            xy_cmd = self.cmd_buf[vel_task_ids, 0, 6:8]
            scale = torch.tensor(self.goal_visualizer.cfg.markers["vel_arrow"].scale, device=self.device).repeat(
                len(vel_task_ids), 1
            )
            scale[:, 0] *= torch.linalg.norm(xy_cmd, dim=1) * 3.0
            heading_angle = torch.atan2(xy_cmd[:, 1], xy_cmd[:, 0])
            zeros = torch.zeros_like(heading_angle)
            quat = quat_mul(identity_quat[: len(vel_task_ids)], quat_from_euler_xyz(zeros, zeros, heading_angle))
            translations.append(base_pos_w)
            orientations.append(quat)
            scales.append(scale)
            marker_indices.append(torch.zeros((len(vel_task_ids),), device=self.device, dtype=torch.long))
        return translations, orientations, scales, marker_indices


class CommandPayloadBaseState(CommandPayloadBase):
    """Base-state command semantics: root-state targets, deltas, observation, error, success."""

    error_names = ("error_pos", "error_rot", "error_linvel", "error_angvel")
    error_dim = 4

    def __init__(self, cfg: StateCommandCfg, env: ManagerBasedEnv, table: RelativeStateTaskTable):
        super().__init__(cfg, env, table)
        if cfg.task_table.pipeline_cfg.asset_cfg is None:
            raise ValueError("CommandPayloadBaseState requires cfg.task_table.pipeline_cfg.asset_cfg.")
        robot = env.scene[cfg.task_table.pipeline_cfg.asset_cfg.name]
        device = env.device
        self.robot = robot
        self.reset_assets = [cfg.task_table.pipeline_cfg.asset_cfg.name]
        self.command_dim = 12
        self.state_dim = 12 + robot.num_joints + 1
        self.mask_dim = 12 + robot.num_joints
        self.num_joints = robot.num_joints
        self.time_idx = 12 + robot.num_joints
        self.cmd_joint_pos_slice = slice(12, 12 + robot.num_joints)
        self.reset_joint_pos_slice = slice(13, 13 + robot.num_joints)
        payload_cfg = cfg.payload
        self.normalize_command_obs = payload_cfg.normalize_command_obs
        self.success_joint_dev_eps = float(payload_cfg.success_joint_dev_eps)
        include = [s.strip() for s in str(payload_cfg.success_joint_dev_include).split(",") if s.strip()]
        if include:
            dev_ids = [i for i, n in enumerate(robot.joint_names) if any(s in n for s in include)]
        else:
            dev_ids = list(range(robot.num_joints))
        if self.success_joint_dev_eps > 0.0 and not dev_ids:
            raise ValueError(
                f"success_joint_dev_include '{payload_cfg.success_joint_dev_include}' matched no joints."
                f" Available: {robot.joint_names}"
            )
        self.success_joint_dev_ids = torch.tensor(dev_ids, device=device, dtype=torch.long)
        self.success_joint_dev_names = [robot.joint_names[i] for i in dev_ids]

        # Optional tracked body. The target side needs forward kinematics over the stored
        # target joint angles, so the Newton body id is resolved alongside the PhysX one.
        self.track_body_offset = torch.tensor(payload_cfg.track_body_offset, device=device)
        self.track_body_id: int | None = None
        self.track_body_newton_id: int | None = None
        if payload_cfg.track_body_name:
            ids, _ = robot.find_bodies(payload_cfg.track_body_name, preserve_order=True)
            if len(ids) != 1:
                raise ValueError(
                    f"track_body_name '{payload_cfg.track_body_name}' matched {len(ids)} bodies; expected exactly"
                    f" one. Available: {robot.body_names}"
                )
            self.track_body_id = int(ids[0])
            if table.target_fk_kin is None:
                raise ValueError("track_body_name requires target_fk_kin on the task table.")
            newton_names = list(table.target_fk_kin.model.body_label)
            matches = [i for i, n in enumerate(newton_names) if n.endswith(payload_cfg.track_body_name)]
            if len(matches) != 1:
                raise ValueError(
                    f"track_body_name '{payload_cfg.track_body_name}' matched {len(matches)} Newton bodies;"
                    f" expected exactly one. Available: {newton_names}"
                )
            self.track_body_newton_id = matches[0]
            model = table.target_fk_kin.model
            self.target_fk_kin = table.target_fk_kin
            self.isaac_to_newton_joint_order = table.isaac_to_newton_joint_order
            self._trk_joint_q = torch.empty(env.num_envs, int(model.joint_coord_count), device=device)
            self._trk_joint_qd = wp.zeros((env.num_envs, int(model.joint_dof_count)), dtype=wp.float32, device=device)
            self._trk_body_q_t = torch.empty(env.num_envs, int(model.body_count), 7, device=device)
            self._trk_body_q = wp.from_torch(self._trk_body_q_t, dtype=wp.transformf)
            self._trk_body_qd = wp.zeros((env.num_envs, int(model.body_count)), dtype=wp.spatial_vectorf, device=device)

        std_attrs = ("pos_std", "rot_std", "lin_vel_std", "ang_vel_std")
        global_stds = (payload_cfg.pos_std, payload_cfg.rot_std, payload_cfg.lin_vel_std, payload_cfg.ang_vel_std)
        reward_scales = torch.empty(len(cfg.commands), self.error_dim, device=device)
        for cmd_idx, cmd_cfg in enumerate(cfg.commands.values()):
            for group_idx, attr in enumerate(std_attrs):
                override = getattr(cmd_cfg, attr, None)
                reward_scales[cmd_idx, group_idx] = global_stds[group_idx] if override is None else override
        self.reward_scales = reward_scales

        obs_group_widths = (3, 3, 3, 3)
        obs_inv_scales = torch.empty(len(cfg.commands), self.command_dim, device=device)
        col = 0
        for group_idx, width in enumerate(obs_group_widths):
            obs_inv_scales[:, col : col + width] = reward_scales[:, group_idx : group_idx + 1].reciprocal()
            col += width
        self.obs_inv_unit_scales = obs_inv_scales
        self._alloc_lifecycle()

    def resample(
        self,
        env_ids: torch.Tensor,
        task_rows: torch.Tensor,
        target_states: torch.Tensor,
        target_origin: torch.Tensor | None,
    ) -> None:
        """Write target state for selected rows."""
        self._store_task_selection(env_ids, task_rows)
        if target_origin is not None:
            target_states = target_states.clone()
            target_states[:, :3] += target_origin
        num_resets = env_ids.numel()
        target_cmd = torch.empty(num_resets, self.state_dim, device=self.cmd_buf.device)
        task_params = self.table.params[task_rows]
        uses_terrain_target = self.table.payload_flags[task_rows, 0]

        target_cmd.zero_()
        target_cmd[:, :3].copy_(target_states[:, :3])
        target_cmd[:, :3].add_(task_params[:, :3])
        target_cmd[:, 3:12].copy_(task_params[:, 3:12])
        target_cmd[:, self.cmd_joint_pos_slice].copy_(target_states[:, self.reset_joint_pos_slice])
        if bool(uses_terrain_target.any()):
            if bool(uses_terrain_target.all()):
                target_cmd[:, :3].copy_(target_states[:, :3])
                target_cmd[:, 3], target_cmd[:, 4], target_cmd[:, 5] = euler_xyz_from_quat(target_states[:, 3:7])
            else:
                terrain_local_ids = uses_terrain_target.nonzero(as_tuple=False).squeeze(-1)
                terrain_target_state = target_states[terrain_local_ids]
                target_cmd[terrain_local_ids, :3] = terrain_target_state[:, :3]
                roll, pitch, yaw = euler_xyz_from_quat(terrain_target_state[:, 3:7])
                target_cmd[terrain_local_ids, 3] = roll
                target_cmd[terrain_local_ids, 4] = pitch
                target_cmd[terrain_local_ids, 5] = yaw
        if self.track_body_newton_id is not None:
            # Replace the root target with where the tracked point ends up once the target
            # joint angles are applied, so "reach the goal" refers to the same frame the
            # error is measured in.
            n = env_ids.numel()
            joint_q = self._trk_joint_q[:n]
            joint_q[:, :7] = target_states[:, :7]
            torch.index_select(
                target_states[:, self.reset_joint_pos_slice], 1, self.isaac_to_newton_joint_order, out=joint_q[:, 7:]
            )
            self.target_fk_kin.eval_fk_batched(
                wp.from_torch(joint_q), self._trk_joint_qd[:n], self._trk_body_q[:n], self._trk_body_qd[:n]
            )
            body_q = self._trk_body_q_t[:n, self.track_body_newton_id]
            target_cmd[:, :3] = body_q[:, :3] + quat_apply(body_q[:, 3:7], self.track_body_offset.expand(n, 3))

        target_cmd[:, self.time_idx].copy_(task_params[:, 12])

        self.cmd_buf[:, 0].index_copy_(0, env_ids, target_cmd)
        self.cmd_buf[:, 2, self.time_idx].index_fill_(0, env_ids, 0.0)

    def update(self, step_dt: float, command_obs: torch.Tensor, error: torch.Tensor) -> None:
        """Update current state, delta, observation, error, and hold progress."""
        cmd_buf = self.cmd_buf
        cmd_mask = self.cmd_mask
        cmd_ids = self.cmd_ids
        current = cmd_buf[:, 2]
        target = cmd_buf[:, 0]
        delta = cmd_buf[:, 1]
        root_state_w = wp.to_torch(self.robot.data.root_state_w)
        root_quat = wp.to_torch(self.robot.data.root_quat_w)
        joint_pos = wp.to_torch(self.robot.data.joint_pos)

        if self.track_body_id is None:
            current[:, :3] = root_state_w[:, :3]
        else:
            body_pos = self.robot.data.body_pos_w.torch[:, self.track_body_id]
            body_quat = self.robot.data.body_quat_w.torch[:, self.track_body_id]
            current[:, :3] = body_pos + quat_apply(body_quat, self.track_body_offset.expand(body_pos.shape[0], 3))
        current[:, 3], current[:, 4], current[:, 5] = euler_xyz_from_quat(root_quat)
        current[:, 6:12] = root_state_w[:, 7:13]
        current[:, self.cmd_joint_pos_slice] = joint_pos

        torch.sub(target[:, :3], current[:, :3], out=delta[:, :3])
        delta[:, :3] *= cmd_mask[:, :3]
        delta[:, :3] = quat_apply_inverse(root_quat, delta[:, :3])

        quat_des = quat_from_euler_xyz(target[:, 3], target[:, 4], target[:, 5])
        delta[:, 3:6] = axis_angle_from_quat(quat_mul(quat_inv(root_quat), quat_des))
        delta[:, 3:6] *= cmd_mask[:, 3:6]

        torch.sub(target[:, 6:9], current[:, 6:9], out=delta[:, 6:9])
        delta[:, 6:9] = quat_apply_inverse(root_quat, delta[:, 6:9])
        torch.sub(target[:, 9:12], current[:, 9:12], out=delta[:, 9:12])
        delta[:, 9:12] = quat_apply_inverse(root_quat, delta[:, 9:12])
        delta[:, 6:12] *= cmd_mask[:, 6:12]
        delta[:, self.cmd_joint_pos_slice].zero_()

        command_obs.copy_(delta[:, :12])
        if self.normalize_command_obs:
            command_obs.mul_(self.obs_inv_unit_scales[cmd_ids])

        error[:, 0] = delta[:, 0:3].norm(dim=-1)
        error[:, 1] = delta[:, 3:6].norm(dim=-1)
        error[:, 2] = delta[:, 6:9].norm(dim=-1)
        error[:, 3] = delta[:, 9:12].norm(dim=-1)

        current[:, self.time_idx] += step_dt * self.success(error, cmd_ids)
        torch.sub(target[:, self.time_idx], current[:, self.time_idx], out=delta[:, self.time_idx])

    def current_state_env(self, env_origins: torch.Tensor) -> torch.Tensor:
        """Return current root state in the per-env local frame."""
        current = self.cmd_buf[:, 2]
        return torch.cat([current[:, :3] - env_origins, current[:, 3:12]], dim=-1)

    def target_state_env(self, env_origins: torch.Tensor) -> torch.Tensor:
        """Return target root state in the per-env local frame."""
        target = self.cmd_buf[:, 0]
        return torch.cat([target[:, :3] - env_origins, target[:, 3:12]], dim=-1)

    def debug_visualize(self, env: ManagerBasedEnv) -> None:
        """Draw payload-specific target and current-velocity debug markers."""
        if not self.robot.is_initialized:
            return
        translations, orientations, scales, marker_indices = self._draw_base_goal_markers(env)
        self.goal_visualizer.visualize(
            translations=torch.cat(translations, dim=0),
            orientations=torch.cat(orientations, dim=0),
            scales=torch.cat(scales, dim=0),
            marker_indices=torch.cat(marker_indices, dim=0),
        )
        self._draw_current_velocity()


class CommandPayloadBaseFootState(CommandPayloadBase):
    """Base-state and target-foot command semantics."""

    error_names = ("error_pos", "error_rot", "error_linvel", "error_angvel", "error_foot_pos")
    error_dim = 5

    def __init__(self, cfg: StateCommandCfg, env: ManagerBasedEnv, table: RelativeStateTaskTable):
        super().__init__(cfg, env, table)
        if cfg.task_table.pipeline_cfg.asset_cfg is None:
            raise ValueError("CommandPayloadBaseFootState requires cfg.task_table.pipeline_cfg.asset_cfg.")
        robot = env.scene[cfg.task_table.pipeline_cfg.asset_cfg.name]
        if (
            table.foot_body_ids is None
            or table.newton_foot_body_ids is None
            or table.isaac_to_newton_joint_order is None
            or table.target_fk_kin is None
        ):
            raise ValueError("CommandPayloadBaseFootState requires foot metadata on the task table.")
        device = env.device
        num_joints = robot.num_joints
        payload_cfg = cfg.payload
        self.robot = robot
        self.reset_assets = [cfg.task_table.pipeline_cfg.asset_cfg.name]
        self.num_joints = num_joints
        self.foot_body_ids = table.foot_body_ids
        self.newton_foot_body_ids = table.newton_foot_body_ids
        self.num_feet = len(self.foot_body_ids)
        self.command_dim = 12 + 3 * self.num_feet
        self.state_dim = 12 + num_joints + 1
        self.mask_dim = 12 + num_joints
        self.time_idx = 12 + num_joints
        self.joint_pos_slice = slice(13, 13 + num_joints)
        self.isaac_to_newton_joint_order = table.isaac_to_newton_joint_order
        self.target_fk_kin = table.target_fk_kin

        std_attrs = ("pos_std", "rot_std", "lin_vel_std", "ang_vel_std")
        global_stds = (payload_cfg.pos_std, payload_cfg.rot_std, payload_cfg.lin_vel_std, payload_cfg.ang_vel_std)
        reward_scales = torch.empty(len(cfg.commands), self.error_dim, device=device)
        for cmd_idx, cmd_cfg in enumerate(cfg.commands.values()):
            for group_idx, attr in enumerate(std_attrs):
                override = getattr(cmd_cfg, attr, None)
                reward_scales[cmd_idx, group_idx] = global_stds[group_idx] if override is None else override
            reward_scales[cmd_idx, 4] = payload_cfg.foot_pos_std
        self.reward_scales = reward_scales
        self.normalize_command_obs = payload_cfg.normalize_command_obs

        obs_group_widths = (3, 3, 3, 3, 3 * self.num_feet)
        obs_inv_scales = torch.empty(len(cfg.commands), self.command_dim, device=device)
        col = 0
        for group_idx, width in enumerate(obs_group_widths):
            obs_inv_scales[:, col : col + width] = reward_scales[:, group_idx : group_idx + 1].reciprocal()
            col += width
        self.obs_inv_unit_scales = obs_inv_scales

        self.target_foot_pos_w = torch.zeros(self.num_envs, self.num_feet, 3, device=device)
        self.current_foot_pos_w = torch.zeros_like(self.target_foot_pos_w)
        self.foot_success_mask = torch.zeros(self.num_envs, device=device, dtype=torch.bool)
        self._target_foot_pos_resample = torch.empty(env.num_envs, self.num_feet, 3, device=device)
        self._target_foot_offset_w = torch.empty_like(self._target_foot_pos_resample)
        self._target_foot_pos_b = torch.zeros_like(self._target_foot_pos_resample)
        self._foot_delta_w = torch.empty_like(self._target_foot_pos_resample)
        self._foot_delta_b = torch.zeros_like(self._target_foot_pos_resample)
        self._foot_cross = torch.empty_like(self._target_foot_pos_resample)
        self._foot_cross2 = torch.empty_like(self._target_foot_pos_resample)
        model = self.target_fk_kin.model
        self._target_fk_joint_q = torch.empty(env.num_envs, int(model.joint_coord_count), device=device)
        self._target_fk_joint_qd = wp.zeros((env.num_envs, int(model.joint_dof_count)), dtype=wp.float32, device=device)
        self._target_fk_body_q_t = torch.empty(env.num_envs, int(model.body_count), 7, device=device)
        self._target_fk_body_q = wp.from_torch(self._target_fk_body_q_t, dtype=wp.transformf)
        self._target_fk_body_qd = wp.zeros(
            (env.num_envs, int(model.body_count)), dtype=wp.spatial_vectorf, device=device
        )
        self._alloc_lifecycle()

    def resample(
        self,
        env_ids: torch.Tensor,
        task_rows: torch.Tensor,
        target_states: torch.Tensor,
        target_origin: torch.Tensor | None,
    ) -> None:
        """Write target state for selected rows."""
        self._store_task_selection(env_ids, task_rows)
        if target_origin is not None:
            target_states = target_states.clone()
            target_states[:, :3] += target_origin
        num_resets = env_ids.numel()
        target_cmd = torch.empty(num_resets, self.state_dim, device=self.device)
        task_params = self.table.params[task_rows]
        uses_terrain_target = self.table.payload_flags[task_rows, 0]
        uses_foot_target = uses_terrain_target

        target_cmd.zero_()
        target_cmd[:, :3].copy_(target_states[:, :3])
        target_cmd[:, :3].add_(task_params[:, :3])
        target_cmd[:, 3:12].copy_(task_params[:, 3:12])
        target_cmd[:, 12 : 12 + self.num_joints].copy_(target_states[:, self.joint_pos_slice])
        if bool(uses_terrain_target.any()):
            if bool(uses_terrain_target.all()):
                target_cmd[:, :3].copy_(target_states[:, :3])
                target_cmd[:, 3], target_cmd[:, 4], target_cmd[:, 5] = euler_xyz_from_quat(target_states[:, 3:7])
            else:
                terrain_local_ids = uses_terrain_target.nonzero(as_tuple=False).squeeze(-1)
                terrain_target_state = target_states[terrain_local_ids]
                target_cmd[terrain_local_ids, :3] = terrain_target_state[:, :3]
                roll, pitch, yaw = euler_xyz_from_quat(terrain_target_state[:, 3:7])
                target_cmd[terrain_local_ids, 3] = roll
                target_cmd[terrain_local_ids, 4] = pitch
                target_cmd[terrain_local_ids, 5] = yaw
        target_cmd[:, self.time_idx].copy_(task_params[:, 12])
        self.cmd_buf[:, 0].index_copy_(0, env_ids, target_cmd)
        self.cmd_buf[:, 2, self.time_idx].index_fill_(0, env_ids, 0.0)

        target_foot_pos = self._target_foot_pos_resample[:num_resets]
        if bool(uses_foot_target.any()):
            joint_q = self._target_fk_joint_q[:num_resets]
            body_q_t = self._target_fk_body_q_t[:num_resets]
            joint_q[:, :7] = target_states[:, :7]
            torch.index_select(
                target_states[:, self.joint_pos_slice],
                1,
                self.isaac_to_newton_joint_order,
                out=joint_q[:, 7:],
            )

            self.target_fk_kin.eval_fk_batched(
                wp.from_torch(joint_q),
                self._target_fk_joint_qd[:num_resets],
                self._target_fk_body_q[:num_resets],
                self._target_fk_body_qd[:num_resets],
            )
            for foot_id, body_id in enumerate(self.newton_foot_body_ids):
                target_foot_pos[:num_resets, foot_id].copy_(body_q_t[:, body_id, :3])
        else:
            target_foot_pos.zero_()
        self.target_foot_pos_w.index_copy_(0, env_ids, target_foot_pos)
        self.foot_success_mask.index_copy_(0, env_ids, uses_foot_target)

    def update(self, step_dt: float, command_obs: torch.Tensor, error: torch.Tensor) -> None:
        """Update current state, delta, observation, error, and hold progress."""
        cmd_buf = self.cmd_buf
        cmd_mask = self.cmd_mask
        cmd_ids = self.cmd_ids
        current = cmd_buf[:, 2]
        target = cmd_buf[:, 0]
        delta = cmd_buf[:, 1]
        root_state_w = wp.to_torch(self.robot.data.root_state_w)
        root_quat = wp.to_torch(self.robot.data.root_quat_w)
        joint_pos = wp.to_torch(self.robot.data.joint_pos)
        body_link_pos_w = wp.to_torch(self.robot.data.body_link_pos_w)

        current[:, :3] = root_state_w[:, :3]
        current[:, 3], current[:, 4], current[:, 5] = euler_xyz_from_quat(root_quat)
        current[:, 6:12] = root_state_w[:, 7:13]
        current[:, 12 : 12 + self.num_joints] = joint_pos

        torch.sub(target[:, :3], current[:, :3], out=delta[:, :3])
        delta[:, :3] *= cmd_mask[:, :3]
        delta[:, :3] = quat_apply_inverse(root_quat, delta[:, :3])

        quat_des = quat_from_euler_xyz(target[:, 3], target[:, 4], target[:, 5])
        delta[:, 3:6] = axis_angle_from_quat(quat_mul(quat_inv(root_quat), quat_des))
        delta[:, 3:6] *= cmd_mask[:, 3:6]

        torch.sub(target[:, 6:9], current[:, 6:9], out=delta[:, 6:9])
        delta[:, 6:9] = quat_apply_inverse(root_quat, delta[:, 6:9])
        torch.sub(target[:, 9:12], current[:, 9:12], out=delta[:, 9:12])
        delta[:, 9:12] = quat_apply_inverse(root_quat, delta[:, 9:12])
        delta[:, 6:12] *= cmd_mask[:, 6:12]
        delta[:, 12 : 12 + self.num_joints].zero_()

        for foot_id, body_id in enumerate(self.foot_body_ids):
            self.current_foot_pos_w[:, foot_id].copy_(body_link_pos_w[:, body_id])
        torch.sub(self.target_foot_pos_w, self.current_foot_pos_w, out=self._foot_delta_w)
        torch.sub(self.target_foot_pos_w, root_state_w[:, None, :3], out=self._target_foot_offset_w)
        quat_xyz = root_quat[:, :3]
        quat_w = root_quat[:, 3:4]
        for foot_id in range(self.num_feet):
            torch.cross(quat_xyz, self._foot_delta_w[:, foot_id], dim=-1, out=self._foot_cross[:, foot_id])
            self._foot_cross[:, foot_id].mul_(2.0)
            torch.mul(quat_w, self._foot_cross[:, foot_id], out=self._foot_delta_b[:, foot_id])
            self._foot_delta_b[:, foot_id].neg_()
            self._foot_delta_b[:, foot_id].add_(self._foot_delta_w[:, foot_id])
            torch.cross(quat_xyz, self._foot_cross[:, foot_id], dim=-1, out=self._foot_cross2[:, foot_id])
            self._foot_delta_b[:, foot_id].add_(self._foot_cross2[:, foot_id])

            target_foot_pos_b = self._target_foot_pos_b[:, foot_id]
            torch.cross(quat_xyz, self._target_foot_offset_w[:, foot_id], dim=-1, out=self._foot_cross[:, foot_id])
            self._foot_cross[:, foot_id].mul_(2.0)
            torch.mul(quat_w, self._foot_cross[:, foot_id], out=target_foot_pos_b)
            target_foot_pos_b.neg_()
            target_foot_pos_b.add_(self._target_foot_offset_w[:, foot_id])
            torch.cross(quat_xyz, self._foot_cross[:, foot_id], dim=-1, out=self._foot_cross2[:, foot_id])
            target_foot_pos_b.add_(self._foot_cross2[:, foot_id])

        self._target_foot_pos_b *= self.foot_success_mask.view(-1, 1, 1)
        command_obs[:, :12].copy_(delta[:, :12])
        command_obs[:, 12:].copy_(self._target_foot_pos_b.flatten(1))
        if self.normalize_command_obs:
            command_obs.mul_(self.obs_inv_unit_scales[cmd_ids])

        error[:, 0] = delta[:, 0:3].norm(dim=-1)
        error[:, 1] = delta[:, 3:6].norm(dim=-1)
        error[:, 2] = delta[:, 6:9].norm(dim=-1)
        error[:, 3] = delta[:, 9:12].norm(dim=-1)
        foot_err = self._foot_delta_b.norm(dim=-1).amax(dim=1)
        error[:, 4] = torch.where(self.foot_success_mask, foot_err, torch.zeros_like(foot_err))

        current[:, self.time_idx] += step_dt * self.success(error, cmd_ids)
        torch.sub(target[:, self.time_idx], current[:, self.time_idx], out=delta[:, self.time_idx])

    def current_state_env(self, env_origins: torch.Tensor) -> torch.Tensor:
        """Return current root and foot state in the per-env local frame."""
        current = self.cmd_buf[:, 2]
        pos_local = current[:, :3] - env_origins
        foot_pos_local = (self.current_foot_pos_w - env_origins[:, None, :]).flatten(1)
        return torch.cat([pos_local, current[:, 3:12], foot_pos_local], dim=-1)

    def target_state_env(self, env_origins: torch.Tensor) -> torch.Tensor:
        """Return target root and foot state in the per-env local frame."""
        target = self.cmd_buf[:, 0]
        pos_local = target[:, :3] - env_origins
        foot_pos_local = (self.target_foot_pos_w - env_origins[:, None, :]).flatten(1)
        return torch.cat([pos_local, target[:, 3:12], foot_pos_local], dim=-1)

    def debug_visualize(self, env: ManagerBasedEnv) -> None:
        """Draw payload-specific target, foot-target, and current-velocity markers."""
        if not self.robot.is_initialized:
            return
        translations, orientations, scales, marker_indices = self._draw_base_goal_markers(env)
        identity_quat = torch.zeros(self.num_envs, 4, device=self.device)
        identity_quat[:, 3] = 1.0
        foot_active_ids = env.scene._ALL_INDICES[self.foot_success_mask]
        if len(foot_active_ids) > 0:
            foot_pos_w = self.target_foot_pos_w[foot_active_ids].reshape(-1, 3)
            n_markers = foot_pos_w.shape[0]
            translations.append(foot_pos_w)
            orientations.append(identity_quat[:1].expand(n_markers, -1))
            scales.append(torch.tensor((1.0, 1.0, 1.0), device=self.device).repeat(n_markers, 1))
            marker_indices.append(torch.full((n_markers,), 3, device=self.device, dtype=torch.long))

        self.goal_visualizer.visualize(
            translations=torch.cat(translations, dim=0),
            orientations=torch.cat(orientations, dim=0),
            scales=torch.cat(scales, dim=0),
            marker_indices=torch.cat(marker_indices, dim=0),
        )
        self._draw_current_velocity()
