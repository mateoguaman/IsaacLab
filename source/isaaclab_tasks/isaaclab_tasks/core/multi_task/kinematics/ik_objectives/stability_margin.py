# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IK objective: stability margin (CoM inside the support region).

Penalizes configurations where the CoM projects outside the convex hull of
the sole outlines of the feet in contact (static instability under gravity).
No gradient inside the region — any interior position is stable — so the
objective does not bias the solve toward its centroid.

Building the region from sole outlines rather than foot origins is what lets
the same objective serve a quadruped on four near-point feet and a humanoid
on one or two flat soles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import newton
import newton.ik as ik
import numpy as np
import warp as wp

from ._kernels import jac_fill_row

if TYPE_CHECKING:
    from isaaclab_tasks.core.multi_task.terrain.retarget.pipeline import RetargetPipeline

    from .cfg import IKObjectiveStabilityMarginCfg


def _build_stability_jac_relations(model: newton.Model, foot_body_ids: list[int]) -> dict:
    """Precompute tables used by the analytic Jacobian kernel.

    For each dof, we need the joint that owns it (so we can index into the
    per-joint subtree COM cache). For each joint, we need its subtree
    bodies (for the per-iteration COM compute) and total mass. For each
    foot, we need a (foot, dof) bitmap saying whether motion of that dof
    moves that foot.
    """
    jp = model.joint_parent.numpy()
    jc = model.joint_child.numpy()
    bm = model.body_mass.numpy()
    qd_start = model.joint_qd_start.numpy()
    n_joints = model.joint_count
    n_bodies = model.body_count
    n_dofs = model.joint_dof_count

    dof_to_joint = np.zeros(n_dofs, dtype=np.int32)
    for jg in range(n_joints):
        end = qd_start[jg + 1] if jg + 1 < len(qd_start) else n_dofs
        for d in range(int(qd_start[jg]), int(end)):
            dof_to_joint[d] = jg

    children: dict[int, list[int]] = {b: [] for b in range(-1, n_bodies)}
    for jg in range(n_joints):
        children[int(jp[jg])].append(int(jc[jg]))

    def _subtree(child_body: int) -> list[int]:
        if child_body < 0:
            return []
        out, queue = [child_body], [child_body]
        while queue:
            x = queue.pop()
            for ch in children[x]:
                out.append(ch)
                queue.append(ch)
        return out

    flat, offsets, masses = [], [0], []
    subtree_set: list[set[int]] = []
    for jg in range(n_joints):
        bodies = _subtree(int(jc[jg]))
        flat.extend(bodies)
        offsets.append(len(flat))
        masses.append(float(sum(bm[b] for b in bodies)))
        subtree_set.append(set(bodies))

    foot_in_subtree = np.zeros((len(foot_body_ids), n_dofs), dtype=np.uint8)
    for fi, f_body in enumerate(foot_body_ids):
        for d in range(n_dofs):
            if int(f_body) in subtree_set[int(dof_to_joint[d])]:
                foot_in_subtree[fi, d] = 1

    return {
        "dof_to_joint": dof_to_joint,
        "joint_subtree_bodies": np.array(flat, dtype=np.int32),
        "joint_subtree_offsets": np.array(offsets, dtype=np.int32),
        "joint_subtree_mass": np.array(masses, dtype=np.float32),
        "foot_in_subtree": foot_in_subtree,
    }


@wp.kernel
def _compute_joint_subtree_origin_coms(
    body_q: wp.array2d(dtype=wp.transform),
    body_mass: wp.array1d(dtype=wp.float32),
    subtree_bodies: wp.array1d(dtype=wp.int32),
    subtree_offsets: wp.array1d(dtype=wp.int32),
    subtree_inv_mass: wp.array1d(dtype=wp.float32),
    out_com: wp.array2d(dtype=wp.vec3),
):
    """Mass-weighted average of body **origin** positions per joint subtree.

    Matches the residual kernel's CoM aggregation, which weights body
    origin positions (not body-COM offsets) by mass. Using origin keeps
    the residual and Jacobian consistent so analytic = derivative of what
    the residual actually computes.
    """
    p, j = wp.tid()
    s = subtree_offsets[j]
    e = subtree_offsets[j + 1]
    com = wp.vec3(0.0, 0.0, 0.0)
    for i in range(s, e):
        b = subtree_bodies[i]
        com = com + body_mass[b] * wp.transform_get_translation(body_q[p, b])
    out_com[p, j] = com * subtree_inv_mass[j]


@wp.kernel
def _stability_margin_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    body_mass: wp.array1d(dtype=wp.float32),
    n_bodies: int,
    foot_body_indices: wp.array1d(dtype=wp.int32),
    is_contact: wp.array2d(dtype=wp.uint8),
    hull_local: wp.array2d(dtype=wp.vec2),
    directions: wp.array1d(dtype=wp.vec2),
    scratch_world: wp.array2d(dtype=wp.vec3),
    scratch_slot: wp.array2d(dtype=wp.int32),
    n_feet: int,
    n_vertices: int,
    n_directions: int,
    total_mass_inv: float,
    weight: float,
    start_idx: int,
    residuals: wp.array2d(dtype=wp.float32),
    active_slot: wp.array1d(dtype=wp.int32),
    active_vertex_w: wp.array1d(dtype=wp.vec3),
    active_direction: wp.array1d(dtype=wp.vec2),
):
    """Residual = ``max(0, -margin)`` where margin is the signed distance
    from the CoM (XY projection) to the boundary of the support region --
    the convex hull of the sole outlines of every foot in contact.

    The margin is evaluated by support function: for each probe direction,
    how far the region reaches minus how far the CoM sits, minimised over
    directions. That needs no hull ordering, which matters because most
    sole vertices lie strictly inside the region, and it stays valid at any
    contact count -- a single sole is a support region in its own right.

    Also writes the cache the analytic Jacobian reads: the winning
    direction, the foot slot owning the supporting vertex, and that
    vertex's world position. By Danskin's theorem the gradient at the
    optimum only involves those, so the chain rule has a single term.
    Sets ``active_slot = -1`` when residual = 0 (hinge, no gradient).
    """
    row = wp.tid()

    # Mass-weighted CoM in XY (over the whole body, not just active feet --
    # physical CoM doesn't care about contact state).
    com_x = float(0.0)
    com_y = float(0.0)
    for b in range(n_bodies):
        pos_b = wp.transform_get_translation(body_q[row, b])
        com_x = com_x + body_mass[b] * pos_b[0]
        com_y = com_y + body_mass[b] * pos_b[1]
    com_x = com_x * total_mass_inv
    com_y = com_y * total_mass_inv

    # Place every sole vertex of every contacting foot in the world once,
    # so the direction sweep below is dot products against cached points
    # rather than repeated rigid transforms.
    n_active = int(0)
    for f in range(n_feet):
        if is_contact[row, f] == wp.uint8(0):
            continue
        tf = body_q[row, foot_body_indices[f]]
        for v in range(n_vertices):
            h = hull_local[f, v]
            scratch_world[row, n_active] = wp.transform_point(tf, wp.vec3(h[0], h[1], 0.0))
            scratch_slot[row, n_active] = f
            n_active = n_active + 1

    if n_active == 0:
        residuals[row, start_idx] = 0.0
        active_slot[row] = -1
        return

    best_margin = float(1.0e9)
    best_slot = int(0)
    best_vertex = wp.vec3(0.0, 0.0, 0.0)
    best_direction = wp.vec2(0.0, 0.0)
    for k in range(n_directions):
        d = directions[k]
        support = float(-1.0e9)
        support_slot = int(0)
        support_vertex = wp.vec3(0.0, 0.0, 0.0)
        for i in range(n_active):
            point = scratch_world[row, i]
            projection = d[0] * point[0] + d[1] * point[1]
            if projection > support:
                support = projection
                support_slot = scratch_slot[row, i]
                support_vertex = point
        margin = support - (d[0] * com_x + d[1] * com_y)
        if margin < best_margin:
            best_margin = margin
            best_slot = support_slot
            best_vertex = support_vertex
            best_direction = d

    violation = wp.max(0.0, -best_margin)
    residuals[row, start_idx] = weight * violation

    if violation > 0.0:
        active_slot[row] = best_slot
        active_vertex_w[row] = best_vertex
        active_direction[row] = best_direction
    else:
        active_slot[row] = -1


@wp.kernel
def _stability_margin_jac_analytic(
    foot_in_subtree: wp.array2d(dtype=wp.uint8),
    joint_S_s: wp.array2d(dtype=wp.spatial_vector),
    dof_to_joint: wp.array1d(dtype=wp.int32),
    joint_subtree_mass: wp.array1d(dtype=wp.float32),
    joint_subtree_com: wp.array2d(dtype=wp.vec3),
    total_mass_inv: float,
    active_slot: wp.array1d(dtype=wp.int32),
    active_vertex_w: wp.array1d(dtype=wp.vec3),
    active_direction: wp.array1d(dtype=wp.vec2),
    weight: float,
    start_idx: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    """One thread per ``(problem, dof)``. Hinge gradient is 0 inside the
    support region (encoded by ``active_slot < 0``).

    At the optimum the margin is ``d . (vertex - com)`` for the winning
    direction ``d`` and supporting sole vertex, and neither the direction
    nor which vertex wins varies to first order, so

        d(residual)/dq = -weight * d . (d(vertex)/dq - d(com)/dq)

    with

    * vertex velocity ``v_d + w_d x vertex``, gated by the precomputed
      ``foot_in_subtree`` table for the foot owning it. Taking the moment
      about the vertex itself (rather than the foot origin) is what makes
      the foot's *rotation* carry gradient, so tipping a sole toward its
      edge is penalised.
    * CoM velocity ``(M_s / M_total) * (v_d + w_d x C_s)`` where ``M_s``
      and ``C_s`` are the mass and COM of the subtree of the dof's joint.

    Assumes ``jacobian`` is zeroed upstream. Returns without writing for
    inactive (problem, dof) pairs.
    """
    p, d = wp.tid()

    slot = active_slot[p]
    if slot < 0:
        return

    vertex = active_vertex_w[p]
    direction = active_direction[p]

    S = joint_S_s[p, d]
    v = wp.vec3(S[0], S[1], S[2])
    omega = wp.vec3(S[3], S[4], S[5])

    d_vertex = wp.vec3(0.0, 0.0, 0.0)
    if foot_in_subtree[slot, d] != wp.uint8(0):
        d_vertex = v + wp.cross(omega, vertex)

    j_d = dof_to_joint[d]
    M_s = joint_subtree_mass[j_d]
    C_s = joint_subtree_com[p, j_d]
    d_com = (M_s * total_mass_inv) * (v + wp.cross(omega, C_s))

    d_margin = direction[0] * (d_vertex[0] - d_com[0]) + direction[1] * (d_vertex[1] - d_com[1])

    jacobian[p, start_idx, d] = -weight * d_margin


class IKObjectiveStabilityMargin(ik.IKObjective):
    """Hinge penalty on CoM projecting outside the support polygon.

    Residual is zero whenever the mass-weighted CoM's XY projection lies
    inside the convex hull of the feet, and grows linearly with distance
    when outside. This matches the physical static-balance condition
    (CoM must lie over the support polygon) without biasing the IK
    toward the polygon centroid.

    Args:
        cfg: :class:`~.cfg.IKObjectiveStabilityMarginCfg` with ``weight``.
        pipeline: Live :class:`RetargetPipeline` — read for
            ``kin.model`` (body masses, kinematic tree), ``foot_body_ids``
            (slot order), and ``buffer.is_contact_t`` (per-problem active
            contacts snapshotted at IK build time).
        wp_mesh: Unused (kept for uniform construction signature).
    """

    def __init__(
        self,
        cfg: IKObjectiveStabilityMarginCfg,
        pipeline: RetargetPipeline,
        wp_mesh: object = None,
    ) -> None:
        super().__init__()
        self.weight = cfg.weight
        self._foot_body_indices_np = np.asarray(pipeline.foot_body_ids, dtype=np.int32)
        self.n_feet = int(self._foot_body_indices_np.shape[0])
        # Sole outline per foot, in the foot's own xy frame. Derived from the
        # robot's collision geometry, so a flat humanoid sole contributes its
        # full outline and a spherical quadruped foot nearly a point.
        self._hull_local_np = pipeline.kin.foot_contact_hulls(
            list(pipeline.foot_body_ids),
            height_tol=cfg.contact_height_tol,
        )["vertices"]
        self.n_vertices = int(self._hull_local_np.shape[1])
        angles = np.arange(cfg.num_directions, dtype=np.float32) * (2.0 * np.pi / cfg.num_directions)
        self._directions_np = np.stack([np.cos(angles), np.sin(angles)], axis=-1).astype(np.float32)
        self.n_directions = int(cfg.num_directions)
        model = pipeline.kin.model
        self.n_bodies = model.body_count
        self.n_joints = model.joint_count
        bm = model.body_mass.numpy()
        self._total_mass_inv = float(1.0 / (bm.sum() + 1e-10))
        self._body_mass_np = bm.astype(np.float32)
        rel = _build_stability_jac_relations(model, list(pipeline.foot_body_ids))
        self._dof_to_joint_np = rel["dof_to_joint"]
        self._joint_subtree_bodies_np = rel["joint_subtree_bodies"]
        self._joint_subtree_offsets_np = rel["joint_subtree_offsets"]
        self._joint_subtree_mass_np = rel["joint_subtree_mass"]
        self._joint_subtree_inv_mass_np = (1.0 / (self._joint_subtree_mass_np + 1e-10)).astype(np.float32)
        self._foot_in_subtree_np = rel["foot_in_subtree"]
        self._pipeline = pipeline

    def supports_analytic(self) -> bool:
        return True

    def residual_dim(self) -> int:
        return 1

    def init_buffers(self, model: newton.Model, jacobian_mode: ik.IKJacobianType) -> None:
        self._require_batch_layout()
        d = self.device
        n = self.n_batch

        self._foot_body_indices = wp.array(self._foot_body_indices_np, dtype=wp.int32, device=d)
        self._body_mass_dev = wp.array(self._body_mass_np, dtype=wp.float32, device=d)

        # Snapshot ``is_contact`` per problem from the populated buffer.
        # The sampler ran before IK objectives were built, so buffer
        # contents are stable. Slot order matches ``foot_body_indices``.
        import torch  # local import to avoid top-level torch dep

        buf = self._pipeline.buffer
        is_c_u8 = buf.is_contact_t[: n * self.n_feet].view(n, self.n_feet).to(torch.uint8).contiguous()
        self._is_contact_t = is_c_u8  # keep torch reference alive for Warp view
        self._is_contact = wp.from_torch(is_c_u8, dtype=wp.uint8)

        self._hull_local = wp.array(self._hull_local_np, dtype=wp.vec2, device=d)
        self._directions = wp.array(self._directions_np, dtype=wp.vec2, device=d)

        n_slots = self.n_feet * self.n_vertices
        self._scratch_world = wp.zeros(shape=(n, n_slots), dtype=wp.vec3, device=d)
        self._scratch_slot = wp.zeros(shape=(n, n_slots), dtype=wp.int32, device=d)

        # Supporting-vertex cache populated by the residual kernel and read by
        # the analytic Jacobian kernel on the same iteration.
        self._active_slot = wp.zeros(shape=(n,), dtype=wp.int32, device=d)
        self._active_vertex_w = wp.zeros(shape=(n,), dtype=wp.vec3, device=d)
        self._active_direction = wp.zeros(shape=(n,), dtype=wp.vec2, device=d)

        # Per-joint subtree COM (refreshed every iteration in compute_residuals).
        self._dof_to_joint = wp.array(self._dof_to_joint_np, dtype=wp.int32, device=d)
        self._joint_subtree_bodies = wp.array(self._joint_subtree_bodies_np, dtype=wp.int32, device=d)
        self._joint_subtree_offsets = wp.array(self._joint_subtree_offsets_np, dtype=wp.int32, device=d)
        self._joint_subtree_mass = wp.array(self._joint_subtree_mass_np, dtype=wp.float32, device=d)
        self._joint_subtree_inv_mass = wp.array(self._joint_subtree_inv_mass_np, dtype=wp.float32, device=d)
        self._joint_subtree_com_buf = wp.zeros(shape=(n, self.n_joints), dtype=wp.vec3, device=d)
        self._foot_in_subtree = wp.array(self._foot_in_subtree_np, dtype=wp.uint8, device=d)

        # Autodiff scratch only when the solver may take that path.
        self._e_array: wp.array | None = None
        if jacobian_mode == ik.IKJacobianType.AUTODIFF:
            e = np.zeros((n, self.total_residuals), dtype=np.float32)
            for b in range(n):
                e[b, self.residual_offset] = 1.0
            self._e_array = wp.array(e.flatten(), dtype=wp.float32, device=d)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx) -> None:
        n = body_q.shape[0]
        wp.launch(
            _compute_joint_subtree_origin_coms,
            dim=[n, self.n_joints],
            inputs=[
                body_q,
                self._body_mass_dev,
                self._joint_subtree_bodies,
                self._joint_subtree_offsets,
                self._joint_subtree_inv_mass,
            ],
            outputs=[self._joint_subtree_com_buf],
            device=self.device,
        )
        wp.launch(
            _stability_margin_residuals,
            dim=n,
            inputs=[
                body_q,
                self._body_mass_dev,
                self.n_bodies,
                self._foot_body_indices,
                self._is_contact,
                self._hull_local,
                self._directions,
                self._scratch_world,
                self._scratch_slot,
                self.n_feet,
                self.n_vertices,
                self.n_directions,
                self._total_mass_inv,
                self.weight,
                start_idx,
            ],
            outputs=[
                residuals,
                self._active_slot,
                self._active_vertex_w,
                self._active_direction,
            ],
            device=self.device,
        )

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx) -> None:
        """Reads the supporting-vertex cache populated by the most recent
        :meth:`compute_residuals` call. Newton's IK solver always evaluates
        residuals before Jacobians per iteration, so the cache is fresh.
        """
        self._require_batch_layout()
        n_dofs = model.joint_dof_count
        wp.launch(
            _stability_margin_jac_analytic,
            dim=[self.n_batch, n_dofs],
            inputs=[
                self._foot_in_subtree,
                joint_S_s,
                self._dof_to_joint,
                self._joint_subtree_mass,
                self._joint_subtree_com_buf,
                self._total_mass_inv,
                self._active_slot,
                self._active_vertex_w,
                self._active_direction,
                self.weight,
                start_idx,
            ],
            outputs=[jacobian],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof) -> None:
        self._require_batch_layout()
        tape.backward(grads={tape.outputs[0]: self._e_array})
        wp.launch(
            jac_fill_row,
            dim=self.n_batch,
            inputs=[tape.gradients[dq_dof], dq_dof.shape[1], start_idx],
            outputs=[jacobian],
            device=self.device,
        )
        tape.zero()
