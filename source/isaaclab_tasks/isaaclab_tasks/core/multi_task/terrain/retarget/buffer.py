# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pre-allocated GPU buffer for the retargeting pipeline.

Single contiguous ``float32`` allocation, field-major layout.
Each field occupies a contiguous slice so warp views are zero-copy.
"""

from __future__ import annotations

import torch
import warp as wp


@wp.kernel
def _scatter_contacts(
    src: wp.array(dtype=wp.vec3),
    dst: wp.array(dtype=wp.vec3),
    n_candidates: int,
    n_contacts: int,
):
    """Deinterleave: src[cand*nc + foot] -> dst[foot*N + cand]."""
    i = wp.tid()
    dst[i] = src[(i % n_candidates) * n_contacts + i // n_candidates]


@wp.kernel
def _scatter_contact_rotations(
    src: wp.array(dtype=wp.vec4),
    dst: wp.array(dtype=wp.vec4),
    n_candidates: int,
    n_contacts: int,
):
    """Deinterleave rotations: src[cand*nc + foot] -> dst[foot*N + cand]."""
    i = wp.tid()
    dst[i] = src[(i % n_candidates) * n_contacts + i // n_candidates]


class RetargetBuffer:
    """Pre-allocated GPU buffer shared across all pipeline stages.

    One ``float32`` tensor holds all float data in field-major order
    (each field is a contiguous block).  Named properties return
    zero-copy torch / warp views into those blocks.

    Bool masks and int selection indices are separate small tensors
    (different dtype).

    The buffer is allocated once at ``max_candidates`` capacity and
    reused across runs via :meth:`reset`. ``max_candidates`` here is the
    *post-FPS* IK-stage workload (what the sampler actually writes after
    its polygon-FPS thinning), not the pre-FPS polygon pool size — see
    :attr:`SamplerSizing.ik_capacity` vs.
    :attr:`SamplerSizing.max_neighborhoods`.
    """

    def __init__(
        self,
        max_candidates: int,
        joint_coord_count: int,
        num_bodies: int,
        num_contacts: int,
        device: str = "cuda:0",
    ):
        self.max_candidates = max_candidates
        self.joint_coord_count = joint_coord_count
        self.num_bodies = num_bodies
        self.num_contacts = num_contacts
        self.device = device

        n = max_candidates
        nc, jc, nb = num_contacts, joint_coord_count, num_bodies

        # Field sizes in floats
        sz_ct = n * nc * 3  # contact_targets
        sz_ji = n * jc  # joint_q_init
        sz_jr = n * jc  # joint_q_result
        sz_bq = n * nb * 7  # body_q
        sz_bp = n * 3  # base_target_pos
        sz_br = n * 4  # base_target_rot
        sz_cr = n * nc * 4  # contact_target_rot

        # Cumulative offsets
        self._o_ct = 0
        self._o_ji = sz_ct
        self._o_jr = sz_ct + sz_ji
        self._o_bq = sz_ct + sz_ji + sz_jr
        self._o_bp = sz_ct + sz_ji + sz_jr + sz_bq
        self._o_br = sz_ct + sz_ji + sz_jr + sz_bq + sz_bp
        self._o_cr = sz_ct + sz_ji + sz_jr + sz_bq + sz_bp + sz_br
        total = sz_ct + sz_ji + sz_jr + sz_bq + sz_bp + sz_br + sz_cr

        torch_dev = torch.device(device)
        self._data = torch.zeros(total, dtype=torch.float32, device=torch_dev)

        # Masks and selection (non-float)
        self._geom_valid = torch.zeros(n, dtype=torch.bool, device=torch_dev)
        self._ik_valid = torch.zeros(n, dtype=torch.bool, device=torch_dev)
        self._selected = torch.zeros(n, dtype=torch.int32, device=torch_dev)
        # Per-slot contact flag (flat, matching contact_targets_t layout).
        # Default ``True`` is the hard-contact fallback; the sampler
        # overwrites it per slot to mark air targets.
        self._is_contact = torch.ones(n * nc, dtype=torch.bool, device=torch_dev)

        # Bookkeeping
        self.num_selected: int = 0
        self.num_written: int = 0
        self.num_geometry_valid: int = 0
        self.num_ik_valid: int = 0
        self.num_final_valid: int = 0

    # -- Torch views (zero-copy, contiguous) --

    @property
    def contact_targets_t(self) -> torch.Tensor:
        """``[max_candidates * num_contacts, 3]``."""
        s = self._o_ct
        return self._data[s : s + self.max_candidates * self.num_contacts * 3].view(-1, 3)

    @property
    def contact_target_rot_t(self) -> torch.Tensor:
        """``[max_candidates * num_contacts, 4]`` xyzw target foot orientations.

        The orientation the terrain demands of each foot: ``+z`` along the
        surface normal under its contact target and ``+x`` along the heading
        the patch admits. Slots the sampler marks as air keep the identity
        rotation, which the rotation objective is not applied to.
        """
        s = self._o_cr
        return self._data[s : s + self.max_candidates * self.num_contacts * 4].view(-1, 4)

    @property
    def is_contact_t(self) -> torch.Tensor:
        """Per-slot contact flag, shape ``[max_candidates * num_contacts]``, bool.

        ``True`` means the slot counts as a support point (stability /
        collision / foot-error criteria). ``False`` marks an air-targeted
        slot whose target is a kinematic reference, not a contact. The
        default ``True`` preserves current behavior.
        """
        return self._is_contact

    @property
    def joint_q_init_t(self) -> torch.Tensor:
        """``[max_candidates, joint_coord_count]``."""
        s = self._o_ji
        return self._data[s : s + self.max_candidates * self.joint_coord_count].view(
            self.max_candidates,
            self.joint_coord_count,
        )

    @property
    def joint_q_result_t(self) -> torch.Tensor:
        """``[max_candidates, joint_coord_count]``."""
        s = self._o_jr
        return self._data[s : s + self.max_candidates * self.joint_coord_count].view(
            self.max_candidates,
            self.joint_coord_count,
        )

    @property
    def body_q_t(self) -> torch.Tensor:
        """``[max_candidates * num_bodies, 7]``."""
        s = self._o_bq
        return self._data[s : s + self.max_candidates * self.num_bodies * 7].view(-1, 7)

    @property
    def base_target_pos_t(self) -> torch.Tensor:
        """``[max_candidates, 3]``."""
        s = self._o_bp
        return self._data[s : s + self.max_candidates * 3].view(-1, 3)

    @property
    def base_target_rot_t(self) -> torch.Tensor:
        """``[max_candidates, 4]``."""
        s = self._o_br
        return self._data[s : s + self.max_candidates * 4].view(-1, 4)

    # -- Warp views (zero-copy, contiguous) --

    @property
    def contact_targets(self) -> wp.array:
        """``[max_candidates * num_contacts]`` of ``vec3``."""
        return wp.from_torch(self.contact_targets_t, dtype=wp.vec3)

    @property
    def contact_target_rot(self) -> wp.array:
        """``[max_candidates * num_contacts]`` of ``vec4`` (xyzw)."""
        return wp.from_torch(self.contact_target_rot_t, dtype=wp.vec4)

    @property
    def joint_q_init(self) -> wp.array:
        """``[max_candidates, joint_coord_count]``."""
        return wp.from_torch(self.joint_q_init_t)

    @property
    def joint_q_result(self) -> wp.array:
        """``[max_candidates, joint_coord_count]``."""
        return wp.from_torch(self.joint_q_result_t)

    @property
    def body_q(self) -> wp.array:
        """``[max_candidates * num_bodies]`` of ``transformf``."""
        return wp.from_torch(self.body_q_t, dtype=wp.transformf)

    @property
    def base_target_pos(self) -> wp.array:
        """``[max_candidates]`` of ``vec3``."""
        return wp.from_torch(self.base_target_pos_t, dtype=wp.vec3)

    @property
    def base_target_rot(self) -> wp.array:
        """``[max_candidates]`` of ``vec4``."""
        return wp.from_torch(self.base_target_rot_t, dtype=wp.vec4)

    @property
    def geometry_valid(self) -> wp.array:
        return wp.from_torch(self._geom_valid)

    @property
    def ik_valid(self) -> wp.array:
        return wp.from_torch(self._ik_valid)

    @property
    def selected(self) -> wp.array:
        return wp.from_torch(self._selected)

    def scatter_contact_targets(
        self,
        objectives: list,
        n_active: int,
        src_offset: int = 0,
    ) -> None:
        """Deinterleave contact targets into per-objective warp arrays.

        The buffer stores contacts interleaved as
        ``[c0_f0, c0_f1, ..., c1_f0, c1_f1, ...]``.
        This method scatters them into each objective's contiguous
        ``target_positions`` array.

        Args:
            objectives: IK position objectives (one per contact body).
            n_active: Number of active candidates.
            src_offset: Starting candidate index into the buffer's
                ``contact_targets`` slab. Combined with ``n_active`` so
                the method can scatter a sliding window of rows for
                chunked IK without copying the unchunked source first.
        """
        nc = self.num_contacts
        ct = self.contact_targets
        if src_offset != 0:
            ct = ct[src_offset * nc : (src_offset + n_active) * nc]
        flat_dst = wp.zeros(nc * n_active, dtype=wp.vec3, device=self.device)
        wp.launch(
            _scatter_contacts,
            dim=nc * n_active,
            inputs=[ct, flat_dst, n_active, nc],
            device=self.device,
        )
        for f_idx, obj in enumerate(objectives):
            wp.copy(obj.target_positions, flat_dst, src_offset=f_idx * n_active, count=n_active)

    def scatter_contact_rotations(
        self,
        objectives: list,
        n_active: int,
        src_offset: int = 0,
    ) -> None:
        """Deinterleave contact target rotations into per-objective warp arrays.

        Mirrors :meth:`scatter_contact_targets` for the orientation each foot
        must adopt to lie on its patch.

        Args:
            objectives: IK rotation objectives (one per contact body).
            n_active: Number of active candidates.
            src_offset: Starting candidate index into the buffer's rotation slab.
        """
        if not objectives:
            return
        nc = self.num_contacts
        src = self.contact_target_rot
        if src_offset != 0:
            src = src[src_offset * nc : (src_offset + n_active) * nc]
        flat_dst = wp.zeros(nc * n_active, dtype=wp.vec4, device=self.device)
        wp.launch(
            _scatter_contact_rotations,
            dim=nc * n_active,
            inputs=[src, flat_dst, n_active, nc],
            device=self.device,
        )
        for f_idx, obj in enumerate(objectives):
            wp.copy(obj.target_rotations, flat_dst, src_offset=f_idx * n_active, count=n_active)

    def reset(self) -> None:
        """Zero masks and counters for a new pipeline run."""
        self._geom_valid.zero_()
        self._ik_valid.zero_()
        self._selected.zero_()
        # Default every slot to hard-contact; the sampler overwrites per
        # slot to mark air targets.
        self._is_contact.fill_(True)
        self.num_selected = 0
        self.num_written = 0
        self.num_geometry_valid = 0
        self.num_ik_valid = 0
        self.num_final_valid = 0

    def set_selected(self, indices: torch.Tensor) -> None:
        """Write the selected-slot list and ``num_selected`` in one call.

        ``indices`` are stored at ``self._selected[:n]`` cast to ``int32``
        (the buffer's storage dtype) and ``self.num_selected`` is set to
        ``n = indices.shape[0]``. Used by FPS-thinning helpers so they
        don't have to reach into the private ``_selected`` field.
        """
        n = int(indices.shape[0])
        if n > 0:
            self._selected[:n] = indices.to(torch.int32)
        self.num_selected = n

    @property
    def memory_bytes(self) -> int:
        """Approximate GPU memory used by this buffer [bytes]."""
        return (
            self._data.nelement() * 4
            + self._geom_valid.nelement()
            + self._ik_valid.nelement()
            + self._selected.nelement() * 4
            + self._is_contact.nelement()
        )
