# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-based kinematic model wrapper.

Wraps :class:`newton.Model` behind a single :class:`NewtonKinematics`
object that owns the model, ordered body/joint names, and default stance.
The description file is parsed exactly once in ``__init__``; both USD and
URDF sources are accepted.

No IsaacSim dependency -- only Newton + Warp.
"""

from __future__ import annotations

import math
import re

import newton
import newton.ik as ik
import numpy as np
import warp as wp
from newton import GeoType, JointType
from newton._src.sim.ik.ik_common import eval_fk_batched as _newton_eval_fk_batched

from .newton_kinematics_cfg import NewtonKinematicsCfg  # re-exported for backcompat

__all__ = ["NewtonKinematics", "NewtonKinematicsCfg", "add_robot_description"]


def add_robot_description(
    builder: newton.ModelBuilder,
    path: str,
    collapse_fixed_joints: bool = False,
    show_colliders: bool = False,
) -> list[str]:
    """Add a robot description to ``builder`` and return its body names.

    Accepts USD or URDF. The two Newton loaders differ in more than the file
    they read: ``add_urdf`` defaults to a fixed base and reports nothing back,
    whereas ``add_usd`` returns prim-path maps. Both are normalised here so
    callers get a floating-base articulation and ordered body names either way.

    Args:
        builder: Model builder to add the articulation to.
        path: Path to a ``.usd``/``.usda`` or ``.urdf`` description.
        collapse_fixed_joints: Merge fixed joints into their parent body.
        show_colliders: Render collision geometry in place of visual meshes.
            Only honoured for URDF sources.

    Returns:
        Body names ordered to match the builder's body indices.
    """
    if path.lower().endswith(".urdf"):
        builder.add_urdf(
            path,
            floating=True,
            collapse_fixed_joints=collapse_fixed_joints,
            hide_visuals=show_colliders,
            force_show_colliders=show_colliders,
        )
        return [label.rsplit("/", 1)[-1] for label in builder.body_label]

    result = builder.add_usd(path, collapse_fixed_joints=collapse_fixed_joints)
    path_body_map: dict[str, int] = result.get("path_body_map", {})
    names = [""] * (max(path_body_map.values(), default=-1) + 1)
    for prim_path, index in path_body_map.items():
        names[index] = prim_path.rsplit("/", 1)[-1]
    return names


def _fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` approximately-uniform points on the unit sphere."""
    if n <= 1:
        return np.array([[0.0, 0.0, -1.0]])
    index = np.arange(n, dtype=float)
    y = 1.0 - 2.0 * index / float(n - 1)
    radius = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = np.pi * (3.0 - np.sqrt(5.0)) * index
    return np.stack([np.cos(theta) * radius, y, np.sin(theta) * radius], axis=-1)


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull of 2D ``points``, via monotone chain.

    Degenerate inputs are handled the way callers want: a single cluster
    collapses to one vertex and a collinear set to its two extremes, so
    point-like feet need no special case.

    Args:
        points: ``[N, 2]`` planar points [m].

    Returns:
        ``[H, 2]`` hull vertices in counter-clockwise order.
    """
    unique = np.unique(np.asarray(points, dtype=np.float64).reshape(-1, 2), axis=0)
    if len(unique) <= 2:
        return unique.astype(np.float32)

    order = np.lexsort((unique[:, 1], unique[:, 0]))
    ordered = unique[order]

    def _half(seq: np.ndarray) -> list[np.ndarray]:
        chain: list[np.ndarray] = []
        for point in seq:
            while len(chain) >= 2:
                a, b = chain[-2], chain[-1]
                if (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]) > 0.0:
                    break
                chain.pop()
            chain.append(point)
        return chain

    lower = _half(ordered)
    upper = _half(ordered[::-1])
    hull = np.array(lower[:-1] + upper[:-1], dtype=np.float32)
    return hull if len(hull) else ordered[:1].astype(np.float32)


def _thin_hull(hull: np.ndarray, max_vertices: int) -> np.ndarray:
    """Reduce ``hull`` to ``max_vertices`` by repeatedly dropping the flattest corner.

    The dropped vertex is the one nearest the line joining its neighbours, so
    each removal moves the outline as little as possible.

    Args:
        hull: ``[H, 2]`` counter-clockwise hull vertices [m].
        max_vertices: Number of vertices to keep.

    Returns:
        ``[max_vertices, 2]`` thinned hull, still counter-clockwise.
    """
    kept = [np.asarray(v, dtype=np.float64) for v in hull]
    while len(kept) > max_vertices:
        deviations = []
        for i in range(len(kept)):
            previous, current, following = kept[i - 1], kept[i], kept[(i + 1) % len(kept)]
            edge = following - previous
            edge_length = float(np.hypot(edge[0], edge[1]))
            offset = current - previous
            cross = abs(edge[0] * offset[1] - edge[1] * offset[0])
            deviations.append(cross / edge_length if edge_length > 1.0e-12 else float(np.hypot(*offset)))
        kept.pop(int(np.argmin(deviations)))
    return np.asarray(kept, dtype=np.float32)


class NewtonKinematics:
    """Newton kinematic model built from a USD file.

    Owns the :class:`newton.Model`, ordered body/joint name lists, and
    the default stance (computed via FK at construction time).

    Args:
        cfg: Kinematics configuration.
    """

    model: newton.Model
    """Finalized Newton model."""

    usd_path: str
    """Absolute path to the USD file used to build this model."""

    body_names: list[str]
    """Ordered body names (index ``i`` corresponds to Newton body ``i``)."""

    joint_names: list[str]
    """Ordered joint names (index ``i`` corresponds to Newton joint ``i``)."""

    def __init__(self, cfg: NewtonKinematicsCfg):
        self.cfg = cfg
        self.usd_path = str(cfg.usd_path)

        self.builder = newton.ModelBuilder()
        if self.usd_path.lower().endswith(".urdf"):
            self.body_names = add_robot_description(
                self.builder, self.usd_path, collapse_fixed_joints=cfg.collapse_fixed_joints
            )
            self.model = self.builder.finalize(device=cfg.device)
            self.joint_names = [label.rsplit("/", 1)[-1] for label in self.builder.joint_label]
        else:
            result = self.builder.add_usd(self.usd_path, collapse_fixed_joints=cfg.collapse_fixed_joints)
            self.model = self.builder.finalize(device=cfg.device)

            path_body_map: dict[str, int] = result.get("path_body_map", {})
            names = [""] * self.model.body_count
            for path, idx in path_body_map.items():
                names[idx] = path.rsplit("/", 1)[-1]
            self.body_names = names

            path_joint_map: dict[str, int] = result.get("path_joint_map", {})
            jnames = [""] * self.model.joint_count
            for path, idx in path_joint_map.items():
                jnames[idx] = path.rsplit("/", 1)[-1]
            self.joint_names = jnames

        # Root coordinate count: 7 for a free-floating base (3 position + 4
        # quaternion), 0 for a fixed base. Non-root joints occupy
        # ``joint_q[n_root_coords:]``. Reading it from the model (instead of
        # assuming a free root) lets the same wrapper drive fixed-base arms.
        self._n_root_coords = self._compute_root_coord_count()

        jq = self.model.joint_q.numpy().copy()
        if self._n_root_coords >= 7:
            # Free-floating base only: the first 7 coords are the root pose.
            jq[0:3] = cfg.default_pos
            jq[3:7] = cfg.default_quat
        if cfg.default_joint_pos is not None:
            resolved = self._resolve_joint_pos_map(cfg.default_joint_pos)
            n = min(len(resolved), len(jq) - self._n_root_coords)
            jq[self._n_root_coords : self._n_root_coords + n] = resolved[:n]
        state = self.eval_fk(wp.array(jq, dtype=float, device=cfg.device))
        self._default_joint_q = jq
        self._default_body_q = state.body_q.numpy()

    def _resolve_joint_pos_map(self, joint_pos_map: dict[str, float]) -> np.ndarray:
        """Resolve a ``{regex: value}`` dict to a flat joint position array.

        Uses ``joint_q_start`` to map each matched joint to its actual
        position in ``joint_q[n_root_coords:]`` (the non-root coordinates),
        accepting single-DoF position joints (revolute or prismatic) and
        skipping fixed/ball/free joints that carry no scalar default here.
        """
        n_root = self._n_root_coords
        n_coords = self.model.joint_coord_count - n_root
        jpos = np.zeros(n_coords, dtype=np.float32)
        q_start = self.model.joint_q_start.numpy()
        joint_type = self.model.joint_type.numpy()
        single_dof = (int(JointType.PRISMATIC), int(JointType.REVOLUTE))
        for pattern, value in joint_pos_map.items():
            regex = re.compile(pattern)
            for jidx in range(1, len(self.joint_names)):
                if not regex.fullmatch(self.joint_names[jidx]):
                    continue
                if int(joint_type[jidx]) not in single_dof:
                    continue
                qi = int(q_start[jidx]) - n_root
                if 0 <= qi < n_coords:
                    jpos[qi] = value
        return jpos

    def _compute_root_coord_count(self) -> int:
        """Number of ``joint_q`` coordinates consumed by the root joint.

        ``7`` for a free-floating base (joint 0 is a ``FREE`` joint: 3
        position + 4 quaternion), ``0`` for a fixed base (joint 0 is
        ``FIXED``). Derived from ``joint_q_start`` so it reflects the actual
        model layout regardless of the root joint type.
        """
        if self.model.joint_count <= 1:
            return int(self.model.joint_coord_count)
        q_start = self.model.joint_q_start.numpy()
        return int(q_start[1] - q_start[0])

    @property
    def device(self) -> str:
        return str(self.model.device)

    @property
    def n_root_coords(self) -> int:
        """Coordinates the root joint consumes (7 free-floating, 0 fixed-base)."""
        return self._n_root_coords

    @property
    def default_joint_q(self) -> np.ndarray:
        """Default joint coordinates ``[joint_coord_count]`` (from FK at init)."""
        return self._default_joint_q

    @property
    def default_body_q(self) -> np.ndarray:
        """Default body transforms ``[body_count, 7]`` (from FK at init)."""
        return self._default_body_q

    def find_body_indices(self, names: list[str]) -> list[int]:
        """Resolve body names to Newton body indices.

        Args:
            names: Body name strings (exact match).

        Returns:
            Corresponding Newton body indices.

        Raises:
            ValueError: If any name is not found.
        """
        indices = []
        for name in names:
            if name not in self.body_names:
                raise ValueError(f"Body '{name}' not found. Available: {self.body_names}")
            indices.append(self.body_names.index(name))
        return indices

    def find_joint_dof_indices(self, pattern: str) -> list[int]:
        """Find revolute-joint DOF indices matching a regex pattern.

        Returns indices into ``joint_q[n_root_coords:]`` (i.e. excluding the
        root coordinates).  Uses ``joint_q_start`` for correct mapping even
        when the model contains non-revolute joints.

        Args:
            pattern: Regex matched against each joint name.

        Returns:
            Sorted list of matching DOF indices.
        """
        regex = re.compile(pattern)
        q_start = self.model.joint_q_start.numpy()
        joint_type = self.model.joint_type.numpy()
        indices = []
        for jidx in range(1, len(self.joint_names)):
            if int(joint_type[jidx]) != 1:
                continue
            if regex.fullmatch(self.joint_names[jidx]):
                indices.append(int(q_start[jidx]) - self._n_root_coords)
        return sorted(indices)

    def foot_geometry(self, foot_body_ids: list[int]) -> dict[str, np.ndarray | float]:
        """Derive foot geometry from the default stance + collision shapes.

        ``foot_ground_offset`` is the z offset from the foot body's origin
        to the lowest point of its collision geometry — a pure-geometric
        quantity independent of the URDF default pose. Pipeline uses it
        to lift contact targets by this offset so IK places the foot's
        *sole* (not body origin) on the terrain surface.

        Args:
            foot_body_ids: Newton body indices for the feet.

        Returns:
            Dict with ``foot_offsets`` (body-to-base xyz at default),
            ``standing_height`` (default base-z minus default foot-mean-z),
            ``foot_ground_offset`` (negated min local-z of foot collision
            geometry, fallback to default foot-z if no shapes attached).
        """
        base_pos = self._default_body_q[0][:3]
        foot_pos = np.array([self._default_body_q[fid][:3] for fid in foot_body_ids])

        # Per-foot local-z-min from attached collision shapes. For each
        # shape type, compute the lowest-z offset the shape reaches in the
        # body frame, honouring the shape's own rotation.
        builder = self.builder
        foot_ids_set = set(int(f) for f in foot_body_ids)
        z_min_local: float | None = None
        for si in range(len(builder.shape_body)):
            bid = int(builder.shape_body[si])
            if bid not in foot_ids_set:
                continue
            zmin = self._shape_local_z_min(
                int(builder.shape_type[si]),
                builder.shape_scale[si],
                builder.shape_transform[si],
                builder.shape_source[si],
            )
            if zmin is None:
                continue
            if z_min_local is None or zmin < z_min_local:
                z_min_local = zmin

        if z_min_local is not None:
            # foot_body_z + (-z_min_local) = terrain_z  →  sole on terrain.
            foot_ground_offset = float(-z_min_local)
        else:
            # Fallback: URDF default pose (assumes default places soles at z = 0).
            foot_ground_offset = float(foot_pos[:, 2].min())

        return {
            "foot_offsets": foot_pos - base_pos,
            "standing_height": float(base_pos[2] - foot_pos[:, 2].mean()),
            "foot_ground_offset": foot_ground_offset,
        }

    def foot_contact_hulls(
        self,
        foot_body_ids: list[int],
        height_tol: float = 0.001,
        max_vertices: int = 8,
        samples_per_shape: int = 128,
    ) -> dict[str, np.ndarray]:
        """Per-foot sole outline, as a 2D convex hull in the foot's own frame.

        Samples the surface of every collision shape attached to each foot,
        keeps the points lying within :paramref:`height_tol` of that foot's
        lowest point, and returns the convex hull of their xy projection. This
        is the patch of the foot that can bear load on flat ground, so it is
        what the support region should be built from.

        The result is derived from whatever collision geometry the robot
        ships, so it needs no per-robot constants and degenerates on its own:
        a spherical foot yields a hull barely wider than a point, while a flat
        humanoid sole yields its full outline.

        Args:
            foot_body_ids: Newton body indices for the feet.
            height_tol: How far above a foot's lowest point a sample may sit
                and still count as sole [m], standing in for how far the foot
                deforms under load. A flat sole is insensitive to it (G1
                measures 20.5 cm at 0.2 mm and 20.8 cm at 5 mm), while a
                curved one scales with it (ANYmal-C's spherical foot spans
                0.6 cm at 0.2 mm and 3.4 cm at 5 mm), which is the intended
                distinction between a foot that lies flat and one that
                touches at a point.
            max_vertices: Upper bound on hull vertices retained per foot.
                Hulls with more are thinned corner by corner; hulls with fewer
                repeat their last vertex as padding.
            samples_per_shape: Surface samples drawn per collision shape.

        Returns:
            Dict with ``vertices`` (``[n_feet, max_vertices, 2]`` xy offsets in
            the foot frame [m]), ``counts`` (``[n_feet]`` real vertex count
            before padding), and ``sole_z`` (``[n_feet]`` body-frame z of each
            foot's contact plane [m]).

        Raises:
            ValueError: If a foot body carries no usable collision geometry.
        """
        builder = self.builder
        body_shapes: dict[int, list[int]] = {}
        for si in range(len(builder.shape_body)):
            body_shapes.setdefault(int(builder.shape_body[si]), []).append(si)

        vertices = np.zeros((len(foot_body_ids), max_vertices, 2), dtype=np.float32)
        counts = np.zeros(len(foot_body_ids), dtype=np.int32)
        sole_z = np.zeros(len(foot_body_ids), dtype=np.float32)

        for slot, body_id in enumerate(foot_body_ids):
            points: list[np.ndarray] = []
            for si in body_shapes.get(int(body_id), []):
                sampled = self._shape_surface_points(
                    int(builder.shape_type[si]),
                    builder.shape_scale[si],
                    builder.shape_transform[si],
                    builder.shape_source[si],
                    samples_per_shape,
                )
                if sampled is not None and len(sampled):
                    points.append(sampled)
            if not points:
                name = self.body_names[int(body_id)]
                raise ValueError(f"Foot body '{name}' has no collision geometry to derive a contact hull from.")

            cloud = np.concatenate(points, axis=0)
            z_min = float(cloud[:, 2].min())
            sole = cloud[cloud[:, 2] <= z_min + height_tol]
            hull = _convex_hull_2d(sole[:, :2])
            if len(hull) > max_vertices:
                hull = _thin_hull(hull, max_vertices)

            counts[slot] = len(hull)
            sole_z[slot] = z_min
            vertices[slot, : len(hull)] = hull
            # Pad by repeating the last vertex. Consumers take a maximum over
            # vertices, which duplicates leave unchanged, so padding needs no
            # masking at the use site.
            vertices[slot, len(hull) :] = hull[-1]

        return {"vertices": vertices, "counts": counts, "sole_z": sole_z}

    @staticmethod
    def _shape_surface_points(
        shape_type: int,
        shape_scale,
        shape_transform,
        shape_source,
        n_samples: int,
    ) -> np.ndarray | None:
        """Surface samples of one collision shape, in body frame [m].

        Applies the shape's full transform, rotation included, so shapes
        attached at an angle land where they actually are.

        Returns ``None`` for geometry types this does not model.
        """
        xform = np.asarray(shape_transform, dtype=float).reshape(-1)
        translation = xform[:3]
        rotation = NewtonKinematics._rotation_matrix(xform[3:7])
        scale = np.asarray(shape_scale, dtype=float).reshape(-1)

        def to_body(local: np.ndarray) -> np.ndarray:
            return local @ rotation.T + translation

        if shape_type in (int(GeoType.MESH), int(GeoType.CONVEX_MESH)):
            if shape_source is None or not hasattr(shape_source, "vertices"):
                return None
            verts = np.asarray(shape_source.vertices, dtype=float).reshape(-1, 3)
            if verts.size == 0:
                return None
            return to_body(verts * scale[:3])

        if shape_type == int(GeoType.SPHERE):
            return to_body(_fibonacci_sphere(n_samples) * float(scale[0]))

        if shape_type == int(GeoType.BOX):
            half = 0.5 * scale[:3]
            signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
            return to_body(signs * half)

        if shape_type in (int(GeoType.CAPSULE), int(GeoType.CYLINDER)):
            radius, half_height = float(scale[0]), float(scale[1])
            n_angular = max(8, int(math.sqrt(n_samples)) * 2)
            n_axial = max(2, n_samples // n_angular)
            angles = np.linspace(0.0, 2.0 * np.pi, n_angular, endpoint=False)
            rim = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=-1)
            # Lateral surface: the load-bearing face when the shape lies on its
            # side, which end-cap samples alone would miss.
            axial = np.linspace(-half_height, half_height, n_axial)
            lateral = np.concatenate(
                [np.column_stack([rim, np.full(n_angular, z)]) for z in axial],
                axis=0,
            )
            if shape_type == int(GeoType.CAPSULE):
                cap = _fibonacci_sphere(max(8, n_samples // 4)) * radius
                caps = np.concatenate([cap + [0.0, 0.0, half_height], cap - [0.0, 0.0, half_height]], axis=0)
                return to_body(np.concatenate([lateral, caps], axis=0))
            return to_body(lateral)

        return None

    @staticmethod
    def _rotation_matrix(quat_xyzw) -> np.ndarray:
        """Return the 3x3 rotation matrix for a Newton ``(x, y, z, w)`` quaternion."""
        x, y, z, w = (float(v) for v in np.asarray(quat_xyzw, dtype=float).reshape(-1)[:4])
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1.0e-12:
            return np.eye(3)
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ]
        )

    @staticmethod
    def _shape_local_z_min(
        shape_type: int,
        shape_scale,
        shape_transform,
        shape_source,
    ) -> float | None:
        """Lowest body-frame z coordinate reachable by a shape's surface.

        Handles the geometry primitives we encounter in practice (mesh,
        sphere, box, capsule, cylinder). Returns ``None`` for unsupported
        types so the caller can fall back or skip.

        The shape's own rotation is applied: primitives are commonly attached
        rotated (foot soles are often cylinders laid flat along the body x
        axis), and treating their local axis as body-frame z overstates how far
        they reach below the body origin by roughly their half-length.
        """
        xform = np.asarray(shape_transform, dtype=float).reshape(-1)
        pos_z = float(xform[2])
        # Body-frame straight-down direction, expressed in the shape's local
        # frame. Support distance along it is how far the shape reaches below
        # its own origin.
        rotation = NewtonKinematics._rotation_matrix(xform[3:7])
        down = -rotation[2, :]

        if shape_type == int(GeoType.MESH) or shape_type == int(GeoType.CONVEX_MESH):
            if shape_source is None or not hasattr(shape_source, "vertices"):
                return None
            verts = np.asarray(shape_source.vertices, dtype=float).reshape(-1, 3)
            if verts.size == 0:
                return None
            scaled = verts * np.asarray(shape_scale, dtype=float).reshape(-1)[:3]
            return pos_z + float((scaled @ rotation[2, :]).min())
        if shape_type == int(GeoType.SPHERE):
            return pos_z - float(shape_scale[0])
        if shape_type == int(GeoType.BOX):
            half_extents = 0.5 * np.asarray(shape_scale, dtype=float).reshape(-1)[:3]
            return pos_z - float(np.abs(down) @ half_extents)
        if shape_type == int(GeoType.CAPSULE):
            radius, half_height = float(shape_scale[0]), float(shape_scale[1])
            return pos_z - (half_height * abs(down[2]) + radius)
        if shape_type == int(GeoType.CYLINDER):
            radius, half_height = float(shape_scale[0]), float(shape_scale[1])
            # Flat end caps: the rim contributes the radius scaled by how much
            # the cylinder's axis is tilted away from the query direction.
            return pos_z - (half_height * abs(down[2]) + radius * math.hypot(down[0], down[1]))
        return None

    def create_ik_solver(
        self,
        objectives: list,
        n_problems: int,
        jacobian_mode: ik.IKJacobianType = ik.IKJacobianType.ANALYTIC,
    ) -> ik.IKSolver:
        """Create an IK solver from user-provided objectives.

        Args:
            objectives: List of IK objectives (position, rotation,
                joint limit, etc.).
            n_problems: Number of parallel IK problems.
            jacobian_mode: Jacobian backend.  Use ``MIXED`` when
                combining analytic objectives with autodiff-only
                objectives.

        Returns:
            Configured :class:`newton.ik.IKSolver`.
        """
        return ik.IKSolver(
            model=self.model,
            n_problems=n_problems,
            objectives=objectives,
            optimizer=ik.IKOptimizer.LM,
            jacobian_mode=jacobian_mode,
        )

    def eval_fk(self, joint_q: wp.array, joint_qd: wp.array | None = None) -> newton.State:
        """Run forward kinematics for a single articulation.

        Args:
            joint_q: Joint coordinates [m or rad].
            joint_qd: Joint velocities (zeros if ``None``).

        Returns:
            Newton state with ``body_q`` populated.
        """
        state = self.model.state()
        if joint_qd is None:
            joint_qd = wp.zeros(self.model.joint_dof_count, dtype=float, device=self.device)
        newton.eval_fk(self.model, joint_q, joint_qd, state)
        return state

    def eval_fk_batched(
        self,
        joint_q: wp.array,
        joint_qd: wp.array | None = None,
        body_q: wp.array | None = None,
        body_qd: wp.array | None = None,
    ) -> tuple[wp.array, wp.array]:
        """Run batched forward kinematics across ``N`` problems on the shared model.

        Wraps Newton's internal batched-FK kernel. All array arguments
        use a leading-``N`` batch dimension over the ``N`` parallel
        problems; all share the same kinematic model. Output arrays are
        allocated lazily when ``None``.

        Args:
            joint_q: Joint coordinates per problem, shape
                ``[N, joint_coord_count]`` [m or rad].
            joint_qd: Joint velocities per problem, shape
                ``[N, joint_dof_count]`` [m/s or rad/s]. Zero-filled if ``None``.
            body_q: Optional pre-allocated output for body transforms,
                shape ``[N, body_count]`` of :class:`warp.transformf`. Allocated if ``None``.
            body_qd: Optional pre-allocated output for body spatial velocities,
                shape ``[N, body_count]`` of :class:`warp.spatial_vectorf`. Allocated if ``None``.

        Returns:
            Tuple ``(body_q, body_qd)`` -- the (possibly freshly allocated) output arrays.
        """
        n = joint_q.shape[0]
        if joint_qd is None:
            joint_qd = wp.zeros((n, self.model.joint_dof_count), dtype=wp.float32, device=self.device)
        if body_q is None:
            body_q = wp.zeros((n, self.model.body_count), dtype=wp.transformf, device=self.device)
        if body_qd is None:
            body_qd = wp.zeros((n, self.model.body_count), dtype=wp.spatial_vectorf, device=self.device)
        _newton_eval_fk_batched(self.model, joint_q, joint_qd, body_q, body_qd)
        return body_q, body_qd
