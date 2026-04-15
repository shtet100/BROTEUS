"""
BROTEUS Grasp Intelligence System
====================================

Layers 2-5 of the pipeline — the core IP.

    Layer 2: Spatial Grid Projection
    Layer 3: Grasp Affordance Scoring
    Layer 4: Hand Configuration Planning
    Layer 5: Manipulation Intelligence

The novel contribution: the moment an object is detected, the grasp affordance
grid is projected simultaneously. Detection and grasp planning fused into a
single real-time pass. The spatial grid scoring algorithm evaluates hundreds
of candidate contact points against physics-based stability criteria in real-time.

Existing work treats vision and grasp planning as separate stages.
BROTEUS fuses them.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from broteus.core.config import GraspConfig
from broteus.detection.detector import Detection, BoundingBox, GraspCategory

logger = logging.getLogger("broteus.grasp")


# ══════════════════════════════════════════════════════════════
# Layer 2: Spatial Grid Projection
# ══════════════════════════════════════════════════════════════

@dataclass
class ContactPoint:
    """A single candidate contact point on the object surface.

    Each point in the spatial grid gets scored across multiple
    physics-based criteria to determine grasp viability.
    """
    position: np.ndarray          # 3D position (x, y, z) in world frame
    surface_normal: np.ndarray    # Outward-facing surface normal at this point
    curvature: float              # Local surface curvature (1/radius, higher = more curved)

    # Scores (computed in Layer 3)
    friction_score: float = 0.0   # How well this point resists slip [0, 1]
    stability_score: float = 0.0  # Contribution to grasp stability [0, 1]
    force_score: float = 0.0      # Force efficiency at this point [0, 1]
    composite_score: float = 0.0  # Weighted combination [0, 1]

    # Classification
    zone: str = "risk"            # "optimal", "acceptable", "risk"

    def to_dict(self) -> dict:
        return {
            "position": self.position.tolist(),
            "normal": self.surface_normal.tolist(),
            "curvature": round(self.curvature, 4),
            "scores": {
                "friction": round(self.friction_score, 4),
                "stability": round(self.stability_score, 4),
                "force": round(self.force_score, 4),
                "composite": round(self.composite_score, 4),
            },
            "zone": self.zone,
        }


class ObjectGeometry(Enum):
    """Parameterized object geometry types for grid projection."""
    BOX = "box"
    CYLINDER = "cylinder"
    SPHERE = "sphere"
    TORUS_HANDLE = "torus_handle"    # Mug handle, tool grip ring
    COMPOSITE = "composite"          # Multi-part (wrench = cylinder + torus)


@dataclass
class SpatialGrid:
    """The 3D spatial grid of candidate contact points around an object.

    This is the output of Layer 2. Each point is a potential grip location
    that gets scored in Layer 3.
    """
    detection: Detection
    points: List[ContactPoint]
    object_center: np.ndarray       # 3D center of the object
    object_extent: np.ndarray       # 3D bounding extent (half-widths)
    geometry_type: ObjectGeometry
    generation_time_ms: float

    @property
    def count(self) -> int:
        return len(self.points)

    @property
    def optimal_points(self) -> List[ContactPoint]:
        return [p for p in self.points if p.zone == "optimal"]

    @property
    def acceptable_points(self) -> List[ContactPoint]:
        return [p for p in self.points if p.zone == "acceptable"]

    @property
    def risk_points(self) -> List[ContactPoint]:
        return [p for p in self.points if p.zone == "risk"]

    def to_dict(self) -> dict:
        return {
            "detection": self.detection.to_dict(),
            "point_count": self.count,
            "optimal": len(self.optimal_points),
            "acceptable": len(self.acceptable_points),
            "risk": len(self.risk_points),
            "object_center": self.object_center.tolist(),
            "object_extent": self.object_extent.tolist(),
            "geometry_type": self.geometry_type.value,
            "generation_time_ms": round(self.generation_time_ms, 2),
            "points": [p.to_dict() for p in self.points],
        }


class SpatialGridProjector:
    """Layer 2: Projects a 3D spatial grid of candidate contact points.

    Given a detection (2D bbox + optional depth), projects a 3D mesh
    of candidate grip locations around the estimated object surface.
    The grid density and shape adapt to the object's grasp category.
    """

    # Geometry mapping from grasp category
    _CATEGORY_GEOMETRY = {
        GraspCategory.TOOL: ObjectGeometry.COMPOSITE,
        GraspCategory.CONTAINER: ObjectGeometry.CYLINDER,
        GraspCategory.SPHERE: ObjectGeometry.SPHERE,
        GraspCategory.CYLINDER: ObjectGeometry.CYLINDER,
        GraspCategory.FLAT: ObjectGeometry.BOX,
        GraspCategory.IRREGULAR: ObjectGeometry.BOX,
        GraspCategory.FRAGILE: ObjectGeometry.SPHERE,
        GraspCategory.DEFORMABLE: ObjectGeometry.BOX,
        GraspCategory.UNKNOWN: ObjectGeometry.BOX,
    }

    def __init__(self, config: GraspConfig):
        self.config = config

    def project(
        self,
        detection: Detection,
        frame_width: int,
        frame_height: int,
        depth_m: float = 1.0,
    ) -> SpatialGrid:
        """Project a spatial grid around a detected object.

        Args:
            detection: Detected object with bbox and category.
            frame_width: Frame width for coordinate normalization.
            frame_height: Frame height for coordinate normalization.
            depth_m: Estimated depth to object (meters). Uses detection
                     depth if available.

        Returns:
            SpatialGrid with scored contact points.
        """
        start = time.time()

        # Use detection depth if available
        if detection.depth_estimate_m is not None:
            depth_m = detection.depth_estimate_m

        # Estimate 3D position and extent from 2D bbox + depth
        center_3d, extent_3d = self._estimate_3d_bounds(
            detection.bbox, frame_width, frame_height, depth_m
        )

        # Select geometry type
        geometry = self._CATEGORY_GEOMETRY.get(
            detection.grasp_category, ObjectGeometry.BOX
        )

        # Generate contact points on the surface
        points = self._generate_surface_points(
            center_3d, extent_3d, geometry, detection.grasp_category
        )

        gen_time = (time.time() - start) * 1000

        return SpatialGrid(
            detection=detection,
            points=points,
            object_center=center_3d,
            object_extent=extent_3d,
            geometry_type=geometry,
            generation_time_ms=gen_time,
        )

    def _estimate_3d_bounds(
        self,
        bbox: BoundingBox,
        img_w: int,
        img_h: int,
        depth_m: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate 3D center and extent from 2D bbox and depth.

        Uses a simple pinhole camera model. For more accuracy,
        use actual camera intrinsics from the frame metadata.
        """
        # Normalize bbox center to [-1, 1] range
        cx_norm = (bbox.center[0] / img_w) * 2 - 1
        cy_norm = (bbox.center[1] / img_h) * 2 - 1

        # Approximate 3D center (assuming ~60° FOV)
        fov_factor = 0.5  # tan(30°) ≈ 0.577
        center_3d = np.array([
            cx_norm * depth_m * fov_factor,
            -cy_norm * depth_m * fov_factor,  # Y is up in world frame
            depth_m,
        ])

        # Estimate 3D extent from bbox size
        # Object angular size → physical size at depth
        w_ratio = bbox.width / img_w
        h_ratio = bbox.height / img_h
        extent_3d = np.array([
            w_ratio * depth_m * fov_factor,
            h_ratio * depth_m * fov_factor,
            min(w_ratio, h_ratio) * depth_m * fov_factor * 0.5,  # Depth extent estimate
        ])

        return center_3d, extent_3d

    def _generate_surface_points(
        self,
        center: np.ndarray,
        extent: np.ndarray,
        geometry: ObjectGeometry,
        category: GraspCategory,
    ) -> List[ContactPoint]:
        """Generate contact points on the estimated object surface."""

        res = self.config.grid_resolution
        points = []

        if geometry == ObjectGeometry.SPHERE:
            points = self._generate_sphere_points(center, extent, res)
        elif geometry == ObjectGeometry.CYLINDER:
            points = self._generate_cylinder_points(center, extent, res, category)
        elif geometry == ObjectGeometry.COMPOSITE:
            points = self._generate_composite_points(center, extent, res, category)
        else:  # BOX
            points = self._generate_box_points(center, extent, res)

        return points

    def _generate_sphere_points(
        self, center: np.ndarray, extent: np.ndarray, res: int
    ) -> List[ContactPoint]:
        """Generate points on a sphere surface (Fibonacci lattice)."""
        points = []
        n = res * 4  # Total points on sphere
        radius = np.mean(extent[:2])  # Average radius

        golden_ratio = (1 + math.sqrt(5)) / 2

        for i in range(n):
            theta = math.acos(1 - 2 * (i + 0.5) / n)
            phi = 2 * math.pi * i / golden_ratio

            x = center[0] + radius * math.sin(theta) * math.cos(phi)
            y = center[1] + radius * math.sin(theta) * math.sin(phi)
            z = center[2] + radius * math.cos(theta)

            pos = np.array([x, y, z])
            normal = (pos - center)
            normal = normal / (np.linalg.norm(normal) + 1e-8)

            # Curvature = 1/radius (constant for sphere)
            curvature = 1.0 / (radius + 1e-8)

            points.append(ContactPoint(
                position=pos,
                surface_normal=normal,
                curvature=curvature,
            ))

        return points

    def _generate_cylinder_points(
        self,
        center: np.ndarray,
        extent: np.ndarray,
        res: int,
        category: GraspCategory,
    ) -> List[ContactPoint]:
        """Generate points on a cylinder surface."""
        points = []
        radius = extent[0]
        half_height = extent[1]
        n_ring = max(8, res // 2)
        n_height = max(4, res // 4)

        for hi in range(n_height):
            y = center[1] - half_height + (2 * half_height * hi / (n_height - 1))

            for ri in range(n_ring):
                theta = 2 * math.pi * ri / n_ring

                x = center[0] + radius * math.cos(theta)
                z = center[2] + radius * math.sin(theta)

                pos = np.array([x, y, z])
                normal = np.array([math.cos(theta), 0, math.sin(theta)])
                curvature = 1.0 / (radius + 1e-8)

                points.append(ContactPoint(
                    position=pos,
                    surface_normal=normal,
                    curvature=curvature,
                ))

        # Add top and bottom cap points
        for cap_y, cap_normal_y in [
            (center[1] - half_height, -1),
            (center[1] + half_height, 1),
        ]:
            for ri in range(max(3, n_ring // 2)):
                r = radius * (ri + 1) / (n_ring // 2)
                for ti in range(max(4, n_ring // 2)):
                    theta = 2 * math.pi * ti / (n_ring // 2)
                    x = center[0] + r * math.cos(theta)
                    z = center[2] + r * math.sin(theta)
                    points.append(ContactPoint(
                        position=np.array([x, cap_y, z]),
                        surface_normal=np.array([0, cap_normal_y, 0], dtype=float),
                        curvature=0.0,  # Flat cap
                    ))

        return points

    def _generate_box_points(
        self, center: np.ndarray, extent: np.ndarray, res: int
    ) -> List[ContactPoint]:
        """Generate points on a box surface."""
        points = []
        n_per_face = max(4, res // 6)

        # Six faces: ±x, ±y, ±z
        for axis in range(3):
            for sign in [-1, 1]:
                normal = np.zeros(3)
                normal[axis] = sign

                # Generate grid on this face
                other_axes = [a for a in range(3) if a != axis]
                for i in range(n_per_face):
                    for j in range(n_per_face):
                        pos = center.copy()
                        pos[axis] += sign * extent[axis]
                        pos[other_axes[0]] += extent[other_axes[0]] * (2 * i / (n_per_face - 1) - 1)
                        pos[other_axes[1]] += extent[other_axes[1]] * (2 * j / (n_per_face - 1) - 1)

                        points.append(ContactPoint(
                            position=pos,
                            surface_normal=normal.copy(),
                            curvature=0.0,  # Flat face
                        ))

        return points

    def _generate_composite_points(
        self,
        center: np.ndarray,
        extent: np.ndarray,
        res: int,
        category: GraspCategory,
    ) -> List[ContactPoint]:
        """Generate points for composite objects (e.g., tool = handle + head)."""
        points = []

        if category == GraspCategory.TOOL:
            # Handle: cylinder along X-axis
            handle_center = center.copy()
            handle_extent = np.array([extent[0] * 0.8, extent[1] * 0.3, extent[2] * 0.3])
            handle_points = self._generate_cylinder_points(
                handle_center, handle_extent, res // 2, category
            )
            # Rotate cylinder to lie along X (swap Y and X in normals)
            for p in handle_points:
                # Simple axis remap: treat cylinder as lying along X
                offset = p.position - handle_center
                p.position = handle_center + np.array([offset[1], offset[0], offset[2]])
                p.surface_normal = np.array([p.surface_normal[1], p.surface_normal[0], p.surface_normal[2]])
            points.extend(handle_points)

            # Head: small sphere offset along X
            head_center = center + np.array([extent[0] * 0.7, 0, 0])
            head_extent = extent * 0.4
            head_points = self._generate_sphere_points(
                head_center, head_extent, res // 3
            )
            points.extend(head_points)

        else:
            # Fallback: box
            points = self._generate_box_points(center, extent, res)

        return points


# ══════════════════════════════════════════════════════════════
# Layer 3: Grasp Affordance Scoring
# ══════════════════════════════════════════════════════════════

class AffordanceScorer:
    """Layer 3: Scores each contact point for grasp viability.

    Evaluates each grid point based on:
    - Surface normal alignment (can a finger approach from this direction?)
    - Curvature fitness (flat vs curved vs edge)
    - Friction estimate (material-dependent slip resistance)
    - Force distribution (how efficiently does this point transfer grip force?)
    - Stability contribution (how much does this point resist object rotation?)

    The composite score determines the green/yellow/red zone classification.
    """

    def __init__(self, config: GraspConfig):
        self.config = config

        # Score weights (tuned for general manipulation)
        self.w_friction = 0.25
        self.w_stability = 0.30
        self.w_force = 0.25
        self.w_curvature = 0.20

    def score(self, grid: SpatialGrid) -> SpatialGrid:
        """Score all contact points in a spatial grid.

        Modifies points in-place and returns the same grid.
        """
        start = time.time()

        category = grid.detection.grasp_category
        friction_est = grid.detection.estimated_friction or self.config.friction_coefficient

        for point in grid.points:
            # ── Friction Score ──────────────────────
            # Higher friction = better grip. Penalize near-vertical normals
            # (gravity pulls object out of grip).
            normal_up_component = abs(point.surface_normal[1])  # Y is up
            # Surfaces facing sideways are better for grip
            lateral_component = math.sqrt(
                point.surface_normal[0] ** 2 + point.surface_normal[2] ** 2
            )
            point.friction_score = (
                friction_est * (0.3 + 0.7 * lateral_component) *
                (1.0 - 0.5 * normal_up_component)
            )

            # ── Stability Score ─────────────────────
            # Points opposing each other contribute to force closure.
            # Points near the object's center of mass are more stable.
            dist_from_center = np.linalg.norm(point.position - grid.object_center)
            max_dist = np.linalg.norm(grid.object_extent) + 1e-8
            proximity = 1.0 - (dist_from_center / max_dist)

            # Prefer points at the object's equator (y near center)
            equator_proximity = 1.0 - abs(
                (point.position[1] - grid.object_center[1]) /
                (grid.object_extent[1] + 1e-8)
            )
            equator_proximity = max(0, min(1, equator_proximity))

            point.stability_score = (
                0.4 * proximity +
                0.3 * equator_proximity +
                0.3 * lateral_component
            )

            # ── Force Score ─────────────────────────
            # How efficiently does this point transfer grip force?
            # Flat surfaces (low curvature) transfer force better.
            # Very high curvature (edges) concentrate stress.
            curvature_normalized = min(1.0, point.curvature * np.mean(grid.object_extent))
            point.force_score = (
                (1.0 - 0.6 * curvature_normalized) *  # Moderate curvature is OK
                lateral_component *                      # Must be approachable
                (0.7 + 0.3 * proximity)                 # Closer to center = better
            )

            # ── Category-Specific Adjustments ───────
            point = self._apply_category_adjustments(
                point, grid, category
            )

            # ── Composite Score ─────────────────────
            point.composite_score = (
                self.w_friction * point.friction_score +
                self.w_stability * point.stability_score +
                self.w_force * point.force_score +
                self.w_curvature * (1.0 - curvature_normalized)
            )
            point.composite_score = max(0, min(1, point.composite_score))

            # ── Zone Classification ─────────────────
            if point.composite_score >= self.config.stability_threshold:
                point.zone = "optimal"
            elif point.composite_score >= self.config.acceptable_threshold:
                point.zone = "acceptable"
            else:
                point.zone = "risk"

        score_time = (time.time() - start) * 1000
        logger.debug(
            f"Scored {grid.count} points in {score_time:.1f}ms: "
            f"{len(grid.optimal_points)} optimal, "
            f"{len(grid.acceptable_points)} acceptable, "
            f"{len(grid.risk_points)} risk"
        )

        return grid

    def _apply_category_adjustments(
        self,
        point: ContactPoint,
        grid: SpatialGrid,
        category: GraspCategory,
    ) -> ContactPoint:
        """Apply object-category-specific scoring adjustments."""

        if category == GraspCategory.TOOL:
            # Boost handle region (points near center along longest axis)
            handle_proximity = 1.0 - abs(
                (point.position[0] - grid.object_center[0]) /
                (grid.object_extent[0] + 1e-8)
            )
            point.friction_score *= (0.6 + 0.4 * max(0, handle_proximity))
            point.stability_score *= (0.7 + 0.3 * max(0, handle_proximity))

        elif category == GraspCategory.CONTAINER:
            # Boost rim and handle-side points
            height_ratio = (
                (point.position[1] - grid.object_center[1]) /
                (grid.object_extent[1] + 1e-8)
            )
            # Upper part of container is better for rim grip
            if height_ratio > 0.3:
                point.stability_score *= 1.2
            # Side with handle (positive X) gets a boost
            if point.position[0] > grid.object_center[0]:
                point.friction_score *= 1.15

        elif category == GraspCategory.SPHERE:
            # Equatorial band is optimal for spheres
            # Already handled in base scoring, but amplify
            equator_dist = abs(
                point.position[1] - grid.object_center[1]
            ) / (grid.object_extent[1] + 1e-8)
            equator_boost = max(0, 1.0 - equator_dist * 1.5)
            point.stability_score *= (0.7 + 0.3 * equator_boost)

        elif category == GraspCategory.FRAGILE:
            # Heavily penalize high-force points
            point.force_score *= 0.5
            # Boost widely distributed points (spread the force)
            point.stability_score *= 1.1

        elif category == GraspCategory.FLAT:
            # Top and bottom faces are best for pinch grasp
            top_bottom = abs(point.surface_normal[1])
            point.stability_score *= (0.5 + 0.5 * top_bottom)

        # Clamp all scores to [0, 1]
        point.friction_score = max(0, min(1, point.friction_score))
        point.stability_score = max(0, min(1, point.stability_score))
        point.force_score = max(0, min(1, point.force_score))

        return point


# ══════════════════════════════════════════════════════════════
# Layer 4: Hand Configuration Planning
# ══════════════════════════════════════════════════════════════

class GripperType(Enum):
    """Available gripper/hand types."""
    PARALLEL_JAW = "parallel_jaw"
    THREE_FINGER = "three_finger"
    FIVE_FINGER = "five_finger"
    PROSTHETIC = "prosthetic"


@dataclass
class FingerPlacement:
    """Planned placement for a single finger/contact pad."""
    finger_id: int                    # 0 = thumb, 1 = index, etc.
    finger_name: str                  # "thumb", "index", "middle", etc.
    contact_point: ContactPoint       # The grid point this finger targets
    approach_normal: np.ndarray       # Direction of approach for this finger
    force_n: float                    # Planned contact force (Newtons)


@dataclass
class GraspPlan:
    """Complete grasp plan — how to pick up this object.

    Output of Layer 4. Specifies exactly how the gripper should
    approach, which points to contact, how much force to apply,
    and the wrist orientation for execution.
    """
    detection: Detection
    gripper_type: GripperType
    finger_placements: List[FingerPlacement]
    approach_vector: np.ndarray       # Direction of approach (world frame)
    approach_angle_deg: float         # Approach angle from vertical
    wrist_orientation: np.ndarray     # Quaternion (w, x, y, z)
    total_force_n: float              # Total grip force
    grasp_confidence: float           # Overall confidence [0, 1]
    planning_time_ms: float
    manipulation_hints: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "detection": self.detection.to_dict(),
            "gripper_type": self.gripper_type.value,
            "finger_count": len(self.finger_placements),
            "approach_vector": self.approach_vector.tolist(),
            "approach_angle_deg": round(self.approach_angle_deg, 1),
            "wrist_orientation": self.wrist_orientation.tolist(),
            "total_force_n": round(self.total_force_n, 2),
            "grasp_confidence": round(self.grasp_confidence, 4),
            "planning_time_ms": round(self.planning_time_ms, 2),
            "fingers": [
                {
                    "id": fp.finger_id,
                    "name": fp.finger_name,
                    "position": fp.contact_point.position.tolist(),
                    "force_n": round(fp.force_n, 2),
                    "zone": fp.contact_point.zone,
                }
                for fp in self.finger_placements
            ],
            "manipulation_hints": self.manipulation_hints,
        }


class HandConfigPlanner:
    """Layer 4: Plans optimal hand/gripper configuration.

    Given a scored spatial grid, selects the optimal combination of
    contact points and computes finger placements, approach vector,
    wrist orientation, and force distribution.

    "It doesn't just say 'grab here' — it says 'approach from 35 degrees
    left, index finger on this ridge, thumb on this flat, apply 2.4N.'"
    """

    # Finger definitions per gripper type
    _FINGER_DEFS = {
        GripperType.PARALLEL_JAW: [
            {"id": 0, "name": "jaw_left"},
            {"id": 1, "name": "jaw_right"},
        ],
        GripperType.THREE_FINGER: [
            {"id": 0, "name": "finger_a"},
            {"id": 1, "name": "finger_b"},
            {"id": 2, "name": "finger_c"},
        ],
        GripperType.FIVE_FINGER: [
            {"id": 0, "name": "thumb"},
            {"id": 1, "name": "index"},
            {"id": 2, "name": "middle"},
            {"id": 3, "name": "ring"},
            {"id": 4, "name": "pinky"},
        ],
    }

    def __init__(self, config: GraspConfig):
        self.config = config

    def plan(
        self,
        grid: SpatialGrid,
        gripper_type: GripperType = GripperType.PARALLEL_JAW,
    ) -> GraspPlan:
        """Generate a grasp plan from a scored spatial grid.

        Selects optimal contact points, computes finger placements,
        approach vector, and force distribution.
        """
        start = time.time()

        finger_defs = self._FINGER_DEFS.get(gripper_type, self._FINGER_DEFS[GripperType.PARALLEL_JAW])
        n_fingers = len(finger_defs)

        # Select best contact points using antipodal heuristic
        selected_points = self._select_contact_points(grid, n_fingers)

        # Compute approach vector (average inward direction of selected points)
        approach_vec = self._compute_approach_vector(selected_points, grid.object_center)

        # Compute approach angle from vertical
        up = np.array([0, 1, 0])
        cos_angle = np.dot(approach_vec, up) / (np.linalg.norm(approach_vec) + 1e-8)
        approach_angle = math.degrees(math.acos(max(-1, min(1, cos_angle))))

        # Compute wrist orientation (simple: align Z with approach)
        wrist_quat = self._approach_to_quaternion(approach_vec)

        # Distribute force across fingers
        total_mass = grid.detection.estimated_mass_kg or 0.5
        gravity_force = total_mass * 9.81
        # Need enough friction force to overcome gravity
        friction = grid.detection.estimated_friction or self.config.friction_coefficient
        min_grip_force = gravity_force / (n_fingers * friction + 1e-8)
        grip_force = min(self.config.force_limit_n, max(min_grip_force * 1.5, 1.0))

        # Create finger placements
        finger_placements = []
        for i, (fdef, point) in enumerate(zip(finger_defs, selected_points)):
            force_per_finger = grip_force / n_fingers
            # Adjust force by point quality
            force_per_finger *= (0.7 + 0.3 * point.composite_score)

            finger_placements.append(FingerPlacement(
                finger_id=fdef["id"],
                finger_name=fdef["name"],
                contact_point=point,
                approach_normal=-point.surface_normal,  # Approach into surface
                force_n=force_per_finger,
            ))

        # Overall confidence
        avg_score = np.mean([p.composite_score for p in selected_points])
        force_margin = 1.0 - (grip_force / self.config.force_limit_n)
        confidence = avg_score * 0.7 + force_margin * 0.3

        planning_time = (time.time() - start) * 1000

        plan = GraspPlan(
            detection=grid.detection,
            gripper_type=gripper_type,
            finger_placements=finger_placements,
            approach_vector=approach_vec,
            approach_angle_deg=approach_angle,
            wrist_orientation=wrist_quat,
            total_force_n=grip_force,
            grasp_confidence=confidence,
            planning_time_ms=planning_time,
        )

        # Layer 5: Add manipulation hints
        plan.manipulation_hints = self._compute_manipulation_hints(grid, plan)

        return plan

    def _select_contact_points(
        self, grid: SpatialGrid, n_points: int
    ) -> List[ContactPoint]:
        """Select optimal contact points for force closure.

        Uses a greedy algorithm:
        1. Start with the highest-scoring point
        2. Add points that maximize angular coverage around the object
        3. Prefer antipodal pairs (opposing normals)
        """
        if not grid.points:
            return []

        # Sort by composite score
        sorted_points = sorted(
            grid.points, key=lambda p: p.composite_score, reverse=True
        )

        # Start with best point
        selected = [sorted_points[0]]
        selected_ids = {id(sorted_points[0])}

        for _ in range(n_points - 1):
            best_candidate = None
            best_diversity = -1

            for candidate in sorted_points:
                if id(candidate) in selected_ids:
                    continue

                # Compute diversity: how different is this point from selected ones?
                min_angle = float("inf")
                for sel in selected:
                    # Angular distance between surface normals
                    cos_sim = np.dot(candidate.surface_normal, sel.surface_normal)
                    angle = math.acos(max(-1, min(1, cos_sim)))
                    min_angle = min(min_angle, angle)

                # Also consider spatial distribution
                min_dist = min(
                    np.linalg.norm(candidate.position - sel.position)
                    for sel in selected
                )

                # Combined diversity metric: angle diversity + spatial spread
                diversity = (
                    candidate.composite_score * 0.4 +
                    (min_angle / math.pi) * 0.35 +
                    min(1, min_dist / (np.mean(grid.object_extent) + 1e-8)) * 0.25
                )

                if diversity > best_diversity:
                    best_diversity = diversity
                    best_candidate = candidate

            if best_candidate is not None:
                selected.append(best_candidate)
                selected_ids.add(id(best_candidate))

        return selected[:n_points]

    def _compute_approach_vector(
        self,
        points: List[ContactPoint],
        object_center: np.ndarray,
    ) -> np.ndarray:
        """Compute the optimal approach direction.

        Average of the inward normals, biased toward top-down for stability.
        """
        if not points:
            return np.array([0, -1, 0])  # Default: straight down

        avg_normal = np.mean([-p.surface_normal for p in points], axis=0)

        # Bias toward top-down approach (gravity-aligned)
        top_down = np.array([0, -1, 0])
        approach = avg_normal * 0.6 + top_down * 0.4
        approach = approach / (np.linalg.norm(approach) + 1e-8)

        return approach

    def _approach_to_quaternion(self, approach: np.ndarray) -> np.ndarray:
        """Convert approach vector to wrist orientation quaternion.

        Simple implementation: align gripper Z-axis with approach direction.
        """
        z = approach / (np.linalg.norm(approach) + 1e-8)

        # Choose an up vector that isn't parallel to z
        up_candidate = np.array([0, 1, 0])
        if abs(np.dot(z, up_candidate)) > 0.95:
            up_candidate = np.array([1, 0, 0])

        x = np.cross(up_candidate, z)
        x = x / (np.linalg.norm(x) + 1e-8)
        y = np.cross(z, x)

        # Rotation matrix to quaternion
        R = np.column_stack([x, y, z])
        trace = R[0, 0] + R[1, 1] + R[2, 2]

        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1)
            w = 0.25 / s
            qx = (R[2, 1] - R[1, 2]) * s
            qy = (R[0, 2] - R[2, 0]) * s
            qz = (R[1, 0] - R[0, 1]) * s
        else:
            # Handle edge cases
            w, qx, qy, qz = 1, 0, 0, 0

        return np.array([w, qx, qy, qz])


    # ══════════════════════════════════════════════════════════
    # Layer 5: Manipulation Intelligence
    # ══════════════════════════════════════════════════════════

    def _compute_manipulation_hints(
        self, grid: SpatialGrid, plan: GraspPlan
    ) -> dict:
        """Layer 5: Post-grasp reasoning.

        Once held, how should the object be rotated, carried, or placed?
        A coffee mug needs to stay upright. A tool needs to be oriented
        for use. A fragile object needs force limits.
        """
        category = grid.detection.grasp_category
        hints = {
            "keep_upright": False,
            "orientation_constraint": None,
            "max_acceleration_mps2": 5.0,     # Default safe acceleration
            "max_rotation_dps": 90.0,          # Default safe rotation speed
            "place_orientation": "natural",
            "fragile": False,
            "notes": [],
        }

        if category == GraspCategory.CONTAINER:
            hints["keep_upright"] = True
            hints["max_acceleration_mps2"] = 2.0
            hints["max_rotation_dps"] = 30.0
            hints["orientation_constraint"] = "upright"
            hints["notes"].append("Container: maintain upright orientation to prevent spilling")

        elif category == GraspCategory.TOOL:
            hints["orientation_constraint"] = "use_ready"
            hints["place_orientation"] = "handle_accessible"
            hints["notes"].append("Tool: orient with working end forward for handoff")

        elif category == GraspCategory.FRAGILE:
            hints["fragile"] = True
            hints["max_acceleration_mps2"] = 1.0
            hints["max_rotation_dps"] = 20.0
            hints["notes"].append(
                f"Fragile: force limit {plan.total_force_n:.1f}N, minimize acceleration"
            )

        elif category == GraspCategory.SPHERE:
            hints["max_rotation_dps"] = 180.0  # Orientation doesn't matter
            hints["notes"].append("Sphere: no orientation constraint")

        elif category == GraspCategory.FLAT:
            hints["orientation_constraint"] = "level"
            hints["notes"].append("Flat object: maintain level during transport")

        elif category == GraspCategory.DEFORMABLE:
            hints["max_acceleration_mps2"] = 2.0
            hints["notes"].append("Deformable: adaptive grip — monitor force feedback")

        return hints


# ══════════════════════════════════════════════════════════════
# Full Grasp Pipeline
# ══════════════════════════════════════════════════════════════

class GraspPipeline:
    """Complete Layers 2-5 pipeline.

    Takes a Detection, produces a GraspPlan.
    The moment an object is detected, the grasp plan is generated
    in a single pass — detection and grasp planning fused.
    """

    def __init__(self, config: Optional[GraspConfig] = None):
        self.config = config or GraspConfig()
        self.projector = SpatialGridProjector(self.config)
        self.scorer = AffordanceScorer(self.config)
        self.planner = HandConfigPlanner(self.config)

    def process(
        self,
        detection: Detection,
        frame_width: int = 640,
        frame_height: int = 480,
        depth_m: float = 1.0,
        gripper_type: GripperType = GripperType.PARALLEL_JAW,
    ) -> Tuple[SpatialGrid, GraspPlan]:
        """Run the full grasp pipeline on a detection.

        Returns:
            (SpatialGrid, GraspPlan) — the scored grid and final grasp plan.
        """
        start = time.time()

        # Layer 2: Spatial Grid Projection
        grid = self.projector.project(detection, frame_width, frame_height, depth_m)

        # Layer 3: Grasp Affordance Scoring
        grid = self.scorer.score(grid)

        # Layer 4 + 5: Hand Configuration + Manipulation Intelligence
        plan = self.planner.plan(grid, gripper_type)

        total_ms = (time.time() - start) * 1000
        logger.info(
            f"Grasp pipeline: {grid.count} points, "
            f"{len(grid.optimal_points)} optimal, "
            f"confidence={plan.grasp_confidence:.3f}, "
            f"{total_ms:.1f}ms total"
        )

        return grid, plan
