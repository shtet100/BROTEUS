"""
BROTEUS Master Pipeline
=========================

The full detection-to-grasp pipeline as a unified real-time system.

    Frame → Detection → Spatial Grid → Affordance Scoring → 
    Hand Config → Manipulation Intelligence → GraspPlan

Existing work treats vision and grasp planning as separate stages.
BROTEUS fuses them — the moment an object is visually detected,
the grasp affordance grid is projected simultaneously, with hand
configuration generated in the same inference pass.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from broteus.core.frame import BroteusFrame
from broteus.core.config import BroteusConfig
from broteus.detection.detector import (
    DetectionModel,
    DetectionResult,
    Detection,
    SimulatedDetector,
)
from broteus.grasp.engine import (
    GraspPipeline,
    GripperType,
    SpatialGrid,
    GraspPlan,
)

logger = logging.getLogger("broteus.pipeline")


@dataclass
class PipelineResult:
    """Complete output of the BROTEUS pipeline for a single frame.

    Contains detection results and grasp plans for every detected object.
    This is what gets sent to the visualization frontend over WebSocket.
    """
    frame_id: str
    timestamp: float
    detections: DetectionResult
    grasp_results: List[Tuple[SpatialGrid, GraspPlan]]
    total_time_ms: float

    @property
    def object_count(self) -> int:
        return self.detections.count

    @property
    def best_grasp(self) -> Optional[GraspPlan]:
        """Return the highest-confidence grasp plan."""
        if not self.grasp_results:
            return None
        return max(self.grasp_results, key=lambda r: r[1].grasp_confidence)[1]

    def to_dict(self) -> dict:
        """Serialize for WebSocket/API transmission."""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "total_time_ms": round(self.total_time_ms, 2),
            "object_count": self.object_count,
            "detections": self.detections.to_dict(),
            "grasp_plans": [
                {
                    "grid_summary": {
                        "point_count": grid.count,
                        "optimal": len(grid.optimal_points),
                        "acceptable": len(grid.acceptable_points),
                        "risk": len(grid.risk_points),
                        "geometry": grid.geometry_type.value,
                        "generation_time_ms": round(grid.generation_time_ms, 2),
                    },
                    "plan": plan.to_dict(),
                }
                for grid, plan in self.grasp_results
            ],
        }

    def to_visualization_payload(self) -> dict:
        """Compact payload optimized for Three.js visualization.

        Sends point positions, scores, and zones without full metadata
        to keep WebSocket messages lean.
        """
        viz_objects = []

        for grid, plan in self.grasp_results:
            # Compact point data: [x, y, z, score, zone_id]
            # zone_id: 0=risk, 1=acceptable, 2=optimal
            zone_map = {"risk": 0, "acceptable": 1, "optimal": 2}
            points_compact = [
                [
                    round(p.position[0], 4),
                    round(p.position[1], 4),
                    round(p.position[2], 4),
                    round(p.composite_score, 3),
                    zone_map.get(p.zone, 0),
                ]
                for p in grid.points
            ]

            # Finger placements: [x, y, z, force]
            fingers_compact = [
                [
                    round(fp.contact_point.position[0], 4),
                    round(fp.contact_point.position[1], 4),
                    round(fp.contact_point.position[2], 4),
                    round(fp.force_n, 2),
                ]
                for fp in plan.finger_placements
            ]

            viz_objects.append({
                "class": grid.detection.class_name,
                "confidence": round(grid.detection.confidence, 3),
                "category": grid.detection.grasp_category.value,
                "center": grid.object_center.tolist(),
                "extent": grid.object_extent.tolist(),
                "points": points_compact,
                "fingers": fingers_compact,
                "approach": plan.approach_vector.tolist(),
                "approach_angle": round(plan.approach_angle_deg, 1),
                "grip_force": round(plan.total_force_n, 2),
                "grasp_confidence": round(plan.grasp_confidence, 3),
                "gripper": plan.gripper_type.value,
                "manipulation": plan.manipulation_hints,
            })

        return {
            "type": "grasp_data",
            "frame_id": self.frame_id,
            "pipeline_ms": round(self.total_time_ms, 1),
            "objects": viz_objects,
        }


class BroteusEngine:
    """The unified BROTEUS engine.

    Chains all 5 layers into a single process() call.
    Feed it frames, get back grasp plans.
    """

    def __init__(
        self,
        config: Optional[BroteusConfig] = None,
        detector: Optional[DetectionModel] = None,
        gripper_type: GripperType = GripperType.PARALLEL_JAW,
    ):
        self.config = config or BroteusConfig()
        self.gripper_type = gripper_type

        # Detection model (Layer 1)
        self.detector = detector or SimulatedDetector(objects=["wrench"])
        self.detector.load(device=self.config.detection.device)

        # Grasp pipeline (Layers 2-5)
        self.grasp_pipeline = GraspPipeline(self.config.grasp)

        # Stats
        self._frames_processed = 0
        self._total_time_ms = 0

        logger.info(
            f"BROTEUS engine initialized: "
            f"detector={self.detector.get_model_info()['name']}, "
            f"gripper={self.gripper_type.value}"
        )

    def process(self, frame: BroteusFrame) -> PipelineResult:
        """Process a single frame through all 5 layers.

        Args:
            frame: Input frame from any camera adapter.

        Returns:
            PipelineResult with detections and grasp plans.
        """
        start = time.time()

        # Layer 1: Detection
        detection_result = self.detector.detect(frame)

        # Layers 2-5: Grasp planning for each detection
        grasp_results = []
        for detection in detection_result.detections:
            grid, plan = self.grasp_pipeline.process(
                detection=detection,
                frame_width=frame.width,
                frame_height=frame.height,
                depth_m=detection.depth_estimate_m or 1.0,
                gripper_type=self.gripper_type,
            )
            grasp_results.append((grid, plan))

        total_ms = (time.time() - start) * 1000

        self._frames_processed += 1
        self._total_time_ms += total_ms

        return PipelineResult(
            frame_id=frame.frame_id,
            timestamp=time.time(),
            detections=detection_result,
            grasp_results=grasp_results,
            total_time_ms=total_ms,
        )

    def set_gripper(self, gripper_type: GripperType) -> None:
        """Switch the active gripper type."""
        self.gripper_type = gripper_type
        logger.info(f"Gripper switched to: {gripper_type.value}")

    def set_objects(self, objects: List[str]) -> None:
        """Update the simulated detector's object list."""
        if isinstance(self.detector, SimulatedDetector):
            self.detector._objects = objects
            logger.info(f"Simulated objects updated: {objects}")

    @property
    def stats(self) -> dict:
        avg_ms = (
            self._total_time_ms / self._frames_processed
            if self._frames_processed > 0 else 0
        )
        return {
            "frames_processed": self._frames_processed,
            "total_time_ms": round(self._total_time_ms, 2),
            "avg_pipeline_ms": round(avg_ms, 2),
            "avg_fps": round(1000 / avg_ms, 1) if avg_ms > 0 else 0,
            "detector": self.detector.get_model_info(),
            "gripper": self.gripper_type.value,
        }
