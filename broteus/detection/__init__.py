"""BROTEUS Detection Layer — Object detection and classification."""

from broteus.detection.detector import (
    Detection,
    DetectionResult,
    BoundingBox,
    GraspCategory,
    DetectionModel,
    YOLODetector,
    SimulatedDetector,
)

__all__ = [
    "Detection", "DetectionResult", "BoundingBox", "GraspCategory",
    "DetectionModel", "YOLODetector", "SimulatedDetector",
]
