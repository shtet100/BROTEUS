"""
BROTEUS Detection Layer
=========================

Layer 1 of the pipeline: real-time object detection, classification,
and bounding box extraction from BroteusFrames.

Designed to be model-agnostic — swap YOLO for any detector that
implements the DetectionModel interface.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from broteus.core.frame import BroteusFrame

logger = logging.getLogger("broteus.detection")


# ── Detection Data Structures ──────────────────────────────────

@dataclass(frozen=True)
class BoundingBox:
    """2D bounding box in pixel coordinates."""
    x1: float   # Top-left x
    y1: float   # Top-left y
    x2: float   # Bottom-right x
    y2: float   # Bottom-right y

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return self.width * self.height

    def normalized(self, img_w: int, img_h: int) -> "BoundingBox":
        """Return box with coordinates normalized to [0, 1]."""
        return BoundingBox(
            x1=self.x1 / img_w, y1=self.y1 / img_h,
            x2=self.x2 / img_w, y2=self.y2 / img_h,
        )

    def to_dict(self) -> dict:
        return {
            "x1": self.x1, "y1": self.y1,
            "x2": self.x2, "y2": self.y2,
            "width": self.width, "height": self.height,
            "center": self.center,
        }


class GraspCategory(Enum):
    """Object categories relevant to grasp planning.

    Each category implies different grasp strategies.
    """
    TOOL = "tool"               # Wrench, screwdriver, hammer → grip handle
    CONTAINER = "container"     # Mug, cup, bowl → rim grip or side grip
    SPHERE = "sphere"           # Ball, apple, orange → enveloping grasp
    CYLINDER = "cylinder"       # Bottle, can, tube → power grasp
    FLAT = "flat"               # Book, phone, card → pinch grasp
    IRREGULAR = "irregular"     # Complex geometry
    FRAGILE = "fragile"         # Egg, glass → force-limited grasp
    DEFORMABLE = "deformable"   # Cloth, rope, bag → adaptive grasp
    UNKNOWN = "unknown"


# Mapping common COCO/YOLO classes to grasp categories
_CLASS_TO_GRASP_CATEGORY = {
    "wrench": GraspCategory.TOOL,
    "scissors": GraspCategory.TOOL,
    "knife": GraspCategory.TOOL,
    "hammer": GraspCategory.TOOL,
    "screwdriver": GraspCategory.TOOL,
    "cup": GraspCategory.CONTAINER,
    "mug": GraspCategory.CONTAINER,
    "bowl": GraspCategory.CONTAINER,
    "wine glass": GraspCategory.CONTAINER,
    "bottle": GraspCategory.CYLINDER,
    "can": GraspCategory.CYLINDER,
    "vase": GraspCategory.CYLINDER,
    "sports ball": GraspCategory.SPHERE,
    "apple": GraspCategory.SPHERE,
    "orange": GraspCategory.SPHERE,
    "tennis ball": GraspCategory.SPHERE,
    "book": GraspCategory.FLAT,
    "cell phone": GraspCategory.FLAT,
    "remote": GraspCategory.FLAT,
    "laptop": GraspCategory.FLAT,
    "mouse": GraspCategory.IRREGULAR,
    "keyboard": GraspCategory.FLAT,
    "banana": GraspCategory.IRREGULAR,
    "teddy bear": GraspCategory.DEFORMABLE,
    "backpack": GraspCategory.DEFORMABLE,
}


@dataclass
class Detection:
    """A single detected object in a frame.

    This is the output of Layer 1, and the input to Layer 2
    (Spatial Grid Projection).
    """
    class_name: str                         # "wrench", "mug", "bottle"
    class_id: int                           # Model-specific class index
    confidence: float                       # Detection confidence [0, 1]
    bbox: BoundingBox                       # 2D bounding box
    grasp_category: GraspCategory = GraspCategory.UNKNOWN
    mask: Optional[np.ndarray] = None       # Instance segmentation mask (if available)
    depth_estimate_m: Optional[float] = None  # Estimated depth to object center

    # Physical property estimates (refined in grasp layer)
    estimated_mass_kg: Optional[float] = None
    estimated_friction: Optional[float] = None

    def __post_init__(self):
        # Auto-assign grasp category from class name
        if self.grasp_category == GraspCategory.UNKNOWN:
            self.grasp_category = _CLASS_TO_GRASP_CATEGORY.get(
                self.class_name.lower(), GraspCategory.UNKNOWN
            )

    def to_dict(self) -> dict:
        return {
            "class_name": self.class_name,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox.to_dict(),
            "grasp_category": self.grasp_category.value,
            "depth_estimate_m": self.depth_estimate_m,
            "estimated_mass_kg": self.estimated_mass_kg,
        }


@dataclass
class DetectionResult:
    """Complete detection output for a single frame."""
    frame_id: str
    detections: List[Detection]
    inference_time_ms: float
    model_name: str
    timestamp: float = field(default_factory=time.time)

    @property
    def count(self) -> int:
        return len(self.detections)

    def filter_by_confidence(self, threshold: float) -> List[Detection]:
        return [d for d in self.detections if d.confidence >= threshold]

    def filter_by_category(self, category: GraspCategory) -> List[Detection]:
        return [d for d in self.detections if d.grasp_category == category]

    def best_detection(self) -> Optional[Detection]:
        """Return highest-confidence detection."""
        if not self.detections:
            return None
        return max(self.detections, key=lambda d: d.confidence)

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "count": self.count,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "model": self.model_name,
            "detections": [d.to_dict() for d in self.detections],
        }


# ── Detection Model Interface ─────────────────────────────────

class DetectionModel(ABC):
    """Abstract interface for object detection models.

    Implement this to plug in YOLO, Detectron2, DETR, or any
    future detection architecture.
    """

    @abstractmethod
    def load(self, device: str = "auto") -> None:
        """Load model weights."""
        ...

    @abstractmethod
    def detect(self, frame: BroteusFrame) -> DetectionResult:
        """Run detection on a single frame."""
        ...

    @abstractmethod
    def get_model_info(self) -> dict:
        """Return model metadata."""
        ...


# ── YOLO Detector ──────────────────────────────────────────────

class YOLODetector(DetectionModel):
    """YOLOv8 object detection via Ultralytics.

    Requires: pip install ultralytics
    """

    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.5):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self._model = None
        self._device = None

    def load(self, device: str = "auto") -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "YOLO detection requires ultralytics: pip install ultralytics"
            )

        self._device = device
        self._model = YOLO(self.model_name)

        if device != "auto":
            self._model.to(device)

        logger.info(f"YOLO model loaded: {self.model_name} on {device}")

    def detect(self, frame: BroteusFrame) -> DetectionResult:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        start = time.time()
        results = self._model(
            frame.to_rgb(),
            conf=self.conf_threshold,
            verbose=False,
        )
        inference_ms = (time.time() - start) * 1000

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu())
                cls_id = int(boxes.cls[i].cpu())
                cls_name = r.names[cls_id]

                det = Detection(
                    class_name=cls_name,
                    class_id=cls_id,
                    confidence=conf,
                    bbox=BoundingBox(
                        x1=float(xyxy[0]), y1=float(xyxy[1]),
                        x2=float(xyxy[2]), y2=float(xyxy[3]),
                    ),
                )

                # Estimate depth from frame if available
                if frame.has_depth:
                    cx, cy = int(det.bbox.center[0]), int(det.bbox.center[1])
                    cx = max(0, min(cx, frame.width - 1))
                    cy = max(0, min(cy, frame.height - 1))
                    depth_m = frame.depth.to_meters()[cy, cx]
                    det.depth_estimate_m = float(depth_m)

                detections.append(det)

        return DetectionResult(
            frame_id=frame.frame_id,
            detections=detections,
            inference_time_ms=inference_ms,
            model_name=self.model_name,
        )

    def get_model_info(self) -> dict:
        return {
            "name": self.model_name,
            "type": "YOLOv8",
            "device": str(self._device),
            "conf_threshold": self.conf_threshold,
            "loaded": self._model is not None,
        }


# ── Simulated Detector (for development without GPU) ──────────

class SimulatedDetector(DetectionModel):
    """Generates synthetic detections for pipeline testing.

    Returns realistic-looking detections for predefined object types
    without requiring any ML model or GPU.
    """

    SIMULATED_OBJECTS = {
        "wrench": {
            "class_id": 0, "category": GraspCategory.TOOL,
            "mass": 0.32, "friction": 0.65,
            "bbox_size": (200, 80),
        },
        "mug": {
            "class_id": 1, "category": GraspCategory.CONTAINER,
            "mass": 0.28, "friction": 0.55,
            "bbox_size": (120, 140),
        },
        "sphere": {
            "class_id": 2, "category": GraspCategory.SPHERE,
            "mass": 0.15, "friction": 0.70,
            "bbox_size": (100, 100),
        },
        "bottle": {
            "class_id": 3, "category": GraspCategory.CYLINDER,
            "mass": 0.45, "friction": 0.50,
            "bbox_size": (80, 180),
        },
    }

    def __init__(self, objects: Optional[List[str]] = None):
        self._objects = objects or ["wrench"]
        self._loaded = False

    def load(self, device: str = "auto") -> None:
        self._loaded = True
        logger.info(f"Simulated detector loaded with objects: {self._objects}")

    def detect(self, frame: BroteusFrame) -> DetectionResult:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        start = time.time()
        detections = []

        for i, obj_name in enumerate(self._objects):
            obj_data = self.SIMULATED_OBJECTS.get(obj_name)
            if obj_data is None:
                continue

            # Place detection near center with slight offset per object
            cx = frame.width / 2 + (i - len(self._objects) / 2) * 150
            cy = frame.height / 2
            bw, bh = obj_data["bbox_size"]

            # Add slight jitter for realism
            jitter_x = np.random.normal(0, 2)
            jitter_y = np.random.normal(0, 2)

            det = Detection(
                class_name=obj_name,
                class_id=obj_data["class_id"],
                confidence=0.85 + np.random.uniform(0, 0.14),
                bbox=BoundingBox(
                    x1=cx - bw / 2 + jitter_x,
                    y1=cy - bh / 2 + jitter_y,
                    x2=cx + bw / 2 + jitter_x,
                    y2=cy + bh / 2 + jitter_y,
                ),
                grasp_category=obj_data["category"],
                estimated_mass_kg=obj_data["mass"],
                estimated_friction=obj_data["friction"],
            )

            # Depth from frame if available
            if frame.has_depth:
                icx = int(max(0, min(cx, frame.width - 1)))
                icy = int(max(0, min(cy, frame.height - 1)))
                det.depth_estimate_m = float(frame.depth.to_meters()[icy, icx])

            detections.append(det)

        inference_ms = (time.time() - start) * 1000

        return DetectionResult(
            frame_id=frame.frame_id,
            detections=detections,
            inference_time_ms=inference_ms,
            model_name="simulated",
        )

    def get_model_info(self) -> dict:
        return {
            "name": "SimulatedDetector",
            "type": "simulated",
            "objects": self._objects,
            "loaded": self._loaded,
        }
