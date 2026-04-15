"""
BROTEUS Configuration
======================

Central configuration for the BROTEUS subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StreamConfig:
    """Configuration for frame streaming."""
    max_fps: float = 30.0
    jpeg_quality: int = 85
    buffer_size: int = 5           # Max frames in the buffer before dropping
    enable_depth: bool = True      # Request depth if available
    target_width: Optional[int] = None   # Resize frames (None = native)
    target_height: Optional[int] = None


@dataclass
class DetectionConfig:
    """Configuration for the detection layer."""
    model_name: str = "yolov8n"    # Default lightweight model
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    max_detections: int = 20
    device: str = "auto"           # "auto", "cuda", "cpu", "mps"


@dataclass
class GraspConfig:
    """Configuration for the grasp intelligence system."""
    grid_resolution: int = 50      # Number of candidate contact points per axis
    min_contact_points: int = 2    # Minimum points for a valid grasp
    max_contact_points: int = 5    # Maximum simultaneous contact points
    friction_coefficient: float = 0.6   # Default friction estimate
    stability_threshold: float = 0.7    # Min score for "green" zone
    acceptable_threshold: float = 0.4   # Min score for "yellow" zone
    force_limit_n: float = 20.0         # Maximum grip force (Newtons)


@dataclass
class VisualizationConfig:
    """Configuration for the Three.js visualization."""
    grid_point_size: float = 0.02       # Size of spatial grid nodes
    optimal_color: str = "#00ff88"      # Green — optimal grip zones
    acceptable_color: str = "#ffaa00"   # Yellow — acceptable grip zones
    risk_color: str = "#ff3344"         # Red — slip-risk zones
    hand_opacity: float = 0.85
    animate_approach: bool = True
    show_force_vectors: bool = True
    show_surface_normals: bool = False


@dataclass
class APIConfig:
    """Configuration for the FastAPI server."""
    host: str = "0.0.0.0"
    port: int = 8100                    # ORION is 8000, ATHENA is 8050
    ws_path: str = "/ws/frames"
    cors_origins: list = field(default_factory=lambda: ["*"])
    enable_docs: bool = True


@dataclass
class BroteusConfig:
    """Master configuration for the BROTEUS subsystem."""
    stream: StreamConfig = field(default_factory=StreamConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    grasp: GraspConfig = field(default_factory=GraspConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    api: APIConfig = field(default_factory=APIConfig)

    # ORION integration
    orion_host: str = "localhost"
    orion_port: int = 8000
    subsystem_id: str = "BROTEUS"
    subsystem_version: str = "0.1.0"
