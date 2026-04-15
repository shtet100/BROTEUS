"""
BROTEUS Standard Frame Format
==============================

Every camera in the world — webcam, RealSense, endoscope, Mars rover nav cam,
satellite downlink — produces one thing: a BroteusFrame.

This is the Universal Camera Adapter Interface's contract. Write one thin adapter
per camera type, deliver frames in this format, and the entire BROTEUS pipeline
works identically regardless of source.

Cameras that don't exist yet will plug in the same way.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class ColorSpace(Enum):
    """Supported color space encodings."""
    RGB = "rgb"
    BGR = "bgr"
    RGBA = "rgba"
    GRAYSCALE = "grayscale"
    # Depth-specific
    DEPTH_UINT16 = "depth_uint16"      # mm precision (RealSense, Kinect)
    DEPTH_FLOAT32 = "depth_float32"    # meter precision


class FrameSource(Enum):
    """Classification of the camera source type."""
    WEBCAM = "webcam"
    VIDEO_FILE = "video_file"
    HTTP_STREAM = "http_stream"
    WEBSOCKET_STREAM = "websocket_stream"
    REALSENSE = "realsense"
    KINECT = "kinect"
    STEREO_PAIR = "stereo_pair"
    DRONE = "drone"
    ENDOSCOPE = "endoscope"
    MICROSCOPE = "microscope"
    AR_VR_PASSTHROUGH = "ar_vr_passthrough"
    SYNTHETIC = "synthetic"            # Generated/simulated frames
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DepthInfo:
    """Optional depth channel metadata.

    When a camera provides depth data (RealSense, Kinect, stereo pair,
    LiDAR-fused), this structure describes how to interpret it.
    """
    depth_map: np.ndarray               # H x W depth array
    color_space: ColorSpace = ColorSpace.DEPTH_UINT16
    min_depth_m: float = 0.1            # Minimum reliable depth (meters)
    max_depth_m: float = 10.0           # Maximum reliable depth (meters)
    depth_scale: float = 0.001          # Multiplier to convert raw values → meters
    is_aligned: bool = True             # Whether depth is aligned to color frame

    def to_meters(self) -> np.ndarray:
        """Convert raw depth values to meters."""
        return self.depth_map.astype(np.float32) * self.depth_scale

    def __post_init__(self):
        if self.depth_map.ndim != 2:
            raise ValueError(
                f"Depth map must be 2D (H x W), got shape {self.depth_map.shape}"
            )


@dataclass(frozen=True)
class IntrinsicParameters:
    """Camera intrinsic parameters for 3D projection.

    Required for spatial grid projection (Layer 2 of BROTEUS pipeline).
    Without these, we can't map 2D detections to 3D grasp coordinates.
    """
    fx: float               # Focal length x (pixels)
    fy: float               # Focal length y (pixels)
    cx: float               # Principal point x (pixels)
    cy: float               # Principal point y (pixels)
    distortion: Optional[np.ndarray] = None  # Distortion coefficients (k1,k2,p1,p2,k3)
    
    @property
    def matrix(self) -> np.ndarray:
        """3x3 camera intrinsic matrix."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)


@dataclass(frozen=True)
class FrameMetadata:
    """Rich metadata attached to every frame.

    This is what makes BROTEUS context-aware. A frame isn't just pixels —
    it carries knowledge about where it came from, when, and how to interpret it.
    """
    source_type: FrameSource = FrameSource.UNKNOWN
    source_id: str = ""                   # Unique identifier for the camera device
    source_name: str = ""                 # Human-readable name ("Left Arm RealSense")
    sequence_number: int = 0              # Frame counter from adapter
    capture_timestamp: float = 0.0        # When the frame was captured (epoch)
    arrival_timestamp: float = 0.0        # When BROTEUS received it (epoch)
    intrinsics: Optional[IntrinsicParameters] = None
    exposure_us: Optional[float] = None   # Exposure time in microseconds
    gain: Optional[float] = None          # Camera gain
    temperature_c: Optional[float] = None # Sensor temperature (for thermal drift)
    custom: dict = field(default_factory=dict)  # Extensible metadata bucket


@dataclass
class BroteusFrame:
    """The universal frame — BROTEUS's atomic unit of perception.

    Every camera adapter in the system produces exactly this. The entire
    downstream pipeline (detection, spatial grid, grasp scoring, hand planning,
    manipulation) consumes exactly this.

    Attributes:
        frame_id: Globally unique frame identifier.
        image: Raw pixel data as numpy array (H x W x C) or (H x W) for grayscale.
        color_space: How to interpret the pixel values.
        width: Frame width in pixels.
        height: Frame height in pixels.
        channels: Number of color channels.
        fps: Reported frame rate from the source.
        depth: Optional depth channel (from RGBD cameras).
        metadata: Rich context about the frame source.
        timestamp: When this frame object was created (epoch seconds).
    """
    image: np.ndarray
    color_space: ColorSpace = ColorSpace.RGB
    fps: float = 30.0
    depth: Optional[DepthInfo] = None
    metadata: FrameMetadata = field(default_factory=FrameMetadata)
    frame_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.image.ndim == 2:
            self.color_space = ColorSpace.GRAYSCALE
        elif self.image.ndim != 3:
            raise ValueError(
                f"Image must be 2D (H x W) or 3D (H x W x C), got {self.image.ndim}D"
            )

        # Validate depth alignment if present
        if self.depth is not None and self.depth.is_aligned:
            dh, dw = self.depth.depth_map.shape
            ih, iw = self.image.shape[:2]
            if (dh, dw) != (ih, iw):
                raise ValueError(
                    f"Aligned depth shape ({dh}x{dw}) != image shape ({ih}x{iw})"
                )

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def channels(self) -> int:
        return self.image.shape[2] if self.image.ndim == 3 else 1

    @property
    def has_depth(self) -> bool:
        return self.depth is not None

    @property
    def has_intrinsics(self) -> bool:
        return self.metadata.intrinsics is not None

    @property
    def latency_ms(self) -> float:
        """Time from capture to BROTEUS arrival, in milliseconds."""
        if self.metadata.capture_timestamp > 0 and self.metadata.arrival_timestamp > 0:
            return (self.metadata.arrival_timestamp - self.metadata.capture_timestamp) * 1000
        return -1.0

    @property
    def shape_str(self) -> str:
        return f"{self.width}x{self.height}x{self.channels}"

    def to_rgb(self) -> np.ndarray:
        """Convert image to RGB regardless of source color space."""
        import cv2
        if self.color_space == ColorSpace.RGB:
            return self.image.copy()
        elif self.color_space == ColorSpace.BGR:
            return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        elif self.color_space == ColorSpace.RGBA:
            return cv2.cvtColor(self.image, cv2.COLOR_RGBA2RGB)
        elif self.color_space == ColorSpace.GRAYSCALE:
            return cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        raise ValueError(f"Cannot convert {self.color_space} to RGB")

    def to_jpeg_bytes(self, quality: int = 85) -> bytes:
        """Encode frame as JPEG bytes for streaming."""
        import cv2
        if self.color_space == ColorSpace.RGB:
            img = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        elif self.color_space == ColorSpace.BGR:
            img = self.image
        else:
            img = cv2.cvtColor(self.to_rgb(), cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()

    def to_dict(self) -> dict:
        """Serialize frame metadata (without pixel data) for API responses."""
        return {
            "frame_id": self.frame_id,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "color_space": self.color_space.value,
            "fps": self.fps,
            "has_depth": self.has_depth,
            "has_intrinsics": self.has_intrinsics,
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
            "source_type": self.metadata.source_type.value,
            "source_name": self.metadata.source_name,
            "sequence_number": self.metadata.sequence_number,
        }

    def __repr__(self) -> str:
        depth_str = f", depth={self.depth.depth_map.shape}" if self.has_depth else ""
        return (
            f"BroteusFrame("
            f"id={self.frame_id}, "
            f"{self.shape_str}, "
            f"{self.color_space.value}, "
            f"{self.fps}fps"
            f"{depth_str}, "
            f"src={self.metadata.source_type.value}"
            f")"
        )
