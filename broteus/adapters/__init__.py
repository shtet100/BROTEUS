"""BROTEUS Camera Adapters — Universal Camera Adapter Interface."""

from broteus.adapters.base import CameraAdapter
from broteus.adapters.webcam import WebcamAdapter
from broteus.adapters.video_file import VideoFileAdapter
from broteus.adapters.synthetic import SyntheticAdapter

__all__ = [
    "CameraAdapter",
    "WebcamAdapter",
    "VideoFileAdapter",
    "SyntheticAdapter",
]
