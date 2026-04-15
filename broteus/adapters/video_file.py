"""
BROTEUS Video File Adapter
============================

Plays back recorded video files (MP4, AVI, MOV, etc.) through the
BROTEUS pipeline. Useful for testing, demos, and offline analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from broteus.adapters.base import CameraAdapter
from broteus.core.frame import (
    BroteusFrame,
    ColorSpace,
    FrameMetadata,
    FrameSource,
)
from broteus.core.config import StreamConfig

logger = logging.getLogger("broteus.adapter.video_file")


class VideoFileAdapter(CameraAdapter):
    """Camera adapter for video file playback.

    Usage:
        async with VideoFileAdapter("demo.mp4") as cam:
            async for frame in cam.stream():
                process(frame)
    """

    def __init__(
        self,
        filepath: str,
        loop: bool = True,
        source_name: Optional[str] = None,
        config: Optional[StreamConfig] = None,
    ):
        self._filepath = Path(filepath)
        self._loop = loop
        self._cap: Optional[cv2.VideoCapture] = None
        self._total_frames: int = 0

        if not self._filepath.exists():
            raise FileNotFoundError(f"Video file not found: {self._filepath}")

        super().__init__(
            source=str(filepath),
            source_type=FrameSource.VIDEO_FILE,
            source_name=source_name or f"Video:{self._filepath.name}",
            config=config,
        )

    def _open(self) -> None:
        self._cap = cv2.VideoCapture(str(self._filepath))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {self._filepath}")

        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        logger.info(
            f"[{self.source_name}] Opened: {w}x{h} @ {fps}fps, "
            f"{self._total_frames} frames, loop={self._loop}"
        )

    def _read_frame(self) -> Optional[BroteusFrame]:
        if self._cap is None:
            return None

        ret, bgr_image = self._cap.read()
        if not ret or bgr_image is None:
            if self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, bgr_image = self._cap.read()
                if not ret:
                    return None
            else:
                self._is_open = False
                return None

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        return BroteusFrame(
            image=rgb_image,
            color_space=ColorSpace.RGB,
            fps=self._cap.get(cv2.CAP_PROP_FPS) or 30.0,
            metadata=FrameMetadata(),
        )

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def progress(self) -> float:
        """Playback progress 0.0 → 1.0."""
        if self._cap is None or self._total_frames == 0:
            return 0.0
        pos = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
        return pos / self._total_frames
