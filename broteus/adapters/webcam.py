"""
BROTEUS Webcam Adapter
========================

OpenCV-based adapter for local USB/built-in cameras.
The most common development and prototyping input source.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import cv2
import numpy as np

from broteus.adapters.base import CameraAdapter
from broteus.core.frame import (
    BroteusFrame,
    ColorSpace,
    FrameMetadata,
    FrameSource,
    IntrinsicParameters,
)
from broteus.core.config import StreamConfig

logger = logging.getLogger("broteus.adapter.webcam")


class WebcamAdapter(CameraAdapter):
    """Camera adapter for local webcams via OpenCV.

    Usage:
        async with WebcamAdapter(device_index=0) as cam:
            async for frame in cam.stream():
                process(frame)
    """

    def __init__(
        self,
        device_index: Union[int, str] = 0,
        source_name: Optional[str] = None,
        config: Optional[StreamConfig] = None,
        backend: int = cv2.CAP_ANY,
    ):
        self._device_index = device_index
        self._backend = backend
        self._cap: Optional[cv2.VideoCapture] = None

        super().__init__(
            source=str(device_index),
            source_type=FrameSource.WEBCAM,
            source_name=source_name or f"Webcam:{device_index}",
            config=config,
        )

    def _open(self) -> None:
        self._cap = cv2.VideoCapture(self._device_index, self._backend)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open webcam at index {self._device_index}"
            )

        # Apply target resolution if specified
        if self.config.target_width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.target_width)
        if self.config.target_height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.target_height)

        # Report actual resolution
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        logger.info(f"[{self.source_name}] Opened: {w}x{h} @ {fps}fps")

    def _read_frame(self) -> Optional[BroteusFrame]:
        if self._cap is None:
            return None

        ret, bgr_image = self._cap.read()
        if not ret or bgr_image is None:
            self._frames_dropped += 1
            return None

        # Convert BGR (OpenCV default) → RGB (BROTEUS standard)
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

    def get_capabilities(self) -> dict:
        caps = super().get_capabilities()
        if self._cap is not None:
            caps.update({
                "width": int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "native_fps": self._cap.get(cv2.CAP_PROP_FPS),
                "backend": self._cap.getBackendName(),
            })
        return caps
