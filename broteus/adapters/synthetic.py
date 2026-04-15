"""
BROTEUS Synthetic Adapter
===========================

Generates synthetic frames for testing, demos, and development.
No camera hardware required — produces patterned or random frames
that exercise the full pipeline.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

import numpy as np

from broteus.adapters.base import CameraAdapter
from broteus.core.frame import (
    BroteusFrame,
    ColorSpace,
    DepthInfo,
    FrameMetadata,
    FrameSource,
    IntrinsicParameters,
)
from broteus.core.config import StreamConfig

logger = logging.getLogger("broteus.adapter.synthetic")


class SyntheticAdapter(CameraAdapter):
    """Generates synthetic frames for testing and development.

    Produces frames with animated patterns, optional simulated depth,
    and mock intrinsics. Perfect for pipeline testing without hardware.

    Usage:
        async with SyntheticAdapter(width=640, height=480) as cam:
            async for frame in cam.stream():
                process(frame)
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        generate_depth: bool = False,
        source_name: Optional[str] = None,
        config: Optional[StreamConfig] = None,
    ):
        self._width = width
        self._height = height
        self._generate_depth = generate_depth

        super().__init__(
            source=f"synthetic:{width}x{height}",
            source_type=FrameSource.SYNTHETIC,
            source_name=source_name or f"Synthetic:{width}x{height}",
            config=config,
        )

    def _open(self) -> None:
        logger.info(
            f"[{self.source_name}] Synthetic camera ready: "
            f"{self._width}x{self._height}, depth={self._generate_depth}"
        )

    def _read_frame(self) -> Optional[BroteusFrame]:
        t = time.time()

        # Generate an animated RGB pattern
        image = self._generate_pattern(t)

        # Optionally generate synthetic depth
        depth = None
        if self._generate_depth:
            depth = self._generate_depth_map(t)

        return BroteusFrame(
            image=image,
            color_space=ColorSpace.RGB,
            fps=self.config.max_fps,
            depth=depth,
            metadata=FrameMetadata(
                capture_timestamp=t,
                intrinsics=self.get_intrinsics(),
            ),
        )

    def _close(self) -> None:
        pass

    def _generate_pattern(self, t: float) -> np.ndarray:
        """Generate an animated test pattern with moving gradients."""
        h, w = self._height, self._width
        y_coords = np.linspace(0, 1, h)[:, np.newaxis]
        x_coords = np.linspace(0, 1, w)[np.newaxis, :]

        # Animated gradient with interference pattern
        phase = t * 0.5
        r = np.clip(
            (np.sin(x_coords * 6 + phase) * 0.5 + 0.5) * 200 + 30, 0, 255
        )
        g = np.clip(
            (np.sin(y_coords * 4 + phase * 0.7) * 0.5 + 0.5) * 200 + 30, 0, 255
        )
        b = np.clip(
            (np.cos((x_coords + y_coords) * 5 + phase * 1.3) * 0.5 + 0.5) * 200 + 30,
            0, 255,
        )

        image = np.stack([r * np.ones_like(y_coords),
                          g * np.ones_like(x_coords),
                          b], axis=-1).astype(np.uint8)
        # Ensure correct shape
        image = image[:h, :w, :3]
        return image

    def _generate_depth_map(self, t: float) -> DepthInfo:
        """Generate a synthetic depth map simulating a table scene."""
        h, w = self._height, self._width
        y_coords = np.linspace(0, 1, h)
        x_coords = np.linspace(0, 1, w)
        xx, yy = np.meshgrid(x_coords, y_coords)

        # Simulated table at ~1m with an object bump
        table_depth = 1000  # 1 meter in mm
        cx, cy = 0.5 + 0.1 * math.sin(t * 0.3), 0.5
        object_bump = 300 * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / 0.02
        )
        depth_map = (table_depth - object_bump).astype(np.uint16)

        return DepthInfo(
            depth_map=depth_map,
            min_depth_m=0.3,
            max_depth_m=3.0,
            depth_scale=0.001,
        )

    def get_intrinsics(self) -> IntrinsicParameters:
        """Return simulated camera intrinsics."""
        fx = fy = self._width  # Simple pinhole approximation
        cx = self._width / 2
        cy = self._height / 2
        return IntrinsicParameters(fx=fx, fy=fy, cx=cx, cy=cy)

    def get_capabilities(self) -> dict:
        caps = super().get_capabilities()
        caps.update({
            "width": self._width,
            "height": self._height,
            "has_depth": self._generate_depth,
            "has_intrinsics": True,
            "synthetic": True,
        })
        return caps
