"""
BROTEUS Camera Adapter — Abstract Base
========================================

This is the contract. Every camera that feeds BROTEUS implements this interface.

Write one thin adapter per camera type, and the entire BROTEUS pipeline works
identically regardless of source. A webcam, a Mars rover nav cam, a surgical
endoscope, a satellite downlink decades from now — same interface, same frames.

Usage:
    class MyCamera(CameraAdapter):
        def _open(self): ...
        def _read_frame(self) -> BroteusFrame: ...
        def _close(self): ...

    async with MyCamera(source="...") as cam:
        async for frame in cam.stream():
            pipeline.process(frame)
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import AsyncIterator, Optional

from broteus.core.frame import BroteusFrame, FrameMetadata, FrameSource
from broteus.core.config import StreamConfig

logger = logging.getLogger("broteus.adapter")


class CameraAdapter(ABC):
    """Abstract base for all BROTEUS camera adapters.

    Every adapter must implement three methods:
        _open()       — Initialize the camera/connection
        _read_frame() — Capture and return one BroteusFrame
        _close()      — Release resources

    The base class handles:
        - Async streaming with frame rate control
        - Frame buffering and drop policy
        - Sequence numbering
        - Latency tracking
        - Graceful lifecycle management
    """

    def __init__(
        self,
        source: str,
        source_type: FrameSource = FrameSource.UNKNOWN,
        source_name: Optional[str] = None,
        config: Optional[StreamConfig] = None,
    ):
        self.source = source
        self.source_type = source_type
        self.source_name = source_name or f"{source_type.value}:{source}"
        self.config = config or StreamConfig()

        # State
        self._is_open = False
        self._sequence = 0
        self._frame_buffer: deque[BroteusFrame] = deque(maxlen=self.config.buffer_size)
        self._frames_captured = 0
        self._frames_dropped = 0
        self._start_time: float = 0.0
        self._last_frame_time: float = 0.0

    # ── Abstract Methods (implement per camera) ────────────────────

    @abstractmethod
    def _open(self) -> None:
        """Initialize camera hardware/connection. Called once on start."""
        ...

    @abstractmethod
    def _read_frame(self) -> Optional[BroteusFrame]:
        """Capture a single frame. Return None if no frame available."""
        ...

    @abstractmethod
    def _close(self) -> None:
        """Release camera resources. Called once on shutdown."""
        ...

    # ── Optional Overrides ─────────────────────────────────────────

    def get_intrinsics(self):
        """Override to provide camera intrinsic parameters."""
        return None

    def get_capabilities(self) -> dict:
        """Override to report camera capabilities."""
        return {
            "source_type": self.source_type.value,
            "source_name": self.source_name,
            "has_depth": False,
            "has_intrinsics": self.get_intrinsics() is not None,
            "max_fps": self.config.max_fps,
        }

    # ── Lifecycle ──────────────────────────────────────────────────

    def open(self) -> None:
        """Open the camera adapter."""
        if self._is_open:
            logger.warning(f"[{self.source_name}] Already open")
            return
        logger.info(f"[{self.source_name}] Opening camera adapter...")
        self._open()
        self._is_open = True
        self._start_time = time.time()
        self._sequence = 0
        logger.info(f"[{self.source_name}] Camera adapter ready")

    def close(self) -> None:
        """Close the camera adapter and release resources."""
        if not self._is_open:
            return
        logger.info(
            f"[{self.source_name}] Closing. "
            f"Captured: {self._frames_captured}, Dropped: {self._frames_dropped}"
        )
        self._close()
        self._is_open = False
        self._frame_buffer.clear()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    async def __aenter__(self):
        self.open()
        return self

    async def __aexit__(self, *exc):
        self.close()

    # ── Frame Capture ──────────────────────────────────────────────

    def capture(self) -> Optional[BroteusFrame]:
        """Capture a single frame with metadata injection."""
        if not self._is_open:
            raise RuntimeError(f"[{self.source_name}] Adapter not open")

        frame = self._read_frame()
        if frame is None:
            return None

        # Inject adapter-level metadata
        self._sequence += 1
        now = time.time()

        frame.metadata = FrameMetadata(
            source_type=self.source_type,
            source_id=self.source,
            source_name=self.source_name,
            sequence_number=self._sequence,
            capture_timestamp=frame.metadata.capture_timestamp or now,
            arrival_timestamp=now,
            intrinsics=frame.metadata.intrinsics or self.get_intrinsics(),
            custom=frame.metadata.custom,
        )

        self._frames_captured += 1
        self._last_frame_time = now
        return frame

    # ── Async Streaming ────────────────────────────────────────────

    async def stream(self) -> AsyncIterator[BroteusFrame]:
        """Async generator that yields frames at the configured rate.

        Usage:
            async for frame in adapter.stream():
                process(frame)
        """
        if not self._is_open:
            raise RuntimeError(f"[{self.source_name}] Adapter not open")

        frame_interval = 1.0 / self.config.max_fps if self.config.max_fps > 0 else 0
        logger.info(
            f"[{self.source_name}] Streaming at {self.config.max_fps} fps "
            f"(interval={frame_interval*1000:.1f}ms)"
        )

        while self._is_open:
            loop_start = time.time()

            frame = self.capture()
            if frame is not None:
                yield frame

            # Frame rate control
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                # We're running behind — yield control briefly
                await asyncio.sleep(0)

    # ── Stats ──────────────────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def stats(self) -> dict:
        elapsed = time.time() - self._start_time if self._start_time else 0
        return {
            "source_name": self.source_name,
            "is_open": self._is_open,
            "frames_captured": self._frames_captured,
            "frames_dropped": self._frames_dropped,
            "elapsed_seconds": round(elapsed, 2),
            "effective_fps": round(self._frames_captured / elapsed, 2) if elapsed > 0 else 0,
            "last_frame_time": self._last_frame_time,
        }

    def __repr__(self) -> str:
        status = "OPEN" if self._is_open else "CLOSED"
        return f"CameraAdapter({self.source_name}, {status}, seq={self._sequence})"
