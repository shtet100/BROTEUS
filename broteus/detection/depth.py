"""
BROTEUS Depth Estimation
==========================

Monocular depth estimation using MiDaS.
Produces a per-pixel relative depth map from a single RGB camera frame.
Combined with segmentation masks, this gives us 3D contact point positions
and surface normal estimation via depth gradients.
"""

import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger("broteus.depth")


class DepthEstimator:
    """MiDaS-based monocular depth estimator.

    Produces relative depth maps from single RGB frames.
    'Relative' means closer objects have higher values, but the
    absolute scale is unknown without a depth camera.
    """

    def __init__(self, model_type: str = "MiDaS_small"):
        self.model_type = model_type
        self.model = None
        self.transform = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False

    def load(self):
        """Load the MiDaS model."""
        logger.info(f"Loading MiDaS ({self.model_type}) on {self.device}...")
        self.model = torch.hub.load(
            "intel-isl/MiDaS", self.model_type, trust_repo=True
        )
        self.model.to(self.device)
        self.model.eval()

        # Load the appropriate transform
        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        )
        if self.model_type == "MiDaS_small":
            self.transform = midas_transforms.small_transform
        else:
            self.transform = midas_transforms.default_transform

        self._loaded = True
        logger.info("MiDaS loaded")

    def estimate(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Estimate depth from a BGR frame.

        Args:
            bgr_frame: OpenCV BGR image (H, W, 3)

        Returns:
            Depth map (H, W) as float32. Higher values = closer to camera.
            Normalized to [0, 1] range.
        """
        if not self._loaded:
            raise RuntimeError("DepthEstimator not loaded. Call load() first.")

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=bgr_frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()

        # Normalize to [0, 1] — higher = closer
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)

        return depth.astype(np.float32)

    def compute_normals(self, depth: np.ndarray) -> np.ndarray:
        """Compute surface normals from a depth map using gradients.

        Args:
            depth: (H, W) depth map

        Returns:
            (H, W, 3) normal map with unit normals.
            Convention: [nx, ny, nz] where nz points toward camera.
        """
        # Compute gradients (change in depth along x and y)
        dz_dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=5) * 0.5
        dz_dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=5) * 0.5

        # Surface normal: cross product of tangent vectors
        # Tangent along x: (1, 0, dz/dx)
        # Tangent along y: (0, 1, dz/dy)
        # Normal = (-dz/dx, -dz/dy, 1) then normalize
        normals = np.zeros((*depth.shape, 3), dtype=np.float32)
        normals[:, :, 0] = -dz_dx
        normals[:, :, 1] = -dz_dy
        normals[:, :, 2] = 1.0

        # Normalize
        mag = np.linalg.norm(normals, axis=2, keepdims=True)
        mag[mag < 1e-8] = 1.0
        normals = normals / mag

        return normals

    @property
    def is_loaded(self):
        return self._loaded
