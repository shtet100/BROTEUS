"""
BROTEUS Object Tracker
=======================

Simple IoU-based multi-object tracker. Assigns persistent IDs to detections
and maintains them across frames even when YOLO temporarily loses an object.

No external dependencies — pure numpy.
"""

import numpy as np
from typing import List, Dict, Optional


class TrackedObject:
    """A tracked object with persistent identity."""

    _next_id = 1

    def __init__(self, detection: dict):
        self.id = TrackedObject._next_id
        TrackedObject._next_id += 1
        self.detection = detection.copy()
        self.bbox = detection['bbox']  # [x1, y1, x2, y2]
        self.class_name = detection.get('class_name', '?')
        self.confidence = detection.get('confidence', 0)
        self.age = 0                # Frames since creation
        self.frames_seen = 1        # Total frames this object was matched
        self.frames_missing = 0     # Consecutive frames without a match
        self.stable_class = None    # Most common CLIP class assignment
        self.class_votes = {}       # {class_name: count} for voting
        self._vote(self.class_name)
        # If no valid votes yet, show placeholder
        if not self.stable_class:
            self.class_name = 'detecting...'

    def _vote(self, class_name: str):
        """Record a classification vote. Ignores placeholder labels."""
        if class_name in ('detecting...', '?', ''):
            return  # Don't vote on placeholders
        self.class_votes[class_name] = self.class_votes.get(class_name, 0) + 1
        # Stable class = most voted
        self.stable_class = max(self.class_votes, key=self.class_votes.get)

    def update(self, detection: dict):
        """Update with a new matching detection."""
        self.detection = detection.copy()
        self.detection['mask'] = detection.get('mask')  # Keep mask reference
        self.bbox = detection['bbox']
        self.confidence = detection.get('confidence', 0)
        self.frames_seen += 1
        self.frames_missing = 0
        self._vote(detection.get('class_name', '?'))
        # Use stable class if we have CLIP votes, otherwise show placeholder
        if self.stable_class:
            self.class_name = self.stable_class
        else:
            self.class_name = 'detecting...'

    def mark_missing(self):
        """Called when no detection matches this track."""
        self.frames_missing += 1

    @property
    def is_confirmed(self) -> bool:
        """Object has been seen enough times to be considered real."""
        return self.frames_seen >= 3

    @property
    def is_lost(self) -> bool:
        """Object hasn't been seen for too long."""
        return self.frames_missing > 5  # ~1 second at 5fps YOLO

    def to_detection_dict(self) -> dict:
        """Export as a detection dict with stable class and track ID."""
        d = self.detection.copy()
        d['class_name'] = self.stable_class
        d['track_id'] = self.id
        d['frames_seen'] = self.frames_seen
        d['confirmed'] = self.is_confirmed
        return d


def compute_iou(box_a, box_b):
    """Compute intersection-over-union between two [x1,y1,x2,y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / (union + 1e-6)


class ObjectTracker:
    """Multi-object tracker using IoU matching.

    Maintains a list of TrackedObjects across frames.
    Handles:
    - Assigning persistent IDs
    - Matching new detections to existing tracks
    - Keeping tracks alive through brief detection gaps
    - Voting on stable class labels (prevents flickering)
    - Filtering out spurious single-frame detections
    """

    def __init__(self, iou_threshold: float = 0.25, max_missing: int = 15):
        self.tracks: List[TrackedObject] = []
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing

    def update(self, detections: List[dict]) -> List[dict]:
        """Process a new set of detections.

        Args:
            detections: List of detection dicts with 'bbox', 'class_name', etc.

        Returns:
            List of stable, tracked detection dicts with 'track_id' added.
            Only returns confirmed tracks (seen >= 2 frames).
        """
        # Age all existing tracks
        for track in self.tracks:
            track.age += 1

        if not detections:
            # No detections — mark all as missing
            for track in self.tracks:
                track.mark_missing()
            self._prune()
            return self._export()

        if not self.tracks:
            # No existing tracks — create new ones for all detections
            for det in detections:
                self.tracks.append(TrackedObject(det))
            return self._export()

        # Compute IoU matrix: tracks × detections
        n_tracks = len(self.tracks)
        n_dets = len(detections)
        iou_matrix = np.zeros((n_tracks, n_dets))

        for ti, track in enumerate(self.tracks):
            for di, det in enumerate(detections):
                iou_matrix[ti, di] = compute_iou(track.bbox, det['bbox'])

        # Greedy matching: highest IoU first
        matched_tracks = set()
        matched_dets = set()

        while True:
            if iou_matrix.size == 0:
                break
            best = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            ti, di = best
            if iou_matrix[ti, di] < self.iou_threshold:
                break
            # Match
            self.tracks[ti].update(detections[di])
            matched_tracks.add(ti)
            matched_dets.add(di)
            # Zero out this row and column
            iou_matrix[ti, :] = 0
            iou_matrix[:, di] = 0

        # Unmatched tracks → missing
        for ti in range(n_tracks):
            if ti not in matched_tracks:
                self.tracks[ti].mark_missing()

        # Unmatched detections → new tracks
        for di in range(n_dets):
            if di not in matched_dets:
                self.tracks.append(TrackedObject(detections[di]))

        self._prune()
        return self._export()

    def _prune(self):
        """Remove tracks that have been lost for too long."""
        self.tracks = [t for t in self.tracks if not t.is_lost]

    def _export(self) -> List[dict]:
        """Export confirmed tracks as detection dicts."""
        return [t.to_detection_dict() for t in self.tracks if t.is_confirmed]

    def get_track_count(self) -> int:
        return len([t for t in self.tracks if t.is_confirmed])

    def reset(self):
        self.tracks = []
        TrackedObject._next_id = 1
