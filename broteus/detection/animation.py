"""
BROTEUS Hand Animation Recognition
=====================================

Recognizes dynamic hand gestures — movements over time, not static poses.
Uses Dynamic Time Warping (DTW) to match temporal sequences of hand features.

Architecture:
- Maintains a sliding window of recent hand feature vectors (~5 seconds)
- Recorded animations are stored as sequences of feature vectors
- Classification: DTW compares the sliding window against stored animations
- DTW handles speed variations (fast wave vs slow wave = same gesture)

Examples of animations vs static gestures:
  Static: open palm (single pose) → "stop"
  Animation: beckoning curl (repeated finger motion) → "come here"
  Static: fist (single pose) → "grab"
  Animation: fist opening slowly (transition) → "release"
  Animation: hand waving side to side → "wave/hello"
  Animation: circular hand motion → "spin/rotate"

Recording flow:
1. User presses RECORD ANIMATION
2. Performs the hand animation (e.g., beckoning motion)
3. Presses STOP — BROTEUS captures the full temporal sequence
4. Names it and assigns an ORION action
5. On classification, BROTEUS continuously compares recent hand movement
   against all stored animations using DTW

Feature compression for temporal sequences:
  Instead of storing all 35 features per frame, animations use a
  compressed 12-dim temporal feature vector focused on movement:
  - 5 finger curl angles (the primary movement signal)
  - 3 palm normal components (hand orientation over time)
  - 2 palm center position (x, y trajectory)
  - 2 hand velocity (dx, dy between frames)
"""

import json
import logging
import math
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from collections import deque
from pathlib import Path

logger = logging.getLogger("broteus.animation")

ANIM_MEMORY_DIR = Path("broteus_memory") / "animations"


@dataclass
class AnimationResult:
    name: str
    action: str
    confidence: float
    matched: bool         # True if an animation was recognized
    progress: float       # 0-1 how much of the animation has been performed
    recording: bool


# ── Temporal Feature Extraction ───────────────────────────────

def _xyz(lm):
    return (lm.x, lm.y, lm.z)

def _angle_3d(a, b, c):
    ba = np.array([a[i]-b[i] for i in range(3)])
    bc = np.array([c[i]-b[i] for i in range(3)])
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return math.degrees(math.acos(np.clip(cos_a, -1, 1)))

FINGERS = [(1,2,3,4), (5,6,7,8), (9,10,11,12), (13,14,15,16), (17,18,19,20)]


def extract_temporal_features(landmarks, prev_center=None) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Extract 12-dim temporal feature vector optimized for motion.
    
    Returns (features, palm_center) where palm_center is used
    to compute velocity on the next frame.
    """
    lm = [_xyz(l) for l in landmarks]
    wrist = lm[0]

    features = []

    # 1. Finger curl angles (5) — primary movement signal
    for f in FINGERS:
        features.append(_angle_3d(lm[f[1]], lm[f[2]], lm[f[3]]) / 180.0)

    # 2. Palm normal (3) — orientation changes during animation
    v_up = np.array([lm[9][i] - wrist[i] for i in range(3)])
    v_side = np.array([lm[17][i] - wrist[i] for i in range(3)])
    normal = np.cross(v_up, v_side)
    norm_len = np.linalg.norm(normal)
    if norm_len > 1e-8:
        normal = normal / norm_len
    features.extend(normal.tolist())

    # 3. Palm center position (2) — trajectory of the hand
    cx = (wrist[0] + lm[9][0]) / 2
    cy = (wrist[1] + lm[9][1]) / 2
    features.append(cx)
    features.append(cy)

    # 4. Velocity (2) — speed and direction of hand movement
    if prev_center is not None:
        dx = cx - prev_center[0]
        dy = cy - prev_center[1]
    else:
        dx, dy = 0.0, 0.0
    features.append(dx * 20)  # Scale up for visibility
    features.append(dy * 20)

    return np.array(features, dtype=np.float32), (cx, cy)


# ── Dynamic Time Warping ─────────────────────────────────────

def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray, band_ratio: float = 0.5) -> float:
    """Compute DTW distance between two sequences with Sakoe-Chiba band.
    
    Args:
        seq_a: (N, D) array — first sequence
        seq_b: (M, D) array — second sequence
        band_ratio: fraction of sequence length for warping band constraint
        
    Returns:
        Normalized DTW distance (lower = more similar)
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return float('inf')

    # Sakoe-Chiba band width
    band = max(int(max(n, m) * band_ratio), abs(n - m) + 1)

    # Cost matrix (inf outside band)
    cost = np.full((n + 1, m + 1), float('inf'))
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - band)
        j_end = min(m, i + band) + 1
        for j in range(j_start, j_end):
            d = np.linalg.norm(seq_a[i-1] - seq_b[j-1])
            cost[i, j] = d + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])

    # Normalize by path length
    path_len = n + m
    return cost[n, m] / path_len


def dtw_similarity(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """Convert DTW distance to similarity score (0-1, higher = more similar)."""
    dist = dtw_distance(seq_a, seq_b)
    # Softer decay — typical distances range 0.1-1.5
    return math.exp(-dist * 1.0)


# ── Animation Memory ─────────────────────────────────────────

class AnimationMemory:
    """Stores recorded hand animation sequences.
    
    Each animation has multiple recorded sequences (from different speeds/angles).
    Classification uses DTW to find the best matching stored sequence.
    """

    def __init__(self, hand_id="default"):
        self._filename = f"animations_{hand_id}.json"
        self.animations: Dict[str, Dict] = {}
        # {name: {sequences: [list of sequences], action: str}}
        # Each sequence is a list of 12-dim feature vectors
        self._load()

    def store(self, name: str, sequence: List[np.ndarray], action: str = "custom"):
        """Store a recorded animation sequence."""
        if name not in self.animations:
            self.animations[name] = {"sequences": [], "action": action}
        self.animations[name]["sequences"].append([s.tolist() for s in sequence])
        self.animations[name]["action"] = action
        self._save()
        n = len(self.animations[name]["sequences"])
        logger.info(f"Animation '{name}': {n} recordings stored")

    def classify(self, window: np.ndarray, threshold: float = 0.35) -> Optional[Tuple[str, str, float]]:
        """Match a sliding window against all stored animations.
        
        Uses only 2 sub-window lengths for speed.
        """
        if not self.animations or len(window) < 10:
            return None

        best_name, best_action, best_sim = None, None, 0.0

        for name, data in self.animations.items():
            for seq in data["sequences"]:
                stored = np.array(seq, dtype=np.float32)
                if len(stored) < 5:
                    continue

                stored_len = len(stored)
                # Try 3 ratios: shorter, exact, and longer window
                for ratio in [0.7, 1.0, 1.3]:
                    sub_len = min(len(window), max(10, int(stored_len * ratio)))
                    sub_window = window[-sub_len:]
                    sim = dtw_similarity(sub_window, stored)
                    if sim > best_sim:
                        best_sim = sim
                        best_name = name
                        best_action = data["action"]

        if best_sim >= threshold:
            logger.info(f"Animation match: '{best_name}' sim={best_sim:.3f}")
            return best_name, best_action, best_sim
        elif best_sim > 0.15 and best_name:
            logger.debug(f"Animation near-miss: '{best_name}' sim={best_sim:.3f} (threshold={threshold})")
        return None

    def get_names(self) -> list:
        return list(self.animations.keys())

    def get_counts(self) -> dict:
        return {n: len(d["sequences"]) for n, d in self.animations.items()}

    def remove(self, name: str) -> bool:
        if name in self.animations:
            del self.animations[name]
            self._save()
            return True
        return False

    def clear(self):
        self.animations = {}
        self._save()

    def _save(self):
        ANIM_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        with open(ANIM_MEMORY_DIR / self._filename, 'w') as f:
            json.dump(self.animations, f)

    def _load(self):
        path = ANIM_MEMORY_DIR / self._filename
        if path.exists():
            try:
                with open(path) as f:
                    self.animations = json.load(f)
                total = sum(len(d["sequences"]) for d in self.animations.values())
                if self.animations:
                    logger.info(f"Loaded {len(self.animations)} animations ({total} recordings) from {self._filename}")
            except Exception:
                self.animations = {}


# ── Animation Recognizer ──────────────────────────────────────

class AnimationRecognizer:
    """Recognizes hand animations (temporal gestures) in real-time.
    
    Pipeline:
    1. Each frame: extract 12-dim temporal features, push to sliding window
    2. Every N frames: compare window against stored animations via DTW
    3. If match found above threshold: fire the animation event
    4. Cooldown period prevents re-triggering the same animation immediately
    
    The sliding window maintains ~5 seconds of history at ~20 FPS (100 frames).
    """

    WINDOW_SIZE = 60       # ~3 seconds at 20 FPS
    CHECK_INTERVAL = 3     # Check DTW every 3 frames — responsive
    COOLDOWN_FRAMES = 20   # ~1 second cooldown between re-triggers

    def __init__(self, hand_id="default"):
        self.hand_id = hand_id
        self.memory = AnimationMemory(hand_id=hand_id)
        self._window = deque(maxlen=self.WINDOW_SIZE)
        self._prev_center = None
        self._frame_count = 0
        self._last_match = None
        self._cooldown = 0
        self._recording = False
        self._recorded_sequence = []
        self._recorded_prev_center = None

    def process_frame(self, landmarks) -> Optional[AnimationResult]:
        """Process one frame and return animation result if detected.
        
        Call this every frame with the hand landmarks.
        Returns AnimationResult when an animation is recognized.
        """
        if not landmarks or len(landmarks) < 21:
            self._prev_center = None
            return AnimationResult(name="", action="", confidence=0,
                                   matched=False, progress=0, recording=self._recording)

        # Extract temporal features
        features, center = extract_temporal_features(landmarks, self._prev_center)
        self._prev_center = center

        # Push to sliding window
        self._window.append(features)

        # Record if in recording mode
        if self._recording:
            rec_features, self._recorded_prev_center = extract_temporal_features(
                landmarks, self._recorded_prev_center
            )
            self._recorded_sequence.append(rec_features)
            return AnimationResult(name="", action="", confidence=0,
                                   matched=False, progress=0, recording=True)

        # Cooldown
        if self._cooldown > 0:
            self._cooldown -= 1

        # Classify every N frames
        self._frame_count += 1
        if self._frame_count % self.CHECK_INTERVAL == 0 and len(self._window) >= 15:
            if self._cooldown <= 0 and self.memory.animations:
                window_arr = np.array(list(self._window), dtype=np.float32)
                match = self.memory.classify(window_arr, threshold=0.35)
                if match:
                    name, action, sim = match
                    self._last_match = (name, action, sim)
                    self._cooldown = self.COOLDOWN_FRAMES
                    logger.info(f"Animation detected: '{name}' ({sim:.2f})")
                    return AnimationResult(name=name, action=action, confidence=sim,
                                           matched=True, progress=1.0, recording=False)

        # Return last match info if in cooldown (keeps showing it)
        if self._last_match and self._cooldown > 0:
            name, action, sim = self._last_match
            return AnimationResult(name=name, action=action, confidence=sim,
                                   matched=True, progress=self._cooldown / self.COOLDOWN_FRAMES,
                                   recording=False)

        return AnimationResult(name="", action="", confidence=0,
                               matched=False, progress=0, recording=self._recording)

    def start_recording(self):
        """Start recording an animation sequence."""
        self._recording = True
        self._recorded_sequence = []
        self._recorded_prev_center = None
        logger.info(f"Animation recording started ({self.hand_id})")

    def stop_recording(self) -> int:
        """Stop recording and return frame count."""
        self._recording = False
        n = len(self._recorded_sequence)
        logger.info(f"Animation recording stopped: {n} frames")
        return n

    def save_animation(self, name: str, action: str = "custom") -> int:
        """Save recorded animation with a name."""
        seq = self._recorded_sequence
        if not seq or len(seq) < 10:
            logger.warning(f"Animation too short ({len(seq)} frames), need at least 10")
            return 0
        self.memory.store(name, seq, action)
        n = len(seq)
        self._recorded_sequence = []
        return n

    def reset(self):
        """Reset the sliding window and state."""
        self._window.clear()
        self._prev_center = None
        self._last_match = None
        self._cooldown = 0