"""
BROTEUS Gesture Recognition v3.1
==================================

Learning-first gesture recognition with 3D rotation capture.

Recording flow:
1. User holds a hand gesture and presses Record
2. BROTEUS prompts: "Now rotate your hand slowly..."
3. Over 5 seconds, BROTEUS captures ~100+ feature samples
   across all orientations the user demonstrates
4. The gesture cluster covers the full rotation space
5. On classification, BROTEUS finds the closest orientation match

Feature vector (35 dimensions):
- 5 finger curl angles (3D, normalized 0-1)
- 5 fingertip-to-palm distances (normalized by hand size)
- 10 inter-fingertip distances (all pairs, normalized)
- 5 fingertip z-depths relative to wrist
- 4 thumb-to-each-finger proximities
- 3 palm normal vector (which way the palm faces)
- 3 palm direction vector (which way the fingers point)
"""

import math
import json
import logging
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger("broteus.gesture")

GESTURE_MEMORY_DIR = Path("broteus_memory") / "gestures"


class Gesture(Enum):
    NONE = "none"
    OPEN_PALM = "open_palm"
    FIST = "fist"
    POINT = "point"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    PEACE = "peace"
    THREE = "three"
    PINCH = "pinch"
    OK_SIGN = "ok_sign"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


DEFAULT_ACTIONS = {
    Gesture.OPEN_PALM: ("stop", "STOP"),
    Gesture.FIST: ("grab", "GRAB"),
    Gesture.POINT: ("select", "SELECT"),
    Gesture.THUMBS_UP: ("confirm", "CONFIRM"),
    Gesture.THUMBS_DOWN: ("reject", "REJECT"),
    Gesture.PEACE: ("release", "RELEASE"),
    Gesture.THREE: ("approach", "COME HERE"),
    Gesture.PINCH: ("precise_grip", "PINCH"),
    Gesture.OK_SIGN: ("ok", "OK"),
    Gesture.UNKNOWN: ("none", "???"),
    Gesture.NONE: ("none", "NONE"),
}


@dataclass
class GestureResult:
    gesture: Gesture
    name: str
    confidence: float
    action: str
    label: str
    finger_curls: list
    extended: list
    hand_center: Tuple[float, float]
    source: str  # "learned" or "geometric"


# ── 3D Math Helpers ───────────────────────────────────────────

def _xyz(lm):
    return (lm.x, lm.y, lm.z)

def _dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def _angle(a, b, c):
    """Angle at b in degrees."""
    ba = np.array([a[i]-b[i] for i in range(3)])
    bc = np.array([c[i]-b[i] for i in range(3)])
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return math.degrees(math.acos(np.clip(cos_a, -1, 1)))


# ── Feature Extraction ────────────────────────────────────────

TIPS = [4, 8, 12, 16, 20]
FINGERS = [(1,2,3,4), (5,6,7,8), (9,10,11,12), (13,14,15,16), (17,18,19,20)]


def extract_features(landmarks) -> np.ndarray:
    """Extract 35-dimensional feature vector from 21 hand landmarks.
    
    29 original features + 6 palm orientation features:
    - 3 palm normal vector components (which way the palm faces)
    - 3 palm plane vectors (wrist-to-middle-MCP and wrist-to-pinky-MCP)
    
    The orientation features make the vector change meaningfully
    when the hand rotates, even if finger poses stay the same.
    """
    lm = [_xyz(l) for l in landmarks]
    wrist = lm[0]
    palm = tuple((lm[0][i] + lm[9][i]) / 2 for i in range(3))
    hand_size = _dist(wrist, lm[9]) + 1e-8

    features = []

    # 1. Finger curl angles (5)
    for f in FINGERS:
        features.append(_angle(lm[f[1]], lm[f[2]], lm[f[3]]) / 180.0)

    # 2. Fingertip-to-palm distances (5)
    for t in TIPS:
        features.append(_dist(lm[t], palm) / hand_size)

    # 3. Inter-fingertip distances (10)
    for i in range(5):
        for j in range(i+1, 5):
            features.append(_dist(lm[TIPS[i]], lm[TIPS[j]]) / hand_size)

    # 4. Fingertip z-depths relative to wrist (5)
    for t in TIPS:
        features.append((wrist[2] - lm[t][2]) * 10)

    # 5. Thumb-to-finger proximity (4)
    for t in [8, 12, 16, 20]:
        features.append(_dist(lm[4], lm[t]) / hand_size)

    # 6. Palm orientation (6 new features)
    # Vector from wrist to middle finger MCP (palm "up" direction)
    v_up = np.array([lm[9][i] - wrist[i] for i in range(3)])
    v_up_norm = v_up / (np.linalg.norm(v_up) + 1e-8)

    # Vector from wrist to pinky MCP (palm "side" direction)
    v_side = np.array([lm[17][i] - wrist[i] for i in range(3)])
    v_side_norm = v_side / (np.linalg.norm(v_side) + 1e-8)

    # Palm normal = cross product (tells which way palm faces)
    palm_normal = np.cross(v_up_norm, v_side_norm)
    palm_normal = palm_normal / (np.linalg.norm(palm_normal) + 1e-8)

    # Add all 6 orientation components
    features.extend(palm_normal.tolist())    # 3 values: palm facing direction
    features.extend(v_up_norm.tolist())      # 3 values: hand pointing direction

    return np.array(features, dtype=np.float32)


def get_curl_angles(landmarks) -> list:
    """Get 5 finger curl angles in degrees."""
    lm = [_xyz(l) for l in landmarks]
    return [_angle(lm[f[1]], lm[f[2]], lm[f[3]]) for f in FINGERS]


def get_extended(landmarks) -> list:
    """Determine which fingers are extended using 3D."""
    lm = [_xyz(l) for l in landmarks]
    wrist = lm[0]
    curls = get_curl_angles(landmarks)
    ext = []
    for i, (f, c) in enumerate(zip(FINGERS, curls)):
        tip_d = _dist(lm[f[3]], wrist)
        pip_d = _dist(lm[f[2]], wrist)
        ext.append(c > 140 or tip_d > pip_d * 1.1)
    return ext


# ── Gesture Memory (Learning System) ─────────────────────────

class GestureMemory:
    """Stores user-taught gestures as clusters of feature vectors.
    
    Each gesture has multiple samples (from different angles/poses).
    Classification computes average similarity to each gesture's samples.
    """

    def __init__(self, filename="gestures.json"):
        self._filename = filename
        self.gestures: Dict[str, Dict] = {}
        # {name: {samples: [np.array], action: str}}
        self._load()

    def teach(self, name: str, samples: List[np.ndarray], action: str = "custom"):
        """Store multiple samples for a gesture."""
        if name not in self.gestures:
            self.gestures[name] = {"samples": [], "action": action}
        for s in samples:
            self.gestures[name]["samples"].append(s.tolist())
        self.gestures[name]["action"] = action
        self._save()
        n = len(self.gestures[name]["samples"])
        logger.info(f"Gesture '{name}': {n} samples stored")

    def classify(self, features: np.ndarray, threshold: float = 0.90) -> Optional[Tuple[str, str, float]]:
        """Match features against all stored gestures.
        
        Uses average cosine similarity across all samples per gesture.
        Returns (name, action, confidence) or None.
        """
        if not self.gestures:
            return None

        f_norm = features / (np.linalg.norm(features) + 1e-8)
        best_name, best_action, best_score = None, None, 0

        for name, data in self.gestures.items():
            samples = [np.array(s, dtype=np.float32) for s in data["samples"]]
            if not samples:
                continue

            # Compute similarity to each sample, take top-3 average
            sims = []
            for s in samples:
                s_norm = s / (np.linalg.norm(s) + 1e-8)
                sims.append(float(np.dot(f_norm, s_norm)))

            sims.sort(reverse=True)
            # Average of top-5 most similar samples (covers closest orientations)
            top_k = sims[:min(5, len(sims))]
            avg_sim = sum(top_k) / len(top_k)

            if avg_sim > best_score:
                best_score = avg_sim
                best_name = name
                best_action = data["action"]

        if best_score >= threshold:
            return best_name, best_action, best_score
        return None

    def get_names(self) -> list:
        return list(self.gestures.keys())

    def get_sample_counts(self) -> dict:
        return {n: len(d["samples"]) for n, d in self.gestures.items()}

    def remove(self, name: str) -> bool:
        if name in self.gestures:
            del self.gestures[name]
            self._save()
            return True
        return False

    def clear(self):
        self.gestures = {}
        self._save()

    def _save(self):
        GESTURE_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        with open(GESTURE_MEMORY_DIR / self._filename, 'w') as f:
            json.dump(self.gestures, f)

    def _load(self):
        path = GESTURE_MEMORY_DIR / self._filename
        if path.exists():
            try:
                with open(path) as f:
                    self.gestures = json.load(f)
                total = sum(len(d["samples"]) for d in self.gestures.values())
                if self.gestures:
                    logger.info(f"Loaded {len(self.gestures)} gestures ({total} samples) from {self._filename}")
            except Exception:
                self.gestures = {}


# ── Geometric Fallback Classifier ─────────────────────────────

def classify_geometric(landmarks) -> Tuple[Gesture, str, float]:
    """Rule-based classification. Used only when no learned gesture matches."""
    lm = [_xyz(l) for l in landmarks]
    ext = get_extended(landmarks)
    curls = get_curl_angles(landmarks)
    thumb, index, middle, ring, pinky = ext
    n_ext = sum(ext)
    hand_size = _dist(lm[0], lm[9]) + 1e-8

    # Thumb-index distance
    ti_dist = _dist(lm[4], lm[8]) / hand_size
    avg_curl = sum(curls[1:]) / 4

    # Pinch
    if ti_dist < 0.4 and not middle and not ring and not pinky:
        return Gesture.PINCH, "PINCH", 0.80

    # OK sign
    if ti_dist < 0.4 and middle and ring:
        return Gesture.OK_SIGN, "OK", 0.75

    # Open palm
    if n_ext >= 5:
        return Gesture.OPEN_PALM, "STOP", 0.85

    # Fist
    if n_ext == 0 and avg_curl < 100:
        return Gesture.FIST, "GRAB", 0.80

    # Thumbs up/down
    if thumb and not index and not middle and not ring and not pinky:
        y_diff = lm[2][1] - lm[4][1]
        if y_diff > 0.04:
            return Gesture.THUMBS_UP, "CONFIRM", 0.75
        else:
            return Gesture.THUMBS_DOWN, "REJECT", 0.70

    # Point
    if index and not middle and not ring and not pinky:
        return Gesture.POINT, "SELECT", 0.80

    # Peace
    if index and middle and not ring and not pinky:
        return Gesture.PEACE, "RELEASE", 0.80

    # Three
    if index and middle and ring and not pinky:
        return Gesture.THREE, "COME HERE", 0.75

    return Gesture.UNKNOWN, "???", 0.2


# ── Main Recognizer ───────────────────────────────────────────

class GestureRecognizer:
    """Learning-first gesture recognizer.
    
    Pipeline:
    1. Extract 29-dim features from landmarks
    2. Check learned gesture memory (multi-sample cosine similarity)
    3. If no match, fall back to geometric rules
    4. Stability filter (N consistent frames)
    """

    def __init__(self, stability_frames=3, hand_id="default"):
        self.hand_id = hand_id
        self.memory = GestureMemory(filename=f"gestures_{hand_id}.json")
        self.stability_frames = stability_frames
        self._history = []
        self._stable = (Gesture.NONE, "NONE")

    def classify(self, landmarks) -> GestureResult:
        if not landmarks or len(landmarks) < 21:
            self._history = []
            return self._result(Gesture.NONE, "NONE", 0, [0]*5, [False]*5, (0.5,0.5), "none")

        features = extract_features(landmarks)
        curls = get_curl_angles(landmarks)
        ext = get_extended(landmarks)
        center = ((landmarks[0].x + landmarks[9].x)/2,
                  (landmarks[0].y + landmarks[9].y)/2)

        # Step 1: Check learned gestures
        match = self.memory.classify(features, threshold=0.82)
        if match:
            name, action, conf = match
            gesture = Gesture.CUSTOM
            source = "learned"
        else:
            # Step 2: Geometric fallback
            gesture, name, conf = classify_geometric(landmarks)
            action = DEFAULT_ACTIONS.get(gesture, ("none", "???"))[0]
            source = "geometric"

        # Stability filter
        self._history.append((gesture, name))
        if len(self._history) > self.stability_frames:
            self._history = self._history[-self.stability_frames:]
        if len(self._history) == self.stability_frames:
            if all(h[1] == self._history[0][1] for h in self._history):
                self._stable = self._history[0]

        sg, sn = self._stable
        if sg == Gesture.CUSTOM:
            sa = "custom"
            for g in self.memory.gestures.values():
                if True:  # find action
                    pass
            m = self.memory.gestures.get(sn)
            sa = m["action"] if m else "custom"
            sl = sn
        else:
            sa = DEFAULT_ACTIONS.get(sg, ("none", "???"))[0]
            sl = DEFAULT_ACTIONS.get(sg, ("none", "???"))[1]
            if sn not in ("???", "NONE"):
                sl = sn

        return GestureResult(
            gesture=sg, name=sn, confidence=conf,
            action=sa, label=sl,
            finger_curls=curls, extended=ext,
            hand_center=center, source=source,
        )

    def start_recording(self):
        """Start recording feature samples for teaching."""
        self._recording = True
        self._recorded_samples = []
        self._recorded_normals = []  # Track palm normals for rotation coverage

    def record_frame(self, landmarks):
        """Record one frame of feature data during teaching."""
        if hasattr(self, '_recording') and self._recording and landmarks:
            features = extract_features(landmarks)
            self._recorded_samples.append(features)
            # Track palm normal for rotation coverage feedback
            lm = [_xyz(l) for l in landmarks]
            v_up = np.array([lm[9][i] - lm[0][i] for i in range(3)])
            v_side = np.array([lm[17][i] - lm[0][i] for i in range(3)])
            normal = np.cross(v_up, v_side)
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-8:
                self._recorded_normals.append((normal / norm_len).tolist())

    def get_rotation_coverage(self) -> float:
        """Estimate how much rotation space has been covered (0-1).
        
        Computed from the angular spread of recorded palm normals.
        0 = all samples from same angle, 1 = full hemisphere covered.
        """
        normals = getattr(self, '_recorded_normals', [])
        if len(normals) < 2:
            return 0.0
        # Compute max angular spread between any pair of normals
        arr = np.array(normals)
        # Use dot products between all pairs
        dots = arr @ arr.T
        dots = np.clip(dots, -1, 1)
        # Max angle seen between any two normals
        min_dot = dots.min()
        max_angle = math.degrees(math.acos(min_dot))
        # Normalize: 180 degrees = full coverage (front to back)
        return min(1.0, max_angle / 120.0)  # 120 degrees = "good" coverage

    def stop_recording(self) -> int:
        """Stop recording and return number of samples captured."""
        self._recording = False
        n = len(getattr(self, '_recorded_samples', []))
        return n

    def save_gesture(self, name: str, action: str = "custom") -> int:
        """Save recorded samples as a named gesture."""
        samples = getattr(self, '_recorded_samples', [])
        if not samples:
            return 0
        self.memory.teach(name, samples, action)
        self._recorded_samples = []
        return len(samples)

    def reset(self):
        self._history = []
        self._stable = (Gesture.NONE, "NONE")