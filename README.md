# BROTEUS

**Biometric Recognition & Object-Tracking Engagement with Universal Sensing**

ORION Subsystem B — the perception layer.

---

BROTEUS is a real-time vision system that detects objects, tracks hands, recognizes gestures, and understands hand animations — all running at once in a single pipeline. It's the eyes of [ORION](https://github.com/shtet100), a modular robotics ecosystem we're building to eventually control a physical robot arm.

The whole thing runs on CPU at ~21 FPS. No GPU needed.

---

## What It Actually Does

You point a camera at a desk. You type "pen" into the search bar. BROTEUS finds the pen. You type "sticky note" — now it's tracking both. Hold up your hand and it draws a skeleton, tells you which gesture you're making, and maps it to a robot command. Wave your hand and it recognizes the motion pattern. Click on a detected object and it shows you where a gripper should grab it.

That's the short version. Here's the longer one.

---

## Object Detection

<div align="center">

https://github.com/user-attachments/assets/PLACEHOLDER_OBJECT_DETECTION

</div>

We use **YOLO-World** — an open-vocabulary detector. Traditional YOLO only knows 80 object classes. YOLO-World takes text queries, so you can tell it to find literally anything.

BROTEUS starts blind. Zero classes. You add what you want through the UI, and it starts looking. Remove it with one click. The list saves to disk, so it remembers across restarts.

Why this matters: most detection systems have a hardcoded vocabulary baked into training. Ours doesn't. The user decides what exists in the scene.

**Numbers:** 79–87% confidence on common desk objects, 21 FPS on CPU, detection every 5th frame with an IoU tracker filling the gaps. The tracker keeps persistent IDs so objects don't flicker when the detector skips a frame.

---

## Gesture Recognition

<div align="center">

https://github.com/user-attachments/assets/PLACEHOLDER_GESTURE_RECOGNITION

</div>

We track up to two hands at once using MediaPipe's HandLandmarker (21 3D keypoints per hand). Left and right are identified and tracked independently — separate classifiers, separate memory, separate everything.

### The Feature Vector

Each hand pose gets compressed into a **35-dimensional vector**:

- **5** finger curl angles (3D joint angles, not just "up or down")
- **5** fingertip-to-palm distances
- **10** inter-fingertip distances (every pair)
- **5** z-depth ratios (catches gestures pointing toward/away from camera)
- **4** thumb-to-finger proximities (pinch detection)
- **3** palm normal (which way the palm faces — this is what makes rotation work)
- **3** palm direction (which way the fingers point)

The palm orientation features are what separate this from the typical "is the finger up or down" approach. When you rotate your hand, the curl angles barely change, but the palm normal flips completely. That's encoded.

### Learning-First

There's a built-in geometric classifier for the basics (open palm → stop, fist → grab, point → select). But the real system is learning-first: you hold a pose, hit record, rotate your hand slowly while holding it, and BROTEUS captures ~30+ samples across different orientations. The UI shows you a live rotation coverage percentage so you know when you've given it enough angles.

On classification, learned gestures always win over geometric rules. Cosine similarity against the top-5 closest stored samples, threshold 0.82.

---

## Animation Recognition

<div align="center">

https://github.com/user-attachments/assets/PLACEHOLDER_ANIMATION_DETECTION

</div>

Static gestures are poses. Animations are movements — a wave, a beckoning curl, a circular motion. Different problem entirely.

We solve it with **Dynamic Time Warping (DTW)**. Each frame, we extract a 12-dim temporal feature vector (finger curls + palm normal + position + velocity) and push it into a sliding window of the last ~3 seconds. Every few frames, DTW compares that window against all stored animation recordings.

DTW handles speed variation: a fast wave and a slow wave have different frame counts but the same shape. DTW warps one sequence onto the other and measures the alignment cost. We use a Sakoe-Chiba band to keep the warping reasonable.

**To teach an animation:** hit Record, perform the motion for 2–3 seconds, stop, name it. Do it 2–3 times at different speeds for better matching. That's it.

---

## Grasp Intelligence

Click a detected object and BROTEUS shows you where to grab it. Every surface point gets scored on four criteria:

1. **Normal alignment** — can a gripper approach from this angle?
2. **Depth consistency** — is this a flat, stable surface?
3. **Edge proximity** — how far from the boundary? (distance transform)
4. **Centroid balance** — is the grasp centered for stability?

Rendered as a green-to-red heatmap overlay. Green = good grip. Red = bad grip.

Depth comes from **MiDaS** monocular estimation — one RGB camera, no depth sensor needed. We compute Sobel-based surface normals from the depth map for the normal alignment criterion.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   BROTEUS Server                     │
│                 FastAPI · Port 8100                   │
│                                                      │
│  ┌────────────┐  ┌──────────────┐  ┌────────────┐   │
│  │ YOLO-World │  │  MediaPipe   │  │   MiDaS    │   │
│  │ Detection  │  │  Dual Hands  │  │   Depth    │   │
│  └─────┬──────┘  └──────┬───────┘  └─────┬──────┘   │
│        │         ┌──────┴───────┐         │          │
│        │         │ Gesture 35D  │         │          │
│        │         │ Anim 12D+DTW │         │          │
│        │         └──────┬───────┘         │          │
│  ┌─────▼────────────────▼─────────────────▼──────┐   │
│  │            IoU Object Tracker                 │   │
│  │    Persistent IDs · Class-vote stability      │   │
│  └─────────────────────┬─────────────────────────┘   │
│  ┌─────────────────────▼─────────────────────────┐   │
│  │          Grasp Affordance Scorer              │   │
│  └─────────────────────┬─────────────────────────┘   │
│  ┌─────────────────────▼─────────────────────────┐   │
│  │        WebSocket Frame Streaming              │   │
│  └───────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
                         │
                   WebSocket / JSON
                         │
┌────────────────────────▼─────────────────────────────┐
│                  Live Dashboard                       │
│               Browser · localhost:8100                │
└──────────────────────────────────────────────────────┘
```

---

## Part of ORION

BROTEUS is one piece of a larger system:

```
ORION (Brain · Port 8000)
 ├── ATHENA    — Navigation · Procedural terrain · A* pathfinding
 ├── BROTEUS   — Perception · Grasp intelligence · Gestures & animations
 ├── CHIRON    — Motor cortex · ROS 2 bridge · Hardware abstraction
 ├── DAEDALUS  — Self-calibrating physics discovery (SINDy)
 └── RL Pipeline — PPO/SAC in sim · ONNX deployment at 50–200 Hz
```

**BROTEUS sees → ORION decides → CHIRON moves → DAEDALUS calibrates.**

The target hardware is an [SO-ARM 101](https://github.com/TheRobotStudio/SO-ARM100) — a 6-DOF arm where BROTEUS is the eye, CHIRON drives the joints, and DAEDALUS closes the sim-to-real gap.

---

## Setup

```bash
# Clone
git clone https://github.com/shtet100/BROTEUS.git
cd BROTEUS

# Environment
conda create -n BROTEUS python=3.11 -y
conda activate BROTEUS
pip install -r requirements.txt

# Download the hand landmark model
python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task', 'hand_landmarker.task')"

# Run
python -m broteus.api.server
```

Open `http://localhost:8100/live` in your browser.

YOLO-World and MiDaS models download automatically on first use.

---

## Tech Stack

| | |
|:---|:---|
| **Detection** | YOLO-World (`yolov8s-worldv2`) |
| **Hands** | MediaPipe HandLandmarker (Tasks API v0.10.33) |
| **Depth** | MiDaS (`MiDaS_small`) |
| **Server** | FastAPI + WebSocket, Python 3.11 |
| **Tracking** | Custom IoU tracker w/ class-vote stabilization |
| **Gestures** | 35-dim features, cosine similarity, geometric fallback |
| **Animations** | 12-dim temporal features, DTW w/ Sakoe-Chiba band |
| **Frontend** | Single-file vanilla JS/CSS dashboard |

---

## Project Structure

```
broteus/
├── api/
│   └── server.py           # FastAPI server — the main entry point
├── adapters/
│   ├── base.py              # Abstract camera adapter
│   ├── webcam.py            # OpenCV webcam
│   ├── video_file.py        # Video playback
│   └── synthetic.py         # Test pattern generator
├── detection/
│   ├── gesture.py           # 35-dim gesture recognition (learning-first)
│   ├── animation.py         # DTW-based animation recognition
│   ├── overlay.py           # Browse + focus mode rendering
│   ├── depth.py             # MiDaS depth estimator
│   └── tracker.py           # IoU multi-object tracker
├── grasp/
│   └── engine.py            # Grasp affordance scoring
├── core/
│   ├── config.py            # Configuration
│   └── frame.py             # Frame data structures
├── visualization/
│   └── live.html            # Browser dashboard
└── pipeline.py              # Pipeline engine
```

---

## Design Decisions

**No hardcoded anything.** The search list starts empty. Gestures are taught, not predefined. Animations are recorded, not scripted. BROTEUS knows nothing until the user teaches it.

**Learning beats rules.** The geometric gesture classifier is a fallback. If you've taught BROTEUS what "thumbs up" looks like, it uses your samples, not the built-in rule.

**The detection model is swappable.** YOLO-World today, NVIDIA Isaac ROS tomorrow. The rest of the pipeline doesn't care what's behind the `get_detector()` call.

**Per-hand independence.** Left and right hands are fully separate systems. Different classifiers, different memory files, different state. They don't interfere.

**Disk persistence.** Search lists, gesture samples, animation recordings — all JSON on disk. Kill the server, restart, everything's still there.

---

*Built by Swan & David Young.*
