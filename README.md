# BROTEUS

**Biometric Recognition & Object-Tracking Engagement with Universal Sensing**

ORION Subsystem B. Perception & Grasp Intelligence.

---

BROTEUS is a real-time vision system that detects objects, tracks hands, recognizes gestures, and understands hand animations, all running simultaneously in a single pipeline. It serves as the perception layer of [ORION](https://github.com/shtet100), a modular robotics ecosystem designed to control a physical robot arm.

The entire system runs on CPU at ~21 FPS. No GPU required.

---

## What It Does

A camera feed is processed through four parallel subsystems: YOLO-World finds objects that the operator has specified, MediaPipe tracks both hands with full 3D skeleton data, a learning-first classifier identifies static hand gestures, and a DTW-based recognizer detects temporal hand animations. Clicking on a detected object triggers a grasp affordance heatmap showing optimal contact surfaces.

---

## Object Detection

<div align="center">

https://github.com/user-attachments/assets/36bcba2e-5a44-4781-a02c-d8595c81ea91

*Real-time object detection with user-driven search list. Classes are added and removed on the fly.*

</div>

BROTEUS uses **YOLO-World**, an open-vocabulary detection model. Traditional YOLO is locked to 80 COCO classes. YOLO-World accepts arbitrary text queries at runtime.

The system starts with zero classes. The operator adds object names through the UI, and BROTEUS begins searching for them. Removing a class is a single click. The search list persists to disk across restarts.

This matters because most detection systems have a hardcoded vocabulary baked into training. BROTEUS has none. The operator decides what exists in the scene.

**Performance:** 79-87% confidence on common desk objects, 21 FPS on CPU, detection every 5th frame with an IoU tracker filling the gaps. The tracker maintains persistent IDs so objects don't flicker when the detector skips a frame.

The detection backbone is designed as a swappable module, structured for future drop-in of NVIDIA Isaac ROS (RT-DETR, FoundationPose) when GPU hardware becomes available.

---

## Gesture Recognition

<div align="center">

https://github.com/user-attachments/assets/6b350a3f-31fd-4277-9c1b-d692c2b9d260

*Dual-hand gesture recognition with independent left/right tracking. Each hand displays its gesture, action, confidence, and finger states in real-time.*

</div>

BROTEUS tracks up to two hands simultaneously using MediaPipe's HandLandmarker (21 3D keypoints per hand). Left and right hands are identified and tracked independently with separate classifiers, separate memory, and separate state.

### The Feature Vector

Each hand pose is compressed into a **35-dimensional vector**:

- **5** finger curl angles (3D joint angles, not binary up/down)
- **5** fingertip-to-palm distances
- **10** inter-fingertip distances (every pair)
- **5** z-depth ratios (captures gestures pointing toward or away from the camera)
- **4** thumb-to-finger proximities (pinch detection)
- **3** palm normal (encodes which direction the palm faces, enabling rotation invariance)
- **3** palm direction (encodes which direction the fingers point)

The palm orientation features are what separate this from typical "is the finger up or down" approaches. When a hand rotates, the curl angles barely change, but the palm normal flips completely. That signal is encoded.

### Teaching New Gestures

1. Hold a hand pose in front of the camera
2. Press **Record** to start capturing feature samples
3. Slowly rotate the hand while holding the pose. This captures the gesture across multiple orientations.
4. The UI displays a live **rotation coverage** percentage reflecting angular variety
5. Press **Stop**, name the gesture, assign an ORION action
6. The gesture is recognized from that point forward, including after restart

Classification checks learned gestures first (cosine similarity against multi-sample clusters), then falls back to geometric rules for common poses like open palm, fist, or point. Left and right hands maintain completely separate memory files.

---

## Animation Recognition

<div align="center">

https://github.com/user-attachments/assets/295dcfd5-870a-42b1-9d6e-d345dccdb10f

*Recognizing a learned hand animation in real-time. The purple banner indicates a temporal gesture match.*

</div>

Static gestures capture frozen poses. Animations capture movements: a beckoning curl, a wave, a circular "spin" motion. This is a fundamentally different recognition problem.

BROTEUS solves it with **Dynamic Time Warping (DTW)**. Each frame, a 12-dimensional temporal feature vector is extracted (finger curls + palm normal + position + velocity) and pushed into a sliding window covering the last ~3 seconds. Every few frames, DTW compares that window against all stored animation recordings.

DTW handles speed variation naturally. A fast wave and a slow wave produce different frame counts but share the same underlying motion shape. DTW warps one sequence onto the other and measures alignment cost. A Sakoe-Chiba band constraint keeps the warping physically reasonable.

**Teaching an animation:** Press Record, perform the motion for 2-3 seconds, press Stop, name it. Recording the same motion 2-3 times at different speeds improves matching robustness.

---

## Grasp Intelligence

Clicking a detected object activates **focus mode**, which computes a grasp affordance heatmap. Every surface point is scored on four criteria:

1. **Normal alignment**: can a gripper approach from this angle?
2. **Depth consistency**: is this a flat, stable region?
3. **Edge proximity**: distance from the object boundary (via distance transform)
4. **Centroid balance**: is the grasp centered for stability?

Scores render as a continuous green-to-red heatmap overlay. Green indicates an ideal contact surface. Red indicates a poor one.

Depth data comes from **MiDaS** monocular estimation. One RGB camera, no depth sensor needed. Sobel-based surface normals are computed from the depth map for the normal alignment criterion.

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
 ├── ATHENA    - Navigation · Procedural terrain · A* pathfinding
 ├── BROTEUS   - Perception · Grasp intelligence · Gestures & animations
 ├── CHIRON    - Motor cortex · ROS 2 bridge · Hardware abstraction
 ├── DAEDALUS  - Self-calibrating physics discovery (SINDy)
 └── RL Pipeline - PPO/SAC in sim · ONNX deployment at 50-200 Hz
```

**BROTEUS sees. ORION decides. CHIRON moves. DAEDALUS calibrates.**

The target hardware is an [SO-ARM 101](https://github.com/TheRobotStudio/SO-ARM100), a 6-DOF arm where BROTEUS provides perception, CHIRON drives the joints, and DAEDALUS closes the sim-to-real gap.

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

Open `http://localhost:8100/live` in a browser.

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
│   └── server.py           # FastAPI server, the main entry point
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

**No hardcoded anything.** The search list starts empty. Gestures are taught, not predefined. Animations are recorded, not scripted. BROTEUS knows nothing until the operator teaches it.

**Learning beats rules.** Learned gestures and animations always take priority over built-in geometric classifiers. The rule-based system is a bootstrap, not the intelligence.

**Swappable detection backbone.** The detection model can be replaced without touching the rest of the pipeline. Today it runs YOLO-World on CPU. The architecture supports dropping in RT-DETR on an NVIDIA Jetson with no pipeline changes.

**Per-hand independence.** Left and right hands run fully separate systems. Different classifiers, different memory files, different state. No interference.

**Disk persistence.** Search lists, gesture samples, animation recordings are all stored as JSON. The server can be killed and restarted without losing any learned data.

---

*Built by Swan Yi Htet & David Young.*