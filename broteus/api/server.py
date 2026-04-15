"""
BROTEUS API Server v2.1
========================

YOLO-World detection + MediaPipe gesture recognition + MiDaS depth.
User-driven search list. Learning-first gesture system.
No hardcoded vocabularies.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from broteus.adapters.base import CameraAdapter
from broteus.adapters.synthetic import SyntheticAdapter
from broteus.core.config import BroteusConfig
from broteus.pipeline import BroteusEngine, PipelineResult
from broteus.grasp.engine import GripperType
from broteus.detection.detector import SimulatedDetector
from broteus.detection.tracker import ObjectTracker
from broteus.detection.gesture import GestureRecognizer
from broteus.detection.animation import AnimationRecognizer

logger = logging.getLogger("broteus.api")


# == YOLO-World ==
_yoloworld = None
def get_detector():
    global _yoloworld
    if _yoloworld is None:
        try:
            from ultralytics import YOLO
            _yoloworld = YOLO("yolov8s-worldv2.pt")
            classes = _search_list.get_classes()
            if classes:
                _yoloworld.set_classes(classes)
                logger.info(f"YOLO-World loaded with {len(classes)} classes")
            else:
                logger.info("YOLO-World loaded (no active classes)")
        except Exception as e:
            logger.warning(f"YOLO-World not available: {e}")
    return _yoloworld

# == MiDaS Depth ==
_depth_estimator = None
def get_depth():
    global _depth_estimator
    if _depth_estimator is None:
        try:
            from broteus.detection.depth import DepthEstimator
            _depth_estimator = DepthEstimator("MiDaS_small")
            _depth_estimator.load()
            logger.info("MiDaS depth loaded")
        except Exception as e:
            logger.warning(f"MiDaS not available: {e}")
    return _depth_estimator

# == MediaPipe Hands ==
_hand_landmarker = None
def get_hand_detector():
    global _hand_landmarker
    if _hand_landmarker is None:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions as MPBase
            from mediapipe.tasks.python.vision import (
                HandLandmarker, HandLandmarkerOptions,
                RunningMode as MPMode,
            )
            model_path = Path("hand_landmarker.task")
            if not model_path.exists():
                logger.warning("hand_landmarker.task not found")
                return None
            opts = HandLandmarkerOptions(
                base_options=MPBase(model_asset_path=str(model_path)),
                num_hands=2, running_mode=MPMode.IMAGE,
            )
            _hand_landmarker = HandLandmarker.create_from_options(opts)
            logger.info("MediaPipe HandLandmarker loaded (2 hands)")
        except Exception as e:
            logger.warning(f"HandLandmarker not available: {e}")
    return _hand_landmarker

# == Gesture Recognizers (one per hand for independent tracking) ==
_gesture_left = GestureRecognizer(stability_frames=3, hand_id="left")
_gesture_right = GestureRecognizer(stability_frames=3, hand_id="right")

# == Animation Recognizers (temporal gesture sequences per hand) ==
_anim_left = AnimationRecognizer(hand_id="left")
_anim_right = AnimationRecognizer(hand_id="right")

# == Search List ==
SEARCH_LIST_FILE = Path("broteus_memory") / "search_list.json"

class SearchList:
    def __init__(self):
        self.classes = []
        self._load()
    def add(self, name):
        clean = name.strip()
        if not clean or clean.lower() in {c.lower() for c in self.classes}: return False
        self.classes.append(clean); self._save(); return True
    def remove(self, name):
        before = len(self.classes)
        self.classes = [c for c in self.classes if c.lower() != name.lower().strip()]
        if len(self.classes) < before: self._save(); return True
        return False
    def get_classes(self): return list(self.classes)
    def clear(self): self.classes = []; self._save()
    def _save(self):
        SEARCH_LIST_FILE.parent.mkdir(exist_ok=True)
        with open(SEARCH_LIST_FILE, 'w') as f: json.dump(self.classes, f)
    def _load(self):
        if SEARCH_LIST_FILE.exists():
            try:
                with open(SEARCH_LIST_FILE) as f: self.classes = json.load(f)
                if self.classes: logger.info(f"Search list: {self.classes}")
            except Exception: self.classes = []

_search_list = SearchList()

_latest_detections = []
_latest_gesture_data = {"left": None, "right": None}
_latest_frame_b64 = None

# == Global State ==
config = BroteusConfig()
active_adapter: Optional[CameraAdapter] = None
engine: Optional[BroteusEngine] = None
connected_clients: set = set()

# == Lifecycle ==
@asynccontextmanager
async def lifespan(app: FastAPI):
    global active_adapter, engine
    logger.info("BROTEUS starting...")
    camera_found = False
    from broteus.adapters.webcam import WebcamAdapter
    for idx in range(5):
        for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
            try:
                active_adapter = WebcamAdapter(device_index=idx, source_name=f"Camera:{idx}",
                                               config=config.stream, backend=backend)
                active_adapter.open()
                logger.info(f"Camera connected (index {idx})")
                camera_found = True; break
            except Exception: continue
        if camera_found: break
    if not camera_found:
        active_adapter = SyntheticAdapter(width=640, height=480, generate_depth=True,
                                          source_name="Synthetic", config=config.stream)
        active_adapter.open()
    engine = BroteusEngine(config=config, detector=SimulatedDetector(objects=["wrench"]),
                           gripper_type=GripperType.PARALLEL_JAW)
    logger.info("BROTEUS v2.1 ready")
    yield
    if active_adapter: active_adapter.close()

# == App ==
app = FastAPI(title="BROTEUS", version="2.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# == Routes ==
@app.get("/")
async def root():
    return {"subsystem": "BROTEUS", "version": "2.1.0", "status": "online",
            "search_list": _search_list.get_classes(),
            "gestures": {
                "left": _gesture_left.memory.get_names(),
                "right": _gesture_right.memory.get_names(),
            },
            "animations": {
                "left": _anim_left.memory.get_names(),
                "right": _anim_right.memory.get_names(),
            }}

@app.get("/live")
async def live_page():
    p = Path(__file__).parent.parent / "visualization" / "live.html"
    if p.exists(): return HTMLResponse(p.read_text(encoding='utf-8'))
    return HTMLResponse("<h1>live.html not found</h1>")

@app.get("/cameras/scan")
async def scan_cameras():
    cameras = []
    for idx in range(10):
        for backend, bname in [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_ANY, "ANY")]:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    for _ in range(3): cap.read()
                    ret, f = cap.read()
                    if ret and f is not None:
                        h, w = f.shape[:2]
                        cameras.append({"index": idx, "backend": bname,
                                        "resolution": f"{w}x{h}",
                                        "label": f"Camera {idx} ({bname}, {w}x{h})"})
                    cap.release(); break
                cap.release()
            except Exception: pass
    return {"cameras": cameras}

@app.post("/adapter/switch/webcam")
async def switch_webcam(source: int = 0):
    global active_adapter
    if active_adapter: active_adapter.close()
    from broteus.adapters.webcam import WebcamAdapter
    for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
        try:
            active_adapter = WebcamAdapter(device_index=source, source_name=f"Camera:{source}",
                                           config=config.stream, backend=backend)
            active_adapter.open()
            return {"status": "switched", "adapter": active_adapter.source_name}
        except Exception: active_adapter = None
    return {"error": f"Cannot open camera {source}"}

@app.get("/search")
async def get_search(): return {"classes": _search_list.get_classes()}

@app.post("/search/add")
async def add_search(name: str):
    added = _search_list.add(name)
    if added and _yoloworld: _yoloworld.set_classes(_search_list.get_classes())
    return {"added": added, "classes": _search_list.get_classes()}

@app.post("/search/remove")
async def remove_search(name: str):
    removed = _search_list.remove(name)
    if removed and _yoloworld and _search_list.classes:
        _yoloworld.set_classes(_search_list.get_classes())
    return {"removed": removed, "classes": _search_list.get_classes()}

@app.post("/search/clear")
async def clear_search(): _search_list.clear(); return {"cleared": True}

@app.get("/gestures")
async def get_gestures():
    return {"right": _gesture_right.memory.get_sample_counts(),
            "left": _gesture_left.memory.get_sample_counts(),
            "right_names": _gesture_right.memory.get_names(),
            "left_names": _gesture_left.memory.get_names()}

@app.post("/gestures/remove")
async def remove_gesture(name: str, hand: str = "right"):
    rec = _gesture_right if hand == "right" else _gesture_left
    return {"removed": rec.memory.remove(name)}

@app.post("/gestures/clear")
async def clear_gestures():
    _gesture_right.memory.clear(); _gesture_left.memory.clear(); return {"cleared": True}


@app.get("/detections")
async def get_detections():
    return {
        "detections": _latest_detections,
        "gesture": _latest_gesture_data,
        "search_list": _search_list.get_classes(),
    }

@app.get("/snapshot")
async def get_snapshot():
    """Return the latest annotated camera frame as base64 JPEG."""
    if _latest_frame_b64:
        return {"success": True, "frame": _latest_frame_b64,
                "detections": _latest_detections,
                "search_list": _search_list.get_classes()}
    return {"success": False, "error": "No frame available. Is the dashboard open?"}

# -- Animations --

@app.get("/animations")
async def get_animations():
    return {"right": _anim_right.memory.get_counts(),
            "left": _anim_left.memory.get_counts(),
            "right_names": _anim_right.memory.get_names(),
            "left_names": _anim_left.memory.get_names()}

@app.post("/animations/remove")
async def remove_animation(name: str, hand: str = "right"):
    rec = _anim_right if hand == "right" else _anim_left
    return {"removed": rec.memory.remove(name)}

@app.post("/animations/clear")
async def clear_animations():
    _anim_right.memory.clear(); _anim_left.memory.clear(); return {"cleared": True}

# == WebSocket ==
@app.websocket("/ws/frames")
async def websocket_frames(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        if not active_adapter:
            await websocket.send_json({"error": "No camera"}); return

        detector = get_detector()
        depth_est = get_depth()
        hand_det = get_hand_detector()
        tracker = ObjectTracker(iou_threshold=0.3, max_missing=5)

        frame_count = 0
        last_dets, tracked = [], []
        last_depth, last_normals = None, None
        focused_idx = None
        gesture_recording = False
        anim_recording = False

        async def listen():
            nonlocal focused_idx, gesture_recording, anim_recording
            try:
                while True:
                    raw = await websocket.receive_text()
                    cmd = json.loads(raw)
                    t = cmd.get('type')
                    if t == 'focus':
                        idx = cmd.get('index', -1)
                        focused_idx = idx if idx >= 0 else None
                    elif t == 'gesture_record_start':
                        gesture_recording = True
                        rec_hand = cmd.get('hand', 'right')
                        rec_obj = _gesture_right if rec_hand == 'right' else _gesture_left
                        rec_obj.start_recording()
                    elif t == 'gesture_record_stop':
                        gesture_recording = False
                        rec_hand = cmd.get('hand', 'right')
                        rec_obj = _gesture_right if rec_hand == 'right' else _gesture_left
                        n = rec_obj.stop_recording()
                        name = cmd.get('name', '')
                        action = cmd.get('action', 'custom')
                        if name and n > 0:
                            rec_obj.save_gesture(name, action)
                            logger.info(f"Gesture '{name}' ({rec_hand}): {n} samples")
                    elif t == 'anim_record_start':
                        anim_recording = True
                        rec_hand = cmd.get('hand', 'right')
                        anim_obj = _anim_right if rec_hand == 'right' else _anim_left
                        anim_obj.start_recording()
                    elif t == 'anim_record_stop':
                        anim_recording = False
                        rec_hand = cmd.get('hand', 'right')
                        anim_obj = _anim_right if rec_hand == 'right' else _anim_left
                        n = anim_obj.stop_recording()
                        name = cmd.get('name', '')
                        action = cmd.get('action', 'custom')
                        if name and n > 0:
                            anim_obj.save_animation(name, action)
                            logger.info(f"Animation '{name}' ({rec_hand}): {n} frames")
            except Exception: pass

        asyncio.create_task(listen())

        async for frame in active_adapter.stream():
            bgr = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
            fh, fw = bgr.shape[:2]

            # == Hand gesture (both hands, every frame) ==
            gesture_data = {"left": None, "right": None}
            if hand_det:
                try:
                    import mediapipe as _mp
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    mp_img = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)
                    hr = hand_det.detect(mp_img)

                    conns = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                             (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
                             (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]

                    # Color per hand: cyan for left, green for right
                    hand_colors = {"Left": ((200,180,0), (255,220,0)),    # cyan-ish
                                   "Right": ((0,200,100), (0,255,0))}     # green

                    detected_hands = set()
                    for h_idx in range(len(hr.hand_landmarks)):
                        lms = hr.hand_landmarks[h_idx]

                        # MediaPipe handedness already matches the user's actual hand
                        mp_label = "Right"
                        if hr.handedness and h_idx < len(hr.handedness):
                            mp_label = hr.handedness[h_idx][0].category_name
                        hand_side = mp_label.lower()  # "left" or "right"
                        detected_hands.add(hand_side)

                        rec = _gesture_right if hand_side == "right" else _gesture_left

                        if gesture_recording:
                            rec.record_frame(lms)

                        gr = rec.classify(lms)

                        # Draw skeleton with hand-specific color
                        line_c, dot_c = hand_colors.get(mp_label, ((0,200,100),(0,255,0)))
                        for a, b in conns:
                            cv2.line(bgr, (int(lms[a].x*fw), int(lms[a].y*fh)),
                                     (int(lms[b].x*fw), int(lms[b].y*fh)), line_c, 2)
                        for lm in lms:
                            cv2.circle(bgr, (int(lm.x*fw), int(lm.y*fh)), 4, dot_c, -1)

                        # Hand label on frame near wrist
                        wx, wy = int(lms[0].x*fw), int(lms[0].y*fh)
                        cv2.putText(bgr, f"{hand_side.upper()}", (wx-15, wy+20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, dot_c, 2)

                        # HUD per hand
                        hud_y = 5 if hand_side == "right" else 60
                        cv2.rectangle(bgr, (fw-230, hud_y), (fw-5, hud_y+50), (30,30,30), -1)
                        clr = (0,255,200) if gr.source == "learned" else dot_c
                        cv2.putText(bgr, f"{hand_side[0].upper()}: {gr.label}", (fw-225, hud_y+20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)
                        cv2.putText(bgr, f"{gr.action} ({gr.confidence:.0%})", (fw-225, hud_y+40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180,180,180), 1)

                        gesture_data[hand_side] = {
                            "name": gr.name, "action": gr.action, "label": gr.label,
                            "confidence": round(gr.confidence, 2), "source": gr.source,
                            "curls": [round(c,1) for c in gr.finger_curls],
                            "extended": gr.extended, "recording": gesture_recording,
                            "hand_center": [round(gr.hand_center[0],3), round(gr.hand_center[1],3)],
                            "rotation_coverage": round(rec.get_rotation_coverage(), 2) if gesture_recording else 0,
                            "samples_count": len(getattr(rec, '_recorded_samples', [])) if gesture_recording else 0,
                        }

                        # == Animation recognition (temporal gestures) ==
                        anim_rec = _anim_right if hand_side == "right" else _anim_left
                        ar = anim_rec.process_frame(lms)
                        if ar.matched:
                            # Show animation detection on frame
                            ax = 5 if hand_side == "left" else fw - 250
                            cv2.rectangle(bgr, (ax, fh-70), (ax+245, fh-35), (180, 0, 255), -1)
                            cv2.putText(bgr, f"ANIM: {ar.name}", (ax+5, fh-50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
                            cv2.putText(bgr, f"{ar.action} ({ar.confidence:.0%})", (ax+5, fh-38),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220,220,220), 1)

                        gesture_data[hand_side]["animation"] = {
                            "name": ar.name, "action": ar.action,
                            "confidence": round(ar.confidence, 2),
                            "matched": ar.matched, "recording": ar.recording,
                            "frames": len(getattr(anim_rec, '_recorded_sequence', [])) if ar.recording else 0,
                        } if ar else None

                    if gesture_recording:
                        # Recording overlay: red bar + rotation coverage + sample count
                        cv2.rectangle(bgr, (0, 0), (fw, 8), (0, 0, 220), -1)
                        for side, rec_obj in [("right", _gesture_right), ("left", _gesture_left)]:
                            cov = rec_obj.get_rotation_coverage()
                            n_samp = len(getattr(rec_obj, '_recorded_samples', []))
                            if n_samp > 0:
                                bar_w = int(fw * 0.4 * cov)
                                bar_x = 10 if side == "left" else fw - 10 - int(fw * 0.4)
                                cv2.rectangle(bgr, (bar_x, fh-30), (bar_x + int(fw*0.4), fh-20), (50,50,50), -1)
                                bar_color = (0, 255, 0) if cov > 0.5 else (0, 200, 255)
                                cv2.rectangle(bgr, (bar_x, fh-30), (bar_x + bar_w, fh-20), bar_color, -1)
                                cv2.putText(bgr, f"Rotation: {cov:.0%}  Samples: {n_samp}",
                                            (bar_x, fh-34), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)

                    if anim_recording:
                        # Animation recording overlay: purple bar + frame count
                        cv2.rectangle(bgr, (0, 0), (fw, 8), (180, 0, 220), -1)
                        for side, anim_obj in [("right", _anim_right), ("left", _anim_left)]:
                            n_frames = len(getattr(anim_obj, '_recorded_sequence', []))
                            if n_frames > 0:
                                bar_x = 10 if side == "left" else fw - 200
                                cv2.putText(bgr, f"ANIM REC: {n_frames} frames",
                                            (bar_x, fh-34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220,150,255), 1)

                    # Reset recognizers for hands not seen
                    if "left" not in detected_hands:
                        _gesture_left.reset(); _anim_left.reset()
                    if "right" not in detected_hands:
                        _gesture_right.reset(); _anim_right.reset()

                    if not hr.hand_landmarks:
                        _gesture_left.reset(); _gesture_right.reset()
                        _anim_left.reset(); _anim_right.reset()
                except Exception: pass

            # == YOLO-World (every 5th) ==
            if detector and _search_list.classes and frame_count % 5 == 0:
                try:
                    if frame_count % 25 == 0: detector.set_classes(_search_list.get_classes())
                    results = detector(bgr, conf=0.08, verbose=False)
                    last_dets = []
                    r = results[0]
                    if r.boxes is not None:
                        for i in range(len(r.boxes)):
                            xyxy = r.boxes.xyxy[i].cpu().numpy()
                            conf = float(r.boxes.conf[i].cpu())
                            cls_id = int(r.boxes.cls[i].cpu())
                            mask = None
                            if r.masks and i < len(r.masks):
                                md = r.masks.data[i].cpu().numpy()
                                mask = cv2.resize(md.astype(np.float32), (fw,fh),
                                                  interpolation=cv2.INTER_LINEAR)
                                mask = (mask > 0.5).astype(np.uint8)
                            last_dets.append({'bbox': [float(xyxy[0]),float(xyxy[1]),
                                                       float(xyxy[2]),float(xyxy[3])],
                                              'confidence': conf, 'class_name': r.names[cls_id],
                                              'mask': mask, 'source': 'yolo-world'})
                except Exception as e: logger.warning(f"Detection: {e}")
            elif not _search_list.classes and frame_count % 5 == 0:
                last_dets = []

            # == Depth (every 20th) ==
            if depth_est and frame_count % 20 == 10:
                try:
                    last_depth = depth_est.estimate(bgr)
                    last_normals = depth_est.compute_normals(last_depth)
                except Exception: pass

            # == Tracker ==
            if frame_count % 5 == 0: tracked = tracker.update(last_dets)
            display = tracked if tracked else last_dets

            # == Overlay ==
            from broteus.detection.overlay import draw_overlay
            sf = focused_idx
            if sf is not None and sf >= len(display): sf = None
            annotated, focus_crop, stats = draw_overlay(bgr, display, focused_idx=sf,
                                                         depth=last_depth, normals=last_normals)

            # == Encode ==
            _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64 = base64.b64encode(buf.tobytes()).decode('ascii')
            b64_crop, fstats = None, None
            if focus_crop is not None:
                _, cb = cv2.imencode('.jpg', focus_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                b64_crop = base64.b64encode(cb.tobytes()).decode('ascii')
                fstats = {"grip_score": stats.get('focused','-'), "mode": "3D DEPTH" if last_depth else "2D"}

            det_json = [{'bbox':d['bbox'],'class_name':d['class_name'],
                         'confidence':d['confidence'],'track_id':d.get('track_id',0)}
                        for d in display]
            
            # Update shared state for REST API
            global _latest_detections, _latest_gesture_data, _latest_frame_b64
            _latest_detections = det_json
            _latest_gesture_data = gesture_data
            _latest_frame_b64 = b64

            msg = {"type":"frame","frame":b64,"metadata":frame.to_dict(),
                   "detections":det_json,"gesture":gesture_data,
                   "focus_crop":b64_crop,"focus_stats":fstats,
                   "search_list":_search_list.get_classes()}

            frame_count += 1
            try: await websocket.send_json(msg)
            except Exception: break

    except WebSocketDisconnect: pass
    finally: connected_clients.discard(websocket)

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("broteus.api.server:app", host="0.0.0.0", port=8100, reload=True)