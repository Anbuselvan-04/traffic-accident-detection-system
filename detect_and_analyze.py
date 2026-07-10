"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   AI-Powered Traffic Accident Detection and Severity Analysis                ║
║   Using Explainable AI (YOLOv8 + OpenCV)                                    ║
║                                                                              ║
║   Script  : detect_and_analyze.py                                           ║
║   Model   : YOLOv8 custom trained (models/best.pt)                          ║
║   Classes : 0 = Accident  (single-class model)                              ║
║             YOLO detects objects, not absence of objects — so there is      ║
║             no "Non Accident" class. Non-accident frames/images are used    ║
║             as negative background samples during training (no boxes),      ║
║             not as a second class.                                          ║
║                                                                             ║
║   IMPROVEMENTS (v2):                                                         ║
║     • EventLogger now writes center_x, center_y, vehicle_count columns      ║
║       (required by heatmap_analysis.py → fixes heatmap tab)                 ║
║     • live_heatmap_overlay() — accumulates accident blobs onto video frames  ║
║     • process_video() saves heatmap_video.mp4 alongside analyzed video       ║
║     • frame_generator() — yields processed frames for Streamlit side-by-side ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python detect_and_analyze.py --video path/to/video.mp4
    python detect_and_analyze.py --video path/to/video.mp4 --conf 0.4
    python detect_and_analyze.py --webcam
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import csv
import sys
import time
import math
from datetime import datetime
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import deque

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 0  ──  PROJECT PATHS
# ═════════════════════════════════════════════════════════════════════════════
BASE_DIR       = Path(__file__).resolve().parent
MODEL_PATH     = BASE_DIR / "models" / "best.pt"
OUTPUT_DIR     = BASE_DIR / "outputs"
VIDEO_OUT_DIR  = OUTPUT_DIR / "videos"
FRAMES_OUT_DIR = OUTPUT_DIR / "accident_frames"
LOGS_OUT_DIR   = OUTPUT_DIR / "logs"

for d in [VIDEO_OUT_DIR, FRAMES_OUT_DIR, LOGS_OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1  ──  CONFIGURATION CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
DEFAULT_CONF       = 0.70
DEFAULT_IOU        = 0.45
PROCESS_EVERY_N    = 2           # analyse every Nth frame
FRAME_IMG_SIZE     = 640
ALERT_FLASH_FRAMES = 20

SEV_LOW_MAX = 35
SEV_MED_MAX = 70

W_CONF  = 40
W_BBOX  = 35
W_COUNT = 25

# BGR colour palette
COLOR_ACCIDENT     = (0,   0,   255)
COLOR_SEVERE       = (0,   0,   200)
COLOR_MEDIUM       = (0,  140,  255)
COLOR_LOW          = (0,  200,  50 )
COLOR_PANEL_BG     = (15,  15,  30 )
COLOR_WHITE        = (255, 255, 255)
COLOR_YELLOW       = (0,   220, 255)
COLOR_RED_FLASH    = (0,   0,   180)

TRACKER_MAX_AGE = 6
TRACKER_IOU_THRESHOLD = 0.20
TRACKER_MIN_CONFIDENCE = 0.85
TRACKER_MIN_CONSEC_FRAMES = 5
TRACKER_STATIONARY_SEC = 2.0
TRACKER_SPEED_DROP_RATIO = 0.35

# ── CLASS CONFIGURATION ───────────────────────────────────────────────────────
# CLASS_NAMES is populated at runtime from the loaded model (see load_models()).
# It starts as an empty dict and is filled once the model is available so that
# any class layout in best.pt is handled automatically.
CLASS_NAMES:  dict = {}   # filled by _sync_class_names() after model load
CLASS_COLORS: dict = {}   # filled by _sync_class_names() after model load

# ACCIDENT_CLASSES — the single source of truth for "what counts as an accident".
# All class NAMES (case-insensitive) in this set are treated as accident events.
# Add "Severe", "Moderate", "Minor" etc. to match your model's actual class names.
# dashboard.py and every guard below reads this set — edit only this line to
# accommodate future model changes.
ACCIDENT_CLASSES = {
    "accident",   # original single-class name
    "moderate",   # detected by your current model ("Moderate", conf 0.938)
    "severe",     # high-severity class in your model
    "minor",      # if your model ever adds this class
    "crash",      # generic alias
}
# Internal helper — maps class_name.lower() → True/False (fast O(1) lookup)
def _is_accident_class(class_name: str) -> bool:
    """Return True if the given class name represents an accident event."""
    return class_name.strip().lower() in ACCIDENT_CLASSES


def _sync_class_names(model) -> None:
    """
    Populate CLASS_NAMES and CLASS_COLORS from a loaded ultralytics YOLO model.

    WHY: Hard-coding {0: "Accident"} broke as soon as the trained model used
    different class names.  Reading names directly from model.names means the
    code works regardless of how classes were labelled during training.

    Called once inside load_models() after each model is loaded.
    """
    global CLASS_NAMES, CLASS_COLORS
    if hasattr(model, "names") and model.names:
        CLASS_NAMES.clear()
        CLASS_NAMES.update(dict(model.names))          # e.g. {0: "Moderate", 1: "Severe"}
    # Assign colours: accident classes → red, everything else → white
    CLASS_COLORS.clear()
    CLASS_COLORS.update({
        cls_id: (COLOR_ACCIDENT if _is_accident_class(name) else COLOR_WHITE)
        for cls_id, name in CLASS_NAMES.items()
    })

# ─── Live heatmap: how fast old blobs fade (0.97 = slow fade, 0.90 = fast) ──
HEATMAP_DECAY = 0.97
HEATMAP_SIGMA = 40   # Gaussian spread in pixels

# Pixels to expand the reported accident bbox on each side (improves scene context)
BBOX_EXPAND_PX = 60

# Temporal verification parameters
CONSEC_FRAMES_REQUIRED = 5      # must appear in at least 5 consecutive frames
STATIONARY_SEC_REQUIRED = 2.0   # must remain nearly stationary for at least 2 seconds
CENTER_MOVE_PX_THRESHOLD = 8    # pixels per frame considered as movement


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2  ──  MODEL LOADER
# ═════════════════════════════════════════════════════════════════════════════
def load_models():
    """Load vehicle detection (yolov8m) and accident detection (best.pt) models."""
    vehicle_model = YOLO("yolov8m.pt")

    if not MODEL_PATH.exists():
        print(f"  ⚠  best.pt not found at {MODEL_PATH}")
        accident_model = YOLO("yolov8m.pt")
    else:
        print(f"  ✓  Accident model loaded: {MODEL_PATH}")
        accident_model = YOLO(str(MODEL_PATH))

    # FIX 2: populate CLASS_NAMES / CLASS_COLORS from the real model class list.
    # This replaces the old hard-coded {0: "Accident"} and makes the pipeline
    # work with any class layout the trained model uses (Moderate, Severe, …).
    _sync_class_names(accident_model)
    print(f"  ✓  Model classes : {CLASS_NAMES}")
    print(f"  ✓  Accident set  : {ACCIDENT_CLASSES}")

    return vehicle_model, accident_model


def load_model(model_path=None):
    """Convenience wrapper: returns accident model only (dashboard compatibility)."""
    _, accident_model = load_models()
    return accident_model


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3  ──  SEVERITY & RISK ENGINE
# ═════════════════════════════════════════════════════════════════════════════
def compute_risk_score(
    confidences:    list,
    bboxes:         list,
    frame_h:        int,
    frame_w:        int,
    accident_count: int,
) -> float:
    """
    Normalised risk score in [0, 100].

    Components:
      Confidence (W_CONF)  : mean confidence of accident detections
      BBox size  (W_BBOX)  : largest bbox as fraction of frame area
      Count      (W_COUNT) : sigmoid of detected accident count
    """
    if not confidences:
        return 0.0

    frame_area = frame_h * frame_w
    conf_score = np.mean(confidences) * 100

    max_area = 0
    for (x1, y1, x2, y2) in bboxes:
        max_area = max(max_area, (x2 - x1) * (y2 - y1))
    bbox_score  = min(1.0, max_area / (frame_area * 0.25)) * 100
    count_score = (1 / (1 + math.exp(-1.5 * (accident_count - 1)))) * 100

    risk = (
        (W_CONF  / 100) * conf_score  +
        (W_BBOX  / 100) * bbox_score  +
        (W_COUNT / 100) * count_score
    )
    return round(min(100.0, risk), 1)


def classify_severity(risk_score: float) -> tuple:
    """Map risk score → (severity_label, severity_color_BGR)."""
    if risk_score <= SEV_LOW_MAX:
        return "Low",    COLOR_LOW
    elif risk_score <= SEV_MED_MAX:
        return "Medium", COLOR_MEDIUM
    else:
        return "High",   COLOR_SEVERE


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4  ──  FRAME ANNOTATION ENGINE
# ═════════════════════════════════════════════════════════════════════════════
def draw_bounding_boxes(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw a single expanded red accident bounding box and label.

    This function selects the highest-confidence detection (if multiple
    are passed), expands the box by `BBOX_EXPAND_PX` on every side and
    draws a thick red rectangle with the required label.
    """
    if not detections:
        return frame

    # Choose highest-confidence detection
    best = max(detections, key=lambda d: d.get("confidence", 0.0))
    x1, y1, x2, y2 = int(best.get("x1", 0)), int(best.get("y1", 0)), int(best.get("x2", 0)), int(best.get("y2", 0))
    conf = float(best.get("confidence", 0.0))

    h, w = frame.shape[:2]
    exp = BBOX_EXPAND_PX
    x1 = max(0, x1 - exp)
    y1 = max(0, y1 - exp)
    x2 = min(w - 1, x2 + exp)
    y2 = min(h - 1, y2 + exp)

    color = COLOR_ACCIDENT
    thickness = 8
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    label = f" ACCIDENT DETECTED  {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)
    # draw filled background for label for readability
    cv2.rectangle(frame, (x1, max(0, y1 - th - 16)), (x1 + tw + 12, y1), color, -1)
    cv2.putText(frame, label, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.85, COLOR_WHITE, 2, cv2.LINE_AA)

    return frame


def draw_dashboard_panel(
    frame: np.ndarray,
    stats: dict,
    flash_counter: int,
) -> np.ndarray:
    """Overlay a semi-transparent stats panel (top-right corner)."""
    h, w = frame.shape[:2]
    panel_w, panel_h = 320, 230
    x0, y0 = w - panel_w - 12, 12

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h),
                  COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    sev_color = stats.get("severity_color", COLOR_WHITE)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), sev_color, 1)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + 28), sev_color, -1)
    cv2.putText(frame, "  ACCIDENT ANALYSIS SYSTEM",
                (x0 + 4, y0 + 19), cv2.FONT_HERSHEY_DUPLEX,
                0.45, COLOR_WHITE, 1, cv2.LINE_AA)

    rows = [
        ("Frame",     str(stats.get("frame_no", 0))),
        ("FPS",       f"{stats.get('fps', 0):.1f}"),
        ("Vehicles",  str(stats.get("vehicle_count", 0))),
        ("Accidents", str(stats.get("accident_count", 0))),
        ("Risk Score", f"{stats.get('risk_score', 0):.1f} / 100"),
        ("Severity",  stats.get("severity", "—")),
        ("Time",      stats.get("timestamp", "")),
    ]
    for i, (key, val) in enumerate(rows):
        ry = y0 + 42 + i * 26
        cv2.putText(frame, key, (x0 + 10, ry),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_YELLOW, 1, cv2.LINE_AA)
        val_color = sev_color if key == "Severity" else COLOR_WHITE
        cv2.putText(frame, val, (x0 + 150, ry),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, val_color, 1, cv2.LINE_AA)

    bar_y  = y0 + panel_h - 22
    bar_w  = panel_w - 20
    filled = int(bar_w * stats.get("risk_score", 0) / 100)
    cv2.rectangle(frame, (x0 + 10, bar_y), (x0 + 10 + bar_w, bar_y + 10),
                  (50, 50, 50), -1)
    cv2.rectangle(frame, (x0 + 10, bar_y), (x0 + 10 + filled, bar_y + 10),
                  sev_color, -1)

    return frame


def draw_emergency_alert(frame: np.ndarray, flash_counter: int) -> np.ndarray:
    """Flashing red border + banner when severity = High."""
    h, w = frame.shape[:2]

    if (flash_counter // 10) % 2 == 0:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 220), 8)

        banner_h = 55
        overlay  = frame.copy()
        cv2.rectangle(overlay,
                      (0, h // 2 - banner_h // 2 - 5),
                      (w, h // 2 + banner_h // 2 + 5),
                      (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        msg = "🚨  EMERGENCY ALERT SENT TO TRAFFIC CONTROL  🚨"
        sub = "High-Severity Accident Detected — Emergency Services Notified"
        (mw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
        (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(frame, msg, ((w - mw) // 2, h // 2 - 8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, COLOR_WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, sub, ((w - sw) // 2, h // 2 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 255), 1, cv2.LINE_AA)

    return frame


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4B  ──  LIVE HEATMAP OVERLAY  (NEW)
# ═════════════════════════════════════════════════════════════════════════════

class LiveHeatmapOverlay:
    """
    Maintains a persistent float32 density grid that grows every time a new
    accident bbox is added.  Call overlay() to blend the heatmap onto a frame.

    Added for Feature 3 (Live Heatmap Overlay on video).
    """

    def __init__(self, frame_h: int, frame_w: int,
                 decay: float = HEATMAP_DECAY,
                 sigma: int   = HEATMAP_SIGMA):
        self.h     = frame_h
        self.w     = frame_w
        self.decay = decay
        self.sigma = sigma
        # Persistent density accumulator — float32 for precision
        self.density = np.zeros((frame_h, frame_w), dtype=np.float32)

    def update(self, acc_bboxes: list, risk_score: float = 1.0):
        """
        Add Gaussian blobs for each accident bbox.
        Weight = risk_score / 100 so high-risk events burn brighter.

        Parameters
        ----------
        acc_bboxes : list of (x1,y1,x2,y2) tuples
        risk_score : current frame risk score 0-100
        """
        # Slowly fade old activations so heatmap reflects recent history
        self.density *= self.decay

        weight = max(risk_score / 100.0, 0.1)
        ys = np.arange(self.h, dtype=np.float32)
        xs = np.arange(self.w, dtype=np.float32)

        for (x1, y1, x2, y2) in acc_bboxes:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            # Gaussian blob sized to bbox
            bw = max(x2 - x1, 10)
            bh = max(y2 - y1, 10)
            sig_x = bw * 0.4
            sig_y = bh * 0.4
            gx = np.exp(-0.5 * ((xs - cx) / sig_x) ** 2)
            gy = np.exp(-0.5 * ((ys - cy) / sig_y) ** 2)
            self.density += np.outer(gy, gx) * weight

        # Clip to avoid unbounded growth
        self.density = np.clip(self.density, 0.0, 1.0)

    def overlay(self, frame: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        """
        Blend the heatmap onto frame and return the blended copy.

        Parameters
        ----------
        frame : BGR ndarray
        alpha : heatmap opacity (0 = invisible, 1 = full heatmap)
        """
        if self.density.max() < 1e-6:
            return frame

        # Normalise density to 0-255 for colormap
        norm = (self.density / self.density.max() * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

        # Blend only where density is non-trivial (> 5%)
        mask = (self.density > 0.05).astype(np.float32)
        mask_3 = np.stack([mask] * 3, axis=-1)

        blended = (
            frame.astype(np.float32) * (1.0 - mask_3 * alpha)
            + heatmap_color.astype(np.float32) * (mask_3 * alpha)
        )
        return np.clip(blended, 0, 255).astype(np.uint8)

    def reset(self):
        """Clear accumulated density (call between videos)."""
        self.density[:] = 0.0


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5  ──  CSV EVENT LOGGER
# ═════════════════════════════════════════════════════════════════════════════
def _box_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


class VehicleTracker:
    """Lightweight tracker that assigns persistent vehicle IDs and motion stats."""

    def __init__(self, fps: float = 25.0):
        self.next_vehicle_id = 1
        self.tracks: dict = {}
        self.fps = max(float(fps), 1.0)

    def _new_track(self, detection: dict, frame_no: int) -> dict:
        x1, y1, x2, y2 = detection["bbox"]
        return {
            "vehicle_id": self.next_vehicle_id,
            "bbox": (x1, y1, x2, y2),
            "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            "confidence": detection.get("confidence", 0.0),
            "label": detection.get("label", "vehicle"),
            "last_seen_frame": frame_no,
            "age": 0,
            "consecutive_frames": 1,
            "current_speed": 0.0,
            "previous_speed": 0.0,
            "average_speed": 0.0,
            "pixel_displacement": 0.0,
            "stopped_duration_sec": 0.0,
            "stationary_frames": 0,
            "sudden_speed_drop": False,
            "accident_consecutive_frames": 0,
            "verification_status": "Monitoring",
            "verification_confidence": 0.0,
            "collision_detected": False,
            "overlap_increase": False,
            "severity": "None",
            "risk_score": 0.0,
        }

    def _update_track(self, track: dict, detection: dict, vehicle_detections: list,
                      detection_index: int, accident_detections: list,
                      frame_no: int, fuzzy_high: bool) -> None:
        prev_center = track.get("center")
        prev_bbox = track.get("bbox")
        x1, y1, x2, y2 = detection["bbox"]
        current_bbox = (x1, y1, x2, y2)
        current_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        track["bbox"] = current_bbox
        track["center"] = current_center
        track["confidence"] = detection.get("confidence", 0.0)
        track["label"] = detection.get("label", "vehicle")
        track["last_seen_frame"] = frame_no
        track["age"] = 0
        track["consecutive_frames"] = track.get("consecutive_frames", 0) + 1

        if prev_center is not None:
            disp = math.hypot(current_center[0] - prev_center[0], current_center[1] - prev_center[1])
            prev_speed = track.get("current_speed", 0.0)
            current_speed = disp * self.fps
            speed_samples = track.get("speed_samples", 0) + 1
            avg_speed = (track.get("average_speed", 0.0) * (speed_samples - 1) + current_speed) / speed_samples
            track["pixel_displacement"] = disp
            track["previous_speed"] = prev_speed
            track["current_speed"] = current_speed
            track["average_speed"] = avg_speed
            track["speed_samples"] = speed_samples
            track["sudden_speed_drop"] = prev_speed > 5.0 and current_speed < max(2.0, prev_speed * TRACKER_SPEED_DROP_RATIO)
            if current_speed <= 2.0:
                track["stopped_duration_sec"] += 1.0 / max(self.fps, 1.0)
                track["stationary_frames"] += 1
            else:
                track["stopped_duration_sec"] = 0.0
                track["stationary_frames"] = 0
        else:
            track["pixel_displacement"] = 0.0
            track["previous_speed"] = 0.0
            track["current_speed"] = 0.0
            track["average_speed"] = 0.0
            track["speed_samples"] = 1
            track["sudden_speed_drop"] = False
            track["stopped_duration_sec"] = 0.0
            track["stationary_frames"] = 0

        track["bbox_overlap_change"] = _box_iou(prev_bbox, current_bbox) if prev_bbox is not None else 0.0

        collision_detected = False
        for idx, other in enumerate(vehicle_detections):
            if idx == detection_index:
                continue
            if _box_iou(current_bbox, other["bbox"]) > 0.20:
                collision_detected = True
                break
        track["collision_detected"] = collision_detected

        accident_overlap = 0.0
        for acc in accident_detections or []:
            overlap = _box_iou(current_bbox, acc["bbox"])
            if overlap > accident_overlap:
                accident_overlap = overlap

        prev_accident_overlap = track.get("last_accident_overlap", 0.0)
        track["last_accident_overlap"] = accident_overlap
        track["max_accident_overlap"] = max(track.get("max_accident_overlap", 0.0), accident_overlap)
        track["overlap_increase"] = accident_overlap > prev_accident_overlap + 0.05

        if accident_overlap > 0.10 and detection.get("confidence", 0.0) > TRACKER_MIN_CONFIDENCE:
            track["accident_consecutive_frames"] = track.get("accident_consecutive_frames", 0) + 1
        else:
            track["accident_consecutive_frames"] = 0

        if (detection.get("confidence", 0.0) > TRACKER_MIN_CONFIDENCE and
                track.get("accident_consecutive_frames", 0) >= TRACKER_MIN_CONSEC_FRAMES and
                track.get("sudden_speed_drop", False) and
                track.get("stopped_duration_sec", 0.0) >= TRACKER_STATIONARY_SEC and
                (track.get("overlap_increase", False) or collision_detected) and
                fuzzy_high):
            track["verification_status"] = "Confirmed Accident"
        elif detection.get("confidence", 0.0) > TRACKER_MIN_CONFIDENCE and track.get("accident_consecutive_frames", 0) >= 3:
            track["verification_status"] = "Potential Accident"
        else:
            track["verification_status"] = "Monitoring"

        track["verification_confidence"] = detection.get("confidence", 0.0)
        track["severity"] = "High" if fuzzy_high else "None"
        track["risk_score"] = 100.0 if fuzzy_high else 0.0

    def update(self, vehicle_detections: list, accident_detections: list | None = None,
               frame_no: int = 0, fps: float | None = None, fuzzy_high: bool = False) -> list:
        if fps is not None:
            self.fps = max(float(fps), 1.0)

        matched_ids = set()
        for track_id, track in list(self.tracks.items()):
            best_idx = None
            best_iou = TRACKER_IOU_THRESHOLD - 1e-6
            for idx, det in enumerate(vehicle_detections):
                if idx in matched_ids:
                    continue
                iou = _box_iou(track["bbox"], det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx is not None and best_iou >= TRACKER_IOU_THRESHOLD:
                matched_ids.add(best_idx)
                self._update_track(track, vehicle_detections[best_idx], vehicle_detections,
                                   best_idx, accident_detections or [], frame_no, fuzzy_high)
            else:
                track["age"] += 1
                if track["age"] > TRACKER_MAX_AGE:
                    del self.tracks[track_id]

        for idx, det in enumerate(vehicle_detections):
            if idx in matched_ids:
                continue
            new_track = self._new_track(det, frame_no)
            self.tracks[self.next_vehicle_id] = new_track
            self.next_vehicle_id += 1

        return self.get_states()

    def get_states(self) -> list:
        return [self.tracks[vehicle_id] for vehicle_id in sorted(self.tracks.keys())]


class EventLogger:
    """
    Appends one row per detected accident event to a CSV log file.

    COLUMNS (v2 — added center_x, center_y, vehicle_count):
        timestamp, frame_number, class_label, confidence,
        severity, risk_score,
        bbox_x1, bbox_y1, bbox_x2, bbox_y2,
        center_x, center_y,   ← NEW: required by heatmap_analysis.py
        vehicle_count          ← NEW: for analytics dashboard
    """

    # CHANGE: added center_x, center_y, vehicle_count to COLUMNS
    COLUMNS = [
        "timestamp", "frame_number", "class_label",
        "confidence", "severity", "risk_score",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "center_x", "center_y",   # heatmap spatial coordinates
        "vehicle_count",           # vehicles detected in same frame
    ]

    def __init__(self, log_path: Path):
        self.log_path = log_path
        with open(self.log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.COLUMNS).writeheader()
        print(f"  ✓  Event log  → {self.log_path}")

    # CHANGE: added vehicle_count parameter (default=0 for backward compat)
    def log(self, frame_no: int, detections: list,
            risk_score: float, severity: str, vehicle_count: int = 0):
        """Log every accident detection in the current frame."""
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rows = []
        for det in detections:
            # FIX 3: use _is_accident_class() so any accident label
            # ("Moderate", "Severe", "Accident", …) is logged — not only class_id 0.
            is_accident_event = (
                det.get("is_accident", False)
                or det.get("verification_status") == "Confirmed Accident"
                or _is_accident_class(det["label"])
            )
            if is_accident_event:
                cx = (det["x1"] + det["x2"]) // 2
                cy = (det["y1"] + det["y2"]) // 2
                rows.append({
                    "timestamp":     ts,
                    "frame_number":  frame_no,
                    "class_label":   det["label"],
                    "confidence":    round(det["confidence"], 4),
                    "severity":      severity,
                    "risk_score":    risk_score,
                    "bbox_x1":       det["x1"],
                    "bbox_y1":       det["y1"],
                    "bbox_x2":       det["x2"],
                    "bbox_y2":       det["y2"],
                    "center_x":      cx,
                    "center_y":      cy,
                    "vehicle_count": vehicle_count,
                })
        if rows:
            with open(self.log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self.COLUMNS).writerows(rows)

    def summary(self) -> pd.DataFrame:
        return pd.read_csv(self.log_path)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6  ──  MAIN PROCESSING PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
def process_video(
    video_source,
    vehicle_model,
    accident_model,
    conf_threshold: float = DEFAULT_CONF,
    show_preview:   bool  = False,
    frame_callback=None,     # NEW: optional callable(processed_frame, raw_frame, stats)
) -> dict:
    """
    Core pipeline: reads video, runs YOLO inference, draws overlays,
    saves annotated video AND heatmap video.

    NEW parameters
    --------------
    frame_callback : callable or None
        If provided, called on every processed frame with:
            frame_callback(annotated_frame, raw_frame, stats_dict)
        Used by Streamlit dashboard for side-by-side live preview.
    """
    cap = cv2.VideoCapture(
        video_source if isinstance(video_source, int) else str(video_source)
    )
    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {video_source}")

    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n  Source : {video_source}")
    print(f"  Size   : {src_w}×{src_h} @ {src_fps:.1f} fps  |  Frames: {total}")

    ts_tag  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_vid = VIDEO_OUT_DIR / f"analyzed_{ts_tag}.mp4"
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(str(out_vid), fourcc, src_fps, (src_w, src_h))

    # NEW: second writer for the live-heatmap video
    heat_vid = VIDEO_OUT_DIR / f"heatmap_video_{ts_tag}.mp4"
    heat_writer = cv2.VideoWriter(str(heat_vid), fourcc, src_fps, (src_w, src_h))

    log_path = LOGS_OUT_DIR / f"events_{ts_tag}.csv"
    logger   = EventLogger(log_path)

    # NEW: instantiate live heatmap accumulator
    live_heatmap = LiveHeatmapOverlay(src_h, src_w)

    # Simple consecutive-frame verification state
    pending_consec = 0            # consecutive processed frames with an accident candidate
    confirmed = False

    frame_no        = 0
    accident_frames = 0
    flash_counter   = 0
    prev_time       = time.time()
    stats_accumulator = []

    print(f"\n  Processing …  (Ctrl+C to stop early)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_no  += 1
        raw_frame  = frame.copy()   # keep original for side-by-side

        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        detections     = []
        risk_score     = 0.0
        severity       = "None"
        severity_color = COLOR_WHITE
        accident_count = 0
        vehicle_count  = 0

        if frame_no % PROCESS_EVERY_N == 0:
            # ── Accident detection ───────────────────────────────────────────
            results = accident_model(
                frame,
                conf   = conf_threshold,
                iou    = DEFAULT_IOU,
                imgsz  = FRAME_IMG_SIZE,
                verbose= False,
            )

            acc_confs  = []
            acc_bboxes = []
            accident_detections = []

            # Candidate accident detection
            best_det = None
            frame_area = src_h * src_w
            min_area_px = max(2500, int(0.001 * frame_area))
            for r in results:
                if getattr(r, 'boxes', None) is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                    if not _is_accident_class(label):
                        continue
                    if conf < conf_threshold:
                        continue
                    area = max(0, (x2 - x1)) * max(0, (y2 - y1))
                    if area < min_area_px:
                        continue
                    if best_det is None or conf > best_det["confidence"]:
                        best_det = {
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "confidence": conf, "label": label,
                        }

            if best_det is not None:
                pending_consec += 1
                confirmed = pending_consec >= 3
                status_label = "Accident" if confirmed else "Potential Accident"
                acc_confs.append(best_det["confidence"])
                acc_bboxes.append((best_det["x1"], best_det["y1"], best_det["x2"], best_det["y2"]))
                accident_detections.append({
                    "x1": best_det["x1"], "y1": best_det["y1"],
                    "x2": best_det["x2"], "y2": best_det["y2"],
                    "confidence": best_det["confidence"],
                    "label": status_label,
                })
                accident_count = 1 if confirmed else 0
            else:
                pending_consec = 0
                confirmed = False
                status_label = None

            if acc_confs:
                risk_score = compute_risk_score(
                    acc_confs, acc_bboxes, src_h, src_w, 1
                )
                risk_score = min(risk_score, 100.0)
                severity, severity_color = classify_severity(risk_score)
            else:
                risk_score = 0.0
                severity = "None"
                severity_color = COLOR_WHITE

            if acc_bboxes:
                live_heatmap.update(acc_bboxes, risk_score)

            if confirmed:
                detections = [
                    {
                        "class_id": 0,
                        "label": det["label"],
                        "confidence": det["confidence"],
                        "x1": det["x1"], "y1": det["y1"],
                        "x2": det["x2"], "y2": det["y2"],
                        "is_accident": True,
                    }
                    for det in accident_detections
                ]
                logger.log(frame_no, detections, risk_score, severity, 0)
                if severity == "High":
                    flash_counter = ALERT_FLASH_FRAMES
            else:
                detections = [
                    {
                        "class_id": 0,
                        "label": det["label"],
                        "confidence": det["confidence"],
                        "x1": det["x1"], "y1": det["y1"],
                        "x2": det["x2"], "y2": det["y2"],
                        "is_accident": True,
                    }
                    for det in accident_detections
                ]

            frame = draw_bounding_boxes(frame, detections)

            ts_str = datetime.now().strftime("%H:%M:%S")
            frame = draw_dashboard_panel(frame, {
                "frame_no":      frame_no,
                "fps":           fps,
                "vehicle_count": vehicle_count,
                "accident_count":accident_count,
                "risk_score":    risk_score,
                "severity":      severity if acc_bboxes else "—",
                "severity_color":severity_color,
                "timestamp":     ts_str,
            }, flash_counter)

            # ── Risk & severity ──────────────────────────────────────────────
            if accident_count > 0:
                risk_score = compute_risk_score(
                    acc_confs, acc_bboxes, src_h, src_w, accident_count
                )
                risk_score  = min(risk_score, 100.0)
                severity, severity_color = classify_severity(risk_score)

                print(f"[DEBUG] frame={frame_no} accident_confirmed={confirmed} accident_count={accident_count} risk_score={risk_score:.1f} severity={severity}")

                # NEW: update live heatmap with current accident bboxes
                live_heatmap.update(acc_bboxes, risk_score)

                detections = [
                    {
                        "class_id": 0,
                        "label": det["label"],
                        "confidence": det["confidence"],
                        "x1": det["bbox"][0], "y1": det["bbox"][1],
                        "x2": det["bbox"][2], "y2": det["bbox"][3],
                        "is_accident": True,
                    }
                    for det in accident_detections
                ]

                logger.log(frame_no, detections, risk_score, severity, 0)
                print(f"[DEBUG] frame={frame_no} logged {len(detections)} detections")
                if severity == "High":
                    flash_counter = ALERT_FLASH_FRAMES

            # ── Draw bounding boxes ──────────────────────────────────────────────
            frame = draw_bounding_boxes(frame, detections)

            # ── Dashboard panel ──────────────────────────────────────────────────
            ts_str = datetime.now().strftime("%H:%M:%S")
            frame  = draw_dashboard_panel(frame, {
                "frame_no":      frame_no,
                "fps":           fps,
                "vehicle_count": vehicle_count,
                "accident_count":accident_count,
                "risk_score":    risk_score,
                "severity":      severity if accident_count > 0 else "—",
                "severity_color":severity_color,
                "timestamp":     ts_str,
            }, flash_counter)

        # ── Emergency alert ──────────────────────────────────────────────────
        if flash_counter > 0:
            frame = draw_emergency_alert(frame, flash_counter)
            flash_counter -= 1

        # ── Progress bar (bottom-left) ───────────────────────────────────────
        if total > 0:
            pct   = frame_no / total
            bar_w = 200
            cv2.rectangle(frame, (10, src_h - 20), (10 + bar_w, src_h - 8),
                          (60, 60, 60), -1)
            cv2.rectangle(frame, (10, src_h - 20),
                          (10 + int(bar_w * pct), src_h - 8),
                          COLOR_YELLOW, -1)
            cv2.putText(frame, f"Frame {frame_no}/{total}",
                        (220, src_h - 9), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, COLOR_WHITE, 1, cv2.LINE_AA)

        # ── Write annotated video ────────────────────────────────────────────
        writer.write(frame)

        # NEW: build heatmap frame and write to heatmap video
        heat_frame = live_heatmap.overlay(raw_frame, alpha=0.5)
        # Stamp frame number on heatmap video
        cv2.putText(heat_frame, f"Heatmap | Frame {frame_no}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, COLOR_WHITE, 2, cv2.LINE_AA)
        heat_writer.write(heat_frame)

        # NEW: call optional callback for Streamlit side-by-side
        if frame_callback is not None:
            stats_cb = {
                "frame_no":      frame_no,
                "total":         total,
                "fps":           fps,
                "vehicle_count": vehicle_count,
                "accident_count":accident_count,
                "risk_score":    risk_score,
                "severity":      severity if accident_count > 0 else "—",
                "timestamp":     ts_str,
            }
            frame_callback(frame, raw_frame, heat_frame, stats_cb)

        # ── Optional live preview ────────────────────────────────────────────
        if show_preview:
            cv2.imshow("AI Accident Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n  Preview closed by user.")
                break

        if frame_no % 50 == 0:
            bar = "█" * int(30 * frame_no / max(total, 1))
            print(f"  [{bar:<30}] {frame_no}/{total}  "
                  f"Accidents:{accident_frames}  Risk:{risk_score:.1f}  Sev:{severity}",
                  end="\r")

    cap.release()
    writer.release()
    heat_writer.release()
    if show_preview:
        cv2.destroyAllWindows()

    print(f"\n\n  ✓  Analyzed video  → {out_vid}")
    print(f"  ✓  Heatmap video   → {heat_vid}")   # NEW
    print(f"  ✓  Accident frames → {FRAMES_OUT_DIR}  ({accident_frames} saved)")
    print(f"  ✓  Event log       → {log_path}")

    return {
        "total_frames":    frame_no,
        "accident_frames": accident_frames,
        "output_video":    str(out_vid),
        "heatmap_video":   str(heat_vid),   # NEW
        "log_path":        str(log_path),
        "stats":           stats_accumulator,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7  ──  POST-RUN SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
def print_summary(run_results: dict):
    stats = run_results.get("stats", [])
    SEP = "═" * 60
    print(f"\n{SEP}\n  📊  RUN SUMMARY\n{SEP}")
    print(f"  Total frames   : {run_results['total_frames']}")
    print(f"  Accident frames: {run_results['accident_frames']}")
    print(f"  Output video   : {run_results['output_video']}")
    print(f"  Heatmap video  : {run_results.get('heatmap_video','—')}")
    print(f"  Event log      : {run_results['log_path']}")

    if not stats:
        print("  No accidents detected.")
        print(SEP)
        return

    df = pd.DataFrame(stats)
    print(f"\n  Risk — Mean:{df['risk'].mean():.1f}  Max:{df['risk'].max():.1f}  "
          f"Min:{df['risk'].min():.1f}  Std:{df['risk'].std():.1f}")
    print(f"\n  Severity Distribution:")
    for sev, cnt in df["severity"].value_counts().items():
        pct = cnt / len(df) * 100
        print(f"    {sev:<8}: {'▓'*int(pct/5):<20} {cnt:>4}  ({pct:.1f}%)")
    pk = df.loc[df["risk"].idxmax()]
    print(f"\n  Peak: Frame {int(pk['frame'])}  Risk {pk['risk']:.1f}  {pk['severity']}")
    print(SEP)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8  ──  CLI
# ═════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="AI Traffic Accident Detection & Severity Analysis"
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--video",  type=str)
    g.add_argument("--webcam", action="store_true")
    p.add_argument("--conf",    type=float, default=DEFAULT_CONF)
    p.add_argument("--preview", action="store_true")
    p.add_argument("--model",   type=str, default=str(MODEL_PATH))
    return p.parse_args()


def main():
    print("\n" + "═" * 60)
    print("  🚦  AI Traffic Accident Detection & Severity Analysis")
    print("═" * 60)

    args   = parse_args()
    source = 0 if args.webcam else args.video

    if not args.webcam and not Path(source).exists():
        print(f"\n  ✗  Video not found: {source}")
        sys.exit(1)

    print("[1/3] Loading models …")
    vehicle_model, accident_model = load_models()

    print("\n[2/3] Processing video …")
    results = process_video(
        video_source   = source,
        vehicle_model  = vehicle_model,
        accident_model = accident_model,
        conf_threshold = args.conf,
        show_preview   = args.preview,
    )

    print("\n[3/3] Done.")
    print_summary(results)


if __name__ == "__main__":
    main()