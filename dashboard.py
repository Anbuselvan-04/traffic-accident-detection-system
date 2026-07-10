"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   AI Traffic Accident Detection — Streamlit Dashboard  (v2 — fixed)         ║
║                                                                              ║
║   Run : streamlit run dashboard.py                                          ║
║                                                                              ║
║   Bug-fixes in this version:                                                 ║
║     • Fixed: load_models import (was load_model → now load_models)          ║
║     • Fixed: LiveHeatmapOverlay import from detect_and_analyze              ║
║     • Fixed: email send_alert() signature (no frame/vehicle_count needed)   ║
║     • Fixed: alert_ph.markdown() — filled in missing severity branches      ║
║     • Fixed: XAI indent error (email block was inside wrong if-block)       ║
║     • Fixed: heatmap update on EVERY accident frame (not only inside email) ║
║     • Fixed: heatmap overlay shown only on actual accident pixels           ║
║     • Fixed: bounding boxes drawn on every processed frame correctly        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ── Standard library ──────────────────────────────────────────────────────────
import io
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from ultralytics import YOLO
from collections import deque
import math

# ── Project root on sys.path ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ── Internal modules ──────────────────────────────────────────────────────────
from detect_and_analyze import (
    load_models,                         # FIX: was load_model (singular)
    compute_risk_score, classify_severity,
    CLASS_NAMES, DEFAULT_CONF, DEFAULT_IOU, FRAME_IMG_SIZE,
    _is_accident_class,
    SEV_LOW_MAX, SEV_MED_MAX,
    LiveHeatmapOverlay,                  # FIX: was missing
    MODEL_PATH, LOGS_OUT_DIR, FRAMES_OUT_DIR, VIDEO_OUT_DIR,
    EventLogger, draw_bounding_boxes, draw_dashboard_panel,
    draw_emergency_alert, COLOR_WHITE,
    CONSEC_FRAMES_REQUIRED, STATIONARY_SEC_REQUIRED, CENTER_MOVE_PX_THRESHOLD, PROCESS_EVERY_N,
)
from heatmap_analysis import (
    load_logs, compute_density_grid, find_hotspots,
    compute_frequency_summary, GAUSSIAN_SIGMA, HOTSPOT_TOPN,
    MIN_CLUSTER_DIST, DEFAULT_W, DEFAULT_H,
)
from fuzzy_severity import fuzzy_classify, infer as fuzzy_infer, RISK_UNIVERSE
from xai_explainer  import (
    XAIExplainer, generate_gradcam_map, compute_feature_contributions,
)
from email_alert import EmailAlertSystem, FIXED_LOCATION
from location_service import get_location, display_location_card, maps_link, LocationInfo


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 0  ──  PAGE CONFIG & DARK THEME
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Traffic Monitoring",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Global ──────────────────────────────────────────────── */
.stApp { background:#0D0D1A; color:#E0E0F0; }

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background:#12122A;
    border-right:1px solid #1F1F3F;
}

/* ── KPI metric cards ────────────────────────────────────── */
[data-testid="metric-container"] {
    background:#1A1A30;
    border:1px solid #2A2A4A;
    border-radius:10px;
    padding:14px 16px;
}
[data-testid="stMetricValue"] { color:#7EB8FF; font-size:1.8rem !important; }
[data-testid="stMetricLabel"] { color:#8888AA; }
[data-testid="stMetricDelta"] { font-size:0.85rem; }

/* ── Tab bar ─────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background:#12122A;
    border-bottom:2px solid #2A2A4A;
}
.stTabs [data-baseweb="tab"] {
    color:#8888AA;
    font-weight:600;
    padding:10px 22px;
}
.stTabs [aria-selected="true"] {
    color:#7EB8FF !important;
    border-bottom:3px solid #7EB8FF;
}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    background:#1F3060; color:white;
    border:1px solid #3050A0; border-radius:8px;
    font-weight:600; transition:all .2s;
}
.stButton > button:hover { background:#2A40A0; border-color:#5070D0; }

/* ── DataFrames ──────────────────────────────────────────── */
[data-testid="stDataFrame"] { background:#1A1A30 !important; }

/* ── Progress bar ────────────────────────────────────────── */
.stProgress > div > div { background:#3A7BD5; }

/* ── Severity badges ─────────────────────────────────────── */
.badge-high   { background:#C0392B; color:white; border-radius:6px;
                padding:3px 10px; font-weight:bold; }
.badge-medium { background:#D68910; color:white; border-radius:6px;
                padding:3px 10px; font-weight:bold; }
.badge-low    { background:#1E8449; color:white; border-radius:6px;
                padding:3px 10px; font-weight:bold; }

/* ── Emergency alert pulse ───────────────────────────────── */
.alert-box {
    background:linear-gradient(135deg,#4A0000,#800000);
    border:2px solid #FF4444; border-radius:10px;
    padding:16px 20px; text-align:center;
    font-size:1.1rem; font-weight:bold; color:white;
    animation:pulse 1.2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:.7;} }

/* ── Alert boxes by severity ─────────────────────────────── */
.alert-medium {
    background:linear-gradient(135deg,#3A2000,#6A4000);
    border:2px solid #FF8C00; border-radius:10px;
    padding:14px 20px; text-align:center;
    font-size:1.05rem; font-weight:bold; color:white;
}
.alert-low {
    background:linear-gradient(135deg,#002A00,#005000);
    border:2px solid #00CC44; border-radius:10px;
    padding:14px 20px; text-align:center;
    font-size:1.05rem; font-weight:bold; color:white;
}

/* ── Live KPI strip ──────────────────────────────────────── */
.kpi-strip {
    display:flex; gap:10px; flex-wrap:wrap;
    background:#12122A; border-radius:10px;
    padding:12px 16px; margin-bottom:10px;
    border:1px solid #2A2A4A;
}
.kpi-item {
    flex:1; min-width:100px; text-align:center;
    background:#1A1A30; border-radius:8px; padding:8px;
}
.kpi-label { font-size:0.72rem; color:#8888AA; }
.kpi-value { font-size:1.1rem; font-weight:bold; color:#7EB8FF; }

/* ── XAI reason card ─────────────────────────────────────── */
.xai-card {
    background:#1A1A30; border-radius:8px;
    border-left:4px solid #7EB8FF;
    padding:12px 16px; margin:6px 0;
}
.xai-card.high  { border-left-color:#E74C3C; }
.xai-card.medium{ border-left-color:#F39C12; }
.xai-card.low   { border-left-color:#2ECC71; }

/* ── Gallery card ────────────────────────────────────────── */
.gallery-card {
    background:#1A1A30; border-radius:10px;
    border:1px solid #2A2A4A; overflow:hidden;
    transition:border-color .2s;
}
.gallery-card:hover { border-color:#7EB8FF; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1  ──  LOCATION INITIALIZATION
# ═════════════════════════════════════════════════════════════════════════════
if "location" not in st.session_state:
    st.session_state["location"] = get_location()

current_location = st.session_state.get("location")
if current_location is None:
    current_location = get_location()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2  ──  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🚦 Traffic AI Monitor")
    st.markdown("---")

    model_ok  = MODEL_PATH.exists()
    log_files = sorted(LOGS_OUT_DIR.glob("*.csv"),
                       key=lambda p: p.stat().st_mtime, reverse=True)

    st.markdown("### 🟢 System Status")
    st.markdown(
        f"{'✅' if model_ok else '⚠️'} **Model:** "
        f"{'best.pt found' if model_ok else 'best.pt missing'}"
    )
    st.markdown(f"📂 **Logs:** {len(log_files)} available")
    if log_files:
        st.caption(f"Latest: {log_files[0].name}")
    st.markdown("---")

    st.markdown("### ⚙️ Detection Settings")
    conf_thresh = st.slider("Confidence Threshold", 0.1, 0.9,
                            float(DEFAULT_CONF), 0.05)
    use_fuzzy   = st.toggle("Fuzzy Logic Engine",  value=True,
                            help="Mamdani FIS severity scoring")
    use_xai     = st.toggle("XAI Explanations",    value=True,
                            help="Grad-CAM + feature contributions")
    st.markdown("---")

    st.markdown("### 📂 Log Source")
    log_options  = ["(auto-latest)"] + [p.name for p in log_files]
    selected_log = st.selectbox("Select event log", log_options)
    st.markdown("---")

    st.markdown("### 📍 Current Location")
    display_location_card(current_location)
    if current_location and getattr(current_location, "source", "") in {"GPS", "IP", "fixed"}:
        st.caption(f"Source: {current_location.source_label}")
    st.markdown("---")

    # ── Email alert settings ──────────────────────────────────────────────────
    st.markdown("### ✉️ Email Alert Settings")
    enable_email   = st.toggle("Enable Email Alerts", value=False)
    email_cooldown = st.number_input("Cooldown (seconds)", 30, 3600, 60, 30)
    smtp_host      = st.text_input("SMTP Host",    value="smtp.gmail.com")
    smtp_port      = st.number_input("SMTP Port",  value=587, step=1)
    sender_addr    = st.text_input("Sender Email", value="")
    sender_pass    = st.text_input("App Password", value="", type="password")
    recip_str      = st.text_input("Recipients (comma-separated)", value="")

    use_live_location = st.toggle(
        "🌐 Use Live Location (IP-based)", value=False,
        help="Looks up an approximate location from the server's public "
             "IP address instead of the fixed address below. This is a "
             "network-based estimate, not exact GPS — accuracy varies and "
             "requires internet access. Falls back to the fixed location "
             "if the lookup fails.",
    )

    if enable_email and sender_addr and sender_pass and recip_str:
        # Rebuild the email system whenever the relevant settings change
        # (session_state alone would otherwise keep using stale config,
        # e.g. if the live-location toggle is flipped after first setup).
        config_signature = (
            smtp_host, int(smtp_port), sender_addr, sender_pass,
            recip_str, int(email_cooldown), use_live_location,
        )
        if st.session_state.get("email_config_signature") != config_signature:
            st.session_state["email_system"] = EmailAlertSystem(
                smtp_host    = smtp_host,
                smtp_port    = int(smtp_port),
                sender       = sender_addr,
                password     = sender_pass,
                recipients   = [r.strip() for r in recip_str.split(",") if r.strip()],
                cooldown_sec = int(email_cooldown),
                use_live_location = use_live_location,
            )
            st.session_state["email_config_signature"] = config_signature

        email_sys_preview = st.session_state["email_system"]
        location_label = (
            email_sys_preview.current_location() if use_live_location
            else FIXED_LOCATION
        )
        st.caption(f"📍 Alert location: **{location_label}**"
                   + (" _(live, IP-based)_" if use_live_location else " _(fixed)_"))

        st.success("✅ Email system configured")
    else:
        st.session_state.pop("email_system", None)
        st.session_state.pop("email_config_signature", None)
        st.caption(f"📍 Alert location: **{FIXED_LOCATION}** _(fixed)_")
        if enable_email:
            st.warning("Fill all email fields to activate alerts.")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2  ──  CACHED HELPERS
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="⏳ Loading AI models...")
def get_models():
    """Load vehicle (yolov8m) and accident (best.pt) models — cached."""
    # FIX: use load_models() which returns (vehicle_model, accident_model)
    vehicle_model, accident_model = load_models()
    return vehicle_model, accident_model


@st.cache_data(show_spinner="⏳ Loading event log …", ttl=30)
def load_event_data(log_name: str) -> pd.DataFrame:
    if log_name == "(auto-latest)":
        if not log_files:
            return pd.DataFrame()
        src = str(log_files[0])
    else:
        src = str(LOGS_OUT_DIR / log_name)
    try:
        return load_logs(src)
    except Exception as e:
        st.error(f"Could not load log: {e}")
        return pd.DataFrame()


def sev_color(sev: str) -> str:
    return {"High": "#E74C3C", "Medium": "#F39C12", "Low": "#2ECC71"}.get(
        sev, "#7EB8FF"
    )


def dark_chart(fig, height: int = 320):
    fig.update_layout(
        height=height,
        paper_bgcolor="#0F0F1E", plot_bgcolor="#0F0F1E",
        font_color="white",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(bgcolor="#0F0F1E"),
        xaxis=dict(gridcolor="#1F1F3F"),
        yaxis=dict(gridcolor="#1F1F3F"),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3  ──  TABS
# ═════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎬 Live Detection",
    "📊 Analytics",
    "🗺  Heatmap",
    "🔍 XAI Panel",
    "🖼  Gallery",
    "📋 Event Log",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1  ──  LIVE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("## 🎬 Live Video Detection")

    uploaded_video = st.file_uploader(
        "Upload a traffic video (MP4 / AVI / MOV / MKV)",
        type=["mp4", "avi", "mov", "mkv"],
    )

    ctl1, ctl2, ctl3, ctl4 = st.columns([2, 1, 1, 1])
    process_btn     = ctl1.button("▶  Analyze Video", type="primary",
                                   disabled=(uploaded_video is None))
    show_raw        = ctl2.checkbox("Show original",  value=True)
    show_heatmap_cb = ctl3.checkbox("Show heatmap",   value=True)
    max_frames      = ctl4.number_input("Max frames (0=all)", 0, 50000, 0, 100)

    alert_ph    = st.empty()
    kpi_ph      = st.empty()
    progress_ph = st.empty()
    status_ph   = st.empty()

    col_orig, col_proc, col_heat = st.columns(3)
    orig_ph  = col_orig.empty()
    proc_ph  = col_proc.empty()
    heat_ph  = col_heat.empty()
    col_orig.caption("📷 Original")
    col_proc.caption("🔍 Detected")
    col_heat.caption("🗺  Heatmap")

    if process_btn and uploaded_video is not None:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            tmp_path = tmp.name

        status_ph.info("⏳ Loading models …")
        vehicle_model, accident_model = get_models()
        status_ph.info("🔄 Running analysis …")

        cap   = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ts_tag   = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = LOGS_OUT_DIR / f"events_{ts_tag}.csv"
        logger   = EventLogger(log_path)

        # FIX: LiveHeatmapOverlay properly imported and initialised
        live_hm    = LiveHeatmapOverlay(h, w)
        xai_engine = XAIExplainer() if use_xai else None
        prog_bar   = progress_ph.progress(0)

        frame_no       = 0
        accident_total = 0
        stats_live     = []
        last_sev       = "—"
        last_risk      = 0.0
        max_risk_seen  = 0.0
        last_xai       = None

        # Simple consecutive-frame verification state for live detection
        pending_consec = 0
        confirmed = False

        if "xai_report" not in st.session_state:
            st.session_state["xai_report"] = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1
            if max_frames > 0 and frame_no > max_frames:
                break

            raw_frame = frame.copy()

            # Skip odd frames for speed (process every 2nd frame)
            if frame_no % 2 != 0:
                prog_bar.progress(min(1.0, frame_no / total))
                continue

            # ── Accident detection ────────────────────────────────────────────
            detections     = []
            acc_confs      = []
            acc_bboxes     = []
            accident_detections = []
            accident_count = 0
            risk_score     = 0.0
            severity       = "—"

            acc_results = accident_model(
                frame,
                conf    = conf_thresh,
                iou     = DEFAULT_IOU,
                imgsz   = FRAME_IMG_SIZE,
                verbose = False,
            )

            best_det = None
            frame_area = h * w
            min_area_px = max(2500, int(0.001 * frame_area))
            for r in acc_results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                    if not _is_accident_class(label):
                        continue
                    if conf < conf_thresh:
                        continue
                    area = max(0, (x2 - x1)) * max(0, (y2 - y1))
                    if area < min_area_px:
                        continue
                    if best_det is None or conf > best_det["confidence"]:
                        best_det = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": conf, "label": label}

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

            if acc_confs:
                risk_score = compute_risk_score(acc_confs, acc_bboxes, h, w, 1)
                risk_score = min(risk_score, 100.0)
                severity, _ = classify_severity(risk_score)

            if acc_bboxes:
                live_hm.update(acc_bboxes, risk_score)

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

            if confirmed:
                logger.log(frame_no, detections, risk_score, severity, 0)
                accident_total += 1
                last_sev = severity
                last_risk = risk_score
                max_risk_seen = max(max_risk_seen, risk_score)
                alert_ph.markdown(
                    '<div class="alert-box">'
                    f'🚨 ACCIDENT DETECTED — {severity} | Risk: {risk_score:.1f} | Conf: {best_det["confidence"]:.2f}'
                    '</div>',
                    unsafe_allow_html=True,
                )
                email_sys = st.session_state.get("email_system")
                detection_frame_path = None
                if "accident_frame_path" not in st.session_state:
                    FRAMES_OUT_DIR.mkdir(parents=True, exist_ok=True)
                    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
                    detection_frame_path = str(FRAMES_OUT_DIR / f"accident_{ts_tag}.jpg")
                    saved = cv2.imwrite(detection_frame_path, frame)
                    if saved:
                        st.session_state["accident_frame_path"] = detection_frame_path
                        print(f"Detection frame saved:\n{detection_frame_path}")
                    else:
                        print(f"  ⚠ Failed to save detection frame: {detection_frame_path}")
                else:
                    detection_frame_path = st.session_state["accident_frame_path"]

                if email_sys and email_sys.should_send():
                    email_sys.send_alert(
                        risk_score = risk_score,
                        severity   = severity,
                        timestamp  = datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                        detection_frame_path = detection_frame_path,
                        location = current_location,
                    )
                fres = None
                if use_xai and xai_engine:
                    last_xai = xai_engine.explain(
                        frame        = raw_frame,
                        detections   = detections,
                        risk_score   = risk_score,
                        severity     = severity,
                        fuzzy_result = fres if use_fuzzy else None,
                        frame_no     = frame_no,
                        save_images  = True,
                    )
                    st.session_state["xai_report"] = last_xai
            else:
                alert_ph.empty()

            annotated = draw_bounding_boxes(frame.copy(), detections)
            sev_bgr = {
                "High":   (0, 0, 200),
                "Medium": (0, 140, 255),
                "Low":    (0, 200, 50),
            }.get(severity, COLOR_WHITE)
            annotated = draw_dashboard_panel(annotated, {
                "frame_no":       frame_no,
                "fps":            fps,
                "accident_count": accident_count,
                "risk_score":     risk_score,
                "severity":       severity,
                "severity_color": sev_bgr,
                "timestamp":      datetime.now().strftime("%H:%M:%S"),
            }, flash_counter=0)

            # ── Live KPI strip ────────────────────────────────────────────────
            kpi_html = f"""
            <div class="kpi-strip">
              <div class="kpi-item">
                <div class="kpi-label">Frame</div>
                <div class="kpi-value">{frame_no}/{total}</div>
              </div>
              <div class="kpi-item">
                <div class="kpi-label">FPS</div>
                <div class="kpi-value">{fps:.1f}</div>
              </div>
              <div class="kpi-item">
                <div class="kpi-label">Accidents</div>
                <div class="kpi-value" style="color:#E74C3C">{accident_count}</div>
              </div>
              <div class="kpi-item">
                <div class="kpi-label">Risk Score</div>
                <div class="kpi-value" style="color:{sev_color(last_sev)}">{last_risk:.1f}</div>
              </div>
              <div class="kpi-item">
                <div class="kpi-label">Severity</div>
                <div class="kpi-value" style="color:{sev_color(last_sev)}">{last_sev}</div>
              </div>
              <div class="kpi-item">
                <div class="kpi-label">Peak Risk</div>
                <div class="kpi-value">{max_risk_seen:.1f}</div>
              </div>
              <div class="kpi-item">
                <div class="kpi-label">Total Events</div>
                <div class="kpi-value">{accident_total}</div>
              </div>
            </div>
            """
            kpi_ph.markdown(kpi_html, unsafe_allow_html=True)

            # ── Side-by-side frame display (every 8 processed frames) ─────────
            if frame_no % 8 == 0:
                if show_raw:
                    orig_ph.image(
                        cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                    )
                proc_ph.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )
                # FIX: heatmap overlay is always up-to-date (updated above)
                if show_heatmap_cb:
                    heat_frame = live_hm.overlay(raw_frame.copy(), alpha=0.5)
                    heat_ph.image(
                        cv2.cvtColor(heat_frame, cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                    )

            prog_bar.progress(min(1.0, frame_no / total))

        cap.release()
        status_ph.success(f"✅ Analysis complete — {frame_no} frames processed")
        alert_ph.empty()
        progress_ph.empty()

        # ── Post-run summary ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📈 Run Summary")
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Frames",          frame_no)
        s2.metric("Accident Events", accident_total)
        s3.metric("Peak Risk",       f"{max_risk_seen:.1f}")
        s4.metric("Max Vehicles",    "—")
        s5.metric("Log",             log_path.name)

        if stats_live:
            df_live = pd.DataFrame(stats_live)
            fig = px.line(
                df_live, x="frame", y="risk",
                color_discrete_sequence=["#3A7BD5"],
                title="Risk Score Over Time",
                labels={"frame": "Frame #", "risk": "Risk Score"},
            )
            fig.add_hrect(y0=70, y1=105, fillcolor="#E74C3C", opacity=0.07,
                          annotation_text="High")
            fig.add_hrect(y0=35, y1=70,  fillcolor="#F39C12", opacity=0.05,
                          annotation_text="Medium")
            st.plotly_chart(dark_chart(fig), use_container_width=True)

        st.info(f"📁 XAI outputs → `outputs/xai/`  |  Log → `{log_path.name}`")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2  ──  ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("## 📊 Analytics Dashboard")

    df2 = load_event_data(selected_log)
    if df2.empty:
        st.warning("⚠️ No event data. Run a video analysis in **🎬 Live Detection** first.")
    else:
        summary = compute_frequency_summary(df2)

        st.markdown("### Key Performance Indicators")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Events",    summary.get("total_events", 0))
        c2.metric("Mean Risk",       f"{summary.get('mean_risk', 0):.1f}")
        c3.metric("Max Risk",        f"{summary.get('max_risk', 0):.1f}",
                  delta="⚠ HIGH" if summary.get("max_risk", 0) > 70 else "")
        c4.metric("Mean Confidence", f"{summary.get('mean_confidence', 0):.3f}")
        ph = summary.get("peak_hour")
        c5.metric("Peak Hour",       f"{ph:02d}:00" if ph is not None else "—")

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Risk Score Timeline")
            df_s = df2.sort_values("frame_number") if "frame_number" in df2.columns else df2
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=df_s.get("frame_number", df_s.index),
                y=df_s["risk_score"],
                mode="lines+markers",
                marker=dict(
                    color=df_s["severity"].map(
                        {"High": "#E74C3C", "Medium": "#F39C12", "Low": "#2ECC71"}
                    ),
                    size=5,
                ),
                line=dict(color="#3A7BD5", width=1.2),
            ))
            fig_line.add_hrect(y0=70, y1=100, fillcolor="#E74C3C", opacity=0.07,
                               annotation_text="High")
            fig_line.add_hrect(y0=35, y1=70,  fillcolor="#F39C12", opacity=0.05,
                               annotation_text="Medium")
            fig_line.update_yaxes(range=[0, 105])
            st.plotly_chart(dark_chart(fig_line), use_container_width=True)

        with col_b:
            st.markdown("#### Severity Distribution")
            sev_counts = df2["severity"].value_counts().reset_index()
            sev_counts.columns = ["Severity", "Count"]
            fig_pie = px.pie(
                sev_counts, names="Severity", values="Count", hole=0.55,
                color="Severity",
                color_discrete_map={"High":"#E74C3C","Medium":"#F39C12","Low":"#2ECC71"},
            )
            fig_pie.update_traces(textfont_size=13, textposition="outside")
            st.plotly_chart(dark_chart(fig_pie), use_container_width=True)

        if "hour" in df2.columns:
            st.markdown("#### Accidents by Hour of Day")
            hourly = df2.groupby(["hour", "severity"])["risk_score"].count().reset_index()
            hourly.columns = ["Hour", "Severity", "Count"]
            fig_hr = px.bar(
                hourly, x="Hour", y="Count", color="Severity", barmode="stack",
                color_discrete_map={"High":"#E74C3C","Medium":"#F39C12","Low":"#2ECC71"},
            )
            st.plotly_chart(dark_chart(fig_hr, 280), use_container_width=True)

        st.markdown("#### Risk Score Distribution")
        fig_hist = px.histogram(
            df2, x="risk_score", nbins=30, color="severity",
            color_discrete_map={"High":"#E74C3C","Medium":"#F39C12","Low":"#2ECC71"},
        )
        fig_hist.update_layout(bargap=0.05)
        st.plotly_chart(dark_chart(fig_hist, 260), use_container_width=True)




# ─────────────────────────────────────────────────────────────────────────────
# TAB 3  ──  HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("## 🗺  Accident Hotspot Heatmap")

    df3 = load_event_data(selected_log)
    if df3.empty:
        st.warning("No data available. Run analysis first.")
    else:
        c_w, c_h = st.columns(2)
        canvas_w = c_w.number_input("Video width (px)",  480, 3840, DEFAULT_W, 160)
        canvas_h = c_h.number_input("Video height (px)", 360, 2160, DEFAULT_H, 90)

        # FIX: guard against missing center_x/center_y columns
        if "center_x" not in df3.columns or "center_y" not in df3.columns:
            st.warning("Log file is missing center_x / center_y columns. "
                       "Re-run analysis with the updated detect_and_analyze.py.")
        else:
            st.markdown("#### Accident Location Scatter")
            fig_scat = px.scatter(
                df3, x="center_x", y="center_y",
                color="severity", size="risk_score", size_max=22, opacity=0.7,
                color_discrete_map={"High":"#E74C3C","Medium":"#F39C12","Low":"#2ECC71"},
                labels={"center_x":"X (px)","center_y":"Y (px)"},
                title="Point size = risk score",
            )
            fig_scat.update_layout(
                xaxis=dict(range=[0, canvas_w]),
                yaxis=dict(range=[canvas_h, 0]),
            )
            st.plotly_chart(dark_chart(fig_scat, 420), use_container_width=True)

            st.markdown("#### Kernel Density Heatmap")
            fig_kde = go.Figure(go.Histogram2dContour(
                x=df3["center_x"], y=df3["center_y"],
                colorscale="Hot", reversescale=False, showscale=True,
                contours=dict(showlabels=True, labelfont=dict(size=8, color="white")),
                line=dict(width=0),
            ))
            fig_kde.add_trace(go.Scatter(
                x=df3["center_x"], y=df3["center_y"],
                mode="markers",
                marker=dict(color="#FFFFFF", size=3, opacity=0.4),
            ))
            fig_kde.update_layout(
                xaxis=dict(range=[0, canvas_w], title="X (px)"),
                yaxis=dict(range=[canvas_h, 0], title="Y (px)"),
            )
            st.plotly_chart(dark_chart(fig_kde, 420), use_container_width=True)

            st.markdown("#### 🔥 Top Hotspots")
            density  = compute_density_grid(
                df3["center_x"].values, df3["center_y"].values,
                int(canvas_w), int(canvas_h),
                sigma=GAUSSIAN_SIGMA, risk_weights=df3["risk_score"].values,
            )
            hotspots = find_hotspots(
                density, df3, int(canvas_w), int(canvas_h),
                top_n=HOTSPOT_TOPN, min_dist=MIN_CLUSTER_DIST,
            )
            if hotspots:
                hs_df = pd.DataFrame(hotspots)[[
                    "rank","cx","cy","event_count","mean_risk","max_risk","dominant_sev"
                ]]
                hs_df.columns = ["Rank","X","Y","Events","Mean Risk","Max Risk","Dominant Sev"]
                st.dataframe(
                    hs_df.style.background_gradient(subset=["Mean Risk"], cmap="Reds"),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info("No hotspots yet — need more spatially-clustered events.")

        heat_vids = sorted(VIDEO_OUT_DIR.glob("heatmap_video_*.mp4"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        if heat_vids:
            st.markdown("#### 🎞️ Latest Heatmap Video")
            st.video(str(heat_vids[0]))


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4  ──  XAI PANEL
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("## 🔍 Explainable AI Panel")

    latest_xai = st.session_state.get("xai_report")

    if latest_xai:
        st.markdown("### 📌 Latest Detection Explanation")

        rpt      = latest_xai["json_report"]
        sev      = rpt["severity"]
        risk     = rpt["risk_score"]
        contribs = latest_xai["contributions"]

        sev_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(sev, "⚪")
        st.markdown(
            f'<div class="xai-card {sev.lower()}">'
            f'<strong>{sev_emoji} {sev} Severity</strong> — '
            f'Risk Score: <strong>{risk:.1f}/100</strong><br/>'
            f'<small>{rpt.get("summary","")}</small>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("#### Why did the AI classify this as an accident?")
        reason_map = {
            "Detection Confidence":   ("🎯", "High model confidence"),
            "Bounding Box Coverage":  ("📐", "Large collision area detected"),
            "Accident Count":         ("🚗", "Multiple vehicle involvement"),
        }
        for c in contribs:
            if c.get("contribution_pct") is None:
                continue
            icon, short = reason_map.get(c["feature"], ("ℹ️", c["feature"]))
            pct  = c["contribution_pct"]
            expl = c["explanation"]
            col_icon, col_text, col_bar = st.columns([0.5, 4, 2])
            col_icon.markdown(f"### {icon}")
            col_text.markdown(f"**{short}**  \n{expl}")
            col_bar.progress(min(1.0, pct / 100))
            col_bar.caption(f"{pct:.1f}% weight")

        top_rules = latest_xai.get("top_rules", [])
        if top_rules:
            st.markdown("#### Active Fuzzy Rules")
            rules_df = pd.DataFrame([{
                "Rule":    r["rule"],
                "Output":  r["output"],
                "Strength":r["strength"],
                "Conf":    r["antecedents"]["conf"][0],
                "BBox":    r["antecedents"]["bbox"][0],
                "Count":   r["antecedents"]["count"][0],
            } for r in top_rules])
            st.dataframe(
                rules_df.style.background_gradient(subset=["Strength"], cmap="Blues"),
                use_container_width=True, hide_index=True,
            )

        col_gc, col_ch = st.columns(2)
        if latest_xai.get("gradcam_path"):
            col_gc.markdown("**Grad-CAM Saliency**")
            col_gc.image(str(latest_xai["gradcam_path"]), use_container_width=True)
        if latest_xai.get("chart_path"):
            col_ch.markdown("**Feature Contributions**")
            col_ch.image(str(latest_xai["chart_path"]), use_container_width=True)

    else:
        st.info("Run a video analysis in **🎬 Live Detection** first to see XAI results here.")

    # Saved XAI image browser
    st.markdown("---")
    st.markdown("### 🗂️ Saved XAI Outputs")
    xai_dir      = BASE_DIR / "outputs" / "xai"
    xai_dir.mkdir(parents=True, exist_ok=True)
    gradcam_imgs = sorted(xai_dir.glob("gradcam_*.jpg"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
    xai_charts   = sorted(xai_dir.glob("xai_chart_*.png"),
                          key=lambda p: p.stat().st_mtime, reverse=True)

    if gradcam_imgs:
        sel = st.selectbox("Select saved result",
                           range(min(len(gradcam_imgs), 20)),
                           format_func=lambda i: gradcam_imgs[i].name)
        c1x, c2x = st.columns(2)
        c1x.image(str(gradcam_imgs[sel]), caption="Grad-CAM",
                  use_container_width=True)
        if sel < len(xai_charts):
            c2x.image(str(xai_charts[sel]), caption="Feature Chart",
                      use_container_width=True)
    else:
        st.info("No saved XAI images yet.")

    # Interactive fuzzy playground
    st.markdown("---")
    st.markdown("### 🧮 Fuzzy Logic Playground")
    p1, p2, p3 = st.columns(3)
    play_conf  = p1.slider("Confidence",      0.0, 1.0, 0.70, 0.01)
    play_bbox  = p2.slider("BBox Ratio",      0.0, 1.0, 0.15, 0.01)
    play_count = p3.slider("Accident Count",    1,  10,   2,    1)

    fres = fuzzy_infer(play_conf, play_bbox, play_count)
    r1, r2, r3 = st.columns(3)
    r1.metric("Risk Score",  f"{fres.risk_score:.1f} / 100")
    r2.metric("Severity",    fres.severity)
    r3.metric("Rules Fired", len(fres.rules_fired))

    mu = fres.membership
    mem_data = {
        "Set": ["conf_low","conf_medium","conf_high",
                "bbox_small","bbox_medium","bbox_large",
                "count_few","count_some","count_many"],
        "μ Value": [
            mu.get("conf_low",0),   mu.get("conf_medium",0), mu.get("conf_high",0),
            mu.get("bbox_small",0), mu.get("bbox_medium",0), mu.get("bbox_large",0),
            mu.get("count_few",0),  mu.get("count_some",0),  mu.get("count_many",0),
        ],
        "Group": ["Confidence"]*3 + ["BBox"]*3 + ["Count"]*3,
    }
    fig_mem = px.bar(
        mem_data, x="Set", y="μ Value", color="Group", barmode="group",
        color_discrete_sequence=["#3A7BD5","#E74C3C","#2ECC71"],
        range_y=[0, 1.05],
    )
    st.plotly_chart(dark_chart(fig_mem, 260), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5  ──  GALLERY
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("## 🖼  Accident Frame Gallery")

    frame_files = sorted(
        FRAMES_OUT_DIR.glob("*.jpg"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )

    if not frame_files:
        st.info("No accident frames saved yet. Run a video analysis first.")
    else:
        st.markdown(f"**{len(frame_files)} frames saved** in `outputs/accident_frames/`")

        sev_filter_gallery = st.multiselect(
            "Filter by severity",
            ["high", "medium", "low"],
            default=["high", "medium", "low"],
        )
        filtered_frames = [
            f for f in frame_files
            if any(sev in f.stem for sev in sev_filter_gallery)
        ]
        st.caption(f"Showing {len(filtered_frames)} frames after filter")

        COLS = 3
        rows = [filtered_frames[i:i+COLS] for i in range(0, len(filtered_frames), COLS)]

        for row in rows:
            cols = st.columns(COLS)
            for col, fpath in zip(cols, row):
                parts    = fpath.stem.split("_")
                frame_no = parts[1] if len(parts) > 1 else "?"
                sev_str  = parts[2].capitalize() if len(parts) > 2 else "?"
                sev_col  = sev_color(sev_str)

                df_log    = load_event_data(selected_log)
                risk_disp = "—"
                ts_disp   = "—"
                if not df_log.empty and "frame_number" in df_log.columns:
                    match = df_log[df_log["frame_number"].astype(str) == frame_no]
                    if not match.empty:
                        risk_disp = f"{match.iloc[0]['risk_score']:.1f}"
                        ts_disp   = str(match.iloc[0].get("timestamp", "—"))[:19]

                with col:
                    st.image(str(fpath), use_container_width=True)
                    st.markdown(
                        f'<div style="background:#1A1A30;border-radius:6px;'
                        f'padding:6px 10px;margin-top:4px;'
                        f'border-left:4px solid {sev_col};">'
                        f'<span style="color:{sev_col};font-weight:bold;">'
                        f'{sev_str}</span>  '
                        f'<span style="color:#8888AA;font-size:0.8rem;">'
                        f'Risk: <strong style="color:#7EB8FF">{risk_disp}</strong>'
                        f'  Frame #{frame_no}</span><br/>'
                        f'<span style="color:#555577;font-size:0.72rem;">{ts_disp}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    with st.expander("🔍 Enlarge / Details"):
                        st.image(str(fpath), use_container_width=True)
                        st.markdown(f"**File:** `{fpath.name}`")
                        st.markdown(f"**Severity:** {sev_str}")
                        st.markdown(f"**Risk Score:** {risk_disp}")
                        st.markdown(f"**Frame Number:** {frame_no}")
                        st.markdown(f"**Timestamp:** {ts_disp}")
                        with open(str(fpath), "rb") as img_f:
                            st.download_button(
                                "⬇️ Download Frame",
                                data      = img_f.read(),
                                file_name = fpath.name,
                                mime      = "image/jpeg",
                            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6  ──  EVENT LOG
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown("## 📋 Event Log Viewer")

    df6 = load_event_data(selected_log)

    if df6.empty:
        st.warning("No log data available.")
    else:
        st.markdown(
            f"**Loaded:** `{selected_log}`  —  "
            f"**{len(df6):,} accident events**"
        )

        search_term = st.text_input(
            "🔎 Search log (searches timestamp, severity, class_label …)",
            value="",
            placeholder="e.g.  High  or  2024-06-01",
        )

        fc1, fc2, fc3 = st.columns(3)
        sev_filter = fc1.multiselect(
            "Severity", ["High","Medium","Low"],
            default=["High","Medium","Low"],
        )
        risk_range = fc2.slider("Risk Score Range", 0, 100, (0, 100))
        conf_range = fc3.slider("Confidence Range", 0.0, 1.0, (0.0, 1.0), 0.01)

        mask = (
            df6["severity"].isin(sev_filter) &
            df6["risk_score"].between(risk_range[0], risk_range[1]) &
            df6["confidence"].between(conf_range[0], conf_range[1])
        )
        df_view = df6[mask].reset_index(drop=True)

        if search_term.strip():
            str_cols  = df_view.select_dtypes(include="object").columns
            text_mask = df_view[str_cols].apply(
                lambda col: col.astype(str).str.contains(
                    search_term.strip(), case=False, na=False
                )
            ).any(axis=1)
            df_view = df_view[text_mask].reset_index(drop=True)

        st.markdown(f"Showing **{len(df_view):,}** rows after filters")

        cols_show = [c for c in [
            "timestamp","frame_number","class_label","confidence",
            "severity","risk_score","vehicle_count",
            "bbox_x1","bbox_y1","bbox_x2","bbox_y2",
        ] if c in df_view.columns]

        def color_severity(val):
            c = {"High":"#5A0000","Medium":"#4A3000","Low":"#003A00"}.get(val,"")
            return f"background-color:{c};color:white;" if c else ""

        st.dataframe(
            df_view[cols_show].style
                .applymap(color_severity, subset=["severity"])
                .format({"risk_score":"{:.1f}","confidence":"{:.3f}"}),
            use_container_width=True, height=450,
        )

        csv_bytes = df_view[cols_show].to_csv(index=False).encode()
        st.download_button(
            "⬇️  Download Filtered CSV",
            data      = csv_bytes,
            file_name = f"accident_log_filtered_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime      = "text/csv",
        )

        if len(df_view) > 0:
            st.markdown("---")
            st.markdown("### Quick Stats (filtered view)")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Events",    len(df_view))
            s2.metric("Mean Risk", f"{df_view['risk_score'].mean():.1f}")
            s3.metric("Max Risk",  f"{df_view['risk_score'].max():.1f}")
            s4.metric("Mean Conf", f"{df_view['confidence'].mean():.3f}")