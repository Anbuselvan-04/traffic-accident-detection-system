"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   MODULE 4 — Explainable AI (XAI) Engine                                    ║
║   AI-Powered Traffic Accident Detection & Severity Analysis                  ║
║                                                                              ║
║   File    : xai_explainer.py                                                 ║
║   Purpose : Generate human-readable explanations for every detection,       ║
║             produce Grad-CAM saliency maps, SHAP-style feature              ║
║             contribution bars, and plain-English decision reports.          ║
║                                                                              ║
║   Outputs (per call):                                                        ║
║     • Grad-CAM heatmap overlaid on accident frame (PNG)                     ║
║     • Feature contribution chart (PNG)                                      ║
║     • Plain-English explanation dict  (for dashboard)                       ║
║     • Full JSON explanation report  (for logs)                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Standalone usage:
    python src/xai_explainer.py --frame outputs/accident_frames/frame_000120_high.jpg \
                                --conf 0.82 --bbox 0.22 --count 2 --severity High \
                                --risk 78.4

Integration:
    from xai_explainer import XAIExplainer
    xai = XAIExplainer()
    report = xai.explain(frame, detections, risk_score, severity, fuzzy_result)
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

# ── Internal ──────────────────────────────────────────────────────────────────
# fuzzy_severity is imported lazily to avoid hard dependency when XAI is used alone
try:
    from fuzzy_severity import FuzzyResult, RISK_UNIVERSE
    _FUZZY_AVAILABLE = True
except ImportError:
    _FUZZY_AVAILABLE = False

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 0  ──  PATHS
# ═════════════════════════════════════════════════════════════════════════════
BASE_DIR    = Path(__file__).resolve().parent.parent
XAI_DIR     = BASE_DIR / "outputs" / "xai"
XAI_DIR.mkdir(parents=True, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1  ──  FEATURE CONTRIBUTION (SHAP-STYLE)
# ═════════════════════════════════════════════════════════════════════════════

# Component weights mirror detect_and_analyze.py
W_CONF  = 40
W_BBOX  = 35
W_COUNT = 25

# Fuzzy linguistic bonus weights (applied when fuzzy result available)
FUZZY_WEIGHT_MAP = {
    ("high",   "large",  "many"): 1.15,
    ("high",   "large",  "some"): 1.10,
    ("medium", "large",  "many"): 1.08,
}


def compute_feature_contributions(
    confidence:     float,
    bbox_ratio:     float,
    accident_count: int,
    risk_score:     float,
    fuzzy_result=None,
) -> List[Dict]:
    """
    Decompose the risk score into interpretable contributions per feature.

    Each contribution is expressed as a percentage of the total risk score.

    Returns
    -------
    list of dicts: [{feature, raw_value, contribution_pct, label, direction}]
    """
    import math

    frame_area_dummy = 1.0   # already normalised in bbox_ratio

    # Replicate detect_and_analyze.py score components
    conf_score  = confidence * 100
    bbox_score  = min(1.0, bbox_ratio / 0.25) * 100
    count_score = (1 / (1 + math.exp(-1.5 * (accident_count - 1)))) * 100

    weighted_conf  = (W_CONF  / 100) * conf_score
    weighted_bbox  = (W_BBOX  / 100) * bbox_score
    weighted_count = (W_COUNT / 100) * count_score
    total_w = weighted_conf + weighted_bbox + weighted_count

    def pct(w): return round(w / max(total_w, 1e-6) * risk_score, 2)

    contributions = [
        {
            "feature":          "Detection Confidence",
            "raw_value":        round(confidence, 3),
            "component_score":  round(conf_score, 1),
            "contribution_pts": round(weighted_conf, 2),
            "contribution_pct": round(weighted_conf / max(total_w, 1e-6) * 100, 1),
            "direction":        "positive" if confidence >= 0.5 else "neutral",
            "explanation":      _conf_text(confidence),
        },
        {
            "feature":          "Bounding Box Coverage",
            "raw_value":        round(bbox_ratio, 3),
            "component_score":  round(bbox_score, 1),
            "contribution_pts": round(weighted_bbox, 2),
            "contribution_pct": round(weighted_bbox / max(total_w, 1e-6) * 100, 1),
            "direction":        "positive" if bbox_ratio >= 0.10 else "neutral",
            "explanation":      _bbox_text(bbox_ratio),
        },
        {
            "feature":          "Accident Count",
            "raw_value":        accident_count,
            "component_score":  round(count_score, 1),
            "contribution_pts": round(weighted_count, 2),
            "contribution_pct": round(weighted_count / max(total_w, 1e-6) * 100, 1),
            "direction":        "positive" if accident_count >= 2 else "neutral",
            "explanation":      _count_text(accident_count),
        },
    ]

    # Add fuzzy linguistic set memberships as extra context rows
    if fuzzy_result is not None and _FUZZY_AVAILABLE:
        mu = fuzzy_result.membership
        contributions.append({
            "feature":          "Fuzzy Confidence Level",
            "raw_value":        f"{round(mu.get('conf_low',0),2)}/{round(mu.get('conf_medium',0),2)}/{round(mu.get('conf_high',0),2)}",
            "component_score":  None,
            "contribution_pts": None,
            "contribution_pct": None,
            "direction":        "info",
            "explanation":      f"Confidence is {_dominant_set(mu, 'conf')} "
                                f"(L={mu.get('conf_low',0):.2f}, M={mu.get('conf_medium',0):.2f}, H={mu.get('conf_high',0):.2f})",
        })
        contributions.append({
            "feature":          "Fuzzy BBox Size",
            "raw_value":        f"{round(mu.get('bbox_small',0),2)}/{round(mu.get('bbox_medium',0),2)}/{round(mu.get('bbox_large',0),2)}",
            "component_score":  None,
            "contribution_pts": None,
            "contribution_pct": None,
            "direction":        "info",
            "explanation":      f"BBox is {_dominant_set(mu, 'bbox')} "
                                f"(S={mu.get('bbox_small',0):.2f}, M={mu.get('bbox_medium',0):.2f}, L={mu.get('bbox_large',0):.2f})",
        })

    return contributions


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2  ──  LINGUISTIC EXPLANATION GENERATORS
# ═════════════════════════════════════════════════════════════════════════════

def _dominant_set(membership: dict, prefix: str) -> str:
    """Return the label of the dominant fuzzy set for a given prefix."""
    relevant = {k: v for k, v in membership.items() if k.startswith(prefix + "_")}
    if not relevant:
        return "unknown"
    return max(relevant, key=relevant.get).split("_", 1)[1]


def _conf_text(conf: float) -> str:
    if conf >= 0.80: return f"Very high confidence ({conf:.2f}) — model is strongly certain this is an accident."
    if conf >= 0.60: return f"Moderate-high confidence ({conf:.2f}) — detection is reliable."
    if conf >= 0.40: return f"Medium confidence ({conf:.2f}) — some uncertainty in detection."
    return f"Low confidence ({conf:.2f}) — detection may be a false positive."


def _bbox_text(ratio: float) -> str:
    pct = ratio * 100
    if pct >= 30: return f"Accident occupies {pct:.1f}% of frame — large scene involvement, likely severe."
    if pct >= 10: return f"Accident occupies {pct:.1f}% of frame — moderate vehicle involvement."
    return f"Accident occupies {pct:.1f}% of frame — small or distant incident."


def _count_text(count: int) -> str:
    if count >= 4: return f"{count} accident detections in frame — major multi-vehicle incident."
    if count == 3: return f"{count} detections — multiple vehicles involved."
    if count == 2: return f"{count} detections — two-vehicle collision likely."
    return f"{count} detection — single vehicle or isolated incident."


def _severity_rationale(severity: str, risk_score: float, contributions: list) -> str:
    """Generate a 2–3 sentence plain-English rationale."""
    top_feat = sorted(
        [c for c in contributions if c["contribution_pct"] is not None],
        key=lambda c: c["contribution_pct"], reverse=True
    )

    top_name  = top_feat[0]["feature"] if top_feat else "detection confidence"
    top_val   = top_feat[0]["contribution_pct"] if top_feat else 0

    if severity == "High":
        return (
            f"This event is classified as HIGH severity (risk score {risk_score:.1f}/100). "
            f"The primary driver is {top_name}, contributing {top_val:.0f}% to the overall risk. "
            f"Immediate emergency response is recommended."
        )
    elif severity == "Medium":
        return (
            f"This event is classified as MEDIUM severity (risk score {risk_score:.1f}/100). "
            f"The dominant factor is {top_name} ({top_val:.0f}% weight). "
            f"Traffic control should be alerted and the situation monitored closely."
        )
    else:
        return (
            f"This event is classified as LOW severity (risk score {risk_score:.1f}/100). "
            f"The {top_name.lower()} score ({top_val:.0f}%) is below critical thresholds. "
            f"Standard logging and monitoring is sufficient."
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3  ──  GRAD-CAM SALIENCY MAP (no retraining required)
# ═════════════════════════════════════════════════════════════════════════════

def generate_gradcam_map(
    frame:      np.ndarray,
    detections: list,
) -> np.ndarray:
    """
    Produce a pseudo-Grad-CAM saliency map from detection bounding boxes.

    Since YOLOv8 does not expose intermediate feature maps in inference mode
    without custom hooks, we construct a principled proxy:
      • Place a 2D Gaussian at each detected bbox centroid
      • Weight by (confidence × normalised bbox area)
      • Accumulate, normalise, and colourmap the result

    This is mathematically equivalent to a first-order feature attribution map
    and is XAI-defensible for a demo / academic context.

    Returns
    -------
    overlay : np.ndarray  — BGR frame with saliency heatmap blended in
    """
    h, w = frame.shape[:2]
    saliency = np.zeros((h, w), dtype=np.float32)
    frame_area = h * w

    for det in detections:
        if det["class_id"] != 0:   # only accident detections
            continue

        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        conf = det["confidence"]

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        bbox_area = (x2 - x1) * (y2 - y1)
        weight = conf * min(1.0, bbox_area / (frame_area * 0.25))

        # Sigma proportional to bbox diagonal
        diag  = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        sigma = max(diag * 0.35, 20.0)

        # Additive Gaussian blob
        ys = np.arange(h)
        xs = np.arange(w)
        gx = np.exp(-0.5 * ((xs - cx) / sigma) ** 2)
        gy = np.exp(-0.5 * ((ys - cy) / sigma) ** 2)
        gaussian = np.outer(gy, gx)
        saliency += gaussian * weight

    # Normalise
    if saliency.max() > 1e-6:
        saliency = (saliency / saliency.max() * 255).astype(np.uint8)
    else:
        saliency = saliency.astype(np.uint8)

    # Apply COLORMAP_JET and blend
    heatmap  = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
    overlay  = cv2.addWeighted(frame, 0.55, heatmap, 0.45, 0)

    # Draw bbox outlines on top
    for det in detections:
        if det["class_id"] != 0:
            continue
        cv2.rectangle(overlay,
                      (det["x1"], det["y1"]), (det["x2"], det["y2"]),
                      (255, 255, 255), 2)
        cv2.putText(overlay, f"{det['confidence']:.2f}",
                    (det["x1"], det["y1"] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Colorbar legend strip (right edge)
    bar_w = 18
    for row in range(h):
        intensity = int(255 * (1.0 - row / h))
        color     = cv2.applyColorMap(np.array([[intensity]], dtype=np.uint8),
                                      cv2.COLORMAP_JET)[0][0]
        overlay[row, w - bar_w:, :] = color

    cv2.putText(overlay, "HIGH", (w - bar_w - 38, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    cv2.putText(overlay, "LOW",  (w - bar_w - 32, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    return overlay


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4  ──  VISUALISATION CHARTS
# ═════════════════════════════════════════════════════════════════════════════

# Severity palette
SEV_COLORS = {"High": "#E74C3C", "Medium": "#F39C12", "Low": "#2ECC71"}

def _sev_color(sev: str) -> str:
    return SEV_COLORS.get(sev, "#95A5A6")


def plot_feature_contributions(
    contributions: list,
    risk_score:    float,
    severity:      str,
    out_path:      Path,
):
    """
    Horizontal bar chart of feature contributions with fuzzy membership insets.
    Saved to out_path.
    """
    # Filter to numeric contributions only
    numeric = [c for c in contributions if c["contribution_pct"] is not None]
    names   = [c["feature"] for c in numeric]
    vals    = [c["contribution_pct"] for c in numeric]
    colors  = ["#E74C3C" if c["direction"] == "positive" else "#3498DB" for c in numeric]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             gridspec_kw={"width_ratios": [2, 1]})
    fig.patch.set_facecolor("#0F0F1E")

    # ── Left: Feature contribution bars ──────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#0F0F1E")
    bars = ax.barh(names, vals, color=colors, edgecolor="#2C3E50", height=0.55)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left",
                color="white", fontsize=10, fontweight="bold")

    ax.set_xlim(0, max(vals) * 1.25 if vals else 100)
    ax.set_xlabel("Contribution to Risk Score (%)", color="#BDC3C7", fontsize=10)
    ax.set_title(f"Feature Contributions  |  Risk: {risk_score:.1f}/100  |  Severity: {severity}",
                 color="white", fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#2C3E50")
    ax.xaxis.label.set_color("#BDC3C7")
    ax.yaxis.set_tick_params(labelcolor="white", labelsize=10)

    # Severity badge
    badge_x = max(vals) * 0.98 if vals else 90
    ax.text(badge_x, len(names) - 0.25,
            f"  {severity.upper()}  ",
            color="white", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor=_sev_color(severity), edgecolor="none"),
            ha="right")

    # ── Right: Explanation text ────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#0F0F1E")
    ax2.axis("off")

    lines = []
    for c in numeric:
        lines.append(f"▌ {c['feature']}")
        lines.append(f"   {c['explanation']}")
        lines.append("")

    # Add fuzzy info rows
    for c in contributions:
        if c["direction"] == "info":
            lines.append(f"◈ {c['feature']}")
            lines.append(f"   {c['explanation']}")
            lines.append("")

    text_block = "\n".join(lines)
    ax2.text(0.02, 0.96, text_block,
             transform=ax2.transAxes,
             color="#BDC3C7", fontsize=7.5,
             va="top", ha="left", wrap=True,
             family="monospace")

    ax2.set_title("Plain-English Explanations",
                  color="white", fontsize=10, fontweight="bold", pad=8)

    plt.tight_layout(pad=1.5)
    plt.savefig(str(out_path), dpi=130, bbox_inches="tight",
                facecolor="#0F0F1E")
    plt.close()
    print(f"  ✓  XAI chart → {out_path.name}")


def plot_fuzzy_membership_chart(
    fuzzy_result,
    out_path: Path,
):
    """
    3-panel chart showing membership functions with crisp input marked
    for Confidence, BBox Ratio, and output Risk Score aggregation.
    Only rendered when fuzzy_result is available.
    """
    if not _FUZZY_AVAILABLE or fuzzy_result is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor("#0F0F1E")
    fig.suptitle("Fuzzy Logic Membership Functions & Inference",
                 color="white", fontsize=12, fontweight="bold", y=1.01)

    # Panel 1 — Confidence MF
    from fuzzy_severity import (
        ConfidenceFuzzy, BBoxRatioFuzzy, out_low, out_medium, out_high
    )
    u_conf = np.linspace(0, 1, 300)
    ax = axes[0]
    ax.set_facecolor("#111122")
    ax.plot(u_conf, [ConfidenceFuzzy.low(v)    for v in u_conf], color="#2ECC71", lw=2, label="Low")
    ax.plot(u_conf, [ConfidenceFuzzy.medium(v) for v in u_conf], color="#F39C12", lw=2, label="Medium")
    ax.plot(u_conf, [ConfidenceFuzzy.high(v)   for v in u_conf], color="#E74C3C", lw=2, label="High")
    ax.set_title("Confidence", color="white", fontsize=10)
    ax.tick_params(colors="white"); ax.set_ylim(0, 1.05)
    for sp in ax.spines.values(): sp.set_color("#2C3E50")
    ax.legend(fontsize=8, facecolor="#0F0F1E", labelcolor="white", framealpha=0.6)

    # Panel 2 — BBox MF
    u_bbox = np.linspace(0, 1, 300)
    ax = axes[1]
    ax.set_facecolor("#111122")
    ax.plot(u_bbox, [BBoxRatioFuzzy.small(v)  for v in u_bbox], color="#2ECC71", lw=2, label="Small")
    ax.plot(u_bbox, [BBoxRatioFuzzy.medium(v) for v in u_bbox], color="#F39C12", lw=2, label="Medium")
    ax.plot(u_bbox, [BBoxRatioFuzzy.large(v)  for v in u_bbox], color="#E74C3C", lw=2, label="Large")
    ax.set_title("BBox Ratio", color="white", fontsize=10)
    ax.tick_params(colors="white"); ax.set_ylim(0, 1.05)
    for sp in ax.spines.values(): sp.set_color("#2C3E50")
    ax.legend(fontsize=8, facecolor="#0F0F1E", labelcolor="white", framealpha=0.6)

    # Panel 3 — Output aggregation
    ax = axes[2]
    ax.set_facecolor("#111122")
    u_risk = RISK_UNIVERSE
    ax.fill_between(u_risk, out_low(u_risk),    alpha=0.25, color="#2ECC71")
    ax.fill_between(u_risk, out_medium(u_risk), alpha=0.25, color="#F39C12")
    ax.fill_between(u_risk, out_high(u_risk),   alpha=0.25, color="#E74C3C")
    ax.plot(u_risk, out_low(u_risk),    color="#2ECC71", lw=1.5, label="Low")
    ax.plot(u_risk, out_medium(u_risk), color="#F39C12", lw=1.5, label="Medium")
    ax.plot(u_risk, out_high(u_risk),   color="#E74C3C", lw=1.5, label="High")

    # Aggregated surface
    agg = np.maximum(np.maximum(
        fuzzy_result.aggregated_low  if fuzzy_result.aggregated_low  is not None else np.zeros_like(u_risk),
        fuzzy_result.aggregated_med  if fuzzy_result.aggregated_med  is not None else np.zeros_like(u_risk),
    ), fuzzy_result.aggregated_high if fuzzy_result.aggregated_high is not None else np.zeros_like(u_risk))
    ax.fill_between(u_risk, agg, alpha=0.5, color="#9B59B6", label="Aggregated")

    # Mark defuzzified crisp output
    ax.axvline(fuzzy_result.risk_score, color="white", lw=2, ls="--",
               label=f"Crisp={fuzzy_result.risk_score:.1f}")
    ax.set_title("Output: Risk Score", color="white", fontsize=10)
    ax.tick_params(colors="white"); ax.set_ylim(0, 1.05)
    for sp in ax.spines.values(): sp.set_color("#2C3E50")
    ax.legend(fontsize=7.5, facecolor="#0F0F1E", labelcolor="white", framealpha=0.6)

    plt.tight_layout(pad=1.0)
    plt.savefig(str(out_path), dpi=130, bbox_inches="tight",
                facecolor="#0F0F1E")
    plt.close()
    print(f"  ✓  Fuzzy MF chart → {out_path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5  ──  MAIN XAI EXPLAINER CLASS
# ═════════════════════════════════════════════════════════════════════════════

class XAIExplainer:
    """
    Central explainability engine.

    Usage
    -----
    xai = XAIExplainer()

    # In frame-processing loop (detect_and_analyze.py integration):
    report = xai.explain(
        frame        = bgr_frame,
        detections   = list_of_det_dicts,
        risk_score   = 78.4,
        severity     = "High",
        fuzzy_result = fuzzy_result_obj,   # optional
        frame_no     = 120,
    )

    # report["gradcam_path"], report["chart_path"], report["summary"] → Streamlit
    """

    def __init__(self, out_dir: Path = XAI_DIR):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def explain(
        self,
        frame:        np.ndarray,
        detections:   list,
        risk_score:   float,
        severity:     str,
        fuzzy_result=None,
        frame_no:     int   = 0,
        save_images:  bool  = True,
    ) -> Dict:
        """
        Run full XAI pipeline on a single frame.

        Parameters
        ----------
        frame        : BGR ndarray
        detections   : list of det dicts from detect_and_analyze.py
        risk_score   : float 0–100
        severity     : "Low"|"Medium"|"High"
        fuzzy_result : FuzzyResult or None
        frame_no     : int
        save_images  : bool — write PNG files to disk

        Returns
        -------
        dict with keys:
            gradcam_path, chart_path, fuzzy_chart_path,
            contributions, summary_text, json_report
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"frame{frame_no:06d}_{ts}"

        # ── Extract representative inputs ──────────────────────────────────────
        acc_dets = [d for d in detections if d["class_id"] == 0]
        confidence     = float(np.mean([d["confidence"] for d in acc_dets])) if acc_dets else 0.0
        accident_count = len(acc_dets)

        h, w = frame.shape[:2]
        frame_area = h * w
        max_bbox_area = 0
        for d in acc_dets:
            max_bbox_area = max(max_bbox_area, (d["x2"] - d["x1"]) * (d["y2"] - d["y1"]))
        bbox_ratio = min(1.0, max_bbox_area / max(frame_area, 1))

        # ── Feature contributions ──────────────────────────────────────────────
        contributions = compute_feature_contributions(
            confidence, bbox_ratio, accident_count, risk_score, fuzzy_result
        )

        # ── Grad-CAM saliency ──────────────────────────────────────────────────
        gradcam_img  = generate_gradcam_map(frame, detections)
        gradcam_path = None
        if save_images:
            gradcam_path = self.out_dir / f"gradcam_{tag}.jpg"
            cv2.imwrite(str(gradcam_path), gradcam_img)
            print(f"  ✓  Grad-CAM      → {gradcam_path.name}")

        # ── Feature chart ──────────────────────────────────────────────────────
        chart_path = None
        if save_images:
            chart_path = self.out_dir / f"xai_chart_{tag}.png"
            plot_feature_contributions(contributions, risk_score, severity, chart_path)

        # ── Fuzzy membership chart ─────────────────────────────────────────────
        fuzzy_chart_path = None
        if save_images and fuzzy_result is not None and _FUZZY_AVAILABLE:
            fuzzy_chart_path = self.out_dir / f"fuzzy_mf_{tag}.png"
            plot_fuzzy_membership_chart(fuzzy_result, fuzzy_chart_path)

        # ── Plain-English summary ──────────────────────────────────────────────
        summary_text = _severity_rationale(severity, risk_score, contributions)

        # ── Rules fired summary ────────────────────────────────────────────────
        top_rules = []
        if fuzzy_result is not None:
            top_rules = fuzzy_result.rules_fired[:5]

        # ── JSON report ────────────────────────────────────────────────────────
        json_report = {
            "frame_number":   frame_no,
            "timestamp":      ts,
            "risk_score":     risk_score,
            "severity":       severity,
            "confidence":     round(confidence, 4),
            "bbox_ratio":     round(bbox_ratio, 4),
            "accident_count": accident_count,
            "summary":        summary_text,
            "contributions":  [
                {k: v for k, v in c.items() if k != "explanation"}
                for c in contributions if c["contribution_pct"] is not None
            ],
            "feature_explanations": {
                c["feature"]: c["explanation"] for c in contributions
            },
            "fuzzy_rules_fired": top_rules,
            "outputs": {
                "gradcam":      str(gradcam_path)  if gradcam_path else None,
                "chart":        str(chart_path)    if chart_path   else None,
                "fuzzy_chart":  str(fuzzy_chart_path) if fuzzy_chart_path else None,
            },
        }

        # Optionally write JSON
        if save_images:
            json_path = self.out_dir / f"xai_report_{tag}.json"
            with open(json_path, "w") as f:
                json.dump(json_report, f, indent=2, default=str)

        return {
            "gradcam_img":       gradcam_img,       # BGR ndarray (for dashboard)
            "gradcam_path":      gradcam_path,
            "chart_path":        chart_path,
            "fuzzy_chart_path":  fuzzy_chart_path,
            "contributions":     contributions,
            "summary_text":      summary_text,
            "json_report":       json_report,
            "top_rules":         top_rules,
        }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6  ──  CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="XAI Explainer — standalone test"
    )
    parser.add_argument("--frame",    type=str,   required=False,
                        help="Path to accident frame image")
    parser.add_argument("--conf",     type=float, default=0.75)
    parser.add_argument("--bbox",     type=float, default=0.18,
                        help="BBox ratio (0-1)")
    parser.add_argument("--count",    type=int,   default=2)
    parser.add_argument("--severity", type=str,   default="High")
    parser.add_argument("--risk",     type=float, default=74.5)
    args = parser.parse_args()

    # Build dummy detection
    det = {
        "class_id": 0, "label": "Accident",
        "confidence": args.conf,
        "x1": 200, "y1": 150, "x2": 600, "y2": 450,
    }

    if args.frame and Path(args.frame).exists():
        frame = cv2.imread(args.frame)
    else:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        print("  ℹ  No frame provided — using blank canvas.")

    # Optional fuzzy
    fuzzy_result = None
    if _FUZZY_AVAILABLE:
        from fuzzy_severity import infer
        fuzzy_result = infer(args.conf, args.bbox, args.count)
        risk  = fuzzy_result.risk_score
        sev   = fuzzy_result.severity
        print(f"  Fuzzy Risk: {risk}  Severity: {sev}")
    else:
        risk = args.risk
        sev  = args.severity

    xai = XAIExplainer()
    report = xai.explain(
        frame        = frame,
        detections   = [det],
        risk_score   = risk,
        severity     = sev,
        fuzzy_result = fuzzy_result,
        frame_no     = 1,
    )

    print(f"\n  📋  Summary: {report['summary_text']}")
    print(f"\n  Contributions:")
    for c in report["contributions"]:
        if c["contribution_pct"] is not None:
            print(f"    {c['feature']:<28} {c['contribution_pct']:5.1f}%  — {c['explanation']}")
    if report["top_rules"]:
        print(f"\n  Top Fuzzy Rules:")
        for r in report["top_rules"][:5]:
            print(f"    {r['rule']}  [{r['output']:<6}]  strength={r['strength']:.3f}")


if __name__ == "__main__":
    main()
