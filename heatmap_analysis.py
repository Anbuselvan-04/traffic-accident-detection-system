"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   MODULE 2 — Accident Heatmap Analytics                                      ║
║   AI-Powered Traffic Accident Detection & Severity Analysis                  ║
║                                                                              ║
║   File    : heatmap_analysis.py                                              ║
║   Reads   : outputs/logs/*.csv   (from detect_and_analyze.py)               ║
║   Writes  : outputs/heatmaps/    (PNG images + summary report)              ║
║                                                                              ║
║   Outputs generated:                                                         ║
║     1. heatmap_overlay.png      — KDE heatmap on blank canvas               ║
║     2. heatmap_severity.png     — one panel per severity level               ║
║     3. heatmap_timeline.png     — hour-of-day accident frequency bar chart   ║
║     4. heatmap_summary.png      — combined 4-panel analytics board           ║
║     5. hotspot_report.csv       — top hotspot clusters with statistics       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python src/heatmap_analysis.py                          # auto-find latest log
    python src/heatmap_analysis.py --log outputs/logs/events_20240601.csv
    python src/heatmap_analysis.py --log outputs/logs/ --width 1920 --height 1080
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import cv2
import matplotlib
matplotlib.use("Agg")                          # headless backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 0  ──  PATHS & CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR     = BASE_DIR / "outputs" / "logs"
HEATMAP_DIR  = BASE_DIR / "outputs" / "heatmaps"
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

# Default canvas size — matches the source video resolution used during detection
DEFAULT_W, DEFAULT_H = 1920, 1080

# Kernel density estimation bandwidth (controls heatmap "spread")
KDE_BANDWIDTH    = 0.08          # relative to canvas size; tune per dataset
GAUSSIAN_SIGMA   = 28            # sigma for scipy gaussian_filter (pixels)
HOTSPOT_TOPN     = 5             # number of top hotspots to report
MIN_CLUSTER_DIST = 80            # px — min distance between independent hotspots

# Severity colour map (BGR for OpenCV, RGB for matplotlib)
SEV_COLORS_MPL = {
    "Low":    "#2ECC71",   # green
    "Medium": "#F39C12",   # amber
    "High":   "#E74C3C",   # red
}
SEV_MARKER_SIZE = {"Low": 20, "Medium": 50, "High": 120}

# Custom green→yellow→red colormap for heatmap
HEATMAP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "accident_heat",
    [
        (0.00, "#00000000"),   # transparent (no events)
        (0.10, "#00FF00"),     # green  (very low)
        (0.35, "#AAFF00"),     # lime
        (0.55, "#FFFF00"),     # yellow (moderate)
        (0.75, "#FF8800"),     # orange
        (1.00, "#FF0000"),     # red    (hotspot)
    ],
)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1  ──  DATA LOADING & VALIDATION
# ═════════════════════════════════════════════════════════════════════════════
def find_latest_log(logs_dir: Path) -> Path:
    """Return the most recently modified CSV in the logs directory."""
    csvs = sorted(logs_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    if not csvs:
        raise FileNotFoundError(f"No CSV log files found in: {logs_dir}")
    latest = csvs[-1]
    print(f"  Auto-selected log: {latest.name}")
    return latest


def load_logs(log_source: str) -> pd.DataFrame:
    """
    Load one CSV file or all CSVs in a directory.
    Validates required columns and cleans data types.

    Parameters
    ----------
    log_source : str   path to a .csv file OR a directory of .csv files

    Returns
    -------
    pd.DataFrame  with columns guaranteed:
        timestamp, frame_number, class_label, confidence,
        severity, risk_score, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
        center_x, center_y   ← computed here
    """
    p = Path(log_source)

    # Collect CSV paths
    if p.is_dir():
        csv_files = sorted(p.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files in directory: {p}")
        frames = [pd.read_csv(f) for f in csv_files]
        df     = pd.concat(frames, ignore_index=True)
        print(f"  Loaded {len(csv_files)} log file(s) → {len(df):,} total rows")
    else:
        df = pd.read_csv(p)
        print(f"  Loaded: {p.name}  →  {len(df):,} rows")

    # ── Required columns ────────────────────────────────────────────────────
    required = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                "severity", "risk_score", "confidence", "timestamp"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # ── Type coercion & cleaning ─────────────────────────────────────────────
    bbox_cols = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    df[bbox_cols]    = df[bbox_cols].apply(pd.to_numeric, errors="coerce")
    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["timestamp"]  = pd.to_datetime(df["timestamp"],  errors="coerce")
    df = df.dropna(subset=bbox_cols + ["risk_score"]).reset_index(drop=True)

    # ── Derived columns ──────────────────────────────────────────────────────
    df["center_x"] = ((df["bbox_x1"] + df["bbox_x2"]) / 2).astype(int)
    df["center_y"] = ((df["bbox_y1"] + df["bbox_y2"]) / 2).astype(int)
    df["bbox_area"] = (df["bbox_x2"] - df["bbox_x1"]) * (df["bbox_y2"] - df["bbox_y1"])
    df["hour"]      = df["timestamp"].dt.hour

    # Keep only accident rows (non-accident detections are not plotted on heatmap)
    acc_df = df[df.get("class_label", pd.Series(["Accident"] * len(df))) == "Accident"].copy()
    print(f"  Accident events : {len(acc_df):,}  "
          f"({len(acc_df)/max(len(df),1)*100:.1f}% of total detections)")

    return acc_df


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2  ──  HEATMAP DENSITY COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════
def compute_density_grid(
    cx: np.ndarray,
    cy: np.ndarray,
    canvas_w: int,
    canvas_h: int,
    sigma: float = GAUSSIAN_SIGMA,
    risk_weights: np.ndarray = None,
) -> np.ndarray:
    """
    Place Gaussian blobs for each accident coordinate on a 2D grid.
    Weighted by risk_score so high-risk events create brighter peaks.

    Parameters
    ----------
    cx, cy         : accident centre coordinates (arrays)
    canvas_w/h     : output canvas dimensions (px)
    sigma          : Gaussian blur spread (pixels)
    risk_weights   : per-event weight (uses risk_score; defaults to ones)

    Returns
    -------
    density : (canvas_h, canvas_w) float32 array, values in [0, 1]
    """
    density = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    if len(cx) == 0:
        return density

    weights = risk_weights if risk_weights is not None else np.ones(len(cx))

    # Clamp coordinates to canvas
    cx_c = np.clip(cx, 0, canvas_w - 1).astype(int)
    cy_c = np.clip(cy, 0, canvas_h - 1).astype(int)

    # Accumulate weighted points
    for x, y, w in zip(cx_c, cy_c, weights):
        density[y, x] += w

    # Gaussian blur → smooth density surface
    density = gaussian_filter(density, sigma=sigma)

    # Normalize to [0, 1]
    if density.max() > 0:
        density /= density.max()

    return density


def density_to_color_image(
    density: np.ndarray,
    cmap=HEATMAP_CMAP,
    alpha_boost: float = 1.4,
) -> np.ndarray:
    """
    Map a normalised density grid to an RGBA colour image using the
    custom green→yellow→red colormap.

    Returns uint8 BGR image (for OpenCV overlay use).
    """
    # Apply colormap (returns RGBA float [0,1])
    rgba = cmap(density)

    # Boost alpha so even mid-density areas are visible
    rgba[..., 3] = np.clip(rgba[..., 3] * alpha_boost, 0, 1)

    # Convert to uint8 RGB
    rgb  = (rgba[..., :3] * 255).astype(np.uint8)
    bgr  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3  ──  HOTSPOT DETECTION  (local maxima clustering)
# ═════════════════════════════════════════════════════════════════════════════
def find_hotspots(
    density: np.ndarray,
    df: pd.DataFrame,
    canvas_w: int,
    canvas_h: int,
    top_n: int = HOTSPOT_TOPN,
    min_dist: int = MIN_CLUSTER_DIST,
) -> list:
    """
    Find the top-N local maxima in the density map.
    For each hotspot, computes:
        - pixel location (cx, cy)
        - peak density value
        - number of accident events within `min_dist` radius
        - mean / max risk score of events in that cluster
        - dominant severity level

    Returns
    -------
    list of dicts, sorted by density descending
    """
    hotspots = []
    temp     = density.copy()

    for rank in range(top_n):
        if temp.max() < 0.01:
            break

        # Find current maximum
        flat_idx = np.argmax(temp)
        peak_y, peak_x = np.unravel_index(flat_idx, temp.shape)
        peak_val = temp[peak_y, peak_x]

        # Events within min_dist of this peak
        dist = np.sqrt((df["center_x"] - peak_x) ** 2 +
                       (df["center_y"] - peak_y) ** 2)
        nearby = df[dist <= min_dist]

        sev_counts = nearby["severity"].value_counts()
        dominant   = sev_counts.index[0] if len(sev_counts) else "Low"

        hotspots.append({
            "rank":          rank + 1,
            "cx":            int(peak_x),
            "cy":            int(peak_y),
            "peak_density":  round(float(peak_val), 4),
            "event_count":   len(nearby),
            "mean_risk":     round(nearby["risk_score"].mean(), 1) if len(nearby) else 0,
            "max_risk":      round(nearby["risk_score"].max(), 1)  if len(nearby) else 0,
            "mean_conf":     round(nearby["confidence"].mean(), 3) if len(nearby) else 0,
            "dominant_sev":  dominant,
        })

        # Suppress this region so next iteration finds a different peak
        y0 = max(0, peak_y - min_dist)
        y1 = min(canvas_h, peak_y + min_dist)
        x0 = max(0, peak_x - min_dist)
        x1 = min(canvas_w, peak_x + min_dist)
        temp[y0:y1, x0:x1] = 0

    return hotspots


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4  ──  INDIVIDUAL HEATMAP PLOTS
# ═════════════════════════════════════════════════════════════════════════════
def plot_main_heatmap(
    density: np.ndarray,
    df: pd.DataFrame,
    hotspots: list,
    canvas_w: int,
    canvas_h: int,
    out_path: Path,
):
    """
    Render the primary KDE heatmap:
        - Smooth density surface (green→yellow→red)
        - Hotspot markers with rank labels
        - Colourbar legend
        - Axis ticks in pixel units
    """
    fig, ax = plt.subplots(figsize=(14, 8), facecolor="#0A0A14")
    ax.set_facecolor("#0A0A14")

    # ── Density surface ──────────────────────────────────────────────────────
    im = ax.imshow(
        density,
        cmap   = HEATMAP_CMAP,
        origin = "upper",
        extent = [0, canvas_w, canvas_h, 0],
        aspect = "auto",
        alpha  = 0.92,
        vmin   = 0,
        vmax   = 1,
    )

    # ── Raw accident scatter (tiny dots, semi-transparent) ────────────────────
    ax.scatter(
        df["center_x"], df["center_y"],
        c       = [SEV_COLORS_MPL.get(s, "#FFFFFF") for s in df["severity"]],
        s       = [SEV_MARKER_SIZE.get(s, 20) for s in df["severity"]],
        alpha   = 0.30,
        edgecolors = "none",
        zorder  = 3,
    )

    # ── Hotspot rings & rank labels ──────────────────────────────────────────
    for hs in hotspots:
        cx, cy = hs["cx"], hs["cy"]

        # Outer glow ring
        circle_outer = plt.Circle((cx, cy), MIN_CLUSTER_DIST * 0.85,
                                   color="white", fill=False,
                                   linewidth=1.4, linestyle="--", alpha=0.55,
                                   zorder=4)
        ax.add_patch(circle_outer)

        # Inner filled circle
        circle_inner = plt.Circle((cx, cy), 8,
                                   color="#FFFFFF", fill=True,
                                   linewidth=2, alpha=0.9, zorder=5)
        ax.add_patch(circle_inner)

        # Rank label
        ax.text(cx + 14, cy - 14,
                f"#{hs['rank']}  {hs['dominant_sev']}\n"
                f"Events: {hs['event_count']}  Risk: {hs['mean_risk']}",
                color="white", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="#1A1A2E",
                          ec="white", alpha=0.75, linewidth=0.8),
                zorder=6)

    # ── Colourbar ────────────────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.01,
                        label="Accident Frequency Density")
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["None", "Low", "Medium", "High", "Critical"])

    # ── Severity legend ───────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=SEV_COLORS_MPL[s], label=f"{s} Severity")
        for s in ["Low", "Medium", "High"]
    ]
    ax.legend(handles=legend_patches, loc="lower left",
              facecolor="#1A1A2E", edgecolor="white",
              labelcolor="white", fontsize=9, framealpha=0.8)

    # ── Labels & styling ─────────────────────────────────────────────────────
    ax.set_title(
        "🔥  Accident Hotspot Heatmap  —  Frequency Density",
        color="white", fontsize=15, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Frame X (pixels)", color="#AAAAAA", fontsize=10)
    ax.set_ylabel("Frame Y (pixels)", color="#AAAAAA", fontsize=10)
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓  Main heatmap     → {out_path.name}")


def plot_severity_panels(
    df: pd.DataFrame,
    canvas_w: int,
    canvas_h: int,
    out_path: Path,
):
    """
    Three side-by-side heatmaps — one per severity level (Low / Medium / High).
    Lets you compare which zones have which kind of accidents.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0A0A14")
    fig.suptitle("Accident Heatmap by Severity Level",
                 color="white", fontsize=14, fontweight="bold", y=1.01)

    for ax, sev in zip(axes, ["Low", "Medium", "High"]):
        sub = df[df["severity"] == sev]
        ax.set_facecolor("#0A0A14")

        if len(sub) > 0:
            den = compute_density_grid(
                sub["center_x"].values,
                sub["center_y"].values,
                canvas_w, canvas_h,
                sigma          = GAUSSIAN_SIGMA,
                risk_weights   = sub["risk_score"].values,
            )
            # Use severity-specific single-colour gradient
            base_color = mcolors.to_rgba(SEV_COLORS_MPL[sev])
            cmap_sev   = mcolors.LinearSegmentedColormap.from_list(
                f"sev_{sev}",
                [(0, (0, 0, 0, 0)),
                 (1, base_color)],
            )
            ax.imshow(den, cmap=cmap_sev, origin="upper",
                      extent=[0, canvas_w, canvas_h, 0], aspect="auto")
            ax.scatter(sub["center_x"], sub["center_y"],
                       color=SEV_COLORS_MPL[sev], s=12, alpha=0.4, edgecolors="none")

        ax.set_title(f"{sev} Severity  ({len(sub)} events)",
                     color=SEV_COLORS_MPL[sev], fontsize=11, fontweight="bold")
        ax.set_xlabel("X", color="#888888", fontsize=9)
        ax.set_ylabel("Y", color="#888888", fontsize=9)
        ax.tick_params(colors="#666666")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333355")

    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="#0A0A14")
    plt.close(fig)
    print(f"  ✓  Severity panels  → {out_path.name}")


def plot_timeline_chart(df: pd.DataFrame, out_path: Path):
    """
    Horizontal bar chart showing accident count by hour of day.
    Highlights peak hours in red and safe hours in green.
    """
    if "hour" not in df.columns or df["hour"].isna().all():
        print("  ⚠  No timestamp data — skipping timeline chart.")
        return

    hourly = df.groupby("hour").agg(
        count      = ("risk_score", "count"),
        mean_risk  = ("risk_score", "mean"),
    ).reindex(range(24), fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 7), facecolor="#0A0A14")
    ax.set_facecolor("#0D0D1A")

    # Colour each bar by mean risk
    bar_colors = []
    for _, row in hourly.iterrows():
        if row["mean_risk"] >= 70:   bar_colors.append("#E74C3C")
        elif row["mean_risk"] >= 35: bar_colors.append("#F39C12")
        else:                        bar_colors.append("#2ECC71")

    bars = ax.barh(hourly.index, hourly["count"], color=bar_colors,
                   edgecolor="#1A1A2E", linewidth=0.5, height=0.7)

    # Count labels
    for bar, cnt in zip(bars, hourly["count"]):
        if cnt > 0:
            ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                    str(int(cnt)), va="center", color="white", fontsize=8)

    ax.set_yticks(range(24))
    ax.set_yticklabels(
        [f"{h:02d}:00" for h in range(24)], fontsize=8, color="#AAAAAA"
    )
    ax.set_xlabel("Number of Accident Events", color="#AAAAAA", fontsize=10)
    ax.set_title("⏰  Accident Frequency by Hour of Day",
                 color="white", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(axis="x", colors="#888888")
    ax.invert_yaxis()   # 00:00 at top

    for sp in ax.spines.values():
        sp.set_edgecolor("#333355")

    # Peak annotation
    peak_hour = hourly["count"].idxmax()
    peak_cnt  = hourly["count"].max()
    ax.annotate(f"Peak: {peak_hour:02d}:00  ({int(peak_cnt)} events)",
                xy=(peak_cnt, peak_hour),
                xytext=(peak_cnt + 2, peak_hour - 2),
                color="white", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#AAAAAA", lw=1.2))

    # Legend
    patches = [
        mpatches.Patch(color="#2ECC71", label="Low Risk Hour"),
        mpatches.Patch(color="#F39C12", label="Medium Risk Hour"),
        mpatches.Patch(color="#E74C3C", label="High Risk Hour"),
    ]
    ax.legend(handles=patches, loc="lower right",
              facecolor="#1A1A2E", edgecolor="white",
              labelcolor="white", fontsize=8, framealpha=0.8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="#0A0A14")
    plt.close(fig)
    print(f"  ✓  Timeline chart   → {out_path.name}")


def plot_summary_dashboard(
    df: pd.DataFrame,
    density: np.ndarray,
    hotspots: list,
    canvas_w: int,
    canvas_h: int,
    out_path: Path,
):
    """
    4-panel combined analytics dashboard:
        [0,0] Main heatmap with hotspot markers
        [0,1] Severity distribution pie chart
        [1,0] Hour-of-day accident frequency
        [1,1] Risk score distribution histogram
    """
    fig = plt.figure(figsize=(18, 11), facecolor="#0A0A14")
    gs  = GridSpec(2, 2, figure=fig,
                   hspace=0.38, wspace=0.28,
                   left=0.05, right=0.96,
                   top=0.92, bottom=0.06)

    fig.suptitle(
        "🚦  AI Traffic Accident Analytics Dashboard  —  Heatmap Report",
        color="white", fontsize=16, fontweight="bold",
    )

    # ── Panel A: Main heatmap ─────────────────────────────────────────────────
    ax_map = fig.add_subplot(gs[0, 0])
    ax_map.set_facecolor("#0D0D1A")
    ax_map.imshow(density, cmap=HEATMAP_CMAP, origin="upper",
                  extent=[0, canvas_w, canvas_h, 0], aspect="auto")
    ax_map.scatter(df["center_x"], df["center_y"],
                   c=[SEV_COLORS_MPL.get(s,"#FFF") for s in df["severity"]],
                   s=8, alpha=0.35, edgecolors="none", zorder=3)
    for hs in hotspots[:3]:
        c = plt.Circle((hs["cx"], hs["cy"]), 30, color="white",
                        fill=False, linewidth=1.5, linestyle="--", alpha=0.7)
        ax_map.add_patch(c)
        ax_map.text(hs["cx"] + 14, hs["cy"] - 14, f"#{hs['rank']}",
                    color="white", fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round", fc="#1A1A2E", ec="white",
                              alpha=0.75, linewidth=0.7))
    ax_map.set_title("Accident Density Heatmap", color="white",
                     fontsize=11, fontweight="bold")
    ax_map.tick_params(colors="#666666")
    for sp in ax_map.spines.values(): sp.set_edgecolor("#333355")

    # ── Panel B: Severity pie ─────────────────────────────────────────────────
    ax_pie = fig.add_subplot(gs[0, 1])
    ax_pie.set_facecolor("#0A0A14")
    sev_counts = df["severity"].value_counts().reindex(
        ["Low", "Medium", "High"], fill_value=0
    )
    colors_pie = [SEV_COLORS_MPL[s] for s in sev_counts.index]

    wedges, texts, autotexts = ax_pie.pie(
        sev_counts.values,
        labels      = [f"{s}\n({v})" for s, v in zip(sev_counts.index, sev_counts.values)],
        colors      = colors_pie,
        autopct     = "%1.1f%%",
        startangle  = 140,
        pctdistance = 0.78,
        wedgeprops  = dict(edgecolor="#0A0A14", linewidth=2),
        textprops   = dict(color="white", fontsize=9),
    )
    for at in autotexts:
        at.set_color("#0A0A14")
        at.set_fontweight("bold")
        at.set_fontsize(8)

    # Donut style
    centre_circle = plt.Circle((0, 0), 0.55, fc="#0A0A14")
    ax_pie.add_patch(centre_circle)
    ax_pie.text(0, 0, f"{len(df)}\nEvents",
                ha="center", va="center", color="white",
                fontsize=11, fontweight="bold")
    ax_pie.set_title("Severity Distribution", color="white",
                     fontsize=11, fontweight="bold")

    # ── Panel C: Hourly bar ───────────────────────────────────────────────────
    ax_time = fig.add_subplot(gs[1, 0])
    ax_time.set_facecolor("#0D0D1A")
    if "hour" in df.columns:
        hourly = df.groupby("hour")["risk_score"].count().reindex(range(24), fill_value=0)
        bar_cols = ["#E74C3C" if hourly.get(h, 0) == hourly.max()
                    else "#F39C12" if hourly.get(h, 0) >= hourly.quantile(0.75)
                    else "#2ECC71"
                    for h in range(24)]
        ax_time.bar(range(24), hourly.values, color=bar_cols,
                    edgecolor="#1A1A2E", linewidth=0.4, width=0.75)
        ax_time.set_xticks(range(24))
        ax_time.set_xticklabels(
            [f"{h}" for h in range(24)], color="#888888", fontsize=7
        )
        ax_time.set_xlabel("Hour of Day", color="#AAAAAA", fontsize=9)
        ax_time.set_ylabel("Accidents", color="#AAAAAA", fontsize=9)
    ax_time.set_title("Accidents by Hour of Day", color="white",
                      fontsize=11, fontweight="bold")
    ax_time.tick_params(axis="y", colors="#888888")
    for sp in ax_time.spines.values(): sp.set_edgecolor("#333355")

    # ── Panel D: Risk score histogram ─────────────────────────────────────────
    ax_hist = fig.add_subplot(gs[1, 1])
    ax_hist.set_facecolor("#0D0D1A")
    risk_vals = df["risk_score"].dropna()
    n, bins, patches_h = ax_hist.hist(
        risk_vals, bins=20, edgecolor="#0A0A14", linewidth=0.5, color="#888888"
    )
    # Colour bars by severity zone
    for patch, left in zip(patches_h, bins[:-1]):
        if left < 35:    patch.set_facecolor(SEV_COLORS_MPL["Low"])
        elif left < 70:  patch.set_facecolor(SEV_COLORS_MPL["Medium"])
        else:            patch.set_facecolor(SEV_COLORS_MPL["High"])

    ax_hist.axvline(risk_vals.mean(), color="white", linestyle="--",
                    linewidth=1.2, label=f"Mean: {risk_vals.mean():.1f}")
    ax_hist.axvline(risk_vals.median(), color="#FFD700", linestyle=":",
                    linewidth=1.2, label=f"Median: {risk_vals.median():.1f}")
    ax_hist.legend(facecolor="#1A1A2E", edgecolor="white",
                   labelcolor="white", fontsize=8)
    ax_hist.set_xlabel("Risk Score (0–100)", color="#AAAAAA", fontsize=9)
    ax_hist.set_ylabel("Frequency",          color="#AAAAAA", fontsize=9)
    ax_hist.set_title("Risk Score Distribution", color="white",
                      fontsize=11, fontweight="bold")
    ax_hist.tick_params(colors="#888888")
    for sp in ax_hist.spines.values(): sp.set_edgecolor("#333355")

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0A0A14")
    plt.close(fig)
    print(f"  ✓  Dashboard        → {out_path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5  ──  FREQUENCY SUMMARY & CONSOLE REPORT
# ═════════════════════════════════════════════════════════════════════════════
def compute_frequency_summary(df: pd.DataFrame) -> dict:
    """
    Compute aggregated statistics across all loaded log events.

    Returns dict with top-level KPIs and per-severity breakdown.
    """
    total = len(df)
    if total == 0:
        return {"total": 0}

    sev_breakdown = {}
    for sev in ["Low", "Medium", "High"]:
        sub = df[df["severity"] == sev]
        sev_breakdown[sev] = {
            "count":     len(sub),
            "pct":       round(len(sub) / total * 100, 1),
            "mean_risk": round(sub["risk_score"].mean(), 1) if len(sub) else 0,
            "max_risk":  round(sub["risk_score"].max(), 1)  if len(sub) else 0,
            "mean_conf": round(sub["confidence"].mean(), 3) if len(sub) else 0,
        }

    peak_hour = (df.groupby("hour")["risk_score"].count().idxmax()
                 if "hour" in df.columns else None)

    return {
        "total_events":    total,
        "mean_risk":       round(df["risk_score"].mean(), 1),
        "max_risk":        round(df["risk_score"].max(), 1),
        "min_risk":        round(df["risk_score"].min(), 1),
        "std_risk":        round(df["risk_score"].std(), 1),
        "mean_confidence": round(df["confidence"].mean(), 3),
        "peak_hour":       int(peak_hour) if peak_hour is not None else None,
        "date_range":      {
            "start": str(df["timestamp"].min()),
            "end":   str(df["timestamp"].max()),
        },
        "severity_breakdown": sev_breakdown,
    }


def save_hotspot_report(hotspots: list, out_path: Path):
    """Write top-N hotspot table to CSV."""
    if not hotspots:
        return
    pd.DataFrame(hotspots).to_csv(out_path, index=False)
    print(f"  ✓  Hotspot report   → {out_path.name}")


def print_console_report(summary: dict, hotspots: list):
    """Print a rich formatted analytics report to stdout."""
    SEP = "═" * 64

    print(f"\n{SEP}")
    print("  📊  ACCIDENT HEATMAP ANALYTICS REPORT")
    print(SEP)

    total = summary.get("total_events", 0)
    print(f"  Total Accident Events : {total:,}")
    print(f"  Risk Score  — Mean    : {summary.get('mean_risk', 0):.1f}")
    print(f"               Max      : {summary.get('max_risk', 0):.1f}")
    print(f"               Min      : {summary.get('min_risk', 0):.1f}")
    print(f"               Std Dev  : {summary.get('std_risk', 0):.1f}")
    print(f"  Mean Confidence       : {summary.get('mean_confidence', 0):.3f}")

    ph = summary.get("peak_hour")
    if ph is not None:
        print(f"  Peak Accident Hour    : {ph:02d}:00 – {ph+1:02d}:00")

    dr = summary.get("date_range", {})
    print(f"  Date Range            : {dr.get('start','—')}  →  {dr.get('end','—')}")

    print(f"\n  {'Severity':<10}  {'Count':>6}  {'%':>6}  {'Mean Risk':>10}  {'Max Risk':>9}")
    print(f"  {'─'*10}  {'─'*6}  {'─'*6}  {'─'*10}  {'─'*9}")
    for sev in ["High", "Medium", "Low"]:
        s = summary.get("severity_breakdown", {}).get(sev, {})
        bar = "▓" * int(s.get("pct", 0) / 5)
        print(f"  {sev:<10}  {s.get('count',0):>6,}  "
              f"{s.get('pct',0):>5.1f}%  "
              f"{s.get('mean_risk',0):>10.1f}  "
              f"{s.get('max_risk',0):>9.1f}  {bar}")

    print(f"\n  🔥  TOP ACCIDENT HOTSPOTS")
    print(f"  {'Rank':<5}  {'Location':^18}  {'Events':>7}  {'Mean Risk':>10}  {'Severity':>10}")
    print(f"  {'─'*5}  {'─'*18}  {'─'*7}  {'─'*10}  {'─'*10}")
    for hs in hotspots:
        loc = f"({hs['cx']}, {hs['cy']})"
        print(f"  #{hs['rank']:<4}  {loc:^18}  {hs['event_count']:>7}  "
              f"{hs['mean_risk']:>10.1f}  {hs['dominant_sev']:>10}")

    print(SEP)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6  ──  MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════
def run(
    log_source: str,
    canvas_w: int = DEFAULT_W,
    canvas_h: int = DEFAULT_H,
):
    """
    Full pipeline:
        load → density → hotspots → plots → report

    Parameters
    ----------
    log_source : str    path to .csv file or directory of .csv files
    canvas_w/h : int    pixel dimensions of original video (for coordinate mapping)
    """
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "═" * 64)
    print("  🗺   Accident Heatmap Analytics  —  Starting")
    print("═" * 64)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("\n[1/5] Loading accident logs …")
    df = load_logs(log_source)

    if len(df) == 0:
        print("  ✗  No accident events found in logs. Exiting.")
        return

    # ── 2. Compute density grid ───────────────────────────────────────────────
    print("\n[2/5] Computing density grid …")
    density = compute_density_grid(
        df["center_x"].values,
        df["center_y"].values,
        canvas_w, canvas_h,
        sigma        = GAUSSIAN_SIGMA,
        risk_weights = df["risk_score"].values,
    )
    print(f"       Grid shape  : {density.shape}")
    print(f"       Peak density: {density.max():.4f}")

    # ── 3. Detect hotspots ────────────────────────────────────────────────────
    print("\n[3/5] Detecting hotspot clusters …")
    hotspots = find_hotspots(density, df, canvas_w, canvas_h,
                             top_n=HOTSPOT_TOPN, min_dist=MIN_CLUSTER_DIST)
    print(f"       Found {len(hotspots)} hotspot(s)")

    # ── 4. Generate plots ─────────────────────────────────────────────────────
    print("\n[4/5] Generating visualisations …")
    plot_main_heatmap(
        density, df, hotspots, canvas_w, canvas_h,
        HEATMAP_DIR / f"heatmap_main_{ts_tag}.png"
    )
    plot_severity_panels(
        df, canvas_w, canvas_h,
        HEATMAP_DIR / f"heatmap_severity_{ts_tag}.png"
    )
    plot_timeline_chart(
        df,
        HEATMAP_DIR / f"heatmap_timeline_{ts_tag}.png"
    )
    plot_summary_dashboard(
        df, density, hotspots, canvas_w, canvas_h,
        HEATMAP_DIR / f"heatmap_dashboard_{ts_tag}.png"
    )

    # ── 5. Reports ────────────────────────────────────────────────────────────
    print("\n[5/5] Building reports …")
    summary = compute_frequency_summary(df)
    save_hotspot_report(
        hotspots,
        HEATMAP_DIR / f"hotspot_report_{ts_tag}.csv"
    )
    print_console_report(summary, hotspots)

    print(f"\n  All outputs saved → {HEATMAP_DIR}\n")
    return summary, hotspots


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7  ──  CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="Accident Heatmap Analytics — Module 2"
    )
    parser.add_argument(
        "--log", type=str,
        default=str(LOGS_DIR),
        help="Path to a single CSV log file OR a directory of logs "
             f"(default: {LOGS_DIR})",
    )
    parser.add_argument(
        "--width",  type=int, default=DEFAULT_W,
        help=f"Video frame width in pixels  (default: {DEFAULT_W})"
    )
    parser.add_argument(
        "--height", type=int, default=DEFAULT_H,
        help=f"Video frame height in pixels (default: {DEFAULT_H})"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # If the log argument is a directory, auto-pick the latest file
    log_path = Path(args.log)
    if log_path.is_dir():
        log_source = str(find_latest_log(log_path))
    else:
        if not log_path.exists():
            print(f"  ✗  Log not found: {log_path}")
            sys.exit(1)
        log_source = str(log_path)

    run(log_source, canvas_w=args.width, canvas_h=args.height)