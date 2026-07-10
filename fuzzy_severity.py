"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   MODULE 3 — Fuzzy Logic Severity Engine                                     ║
║   AI-Powered Traffic Accident Detection & Severity Analysis                  ║
║                                                                              ║
║   File    : fuzzy_severity.py                                                ║
║   Purpose : Replace the hard-threshold severity classifier in                ║
║             detect_and_analyze.py with a smooth, interpretable Fuzzy        ║
║             Inference System (Mamdani-style).                               ║
║                                                                              ║
║   Inputs  : confidence  (0–1)                                                ║
║             bbox_ratio  (0–1)   largest bbox area / frame area              ║
║             accident_count (int) number of accident detections              ║
║                                                                              ║
║   Outputs : risk_score  (0–100)   crisp score via centroid defuzzification  ║
║             severity    "Low" | "Medium" | "High"                            ║
║             membership  dict  — μ values for explainability                 ║
║             rules_fired list  — which fuzzy rules fired and at what strength ║
╚══════════════════════════════════════════════════════════════════════════════╝

Standalone usage:
    python src/fuzzy_severity.py --conf 0.72 --bbox 0.18 --count 2
    python src/fuzzy_severity.py --demo       # run a sweep of sample inputs
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1  ──  MEMBERSHIP FUNCTION PRIMITIVES
# ═════════════════════════════════════════════════════════════════════════════

def trimf(x: float, a: float, b: float, c: float) -> float:
    """
    Triangular membership function.
    μ = 0 for x ≤ a or x ≥ c
    μ = 1 at x = b (peak)
    """
    if x <= a or x >= c:
        return 0.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    """
    Trapezoidal membership function.
    μ = 0   for x ≤ a or x ≥ d
    μ = 1   for b ≤ x ≤ c  (flat top)
    μ ramps between corners
    """
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)


def sigmf(x: float, a: float, c: float) -> float:
    """
    Sigmoid membership function — smooth one-sided ramp.
    a : slope (positive → rising, negative → falling)
    c : crossover point (μ = 0.5)
    """
    return 1.0 / (1.0 + math.exp(-a * (x - c)))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2  ──  INPUT FUZZY SETS
# ═════════════════════════════════════════════════════════════════════════════

class ConfidenceFuzzy:
    """
    Universe: [0, 1]  — YOLO detection confidence

    Sets:
      Low    : [0, 0, 0.4, 0.55]    (trap)
      Medium : [0.35, 0.55, 0.70]   (tri)
      High   : [0.60, 0.75, 1, 1]   (trap)
    """
    @staticmethod
    def low(x: float)    -> float: return trapmf(x, 0.0, 0.0, 0.40, 0.55)

    @staticmethod
    def medium(x: float) -> float: return trimf (x, 0.35, 0.55, 0.75)

    @staticmethod
    def high(x: float)   -> float: return trapmf(x, 0.60, 0.75, 1.0, 1.0)

    @classmethod
    def memberships(cls, x: float) -> Dict[str, float]:
        return {
            "conf_low":    cls.low(x),
            "conf_medium": cls.medium(x),
            "conf_high":   cls.high(x),
        }


class BBoxRatioFuzzy:
    """
    Universe: [0, 1]  — max bbox area / frame area

    Sets:
      Small  : [0, 0, 0.08, 0.15]   very small bbox (minor incident)
      Medium : [0.08, 0.18, 0.30]   moderate coverage
      Large  : [0.22, 0.40, 1, 1]   large coverage (severe crash)
    """
    @staticmethod
    def small(x: float)  -> float: return trapmf(x, 0.00, 0.00, 0.08, 0.15)

    @staticmethod
    def medium(x: float) -> float: return trimf (x, 0.08, 0.18, 0.32)

    @staticmethod
    def large(x: float)  -> float: return trapmf(x, 0.22, 0.40, 1.00, 1.00)

    @classmethod
    def memberships(cls, x: float) -> Dict[str, float]:
        return {
            "bbox_small":  cls.small(x),
            "bbox_medium": cls.medium(x),
            "bbox_large":  cls.large(x),
        }


class CountFuzzy:
    """
    Universe: [0, 10]  — number of accident detections

    Sets:
      Few    : [0, 1]      1 detection
      Some   : [1, 3]      2–3 detections
      Many   : [3, 10]     4+ detections
    """
    @staticmethod
    def few(x: float)    -> float: return trapmf(x, 0.0, 0.0, 1.0, 2.0)

    @staticmethod
    def some(x: float)   -> float: return trimf (x, 1.0, 2.5, 4.0)

    @staticmethod
    def many(x: float)   -> float: return trapmf(x, 3.0, 5.0, 10.0, 10.0)

    @classmethod
    def memberships(cls, x: float) -> Dict[str, float]:
        return {
            "count_few":  cls.few(x),
            "count_some": cls.some(x),
            "count_many": cls.many(x),
        }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3  ──  OUTPUT FUZZY SETS  (Risk Score)
# ═════════════════════════════════════════════════════════════════════════════

# Output universe: 0–100 (risk score)
RISK_UNIVERSE = np.linspace(0, 100, 500)

def out_low(x: np.ndarray)    -> np.ndarray:
    """Low risk: [0, 0, 25, 40]"""
    return np.vectorize(lambda v: trapmf(v, 0, 0, 25, 40))(x)

def out_medium(x: np.ndarray) -> np.ndarray:
    """Medium risk: [25, 45, 65]"""
    return np.vectorize(lambda v: trimf(v, 25, 45, 65))(x)

def out_high(x: np.ndarray)   -> np.ndarray:
    """High risk: [55, 75, 100, 100]"""
    return np.vectorize(lambda v: trapmf(v, 55, 75, 100, 100))(x)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4  ──  RULE BASE  (27 Mamdani rules)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class FuzzyRule:
    """A single IF-THEN fuzzy rule."""
    name:           str
    conf_set:       str        # "low" | "medium" | "high"
    bbox_set:       str        # "small" | "medium" | "large"
    count_set:      str        # "few" | "some" | "many"
    output_set:     str        # "low" | "medium" | "high"
    weight:         float = 1.0


# Full 3×3×3 rule table — designed by traffic safety heuristics
RULE_BASE: List[FuzzyRule] = [
    # ── Low confidence detections ──────────────────────────────────────────────
    FuzzyRule("R01", "low",    "small",  "few",  "low",    weight=1.0),
    FuzzyRule("R02", "low",    "small",  "some", "low",    weight=1.0),
    FuzzyRule("R03", "low",    "small",  "many", "medium", weight=0.9),
    FuzzyRule("R04", "low",    "medium", "few",  "low",    weight=1.0),
    FuzzyRule("R05", "low",    "medium", "some", "medium", weight=0.9),
    FuzzyRule("R06", "low",    "medium", "many", "medium", weight=0.9),
    FuzzyRule("R07", "low",    "large",  "few",  "medium", weight=0.9),
    FuzzyRule("R08", "low",    "large",  "some", "medium", weight=0.9),
    FuzzyRule("R09", "low",    "large",  "many", "high",   weight=0.8),

    # ── Medium confidence detections ───────────────────────────────────────────
    FuzzyRule("R10", "medium", "small",  "few",  "low",    weight=1.0),
    FuzzyRule("R11", "medium", "small",  "some", "medium", weight=1.0),
    FuzzyRule("R12", "medium", "small",  "many", "medium", weight=1.0),
    FuzzyRule("R13", "medium", "medium", "few",  "medium", weight=1.0),
    FuzzyRule("R14", "medium", "medium", "some", "medium", weight=1.0),
    FuzzyRule("R15", "medium", "medium", "many", "high",   weight=1.0),
    FuzzyRule("R16", "medium", "large",  "few",  "medium", weight=1.0),
    FuzzyRule("R17", "medium", "large",  "some", "high",   weight=1.0),
    FuzzyRule("R18", "medium", "large",  "many", "high",   weight=1.0),

    # ── High confidence detections ─────────────────────────────────────────────
    FuzzyRule("R19", "high",   "small",  "few",  "medium", weight=1.0),
    FuzzyRule("R20", "high",   "small",  "some", "medium", weight=1.0),
    FuzzyRule("R21", "high",   "small",  "many", "high",   weight=1.0),
    FuzzyRule("R22", "high",   "medium", "few",  "medium", weight=1.0),
    FuzzyRule("R23", "high",   "medium", "some", "high",   weight=1.0),
    FuzzyRule("R24", "high",   "medium", "many", "high",   weight=1.0),
    FuzzyRule("R25", "high",   "large",  "few",  "high",   weight=1.0),
    FuzzyRule("R26", "high",   "large",  "some", "high",   weight=1.0),
    FuzzyRule("R27", "high",   "large",  "many", "high",   weight=1.0),
]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5  ──  INFERENCE ENGINE
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class FuzzyResult:
    """Full result of one fuzzy inference call."""
    risk_score:      float                      # crisp 0–100
    severity:        str                        # Low | Medium | High
    membership:      Dict[str, float]           # all μ values
    rules_fired:     List[Dict]                 # active rules + strength
    aggregated_low:  np.ndarray = field(repr=False, default=None)
    aggregated_med:  np.ndarray = field(repr=False, default=None)
    aggregated_high: np.ndarray = field(repr=False, default=None)


# Dispatch tables
_CONF_MF  = {"low": ConfidenceFuzzy.low,  "medium": ConfidenceFuzzy.medium,  "high": ConfidenceFuzzy.high}
_BBOX_MF  = {"small": BBoxRatioFuzzy.small, "medium": BBoxRatioFuzzy.medium, "large": BBoxRatioFuzzy.large}
_COUNT_MF = {"few": CountFuzzy.few,       "some": CountFuzzy.some,           "many": CountFuzzy.many}
_OUT_MF   = {"low": out_low, "medium": out_medium, "high": out_high}


def infer(
    confidence:     float,
    bbox_ratio:     float,
    accident_count: int,
) -> FuzzyResult:
    """
    Mamdani fuzzy inference.

    Parameters
    ----------
    confidence     : YOLO detection confidence (0–1)
    bbox_ratio     : max bbox area / frame area (0–1)
    accident_count : number of accident bboxes in frame

    Returns
    -------
    FuzzyResult
    """
    # ── Fuzzification ──────────────────────────────────────────────────────────
    mu_conf  = ConfidenceFuzzy.memberships(confidence)
    mu_bbox  = BBoxRatioFuzzy.memberships(bbox_ratio)
    mu_count = CountFuzzy.memberships(accident_count)

    all_memberships = {**mu_conf, **mu_bbox, **mu_count}

    # ── Rule evaluation (min T-norm, weighted) ────────────────────────────────
    agg_low  = np.zeros_like(RISK_UNIVERSE)
    agg_med  = np.zeros_like(RISK_UNIVERSE)
    agg_high = np.zeros_like(RISK_UNIVERSE)

    rules_fired = []

    for rule in RULE_BASE:
        # Antecedent strength via min (AND)
        mu_c  = _CONF_MF [rule.conf_set ](confidence)
        mu_b  = _BBOX_MF [rule.bbox_set ](bbox_ratio)
        mu_n  = _COUNT_MF[rule.count_set](accident_count)

        strength = min(mu_c, mu_b, mu_n) * rule.weight

        if strength < 1e-6:
            continue

        rules_fired.append({
            "rule":     rule.name,
            "output":   rule.output_set,
            "strength": round(strength, 4),
            "antecedents": {
                "conf":  (rule.conf_set,  round(mu_c, 4)),
                "bbox":  (rule.bbox_set,  round(mu_b, 4)),
                "count": (rule.count_set, round(mu_n, 4)),
            },
        })

        # Implication: clip output MF at strength (Mamdani)
        clipped = np.minimum(_OUT_MF[rule.output_set](RISK_UNIVERSE), strength)

        if rule.output_set == "low":
            agg_low  = np.maximum(agg_low,  clipped)
        elif rule.output_set == "medium":
            agg_med  = np.maximum(agg_med,  clipped)
        else:
            agg_high = np.maximum(agg_high, clipped)

    # ── Aggregation (max) ──────────────────────────────────────────────────────
    aggregated = np.maximum(np.maximum(agg_low, agg_med), agg_high)

    # ── Defuzzification (centroid) ─────────────────────────────────────────────
    denom = np.sum(aggregated)
    if denom < 1e-9:
        # No rules fired → default to low risk
        risk_score = 10.0
    else:
        risk_score = float(np.sum(RISK_UNIVERSE * aggregated) / denom)

    risk_score = round(min(100.0, max(0.0, risk_score)), 1)

    # ── Severity label ─────────────────────────────────────────────────────────
    if risk_score <= 35:
        severity = "Low"
    elif risk_score <= 68:
        severity = "Medium"
    else:
        severity = "High"

    # Sort rules by firing strength
    rules_fired.sort(key=lambda r: r["strength"], reverse=True)

    return FuzzyResult(
        risk_score      = risk_score,
        severity        = severity,
        membership      = all_memberships,
        rules_fired     = rules_fired,
        aggregated_low  = agg_low,
        aggregated_med  = agg_med,
        aggregated_high = agg_high,
    )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6  ──  CONVENIENCE WRAPPER  (drop-in for detect_and_analyze.py)
# ═════════════════════════════════════════════════════════════════════════════

def fuzzy_classify(
    confidences:    list,
    bboxes:         list,
    frame_h:        int,
    frame_w:        int,
    accident_count: int,
) -> Tuple[float, str, FuzzyResult]:
    """
    Drop-in replacement for compute_risk_score() + classify_severity()
    from detect_and_analyze.py.

    Parameters
    ----------
    confidences    : list of float  — confidence per accident detection
    bboxes         : list of (x1,y1,x2,y2) tuples
    frame_h/w      : frame dimensions
    accident_count : total accident detections

    Returns
    -------
    (risk_score: float, severity: str, fuzzy_result: FuzzyResult)
    """
    if not confidences:
        empty = FuzzyResult(
            risk_score=0.0, severity="Low",
            membership={}, rules_fired=[],
            aggregated_low=np.zeros_like(RISK_UNIVERSE),
            aggregated_med=np.zeros_like(RISK_UNIVERSE),
            aggregated_high=np.zeros_like(RISK_UNIVERSE),
        )
        return 0.0, "Low", empty

    frame_area = frame_h * frame_w

    # Representative inputs
    mean_conf = float(np.mean(confidences))

    max_area  = 0
    for (x1, y1, x2, y2) in bboxes:
        max_area = max(max_area, (x2 - x1) * (y2 - y1))
    bbox_ratio = min(1.0, max_area / max(frame_area, 1))

    result = infer(
        confidence     = mean_conf,
        bbox_ratio     = bbox_ratio,
        accident_count = accident_count,
    )

    return result.risk_score, result.severity, result


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7  ──  PRETTY PRINTER
# ═════════════════════════════════════════════════════════════════════════════

def print_result(result: FuzzyResult, title: str = "Fuzzy Inference Result"):
    SEP = "═" * 62
    print(f"\n{SEP}")
    print(f"  🔢  {title}")
    print(SEP)
    print(f"  Risk Score  : {result.risk_score:.1f} / 100")
    print(f"  Severity    : {result.severity}")

    print(f"\n  Input Memberships:")
    groups = [
        ("Confidence", ["conf_low", "conf_medium", "conf_high"]),
        ("BBox Ratio", ["bbox_small", "bbox_medium", "bbox_large"]),
        ("Count",      ["count_few", "count_some", "count_many"]),
    ]
    for grp_name, keys in groups:
        vals = "  ".join(f"{k.split('_',1)[1]}: {result.membership.get(k, 0):.3f}"
                         for k in keys)
        print(f"    {grp_name:<14}: {vals}")

    print(f"\n  Rules Fired ({len(result.rules_fired)}):")
    for r in result.rules_fired[:8]:   # show top 8
        ant = r["antecedents"]
        print(f"    {r['rule']}  [{r['output']:<6}]  strength={r['strength']:.3f}"
              f"  conf={ant['conf'][0]}({ant['conf'][1]:.2f})"
              f"  bbox={ant['bbox'][0]}({ant['bbox'][1]:.2f})"
              f"  count={ant['count'][0]}({ant['count'][1]:.2f})")
    print(SEP)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8  ──  CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def _demo():
    """Run a set of representative test cases."""
    cases = [
        ("Minor fender-bender",     0.42, 0.04, 1),
        ("Moderate 2-car crash",    0.65, 0.14, 2),
        ("Major pile-up",           0.88, 0.35, 4),
        ("High-conf large bbox",    0.91, 0.50, 3),
        ("Low-conf small bbox",     0.31, 0.03, 1),
        ("Multi-vehicle high-conf", 0.78, 0.22, 5),
    ]
    for name, conf, bbox, count in cases:
        r = infer(conf, bbox, count)
        print_result(r, f"{name}  (conf={conf}, bbox={bbox}, count={count})")


def main():
    parser = argparse.ArgumentParser(
        description="Fuzzy Logic Severity Engine — standalone test"
    )
    parser.add_argument("--conf",  type=float, default=None, help="Confidence (0-1)")
    parser.add_argument("--bbox",  type=float, default=None, help="BBox ratio (0-1)")
    parser.add_argument("--count", type=int,   default=None, help="Accident count")
    parser.add_argument("--demo",  action="store_true",      help="Run demo sweep")
    args = parser.parse_args()

    if args.demo or (args.conf is None):
        _demo()
        return

    result = infer(args.conf, args.bbox or 0.1, args.count or 1)
    print_result(result, "Custom Input")


if __name__ == "__main__":
    main()
