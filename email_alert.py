"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   EMAIL ALERT MODULE  (v7 — full location + annotated detection frame)      ║
║   AI-Powered Traffic Accident Detection & Severity Analysis                  ║
║                                                                              ║
║   What changed vs v6:                                                        ║
║                                                                              ║
║   1. Dynamic email subject includes city name:                              ║
║      "🚨 Traffic Accident Alert | High Risk | Chennai"                      ║
║      City is extracted from the resolved location automatically.            ║
║                                                                              ║
║   2. Detection frame saved WITH bounding boxes drawn on it:                 ║
║      save_detection_frame(frame_bgr, detections, path) — new public helper  ║
║      Draws red bounding boxes + confidence labels before saving the JPEG.   ║
║      Called by dashboard.py BEFORE passing the path to send_alert().        ║
║                                                                              ║
║   3. Full IP-based auto-location section in the email body:                 ║
║      ┌─────────────────────────────┐                                        ║
║      │  📍 CURRENT LOCATION        │                                        ║
║      │  City / State / Country     │                                        ║
║      │  Latitude / Longitude       │                                        ║
║      │  Full Address               │                                        ║
║      │  Location Source            │                                        ║
║      │  Google Maps (clickable)    │                                        ║
║      └─────────────────────────────┘                                        ║
║      Auto-populated from IP lookup when use_live_location=True.             ║
║      Falls back to camera_location string when IP lookup unavailable.       ║
║                                                                              ║
║   4. Evidence section in email body:                                         ║
║      Shows "📎 Detection_Frame.jpg — attached" when file exists.            ║
║                                                                              ║
║   5. All v6 features fully preserved:                                        ║
║      • detection_frame_path file attachment                                  ║
║      • camera_location sidebar input                                         ║
║      • use_live_location + current_location() backward compat               ║
║      • inline cid: frame (numpy BGR array)                                  ║
║      • SMTP / cooldown / TLS / from_env() unchanged                         ║
║      • No other module modified                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import smtplib
import time
import urllib.request
import json as _json
from datetime import datetime
from email.mime.image     import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from typing               import Optional, List, Dict, Tuple

try:
    from location_service import LocationInfo, maps_link
except Exception:  # pragma: no cover - optional import
    LocationInfo = None
    def maps_link(loc):
        return ""

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
FIXED_LOCATION = "GST Road, Chennai, Tamil Nadu, India"

_IP_API_URL   = "http://ip-api.com/json/?fields=status,city,regionName,country,lat,lon,query"
_HTTP_TIMEOUT = 6   # seconds
_DEFAULT_ACCIDENT_FRAME_DIR = os.path.abspath(os.path.join("outputs", "accident_frames"))
_SUPPORTED_FRAME_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Bounding box drawing constants (used by save_detection_frame)
_BOX_COLOR     = (0,   0,   255)   # red  (BGR)
_BOX_THICKNESS = 2
_FONT          = cv2.FONT_HERSHEY_DUPLEX
_FONT_SCALE    = 0.55
_FONT_THICK    = 1


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight location object
# Avoids hard dependency on location_service.py.
# Attribute names match what _build_message() reads via getattr().
# ─────────────────────────────────────────────────────────────────────────────
class _SimpleLocation:
    """Minimal location data object compatible with email _build_message()."""
    def __init__(
        self,
        address:      str,
        city:         str,
        state:        str,
        country:      str,
        latitude:     float,
        longitude:    float,
        accuracy:     float,
        source_label: str,
    ):
        self.address      = address
        self.city         = city
        self.state        = state
        self.country      = country
        self.latitude     = latitude
        self.longitude    = longitude
        self.accuracy     = accuracy
        self.source_label = source_label


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC HELPER — save annotated detection frame
# ─────────────────────────────────────────────────────────────────────────────
def save_detection_frame(
    frame:      np.ndarray,
    detections: List[Dict],
    save_path:  str,
    jpeg_quality: int = 90,
) -> bool:
    """
    Draw accident bounding boxes on `frame` and save as JPEG.

    Parameters
    ----------
    frame      : BGR numpy array (the raw video frame)
    detections : list of detection dicts, each with keys:
                   x1, y1, x2, y2, label, confidence, is_accident (bool)
                 Any detection where is_accident=True (or label contains
                 "accident"/"moderate"/"severe") gets a red bounding box.
    save_path  : destination JPEG path (will be created/overwritten)
    jpeg_quality : JPEG compression quality 1-100

    Returns True on success, False on any error.

    Usage in dashboard.py:
        from email_alert import save_detection_frame
        det_path = str(FRAMES_OUT_DIR / f"detection_{frame_no:06d}.jpg")
        save_detection_frame(raw_frame, detections, det_path)
        email_sys.send_alert(..., detection_frame_path=det_path)
    """
    try:
        annotated = frame.copy()

        for det in detections:
            # Determine if this is an accident detection
            is_acc = det.get("is_accident", False)
            if not is_acc:
                label = det.get("label", "").lower()
                is_acc = any(kw in label for kw in
                             ["accident", "moderate", "severe", "minor", "crash"])
            if not is_acc:
                continue

            x1   = int(det.get("x1", 0))
            y1   = int(det.get("y1", 0))
            x2   = int(det.get("x2", 0))
            y2   = int(det.get("y2", 0))
            conf = float(det.get("confidence", 0))
            lbl  = det.get("label", "Accident")

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), _BOX_COLOR, _BOX_THICKNESS)

            # Corner accent marks
            clen, cthick = 12, 3
            for cx, cy, dx, dy in [
                (x1, y1,  1,  1), (x2, y1, -1,  1),
                (x1, y2,  1, -1), (x2, y2, -1, -1),
            ]:
                cv2.line(annotated, (cx, cy), (cx + dx * clen, cy),
                         _BOX_COLOR, cthick)
                cv2.line(annotated, (cx, cy), (cx, cy + dy * clen),
                         _BOX_COLOR, cthick)

            # Label background + text
            tag = f"  {lbl}  {conf:.2f}  "
            (tw, th), _ = cv2.getTextSize(tag, _FONT, _FONT_SCALE, _FONT_THICK)
            cv2.rectangle(annotated,
                          (x1, y1 - th - 10), (x1 + tw, y1),
                          _BOX_COLOR, -1)
            cv2.putText(annotated, tag, (x1, y1 - 5),
                        _FONT, _FONT_SCALE, (255, 255, 255),
                        _FONT_THICK, cv2.LINE_AA)

        # Timestamp watermark bottom-left
        ts_txt = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(annotated, ts_txt, (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        success, buf = cv2.imencode(
            ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        )
        if not success:
            print(f"  ⚠  save_detection_frame: imencode failed")
            return False

        with open(save_path, "wb") as fh:
            fh.write(buf.tobytes())

        print(f"  📸  Detection frame saved → {os.path.basename(save_path)}")
        return True

    except Exception as exc:
        print(f"  ⚠  save_detection_frame error: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# IP LOCATION FETCH  (module-level, reused by class and helper)
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_ip_location() -> Optional["_SimpleLocation"]:
    """
    Query ip-api.com and return a _SimpleLocation, or None on failure.
    Returned object has: city, state, country, address, latitude, longitude,
    accuracy=0, source_label="🌐 IP-based location".
    """
    try:
        req = urllib.request.Request(
            _IP_API_URL,
            headers={"User-Agent": "AI-Traffic-Monitor/1.0"},
        )
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            data = _json.loads(resp.read().decode())

        if data.get("status") != "success":
            return None

        city    = data.get("city",       "")
        state   = data.get("regionName", "")
        country = data.get("country",    "")
        lat     = float(data.get("lat",  0))
        lon     = float(data.get("lon",  0))
        address = ", ".join(p for p in [city, state, country] if p) or FIXED_LOCATION

        return _SimpleLocation(
            address      = address,
            city         = city,
            state        = state,
            country      = country,
            latitude     = lat,
            longitude    = lon,
            accuracy     = 0.0,
            source_label = "🌐 IP-based location",
        )

    except Exception as exc:
        print(f"  [email_alert] IP location fetch failed: {exc}")
        return None


def _fixed_location_obj() -> "_SimpleLocation":
    """Return a _SimpleLocation for the hardcoded FIXED_LOCATION."""
    return _SimpleLocation(
        address      = FIXED_LOCATION,
        city         = "Chennai",
        state        = "Tamil Nadu",
        country      = "India",
        latitude     = 12.9716,
        longitude    = 80.2209,
        accuracy     = 0.0,
        source_label = "📌 Fixed (fallback)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────
class EmailAlertSystem:
    """
    Sends accident alert emails via SMTP.

    Email layout (v7)
    ──────────────────
        🚨 HEADER (severity colour)
        ──────────────────────────────────────
        DETECTION DETAILS
            ⏰ Date & Time
            ⚠️ Severity
            📊 Risk Score
            🎯 Detection Confidence
            🔍 Detection Status
        ──────────────────────────────────────
        📍 CURRENT LOCATION
            📷 Camera Location
            🏙 City / State / Country
            🌐 Latitude / Longitude
            🏠 Full Address
            📡 Location Source
            🗺  Google Maps (clickable link)
        ──────────────────────────────────────
        🖼  EVIDENCE
            Inline accident frame image
            📎 Detection_Frame.jpg — attached
        ──────────────────────────────────────
        Footer
    """

    def __init__(
        self,
        smtp_host:         str,
        smtp_port:         int,
        sender:            str,
        password:          str,
        recipients:        list,
        cooldown_sec:      int  = 60,
        use_live_location: bool = False,
        camera_location:   str  = "",
    ):
        self.smtp_host         = smtp_host
        self.smtp_port         = smtp_port
        self.sender            = sender
        self.password          = password
        self.recipients        = [r for r in recipients if r]
        self.cooldown_sec      = cooldown_sec
        self.use_live_location = use_live_location
        self.camera_location   = camera_location

        self._last_sent        = 0.0
        self._send_count       = 0
        self._cached_ip_loc    = None          # cache _SimpleLocation from IP
        self._cached_loc_str   = None          # cache plain string for display

    # ── Backward-compat display helper ────────────────────────────────────────

    def current_location(self) -> str:
        """
        Return a human-readable location string for sidebar display.
        Priority: camera_location → IP lookup → FIXED_LOCATION.
        """
        if self.camera_location:
            return self.camera_location
        if not self.use_live_location:
            return FIXED_LOCATION
        if self._cached_loc_str is not None:
            return self._cached_loc_str
        loc = _fetch_ip_location()
        self._cached_loc_str = loc.address if loc else FIXED_LOCATION
        return self._cached_loc_str

    # ── Public API ────────────────────────────────────────────────────────────

    def should_send(self) -> bool:
        """True if cooldown has elapsed since last send."""
        return (time.time() - self._last_sent) >= self.cooldown_sec

    def send_alert(
        self,
        risk_score:           float,
        severity:             str,
        timestamp:            Optional[str]        = None,
        confidence:           float                = 0.0,
        location              = None,
        frame:                Optional[np.ndarray] = None,
        detection_frame_path: Optional[str]        = None,
        camera_location:      Optional[str]        = None,
        # backward-compat params (accepted, not used in email body)
        vehicle_count:  int = 0,
        accident_count: int = 0,
        frame_no:       int = 0,
    ) -> bool:
        """
        Send the accident alert email.

        Parameters
        ----------
        risk_score            : 0-100
        severity              : "High" | "Medium" | "Low"
        timestamp             : readable string, defaults to now
        confidence            : detection confidence 0-1 (shown in body)
        location              : LocationInfo from location_service, or None
        frame                 : BGR numpy array — embedded INLINE via cid:
        detection_frame_path  : path to annotated detection frame JPEG
                                → attached as "Detection_Frame.jpg"
                                → if missing/None, email sends without attachment
        camera_location       : per-call override for camera/road name

        Returns True if sent, False if blocked or config missing.
        """
        if not self.should_send():
            remaining = int(self.cooldown_sec - (time.time() - self._last_sent))
            print(f"  ⏳  Email cooldown — {remaining}s remaining")
            return False

        if not self.recipients:
            print("  ⚠  No recipients configured")
            return False

        ts      = timestamp or datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        cam_loc = camera_location or self.camera_location or ""

        # ── Resolve location object ───────────────────────────────────────────
        # Priority:
        #   1. Explicit LocationInfo passed by caller (from location_service)
        #   2. IP-based auto-lookup (when use_live_location=True)
        #   3. Camera location string → wrapped into _SimpleLocation
        #   4. Hardcoded fallback
        resolved_loc = self._resolve_location(location, cam_loc)

        resolved_frame_path = detection_frame_path

        try:
            msg = self._build_message(
                risk_score           = risk_score,
                severity             = severity,
                ts                   = ts,
                confidence           = confidence,
                loc                  = resolved_loc,
                cam_loc_label        = cam_loc,
                frame                = frame,
                detection_frame_path = resolved_frame_path,
            )
            self._send(msg)
            self._last_sent   = time.time()
            self._send_count += 1
            print(f"  ✉  Alert #{self._send_count} sent → {self.recipients}")
            if resolved_frame_path and os.path.isfile(resolved_frame_path):
                print(f"Email attachment:\n{os.path.abspath(resolved_frame_path)}")
            else:
                print(f"  ⚠  Email attachment missing: {resolved_frame_path}")
            return True
        except Exception as exc:
            print(f"  ✗  Email send failed: {exc}")
            return False

    # ── Location resolution ───────────────────────────────────────────────────

    def _resolve_location(
        self,
        explicit_location,   # LocationInfo | None
        cam_loc_label: str,
    ) -> "_SimpleLocation":
        """
        Resolve the best available location to a _SimpleLocation.

        Tries in order:
            1. Explicit LocationInfo (from location_service.get_location())
            2. Cached IP-based location
            3. Live IP-based lookup (if use_live_location=True)
            4. Build from camera_location label string
            5. Hardcoded FIXED_LOCATION fallback
        """
        # 1. Explicit LocationInfo from location_service
        if explicit_location is not None:
            address = getattr(explicit_location, "address", FIXED_LOCATION) or FIXED_LOCATION
            city = getattr(explicit_location, "city", "") or ""
            state = getattr(explicit_location, "state", "") or ""
            country = getattr(explicit_location, "country", "") or ""
            latitude = getattr(explicit_location, "latitude", 0.0) or 0.0
            longitude = getattr(explicit_location, "longitude", 0.0) or 0.0
            accuracy = getattr(explicit_location, "accuracy", 0.0) or 0.0
            source_label = getattr(explicit_location, "source_label", None)
            if source_label is None:
                source_obj = getattr(explicit_location, "source", "")
                source_label = {
                    "GPS": "📡 GPS (browser)",
                    "IP": "🌐 IP-based location",
                    "fixed": "📌 Fixed location (fallback)",
                    "pending": "⏳ Detecting location",
                    "unavailable": "⚠️ Location unavailable",
                }.get(source_obj, "—")
            return _SimpleLocation(
                address      = address,
                city         = city,
                state        = state,
                country      = country,
                latitude     = latitude,
                longitude    = longitude,
                accuracy     = accuracy,
                source_label = source_label,
            )

        # 2. Cached IP result
        if self._cached_ip_loc is not None:
            return self._cached_ip_loc

        # 3. Fresh IP lookup
        if self.use_live_location:
            loc = _fetch_ip_location()
            if loc:
                self._cached_ip_loc = loc
                return loc

        # 4. Camera location label from sidebar (no coords — text only)
        if cam_loc_label:
            return _SimpleLocation(
                address      = cam_loc_label,
                city         = cam_loc_label.split(",")[0].strip(),
                state        = "",
                country      = "",
                latitude     = 0.0,
                longitude    = 0.0,
                accuracy     = 0.0,
                source_label = "📷 Camera (user-configured)",
            )

        # 5. Hardcoded fallback
        return _fixed_location_obj()

    # ── Email builder ─────────────────────────────────────────────────────────

    def _build_message(
        self,
        risk_score:           float,
        severity:             str,
        ts:                   str,
        confidence:           float,
        loc:                  "_SimpleLocation",
        cam_loc_label:        str,
        frame:                Optional[np.ndarray],
        detection_frame_path: Optional[str],
    ) -> MIMEMultipart:
        """
        Build the full multipart/mixed email with:
            - HTML body (detection details + location + evidence sections)
            - Plain-text fallback
            - Optional inline cid: frame (from numpy array)
            - Optional file attachment (Detection_Frame.jpg)

        MIME structure:
            multipart/mixed
              └── multipart/related
                    └── multipart/alternative
                          ├── text/plain
                          └── text/html
                    └── image/jpeg (Content-ID: accident_frame)  ← inline
              └── image/jpeg  Content-Disposition: attachment    ← file
        """
        sev_hex = {
            "High":   "#C0392B",
            "Medium": "#D68910",
            "Low":    "#1E8449",
        }.get(severity, "#7F8C8D")

        # ── Dynamic subject with city ─────────────────────────────────────────
        city_tag = loc.city or "Unknown"
        subject  = f"🚨 Traffic Accident Alert | {severity} Risk | {city_tag}"

        # ── Location data ─────────────────────────────────────────────────────
        has_coords   = (loc.latitude != 0.0 or loc.longitude != 0.0)
        lat_str      = f"{loc.latitude:.6f}"  if has_coords else "—"
        lon_str      = f"{loc.longitude:.6f}" if has_coords else "—"
        maps_url     = maps_link(
            type("_LocationShim", (), {
                "latitude": loc.latitude,
                "longitude": loc.longitude,
                "source": "GPS",
            })
        ) if has_coords else ""
        maps_html    = (
            f'<a href="{maps_url}" style="color:#5599FF;text-decoration:none;'
            f'font-weight:bold;">🗺 View on Google Maps ↗</a>'
        ) if maps_url else "—"

        city_state   = ", ".join(p for p in [loc.city, loc.state, loc.country] if p)
        display_addr = loc.address or cam_loc_label or FIXED_LOCATION

        # ── File attachment check ─────────────────────────────────────────────
        frame_file_ok = (
            detection_frame_path is not None
            and isinstance(detection_frame_path, str)
            and os.path.isfile(detection_frame_path)
        )

        # ── Inline cid frame ──────────────────────────────────────────────────
        frame_cid  = "accident_frame_inline"
        has_inline = frame is not None

        # ── Plain-text body ───────────────────────────────────────────────────
        attachment_display = "Attachment: Accident_Detection_Frame.jpg" if frame_file_ok else "Attachment: Not Available"
        plain = (
            f"🚨 Accident Detected\n"
            f"{'='*45}\n\n"
            f"DETECTION DETAILS\n"
            f"  Date & Time   : {ts}\n"
            f"  Severity      : {severity}\n"
            f"  Risk Score    : {risk_score:.1f} / 100\n"
            f"  Confidence    : {confidence:.2f}\n"
            f"  Status        : Accident Detected\n\n"
            f"CURRENT LOCATION\n"
            f"  Camera Label  : {cam_loc_label or '—'}\n"
            f"  City          : {loc.city or '—'}\n"
            f"  State         : {loc.state or '—'}\n"
            f"  Country       : {loc.country or '—'}\n"
            f"  Latitude      : {lat_str}\n"
            f"  Longitude     : {lon_str}\n"
            f"  Full Address  : {display_addr}\n"
            f"  Source        : {loc.source_label}\n"
            f"  Google Maps   : {maps_url or '—'}\n\n"
            f"EVIDENCE\n"
            f"  {attachment_display}\n"
            f"\nPlease verify and respond immediately.\n"
            f"Contact traffic control or emergency services if required."
        )

        # ── HTML body ─────────────────────────────────────────────────────────
        # Evidence section
        evidence_rows = ""
        if has_inline:
            evidence_rows += self._row(
                "🖼 &nbsp;Frame Preview",
                "(see inline image below)",
                sev_hex,
            )
        attachment_label = "Attachment: Accident_Detection_Frame.jpg" if frame_file_ok else "Attachment: Not Available"
        if frame_file_ok:
            evidence_rows += self._row(
                "📎 &nbsp;Attachment",
                f'<span style="color:#2ECC71;font-weight:bold;">{attachment_label}</span>',
                sev_hex,
            )
        else:
            evidence_rows += self._row(
                "📎 &nbsp;Attachment",
                f'<span style="color:#888888;">{attachment_label}</span>',
                sev_hex,
            )

        # Inline image block (rendered after data rows)
        inline_img_block = ""
        if has_inline:
            inline_img_block = f"""
          <!-- ── Evidence: inline annotated frame ──────────────── -->
          <tr>
            <td style="padding:4px 28px 24px;">
              <img src="cid:{frame_cid}"
                   width="504"
                   style="width:100%;max-width:504px;
                          border-radius:8px;
                          border:2px solid {sev_hex};
                          display:block;" />
            </td>
          </tr>"""

        html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
</head>
<body style="margin:0;padding:0;background:#0D0D1A;
             font-family:Arial,Helvetica,sans-serif;">

  <table width="100%" cellpadding="0" cellspacing="0"
         style="background:#0D0D1A;padding:30px 0;">
    <tr>
      <td align="center">
        <table width="580" cellpadding="0" cellspacing="0"
               style="background:#1A1A30;border-radius:12px;
                      border:2px solid {sev_hex};overflow:hidden;">

          <!-- ══ HEADER ══════════════════════════════════════════ -->
          <tr>
            <td style="background:{sev_hex};padding:20px 28px;">
              <h1 style="margin:0;color:#FFFFFF;font-size:1.3rem;
                         letter-spacing:0.5px;">
                🚨 &nbsp;Accident Detected
              </h1>
              <p style="margin:6px 0 0;color:rgba(255,255,255,0.80);
                        font-size:0.82rem;">
                AI Traffic Monitoring System — Automated Alert
              </p>
            </td>
          </tr>

          <!-- ══ DETECTION DETAILS ════════════════════════════════ -->
          <tr>
            <td style="padding:20px 28px 8px;">
              <p style="margin:0 0 10px;color:#7EB8FF;font-size:0.78rem;
                        font-weight:700;letter-spacing:1px;
                        text-transform:uppercase;">
                Detection Details
              </p>
              <table width="100%" cellpadding="0" cellspacing="0">
                {self._row("⏰ &nbsp;Date &amp; Time",       ts,                        sev_hex)}
                {self._row("⚠️ &nbsp;Severity",              severity,                  sev_hex)}
                {self._row("📊 &nbsp;Risk Score",            f"{risk_score:.1f} / 100", sev_hex)}
                {self._row("🎯 &nbsp;Detection Confidence",  f"{confidence:.2f}",       sev_hex)}
                {self._row("🔍 &nbsp;Detection Status",      "Accident Detected",        sev_hex)}
              </table>
            </td>
          </tr>

          <!-- ══ CURRENT LOCATION ═════════════════════════════════ -->
          <tr>
            <td style="padding:8px 28px;">
              <div style="background:#12122A;border-radius:8px;
                          border-left:4px solid {sev_hex};
                          padding:14px 16px;margin-bottom:4px;">
                <p style="margin:0 0 12px;color:#7EB8FF;font-size:0.78rem;
                           font-weight:700;letter-spacing:1px;
                           text-transform:uppercase;">
                  📍 Current Location
                </p>
                <table width="100%" cellpadding="0" cellspacing="0">
                  {self._row_inner("📷 Camera Label",     cam_loc_label or "—",   sev_hex)}
                  {self._row_inner("🏙 City",              loc.city or "—",        sev_hex)}
                  {self._row_inner("🏛 State",             loc.state or "—",       sev_hex)}
                  {self._row_inner("🌍 Country",           loc.country or "—",     sev_hex)}
                  {self._row_inner("🌐 Latitude",          lat_str,                sev_hex)}
                  {self._row_inner("🌐 Longitude",         lon_str,                sev_hex)}
                  {self._row_inner("🏠 Full Address",      display_addr,           sev_hex)}
                  {self._row_inner("📡 Location Source",   loc.source_label,       sev_hex)}
                  {self._row_inner("🗺 Google Maps",       maps_html,              sev_hex)}
                </table>
              </div>
            </td>
          </tr>

          <!-- ══ EVIDENCE ═════════════════════════════════════════ -->
          <tr>
            <td style="padding:8px 28px 20px;">
              <p style="margin:0 0 10px;color:#7EB8FF;font-size:0.78rem;
                        font-weight:700;letter-spacing:1px;
                        text-transform:uppercase;">
                Evidence
              </p>
              <table width="100%" cellpadding="0" cellspacing="0">
                {evidence_rows}
              </table>
            </td>
          </tr>

          {inline_img_block}

          <!-- ══ FOOTER ══════════════════════════════════════════ -->
          <tr>
            <td style="background:#12122A;padding:14px 28px;
                       text-align:center;color:#44446A;font-size:0.75rem;">
              AI Traffic Monitoring System &nbsp;•&nbsp;
              Auto-generated alert &nbsp;•&nbsp;
              Do not reply to this email
            </td>
          </tr>

        </table>
      </td>
    </tr>
  </table>

</body>
</html>"""

        # ── Assemble MIME ─────────────────────────────────────────────────────
        msg_mixed            = MIMEMultipart("mixed")
        msg_mixed["Subject"] = subject
        msg_mixed["From"]    = self.sender
        msg_mixed["To"]      = ", ".join(self.recipients)

        msg_related = MIMEMultipart("related")
        msg_alt     = MIMEMultipart("alternative")
        msg_alt.attach(MIMEText(plain, "plain"))
        msg_alt.attach(MIMEText(html,  "html"))
        msg_related.attach(msg_alt)

        # Inline numpy frame
        if has_inline:
            img_part = self._encode_frame_bytes(frame, frame_cid)
            if img_part is not None:
                msg_related.attach(img_part)

        msg_mixed.attach(msg_related)

        # File attachment — ONLY Detection_Frame.jpg, nothing else
        if frame_file_ok:
            self._attach_file(msg_mixed, detection_frame_path)

        return msg_mixed

    # ── HTML row helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _row(label: str, value: str, accent: str) -> str:
        """Standard full-width data row (Detection Details & Evidence sections)."""
        return f"""
        <tr>
          <td style="padding:10px 0;border-bottom:1px solid #22223A;">
            <table width="100%" cellpadding="0" cellspacing="0">
              <tr>
                <td width="42%"
                    style="color:#8888AA;font-size:0.82rem;
                           padding-right:12px;vertical-align:top;">
                  {label}
                </td>
                <td style="color:#FFFFFF;font-size:0.95rem;font-weight:bold;
                           border-left:3px solid {accent};
                           padding-left:12px;">
                  {value}
                </td>
              </tr>
            </table>
          </td>
        </tr>"""

    @staticmethod
    def _row_inner(label: str, value: str, accent: str) -> str:
        """Compact row used inside the location card (slightly smaller text)."""
        return f"""
        <tr>
          <td style="padding:7px 0;border-bottom:1px solid #1A1A2E;">
            <table width="100%" cellpadding="0" cellspacing="0">
              <tr>
                <td width="42%"
                    style="color:#888899;font-size:0.78rem;
                           padding-right:10px;vertical-align:top;">
                  {label}
                </td>
                <td style="color:#E0E0F0;font-size:0.85rem;font-weight:600;
                           border-left:2px solid {accent};
                           padding-left:10px;">
                  {value}
                </td>
              </tr>
            </table>
          </td>
        </tr>"""

    # ── File attachment ───────────────────────────────────────────────────────

    @staticmethod
    def _attach_file(msg: MIMEMultipart, file_path: str) -> bool:
        """
        Attach file at file_path to the email as 'Accident_Detection_Frame.jpg'.
        Uses MIMEImage so the recipient receives it as a downloadable attachment.
        Returns True on success, False on any error — never raises.
        """
        try:
            if not os.path.isfile(file_path):
                print(f"  ⚠  Email attachment missing: {file_path}")
                return False

            with open(file_path, "rb") as fh:
                payload = fh.read()

            lower = file_path.lower()
            if lower.endswith((".jpg", ".jpeg")):
                maintype, subtype = "image", "jpeg"
            elif lower.endswith(".png"):
                maintype, subtype = "image", "png"
            else:
                maintype, subtype = "application", "octet-stream"

            part = MIMEImage(payload, _subtype=subtype)
            part.add_header(
                "Content-Disposition",
                "attachment",
                filename="Accident_Detection_Frame.jpg",
            )
            msg.attach(part)
            print(f"Email attachment:\n{os.path.abspath(file_path)}")
            return True

        except Exception as exc:
            print(f"  ⚠  Attachment error: {exc}")
            return False

    # ── Inline frame encoder ──────────────────────────────────────────────────

    @staticmethod
    def _encode_frame_bytes(
        frame:        np.ndarray,
        content_id:   str,
        max_width:    int = 800,
        jpeg_quality: int = 85,
    ) -> Optional[MIMEImage]:
        """Encode BGR numpy array → JPEG → MIMEImage for cid: inline use."""
        try:
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / w
                frame = cv2.resize(frame, (max_width, int(h * scale)),
                                   interpolation=cv2.INTER_AREA)
            success, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            if not success:
                return None
            img = MIMEImage(buf.tobytes(), _subtype="jpeg")
            img.add_header("Content-ID",          f"<{content_id}>")
            img.add_header("Content-Disposition", "inline",
                           filename="accident_frame_inline.jpg")
            return img
        except Exception as exc:
            print(f"  ⚠  Frame encode error: {exc}")
            return None

    # ── SMTP sender ───────────────────────────────────────────────────────────

    def _send(self, msg: MIMEMultipart):
        """Open a TLS-upgraded SMTP connection and deliver the message."""
        with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(self.sender, self.password)
            server.sendmail(self.sender, self.recipients, msg.as_string())


# ── Convenience factory from environment variables ────────────────────────────
def from_env(cooldown_sec: int = 60) -> Optional["EmailAlertSystem"]:
    """
    Build from environment variables:
        ALERT_SMTP_HOST, ALERT_SMTP_PORT, ALERT_SENDER,
        ALERT_PASSWORD, ALERT_RECIPIENTS (comma-separated),
        ALERT_USE_LIVE_LOCATION ("true"/"false"),
        ALERT_CAMERA_LOCATION   (optional string)
    Returns None if any required variable is missing.
    """
    host       = os.getenv("ALERT_SMTP_HOST")
    port_str   = os.getenv("ALERT_SMTP_PORT", "587")
    sender     = os.getenv("ALERT_SENDER")
    password   = os.getenv("ALERT_PASSWORD")
    recipients = os.getenv("ALERT_RECIPIENTS", "")
    live_loc   = os.getenv("ALERT_USE_LIVE_LOCATION", "false").lower() == "true"
    cam_loc    = os.getenv("ALERT_CAMERA_LOCATION", "")

    if not all([host, sender, password, recipients]):
        return None

    return EmailAlertSystem(
        smtp_host         = host,
        smtp_port         = int(port_str),
        sender            = sender,
        password          = password,
        recipients        = [r.strip() for r in recipients.split(",") if r.strip()],
        cooldown_sec      = cooldown_sec,
        use_live_location = live_loc,
        camera_location   = cam_loc,
    )