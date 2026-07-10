"""
location_service.py
═══════════════════════════════════════════════════════════════════════════════
Browser GPS → reverse geocode → Streamlit session state.

Responsibilities
────────────────
  1. Inject a tiny JS snippet into Streamlit via st.components.v1.html()
     that calls navigator.geolocation.getCurrentPosition().
  2. Pass the result back to Python through a hidden Streamlit text_input
     (the only reliable bridge between browser JS and Python in Streamlit).
  3. Parse the returned JSON and reverse-geocode to a human address using
     the Nominatim OpenStreetMap API (zero extra dependencies — uses urllib).
  4. Fall back to IP-based geolocation (ip-api.com JSON endpoint) if the
     user denies permission or GPS times out.
  5. Store the final LocationInfo in st.session_state["location"] so the
     lookup runs exactly once per session.

Public API
──────────
  get_location() -> LocationInfo
      Call this once at the top of dashboard.py, before the sidebar.
      Returns a LocationInfo dataclass ready for display and email.

  format_for_email(loc: LocationInfo) -> str
      Returns a one-line string safe to embed in the email HTML body.

  maps_link(loc: LocationInfo) -> str
      Returns a Google Maps URL for the detected coordinates.

No modifications to detect_and_analyze.py, heatmap_analysis.py,
fuzzy_severity.py, xai_explainer.py, or the YOLO/logging/risk pipeline.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

try:
    from geopy.geocoders import Nominatim as _GeopyNominatim
    from geopy.exc import GeopyError as _GeopyError
except Exception:  # pragma: no cover - geopy may be absent in some environments
    _GeopyNominatim = None
    _GeopyError = Exception

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — change these constants to customise fallback behaviour
# ─────────────────────────────────────────────────────────────────────────────

# Shown when BOTH GPS and IP lookup fail completely
FALLBACK_ADDRESS = "GST Road, Chennai, Tamil Nadu, India"

# Nominatim reverse-geocode endpoint (OpenStreetMap, no API key needed)
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"

# ip-api.com — free, no key, works from any server/desktop
IP_API_URL = "http://ip-api.com/json/?fields=status,city,regionName,country,lat,lon"

# HTTP timeout for both geocoding and IP lookup requests (seconds)
HTTP_TIMEOUT = 6

# Key used to persist the result in Streamlit session state
SESSION_KEY = "location"

# Internal key used to read the JS-posted GPS payload from the hidden widget
_GPS_INPUT_KEY = "_gps_payload"


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODEL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LocationInfo:
    """
    Holds everything the dashboard and email system need about the
    user's current location.

    source : "GPS"  — obtained from navigator.geolocation (browser)
             "IP"   — obtained from ip-api.com
             "fixed"— hardcoded fallback (both APIs failed)
             "pending" — waiting for browser permission / response
             "unavailable" — browser denied or failed and no fallback was available
    """
    address:  str             # Human-readable address
    latitude:  float          # Decimal degrees
    longitude: float          # Decimal degrees
    accuracy:  float          # Metres (GPS) or 0 for IP / fixed
    source:    str            # "GPS" | "IP" | "fixed"
    raw:       dict = field(default_factory=dict, repr=False)  # raw API response

    @property
    def is_gps(self) -> bool:
        return self.source == "GPS"

    @property
    def source_label(self) -> str:
        labels = {
            "GPS":         "📡 GPS (browser)",
            "IP":          "🌐 IP-based location",
            "fixed":       "📌 Fixed location (fallback)",
            "pending":     "⏳ Detecting location",
            "unavailable": "⚠️ Location unavailable",
        }
        return labels.get(self.source, self.source)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — JAVASCRIPT INJECTOR
# Injects a <script> that calls navigator.geolocation and posts the result
# to a hidden Streamlit text_input via React's nativeInputValueSetter trick.
# This is the standard Streamlit pattern for JS→Python communication.
# ─────────────────────────────────────────────────────────────────────────────

_GPS_JS = """
<script>
(function() {
    // Find the hidden text input Streamlit renders for _GPS_INPUT_KEY.
    // Streamlit renders inputs as: <input data-testid="stTextInput" ...>
    // We locate it by its aria-label which matches the label we pass to
    // st.text_input().  Using a short poll loop because the DOM may not be
    // ready at injection time.
    var MAX_TRIES = 40;
    var tries = 0;
    var LABEL = "_gps_payload";

    function findInput() {
        // Match by aria-label (Streamlit ≥1.20 sets this from the label arg)
        var inputs = document.querySelectorAll('input[aria-label="' + LABEL + '"]');
        if (inputs.length) return inputs[0];
        // Wider fallback — any text input containing "gps_payload" in aria-label
        var all = document.querySelectorAll('input[type="text"]');
        for (var i = 0; i < all.length; i++) {
            if (all[i].getAttribute('aria-label') &&
                all[i].getAttribute('aria-label').indexOf('gps_payload') !== -1)
                return all[i];
        }
        return null;
    }

    function postResult(payload) {
        var input = findInput();
        if (!input) return;
        // React controlled input: must use the nativeInputValueSetter trick
        var nativeSetter = Object.getOwnPropertyDescriptor(
            window.HTMLInputElement.prototype, 'value'
        ).set;
        nativeSetter.call(input, JSON.stringify(payload));
        input.dispatchEvent(new Event('input', { bubbles: true }));
    }

    function tryGeolocation() {
        if (!navigator.geolocation) {
            postResult({ error: "Geolocation API not supported in this browser." });
            return;
        }

        navigator.geolocation.getCurrentPosition(
            function(pos) {
                postResult({
                    ok:       true,
                    lat:      pos.coords.latitude,
                    lon:      pos.coords.longitude,
                    accuracy: pos.coords.accuracy
                });
            },
            function(err) {
                // err.code: 1=PERMISSION_DENIED, 2=UNAVAILABLE, 3=TIMEOUT
                postResult({
                    error:     err.message,
                    errorCode: err.code
                });
            },
            {
                enableHighAccuracy: true,
                timeout:            10000,
                maximumAge:         0
            }
        );
    }

    function waitForInputThenGeo() {
        tries++;
        if (findInput()) {
            tryGeolocation();
        } else if (tries < MAX_TRIES) {
            setTimeout(waitForInputThenGeo, 150);
        } else {
            console.warn("location_service: hidden input not found after " + MAX_TRIES + " tries");
        }
    }

    waitForInputThenGeo();
})();
</script>
"""


def _inject_gps_js():
    """
    Render an invisible iframe that executes the GPS JavaScript.
    height=0 keeps it visually hidden.
    """
    components.html(_GPS_JS, height=0, scrolling=False)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — REVERSE GEOCODING (Nominatim / OpenStreetMap)
# No API key required. Nominatim's usage policy requires a unique User-Agent.
# ─────────────────────────────────────────────────────────────────────────────

def _reverse_geocode(lat: float, lon: float) -> str:
    """
    Convert (lat, lon) → human-readable address string via Nominatim.
    Prefers geopy when installed, and falls back to the raw Nominatim API.
    """
    if _GeopyNominatim is not None:
        try:
            geolocator = _GeopyNominatim(user_agent="AI-Traffic-Monitor/1.0")
            location = geolocator.reverse((lat, lon), language="en")
            if location is not None and location.address:
                return location.address
        except Exception as exc:
            print(f"  [location_service] geopy reverse geocode failed: {exc}")

    params = urllib.parse.urlencode({
        "lat":            lat,
        "lon":            lon,
        "format":         "json",
        "addressdetails": 1,
    })
    url = f"{NOMINATIM_URL}?{params}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "AI-Traffic-Monitor/1.0 (streamlit-dashboard)"
    })
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())

        addr  = data.get("address", {})
        parts = []

        for field_key in ["road", "suburb", "city", "town", "village",
                          "state_district", "state", "country"]:
            val = addr.get(field_key)
            if val and val not in parts:
                parts.append(val)
            if len(parts) >= 4:
                break

        return ", ".join(parts) if parts else f"{lat:.5f}, {lon:.5f}"

    except Exception as exc:
        print(f"  [location_service] Nominatim error: {exc}")
        return f"{lat:.5f}, {lon:.5f}"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — IP-BASED FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def _ip_location() -> Optional[LocationInfo]:
    """
    Fetch approximate location from ip-api.com.
    Returns a LocationInfo with source="IP", or None on failure.
    """
    try:
        req = urllib.request.Request(IP_API_URL, headers={
            "User-Agent": "AI-Traffic-Monitor/1.0"
        })
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())

        if data.get("status") != "success":
            return None

        lat  = float(data.get("lat", 0))
        lon  = float(data.get("lon", 0))
        city = data.get("city", "")
        region = data.get("regionName", "")
        country = data.get("country", "")
        address = ", ".join(p for p in [city, region, country] if p)

        return LocationInfo(
            address   = address or f"{lat:.4f}, {lon:.4f}",
            latitude  = lat,
            longitude = lon,
            accuracy  = 0.0,      # IP location has no accuracy metric
            source    = "IP",
            raw       = data,
        )

    except Exception as exc:
        print(f"  [location_service] IP lookup error: {exc}")
        return None


def _fixed_fallback() -> LocationInfo:
    """Return the hardcoded fallback when everything else fails."""
    return LocationInfo(
        address   = FALLBACK_ADDRESS,
        latitude  = 12.9716,
        longitude = 80.2209,
        accuracy  = 0.0,
        source    = "fixed",
        raw       = {},
    )


def _pending_location() -> LocationInfo:
    """Return an intermediate placeholder while the browser is prompting for permission."""
    return LocationInfo(
        address   = "Detecting your location...",
        latitude  = 0.0,
        longitude = 0.0,
        accuracy  = 0.0,
        source    = "pending",
        raw       = {},
    )


def _unavailable_location(error: Optional[str] = None) -> LocationInfo:
    """Return a safe placeholder when the browser denied permission or GPS failed."""
    return LocationInfo(
        address   = "Location unavailable",
        latitude  = 0.0,
        longitude = 0.0,
        accuracy  = 0.0,
        source    = "unavailable",
        raw       = {"error": error} if error else {},
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def get_location() -> LocationInfo:
    """
    Attempt to get the user's real location in this order:
        1. Return cached value if already fetched this session.
        2. Inject GPS JS and wait for the browser to respond.
        3. Parse GPS result → reverse-geocode to an address.
        4. If GPS is denied / unavailable, fall back to IP-based lookup.
        5. If the IP lookup fails, use the hardcoded fallback.

    This function intentionally avoids using IP-based lookup while the browser
    is still prompting for permission, so the real GPS path is preferred.
    """
    cached = st.session_state.get(SESSION_KEY)
    if cached is not None and getattr(cached, "source", "") != "pending":
        _render_hidden_input()
        return cached

    payload_raw = _render_hidden_input()
    _inject_gps_js()

    loc = _parse_gps_payload(payload_raw)
    if loc is not None:
        if loc.source == "GPS":
            print(f"  [location_service] Browser permission granted: {loc.latitude:.6f}, {loc.longitude:.6f}")
            print(f"  [location_service] Address: {loc.address}")
            print(f"  [location_service] Google Maps: {maps_link(loc)}")
        st.session_state[SESSION_KEY] = loc
        return loc

    if payload_raw and payload_raw.strip() not in ("", "{}"):
        ip_loc = _ip_location()
        result = ip_loc if ip_loc else _fixed_fallback()
        st.session_state[SESSION_KEY] = result
        return result

    pending = _pending_location()
    st.session_state[SESSION_KEY] = pending
    return pending


def _render_hidden_input() -> str:
    """
    Render the hidden st.text_input that JS will write GPS data into.
    Returns the current value of the input (empty string initially).

    The input is visually hidden via the CSS rule injected at app startup.
    """
    # The label must match LABEL in the JS exactly
    val = st.text_input(
        label       = _GPS_INPUT_KEY,   # JS searches for aria-label == this
        value       = "",
        key         = _GPS_INPUT_KEY,
        label_visibility = "hidden",    # hidden from Streamlit UI
    )
    return val or ""


def _parse_gps_payload(raw: str) -> Optional[LocationInfo]:
    """
    Parse the JSON string posted by the browser JS.
    Returns a LocationInfo on success, or an unavailable placeholder if the
    browser denied permission or the GPS request failed.
    """
    raw = raw.strip()
    if not raw or raw == "{}":
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if data.get("error") or not data.get("ok"):
        err_text = data.get("error") or "Browser geolocation failed"
        print(f"  [location_service] GPS denied/failed: {err_text}")
        return _unavailable_location(err_text)

    lat      = float(data["lat"])
    lon      = float(data["lon"])
    accuracy = float(data.get("accuracy", 0))
    address  = _reverse_geocode(lat, lon)

    return LocationInfo(
        address   = address,
        latitude  = lat,
        longitude = lon,
        accuracy  = round(accuracy, 1),
        source    = "GPS",
        raw       = data,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — EMAIL / DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def maps_link(loc: LocationInfo) -> str:
    """
    Return a Google Maps URL for the detected coordinates.
    Example: https://maps.google.com/?q=12.9716,80.2209
    """
    if loc is None or getattr(loc, "source", "") in {"pending", "unavailable"}:
        return ""
    return f"https://maps.google.com/?q={loc.latitude},{loc.longitude}"


def format_for_email(loc: LocationInfo) -> str:
    """
    Returns a single-line string safe to embed in HTML email.
    Includes the address and a '(Source)' tag.
    """
    tag = {"GPS": "GPS", "IP": "IP-based", "fixed": "Fixed"}.get(loc.source, loc.source)
    return f"{loc.address}  [{tag}]"


def display_location_card(loc: LocationInfo):
    """
    Render a compact location info card inside Streamlit.
    Call this wherever you want to show the detected location in the UI.
    """
    src_color = {"GPS": "#2ECC71", "IP": "#F39C12", "fixed": "#888888", "pending": "#7EB8FF", "unavailable": "#E74C3C"}.get(
        loc.source, "#888888"
    )
    acc_text  = f"{loc.accuracy:.0f} m" if loc.accuracy > 0 else "N/A"
    maps_url  = maps_link(loc)
    address_text = loc.address if loc.address else "Location unavailable"

    st.markdown(f"""
    <div style="background:#1A1A30;border-radius:10px;border:1px solid #2A2A4A;
                padding:14px 18px;margin:8px 0;">
      <div style="display:flex;align-items:center;justify-content:space-between;
                  margin-bottom:10px;">
        <span style="color:#7EB8FF;font-weight:600;font-size:0.95rem;">
          📍 Current Location
        </span>
        <span style="background:{src_color}22;color:{src_color};
                     border:1px solid {src_color}44;
                     border-radius:5px;padding:2px 10px;font-size:0.78rem;
                     font-weight:600;">
          {loc.source_label}
        </span>
      </div>
      <div style="color:#E0E0F0;font-size:1rem;margin-bottom:10px;
                  line-height:1.5;">
        {address_text}
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;">
        <div style="background:#12122A;border-radius:6px;padding:8px 10px;">
          <div style="color:#8888AA;font-size:0.72rem;">Latitude</div>
          <div style="color:#7EB8FF;font-size:0.92rem;font-weight:600;">
            {loc.latitude:.6f}
          </div>
        </div>
        <div style="background:#12122A;border-radius:6px;padding:8px 10px;">
          <div style="color:#8888AA;font-size:0.72rem;">Longitude</div>
          <div style="color:#7EB8FF;font-size:0.92rem;font-weight:600;">
            {loc.longitude:.6f}
          </div>
        </div>
        <div style="background:#12122A;border-radius:6px;padding:8px 10px;">
          <div style="color:#8888AA;font-size:0.72rem;">Accuracy</div>
          <div style="color:#7EB8FF;font-size:0.92rem;font-weight:600;">
            {acc_text}
          </div>
        </div>
      </div>
      <div style="margin-top:10px;">
        <a href="{maps_url}" target="_blank"
           style="color:#5599FF;font-size:0.82rem;text-decoration:none;">
          🗺 View on Google Maps ↗
        </a>
      </div>
    </div>
    """, unsafe_allow_html=True)