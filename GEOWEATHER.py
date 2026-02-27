# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Streamlit app: NWS Forecast Reporter (supports "X miles north of <place>")

# Run:
#   pip install streamlit requests
#   # (recommended) set a real email to reduce Nominatim 403s:
#   # Windows PowerShell:
#   #   setx WEATHER_CONTACT_EMAIL "you@domain.com"
#   # New terminal after setx, or set for current session:
#   #   $env:WEATHER_CONTACT_EMAIL="you@domain.com"
#   streamlit run app.py

# Notes:
# - Primary geocoder: Nominatim (OSM). Fallback: Photon.
# - Forecast source: NWS api.weather.gov (U.S. locations only).
# """

# import os
# import re
# import time
# import math
# import requests
# import streamlit as st
# from textwrap import shorten

# # ───────────────────────────────────────────────────────────────────────────────
# # CONFIG
# # ───────────────────────────────────────────────────────────────────────────────
# CONTACT_EMAIL = os.getenv("WEATHER_CONTACT_EMAIL", "").strip()
# USER_AGENT = f"RCVFD-WeatherStreamlit/2.1 ({CONTACT_EMAIL or 'no-email-provided'})"
# NWS_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/geo+json, application/json"}

# # ───────────────────────────────────────────────────────────────────────────────
# # HTTP helpers
# # ───────────────────────────────────────────────────────────────────────────────
# def _get_json(url, params=None, headers=None, timeout=30):
#     r = requests.get(url, params=params, headers=headers or {}, timeout=timeout)
#     r.raise_for_status()
#     return r.json()

# # ───────────────────────────────────────────────────────────────────────────────
# # GEOCODERS
# # ───────────────────────────────────────────────────────────────────────────────
# def geocode_nominatim(query: str):
#     url = "https://nominatim.openstreetmap.org/search"
#     headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
#     params = {"q": query, "format": "json", "limit": 1, "addressdetails": 0}
#     if CONTACT_EMAIL:
#         params["email"] = CONTACT_EMAIL

#     r = requests.get(url, params=params, headers=headers, timeout=30)
#     if r.status_code == 403:
#         raise RuntimeError("Nominatim 403. Set WEATHER_CONTACT_EMAIL to a real email; avoid rapid requests.")
#     r.raise_for_status()

#     results = r.json()
#     if not results:
#         raise RuntimeError("Nominatim returned no results.")
#     lat = float(results[0]["lat"])
#     lon = float(results[0]["lon"])
#     name = results[0].get("display_name", query)
#     return lat, lon, name

# def geocode_photon(query: str):
#     """
#     Photon (OSM-based) forward geocoder. Often more permissive.
#     https://photon.komoot.io/
#     """
#     url = "https://photon.komoot.io/api/"
#     headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
#     params = {"q": query, "limit": 1}
#     data = _get_json(url, params=params, headers=headers)

#     feats = data.get("features") or []
#     if not feats:
#         raise RuntimeError("Photon returned no results.")
#     props = feats[0].get("properties") or {}
#     coords = feats[0].get("geometry", {}).get("coordinates") or [None, None]
#     lon, lat = coords[0], coords[1]
#     if lat is None or lon is None:
#         raise RuntimeError("Photon result missing coordinates.")

#     label_parts = []
#     for k in ("name", "city", "state", "country"):
#         v = props.get(k)
#         if v and v not in label_parts:
#             label_parts.append(v)
#     name = ", ".join(label_parts) if label_parts else query
#     return float(lat), float(lon), name

# @st.cache_data(ttl=60 * 60, show_spinner=False)  # 1 hour
# def geocode_place_cached(query: str):
#     # Never pass relative phrases here; resolve_location() handles that.
#     try:
#         return geocode_nominatim(query)
#     except Exception:
#         return geocode_photon(query)

# # ───────────────────────────────────────────────────────────────────────────────
# # NWS FORECAST
# # ───────────────────────────────────────────────────────────────────────────────
# @st.cache_data(ttl=10 * 60, show_spinner=False)  # 10 min
# def nws_points_cached(lat: float, lon: float):
#     url = f"https://api.weather.gov/points/{lat:.6f},{lon:.6f}"
#     return _get_json(url, headers=NWS_HEADERS)

# @st.cache_data(ttl=10 * 60, show_spinner=False)  # 10 min
# def nws_periods_cached(lat: float, lon: float, hourly: bool):
#     points = nws_points_cached(lat, lon)
#     props = points.get("properties", {})
#     forecast_url = props.get("forecastHourly") if hourly else props.get("forecast")
#     if not forecast_url:
#         raise RuntimeError("NWS points response missing forecast URL.")
#     fc = _get_json(forecast_url, headers=NWS_HEADERS)
#     periods = fc.get("properties", {}).get("periods", [])
#     return periods, forecast_url, props

# def first_daytime(periods):
#     for p in periods:
#         if p.get("isDaytime") is True:
#             return p
#     return periods[0] if periods else None

# # ───────────────────────────────────────────────────────────────────────────────
# # RELATIVE LOCATION PARSING + OFFSETS
# # ───────────────────────────────────────────────────────────────────────────────
# REL_RE = re.compile(
#     r"^\s*(?P<miles>\d+(?:\.\d+)?)\s*miles?\s*(?P<dir>north|south|east|west|n|s|e|w)\s*of\s*(?P<place>.+?)\s*$",
#     re.IGNORECASE,
# )

# def offset_latlon(lat: float, lon: float, miles: float, direction: str):
#     """
#     Offset a lat/lon by a given miles in a cardinal direction.
#     Spherical approximation:
#       1 deg lat  ~= 69.0 miles
#       1 deg lon  ~= 69.0 * cos(lat) miles
#     """
#     direction = direction.lower()
#     miles_per_deg_lat = 69.0
#     miles_per_deg_lon = 69.0 * max(0.01, abs(math.cos(math.radians(lat))))

#     dlat = 0.0
#     dlon = 0.0
#     if direction in ("north", "n"):
#         dlat = miles / miles_per_deg_lat
#     elif direction in ("south", "s"):
#         dlat = -miles / miles_per_deg_lat
#     elif direction in ("east", "e"):
#         dlon = miles / miles_per_deg_lon
#     elif direction in ("west", "w"):
#         dlon = -miles / miles_per_deg_lon
#     else:
#         raise ValueError(f"Unsupported direction: {direction}")

#     return lat + dlat, lon + dlon

# def resolve_location(description: str):
#     """
#     Returns (lat, lon, label).
#     Supports:
#       - "<miles> miles <dir> of <place>"  (north/south/east/west)
#       - "<place>"
#     """
#     s = (description or "").strip()
#     if not s:
#         raise ValueError("Location is empty.")

#     m = REL_RE.match(s)
#     if m:
#         miles = float(m.group("miles"))
#         direction = m.group("dir")
#         place = m.group("place").strip()

#         base_lat, base_lon, base_name = geocode_place_cached(place)
#         lat, lon = offset_latlon(base_lat, base_lon, miles, direction)
#         label = f"{miles:g} miles {direction.lower()} of {base_name}"
#         return lat, lon, label

#     lat, lon, name = geocode_place_cached(s)
#     return lat, lon, name

# # ───────────────────────────────────────────────────────────────────────────────
# # FORMATTING
# # ───────────────────────────────────────────────────────────────────────────────
# def _fmt_prob(p):
#     if isinstance(p, dict):
#         v = p.get("value", None)
#         return "—" if v is None else f"{int(round(v))}%"
#     return "—" if p is None else str(p)

# def _fmt_humidity(p):
#     rh = p.get("relativeHumidity")
#     if isinstance(rh, dict):
#         v = rh.get("value", None)
#         return "—" if v is None else f"{int(round(v))}%"
#     return "—"

# def _fmt_temp(p):
#     t = p.get("temperature")
#     u = p.get("temperatureUnit", "")
#     return "—" if t is None else f"{t}{u}"

# def _fmt_wind(p):
#     ws = p.get("windSpeed") or "—"
#     wd = p.get("windDirection") or ""
#     return (ws + (" " + wd if wd else "")).strip()

# def _wrap(text, width=92):
#     if not text:
#         return ""
#     words = text.split()
#     lines, cur, n = [], [], 0
#     for w in words:
#         if n + len(w) + (1 if cur else 0) > width:
#             lines.append(" ".join(cur))
#             cur = [w]
#             n = len(w)
#         else:
#             cur.append(w)
#             n += len(w) + (1 if len(cur) > 1 else 0)
#     if cur:
#         lines.append(" ".join(cur))
#     return "\n".join(lines)

# def build_report_text(description: str, include_hourly: bool, n_daily: int, n_hourly: int):
#     lat, lon, label = resolve_location(description)

#     daily_periods, daily_url, props = nws_periods_cached(lat, lon, hourly=False)
#     if not daily_periods:
#         raise RuntimeError("No daily forecast periods returned from NWS (are you in the U.S.?).")

#     current = daily_periods[0]
#     today_day = first_daytime(daily_periods)

#     hourly_periods, hourly_url = ([], None)
#     if include_hourly:
#         hourly_periods, hourly_url, _ = nws_periods_cached(lat, lon, hourly=True)

#     grid_id = props.get("gridId", "—")
#     grid_x = props.get("gridX", "—")
#     grid_y = props.get("gridY", "—")
#     cwa = props.get("cwa", "—")
#     radar = props.get("radarStation", "—")
#     rel_loc = props.get("relativeLocation", {}).get("properties", {})
#     near_city = rel_loc.get("city")
#     near_state = rel_loc.get("state")
#     near_str = f"{near_city}, {near_state}" if near_city and near_state else "—"

#     ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

#     out = []
#     out.append("=" * 96)
#     out.append("FULL WEATHER REPORT (NWS api.weather.gov)")
#     out.append("=" * 96)
#     out.append(f"Input location     : {description}")
#     out.append(f"Resolved location  : {label}")
#     out.append(f"Nearest named place: {near_str}")
#     out.append(f"Coordinates        : {lat:.6f}, {lon:.6f}")
#     out.append(f"NWS grid           : {grid_id} ({grid_x},{grid_y}) | CWA {cwa} | Radar {radar}")
#     out.append(f"Generated          : {ts}")
#     out.append("-" * 96)

#     if today_day:
#         out.append("HIGHLIGHT (first daytime period)")
#         out.append(f"  Period   : {today_day.get('name','—')}")
#         out.append(f"  Temp     : {_fmt_temp(today_day)}")
#         out.append(f"  Wind     : {_fmt_wind(today_day)}")
#         out.append(f"  POP      : {_fmt_prob(today_day.get('probabilityOfPrecipitation'))}")
#         out.append(f"  RH       : {_fmt_humidity(today_day)}")
#         out.append(f"  Summary  : {today_day.get('shortForecast','—')}")
#         det = today_day.get("detailedForecast", "")
#         if det:
#             out.append("  Details  :")
#             out.append("    " + _wrap(det, width=88).replace("\n", "\n    "))
#         out.append("-" * 96)

#     out.append(f"DAILY FORECAST (next {min(n_daily, len(daily_periods))} periods)")
#     header = f"{'Period':18} {'Temp':8} {'Wind':18} {'POP':6} {'RH':6} {'Short forecast'}"
#     out.append(header)
#     out.append("-" * len(header))
#     for p in daily_periods[:n_daily]:
#         period = (p.get("name") or "—")[:18]
#         temp = _fmt_temp(p)[:8]
#         wind = _fmt_wind(p)[:18]
#         pop = _fmt_prob(p.get("probabilityOfPrecipitation")).rjust(6)
#         rh = _fmt_humidity(p).rjust(6)
#         short_fc = shorten(p.get("shortForecast") or "—", width=52, placeholder="…")
#         out.append(f"{period:18} {temp:8} {wind:18} {pop:6} {rh:6} {short_fc}")
#     out.append("-" * 96)
#     out.append(f"Daily forecast URL : {daily_url}")
#     if hourly_url:
#         out.append(f"Hourly forecast URL: {hourly_url}")
#     out.append("-" * 96)

#     if include_hourly and hourly_periods:
#         out.append(f"HOURLY SNAPSHOT (next {min(n_hourly, len(hourly_periods))} hours)")
#         header = f"{'Start':20} {'Temp':8} {'Wind':18} {'POP':6} {'RH':6} {'Short forecast'}"
#         out.append(header)
#         out.append("-" * len(header))
#         for p in hourly_periods[:n_hourly]:
#             start = (p.get("startTime") or "—")[:20]
#             temp = _fmt_temp(p)[:8]
#             wind = _fmt_wind(p)[:18]
#             pop = _fmt_prob(p.get("probabilityOfPrecipitation")).rjust(6)
#             rh = _fmt_humidity(p).rjust(6)
#             short_fc = shorten(p.get("shortForecast") or "—", width=52, placeholder="…")
#             out.append(f"{start:20} {temp:8} {wind:18} {pop:6} {rh:6} {short_fc}")
#         out.append("-" * 96)

#     out.append("CURRENT PERIOD (daily period[0])")
#     out.append(f"  Period   : {current.get('name','—')}")
#     out.append(f"  Temp     : {_fmt_temp(current)}")
#     out.append(f"  Wind     : {_fmt_wind(current)}")
#     out.append(f"  POP      : {_fmt_prob(current.get('probabilityOfPrecipitation'))}")
#     out.append(f"  RH       : {_fmt_humidity(current)}")
#     out.append(f"  Summary  : {current.get('shortForecast','—')}")
#     det = current.get("detailedForecast", "")
#     if det:
#         out.append("  Details  :")
#         out.append("    " + _wrap(det, width=88).replace("\n", "\n    "))
#     out.append("=" * 96)

#     meta = {
#         "input": description,
#         "resolved_location": label,
#         "nearest_named_place": near_str,
#         "lat": lat,
#         "lon": lon,
#         "nws_grid": {"gridId": grid_id, "gridX": grid_x, "gridY": grid_y, "cwa": cwa, "radar": radar},
#         "daily_url": daily_url,
#         "hourly_url": hourly_url,
#         "generated_utc": ts,
#     }
#     return "\n".join(out), meta

# # ───────────────────────────────────────────────────────────────────────────────
# # STREAMLIT UI
# # ───────────────────────────────────────────────────────────────────────────────
# st.set_page_config(page_title="NWS Forecast Reporter", layout="centered")

# st.title("NWS Forecast Reporter")
# st.caption('Type a place, or a relative location like: "15 miles north of Miami, Florida"')

# if not CONTACT_EMAIL:
#     st.info("Tip: set WEATHER_CONTACT_EMAIL to a real email to reduce Nominatim 403s.")

# location = st.text_input("Location", value="15 miles north of Miami, Florida")

# col1, col2, col3 = st.columns([1, 1, 1])
# with col1:
#     include_hourly = st.checkbox("Include hourly", value=True)
# with col2:
#     n_daily = st.number_input("Daily periods", min_value=1, max_value=14, value=8, step=1)
# with col3:
#     n_hourly = st.number_input("Hourly hours", min_value=1, max_value=48, value=18, step=1)

# if st.button("Get forecast", type="primary"):
#     try:
#         with st.spinner("Fetching forecast..."):
#             text_report, meta = build_report_text(
#                 description=location,
#                 include_hourly=include_hourly,
#                 n_daily=int(n_daily),
#                 n_hourly=int(n_hourly),
#             )

#         st.success("Forecast ready.")
#         st.text_area("Forecast", value=text_report, height=520)

#         with st.expander("Details (resolved location + URLs)", expanded=False):
#             st.write(meta)

#     except requests.HTTPError as e:
#         st.error(f"HTTP error: {e}")
#     except Exception as e:
#         st.error(str(e))


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit app: NWS Forecast Reporter
- Geocoding: Photon primary (with proper User-Agent), Nominatim fallback (no email)
- Supports:
    "500 miles north of Fort Collins, CO"
    "200 km south of Fort Collins, CO"
    "10 mi SW of Boulder, CO"
    "200 miels N of Fort Collins, CO"  (common typo)
- Forecast: NWS api.weather.gov (U.S. only)

Run:
  pip install streamlit requests
  streamlit run app.py
"""

import re
import math
import time
import requests
import streamlit as st

USER_AGENT = "WeatherStreamlitApp/1.3"
COMMON_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}
NWS_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/geo+json, application/json"}

# ───────────────────────────────────────────────────────────────────────────────
# HTTP
# ───────────────────────────────────────────────────────────────────────────────
def _get_json(url, params=None, headers=None, timeout=30):
    r = requests.get(url, params=params, headers=headers or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ───────────────────────────────────────────────────────────────────────────────
# GEOCODING
# ───────────────────────────────────────────────────────────────────────────────
def geocode_photon(query: str):
    url = "https://photon.komoot.io/api/"
    params = {"q": query, "limit": 1}
    data = _get_json(url, params=params, headers=COMMON_HEADERS)

    feats = data.get("features") or []
    if not feats:
        raise RuntimeError("No location found (Photon).")

    feat = feats[0]
    props = feat.get("properties") or {}
    lon, lat = feat.get("geometry", {}).get("coordinates", [None, None])
    if lat is None or lon is None:
        raise RuntimeError("Photon result missing coordinates.")

    label_parts = []
    for k in ("name", "city", "state", "country"):
        v = props.get(k)
        if v and v not in label_parts:
            label_parts.append(v)
    label = ", ".join(label_parts) if label_parts else query

    return float(lat), float(lon), label

def geocode_nominatim(query: str):
    # No email usage here; just a proper User-Agent.
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 1, "addressdetails": 0}
    data = _get_json(url, params=params, headers=COMMON_HEADERS)

    if not data:
        raise RuntimeError("No location found (Nominatim).")

    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    name = data[0].get("display_name", query)
    return lat, lon, name

@st.cache_data(ttl=3600)
def geocode_place(query: str):
    # Photon first; fallback to Nominatim if Photon blocks/rate-limits.
    try:
        return geocode_photon(query)
    except Exception:
        return geocode_nominatim(query)

# ───────────────────────────────────────────────────────────────────────────────
# RELATIVE LOCATION SUPPORT (miles + kilometers)
# ───────────────────────────────────────────────────────────────────────────────
REL_RE = re.compile(
    r"""
    ^\s*
    (?P<distance>\d+(?:\.\d+)?)\s*
    (?P<unit>mi|mile|miles|miels|km|kms|kilometer|kilometers)\s*
    (?P<dir>north|south|east|west|n|s|e|w|ne|nw|se|sw|
            northeast|northwest|southeast|southwest)\s*
    of\s*
    (?P<place>.+?)\s*
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)

def _dir_to_bearing(dir_str: str) -> float:
    d = dir_str.strip().lower()
    mapping = {
        "n": 0.0, "north": 0.0,
        "ne": 45.0, "northeast": 45.0,
        "e": 90.0, "east": 90.0,
        "se": 135.0, "southeast": 135.0,
        "s": 180.0, "south": 180.0,
        "sw": 225.0, "southwest": 225.0,
        "w": 270.0, "west": 270.0,
        "nw": 315.0, "northwest": 315.0,
    }
    if d not in mapping:
        raise ValueError(f"Unsupported direction: {dir_str}")
    return mapping[d]

def _to_miles(distance: float, unit: str) -> float:
    u = unit.strip().lower()
    if u in ("km", "kms", "kilometer", "kilometers"):
        return distance * 0.621371
    return distance  # mi/mile/miles/miels -> miles

def offset_latlon_bearing(lat: float, lon: float, miles: float, bearing_deg: float):
    # simple approximation good for small/moderate offsets
    miles_per_deg_lat = 69.0
    dlat = (miles / miles_per_deg_lat) * math.cos(math.radians(bearing_deg))

    miles_per_deg_lon = 69.0 * max(0.01, abs(math.cos(math.radians(lat))))
    dlon = (miles / miles_per_deg_lon) * math.sin(math.radians(bearing_deg))

    return lat + dlat, lon + dlon

def resolve_location(description: str):
    s = (description or "").strip()
    if not s:
        raise ValueError("Location is empty.")

    m = REL_RE.match(s)
    if m:
        distance = float(m.group("distance"))
        unit = m.group("unit")
        direction = m.group("dir")
        place = m.group("place").strip()

        base_lat, base_lon, base_label = geocode_place(place)
        miles = _to_miles(distance, unit)
        bearing = _dir_to_bearing(direction)
        lat, lon = offset_latlon_bearing(base_lat, base_lon, miles, bearing)

        label = f"{distance:g} {unit} {direction.lower()} of {base_label}"
        return lat, lon, label

    return geocode_place(s)

# ───────────────────────────────────────────────────────────────────────────────
# NWS
# ───────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def nws_periods(lat, lon, hourly=False):
    points = _get_json(f"https://api.weather.gov/points/{lat:.6f},{lon:.6f}", headers=NWS_HEADERS)
    props = points["properties"]
    url = props["forecastHourly"] if hourly else props["forecast"]
    fc = _get_json(url, headers=NWS_HEADERS)
    return fc["properties"]["periods"], url

# ───────────────────────────────────────────────────────────────────────────────
# UI
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Weather", layout="centered")
st.title("Weather Forecast")

location = st.text_input("Enter location", "500 miles north of Fort Collins, CO")
include_hourly = st.checkbox("Include hourly forecast", True)

if st.button("Get Forecast", type="primary"):
    try:
        with st.spinner("Fetching forecast..."):
            lat, lon, label = resolve_location(location)
            daily, daily_url = nws_periods(lat, lon, hourly=False)
            hourly = []
            hourly_url = None
            if include_hourly:
                hourly, hourly_url = nws_periods(lat, lon, hourly=True)

        st.success(f"Resolved: {label} ({lat:.4f}, {lon:.4f})")

        st.subheader("Daily Forecast")
        for p in daily[:8]:
            st.write(
                f"**{p.get('name','—')}** — {p.get('temperature','—')}°{p.get('temperatureUnit','')} | "
                f"{p.get('windSpeed','—')} {p.get('windDirection','')} | "
                f"{p.get('shortForecast','—')}"
            )

        if include_hourly:
            st.subheader("Hourly Snapshot")
            for p in hourly[:12]:
                start = (p.get("startTime") or "—")[:16]
                st.write(
                    f"{start} — {p.get('temperature','—')}°{p.get('temperatureUnit','')} | "
                    f"{p.get('shortForecast','—')}"
                )

        with st.expander("Raw forecast URLs"):
            st.write("Daily:", daily_url)
            if hourly_url:
                st.write("Hourly:", hourly_url)

    except Exception as e:
        st.error(str(e))
