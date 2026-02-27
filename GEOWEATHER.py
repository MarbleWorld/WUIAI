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


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Streamlit app: NWS Forecast Reporter
# - Geocoding: Photon primary (with proper User-Agent), Nominatim fallback (no email)
# - Supports:
#     "500 miles north of Fort Collins, CO"
#     "200 km south of Fort Collins, CO"
#     "10 mi SW of Boulder, CO"
#     "200 miels N of Fort Collins, CO"  (common typo)
# - Forecast: NWS api.weather.gov (U.S. only)

# Run:
#   pip install streamlit requests
#   streamlit run app.py
# """

# import re
# import math
# import time
# import requests
# import streamlit as st

# USER_AGENT = "WeatherStreamlitApp/1.3"
# COMMON_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}
# NWS_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/geo+json, application/json"}

# # ───────────────────────────────────────────────────────────────────────────────
# # HTTP
# # ───────────────────────────────────────────────────────────────────────────────
# def _get_json(url, params=None, headers=None, timeout=30):
#     r = requests.get(url, params=params, headers=headers or {}, timeout=timeout)
#     r.raise_for_status()
#     return r.json()

# # ───────────────────────────────────────────────────────────────────────────────
# # GEOCODING
# # ───────────────────────────────────────────────────────────────────────────────
# def geocode_photon(query: str):
#     url = "https://photon.komoot.io/api/"
#     params = {"q": query, "limit": 1}
#     data = _get_json(url, params=params, headers=COMMON_HEADERS)

#     feats = data.get("features") or []
#     if not feats:
#         raise RuntimeError("No location found (Photon).")

#     feat = feats[0]
#     props = feat.get("properties") or {}
#     lon, lat = feat.get("geometry", {}).get("coordinates", [None, None])
#     if lat is None or lon is None:
#         raise RuntimeError("Photon result missing coordinates.")

#     label_parts = []
#     for k in ("name", "city", "state", "country"):
#         v = props.get(k)
#         if v and v not in label_parts:
#             label_parts.append(v)
#     label = ", ".join(label_parts) if label_parts else query

#     return float(lat), float(lon), label

# def geocode_nominatim(query: str):
#     # No email usage here; just a proper User-Agent.
#     url = "https://nominatim.openstreetmap.org/search"
#     params = {"q": query, "format": "json", "limit": 1, "addressdetails": 0}
#     data = _get_json(url, params=params, headers=COMMON_HEADERS)

#     if not data:
#         raise RuntimeError("No location found (Nominatim).")

#     lat = float(data[0]["lat"])
#     lon = float(data[0]["lon"])
#     name = data[0].get("display_name", query)
#     return lat, lon, name

# @st.cache_data(ttl=3600)
# def geocode_place(query: str):
#     # Photon first; fallback to Nominatim if Photon blocks/rate-limits.
#     try:
#         return geocode_photon(query)
#     except Exception:
#         return geocode_nominatim(query)

# # ───────────────────────────────────────────────────────────────────────────────
# # RELATIVE LOCATION SUPPORT (miles + kilometers)
# # ───────────────────────────────────────────────────────────────────────────────
# REL_RE = re.compile(
#     r"""
#     ^\s*
#     (?P<distance>\d+(?:\.\d+)?)\s*
#     (?P<unit>mi|mile|miles|miels|km|kms|kilometer|kilometers)\s*
#     (?P<dir>north|south|east|west|n|s|e|w|ne|nw|se|sw|
#             northeast|northwest|southeast|southwest)\s*
#     of\s*
#     (?P<place>.+?)\s*
#     $
#     """,
#     re.IGNORECASE | re.VERBOSE,
# )

# def _dir_to_bearing(dir_str: str) -> float:
#     d = dir_str.strip().lower()
#     mapping = {
#         "n": 0.0, "north": 0.0,
#         "ne": 45.0, "northeast": 45.0,
#         "e": 90.0, "east": 90.0,
#         "se": 135.0, "southeast": 135.0,
#         "s": 180.0, "south": 180.0,
#         "sw": 225.0, "southwest": 225.0,
#         "w": 270.0, "west": 270.0,
#         "nw": 315.0, "northwest": 315.0,
#     }
#     if d not in mapping:
#         raise ValueError(f"Unsupported direction: {dir_str}")
#     return mapping[d]

# def _to_miles(distance: float, unit: str) -> float:
#     u = unit.strip().lower()
#     if u in ("km", "kms", "kilometer", "kilometers"):
#         return distance * 0.621371
#     return distance  # mi/mile/miles/miels -> miles

# def offset_latlon_bearing(lat: float, lon: float, miles: float, bearing_deg: float):
#     # simple approximation good for small/moderate offsets
#     miles_per_deg_lat = 69.0
#     dlat = (miles / miles_per_deg_lat) * math.cos(math.radians(bearing_deg))

#     miles_per_deg_lon = 69.0 * max(0.01, abs(math.cos(math.radians(lat))))
#     dlon = (miles / miles_per_deg_lon) * math.sin(math.radians(bearing_deg))

#     return lat + dlat, lon + dlon

# def resolve_location(description: str):
#     s = (description or "").strip()
#     if not s:
#         raise ValueError("Location is empty.")

#     m = REL_RE.match(s)
#     if m:
#         distance = float(m.group("distance"))
#         unit = m.group("unit")
#         direction = m.group("dir")
#         place = m.group("place").strip()

#         base_lat, base_lon, base_label = geocode_place(place)
#         miles = _to_miles(distance, unit)
#         bearing = _dir_to_bearing(direction)
#         lat, lon = offset_latlon_bearing(base_lat, base_lon, miles, bearing)

#         label = f"{distance:g} {unit} {direction.lower()} of {base_label}"
#         return lat, lon, label

#     return geocode_place(s)

# # ───────────────────────────────────────────────────────────────────────────────
# # NWS
# # ───────────────────────────────────────────────────────────────────────────────
# @st.cache_data(ttl=600)
# def nws_periods(lat, lon, hourly=False):
#     points = _get_json(f"https://api.weather.gov/points/{lat:.6f},{lon:.6f}", headers=NWS_HEADERS)
#     props = points["properties"]
#     url = props["forecastHourly"] if hourly else props["forecast"]
#     fc = _get_json(url, headers=NWS_HEADERS)
#     return fc["properties"]["periods"], url

# # ───────────────────────────────────────────────────────────────────────────────
# # UI
# # ───────────────────────────────────────────────────────────────────────────────
# st.set_page_config(page_title="Weather", layout="centered")
# st.title("Weather Forecast")

# location = st.text_input("Enter location", "500 miles north of Fort Collins, CO")
# include_hourly = st.checkbox("Include hourly forecast", True)

# if st.button("Get Forecast", type="primary"):
#     try:
#         with st.spinner("Fetching forecast..."):
#             lat, lon, label = resolve_location(location)
#             daily, daily_url = nws_periods(lat, lon, hourly=False)
#             hourly = []
#             hourly_url = None
#             if include_hourly:
#                 hourly, hourly_url = nws_periods(lat, lon, hourly=True)

#         st.success(f"Resolved: {label} ({lat:.4f}, {lon:.4f})")

#         st.subheader("Daily Forecast")
#         for p in daily[:8]:
#             st.write(
#                 f"**{p.get('name','—')}** — {p.get('temperature','—')}°{p.get('temperatureUnit','')} | "
#                 f"{p.get('windSpeed','—')} {p.get('windDirection','')} | "
#                 f"{p.get('shortForecast','—')}"
#             )

#         if include_hourly:
#             st.subheader("Hourly Snapshot")
#             for p in hourly[:12]:
#                 start = (p.get("startTime") or "—")[:16]
#                 st.write(
#                     f"{start} — {p.get('temperature','—')}°{p.get('temperatureUnit','')} | "
#                     f"{p.get('shortForecast','—')}"
#                 )

#         with st.expander("Raw forecast URLs"):
#             st.write("Daily:", daily_url)
#             if hourly_url:
#                 st.write("Hourly:", hourly_url)

#     except Exception as e:
#         st.error(str(e))


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Streamlit app: FULL weather report (nicely formatted) like your CLI script,
# but as a simple UI:
#   - Text input ("500 miles north of Fort Collins, CO" / "200 km south of ...")
#   - Button: Get Forecast
#   - Output: one big formatted text report (daily + optional hourly snapshot)

# Geocoding:
#   - Photon primary (with proper User-Agent header)
#   - Nominatim fallback (with proper User-Agent header)
#   - NO email anywhere

# Forecast:
#   - NWS api.weather.gov (U.S. locations only)

# Run:
#   pip install streamlit requests
#   streamlit run app.py
# """

# import re
# import time
# import math
# import requests
# import streamlit as st
# from textwrap import shorten

# # ───────────────────────────────────────────────────────────────────────────────
# # CONFIG
# # ───────────────────────────────────────────────────────────────────────────────
# USER_AGENT = "WeatherStreamlitApp/2.0"
# COMMON_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}
# NWS_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/geo+json, application/json"}

# def _get_json(url, params=None, headers=None, timeout=30):
#     r = requests.get(url, params=params, headers=headers or {}, timeout=timeout)
#     r.raise_for_status()
#     return r.json()

# # ───────────────────────────────────────────────────────────────────────────────
# # GEOCODERS (Photon primary, Nominatim fallback) — NO EMAIL
# # ───────────────────────────────────────────────────────────────────────────────
# def geocode_photon(query: str):
#     url = "https://photon.komoot.io/api/"
#     params = {"q": query, "limit": 1}
#     data = _get_json(url, params=params, headers=COMMON_HEADERS)

#     feats = data.get("features") or []
#     if not feats:
#         raise RuntimeError("Photon returned no results.")

#     feat = feats[0]
#     props = feat.get("properties") or {}
#     lon, lat = feat.get("geometry", {}).get("coordinates", [None, None])
#     if lat is None or lon is None:
#         raise RuntimeError("Photon result missing coordinates.")

#     label_parts = []
#     for k in ("name", "city", "state", "country"):
#         v = props.get(k)
#         if v and v not in label_parts:
#             label_parts.append(v)
#     label = ", ".join(label_parts) if label_parts else query

#     return float(lat), float(lon), label

# def geocode_nominatim(query: str):
#     url = "https://nominatim.openstreetmap.org/search"
#     params = {"q": query, "format": "json", "limit": 1, "addressdetails": 0}
#     data = _get_json(url, params=params, headers=COMMON_HEADERS)

#     if not data:
#         raise RuntimeError("Nominatim returned no results.")

#     lat = float(data[0]["lat"])
#     lon = float(data[0]["lon"])
#     name = data[0].get("display_name", query)
#     return lat, lon, name

# @st.cache_data(ttl=3600)
# def geocode_place(query: str):
#     try:
#         return geocode_photon(query)
#     except Exception:
#         return geocode_nominatim(query)

# # ───────────────────────────────────────────────────────────────────────────────
# # NWS FORECAST
# # ───────────────────────────────────────────────────────────────────────────────
# @st.cache_data(ttl=600)
# def nws_points(lat, lon):
#     url = f"https://api.weather.gov/points/{lat:.6f},{lon:.6f}"
#     return _get_json(url, headers=NWS_HEADERS)

# @st.cache_data(ttl=600)
# def nws_periods(lat, lon, hourly=False):
#     points = nws_points(lat, lon)
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
# # RELATIVE LOCATION PARSING + OFFSETS (miles + km, cardinal + diagonals, typo "miels")
# # ───────────────────────────────────────────────────────────────────────────────
# REL_RE = re.compile(
#     r"""
#     ^\s*
#     (?P<distance>\d+(?:\.\d+)?)\s*
#     (?P<unit>mi|mile|miles|miels|km|kms|kilometer|kilometers)\s*
#     (?P<dir>north|south|east|west|n|s|e|w|ne|nw|se|sw|
#             northeast|northwest|southeast|southwest)\s*
#     of\s*
#     (?P<place>.+?)\s*
#     $
#     """,
#     re.IGNORECASE | re.VERBOSE,
# )

# def _dir_to_bearing(dir_str: str) -> float:
#     d = dir_str.strip().lower()
#     mapping = {
#         "n": 0.0, "north": 0.0,
#         "ne": 45.0, "northeast": 45.0,
#         "e": 90.0, "east": 90.0,
#         "se": 135.0, "southeast": 135.0,
#         "s": 180.0, "south": 180.0,
#         "sw": 225.0, "southwest": 225.0,
#         "w": 270.0, "west": 270.0,
#         "nw": 315.0, "northwest": 315.0,
#     }
#     if d not in mapping:
#         raise ValueError(f"Unsupported direction: {dir_str}")
#     return mapping[d]

# def _to_miles(distance: float, unit: str) -> float:
#     u = unit.strip().lower()
#     if u in ("km", "kms", "kilometer", "kilometers"):
#         return distance * 0.621371
#     return distance  # mi/mile/miles/miels -> miles

# def offset_latlon_bearing(lat: float, lon: float, miles: float, bearing_deg: float):
#     miles_per_deg_lat = 69.0
#     dlat = (miles / miles_per_deg_lat) * math.cos(math.radians(bearing_deg))

#     miles_per_deg_lon = 69.0 * max(0.01, abs(math.cos(math.radians(lat))))
#     dlon = (miles / miles_per_deg_lon) * math.sin(math.radians(bearing_deg))

#     return lat + dlat, lon + dlon

# def resolve_location(description: str):
#     s = (description or "").strip()
#     if not s:
#         raise ValueError("Location is empty.")

#     m = REL_RE.match(s)
#     if m:
#         distance = float(m.group("distance"))
#         unit = m.group("unit")
#         direction = m.group("dir")
#         place = m.group("place").strip()

#         base_lat, base_lon, base_name = geocode_place(place)
#         miles = _to_miles(distance, unit)
#         bearing = _dir_to_bearing(direction)
#         lat, lon = offset_latlon_bearing(base_lat, base_lon, miles, bearing)
#         label = f"{distance:g} {unit} {direction.lower()} of {base_name}"
#         return lat, lon, label

#     lat, lon, name = geocode_place(s)
#     return lat, lon, name

# # ───────────────────────────────────────────────────────────────────────────────
# # PRETTY FORMAT (same output style as your CLI print_full_report)
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

# def build_full_report_text(description: str, include_hourly=True, n_daily=8, n_hourly=18):
#     lat, lon, label = resolve_location(description)

#     daily_periods, daily_url, props = nws_periods(lat, lon, hourly=False)
#     if not daily_periods:
#         raise RuntimeError("No daily forecast periods returned from NWS (U.S. only).")

#     current = daily_periods[0]
#     today_day = first_daytime(daily_periods)

#     hourly_periods, hourly_url = ([], None)
#     if include_hourly:
#         hourly_periods, hourly_url, _ = nws_periods(lat, lon, hourly=True)

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
#     out.append("\n" + "=" * 96)
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
#     out.append("=" * 96 + "\n")

#     return "\n".join(out)

# # ───────────────────────────────────────────────────────────────────────────────
# # STREAMLIT UI
# # ───────────────────────────────────────────────────────────────────────────────
# st.set_page_config(page_title="Weather", layout="centered")
# st.title("Weather Forecast (Full Report)")

# location = st.text_input("Enter location", "500 miles north of Fort Collins, CO")

# c1, c2, c3 = st.columns([1, 1, 1])
# with c1:
#     include_hourly = st.checkbox("Include hourly", True)
# with c2:
#     n_daily = st.number_input("Daily periods", min_value=1, max_value=14, value=8, step=1)
# with c3:
#     n_hourly = st.number_input("Hourly hours", min_value=1, max_value=72, value=18, step=1)

# if st.button("Get Forecast", type="primary"):
#     try:
#         with st.spinner("Fetching forecast..."):
#             report_text = build_full_report_text(
#                 location,
#                 include_hourly=include_hourly,
#                 n_daily=int(n_daily),
#                 n_hourly=int(n_hourly),
#             )
#         st.text_area("Forecast report", value=report_text, height=600)
#     except Exception as e:
#         st.error(str(e))




# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Streamlit app: FULL WEATHER REPORT (exactly like your CLI printout style),
# but in Streamlit.

# UI:
#   - One text box for location
#   - One button: Get Forecast
#   - Output is ONE big formatted text report (same layout as your CLI report)
#   - No "daily periods" or "hourly hours" settings

# Geocoding:
#   - Photon primary (with User-Agent)
#   - Nominatim fallback (with User-Agent)
#   - NO email anywhere

# Relative locations:
#   - Supports miles + kilometers and diagonals + typo "miels"
#     e.g. "500 miles north of Fort Collins, CO"
#          "200 km NE of Fort Collins, CO"
#          "10 miels west of Denver, CO"

# Run:
#   pip install streamlit requests
#   streamlit run app.py
# """

# import re
# import time
# import math
# import requests
# import streamlit as st
# from textwrap import shorten

# USER_AGENT = "WeatherStreamlitApp/4.0"
# COMMON_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}
# NWS_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/geo+json, application/json"}

# # Fixed output choices (no UI settings)
# INCLUDE_HOURLY = True
# N_DAILY = 14       # show ALL daily periods typically returned (usually 14)
# N_HOURLY = 24      # show next 24 hours

# # ───────────────────────────────────────────────────────────────────────────────
# # HTTP
# # ───────────────────────────────────────────────────────────────────────────────
# def _get_json(url, params=None, headers=None, timeout=30):
#     r = requests.get(url, params=params, headers=headers or {}, timeout=timeout)
#     r.raise_for_status()
#     return r.json()

# # ───────────────────────────────────────────────────────────────────────────────
# # GEOCODING (Photon primary, Nominatim fallback) — NO EMAIL
# # ───────────────────────────────────────────────────────────────────────────────
# def geocode_photon(query: str):
#     url = "https://photon.komoot.io/api/"
#     params = {"q": query, "limit": 1}
#     data = _get_json(url, params=params, headers=COMMON_HEADERS)

#     feats = data.get("features") or []
#     if not feats:
#         raise RuntimeError("Photon returned no results.")

#     feat = feats[0]
#     props = feat.get("properties") or {}
#     lon, lat = feat.get("geometry", {}).get("coordinates", [None, None])
#     if lat is None or lon is None:
#         raise RuntimeError("Photon result missing coordinates.")

#     label_parts = []
#     for k in ("name", "city", "state", "country"):
#         v = props.get(k)
#         if v and v not in label_parts:
#             label_parts.append(v)
#     label = ", ".join(label_parts) if label_parts else query

#     return float(lat), float(lon), label

# def geocode_nominatim(query: str):
#     url = "https://nominatim.openstreetmap.org/search"
#     params = {"q": query, "format": "json", "limit": 1, "addressdetails": 0}
#     data = _get_json(url, params=params, headers=COMMON_HEADERS)

#     if not data:
#         raise RuntimeError("Nominatim returned no results.")

#     lat = float(data[0]["lat"])
#     lon = float(data[0]["lon"])
#     name = data[0].get("display_name", query)
#     return lat, lon, name

# @st.cache_data(ttl=3600)
# def geocode_place(query: str):
#     try:
#         return geocode_photon(query)
#     except Exception:
#         return geocode_nominatim(query)

# # ───────────────────────────────────────────────────────────────────────────────
# # RELATIVE LOCATION PARSING + OFFSETS (miles + km, diagonals, typo "miels")
# # ───────────────────────────────────────────────────────────────────────────────
# REL_RE = re.compile(
#     r"""
#     ^\s*
#     (?P<distance>\d+(?:\.\d+)?)\s*
#     (?P<unit>mi|mile|miles|miels|km|kms|kilometer|kilometers)\s*
#     (?P<dir>north|south|east|west|n|s|e|w|ne|nw|se|sw|
#             northeast|northwest|southeast|southwest)\s*
#     of\s*
#     (?P<place>.+?)\s*
#     $
#     """,
#     re.IGNORECASE | re.VERBOSE,
# )

# def _dir_to_bearing(dir_str: str) -> float:
#     d = dir_str.strip().lower()
#     mapping = {
#         "n": 0.0, "north": 0.0,
#         "ne": 45.0, "northeast": 45.0,
#         "e": 90.0, "east": 90.0,
#         "se": 135.0, "southeast": 135.0,
#         "s": 180.0, "south": 180.0,
#         "sw": 225.0, "southwest": 225.0,
#         "w": 270.0, "west": 270.0,
#         "nw": 315.0, "northwest": 315.0,
#     }
#     if d not in mapping:
#         raise ValueError(f"Unsupported direction: {dir_str}")
#     return mapping[d]

# def _to_miles(distance: float, unit: str) -> float:
#     u = unit.strip().lower()
#     if u in ("km", "kms", "kilometer", "kilometers"):
#         return distance * 0.621371
#     return distance

# def offset_latlon_bearing(lat: float, lon: float, miles: float, bearing_deg: float):
#     miles_per_deg_lat = 69.0
#     dlat = (miles / miles_per_deg_lat) * math.cos(math.radians(bearing_deg))

#     miles_per_deg_lon = 69.0 * max(0.01, abs(math.cos(math.radians(lat))))
#     dlon = (miles / miles_per_deg_lon) * math.sin(math.radians(bearing_deg))

#     return lat + dlat, lon + dlon

# def resolve_location(description: str):
#     s = (description or "").strip()
#     if not s:
#         raise ValueError("Location is empty.")

#     m = REL_RE.match(s)
#     if m:
#         distance = float(m.group("distance"))
#         unit = m.group("unit")
#         direction = m.group("dir")
#         place = m.group("place").strip()

#         base_lat, base_lon, base_name = geocode_place(place)
#         miles = _to_miles(distance, unit)
#         bearing = _dir_to_bearing(direction)
#         lat, lon = offset_latlon_bearing(base_lat, base_lon, miles, bearing)
#         label = f"{distance:g} {unit} {direction.lower()} of {base_name}"
#         return lat, lon, label

#     lat, lon, name = geocode_place(s)
#     return lat, lon, name

# # ───────────────────────────────────────────────────────────────────────────────
# # NWS FORECAST
# # ───────────────────────────────────────────────────────────────────────────────
# @st.cache_data(ttl=600)
# def nws_points(lat, lon):
#     return _get_json(f"https://api.weather.gov/points/{lat:.6f},{lon:.6f}", headers=NWS_HEADERS)

# @st.cache_data(ttl=600)
# def nws_periods(lat, lon, hourly=False):
#     points = nws_points(lat, lon)
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
# # PRETTY REPORT (returns a big string)
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

# def build_full_report_text(description: str):
#     lat, lon, label = resolve_location(description)

#     daily_periods, daily_url, props = nws_periods(lat, lon, hourly=False)
#     if not daily_periods:
#         raise RuntimeError("No daily forecast periods returned from NWS (U.S. only).")

#     current = daily_periods[0]
#     today_day = first_daytime(daily_periods)

#     hourly_periods, hourly_url = ([], None)
#     if INCLUDE_HOURLY:
#         hourly_periods, hourly_url, _ = nws_periods(lat, lon, hourly=True)

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
#     out.append("\n" + "=" * 96)
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

#     daily_show = daily_periods[:N_DAILY]
#     out.append(f"DAILY FORECAST (next {min(N_DAILY, len(daily_periods))} periods)")
#     header = f"{'Period':18} {'Temp':8} {'Wind':18} {'POP':6} {'RH':6} {'Short forecast'}"
#     out.append(header)
#     out.append("-" * len(header))
#     for p in daily_show:
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

#     if INCLUDE_HOURLY and hourly_periods:
#         hourly_show = hourly_periods[:N_HOURLY]
#         out.append(f"HOURLY SNAPSHOT (next {min(N_HOURLY, len(hourly_periods))} hours)")
#         header = f"{'Start':20} {'Temp':8} {'Wind':18} {'POP':6} {'RH':6} {'Short forecast'}"
#         out.append(header)
#         out.append("-" * len(header))
#         for p in hourly_show:
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
#     out.append("=" * 96 + "\n")

#     return "\n".join(out)

# # ───────────────────────────────────────────────────────────────────────────────
# # STREAMLIT UI
# # ───────────────────────────────────────────────────────────────────────────────
# st.set_page_config(page_title="Weather", layout="centered")
# st.title("Weather Forecast")

# location = st.text_input("Enter location", "500 miles north of Fort Collins, CO")

# if st.button("Get Forecast", type="primary"):
#     try:
#         with st.spinner("Fetching forecast..."):
#             report_text = build_full_report_text(location)
#         st.text_area("Forecast report", value=report_text, height=700)
#     except Exception as e:
#         st.error(str(e))






# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Streamlit app: FULL WEATHER REPORT (CLI-style big text report) + Geolocation.

# This version uses the NWS API forecast logic from your first script (nws_forecast),
# and keeps the Streamlit app structure + geocoding/relative-location parsing from your second script.

# Run:
#   pip install streamlit requests
#   streamlit run app.py
# """

# import re
# import time
# import math
# import requests
# import streamlit as st
# from textwrap import shorten

# # ───────────────────────────────────────────────────────────────────────────────
# # HEADERS
# # ───────────────────────────────────────────────────────────────────────────────
# USER_AGENT = "RCVFD-WeatherStreamlit/1.0"
# COMMON_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}
# NWS_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/geo+json, application/json"}

# # Fixed output choices (no UI settings)
# INCLUDE_HOURLY = True
# N_DAILY = 14
# N_HOURLY = 24

# # ───────────────────────────────────────────────────────────────────────────────
# # HTTP
# # ───────────────────────────────────────────────────────────────────────────────
# def _get_json(url, params=None, headers=None, timeout=30):
#     r = requests.get(url, params=params, headers=headers or {}, timeout=timeout)
#     r.raise_for_status()
#     return r.json()

# # ───────────────────────────────────────────────────────────────────────────────
# # GEOCODING (Photon primary, Nominatim fallback) — NO EMAIL
# # ───────────────────────────────────────────────────────────────────────────────
# def geocode_photon(query: str):
#     url = "https://photon.komoot.io/api/"
#     params = {"q": query, "limit": 1}
#     data = _get_json(url, params=params, headers=COMMON_HEADERS)

#     feats = data.get("features") or []
#     if not feats:
#         raise RuntimeError("Photon returned no results.")

#     feat = feats[0]
#     props = feat.get("properties") or {}
#     lon, lat = feat.get("geometry", {}).get("coordinates", [None, None])
#     if lat is None or lon is None:
#         raise RuntimeError("Photon result missing coordinates.")

#     label_parts = []
#     for k in ("name", "city", "state", "country"):
#         v = props.get(k)
#         if v and v not in label_parts:
#             label_parts.append(v)
#     label = ", ".join(label_parts) if label_parts else query

#     return float(lat), float(lon), label

# def geocode_nominatim(query: str):
#     url = "https://nominatim.openstreetmap.org/search"
#     params = {"q": query, "format": "json", "limit": 1, "addressdetails": 0}
#     data = _get_json(url, params=params, headers=COMMON_HEADERS)

#     if not data:
#         raise RuntimeError("Nominatim returned no results.")

#     lat = float(data[0]["lat"])
#     lon = float(data[0]["lon"])
#     name = data[0].get("display_name", query)
#     return lat, lon, name

# @st.cache_data(ttl=3600)
# def geocode_place(query: str):
#     try:
#         return geocode_photon(query)
#     except Exception:
#         return geocode_nominatim(query)

# # ───────────────────────────────────────────────────────────────────────────────
# # RELATIVE LOCATION PARSING + OFFSETS (miles + km, diagonals, typo "miels")
# # ───────────────────────────────────────────────────────────────────────────────
# REL_RE = re.compile(
#     r"""
#     ^\s*
#     (?P<distance>\d+(?:\.\d+)?)\s*
#     (?P<unit>mi|mile|miles|miels|km|kms|kilometer|kilometers)\s*
#     (?P<dir>north|south|east|west|n|s|e|w|ne|nw|se|sw|
#             northeast|northwest|southeast|southwest)\s*
#     of\s*
#     (?P<place>.+?)\s*
#     $
#     """,
#     re.IGNORECASE | re.VERBOSE,
# )

# def _dir_to_bearing(dir_str: str) -> float:
#     d = dir_str.strip().lower()
#     mapping = {
#         "n": 0.0, "north": 0.0,
#         "ne": 45.0, "northeast": 45.0,
#         "e": 90.0, "east": 90.0,
#         "se": 135.0, "southeast": 135.0,
#         "s": 180.0, "south": 180.0,
#         "sw": 225.0, "southwest": 225.0,
#         "w": 270.0, "west": 270.0,
#         "nw": 315.0, "northwest": 315.0,
#     }
#     if d not in mapping:
#         raise ValueError(f"Unsupported direction: {dir_str}")
#     return mapping[d]

# def _to_miles(distance: float, unit: str) -> float:
#     u = unit.strip().lower()
#     if u in ("km", "kms", "kilometer", "kilometers"):
#         return distance * 0.621371
#     return distance

# def offset_latlon_bearing(lat: float, lon: float, miles: float, bearing_deg: float):
#     miles_per_deg_lat = 69.0
#     dlat = (miles / miles_per_deg_lat) * math.cos(math.radians(bearing_deg))

#     miles_per_deg_lon = 69.0 * max(0.01, abs(math.cos(math.radians(lat))))
#     dlon = (miles / miles_per_deg_lon) * math.sin(math.radians(bearing_deg))

#     return lat + dlat, lon + dlon

# def resolve_location(description: str):
#     s = (description or "").strip()
#     if not s:
#         raise ValueError("Location is empty.")

#     m = REL_RE.match(s)
#     if m:
#         distance = float(m.group("distance"))
#         unit = m.group("unit")
#         direction = m.group("dir")
#         place = m.group("place").strip()

#         base_lat, base_lon, base_name = geocode_place(place)
#         miles = _to_miles(distance, unit)
#         bearing = _dir_to_bearing(direction)
#         lat, lon = offset_latlon_bearing(base_lat, base_lon, miles, bearing)
#         label = f"{distance:g} {unit} {direction.lower()} of {base_name}"
#         return lat, lon, label

#     lat, lon, name = geocode_place(s)
#     return lat, lon, name

# # ───────────────────────────────────────────────────────────────────────────────
# # NWS FORECAST (from your first script logic)
# # ───────────────────────────────────────────────────────────────────────────────
# def _req_json(url, headers=None, timeout=30):
#     r = requests.get(url, headers=headers or {}, timeout=timeout)
#     r.raise_for_status()
#     return r.json()

# @st.cache_data(ttl=600)
# def nws_forecast(lat, lon, hourly=False):
#     """
#     Returns a dict with NWS forecast periods.
#     """
#     points_url = f"https://api.weather.gov/points/{lat:.6f},{lon:.6f}"
#     points = _req_json(points_url, headers=NWS_HEADERS)
#     props = points.get("properties", {})
#     forecast_url = props.get("forecastHourly") if hourly else props.get("forecast")
#     if not forecast_url:
#         raise RuntimeError("NWS points response missing forecast URL.")

#     fc = _req_json(forecast_url, headers=NWS_HEADERS)
#     periods = fc.get("properties", {}).get("periods", [])
#     return {
#         "source": "NWS api.weather.gov",
#         "lat": lat,
#         "lon": lon,
#         "hourly": bool(hourly),
#         "forecast_url": forecast_url,
#         "periods": periods,
#         "points_properties": props,
#         "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
#     }

# def first_daytime(periods):
#     for p in periods:
#         if p.get("isDaytime") is True:
#             return p
#     return periods[0] if periods else None

# # ───────────────────────────────────────────────────────────────────────────────
# # PRETTY REPORT (returns a big string)
# # ───────────────────────────────────────────────────────────────────────────────
# def _fmt_prob(p):
#     if isinstance(p, dict):
#         v = p.get("value", None)
#         return "—" if v is None else f"{int(round(v))}%"
#     return "—" if p is None else str(p)

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

# def build_full_report_text(description: str):
#     lat, lon, label = resolve_location(description)

#     daily = nws_forecast(lat, lon, hourly=False)
#     daily_periods = daily.get("periods") or []
#     if not daily_periods:
#         raise RuntimeError("No daily forecast periods returned from NWS (U.S. only).")

#     props = daily.get("points_properties") or {}
#     daily_url = daily.get("forecast_url")

#     current = daily_periods[0]
#     today_day = first_daytime(daily_periods)

#     hourly_periods, hourly_url = ([], None)
#     if INCLUDE_HOURLY:
#         hourly = nws_forecast(lat, lon, hourly=True)
#         hourly_periods = hourly.get("periods") or []
#         hourly_url = hourly.get("forecast_url")

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
#     out.append("\n" + "=" * 96)
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
#         out.append(f"  Summary  : {today_day.get('shortForecast','—')}")
#         det = today_day.get("detailedForecast", "")
#         if det:
#             out.append("  Details  :")
#             out.append("    " + _wrap(det, width=88).replace("\n", "\n    "))
#         out.append("-" * 96)

#     daily_show = daily_periods[:N_DAILY]
#     out.append(f"DAILY FORECAST (next {min(N_DAILY, len(daily_periods))} periods)")
#     header = f"{'Period':18} {'Temp':8} {'Wind':18} {'POP':6} {'Short forecast'}"
#     out.append(header)
#     out.append("-" * len(header))
#     for p in daily_show:
#         period = (p.get("name") or "—")[:18]
#         temp = _fmt_temp(p)[:8]
#         wind = _fmt_wind(p)[:18]
#         pop = _fmt_prob(p.get("probabilityOfPrecipitation")).rjust(6)
#         short_fc = shorten(p.get("shortForecast") or "—", width=62, placeholder="…")
#         out.append(f"{period:18} {temp:8} {wind:18} {pop:6} {short_fc}")
#     out.append("-" * 96)
#     out.append(f"Daily forecast URL : {daily_url}")
#     if hourly_url:
#         out.append(f"Hourly forecast URL: {hourly_url}")
#     out.append("-" * 96)

#     if INCLUDE_HOURLY and hourly_periods:
#         hourly_show = hourly_periods[:N_HOURLY]
#         out.append(f"HOURLY SNAPSHOT (next {min(N_HOURLY, len(hourly_periods))} hours)")
#         header = f"{'Start':20} {'Temp':8} {'Wind':18} {'POP':6} {'Short forecast'}"
#         out.append(header)
#         out.append("-" * len(header))
#         for p in hourly_show:
#             start = (p.get("startTime") or "—")[:20]
#             temp = _fmt_temp(p)[:8]
#             wind = _fmt_wind(p)[:18]
#             pop = _fmt_prob(p.get("probabilityOfPrecipitation")).rjust(6)
#             short_fc = shorten(p.get("shortForecast") or "—", width=62, placeholder="…")
#             out.append(f"{start:20} {temp:8} {wind:18} {pop:6} {short_fc}")
#         out.append("-" * 96)

#     out.append("CURRENT PERIOD (daily period[0])")
#     out.append(f"  Period   : {current.get('name','—')}")
#     out.append(f"  Temp     : {_fmt_temp(current)}")
#     out.append(f"  Wind     : {_fmt_wind(current)}")
#     out.append(f"  POP      : {_fmt_prob(current.get('probabilityOfPrecipitation'))}")
#     out.append(f"  Summary  : {current.get('shortForecast','—')}")
#     det = current.get("detailedForecast", "")
#     if det:
#         out.append("  Details  :")
#         out.append("    " + _wrap(det, width=88).replace("\n", "\n    "))
#     out.append("=" * 96 + "\n")

#     return "\n".join(out)

# # ───────────────────────────────────────────────────────────────────────────────
# # STREAMLIT UI
# # ───────────────────────────────────────────────────────────────────────────────
# st.set_page_config(page_title="Weather", layout="centered")
# st.title("Weather Forecast")

# location = st.text_input("Enter location", "500 miles north of Fort Collins, CO")

# if st.button("Get Forecast", type="primary"):
#     try:
#         with st.spinner("Fetching forecast..."):
#             report_text = build_full_report_text(location)
#         st.text_area("Forecast report", value=report_text, height=700)
#     except Exception as e:
#         st.error(str(e))






# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Streamlit app: FULL WEATHER REPORT (your big CLI-style report)
# PLUS:
#   - the small "=== NWS Daily Forecast (first 6 periods) ===" block
#   - the "=== ChatGPT Web Forecast ===" block (optional; requires OPENAI_API_KEY + Responses API web_search tool)

# Geocoding:
#   - Photon primary (with User-Agent)
#   - Nominatim fallback (with User-Agent)
#   - NO email anywhere

# Relative locations:
#   - Supports miles + kilometers and diagonals + typo "miels"
#     e.g. "500 miles north of Fort Collins, CO"
#          "200 km NE of Fort Collins, CO"
#          "10 miels west of Denver, CO"

# Run:
#   pip install streamlit requests openai
#   streamlit run app.py

# Notes:
#   - If OPENAI_API_KEY is not set (or web_search isn't enabled), the app still works (NWS-only).
#   - NWS alerts are also included via api.weather.gov/alerts/active?point=lat,lon (this replaces the “Red Flag Warning” style content you showed).
# """

# import os
# import re
# import json
# import time
# import math
# import requests
# import streamlit as st
# from textwrap import shorten

# # Optional OpenAI (web_search)
# try:
#     from openai import OpenAI
# except Exception:
#     OpenAI = None

# # ───────────────────────────────────────────────────────────────────────────────
# # HEADERS
# # ───────────────────────────────────────────────────────────────────────────────
# USER_AGENT = "RCVFD-WeatherStreamlit/1.1"
# COMMON_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}
# NWS_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/geo+json, application/json"}

# # Output settings (fixed; no UI toggles)
# INCLUDE_HOURLY = True
# N_DAILY = 14
# N_HOURLY = 24
# NWS_FIRST6 = 6

# # OpenAI web_search settings
# DEFAULT_WEB_DAYS = 2
# DEFAULT_WEB_MODEL = "gpt-4.1-mini"

# # ───────────────────────────────────────────────────────────────────────────────
# # HTTP
# # ───────────────────────────────────────────────────────────────────────────────
# def _get_json(url, params=None, headers=None, timeout=30):
#     r = requests.get(url, params=params, headers=headers or {}, timeout=timeout)
#     r.raise_for_status()
#     return r.json()

# def _req_json(url, headers=None, timeout=30):
#     r = requests.get(url, headers=headers or {}, timeout=timeout)
#     r.raise_for_status()
#     return r.json()

# # ───────────────────────────────────────────────────────────────────────────────
# # GEOCODING (Photon primary, Nominatim fallback) — NO EMAIL
# # ───────────────────────────────────────────────────────────────────────────────
# def geocode_photon(query: str):
#     url = "https://photon.komoot.io/api/"
#     params = {"q": query, "limit": 1}
#     data = _get_json(url, params=params, headers=COMMON_HEADERS)

#     feats = data.get("features") or []
#     if not feats:
#         raise RuntimeError("Photon returned no results.")

#     feat = feats[0]
#     props = feat.get("properties") or {}
#     lon, lat = feat.get("geometry", {}).get("coordinates", [None, None])
#     if lat is None or lon is None:
#         raise RuntimeError("Photon result missing coordinates.")

#     label_parts = []
#     for k in ("name", "city", "state", "country"):
#         v = props.get(k)
#         if v and v not in label_parts:
#             label_parts.append(v)
#     label = ", ".join(label_parts) if label_parts else query

#     return float(lat), float(lon), label

# def geocode_nominatim(query: str):
#     url = "https://nominatim.openstreetmap.org/search"
#     params = {"q": query, "format": "json", "limit": 1, "addressdetails": 0}
#     data = _get_json(url, params=params, headers=COMMON_HEADERS)

#     if not data:
#         raise RuntimeError("Nominatim returned no results.")

#     lat = float(data[0]["lat"])
#     lon = float(data[0]["lon"])
#     name = data[0].get("display_name", query)
#     return lat, lon, name

# @st.cache_data(ttl=3600)
# def geocode_place(query: str):
#     try:
#         return geocode_photon(query)
#     except Exception:
#         return geocode_nominatim(query)

# # ───────────────────────────────────────────────────────────────────────────────
# # RELATIVE LOCATION PARSING + OFFSETS (miles + km, diagonals, typo "miels")
# # ───────────────────────────────────────────────────────────────────────────────
# REL_RE = re.compile(
#     r"""
#     ^\s*
#     (?P<distance>\d+(?:\.\d+)?)\s*
#     (?P<unit>mi|mile|miles|miels|km|kms|kilometer|kilometers)\s*
#     (?P<dir>north|south|east|west|n|s|e|w|ne|nw|se|sw|
#             northeast|northwest|southeast|southwest)\s*
#     of\s*
#     (?P<place>.+?)\s*
#     $
#     """,
#     re.IGNORECASE | re.VERBOSE,
# )

# def _dir_to_bearing(dir_str: str) -> float:
#     d = dir_str.strip().lower()
#     mapping = {
#         "n": 0.0, "north": 0.0,
#         "ne": 45.0, "northeast": 45.0,
#         "e": 90.0, "east": 90.0,
#         "se": 135.0, "southeast": 135.0,
#         "s": 180.0, "south": 180.0,
#         "sw": 225.0, "southwest": 225.0,
#         "w": 270.0, "west": 270.0,
#         "nw": 315.0, "northwest": 315.0,
#     }
#     if d not in mapping:
#         raise ValueError(f"Unsupported direction: {dir_str}")
#     return mapping[d]

# def _to_miles(distance: float, unit: str) -> float:
#     u = unit.strip().lower()
#     if u in ("km", "kms", "kilometer", "kilometers"):
#         return distance * 0.621371
#     return distance

# def offset_latlon_bearing(lat: float, lon: float, miles: float, bearing_deg: float):
#     miles_per_deg_lat = 69.0
#     dlat = (miles / miles_per_deg_lat) * math.cos(math.radians(bearing_deg))

#     miles_per_deg_lon = 69.0 * max(0.01, abs(math.cos(math.radians(lat))))
#     dlon = (miles / miles_per_deg_lon) * math.sin(math.radians(bearing_deg))

#     return lat + dlat, lon + dlon

# def resolve_location(description: str):
#     s = (description or "").strip()
#     if not s:
#         raise ValueError("Location is empty.")

#     m = REL_RE.match(s)
#     if m:
#         distance = float(m.group("distance"))
#         unit = m.group("unit")
#         direction = m.group("dir")
#         place = m.group("place").strip()

#         base_lat, base_lon, base_name = geocode_place(place)
#         miles = _to_miles(distance, unit)
#         bearing = _dir_to_bearing(direction)
#         lat, lon = offset_latlon_bearing(base_lat, base_lon, miles, bearing)
#         label = f"{distance:g} {unit} {direction.lower()} of {base_name}"
#         return lat, lon, label

#     lat, lon, name = geocode_place(s)
#     return lat, lon, name

# # ───────────────────────────────────────────────────────────────────────────────
# # NWS FORECAST (A) + ALERTS
# # ───────────────────────────────────────────────────────────────────────────────
# @st.cache_data(ttl=600)
# def nws_forecast(lat, lon, hourly=False):
#     points_url = f"https://api.weather.gov/points/{lat:.6f},{lon:.6f}"
#     points = _req_json(points_url, headers=NWS_HEADERS)
#     props = points.get("properties", {})
#     forecast_url = props.get("forecastHourly") if hourly else props.get("forecast")
#     if not forecast_url:
#         raise RuntimeError("NWS points response missing forecast URL.")

#     fc = _req_json(forecast_url, headers=NWS_HEADERS)
#     periods = fc.get("properties", {}).get("periods", [])
#     return {
#         "source": "NWS api.weather.gov",
#         "lat": lat,
#         "lon": lon,
#         "hourly": bool(hourly),
#         "forecast_url": forecast_url,
#         "periods": periods,
#         "points_properties": props,
#         "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
#     }

# @st.cache_data(ttl=300)
# def nws_alerts(lat, lon):
#     # Active alerts by point
#     url = "https://api.weather.gov/alerts/active"
#     params = {"point": f"{lat:.6f},{lon:.6f}"}
#     data = _get_json(url, params=params, headers=NWS_HEADERS)
#     feats = data.get("features") or []
#     alerts = []
#     for f in feats:
#         p = (f.get("properties") or {})
#         alerts.append({
#             "event": p.get("event"),
#             "headline": p.get("headline"),
#             "severity": p.get("severity"),
#             "certainty": p.get("certainty"),
#             "urgency": p.get("urgency"),
#             "effective": p.get("effective"),
#             "onset": p.get("onset"),
#             "ends": p.get("ends"),
#             "expires": p.get("expires"),
#             "senderName": p.get("senderName"),
#             "description": p.get("description"),
#             "instruction": p.get("instruction"),
#             "web": p.get("web"),
#             "areaDesc": p.get("areaDesc"),
#         })
#     return alerts

# def first_daytime(periods):
#     for p in periods:
#         if p.get("isDaytime") is True:
#             return p
#     return periods[0] if periods else None

# # ───────────────────────────────────────────────────────────────────────────────
# # OpenAI web_search (B) - optional
# # ───────────────────────────────────────────────────────────────────────────────
# def _openai_client():
#     api_key = os.getenv("OPENAI_API_KEY", "").strip()
#     if not api_key or OpenAI is None:
#         return None
#     return OpenAI(api_key=api_key)

# @st.cache_data(ttl=900)
# def chatgpt_web_weather(lat, lon, days=DEFAULT_WEB_DAYS, model=DEFAULT_WEB_MODEL):
#     client = _openai_client()
#     if client is None:
#         return {"ok": False, "reason": "OPENAI_API_KEY not set or openai package unavailable.", "data": None, "raw": ""}

#     prompt = f"""
# Look up the weather forecast for coordinates ({lat:.6f}, {lon:.6f}) for the next {days} days.
# Use web search. Prefer authoritative sources (NWS, NOAA, official forecast pages).
# Return STRICT JSON with:
# {{
#   "location_name": "<best guess place name>",
#   "forecast_summary": "<short summary>",
#   "high_level_hazards": ["<wind>", "<snow>", "<red flag>", "..."],
#   "periods": [
#     {{
#       "name": "<e.g., Today, Tonight, Mon>",
#       "temp": "<value + units if available>",
#       "wind": "<value + units/direction if available>",
#       "precip": "<if available>",
#       "summary": "<one sentence>"
#     }}
#   ]
# }}
# """.strip()

#     resp = client.responses.create(
#         model=model,
#         input=prompt,
#         tools=[{"type": "web_search"}],
#     )

#     out_text = getattr(resp, "output_text", None)
#     if not out_text:
#         out_text_parts = []
#         for item in getattr(resp, "output", []) or []:
#             if getattr(item, "type", "") == "message":
#                 for c in getattr(item, "content", []) or []:
#                     if getattr(c, "type", "") in ("output_text", "text"):
#                         out_text_parts.append(getattr(c, "text", ""))
#         out_text = "\n".join([t for t in out_text_parts if t]).strip()

#     try:
#         data = json.loads(out_text)
#         return {"ok": True, "data": data, "raw": out_text}
#     except Exception:
#         return {"ok": True, "data": None, "raw": out_text}

# # ───────────────────────────────────────────────────────────────────────────────
# # PRETTY REPORT
# # ───────────────────────────────────────────────────────────────────────────────
# def _fmt_prob(p):
#     if isinstance(p, dict):
#         v = p.get("value", None)
#         return "—" if v is None else f"{int(round(v))}%"
#     return "—" if p is None else str(p)

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

# def _fmt_iso(iso_str):
#     if not iso_str:
#         return "—"
#     # keep as-is (ISO), but strip trailing Z if present
#     return str(iso_str)

# def build_full_report_text(description: str, include_web=True):
#     lat, lon, label = resolve_location(description)

#     daily = nws_forecast(lat, lon, hourly=False)
#     daily_periods = daily.get("periods") or []
#     if not daily_periods:
#         raise RuntimeError("No daily forecast periods returned from NWS (U.S. only).")

#     props = daily.get("points_properties") or {}
#     daily_url = daily.get("forecast_url")

#     current = daily_periods[0]
#     today_day = first_daytime(daily_periods)

#     hourly_periods, hourly_url = ([], None)
#     if INCLUDE_HOURLY:
#         hourly = nws_forecast(lat, lon, hourly=True)
#         hourly_periods = hourly.get("periods") or []
#         hourly_url = hourly.get("forecast_url")

#     alerts = nws_alerts(lat, lon)

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

#     # --- This is the missing block you wanted back ---
#     out.append("\n=== NWS Daily Forecast (first 6 periods) ===")
#     for p in daily_periods[:NWS_FIRST6]:
#         out.append(
#             f"- {p.get('name','—')}: "
#             f"{p.get('temperature','—')}{p.get('temperatureUnit','')} | "
#             f"{p.get('windSpeed','—')} {p.get('windDirection','')} | "
#             f"{p.get('shortForecast','—')}"
#         )

#     # Optional OpenAI web_search block
#     if include_web:
#         res = chatgpt_web_weather(lat, lon, days=DEFAULT_WEB_DAYS, model=DEFAULT_WEB_MODEL)
#         out.append("\n=== ChatGPT Web Forecast ===")
#         if not res.get("ok", False):
#             out.append(f"[web_search unavailable] {res.get('reason','')}")
#         else:
#             if res.get("data") is not None:
#                 out.append(json.dumps(res["data"], indent=2))
#             else:
#                 out.append(res.get("raw", "").strip() or "[No text returned]")

#     # Main report (your big block)
#     out.append("\n" + "=" * 96)
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
#         out.append(f"  Summary  : {today_day.get('shortForecast','—')}")
#         det = today_day.get("detailedForecast", "")
#         if det:
#             out.append("  Details  :")
#             out.append("    " + _wrap(det, width=88).replace("\n", "\n    "))
#         out.append("-" * 96)

#     # Alerts section (gives you Red Flag Warning–style content from NWS directly)
#     out.append("SEVERE WEATHER ALERTS (NWS alerts/active)")
#     if not alerts:
#         out.append("  None.")
#     else:
#         for i, a in enumerate(alerts[:8], start=1):
#             out.append(f"* [{i}] {a.get('event','—')}")
#             if a.get("headline"):
#                 out.append(f"  Headline : {a.get('headline')}")
#             if a.get("areaDesc"):
#                 out.append(f"  Area     : {a.get('areaDesc')}")
#             out.append(f"  Severity : {a.get('severity','—')} | Urgency {a.get('urgency','—')} | Certainty {a.get('certainty','—')}")
#             out.append(f"  Effective: {_fmt_iso(a.get('effective'))}")
#             out.append(f"  Onset    : {_fmt_iso(a.get('onset'))}")
#             out.append(f"  Ends     : {_fmt_iso(a.get('ends'))}")
#             out.append(f"  Expires  : {_fmt_iso(a.get('expires'))}")
#             desc = (a.get("description") or "").strip()
#             if desc:
#                 out.append("  Details  :")
#                 out.append("    " + _wrap(desc, width=88).replace("\n", "\n    "))
#             instr = (a.get("instruction") or "").strip()
#             if instr:
#                 out.append("  Instruction:")
#                 out.append("    " + _wrap(instr, width=88).replace("\n", "\n    "))
#             if a.get("web"):
#                 out.append(f"  More info: {a.get('web')}")
#             out.append("")
#     out.append("-" * 96)

#     daily_show = daily_periods[:N_DAILY]
#     out.append(f"DAILY FORECAST (next {min(N_DAILY, len(daily_periods))} periods)")
#     header = f"{'Period':18} {'Temp':8} {'Wind':18} {'POP':6} {'Short forecast'}"
#     out.append(header)
#     out.append("-" * len(header))
#     for p in daily_show:
#         period = (p.get("name") or "—")[:18]
#         temp = _fmt_temp(p)[:8]
#         wind = _fmt_wind(p)[:18]
#         pop = _fmt_prob(p.get("probabilityOfPrecipitation")).rjust(6)
#         short_fc = shorten(p.get("shortForecast") or "—", width=62, placeholder="…")
#         out.append(f"{period:18} {temp:8} {wind:18} {pop:6} {short_fc}")
#     out.append("-" * 96)
#     out.append(f"Daily forecast URL : {daily_url}")
#     if hourly_url:
#         out.append(f"Hourly forecast URL: {hourly_url}")
#     out.append("-" * 96)

#     if INCLUDE_HOURLY and hourly_periods:
#         hourly_show = hourly_periods[:N_HOURLY]
#         out.append(f"HOURLY SNAPSHOT (next {min(N_HOURLY, len(hourly_periods))} hours)")
#         header = f"{'Start':20} {'Temp':8} {'Wind':18} {'POP':6} {'Short forecast'}"
#         out.append(header)
#         out.append("-" * len(header))
#         for p in hourly_show:
#             start = (p.get("startTime") or "—")[:20]
#             temp = _fmt_temp(p)[:8]
#             wind = _fmt_wind(p)[:18]
#             pop = _fmt_prob(p.get("probabilityOfPrecipitation")).rjust(6)
#             short_fc = shorten(p.get("shortForecast") or "—", width=62, placeholder="…")
#             out.append(f"{start:20} {temp:8} {wind:18} {pop:6} {short_fc}")
#         out.append("-" * 96)

#     out.append("CURRENT PERIOD (daily period[0])")
#     out.append(f"  Period   : {current.get('name','—')}")
#     out.append(f"  Temp     : {_fmt_temp(current)}")
#     out.append(f"  Wind     : {_fmt_wind(current)}")
#     out.append(f"  POP      : {_fmt_prob(current.get('probabilityOfPrecipitation'))}")
#     out.append(f"  Summary  : {current.get('shortForecast','—')}")
#     det = current.get("detailedForecast", "")
#     if det:
#         out.append("  Details  :")
#         out.append("    " + _wrap(det, width=88).replace("\n", "\n    "))
#     out.append("=" * 96 + "\n")

#     return "\n".join(out)

# # ───────────────────────────────────────────────────────────────────────────────
# # STREAMLIT UI
# # ───────────────────────────────────────────────────────────────────────────────
# st.set_page_config(page_title="Weather", layout="centered")
# st.title("Weather Forecast")

# location = st.text_input("Enter location", "55 miles north of Fort Collins, CO")

# col1, col2 = st.columns([1, 1])
# with col1:
#     include_web = st.checkbox("Include ChatGPT web forecast (requires OPENAI_API_KEY + web_search)", value=True)
# with col2:
#     st.caption("NWS-only will always work (U.S. points only).")

# if st.button("Get Forecast", type="primary"):
#     try:
#         with st.spinner("Fetching forecast..."):
#             report_text = build_full_report_text(location, include_web=include_web)
#         st.text_area("Forecast report", value=report_text, height=780)
#     except Exception as e:
#         st.error(str(e))

#  #THISWORKES
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Streamlit app: FULL WEATHER REPORT (your big CLI-style report)
# PLUS:
#   - the small "=== NWS Daily Forecast (first 6 periods) ===" block
#   - the "=== ChatGPT Web Forecast ===" block (optional; requires OPENAI_API_KEY + Responses API web_search tool)

# Geocoding:
#   - Photon primary (with User-Agent)
#   - Nominatim fallback (with User-Agent)
#   - NO email anywhere

# Relative locations:
#   - Supports miles + kilometers and diagonals + typo "miels"
#     e.g. "500 miles north of Fort Collins, CO"
#          "200 km NE of Fort Collins, CO"
#          "10 miels west of Denver, CO"

# Run:
#   pip install streamlit requests openai
#   streamlit run app.py

# Notes:
#   - If OPENAI_API_KEY is not set (or web_search isn't enabled), the app still works (NWS-only).
#   - NWS alerts are included via api.weather.gov/alerts/active?point=lat,lon
# """

# import os
# import re
# import json
# import time
# import math
# import requests
# import streamlit as st
# from textwrap import shorten

# # Optional OpenAI (web_search)
# try:
#     from openai import OpenAI
# except Exception:
#     OpenAI = None

# # ───────────────────────────────────────────────────────────────────────────────
# # HEADERS
# # ───────────────────────────────────────────────────────────────────────────────
# USER_AGENT = "RCVFD-WeatherStreamlit/1.1"
# COMMON_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}
# NWS_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/geo+json, application/json"}

# # Output settings (fixed; no UI toggles)
# INCLUDE_HOURLY = True
# N_DAILY = 14
# N_HOURLY = 24
# NWS_FIRST6 = 6

# # OpenAI web_search settings
# DEFAULT_WEB_DAYS = 2
# DEFAULT_WEB_MODEL = "gpt-4.1-mini"

# # ───────────────────────────────────────────────────────────────────────────────
# # HTTP
# # ───────────────────────────────────────────────────────────────────────────────
# def _get_json(url, params=None, headers=None, timeout=30):
#     r = requests.get(url, params=params, headers=headers or {}, timeout=timeout)
#     r.raise_for_status()
#     return r.json()

# def _req_json(url, headers=None, timeout=30):
#     r = requests.get(url, headers=headers or {}, timeout=timeout)
#     r.raise_for_status()
#     return r.json()

# # ───────────────────────────────────────────────────────────────────────────────
# # GEOCODING (Photon primary, Nominatim fallback) — NO EMAIL
# # ───────────────────────────────────────────────────────────────────────────────
# def geocode_photon(query: str):
#     url = "https://photon.komoot.io/api/"
#     params = {"q": query, "limit": 1}
#     data = _get_json(url, params=params, headers=COMMON_HEADERS)

#     feats = data.get("features") or []
#     if not feats:
#         raise RuntimeError("Photon returned no results.")

#     feat = feats[0]
#     props = feat.get("properties") or {}
#     lon, lat = feat.get("geometry", {}).get("coordinates", [None, None])
#     if lat is None or lon is None:
#         raise RuntimeError("Photon result missing coordinates.")

#     label_parts = []
#     for k in ("name", "city", "state", "country"):
#         v = props.get(k)
#         if v and v not in label_parts:
#             label_parts.append(v)
#     label = ", ".join(label_parts) if label_parts else query

#     return float(lat), float(lon), label

# def geocode_nominatim(query: str):
#     url = "https://nominatim.openstreetmap.org/search"
#     params = {"q": query, "format": "json", "limit": 1, "addressdetails": 0}
#     data = _get_json(url, params=params, headers=COMMON_HEADERS)

#     if not data:
#         raise RuntimeError("Nominatim returned no results.")

#     lat = float(data[0]["lat"])
#     lon = float(data[0]["lon"])
#     name = data[0].get("display_name", query)
#     return lat, lon, name

# @st.cache_data(ttl=3600)
# def geocode_place(query: str):
#     try:
#         return geocode_photon(query)
#     except Exception:
#         return geocode_nominatim(query)

# # ───────────────────────────────────────────────────────────────────────────────
# # RELATIVE LOCATION PARSING + OFFSETS (miles + km, diagonals, typo "miels")
# # ───────────────────────────────────────────────────────────────────────────────
# REL_RE = re.compile(
#     r"""
#     ^\s*
#     (?P<distance>\d+(?:\.\d+)?)\s*
#     (?P<unit>mi|mile|miles|miels|km|kms|kilometer|kilometers)\s*
#     (?P<dir>north|south|east|west|n|s|e|w|ne|nw|se|sw|
#             northeast|northwest|southeast|southwest)\s*
#     of\s*
#     (?P<place>.+?)\s*
#     $
#     """,
#     re.IGNORECASE | re.VERBOSE,
# )

# def _dir_to_bearing(dir_str: str) -> float:
#     d = dir_str.strip().lower()
#     mapping = {
#         "n": 0.0, "north": 0.0,
#         "ne": 45.0, "northeast": 45.0,
#         "e": 90.0, "east": 90.0,
#         "se": 135.0, "southeast": 135.0,
#         "s": 180.0, "south": 180.0,
#         "sw": 225.0, "southwest": 225.0,
#         "w": 270.0, "west": 270.0,
#         "nw": 315.0, "northwest": 315.0,
#     }
#     if d not in mapping:
#         raise ValueError(f"Unsupported direction: {dir_str}")
#     return mapping[d]

# def _to_miles(distance: float, unit: str) -> float:
#     u = unit.strip().lower()
#     if u in ("km", "kms", "kilometer", "kilometers"):
#         return distance * 0.621371
#     return distance

# def offset_latlon_bearing(lat: float, lon: float, miles: float, bearing_deg: float):
#     miles_per_deg_lat = 69.0
#     dlat = (miles / miles_per_deg_lat) * math.cos(math.radians(bearing_deg))

#     miles_per_deg_lon = 69.0 * max(0.01, abs(math.cos(math.radians(lat))))
#     dlon = (miles / miles_per_deg_lon) * math.sin(math.radians(bearing_deg))

#     return lat + dlat, lon + dlon

# def resolve_location(description: str):
#     s = (description or "").strip()
#     if not s:
#         raise ValueError("Location is empty.")

#     m = REL_RE.match(s)
#     if m:
#         distance = float(m.group("distance"))
#         unit = m.group("unit")
#         direction = m.group("dir")
#         place = m.group("place").strip()

#         base_lat, base_lon, base_name = geocode_place(place)
#         miles = _to_miles(distance, unit)
#         bearing = _dir_to_bearing(direction)
#         lat, lon = offset_latlon_bearing(base_lat, base_lon, miles, bearing)
#         label = f"{distance:g} {unit} {direction.lower()} of {base_name}"
#         return lat, lon, label

#     lat, lon, name = geocode_place(s)
#     return lat, lon, name

# # ───────────────────────────────────────────────────────────────────────────────
# # NWS FORECAST (A) + ALERTS
# # ───────────────────────────────────────────────────────────────────────────────
# @st.cache_data(ttl=600)
# def nws_forecast(lat, lon, hourly=False):
#     points_url = f"https://api.weather.gov/points/{lat:.6f},{lon:.6f}"
#     points = _req_json(points_url, headers=NWS_HEADERS)
#     props = points.get("properties", {})
#     forecast_url = props.get("forecastHourly") if hourly else props.get("forecast")
#     if not forecast_url:
#         raise RuntimeError("NWS points response missing forecast URL.")

#     fc = _req_json(forecast_url, headers=NWS_HEADERS)
#     periods = fc.get("properties", {}).get("periods", [])
#     return {
#         "source": "NWS api.weather.gov",
#         "lat": lat,
#         "lon": lon,
#         "hourly": bool(hourly),
#         "forecast_url": forecast_url,
#         "periods": periods,
#         "points_properties": props,
#         "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
#     }

# @st.cache_data(ttl=300)
# def nws_alerts(lat, lon):
#     url = "https://api.weather.gov/alerts/active"
#     params = {"point": f"{lat:.6f},{lon:.6f}"}
#     data = _get_json(url, params=params, headers=NWS_HEADERS)
#     feats = data.get("features") or []
#     alerts = []
#     for f in feats:
#         p = (f.get("properties") or {})
#         alerts.append({
#             "event": p.get("event"),
#             "headline": p.get("headline"),
#             "severity": p.get("severity"),
#             "certainty": p.get("certainty"),
#             "urgency": p.get("urgency"),
#             "effective": p.get("effective"),
#             "onset": p.get("onset"),
#             "ends": p.get("ends"),
#             "expires": p.get("expires"),
#             "senderName": p.get("senderName"),
#             "description": p.get("description"),
#             "instruction": p.get("instruction"),
#             "web": p.get("web"),
#             "areaDesc": p.get("areaDesc"),
#         })
#     return alerts

# def first_daytime(periods):
#     for p in periods:
#         if p.get("isDaytime") is True:
#             return p
#     return periods[0] if periods else None

# # ───────────────────────────────────────────────────────────────────────────────
# # OpenAI web_search (B) - optional
# # ───────────────────────────────────────────────────────────────────────────────
# def _openai_client():
#     api_key = os.getenv("OPENAI_API_KEY", "").strip()
#     if not api_key or OpenAI is None:
#         return None
#     return OpenAI(api_key=api_key)

# @st.cache_data(ttl=900)
# def chatgpt_web_weather(lat, lon, days=DEFAULT_WEB_DAYS, model=DEFAULT_WEB_MODEL):
#     client = _openai_client()
#     if client is None:
#         return {"ok": False, "reason": "OPENAI_API_KEY not set or openai package unavailable.", "data": None, "raw": ""}

#     prompt = f"""
# Look up the weather forecast for coordinates ({lat:.6f}, {lon:.6f}) for the next {days} days.
# Use web search. Prefer authoritative sources (NWS, NOAA, official forecast pages).
# Return STRICT JSON with:
# {{
#   "location_name": "<best guess place name>",
#   "forecast_summary": "<short summary>",
#   "high_level_hazards": ["<wind>", "<snow>", "<red flag>", "..."],
#   "periods": [
#     {{
#       "name": "<e.g., Today, Tonight, Mon>",
#       "temp": "<value + units if available>",
#       "wind": "<value + units/direction if available>",
#       "precip": "<if available>",
#       "summary": "<one sentence>"
#     }}
#   ]
# }}
# """.strip()

#     resp = client.responses.create(
#         model=model,
#         input=prompt,
#         tools=[{"type": "web_search"}],
#     )

#     out_text = getattr(resp, "output_text", None)
#     if not out_text:
#         out_text_parts = []
#         for item in getattr(resp, "output", []) or []:
#             if getattr(item, "type", "") == "message":
#                 for c in getattr(item, "content", []) or []:
#                     if getattr(c, "type", "") in ("output_text", "text"):
#                         out_text_parts.append(getattr(c, "text", ""))
#         out_text = "\n".join([t for t in out_text_parts if t]).strip()

#     try:
#         data = json.loads(out_text)
#         return {"ok": True, "data": data, "raw": out_text}
#     except Exception:
#         return {"ok": True, "data": None, "raw": out_text}

# # ───────────────────────────────────────────────────────────────────────────────
# # PRETTY REPORT
# # ───────────────────────────────────────────────────────────────────────────────
# def _fmt_prob(p):
#     if isinstance(p, dict):
#         v = p.get("value", None)
#         return "—" if v is None else f"{int(round(v))}%"
#     return "—" if p is None else str(p)

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

# def _fmt_iso(iso_str):
#     return "—" if not iso_str else str(iso_str)

# def build_full_report_text(description: str):
#     lat, lon, label = resolve_location(description)

#     daily = nws_forecast(lat, lon, hourly=False)
#     daily_periods = daily.get("periods") or []
#     if not daily_periods:
#         raise RuntimeError("No daily forecast periods returned from NWS (U.S. only).")

#     props = daily.get("points_properties") or {}
#     daily_url = daily.get("forecast_url")

#     current = daily_periods[0]
#     today_day = first_daytime(daily_periods)

#     hourly_periods, hourly_url = ([], None)
#     if INCLUDE_HOURLY:
#         hourly = nws_forecast(lat, lon, hourly=True)
#         hourly_periods = hourly.get("periods") or []
#         hourly_url = hourly.get("forecast_url")

#     alerts = nws_alerts(lat, lon)

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

#     # This is the missing block you wanted back
#     out.append("\n=== NWS Daily Forecast (first 6 periods) ===")
#     for p in daily_periods[:NWS_FIRST6]:
#         out.append(
#             f"- {p.get('name','—')}: "
#             f"{p.get('temperature','—')}{p.get('temperatureUnit','')} | "
#             f"{p.get('windSpeed','—')} {p.get('windDirection','')} | "
#             f"{p.get('shortForecast','—')}"
#         )

#     # Always attempt ChatGPT web forecast, but fall back gracefully (no checkbox)
#     out.append("\n=== ChatGPT Web Forecast ===")
#     res = chatgpt_web_weather(lat, lon, days=DEFAULT_WEB_DAYS, model=DEFAULT_WEB_MODEL)
#     if not res.get("ok", False):
#         out.append(f"[OpenAI web_search unavailable; falling back to NWS only]\nReason: {res.get('reason','')}")
#     else:
#         if res.get("data") is not None:
#             out.append(json.dumps(res["data"], indent=2))
#         else:
#             out.append(res.get("raw", "").strip() or "[No text returned]")

#     # Main big report
#     out.append("\n" + "=" * 96)
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
#         out.append(f"  Summary  : {today_day.get('shortForecast','—')}")
#         det = today_day.get("detailedForecast", "")
#         if det:
#             out.append("  Details  :")
#             out.append("    " + _wrap(det, width=88).replace("\n", "\n    "))
#         out.append("-" * 96)

#     out.append("SEVERE WEATHER ALERTS (NWS alerts/active)")
#     if not alerts:
#         out.append("  None.")
#     else:
#         for i, a in enumerate(alerts[:8], start=1):
#             out.append(f"* [{i}] {a.get('event','—')}")
#             if a.get("headline"):
#                 out.append(f"  Headline : {a.get('headline')}")
#             if a.get("areaDesc"):
#                 out.append(f"  Area     : {a.get('areaDesc')}")
#             out.append(f"  Severity : {a.get('severity','—')} | Urgency {a.get('urgency','—')} | Certainty {a.get('certainty','—')}")
#             out.append(f"  Effective: {_fmt_iso(a.get('effective'))}")
#             out.append(f"  Onset    : {_fmt_iso(a.get('onset'))}")
#             out.append(f"  Ends     : {_fmt_iso(a.get('ends'))}")
#             out.append(f"  Expires  : {_fmt_iso(a.get('expires'))}")
#             desc = (a.get("description") or "").strip()
#             if desc:
#                 out.append("  Details  :")
#                 out.append("    " + _wrap(desc, width=88).replace("\n", "\n    "))
#             instr = (a.get("instruction") or "").strip()
#             if instr:
#                 out.append("  Instruction:")
#                 out.append("    " + _wrap(instr, width=88).replace("\n", "\n    "))
#             if a.get("web"):
#                 out.append(f"  More info: {a.get('web')}")
#             out.append("")
#     out.append("-" * 96)

#     daily_show = daily_periods[:N_DAILY]
#     out.append(f"DAILY FORECAST (next {min(N_DAILY, len(daily_periods))} periods)")
#     header = f"{'Period':18} {'Temp':8} {'Wind':18} {'POP':6} {'Short forecast'}"
#     out.append(header)
#     out.append("-" * len(header))
#     for p in daily_show:
#         period = (p.get("name") or "—")[:18]
#         temp = _fmt_temp(p)[:8]
#         wind = _fmt_wind(p)[:18]
#         pop = _fmt_prob(p.get("probabilityOfPrecipitation")).rjust(6)
#         short_fc = shorten(p.get("shortForecast") or "—", width=62, placeholder="…")
#         out.append(f"{period:18} {temp:8} {wind:18} {pop:6} {short_fc}")
#     out.append("-" * 96)
#     out.append(f"Daily forecast URL : {daily_url}")
#     if hourly_url:
#         out.append(f"Hourly forecast URL: {hourly_url}")
#     out.append("-" * 96)

#     if INCLUDE_HOURLY and hourly_periods:
#         hourly_show = hourly_periods[:N_HOURLY]
#         out.append(f"HOURLY SNAPSHOT (next {min(N_HOURLY, len(hourly_periods))} hours)")
#         header = f"{'Start':20} {'Temp':8} {'Wind':18} {'POP':6} {'Short forecast'}"
#         out.append(header)
#         out.append("-" * len(header))
#         for p in hourly_show:
#             start = (p.get("startTime") or "—")[:20]
#             temp = _fmt_temp(p)[:8]
#             wind = _fmt_wind(p)[:18]
#             pop = _fmt_prob(p.get("probabilityOfPrecipitation")).rjust(6)
#             short_fc = shorten(p.get("shortForecast") or "—", width=62, placeholder="…")
#             out.append(f"{start:20} {temp:8} {wind:18} {pop:6} {short_fc}")
#         out.append("-" * 96)

#     out.append("CURRENT PERIOD (daily period[0])")
#     out.append(f"  Period   : {current.get('name','—')}")
#     out.append(f"  Temp     : {_fmt_temp(current)}")
#     out.append(f"  Wind     : {_fmt_wind(current)}")
#     out.append(f"  POP      : {_fmt_prob(current.get('probabilityOfPrecipitation'))}")
#     out.append(f"  Summary  : {current.get('shortForecast','—')}")
#     det = current.get("detailedForecast", "")
#     if det:
#         out.append("  Details  :")
#         out.append("    " + _wrap(det, width=88).replace("\n", "\n    "))
#     out.append("=" * 96 + "\n")

#     return "\n".join(out)

# # ───────────────────────────────────────────────────────────────────────────────
# # STREAMLIT UI
# # ───────────────────────────────────────────────────────────────────────────────
# st.set_page_config(page_title="Weather", layout="centered")
# st.title("Weather Forecast")

# location = st.text_input("Enter location", "55 miles north of Fort Collins, CO")

# if st.button("Get Forecast", type="primary"):
#     try:
#         with st.spinner("Fetching forecast..."):
#             report_text = build_full_report_text(location)
#         st.text_area("Forecast report", value=report_text, height=780)
#     except Exception as e:
#         st.error(str(e))




# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Streamlit app: GPT-only weather lookup (via OpenAI Responses API + web_search tool)
# + Geolocation (Photon primary, Nominatim fallback) + relative offsets.

# Run:
#   pip install streamlit requests openai
#   set OPENAI_API_KEY=...
#   streamlit run app.py

# Notes:
#   - This does NOT call api.weather.gov for forecast/alerts.
#   - It uses web_search, so the model is synthesizing from live web sources.
# """

# import os
# import re
# import json
# import time
# import math
# import requests
# import streamlit as st

# from openai import OpenAI

# # ───────────────────────────────────────────────────────────────────────────────
# # HEADERS
# # ───────────────────────────────────────────────────────────────────────────────
# USER_AGENT = "RCVFD-WeatherStreamlit-GPTOnly/1.0"
# COMMON_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}

# DEFAULT_WEB_DAYS = 2
# DEFAULT_WEB_MODEL = "gpt-4.1-mini"

# # ───────────────────────────────────────────────────────────────────────────────
# # HTTP
# # ───────────────────────────────────────────────────────────────────────────────
# def _get_json(url, params=None, headers=None, timeout=30):
#     r = requests.get(url, params=params, headers=headers or {}, timeout=timeout)
#     r.raise_for_status()
#     return r.json()

# # ───────────────────────────────────────────────────────────────────────────────
# # GEOCODING (Photon primary, Nominatim fallback) — NO EMAIL
# # ───────────────────────────────────────────────────────────────────────────────
# def geocode_photon(query: str):
#     url = "https://photon.komoot.io/api/"
#     params = {"q": query, "limit": 1}
#     data = _get_json(url, params=params, headers=COMMON_HEADERS)

#     feats = data.get("features") or []
#     if not feats:
#         raise RuntimeError("Photon returned no results.")

#     feat = feats[0]
#     props = feat.get("properties") or {}
#     lon, lat = feat.get("geometry", {}).get("coordinates", [None, None])
#     if lat is None or lon is None:
#         raise RuntimeError("Photon result missing coordinates.")

#     label_parts = []
#     for k in ("name", "city", "state", "country"):
#         v = props.get(k)
#         if v and v not in label_parts:
#             label_parts.append(v)
#     label = ", ".join(label_parts) if label_parts else query

#     return float(lat), float(lon), label

# def geocode_nominatim(query: str):
#     url = "https://nominatim.openstreetmap.org/search"
#     params = {"q": query, "format": "json", "limit": 1, "addressdetails": 0}
#     data = _get_json(url, params=params, headers=COMMON_HEADERS)

#     if not data:
#         raise RuntimeError("Nominatim returned no results.")

#     lat = float(data[0]["lat"])
#     lon = float(data[0]["lon"])
#     name = data[0].get("display_name", query)
#     return lat, lon, name

# @st.cache_data(ttl=3600)
# def geocode_place(query: str):
#     try:
#         return geocode_photon(query)
#     except Exception:
#         return geocode_nominatim(query)

# # ───────────────────────────────────────────────────────────────────────────────
# # RELATIVE LOCATION PARSING + OFFSETS (miles + km, diagonals, typo "miels")
# # ───────────────────────────────────────────────────────────────────────────────
# REL_RE = re.compile(
#     r"""
#     ^\s*
#     (?P<distance>\d+(?:\.\d+)?)\s*
#     (?P<unit>mi|mile|miles|miels|km|kms|kilometer|kilometers)\s*
#     (?P<dir>north|south|east|west|n|s|e|w|ne|nw|se|sw|
#             northeast|northwest|southeast|southwest)\s*
#     of\s*
#     (?P<place>.+?)\s*
#     $
#     """,
#     re.IGNORECASE | re.VERBOSE,
# )

# def _dir_to_bearing(dir_str: str) -> float:
#     d = dir_str.strip().lower()
#     mapping = {
#         "n": 0.0, "north": 0.0,
#         "ne": 45.0, "northeast": 45.0,
#         "e": 90.0, "east": 90.0,
#         "se": 135.0, "southeast": 135.0,
#         "s": 180.0, "south": 180.0,
#         "sw": 225.0, "southwest": 225.0,
#         "w": 270.0, "west": 270.0,
#         "nw": 315.0, "northwest": 315.0,
#     }
#     if d not in mapping:
#         raise ValueError(f"Unsupported direction: {dir_str}")
#     return mapping[d]

# def _to_miles(distance: float, unit: str) -> float:
#     u = unit.strip().lower()
#     if u in ("km", "kms", "kilometer", "kilometers"):
#         return distance * 0.621371
#     return distance

# def offset_latlon_bearing(lat: float, lon: float, miles: float, bearing_deg: float):
#     miles_per_deg_lat = 69.0
#     dlat = (miles / miles_per_deg_lat) * math.cos(math.radians(bearing_deg))

#     miles_per_deg_lon = 69.0 * max(0.01, abs(math.cos(math.radians(lat))))
#     dlon = (miles / miles_per_deg_lon) * math.sin(math.radians(bearing_deg))

#     return lat + dlat, lon + dlon

# def resolve_location(description: str):
#     s = (description or "").strip()
#     if not s:
#         raise ValueError("Location is empty.")

#     m = REL_RE.match(s)
#     if m:
#         distance = float(m.group("distance"))
#         unit = m.group("unit")
#         direction = m.group("dir")
#         place = m.group("place").strip()

#         base_lat, base_lon, base_name = geocode_place(place)
#         miles = _to_miles(distance, unit)
#         bearing = _dir_to_bearing(direction)
#         lat, lon = offset_latlon_bearing(base_lat, base_lon, miles, bearing)
#         label = f"{distance:g} {unit} {direction.lower()} of {base_name}"
#         return lat, lon, label

#     lat, lon, name = geocode_place(s)
#     return lat, lon, name

# # ───────────────────────────────────────────────────────────────────────────────
# # GPT WEB WEATHER ONLY
# # ───────────────────────────────────────────────────────────────────────────────
# @st.cache_data(ttl=900)
# def gpt_web_weather(lat, lon, days=DEFAULT_WEB_DAYS, model=DEFAULT_WEB_MODEL):
#     api_key = os.getenv("OPENAI_API_KEY", "").strip()
#     if not api_key:
#         raise RuntimeError("Set OPENAI_API_KEY in your environment.")
#     client = OpenAI(api_key=api_key)

#     prompt = f"""
# Look up the weather forecast for coordinates ({lat:.6f}, {lon:.6f}) for the next {days} days.
# Use web search. Prefer authoritative sources (NWS, NOAA, official forecast pages).
# Return STRICT JSON with:
# {{
#   "location_name": "<best guess place name>",
#   "forecast_summary": "<short summary>",
#   "high_level_hazards": ["<wind>", "<snow>", "<red flag>", "..."],
#   "periods": [
#     {{
#       "name": "<e.g., Today, Tonight, Mon>",
#       "temp": "<value + units if available>",
#       "wind": "<value + units/direction if available>",
#       "precip": "<if available>",
#       "summary": "<one sentence>"
#     }}
#   ]
# }}
# """.strip()

#     resp = client.responses.create(
#         model=model,
#         input=prompt,
#         tools=[{"type": "web_search"}],
#     )

#     out_text = getattr(resp, "output_text", None)
#     if not out_text:
#         out_text_parts = []
#         for item in getattr(resp, "output", []) or []:
#             if getattr(item, "type", "") == "message":
#                 for c in getattr(item, "content", []) or []:
#                     if getattr(c, "type", "") in ("output_text", "text"):
#                         out_text_parts.append(getattr(c, "text", ""))
#         out_text = "\n".join([t for t in out_text_parts if t]).strip()

#     try:
#         return json.loads(out_text)
#     except Exception:
#         raise RuntimeError(f"Model did not return valid JSON.\nRaw:\n{out_text}")

# def build_report(location_str: str):
#     lat, lon, label = resolve_location(location_str)
#     ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

#     data = gpt_web_weather(lat, lon)

#     out = []
#     out.append("\n" + "=" * 96)
#     out.append("FULL WEATHER REPORT (GPT web_search)")
#     out.append("=" * 96)
#     out.append(f"Input location     : {location_str}")
#     out.append(f"Resolved location  : {label}")
#     out.append(f"Coordinates        : {lat:.6f}, {lon:.6f}")
#     out.append(f"Generated          : {ts}")
#     out.append("-" * 96)
#     out.append(json.dumps(data, indent=2))
#     out.append("=" * 96 + "\n")
#     return "\n".join(out)

# # ───────────────────────────────────────────────────────────────────────────────
# # STREAMLIT UI
# # ───────────────────────────────────────────────────────────────────────────────
# st.set_page_config(page_title="Weather (GPT-only)", layout="centered")
# st.title("Weather Forecast (GPT-only)")

# location = st.text_input("Enter location", "55 miles north of Fort Collins, CO")

# if st.button("Get Forecast", type="primary"):
#     try:
#         with st.spinner("Fetching forecast (web_search)..."):
#             report_text = build_report(location)
#         st.text_area("Forecast report", value=report_text, height=780)
#     except Exception as e:
#         st.error(str(e))







#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit app: GPT-only weather lookup (via OpenAI Responses API + web_search tool)
+ Geolocation (Photon primary, Nominatim fallback) + relative offsets.

Run:
  pip install streamlit requests openai
  streamlit run app.py

Secrets / Env:
  - Streamlit Cloud:
      [openai]
      api_key = "..."
    or
  - Local:
      set OPENAI_API_KEY=...

Notes:
  - This does NOT call api.weather.gov for forecast/alerts.
  - It uses web_search, so the model is synthesizing from live web sources.
"""

import os
import re
import json
import time
import math
import requests
import streamlit as st

from openai import OpenAI

# ───────────────────────────────────────────────────────────────────────────────
# HEADERS
# ───────────────────────────────────────────────────────────────────────────────
USER_AGENT = "RCVFD-WeatherStreamlit-GPTOnly/1.1"
COMMON_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}

DEFAULT_WEB_DAYS = 2
DEFAULT_WEB_MODEL = "gpt-4.1-mini"

# ───────────────────────────────────────────────────────────────────────────────
# HTTP
# ───────────────────────────────────────────────────────────────────────────────
def _get_json(url, params=None, headers=None, timeout=30):
    r = requests.get(url, params=params, headers=headers or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ───────────────────────────────────────────────────────────────────────────────
# GEOCODING (Photon primary, Nominatim fallback) — NO EMAIL
# ───────────────────────────────────────────────────────────────────────────────
def geocode_photon(query: str):
    url = "https://photon.komoot.io/api/"
    params = {"q": query, "limit": 1}
    data = _get_json(url, params=params, headers=COMMON_HEADERS)

    feats = data.get("features") or []
    if not feats:
        raise RuntimeError("Photon returned no results.")

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
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 1, "addressdetails": 0}
    data = _get_json(url, params=params, headers=COMMON_HEADERS)

    if not data:
        raise RuntimeError("Nominatim returned no results.")

    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    name = data[0].get("display_name", query)
    return lat, lon, name

@st.cache_data(ttl=3600)
def geocode_place(query: str):
    try:
        return geocode_photon(query)
    except Exception:
        return geocode_nominatim(query)

# ───────────────────────────────────────────────────────────────────────────────
# RELATIVE LOCATION PARSING + OFFSETS (miles + km, diagonals, typo "miels")
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
    return distance

def offset_latlon_bearing(lat: float, lon: float, miles: float, bearing_deg: float):
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

        base_lat, base_lon, base_name = geocode_place(place)
        miles = _to_miles(distance, unit)
        bearing = _dir_to_bearing(direction)
        lat, lon = offset_latlon_bearing(base_lat, base_lon, miles, bearing)
        label = f"{distance:g} {unit} {direction.lower()} of {base_name}"
        return lat, lon, label

    lat, lon, name = geocode_place(s)
    return lat, lon, name

# ───────────────────────────────────────────────────────────────────────────────
# OPENAI KEY RESOLUTION (secrets first, then env)
# ───────────────────────────────────────────────────────────────────────────────
def get_openai_api_key() -> str:
    key = ""
    try:
        key = (st.secrets.get("openai", {}) or {}).get("api_key", "") or ""
    except Exception:
        key = ""
    if not key:
        key = os.getenv("OPENAI_API_KEY", "") or ""
    key = str(key).strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to Streamlit secrets:\n"
            "[openai]\napi_key = \"...\"\n"
            "or set environment variable OPENAI_API_KEY."
        )
    return key

# ───────────────────────────────────────────────────────────────────────────────
# GPT WEB WEATHER ONLY
# ───────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=900)
def gpt_web_weather(lat, lon, days=DEFAULT_WEB_DAYS, model=DEFAULT_WEB_MODEL):
    api_key = get_openai_api_key()
    client = OpenAI(api_key=api_key)

    prompt = f"""
Look up the weather forecast for coordinates ({lat:.6f}, {lon:.6f}) for the next {days} days.
Use web search. Prefer authoritative sources (NWS, NOAA, official forecast pages).
Return STRICT JSON with:
{{
  "location_name": "<best guess place name>",
  "forecast_summary": "<short summary>",
  "high_level_hazards": ["<wind>", "<snow>", "<red flag>", "..."],
  "periods": [
    {{
      "name": "<e.g., Today, Tonight, Mon>",
      "temp": "<value + units if available>",
      "wind": "<value + units/direction if available>",
      "precip": "<if available>",
      "summary": "<one sentence>"
    }}
  ]
}}
""".strip()

    resp = client.responses.create(
        model=model,
        input=prompt,
        tools=[{"type": "web_search"}],
    )

    out_text = getattr(resp, "output_text", None)
    if not out_text:
        out_text_parts = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", "") == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", "") in ("output_text", "text"):
                        out_text_parts.append(getattr(c, "text", ""))
        out_text = "\n".join([t for t in out_text_parts if t]).strip()

    try:
        return json.loads(out_text)
    except Exception:
        raise RuntimeError(f"Model did not return valid JSON.\nRaw:\n{out_text}")

def build_report(location_str: str):
    lat, lon, label = resolve_location(location_str)
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    data = gpt_web_weather(lat, lon)

    out = []
    out.append("\n" + "=" * 96)
    out.append("FULL WEATHER REPORT (GPT web_search)")
    out.append("=" * 96)
    out.append(f"Input location     : {location_str}")
    out.append(f"Resolved location  : {label}")
    out.append(f"Coordinates        : {lat:.6f}, {lon:.6f}")
    out.append(f"Generated          : {ts}")
    out.append("-" * 96)
    out.append(json.dumps(data, indent=2))
    out.append("=" * 96 + "\n")
    return "\n".join(out)

# ───────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Weather (GPT-only)", layout="centered")
st.title("Weather Forecast (GPT-only)")

location = st.text_input("Enter location", "55 miles north of Fort Collins, CO")

if st.button("Get Forecast", type="primary"):
    try:
        with st.spinner("Fetching forecast (web_search)..."):
            report_text = build_report(location)
        st.text_area("Forecast report", value=report_text, height=780)
    except Exception as e:
        st.error(str(e))
