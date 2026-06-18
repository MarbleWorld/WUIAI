# streamlit_app_naip_wui_mapclick.py
# Run:
#   pip install streamlit streamlit-folium folium geopandas rasterio pystac-client planetary-computer shapely pillow numpy openai matplotlib
#   streamlit run streamlit_app_naip_wui_mapclick.py


import os
import json
import math
import base64
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image

import folium
from streamlit_folium import st_folium

import rasterio
from rasterio.mask import mask

from shapely.geometry import Point, box, mapping, shape
import geopandas as gpd

import pystac_client
import planetary_computer

from openai import OpenAI


# =========================
# DEFAULTS
# =========================
DEFAULT_MODEL = "gpt-4o"
DEFAULT_SAVE_DIR = "C:\\Users"

DEFAULT_SYSTEM = "You are a land cover photo-analyst. Use only what is visible."

DEFAULT_QUESTION = r"""
Analyze this NAIP RGB aerial image (~0.6–1 m resolution). Use only visible information in the image. Do not assume information not directly observable.

Context:
This image represents a fixed-area crop (~500 m × 500 m). The task is to characterize Wildland–Urban Interface (WUI) exposure and structural defensibility indicators based on visible housing patterns, vegetation proximity, and access.

Tasks:

1. Housing Presence & Density
- Identify all visible residential structures (single-family homes, cabins, outbuildings associated with residences).
- Estimate the number of residential structures visible.
- Classify housing density within the image as one of:
  • Very Low (isolated structures, >40 acres per home equivalent)
  • Low (scattered homes, rural residential)
  • Medium (exurban/suburban edge)
  • High (dense suburban/urban)
- Briefly justify using spacing and clustering visible in the image.

2. Vegetation Proximity & Defensible Space
- Assess vegetation immediately surrounding structures.
- Note whether trees and shrubs are:
  • In direct contact with structures
  • Within ~0–10 m
  • Within ~10–30 m
  • Mostly cleared beyond ~30 m
- Based on visible cues only, classify overall defensible space as:
  • Poor
  • Limited
  • Moderate
  • Good
- Cite concrete visual indicators (e.g., canopy overhang, ladder fuels, cleared yards, mowed areas).

3. WUI Classification
- Based on housing density and surrounding wildland vegetation, classify the scene as:
  • Not WUI
  • Intermix WUI
  • Interface WUI
- Briefly explain the classification using observable patterns.

4. Road Network & Access
- Identify visible roads (paved vs unpaved if distinguishable).
- Note apparent access characteristics:
  • Multiple access routes vs single ingress/egress
  • Road width (narrow vs standard two-lane)
  • Connectivity (grid vs dead-end/driveways)
- Comment on potential access constraints for fire response based only on visibility.

5. Overall WUI Risk Indicator
- Rate overall wildfire exposure risk to structures as:
  • Low
  • Moderate
  • High
  • Very High
- Base this rating strictly on housing density, vegetation proximity, and access—do not incorporate weather, slope, or fire history unless directly visible.

Uncertainty:
- Provide an uncertainty score from 0–1 reflecting confidence in your assessment given image resolution, shadows, and occlusions.

Output strictly as JSON:
{
  "estimated_structure_count": <integer>,
  "housing_density_class": "<Very Low|Low|Medium|High>",
  "defensible_space_class": "<Poor|Limited|Moderate|Good>",
  "wui_class": "<Not WUI|Intermix WUI|Interface WUI>",
  "road_access_notes": "<concise description>",
  "overall_wui_risk": "<Low|Moderate|High|Very High>",
  "evidence": "<≤75 words citing visible features>",
  "uncertainty": <float 0–1>
}
""".strip()


# =========================
# CONFIG (behind-the-scenes)
# =========================
NAIP_SEARCH_START = "2010-01-01"
NAIP_SEARCH_END = "2035-01-01"
HIDDEN_PAD_DEG = 0.0025

DEFAULT_CENTER = [40.655527, -105.307652]
DEFAULT_ZOOM = 13


# =========================
# CACHED RESOURCES
# =========================
@st.cache_resource
def get_catalog():
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


# =========================
# HELPERS
# =========================
def pct_scale_to_u8(rgb_arr):
    if rgb_arr.dtype == np.uint8:
        return rgb_arr
    out = []
    for b in range(rgb_arr.shape[0]):
        band = rgb_arr[b].astype(np.float32)
        lo, hi = np.nanpercentile(band, [0.0, 99.5])
        if not np.isfinite(lo): lo = 0.0
        if not np.isfinite(hi) or hi <= lo: hi = lo + 1.0
        band = (np.clip((band - lo) / (hi - lo), 0, 1) * 255.0).astype(np.uint8)
        out.append(band)
    return np.stack(out, axis=0)

def _aoi_poly(lon, lat, pad_deg):
    return {
        "type": "Polygon",
        "coordinates": [[
            [lon - pad_deg, lat - pad_deg],
            [lon + pad_deg, lat - pad_deg],
            [lon + pad_deg, lat + pad_deg],
            [lon - pad_deg, lat + pad_deg],
            [lon - pad_deg, lat - pad_deg],
        ]]
    }

def _item_datetime_str(it):
    dt = getattr(it, "datetime", None)
    if dt is not None:
        try:
            return dt.isoformat()
        except Exception:
            pass
    props = getattr(it, "properties", {}) or {}
    return props.get("datetime") or props.get("start_datetime") or ""

def find_most_recent_naip_item(catalog, lon, lat, pad_deg=HIDDEN_PAD_DEG):
    aoi = _aoi_poly(lon, lat, pad_deg=pad_deg)

    items = catalog.search(
        collections=["naip"],
        intersects=aoi,
        datetime=f"{NAIP_SEARCH_START}/{NAIP_SEARCH_END}",
        max_items=200,
    ).item_collection()

    if not items or len(items) == 0:
        return None, aoi

    aoi_shape = shape(aoi)
    keep = []
    for it in items:
        try:
            if shape(it.geometry).intersects(aoi_shape):
                keep.append(it)
        except Exception:
            continue

    if len(keep) == 0:
        return None, aoi

    def sort_key(it):
        dt = _item_datetime_str(it)
        try:
            area = shape(it.geometry).intersection(aoi_shape).area
        except Exception:
            area = 0.0
        return (dt, area)

    keep_sorted = sorted(keep, key=sort_key, reverse=True)
    return keep_sorted[0], aoi

def item_to_href(item):
    href = None
    if "image" in item.assets:
        href = item.assets["image"].href
    else:
        for a in item.assets.values():
            if a.href.lower().endswith((".tif", ".tiff")):
                href = a.href
                break
    return href

def crop_naip_at_point(href, lon, lat, half_m):
    with rasterio.open(href) as src:
        pt = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs=4326).to_crs(src.crs)
        cx, cy = pt.geometry.iloc[0].x, pt.geometry.iloc[0].y
        crop_box = box(cx - half_m, cy - half_m, cx + half_m, cy + half_m)
        crop_geom = [mapping(crop_box)]
        data, _ = mask(src, crop_geom, crop=True)

        if data.shape[0] >= 3:
            rgb = data[:3, :, :]
        else:
            rgb = np.repeat(data[0:1, :, :], 3, axis=0)

        rgb_u8 = pct_scale_to_u8(rgb)
        pil = Image.fromarray(np.transpose(rgb_u8, (1, 2, 0)))
        return pil

def ask_image_question(client, pil_image, question, system_preamble, model, temperature=0.0):
    buf = BytesIO()
    pil_image.save(buf, format="PNG")
    img_data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_preamble},
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": img_data_url}},
            ]},
        ],
        temperature=float(temperature),
    )
    return resp.choices[0].message.content.strip()

def try_parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        t = text.strip()
        if t.startswith("```"):
            t = t.strip("`")
            t = t.split("\n", 1)[-1]
        try:
            return json.loads(t)
        except Exception:
            return None


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="NAIP WUI Map-Click", layout="wide")
st.title("NAIP WUI Map-Click")

api_key = os.getenv("OPENAI_API_KEY", "") or (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else "")
if not api_key:
    st.error("OPENAI_API_KEY not found. Add it in Streamlit → Manage app → Secrets.")
    st.stop()

# ---- reset + initialize session state
def do_reset():
    st.session_state.clicked_lat = None
    st.session_state.clicked_lon = None
    st.session_state.map_center = DEFAULT_CENTER[:]
    st.session_state.map_zoom = DEFAULT_ZOOM
    st.session_state.last_answer = None
    st.session_state.last_meta = None
    st.session_state.last_image = None

if "initialized" not in st.session_state:
    st.session_state.initialized = True
    do_reset()

colA, colB, colC = st.columns([1, 1, 3])
with colA:
    if st.button("Reset selection", type="secondary"):
        do_reset()

with st.sidebar:
    st.header("Crop")
    side_m = st.selectbox("Crop size (meters)", options=[250, 500, 750, 1000], index=1)
    half_m = float(side_m) / 2.0

    st.header("Model")
    model = st.text_input("OpenAI model", value=DEFAULT_MODEL)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

system_preamble = st.text_area("System preamble", value=DEFAULT_SYSTEM, height=90)
question = st.text_area("Question prompt", value=DEFAULT_QUESTION, height=280)

st.subheader("1) Pseudo Double-Click a location on the map")
st.caption("No point is selected at startup. Click to drop a point; then the crop box appears and you can run analysis.")

m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom, control_scale=True)

if st.session_state.clicked_lat is not None and st.session_state.clicked_lon is not None:
    folium.CircleMarker(
        location=[st.session_state.clicked_lat, st.session_state.clicked_lon],
        radius=8,
        weight=2,
        color="yellow",
        fill=True,
        fill_opacity=0.25,
        tooltip="Selected point",
    ).add_to(m)

    deg_lat = (half_m / 111320.0)
    deg_lon = (half_m / (111320.0 * max(0.2, math.cos(math.radians(st.session_state.clicked_lat)))))
    bounds = [
        [st.session_state.clicked_lat - deg_lat, st.session_state.clicked_lon - deg_lon],
        [st.session_state.clicked_lat + deg_lat, st.session_state.clicked_lon + deg_lon],
    ]
    folium.Rectangle(bounds=bounds, color="yellow", weight=2, fill=False).add_to(m)

map_out = st_folium(m, height=560, width=None, key="wui_map")

if map_out:
    if map_out.get("center"):
        st.session_state.map_center = [float(map_out["center"]["lat"]), float(map_out["center"]["lng"])]
    if map_out.get("zoom") is not None:
        st.session_state.map_zoom = int(map_out["zoom"])
    if map_out.get("last_clicked"):
        st.session_state.clicked_lat = float(map_out["last_clicked"]["lat"])
        st.session_state.clicked_lon = float(map_out["last_clicked"]["lng"])

if st.session_state.clicked_lat is None:
    st.info("No point selected yet. Click on the map to select a location.")
else:
    st.write(f"Selected point: **lat={st.session_state.clicked_lat:.6f}, lon={st.session_state.clicked_lon:.6f}**")

run_disabled = (st.session_state.clicked_lat is None or st.session_state.clicked_lon is None)
run = st.button("2) Run analysis for selected point", type="primary", disabled=run_disabled)

if run:
    lon = st.session_state.clicked_lon
    lat = st.session_state.clicked_lat

    catalog = get_catalog()

    st.info("Finding most recent NAIP scene for that point...")
    try:
        item, _aoi = find_most_recent_naip_item(catalog, lon, lat, pad_deg=HIDDEN_PAD_DEG)
        if item is None:
            st.error("No NAIP scene found for that location.")
            st.stop()

        href = item_to_href(item)
        if href is None:
            st.error("Could not find NAIP COG asset on the selected item.")
            st.stop()

        naip_id = item.id
        naip_dt = _item_datetime_str(item)

    except Exception as e:
        st.exception(e)
        st.stop()

    st.info(f"Cropping NAIP to {int(side_m)} m × {int(side_m)} m...")
    try:
        pil = crop_naip_at_point(href, lon, lat, half_m=half_m)
        if pil is None:
            st.error("Crop returned no image.")
            st.stop()
    except Exception as e:
        st.exception(e)
        st.stop()

    st.info("Calling OpenAI model...")
    try:
        client = OpenAI(api_key=api_key)
        answer_text = ask_image_question(
            client=client,
            pil_image=pil,
            question=question,
            system_preamble=system_preamble,
            model=model,
            temperature=temperature,
        )
    except Exception as e:
        st.exception(e)
        st.stop()

    parsed = try_parse_json(answer_text)

    st.session_state.last_answer = parsed if parsed is not None else answer_text
    st.session_state.last_meta = {
        "lat": lat,
        "lon": lon,
        "crop_side_m": int(side_m),
        "naip_id": naip_id,
        "naip_datetime": naip_dt,
        "naip_href": href,
        "model": model,
        "temperature": float(temperature),
    }
    st.session_state.last_image = pil

if st.session_state.last_answer is not None and st.session_state.last_image is not None:
    st.subheader("Results")
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("NAIP crop")
        meta = st.session_state.last_meta or {}
        cap = f"{meta.get('naip_id','')} | {meta.get('naip_datetime','')} | crop={meta.get('crop_side_m','')}m"
        st.image(np.array(st.session_state.last_image), caption=cap, width=650)
        if meta.get("naip_href"):
            st.caption(f"COG href: {meta['naip_href']}")

    with c2:
        st.subheader("Model answer")
        if isinstance(st.session_state.last_answer, dict):
            st.json(st.session_state.last_answer)
        else:
            st.text_area("Raw answer (not valid JSON)", value=str(st.session_state.last_answer), height=320)

    st.subheader("Run metadata")
    st.code(json.dumps(st.session_state.last_meta, indent=2), language="json")
