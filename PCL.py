


# PCL_basic_app.py
# Run:
#   pip install streamlit rasterio pyproj numpy folium streamlit-folium matplotlib requests openai
#   export OPENAI_API_KEY="your_key_here"   # mac/linux
#   set OPENAI_API_KEY=your_key_here        # windows cmd
#   $env:OPENAI_API_KEY="your_key_here"     # windows powershell
#   streamlit run PCL_basic_app.py

import base64
import io
import json
import math
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import requests
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import folium
from streamlit_folium import st_folium
from openai import OpenAI

import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from pyproj import Transformer


# -------------------------
# CONFIG
# -------------------------
DEFAULT_TIF_URL = "https://raw.githubusercontent.com/MarbleWorld/WUIAI/main/PCL_RCVFD.tif"
REQUEST_TIMEOUT = 120

OPENAI_MODEL = "gpt-4.1"
PCL_THRESHOLD_DEFAULT = 15.0
SEARCH_PATCH_HALF_PX = 1800
MAX_CANDIDATES_FOR_GPT = 8
PATCH_DISPLAY_HALF_PX = 450


# -------------------------
# Helpers
# -------------------------
def robust_minmax(a, lo=2, hi=98):
    x = a[np.isfinite(a)]
    if x.size == 0:
        return 0.0, 1.0
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))


def downsample_for_display(arr2d, max_dim=1400):
    h, w = arr2d.shape
    scale = max(h / max_dim, w / max_dim, 1.0)
    step = int(np.ceil(scale))
    return arr2d[::step, ::step], step


def render_overlay_png(arr2d_float, cmap="viridis", annotate_click=None):
    arr = arr2d_float.astype("float32", copy=False)
    vmin, vmax = robust_minmax(arr)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(arr)) if np.isfinite(np.nanmin(arr)) else 0.0
        vmax = float(np.nanmax(arr)) if np.isfinite(np.nanmax(arr)) else 1.0
        if vmin == vmax:
            vmax = vmin + 1.0

    fig = plt.figure(figsize=(6, 6), dpi=220)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    if annotate_click is not None:
        rr, cc = annotate_click
        ax.scatter([cc], [rr], s=80, marker="x")

    ax.set_facecolor((0, 0, 0, 0))
    fig.patch.set_alpha(0)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def png_bytes_to_data_url(png_bytes):
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")


def latlon_to_rowcol(ds, lat, lon):
    transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    row, col = ds.index(x, y)
    return row, col


def rowcol_to_latlon(ds, row, col):
    x, y = ds.xy(row, col)
    transformer = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return float(lat), float(lon)


def sample_pcl_at_latlon(ds, lat, lon):
    row, col = latlon_to_rowcol(ds, lat, lon)
    if row < 0 or col < 0 or row >= ds.height or col >= ds.width:
        return None
    win = Window(col, row, 1, 1)
    val = ds.read(1, window=win, masked=True)[0, 0]
    if np.ma.is_masked(val):
        return None
    return float(val)


def read_patch(ds, center_row, center_col, half_px):
    r0 = max(0, center_row - half_px)
    r1 = min(ds.height, center_row + half_px + 1)
    c0 = max(0, center_col - half_px)
    c1 = min(ds.width, center_col + half_px + 1)
    win = Window(c0, r0, c1 - c0, r1 - r0)
    block = ds.read(1, window=win, masked=True).astype("float32")
    arr = block.data.copy()
    if hasattr(block, "mask"):
        arr[block.mask] = np.nan
    return arr, r0, c0


def connected_components_8(mask):
    h, w = mask.shape
    labels = -np.ones((h, w), dtype=np.int32)
    comps = []
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    cid = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue
            stack = [(r, c)]
            labels[r, c] = cid
            pts = [(r, c)]
            while stack:
                rr, cc = stack.pop()
                for dr, dc in nbrs:
                    r2, c2 = rr + dr, cc + dc
                    if 0 <= r2 < h and 0 <= c2 < w and mask[r2, c2] and labels[r2, c2] == -1:
                        labels[r2, c2] = cid
                        stack.append((r2, c2))
                        pts.append((r2, c2))
            comps.append(pts)
            cid += 1
    return comps


def approx_distance_m(ds, d_pixels, fallback_px_m=30.0):
    try:
        px = float(max(abs(ds.res[0]), abs(ds.res[1])))
        if np.isfinite(px) and px > 0:
            return float(d_pixels) * px, px
    except Exception:
        pass
    return float(d_pixels) * float(fallback_px_m), float(fallback_px_m)


def component_orientation_deg(points_rowcol):
    if len(points_rowcol) < 2:
        return 0.0
    arr = np.array(points_rowcol, dtype=np.float64)
    rr = arr[:, 0]
    cc = arr[:, 1]
    x = cc - np.mean(cc)
    y = rr - np.mean(rr)
    cov = np.cov(np.vstack([x, y]))
    vals, vecs = np.linalg.eig(cov)
    v = vecs[:, np.argmax(vals)]
    angle = math.degrees(math.atan2(v[1], v[0]))
    angle = (angle + 180.0) % 180.0
    return float(angle)


def nearest_component_candidates(
    ds,
    arr_patch,
    r0,
    c0,
    center_row,
    center_col,
    thr,
    min_comp_pixels=25,
    max_candidates=8,
):
    if arr_patch is None or arr_patch.size == 0:
        return []

    mask = np.isfinite(arr_patch) & (arr_patch >= thr)
    if not mask.any():
        return []

    comps = connected_components_8(mask)

    pr = center_row - r0
    pc = center_col - c0

    out = []
    for pts in comps:
        if len(pts) < int(min_comp_pixels):
            continue

        nearest = min(pts, key=lambda p: (p[0] - pr) ** 2 + (p[1] - pc) ** 2)
        d2 = (nearest[0] - pr) ** 2 + (nearest[1] - pc) ** 2

        comp_vals = np.array([arr_patch[r, c] for r, c in pts], dtype=np.float32)
        rr_mean = int(round(np.mean([p[0] for p in pts])))
        cc_mean = int(round(np.mean([p[1] for p in pts])))

        comp_full = [(r0 + rr, c0 + cc) for rr, cc in pts]
        near_full = (r0 + nearest[0], c0 + nearest[1])
        cent_full = (r0 + rr_mean, c0 + cc_mean)

        dist_px = float(np.sqrt(d2))
        dist_m, px_m = approx_distance_m(ds, dist_px, fallback_px_m=30.0)

        angle = component_orientation_deg(pts)
        east_west_score = 1.0 - min(abs(angle - 0.0), abs(angle - 180.0), abs(angle - 180.0)) / 90.0
        east_west_score = float(max(0.0, min(1.0, east_west_score)))

        out.append(
            {
                "component_points_rowcol": comp_full,
                "nearest_pixel_rowcol": (int(near_full[0]), int(near_full[1])),
                "centroid_rowcol": (int(cent_full[0]), int(cent_full[1])),
                "min_distance_pixels": dist_px,
                "distance_m": float(dist_m),
                "px_m": float(px_m),
                "size_pixels": int(len(comp_full)),
                "pcl_mean": float(np.nanmean(comp_vals)),
                "pcl_max": float(np.nanmax(comp_vals)),
                "orientation_deg": angle,
                "east_west_score": east_west_score,
            }
        )

    out.sort(key=lambda x: (x["distance_m"], -x["pcl_mean"], -x["size_pixels"]))
    return out[:max_candidates]


def build_300m_polyline_from_component(ds, comp_points_rowcol, prefer_east=True, target_len_m=300.0, fallback_px_m=30.0):
    if not comp_points_rowcol:
        return None

    try:
        px_m = float(max(abs(ds.res[0]), abs(ds.res[1])))
        if not np.isfinite(px_m) or px_m <= 0:
            px_m = float(fallback_px_m)
    except Exception:
        px_m = float(fallback_px_m)

    target_steps = max(3, int(round(target_len_m / px_m)))

    pts = comp_points_rowcol
    pts_set = set(pts)

    cols = [c for _, c in pts]
    if prefer_east:
        start_col = max(cols)
    else:
        start_col = min(cols)

    candidates = [p for p in pts if p[1] == start_col]
    start = candidates[len(candidates) // 2] if candidates else pts[len(pts) // 2]

    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    chain = [start]
    visited = {start}

    desired_sign = 1 if not prefer_east else -1

    for _ in range(target_steps - 1):
        r, c = chain[-1]
        neigh = []
        for dr, dc in nbrs:
            p = (r + dr, c + dc)
            if p in pts_set and p not in visited:
                neigh.append(p)
        if not neigh:
            break

        prev = chain[-2] if len(chain) >= 2 else None
        best_p = None
        best_score = None
        for p in neigh:
            rr, cc = p
            s = desired_sign * (cc - c)

            if prev is not None:
                pr, pc = prev
                v1 = (r - pr, c - pc)
                v2 = (rr - r, cc - c)
                dot = v1[0] * v2[0] + v1[1] * v2[1]
            else:
                dot = 0

            score = (10.0 * s) + (1.0 * dot)
            if best_score is None or score > best_score:
                best_score = score
                best_p = p

        if best_p is None:
            break
        chain.append(best_p)
        visited.add(best_p)

    length_m = max(0.0, (len(chain) - 1) * px_m)
    latlons = [rowcol_to_latlon(ds, r, c) for r, c in chain]
    return latlons, float(length_m), chain


def render_local_patch_png(arr_patch, click_local_row, click_local_col, candidates):
    arr = arr_patch.astype("float32", copy=False)
    vmin, vmax = robust_minmax(arr)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(arr)) if np.isfinite(np.nanmin(arr)) else 0.0
        vmax = float(np.nanmax(arr)) if np.isfinite(np.nanmax(arr)) else 1.0
        if vmin == vmax:
            vmax = vmin + 1.0

    fig, ax = plt.subplots(figsize=(6, 6), dpi=220)
    ax.imshow(arr, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.scatter([click_local_col], [click_local_row], marker="x", s=90)
    for idx, cand in enumerate(candidates, start=1):
        rr, cc = cand["nearest_pixel_local"]
        ax.text(cc + 6, rr + 6, str(idx), fontsize=9)
        ax.scatter([cc], [rr], s=45)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout(pad=0)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def safe_json_loads(s: str) -> Dict[str, Any]:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        lines = s.splitlines()
        if lines and lines[0].lower().strip() == "json":
            s = "\n".join(lines[1:])
    return json.loads(s)


def resolve_github_raw_url(url: str) -> str:
    url = url.strip()
    if "github.com" in url and "/blob/" in url:
        url = url.replace("https://github.com/", "https://raw.githubusercontent.com/")
        url = url.replace("/blob/", "/")
    return url


@st.cache_data(show_spinner=False)
def download_tif_from_url(url):
    if not url or not url.strip():
        raise ValueError("GeoTIFF URL is empty.")

    url = resolve_github_raw_url(url)

    suffix = ".tif"
    if url.lower().endswith(".tiff"):
        suffix = ".tiff"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()

    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "").lower()
        if "text/html" in content_type:
            raise ValueError(f"URL returned HTML instead of a GeoTIFF: {url}")
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    return tmp_path


def get_openai_client():
    api_key = st.secrets.get("openai", {}).get("api_key") or os.getenv("OPENAI_API_KEY", "")
    api_key = str(api_key).strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to Streamlit secrets under [openai] api_key "
            "or set it as an environment variable."
        )
    return OpenAI(api_key=api_key)

def call_gpt_for_pcl_decision(
    prompt_text: str,
    map_patch_png_bytes: bytes,
    clicked_lat: float,
    clicked_lon: float,
    clicked_pcl: float,
    threshold: float,
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    client = get_openai_client()

    image_data_url = png_bytes_to_data_url(map_patch_png_bytes)

    schema = {
        "name": "pcl_decision",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "selected_candidate_index": {"type": "integer"},
                "use_candidate": {"type": "boolean"},
                "reasoning_summary": {"type": "string"},
                "threshold_used": {"type": "number"},
                "requested_line_length_m": {"type": "number"},
                "prefer_east": {"type": "boolean"},
                "upwind_side": {"type": "string"},
                "downwind_side": {"type": "string"},
                "confidence": {"type": "number"},
                "answer_markdown": {"type": "string"},
            },
            "required": [
                "selected_candidate_index",
                "use_candidate",
                "reasoning_summary",
                "threshold_used",
                "requested_line_length_m",
                "prefer_east",
                "upwind_side",
                "downwind_side",
                "confidence",
                "answer_markdown",
            ],
        },
    }

    candidate_text_lines = []
    for i, c in enumerate(candidates, start=1):
        candidate_text_lines.append(
            (
                f"Candidate {i}: "
                f"distance_m={c['distance_m']:.1f}, "
                f"size_pixels={c['size_pixels']}, "
                f"pcl_mean={c['pcl_mean']:.2f}, "
                f"pcl_max={c['pcl_max']:.2f}, "
                f"orientation_deg={c['orientation_deg']:.1f}, "
                f"east_west_score={c['east_west_score']:.2f}, "
                f"nearest_lat={c['nearest_lat']:.6f}, nearest_lon={c['nearest_lon']:.6f}, "
                f"centroid_lat={c['centroid_lat']:.6f}, centroid_lon={c['centroid_lon']:.6f}"
            )
        )
    candidate_text = "\n".join(candidate_text_lines)

    system_prompt = (
        "You are assisting with wildfire control-opportunity interpretation from a PCL raster. "
        "You must answer strictly from the user prompt, the clicked point, the candidate summaries, "
        "and the supplied patch image. "
        "You are not allowed to invent a new candidate. "
        "Select the best candidate from the provided list if one satisfies the prompt. "
        "When the prompt asks for a continuous line, prefer candidates that are nearby, high PCL, "
        "and reasonably east-west oriented when wind guidance says west-to-east. "
        "Return valid JSON only."
    )

    user_text = f"""
User prompt:
{prompt_text}

Clicked point:
- lat={clicked_lat:.6f}
- lon={clicked_lon:.6f}
- pcl_at_click={clicked_pcl:.4f}

Interpretation:
- low PCL = lower probability of control
- high PCL = higher probability of control
- default threshold target is {threshold:.1f}

Candidate features:
{candidate_text}

Image:
- The image is a local patch around the clicked point.
- The clicked point is marked with an X.
- Candidate nearest points are labeled 1..N on the image.

Instructions:
- Choose the best candidate index from the provided list.
- If none are suitable, set use_candidate=false and still pick the nearest-best fallback index.
- Keep requested_line_length_m near 300 if the prompt asks for approximately 300 m.
- Wind is west -> east at 15 mph, so west is upwind and east is downwind.
- prefer_east=false means the traced segment should start on the west/upwind side when possible.
- answer_markdown should be a concise user-facing answer.
""".strip()

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.1,
        response_format={
            "type": "json_schema",
            "json_schema": schema,
        },
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    )

    content = resp.choices[0].message.content
    return safe_json_loads(content)


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="PCL GPT Map-Click", layout="wide")
st.title("PCL GPT Map-Click")
st.caption("Workflow: 1) Load GeoTIFF from GitHub URL → 2) See PCL overlay → 3) Click a point → 4) Type question → 5) RUN with GPT")

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_result_map_key" not in st.session_state:
    st.session_state.last_result_map_key = 0

if "clicked_lat" not in st.session_state:
    st.session_state.clicked_lat = None
    st.session_state.clicked_lon = None
    st.session_state.map_center = None
    st.session_state.map_zoom = 12

if "loaded_tif_url" not in st.session_state:
    st.session_state.loaded_tif_url = DEFAULT_TIF_URL

if "loaded_tif_path" not in st.session_state:
    st.session_state.loaded_tif_path = None

st.subheader("1) Load PCL GeoTIFF from GitHub")
tif_url = st.text_input(
    "GeoTIFF URL",
    value=st.session_state.loaded_tif_url,
    help="Raw GitHub URL preferred. Blob URLs are converted automatically.",
)

load_col1, load_col2 = st.columns([1, 4])
with load_col1:
    load_btn = st.button("Load raster", type="primary", use_container_width=True)

if load_btn or (st.session_state.loaded_tif_path is None and tif_url.strip()):
    try:
        with st.spinner("Downloading GeoTIFF from URL..."):
            local_tif_path = download_tif_from_url(tif_url)
        st.session_state.loaded_tif_url = resolve_github_raw_url(tif_url)
        st.session_state.loaded_tif_path = local_tif_path
        st.success("Raster loaded.")
    except Exception as e:
        st.error(f"Could not download/load raster from URL: {e}")
        st.stop()

if not st.session_state.loaded_tif_path or not os.path.exists(st.session_state.loaded_tif_path):
    st.warning("No raster loaded.")
    st.stop()

try:
    ds = rasterio.open(st.session_state.loaded_tif_path)
except Exception as e:
    st.error(f"Could not open raster: {e}")
    st.stop()

band_full = ds.read(1, masked=True).astype("float32")
arr_full = band_full.data.copy()
if hasattr(band_full, "mask"):
    arr_full[band_full.mask] = np.nan

with st.spinner("Rendering PCL overlay..."):
    arr_ds, _ = downsample_for_display(arr_full, max_dim=1400)
    png_bytes = render_overlay_png(arr_ds, cmap="viridis")
    img_url = png_bytes_to_data_url(png_bytes)

bounds_ll = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
minx, miny, maxx, maxy = bounds_ll
center_lat = (miny + maxy) / 2
center_lon = (minx + maxx) / 2

if st.session_state.map_center is None:
    st.session_state.map_center = [center_lat, center_lon]

DEFAULT_PROMPT = """Where is the closest area to this point with high PCL values that forms a continuous line?

Constraints / interpretation:
- Treat PCL >= 15 as a good candidate control opportunity (relative to this raster).
- Find the nearest connected (8-neighborhood) “continuous” feature made of PCL>=15 pixels.
- From that feature, highlight an approximately 300 m-long line segment along the feature.
- Wind is blowing West -> East at 15 mph:
  * Note: PCL is not wind-aware by itself. Use wind only to orient the highlighted segment (prefer an E-W oriented line; and describe which side is upwind/downwind).
Output:
- Report nearest feature location (nearest pixel and centroid), distance from click (meters), and PCL threshold used (15).
- Draw the ~300 m line on the map.
""".strip()

colL, colR = st.columns([1.3, 1.0], gap="large")

with colL:
    st.subheader("2) PCL overlay (click a point)")
    m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom, control_scale=True, tiles="CartoDB positron")

    folium.raster_layers.ImageOverlay(
        name="PCL",
        image=img_url,
        bounds=[[miny, minx], [maxy, maxx]],
        opacity=0.75,
        interactive=True,
        cross_origin=False,
        zindex=2,
    ).add_to(m)
    folium.LayerControl().add_to(m)

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

    out = st_folium(
        m,
        height=650,
        width=None,
        returned_objects=["last_clicked", "center", "zoom"],
        key="pcl_basic_map",
    )

    if out:
        if out.get("center"):
            st.session_state.map_center = [float(out["center"]["lat"]), float(out["center"]["lng"])]
        if out.get("zoom") is not None:
            st.session_state.map_zoom = int(out["zoom"])
        if out.get("last_clicked"):
            st.session_state.clicked_lat = float(out["last_clicked"]["lat"])
            st.session_state.clicked_lon = float(out["last_clicked"]["lng"])

with colR:
    st.subheader("3) Click → ask → RUN")

    if st.session_state.clicked_lat is None:
        st.info("Click on the map to select a location.")
    else:
        lat = float(st.session_state.clicked_lat)
        lon = float(st.session_state.clicked_lon)
        st.markdown(f"**Selected:** lat={lat:.6f}, lon={lon:.6f}")

        pcl_val = sample_pcl_at_latlon(ds, lat, lon)
        if pcl_val is None:
            st.warning("No valid PCL here (outside raster or NoData). Click somewhere else.")
        else:
            st.markdown(f"**PCL at selected point:** `{pcl_val:.4f}`")
            st.caption("Interpretation: low PCL = low probability of control, high PCL = high probability of control.")

            threshold = st.number_input("PCL threshold", min_value=0.0, max_value=1000.0, value=float(PCL_THRESHOLD_DEFAULT), step=1.0)
            question = st.text_area("Question prompt", value=DEFAULT_PROMPT, height=240)

            st.markdown("---")
            run = st.button("RUN with GPT", type="primary", use_container_width=True)

            if run:
                try:
                    center_row, center_col = latlon_to_rowcol(ds, lat, lon)

                    search_patch, r0, c0 = read_patch(ds, center_row, center_col, SEARCH_PATCH_HALF_PX)
                    candidates = nearest_component_candidates(
                        ds=ds,
                        arr_patch=search_patch,
                        r0=r0,
                        c0=c0,
                        center_row=center_row,
                        center_col=center_col,
                        thr=float(threshold),
                        min_comp_pixels=25,
                        max_candidates=MAX_CANDIDATES_FOR_GPT,
                    )

                    if not candidates:
                        st.session_state.last_result = {
                            "type": "warning",
                            "text": f"No connected high-PCL (>= {threshold:.1f}) feature found in the search window."
                        }
                        st.session_state.last_result_map_key += 1
                    else:
                        patch_for_gpt, pg_r0, pg_c0 = read_patch(ds, center_row, center_col, PATCH_DISPLAY_HALF_PX)
                        click_local_row = center_row - pg_r0
                        click_local_col = center_col - pg_c0

                        gpt_candidates = []
                        for i, cand in enumerate(candidates, start=1):
                            near_r, near_c = cand["nearest_pixel_rowcol"]
                            cent_r, cent_c = cand["centroid_rowcol"]
                            near_lat, near_lon = rowcol_to_latlon(ds, near_r, near_c)
                            cent_lat, cent_lon = rowcol_to_latlon(ds, cent_r, cent_c)

                            gpt_candidates.append(
                                {
                                    **cand,
                                    "nearest_lat": float(near_lat),
                                    "nearest_lon": float(near_lon),
                                    "centroid_lat": float(cent_lat),
                                    "centroid_lon": float(cent_lon),
                                    "nearest_pixel_local": (near_r - pg_r0, near_c - pg_c0),
                                }
                            )

                        patch_png = render_local_patch_png(
                            patch_for_gpt,
                            click_local_row,
                            click_local_col,
                            gpt_candidates,
                        )

                        decision = call_gpt_for_pcl_decision(
                            prompt_text=question,
                            map_patch_png_bytes=patch_png,
                            clicked_lat=lat,
                            clicked_lon=lon,
                            clicked_pcl=float(pcl_val),
                            threshold=float(threshold),
                            candidates=gpt_candidates,
                        )

                        idx = int(decision["selected_candidate_index"])
                        idx = max(1, min(idx, len(gpt_candidates)))
                        chosen = gpt_candidates[idx - 1]

                        line_len_target = float(decision.get("requested_line_length_m", 300.0))
                        line_len_target = max(50.0, min(line_len_target, 1200.0))
                        prefer_east = bool(decision.get("prefer_east", False))

                        line_result = build_300m_polyline_from_component(
                            ds=ds,
                            comp_points_rowcol=chosen["component_points_rowcol"],
                            prefer_east=prefer_east,
                            target_len_m=line_len_target,
                            fallback_px_m=30.0,
                        )

                        if line_result is None or len(line_result[0]) < 2:
                            st.session_state.last_result = {
                                "type": "warning",
                                "text": "GPT selected a candidate, but the line could not be traced from that component."
                            }
                            st.session_state.last_result_map_key += 1
                        else:
                            line_latlons, line_len_m, line_chain = line_result
                            near_r, near_c = chosen["nearest_pixel_rowcol"]
                            cent_r, cent_c = chosen["centroid_rowcol"]
                            near_lat, near_lon = rowcol_to_latlon(ds, near_r, near_c)
                            cent_lat, cent_lon = rowcol_to_latlon(ds, cent_r, cent_c)

                            st.session_state.last_result = {
                                "type": "answer",
                                "clicked_lat": lat,
                                "clicked_lon": lon,
                                "pcl_at_click": float(pcl_val),
                                "thr": float(decision.get("threshold_used", threshold)),
                                "nearest_lat": float(near_lat),
                                "nearest_lon": float(near_lon),
                                "centroid_lat": float(cent_lat),
                                "centroid_lon": float(cent_lon),
                                "distance_m": float(chosen["distance_m"]),
                                "distance_px": float(chosen["min_distance_pixels"]),
                                "px_m": float(chosen["px_m"]),
                                "component_size_pixels": int(chosen["size_pixels"]),
                                "component_pcl_mean": float(chosen["pcl_mean"]),
                                "component_pcl_max": float(chosen["pcl_max"]),
                                "orientation_deg": float(chosen["orientation_deg"]),
                                "east_west_score": float(chosen["east_west_score"]),
                                "line_latlons": line_latlons,
                                "line_len_m": float(line_len_m),
                                "line_target_m": float(line_len_target),
                                "gpt_answer_markdown": decision.get("answer_markdown", ""),
                                "gpt_reasoning_summary": decision.get("reasoning_summary", ""),
                                "gpt_confidence": float(decision.get("confidence", 0.0)),
                                "upwind_side": decision.get("upwind_side", "west"),
                                "downwind_side": decision.get("downwind_side", "east"),
                                "selected_candidate_index": idx,
                                "use_candidate": bool(decision.get("use_candidate", True)),
                                "wind_note": "Wind W→E @ 15 mph. West is upwind, east is downwind.",
                            }
                            st.session_state.last_result_map_key += 1

                except Exception as e:
                    st.session_state.last_result = {
                        "type": "warning",
                        "text": f"RUN failed: {e}"
                    }
                    st.session_state.last_result_map_key += 1

st.markdown("---")
st.header("Results")

r = st.session_state.last_result
if r is None:
    st.info("No results yet. Click a point, then RUN with GPT.")
else:
    if r["type"] == "warning":
        st.warning(r["text"])
    else:
        st.markdown(r["gpt_answer_markdown"])
        st.markdown(f"**GPT confidence:** `{r['gpt_confidence']:.2f}`")
        st.markdown(f"**Selected candidate:** `{r['selected_candidate_index']}`")
        st.markdown(f"**Threshold used:** `PCL ≥ {r['thr']:.1f}`")
        st.markdown(f"**PCL at clicked point:** `{r['pcl_at_click']:.4f}`")
        st.markdown(f"**Nearest feature point:** lat={r['nearest_lat']:.6f}, lon={r['nearest_lon']:.6f}")
        st.markdown(f"**Feature centroid:** lat={r['centroid_lat']:.6f}, lon={r['centroid_lon']:.6f}")
        st.markdown(f"**Distance from click to nearest high-PCL pixel:** **{r['distance_m']:.1f} m** (~{r['distance_px']:.2f} px; px≈{r['px_m']:.1f} m)")
        st.markdown(f"**Component size:** **{r['component_size_pixels']} px**")
        st.markdown(f"**Component mean / max PCL:** **{r['component_pcl_mean']:.2f} / {r['component_pcl_max']:.2f}**")
        st.markdown(f"**Component orientation (deg):** **{r['orientation_deg']:.1f}**")
        st.markdown(f"**East-west score:** **{r['east_west_score']:.2f}**")
        st.markdown(f"**Highlighted line length:** ~**{r['line_len_m']:.1f} m** (target ≈ {r['line_target_m']:.1f} m)")
        st.caption(r["wind_note"])
        st.caption(f"Upwind side: {r['upwind_side']} | Downwind side: {r['downwind_side']}")
        st.caption(r["gpt_reasoning_summary"])

        m2 = folium.Map(
            location=[r["clicked_lat"], r["clicked_lon"]],
            zoom_start=st.session_state.map_zoom,
            control_scale=True,
            tiles="CartoDB positron",
        )

        folium.raster_layers.ImageOverlay(
            name="PCL",
            image=img_url,
            bounds=[[miny, minx], [maxy, maxx]],
            opacity=0.75,
            interactive=True,
            cross_origin=False,
            zindex=2,
        ).add_to(m2)

        folium.CircleMarker(
            location=[r["clicked_lat"], r["clicked_lon"]],
            radius=8,
            weight=2,
            color="yellow",
            fill=True,
            fill_opacity=0.35,
            tooltip="Clicked point",
        ).add_to(m2)

        folium.CircleMarker(
            location=[r["nearest_lat"], r["nearest_lon"]],
            radius=8,
            weight=2,
            color="red",
            fill=True,
            fill_opacity=0.35,
            tooltip="Selected feature nearest point",
        ).add_to(m2)

        folium.PolyLine(
            locations=r["line_latlons"],
            weight=5,
            opacity=0.95,
            tooltip="GPT-selected line",
        ).add_to(m2)

        folium.LayerControl().add_to(m2)

        st.subheader("Map of result")
        st_folium(m2, height=520, width=None, key=f"pcl_result_map_{st.session_state.last_result_map_key}")



