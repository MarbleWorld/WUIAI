# # app_pcl_query.py
# # pip install streamlit rasterio pyproj numpy matplotlib folium streamlit-folium
# import base64

# import io
# import os
# import tempfile
# import numpy as np
# import matplotlib.pyplot as plt

# import streamlit as st
# from streamlit_folium import st_folium

# import rasterio
# from rasterio.windows import Window
# from rasterio.warp import transform_bounds
# from pyproj import Transformer

# import folium
# #from branca.utilities import image_to_url


# def robust_minmax(a, lo=2, hi=98):
#     a = a[np.isfinite(a)]
#     if a.size == 0:
#         return 0.0, 1.0
#     return np.percentile(a, lo), np.percentile(a, hi)


# def render_overlay_png(data2d, nodata=None, cmap="viridis"):
#     arr = data2d.astype("float32", copy=False)
#     if nodata is not None:
#         arr = np.where(arr == nodata, np.nan, arr)

#     vmin, vmax = robust_minmax(arr)
#     if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
#         vmin, vmax = np.nanmin(arr), np.nanmax(arr)
#         if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
#             vmin, vmax = 0.0, 1.0

#     fig = plt.figure(figsize=(6, 6), dpi=200)
#     ax = plt.axes([0, 0, 1, 1])
#     ax.set_axis_off()
#     ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
#     ax.set_facecolor((0, 0, 0, 0))
#     fig.patch.set_alpha(0)

#     buf = io.BytesIO()
#     plt.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0)
#     plt.close(fig)
#     buf.seek(0)
#     return buf.getvalue()


# def downsample_for_display(data, max_dim=1200):
#     h, w = data.shape
#     scale = max(h / max_dim, w / max_dim, 1.0)
#     step = int(np.ceil(scale))
#     return data[::step, ::step], step


# def sample_at_latlon(ds, lat, lon):
#     transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
#     x, y = transformer.transform(lon, lat)
#     row, col = ds.index(x, y)

#     if row < 0 or col < 0 or row >= ds.height or col >= ds.width:
#         return None

#     win = Window(col, row, 1, 1)
#     val = ds.read(1, window=win, masked=True)[0, 0]
#     if np.ma.is_masked(val):
#         return None
#     return float(val)


# def local_stats_at_latlon(ds, lat, lon, radius_m=150):
#     transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
#     x, y = transformer.transform(lon, lat)
#     row, col = ds.index(x, y)

#     if row < 0 or col < 0 or row >= ds.height or col >= ds.width:
#         return None

#     try:
#         resx, resy = ds.res
#         px = max(abs(resx), abs(resy))
#     except Exception:
#         px = 30.0

#     r = max(1, int(np.ceil(radius_m / px)))
#     r0 = max(0, row - r)
#     r1 = min(ds.height, row + r + 1)
#     c0 = max(0, col - r)
#     c1 = min(ds.width, col + r + 1)

#     win = Window(c0, r0, c1 - c0, r1 - r0)
#     block = ds.read(1, window=win, masked=True)
#     if block.count() == 0:
#         return None

#     vals = block.compressed().astype("float32")
#     return {
#         "n": int(vals.size),
#         "mean": float(np.mean(vals)),
#         "median": float(np.median(vals)),
#         "min": float(np.min(vals)),
#         "max": float(np.max(vals)),
#         "std": float(np.std(vals)),
#         "radius_m": float(radius_m),
#     }


# st.set_page_config(page_title="RCVFD PCL Click + Query", layout="wide")
# st.title("RCVFD PCL: click the map, sample the raster")

# uploaded = st.file_uploader("Upload your PCL GeoTIFF (.tif)", type=["tif", "tiff"])

# if uploaded is None:
#     st.info("Upload the PCL_RCVFD.tif file to start.")
#     st.stop()

# # Write upload to a temp file so rasterio can open it
# suffix = ".tif"
# with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#     tmp.write(uploaded.getbuffer())
#     tmp_path = tmp.name

# # Open raster
# try:
#     ds = rasterio.open(tmp_path)
# except Exception as e:
#     st.error(f"Failed to open the uploaded raster: {e}")
#     st.stop()

# # Build display overlay
# with st.spinner("Preparing map overlay..."):
#     data = ds.read(1, masked=True).astype("float32")
#     data_ds, step = downsample_for_display(data.filled(np.nan), max_dim=1400)
#     png_bytes = render_overlay_png(data_ds, nodata=None, cmap="viridis")

# bounds_ll = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
# minx, miny, maxx, maxy = bounds_ll
# center_lat = (miny + maxy) / 2
# center_lon = (minx + maxx) / 2

# m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")

# #img_url = image_to_url(png_bytes, origin="upper")
# img_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")
# folium.raster_layers.ImageOverlay(
#     name="PCL",
#     image=img_url,
#     bounds=[[miny, minx], [maxy, maxx]],
#     opacity=0.75,
#     interactive=True,
#     cross_origin=False,
#     zindex=2,
# ).add_to(m)

# folium.LayerControl().add_to(m)

# colA, colB = st.columns([1.2, 1.0], gap="large")

# with colA:
#     st.subheader("Map (click to query PCL)")
#     out = st_folium(m, width=900, height=700, returned_objects=["last_clicked"])
#     clicked = out.get("last_clicked", None)

# with colB:
#     st.subheader("Result")
#     if clicked is None:
#         st.info("Click on the map to sample PCL.")
#         st.stop()

#     lat = float(clicked["lat"])
#     lon = float(clicked["lng"])
#     st.markdown(f"**Clicked:** lat={lat:.6f}, lon={lon:.6f}")

#     pcl_val = sample_at_latlon(ds, lat, lon)
#     if pcl_val is None:
#         st.warning("No valid PCL value here (outside raster or NoData).")
#         st.stop()

#     radius_m = st.slider("Neighborhood radius (meters)", 30, 1000, 150, 10)
#     stats = local_stats_at_latlon(ds, lat, lon, radius_m=radius_m)

#     st.markdown(f"**PCL at point:** {pcl_val:.4f}")
#     if stats is not None:
#         st.markdown(
#             f"**Local stats (~{int(stats['radius_m'])} m, n={stats['n']}):** "
#             f"mean={stats['mean']:.4f}, median={stats['median']:.4f}, "
#             f"min={stats['min']:.4f}, max={stats['max']:.4f}, std={stats['std']:.4f}"
#         )

# # cleanup temp file on rerun/exit (best-effort)
# try:
#     if os.path.exists(tmp_path):
#         os.remove(tmp_path)
# except Exception:
#     pass







# app_pcl_query_llm.py
# Run:
#   pip install streamlit rasterio pyproj numpy folium streamlit-folium openai
#   streamlit run app_pcl_query_llm.py
#
# Streamlit Cloud:
#   - Add OPENAI_API_KEY in Secrets
#   - Prefer runtime.txt with: python-3.11

import os
import json
import base64
import tempfile
import numpy as np
import streamlit as st

import folium
from streamlit_folium import st_folium

import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_bounds
from pyproj import Transformer

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


DEFAULT_MODEL = "gpt-4o-mini"

DEFAULT_SYSTEM = """You are a wildfire operations GIS assistant.
You will be given:
- a user question
- a point location (lat/lon)
- the PCL value at that point
- summary statistics of PCL in a small neighborhood around the point

PCL = Potential Control Location (higher usually indicates more favorable control potential, but interpretation depends on the raster's scale and how it was produced).
Answer clearly and concisely. If the user asks about thresholds or "good/bad", state uncertainty unless thresholds are provided.
"""

DEFAULT_QUESTION = """At this location, explain what the PCL implies operationally.
If the value is high, describe why it might be a favorable control area.
If it is low, describe what that might imply.
If you need additional context (fuel type, roads, slope, access), say what you'd want next.
Return short bullet points.
"""


def to_data_url_png(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")


def _nan_stats(arr: np.ndarray):
    x = arr[np.isfinite(arr)]
    if x.size == 0:
        return None
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "std": float(np.std(x)),
    }


def sample_point_and_patch(ds, lat, lon, half_window_px=12):
    """
    Returns:
      pcl_val (float|None),
      patch_stats (dict|None),
      patch_px (int),
      ds_xy (x,y) in dataset CRS (float,float)
    """
    transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
    x, y = transformer.transform(lon, lat)

    try:
        row, col = ds.index(x, y)
    except Exception:
        return None, None, half_window_px, (x, y)

    if row < 0 or col < 0 or row >= ds.height or col >= ds.width:
        return None, None, half_window_px, (x, y)

    # point value
    win_pt = Window(col, row, 1, 1)
    v = ds.read(1, window=win_pt, masked=True)[0, 0]
    pcl_val = None if np.ma.is_masked(v) else float(v)

    # neighborhood patch
    r0 = max(0, row - half_window_px)
    r1 = min(ds.height, row + half_window_px + 1)
    c0 = max(0, col - half_window_px)
    c1 = min(ds.width, col + half_window_px + 1)

    win = Window(c0, r0, c1 - c0, r1 - r0)
    block = ds.read(1, window=win, masked=True).astype("float32")

    arr = block.data.copy()
    if hasattr(block, "mask"):
        arr[block.mask] = np.nan
    patch_stats = _nan_stats(arr)

    return pcl_val, patch_stats, half_window_px, (x, y)


def call_llm(api_key, model, system_preamble, question, payload):
    if OpenAI is None:
        return "OpenAI package not available. Install openai or run without LLM mode."

    client = OpenAI(api_key=api_key)

    # Keep it simple: just pass structured JSON + the user's question
    msg = (
        "User question:\n"
        f"{question}\n\n"
        "Context JSON:\n"
        f"{json.dumps(payload, indent=2)}\n"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_preamble},
            {"role": "user", "content": msg},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


st.set_page_config(page_title="PCL Map-Click Q&A", layout="wide")
st.title("PCL Map-Click Q&A (upload GeoTIFF → click → ask)")

with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("Upload PCL GeoTIFF (.tif)", type=["tif", "tiff"])
    half_window_px = st.slider("Neighborhood half-window (pixels)", 1, 60, 12, 1)

    st.header("LLM (optional)")
    use_llm = st.checkbox("Use OpenAI to answer (optional)", value=False)
    model = st.text_input("Model", value=DEFAULT_MODEL)
    system_preamble = st.text_area("System preamble", value=DEFAULT_SYSTEM, height=160)
    question = st.text_area("Question", value=DEFAULT_QUESTION, height=160)

    api_key = os.getenv("OPENAI_API_KEY", "") or (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else "")
    if use_llm and not api_key:
        st.error("OPENAI_API_KEY not found. Add it in Streamlit → Manage app → Secrets.")

if uploaded is None:
    st.info("Upload your PCL GeoTIFF to start.")
    st.stop()

# write upload to temp, open with rasterio
with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
    tmp.write(uploaded.getbuffer())
    tmp_path = tmp.name

try:
    ds = rasterio.open(tmp_path)
except Exception as e:
    st.error(f"Failed to open raster: {e}")
    st.stop()

# map bounds in lat/lon
bounds_ll = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
minx, miny, maxx, maxy = bounds_ll
center_lat = (miny + maxy) / 2
center_lon = (minx + maxx) / 2

# init session state (pseudo double-click behavior)
if "pcl_clicked_lat" not in st.session_state:
    st.session_state.pcl_clicked_lat = None
    st.session_state.pcl_clicked_lon = None
    st.session_state.pcl_center = [center_lat, center_lon]
    st.session_state.pcl_zoom = 12
    st.session_state.last_answer = None
    st.session_state.last_payload = None

colA, colB = st.columns([1.2, 1.0], gap="large")

with colA:
    st.subheader("1) Click a location on the map")
    st.caption("Click to drop a point. Then ask a question about PCL at that point.")

    m = folium.Map(location=st.session_state.pcl_center, zoom_start=st.session_state.pcl_zoom, control_scale=True)

    # show raster footprint
    folium.Rectangle(bounds=[[miny, minx], [maxy, maxx]], color="black", weight=2, fill=False).add_to(m)

    if st.session_state.pcl_clicked_lat is not None and st.session_state.pcl_clicked_lon is not None:
        folium.CircleMarker(
            location=[st.session_state.pcl_clicked_lat, st.session_state.pcl_clicked_lon],
            radius=8,
            weight=2,
            color="yellow",
            fill=True,
            fill_opacity=0.25,
            tooltip="Selected point",
        ).add_to(m)

    out = st_folium(m, height=620, width=None, key="pcl_map")

    if out:
        if out.get("center"):
            st.session_state.pcl_center = [float(out["center"]["lat"]), float(out["center"]["lng"])]
        if out.get("zoom") is not None:
            st.session_state.pcl_zoom = int(out["zoom"])
        if out.get("last_clicked"):
            st.session_state.pcl_clicked_lat = float(out["last_clicked"]["lat"])
            st.session_state.pcl_clicked_lon = float(out["last_clicked"]["lng"])

with colB:
    st.subheader("2) Ask about PCL at the clicked point")

    if st.session_state.pcl_clicked_lat is None:
        st.info("No point selected yet. Click on the map.")
        st.stop()

    lat = float(st.session_state.pcl_clicked_lat)
    lon = float(st.session_state.pcl_clicked_lon)

    st.write(f"Selected point: **lat={lat:.6f}, lon={lon:.6f}**")

    pcl_val, patch_stats, hw, (x, y) = sample_point_and_patch(ds, lat, lon, half_window_px=half_window_px)

    if pcl_val is None and patch_stats is None:
        st.warning("No valid PCL data here (outside raster or NoData).")
        st.stop()

    # build payload for either local answer or LLM
    payload = {
        "pcl_raster": {
            "crs": str(ds.crs),
            "dtype": str(ds.dtypes[0]),
            "nodata": None if ds.nodata is None else float(ds.nodata),
            "bounds_latlon": {"min_lon": float(minx), "min_lat": float(miny), "max_lon": float(maxx), "max_lat": float(maxy)},
            "pixel_size": {"x": float(ds.res[0]), "y": float(ds.res[1])},
        },
        "query_point": {
            "lat": float(lat),
            "lon": float(lon),
            "x": float(x),
            "y": float(y),
        },
        "pcl_value_at_point": pcl_val,
        "neighborhood_half_window_pixels": int(hw),
        "neighborhood_stats": patch_stats,
    }

    st.session_state.last_payload = payload

    # default local response if no llm
    if not use_llm:
        st.markdown(f"**PCL at point:** {pcl_val if pcl_val is not None else 'NoData'}")
        if patch_stats is not None:
            st.markdown(
                f"**Neighborhood stats (±{hw}px):** "
                f"mean={patch_stats['mean']:.4f}, median={patch_stats['median']:.4f}, "
                f"min={patch_stats['min']:.4f}, max={patch_stats['max']:.4f}, std={patch_stats['std']:.4f}, n={patch_stats['n']}"
            )
        st.caption("Enable LLM mode in the sidebar if you want natural-language answers.")
    else:
        run = st.button("Run Q&A", type="primary", disabled=(not api_key))
        if run:
            with st.spinner("Asking the model..."):
                ans = call_llm(
                    api_key=api_key,
                    model=model,
                    system_preamble=system_preamble,
                    question=question,
                    payload=payload,
                )
            st.session_state.last_answer = ans

    if st.session_state.last_answer is not None:
        st.subheader("Answer")
        st.write(st.session_state.last_answer)

    with st.expander("Context JSON (what the model sees)"):
        st.code(json.dumps(payload, indent=2), language="json")

# cleanup temp file (best effort)
try:
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
except Exception:
    pass


