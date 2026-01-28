# app_pcl_query.py
# pip install streamlit rasterio pyproj numpy matplotlib folium streamlit-folium

import io
import re
import base64
import numpy as np
import matplotlib.pyplot as plt

import streamlit as st
from streamlit_folium import st_folium

import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_bounds
from pyproj import Transformer

import folium
from branca.utilities import image_to_url


PCL_TIF = r"C:\Users\magst\Desktop\RCVFD_gis\RCVFD\PCL_RCVFD.tif"


def robust_minmax(a, lo=2, hi=98):
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 1.0
    return np.percentile(a, lo), np.percentile(a, hi)


def render_overlay_png(data2d, nodata=None, cmap="viridis"):
    arr = data2d.astype("float32", copy=False)

    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)

    vmin, vmax = robust_minmax(arr)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = np.nanmin(arr), np.nanmax(arr)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0

    fig = plt.figure(figsize=(6, 6), dpi=200)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_axis_off()
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_facecolor((0, 0, 0, 0))
    fig.patch.set_alpha(0)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def downsample_for_display(data, max_dim=1200):
    h, w = data.shape
    scale = max(h / max_dim, w / max_dim, 1.0)
    step = int(np.ceil(scale))
    return data[::step, ::step], step


def sample_at_latlon(ds, lat, lon):
    # lat/lon -> dataset CRS -> row/col
    transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    row, col = ds.index(x, y)

    if row < 0 or col < 0 or row >= ds.height or col >= ds.width:
        return None

    win = Window(col, row, 1, 1)
    val = ds.read(1, window=win, masked=True)[0, 0]
    if np.ma.is_masked(val):
        return None

    return float(val)


def local_stats_at_latlon(ds, lat, lon, radius_m=150):
    transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    row, col = ds.index(x, y)

    if row < 0 or col < 0 or row >= ds.height or col >= ds.width:
        return None

    # approximate meters per pixel (works fine for projected CRS)
    try:
        resx, resy = ds.res
        px = max(abs(resx), abs(resy))
    except Exception:
        px = 30.0

    r = max(1, int(np.ceil(radius_m / px)))
    r0 = max(0, row - r)
    r1 = min(ds.height, row + r + 1)
    c0 = max(0, col - r)
    c1 = min(ds.width, col + r + 1)

    win = Window(c0, r0, c1 - c0, r1 - r0)
    block = ds.read(1, window=win, masked=True)

    if block.count() == 0:
        return None

    vals = block.compressed().astype("float32")
    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "std": float(np.std(vals)),
        "radius_m": float(radius_m),
    }


def answer_question(question, pcl_value, stats=None):
    q = (question or "").strip().lower()

    if pcl_value is None:
        return "No valid PCL value at that clicked location (outside raster or NoData)."

    # lightweight intent detection
    want_value = bool(re.search(r"\b(value|pcl|score|what is|whats|what's)\b", q)) or q == ""
    want_explain = bool(re.search(r"\b(mean|median|min|max|std|nearby|around|within|radius|local)\b", q))
    want_interpret = bool(re.search(r"\b(good|bad|high|low|interpret|meaning|does it mean|should i)\b", q))

    lines = []

    if want_value or (not want_explain and not want_interpret):
        lines.append(f"PCL at the clicked point: **{pcl_value:.4f}**")

    if want_explain:
        if stats is None:
            lines.append("Could not compute local stats (NoData neighborhood or outside raster).")
        else:
            lines.append(
                f"Local PCL stats within ~{int(stats['radius_m'])} m "
                f"(n={stats['n']}): mean={stats['mean']:.4f}, median={stats['median']:.4f}, "
                f"min={stats['min']:.4f}, max={stats['max']:.4f}, std={stats['std']:.4f}"
            )

    if want_interpret:
        lines.append(
            "Interpretation (generic): higher PCL typically indicates *more favorable* potential control location "
            "relative to lower PCL, but the exact meaning depends on how your PCL raster was produced "
            "(normalization, thresholds, and covariates)."
        )

    return "\n\n".join(lines)


st.set_page_config(page_title="RCVFD PCL Click + Query", layout="wide")
st.title("RCVFD PCL: click the map, then ask about PCL at that location")

@st.cache_resource
def load_ds(path):
    return rasterio.open(path)

ds = load_ds(PCL_TIF)

# Read a display version (downsampled)
with st.spinner("Loading raster for display..."):
    data = ds.read(1, masked=True)
    nodata = ds.nodata
    data_ds, step = downsample_for_display(data.filled(np.nan), max_dim=1400)
    png_bytes = render_overlay_png(data_ds, nodata=None, cmap="viridis")

# Dataset bounds -> lat/lon bounds for folium
bounds_ll = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
minx, miny, maxx, maxy = bounds_ll  # lon/lat
center_lat = (miny + maxy) / 2
center_lon = (minx + maxx) / 2

m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")

# Add raster overlay
img_url = image_to_url(png_bytes, origin="upper")
folium.raster_layers.ImageOverlay(
    name="PCL",
    image=img_url,
    bounds=[[miny, minx], [maxy, maxx]],  # [[south, west], [north, east]]
    opacity=0.75,
    interactive=True,
    cross_origin=False,
    zindex=2,
).add_to(m)

folium.LayerControl().add_to(m)

colA, colB = st.columns([1.2, 1.0], gap="large")

with colA:
    st.subheader("Map (click anywhere on the overlay)")
    out = st_folium(m, width=900, height=700, returned_objects=["last_clicked"])
    clicked = out.get("last_clicked", None)

with colB:
    st.subheader("Query")
    if clicked is None:
        st.info("Click a point on the map to sample PCL.")
        st.stop()

    lat = float(clicked["lat"])
    lon = float(clicked["lng"])

    pcl_val = sample_at_latlon(ds, lat, lon)
    st.markdown(f"**Clicked:** lat={lat:.6f}, lon={lon:.6f}")

    # Optional neighborhood stats controls
    radius_m = st.slider("Neighborhood radius (meters)", min_value=30, max_value=1000, value=150, step=10)
    stats = local_stats_at_latlon(ds, lat, lon, radius_m=radius_m)

    q = st.text_input(
        "Ask a question about PCL at this location",
        value="What is the PCL value here, and what are nearby stats?",
        placeholder="e.g., 'Is this a good control location?' or 'nearby mean/median within 200m'",
    )

    resp = answer_question(q, pcl_val, stats=stats)
    st.markdown(resp)
