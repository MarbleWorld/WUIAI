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







# # PCL_basic_app.py
# # Run:
# #   pip install streamlit rasterio pyproj numpy folium streamlit-folium matplotlib
# #   streamlit run PCL_basic_app.py
# #
# # Streamlit Cloud tips:
# #   - Use runtime.txt: python-3.11
# #   - requirements.txt: streamlit, rasterio, pyproj, numpy, folium, streamlit-folium, matplotlib

# import base64
# import io
# import os
# import tempfile

# import numpy as np
# import matplotlib.pyplot as plt
# import streamlit as st
# import folium
# from streamlit_folium import st_folium

# import rasterio
# from rasterio.warp import transform_bounds
# from rasterio.windows import Window
# from pyproj import Transformer


# # -------------------------
# # Helpers
# # -------------------------
# def robust_minmax(a, lo=2, hi=98):
#     x = a[np.isfinite(a)]
#     if x.size == 0:
#         return 0.0, 1.0
#     return float(np.percentile(x, lo)), float(np.percentile(x, hi))


# def downsample_for_display(arr2d, max_dim=1400):
#     h, w = arr2d.shape
#     scale = max(h / max_dim, w / max_dim, 1.0)
#     step = int(np.ceil(scale))
#     return arr2d[::step, ::step], step


# def render_overlay_png(arr2d_float, cmap="viridis"):
#     arr = arr2d_float.astype("float32", copy=False)

#     vmin, vmax = robust_minmax(arr)
#     if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
#         vmin = float(np.nanmin(arr)) if np.isfinite(np.nanmin(arr)) else 0.0
#         vmax = float(np.nanmax(arr)) if np.isfinite(np.nanmax(arr)) else 1.0
#         if vmin == vmax:
#             vmax = vmin + 1.0

#     fig = plt.figure(figsize=(6, 6), dpi=220)
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


# def png_bytes_to_data_url(png_bytes):
#     return "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")


# def sample_pcl_at_latlon(ds, lat, lon):
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


# # -------------------------
# # UI
# # -------------------------
# st.set_page_config(page_title="PCL Basic Map-Click", layout="wide")
# st.title("PCL Basic Map-Click")

# st.caption("Workflow: 1) Upload GeoTIFF → 2) See PCL overlay → 3) Click a point → 4) Type question → 5) RUN")

# # Session state
# if "clicked_lat" not in st.session_state:
#     st.session_state.clicked_lat = None
#     st.session_state.clicked_lon = None
#     st.session_state.map_center = None
#     st.session_state.map_zoom = 12

# # Step 1: drag & drop
# uploaded = st.file_uploader("1) Drag & drop your PCL GeoTIFF (.tif)", type=["tif", "tiff"])

# if uploaded is None:
#     st.stop()

# # Write upload to temp, open raster
# with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
#     tmp.write(uploaded.getbuffer())
#     tmp_path = tmp.name

# try:
#     ds = rasterio.open(tmp_path)
# except Exception as e:
#     st.error(f"Could not open raster: {e}")
#     st.stop()

# # Step 2: build overlay + map
# with st.spinner("Rendering PCL overlay..."):
#     band = ds.read(1, masked=True).astype("float32")
#     arr = band.data.copy()
#     if hasattr(band, "mask"):
#         arr[band.mask] = np.nan

#     arr_ds, step = downsample_for_display(arr, max_dim=1400)
#     png_bytes = render_overlay_png(arr_ds, cmap="viridis")
#     img_url = png_bytes_to_data_url(png_bytes)

# bounds_ll = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
# minx, miny, maxx, maxy = bounds_ll
# center_lat = (miny + maxy) / 2
# center_lon = (minx + maxx) / 2

# if st.session_state.map_center is None:
#     st.session_state.map_center = [center_lat, center_lon]

# colL, colR = st.columns([1.3, 1.0], gap="large")

# with colL:
#     st.subheader("2) PCL overlay (click a point)")
#     m = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom, control_scale=True, tiles="CartoDB positron")

#     folium.raster_layers.ImageOverlay(
#         name="PCL",
#         image=img_url,
#         bounds=[[miny, minx], [maxy, maxx]],
#         opacity=0.75,
#         interactive=True,
#         cross_origin=False,
#         zindex=2,
#     ).add_to(m)

#     folium.LayerControl().add_to(m)

#     if st.session_state.clicked_lat is not None and st.session_state.clicked_lon is not None:
#         folium.CircleMarker(
#             location=[st.session_state.clicked_lat, st.session_state.clicked_lon],
#             radius=8,
#             weight=2,
#             color="yellow",
#             fill=True,
#             fill_opacity=0.25,
#             tooltip="Selected point",
#         ).add_to(m)

#     out = st_folium(m, height=650, width=None, returned_objects=["last_clicked", "center", "zoom"], key="pcl_basic_map")

#     if out:
#         if out.get("center"):
#             st.session_state.map_center = [float(out["center"]["lat"]), float(out["center"]["lng"])]
#         if out.get("zoom") is not None:
#             st.session_state.map_zoom = int(out["zoom"])
#         if out.get("last_clicked"):
#             st.session_state.clicked_lat = float(out["last_clicked"]["lat"])
#             st.session_state.clicked_lon = float(out["last_clicked"]["lng"])

# with colR:
#     st.subheader("3) Click → ask → RUN")

#     # Auto-populate as steps happen
#     if st.session_state.clicked_lat is None:
#         st.info("Click on the map to select a location.")
#         st.stop()

#     lat = float(st.session_state.clicked_lat)
#     lon = float(st.session_state.clicked_lon)
#     st.markdown(f"**Selected:** lat={lat:.6f}, lon={lon:.6f}")

#     pcl_val = sample_pcl_at_latlon(ds, lat, lon)
#     if pcl_val is None:
#         st.warning("No valid PCL here (outside raster or NoData). Click somewhere else.")
#         st.stop()

#     st.markdown(f"**PCL at selected point:** `{pcl_val:.4f}`")

#     question = st.text_area(
#         "Question about PCL at this point",
#         value="Does this PCL suggest a potentially favorable control location? What would you want to check next?",
#         height=120,
#     )

#     st.markdown("---")
#     run = st.button("RUN", type="primary", use_container_width=True)

#     if run:
#         # Basic (no OpenAI): answer using simple templated logic
#         # (You can swap this block for an OpenAI call later.)
#         lines = []
#         lines.append(f"**PCL={pcl_val:.4f}** at the clicked location.")
#         lines.append("")
#         lines.append("**How to interpret (generic):**")
#         lines.append("- Higher PCL generally indicates more favorable potential control opportunity relative to lower values.")
#         lines.append("- Exact meaning depends on how this raster was built (scaling, inputs, thresholds).")
#         lines.append("")
#         lines.append("**What I’d check next (to make it operational):**")
#         lines.append("- Access: road proximity / dozer suitability / safety zones")
#         lines.append("- Fuel breaks / canopy / surface fuel continuity near the point")
#         lines.append("- Slope/aspect and wind alignment")
#         lines.append("- Nearby anchor points (ridges, rivers, trails, previous lines)")
#         lines.append("")
#         lines.append("**Your question:**")
#         lines.append(f"> {question.strip()}")
#         st.markdown("\n".join(lines))

# # cleanup temp file (best effort)
# try:
#     if os.path.exists(tmp_path):
#         os.remove(tmp_path)
# except Exception:
#     pass
























# PCL_basic_app.py
# Run:
#   pip install streamlit rasterio pyproj numpy folium streamlit-folium matplotlib
#   streamlit run PCL_basic_app.py

import base64
import io
import os
import tempfile
import heapq
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import folium
from streamlit_folium import st_folium

import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from pyproj import Transformer


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


def render_overlay_png(arr2d_float, cmap="viridis"):
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
    """
    Label 8-connected components in a boolean mask.
    Returns labels array (int32) and list of components: each is list of (r,c).
    """
    h, w = mask.shape
    labels = -np.ones((h, w), dtype=np.int32)
    comps = []
    nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
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
    return labels, comps


def find_nearest_high_pcl_line(ds, lat, lon, arr_patch, r0, c0, center_row, center_col, thr, min_comp_pixels=25):
    """
    Interprets "continuous line" as a connected component of high-PCL pixels (8-connected).
    Finds nearest component to the clicked point, returns:
      - best component centroid (row,col) in full raster coords
      - min distance in pixels from point to component
      - component size
      - polyline-ish representative points (sampled)
    """
    if arr_patch is None or arr_patch.size == 0:
        return None

    mask = np.isfinite(arr_patch) & (arr_patch >= thr)
    if not mask.any():
        return None

    labels, comps = connected_components_8(mask)

    # coordinates of clicked point in patch coordinates
    pr = center_row - r0
    pc = center_col - c0

    best = None
    best_d2 = None

    # precompute high pixel coords per comp
    for pts in comps:
        if len(pts) < int(min_comp_pixels):
            continue

        # min distance from clicked point to any pixel in comp
        d2 = min((rr - pr) ** 2 + (cc - pc) ** 2 for rr, cc in pts)

        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best = pts

    if best is None:
        return None

    # centroid in patch coords
    rr_mean = int(round(np.mean([p[0] for p in best])))
    cc_mean = int(round(np.mean([p[1] for p in best])))

    # convert to full raster row/col
    full_r = r0 + rr_mean
    full_c = c0 + cc_mean

    # representative points along the component for drawing (sample every N points)
    step = max(1, len(best) // 50)
    reps_full = [(r0 + rr, c0 + cc) for rr, cc in best[::step]]

    return {
        "nearest_component_size_pixels": int(len(best)),
        "nearest_component_centroid_rowcol": (int(full_r), int(full_c)),
        "min_distance_pixels": float(np.sqrt(best_d2)) if best_d2 is not None else None,
        "rep_points_rowcol": reps_full,
    }


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="PCL Basic Map-Click", layout="wide")
st.title("PCL Basic Map-Click")
st.caption("Workflow: 1) Upload GeoTIFF → 2) See PCL overlay → 3) Click a point → 4) Type question → 5) RUN")

# Session state
if "clicked_lat" not in st.session_state:
    st.session_state.clicked_lat = None
    st.session_state.clicked_lon = None
    st.session_state.map_center = None
    st.session_state.map_zoom = 12

# Step 1: drag & drop
uploaded = st.file_uploader("1) Drag & drop your PCL GeoTIFF (.tif)", type=["tif", "tiff"])
if uploaded is None:
    st.stop()

# Write upload to temp, open raster
with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
    tmp.write(uploaded.getbuffer())
    tmp_path = tmp.name

try:
    ds = rasterio.open(tmp_path)
except Exception as e:
    st.error(f"Could not open raster: {e}")
    st.stop()

# Read full band once (used for overlay + analysis)
band_full = ds.read(1, masked=True).astype("float32")
arr_full = band_full.data.copy()
if hasattr(band_full, "mask"):
    arr_full[band_full.mask] = np.nan

# Step 2: overlay for map
with st.spinner("Rendering PCL overlay..."):
    arr_ds, ds_step = downsample_for_display(arr_full, max_dim=1400)
    png_bytes = render_overlay_png(arr_ds, cmap="viridis")
    img_url = png_bytes_to_data_url(png_bytes)

bounds_ll = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
minx, miny, maxx, maxy = bounds_ll
center_lat = (miny + maxy) / 2
center_lon = (minx + maxx) / 2

if st.session_state.map_center is None:
    st.session_state.map_center = [center_lat, center_lon]

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

    out = st_folium(m, height=650, width=None, returned_objects=["last_clicked", "center", "zoom"], key="pcl_basic_map")

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
        st.stop()

    lat = float(st.session_state.clicked_lat)
    lon = float(st.session_state.clicked_lon)
    st.markdown(f"**Selected:** lat={lat:.6f}, lon={lon:.6f}")

    # point value
    pcl_val = sample_pcl_at_latlon(ds, lat, lon)
    if pcl_val is None:
        st.warning("No valid PCL here (outside raster or NoData). Click somewhere else.")
        st.stop()

    st.markdown(f"**PCL at selected point:** `{pcl_val:.4f}`")

    question = st.text_area(
        "Question about PCL at this point",
        value="Where is the closest area to this point with high PCL values that forms a continuous line?",
        height=120,
    )

    st.markdown("---")
    run = st.button("RUN", type="primary", use_container_width=True)

    if run:
        q = question.strip().lower()

        # Heuristic: if user asks about nearest high continuous line, do that analysis
        wants_line = ("closest" in q or "nearest" in q) and ("line" in q or "continuous" in q) and ("high" in q)

        if not wants_line:
            st.markdown(
                "\n".join([
                    f"**PCL={pcl_val:.4f}** at the clicked location.",
                    "",
                    "I can answer operational questions like:",
                    "- closest high-PCL continuous line",
                    "- nearest high-PCL patch",
                    "- summarize PCL distribution nearby",
                    "",
                    "Ask using keywords like: *closest / high / continuous / line*.",
                ])
            )
        else:
            # Define "high" threshold as the 90th percentile of all finite PCL values
            finite = arr_full[np.isfinite(arr_full)]
            if finite.size < 100:
                st.error("Not enough finite PCL values to compute a threshold.")
            else:
                thr = float(np.percentile(finite, 90))

                # Search within a patch around the clicked point
                center_row, center_col = latlon_to_rowcol(ds, lat, lon)
                patch_half_px = 1200  # ~large search window; adjust if needed
                patch, r0, c0 = read_patch(ds, center_row, center_col, patch_half_px)

                res = find_nearest_high_pcl_line(
                    ds=ds,
                    lat=lat,
                    lon=lon,
                    arr_patch=patch,
                    r0=r0,
                    c0=c0,
                    center_row=center_row,
                    center_col=center_col,
                    thr=thr,
                    min_comp_pixels=25,
                )

                st.markdown(f"**Interpreting “high” as ≥ 90th percentile:** threshold = `{thr:.4f}`")

                if res is None:
                    st.warning("No sufficiently large continuous high-PCL component found in the search window. Try clicking closer to a ridge/road network or increase the search window / lower threshold in code.")
                else:
                    cent_r, cent_c = res["nearest_component_centroid_rowcol"]
                    tgt_lat, tgt_lon = rowcol_to_latlon(ds, cent_r, cent_c)

                    # Convert pixel distance to approximate meters (only meaningful in projected CRS)
                    try:
                        px = max(abs(ds.res[0]), abs(ds.res[1]))
                        dist_m = res["min_distance_pixels"] * px
                    except Exception:
                        dist_m = None

                    st.markdown(
                        "\n".join([
                            f"**Nearest continuous high-PCL ‘line’ (connected area) found.**",
                            f"- Component size: **{res['nearest_component_size_pixels']} pixels**",
                            f"- Component centroid: **lat={tgt_lat:.6f}, lon={tgt_lon:.6f}**",
                            f"- Approx distance from click: **{dist_m:.1f} m**" if dist_m is not None else f"- Distance from click: **{res['min_distance_pixels']:.1f} pixels**",
                        ])
                    )

                    # Draw the representative points on the map (as a polyline) + centroid marker
                    m2 = folium.Map(location=[lat, lon], zoom_start=st.session_state.map_zoom, control_scale=True, tiles="CartoDB positron")
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
                        location=[lat, lon],
                        radius=8,
                        weight=2,
                        color="yellow",
                        fill=True,
                        fill_opacity=0.35,
                        tooltip="Clicked point",
                    ).add_to(m2)

                    folium.CircleMarker(
                        location=[tgt_lat, tgt_lon],
                        radius=8,
                        weight=2,
                        color="red",
                        fill=True,
                        fill_opacity=0.35,
                        tooltip="Nearest high-PCL component centroid",
                    ).add_to(m2)

                    # Convert rep points to lat/lon
                    rep_latlons = [rowcol_to_latlon(ds, rr, cc) for rr, cc in res["rep_points_rowcol"]]
                    folium.PolyLine(locations=rep_latlons, weight=4, opacity=0.9).add_to(m2)

                    folium.LayerControl().add_to(m2)
                    st.subheader("Nearest high-PCL continuous feature (highlighted)")
                    st_folium(m2, height=520, width=None, key="pcl_result_map")

# cleanup temp file (best effort)
try:
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
except Exception:
    pass

















