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

















# # PCL_basic_app.py
# # Run:
# #   pip install streamlit rasterio pyproj numpy folium streamlit-folium matplotlib
# #   streamlit run PCL_basic_app.py

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


# def latlon_to_rowcol(ds, lat, lon):
#     transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
#     x, y = transformer.transform(lon, lat)
#     row, col = ds.index(x, y)
#     return row, col


# def rowcol_to_latlon(ds, row, col):
#     x, y = ds.xy(row, col)
#     transformer = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
#     lon, lat = transformer.transform(x, y)
#     return float(lat), float(lon)


# def sample_pcl_at_latlon(ds, lat, lon):
#     row, col = latlon_to_rowcol(ds, lat, lon)
#     if row < 0 or col < 0 or row >= ds.height or col >= ds.width:
#         return None
#     win = Window(col, row, 1, 1)
#     val = ds.read(1, window=win, masked=True)[0, 0]
#     if np.ma.is_masked(val):
#         return None
#     return float(val)


# def read_patch(ds, center_row, center_col, half_px):
#     r0 = max(0, center_row - half_px)
#     r1 = min(ds.height, center_row + half_px + 1)
#     c0 = max(0, center_col - half_px)
#     c1 = min(ds.width, center_col + half_px + 1)
#     win = Window(c0, r0, c1 - c0, r1 - r0)
#     block = ds.read(1, window=win, masked=True).astype("float32")
#     arr = block.data.copy()
#     if hasattr(block, "mask"):
#         arr[block.mask] = np.nan
#     return arr, r0, c0


# def connected_components_8(mask):
#     h, w = mask.shape
#     labels = -np.ones((h, w), dtype=np.int32)
#     comps = []
#     nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
#     cid = 0
#     for r in range(h):
#         for c in range(w):
#             if not mask[r, c] or labels[r, c] != -1:
#                 continue
#             stack = [(r, c)]
#             labels[r, c] = cid
#             pts = [(r, c)]
#             while stack:
#                 rr, cc = stack.pop()
#                 for dr, dc in nbrs:
#                     r2, c2 = rr + dr, cc + dc
#                     if 0 <= r2 < h and 0 <= c2 < w and mask[r2, c2] and labels[r2, c2] == -1:
#                         labels[r2, c2] = cid
#                         stack.append((r2, c2))
#                         pts.append((r2, c2))
#             comps.append(pts)
#             cid += 1
#     return comps


# def find_nearest_high_pcl_line(arr_patch, r0, c0, center_row, center_col, thr, min_comp_pixels=25):
#     if arr_patch is None or arr_patch.size == 0:
#         return None

#     mask = np.isfinite(arr_patch) & (arr_patch >= thr)
#     if not mask.any():
#         return None

#     comps = connected_components_8(mask)

#     pr = center_row - r0
#     pc = center_col - c0

#     best = None
#     best_d2 = None
#     best_nearest = None

#     for pts in comps:
#         if len(pts) < int(min_comp_pixels):
#             continue

#         nearest = min(pts, key=lambda p: (p[0] - pr) ** 2 + (p[1] - pc) ** 2)
#         d2 = (nearest[0] - pr) ** 2 + (nearest[1] - pc) ** 2

#         if best_d2 is None or d2 < best_d2:
#             best_d2 = d2
#             best = pts
#             best_nearest = nearest

#     if best is None:
#         return None

#     rr_mean = int(round(np.mean([p[0] for p in best])))
#     cc_mean = int(round(np.mean([p[1] for p in best])))

#     full_cent_r = r0 + rr_mean
#     full_cent_c = c0 + cc_mean

#     full_near_r = r0 + best_nearest[0]
#     full_near_c = c0 + best_nearest[1]

#     step = max(1, len(best) // 80)
#     reps_full = [(r0 + rr, c0 + cc) for rr, cc in best[::step]]

#     return {
#         "nearest_component_size_pixels": int(len(best)),
#         "component_centroid_rowcol": (int(full_cent_r), int(full_cent_c)),
#         "nearest_pixel_rowcol": (int(full_near_r), int(full_near_c)),
#         "min_distance_pixels": float(np.sqrt(best_d2)) if best_d2 is not None else None,
#         "rep_points_rowcol": reps_full,
#     }


# def approx_distance_m(ds, d_pixels):
#     try:
#         px = float(max(abs(ds.res[0]), abs(ds.res[1])))
#         if np.isfinite(px) and px > 0:
#             return float(d_pixels) * px
#     except Exception:
#         pass
#     return float(d_pixels) * 30.0


# # -------------------------
# # UI
# # -------------------------
# st.set_page_config(page_title="PCL Basic Map-Click", layout="wide")
# st.title("PCL Basic Map-Click")
# st.caption("Workflow: 1) Upload GeoTIFF → 2) See PCL overlay → 3) Click a point → 4) Type question → 5) RUN")

# # Persist outputs so they don't disappear
# if "last_result" not in st.session_state:
#     st.session_state.last_result = None
# if "last_result_map_key" not in st.session_state:
#     st.session_state.last_result_map_key = 0

# # Session state
# if "clicked_lat" not in st.session_state:
#     st.session_state.clicked_lat = None
#     st.session_state.clicked_lon = None
#     st.session_state.map_center = None
#     st.session_state.map_zoom = 12

# uploaded = st.file_uploader("1) Drag & drop your PCL GeoTIFF (.tif)", type=["tif", "tiff"])
# if uploaded is None:
#     st.stop()

# with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
#     tmp.write(uploaded.getbuffer())
#     tmp_path = tmp.name

# try:
#     ds = rasterio.open(tmp_path)
# except Exception as e:
#     st.error(f"Could not open raster: {e}")
#     st.stop()

# band_full = ds.read(1, masked=True).astype("float32")
# arr_full = band_full.data.copy()
# if hasattr(band_full, "mask"):
#     arr_full[band_full.mask] = np.nan

# with st.spinner("Rendering PCL overlay..."):
#     arr_ds, _ = downsample_for_display(arr_full, max_dim=1400)
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

#     out = st_folium(
#         m,
#         height=650,
#         width=None,
#         returned_objects=["last_clicked", "center", "zoom"],
#         key="pcl_basic_map",
#     )

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

#     if st.session_state.clicked_lat is None:
#         st.info("Click on the map to select a location.")
#     else:
#         lat = float(st.session_state.clicked_lat)
#         lon = float(st.session_state.clicked_lon)
#         st.markdown(f"**Selected:** lat={lat:.6f}, lon={lon:.6f}")

#         pcl_val = sample_pcl_at_latlon(ds, lat, lon)
#         if pcl_val is None:
#             st.warning("No valid PCL here (outside raster or NoData). Click somewhere else.")
#         else:
#             st.markdown(f"**PCL at selected point:** `{pcl_val:.4f}`")
#             st.caption("Interpretation: **low PCL = low probability of control**, **high PCL = high probability of control** (relative to this raster).")

#             question = st.text_area(
#                 "Question about PCL at this point",
#                 value="Where is the closest area to this point with high PCL values that forms a continuous line?",
#                 height=120,
#             )

#             st.markdown("---")
#             run = st.button("RUN", type="primary", use_container_width=True)

#             if run:
#                 q = question.strip().lower()
#                 wants_line = ("closest" in q or "nearest" in q) and ("line" in q or "continuous" in q) and ("high" in q)

#                 if not wants_line:
#                     st.session_state.last_result = {"type": "info", "text": "Try: 'closest high continuous line'."}
#                     st.session_state.last_result_map_key += 1
#                 else:
#                     finite = arr_full[np.isfinite(arr_full)]
#                     if finite.size < 100:
#                         st.session_state.last_result = {"type": "error", "text": "Not enough finite PCL values to compute a threshold."}
#                         st.session_state.last_result_map_key += 1
#                     else:
#                         thr = float(np.percentile(finite, 90))

#                         center_row, center_col = latlon_to_rowcol(ds, lat, lon)
#                         patch_half_px = 1200
#                         patch, r0, c0 = read_patch(ds, center_row, center_col, patch_half_px)

#                         res = find_nearest_high_pcl_line(
#                             arr_patch=patch,
#                             r0=r0,
#                             c0=c0,
#                             center_row=center_row,
#                             center_col=center_col,
#                             thr=thr,
#                             min_comp_pixels=25,
#                         )

#                         if res is None:
#                             st.session_state.last_result = {
#                                 "type": "warning",
#                                 "text": f"No sufficiently large continuous high-PCL component found (thr={thr:.4f}). Try another click."
#                             }
#                             st.session_state.last_result_map_key += 1
#                         else:
#                             near_r, near_c = res["nearest_pixel_rowcol"]
#                             cent_r, cent_c = res["component_centroid_rowcol"]

#                             near_lat, near_lon = rowcol_to_latlon(ds, near_r, near_c)
#                             cent_lat, cent_lon = rowcol_to_latlon(ds, cent_r, cent_c)

#                             dpx = float(res["min_distance_pixels"])
#                             dm = approx_distance_m(ds, dpx)

#                             # store EVERYTHING needed for rendering + guard against older session_state dicts
#                             st.session_state.last_result = {
#                                 "type": "answer",
#                                 "thr": thr,
#                                 "component_size_pixels": res["nearest_component_size_pixels"],
#                                 "nearest_lat": near_lat,
#                                 "nearest_lon": near_lon,
#                                 "centroid_lat": cent_lat,
#                                 "centroid_lon": cent_lon,
#                                 "distance_m": dm,
#                                 "distance_px": dpx,
#                                 "rep_points_rowcol": res["rep_points_rowcol"],
#                                 "clicked_lat": lat,
#                                 "clicked_lon": lon,
#                             }
#                             st.session_state.last_result_map_key += 1

# # -------------------------
# # Persistent Results Section (ALWAYS BELOW)
# # -------------------------
# st.markdown("---")
# st.header("Results")

# r = st.session_state.last_result

# # IMPORTANT: clear old/stale results that don't have the new keys (fixes your KeyError)
# required_keys = {"type"}
# if r is not None and isinstance(r, dict):
#     if r.get("type") == "answer":
#         required_keys = {
#             "type", "thr", "component_size_pixels", "nearest_lat", "nearest_lon",
#             "centroid_lat", "centroid_lon", "distance_m", "distance_px",
#             "rep_points_rowcol", "clicked_lat", "clicked_lon"
#         }
#     if not required_keys.issubset(set(r.keys())):
#         st.warning("Old cached result structure detected. Clearing previous result—hit RUN again.")
#         st.session_state.last_result = None
#         r = None

# if r is None:
#     st.info("No results yet. Click a point, type a question, then hit RUN.")
# else:
#     if r["type"] == "error":
#         st.error(r["text"])
#     elif r["type"] == "warning":
#         st.warning(r["text"])
#     elif r["type"] == "info":
#         st.info(r["text"])
#     else:
#         st.markdown(f"**High threshold used:** 90th percentile = `{r['thr']:.4f}`")
#         st.markdown(f"**Nearest continuous high-PCL component:** size = **{r['component_size_pixels']} px**")
#         st.markdown(f"**Nearest point on that feature:** lat={r['nearest_lat']:.6f}, lon={r['nearest_lon']:.6f}")
#         st.markdown(f"**Component centroid:** lat={r['centroid_lat']:.6f}, lon={r['centroid_lon']:.6f}")
#         st.markdown(f"**Approx distance from click to nearest high-PCL pixel:** **{r['distance_m']:.1f} m** (~{r['distance_px']:.2f} px)")
#         st.caption("Interpretation: **low PCL = low probability of control**, **high PCL = high probability of control** (relative to this raster).")

#         m2 = folium.Map(
#             location=[r["clicked_lat"], r["clicked_lon"]],
#             zoom_start=st.session_state.map_zoom,
#             control_scale=True,
#             tiles="CartoDB positron",
#         )

#         folium.raster_layers.ImageOverlay(
#             name="PCL",
#             image=img_url,
#             bounds=[[miny, minx], [maxy, maxx]],
#             opacity=0.75,
#             interactive=True,
#             cross_origin=False,
#             zindex=2,
#         ).add_to(m2)

#         folium.CircleMarker(
#             location=[r["clicked_lat"], r["clicked_lon"]],
#             radius=8,
#             weight=2,
#             color="yellow",
#             fill=True,
#             fill_opacity=0.35,
#             tooltip="Clicked point",
#         ).add_to(m2)

#         folium.CircleMarker(
#             location=[r["nearest_lat"], r["nearest_lon"]],
#             radius=8,
#             weight=2,
#             color="red",
#             fill=True,
#             fill_opacity=0.35,
#             tooltip="Nearest high-PCL pixel (on continuous feature)",
#         ).add_to(m2)

#         rep_latlons = [rowcol_to_latlon(ds, rr, cc) for rr, cc in r["rep_points_rowcol"]]
#         folium.PolyLine(locations=rep_latlons, weight=4, opacity=0.9).add_to(m2)

#         folium.LayerControl().add_to(m2)

#         st.subheader("Map of result (highlighted)")
#         st_folium(m2, height=520, width=None, key=f"pcl_result_map_persist_{st.session_state.last_result_map_key}")

# # cleanup temp file (best effort)
# try:
#     if os.path.exists(tmp_path):
#         os.remove(tmp_path)
# except Exception:
#     pass








# # PCL_basic_app.py
# # Run:
# #   pip install streamlit rasterio pyproj numpy folium streamlit-folium matplotlib
# #   streamlit run PCL_basic_app.py

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


# def latlon_to_rowcol(ds, lat, lon):
#     transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
#     x, y = transformer.transform(lon, lat)
#     row, col = ds.index(x, y)
#     return row, col


# def rowcol_to_latlon(ds, row, col):
#     x, y = ds.xy(row, col)
#     transformer = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
#     lon, lat = transformer.transform(x, y)
#     return float(lat), float(lon)


# def sample_pcl_at_latlon(ds, lat, lon):
#     row, col = latlon_to_rowcol(ds, lat, lon)
#     if row < 0 or col < 0 or row >= ds.height or col >= ds.width:
#         return None
#     win = Window(col, row, 1, 1)
#     val = ds.read(1, window=win, masked=True)[0, 0]
#     if np.ma.is_masked(val):
#         return None
#     return float(val)


# def read_patch(ds, center_row, center_col, half_px):
#     r0 = max(0, center_row - half_px)
#     r1 = min(ds.height, center_row + half_px + 1)
#     c0 = max(0, center_col - half_px)
#     c1 = min(ds.width, center_col + half_px + 1)
#     win = Window(c0, r0, c1 - c0, r1 - r0)
#     block = ds.read(1, window=win, masked=True).astype("float32")
#     arr = block.data.copy()
#     if hasattr(block, "mask"):
#         arr[block.mask] = np.nan
#     return arr, r0, c0


# def connected_components_8(mask):
#     h, w = mask.shape
#     labels = -np.ones((h, w), dtype=np.int32)
#     comps = []
#     nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
#     cid = 0
#     for r in range(h):
#         for c in range(w):
#             if not mask[r, c] or labels[r, c] != -1:
#                 continue
#             stack = [(r, c)]
#             labels[r, c] = cid
#             pts = [(r, c)]
#             while stack:
#                 rr, cc = stack.pop()
#                 for dr, dc in nbrs:
#                     r2, c2 = rr + dr, cc + dc
#                     if 0 <= r2 < h and 0 <= c2 < w and mask[r2, c2] and labels[r2, c2] == -1:
#                         labels[r2, c2] = cid
#                         stack.append((r2, c2))
#                         pts.append((r2, c2))
#             comps.append(pts)
#             cid += 1
#     return comps


# def approx_distance_m(ds, d_pixels, fallback_px_m=30.0):
#     try:
#         px = float(max(abs(ds.res[0]), abs(ds.res[1])))
#         if np.isfinite(px) and px > 0:
#             return float(d_pixels) * px, px
#     except Exception:
#         pass
#     return float(d_pixels) * float(fallback_px_m), float(fallback_px_m)


# def nearest_component_by_pixel_distance(arr_patch, r0, c0, center_row, center_col, thr, min_comp_pixels=25):
#     """
#     Find nearest connected component of arr_patch >= thr.
#     Distance computed to nearest pixel in component (NOT centroid).
#     Returns dict with component points in FULL raster coords.
#     """
#     if arr_patch is None or arr_patch.size == 0:
#         return None

#     mask = np.isfinite(arr_patch) & (arr_patch >= thr)
#     if not mask.any():
#         return None

#     comps = connected_components_8(mask)

#     pr = center_row - r0
#     pc = center_col - c0

#     best = None
#     best_d2 = None
#     best_nearest = None

#     for pts in comps:
#         if len(pts) < int(min_comp_pixels):
#             continue
#         nearest = min(pts, key=lambda p: (p[0] - pr) ** 2 + (p[1] - pc) ** 2)
#         d2 = (nearest[0] - pr) ** 2 + (nearest[1] - pc) ** 2
#         if best_d2 is None or d2 < best_d2:
#             best_d2 = d2
#             best = pts
#             best_nearest = nearest

#     if best is None:
#         return None

#     # full coords list for component
#     comp_full = [(r0 + rr, c0 + cc) for rr, cc in best]
#     near_full = (r0 + best_nearest[0], c0 + best_nearest[1])

#     rr_mean = int(round(np.mean([p[0] for p in best])))
#     cc_mean = int(round(np.mean([p[1] for p in best])))
#     cent_full = (r0 + rr_mean, c0 + cc_mean)

#     return {
#         "component_points_rowcol": comp_full,
#         "nearest_pixel_rowcol": (int(near_full[0]), int(near_full[1])),
#         "centroid_rowcol": (int(cent_full[0]), int(cent_full[1])),
#         "min_distance_pixels": float(np.sqrt(best_d2)) if best_d2 is not None else None,
#     }


# def build_300m_polyline_from_component(ds, comp_points_rowcol, prefer_east=True, target_len_m=300.0, fallback_px_m=30.0):
#     """
#     Construct an approximately target_len_m polyline along the component by:
#       - choosing a start near the most-west or most-east side (wind-aware preference)
#       - greedily stepping to neighboring component pixels (8-neigh) to form a chain
#     Returns:
#       - list of (lat,lon) for the chain
#       - achieved length (m)
#       - list of (row,col) in chain
#     """
#     if not comp_points_rowcol:
#         return None

#     # pixel size (m)
#     try:
#         px_m = float(max(abs(ds.res[0]), abs(ds.res[1])))
#         if not np.isfinite(px_m) or px_m <= 0:
#             px_m = float(fallback_px_m)
#     except Exception:
#         px_m = float(fallback_px_m)

#     target_steps = max(3, int(round(target_len_m / px_m)))

#     pts = comp_points_rowcol
#     pts_set = set(pts)

#     # choose start based on "wind from W->E" (prefer line segments oriented E-W and start on upwind side)
#     # For W->E wind, "upwind" is west. So pick a WEST-most start. (prefer_east=False means prefer westmost start)
#     cols = [c for _, c in pts]
#     if prefer_east:
#         start_col = max(cols)
#     else:
#         start_col = min(cols)

#     candidates = [p for p in pts if p[1] == start_col]
#     start = candidates[len(candidates) // 2] if candidates else pts[len(pts) // 2]

#     # greedy chain build: always pick an unvisited neighbor that pushes overall direction E-W (or W-E)
#     nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
#     chain = [start]
#     visited = {start}

#     # define desired dx: for W->E wind, we want chain to extend eastward (increasing col)
#     desired_sign = 1 if not prefer_east else -1  # if start is westmost, move east (+col). If start eastmost, move west (-col).

#     for _ in range(target_steps - 1):
#         r, c = chain[-1]
#         neigh = []
#         for dr, dc in nbrs:
#             p = (r + dr, c + dc)
#             if p in pts_set and p not in visited:
#                 neigh.append(p)
#         if not neigh:
#             break

#         # score: prioritize step in desired col direction, then keep going straight-ish
#         prev = chain[-2] if len(chain) >= 2 else None
#         best_p = None
#         best_score = None
#         for p in neigh:
#             rr, cc = p
#             # col progress
#             s = desired_sign * (cc - c)

#             # small penalty for reversing / zigzag
#             if prev is not None:
#                 pr, pc = prev
#                 # vector prev->cur and cur->p
#                 v1 = (r - pr, c - pc)
#                 v2 = (rr - r, cc - c)
#                 # encourage similar direction (dot product)
#                 dot = v1[0]*v2[0] + v1[1]*v2[1]
#             else:
#                 dot = 0

#             score = (10.0 * s) + (1.0 * dot)
#             if best_score is None or score > best_score:
#                 best_score = score
#                 best_p = p

#         if best_p is None:
#             break
#         chain.append(best_p)
#         visited.add(best_p)

#     # length estimate
#     length_m = max(0.0, (len(chain) - 1) * px_m)

#     # latlon chain
#     latlons = [rowcol_to_latlon(ds, r, c) for r, c in chain]
#     return latlons, float(length_m), chain


# # -------------------------
# # UI
# # -------------------------
# st.set_page_config(page_title="PCL Basic Map-Click", layout="wide")
# st.title("PCL Basic Map-Click")
# st.caption("Workflow: 1) Upload GeoTIFF → 2) See PCL overlay → 3) Click a point → 4) Type question → 5) RUN")

# if "last_result" not in st.session_state:
#     st.session_state.last_result = None
# if "last_result_map_key" not in st.session_state:
#     st.session_state.last_result_map_key = 0

# if "clicked_lat" not in st.session_state:
#     st.session_state.clicked_lat = None
#     st.session_state.clicked_lon = None
#     st.session_state.map_center = None
#     st.session_state.map_zoom = 12

# uploaded = st.file_uploader("1) Drag & drop your PCL GeoTIFF (.tif)", type=["tif", "tiff"])
# if uploaded is None:
#     st.stop()

# with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
#     tmp.write(uploaded.getbuffer())
#     tmp_path = tmp.name

# try:
#     ds = rasterio.open(tmp_path)
# except Exception as e:
#     st.error(f"Could not open raster: {e}")
#     st.stop()

# band_full = ds.read(1, masked=True).astype("float32")
# arr_full = band_full.data.copy()
# if hasattr(band_full, "mask"):
#     arr_full[band_full.mask] = np.nan

# with st.spinner("Rendering PCL overlay..."):
#     arr_ds, _ = downsample_for_display(arr_full, max_dim=1400)
#     png_bytes = render_overlay_png(arr_ds, cmap="viridis")
#     img_url = png_bytes_to_data_url(png_bytes)

# bounds_ll = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
# minx, miny, maxx, maxy = bounds_ll
# center_lat = (miny + maxy) / 2
# center_lon = (minx + maxx) / 2

# if st.session_state.map_center is None:
#     st.session_state.map_center = [center_lat, center_lon]

# # DEFAULT PROMPT (all-encompassing)
# DEFAULT_PROMPT = """Where is the closest area to this point with high PCL values that forms a continuous line?

# Constraints / interpretation:
# - Treat PCL >= 15 as a good candidate control opportunity (relative to this raster).
# - Find the nearest connected (8-neighborhood) “continuous” feature made of PCL>=15 pixels.
# - From that feature, highlight an approximately 300 m-long line segment along the feature.
# - Wind is blowing West -> East at 15 mph:
#   * Note: PCL is not wind-aware by itself. Use wind only to *orient* the highlighted segment (prefer an E-W oriented line; and describe which side is upwind/downwind).
# Output:
# - Report nearest feature location (nearest pixel and centroid), distance from click (meters), and PCL threshold used (15).
# - Draw the ~300 m line on the map.
# """.strip()

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

#     out = st_folium(
#         m,
#         height=650,
#         width=None,
#         returned_objects=["last_clicked", "center", "zoom"],
#         key="pcl_basic_map",
#     )

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

#     if st.session_state.clicked_lat is None:
#         st.info("Click on the map to select a location.")
#     else:
#         lat = float(st.session_state.clicked_lat)
#         lon = float(st.session_state.clicked_lon)
#         st.markdown(f"**Selected:** lat={lat:.6f}, lon={lon:.6f}")

#         pcl_val = sample_pcl_at_latlon(ds, lat, lon)
#         if pcl_val is None:
#             st.warning("No valid PCL here (outside raster or NoData). Click somewhere else.")
#         else:
#             st.markdown(f"**PCL at selected point:** `{pcl_val:.4f}`")
#             st.caption("Interpretation: **low PCL = low probability of control**, **high PCL = high probability of control** (relative to this raster).")

#             question = st.text_area("Question prompt", value=DEFAULT_PROMPT, height=240)

#             st.markdown("---")
#             run = st.button("RUN", type="primary", use_container_width=True)

#             if run:
#                 q = question.strip().lower()

#                 # We treat this as a fixed "closest continuous high line" task with explicit threshold 15+
#                 thr = 15.0

#                 center_row, center_col = latlon_to_rowcol(ds, lat, lon)

#                 # Search window: big enough to find nearest line; adjust if needed
#                 patch_half_px = 1800
#                 patch, r0, c0 = read_patch(ds, center_row, center_col, patch_half_px)

#                 comp = nearest_component_by_pixel_distance(
#                     arr_patch=patch,
#                     r0=r0,
#                     c0=c0,
#                     center_row=center_row,
#                     center_col=center_col,
#                     thr=thr,
#                     min_comp_pixels=25,
#                 )

#                 if comp is None:
#                     st.session_state.last_result = {
#                         "type": "warning",
#                         "text": "No connected high-PCL (>=15) feature found in the search window. Try clicking nearer to where you expect control opportunities, or increase patch_half_px in code."
#                     }
#                     st.session_state.last_result_map_key += 1
#                 else:
#                     near_r, near_c = comp["nearest_pixel_rowcol"]
#                     cent_r, cent_c = comp["centroid_rowcol"]

#                     near_lat, near_lon = rowcol_to_latlon(ds, near_r, near_c)
#                     cent_lat, cent_lon = rowcol_to_latlon(ds, cent_r, cent_c)

#                     dpx = float(comp["min_distance_pixels"])
#                     dist_m, px_m = approx_distance_m(ds, dpx, fallback_px_m=30.0)

#                     # Wind W->E: start on WEST-most side and extend EAST if possible
#                     # prefer_east=False means choose westmost start then step eastward
#                     line_latlons, line_len_m, line_chain = build_300m_polyline_from_component(
#                         ds=ds,
#                         comp_points_rowcol=comp["component_points_rowcol"],
#                         prefer_east=False,
#                         target_len_m=300.0,
#                         fallback_px_m=30.0,
#                     )

#                     if line_latlons is None or len(line_latlons) < 2:
#                         st.session_state.last_result = {
#                             "type": "warning",
#                             "text": "Found a high-PCL feature, but couldn't trace a ~300 m segment from it (component may be too small / blob-like). Try another area or lower min_comp_pixels."
#                         }
#                         st.session_state.last_result_map_key += 1
#                     else:
#                         st.session_state.last_result = {
#                             "type": "answer",
#                             "thr": thr,
#                             "clicked_lat": lat,
#                             "clicked_lon": lon,
#                             "pcl_at_click": float(pcl_val),
#                             "nearest_lat": float(near_lat),
#                             "nearest_lon": float(near_lon),
#                             "centroid_lat": float(cent_lat),
#                             "centroid_lon": float(cent_lon),
#                             "distance_m": float(dist_m),
#                             "distance_px": float(dpx),
#                             "px_m": float(px_m),
#                             "component_size_pixels": int(len(comp["component_points_rowcol"])),
#                             "line_latlons": line_latlons,
#                             "line_len_m": float(line_len_m),
#                             "wind_note": "Wind W→E @ 15 mph. PCL is not wind-aware; wind is used only to orient the highlighted segment (prefer an E–W-ish segment starting on the upwind/west side when possible).",
#                         }
#                         st.session_state.last_result_map_key += 1

# # -------------------------
# # Persistent Results Section
# # -------------------------
# st.markdown("---")
# st.header("Results")

# r = st.session_state.last_result
# if r is None:
#     st.info("No results yet. Click a point, then RUN.")
# else:
#     if r["type"] == "warning":
#         st.warning(r["text"])
#     else:
#         st.markdown(f"**Threshold used:** `PCL ≥ {r['thr']:.1f}` (you defined this as a good candidate)")
#         st.markdown(f"**PCL at clicked point:** `{r['pcl_at_click']:.4f}`")
#         st.markdown(f"**Nearest connected high-PCL feature:** size = **{r['component_size_pixels']} px**")
#         st.markdown(f"**Nearest point on that feature:** lat={r['nearest_lat']:.6f}, lon={r['nearest_lon']:.6f}")
#         st.markdown(f"**Feature centroid:** lat={r['centroid_lat']:.6f}, lon={r['centroid_lon']:.6f}")
#         st.markdown(f"**Distance from click to nearest high-PCL pixel:** **{r['distance_m']:.1f} m** (~{r['distance_px']:.2f} px; px≈{r['px_m']:.1f} m)")
#         st.markdown(f"**Highlighted line length:** ~**{r['line_len_m']:.1f} m** (target ≈ 300 m)")
#         st.caption(r["wind_note"])
#         st.caption("Interpretation reminder: **low PCL = low probability of control**, **high PCL = high probability of control** (relative to this raster).")

#         m2 = folium.Map(
#             location=[r["clicked_lat"], r["clicked_lon"]],
#             zoom_start=st.session_state.map_zoom,
#             control_scale=True,
#             tiles="CartoDB positron",
#         )

#         folium.raster_layers.ImageOverlay(
#             name="PCL",
#             image=img_url,
#             bounds=[[miny, minx], [maxy, maxx]],
#             opacity=0.75,
#             interactive=True,
#             cross_origin=False,
#             zindex=2,
#         ).add_to(m2)

#         folium.CircleMarker(
#             location=[r["clicked_lat"], r["clicked_lon"]],
#             radius=8,
#             weight=2,
#             color="yellow",
#             fill=True,
#             fill_opacity=0.35,
#             tooltip="Clicked point",
#         ).add_to(m2)

#         folium.CircleMarker(
#             location=[r["nearest_lat"], r["nearest_lon"]],
#             radius=8,
#             weight=2,
#             color="red",
#             fill=True,
#             fill_opacity=0.35,
#             tooltip="Nearest high-PCL pixel (PCL>=15 feature)",
#         ).add_to(m2)

#         folium.PolyLine(locations=r["line_latlons"], weight=5, opacity=0.95).add_to(m2)

#         folium.LayerControl().add_to(m2)

#         st.subheader("Map of result (highlighted ~300 m line on nearest PCL≥15 feature)")
#         st_folium(m2, height=520, width=None, key=f"pcl_result_map_{st.session_state.last_result_map_key}")

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
    return comps


def approx_distance_m(ds, d_pixels, fallback_px_m=30.0):
    try:
        px = float(max(abs(ds.res[0]), abs(ds.res[1])))
        if np.isfinite(px) and px > 0:
            return float(d_pixels) * px, px
    except Exception:
        pass
    return float(d_pixels) * float(fallback_px_m), float(fallback_px_m)


def nearest_component_by_pixel_distance(arr_patch, r0, c0, center_row, center_col, thr, min_comp_pixels=25):
    if arr_patch is None or arr_patch.size == 0:
        return None

    mask = np.isfinite(arr_patch) & (arr_patch >= thr)
    if not mask.any():
        return None

    comps = connected_components_8(mask)
    pr = center_row - r0
    pc = center_col - c0

    best = None
    best_d2 = None
    best_nearest = None

    for pts in comps:
        if len(pts) < int(min_comp_pixels):
            continue
        nearest = min(pts, key=lambda p: (p[0] - pr) ** 2 + (p[1] - pc) ** 2)
        d2 = (nearest[0] - pr) ** 2 + (nearest[1] - pc) ** 2
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best = pts
            best_nearest = nearest

    if best is None:
        return None

    comp_full = [(r0 + rr, c0 + cc) for rr, cc in best]
    near_full = (r0 + best_nearest[0], c0 + best_nearest[1])

    rr_mean = int(round(np.mean([p[0] for p in best])))
    cc_mean = int(round(np.mean([p[1] for p in best])))
    cent_full = (r0 + rr_mean, c0 + cc_mean)

    return {
        "component_points_rowcol": comp_full,
        "nearest_pixel_rowcol": (int(near_full[0]), int(near_full[1])),
        "centroid_rowcol": (int(cent_full[0]), int(cent_full[1])),
        "min_distance_pixels": float(np.sqrt(best_d2)) if best_d2 is not None else None,
    }


def build_300m_polyline_from_component(ds, comp_points_rowcol, start_west=True, target_len_m=300.0, fallback_px_m=30.0):
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
    start_col = min(cols) if start_west else max(cols)
    candidates = [p for p in pts if p[1] == start_col]
    start = candidates[len(candidates) // 2] if candidates else pts[len(pts) // 2]

    nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    chain = [start]
    visited = {start}

    # if starting on west edge, try to extend east (+col). if starting east edge, extend west (-col).
    desired_sign = 1 if start_west else -1

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
                dot = v1[0]*v2[0] + v1[1]*v2[1]
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


def classify_intent(question_text):
    """
    Minimal intent classifier:
      - if user asks about closest/nearest + line/continuous + high/good/PCL>=... => run "nearest_line"
      - else: fallback generic response (still grounded)
    """
    q = (question_text or "").strip().lower()
    if not q:
        return "nearest_line"
    tokens = q
    wants_nearest = ("closest" in tokens) or ("nearest" in tokens)
    wants_line = ("line" in tokens) or ("continuous" in tokens) or ("corridor" in tokens)
    wants_high = ("high" in tokens) or ("good" in tokens) or (">=" in tokens) or ("15" in tokens) or ("pcl" in tokens)
    if wants_nearest and wants_line and wants_high:
        return "nearest_line"
    return "generic"


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="PCL Basic Map-Click", layout="wide")
st.title("PCL Basic Map-Click")
st.caption("Workflow: 1) Upload GeoTIFF → 2) See PCL overlay → 3) Click a point → 4) Ask a question → 5) RUN")

# Hidden "behind-the-scenes" defaults
PCL_GOOD_THRESHOLD = 15.0
TARGET_LINE_M = 300.0
WIND_DIR_TEXT = "West → East"
WIND_MPH = 15.0
MIN_COMP_PIXELS = 25
PATCH_HALF_PX = 1800

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_result_map_key" not in st.session_state:
    st.session_state.last_result_map_key = 0
if "clicked_lat" not in st.session_state:
    st.session_state.clicked_lat = None
    st.session_state.clicked_lon = None
    st.session_state.map_center = None
    st.session_state.map_zoom = 12

uploaded = st.file_uploader("1) Drag & drop your PCL GeoTIFF (.tif)", type=["tif", "tiff"])
if uploaded is None:
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
    tmp.write(uploaded.getbuffer())
    tmp_path = tmp.name

try:
    ds = rasterio.open(tmp_path)
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

colL, colR = st.columns([1.3, 1.0], gap="large")

with colL:
    st.subheader("2) PCL overlay (click a point)")
    m = folium.Map(
        location=st.session_state.map_center,
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
    st.subheader("3) Ask → RUN")

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
            st.caption("Interpretation: **low PCL = low probability of control**, **high PCL = high probability of control** (relative to this raster).")

            # Blank-ish question box for the user, while app still "knows" defaults
            question = st.text_area(
                "Ask a question about PCL at/near this point",
                value="",
                placeholder="Example: Where is the closest continuous line of high PCL near this point?",
                height=120,
            )

            st.markdown("---")
            run = st.button("RUN", type="primary", use_container_width=True)

            if run:
                intent = classify_intent(question)

                # Default to nearest_line if blank (so button still does something useful)
                if intent == "nearest_line":
                    thr = float(PCL_GOOD_THRESHOLD)

                    center_row, center_col = latlon_to_rowcol(ds, lat, lon)
                    patch, r0, c0 = read_patch(ds, center_row, center_col, PATCH_HALF_PX)

                    comp = nearest_component_by_pixel_distance(
                        arr_patch=patch,
                        r0=r0,
                        c0=c0,
                        center_row=center_row,
                        center_col=center_col,
                        thr=thr,
                        min_comp_pixels=MIN_COMP_PIXELS,
                    )

                    if comp is None:
                        st.session_state.last_result = {
                            "type": "warning",
                            "text": f"No connected high-PCL (≥{thr:g}) feature found in the search window. Try clicking nearer to where you expect control opportunities or increase PATCH_HALF_PX / reduce MIN_COMP_PIXELS.",
                        }
                        st.session_state.last_result_map_key += 1
                    else:
                        near_r, near_c = comp["nearest_pixel_rowcol"]
                        cent_r, cent_c = comp["centroid_rowcol"]

                        near_lat, near_lon = rowcol_to_latlon(ds, near_r, near_c)
                        cent_lat, cent_lon = rowcol_to_latlon(ds, cent_r, cent_c)

                        dpx = float(comp["min_distance_pixels"])
                        dist_m, px_m = approx_distance_m(ds, dpx, fallback_px_m=30.0)

                        # Wind W->E => start on west edge of feature, extend east where possible
                        line_latlons, line_len_m, _ = build_300m_polyline_from_component(
                            ds=ds,
                            comp_points_rowcol=comp["component_points_rowcol"],
                            start_west=True,
                            target_len_m=float(TARGET_LINE_M),
                            fallback_px_m=30.0,
                        )

                        if line_latlons is None or len(line_latlons) < 2:
                            st.session_state.last_result = {
                                "type": "warning",
                                "text": "Found a high-PCL feature, but couldn't trace a ~300 m segment (component may be too compact). Try another area or reduce MIN_COMP_PIXELS.",
                            }
                            st.session_state.last_result_map_key += 1
                        else:
                            st.session_state.last_result = {
                                "type": "answer",
                                "question": question.strip(),
                                "thr": thr,
                                "good_threshold_note": f"PCL ≥ {thr:g} treated as a good candidate control opportunity (relative to this raster).",
                                "wind_note": f"Wind {WIND_DIR_TEXT} @ {WIND_MPH:.0f} mph. PCL is not wind-aware; wind is used only to orient the highlighted segment and describe upwind/downwind sides.",
                                "clicked_lat": lat,
                                "clicked_lon": lon,
                                "pcl_at_click": float(pcl_val),
                                "nearest_lat": float(near_lat),
                                "nearest_lon": float(near_lon),
                                "centroid_lat": float(cent_lat),
                                "centroid_lon": float(cent_lon),
                                "distance_m": float(dist_m),
                                "distance_px": float(dpx),
                                "px_m": float(px_m),
                                "component_size_pixels": int(len(comp["component_points_rowcol"])),
                                "line_latlons": line_latlons,
                                "line_len_m": float(line_len_m),
                            }
                            st.session_state.last_result_map_key += 1

                else:
                    # Generic fallback: keep it grounded and consistent with your interpretation
                    st.session_state.last_result = {
                        "type": "generic",
                        "question": question.strip(),
                        "text": (
                            "I can answer closest-high-line questions (PCL≥15) most reliably right now. "
                            "If you want something else, try phrasing with: closest/nearest + continuous/line + high/good PCL."
                        ),
                    }
                    st.session_state.last_result_map_key += 1

# -------------------------
# Persistent Results Section
# -------------------------
st.markdown("---")
st.header("Results")

r = st.session_state.last_result
if r is None:
    st.info("No results yet. Click a point, ask a question, then RUN.")
else:
    if r["type"] == "warning":
        st.warning(r["text"])
    elif r["type"] == "generic":
        st.markdown(f"**Question:** {r.get('question','')}")
        st.info(r["text"])
    else:
        st.markdown(f"**Question:** {r.get('question','(blank)')}")
        st.markdown(f"**Threshold used:** `PCL ≥ {r['thr']:.1f}`")
        st.caption(r["good_threshold_note"])
        st.caption("Interpretation reminder: **low PCL = low probability of control**, **high PCL = high probability of control** (relative to this raster).")
        st.caption(r["wind_note"])

        st.markdown(f"**PCL at clicked point:** `{r['pcl_at_click']:.4f}`")
        st.markdown(f"**Nearest connected high-PCL feature:** size = **{r['component_size_pixels']} px**")
        st.markdown(f"**Nearest point on that feature:** lat={r['nearest_lat']:.6f}, lon={r['nearest_lon']:.6f}")
        st.markdown(f"**Feature centroid:** lat={r['centroid_lat']:.6f}, lon={r['centroid_lon']:.6f}")
        st.markdown(f"**Distance from click to nearest high-PCL pixel:** **{r['distance_m']:.1f} m** (~{r['distance_px']:.2f} px; px≈{r['px_m']:.1f} m)")
        st.markdown(f"**Highlighted line length:** ~**{r['line_len_m']:.1f} m** (target ≈ {TARGET_LINE_M:.0f} m)")

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
            tooltip="Nearest high-PCL pixel (PCL>=15 feature)",
        ).add_to(m2)

        folium.PolyLine(locations=r["line_latlons"], weight=5, opacity=0.95).add_to(m2)

        folium.LayerControl().add_to(m2)

        st.subheader("Map of result (highlighted ~300 m line on nearest PCL≥15 feature)")
        st_folium(m2, height=520, width=None, key=f"pcl_result_map_{st.session_state.last_result_map_key}")

# cleanup temp file (best effort)
try:
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
except Exception:
    pass







# PCL_basic_app.py
# Run:
#   pip install streamlit rasterio pyproj numpy folium streamlit-folium matplotlib
#   streamlit run PCL_basic_app.py

# import base64
# import io
# import os
# import re
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


# def latlon_to_rowcol(ds, lat, lon):
#     transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
#     x, y = transformer.transform(lon, lat)
#     row, col = ds.index(x, y)
#     return row, col


# def rowcol_to_latlon(ds, row, col):
#     x, y = ds.xy(row, col)
#     transformer = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
#     lon, lat = transformer.transform(x, y)
#     return float(lat), float(lon)


# def sample_pcl_at_latlon(ds, lat, lon):
#     row, col = latlon_to_rowcol(ds, lat, lon)
#     if row < 0 or col < 0 or row >= ds.height or col >= ds.width:
#         return None
#     win = Window(col, row, 1, 1)
#     val = ds.read(1, window=win, masked=True)[0, 0]
#     if np.ma.is_masked(val):
#         return None
#     return float(val)


# def read_patch(ds, center_row, center_col, half_px):
#     r0 = max(0, center_row - half_px)
#     r1 = min(ds.height, center_row + half_px + 1)
#     c0 = max(0, center_col - half_px)
#     c1 = min(ds.width, center_col + half_px + 1)
#     win = Window(c0, r0, c1 - c0, r1 - r0)
#     block = ds.read(1, window=win, masked=True).astype("float32")
#     arr = block.data.copy()
#     if hasattr(block, "mask"):
#         arr[block.mask] = np.nan
#     return arr, r0, c0


# def connected_components_8(mask):
#     h, w = mask.shape
#     labels = -np.ones((h, w), dtype=np.int32)
#     comps = []
#     nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
#     cid = 0
#     for r in range(h):
#         for c in range(w):
#             if not mask[r, c] or labels[r, c] != -1:
#                 continue
#             stack = [(r, c)]
#             labels[r, c] = cid
#             pts = [(r, c)]
#             while stack:
#                 rr, cc = stack.pop()
#                 for dr, dc in nbrs:
#                     r2, c2 = rr + dr, cc + dc
#                     if 0 <= r2 < h and 0 <= c2 < w and mask[r2, c2] and labels[r2, c2] == -1:
#                         labels[r2, c2] = cid
#                         stack.append((r2, c2))
#                         pts.append((r2, c2))
#             comps.append(pts)
#             cid += 1
#     return comps


# def approx_distance_m(ds, d_pixels, fallback_px_m=30.0):
#     try:
#         px = float(max(abs(ds.res[0]), abs(ds.res[1])))
#         if np.isfinite(px) and px > 0:
#             return float(d_pixels) * px, px
#     except Exception:
#         pass
#     return float(d_pixels) * float(fallback_px_m), float(fallback_px_m)


# def nearest_component_by_pixel_distance(arr_patch, r0, c0, center_row, center_col, thr, min_comp_pixels=25):
#     if arr_patch is None or arr_patch.size == 0:
#         return None

#     mask = np.isfinite(arr_patch) & (arr_patch >= thr)
#     if not mask.any():
#         return None

#     comps = connected_components_8(mask)
#     pr = center_row - r0
#     pc = center_col - c0

#     best = None
#     best_d2 = None
#     best_nearest = None

#     for pts in comps:
#         if len(pts) < int(min_comp_pixels):
#             continue
#         nearest = min(pts, key=lambda p: (p[0] - pr) ** 2 + (p[1] - pc) ** 2)
#         d2 = (nearest[0] - pr) ** 2 + (nearest[1] - pc) ** 2
#         if best_d2 is None or d2 < best_d2:
#             best_d2 = d2
#             best = pts
#             best_nearest = nearest

#     if best is None:
#         return None

#     comp_full = [(r0 + rr, c0 + cc) for rr, cc in best]
#     near_full = (r0 + best_nearest[0], c0 + best_nearest[1])

#     rr_mean = int(round(np.mean([p[0] for p in best])))
#     cc_mean = int(round(np.mean([p[1] for p in best])))
#     cent_full = (r0 + rr_mean, c0 + cc_mean)

#     return {
#         "component_points_rowcol": comp_full,
#         "nearest_pixel_rowcol": (int(near_full[0]), int(near_full[1])),
#         "centroid_rowcol": (int(cent_full[0]), int(cent_full[1])),
#         "min_distance_pixels": float(np.sqrt(best_d2)) if best_d2 is not None else None,
#     }


# def build_polyline_from_component(ds, comp_points_rowcol, start_west=True, target_len_m=300.0, fallback_px_m=30.0):
#     if not comp_points_rowcol:
#         return None

#     try:
#         px_m = float(max(abs(ds.res[0]), abs(ds.res[1])))
#         if not np.isfinite(px_m) or px_m <= 0:
#             px_m = float(fallback_px_m)
#     except Exception:
#         px_m = float(fallback_px_m)

#     target_steps = max(2, int(round(target_len_m / px_m)))

#     pts = comp_points_rowcol
#     pts_set = set(pts)

#     cols = [c for _, c in pts]
#     start_col = min(cols) if start_west else max(cols)
#     candidates = [p for p in pts if p[1] == start_col]
#     start = candidates[len(candidates) // 2] if candidates else pts[len(pts) // 2]

#     nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
#     chain = [start]
#     visited = {start}
#     desired_sign = 1 if start_west else -1

#     for _ in range(max(1, target_steps - 1)):
#         r, c = chain[-1]
#         neigh = []
#         for dr, dc in nbrs:
#             p = (r + dr, c + dc)
#             if p in pts_set and p not in visited:
#                 neigh.append(p)
#         if not neigh:
#             break

#         prev = chain[-2] if len(chain) >= 2 else None
#         best_p = None
#         best_score = None
#         for p in neigh:
#             rr, cc = p
#             s = desired_sign * (cc - c)
#             if prev is not None:
#                 pr, pc = prev
#                 v1 = (r - pr, c - pc)
#                 v2 = (rr - r, cc - c)
#                 dot = v1[0]*v2[0] + v1[1]*v2[1]
#             else:
#                 dot = 0
#             score = (10.0 * s) + (1.0 * dot)
#             if best_score is None or score > best_score:
#                 best_score = score
#                 best_p = p

#         if best_p is None:
#             break
#         chain.append(best_p)
#         visited.add(best_p)

#     length_m = max(0.0, (len(chain) - 1) * px_m)
#     latlons = [rowcol_to_latlon(ds, r, c) for r, c in chain]
#     return latlons, float(length_m), chain


# def parse_user_controls(question_text):
#     """
#     Parse optional overrides from the user's question.
#     Supported patterns (examples):
#       - "300m", "300 m", "0.3 km", "0.25km"
#       - "pcl>=20", "threshold 18", "pcl 15+"
#       - "wind west to east", "wind W->E", "wind 15 mph", "15mph"
#     Returns dict with possibly-updated:
#       target_len_m, thr, wind_dir_text, wind_mph
#     """
#     q = (question_text or "").lower()

#     # defaults
#     out = {
#         "target_len_m": None,
#         "thr": None,
#         "wind_dir_text": None,
#         "wind_mph": None,
#     }

#     # length
#     m = re.search(r"(\d+(?:\.\d+)?)\s*km\b", q)
#     if m:
#         out["target_len_m"] = float(m.group(1)) * 1000.0
#     else:
#         m = re.search(r"(\d+(?:\.\d+)?)\s*m\b", q)
#         if m:
#             out["target_len_m"] = float(m.group(1))

#     # threshold
#     m = re.search(r"pcl\s*>=\s*(\d+(?:\.\d+)?)", q)
#     if m:
#         out["thr"] = float(m.group(1))
#     else:
#         m = re.search(r"threshold\s*(\d+(?:\.\d+)?)", q)
#         if m:
#             out["thr"] = float(m.group(1))
#         else:
#             m = re.search(r"pcl\s*(\d+(?:\.\d+)?)\s*\+", q)
#             if m:
#                 out["thr"] = float(m.group(1))

#     # wind mph
#     m = re.search(r"(\d+(?:\.\d+)?)\s*mph\b", q)
#     if m:
#         out["wind_mph"] = float(m.group(1))

#     # wind direction (very simple)
#     if ("west" in q and "east" in q) or ("w->e" in q) or ("w→e" in q):
#         out["wind_dir_text"] = "West → East"
#     elif ("east" in q and "west" in q) or ("e->w" in q) or ("e→w" in q):
#         out["wind_dir_text"] = "East → West"
#     elif ("north" in q and "south" in q) or ("n->s" in q) or ("n→s" in q):
#         out["wind_dir_text"] = "North → South"
#     elif ("south" in q and "north" in q) or ("s->n" in q) or ("s→n" in q):
#         out["wind_dir_text"] = "South → North"

#     return out


# def classify_intent(question_text):
#     q = (question_text or "").strip().lower()
#     # If they mention line/continuous/corridor or explicitly a length/threshold, treat as nearest_line.
#     if ("line" in q) or ("continuous" in q) or ("corridor" in q):
#         return "nearest_line"
#     if re.search(r"\b\d+(?:\.\d+)?\s*(m|km)\b", q):
#         return "nearest_line"
#     if ("closest" in q) or ("nearest" in q):
#         return "nearest_line"
#     # blank -> nearest_line default
#     if not q:
#         return "nearest_line"
#     return "generic"


# # -------------------------
# # UI
# # -------------------------
# st.set_page_config(page_title="PCL Basic Map-Click", layout="wide")
# st.title("PCL Basic Map-Click")
# st.caption("Workflow: 1) Upload GeoTIFF → 2) See PCL overlay → 3) Click a point → 4) Ask a question → 5) RUN")

# # Hidden defaults (app knows these even if user doesn't type them)
# DEFAULT_PCL_GOOD_THRESHOLD = 15.0
# DEFAULT_TARGET_LINE_M = 300.0
# DEFAULT_WIND_DIR_TEXT = "West → East"
# DEFAULT_WIND_MPH = 15.0
# MIN_COMP_PIXELS = 25
# PATCH_HALF_PX = 1800

# if "last_result" not in st.session_state:
#     st.session_state.last_result = None
# if "last_result_map_key" not in st.session_state:
#     st.session_state.last_result_map_key = 0
# if "clicked_lat" not in st.session_state:
#     st.session_state.clicked_lat = None
#     st.session_state.clicked_lon = None
#     st.session_state.map_center = None
#     st.session_state.map_zoom = 12

# uploaded = st.file_uploader("1) Drag & drop your PCL GeoTIFF (.tif)", type=["tif", "tiff"])
# if uploaded is None:
#     st.stop()

# with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
#     tmp.write(uploaded.getbuffer())
#     tmp_path = tmp.name

# try:
#     ds = rasterio.open(tmp_path)
# except Exception as e:
#     st.error(f"Could not open raster: {e}")
#     st.stop()

# band_full = ds.read(1, masked=True).astype("float32")
# arr_full = band_full.data.copy()
# if hasattr(band_full, "mask"):
#     arr_full[band_full.mask] = np.nan

# with st.spinner("Rendering PCL overlay..."):
#     arr_ds, _ = downsample_for_display(arr_full, max_dim=1400)
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
#     m = folium.Map(
#         location=st.session_state.map_center,
#         zoom_start=st.session_state.map_zoom,
#         control_scale=True,
#         tiles="CartoDB positron",
#     )

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

#     out = st_folium(
#         m,
#         height=650,
#         width=None,
#         returned_objects=["last_clicked", "center", "zoom"],
#         key="pcl_basic_map",
#     )

#     if out:
#         if out.get("center"):
#             st.session_state.map_center = [float(out["center"]["lat"]), float(out["center"]["lng"])]
#         if out.get("zoom") is not None:
#             st.session_state.map_zoom = int(out["zoom"])
#         if out.get("last_clicked"):
#             st.session_state.clicked_lat = float(out["last_clicked"]["lat"])
#             st.session_state.clicked_lon = float(out["last_clicked"]["lng"])

# with colR:
#     st.subheader("3) Ask → RUN")

#     if st.session_state.clicked_lat is None:
#         st.info("Click on the map to select a location.")
#     else:
#         lat = float(st.session_state.clicked_lat)
#         lon = float(st.session_state.clicked_lon)
#         st.markdown(f"**Selected:** lat={lat:.6f}, lon={lon:.6f}")

#         pcl_val = sample_pcl_at_latlon(ds, lat, lon)
#         if pcl_val is None:
#             st.warning("No valid PCL here (outside raster or NoData). Click somewhere else.")
#         else:
#             st.markdown(f"**PCL at selected point:** `{pcl_val:.4f}`")
#             st.caption("Interpretation: **low PCL = low probability of control**, **high PCL = high probability of control** (relative to this raster).")

#             question = st.text_area(
#                 "Ask a question (you can override settings inline)",
#                 value="",
#                 placeholder=(
#                     "Examples:\n"
#                     "- closest continuous high-PCL line, 500 m long\n"
#                     "- nearest line with pcl>=20, 0.25 km\n"
#                     "- closest continuous line, 400 m, wind east to west 10 mph\n"
#                 ),
#                 height=140,
#             )

#             run = st.button("RUN", type="primary", use_container_width=True)

#             if run:
#                 intent = classify_intent(question)

#                 # apply user overrides if present; otherwise fall back to hidden defaults
#                 overrides = parse_user_controls(question)
#                 thr = overrides["thr"] if overrides["thr"] is not None else float(DEFAULT_PCL_GOOD_THRESHOLD)
#                 target_len_m = overrides["target_len_m"] if overrides["target_len_m"] is not None else float(DEFAULT_TARGET_LINE_M)
#                 wind_dir_text = overrides["wind_dir_text"] if overrides["wind_dir_text"] is not None else str(DEFAULT_WIND_DIR_TEXT)
#                 wind_mph = overrides["wind_mph"] if overrides["wind_mph"] is not None else float(DEFAULT_WIND_MPH)

#                 # decide start side based on wind_dir_text
#                 # "West → East" => start west. "East → West" => start east. else default start west.
#                 start_west = True
#                 if wind_dir_text.strip().lower().startswith("east"):
#                     start_west = False

#                 if intent != "nearest_line":
#                     st.session_state.last_result = {
#                         "type": "generic",
#                         "question": question.strip(),
#                         "text": (
#                             "Right now, the app is best at: nearest/closest continuous line of high PCL.\n"
#                             "Try asking with: closest/nearest + continuous/line and optionally specify length + threshold."
#                         ),
#                     }
#                     st.session_state.last_result_map_key += 1
#                 else:
#                     center_row, center_col = latlon_to_rowcol(ds, lat, lon)
#                     patch, r0, c0 = read_patch(ds, center_row, center_col, PATCH_HALF_PX)

#                     comp = nearest_component_by_pixel_distance(
#                         arr_patch=patch,
#                         r0=r0,
#                         c0=c0,
#                         center_row=center_row,
#                         center_col=center_col,
#                         thr=thr,
#                         min_comp_pixels=MIN_COMP_PIXELS,
#                     )

#                     if comp is None:
#                         st.session_state.last_result = {
#                             "type": "warning",
#                             "text": f"No connected high-PCL (≥{thr:g}) feature found in the search window. Try clicking elsewhere, lower threshold, or increase PATCH_HALF_PX.",
#                         }
#                         st.session_state.last_result_map_key += 1
#                     else:
#                         near_r, near_c = comp["nearest_pixel_rowcol"]
#                         cent_r, cent_c = comp["centroid_rowcol"]

#                         near_lat, near_lon = rowcol_to_latlon(ds, near_r, near_c)
#                         cent_lat, cent_lon = rowcol_to_latlon(ds, cent_r, cent_c)

#                         dpx = float(comp["min_distance_pixels"])
#                         dist_m, px_m = approx_distance_m(ds, dpx, fallback_px_m=30.0)

#                         line_latlons, line_len_m, _ = build_polyline_from_component(
#                             ds=ds,
#                             comp_points_rowcol=comp["component_points_rowcol"],
#                             start_west=start_west,
#                             target_len_m=float(target_len_m),
#                             fallback_px_m=30.0,
#                         )

#                         if line_latlons is None or len(line_latlons) < 2:
#                             st.session_state.last_result = {
#                                 "type": "warning",
#                                 "text": "Found a high-PCL feature, but couldn't trace the requested line length (feature may be too compact). Try a shorter length or a different area.",
#                             }
#                             st.session_state.last_result_map_key += 1
#                         else:
#                             st.session_state.last_result = {
#                                 "type": "answer",
#                                 "question": question.strip(),
#                                 "thr": float(thr),
#                                 "target_len_m": float(target_len_m),
#                                 "wind_dir_text": wind_dir_text,
#                                 "wind_mph": float(wind_mph),
#                                 "clicked_lat": lat,
#                                 "clicked_lon": lon,
#                                 "pcl_at_click": float(pcl_val),
#                                 "nearest_lat": float(near_lat),
#                                 "nearest_lon": float(near_lon),
#                                 "centroid_lat": float(cent_lat),
#                                 "centroid_lon": float(cent_lon),
#                                 "distance_m": float(dist_m),
#                                 "distance_px": float(dpx),
#                                 "px_m": float(px_m),
#                                 "component_size_pixels": int(len(comp["component_points_rowcol"])),
#                                 "line_latlons": line_latlons,
#                                 "line_len_m": float(line_len_m),
#                             }
#                             st.session_state.last_result_map_key += 1

# # -------------------------
# # Persistent Results Section
# # -------------------------
# st.markdown("---")
# st.header("Results")

# r = st.session_state.last_result
# if r is None:
#     st.info("No results yet. Click a point, ask a question, then RUN.")
# else:
#     if r["type"] == "warning":
#         st.warning(r["text"])
#     elif r["type"] == "generic":
#         st.markdown(f"**Question:** {r.get('question','')}")
#         st.info(r["text"])
#     else:
#         st.markdown(f"**Question:** {r.get('question','(blank)')}")
#         st.markdown(f"**Threshold used:** `PCL ≥ {r['thr']:.1f}`")
#         st.caption("Interpretation: **low PCL = low probability of control**, **high PCL = high probability of control** (relative to this raster).")
#         st.caption(
#             f"Wind {r['wind_dir_text']} @ {r['wind_mph']:.0f} mph. PCL is not wind-aware; wind is only used to choose the segment start side (upwind edge when possible)."
#         )

#         st.markdown(f"**PCL at clicked point:** `{r['pcl_at_click']:.4f}`")
#         st.markdown(f"**Nearest connected high-PCL feature:** size = **{r['component_size_pixels']} px**")
#         st.markdown(f"**Nearest point on that feature:** lat={r['nearest_lat']:.6f}, lon={r['nearest_lon']:.6f}")
#         st.markdown(f"**Feature centroid:** lat={r['centroid_lat']:.6f}, lon={r['centroid_lon']:.6f}")
#         st.markdown(f"**Distance from click to nearest high-PCL pixel:** **{r['distance_m']:.1f} m** (~{r['distance_px']:.2f} px; px≈{r['px_m']:.1f} m)")
#         st.markdown(f"**Highlighted line length:** ~**{r['line_len_m']:.1f} m** (requested ≈ {r['target_len_m']:.0f} m)")

#         m2 = folium.Map(
#             location=[r["clicked_lat"], r["clicked_lon"]],
#             zoom_start=st.session_state.map_zoom,
#             control_scale=True,
#             tiles="CartoDB positron",
#         )

#         folium.raster_layers.ImageOverlay(
#             name="PCL",
#             image=img_url,
#             bounds=[[miny, minx], [maxy, maxx]],
#             opacity=0.75,
#             interactive=True,
#             cross_origin=False,
#             zindex=2,
#         ).add_to(m2)

#         folium.CircleMarker(
#             location=[r["clicked_lat"], r["clicked_lon"]],
#             radius=8,
#             weight=2,
#             color="yellow",
#             fill=True,
#             fill_opacity=0.35,
#             tooltip="Clicked point",
#         ).add_to(m2)

#         folium.CircleMarker(
#             location=[r["nearest_lat"], r["nearest_lon"]],
#             radius=8,
#             weight=2,
#             color="red",
#             fill=True,
#             fill_opacity=0.35,
#             tooltip=f"Nearest high-PCL pixel (PCL≥{r['thr']:.1f})",
#         ).add_to(m2)

#         folium.PolyLine(locations=r["line_latlons"], weight=5, opacity=0.95).add_to(m2)

#         folium.LayerControl().add_to(m2)

#         st.subheader("Map of result (highlighted line on nearest high-PCL feature)")
#         st_folium(m2, height=520, width=None, key=f"pcl_result_map_{st.session_state.last_result_map_key}")

# # cleanup temp file (best effort)
# try:
#     if os.path.exists(tmp_path):
#         os.remove(tmp_path)
# except Exception:
#     pass
