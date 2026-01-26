# # AVIATION.py
# import os
# import io
# import time
# import base64
# from io import BytesIO

# import requests
# import pandas as pd
# import matplotlib.pyplot as plt

# import streamlit as st

# # ---- OPTIONAL GIS STACK (so the app runs even if shapely/geopandas aren't installed)
# GIS_OK = True
# GIS_ERR = ""
# try:
#     import geopandas as gpd
#     from shapely.geometry import Point
#     import contextily as cx
# except Exception as e:
#     GIS_OK = False
#     GIS_ERR = str(e)
#     gpd = None
#     Point = None
#     cx = None

# try:
#     from openai import OpenAI
# except Exception:
#     OpenAI = None


# # =========================
# # STREAMLIT PAGE
# # =========================
# st.set_page_config(page_title="OpenSky USFS/CALFIRE Live Snapshot", layout="wide")

# st.markdown(
#     """
#     <style>
#     .big-button button {
#         background: linear-gradient(90deg, #1f2937, #111827);
#         color: white !important;
#         border-radius: 14px;
#         padding: 0.75rem 1.25rem;
#         border: 1px solid rgba(255,255,255,0.15);
#         font-weight: 700;
#         letter-spacing: 0.02em;
#         box-shadow: 0 10px 30px rgba(0,0,0,0.20);
#         transition: transform 0.06s ease-in-out;
#     }
#     .big-button button:hover {
#         transform: translateY(-1px);
#         border: 1px solid rgba(255,255,255,0.30);
#     }
#     .muted {
#         color: rgba(0,0,0,0.65);
#         font-size: 0.95rem;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# st.title("OpenSky Live Snapshot — USFS / CAL FIRE")
# st.markdown(
#     '<div class="muted">Queries current OpenSky “states” in a Western US bounding box, matches to your USFS/CALFIRE masterlist, then answers a question (optionally using OpenAI).</div>',
#     unsafe_allow_html=True,
# )

# if not GIS_OK:
#     st.warning(
#         "Map rendering dependencies are missing in this environment (geopandas/shapely/contextily). "
#         "The app will still run and show tables + answers. "
#         f"Import error: {GIS_ERR}"
#     )

# # =========================
# # CONFIG
# # =========================
# BBOX = (31.0, 49.5, -125.0, -102.0)  # (min_lat, max_lat, min_lon, max_lon)

# STATES_URL = "https://opensky-network.org/api/states/all"
# TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
# UA = "opensky-live-usfs-calfire/1.0 (+research)"

# # Basemap (only used if GIS_OK)
# BASEMAP = None
# BASEMAP_ZOOM = 6
# if GIS_OK:
#     BASEMAP = cx.providers.Esri.WorldTopoMap

# OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
# OPENAI_API_KEY = st.secrets.get("openai", {}).get("api_key") or os.getenv("OPENAI_API_KEY")


# # =========================
# # MASTERLIST FROM SECRETS
# # =========================
# @st.cache_data(show_spinner=False)
# def load_master_from_text(master_csv: str) -> pd.DataFrame:
#     return pd.read_csv(io.StringIO(master_csv))

# # usage
# master_csv = st.secrets["masterlist"]["csv"]
# master_raw = load_master_from_text(master_csv)

# def normalize_icao24(x) -> str:
#     if pd.isna(x):
#         return ""
#     s = str(x).strip().lower()
#     return s.replace("0x", "").replace(" ", "")

# def load_masterlist_df(df: pd.DataFrame) -> pd.DataFrame:
#     expected = {"TailNumber", "ADSB", "Agency"}
#     missing = expected - set(df.columns)
#     if missing:
#         raise RuntimeError(f"Masterlist missing columns: {missing}. Found columns: {list(df.columns)}")

#     df = df.copy()
#     df["icao24"] = df["ADSB"].apply(normalize_icao24)
#     df = df[df["icao24"] != ""].drop_duplicates(subset=["icao24"]).reset_index(drop=True)

#     df["Agency"] = df["Agency"].astype(str).str.strip().str.upper()
#     if "Type" in df.columns:
#         df["Type"] = df["Type"].astype(str).str.strip()
#     else:
#         df["Type"] = ""
#     return df


# # =========================
# # AUTH (OpenSky)
# # =========================
# def get_access_token(client_id: str, client_secret: str, timeout=30) -> str:
#     r = requests.post(
#         TOKEN_URL,
#         headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": UA},
#         data={
#             "grant_type": "client_credentials",
#             "client_id": client_id,
#             "client_secret": client_secret,
#         },
#         timeout=timeout,
#     )
#     if r.status_code == 403:
#         raise RuntimeError("403 from token endpoint.\n" + r.text[:800])
#     r.raise_for_status()
#     tok = r.json()
#     access_token = tok.get("access_token")
#     if not access_token:
#         raise RuntimeError(f"Token response missing access_token: {tok}")
#     return access_token


# # =========================
# # FETCH
# # =========================
# def fetch_states(token: str, bbox=None, timeout=30) -> dict:
#     params = {}
#     if bbox is not None:
#         min_lat, max_lat, min_lon, max_lon = bbox
#         params.update({"lamin": min_lat, "lamax": max_lat, "lomin": min_lon, "lomax": max_lon})
#     r = requests.get(
#         STATES_URL,
#         headers={"Authorization": f"Bearer {token}", "User-Agent": UA},
#         params=params,
#         timeout=timeout,
#     )
#     r.raise_for_status()
#     return r.json()


# # =========================
# # DATA WRANGLING
# # =========================
# def states_to_df(data: dict) -> pd.DataFrame:
#     cols = [
#         "icao24", "callsign", "origin_country", "time_position", "last_contact",
#         "longitude", "latitude", "baro_altitude", "on_ground", "velocity",
#         "true_track", "vertical_rate", "sensors", "geo_altitude", "squawk",
#         "spi", "position_source",
#     ]
#     states = data.get("states") or []
#     df = pd.DataFrame(states, columns=cols)

#     df["icao24"] = df["icao24"].apply(normalize_icao24)
#     df["callsign"] = df["callsign"].astype(str).str.strip().replace({"None": ""})

#     for c in ["longitude", "latitude", "baro_altitude", "geo_altitude", "velocity", "true_track"]:
#         df[c] = pd.to_numeric(df[c], errors="coerce")

#     df = df[df["longitude"].between(-180, 180) & df["latitude"].between(-90, 90)].copy()
#     df["on_ground"] = df["on_ground"].astype("boolean")
#     return df

# def join_states_master(states_df: pd.DataFrame, master_df: pd.DataFrame) -> pd.DataFrame:
#     out = states_df.merge(master_df, how="inner", on="icao24")
#     out["alt_m"] = out["geo_altitude"].fillna(out["baro_altitude"])
#     return out


# # =========================
# # MAP (optional)
# # =========================
# def bbox_to_webmercator(bbox, pad_frac=0.06):
#     min_lat, max_lat, min_lon, max_lon = bbox
#     lat_pad = (max_lat - min_lat) * pad_frac
#     lon_pad = (max_lon - min_lon) * pad_frac
#     min_lat -= lat_pad
#     max_lat += lat_pad
#     min_lon -= lon_pad
#     max_lon += lon_pad

#     p1 = gpd.GeoSeries([Point(min_lon, min_lat)], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
#     p2 = gpd.GeoSeries([Point(max_lon, max_lat)], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
#     return p1.x, p2.x, p1.y, p2.y

# def to_gdf_webmercator(df: pd.DataFrame) -> "gpd.GeoDataFrame":
#     d = df.dropna(subset=["longitude", "latitude"]).copy()
#     gdf2 = gpd.GeoDataFrame(
#         d,
#         geometry=[Point(xy) for xy in zip(d["longitude"].astype(float), d["latitude"].astype(float))],
#         crs="EPSG:4326",
#     )
#     return gdf2.to_crs(epsg=3857)

# def plot_snapshot_basemap(df: pd.DataFrame, title: str, bbox=None):
#     fig, ax = plt.subplots(figsize=(12.5, 8.0))
#     ax.set_facecolor("white")

#     if bbox is not None:
#         minx, maxx, miny, maxy = bbox_to_webmercator(bbox, pad_frac=0.06)
#         ax.set_xlim(minx, maxx)
#         ax.set_ylim(miny, maxy)

#     cx.add_basemap(ax, source=BASEMAP, zoom=BASEMAP_ZOOM, attribution_size=7)

#     if df is not None and len(df) > 0:
#         gdf2 = to_gdf_webmercator(df)
#         for agency, g in gdf2.groupby("Agency"):
#             speed = pd.to_numeric(g["velocity"], errors="coerce").fillna(0.0)
#             sizes = (speed.clip(0, 220) / 220.0) * 110.0 + 28.0
#             ax.scatter(
#                 g.geometry.x, g.geometry.y,
#                 s=sizes,
#                 label=f"{agency} (n={len(g)})",
#                 alpha=0.9,
#                 linewidths=0.8,
#                 edgecolors="white",
#                 zorder=10,
#             )
#         ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95)
#     else:
#         ax.text(
#             0.02, 0.02,
#             "No matching USFS/CALFIRE aircraft in this snapshot.",
#             transform=ax.transAxes,
#             fontsize=11,
#             bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
#             zorder=20,
#         )

#     ax.set_title(title)
#     ax.set_axis_off()
#     plt.tight_layout()
#     return fig

# def fig_to_data_url(fig) -> str:
#     buf = BytesIO()
#     fig.savefig(buf, format="PNG", dpi=160, bbox_inches="tight")
#     plt.close(fig)
#     b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
#     return "data:image/png;base64," + b64


# # =========================
# # SUMMARY + QUESTION ANSWERING
# # =========================
# def classify_airframe_is_heli(type_str: str, callsign: str = "") -> bool:
#     t = (type_str or "").strip().lower()
#     c = (callsign or "").strip().lower()
#     if any(k in t for k in ["heli", "helic", "rotor", "rotary"]):
#         return True
#     if t in {"h", "hel", "heli"}:
#         return True
#     if t.startswith("h ") or t.endswith(" h") or t.startswith("h-") or t.startswith("h_"):
#         return True
#     if t.replace("-", " ").replace("_", " ").strip() in {"type 1", "type1", "type 2", "type2"}:
#         return True
#     return False

# def summarize_snapshot(matched: pd.DataFrame) -> dict:
#     if matched is None or matched.empty:
#         return {
#             "matched_total": 0,
#             "airborne_total": 0,
#             "agencies_airborne": {},
#             "helicopters_airborne_total": 0,
#             "helicopters_by_agency": {},
#             "note": "No matches in snapshot.",
#         }

#     m = matched.copy()
#     m["Agency"] = m["Agency"].astype(str).str.strip().str.upper()
#     m["is_airborne"] = (~m["on_ground"].fillna(False)).astype(bool)

#     airborne = m[m["is_airborne"]].copy()
#     agencies_airborne = airborne["Agency"].value_counts().to_dict() if not airborne.empty else {}

#     airborne["is_heli"] = [
#         classify_airframe_is_heli(t, cs) for t, cs in zip(airborne.get("Type", ""), airborne.get("callsign", ""))
#     ]
#     helis = airborne[airborne["is_heli"]].copy()

#     return {
#         "matched_total": int(len(m)),
#         "airborne_total": int(len(airborne)),
#         "agencies_airborne": agencies_airborne,
#         "helicopters_airborne_total": int(len(helis)),
#         "helicopters_by_agency": helis["Agency"].value_counts().to_dict() if not helis.empty else {},
#         "sample_airborne": airborne[
#             ["Agency", "TailNumber", "icao24", "callsign", "Type", "latitude", "longitude", "alt_m", "velocity", "on_ground"]
#         ].sort_values(["Agency", "TailNumber"]).head(25).to_dict(orient="records") if not airborne.empty else [],
#     }

# OPENAI_PRICING_PER_1M = {
#     "gpt-4o": {"in": 2.50, "out": 10.00},
#     "gpt-4o-2024-08-06": {"in": 2.50, "out": 10.00},
#     "chatgpt-4o-latest": {"in": 5.00, "out": 15.00},
# }

# def estimate_openai_cost_usd(model: str, usage_obj) -> dict:
#     if usage_obj is None:
#         return {"model": model, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated_usd": 0.0}

#     prompt_tokens = getattr(usage_obj, "prompt_tokens", None)
#     completion_tokens = getattr(usage_obj, "completion_tokens", None)
#     total_tokens = getattr(usage_obj, "total_tokens", None)

#     if prompt_tokens is None and isinstance(usage_obj, dict):
#         prompt_tokens = usage_obj.get("prompt_tokens", 0)
#         completion_tokens = usage_obj.get("completion_tokens", 0)
#         total_tokens = usage_obj.get("total_tokens", (prompt_tokens or 0) + (completion_tokens or 0))

#     prompt_tokens = int(prompt_tokens or 0)
#     completion_tokens = int(completion_tokens or 0)
#     total_tokens = int(total_tokens or (prompt_tokens + completion_tokens))

#     rates = OPENAI_PRICING_PER_1M.get(model)
#     if rates is None:
#         return {"model": model, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens, "estimated_usd": None}

#     est = (prompt_tokens / 1_000_000.0) * rates["in"] + (completion_tokens / 1_000_000.0) * rates["out"]
#     return {"model": model, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens, "estimated_usd": float(est)}

# def ask_snapshot_question(snapshot_summary: dict, question: str, map_data_url: str = None) -> tuple[str, dict]:
#     # HARD REQUIREMENT
#     if not OPENAI_API_KEY:
#         raise RuntimeError(
#             "OPENAI_API_KEY is not set. "
#             "This app is configured to ALWAYS use ChatGPT. "
#             "Add OPENAI_API_KEY to Streamlit secrets or environment variables."
#         )

#     if OpenAI is None:
#         raise RuntimeError(
#             "openai package not installed. "
#             "Add `openai` to requirements.txt."
#         )

#     client = OpenAI(api_key=OPENAI_API_KEY)

#     sys_msg = (
#         "You are an operational aviation analyst for wildfire response. "
#         "You MUST answer the user's question even if the snapshot shows zero aircraft. "
#         #"If zero aircraft are airborne, explain why that can occur operationally. "
#         "Do not refuse to answer."
#     )

#     compact = {
#         "matched_total": snapshot_summary.get("matched_total", 0),
#         "airborne_total": snapshot_summary.get("airborne_total", 0),
#         "agencies_airborne": snapshot_summary.get("agencies_airborne", {}),
#         "helicopters_airborne_total": snapshot_summary.get("helicopters_airborne_total", 0),
#         "helicopters_by_agency": snapshot_summary.get("helicopters_by_agency", {}),
#         "sample_airborne": snapshot_summary.get("sample_airborne", [])[:25],
#     }

#     user_content = [
#         {
#             "type": "text",
#             "text": (
#                 "Here is a live aircraft snapshot summary:\n"
#                 f"{compact}\n\n"
#                 f"Question: {question}\n\n"
#                 "If counts are zero, just answer based on their question"
#             ),
#         }
#     ]

#     if map_data_url is not None:
#         user_content.append(
#             {"type": "image_url", "image_url": {"url": map_data_url}}
#         )

#     resp = client.chat.completions.create(
#         model=OPENAI_MODEL,
#         messages=[
#             {"role": "system", "content": sys_msg},
#             {"role": "user", "content": user_content},
#         ],
#         temperature=0,
#     )

#     usage = getattr(resp, "usage", None)
#     cost = estimate_openai_cost_usd(OPENAI_MODEL, usage)

#     return resp.choices[0].message.content.strip(), cost


# # =========================
# # UI CONTROLS
# # =========================
# with st.sidebar:
#     st.header("Settings")
#     st.caption("Put OpenSky OAuth and masterlist CSV in Streamlit secrets.")

#     bbox_on = st.checkbox("Use Western US bbox", value=True)
#     bbox = BBOX if bbox_on else None

#     agencies_filter = st.multiselect("Agencies to display", ["USFS", "CALFIRE"], default=["USFS", "CALFIRE"])

#     st.markdown("---")
#     st.subheader("OpenSky credentials (from secrets)")
#     st.caption("Expected: st.secrets['opensky']['client_id'] and ['client_secret']")

# question = st.text_input(
#     "Ask a question about the CURRENT aircraft snapshot:",
#     value="How many helicopters are in the air and what agencies are they from?",
# )

# col_a, col_b = st.columns([0.22, 0.78], vertical_alignment="bottom")
# with col_a:
#     run = st.container()
#     run.markdown('<div class="big-button">', unsafe_allow_html=True)
#     go = st.button("Run snapshot", use_container_width=True)
#     run.markdown("</div>", unsafe_allow_html=True)

# with col_b:
#     st.caption("HELLO WORLD")


# # =========================
# # RUN
# # =========================
# if go:
#     t0 = time.time()

#     try:
#         client_id = st.secrets["opensky"]["client_id"]
#         client_secret = st.secrets["opensky"]["client_secret"]
#     except Exception:
#         st.error("Missing OpenSky credentials in secrets. Add [opensky] client_id and client_secret.")
#         st.stop()

#     # with st.spinner("Loading masterlist..."):
#     #     master_raw = load_master_from_secrets()
#     #     master = load_masterlist_df(master_raw)

#     with st.spinner("Loading masterlist..."):
#         master_csv = st.secrets["masterlist"]["csv"]
#         master_raw = load_master_from_text(master_csv)
#         master = load_masterlist_df(master_raw)
    
#     with st.spinner("Fetching current OpenSky states..."):
#         token = get_access_token(client_id, client_secret)
#         data = fetch_states(token, bbox=bbox)

#     states = states_to_df(data)
#     matched = join_states_master(states, master)

#     if agencies_filter:
#         matched = matched[matched["Agency"].astype(str).str.upper().isin([a.upper() for a in agencies_filter])].copy()

#     api_time = data.get("time")
#     elapsed = time.time() - t0

#     st.subheader("Snapshot results")

#     k1, k2, k3, k4 = st.columns(4)
#     k1.metric("Masterlist aircraft", f"{len(master):,}")
#     k2.metric("Matched in snapshot", f"{len(matched):,}")
#     k3.metric("API time (epoch)", str(api_time))
#     k4.metric("Fetch + render", f"{elapsed:.2f}s")

#     if matched.empty:
#         st.warning("No matching USFS/CALFIRE aircraft found in this snapshot.")
#     else:
#         st.markdown("**Counts by agency**")
#         st.dataframe(matched["Agency"].value_counts(dropna=False).rename_axis("Agency").reset_index(name="Count"), use_container_width=True)

#         st.markdown("**Sample matched rows**")
#         show_cols = ["Agency", "TailNumber", "icao24", "callsign", "Type", "latitude", "longitude", "alt_m", "velocity", "on_ground"]
#         st.dataframe(
#             matched[show_cols].sort_values(["Agency", "TailNumber"]).head(100),
#             use_container_width=True,
#             hide_index=True,
#         )

#     if GIS_OK:
#         st.markdown("**Map**")
#         title = f"OpenSky CURRENT states | {'WESTERN US bbox' if bbox else 'GLOBAL'} | matched={len(matched)}"
#         fig = plot_snapshot_basemap(matched, title=title, bbox=bbox or BBOX)
#         st.pyplot(fig, use_container_width=True)

#         snapshot_summary = summarize_snapshot(matched)
#         map_url = fig_to_data_url(fig)
#     else:
#         snapshot_summary = summarize_snapshot(matched)
#         map_url = None

#     st.markdown("**Answer**")
#     with st.spinner("Answering question..."):
#         answer, cost = ask_snapshot_question(snapshot_summary, question=question, map_data_url=map_url)

#     st.write(answer)

#     if cost.get("model"):
#         st.caption(
#             f"OpenAI usage: model={cost.get('model')} | prompt={cost.get('prompt_tokens')} | completion={cost.get('completion_tokens')} | total={cost.get('total_tokens')} | est=${(cost.get('estimated_usd') or 0):.6f}"
#         )






















# # AVIATION.py
# import os
# import io
# import time
# import base64
# from io import BytesIO

# import requests
# import pandas as pd
# import matplotlib.pyplot as plt

# import streamlit as st
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry

# # ---- OPTIONAL GIS STACK (so the app runs even if shapely/geopandas aren't installed)
# GIS_OK = True
# GIS_ERR = ""
# try:
#     import geopandas as gpd
#     from shapely.geometry import Point
#     import contextily as cx
# except Exception as e:
#     GIS_OK = False
#     GIS_ERR = str(e)
#     gpd = None
#     Point = None
#     cx = None

# try:
#     from openai import OpenAI
# except Exception:
#     OpenAI = None


# # =========================
# # STREAMLIT PAGE
# # =========================
# st.set_page_config(page_title="OpenSky USFS/CALFIRE Live Snapshot", layout="wide")

# # Hide the sidebar entirely + Streamlit chrome (menu/footer/header)
# st.markdown(
#     """
#     <style>
#       section[data-testid="stSidebar"] {display: none;}
#       div[data-testid="stSidebarNav"] {display: none;}
#       #MainMenu {visibility: hidden;}
#       footer {visibility: hidden;}
#       header {visibility: hidden;}
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# st.markdown(
#     """
#     <style>
#     .big-button button {
#         background: linear-gradient(90deg, #1f2937, #111827);
#         color: white !important;
#         border-radius: 14px;
#         padding: 0.75rem 1.25rem;
#         border: 1px solid rgba(255,255,255,0.15);
#         font-weight: 700;
#         letter-spacing: 0.02em;
#         box-shadow: 0 10px 30px rgba(0,0,0,0.20);
#         transition: transform 0.06s ease-in-out;
#     }
#     .big-button button:hover {
#         transform: translateY(-1px);
#         border: 1px solid rgba(255,255,255,0.30);
#     }
#     .muted {
#         color: rgba(0,0,0,0.65);
#         font-size: 0.95rem;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# st.title("OpenSky Live Snapshot — USFS / CAL FIRE")
# st.markdown(
#     '<div class="muted">Queries current OpenSky “states” in a Western US bounding box, matches to your USFS/CALFIRE masterlist, then answers a question (optionally using OpenAI).</div>',
#     unsafe_allow_html=True,
# )

# if not GIS_OK:
#     st.warning(
#         "Map rendering dependencies are missing in this environment (geopandas/shapely/contextily). "
#         "The app will still run and show tables + answers. "
#         f"Import error: {GIS_ERR}"
#     )

# # =========================
# # CONFIG
# # =========================
# BBOX = (31.0, 49.5, -125.0, -102.0)  # (min_lat, max_lat, min_lon, max_lon)

# STATES_URL = "https://opensky-network.org/api/states/all"
# TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
# UA = "opensky-live-usfs-calfire/1.0 (+research)"

# # Basemap (only used if GIS_OK)
# BASEMAP = None
# BASEMAP_ZOOM = 6
# if GIS_OK:
#     BASEMAP = cx.providers.Esri.WorldTopoMap

# OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
# OPENAI_API_KEY = st.secrets.get("openai", {}).get("api_key") or os.getenv("OPENAI_API_KEY")

# # Hardcoded (no sidebar)
# bbox = BBOX
# agencies_filter = ["USFS", "CALFIRE"]


# # =========================
# # MASTERLIST FROM SECRETS
# # =========================
# @st.cache_data(show_spinner=False)
# def load_master_from_text(master_csv: str) -> pd.DataFrame:
#     return pd.read_csv(io.StringIO(master_csv))

# def normalize_icao24(x) -> str:
#     if pd.isna(x):
#         return ""
#     s = str(x).strip().lower()
#     return s.replace("0x", "").replace(" ", "")

# def load_masterlist_df(df: pd.DataFrame) -> pd.DataFrame:
#     expected = {"TailNumber", "ADSB", "Agency"}
#     missing = expected - set(df.columns)
#     if missing:
#         raise RuntimeError(f"Masterlist missing columns: {missing}. Found columns: {list(df.columns)}")

#     df = df.copy()
#     df["icao24"] = df["ADSB"].apply(normalize_icao24)
#     df = df[df["icao24"] != ""].drop_duplicates(subset=["icao24"]).reset_index(drop=True)

#     df["Agency"] = df["Agency"].astype(str).str.strip().str.upper()
#     if "Type" in df.columns:
#         df["Type"] = df["Type"].astype(str).str.strip()
#     else:
#         df["Type"] = ""
#     return df


# # =========================
# # AUTH (OpenSky)
# # =========================
# # def get_access_token(client_id: str, client_secret: str, timeout=30) -> str:
# #     r = requests.post(
# #         TOKEN_URL,
# #         headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": UA},
# #         data={
# #             "grant_type": "client_credentials",
# #             "client_id": client_id,
# #             "client_secret": client_secret,
# #         },
# #         timeout=timeout,
# #     )
# #     if r.status_code == 403:
# #         raise RuntimeError("403 from token endpoint.\n" + r.text[:800])
# #     r.raise_for_status()
# #     tok = r.json()
# #     access_token = tok.get("access_token")
# #     if not access_token:
# #         raise RuntimeError(f"Token response missing access_token: {tok}")
# #     return access_token


# # =========================
# # FETCH
# # =========================
# def fetch_states(token: str, bbox=None, timeout=30) -> dict:
#     params = {}
#     if bbox is not None:
#         min_lat, max_lat, min_lon, max_lon = bbox
#         params.update({"lamin": min_lat, "lamax": max_lat, "lomin": min_lon, "lomax": max_lon})
#     r = requests.get(
#         STATES_URL,
#         headers={"Authorization": f"Bearer {token}", "User-Agent": UA},
#         params=params,
#         timeout=timeout,
#     )
#     r.raise_for_status()
#     return r.json()


# # =========================
# # DATA WRANGLING
# # =========================
# def states_to_df(data: dict) -> pd.DataFrame:
#     cols = [
#         "icao24", "callsign", "origin_country", "time_position", "last_contact",
#         "longitude", "latitude", "baro_altitude", "on_ground", "velocity",
#         "true_track", "vertical_rate", "sensors", "geo_altitude", "squawk",
#         "spi", "position_source",
#     ]
#     states = data.get("states") or []
#     df = pd.DataFrame(states, columns=cols)

#     df["icao24"] = df["icao24"].apply(normalize_icao24)
#     df["callsign"] = df["callsign"].astype(str).str.strip().replace({"None": ""})

#     for c in ["longitude", "latitude", "baro_altitude", "geo_altitude", "velocity", "true_track"]:
#         df[c] = pd.to_numeric(df[c], errors="coerce")

#     df = df[df["longitude"].between(-180, 180) & df["latitude"].between(-90, 90)].copy()
#     df["on_ground"] = df["on_ground"].astype("boolean")
#     return df

# def join_states_master(states_df: pd.DataFrame, master_df: pd.DataFrame) -> pd.DataFrame:
#     out = states_df.merge(master_df, how="inner", on="icao24")
#     out["alt_m"] = out["geo_altitude"].fillna(out["baro_altitude"])
#     return out


# # =========================
# # MAP (optional)
# # =========================
# def bbox_to_webmercator(bbox, pad_frac=0.06):
#     min_lat, max_lat, min_lon, max_lon = bbox
#     lat_pad = (max_lat - min_lat) * pad_frac
#     lon_pad = (max_lon - min_lon) * pad_frac
#     min_lat -= lat_pad
#     max_lat += lat_pad
#     min_lon -= lon_pad
#     max_lon += lon_pad

#     p1 = gpd.GeoSeries([Point(min_lon, min_lat)], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
#     p2 = gpd.GeoSeries([Point(max_lon, max_lat)], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
#     return p1.x, p2.x, p1.y, p2.y

# def to_gdf_webmercator(df: pd.DataFrame) -> "gpd.GeoDataFrame":
#     d = df.dropna(subset=["longitude", "latitude"]).copy()
#     gdf2 = gpd.GeoDataFrame(
#         d,
#         geometry=[Point(xy) for xy in zip(d["longitude"].astype(float), d["latitude"].astype(float))],
#         crs="EPSG:4326",
#     )
#     return gdf2.to_crs(epsg=3857)

# def plot_snapshot_basemap(df: pd.DataFrame, title: str, bbox=None):
#     fig, ax = plt.subplots(figsize=(12.5, 8.0))
#     ax.set_facecolor("white")

#     if bbox is not None:
#         minx, maxx, miny, maxy = bbox_to_webmercator(bbox, pad_frac=0.06)
#         ax.set_xlim(minx, maxx)
#         ax.set_ylim(miny, maxy)

#     cx.add_basemap(ax, source=BASEMAP, zoom=BASEMAP_ZOOM, attribution_size=7)

#     # Agency colors: USFS green, CALFIRE blue
#     agency_color = {
#         "USFS": "green",
#         "CALFIRE": "blue",
#     }

#     if df is not None and len(df) > 0:
#         gdf2 = to_gdf_webmercator(df)
#         for agency, g in gdf2.groupby("Agency"):
#             speed = pd.to_numeric(g["velocity"], errors="coerce").fillna(0.0)
#             sizes = (speed.clip(0, 220) / 220.0) * 110.0 + 28.0
#             ax.scatter(
#                 g.geometry.x, g.geometry.y,
#                 s=sizes,
#                 label=f"{agency} (n={len(g)})",
#                 alpha=0.9,
#                 linewidths=0.8,
#                 edgecolors="white",
#                 c=agency_color.get(str(agency).upper(), "gray"),
#                 zorder=10,
#             )
#         ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95)
#     else:
#         ax.text(
#             0.02, 0.02,
#             "No matching USFS/CALFIRE aircraft in this snapshot.",
#             transform=ax.transAxes,
#             fontsize=11,
#             bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
#             zorder=20,
#         )

#     ax.set_title(title)
#     ax.set_axis_off()
#     plt.tight_layout()
#     return fig

# def fig_to_data_url(fig) -> str:
#     buf = BytesIO()
#     fig.savefig(buf, format="PNG", dpi=160, bbox_inches="tight")
#     plt.close(fig)
#     b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
#     return "data:image/png;base64," + b64


# # =========================
# # SUMMARY + QUESTION ANSWERING
# # =========================
# def classify_airframe_is_heli(type_str: str, callsign: str = "") -> bool:
#     t = (type_str or "").strip().lower()
#     c = (callsign or "").strip().lower()
#     if any(k in t for k in ["heli", "helic", "rotor", "rotary"]):
#         return True
#     if t in {"h", "hel", "heli"}:
#         return True
#     if t.startswith("h ") or t.endswith(" h") or t.startswith("h-") or t.startswith("h_"):
#         return True
#     if t.replace("-", " ").replace("_", " ").strip() in {"type 1", "type1", "type 2", "type2"}:
#         return True
#     return False

# def summarize_snapshot(matched: pd.DataFrame) -> dict:
#     if matched is None or matched.empty:
#         return {
#             "matched_total": 0,
#             "airborne_total": 0,
#             "agencies_airborne": {},
#             "helicopters_airborne_total": 0,
#             "helicopters_by_agency": {},
#             "note": "No matches in snapshot.",
#         }

#     m = matched.copy()
#     m["Agency"] = m["Agency"].astype(str).str.strip().str.upper()
#     m["is_airborne"] = (~m["on_ground"].fillna(False)).astype(bool)

#     airborne = m[m["is_airborne"]].copy()
#     agencies_airborne = airborne["Agency"].value_counts().to_dict() if not airborne.empty else {}

#     airborne["is_heli"] = [
#         classify_airframe_is_heli(t, cs) for t, cs in zip(airborne.get("Type", ""), airborne.get("callsign", ""))
#     ]
#     helis = airborne[airborne["is_heli"]].copy()

#     return {
#         "matched_total": int(len(m)),
#         "airborne_total": int(len(airborne)),
#         "agencies_airborne": agencies_airborne,
#         "helicopters_airborne_total": int(len(helis)),
#         "helicopters_by_agency": helis["Agency"].value_counts().to_dict() if not helis.empty else {},
#         "sample_airborne": airborne[
#             ["Agency", "TailNumber", "icao24", "callsign", "Type", "latitude", "longitude", "alt_m", "velocity", "on_ground"]
#         ].sort_values(["Agency", "TailNumber"]).head(25).to_dict(orient="records") if not airborne.empty else [],
#     }

# OPENAI_PRICING_PER_1M = {
#     "gpt-4o": {"in": 2.50, "out": 10.00},
#     "gpt-4o-2024-08-06": {"in": 2.50, "out": 10.00},
#     "chatgpt-4o-latest": {"in": 5.00, "out": 15.00},
# }

# def estimate_openai_cost_usd(model: str, usage_obj) -> dict:
#     if usage_obj is None:
#         return {"model": model, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated_usd": 0.0}

#     prompt_tokens = getattr(usage_obj, "prompt_tokens", None)
#     completion_tokens = getattr(usage_obj, "completion_tokens", None)
#     total_tokens = getattr(usage_obj, "total_tokens", None)

#     if prompt_tokens is None and isinstance(usage_obj, dict):
#         prompt_tokens = usage_obj.get("prompt_tokens", 0)
#         completion_tokens = usage_obj.get("completion_tokens", 0)
#         total_tokens = usage_obj.get("total_tokens", (prompt_tokens or 0) + (completion_tokens or 0))

#     prompt_tokens = int(prompt_tokens or 0)
#     completion_tokens = int(completion_tokens or 0)
#     total_tokens = int(total_tokens or (prompt_tokens + completion_tokens))

#     rates = OPENAI_PRICING_PER_1M.get(model)
#     if rates is None:
#         return {"model": model, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens, "estimated_usd": None}

#     est = (prompt_tokens / 1_000_000.0) * rates["in"] + (completion_tokens / 1_000_000.0) * rates["out"]
#     return {"model": model, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens, "estimated_usd": float(est)}

# def ask_snapshot_question(snapshot_summary: dict, question: str, map_data_url: str = None) -> tuple[str, dict]:
#     if not OPENAI_API_KEY:
#         raise RuntimeError(
#             "OPENAI_API_KEY is not set. "
#             "Add OPENAI_API_KEY to Streamlit secrets or environment variables."
#         )

#     if OpenAI is None:
#         raise RuntimeError("openai package not installed. Add `openai` to requirements.txt.")

#     client = OpenAI(api_key=OPENAI_API_KEY)

#     sys_msg = (
#         "You are an operational aviation analyst for wildfire response. "
#         "You MUST answer the user's question even if the snapshot shows zero aircraft. "
#         "Do not refuse to answer."
#     )

#     compact = {
#         "matched_total": snapshot_summary.get("matched_total", 0),
#         "airborne_total": snapshot_summary.get("airborne_total", 0),
#         "agencies_airborne": snapshot_summary.get("agencies_airborne", {}),
#         "helicopters_airborne_total": snapshot_summary.get("helicopters_airborne_total", 0),
#         "helicopters_by_agency": snapshot_summary.get("helicopters_by_agency", {}),
#         "sample_airborne": snapshot_summary.get("sample_airborne", [])[:25],
#     }

#     user_content = [
#         {
#             "type": "text",
#             "text": (
#                 "Here is a live aircraft snapshot summary:\n"
#                 f"{compact}\n\n"
#                 f"Question: {question}\n\n"
#                 "If counts are zero, just answer based on their question"
#             ),
#         }
#     ]

#     if map_data_url is not None:
#         user_content.append({"type": "image_url", "image_url": {"url": map_data_url}})

#     resp = client.chat.completions.create(
#         model=OPENAI_MODEL,
#         messages=[
#             {"role": "system", "content": sys_msg},
#             {"role": "user", "content": user_content},
#         ],
#         temperature=0,
#     )

#     usage = getattr(resp, "usage", None)
#     cost = estimate_openai_cost_usd(OPENAI_MODEL, usage)

#     return resp.choices[0].message.content.strip(), cost


# # =========================
# # UI CONTROLS (no sidebar)
# # =========================
# question = st.text_input(
#     "Ask a question about the CURRENT aircraft snapshot:",
#     value="How many helicopters are in the air and what agencies are they from?",
# )

# col_a, col_b = st.columns([0.22, 0.78], vertical_alignment="bottom")
# with col_a:
#     run = st.container()
#     run.markdown('<div class="big-button">', unsafe_allow_html=True)
#     go = st.button("Run snapshot", use_container_width=True)
#     run.markdown("</div>", unsafe_allow_html=True)

# with col_b:
#     st.caption("")


# # =========================
# # RUN
# # =========================
# if go:
#     t0 = time.time()

#     try:
#         client_id = st.secrets["opensky"]["client_id"]
#         client_secret = st.secrets["opensky"]["client_secret"]
#     except Exception:
#         st.error("Missing OpenSky credentials in secrets. Add [opensky] client_id and client_secret.")
#         st.stop()

#     with st.spinner("Loading masterlist..."):
#         master_csv = st.secrets["masterlist"]["csv"]
#         master_raw = load_master_from_text(master_csv)
#         master = load_masterlist_df(master_raw)

#     with st.spinner("Fetching current OpenSky states..."):
#         token = get_access_token(client_id, client_secret)
#         data = fetch_states(token, bbox=bbox)

#     states = states_to_df(data)
#     matched = join_states_master(states, master)

#     if agencies_filter:
#         matched = matched[matched["Agency"].astype(str).str.upper().isin([a.upper() for a in agencies_filter])].copy()

#     api_time = data.get("time")
#     elapsed = time.time() - t0

#     st.subheader("Snapshot results")

#     k1, k2, k3, k4 = st.columns(4)
#     k1.metric("Masterlist aircraft", f"{len(master):,}")
#     k2.metric("Matched in snapshot", f"{len(matched):,}")
#     k3.metric("API time (epoch)", str(api_time))
#     k4.metric("Fetch + render", f"{elapsed:.2f}s")

#     if matched.empty:
#         st.warning("No matching USFS/CALFIRE aircraft found in this snapshot.")
#     else:
#         st.markdown("**Counts by agency**")
#         st.dataframe(
#             matched["Agency"].value_counts(dropna=False).rename_axis("Agency").reset_index(name="Count"),
#             use_container_width=True
#         )

#         st.markdown("**Sample matched rows**")
#         show_cols = ["Agency", "TailNumber", "icao24", "callsign", "Type", "latitude", "longitude", "alt_m", "velocity", "on_ground"]
#         st.dataframe(
#             matched[show_cols].sort_values(["Agency", "TailNumber"]).head(100),
#             use_container_width=True,
#             hide_index=True,
#         )

#     if GIS_OK:
#         st.markdown("**Map**")
#         title = f"OpenSky CURRENT states | WESTERN US bbox | matched={len(matched)}"
#         fig = plot_snapshot_basemap(matched, title=title, bbox=bbox)
#         st.pyplot(fig, use_container_width=True)

#         snapshot_summary = summarize_snapshot(matched)
#         map_url = fig_to_data_url(fig)
#     else:
#         snapshot_summary = summarize_snapshot(matched)
#         map_url = None

#     st.markdown("**Answer**")
#     with st.spinner("Answering question..."):
#         answer, cost = ask_snapshot_question(snapshot_summary, question=question, map_data_url=map_url)

#     st.write(answer)

#     if cost.get("model"):
#         st.caption(
#             f"OpenAI usage: model={cost.get('model')} | prompt={cost.get('prompt_tokens')} | completion={cost.get('completion_tokens')} | total={cost.get('total_tokens')} | est=${(cost.get('estimated_usd') or 0):.6f}"
#         )












# AVIATION.py
import os
import io
import time
import base64
from io import BytesIO

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st

# ---- OPTIONAL GIS STACK (so the app runs even if shapely/geopandas aren't installed)
GIS_OK = True
GIS_ERR = ""
try:
    import geopandas as gpd
    from shapely.geometry import Point
    import contextily as cx
except Exception as e:
    GIS_OK = False
    GIS_ERR = str(e)
    gpd = None
    Point = None
    cx = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# =========================
# STREAMLIT PAGE
# =========================
st.set_page_config(page_title="OpenSky USFS/CALFIRE Live Snapshot", layout="wide")

# Hide the sidebar entirely + Streamlit chrome (menu/footer/header)
st.markdown(
    """
    <style>
      section[data-testid="stSidebar"] {display: none;}
      div[data-testid="stSidebarNav"] {display: none;}
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .big-button button {
        background: linear-gradient(90deg, #1f2937, #111827);
        color: white !important;
        border-radius: 14px;
        padding: 0.75rem 1.25rem;
        border: 1px solid rgba(255,255,255,0.15);
        font-weight: 700;
        letter-spacing: 0.02em;
        box-shadow: 0 10px 30px rgba(0,0,0,0.20);
        transition: transform 0.06s ease-in-out;
    }
    .big-button button:hover {
        transform: translateY(-1px);
        border: 1px solid rgba(255,255,255,0.30);
    }
    .muted {
        color: rgba(0,0,0,0.65);
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("OpenSky Live Snapshot — USFS / CAL FIRE")
st.markdown(
    '<div class="muted">Queries current OpenSky “states” in a Western US bounding box, matches to your USFS/CALFIRE masterlist, then answers a question (optionally using OpenAI).</div>',
    unsafe_allow_html=True,
)

if not GIS_OK:
    st.warning(
        "Map rendering dependencies are missing in this environment (geopandas/shapely/contextily). "
        "The app will still run and show tables + answers. "
        f"Import error: {GIS_ERR}"
    )

# =========================
# CONFIG
# =========================
BBOX = (31.0, 49.5, -125.0, -102.0)  # (min_lat, max_lat, min_lon, max_lon)

STATES_URL = "https://opensky-network.org/api/states/all"
TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
UA = "opensky-live-usfs-calfire/1.0 (+research)"

# Basemap (only used if GIS_OK)
BASEMAP = None
BASEMAP_ZOOM = 6
if GIS_OK:
    BASEMAP = cx.providers.Esri.WorldTopoMap

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_API_KEY = st.secrets.get("openai", {}).get("api_key") or os.getenv("OPENAI_API_KEY")

# Hardcoded (no sidebar)
bbox = BBOX
agencies_filter = ["USFS", "CALFIRE"]

# Network timeouts (connect, read)
HTTP_TIMEOUT = (8, 45)


# =========================
# HTTP SESSION WITH RETRIES
# =========================
@st.cache_resource(show_spinner=False)
def http_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


# =========================
# MASTERLIST FROM SECRETS
# =========================
@st.cache_data(show_spinner=False)
def load_master_from_text(master_csv: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(master_csv))

def normalize_icao24(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    return s.replace("0x", "").replace(" ", "")

def load_masterlist_df(df: pd.DataFrame) -> pd.DataFrame:
    expected = {"TailNumber", "ADSB", "Agency"}
    missing = expected - set(df.columns)
    if missing:
        raise RuntimeError(f"Masterlist missing columns: {missing}. Found columns: {list(df.columns)}")

    df = df.copy()
    df["icao24"] = df["ADSB"].apply(normalize_icao24)
    df = df[df["icao24"] != ""].drop_duplicates(subset=["icao24"]).reset_index(drop=True)

    df["Agency"] = df["Agency"].astype(str).str.strip().str.upper()
    if "Type" in df.columns:
        df["Type"] = df["Type"].astype(str).str.strip()
    else:
        df["Type"] = ""
    return df


# =========================
# AUTH (OpenSky)
# =========================
def get_access_token(client_id: str, client_secret: str, timeout=HTTP_TIMEOUT) -> str:
    sess = http_session()
    try:
        r = sess.post(
            TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": UA},
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=timeout,
        )
    except requests.exceptions.ConnectTimeout as e:
        raise RuntimeError(
            f"ConnectTimeout to OpenSky token endpoint: {TOKEN_URL}. "
            "If this happens on Streamlit Cloud, configure Basic Auth fallback (opensky username/password) "
            "or deploy somewhere with reliable egress to OpenSky."
        ) from e

    if r.status_code == 403:
        raise RuntimeError("403 from token endpoint.\n" + (r.text or "")[:800])

    r.raise_for_status()
    tok = r.json()
    access_token = tok.get("access_token")
    if not access_token:
        raise RuntimeError(f"Token response missing access_token: {tok}")
    return access_token


# =========================
# FETCH (OpenSky states)
# =========================
def fetch_states(bbox=None, token=None, basic_auth=None, timeout=HTTP_TIMEOUT) -> dict:
    sess = http_session()
    params = {}
    if bbox is not None:
        min_lat, max_lat, min_lon, max_lon = bbox
        params.update({"lamin": min_lat, "lamax": max_lat, "lomin": min_lon, "lomax": max_lon})

    headers = {"User-Agent": UA}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    r = sess.get(
        STATES_URL,
        headers=headers,
        params=params,
        auth=basic_auth,  # tuple (username, password) or None
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


# =========================
# DATA WRANGLING
# =========================
def states_to_df(data: dict) -> pd.DataFrame:
    cols = [
        "icao24", "callsign", "origin_country", "time_position", "last_contact",
        "longitude", "latitude", "baro_altitude", "on_ground", "velocity",
        "true_track", "vertical_rate", "sensors", "geo_altitude", "squawk",
        "spi", "position_source",
    ]
    states = data.get("states") or []
    df = pd.DataFrame(states, columns=cols)

    df["icao24"] = df["icao24"].apply(normalize_icao24)
    df["callsign"] = df["callsign"].astype(str).str.strip().replace({"None": ""})

    for c in ["longitude", "latitude", "baro_altitude", "geo_altitude", "velocity", "true_track"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["longitude"].between(-180, 180) & df["latitude"].between(-90, 90)].copy()
    df["on_ground"] = df["on_ground"].astype("boolean")
    return df

def join_states_master(states_df: pd.DataFrame, master_df: pd.DataFrame) -> pd.DataFrame:
    out = states_df.merge(master_df, how="inner", on="icao24")
    out["alt_m"] = out["geo_altitude"].fillna(out["baro_altitude"])
    return out


# =========================
# MAP (optional)
# =========================
def bbox_to_webmercator(bbox, pad_frac=0.06):
    min_lat, max_lat, min_lon, max_lon = bbox
    lat_pad = (max_lat - min_lat) * pad_frac
    lon_pad = (max_lon - min_lon) * pad_frac
    min_lat -= lat_pad
    max_lat += lat_pad
    min_lon -= lon_pad
    max_lon += lon_pad

    p1 = gpd.GeoSeries([Point(min_lon, min_lat)], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
    p2 = gpd.GeoSeries([Point(max_lon, max_lat)], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
    return p1.x, p2.x, p1.y, p2.y

def to_gdf_webmercator(df: pd.DataFrame) -> "gpd.GeoDataFrame":
    d = df.dropna(subset=["longitude", "latitude"]).copy()
    gdf2 = gpd.GeoDataFrame(
        d,
        geometry=[Point(xy) for xy in zip(d["longitude"].astype(float), d["latitude"].astype(float))],
        crs="EPSG:4326",
    )
    return gdf2.to_crs(epsg=3857)

def plot_snapshot_basemap(df: pd.DataFrame, title: str, bbox=None):
    fig, ax = plt.subplots(figsize=(12.5, 8.0))
    ax.set_facecolor("white")

    if bbox is not None:
        minx, maxx, miny, maxy = bbox_to_webmercator(bbox, pad_frac=0.06)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

    cx.add_basemap(ax, source=BASEMAP, zoom=BASEMAP_ZOOM, attribution_size=7)

    agency_color = {"USFS": "green", "CALFIRE": "blue"}

    if df is not None and len(df) > 0:
        gdf2 = to_gdf_webmercator(df)
        for agency, g in gdf2.groupby("Agency"):
            speed = pd.to_numeric(g["velocity"], errors="coerce").fillna(0.0)
            sizes = (speed.clip(0, 220) / 220.0) * 110.0 + 28.0
            ax.scatter(
                g.geometry.x, g.geometry.y,
                s=sizes,
                label=f"{agency} (n={len(g)})",
                alpha=0.9,
                linewidths=0.8,
                edgecolors="white",
                c=agency_color.get(str(agency).upper(), "gray"),
                zorder=10,
            )
        ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95)
    else:
        ax.text(
            0.02, 0.02,
            "No matching USFS/CALFIRE aircraft in this snapshot.",
            transform=ax.transAxes,
            fontsize=11,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
            zorder=20,
        )

    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    return fig

def fig_to_data_url(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="PNG", dpi=160, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return "data:image/png;base64," + b64


# =========================
# SUMMARY + QUESTION ANSWERING
# =========================
def classify_airframe_is_heli(type_str: str, callsign: str = "") -> bool:
    t = (type_str or "").strip().lower()
    c = (callsign or "").strip().lower()
    if any(k in t for k in ["heli", "helic", "rotor", "rotary"]):
        return True
    if t in {"h", "hel", "heli"}:
        return True
    if t.startswith("h ") or t.endswith(" h") or t.startswith("h-") or t.startswith("h_"):
        return True
    if t.replace("-", " ").replace("_", " ").strip() in {"type 1", "type1", "type 2", "type2"}:
        return True
    return False

def summarize_snapshot(matched: pd.DataFrame) -> dict:
    if matched is None or matched.empty:
        return {
            "matched_total": 0,
            "airborne_total": 0,
            "agencies_airborne": {},
            "helicopters_airborne_total": 0,
            "helicopters_by_agency": {},
            "note": "No matches in snapshot.",
        }

    m = matched.copy()
    m["Agency"] = m["Agency"].astype(str).str.strip().str.upper()
    m["is_airborne"] = (~m["on_ground"].fillna(False)).astype(bool)

    airborne = m[m["is_airborne"]].copy()
    agencies_airborne = airborne["Agency"].value_counts().to_dict() if not airborne.empty else {}

    airborne["is_heli"] = [
        classify_airframe_is_heli(t, cs) for t, cs in zip(airborne.get("Type", ""), airborne.get("callsign", ""))
    ]
    helis = airborne[airborne["is_heli"]].copy()

    return {
        "matched_total": int(len(m)),
        "airborne_total": int(len(airborne)),
        "agencies_airborne": agencies_airborne,
        "helicopters_airborne_total": int(len(helis)),
        "helicopters_by_agency": helis["Agency"].value_counts().to_dict() if not helis.empty else {},
        "sample_airborne": airborne[
            ["Agency", "TailNumber", "icao24", "callsign", "Type", "latitude", "longitude", "alt_m", "velocity", "on_ground"]
        ].sort_values(["Agency", "TailNumber"]).head(25).to_dict(orient="records") if not airborne.empty else [],
    }

OPENAI_PRICING_PER_1M = {
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "gpt-4o-2024-08-06": {"in": 2.50, "out": 10.00},
    "chatgpt-4o-latest": {"in": 5.00, "out": 15.00},
}

def estimate_openai_cost_usd(model: str, usage_obj) -> dict:
    if usage_obj is None:
        return {"model": model, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated_usd": 0.0}

    prompt_tokens = getattr(usage_obj, "prompt_tokens", None)
    completion_tokens = getattr(usage_obj, "completion_tokens", None)
    total_tokens = getattr(usage_obj, "total_tokens", None)

    if prompt_tokens is None and isinstance(usage_obj, dict):
        prompt_tokens = usage_obj.get("prompt_tokens", 0)
        completion_tokens = usage_obj.get("completion_tokens", 0)
        total_tokens = usage_obj.get("total_tokens", (prompt_tokens or 0) + (completion_tokens or 0))

    prompt_tokens = int(prompt_tokens or 0)
    completion_tokens = int(completion_tokens or 0)
    total_tokens = int(total_tokens or (prompt_tokens + completion_tokens))

    rates = OPENAI_PRICING_PER_1M.get(model)
    if rates is None:
        return {"model": model, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens, "estimated_usd": None}

    est = (prompt_tokens / 1_000_000.0) * rates["in"] + (completion_tokens / 1_000_000.0) * rates["out"]
    return {"model": model, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens, "estimated_usd": float(est)}

def ask_snapshot_question(snapshot_summary: dict, question: str, map_data_url: str = None) -> tuple[str, dict]:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Add OPENAI_API_KEY to Streamlit secrets or environment variables."
        )

    if OpenAI is None:
        raise RuntimeError("openai package not installed. Add `openai` to requirements.txt.")

    client = OpenAI(api_key=OPENAI_API_KEY)

    sys_msg = (
        "You are an operational aviation analyst for wildfire response. "
        "You MUST answer the user's question even if the snapshot shows zero aircraft. "
        "Do not refuse to answer."
    )

    compact = {
        "matched_total": snapshot_summary.get("matched_total", 0),
        "airborne_total": snapshot_summary.get("airborne_total", 0),
        "agencies_airborne": snapshot_summary.get("agencies_airborne", {}),
        "helicopters_airborne_total": snapshot_summary.get("helicopters_airborne_total", 0),
        "helicopters_by_agency": snapshot_summary.get("helicopters_by_agency", {}),
        "sample_airborne": snapshot_summary.get("sample_airborne", [])[:25],
    }

    user_content = [
        {
            "type": "text",
            "text": (
                "Here is a live aircraft snapshot summary:\n"
                f"{compact}\n\n"
                f"Question: {question}\n\n"
                "If counts are zero, just answer based on their question"
            ),
        }
    ]

    if map_data_url is not None:
        user_content.append({"type": "image_url", "image_url": {"url": map_data_url}})

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
    )

    usage = getattr(resp, "usage", None)
    cost = estimate_openai_cost_usd(OPENAI_MODEL, usage)

    return resp.choices[0].message.content.strip(), cost


# =========================
# UI CONTROLS (no sidebar)
# =========================
question = st.text_input(
    "Ask a question about the CURRENT aircraft snapshot:",
    value="How many helicopters are in the air and what agencies are they from?",
)

col_a, col_b = st.columns([0.22, 0.78], vertical_alignment="bottom")
with col_a:
    run = st.container()
    run.markdown('<div class="big-button">', unsafe_allow_html=True)
    go = st.button("Run snapshot", use_container_width=True)
    run.markdown("</div>", unsafe_allow_html=True)

with col_b:
    st.caption("")


# =========================
# RUN
# =========================
if go:
    t0 = time.time()

    try:
        client_id = st.secrets["opensky"]["client_id"]
        client_secret = st.secrets["opensky"]["client_secret"]
    except Exception:
        st.error("Missing OpenSky credentials in secrets. Add [opensky] client_id and client_secret.")
        st.stop()

    # Optional Basic Auth fallback for /states/all
    basic_auth = None
    if "username" in st.secrets.get("opensky", {}) and "password" in st.secrets.get("opensky", {}):
        basic_auth = (st.secrets["opensky"]["username"], st.secrets["opensky"]["password"])

    with st.spinner("Loading masterlist..."):
        master_csv = st.secrets["masterlist"]["csv"]
        master_raw = load_master_from_text(master_csv)
        master = load_masterlist_df(master_raw)

    with st.spinner("Fetching current OpenSky states..."):
        token = None
        token_err = None
        try:
            token = get_access_token(client_id, client_secret)
        except Exception as e:
            token_err = str(e)

        try:
            if token:
                data = fetch_states(bbox=bbox, token=token, basic_auth=None)
            elif basic_auth is not None:
                st.warning("OAuth token fetch failed; falling back to Basic Auth for OpenSky states.")
                data = fetch_states(bbox=bbox, token=None, basic_auth=basic_auth)
            else:
                st.error(
                    "OpenSky OAuth token fetch failed and no Basic Auth fallback is configured.\n\n"
                    f"OAuth error:\n{token_err}"
                )
                st.stop()
        except Exception as e:
            st.error(f"OpenSky request failed: {e}\n\nOAuth error (earlier): {token_err or '(none)'}")
            st.stop()

    states = states_to_df(data)
    matched = join_states_master(states, master)

    if agencies_filter:
        matched = matched[matched["Agency"].astype(str).str.upper().isin([a.upper() for a in agencies_filter])].copy()

    api_time = data.get("time")
    elapsed = time.time() - t0

    st.subheader("Snapshot results")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Masterlist aircraft", f"{len(master):,}")
    k2.metric("Matched in snapshot", f"{len(matched):,}")
    k3.metric("API time (epoch)", str(api_time))
    k4.metric("Fetch + render", f"{elapsed:.2f}s")

    if matched.empty:
        st.warning("No matching USFS/CALFIRE aircraft found in this snapshot.")
    else:
        st.markdown("**Counts by agency**")
        st.dataframe(
            matched["Agency"].value_counts(dropna=False).rename_axis("Agency").reset_index(name="Count"),
            use_container_width=True
        )

        st.markdown("**Sample matched rows**")
        show_cols = ["Agency", "TailNumber", "icao24", "callsign", "Type", "latitude", "longitude", "alt_m", "velocity", "on_ground"]
        st.dataframe(
            matched[show_cols].sort_values(["Agency", "TailNumber"]).head(100),
            use_container_width=True,
            hide_index=True,
        )

    if GIS_OK:
        st.markdown("**Map**")
        title = f"OpenSky CURRENT states | WESTERN US bbox | matched={len(matched)}"
        fig = plot_snapshot_basemap(matched, title=title, bbox=bbox)
        st.pyplot(fig, use_container_width=True)

        snapshot_summary = summarize_snapshot(matched)
        map_url = fig_to_data_url(fig)
    else:
        snapshot_summary = summarize_snapshot(matched)
        map_url = None

    st.markdown("**Answer**")
    with st.spinner("Answering question..."):
        answer, cost = ask_snapshot_question(snapshot_summary, question=question, map_data_url=map_url)

    st.write(answer)

    if cost.get("model"):
        st.caption(
            f"OpenAI usage: model={cost.get('model')} | prompt={cost.get('prompt_tokens')} | completion={cost.get('completion_tokens')} | total={cost.get('total_tokens')} | est=${(cost.get('estimated_usd') or 0):.6f}"
        )




