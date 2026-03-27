

# import os
# import io
# import time
# import base64
# from io import BytesIO

# import requests
# import pandas as pd
# import matplotlib.pyplot as plt

# import streamlit as st

# st.set_page_config(
#     page_title="LLM-Powered Wildfire Aviation Dashboard",
#     layout="wide",
# )
# st.markdown("""
# <style>
#   .block-container {
#     max-width: 100% !important;
#     padding-left: 2.5rem;
#     padding-right: 2.5rem;
#   }
# </style>
# """, unsafe_allow_html=True)


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
# # STREAMLIT PAGE (ONE UI SECTION ONLY)
# # =========================
# # If you use this, it MUST be the first Streamlit call in the file.
# # st.set_page_config(page_title="OpenSky USFS/CALFIRE Live Snapshot", layout="wide")

# st.title("LLM-Powered Wildfire Aviation Dashboard")
# st.caption("Current aircraft locations (not historical tracks)")
# st.divider()

# st.markdown(
#     """
#     <style>
#       /* Hide Streamlit chrome */
#       section[data-testid="stSidebar"] {display: none !important;}
#       div[data-testid="stSidebarNav"] {display: none !important;}
#       #MainMenu {visibility: hidden;}
#       footer {visibility: hidden;}
#       header {visibility: hidden;}

#       /* Typography + spacing */
#       .app-subtitle {
#         color: rgba(255,255,255,0.72);
#         font-size: 0.95rem;
#         line-height: 1.35;
#         margin-top: 0.15rem;
#         margin-bottom: 0.90rem;
#       }

#       /* Inputs readable on dark background */
#       textarea, input, [data-baseweb="textarea"] textarea, [data-baseweb="input"] input {
#         color: #111827 !important;
#         background: #ffffff !important;
#         -webkit-text-fill-color: #111827 !important;
#         caret-color: #111827 !important;
#       }
#       [data-baseweb="textarea"] textarea::placeholder,
#       [data-baseweb="input"] input::placeholder {
#         color: rgba(17,24,39,0.45) !important;
#         -webkit-text-fill-color: rgba(17,24,39,0.45) !important;
#       }

#       /* Tighten top padding */
#       .block-container { padding-top: 1.15rem; }

#       /* Big RUN button */
#       .big-button button {
#         background: linear-gradient(90deg, #ef4444, #fb7185) !important;
#         color: white !important;
#         border-radius: 14px !important;
#         padding: 0.80rem 1.25rem !important;
#         border: 1px solid rgba(255,255,255,0.18) !important;
#         font-weight: 900 !important;
#         letter-spacing: 0.02em !important;
#         box-shadow: 0 10px 30px rgba(0,0,0,0.20) !important;
#         transition: transform 0.06s ease-in-out;
#       }
#       .big-button button:hover {
#         transform: translateY(-1px);
#         border: 1px solid rgba(255,255,255,0.30) !important;
#       }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# st.markdown(
#     '<div class="app-subtitle">'
#     #'Pulls current OpenSky “states” in a Western US bounding box, matches to your USFS/CALFIRE masterlist, '
#    # 'then answers questions about <b>this live snapshot</b> (optional OpenAI).'
#     '</div>',
#     unsafe_allow_html=True,
# )

# # ONE question widget (keep this as text_area so it matches your screenshot)
# question = st.text_area(
#     "Ask a question about the CURRENT wildland aircraft snapshot:",
#     value="",
#     height=110,
# )

# st.markdown('<div class="big-button">', unsafe_allow_html=True)
# go = st.button("RUN", use_container_width=True, type="primary")
# st.markdown("</div>", unsafe_allow_html=True)

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
# def get_openai_client():
#     api_key = (
#         st.secrets.get("openai", {}).get("api_key")
#         or os.getenv("OPENAI_API_KEY")
#         or ""
#     ).strip()

#     if not api_key:
#         raise RuntimeError(
#             "OPENAI_API_KEY is not set. Add [openai] api_key to Streamlit secrets or set OPENAI_API_KEY as an environment variable."
#         )

#     return OpenAI(api_key=api_key)


# # =========================
# # FETCH
# # =========================
# def fetch_states(token: str, bbox=None, timeout=90) -> dict:
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
#     fig, ax = plt.subplots(figsize=(8, 5.0))
#     ax.set_facecolor("white")

#     if bbox is not None:
#         minx, maxx, miny, maxy = bbox_to_webmercator(bbox, pad_frac=0.06)
#         ax.set_xlim(minx, maxx)
#         ax.set_ylim(miny, maxy)

#     cx.add_basemap(ax, source=BASEMAP, zoom=BASEMAP_ZOOM, attribution_size=7)

#     agency_color = {"USFS": "green", "CALFIRE": "blue"}

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
#             "OPENAI_API_KEY is not set. Add OPENAI_API_KEY to Streamlit secrets or environment variables."
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
#                 "If counts are zero, still answer the question based on this context."
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
# def get_access_token(client_id: str, client_secret: str, timeout=(15, 60), max_retries=3) -> str:
#     last_err = None

#     for attempt in range(1, max_retries + 1):
#         try:
#             r = requests.post(
#                 TOKEN_URL,
#                 data={
#                     "grant_type": "client_credentials",
#                     "client_id": client_id,
#                     "client_secret": client_secret,
#                 },
#                 headers={"User-Agent": UA},
#                 timeout=timeout,
#             )
#             r.raise_for_status()

#             payload = r.json()
#             access_token = payload.get("access_token", "").strip()
#             if not access_token:
#                 raise RuntimeError(f"OpenSky token response missing access_token. Response keys: {list(payload.keys())}")

#             return access_token

#         except requests.exceptions.ConnectTimeout as e:
#             last_err = e
#             time.sleep(2 * attempt)

#         except requests.exceptions.ReadTimeout as e:
#             last_err = e
#             time.sleep(2 * attempt)

#         except requests.exceptions.RequestException as e:
#             try:
#                 detail = e.response.text[:500]
#             except Exception:
#                 detail = str(e)
#             raise RuntimeError(f"OpenSky token request failed: {detail}") from e

#     raise RuntimeError(f"OpenSky token request timed out after {max_retries} attempts: {last_err}")

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
#         try:
#             master_csv = st.secrets["masterlist"]["csv"]
#         except Exception:
#             st.error("Missing masterlist in secrets. Add [masterlist] csv = \"\"\"...\"\"\"")
#             st.stop()

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
#         st.pyplot(fig, use_container_width=False)

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
#         est = cost.get("estimated_usd")
#         st.caption(
#             f"OpenAI usage: model={cost.get('model')} | prompt={cost.get('prompt_tokens')} | "
#             f"completion={cost.get('completion_tokens')} | total={cost.get('total_tokens')} | "
#             f"est=${(est or 0):.6f}" if est is not None else
#             f"OpenAI usage: model={cost.get('model')} | prompt={cost.get('prompt_tokens')} | "
#             f"completion={cost.get('completion_tokens')} | total={cost.get('total_tokens')} | est=N/A"
#         )





import os
import io
import time
from io import BytesIO

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# =========================
# PAGE
# =========================
st.set_page_config(
    page_title="LLM-Powered Wildfire Aviation Dashboard",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {
        max-width: 100% !important;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        padding-top: 1.15rem;
      }
      section[data-testid="stSidebar"] {display: none !important;}
      div[data-testid="stSidebarNav"] {display: none !important;}
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}

      textarea, input, [data-baseweb="textarea"] textarea, [data-baseweb="input"] input {
        color: #111827 !important;
        background: #ffffff !important;
        -webkit-text-fill-color: #111827 !important;
        caret-color: #111827 !important;
      }
      [data-baseweb="textarea"] textarea::placeholder,
      [data-baseweb="input"] input::placeholder {
        color: rgba(17,24,39,0.45) !important;
        -webkit-text-fill-color: rgba(17,24,39,0.45) !important;
      }
      .big-button button {
        background: linear-gradient(90deg, #ef4444, #fb7185) !important;
        color: white !important;
        border-radius: 14px !important;
        padding: 0.80rem 1.25rem !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        font-weight: 900 !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.20) !important;
        transition: transform 0.06s ease-in-out;
      }
      .big-button button:hover {
        transform: translateY(-1px);
        border: 1px solid rgba(255,255,255,0.30) !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("LLM-Powered Wildfire Aviation Dashboard")
st.caption("Current aircraft locations from an OpenSky snapshot, matched to your masterlist")

question = st.text_area(
    "Ask a question about the CURRENT wildland aircraft snapshot:",
    value="",
    height=110,
)

st.markdown('<div class="big-button">', unsafe_allow_html=True)
go = st.button("RUN", use_container_width=True, type="primary")
st.markdown("</div>", unsafe_allow_html=True)

# =========================
# CONFIG
# =========================
BBOX = (31.0, 49.5, -125.0, -102.0)  # (min_lat, max_lat, min_lon, max_lon)
STATES_URL = "https://opensky-network.org/api/states/all"
TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
UA = "opensky-live-usfs-calfire/1.0 (+research)"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
bbox = BBOX
agencies_filter = ["USFS", "CALFIRE"]

OPENAI_PRICING_PER_1M = {
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "chatgpt-4o-latest": {"in": 5.00, "out": 15.00},
}

# =========================
# HELPERS
# =========================
def get_secret(section: str, key: str, default=None):
    try:
        return st.secrets[section][key]
    except Exception:
        return default


def build_session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


SESSION = build_session()


def normalize_icao24(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    return s.replace("0x", "").replace(" ", "")


@st.cache_data(show_spinner=False)
def load_master_from_text(master_csv: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(master_csv))


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


def get_openai_client():
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Add `openai` to requirements.txt.")

    api_key = (
        get_secret("openai", "api_key", None)
        or os.getenv("OPENAI_API_KEY")
        or ""
    ).strip()

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add [openai] api_key to Streamlit secrets or set OPENAI_API_KEY in the environment."
        )

    return OpenAI(api_key=api_key)


def get_access_token(client_id: str, client_secret: str, max_retries: int = 3) -> str:
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            r = SESSION.post(
                TOKEN_URL,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
                timeout=(20, 90),
            )

            if r.status_code == 403:
                raise RuntimeError(f"OpenSky token endpoint returned 403: {r.text[:800]}")
            if r.status_code >= 400:
                raise RuntimeError(f"OpenSky token endpoint returned {r.status_code}: {r.text[:800]}")

            payload = r.json()
            access_token = str(payload.get("access_token", "")).strip()
            if not access_token:
                raise RuntimeError(f"OpenSky token response missing access_token. Response keys: {list(payload.keys())}")

            return access_token

        except requests.exceptions.ConnectTimeout as e:
            last_err = e
            time.sleep(2 * attempt)
        except requests.exceptions.ReadTimeout as e:
            last_err = e
            time.sleep(2 * attempt)
        except requests.exceptions.RequestException as e:
            last_err = e
            time.sleep(2 * attempt)
        except Exception as e:
            raise RuntimeError(f"OpenSky token fetch failed: {e}") from e

    raise RuntimeError(
        "Could not connect to the OpenSky auth server after multiple attempts. "
        f"Last error: {type(last_err).__name__}: {last_err}"
    )


def fetch_states(token: str, bbox=None) -> dict:
    params = {}
    if bbox is not None:
        min_lat, max_lat, min_lon, max_lon = bbox
        params.update(
            {"lamin": min_lat, "lamax": max_lat, "lomin": min_lon, "lomax": max_lon}
        )

    r = SESSION.get(
        STATES_URL,
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        timeout=(20, 120),
    )

    if r.status_code >= 400:
        raise RuntimeError(f"OpenSky states endpoint returned {r.status_code}: {r.text[:800]}")

    return r.json()


def states_to_df(data: dict) -> pd.DataFrame:
    cols = [
        "icao24", "callsign", "origin_country", "time_position", "last_contact",
        "longitude", "latitude", "baro_altitude", "on_ground", "velocity",
        "true_track", "vertical_rate", "sensors", "geo_altitude", "squawk",
        "spi", "position_source",
    ]
    states = data.get("states") or []
    if not states:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(states, columns=cols)
    df["icao24"] = df["icao24"].apply(normalize_icao24)
    df["callsign"] = df["callsign"].astype(str).str.strip().replace({"None": ""})

    for c in ["longitude", "latitude", "baro_altitude", "geo_altitude", "velocity", "true_track", "vertical_rate"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["longitude"].between(-180, 180) & df["latitude"].between(-90, 90)].copy()
    df["on_ground"] = df["on_ground"].astype("boolean")
    return df


def join_states_master(states_df: pd.DataFrame, master_df: pd.DataFrame) -> pd.DataFrame:
    if states_df.empty:
        out = master_df.iloc[0:0].copy()
        out["callsign"] = pd.Series(dtype="object")
        out["longitude"] = pd.Series(dtype="float64")
        out["latitude"] = pd.Series(dtype="float64")
        out["baro_altitude"] = pd.Series(dtype="float64")
        out["geo_altitude"] = pd.Series(dtype="float64")
        out["velocity"] = pd.Series(dtype="float64")
        out["true_track"] = pd.Series(dtype="float64")
        out["vertical_rate"] = pd.Series(dtype="float64")
        out["on_ground"] = pd.Series(dtype="boolean")
        out["alt_m"] = pd.Series(dtype="float64")
        return out

    out = states_df.merge(master_df, how="inner", on="icao24")
    out["alt_m"] = out["geo_altitude"].fillna(out["baro_altitude"])
    return out


def meters_to_feet(x):
    if pd.isna(x):
        return None
    return float(x) * 3.28084


def mps_to_knots(x):
    if pd.isna(x):
        return None
    return float(x) * 1.94384


def summarize_snapshot(matched: pd.DataFrame) -> dict:
    if matched.empty:
        return {
            "matched_total": 0,
            "airborne_total": 0,
            "agencies_airborne": {},
            "helicopters_airborne_total": 0,
            "helicopters_by_agency": {},
            "sample_airborne": [],
        }

    df = matched.copy()

    if "Type" not in df.columns:
        df["Type"] = ""

    airborne = df[(df["on_ground"] != True) | (df["alt_m"].fillna(0) > 30)].copy()

    heli_mask = df["Type"].astype(str).str.contains("helicopter|helo|helitack", case=False, na=False)
    airborne_heli = airborne[heli_mask.loc[airborne.index]]

    sample_cols = ["TailNumber", "Agency", "Type", "callsign", "longitude", "latitude", "alt_m", "velocity", "true_track"]
    existing_sample_cols = [c for c in sample_cols if c in airborne.columns]
    sample = airborne[existing_sample_cols].head(25).copy()

    if "alt_m" in sample.columns:
        sample["alt_ft"] = sample["alt_m"].apply(meters_to_feet)
    if "velocity" in sample.columns:
        sample["speed_kt"] = sample["velocity"].apply(mps_to_knots)

    return {
        "matched_total": int(len(df)),
        "airborne_total": int(len(airborne)),
        "agencies_airborne": airborne["Agency"].value_counts(dropna=False).to_dict(),
        "helicopters_airborne_total": int(len(airborne_heli)),
        "helicopters_by_agency": airborne_heli["Agency"].value_counts(dropna=False).to_dict(),
        "sample_airborne": sample.to_dict(orient="records"),
    }


def estimate_openai_cost_usd(model: str, usage_obj) -> dict:
    if usage_obj is None:
        return {
            "model": model,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_usd": 0.0,
        }

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
        return {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_usd": None,
        }

    est = (prompt_tokens / 1_000_000.0) * rates["in"] + (completion_tokens / 1_000_000.0) * rates["out"]
    return {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_usd": float(est),
    }


def ask_snapshot_question(snapshot_summary: dict, question: str) -> tuple[str, dict]:
    client = get_openai_client()

    sys_msg = (
        "You are an operational aviation analyst for wildfire response. "
        "Answer only from the provided snapshot summary. "
        "If counts are zero, say that clearly and do not invent aircraft."
    )

    compact = {
        "matched_total": snapshot_summary.get("matched_total", 0),
        "airborne_total": snapshot_summary.get("airborne_total", 0),
        "agencies_airborne": snapshot_summary.get("agencies_airborne", {}),
        "helicopters_airborne_total": snapshot_summary.get("helicopters_airborne_total", 0),
        "helicopters_by_agency": snapshot_summary.get("helicopters_by_agency", {}),
        "sample_airborne": snapshot_summary.get("sample_airborne", [])[:25],
    }

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": sys_msg},
            {
                "role": "user",
                "content": (
                    "Here is a live aircraft snapshot summary:\n"
                    f"{compact}\n\n"
                    f"Question: {question}\n\n"
                    "If counts are zero, still answer the question directly from this context."
                ),
            },
        ],
        temperature=0,
    )

    usage = getattr(resp, "usage", None)
    cost = estimate_openai_cost_usd(OPENAI_MODEL, usage)
    return resp.choices[0].message.content.strip(), cost


def plot_points(matched: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 7))

    if matched.empty:
        ax.text(0.5, 0.5, "No matching aircraft in snapshot", ha="center", va="center", transform=ax.transAxes, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig

    colors = matched["Agency"].astype(str).map({
        "USFS": "tab:orange",
        "CALFIRE": "tab:red",
    }).fillna("tab:blue")

    ax.scatter(
        matched["longitude"],
        matched["latitude"],
        s=50,
        c=colors.tolist(),
        alpha=0.85,
    )

    for _, row in matched.head(100).iterrows():
        label = row.get("TailNumber") or row.get("callsign") or row.get("icao24")
        if pd.notna(row.get("longitude")) and pd.notna(row.get("latitude")):
            ax.text(row["longitude"], row["latitude"], str(label), fontsize=7)

    min_lat, max_lat, min_lon, max_lon = bbox
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Matched aircraft in current OpenSky snapshot")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


# =========================
# RUN
# =========================
if go:
    t0 = time.time()

    client_id = get_secret("opensky", "client_id", None)
    client_secret = get_secret("opensky", "client_secret", None)
    if not client_id or not client_secret:
        st.error("Missing OpenSky credentials in secrets. Add [opensky] client_id and client_secret.")
        st.stop()

    master_csv = get_secret("masterlist", "csv", None)
    if not master_csv:
        st.error('Missing masterlist in secrets. Add [masterlist] csv = """..."""')
        st.stop()

    with st.spinner("Loading masterlist..."):
        try:
            master_raw = load_master_from_text(master_csv)
            master = load_masterlist_df(master_raw)
        except Exception as e:
            st.error(f"Could not load masterlist: {e}")
            st.stop()

    try:
        with st.spinner("Fetching OpenSky access token..."):
            token = get_access_token(client_id, client_secret)

        with st.spinner("Fetching current OpenSky states..."):
            data = fetch_states(token, bbox=bbox)

    except Exception as e:
        st.error(f"OpenSky fetch failed: {e}")
        st.stop()

    states = states_to_df(data)
    matched = join_states_master(states, master)

    if agencies_filter:
        matched = matched[
            matched["Agency"].astype(str).str.upper().isin([a.upper() for a in agencies_filter])
        ].copy()

    api_time = data.get("time")
    elapsed = time.time() - t0

    st.subheader("Snapshot results")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Masterlist aircraft", f"{len(master):,}")
    k2.metric("Matched in snapshot", f"{len(matched):,}")
    k3.metric("API time (epoch)", str(api_time))
    k4.metric("Fetch + render", f"{elapsed:.2f}s")

    summary = summarize_snapshot(matched)

    if matched.empty:
        st.warning("No matching USFS/CALFIRE aircraft found in this snapshot.")
    else:
        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown("**Counts by agency**")
            st.dataframe(
                matched["Agency"].value_counts(dropna=False).rename_axis("Agency").reset_index(name="Count"),
                use_container_width=True,
                hide_index=True,
            )

        with c2:
            airborne = matched[(matched["on_ground"] != True) | (matched["alt_m"].fillna(0) > 30)].copy()
            st.markdown("**Airborne counts by agency**")
            if airborne.empty:
                st.info("No airborne matches in this snapshot.")
            else:
                st.dataframe(
                    airborne["Agency"].value_counts(dropna=False).rename_axis("Agency").reset_index(name="Count"),
                    use_container_width=True,
                    hide_index=True,
                )

        display_df = matched.copy()
        if "alt_m" in display_df.columns:
            display_df["alt_ft"] = display_df["alt_m"].apply(meters_to_feet)
        if "velocity" in display_df.columns:
            display_df["speed_kt"] = display_df["velocity"].apply(mps_to_knots)

        preferred_cols = [
            "TailNumber", "Agency", "Type", "callsign", "longitude", "latitude",
            "on_ground", "alt_ft", "speed_kt", "true_track", "icao24"
        ]
        existing_cols = [c for c in preferred_cols if c in display_df.columns]

        st.markdown("**Matched aircraft table**")
        st.dataframe(display_df[existing_cols], use_container_width=True, hide_index=True)

        st.markdown("**Map**")
        fig = plot_points(matched)
        st.pyplot(fig, clear_figure=True)

    if question.strip():
        st.subheader("LLM answer")
        try:
            with st.spinner("Generating answer..."):
                answer, usage = ask_snapshot_question(summary, question.strip())
            st.write(answer)

            st.caption(
                f"Model: {usage['model']} | "
                f"Prompt tokens: {usage['prompt_tokens']} | "
                f"Completion tokens: {usage['completion_tokens']} | "
                f"Estimated cost (USD): {usage['estimated_usd']}"
            )
        except Exception as e:
            st.error(f"LLM question failed: {e}")


