import os
import time
import base64
from io import BytesIO

import requests
import pandas as pd
import matplotlib.pyplot as plt

import geopandas as gpd
from shapely.geometry import Point, box
import contextily as cx

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# =========================
# CONFIG
# =========================
MASTER_XLSX = r"C:\Users\magst\Downloads\ADSBCODE_MASTERLIST_USFSCALFIRE.xlsx"  # change if needed

# Western US bbox (min_lat, max_lat, min_lon, max_lon)
BBOX = (31.0, 49.5, -125.0, -102.0)

STATES_URL = "https://opensky-network.org/api/states/all"
TOKEN_URL  = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"

UA = "opensky-live-usfs-calfire/1.0 (+research)"

# Basemap
BASEMAP = cx.providers.Esri.WorldTopoMap
BASEMAP_ZOOM = 6

# OpenAI (optional)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # optional

# OpenSky OAuth client creds (recommended via env vars)
# CLIENT_ID = os.getenv("OPENSKY_CLIENT_ID")
# CLIENT_SECRET = os.getenv("OPENSKY_CLIENT_SECRET")
CLIENT_ID = "magstadt.shayne@hotmail.com-api-client"
CLIENT_SECRET = "ELrE549ykC0B26LI1EBoHppXqqi7l5HG"

# -------------------------
# OPTIONAL BACKGROUND CONTEXT LAYERS (drawn under points)
# Provide local paths to vector data (SHP / GeoJSON / GPKG etc.)
BACKGROUND_LAYERS = []

# Turn off outline + bbox frame (per request)
SHOW_US_OUTLINE = False
SHOW_BBOX_FRAME = False


# =========================
# AUTH
# =========================
def get_access_token(client_id: str, client_secret: str, timeout=30) -> str:
    if not (client_id and client_secret):
        raise RuntimeError(
            "Missing OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET environment variables.\n"
            "Set them (recommended) or pass them in explicitly."
        )

    r = requests.post(
        TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": UA},
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=timeout,
    )

    if r.status_code == 403:
        raise RuntimeError("403 from token endpoint.\n" + r.text[:800])

    r.raise_for_status()
    tok = r.json()
    access_token = tok.get("access_token")
    if not access_token:
        raise RuntimeError(f"Token response missing access_token: {tok}")
    return access_token


# =========================
# FETCH
# =========================
def fetch_states(token: str, bbox=None, timeout=30) -> dict:
    params = {}
    if bbox is not None:
        min_lat, max_lat, min_lon, max_lon = bbox
        params.update({"lamin": min_lat, "lamax": max_lat, "lomin": min_lon, "lomax": max_lon})

    r = requests.get(
        STATES_URL,
        headers={"Authorization": f"Bearer {token}", "User-Agent": UA},
        params=params,
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


# =========================
# DATA WRANGLING
# =========================
def normalize_icao24(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    return s.replace("0x", "").replace(" ", "")

def load_masterlist(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
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

def to_gdf_webmercator(df: pd.DataFrame) -> gpd.GeoDataFrame:
    d = df.dropna(subset=["longitude", "latitude"]).copy()
    gdf = gpd.GeoDataFrame(
        d,
        geometry=[Point(xy) for xy in zip(d["longitude"].astype(float), d["latitude"].astype(float))],
        crs="EPSG:4326",
    )
    return gdf.to_crs(epsg=3857)

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
    airborne_total = int(len(airborne))

    airborne["is_heli"] = [
        classify_airframe_is_heli(t, cs) for t, cs in zip(airborne.get("Type", ""), airborne.get("callsign", ""))
    ]
    helis = airborne[airborne["is_heli"]].copy()

    return {
        "matched_total": int(len(m)),
        "airborne_total": airborne_total,
        "agencies_airborne": agencies_airborne,
        "helicopters_airborne_total": int(len(helis)),
        "helicopters_by_agency": helis["Agency"].value_counts().to_dict() if not helis.empty else {},
        "sample_airborne": airborne[["Agency", "TailNumber", "icao24", "callsign", "Type", "latitude", "longitude", "alt_m", "velocity", "on_ground"]]
            .sort_values(["Agency", "TailNumber"]).head(25).to_dict(orient="records") if not airborne.empty else [],
    }


# =========================
# BACKGROUND LAYER HELPERS
# =========================
def _safe_read_vector(path: str) -> gpd.GeoDataFrame:
    g = gpd.read_file(path)
    if g.empty:
        return g
    if g.crs is None:
        g = g.set_crs("EPSG:4326", allow_override=True)
    return g.to_crs(epsg=3857)

def _plot_background_layers(ax, layers):
    for layer in (layers or []):
        p = layer.get("path")
        if not p:
            continue
        try:
            g = _safe_read_vector(p)
            if g is None or g.empty:
                continue

            edgecolor = layer.get("edgecolor", "black")
            facecolor = layer.get("facecolor", "none")
            alpha = float(layer.get("alpha", 0.7))
            linewidth = float(layer.get("linewidth", 1.0))
            zorder = int(layer.get("zorder", 5))

            g.plot(
                ax=ax,
                edgecolor=edgecolor,
                facecolor=facecolor,
                alpha=alpha,
                linewidth=linewidth,
                zorder=zorder,
            )
        except Exception as e:
            print(f"[WARN] Failed to draw background layer '{p}': {e}")


# =========================
# PLOT (ALWAYS BASEMAP + OPTIONAL BACKGROUND)
# =========================
def plot_snapshot_basemap(df: pd.DataFrame, title: str, bbox=None, save_png=None):
    fig, ax = plt.subplots(figsize=(12.5, 8.5))
    ax.set_facecolor("white")

    if bbox is not None:
        minx, maxx, miny, maxy = bbox_to_webmercator(bbox, pad_frac=0.06)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
    else:
        if df is None or len(df) == 0:
            ax.set_xlim(-1.45e7, -1.05e7)
            ax.set_ylim(3.7e6, 6.2e6)
        else:
            gtmp = to_gdf_webmercator(df)
            ax.set_xlim(gtmp.geometry.x.min(), gtmp.geometry.x.max())
            ax.set_ylim(gtmp.geometry.y.min(), gtmp.geometry.y.max())

    cx.add_basemap(ax, source=BASEMAP, zoom=BASEMAP_ZOOM, attribution_size=7)

    if BACKGROUND_LAYERS:
        _plot_background_layers(ax, BACKGROUND_LAYERS)

    if df is not None and len(df) > 0:
        gdf = to_gdf_webmercator(df)
        for agency, g in gdf.groupby("Agency"):
            speed = pd.to_numeric(g["velocity"], errors="coerce").fillna(0.0)
            sizes = (speed.clip(0, 220) / 220.0) * 110.0 + 28.0
            ax.scatter(
                g.geometry.x, g.geometry.y,
                s=sizes,
                label=f"{agency} (n={len(g)})",
                alpha=0.9,
                linewidths=0.8,
                edgecolors="white",
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

    if save_png:
        plt.savefig(save_png, dpi=160, bbox_inches="tight")
    plt.show()
    return fig


def fig_to_data_url(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="PNG", dpi=160, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return "data:image/png;base64," + b64



# =========================
# COST ESTIMATION (OpenAI)
# =========================

# Current token pricing (USD per 1M tokens) — update if you switch models.
# Source: OpenAI model pricing pages (GPT-4o: $2.50 in / $10.00 out per 1M tokens).
OPENAI_PRICING_PER_1M = {
    "gpt-4o": {"in": 2.50, "out": 10.00},
    # Optional: if you ever use these model names
    "gpt-4o-2024-08-06": {"in": 2.50, "out": 10.00},
    "chatgpt-4o-latest": {"in": 5.00, "out": 15.00},
}

def estimate_openai_cost_usd(model: str, usage_obj) -> dict:
    """
    Estimate cost from OpenAI Chat Completions response usage.
    Returns dict with tokens + estimated USD.
    """
    if usage_obj is None:
        return {"model": model, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated_usd": 0.0}

    # usage can be a pydantic object or dict-like; handle both
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

    rates = OPENAI_PRICING_PER_1M.get(model, None)
    if rates is None:
        # Unknown model pricing; report tokens but do not guess dollars.
        return {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_usd": None,
            "note": f"No pricing entry for model '{model}'. Add it to OPENAI_PRICING_PER_1M.",
        }

    est = (prompt_tokens / 1_000_000.0) * rates["in"] + (completion_tokens / 1_000_000.0) * rates["out"]
    return {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_usd": float(est),
    }


# =========================
# PATCH: in ask_snapshot_question()
# =========================
# Replace the end of your ask_snapshot_question() with this version of the API call + prints:

def ask_snapshot_question(snapshot_summary: dict, question: str, map_data_url: str = None) -> str:
    if (not OPENAI_API_KEY) or (OpenAI is None):
        airborne_total = snapshot_summary.get("airborne_total", 0)
        agencies = snapshot_summary.get("agencies_airborne", {}) or {}
        heli_total = snapshot_summary.get("helicopters_airborne_total", 0)
        heli_by_ag = snapshot_summary.get("helicopters_by_agency", {}) or {}
        return (
            f"Airborne aircraft in matched set: {airborne_total}\n"
            f"Agencies airborne: {agencies}\n"
            f"Helicopters airborne (best-effort from masterlist Type): {heli_total}\n"
            f"Helicopters by agency: {heli_by_ag}\n"
            f"(Set OPENAI_API_KEY + install openai to enable free-form Q&A.)"
        )

    client = OpenAI(api_key=OPENAI_API_KEY)

    sys_msg = (
        "You are an operational aviation analyst for wildfire response. "
        "Answer strictly using the provided snapshot summary (and optional map image). "
        "If helicopter identification is ambiguous, say so explicitly. "
        "In this dataset, masterlist Type values 'Type 1' and 'Type 2' indicate helicopters."
    )

    compact = {
        "matched_total": snapshot_summary.get("matched_total", 0),
        "airborne_total": snapshot_summary.get("airborne_total", 0),
        "agencies_airborne": snapshot_summary.get("agencies_airborne", {}),
        "helicopters_airborne_total": snapshot_summary.get("helicopters_airborne_total", 0),
        "helicopters_by_agency": snapshot_summary.get("helicopters_by_agency", {}),
        "sample_airborne": snapshot_summary.get("sample_airborne", [])[:25],
    }

    user_content = [{"type": "text", "text": f"Snapshot summary (JSON):\n{compact}\n\nQuestion: {question}"}]
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

    # ---- PRINT COST ESTIMATE
    usage = getattr(resp, "usage", None)
    cost = estimate_openai_cost_usd(OPENAI_MODEL, usage)
    print("\n=== OPENAI COST ESTIMATE ===")
    print(f"Model: {cost.get('model')}")
    print(f"prompt_tokens: {cost.get('prompt_tokens')} | completion_tokens: {cost.get('completion_tokens')} | total_tokens: {cost.get('total_tokens')}")
    if cost.get("estimated_usd") is None:
        print(f"estimated_usd: (unknown)  note: {cost.get('note')}")
    else:
        print(f"estimated_usd: ${cost.get('estimated_usd'):.6f}")

    return resp.choices[0].message.content.strip()





# =========================
# SNAPSHOT Q&A
# =========================
def ask_snapshot_question(snapshot_summary: dict, question: str, map_data_url: str = None) -> str:
    # Force visibility of why you're not seeing cost
    print(f"[DEBUG] OPENAI_API_KEY set? {bool(OPENAI_API_KEY)} | OpenAI import ok? {OpenAI is not None} | model={OPENAI_MODEL}")

    if (not OPENAI_API_KEY) or (OpenAI is None):
        airborne_total = snapshot_summary.get("airborne_total", 0)
        agencies = snapshot_summary.get("agencies_airborne", {}) or {}
        heli_total = snapshot_summary.get("helicopters_airborne_total", 0)
        heli_by_ag = snapshot_summary.get("helicopters_by_agency", {}) or {}
        return (
            f"Airborne aircraft in matched set: {airborne_total}\n"
            f"Agencies airborne: {agencies}\n"
            f"Helicopters airborne (best-effort from masterlist Type): {heli_total}\n"
            f"Helicopters by agency: {heli_by_ag}\n"
            f"(Set OPENAI_API_KEY + install openai to enable free-form Q&A.)"
        )

    client = OpenAI(api_key=OPENAI_API_KEY)

    sys_msg = (
        "You are an operational aviation analyst for wildfire response. "
        "Answer strictly using the provided snapshot summary (and optional map image). "
        "If helicopter identification is ambiguous, say so explicitly. "
        "In this dataset, masterlist Type values 'Type 1' and 'Type 2' indicate helicopters."
    )

    compact = {
        "matched_total": snapshot_summary.get("matched_total", 0),
        "airborne_total": snapshot_summary.get("airborne_total", 0),
        "agencies_airborne": snapshot_summary.get("agencies_airborne", {}),
        "helicopters_airborne_total": snapshot_summary.get("helicopters_airborne_total", 0),
        "helicopters_by_agency": snapshot_summary.get("helicopters_by_agency", {}),
        "sample_airborne": snapshot_summary.get("sample_airborne", [])[:25],
    }

    user_content = [{"type": "text", "text": f"Snapshot summary (JSON):\n{compact}\n\nQuestion: {question}"}]
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

    print("\n=== OPENAI COST ESTIMATE ===")
    print(f"Model: {cost.get('model')}")
    print(f"prompt_tokens: {cost.get('prompt_tokens')} | completion_tokens: {cost.get('completion_tokens')} | total_tokens: {cost.get('total_tokens')}")
    if cost.get("estimated_usd") is None:
        print(f"estimated_usd: (unknown)  note: {cost.get('note')}")
    else:
        print(f"estimated_usd: ${cost.get('estimated_usd'):.6f}")

    return resp.choices[0].message.content.strip()


# =========================
# MAIN
# =========================
def main(question: str = "How many helicopters and what agencies are currently in the air?"):
    t0 = time.time()

    master = load_masterlist(MASTER_XLSX)
    token = get_access_token(CLIENT_ID, CLIENT_SECRET)
    data = fetch_states(token, bbox=BBOX)

    states = states_to_df(data)
    matched = join_states_master(states, master)

    api_time = data.get("time")
    print(f"OpenSky api_time={api_time} | bbox={BBOX if BBOX else 'global'}")
    print(f"Masterlist aircraft: {len(master)} | Matched in snapshot: {len(matched)}")

    if matched.empty:
        print("\nCounts by Agency:\n(none)")
        print("\nSample rows:\n(none)")
    else:
        print("\nCounts by Agency:")
        print(matched["Agency"].value_counts(dropna=False).to_string())

        show_cols = ["Agency", "TailNumber", "icao24", "callsign", "Type", "latitude", "longitude", "alt_m", "velocity", "on_ground"]
        print("\nSample rows:")
        print(matched[show_cols].sort_values(["Agency", "TailNumber"]).head(25).to_string(index=False))

    title = f"OpenSky CURRENT states | WESTERN US | matched={len(matched)} | fetched_in={time.time()-t0:.2f}s"
    fig = plot_snapshot_basemap(matched, title=title, bbox=BBOX, save_png=None)

    snapshot_summary = summarize_snapshot(matched)
    map_url = fig_to_data_url(fig)

    answer = ask_snapshot_question(snapshot_summary, question=question, map_data_url=map_url)
    print("\n=== SNAPSHOT Q&A ===")
    print("Q:", question)
    print("A:", answer)


if __name__ == "__main__":
    main(question="How amny hilcopter are inthe ari and tell me the states they are in  ")
