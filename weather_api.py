"""
Live port conditions via Open-Meteo — free, no API key required.
https://open-meteo.com/en/docs (standard forecast) and
https://open-meteo.com/en/docs/marine-weather-api (marine/wave forecast).

IMPORTANT: fetches ALL ports in a single batched request per endpoint (Open-Meteo
supports up to 1000 comma-separated locations per call), rather than one request
per port. The original per-port implementation made 2 HTTP requests every time a
port was selected or auto-refresh fired — with 11 ports that's up to 22 requests
per refresh cycle, which is what triggered 429 Too Many Requests in practice.
Batching cuts this to exactly 2 requests total, regardless of port count.

NOTE ON TESTING: this module was built and logic-tested against Open-Meteo's
documented multi-location request/response schema, but the dev sandbox this was
built in cannot reach open-meteo.com (restricted egress — confirmed via direct
curl returning 403 from the sandbox's own proxy), so the live HTTP round-trip
itself could not be executed end-to-end here. Streamlit Cloud has open internet
access, where this was confirmed reachable (the reported 429 proves the earlier
per-port version *did* reach Open-Meteo successfully — it was just calling it
too often).
"""
import time
import requests
import numpy as np
import pandas as pd

MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT = 12
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 2


def _get_with_retry(url: str, params: dict) -> requests.Response:
    """GET with exponential backoff specifically for 429 responses. Honors a
    Retry-After header if Open-Meteo sends one, otherwise backs off 2s/4s/8s."""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                if attempt < MAX_RETRIES - 1:
                    wait = float(resp.headers.get("Retry-After", BACKOFF_BASE_SECONDS * (2 ** attempt)))
                    time.sleep(min(wait, 20))
                    continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt < MAX_RETRIES - 1 and getattr(e.response, "status_code", None) == 429:
                time.sleep(BACKOFF_BASE_SECONDS * (2 ** attempt))
                continue
            raise
    if last_exc:
        raise last_exc


def fetch_all_ports_conditions(city_coords: dict) -> dict:
    """Fetch current + 7-day hourly marine and wind conditions for ALL ports in
    exactly 2 HTTP requests (one marine, one forecast), using Open-Meteo's
    multi-location comma-separated coordinate support.

    city_coords: {city_name: (lat, lon)}
    Returns: {city_name: per_port_result_dict}, where each per_port_result_dict
    matches the same shape as the old single-port fetch (so the UI code doesn't
    need to change), with 'ok': False and an 'error' message for any port whose
    parsing failed even if the batched HTTP call itself succeeded.
    """
    cities = list(city_coords.keys())
    lats = ",".join(str(city_coords[c][0]) for c in cities)
    lons = ",".join(str(city_coords[c][1]) for c in cities)

    try:
        marine_resp = _get_with_retry(MARINE_URL, {
            "latitude": lats, "longitude": lons,
            "hourly": "wave_height,wave_period,swell_wave_height",
            "timezone": "UTC",
        })
        marine_data = marine_resp.json()

        wind_resp = _get_with_retry(FORECAST_URL, {
            "latitude": lats, "longitude": lons,
            "hourly": "wind_speed_10m,precipitation",
            "current": "wind_speed_10m,precipitation",
            "timezone": "UTC",
        })
        wind_data = wind_resp.json()

    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status == 429:
            msg = (
                "Rate limited by Open-Meteo (429) even after retrying. This is a shared free-tier "
                "limit, not a permissions problem — wait a minute before fetching again, and avoid "
                "setting auto-refresh below 5 minutes if this keeps happening."
            )
        else:
            msg = str(e)
        return {c: {"ok": False, "error": msg} for c in cities}
    except requests.exceptions.RequestException as e:
        return {c: {"ok": False, "error": str(e)} for c in cities}

    # Open-Meteo returns a plain object (not a list) if only ONE location was
    # requested, and a list of objects for multiple locations. Normalize both.
    marine_list = marine_data if isinstance(marine_data, list) else [marine_data]
    wind_list = wind_data if isinstance(wind_data, list) else [wind_data]

    results = {}
    for i, city in enumerate(cities):
        try:
            marine = marine_list[i]
            wind = wind_list[i]
            results[city] = _parse_single_port(marine, wind)
        except (IndexError, KeyError, TypeError) as e:
            results[city] = {"ok": False, "error": f"Could not parse response for {city}: {e}"}
    return results


def _parse_single_port(marine: dict, wind: dict) -> dict:
    hourly_time = marine.get("hourly", {}).get("time", [])
    wave_height = marine.get("hourly", {}).get("wave_height", [])
    wave_period = marine.get("hourly", {}).get("wave_period", [])
    swell_height = marine.get("hourly", {}).get("swell_wave_height", [])
    wind_speed_hourly = wind.get("hourly", {}).get("wind_speed_10m", [])
    precip_hourly = wind.get("hourly", {}).get("precipitation", [])

    if not hourly_time or not wave_height:
        return {"ok": False, "error": "Empty response from Open-Meteo (no hourly data returned)."}

    current_wind = wind.get("current", {}).get("wind_speed_10m")
    current_precip = wind.get("current", {}).get("precipitation")

    df = pd.DataFrame({
        "time": pd.to_datetime(hourly_time),
        "wave_height_m": wave_height,
        "wave_period_s": wave_period if len(wave_period) == len(hourly_time) else [None] * len(hourly_time),
        "swell_height_m": swell_height if len(swell_height) == len(hourly_time) else [None] * len(hourly_time),
        "wind_speed_kmh": wind_speed_hourly if len(wind_speed_hourly) == len(hourly_time) else [None] * len(hourly_time),
        "precipitation_mm": precip_hourly if len(precip_hourly) == len(hourly_time) else [None] * len(hourly_time),
    })

    current_wave = df["wave_height_m"].iloc[0] if len(df) else None
    next_72h = df.head(72)
    peak_wave_72h = next_72h["wave_height_m"].max() if len(next_72h) else None
    peak_wind_72h = next_72h["wind_speed_kmh"].max() if len(next_72h) else None

    return {
        "ok": True,
        "df": df,
        "current_wave_height_m": current_wave,
        "current_wind_kmh": current_wind,
        "current_precip_mm": current_precip,
        "peak_wave_72h_m": peak_wave_72h,
        "peak_wind_72h_kmh": peak_wind_72h,
    }


def fetch_port_conditions(lat: float, lon: float) -> dict:
    """Single-port fetch, kept for backward compatibility / one-off use.
    Prefer fetch_all_ports_conditions() when displaying multiple ports, since
    that batches into 2 requests instead of 2-per-port."""
    result = fetch_all_ports_conditions({"_single": (lat, lon)})
    return result["_single"]


# Operational thresholds loosely based on standard maritime wave-height / wind
# hazard bands (Douglas sea state / Beaufort scale) — used to translate raw
# live readings into a simple traffic-light flag for non-technical users.
def classify_conditions(wave_height_m, wind_kmh) -> dict:
    if wave_height_m is None:
        return {"level": "Unknown", "color": "#8b93a7"}

    if wave_height_m >= 4.0 or (wind_kmh is not None and wind_kmh >= 62):
        return {"level": "Rough — Elevated Risk", "color": "#ef4444"}
    elif wave_height_m >= 2.5 or (wind_kmh is not None and wind_kmh >= 39):
        return {"level": "Moderate", "color": "#f5b942"}
    else:
        return {"level": "Calm", "color": "#2dd4bf"}
