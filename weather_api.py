"""
Live port conditions via Open-Meteo — free, no API key required.
https://open-meteo.com/en/docs (standard forecast) and
https://open-meteo.com/en/docs/marine-weather-api (marine/wave forecast).

NOTE ON TESTING: this module was built and logic-tested against Open-Meteo's
documented request/response schema, but the dev sandbox this was built in
cannot reach open-meteo.com (restricted egress), so the live HTTP round-trip
itself could not be executed end-to-end before deployment. Streamlit Cloud
has open internet access, so this should work once deployed — the error
handling below is deliberately defensive in case a call fails or times out.
"""
import requests
import numpy as np
import pandas as pd

MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT = 8  # seconds


def fetch_port_conditions(lat: float, lon: float) -> dict:
    """Fetch current + 7-day hourly marine and wind conditions for one port.
    Returns a dict with 'ok': False and an 'error' message on failure, so the
    UI can degrade gracefully instead of crashing the whole page."""
    try:
        marine_resp = requests.get(
            MARINE_URL,
            params={
                "latitude": lat, "longitude": lon,
                "hourly": "wave_height,wave_period,swell_wave_height",
                "timezone": "UTC",
            },
            timeout=REQUEST_TIMEOUT,
        )
        marine_resp.raise_for_status()
        marine = marine_resp.json()

        wind_resp = requests.get(
            FORECAST_URL,
            params={
                "latitude": lat, "longitude": lon,
                "hourly": "wind_speed_10m,precipitation",
                "current": "wind_speed_10m,precipitation",
                "timezone": "UTC",
            },
            timeout=REQUEST_TIMEOUT,
        )
        wind_resp.raise_for_status()
        wind = wind_resp.json()

    except requests.exceptions.RequestException as e:
        return {"ok": False, "error": str(e)}

    try:
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
    except (KeyError, IndexError, TypeError) as e:
        return {"ok": False, "error": f"Unexpected response shape from Open-Meteo: {e}"}


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
