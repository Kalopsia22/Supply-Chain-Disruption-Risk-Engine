"""
Ties the shipment dataset to REAL historical weather (not the synthetic
Weather_Severity_Index) via Open-Meteo's Historical Weather API
(archive-api.open-meteo.com/v1/archive — ERA5/IFS reanalysis, 1940-present,
free, no API key).

Strategy: one API call per port (11 total), each covering that port's full
Order_Date range in the dataset, requesting DAILY aggregates. This is far
more efficient than one call per shipment (10,000 calls) and stays well
within the free tier (10,000 calls/day) — 11 calls total for the whole
dataset. Results are merged back onto each shipment by (Origin_City, date).

NOTE ON TESTING: this module was built and merge/correlation logic was
verified against mocked API responses shaped exactly like Open-Meteo's
documented JSON schema, because the dev sandbox this was built in cannot
reach open-meteo.com (restricted egress — confirmed via a direct curl test
that returned 403 from the sandbox's own proxy, not from Open-Meteo).
Streamlit Cloud has open internet access, so the live fetch should work
there — run it from the dashboard's Historical Validation tab and check
the result there first.
"""
import requests
import pandas as pd
import numpy as np

from data_utils import CITY_COORDS

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
REQUEST_TIMEOUT = 15


def fetch_port_historical_daily(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    """Fetch daily wind/precipitation aggregates for one port over a date range.
    Returns {'ok': False, 'error': ...} on failure so callers can degrade gracefully."""
    try:
        resp = requests.get(
            ARCHIVE_URL,
            params={
                "latitude": lat, "longitude": lon,
                "start_date": start_date, "end_date": end_date,
                "daily": "wind_speed_10m_max,wind_gusts_10m_max,precipitation_sum",
                "timezone": "UTC",
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        return {"ok": False, "error": str(e)}

    try:
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        if not dates:
            return {"ok": False, "error": "Empty response (no daily data returned)."}

        df = pd.DataFrame({
            "date": pd.to_datetime(dates).date,
            "real_wind_speed_max_kmh": daily.get("wind_speed_10m_max", [None] * len(dates)),
            "real_wind_gusts_max_kmh": daily.get("wind_gusts_10m_max", [None] * len(dates)),
            "real_precipitation_sum_mm": daily.get("precipitation_sum", [None] * len(dates)),
        })
        return {"ok": True, "df": df}
    except (KeyError, IndexError, TypeError, ValueError) as e:
        return {"ok": False, "error": f"Unexpected response shape: {e}"}


def fetch_all_ports_historical_batched(city_coords: dict, start_date: str, end_date: str) -> dict:
    """Fetch historical daily weather for ALL given ports in a SINGLE batched
    HTTP request (one shared date range across all ports), using Open-Meteo's
    multi-location comma-separated coordinate support — 1 request total instead
    of 1-per-port. Falls back gracefully per-port if the batch call fails."""
    cities = list(city_coords.keys())
    lats = ",".join(str(city_coords[c][0]) for c in cities)
    lons = ",".join(str(city_coords[c][1]) for c in cities)

    try:
        resp = requests.get(
            ARCHIVE_URL,
            params={
                "latitude": lats, "longitude": lons,
                "start_date": start_date, "end_date": end_date,
                "daily": "wind_speed_10m_max,wind_gusts_10m_max,precipitation_sum",
                "timezone": "UTC",
            },
            timeout=30,
        )
        if resp.status_code == 429:
            return {c: {"ok": False, "error": "Rate limited by Open-Meteo (429) — this is a shared free-tier limit, wait a minute and retry."} for c in cities}
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        return {c: {"ok": False, "error": str(e)} for c in cities}

    # Same single-vs-list normalization as the live weather module.
    data_list = data if isinstance(data, list) else [data]

    results = {}
    for i, city in enumerate(cities):
        try:
            daily = data_list[i].get("daily", {})
            dates = daily.get("time", [])
            if not dates:
                results[city] = {"ok": False, "error": "Empty response (no daily data returned)."}
                continue
            df = pd.DataFrame({
                "date": pd.to_datetime(dates).date,
                "real_wind_speed_max_kmh": daily.get("wind_speed_10m_max", [None] * len(dates)),
                "real_wind_gusts_max_kmh": daily.get("wind_gusts_10m_max", [None] * len(dates)),
                "real_precipitation_sum_mm": daily.get("precipitation_sum", [None] * len(dates)),
            })
            results[city] = {"ok": True, "df": df}
        except (IndexError, KeyError, TypeError, ValueError) as e:
            results[city] = {"ok": False, "error": f"Could not parse response for {city}: {e}"}
    return results


def fetch_all_ports_historical(order_dates_by_city: dict, progress_callback=None) -> dict:
    """order_dates_by_city: {city_name: (min_date_str, max_date_str)}.
    Returns {city_name: fetch_result_dict} for every port, one API call each.
    Prefer fetch_all_ports_historical_batched() when all ports can share a
    single date range — that's 1 request total instead of N."""
    results = {}
    cities = list(order_dates_by_city.keys())
    for i, city in enumerate(cities):
        if city not in CITY_COORDS:
            results[city] = {"ok": False, "error": f"No coordinates on file for {city}"}
            continue
        lat, lon = CITY_COORDS[city]
        start_date, end_date = order_dates_by_city[city]
        results[city] = fetch_port_historical_daily(lat, lon, start_date, end_date)
        if progress_callback:
            progress_callback(i + 1, len(cities), city)
    return results


def merge_historical_weather(df: pd.DataFrame, fetch_results: dict) -> pd.DataFrame:
    """Merge fetched real historical weather onto each shipment by
    (Origin_City, Order_Date). Shipments whose port fetch failed get NaN."""
    df = df.copy()
    df["Order_Date_only"] = pd.to_datetime(df["Order_Date"]).dt.date

    merged_frames = []
    for city, result in fetch_results.items():
        city_df = df[df["Origin_City"] == city].copy()
        if not result.get("ok"):
            city_df["real_wind_speed_max_kmh"] = np.nan
            city_df["real_wind_gusts_max_kmh"] = np.nan
            city_df["real_precipitation_sum_mm"] = np.nan
        else:
            weather_df = result["df"]
            city_df = city_df.merge(
                weather_df, left_on="Order_Date_only", right_on="date", how="left"
            ).drop(columns=["date"])
        merged_frames.append(city_df)

    out = pd.concat(merged_frames, ignore_index=True) if merged_frames else df
    return out.drop(columns=["Order_Date_only"], errors="ignore")


def compare_ground_truth_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Compare the synthetic Weather_Severity_Index against real historical
    weather variables, both correlated with the actual delay outcome."""
    rows = []
    rows.append({
        "feature": "Weather_Severity_Index (synthetic)",
        "correlation_with_delay": df["Weather_Severity_Index"].corr(df["Is_Delayed"]),
        "n_shipments": df["Weather_Severity_Index"].notna().sum(),
    })
    for col, label in [
        ("real_wind_speed_max_kmh", "Real Wind Speed Max (Open-Meteo, historical)"),
        ("real_wind_gusts_max_kmh", "Real Wind Gusts Max (Open-Meteo, historical)"),
        ("real_precipitation_sum_mm", "Real Precipitation Sum (Open-Meteo, historical)"),
    ]:
        if col in df.columns:
            rows.append({
                "feature": label,
                "correlation_with_delay": df[col].corr(df["Is_Delayed"]),
                "n_shipments": df[col].notna().sum(),
            })
    return pd.DataFrame(rows)
