"""
build_dataset.py
----------------
Full pipeline for building the rain gauge dataset from the BOM WaterData API.

Usage from a notebook:
    from build_dataset import build_dataset

    df = build_dataset(
        station_ids=STATION_IDS,
        start_date="2021-01-01",
        end_date="2025-12-31",
        output_dir="database/australia/raw",
        combined_output="all_stations_rainfall_hourly_combined.csv",
    )
"""

import time
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from typing import Optional

# ── station IDs ───────────────────────────────────────────────
STATION_IDS = [
    # fill in station IDs
]

# ── Config ────────────────────────────────────────────────────────────────────
BOM_URL    = "https://www.bom.gov.au/waterdata/services"
NS         = {"wml2": "http://www.opengis.net/waterml/2.0"}
SLEEP_SECS = 2   # DO NOT REMOVE — prevents triggering BOM rate limiting


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: fetch + parse one station from the BOM API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_station(station_id: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Download rainfall XML from BOM WaterData API and parse to a DataFrame.

    Returns a DataFrame with columns [timestamp, rainfall_mm],
    or None if the request fails or the response contains no data.
    """
    params = {
        "service":          "SOS",
        "version":          "2.0",
        "request":          "GetObservation",
        "observedProperty": "http://bom.gov.au/waterdata/services/parameters/Rainfall",
        "featureOfInterest": f"http://bom.gov.au/waterdata/services/stations/{station_id}",
        "temporalFilter":   f"om:phenomenonTime,{start_date}/{end_date}",
    }

    try:
        response = requests.get(BOM_URL, params=params, timeout=60)
    except requests.RequestException as e:
        print(f"  [!] Network error: {e}")
        return None

    if response.status_code != 200:
        print(f"  [!] HTTP {response.status_code}")
        return None

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as e:
        print(f"  [!] XML parse error: {e}")
        return None

    # Surface any exception message from the API
    exception = root.find(".//{http://www.opengis.net/ows/1.1}ExceptionText")
    if exception is not None:
        print(f"  [!] API error: {exception.text.strip()}")
        return None

    times, values = [], []
    for point in root.findall(".//wml2:MeasurementTVP", NS):
        t = point.find("wml2:time",  NS)
        v = point.find("wml2:value", NS)
        if t is not None and v is not None:
            times.append(t.text)
            values.append(v.text)

    if not times:
        print(f"  [!] No data returned (station may not have rainfall observations)")
        return None

    df = pd.DataFrame({"timestamp": times, "rainfall_mm": values})
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: resample one station's raw data to hourly
# ─────────────────────────────────────────────────────────────────────────────

def resample_to_hourly(df: pd.DataFrame, hourly_range: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Convert raw sub-hourly BOM data to a complete hourly time series.
    Missing hours are filled with 0.
    """
    df = df.copy()
    df["timestamp"]   = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"]   = df["timestamp"].dt.tz_convert("Australia/Sydney").dt.tz_localize(None)
    df["rainfall_mm"] = pd.to_numeric(df["rainfall_mm"], errors="coerce")

    # Keep only legitimate tipping bucket readings (0.0 = no rain, 0.5 = one tip).
    # Everything else (7.0, 14.0, 164.0, 1208.5 etc.) are sub-daily/period aggregates
    # injected by BOM at reporting boundaries and must be excluded before resampling.
    df = df[df["rainfall_mm"].isin([0.0, 0.5])]

    df.set_index("timestamp", inplace=True)
    hourly = df.resample("h").sum()
    hourly = hourly.reindex(hourly_range, fill_value=0)

    # Mark hours with physically impossible totals as NaN (sensor malfunction).
    # A real 0.5 mm tipping bucket cannot tip faster than once every ~6 seconds,
    # so any hourly sum above 300 tips × 0.5 mm = 150 mm implies a stuck sensor.
    # NaN preserves the distinction between "no rain" (0.0) and "bad data" (NaN).
    hourly.loc[hourly["rainfall_mm"] > 150.0, "rainfall_mm"] = float("nan")

    hourly.reset_index(inplace=True)
    hourly.columns = ["timestamp", "rainfall_mm"]
    return hourly


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: run the full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    station_ids: list,
    start_date: str = "2021-01-01",
    end_date:   str = "2025-12-31",
    output_dir: str = "database/australia/raw",
    combined_output: str = "all_stations_rainfall_hourly_combined.csv",
) -> pd.DataFrame:
    """
    Fetch, parse, resample, and combine rainfall data for all stations.

    Saves each station's hourly CSV to output_dir/{station_id}_rainfall_hourly.csv.
    Saves the combined CSV to combined_output.
    Returns the combined DataFrame.

    Parameters
    ----------
    station_ids     : list of BOM station ID strings
    start_date      : "YYYY-MM-DD"
    end_date        : "YYYY-MM-DD"
    output_dir      : directory for per-station intermediate CSVs
    combined_output : path for the final combined CSV
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hourly_range = pd.date_range(
        start=f"{start_date} 00:00:00",
        end=f"{end_date} 23:00:00",
        freq="h",
    )

    # Deduplicate station IDs — duplicates cause the same data to be added twice
    original_count = len(station_ids)
    station_ids = list(dict.fromkeys(station_ids))
    if len(station_ids) < original_count:
        print(f"⚠  Removed {original_count - len(station_ids)} duplicate station IDs")

    # Threshold above which an hourly total is treated as a sensor malfunction.
    # Matches the NaN logic in resample_to_hourly() — keeps cache loading consistent.
    MAX_HOURLY_MM = 150

    all_data   = []
    successful = failed = skipped = 0
    n = len(station_ids)

    print(f"Fetching {n} stations  |  {start_date} → {end_date}")
    print("=" * 70)

    for i, station in enumerate(station_ids, 1):
        out_path = out_dir / f"{station}_rainfall_hourly.csv"

        # Resume: skip stations already downloaded
        if out_path.exists():
            print(f"[{i:>3}/{n}] {station}  — already downloaded, loading from disk")
            hourly_df = pd.read_csv(out_path)
            hourly_df["station_id"] = station
            # NaN any values that exceed the physical maximum (sensor malfunction guard
            # for cached files built by older versions of this script)
            hourly_df.loc[hourly_df["rainfall_mm"] > MAX_HOURLY_MM, "rainfall_mm"] = float("nan")
            all_data.append(hourly_df[["timestamp", "station_id", "rainfall_mm"]])
            skipped += 1
            continue

        print(f"[{i:>3}/{n}] {station}  ", end="", flush=True)

        raw_df = fetch_station(station, start_date, end_date)
        if raw_df is None:
            failed += 1
            time.sleep(SLEEP_SECS)
            continue

        hourly_df = resample_to_hourly(raw_df, hourly_range)
        hourly_df["station_id"] = station
        hourly_df = hourly_df[["timestamp", "station_id", "rainfall_mm"]]

        hourly_df.to_csv(out_path, index=False)

        rainy_hours = (hourly_df["rainfall_mm"] > 0).sum()
        total_mm    = hourly_df["rainfall_mm"].sum()
        print(f"✓  {len(raw_df):>6} records → {rainy_hours} rainy hours  |  {total_mm:.1f} mm")

        all_data.append(hourly_df)
        successful += 1
        time.sleep(SLEEP_SECS)

    print("=" * 70)
    print(f"Done — successful: {successful}  |  skipped (cached): {skipped}  |  failed: {failed}")

    if not all_data:
        print("[!] No data collected — returning empty DataFrame")
        return pd.DataFrame(columns=["timestamp", "station_id", "rainfall_mm"])

    combined = pd.concat(all_data, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    combined.sort_values(["timestamp", "station_id"], inplace=True)
    combined.to_csv(combined_output, index=False)

    print(f"\nCombined CSV saved → {combined_output}")
    print(f"  Rows      : {len(combined):,}")
    print(f"  Stations  : {combined['station_id'].nunique()}")
    print(f"  Date range: {combined['timestamp'].min()}  →  {combined['timestamp'].max()}")

    return combined
