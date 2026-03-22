import pandas as pd
from src.raingauge.utils import filter_uptime


def load_australia_raingauge_dataset(
    csv_path: str,
    metadata_path: str,
    uptime_threshold: float = 0.9,
) -> tuple:
    """
    Loads Australian raingauge dataset from a single combined CSV file.

    csv_path:      path to all_stations_rainfall_hourly_combined.csv
                   Expected columns: timestamp, station_id, rainfall_mm
    metadata_path: path to station metadata CSV
                   Expected columns: id, latitude, longitude
                   (Create this file with BOM station coordinates.
                    The 'id' column must match station_id values in csv_path.)
    uptime_threshold: fraction of non-NaN timesteps required to keep a station

    Returns
    -------
    formatted_gauge_df       : DataFrame [timestamp × station_id]
    station_metadata_df      : DataFrame with columns id, latitude, longitude, order
    """

    print(f"Loading Australian raingauge data from {csv_path}")
    gauge_df = pd.read_csv(csv_path)

    # Parse timestamp - format: "2021-01-01 00:00:00"
    gauge_df["timestamp"] = pd.to_datetime(gauge_df["timestamp"])

    # Pivot to [timestamp × station_id]
    # Note: data is already in mm/h - no unit conversion needed (unlike Singapore *12)
    formatted_gauge_df = gauge_df.pivot(
        index="timestamp", columns="station_id", values="rainfall_mm"
    )
    print(f"Dataframe shape: {formatted_gauge_df.shape}")

    # Filter stations by uptime threshold (reuse Singapore helper)
    if uptime_threshold:
        print(f"Filtering raingauge uptime. Threshold = {uptime_threshold}")
        filtered_stations_index = filter_uptime(
            formatted_gauge_df, uptime_threshold=uptime_threshold
        ).index
        formatted_gauge_df = formatted_gauge_df[filtered_stations_index]

    # Load station metadata (lat/lon coordinates)
    # File format: no header, columns = name, id, network, latitude, longitude
    print(f"Loading station metadata from {metadata_path}")
    station_metadata_df = pd.read_csv(
        metadata_path,
        header=None,
        names=["name", "id", "network", "latitude", "longitude"],
    )
    station_metadata_df["id"] = station_metadata_df["id"].astype(int)
    station_metadata_df["latitude"] = station_metadata_df["latitude"].astype(float)
    station_metadata_df["longitude"] = station_metadata_df["longitude"].astype(float)

    # Keep only stations present in the (filtered) rainfall data
    station_metadata_df = station_metadata_df[
        station_metadata_df["id"].isin(formatted_gauge_df.columns)
    ].copy()

    print(f"Mapping df shape: {station_metadata_df.shape}")
    station_metadata_df["order"] = range(station_metadata_df.shape[0])
    station_metadata_df.reset_index(inplace=True, drop=True)

    return formatted_gauge_df, station_metadata_df
