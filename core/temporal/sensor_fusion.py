
import os
import sys

import numpy as np
import pandas as pd

SYSTEM_FILE = os.path.join("data", "system_telemetry.csv")
SENSOR_FILE = os.path.join("data", "physical_sensors.csv")
OUTPUT_FILE = os.path.join("data", "fused_telemetry.csv")

MATCH_TOLERANCE_SECONDS = 2


def parse_local_timestamp(value):
    """Normalize timestamps to local wall time for matching.

    The current computer collector uses naive local timestamps,
    while the simulator writes timestamps with a timezone offset.
    Preserve the local clock time so both formats can be matched.
    """
    if pd.isna(value):
        return pd.NaT

    try:
        stamp = pd.Timestamp(value)

        if stamp.tzinfo is not None:
            stamp = stamp.tz_localize(None)

        return stamp
    except (ValueError, TypeError, OverflowError):
        return pd.NaT


def load_csv(path):
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)

    if df.empty:
        print(f"ERROR: File contains no rows: {path}")
        sys.exit(1)

    return df


def main():
    print("\nPHYSOS Sensor Fusion Engine")
    print("=" * 42)

    system = load_csv(SYSTEM_FILE)
    sensors = load_csv(SENSOR_FILE)

    # Support both known frequency column names.
    if (
        "cpu_frequency" not in system.columns
        and "cpu_frequency_mhz" in system.columns
    ):
        system = system.rename(
            columns={"cpu_frequency_mhz": "cpu_frequency"}
        )

    system_required = [
        "timestamp",
        "cpu_percent",
        "ram_percent",
        "cpu_frequency",
        "disk_percent",
        "bytes_sent",
        "bytes_received",
    ]

    sensor_required = [
        "timestamp",
        "temperature_c",
        "humidity_percent",
    ]

    missing_system = [
        c for c in system_required if c not in system.columns
    ]
    missing_sensor = [
        c for c in sensor_required if c not in sensors.columns
    ]

    if missing_system or missing_sensor:
        print("ERROR: Required columns are missing.")
        print("System columns missing:", missing_system)
        print("Sensor columns missing:", missing_sensor)
        sys.exit(1)

    # Normalize timestamps before joining.
    system["timestamp"] = system["timestamp"].map(
        parse_local_timestamp
    )
    sensors["timestamp"] = sensors["timestamp"].map(
        parse_local_timestamp
    )

    # Convert measurement columns to numeric.
    system_numeric = [
        "cpu_percent",
        "ram_percent",
        "cpu_frequency",
        "disk_percent",
        "bytes_sent",
        "bytes_received",
    ]

    sensor_numeric = [
        "temperature_c",
        "humidity_percent",
    ]

    for column in system_numeric:
        system[column] = pd.to_numeric(
            system[column], errors="coerce"
        )

    for column in sensor_numeric:
        sensors[column] = pd.to_numeric(
            sensors[column], errors="coerce"
        )

    system = (
        system.dropna(subset=["timestamp"] + system_numeric)
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )

    sensors = (
        sensors.dropna(subset=["timestamp"] + sensor_numeric)
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )

    if system.empty or sensors.empty:
        print("ERROR: No valid system or sensor readings remain.")
        sys.exit(1)

    # Keep sensor provenance so simulated data cannot be mistaken
    # for physical measurements.
    if "source" not in sensors.columns:
        sensors["source"] = "UNKNOWN"

    system = system.rename(
        columns={"timestamp": "system_timestamp"}
    )
    sensors = sensors.rename(
        columns={"timestamp": "sensor_timestamp"}
    )

    # Both timestamps are now sorted local wall-clock timestamps.
    # Only pair measurements taken close together.
    fused = pd.merge_asof(
        system.sort_values("system_timestamp"),
        sensors.sort_values("sensor_timestamp"),
        left_on="system_timestamp",
        right_on="sensor_timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(
            seconds=MATCH_TOLERANCE_SECONDS
        ),
    )

    fused["time_difference_seconds"] = (
        fused["system_timestamp"] - fused["sensor_timestamp"]
    ).dt.total_seconds().abs()

    # Add a short rolling relationship estimate.
    fused["cpu_temperature_rolling_corr"] = (
        fused["cpu_percent"]
        .rolling(window=10, min_periods=10)
        .corr(fused["temperature_c"])
    )

    fused["ram_temperature_rolling_corr"] = (
        fused["ram_percent"]
        .rolling(window=10, min_periods=10)
        .corr(fused["temperature_c"])
    )

    os.makedirs("data", exist_ok=True)
    fused.to_csv(OUTPUT_FILE, index=False)

    matched = fused["temperature_c"].notna()
    matched_count = int(matched.sum())
    total_count = len(fused)

    print(f"System readings: {len(system)}")
    print(f"Sensor readings: {len(sensors)}")
    print(f"System readings matched to a sensor: {matched_count}/{total_count}")
    print(f"Output saved to: {OUTPUT_FILE}")

    if matched_count:
        print("\nLatest fused readings:")
        print(
            fused.loc[
                matched,
                [
                    "system_timestamp",
                    "cpu_percent",
                    "ram_percent",
                    "temperature_c",
                    "humidity_percent",
                    "source",
                    "time_difference_seconds",
                ],
            ].tail(10).to_string(index=False)
        )

        if matched_count >= 2:
            matched_data = fused.loc[
                matched, ["cpu_percent", "temperature_c"]
            ].dropna()

            if len(matched_data) >= 2:
                correlation = matched_data.corr().iloc[0, 1]
                print(
                    "\nOverall CPU-temperature correlation: "
                    f"{correlation:.3f}"
                )
                print(
                    "This is descriptive only; it does not establish "
                    "causation or predict failure."
                )
    else:
        print(
            "\nNo timestamps matched within the tolerance. "
            "Run the collector and sensor simulator at the same time, "
            "then try again."
        )


if __name__ == "__main__":
    main()
