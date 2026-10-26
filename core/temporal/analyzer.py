
import os
import sys
import pandas as pd
import numpy as np

INPUT_FILE = os.path.join("data", "system_telemetry.csv")
OUTPUT_FILE = os.path.join("data", "temporal_analysis.csv")

# At 1 reading per second, this is a 10-second window.
WINDOW_SIZE = 10
ANOMALY_THRESHOLD = 3.5

METRICS = [
    "cpu_percent",
    "ram_percent",
    "cpu_frequency",
    "disk_percent",
    "bytes_sent",
    "bytes_received",
]

def robust_z_score(series):
    """Estimate how unusual each value is using a rolling median and MAD."""
    rolling_median = series.rolling(
        WINDOW_SIZE, min_periods=WINDOW_SIZE
    ).median()

    rolling_mad = series.rolling(
        WINDOW_SIZE, min_periods=WINDOW_SIZE
    ).apply(
        lambda values: np.median(np.abs(values - np.median(values))),
        raw=True,
    )

    # MAD is robust to extreme values. Fall back to rolling std
    # when MAD is zero or too small.
    robust_scale = 1.4826 * rolling_mad

    rolling_std = series.rolling(
        WINDOW_SIZE, min_periods=WINDOW_SIZE
    ).std()

    scale = robust_scale.where(robust_scale > 1e-9, rolling_std)
    score = (series - rolling_median) / scale.where(scale > 1e-9)

    return score.replace([np.inf, -np.inf], np.nan)


def analyze():
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        print("Run the telemetry collector first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE)

    # Normalize the CPU frequency column name across collector versions.
    if "cpu_frequency" not in df.columns and "cpu_frequency_mhz" in df.columns:
        df = df.rename(columns={"cpu_frequency_mhz": "cpu_frequency"})

    if df.empty:
        print("ERROR: Telemetry CSV contains no readings.")
        sys.exit(1)

    required = ["timestamp"] + METRICS
    missing = [column for column in required if column not in df.columns]

    if missing:
        print("ERROR: Missing required CSV columns:", missing)
        print("Check that the collector and analyzer use matching column names.")
        sys.exit(1)

    # Convert columns to the expected data types.
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    df = (
        df.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )

    for metric in METRICS:
        df[metric] = df[metric].replace([np.inf, -np.inf], np.nan)
        df[metric] = df[metric].ffill()

    df = df.dropna(subset=METRICS).reset_index(drop=True)

    if len(df) < WINDOW_SIZE:
        print(
            f"Collect at least {WINDOW_SIZE} valid readings. "
            f"Currently available: {len(df)}."
        )
        sys.exit(1)

    print("\nPHYSOS Temporal Analysis Engine")
    print("=" * 42)
    print(f"Valid readings: {len(df)}")
    print(f"Window size: {WINDOW_SIZE} readings")
    print("Analyzing trends and relationships...\n")

    # Rolling trends and changes between consecutive readings.
    for metric in METRICS:
        df[f"{metric}_rolling_mean"] = (
            df[metric].rolling(WINDOW_SIZE, min_periods=WINDOW_SIZE).mean()
        )

        df[f"{metric}_rolling_std"] = (
            df[metric].rolling(WINDOW_SIZE, min_periods=WINDOW_SIZE).std()
        )

        df[f"{metric}_change"] = df[metric].diff()

        df[f"{metric}_anomaly_score"] = robust_z_score(df[metric])

    # A reading is flagged if any metric deviates strongly from its
    # recent baseline. The initial window has no anomaly decision.
    score_columns = [f"{metric}_anomaly_score" for metric in METRICS]
    df["max_abs_anomaly_score"] = df[score_columns].abs().max(axis=1)

    df["anomaly_flag"] = (
        df["max_abs_anomaly_score"] >= ANOMALY_THRESHOLD
    ).fillna(False)

    df["anomaly_metrics"] = df.apply(
        lambda row: ", ".join(
            metric
            for metric in METRICS
            if pd.notna(row[f"{metric}_anomaly_score"])
            and abs(row[f"{metric}_anomaly_score"]) >= ANOMALY_THRESHOLD
        ),
        axis=1,
    )

    # Rolling correlations reveal whether two signals have been moving
    # together. Correlation is descriptive, not proof of causation.
    df["cpu_ram_rolling_corr"] = (
        df["cpu_percent"].rolling(WINDOW_SIZE).corr(df["ram_percent"])
    )

    df["cpu_frequency_rolling_corr"] = (
        df["cpu_percent"].rolling(WINDOW_SIZE).corr(df["cpu_frequency"])
    )

    df["cpu_network_rolling_corr"] = (
        df["cpu_percent"].rolling(WINDOW_SIZE).corr(df["bytes_sent"])
    )

    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    analyzed = df["max_abs_anomaly_score"].notna()
    anomaly_count = int(df.loc[analyzed, "anomaly_flag"].sum())
    analyzed_count = int(analyzed.sum())

    print("Analysis complete.")
    print(f"Output saved to: {OUTPUT_FILE}")
    print(f"Readings evaluated after warm-up: {analyzed_count}")
    print(f"Readings flagged as unusual: {anomaly_count}")

    print("\nLatest system readings:")
    print(
        df[[
            "timestamp",
            "cpu_percent",
            "ram_percent",
            "cpu_frequency",
            "max_abs_anomaly_score",
            "anomaly_flag",
        ]].tail(10).to_string(index=False)
    )

    print("\nLatest rolling signal relationships:")
    print(
        df[[
            "timestamp",
            "cpu_ram_rolling_corr",
            "cpu_frequency_rolling_corr",
            "cpu_network_rolling_corr",
        ]].tail(5).to_string(index=False)
    )

    print(
        "\nResearch note: anomaly flags identify statistical deviations, "
        "not confirmed faults or predicted hardware failures."
    )


if __name__ == "__main__":
    analyze()
