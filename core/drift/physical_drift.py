
import os
import sys

import numpy as np
import pandas as pd

INPUT_FILE = os.path.join("data", "system_telemetry.csv")
OUTPUT_FILE = os.path.join("data", "physical_drift.csv")

# At approximately one reading per second:
BASELINE_WINDOW = 30
RECENT_WINDOW = 10

# Experimental thresholds; these require validation.
DRIFT_THRESHOLD = 2.5
MIN_SIGNALS_FOR_DRIFT = 2
PERSISTENCE_WINDOW = 5
MIN_PERSISTENT_FLAGS = 3

METRICS = [
    "cpu_percent",
    "ram_percent",
    "cpu_frequency",
    "disk_percent",
    "bytes_sent",
    "bytes_received",
]


def robust_scale(series):
    """Estimate baseline variability using median absolute deviation."""
    median = series.median()
    mad = (series - median).abs().median()
    return 1.4826 * mad


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        print("Run the PHYSOS telemetry collector first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE)

    # Support both versions of the CPU frequency column.
    if (
        "cpu_frequency" not in df.columns
        and "cpu_frequency_mhz" in df.columns
    ):
        df = df.rename(
            columns={"cpu_frequency_mhz": "cpu_frequency"}
        )

    required = ["timestamp"] + METRICS
    missing = [column for column in required if column not in df.columns]

    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        sys.exit(1)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )

    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        df[metric] = df[metric].replace([np.inf, -np.inf], np.nan)

    df = (
        df.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )

    for metric in METRICS:
        df[metric] = df[metric].ffill()

    df = df.dropna(subset=METRICS).reset_index(drop=True)

    minimum_rows = BASELINE_WINDOW + RECENT_WINDOW

    if len(df) < minimum_rows:
        print("Not enough data to establish a drift baseline.")
        print(f"Valid readings: {len(df)}")
        print(f"Minimum required: {minimum_rows}")
        print("Collect more telemetry, then run this script again.")
        sys.exit(1)

    print("\nPHYSOS Physical Drift Engine")
    print("=" * 42)
    print(f"Valid readings: {len(df)}")
    print(f"Historical baseline: {BASELINE_WINDOW} readings")
    print(f"Recent window: {RECENT_WINDOW} readings")
    print("Comparing recent behavior against earlier behavior...\n")

    # The baseline excludes the current recent window.
    # This helps avoid comparing a signal against itself.
    drift_score_columns = []

    for metric in METRICS:
        past = df[metric].shift(RECENT_WINDOW)

        baseline_median = past.rolling(
            BASELINE_WINDOW,
            min_periods=BASELINE_WINDOW,
        ).median()

        baseline_mad = past.rolling(
            BASELINE_WINDOW,
            min_periods=BASELINE_WINDOW,
        ).apply(
            lambda values: np.median(
                np.abs(values - np.median(values))
            ),
            raw=True,
        )

        baseline_std = past.rolling(
            BASELINE_WINDOW,
            min_periods=BASELINE_WINDOW,
        ).std()

        recent_median = df[metric].rolling(
            RECENT_WINDOW,
            min_periods=RECENT_WINDOW,
        ).median()

        scale = 1.4826 * baseline_mad
        scale = scale.where(scale > 1e-9, baseline_std)
        scale = scale.where(scale > 1e-9, np.nan)

        score_column = f"{metric}_drift_score"

        df[score_column] = (
            (recent_median - baseline_median) / scale
        ).replace([np.inf, -np.inf], np.nan)

        drift_score_columns.append(score_column)

        df[f"{metric}_baseline"] = baseline_median
        df[f"{metric}_recent_median"] = recent_median

    # Count how many signals have changed substantially.
    df["drifting_signal_count"] = (
        df[drift_score_columns].abs()
        .ge(DRIFT_THRESHOLD)
        .sum(axis=1)
    )

    # A candidate multivariate drift requires multiple signals
    # to deviate from their own historical baselines.
    valid_baseline = df[drift_score_columns].notna().all(axis=1)

    df["multi_signal_drift"] = (
        valid_baseline
        & (df["drifting_signal_count"] >= MIN_SIGNALS_FOR_DRIFT)
    )

    # Flag persistence rather than relying on a single reading.
    df["persistent_drift"] = (
        df["multi_signal_drift"]
        .rolling(
            PERSISTENCE_WINDOW,
            min_periods=PERSISTENCE_WINDOW,
        )
        .sum()
        .ge(MIN_PERSISTENT_FLAGS)
    )

    df["drifting_signals"] = df.apply(
        lambda row: ", ".join(
            metric
            for metric in METRICS
            if pd.notna(row[f"{metric}_drift_score"])
            and abs(row[f"{metric}_drift_score"]) >= DRIFT_THRESHOLD
        ),
        axis=1,
    )

    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("Analysis complete.")
    print(f"Output saved to: {OUTPUT_FILE}")

    valid_rows = int(valid_baseline.sum())
    candidate_count = int(
        df.loc[valid_baseline, "multi_signal_drift"].sum()
    )
    persistent_count = int(df["persistent_drift"].sum())

    print(f"Rows with established baselines: {valid_rows}")
    print(f"Candidate multi-signal drift rows: {candidate_count}")
    print(f"Persistent drift rows: {persistent_count}")

    print("\nLatest results:")
    columns_to_show = [
        "timestamp",
        "drifting_signal_count",
        "multi_signal_drift",
        "persistent_drift",
        "drifting_signals",
    ]

    print(df[columns_to_show].tail(10).to_string(index=False))

    print(
        "\nResearch caution: these are statistical drift candidates, "
        "not confirmed faults or predictions of future failure."
    )


if __name__ == "__main__":
    main()
