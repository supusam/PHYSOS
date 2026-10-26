
"""
PHYSOS v1.8 — Persistence Filter Optimization

Compares:
    1. Raw temporal detector
    2. 2-of-3 persistence filter
    3. 3-of-5 persistence filter
    4. 4-of-7 persistence filter

Uses the v1.7 synthetic scenario generator and frozen detector
thresholds. Uses a new seed range for a held-out comparison.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the tested v1.7 synthetic generator and raw detector.
from persistence_v16 import (
    SCENARIOS,
    N_TRIALS,
    BASE_SEED,
    generate_scenario,
    calculate_temporal_features,
    detect_raw_alerts,
    measure,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

TRIALS_PATH = DATA_DIR / "persistence_v18_trials.csv"
SUMMARY_PATH = DATA_DIR / "persistence_v18_summary.csv"

# New seed range, separate from v1.7.
V18_SEED_BASE = 80000

FILTERS = [
    ("persistent_2_of_3", 2, 3),
    ("persistent_3_of_5", 3, 5),
    ("persistent_4_of_7", 4, 7),
]


def apply_persistence(raw_alerts, required, window):
    """Apply a causal rolling persistence rule."""
    rolling_count = pd.Series(
        raw_alerts.astype(int)
    ).rolling(
        window=window,
        min_periods=window,
    ).sum()

    return (
        rolling_count >= required
    ).fillna(False).to_numpy(dtype=bool)


def run_experiment():
    rows = []

    total = len(SCENARIOS) * N_TRIALS
    completed = 0

    print("=" * 80)
    print("PHYSOS v1.8 — PERSISTENCE FILTER OPTIMIZATION")
    print("=" * 80)
    print(f"Scenarios: {len(SCENARIOS)}")
    print(f"Trials per scenario: {N_TRIALS}")
    print(f"Seed base: {V18_SEED_BASE}")
    print("Filters: raw, 2-of-3, 3-of-5, 4-of-7")
    print()

    for scenario_index, scenario in enumerate(SCENARIOS):
        for trial in range(N_TRIALS):
            seed = (
                V18_SEED_BASE
                + scenario_index * 1000
                + trial
            )

            df = generate_scenario(scenario, seed)
            df = calculate_temporal_features(df)

            raw_alerts = detect_raw_alerts(df)
            event_mask = df["event"].to_numpy(dtype=bool)

            methods = [("raw", raw_alerts)]

            for name, required, window in FILTERS:
                filtered = apply_persistence(
                    raw_alerts,
                    required=required,
                    window=window,
                )
                methods.append((name, filtered))

            for method, alerts in methods:
                metrics = measure(alerts, event_mask)

                rows.append({
                    "scenario": scenario,
                    "trial": trial + 1,
                    "seed": seed,
                    "method": method,
                    **metrics,
                })

            completed += 1

        print(
            f"Completed scenario {scenario_index + 1}/"
            f"{len(SCENARIOS)}: {scenario}"
        )

    return pd.DataFrame(rows)


def build_summary(results):
    return results.groupby(
        ["scenario", "method"],
        as_index=False,
    ).agg(
        trials=("trial", "count"),
        event_detection_rate=("event_detected", "mean"),
        mean_delay_samples=("delay_samples", "mean"),
        median_delay_samples=("delay_samples", "median"),
        mean_event_alert_samples=("event_alert_samples", "mean"),
        mean_false_alarm_samples=("false_alarm_samples", "mean"),
        mean_false_alarm_episodes=("false_alarm_episodes", "mean"),
        mean_total_alert_samples=("total_alert_samples", "mean"),
    )


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    results = run_experiment()
    summary = build_summary(results)

    results.to_csv(TRIALS_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\n" + "=" * 80)
    print("PHYSOS v1.8 SUMMARY")
    print("=" * 80)

    columns = [
        "scenario",
        "method",
        "event_detection_rate",
        "mean_delay_samples",
        "mean_false_alarm_samples",
        "mean_false_alarm_episodes",
    ]

    print(summary[columns].round(3).to_string(index=False))

    print("\nSaved files:")
    print(TRIALS_PATH)
    print(SUMMARY_PATH)

    print(
        "\nResearch note: this is a synthetic comparison using "
        "the v1.7 generator. It is not physical hardware validation."
    )


if __name__ == "__main__":
    main()
