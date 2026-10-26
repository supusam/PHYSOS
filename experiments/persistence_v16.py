
"""
PHYSOS v1.7 — Held-Out Persistence Filter Experiment

Purpose:
    Compare a raw temporal drift detector with a causal persistence filter
    on held-out synthetic scenarios.

Frozen thresholds:
    Temperature slope > 0.03 degrees C/sample
    Frequency slope < -1.0 MHz/sample

Persistence rule:
    At least 3 raw alerts among the latest 5 samples.

Important:
    All inputs are synthetic. Results do not establish performance on
    real computers, sensors, or physical systems.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

N_SAMPLES = 600
EVENT_START = 300
EVENT_END = 360
N_TRIALS = 30

WINDOW = 20

TEMP_THRESHOLD = 0.03
FREQ_THRESHOLD = -1.0

PERSISTENCE_WINDOW = 5
PERSISTENCE_COUNT = 3

# New seed range, separate from the earlier v1.6 experiment.
BASE_SEED = 70000

TRIALS_PATH = DATA_DIR / "persistence_v17_heldout_trials.csv"
SUMMARY_PATH = DATA_DIR / "persistence_v17_heldout_summary.csv"


# ============================================================
# SYNTHETIC DATA GENERATOR
# ============================================================

SCENARIOS = [
    "normal",
    "frequency_only",
    "gradual_drift",
    "noisy_gradual_drift",
    "sudden_change",
    "temperature_only",
    "weak_gradual_drift",
]


def generate_scenario(scenario, seed):
    """
    Generate a reproducible synthetic temperature/frequency time series.

    The event occupies samples [EVENT_START, EVENT_END).
    Temperature is in degrees Celsius.
    Frequency is in MHz.

    This generator is a simplified test model, not a physical model
    of any specific computer or sensor.
    """

    rng = np.random.default_rng(seed)

    t = np.arange(N_SAMPLES)

    temperature = (
        28.0
        + 0.003 * t
        + rng.normal(0, 0.08, N_SAMPLES)
    )

    frequency = (
        2400.0
        + rng.normal(0, 3.0, N_SAMPLES)
    )

    event_mask = (
        (t >= EVENT_START)
        & (t < EVENT_END)
    )

    event_position = np.zeros(N_SAMPLES, dtype=float)
    event_position[event_mask] = (
        t[event_mask] - EVENT_START
    )

    duration = EVENT_END - EVENT_START

    if scenario == "normal":
        # Baseline noise only.

        pass

    elif scenario == "frequency_only":
        # Frequency declines without a coordinated temperature rise.

        frequency[event_mask] -= (
            2.0 * event_position[event_mask]
        )

    elif scenario == "gradual_drift":
        # Coordinated, gradual temperature rise and frequency decline.

        temperature[event_mask] += (
            0.055 * event_position[event_mask]
        )

        frequency[event_mask] -= (
            1.6 * event_position[event_mask]
        )

    elif scenario == "noisy_gradual_drift":
        # Similar drift with additional measurement noise.

        temperature[event_mask] += (
            0.055 * event_position[event_mask]
        )

        frequency[event_mask] -= (
            1.6 * event_position[event_mask]
        )

        temperature[event_mask] += rng.normal(
            0, 0.20, event_mask.sum()
        )

        frequency[event_mask] += rng.normal(
            0, 12.0, event_mask.sum()
        )

    elif scenario == "sudden_change":
        # Abrupt changes in both signals.

        temperature[event_mask] += 3.0
        frequency[event_mask] -= 100.0

    elif scenario == "temperature_only":
        # Temperature changes without a matching frequency decline.

        temperature[event_mask] += (
            0.055 * event_position[event_mask]
        )

    elif scenario == "weak_gradual_drift":
        # Smaller coordinated changes.

        temperature[event_mask] += (
            0.025 * event_position[event_mask]
        )

        frequency[event_mask] -= (
            0.65 * event_position[event_mask]
        )

    else:
        raise ValueError(
            f"Unknown scenario: {scenario}"
        )

    return pd.DataFrame({
        "sample": t,
        "temperature_c": temperature,
        "frequency_mhz": frequency,
        "event": event_mask,
        "scenario": scenario,
        "source": "SIMULATED",
    })


# ============================================================
# TEMPORAL FEATURE CALCULATION
# ============================================================

def calculate_temporal_features(df):
    """
    Estimate causal rolling linear-regression slopes.

    The rolling window uses only the current and previous samples.
    No future samples are used to calculate a detection.
    """

    result = df.copy()

    x = np.arange(WINDOW, dtype=float)
    x_centered = x - x.mean()
    denominator = np.sum(x_centered ** 2)

    def rolling_slope(series):
        def slope(window_values):
            y = np.asarray(window_values, dtype=float)
            y_centered = y - y.mean()

            return np.sum(
                x_centered * y_centered
            ) / denominator

        return series.rolling(
            window=WINDOW,
            min_periods=WINDOW,
        ).apply(slope, raw=True)

    result["temperature_slope"] = rolling_slope(
        result["temperature_c"]
    )

    result["frequency_slope"] = rolling_slope(
        result["frequency_mhz"]
    )

    return result


# ============================================================
# DETECTORS
# ============================================================

def detect_raw_alerts(df):
    """
    Raw detector:
      1. Temperature slope exceeds its frozen threshold.
      2. Frequency slope is below its frozen threshold.

    Both conditions must hold simultaneously.
    """

    temperature_condition = (
        df["temperature_slope"] > TEMP_THRESHOLD
    )

    frequency_condition = (
        df["frequency_slope"] < FREQ_THRESHOLD
    )

    return (
        temperature_condition
        & frequency_condition
    ).fillna(False).to_numpy(dtype=bool)


def apply_persistence_filter(raw_alerts):
    """
    Causal persistence filter.

    At sample i, count raw alerts from the current sample and up to
    the previous PERSISTENCE_WINDOW - 1 samples.

    Alert when at least PERSISTENCE_COUNT raw alerts occur.
    """

    raw_series = pd.Series(
        raw_alerts.astype(int)
    )

    recent_count = raw_series.rolling(
        window=PERSISTENCE_WINDOW,
        min_periods=PERSISTENCE_WINDOW,
    ).sum()

    return (
        recent_count >= PERSISTENCE_COUNT
    ).fillna(False).to_numpy(dtype=bool)


# ============================================================
# EVENT-LEVEL METRICS
# ============================================================

def count_episodes(alerts):
    """Count separate alert episodes, including after quiet periods."""

    alerts = np.asarray(alerts, dtype=bool)

    if len(alerts) == 0:
        return 0

    starts = alerts & ~np.r_[False, alerts[:-1]]

    return int(starts.sum())


def measure(alerts, event_mask):
    """
    Calculate sample-level and event-level metrics.

    Event detection:
        At least one alert occurs during the injected event.

    Detection delay:
        Index of the first alert during the event minus EVENT_START.

    False-alarm samples:
        Alerts outside the injected event.

    False-alarm episodes:
        Separate alert episodes outside the event interval.
    """

    alerts = np.asarray(alerts, dtype=bool)
    event_mask = np.asarray(event_mask, dtype=bool)

    event_alerts = alerts & event_mask
    false_alerts = alerts & ~event_mask

    event_detected = bool(event_alerts.any())

    if event_detected:
        first_alert = int(np.flatnonzero(event_alerts)[0])
        delay_samples = first_alert - EVENT_START
    else:
        delay_samples = np.nan

    return {
        "event_detected": event_detected,
        "delay_samples": delay_samples,
        "event_alert_samples": int(event_alerts.sum()),
        "false_alarm_samples": int(false_alerts.sum()),
        "false_alarm_episodes": count_episodes(false_alerts),
        "total_alert_samples": int(alerts.sum()),
    }


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_experiment():
    rows = []

    print("=" * 88)
    print("PHYSOS v1.7 — HELD-OUT PERSISTENCE EXPERIMENT")
    print("=" * 88)
    print(f"Samples per trial: {N_SAMPLES}")
    print(f"Trials per scenario: {N_TRIALS}")
    print(
        f"Injected event interval: "
        f"[{EVENT_START}, {EVENT_END})"
    )
    print(f"Temperature slope threshold: > {TEMP_THRESHOLD}")
    print(f"Frequency slope threshold: < {FREQ_THRESHOLD}")
    print(
        f"Persistence: {PERSISTENCE_COUNT} alerts "
        f"within {PERSISTENCE_WINDOW} samples"
    )
    print(f"Seed base: {BASE_SEED}")
    print("\nRunning trials...\n")

    for scenario_index, scenario in enumerate(SCENARIOS):
        for trial in range(N_TRIALS):

            seed = (
                BASE_SEED
                + scenario_index * 1000
                + trial
            )

            df = generate_scenario(
                scenario=scenario,
                seed=seed,
            )

            df = calculate_temporal_features(df)

            raw_alerts = detect_raw_alerts(df)

            filtered_alerts = apply_persistence_filter(
                raw_alerts
            )

            event_mask = df["event"].to_numpy(dtype=bool)

            for method, alerts in [
                ("raw", raw_alerts),
                ("persistent_3_of_5", filtered_alerts),
            ]:
                metrics = measure(
                    alerts=alerts,
                    event_mask=event_mask,
                )

                rows.append({
                    "scenario": scenario,
                    "trial": trial + 1,
                    "seed": seed,
                    "method": method,
                    **metrics,
                })

        print(
            f"Completed {scenario_index + 1}/"
            f"{len(SCENARIOS)}: {scenario}"
        )

    return pd.DataFrame(rows)


# ============================================================
# SUMMARY AND OUTPUT
# ============================================================

def build_summary(results):
    summary = results.groupby(
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

    return summary


def main():
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = run_experiment()

    results.to_csv(
        TRIALS_PATH,
        index=False,
    )

    summary = build_summary(results)

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print("\n" + "=" * 88)
    print("EXPERIMENT SUMMARY")
    print("=" * 88)

    display_columns = [
        "scenario",
        "method",
        "trials",
        "event_detection_rate",
        "mean_delay_samples",
        "mean_false_alarm_samples",
        "mean_false_alarm_episodes",
    ]

    print(
        summary[display_columns]
        .round(3)
        .to_string(index=False)
    )

    print("\nFiles saved:")
    print(f"Trial results: {TRIALS_PATH}")
    print(f"Summary:       {SUMMARY_PATH}")

    print("\nInterpretation notes:")
    print("- All generated measurements are synthetic.")
    print("- Event detection means an alert occurred during the event.")
    print("- Detection delay is measured in samples, not seconds.")
    print("- This is a held-out seed test within the same simulator.")
    print("- It is not independent validation on physical hardware.")


if __name__ == "__main__":
    main()
