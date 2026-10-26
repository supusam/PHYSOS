
"""
PHYSOS v1.9 — Synthetic Stress Testing

Compares:
    1. Raw temporal detector
    2. 2-of-3 persistence detector

Scenarios:
    normal, noisy normal, temperature-only change, frequency-only change,
    strong drift, moderate drift, weak drift, noisy drift, sudden change.

All thresholds are fixed before evaluation. All data are synthetic.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

N_SAMPLES = 600
EVENT_START = 300
EVENT_END = 360
N_TRIALS = 30

WINDOW = 20
TEMP_THRESHOLD = 0.03
FREQ_THRESHOLD = -1.0

PERSISTENCE_WINDOW = 3
PERSISTENCE_COUNT = 2

SEED_BASE = 90000

TRIALS_PATH = DATA_DIR / "stress_test_v19_trials.csv"
SUMMARY_PATH = DATA_DIR / "stress_test_v19_summary.csv"

SCENARIOS = [
    "normal",
    "noisy_normal",
    "temperature_only",
    "frequency_only",
    "strong_drift",
    "moderate_drift",
    "weak_drift",
    "noisy_gradual_drift",
    "sudden_change",
]


# ------------------------------------------------------------
# SYNTHETIC SCENARIO GENERATOR
# ------------------------------------------------------------

def generate_data(scenario, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(N_SAMPLES)

    temp_noise = 0.08
    freq_noise = 3.0

    if scenario == "noisy_normal":
        temp_noise = 0.25
        freq_noise = 12.0

    temperature = (
        28.0
        + 0.003 * t
        + rng.normal(0, temp_noise, N_SAMPLES)
    )

    frequency = (
        2400.0
        + rng.normal(0, freq_noise, N_SAMPLES)
    )

    event = (
        (t >= EVENT_START)
        & (t < EVENT_END)
    )

    position = np.zeros(N_SAMPLES, dtype=float)
    position[event] = t[event] - EVENT_START

    if scenario == "normal" or scenario == "noisy_normal":
        pass

    elif scenario == "temperature_only":
        temperature[event] += 0.055 * position[event]

    elif scenario == "frequency_only":
        frequency[event] -= 1.6 * position[event]

    elif scenario == "strong_drift":
        temperature[event] += 0.055 * position[event]
        frequency[event] -= 1.6 * position[event]

    elif scenario == "moderate_drift":
        temperature[event] += 0.040 * position[event]
        frequency[event] -= 1.2 * position[event]

    elif scenario == "weak_drift":
        temperature[event] += 0.025 * position[event]
        frequency[event] -= 0.65 * position[event]

    elif scenario == "noisy_gradual_drift":
        temperature[event] += 0.055 * position[event]
        frequency[event] -= 1.6 * position[event]

        temperature[event] += rng.normal(
            0, 0.25, event.sum()
        )
        frequency[event] += rng.normal(
            0, 12.0, event.sum()
        )

    elif scenario == "sudden_change":
        temperature[event] += 3.0
        frequency[event] -= 100.0

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return pd.DataFrame({
        "sample": t,
        "temperature_c": temperature,
        "frequency_mhz": frequency,
        "event": event,
        "scenario": scenario,
        "source": "SIMULATED",
    })


# ------------------------------------------------------------
# CAUSAL ROLLING SLOPES
# ------------------------------------------------------------


def rolling_slope(series, window):
    """Calculate causal rolling linear-regression slope per sample."""

    x = np.arange(window, dtype=float)
    x_centered = x - np.mean(x)
    denominator = np.sum(x_centered * x_centered)

    def calculate(values):
        y = np.asarray(values, dtype=float)

        # Center a fresh copy to avoid modifying input arrays.
        y_centered = y - np.mean(y)

        return float(
            np.dot(x_centered, y_centered) / denominator
        )

    return series.rolling(
        window=window,
        min_periods=window,
    ).apply(calculate, raw=True)



def add_features(df):
    result = df.copy()

    result["temperature_slope"] = rolling_slope(
        result["temperature_c"], WINDOW
    )

    result["frequency_slope"] = rolling_slope(
        result["frequency_mhz"], WINDOW
    )

    return result


# ------------------------------------------------------------
# DETECTORS
# ------------------------------------------------------------

def raw_detector(df):
    temp_condition = (
        df["temperature_slope"] > TEMP_THRESHOLD
    )

    freq_condition = (
        df["frequency_slope"] < FREQ_THRESHOLD
    )

    return (
        temp_condition & freq_condition
    ).fillna(False).to_numpy(dtype=bool)


def persistence_detector(raw_alerts):
    counts = pd.Series(
        raw_alerts.astype(int)
    ).rolling(
        window=PERSISTENCE_WINDOW,
        min_periods=PERSISTENCE_WINDOW,
    ).sum()

    return (
        counts >= PERSISTENCE_COUNT
    ).fillna(False).to_numpy(dtype=bool)


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

def count_episodes(alerts):
    alerts = np.asarray(alerts, dtype=bool)

    if not alerts.any():
        return 0

    starts = alerts & ~np.r_[False, alerts[:-1]]
    return int(starts.sum())


def calculate_metrics(alerts, event_mask):
    alerts = np.asarray(alerts, dtype=bool)
    event_mask = np.asarray(event_mask, dtype=bool)

    event_alerts = alerts & event_mask
    false_alerts = alerts & ~event_mask

    detected = bool(event_alerts.any())

    if detected:
        first_alert = int(np.flatnonzero(event_alerts)[0])
        delay = first_alert - EVENT_START
    else:
        delay = np.nan

    return {
        "event_detected": detected,
        "delay_samples": delay,
        "event_alert_samples": int(event_alerts.sum()),
        "false_alarm_samples": int(false_alerts.sum()),
        "false_alarm_episodes": count_episodes(false_alerts),
        "total_alert_samples": int(alerts.sum()),
    }


# ------------------------------------------------------------
# RUN TRIALS
# ------------------------------------------------------------

def run_experiment():
    rows = []

    print("=" * 82)
    print("PHYSOS v1.9 — SYNTHETIC STRESS TEST")
    print("=" * 82)
    print(f"Trials per scenario: {N_TRIALS}")
    print(f"Samples per trial: {N_SAMPLES}")
    print(f"Event interval: [{EVENT_START}, {EVENT_END})")
    print(f"Temperature slope threshold: > {TEMP_THRESHOLD}")
    print(f"Frequency slope threshold: < {FREQ_THRESHOLD}")
    print("Methods: raw, persistent_2_of_3")
    print()

    for scenario_index, scenario in enumerate(SCENARIOS):
        for trial in range(N_TRIALS):
            seed = (
                SEED_BASE
                + scenario_index * 1000
                + trial
            )

            df = generate_data(scenario, seed)
            df = add_features(df)

            raw = raw_detector(df)
            persistent = persistence_detector(raw)

            event_mask = df["event"].to_numpy(dtype=bool)

            for method, alerts in [
                ("raw", raw),
                ("persistent_2_of_3", persistent),
            ]:
                metrics = calculate_metrics(
                    alerts, event_mask
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


# ------------------------------------------------------------
# SUMMARY AND OUTPUT
# ------------------------------------------------------------

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    results = run_experiment()

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

    results.to_csv(TRIALS_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\n" + "=" * 82)
    print("PHYSOS v1.9 SUMMARY")
    print("=" * 82)

    columns = [
        "scenario",
        "method",
        "event_detection_rate",
        "mean_delay_samples",
        "mean_false_alarm_samples",
        "mean_false_alarm_episodes",
    ]

    print(summary[columns].round(3).to_string(index=False))

    print("\nFiles saved:")
    print(TRIALS_PATH)
    print(SUMMARY_PATH)

    print("\nInterpretation:")
    print("- Detection rate is the fraction of trials with an alert during the event.")
    print("- False alarms are alerts outside the injected event interval.")
    print("- Delay is measured in samples, not seconds.")
    print("- Results are synthetic and do not establish hardware performance.")


if __name__ == "__main__":
    main()
