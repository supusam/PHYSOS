
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

N = 600
START, END = 300, 360
WINDOW = 20
TRIALS = 30

# Frozen v1.3 thresholds
TEMP_THRESHOLD = 0.03
FREQ_THRESHOLD = -1.0

SCENARIOS = [
    "normal",
    "gradual_drift",
    "weak_gradual_drift",
    "temperature_only",
    "frequency_only",
    "sudden_change",
    "noisy_gradual_drift",
]


def rolling_slope(series, window):
    x = np.arange(window, dtype=float)
    x -= x.mean()
    denominator = np.sum(x ** 2)

    def calculate(y):
        y = y - y.mean()
        return np.dot(x, y) / denominator

    return series.rolling(
        window, min_periods=window
    ).apply(calculate, raw=True)


def generate(seed, scenario):
    rng = np.random.default_rng(seed)

    temp_noise = 0.8 if scenario == "noisy_gradual_drift" else 0.5
    freq_noise = 70.0 if scenario == "noisy_gradual_drift" else 45.0

    temperature = 35 + rng.normal(0, temp_noise, N)
    frequency = 2400 + rng.normal(0, freq_noise, N)

    ramp = np.linspace(0, 1, END - START)

    if scenario == "gradual_drift":
        temperature[START:END] += 5 * ramp
        frequency[START:END] -= 250 * ramp

    elif scenario == "weak_gradual_drift":
        temperature[START:END] += 2 * ramp
        frequency[START:END] -= 100 * ramp

    elif scenario == "temperature_only":
        temperature[START:END] += 5 * ramp

    elif scenario == "frequency_only":
        frequency[START:END] -= 250 * ramp

    elif scenario == "sudden_change":
        temperature[START:END] += 4
        frequency[START:END] -= 200

    elif scenario == "noisy_gradual_drift":
        temperature[START:END] += 5 * ramp
        frequency[START:END] -= 250 * ramp

    return pd.DataFrame({
        "temperature_c": temperature,
        "cpu_frequency_mhz": np.clip(frequency, 0, None),
    })


def alarm_episodes(mask):
    """Count contiguous runs of True values as separate alarm episodes."""
    mask = np.asarray(mask, dtype=bool)

    if len(mask) == 0:
        return 0

    starts = mask & ~np.r_[False, mask[:-1]]
    return int(starts.sum())


def evaluate_trial(df, scenario):
    temp_slope = rolling_slope(df["temperature_c"], WINDOW)
    freq_slope = rolling_slope(df["cpu_frequency_mhz"], WINDOW)

    alerts = (
        (temp_slope > TEMP_THRESHOLD)
        & (freq_slope < FREQ_THRESHOLD)
    ).fillna(False).to_numpy(dtype=bool)

    is_event = scenario != "normal"
    event_alerts = alerts[START:END] if is_event else np.array([], dtype=bool)
    outside_event = np.ones(N, dtype=bool)

    if is_event:
        outside_event[START:END] = False

    outside_alerts = alerts & outside_event

    event_detected = bool(event_alerts.any()) if is_event else False

    if event_detected:
        first_local = int(np.flatnonzero(event_alerts)[0])
        detection_delay_samples = first_local
    else:
        detection_delay_samples = np.nan

    return {
        "event_detected": int(event_detected),
        "detection_delay_samples": detection_delay_samples,
        "event_alert_samples": int(event_alerts.sum()),
        "false_alarm_samples": int(outside_alerts.sum()),
        "false_alarm_episodes": alarm_episodes(outside_alerts),
        "total_alert_samples": int(alerts.sum()),
    }


def main():
    rows = []

    for scenario_index, scenario in enumerate(SCENARIOS):
        for trial in range(TRIALS):
            seed = 50000 + scenario_index * 1000 + trial
            df = generate(seed, scenario)

            metrics = evaluate_trial(df, scenario)

            rows.append({
                "scenario": scenario,
                "trial": trial + 1,
                **metrics,
            })

    trial_results = pd.DataFrame(rows)

    trials_path = ROOT / "data" / "event_metrics_v15_trials.csv"
    trials_path.parent.mkdir(parents=True, exist_ok=True)
    trial_results.to_csv(trials_path, index=False)

    summary_rows = []

    for scenario, group in trial_results.groupby("scenario"):
        has_event = scenario != "normal"

        summary_rows.append({
            "scenario": scenario,
            "trials": len(group),
            "event_detection_rate": (
                group["event_detected"].mean() if has_event else np.nan
            ),
            "mean_delay_samples_when_detected": (
                group["detection_delay_samples"].mean()
                if has_event and group["event_detected"].sum() > 0
                else np.nan
            ),
            "median_delay_samples_when_detected": (
                group["detection_delay_samples"].median()
                if has_event and group["event_detected"].sum() > 0
                else np.nan
            ),
            "mean_event_alert_samples": (
                group["event_alert_samples"].mean()
                if has_event else np.nan
            ),
            "mean_false_alarm_samples": group["false_alarm_samples"].mean(),
            "mean_false_alarm_episodes": group["false_alarm_episodes"].mean(),
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = ROOT / "data" / "event_metrics_v15_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nPHYSOS v1.5 — EVENT-LEVEL EVALUATION")
    print("=" * 100)
    print(f"Trials per scenario: {TRIALS}")
    print(f"Frozen thresholds: temp slope > {TEMP_THRESHOLD}; "
          f"frequency slope < {FREQ_THRESHOLD}")
    print(f"Event window: samples {START}–{END - 1}")
    print("Delay is measured in samples, not seconds.")

    print("\nSummary:")
    print(summary.round(3).to_string(index=False))

    print("\nSaved:")
    print(trials_path)
    print(summary_path)

    print("\nSynthetic data only; no physical hardware validation.")


if __name__ == "__main__":
    main()
