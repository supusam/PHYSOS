
"""
PHYSOS v2.0: Reproducible Synthetic Validation Framework

Research purpose:
Compare single-signal and joint temporal detectors under controlled
synthetic scenarios.

IMPORTANT:
- All measurements are synthetic.
- Thresholds are frozen before evaluation.
- Calibration seeds and held-out test seeds are disjoint.
- This is not evidence of real-hardware performance.
"""

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd


# ============================================================
# 1. EXPERIMENT CONFIGURATION — FREEZE BEFORE HELD-OUT TESTING
# ============================================================

VERSION = "2.0"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

N_SAMPLES = 600
EVENT_START = 300
EVENT_END = 360
WINDOW = 20

N_CALIBRATION_TRIALS = 20
N_TEST_TRIALS = 100

CALIBRATION_SEED = 12000
TEST_SEED = 82000

# Frozen thresholds inherited from the v1.9 joint-slope approach.
TEMP_SLOPE_THRESHOLD = 0.03
FREQ_SLOPE_THRESHOLD = -1.0

# Baseline-relative thresholds for simple level detectors.
TEMP_LEVEL_INCREASE = 1.0
FREQ_LEVEL_DECREASE = 50.0

PERSISTENCE_REQUIRED = 2
PERSISTENCE_WINDOW = 3

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

METHODS = [
    "temperature_level",
    "frequency_level",
    "temperature_slope",
    "frequency_slope",
    "joint_slope",
    "joint_slope_persistent_2of3",
]

EVENT_SCENARIOS = [
    scenario for scenario in SCENARIOS
    if scenario not in ("normal", "noisy_normal")
]


# ============================================================
# 2. SYNTHETIC DATA GENERATION
# ============================================================

def generate_data(scenario, seed):
    """Generate one reproducible synthetic trial."""

    rng = np.random.default_rng(seed)
    t = np.arange(N_SAMPLES, dtype=float)

    noisy = scenario in ("noisy_normal", "noisy_gradual_drift")

    temp_noise = 0.25 if noisy else 0.08
    freq_noise = 12.0 if noisy else 3.0

    temperature = (
        28.0
        + 0.0005 * t
        + rng.normal(0, temp_noise, N_SAMPLES)
    )

    frequency = (
        2400.0
        + rng.normal(0, freq_noise, N_SAMPLES)
    )

    event = np.zeros(N_SAMPLES, dtype=bool)
    event[EVENT_START:EVENT_END] = True

    position = np.arange(EVENT_END - EVENT_START, dtype=float)

    temp_ramp = 0.0
    freq_ramp = 0.0

    if scenario == "temperature_only":
        temp_ramp = 0.055

    elif scenario == "frequency_only":
        freq_ramp = 1.6

    elif scenario == "strong_drift":
        temp_ramp = 0.055
        freq_ramp = 1.6

    elif scenario == "moderate_drift":
        temp_ramp = 0.040
        freq_ramp = 1.2

    elif scenario == "weak_drift":
        temp_ramp = 0.025
        freq_ramp = 0.65

    elif scenario == "noisy_gradual_drift":
        temp_ramp = 0.055
        freq_ramp = 1.6

    elif scenario == "sudden_change":
        temperature[EVENT_START:EVENT_END] += 3.0
        frequency[EVENT_START:EVENT_END] -= 100.0

    temperature[EVENT_START:EVENT_END] += temp_ramp * position
    frequency[EVENT_START:EVENT_END] -= freq_ramp * position

    if scenario == "noisy_gradual_drift":
        temperature[EVENT_START:EVENT_END] += rng.normal(
            0, 0.25, EVENT_END - EVENT_START
        )
        frequency[EVENT_START:EVENT_END] += rng.normal(
            0, 12.0, EVENT_END - EVENT_START
        )

    return pd.DataFrame({
        "sample": np.arange(N_SAMPLES),
        "temperature_c": temperature,
        "frequency_mhz": frequency,
        "event": event,
        "scenario": scenario,
        "source": "SIMULATED",
    })


# ============================================================
# 3. CAUSAL ROLLING LINEAR-REGRESSION SLOPE
# ============================================================

def rolling_slope(values, window):
    """
    Compute slope per sample using trailing windows only.

    The calculation never modifies the original input array.
    A slope of 0.04 means approximately +0.04 units per sample.
    """

    values = np.asarray(values, dtype=float)

    result = np.full(len(values), np.nan, dtype=float)

    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denominator = np.dot(x_centered, x_centered)

    if len(values) < window:
        return result

    # Each row is one trailing window, ending at its current sample.
    windows = np.lib.stride_tricks.sliding_window_view(
        values, window_shape=window
    )

    # Centering x makes the slope independent of the y mean.
    slopes = windows @ x_centered / denominator

    result[window - 1:] = slopes

    return result


def add_features(df):
    """Add temporal features without using future observations."""

    df = df.copy()

    df["temperature_slope"] = rolling_slope(
        df["temperature_c"].to_numpy(), WINDOW
    )

    df["frequency_slope"] = rolling_slope(
        df["frequency_mhz"].to_numpy(), WINDOW
    )

    # Baseline is restricted to the first 200 samples,
    # well before the injected event begins.
    baseline = df.loc[df["sample"] < 200]

    temp_baseline = baseline["temperature_c"].median()
    freq_baseline = baseline["frequency_mhz"].median()

    df["temperature_baseline"] = temp_baseline
    df["frequency_baseline"] = freq_baseline

    return df


# ============================================================
# 4. DETECTORS
# ============================================================

def persistence_filter(raw, required=2, window=3):
    """Causal trailing persistence: at least 2 alerts in 3 samples."""

    raw = np.asarray(raw, dtype=bool)
    result = np.zeros(len(raw), dtype=bool)

    for i in range(len(raw)):
        start = max(0, i - window + 1)
        result[i] = (
            np.sum(raw[start:i + 1]) >= required
        )

    return result


def run_detectors(df):
    """Run every detector on the same trial."""

    temp = df["temperature_c"].to_numpy()
    freq = df["frequency_mhz"].to_numpy()

    temp_slope = df["temperature_slope"].to_numpy()
    freq_slope = df["frequency_slope"].to_numpy()

    temp_baseline = float(df["temperature_baseline"].iloc[0])
    freq_baseline = float(df["frequency_baseline"].iloc[0])

    # NaN slope values are treated as non-alerts during warm-up.
    temp_slope_ok = (
        np.isfinite(temp_slope)
        & (temp_slope > TEMP_SLOPE_THRESHOLD)
    )

    freq_slope_ok = (
        np.isfinite(freq_slope)
        & (freq_slope < FREQ_SLOPE_THRESHOLD)
    )

    joint = temp_slope_ok & freq_slope_ok

    return {
        "temperature_level": (
            temp > temp_baseline + TEMP_LEVEL_INCREASE
        ),

        "frequency_level": (
            freq < freq_baseline - FREQ_LEVEL_DECREASE
        ),

        "temperature_slope": temp_slope_ok,

        "frequency_slope": freq_slope_ok,

        "joint_slope": joint,

        "joint_slope_persistent_2of3": persistence_filter(
            joint,
            required=PERSISTENCE_REQUIRED,
            window=PERSISTENCE_WINDOW,
        ),
    }


# ============================================================
# 5. METRICS
# ============================================================

def count_episodes(alerts):
    """Count contiguous alert episodes."""

    alerts = np.asarray(alerts, dtype=bool)

    if len(alerts) == 0:
        return 0

    starts = alerts & ~np.r_[False, alerts[:-1]]
    return int(starts.sum())


def wilson_interval(successes, total, z=1.96):
    """95% Wilson confidence interval for a binomial proportion."""

    if total == 0:
        return (np.nan, np.nan)

    p = successes / total
    denominator = 1 + z * z / total

    center = (
        p + z * z / (2 * total)
    ) / denominator

    margin = (
        z
        * math.sqrt(
            p * (1 - p) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )

    return (
        max(0.0, center - margin),
        min(1.0, center + margin),
    )


def evaluate_method(df, scenario, method, alerts):
    """Calculate event-level and sample-level metrics."""

    alerts = np.asarray(alerts, dtype=bool)

    is_event_scenario = scenario in EVENT_SCENARIOS

    truth = df["event"].to_numpy(dtype=bool)

    tp = int(np.sum(alerts & truth))
    fp = int(np.sum(alerts & ~truth))
    fn = int(np.sum(~alerts & truth))
    tn = int(np.sum(~alerts & ~truth))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    # False alarms are alerts outside the injected event.
    outside_event = ~truth
    false_alarm_samples = int(np.sum(alerts & outside_event))

    false_alarm_episodes = count_episodes(
        alerts & outside_event
    )

    event_alerts = np.flatnonzero(alerts & truth)

    if is_event_scenario and len(event_alerts) > 0:
        event_detected = 1
        first_alert = int(event_alerts[0])
        delay = first_alert - EVENT_START
    elif is_event_scenario:
        event_detected = 0
        delay = np.nan
    else:
        # Event detection rate is undefined for normal scenarios.
        event_detected = np.nan
        delay = np.nan

    return {
        "scenario": scenario,
        "method": method,
        "event_detected": event_detected,
        "detection_delay_samples": delay,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sample_precision": precision,
        "sample_recall": recall,
        "specificity": specificity,
        "false_alarm_samples": false_alarm_samples,
        "false_alarm_episodes": false_alarm_episodes,
        "total_alert_samples": int(alerts.sum()),
    }


# ============================================================
# 6. ONE TRIAL
# ============================================================

def run_trial(scenario, seed, phase, trial_number):
    df = add_features(generate_data(scenario, seed))
    detectors = run_detectors(df)

    rows = []

    for method in METHODS:
        metrics = evaluate_method(
            df=df,
            scenario=scenario,
            method=method,
            alerts=detectors[method],
        )

        metrics.update({
            "phase": phase,
            "trial": trial_number,
            "seed": seed,
            "n_samples": N_SAMPLES,
            "event_start": EVENT_START,
            "event_end": EVENT_END,
            "window": WINDOW,
            "temp_slope_threshold": TEMP_SLOPE_THRESHOLD,
            "freq_slope_threshold": FREQ_SLOPE_THRESHOLD,
            "source": "SIMULATED",
        })

        rows.append(metrics)

    return rows


# ============================================================
# 7. RUN CALIBRATION AND HELD-OUT EVALUATION
# ============================================================

def run_phase(phase, trials_per_scenario, base_seed):
    rows = []

    total = len(SCENARIOS) * trials_per_scenario
    completed = 0

    for scenario_index, scenario in enumerate(SCENARIOS):
        for trial in range(trials_per_scenario):
            seed = (
                base_seed
                + scenario_index * 100_000
                + trial
            )

            rows.extend(
                run_trial(
                    scenario=scenario,
                    seed=seed,
                    phase=phase,
                    trial_number=trial,
                )
            )

            completed += 1

        print(
            f"{phase}: completed "
            f"{completed}/{total} trials; latest={scenario}"
        )

    return pd.DataFrame(rows)


def summarize_results(trials):
    """Aggregate trial-level results by scenario and method."""

    rows = []

    for (scenario, method), group in trials.groupby(
        ["scenario", "method"]
    ):
        is_event_scenario = scenario in EVENT_SCENARIOS

        if is_event_scenario:
            detections = group["event_detected"].dropna()
            successes = int(detections.sum())
            total_events = len(detections)

            event_rate = (
                successes / total_events if total_events else np.nan
            )

            ci_low, ci_high = wilson_interval(
                successes, total_events
            )
        else:
            event_rate = np.nan
            ci_low = np.nan
            ci_high = np.nan

        rows.append({
            "scenario": scenario,
            "method": method,
            "trials": len(group),
            "event_detection_rate": event_rate,
            "event_detection_ci95_low": ci_low,
            "event_detection_ci95_high": ci_high,
            "mean_delay_samples": (
                group["detection_delay_samples"].mean()
            ),
            "median_delay_samples": (
                group["detection_delay_samples"].median()
            ),
            "mean_sample_precision": (
                group["sample_precision"].mean()
            ),
            "mean_sample_recall": (
                group["sample_recall"].mean()
            ),
            "pooled_sample_precision": (
                group["tp"].sum()
                / max(1, group["tp"].sum() + group["fp"].sum())
            ),
            "pooled_sample_recall": (
                group["tp"].sum()
                / max(1, group["tp"].sum() + group["fn"].sum())
            ),
            "mean_false_alarm_samples": (
                group["false_alarm_samples"].mean()
            ),
            "mean_false_alarm_episodes": (
                group["false_alarm_episodes"].mean()
            ),
            "mean_total_alert_samples": (
                group["total_alert_samples"].mean()
            ),
            "source": "SIMULATED",
        })

    return pd.DataFrame(rows)


def save_configuration():
    """Save the exact experiment settings for reproducibility."""

    config = {
        "version": VERSION,
        "source": "SIMULATED",
        "n_samples": N_SAMPLES,
        "event_interval": [EVENT_START, EVENT_END],
        "window": WINDOW,
        "calibration_trials_per_scenario": N_CALIBRATION_TRIALS,
        "heldout_trials_per_scenario": N_TEST_TRIALS,
        "calibration_seed": CALIBRATION_SEED,
        "heldout_seed": TEST_SEED,
        "temperature_slope_threshold": TEMP_SLOPE_THRESHOLD,
        "frequency_slope_threshold": FREQ_SLOPE_THRESHOLD,
        "temperature_level_increase": TEMP_LEVEL_INCREASE,
        "frequency_level_decrease": FREQ_LEVEL_DECREASE,
        "persistence_required": PERSISTENCE_REQUIRED,
        "persistence_window": PERSISTENCE_WINDOW,
        "scenarios": SCENARIOS,
        "methods": METHODS,
        "threshold_policy": (
            "Predeclared thresholds; do not change after "
            "inspecting held-out results."
        ),
    }

    path = DATA_DIR / "physos_v20_config.json"

    with path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    return path


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("PHYSOS v2.0 — SYNTHETIC VALIDATION")
    print("=" * 76)
    print("Scenarios:", len(SCENARIOS))
    print("Methods:", ", ".join(METHODS))
    print("Calibration trials/scenario:", N_CALIBRATION_TRIALS)
    print("Held-out trials/scenario:", N_TEST_TRIALS)
    print("Samples/trial:", N_SAMPLES)
    print("Event interval:", (EVENT_START, EVENT_END))
    print("Temperature slope threshold:", TEMP_SLOPE_THRESHOLD)
    print("Frequency slope threshold:", FREQ_SLOPE_THRESHOLD)
    print("\nAll data are synthetic. No hardware claims are made.\n")

    config_path = save_configuration()

    calibration = run_phase(
        phase="calibration",
        trials_per_scenario=N_CALIBRATION_TRIALS,
        base_seed=CALIBRATION_SEED,
    )

    calibration_trials_path = (
        DATA_DIR / "physos_v20_calibration_trials.csv"
    )
    calibration_summary_path = (
        DATA_DIR / "physos_v20_calibration_summary.csv"
    )

    calibration.to_csv(calibration_trials_path, index=False)
    summarize_results(calibration).to_csv(
        calibration_summary_path, index=False
    )

    print("\nCalibration complete.")
    print("Thresholds remain frozen for held-out testing.\n")

    heldout = run_phase(
        phase="heldout",
        trials_per_scenario=N_TEST_TRIALS,
        base_seed=TEST_SEED,
    )

    heldout_trials_path = (
        DATA_DIR / "physos_v20_heldout_trials.csv"
    )
    heldout_summary_path = (
        DATA_DIR / "physos_v20_heldout_summary.csv"
    )

    heldout.to_csv(heldout_trials_path, index=False)

    summary = summarize_results(heldout)
    summary.to_csv(heldout_summary_path, index=False)

    print("\n" + "=" * 76)
    print("PHYSOS v2.0 — HELD-OUT SUMMARY")
    print("=" * 76)

    columns = [
        "scenario",
        "method",
        "event_detection_rate",
        "event_detection_ci95_low",
        "event_detection_ci95_high",
        "mean_delay_samples",
        "pooled_sample_precision",
        "pooled_sample_recall",
        "mean_false_alarm_samples",
    ]

    print(
        summary[columns].round(4).to_string(index=False)
    )

    print("\nFiles saved:")
    print(config_path)
    print(calibration_trials_path)
    print(calibration_summary_path)
    print(heldout_trials_path)
    print(heldout_summary_path)

    print("\nInterpretation rules:")
    print("- Use held-out results for the primary comparison.")
    print("- Compare methods within the same scenario.")
    print("- NaN event-detection rates for normal scenarios are intentional.")
    print("- Delays are measured in samples, not seconds.")
    print("- Confidence intervals describe trial-level event detection.")
    print("- Synthetic success does not establish hardware performance.")


if __name__ == "__main__":
    main()
