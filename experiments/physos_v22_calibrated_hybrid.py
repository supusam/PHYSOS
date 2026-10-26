"""
PHYSOS v2.2
Calibrated Hybrid Physical Drift Detector

Research objective:
Can a calibrated hybrid temporal detector detect heterogeneous
physical degradation while maintaining a predefined false-alarm budget?

IMPORTANT:
- Calibration data is used ONLY for threshold selection.
- Final test data uses completely new random seeds.
- All data in this experiment are SYNTHETIC.
- Results do not establish real-world hardware performance.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

CAL_TRIALS = 100
TEST_TRIALS = 200

CAL_SEED_START = 30000
TEST_SEED_START = 50000

N_SAMPLES = 600

EVENT_START = 300
EVENT_END = 360

WINDOW = 20

# Maximum acceptable fraction of calibration trials
# containing at least one false alarm.
FALSE_ALARM_BUDGET = 0.01

# Hybrid detector persistence
PERSISTENCE_WINDOW = 3
PERSISTENCE_REQUIRED = 2

# ============================================================
# SCENARIOS
# ============================================================

SCENARIOS = [
    "normal",
    "noisy_normal",
    "temperature_only",
    "frequency_only",
    "weak_drift",
    "moderate_drift",
    "strong_drift",
    "noisy_gradual_drift",
    "sudden_change",
]


# ============================================================
# DATA GENERATION
# ============================================================

def generate_data(scenario, seed):
    """
    Generate synthetic heterogeneous sensor data.

    Temperature:
        degrees C

    Frequency:
        MHz

    All generated values are synthetic.
    """

    rng = np.random.default_rng(seed)

    t = np.arange(N_SAMPLES)

    # --------------------------------------------------------
    # Baseline noise
    # --------------------------------------------------------

    if scenario in ["noisy_normal", "noisy_gradual_drift"]:
        temp_noise = 0.25
        freq_noise = 12.0
    else:
        temp_noise = 0.08
        freq_noise = 3.0

    temperature = (
        28.0
        + 0.0005 * t
        + rng.normal(0, temp_noise, N_SAMPLES)
    )

    frequency = (
        2400.0
        + rng.normal(0, freq_noise, N_SAMPLES)
    )

    # --------------------------------------------------------
    # Degradation event
    # --------------------------------------------------------

    event_mask = (
        (t >= EVENT_START)
        & (t < EVENT_END)
    )

    event_index = np.maximum(
        t - EVENT_START,
        0
    )

    if scenario == "temperature_only":

        temperature[event_mask] += (
            0.055 * event_index[event_mask]
        )

    elif scenario == "frequency_only":

        frequency[event_mask] -= (
            1.6 * event_index[event_mask]
        )

    elif scenario == "weak_drift":

        temperature[event_mask] += (
            0.025 * event_index[event_mask]
        )

        frequency[event_mask] -= (
            0.65 * event_index[event_mask]
        )

    elif scenario == "moderate_drift":

        temperature[event_mask] += (
            0.040 * event_index[event_mask]
        )

        frequency[event_mask] -= (
            1.20 * event_index[event_mask]
        )

    elif scenario == "strong_drift":

        temperature[event_mask] += (
            0.055 * event_index[event_mask]
        )

        frequency[event_mask] -= (
            1.60 * event_index[event_mask]
        )

    elif scenario == "noisy_gradual_drift":

        temperature[event_mask] += (
            0.055 * event_index[event_mask]
        )

        frequency[event_mask] -= (
            1.60 * event_index[event_mask]
        )

    elif scenario == "sudden_change":

        temperature[event_mask] += 3.0

        frequency[event_mask] -= 100.0

    return pd.DataFrame({
        "sample": t,
        "temperature": temperature,
        "frequency": frequency,
        "event": event_mask.astype(int),
        "source": "SIMULATED",
        "scenario": scenario,
        "seed": seed,
    })


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def rolling_slope(series, window):
    """
    Causal rolling linear-regression slope.
    """

    values = np.asarray(series, dtype=float)

    result = np.full(len(values), np.nan)

    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()

    denominator = np.sum(x_centered ** 2)

    for i in range(window - 1, len(values)):

        y = values[
            i - window + 1:
            i + 1
        ]

        y_centered = y - y.mean()

        result[i] = (
            np.sum(x_centered * y_centered)
            / denominator
        )

    return result


def add_features(df):
    """
    Create level and temporal features.
    """

    out = df.copy()

    baseline_temp = np.median(
        out["temperature"].iloc[:200]
    )

    baseline_freq = np.median(
        out["frequency"].iloc[:200]
    )

    out["temp_deviation"] = (
        out["temperature"] - baseline_temp
    )

    out["freq_deviation"] = (
        out["frequency"] - baseline_freq
    )

    out["temp_slope"] = rolling_slope(
        out["temperature"],
        WINDOW
    )

    out["freq_slope"] = rolling_slope(
        out["frequency"],
        WINDOW
    )

    # Combined temporal relationship
    out["joint_slope_score"] = (
        np.maximum(out["temp_slope"], 0)
        +
        np.maximum(-out["freq_slope"], 0)
    )

    return out


# ============================================================
# BASELINE FEATURE DISTRIBUTION
# ============================================================

def calibration_features(df):
    """
    Extract pre-event baseline features.

    Calibration is intentionally performed using only normal
    and noisy-normal scenarios.
    """

    features = add_features(df)

    # Only pre-event observations
    baseline = features[
        features["sample"] < EVENT_START
    ]

    return baseline


# ============================================================
# THRESHOLD CALIBRATION
# ============================================================

def percentile_threshold(values, percentile):
    values = pd.Series(values).dropna()

    if len(values) == 0:
        return np.nan

    return float(
        np.percentile(values, percentile)
    )


def calibrate_thresholds(calibration_data):
    """
    Select thresholds from calibration data.

    We use high percentiles of normal behavior.

    The thresholds are then frozen before the final test.
    """

    temp_dev = []
    freq_dev = []
    temp_slope = []
    freq_slope = []
    joint_score = []

    for df in calibration_data:

        baseline = calibration_features(df)

        temp_dev.extend(
            np.abs(
                baseline["temp_deviation"]
            ).dropna()
        )

        freq_dev.extend(
            np.abs(
                baseline["freq_deviation"]
            ).dropna()
        )

        temp_slope.extend(
            np.abs(
                baseline["temp_slope"]
            ).dropna()
        )

        freq_slope.extend(
            np.abs(
                baseline["freq_slope"]
            ).dropna()
        )

        joint_score.extend(
            baseline["joint_slope_score"]
            .dropna()
        )

    thresholds = {

        "temperature_level": percentile_threshold(
            temp_dev,
            99.0
        ),

        "frequency_level": percentile_threshold(
            freq_dev,
            99.0
        ),

        "temperature_slope": percentile_threshold(
            temp_slope,
            99.0
        ),

        "frequency_slope": percentile_threshold(
            freq_slope,
            99.0
        ),

        "joint_slope": percentile_threshold(
            joint_score,
            99.0
        ),
    }

    return thresholds


# ============================================================
# INDIVIDUAL DETECTORS
# ============================================================

def detector_temperature_level(df, thresholds):

    return (
        df["temp_deviation"]
        > thresholds["temperature_level"]
    )


def detector_frequency_level(df, thresholds):

    return (
        df["freq_deviation"]
        > thresholds["frequency_level"]
    )


def detector_temperature_slope(df, thresholds):

    return (
        df["temp_slope"]
        > thresholds["temperature_slope"]
    )


def detector_frequency_slope(df, thresholds):

    return (
        df["freq_slope"]
        < -thresholds["frequency_slope"]
    )


def detector_joint_slope(df, thresholds):

    temp_signal = (
        df["temp_slope"]
        > thresholds["temperature_slope"]
    )

    freq_signal = (
        df["freq_slope"]
        < -thresholds["frequency_slope"]
    )

    return (
        temp_signal
        & freq_signal
    )


# ============================================================
# HYBRID EVIDENCE DETECTOR
# ============================================================

def hybrid_detector(df, thresholds):

    temp_level = (
        df["temp_deviation"]
        > thresholds["temperature_level"]
    )

    freq_level = (
        df["freq_deviation"]
        > thresholds["frequency_level"]
    )

    temp_slope = (
        df["temp_slope"]
        > thresholds["temperature_slope"]
    )

    freq_slope = (
        df["freq_slope"]
        < -thresholds["frequency_slope"]
    )

    joint = (
        temp_slope
        & freq_slope
    )

    # --------------------------------------------------------
    # Evidence score
    # --------------------------------------------------------

    evidence = (
        temp_level.astype(int)
        + freq_level.astype(int)
        + temp_slope.astype(int)
        + freq_slope.astype(int)
        + (joint.astype(int) * 2)
    )

    # Require at least two independent pieces of evidence.
    raw_alert = evidence >= 2

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    alert = (
        raw_alert
        .rolling(
            PERSISTENCE_WINDOW,
            min_periods=PERSISTENCE_WINDOW
        )
        .sum()
        >= PERSISTENCE_REQUIRED
    )

    return alert.fillna(False)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_detector(
    df,
    predictions,
    method,
):
    """

    Calculate event detection, delay, false alarms,
    precision and recall.
    """

    actual = (
        df["event"]
        .astype(bool)
        .to_numpy()
    )

    pred = (
        predictions
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )

    inside_event = actual
    outside_event = ~actual

    tp = np.sum(pred & inside_event)
    fp = np.sum(pred & outside_event)
    fn = np.sum((~pred) & inside_event)
    tn = np.sum((~pred) & outside_event)

    # --------------------------------------------------------
    # Event detection
    # --------------------------------------------------------

    event_alerts = np.where(
        pred & inside_event
    )[0]

    if len(event_alerts) > 0:

        event_detected = 1

        delay = (
            int(event_alerts[0])
            - EVENT_START
        )

    else:

        event_detected = 0
        delay = np.nan

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    false_alarm_samples = int(fp)

    false_alarm_trial = int(
        false_alarm_samples > 0
    )

    return {
        "method": method,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "precision": precision,
        "recall": recall,
        "event_detected": event_detected,
        "delay_samples": delay,
        "false_alarm_samples": false_alarm_samples,
        "false_alarm_trial": false_alarm_trial,
    }


# ============================================================
# RUN CALIBRATION
# ============================================================

def build_calibration_dataset():

    datasets = []

    for scenario in [
        "normal",
        "noisy_normal"
    ]:

        for trial in range(CAL_TRIALS):

            seed = (
                CAL_SEED_START
                + trial
                + (
                    1000
                    if scenario == "noisy_normal"
                    else 0
                )
            )

            df = generate_data(
                scenario,
                seed
            )

            datasets.append(df)

    return datasets


# ============================================================
# TEST EXPERIMENT
# ============================================================

def run_test(thresholds):

    trial_results = []

    detailed_results = []

    methods = [
        "temperature_level",
        "frequency_level",
        "temperature_slope",
        "frequency_slope",
        "joint_slope",
        "hybrid",
    ]

    for scenario_index, scenario in enumerate(SCENARIOS):

        for trial in range(TEST_TRIALS):

            seed = (
                TEST_SEED_START
                + scenario_index * 10000
                + trial
            )

            df = generate_data(
                scenario,
                seed
            )

            df = add_features(df)

            detectors = {

                "temperature_level":
                    detector_temperature_level(
                        df,
                        thresholds
                    ),

                "frequency_level":
                    detector_frequency_level(
                        df,
                        thresholds
                    ),

                "temperature_slope":
                    detector_temperature_slope(
                        df,
                        thresholds
                    ),

                "frequency_slope":
                    detector_frequency_slope(
                        df,
                        thresholds
                    ),

                "joint_slope":
                    detector_joint_slope(
                        df,
                        thresholds
                    ),

                "hybrid":
                    hybrid_detector(
                        df,
                        thresholds
                    ),
            }

            for method in methods:

                result = evaluate_detector(
                    df,
                    detectors[method],
                    method
                )

                result.update({
                    "scenario": scenario,
                    "trial": trial,
                    "seed": seed,
                    "source": "SIMULATED",
                })

                trial_results.append(result)

                detailed_results.append(
                    result.copy()
                )

    return (
        pd.DataFrame(trial_results),
        pd.DataFrame(detailed_results)
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize_results(results):

    grouped = []

    for (scenario, method), group in results.groupby(
        ["scenario", "method"]
    ):

        grouped.append({

            "scenario":
                scenario,

            "method":
                method,

            "trials":
                len(group),

            "event_detection_rate":
                group["event_detected"].mean(),

            "mean_delay_samples":
                group["delay_samples"].mean(),

            "median_delay_samples":
                group["delay_samples"].median(),

            "precision":
                group["precision"].mean(),

            "recall":
                group["recall"].mean(),

            "mean_false_alarm_samples":
                group["false_alarm_samples"].mean(),

            "false_alarm_trial_rate":
                group["false_alarm_trial"].mean(),

        })

    return pd.DataFrame(grouped)


# ============================================================
# MAIN
# ============================================================

def main():

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 70)
    print("PHYSOS v2.2 CALIBRATED HYBRID DETECTOR")
    print("=" * 70)

    print()
    print("Generating calibration data...")

    calibration_data = (
        build_calibration_dataset()
    )

    print(
        f"Calibration datasets: "
        f"{len(calibration_data)}"
    )

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    print()
    print("Calibrating thresholds...")

    thresholds = calibrate_thresholds(
        calibration_data
    )

    print()
    print("CALIBRATED THRESHOLDS")
    print("-" * 40)

    for name, value in thresholds.items():

        print(
            f"{name:25s}: "
            f"{value:.6f}"
        )

    # Save thresholds
    threshold_file = (
        DATA_DIR
        / "physos_v22_calibrated_thresholds.json"
    )

    with open(
        threshold_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            thresholds,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    print()
    print("Generating NEW untouched test dataset...")

    results, detailed = run_test(
        thresholds
    )

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    detailed_file = (
        DATA_DIR
        / "physos_v22_test_trials.csv"
    )

    detailed.to_csv(
        detailed_file,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = summarize_results(
        results
    )

    summary_file = (
        DATA_DIR
        / "physos_v22_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PHYSOS v2.2 RESULTS")
    print("=" * 70)

    for scenario in SCENARIOS:

        print()
        print(f"SCENARIO: {scenario}")
        print("-" * 70)

        subset = summary[
            summary["scenario"] == scenario
        ]

        print(
            subset[
                [
                    "method",
                    "event_detection_rate",
                    "mean_delay_samples",
                    "precision",
                    "recall",
                    "false_alarm_trial_rate",
                ]
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Normal-data false alarm assessment
    # --------------------------------------------------------

    normal_results = results[
        results["scenario"].isin([
            "normal",
            "noisy_normal"
        ])
    ]

    hybrid_normal = normal_results[
        normal_results["method"] == "hybrid"
    ]

    false_alarm_rate = (
        hybrid_normal["false_alarm_trial"]
        .mean()
    )

    print()
    print("=" * 70)
    print("FALSE-ALARM BUDGET CHECK")
    print("=" * 70)

    print(
        f"Target budget       : "
        f"{FALSE_ALARM_BUDGET:.2%}"
    )

    print(
        f"Observed hybrid FA  : "
        f"{false_alarm_rate:.2%}"
    )

    if false_alarm_rate <= FALSE_ALARM_BUDGET:

        print(
            "STATUS: PASS"
        )

    else:

        print(
            "STATUS: FAIL"
        )

    # --------------------------------------------------------
    # Save experiment configuration
    # --------------------------------------------------------

    config = {

        "experiment":
            "PHYSOS v2.2",

        "calibration_trials":
            CAL_TRIALS,

        "test_trials_per_scenario":
            TEST_TRIALS,

        "calibration_seed_start":
            CAL_SEED_START,

        "test_seed_start":
            TEST_SEED_START,

        "window":
            WINDOW,

        "persistence_window":
            PERSISTENCE_WINDOW,

        "persistence_required":
            PERSISTENCE_REQUIRED,

        "false_alarm_budget":
            FALSE_ALARM_BUDGET,

        "event_start":
            EVENT_START,

        "event_end":
            EVENT_END,

        "data_source":
            "SIMULATED",

        "thresholds":
            thresholds,
    }

    config_file = (
        DATA_DIR
        / "physos_v22_config.json"
    )

    with open(
        config_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            indent=4
        )

    print()
    print("=" * 70)
    print("FILES CREATED")
    print("=" * 70)

    print(
        threshold_file
    )

    print(
        detailed_file
    )

    print(
        summary_file
    )

    print(
        config_file
    )

    print()
    print("PHYSOS v2.2 COMPLETE.")
    print()


if __name__ == "__main__":
    main()