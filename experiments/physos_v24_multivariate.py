
"""
PHYSOS v2.4 — Multivariate Temporal Anomaly Detection

Purpose:
Compare:
  1. Baseline: individual sensor level and slope features
  2. Relationship: cross-sensor temporal relationship features
  3. Isolation Forest: multivariate anomaly detection

Data source: SYNTHETIC ONLY.

Experimental safeguards:
  - Calibration and final testing use different random seeds.
  - Isolation Forest is trained only on normal calibration data.
  - Alert thresholds are calibrated on separate normal calibration trials.
  - Final test results must not be used to tune this experiment.

Run:
  python experiments/physos_v24_multivariate.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 600
EVENT_START = 300
EVENT_END = 360

SLOPE_WINDOW = 20
RELATION_WINDOW = 40

CALIBRATION_TRIALS = 100
TRAIN_TRIALS_PER_SCENARIO = 60
THRESHOLD_PERCENTILE = 95.0

CALIBRATION_SEED = 910000
TEST_SEED = 990000

CONTAMINATION = 0.05

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
    "hidden_relationship",
]

BASELINE_FEATURES = [
    "abs_temp_deviation",
    "abs_freq_deviation",
    "positive_temp_slope",
    "negative_freq_slope",
]

RELATION_FEATURES = [
    "rolling_correlation",
    "lag_corr_1",
    "lag_corr_3",
    "lag_corr_5",
    "rolling_covariance",
    "slope_coupling",
]

MODEL_FEATURES = [
    "temperature_deviation",
    "frequency_deviation",
    "temperature_slope",
    "frequency_slope",
    "rolling_correlation",
    "lag_corr_3",
    "rolling_covariance",
    "slope_coupling",
]


# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================

def generate_data(scenario, seed):
    """Generate synthetic temperature and frequency measurements."""

    rng = np.random.default_rng(seed)
    t = np.arange(N_SAMPLES)

    noisy = scenario in {
        "noisy_normal",
        "noisy_gradual_drift",
    }

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

    event_mask = (
        (t >= EVENT_START)
        & (t < EVENT_END)
    )

    event_index = np.maximum(t - EVENT_START, 0)

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

    elif scenario in {"strong_drift", "noisy_gradual_drift"}:
        temperature[event_mask] += (
            0.055 * event_index[event_mask]
        )
        frequency[event_mask] -= (
            1.60 * event_index[event_mask]
        )

    elif scenario == "sudden_change":
        temperature[event_mask] += 3.0
        frequency[event_mask] -= 100.0

    elif scenario == "hidden_relationship":
        # A coordinated oscillatory pattern plus small drift.
        # This is a designed synthetic challenge, not evidence
        # that this relationship occurs in real equipment.
        phase = event_index[event_mask] / 5.0

        temperature[event_mask] += 0.35 * np.sin(phase)
        frequency[event_mask] += 18.0 * np.sin(phase)

        temperature[event_mask] += (
            0.008 * event_index[event_mask]
        )
        frequency[event_mask] -= (
            0.15 * event_index[event_mask]
        )

    return pd.DataFrame({
        "sample": t,
        "temperature": temperature,
        "frequency": frequency,
        "event": event_mask.astype(int),
        "scenario": scenario,
        "seed": seed,
        "source": "SIMULATED",
    })


# ============================================================
# CAUSAL ROLLING FEATURES
# ============================================================

def rolling_slope(values, window):
    """Calculate causal rolling linear-regression slope."""

    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)

    x = np.arange(window, dtype=float)
    x = x - x.mean()
    denominator = np.sum(x ** 2)

    windows = np.lib.stride_tricks.sliding_window_view(
        values, window
    )

    slopes = (
        (windows - windows.mean(axis=1, keepdims=True))
        @ x
    ) / denominator

    result[window - 1:] = slopes
    return result


def add_features(df):
    """Calculate sensor level, slope, and relationship features."""

    out = df.copy()

    # Baseline uses the initial 200 samples only.
    # The synthetic event starts at sample 300.
    temp_baseline = np.median(
        out["temperature"].iloc[:200]
    )
    freq_baseline = np.median(
        out["frequency"].iloc[:200]
    )

    out["temperature_deviation"] = (
        out["temperature"] - temp_baseline
    )
    out["frequency_deviation"] = (
        out["frequency"] - freq_baseline
    )

    out["abs_temp_deviation"] = (
        out["temperature_deviation"].abs()
    )
    out["abs_freq_deviation"] = (
        out["frequency_deviation"].abs()
    )

    out["temperature_slope"] = rolling_slope(
        out["temperature"], SLOPE_WINDOW
    )
    out["frequency_slope"] = rolling_slope(
        out["frequency"], SLOPE_WINDOW
    )

    out["positive_temp_slope"] = (
        out["temperature_slope"].clip(lower=0)
    )
    out["negative_freq_slope"] = (
        -out["frequency_slope"]
    ).clip(lower=0)

    temp = out["temperature"]
    freq = out["frequency"]

    # Current temperature versus current frequency.
    out["rolling_correlation"] = (
        temp.rolling(RELATION_WINDOW).corr(freq)
    )

    # Compare current temperature with earlier frequency.
    for lag in (1, 3, 5):
        out[f"lag_corr_{lag}"] = (
            temp.rolling(RELATION_WINDOW)
            .corr(freq.shift(lag))
        )

    out["rolling_covariance"] = (
        temp.rolling(RELATION_WINDOW).cov(freq)
    )

    out["slope_coupling"] = (
        out["positive_temp_slope"]
        * out["negative_freq_slope"]
    )

    return out


# ============================================================
# CALIBRATION DATA
# ============================================================

def make_calibration_data():
    """
    Split normal calibration trials into:
      - training trials for the Isolation Forest
      - independent calibration trials for thresholds
    """

    training = []
    threshold_calibration = []

    for scenario_index, scenario in enumerate(
        ["normal", "noisy_normal"]
    ):
        for trial in range(CALIBRATION_TRIALS):
            seed = (
                CALIBRATION_SEED
                + scenario_index * 10000
                + trial
            )

            df = add_features(generate_data(scenario, seed))

            # Do not train or calibrate using event-period samples.
            pre_event = df[df["sample"] < EVENT_START].copy()

            if trial < TRAIN_TRIALS_PER_SCENARIO:
                training.append(pre_event)
            else:
                threshold_calibration.append(pre_event)

    return training, threshold_calibration


# ============================================================
# ROBUST FEATURE STATISTICS
# ============================================================

def robust_location_scale(values):
    values = pd.Series(values).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if values.empty:
        return 0.0, 1.0

    median = float(values.median())
    mad = float(np.median(np.abs(values - median)))

    # Robust estimate of standard deviation.
    scale = max(1.4826 * mad, 1e-8)

    return median, scale


def fit_robust_statistics(training, feature_names):
    stats = {}

    for feature in feature_names:
        values = pd.concat(
            [df[feature] for df in training],
            ignore_index=True,
        )

        stats[feature] = robust_location_scale(values)

    return stats


def robust_score(df, stats, feature_names, absolute_deviation=True):
    """
    Return the largest standardized feature deviation per sample.
    Higher scores mean more unusual behavior.
    """

    columns = []

    for feature in feature_names:
        median, scale = stats[feature]
        delta = df[feature] - median

        if absolute_deviation:
            delta = delta.abs()

        columns.append(delta / scale)

    scores = pd.concat(columns, axis=1)
    scores.columns = feature_names

    return scores.max(axis=1, skipna=False)


# ============================================================
# ISOLATION FOREST
# ============================================================

def fit_isolation_forest(training):
    """
    Train on normal calibration observations only.
    RobustScaler is fitted only on training data.
    """

    frames = [
        df[MODEL_FEATURES].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        for df in training
    ]

    X_train = pd.concat(
        frames, ignore_index=True
    )

    model = Pipeline([
        ("scaler", RobustScaler()),
        (
            "forest",
            IsolationForest(
                n_estimators=200,
                max_samples="auto",
                contamination=CONTAMINATION,
                random_state=24024,
                n_jobs=-1,
            ),
        ),
    ])

    model.fit(X_train)

    return model


def isolation_forest_score(df, model):
    """
    Higher scores represent more anomalous samples.
    Invalid warm-up rows remain NaN.
    """

    X = df[MODEL_FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )

    valid = X.notna().all(axis=1)
    scores = pd.Series(np.nan, index=df.index)

    if valid.any():
        scores.loc[valid] = (
            -model.score_samples(X.loc[valid])
        )

    return scores


# ============================================================
# THRESHOLD CALIBRATION
# ============================================================

def finite_max(series):
    values = pd.Series(series).dropna()
    return float(values.max()) if not values.empty else np.nan


def calibrate_score_thresholds(
    threshold_calibration,
    baseline_stats,
    relationship_stats,
    forest,
):
    """
    Set each threshold from the 95th percentile of independent
    normal calibration-trial maximum scores.

    This targets approximately 5% trial-level false alarms on
    calibration data. It does not guarantee the same test rate.
    """

    maxima = {
        "baseline": [],
        "relationship": [],
        "isolation_forest": [],
    }

    for df in threshold_calibration:
        b_score = robust_score(
            df,
            baseline_stats,
            BASELINE_FEATURES,
        )

        r_score = robust_score(
            df,
            relationship_stats,
            RELATION_FEATURES,
        )

        f_score = isolation_forest_score(df, forest)

        maxima["baseline"].append(finite_max(b_score))
        maxima["relationship"].append(finite_max(r_score))
        maxima["isolation_forest"].append(finite_max(f_score))

    thresholds = {}

    for method, values in maxima.items():
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            raise RuntimeError(
                f"No valid calibration scores for {method}"
            )

        thresholds[method] = float(
            np.percentile(values, THRESHOLD_PERCENTILE)
        )

    return thresholds


# ============================================================
# EVALUATION
# ============================================================

def wilson_interval(successes, total, z=1.959963984540054):
    """95% Wilson confidence interval for a binomial rate."""

    if total == 0:
        return np.nan, np.nan

    p = successes / total
    denom = 1 + z * z / total

    center = (p + z * z / (2 * total)) / denom

    margin = (
        z
        * np.sqrt(
            p * (1 - p) / total
            + z * z / (4 * total * total)
        )
        / denom
    )

    return max(0.0, center - margin), min(1.0, center + margin)


def evaluate_trial(df, prediction, method):
    actual = df["event"].to_numpy(dtype=bool)

    pred = (
        pd.Series(prediction, index=df.index)
        .fillna(False)
        .to_numpy(dtype=bool)
    )

    tp = int(np.sum(pred & actual))
    fp = int(np.sum(pred & ~actual))
    fn = int(np.sum(~pred & actual))
    tn = int(np.sum(~pred & ~actual))

    event_positions = np.flatnonzero(pred & actual)

    detected = int(len(event_positions) > 0)
    delay = (
        int(event_positions[0] - EVENT_START)
        if detected
        else np.nan
    )

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    return {
        "method": method,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "event_detected": detected,
        "delay_samples": delay,
        "false_alarm_samples": fp,
        "false_alarm_trial": int(fp > 0),
        "source": "SIMULATED",
    }


# ============================================================
# FRESH FINAL TEST DATA
# ============================================================

def run_test(
    baseline_stats,
    relationship_stats,
    forest,
    thresholds,
):
    results = []

    for scenario_index, scenario in enumerate(SCENARIOS):
        for trial in range(200):
            seed = (
                TEST_SEED
                + scenario_index * 10000
                + trial
            )

            df = add_features(generate_data(scenario, seed))

            baseline_score = robust_score(
                df,
                baseline_stats,
                BASELINE_FEATURES,
            )

            relationship_score = robust_score(
                df,
                relationship_stats,
                RELATION_FEATURES,
            )

            forest_score = isolation_forest_score(df, forest)

            predictions = {
                "baseline": (
                    baseline_score > thresholds["baseline"]
                ),
                "relationship": (
                    relationship_score > thresholds["relationship"]
                ),
                "isolation_forest": (
                    forest_score > thresholds["isolation_forest"]
                ),
            }

            for method, prediction in predictions.items():
                result = evaluate_trial(df, prediction, method)

                result.update({
                    "scenario": scenario,
                    "trial": trial,
                    "seed": seed,
                })

                results.append(result)

    return pd.DataFrame(results)


# ============================================================
# SUMMARY AND REPORT
# ============================================================

def summarize_results(results):
    rows = []

    for (scenario, method), group in results.groupby(
        ["scenario", "method"],
        sort=False,
    ):
        n = len(group)
        successes = int(group["event_detected"].sum())

        ci_low, ci_high = wilson_interval(successes, n)

        rows.append({
            "scenario": scenario,
            "method": method,
            "trials": n,
            "event_detection_rate": successes / n if n else np.nan,
            "event_detection_ci95_low": ci_low,
            "event_detection_ci95_high": ci_high,
            "mean_delay_samples": group["delay_samples"].mean(),
            "median_delay_samples": group["delay_samples"].median(),
            "mean_trial_precision": group["precision"].mean(),
            "mean_trial_recall": group["recall"].mean(),
            "mean_false_alarm_samples": (
                group["false_alarm_samples"].mean()
            ),
            "false_alarm_trial_rate": (
                group["false_alarm_trial"].mean()
            ),
        })

    return pd.DataFrame(rows)


def main():
    print("=" * 76)
    print("PHYSOS v2.4 — MULTIVARIATE TEMPORAL ANOMALY DETECTION")
    print("=" * 76)
    print("Data source: SYNTHETIC")
    print("Threshold percentile:", THRESHOLD_PERCENTILE)
    print()

    # --------------------------------------------------------
    # 1. Calibration
    # --------------------------------------------------------

    print("1/5 Creating calibration splits...")

    training, threshold_calibration = make_calibration_data()

    print("Training trials:", len(training))
    print("Threshold-calibration trials:", len(threshold_calibration))

    baseline_stats = fit_robust_statistics(
        training, BASELINE_FEATURES
    )

    relationship_stats = fit_robust_statistics(
        training, RELATION_FEATURES
    )

    # --------------------------------------------------------
    # 2. Train multivariate model
    # --------------------------------------------------------

    print("\n2/5 Training Isolation Forest on normal data...")

    forest = fit_isolation_forest(training)

    # --------------------------------------------------------
    # 3. Calibrate thresholds on separate calibration trials
    # --------------------------------------------------------

    print("\n3/5 Calibrating thresholds...")

    thresholds = calibrate_score_thresholds(
        threshold_calibration,
        baseline_stats,
        relationship_stats,
        forest,
    )

    for name, value in thresholds.items():
        print(f"{name:22s}: {value:.6f}")

    threshold_file = DATA_DIR / "physos_v24_thresholds.json"

    with open(threshold_file, "w", encoding="utf-8") as f:
        json.dump({
            "version": "PHYSOS v2.4",
            "data_source": "SIMULATED",
            "threshold_percentile": THRESHOLD_PERCENTILE,
            "thresholds": thresholds,
            "baseline_statistics": {
                k: list(v) for k, v in baseline_stats.items()
            },
            "relationship_statistics": {
                k: list(v) for k, v in relationship_stats.items()
            },
        }, f, indent=2)

    # --------------------------------------------------------
    # 4. Fresh test
    # --------------------------------------------------------

    print("\n4/5 Running fresh final test trials...")
    print("Test trials per scenario: 200")

    results = run_test(
        baseline_stats,
        relationship_stats,
        forest,
        thresholds,
    )

    trials_file = DATA_DIR / "physos_v24_test_trials.csv"
    results.to_csv(trials_file, index=False)

    # --------------------------------------------------------
    # 5. Summarize
    # --------------------------------------------------------

    print("\n5/5 Summarizing results...")

    summary = summarize_results(results)

    summary_file = DATA_DIR / "physos_v24_summary.csv"
    summary.to_csv(summary_file, index=False)

    config = {
        "version": "PHYSOS v2.4",
        "data_source": "SIMULATED",
        "samples_per_trial": N_SAMPLES,
        "event_start": EVENT_START,
        "event_end_exclusive": EVENT_END,
        "slope_window": SLOPE_WINDOW,
        "relationship_window": RELATION_WINDOW,
        "calibration_trials_per_scenario": CALIBRATION_TRIALS,
        "training_trials_per_scenario": TRAIN_TRIALS_PER_SCENARIO,
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "test_trials_per_scenario": 200,
        "test_seed_start": TEST_SEED,
        "methods": [
            "baseline",
            "relationship",
            "isolation_forest",
        ],
        "thresholds": thresholds,
        "note": (
            "All data are synthetic. Results are not evidence of "
            "real-world equipment-failure prediction."
        ),
    }

    config_file = DATA_DIR / "physos_v24_config.json"

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # --------------------------------------------------------
    # Display selected scenarios
    # --------------------------------------------------------

    print("\n" + "=" * 76)
    print("SELECTED RESULTS")
    print("=" * 76)

    selected = summary[
        summary["scenario"].isin([
            "normal",
            "noisy_normal",
            "moderate_drift",
            "noisy_gradual_drift",
            "hidden_relationship",
        ])
    ]

    columns = [
        "scenario",
        "method",
        "event_detection_rate",
        "event_detection_ci95_low",
        "event_detection_ci95_high",
        "mean_delay_samples",
        "false_alarm_trial_rate",
    ]

    print(selected[columns].to_string(index=False))

    # --------------------------------------------------------
    # False-alarm assessment
    # --------------------------------------------------------

    print("\n" + "=" * 76)
    print("NORMAL-CONDITION FALSE-ALARM ASSESSMENT")
    print("=" * 76)

    normal_summary = summary[
        summary["scenario"].isin(["normal", "noisy_normal"])
    ]

    print(
        normal_summary[
            [
                "scenario",
                "method",
                "false_alarm_trial_rate",
                "mean_false_alarm_samples",
            ]
        ].to_string(index=False)
    )

    print("\nFILES CREATED:")
    print(threshold_file)
    print(trials_file)
    print(summary_file)
    print(config_file)

    print("\nPHYSOS v2.4 COMPLETE.")
    print("All results are synthetic and require real-data validation.")


if __name__ == "__main__":
    main()
