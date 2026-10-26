"""
PHYSOS v2.3
Temporal Relationship Engine

Research question:
Do temporal relationships between heterogeneous physical sensors
contain degradation information that is not visible through
single-sensor threshold monitoring?

Features:
1. Temperature level deviation
2. Frequency level deviation
3. Temperature slope
4. Frequency slope
5. Rolling correlation
6. Lagged correlation
7. Rolling covariance
8. Slope-coupling score
9. Relationship anomaly score
10. Persistence-based temporal relationship detector

IMPORTANT:
- This experiment uses SYNTHETIC sensor data only.
- It does NOT establish real-world hardware performance.
- Test seeds are separate from earlier PHYSOS experiments.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

N_SAMPLES = 600

EVENT_START = 300
EVENT_END = 360

SLOPE_WINDOW = 20

RELATION_WINDOW = 40

PERSISTENCE_WINDOW = 5
PERSISTENCE_REQUIRED = 3

CALIBRATION_TRIALS = 100
TEST_TRIALS = 200

CALIBRATION_SEED = 70000
TEST_SEED = 80000


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

    # Special relationship scenario
    "hidden_relationship",

]


# ============================================================
# SYNTHETIC SENSOR GENERATOR
# ============================================================

def generate_data(
    scenario,
    seed
):
    """
    Generate synthetic temperature and frequency sensors.

    The hidden_relationship scenario is specifically designed
    to test whether a relationship between sensors can reveal
    degradation that is less obvious in individual signals.
    """

    rng = np.random.default_rng(seed)

    t = np.arange(N_SAMPLES)

    # --------------------------------------------------------
    # Noise configuration
    # --------------------------------------------------------

    if scenario in [
        "noisy_normal",
        "noisy_gradual_drift"
    ]:

        temperature_noise = 0.25
        frequency_noise = 12.0

    else:

        temperature_noise = 0.08
        frequency_noise = 3.0

    # --------------------------------------------------------
    # Normal physical behavior
    # --------------------------------------------------------

    temperature = (
        28.0
        + 0.0005 * t
        + rng.normal(
            0,
            temperature_noise,
            N_SAMPLES
        )
    )

    frequency = (
        2400.0
        + rng.normal(
            0,
            frequency_noise,
            N_SAMPLES
        )
    )

    event_mask = (
        (t >= EVENT_START)
        &
        (t < EVENT_END)
    )

    event_index = np.maximum(
        t - EVENT_START,
        0
    )

    # --------------------------------------------------------
    # Conventional degradation scenarios
    # --------------------------------------------------------

    if scenario == "temperature_only":

        temperature[event_mask] += (
            0.055
            * event_index[event_mask]
        )

    elif scenario == "frequency_only":

        frequency[event_mask] -= (
            1.6
            * event_index[event_mask]
        )

    elif scenario == "weak_drift":

        temperature[event_mask] += (
            0.025
            * event_index[event_mask]
        )

        frequency[event_mask] -= (
            0.65
            * event_index[event_mask]
        )

    elif scenario == "moderate_drift":

        temperature[event_mask] += (
            0.040
            * event_index[event_mask]
        )

        frequency[event_mask] -= (
            1.20
            * event_index[event_mask]
        )

    elif scenario == "strong_drift":

        temperature[event_mask] += (
            0.055
            * event_index[event_mask]
        )

        frequency[event_mask] -= (
            1.60
            * event_index[event_mask]
        )

    elif scenario == "noisy_gradual_drift":

        temperature[event_mask] += (
            0.055
            * event_index[event_mask]
        )

        frequency[event_mask] -= (
            1.60
            * event_index[event_mask]
        )

    elif scenario == "sudden_change":

        temperature[event_mask] += 3.0

        frequency[event_mask] -= 100.0

    # --------------------------------------------------------
    # HIDDEN RELATIONSHIP SCENARIO
    # --------------------------------------------------------
    #
    # The two sensors develop a synchronized relationship.
    #
    # Temperature remains relatively small in magnitude.
    # Frequency also remains relatively small in magnitude.
    #
    # However, their temporal movements become coupled.
    #
    # This allows us to test the central PHYSOS hypothesis.
    # --------------------------------------------------------

    elif scenario == "hidden_relationship":

        relationship_signal = np.zeros(
            N_SAMPLES
        )

        relationship_signal[event_mask] = (
            np.sin(
                event_index[event_mask]
                / 5.0
            )
        )

        temperature[event_mask] += (
            0.35
            * relationship_signal[event_mask]
        )

        frequency[event_mask] += (
            18.0
            * relationship_signal[event_mask]
        )

        # Small coordinated drift
        temperature[event_mask] += (
            0.008
            * event_index[event_mask]
        )

        frequency[event_mask] -= (
            0.15
            * event_index[event_mask]
        )

    return pd.DataFrame({

        "sample": t,

        "temperature":
            temperature,

        "frequency":
            frequency,

        "event":
            event_mask.astype(int),

        "scenario":
            scenario,

        "seed":
            seed,

        "source":
            "SIMULATED",
    })


# ============================================================
# ROLLING SLOPE
# ============================================================

def rolling_slope(
    values,
    window
):
    """
    Calculate causal rolling linear-regression slope.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    result = np.full(
        len(values),
        np.nan
    )

    x = np.arange(
        window,
        dtype=float
    )

    x_centered = (
        x - x.mean()
    )

    denominator = np.sum(
        x_centered ** 2
    )

    for i in range(
        window - 1,
        len(values)
    ):

        y = values[
            i - window + 1:
            i + 1
        ]

        y_centered = (
            y - y.mean()
        )

        result[i] = (
            np.sum(
                x_centered
                * y_centered
            )
            / denominator
        )

    return result


# ============================================================
# ROLLING CORRELATION
# ============================================================

def rolling_correlation(
    x,
    y,
    window
):
    """
    Pearson correlation calculated over a causal
    rolling window.
    """

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    result = np.full(
        len(x),
        np.nan
    )

    for i in range(
        window - 1,
        len(x)
    ):

        xw = x[
            i - window + 1:
            i + 1
        ]

        yw = y[
            i - window + 1:
            i + 1
        ]

        x_std = np.std(xw)
        y_std = np.std(yw)

        if (
            x_std < 1e-12
            or y_std < 1e-12
        ):
            result[i] = 0.0

        else:

            result[i] = np.corrcoef(
                xw,
                yw
            )[0, 1]

    return result


# ============================================================
# ROLLING COVARIANCE
# ============================================================

def rolling_covariance(
    x,
    y,
    window
):
    """
    Rolling covariance.
    """

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    result = np.full(
        len(x),
        np.nan
    )

    for i in range(
        window - 1,
        len(x)
    ):

        xw = x[
            i - window + 1:
            i + 1
        ]

        yw = y[
            i - window + 1:
            i + 1
        ]

        result[i] = np.cov(
            xw,
            yw,
            ddof=1
        )[0, 1]

    return result


# ============================================================
# LAGGED CORRELATION
# ============================================================

def lagged_correlation(
    x,
    y,
    window,
    lag
):
    """
    Calculate rolling correlation between:

        x(t)

    and

        y(t-lag)

    Positive lag means x is compared with an earlier y.
    """

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    result = np.full(
        len(x),
        np.nan
    )

    for i in range(
        window + lag - 1,
        len(x)
    ):

        xw = x[
            i - window + 1:
            i + 1
        ]

        yw = y[
            i - window - lag + 1:
            i - lag + 1
        ]

        if len(xw) != len(yw):
            continue

        if (
            np.std(xw) < 1e-12
            or np.std(yw) < 1e-12
        ):
            result[i] = 0.0

        else:

            result[i] = np.corrcoef(
                xw,
                yw
            )[0, 1]

    return result


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_relationship_features(
    df
):
    """
    Generate all temporal relationship features.
    """

    out = df.copy()

    # --------------------------------------------------------
    # Baselines
    # --------------------------------------------------------

    baseline_temperature = np.median(
        out["temperature"].iloc[:200]
    )

    baseline_frequency = np.median(
        out["frequency"].iloc[:200]
    )

    out["temperature_deviation"] = (
        out["temperature"]
        - baseline_temperature
    )

    out["frequency_deviation"] = (
        out["frequency"]
        - baseline_frequency
    )

    # --------------------------------------------------------
    # Slopes
    # --------------------------------------------------------

    out["temperature_slope"] = (
        rolling_slope(
            out["temperature"],
            SLOPE_WINDOW
        )
    )

    out["frequency_slope"] = (
        rolling_slope(
            out["frequency"],
            SLOPE_WINDOW
        )
    )

    # --------------------------------------------------------
    # Rolling correlation
    # --------------------------------------------------------

    out["rolling_correlation"] = (
        rolling_correlation(
            out["temperature"],
            out["frequency"],
            RELATION_WINDOW
        )
    )

    # --------------------------------------------------------
    # Lagged correlations
    # --------------------------------------------------------

    out["lag_corr_1"] = (
        lagged_correlation(
            out["temperature"],
            out["frequency"],
            RELATION_WINDOW,
            1
        )
    )

    out["lag_corr_3"] = (
        lagged_correlation(
            out["temperature"],
            out["frequency"],
            RELATION_WINDOW,
            3
        )
    )

    out["lag_corr_5"] = (
        lagged_correlation(
            out["temperature"],
            out["frequency"],
            RELATION_WINDOW,
            5
        )
    )

    # --------------------------------------------------------
    # Rolling covariance
    # --------------------------------------------------------

    out["rolling_covariance"] = (
        rolling_covariance(
            out["temperature"],
            out["frequency"],
            RELATION_WINDOW
        )
    )

    # --------------------------------------------------------
    # Slope coupling
    # --------------------------------------------------------

    temp_slope_positive = (
        np.maximum(
            out["temperature_slope"],
            0
        )
    )

    freq_slope_negative = (
        np.maximum(
            -out["frequency_slope"],
            0
        )
    )

    out["slope_coupling"] = (
        temp_slope_positive
        * freq_slope_negative
    )

    # --------------------------------------------------------
    # Directional relationship
    # --------------------------------------------------------

    out["directional_coupling"] = (
        (
            out["temperature_slope"] > 0
        )
        &
        (
            out["frequency_slope"] < 0
        )
    ).astype(int)

    return out


# ============================================================
# CALIBRATION
# ============================================================

def build_calibration_data():

    datasets = []

    for scenario in [
        "normal",
        "noisy_normal"
    ]:

        scenario_offset = (
            10000
            if scenario == "noisy_normal"
            else 0
        )

        for trial in range(
            CALIBRATION_TRIALS
        ):

            seed = (
                CALIBRATION_SEED
                + scenario_offset
                + trial
            )

            df = generate_data(
                scenario,
                seed
            )

            df = add_relationship_features(
                df
            )

            datasets.append(df)

    return datasets


# ============================================================
# ROBUST BASELINE STATISTICS
# ============================================================

def robust_stats(values):

    values = pd.Series(
        values
    ).dropna()

    median = float(
        values.median()
    )

    mad = float(
        np.median(
            np.abs(
                values - median
            )
        )
    )

    # Prevent zero MAD
    scale = max(
        1.4826 * mad,
        1e-9
    )

    return median, scale


# ============================================================
# CALIBRATE RELATIONSHIP BASELINES
# ============================================================

def calibrate_relationship_model(
    calibration_data
):
    """
    Learn the normal relationship distribution.

    Only normal and noisy-normal data are used.
    """

    feature_names = [

        "rolling_correlation",

        "lag_corr_1",

        "lag_corr_3",

        "lag_corr_5",

        "rolling_covariance",

        "slope_coupling",

    ]

    statistics = {}

    for feature in feature_names:

        values = []

        for df in calibration_data:

            baseline = df[
                df["sample"]
                < EVENT_START
            ]

            values.extend(
                baseline[feature]
                .dropna()
                .tolist()
            )

        median, scale = robust_stats(
            values
        )

        statistics[feature] = {

            "median":
                median,

            "scale":
                scale,
        }

    return statistics


# ============================================================
# RELATIONSHIP ANOMALY SCORE
# ============================================================

def calculate_relationship_score(
    df,
    statistics
):
    """
    Calculate robust anomaly scores for relationship features.

    Score approximately represents the number of robust standard
    deviations from normal behavior.
    """

    scores = pd.DataFrame(
        index=df.index
    )

    for feature, stats in (
        statistics.items()
    ):

        median = stats["median"]

        scale = stats["scale"]

        scores[
            feature
        ] = np.abs(
            df[feature] - median
        ) / scale

    # --------------------------------------------------------
    # Aggregate evidence
    # --------------------------------------------------------

    scores["relationship_score"] = (
        scores.mean(
            axis=1
        )
    )

    scores["relationship_max"] = (
        scores.max(
            axis=1
        )
    )

    return scores


# ============================================================
# RELATIONSHIP DETECTOR
# ============================================================

def relationship_detector(
    df,
    statistics,
    threshold
):
    """
    Relationship detector.

    Alert if multiple relationship features simultaneously
    deviate from their normal behavior.
    """

    scores = calculate_relationship_score(
        df,
        statistics
    )

    # Number of relationship features
    # with strong deviation
    evidence_count = (
        scores[
            [
                "rolling_correlation",
                "lag_corr_1",
                "lag_corr_3",
                "lag_corr_5",
                "rolling_covariance",
                "slope_coupling",
            ]
        ]
        > threshold
    ).sum(
        axis=1
    )

    raw_alert = (
        evidence_count >= 2
    )

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    persistent_alert = (
        raw_alert
        .rolling(
            PERSISTENCE_WINDOW,
            min_periods=PERSISTENCE_WINDOW
        )
        .sum()
        >= PERSISTENCE_REQUIRED
    )

    return persistent_alert.fillna(
        False
    )


# ============================================================
# HYBRID RELATIONSHIP DETECTOR
# ============================================================

def hybrid_relationship_detector(
    df,
    statistics,
    threshold
):
    """
    Combine direct physical evidence with relationship evidence.
    """

    relationship_scores = (
        calculate_relationship_score(
            df,
            statistics
        )
    )

    # Relationship evidence
    relationship_evidence = (
        relationship_scores[
            [
                "rolling_correlation",
                "lag_corr_1",
                "lag_corr_3",
                "lag_corr_5",
                "rolling_covariance",
                "slope_coupling",
            ]
        ]
        > threshold
    ).sum(
        axis=1
    )

    relationship_signal = (
        relationship_evidence >= 2
    )

    # --------------------------------------------------------
    # Direct temporal evidence
    # --------------------------------------------------------

    temp_slope_signal = (
        df["temperature_slope"]
        > 0.02
    )

    freq_slope_signal = (
        df["frequency_slope"]
        < -0.5
    )

    direct_signal = (
        temp_slope_signal
        | freq_slope_signal
    )

    # --------------------------------------------------------
    # Combined detector
    # --------------------------------------------------------

    combined_signal = (
        relationship_signal
        & direct_signal
    )

    # Persistence
    alert = (
        combined_signal
        .rolling(
            PERSISTENCE_WINDOW,
            min_periods=PERSISTENCE_WINDOW
        )
        .sum()
        >= PERSISTENCE_REQUIRED
    )

    return alert.fillna(
        False
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    df,
    prediction,
    method
):

    actual = (
        df["event"]
        .astype(bool)
        .to_numpy()
    )

    predicted = (
        prediction
        .astype(bool)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    tp = np.sum(
        predicted & actual
    )

    fp = np.sum(
        predicted & ~actual
    )

    fn = np.sum(
        ~predicted & actual
    )

    tn = np.sum(
        ~predicted & ~actual
    )

    # --------------------------------------------------------
    # Event detection
    # --------------------------------------------------------

    event_alerts = np.where(
        predicted & actual
    )[0]

    if len(event_alerts) > 0:

        detected = 1

        delay = (
            int(event_alerts[0])
            - EVENT_START
        )

    else:

        detected = 0

        delay = np.nan

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    return {

        "method":
            method,

        "TP":
            int(tp),

        "FP":
            int(fp),

        "FN":
            int(fn),

        "TN":
            int(tn),

        "precision":
            precision,

        "recall":
            recall,

        "event_detected":
            detected,

        "delay_samples":
            delay,

        "false_alarm_samples":
            int(fp),

        "false_alarm_trial":
            int(fp > 0),
    }


# ============================================================
# RUN TEST
# ============================================================

def run_test(
    statistics,
    relationship_threshold
):

    results = []

    for scenario_index, scenario in enumerate(
        SCENARIOS
    ):

        for trial in range(
            TEST_TRIALS
        ):

            seed = (
                TEST_SEED
                + scenario_index * 10000
                + trial
            )

            df = generate_data(
                scenario,
                seed
            )

            df = add_relationship_features(
                df
            )

            # ------------------------------------------------
            # Relationship detector
            # ------------------------------------------------

            relationship_prediction = (
                relationship_detector(
                    df,
                    statistics,
                    relationship_threshold
                )
            )

            # ------------------------------------------------
            # Hybrid relationship detector
            # ------------------------------------------------

            hybrid_prediction = (
                hybrid_relationship_detector(
                    df,
                    statistics,
                    relationship_threshold
                )
            )

            # ------------------------------------------------
            # Direct slope detector
            # ------------------------------------------------

            slope_prediction = (

                (
                    df["temperature_slope"]
                    > 0.02
                )

                |

                (
                    df["frequency_slope"]
                    < -0.5
                )
            )

            detectors = {

                "slope_baseline":
                    slope_prediction,

                "relationship":
                    relationship_prediction,

                "hybrid_relationship":
                    hybrid_prediction,
            }

            for method, prediction in (
                detectors.items()
            ):

                result = evaluate(
                    df,
                    prediction,
                    method
                )

                result.update({

                    "scenario":
                        scenario,

                    "trial":
                        trial,

                    "seed":
                        seed,

                    "source":
                        "SIMULATED",
                })

                results.append(
                    result
                )

    return pd.DataFrame(
        results
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize(
    results
):

    rows = []

    for (
        scenario,
        method
    ), group in results.groupby(
        [
            "scenario",
            "method"
        ]
    ):

        rows.append({

            "scenario":
                scenario,

            "method":
                method,

            "trials":
                len(group),

            "event_detection_rate":
                group[
                    "event_detected"
                ].mean(),

            "mean_delay_samples":
                group[
                    "delay_samples"
                ].mean(),

            "median_delay_samples":
                group[
                    "delay_samples"
                ].median(),

            "precision":
                group[
                    "precision"
                ].mean(),

            "recall":
                group[
                    "recall"
                ].mean(),

            "mean_false_alarm_samples":
                group[
                    "false_alarm_samples"
                ].mean(),

            "false_alarm_trial_rate":
                group[
                    "false_alarm_trial"
                ].mean(),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print("PHYSOS v2.3")
    print("TEMPORAL RELATIONSHIP ENGINE")
    print("=" * 75)

    print()
    print("Research question:")
    print(
        "Can temporal relationships between heterogeneous "
        "sensors reveal degradation?"
    )

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    print()
    print("1. Generating calibration data...")

    calibration_data = (
        build_calibration_data()
    )

    print(
        f"Calibration datasets: "
        f"{len(calibration_data)}"
    )

    print()
    print(
        "2. Learning normal temporal relationships..."
    )

    statistics = (
        calibrate_relationship_model(
            calibration_data
        )
    )

    print()
    print(
        "CALIBRATION STATISTICS"
    )

    print("-" * 75)

    for feature, stats in (
        statistics.items()
    ):

        print(
            f"{feature:25s}"
            f" median={stats['median']:.6f}"
            f" scale={stats['scale']:.6f}"
        )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    relationship_threshold = 3.0

    print()
    print(
        "Relationship anomaly threshold:",
        relationship_threshold
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    print()
    print(
        "3. Running NEW untouched test dataset..."
    )

    results = run_test(
        statistics,
        relationship_threshold
    )

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    detailed_file = (
        DATA_DIR
        / "physos_v23_relationship_trials.csv"
    )

    results.to_csv(
        detailed_file,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = summarize(
        results
    )

    summary_file = (
        DATA_DIR
        / "physos_v23_relationship_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    # --------------------------------------------------------
    # Save calibration model
    # --------------------------------------------------------

    model_file = (
        DATA_DIR
        / "physos_v23_relationship_model.json"
    )

    model = {

        "version":
            "PHYSOS v2.3",

        "source":
            "SIMULATED",

        "slope_window":
            SLOPE_WINDOW,

        "relationship_window":
            RELATION_WINDOW,

        "persistence_window":
            PERSISTENCE_WINDOW,

        "persistence_required":
            PERSISTENCE_REQUIRED,

        "relationship_threshold":
            relationship_threshold,

        "statistics":
            statistics,
    }

    with open(
        model_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            model,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("PHYSOS v2.3 RESULTS")
    print("=" * 75)

    for scenario in SCENARIOS:

        print()
        print(
            f"SCENARIO: {scenario}"
        )

        print("-" * 75)

        subset = summary[
            summary["scenario"]
            == scenario
        ]

        display_columns = [

            "method",

            "event_detection_rate",

            "mean_delay_samples",

            "precision",

            "recall",

            "false_alarm_trial_rate",
        ]

        print(
            subset[
                display_columns
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Hidden relationship result
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("HIDDEN RELATIONSHIP EXPERIMENT")
    print("=" * 75)

    hidden = summary[
        summary["scenario"]
        == "hidden_relationship"
    ]

    print(
        hidden[
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
    # Save configuration
    # --------------------------------------------------------

    config = {

        "experiment":
            "PHYSOS v2.3",

        "description":
            "Temporal relationship engine",

        "data_source":
            "SIMULATED",

        "n_samples":
            N_SAMPLES,

        "event_start":
            EVENT_START,

        "event_end":
            EVENT_END,

        "slope_window":
            SLOPE_WINDOW,

        "relationship_window":
            RELATION_WINDOW,

        "calibration_trials":
            CALIBRATION_TRIALS,

        "test_trials":
            TEST_TRIALS,

        "calibration_seed":
            CALIBRATION_SEED,

        "test_seed":
            TEST_SEED,

        "relationship_threshold":
            relationship_threshold,
    }

    config_file = (
        DATA_DIR
        / "physos_v23_config.json"
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

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("FILES CREATED")
    print("=" * 75)

    print(
        detailed_file
    )

    print(
        summary_file
    )

    print(
        model_file
    )

    print(
        config_file
    )

    print()
    print(
        "PHYSOS v2.3 COMPLETE."
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()