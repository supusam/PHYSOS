
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

N = 600
START, END = 300, 360
WINDOW = 20

CALIBRATION_TRIALS = 20
TEST_TRIALS = 30
MAX_CALIBRATION_FPR = 0.02

TEMP_THRESHOLDS = [0.01, 0.02, 0.03, 0.04, 0.05]
FREQ_THRESHOLDS = [-1.0, -2.0, -3.0, -4.0, -5.0]


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


def generate(seed, inject_drift):
    rng = np.random.default_rng(seed)

    temperature = 35 + rng.normal(0, 0.5, N)
    frequency = 2400 + rng.normal(0, 45, N)

    if inject_drift:
        ramp = np.linspace(0, 1, END - START)
        temperature[START:END] += 5 * ramp
        frequency[START:END] -= 250 * ramp

    df = pd.DataFrame({
        "temperature_c": temperature,
        "cpu_frequency_mhz": np.clip(frequency, 0, None),
    })

    df["temp_slope"] = rolling_slope(
        df["temperature_c"], WINDOW
    )
    df["freq_slope"] = rolling_slope(
        df["cpu_frequency_mhz"], WINDOW
    )

    return df


def score_trial(df, inject_drift, temp_threshold, freq_threshold):
    predictions = (
        (df["temp_slope"] > temp_threshold)
        & (df["freq_slope"] < freq_threshold)
    ).fillna(False).to_numpy(dtype=bool)

    labels = np.zeros(N, dtype=bool)
    if inject_drift:
        labels[START:END] = True

    return {
        "TP": int(np.sum(predictions & labels)),
        "FP": int(np.sum(predictions & ~labels)),
        "FN": int(np.sum(~predictions & labels)),
        "TN": int(np.sum(~predictions & ~labels)),
    }


def evaluate(seeds, temp_threshold, freq_threshold):
    totals = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

    for seed in seeds:
        for scenario, inject in [
            ("normal", False),
            ("gradual_drift", True),
        ]:
            df = generate(seed, inject)
            scores = score_trial(
                df, inject, temp_threshold, freq_threshold
            )

            for key in totals:
                totals[key] += scores[key]

    recall_denominator = totals["TP"] + totals["FN"]
    fpr_denominator = totals["FP"] + totals["TN"]

    totals["recall"] = (
        totals["TP"] / recall_denominator
        if recall_denominator else np.nan
    )
    totals["false_positive_rate"] = (
        totals["FP"] / fpr_denominator
        if fpr_denominator else np.nan
    )

    return totals


def main():
    # Calibration and test use disjoint random seeds.
    calibration_seeds = range(3000, 3000 + CALIBRATION_TRIALS)
    test_seeds = range(9000, 9000 + TEST_TRIALS)

    candidates = []

    for temp_threshold in TEMP_THRESHOLDS:
        for freq_threshold in FREQ_THRESHOLDS:
            metrics = evaluate(
                calibration_seeds,
                temp_threshold,
                freq_threshold,
            )

            candidates.append({
                "temp_threshold": temp_threshold,
                "freq_threshold": freq_threshold,
                **metrics,
            })

    grid = pd.DataFrame(candidates)

    eligible = grid[
        grid["false_positive_rate"] <= MAX_CALIBRATION_FPR
    ]

    if eligible.empty:
        print(
            "No candidate met the calibration false-positive "
            "rate limit. Choose a different threshold grid or "
            "review the detector design."
        )
        print(grid.sort_values("false_positive_rate").to_string(index=False))
        return

    # Maximize calibration recall; use lower FPR to break ties.
    chosen = eligible.sort_values(
        ["recall", "false_positive_rate"],
        ascending=[False, True],
    ).iloc[0]

    temp_threshold = float(chosen["temp_threshold"])
    freq_threshold = float(chosen["freq_threshold"])

    test_metrics = evaluate(
        test_seeds,
        temp_threshold,
        freq_threshold,
    )

    output = ROOT / "data" / "calibration_v13_grid.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output, index=False)

    print("\nPHYSOS v1.3 — THRESHOLD CALIBRATION")
    print("=" * 72)
    print(f"Calibration trials: {CALIBRATION_TRIALS}")
    print(f"Held-out test trials: {TEST_TRIALS}")
    print(f"Maximum calibration FPR: {MAX_CALIBRATION_FPR:.1%}")

    print("\nSelected thresholds:")
    print(f"Temperature slope > {temp_threshold}")
    print(f"Frequency slope < {freq_threshold}")

    print("\nCalibration performance:")
    print(f"Recall: {chosen['recall']:.4f}")
    print(f"FPR:    {chosen['false_positive_rate']:.4f}")

    print("\nHELD-OUT TEST PERFORMANCE")
    for key, value in test_metrics.items():
        if isinstance(value, (float, np.floating)):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nFull threshold grid saved to:")
    print(output)
    print("\nSynthetic data only; not physical validation.")


if __name__ == "__main__":
    main()
