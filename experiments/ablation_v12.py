
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

N = 600
START, END = 300, 360
WINDOW = 20
TRIALS = 30

TEMP_THRESHOLD = 0.02
FREQ_THRESHOLD = -2.0
CORR_THRESHOLD = -0.30


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

    return pd.DataFrame({
        "temperature_c": temperature,
        "cpu_frequency_mhz": np.clip(frequency, 0, None),
    })


def get_features(df):
    temp = df["temperature_c"]
    freq = df["cpu_frequency_mhz"]

    temp_slope = rolling_slope(temp, WINDOW)
    freq_slope = rolling_slope(freq, WINDOW)

    temp_change = temp.diff()
    freq_change = freq.diff()

    change_corr = temp_change.rolling(
        WINDOW, min_periods=WINDOW
    ).corr(freq_change)

    return pd.DataFrame({
        "temperature": temp_slope > TEMP_THRESHOLD,
        "frequency": freq_slope < FREQ_THRESHOLD,
        "correlation": change_corr < CORR_THRESHOLD,
    }).fillna(False)


def detector_variants(features):
    t = features["temperature"]
    f = features["frequency"]
    c = features["correlation"]

    return {
        "temperature_only": t,
        "frequency_only": f,
        "correlation_only": c,
        "temperature_and_frequency": t & f,
        "temperature_and_correlation": t & c,
        "all_three": t & f & c,
    }


def main():
    rows = []

    for trial in range(TRIALS):
        seed = 204002 + trial

        for scenario, inject in [
            ("normal", False),
            ("gradual_drift", True),
        ]:
            df = generate(seed, inject)
            features = get_features(df)
            detectors = detector_variants(features)

            labels = np.zeros(N, dtype=bool)
            if inject:
                labels[START:END] = True

            for name, predictions in detectors.items():
                predictions = predictions.to_numpy(dtype=bool)

                tp = int(np.sum(predictions & labels))
                fp = int(np.sum(predictions & ~labels))
                fn = int(np.sum(~predictions & labels))
                tn = int(np.sum(~predictions & ~labels))

                rows.append({
                    "trial": trial + 1,
                    "scenario": scenario,
                    "detector": name,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "TN": tn,
                })

    results = pd.DataFrame(rows)
    output = ROOT / "data" / "ablation_v12_trials.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)

    summary = results.groupby(
        ["scenario", "detector"]
    )[["TP", "FP", "FN", "TN"]].sum().reset_index()

    summary["recall"] = (
        summary["TP"] /
        (summary["TP"] + summary["FN"]).replace(0, np.nan)
    )
    summary["false_positive_rate"] = (
        summary["FP"] /
        (summary["FP"] + summary["TN"]).replace(0, np.nan)
    )

    summary_path = ROOT / "data" / "ablation_v12_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nPHYSOS v1.2 — ABLATION STUDY")
    print("=" * 76)
    print("Trials:", TRIALS)
    print("Scenarios: normal and gradual drift")
    print("Event window: samples 300–359")
    print("\nPooled performance:")
    print(summary.round(4).to_string(index=False))

    print("\nSaved:")
    print(output)
    print(summary_path)
    print("\nSynthetic data only; no physical hardware validation.")


if __name__ == "__main__":
    main()
