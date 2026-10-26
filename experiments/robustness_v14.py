
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

N = 600
START, END = 300, 360
WINDOW = 20
TRIALS = 30

# Freeze the v1.3 thresholds. Do not tune on these test scenarios.
TEMP_THRESHOLD = 0.03
FREQ_THRESHOLD = -1.0


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

    temp_noise = 0.5
    freq_noise = 45.0

    if scenario == "noisy_gradual_drift":
        temp_noise = 0.8
        freq_noise = 70.0

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

    # "normal" has no injected event.
    return pd.DataFrame({
        "temperature_c": temperature,
        "cpu_frequency_mhz": np.clip(frequency, 0, None),
    })


def evaluate_trial(df, scenario):
    temp_slope = rolling_slope(
        df["temperature_c"], WINDOW
    )
    freq_slope = rolling_slope(
        df["cpu_frequency_mhz"], WINDOW
    )

    predictions = (
        (temp_slope > TEMP_THRESHOLD)
        & (freq_slope < FREQ_THRESHOLD)
    ).fillna(False).to_numpy(dtype=bool)

    labels = np.zeros(N, dtype=bool)

    if scenario != "normal":
        labels[START:END] = True

    return {
        "TP": int(np.sum(predictions & labels)),
        "FP": int(np.sum(predictions & ~labels)),
        "FN": int(np.sum(~predictions & labels)),
        "TN": int(np.sum(~predictions & ~labels)),
    }


def main():
    scenarios = [
        "normal",
        "gradual_drift",
        "weak_gradual_drift",
        "temperature_only",
        "frequency_only",
        "sudden_change",
        "noisy_gradual_drift",
    ]

    rows = []

    for scenario_index, scenario in enumerate(scenarios):
        for trial in range(TRIALS):
            # Independent seeds for each scenario and trial.
            seed = 50000 + scenario_index * 1000 + trial

            df = generate(seed, scenario)
            metrics = evaluate_trial(df, scenario)

            rows.append({
                "scenario": scenario,
                "trial": trial + 1,
                **metrics,
            })

    results = pd.DataFrame(rows)

    output = ROOT / "data" / "robustness_v14_trials.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)

    summary = results.groupby("scenario")[
        ["TP", "FP", "FN", "TN"]
    ].sum().reset_index()

    positive_count = summary["TP"] + summary["FN"]
    negative_count = summary["FP"] + summary["TN"]

    summary["recall"] = (
        summary["TP"] / positive_count.replace(0, np.nan)
    )
    summary["false_positive_rate"] = (
        summary["FP"] / negative_count.replace(0, np.nan)
    )

    summary_path = ROOT / "data" / "robustness_v14_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nPHYSOS v1.4 — CROSS-CONDITION ROBUSTNESS")
    print("=" * 90)
    print(f"Trials per scenario: {TRIALS}")
    print(f"Frozen temperature slope threshold: > {TEMP_THRESHOLD}")
    print(f"Frozen frequency slope threshold: < {FREQ_THRESHOLD}")
    print(f"Event window: samples {START}–{END - 1}")

    print("\nPooled results:")
    print(summary.round(4).to_string(index=False))

    print("\nSaved:")
    print(output)
    print(summary_path)

    print(
        "\nAll observations are synthetic. "
        "No real hardware or physical failure was tested."
    )


if __name__ == "__main__":
    main()
