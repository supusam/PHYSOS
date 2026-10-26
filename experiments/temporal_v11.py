
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"

TRIALS = 30
N = 600
BASELINE_END = 200
EVENT_START = 300
EVENT_END = 360
WINDOW = 20

def generate(scenario, seed):
    rng = np.random.default_rng(seed)

    cpu = 30 + rng.normal(0, 4, N)
    ram = 55 + rng.normal(0, 2.5, N)
    temp = 35 + rng.normal(0, 0.5, N)
    freq = 2400 + rng.normal(0, 45, N)
    truth = np.zeros(N, dtype=bool)

    s, e = EVENT_START, EVENT_END
    length = e - s

    if scenario != "normal":
        truth[s:e] = True

    if scenario == "cpu_only_spike":
        cpu[s:e] += rng.uniform(15, 35)

    elif scenario == "gradual_physical_drift":
        ramp = np.linspace(0, 1, length)
        temp[s:e] += rng.uniform(3, 7) * ramp
        freq[s:e] -= rng.uniform(150, 400) * ramp

    elif scenario == "multisignal_event":
        cpu[s:e] += rng.uniform(15, 25)
        ram[s:e] += rng.uniform(5, 10)
        temp[s:e] += np.linspace(0, rng.uniform(3, 6), length)
        freq[s:e] -= rng.uniform(150, 350)

    df = pd.DataFrame({
        "cpu_percent": np.clip(cpu, 0, 100),
        "ram_percent": np.clip(ram, 0, 100),
        "temperature_c": temp,
        "cpu_frequency_mhz": np.clip(freq, 0, None),
    })

    return df, truth

def rolling_slope(series, window):
    """Least-squares slope for each rolling window."""
    x = np.arange(window, dtype=float)
    x -= x.mean()

    denominator = np.sum(x ** 2)

    def slope(values):
        y = values - np.mean(values)
        return np.dot(x, y) / denominator

    return series.rolling(window, min_periods=window).apply(
        slope, raw=True
    )

def detect(df):
    baseline = df.iloc[:BASELINE_END]
    z = pd.DataFrame(index=df.index)

    for col in df.columns:
        mean = baseline[col].mean()
        std = baseline[col].std(ddof=1)

        if not np.isfinite(std) or std < 1e-9:
            std = 1.0

        z[col] = (df[col] - mean) / std

    # Existing baseline 1: fixed CPU threshold.
    cpu_threshold = df["cpu_percent"].to_numpy() > 60

    # Existing baseline 2: at least two unusual signals.
    multisignal = (
        (z.abs() > 3).sum(axis=1).to_numpy() >= 2
    )

    temp = df["temperature_c"]
    freq = df["cpu_frequency_mhz"]

    # New feature A: slopes across a 20-sample window.
    temp_slope = rolling_slope(temp, WINDOW)
    freq_slope = rolling_slope(freq, WINDOW)

    # New feature B: correlation between sensor changes.
    temp_change = temp.diff()
    freq_change = freq.diff()

    rolling_corr = temp_change.rolling(
        WINDOW, min_periods=WINDOW
    ).corr(freq_change)

    # Experimental thresholds; not physical safety limits.
    raw_candidate = (
        (temp_slope > 0.02)
        & (freq_slope < -2.0)
        & (rolling_corr < -0.30)
    ).fillna(False)

    # Persistence: candidate must hold in at least 2 of
    # the last 3 samples to produce an alert.
    temporal_alert = (
        raw_candidate.astype(int)
        .rolling(3, min_periods=3)
        .sum()
        .ge(2)
        .fillna(False)
        .to_numpy()
    )

    return {
        "CPU threshold": cpu_threshold,
        "Multi-signal z-score": multisignal,
        "Temporal slope-correlation": temporal_alert,
    }

def evaluate(prediction, truth):
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(truth, dtype=bool)

    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    tn = int(np.sum(~pred & ~truth))

    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan

    # Detection delay is measured from event onset.
    # Only detections within the injected event count.
    event_prediction = pred[EVENT_START:EVENT_END]
    detected_indices = np.flatnonzero(event_prediction)

    delay = (
        int(detected_indices[0])
        if len(detected_indices) else np.nan
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "FPR": fpr,
        "event_detected": (
            bool(len(detected_indices))
            if np.any(truth) else np.nan
        ),
        "delay_samples": delay,
    }

def main():
    scenarios = [
        "normal",
        "cpu_only_spike",
        "gradual_physical_drift",
        "multisignal_event",
    ]

    rows = []

    for trial in range(TRIALS):
        for scenario_index, scenario in enumerate(scenarios):
            seed = 204000 + trial * 100 + scenario_index
            df, truth = generate(scenario, seed)

            for detector, prediction in detect(df).items():
                row = {
                    "trial": trial + 1,
                    "scenario": scenario,
                    "detector": detector,
                }
                row.update(evaluate(prediction, truth))
                rows.append(row)

    raw = pd.DataFrame(rows)

    OUT.mkdir(parents=True, exist_ok=True)
    raw.to_csv(OUT / "v11_trial_results.csv", index=False)

    summaries = []

    for (scenario, detector), group in raw.groupby(
        ["scenario", "detector"], sort=False
    ):
        tp = int(group["TP"].sum())
        fp = int(group["FP"].sum())
        fn = int(group["FN"].sum())
        tn = int(group["TN"].sum())

        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        fpr = fp / (fp + tn) if fp + tn else np.nan

        valid_delays = group["delay_samples"].dropna()

        summaries.append({
            "scenario": scenario,
            "detector": detector,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "pooled_precision": precision,
            "pooled_recall": recall,
            "pooled_FPR": fpr,
            "event_detection_rate": (
                group["event_detected"].mean()
                if scenario != "normal" else np.nan
            ),
            "mean_detection_delay_samples": (
                valid_delays.mean()
                if len(valid_delays) else np.nan
            ),
        })

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "v11_summary.csv", index=False)

    print("\nPHYSOS v1.1 — TEMPORAL RELATIONSHIP ENGINE")
    print("=" * 105)
    print(summary.round(4).to_string(index=False))
    print(f"\nTrial data: {OUT / 'v11_trial_results.csv'}")
    print(f"Summary:    {OUT / 'v11_summary.csv'}")
    print("\nSynthetic benchmark only; no real hardware validation.")

if __name__ == "__main__":
    main()
