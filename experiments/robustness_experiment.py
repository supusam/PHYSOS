
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "robustness_results.csv"
RNG = np.random.default_rng(2026)

N = 600
BASELINE_END = 200
EVENT_START = 300
EVENT_END = 360

def make_data(scenario):
    cpu = 30 + RNG.normal(0, 4, N)
    ram = 55 + RNG.normal(0, 2.5, N)
    temp = 35 + RNG.normal(0, 0.5, N)
    freq = 2400 + RNG.normal(0, 45, N)
    truth = np.zeros(N, dtype=int)

    s, e = EVENT_START, EVENT_END

    if scenario == "cpu_only_spike":
        truth[s:e] = 1
        cpu[s:e] += 25

    elif scenario == "gradual_physical_drift":
        truth[s:e] = 1
        ramp = np.linspace(0, 1, e - s)
        temp[s:e] += 5 * ramp
        freq[s:e] -= 300 * ramp

    elif scenario == "multisignal_event":
        truth[s:e] = 1
        cpu[s:e] += 20
        ram[s:e] += 8
        temp[s:e] += np.linspace(0, 5, e - s)
        freq[s:e] -= 250

    elif scenario == "normal":
        # No injected event: labels remain zero.
        pass

    return pd.DataFrame({
        "cpu_percent": np.clip(cpu, 0, 100),
        "ram_percent": np.clip(ram, 0, 100),
        "temperature_c": temp,
        "cpu_frequency_mhz": np.clip(freq, 0, None),
        "ground_truth": truth,
    })

def detect(df):
    baseline = df.iloc[:BASELINE_END]
    features = [
        "cpu_percent",
        "ram_percent",
        "temperature_c",
        "cpu_frequency_mhz",
    ]

    z = pd.DataFrame(index=df.index)

    for col in features:
        mu = baseline[col].mean()
        sigma = baseline[col].std(ddof=1)
        if not np.isfinite(sigma) or sigma < 1e-9:
            sigma = 1.0
        z[col] = (df[col] - mu) / sigma

    # Fixed conventional CPU threshold.
    cpu_only = df["cpu_percent"].to_numpy() > 60

    # Flag when at least two signals exceed 3 baseline standard deviations.
    multi_signal = (z.abs() > 3).sum(axis=1).to_numpy() >= 2

    # Experimental drift detector: temperature rises and frequency falls.
    drift = (
        (z["temperature_c"].to_numpy() > 2)
        & (z["cpu_frequency_mhz"].to_numpy() < -2)
    )

    return {
        "CPU threshold": cpu_only,
        "Multi-signal z-score": multi_signal,
        "Temperature/frequency drift": drift,
    }

def metrics(pred, truth):
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)

    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": round(tp / (tp + fp), 3)
            if tp + fp else 0.0,
        "recall": round(tp / (tp + fn), 3)
            if tp + fn else 0.0,
        "false_positive_rate": round(fp / (fp + tn), 3)
            if fp + tn else 0.0,
    }

def main():
    scenarios = [
        "normal",
        "cpu_only_spike",
        "gradual_physical_drift",
        "multisignal_event",
    ]
    rows = []

    for scenario in scenarios:
        df = make_data(scenario)
        truth = df["ground_truth"].to_numpy()

        for detector_name, prediction in detect(df).items():
            row = {
                "scenario": scenario,
                "detector": detector_name,
            }
            row.update(metrics(prediction, truth))
            rows.append(row)

    result = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)

    print("\nPHYSOS v0.8 — ROBUSTNESS EXPERIMENT")
    print("=" * 78)
    print(result.to_string(index=False))
    print(f"\nSaved to: {OUT}")
    print("\nSynthetic results only; not validated hardware predictions.")

if __name__ == "__main__":
    main()
