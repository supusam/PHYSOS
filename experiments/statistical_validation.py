
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

def generate_data(scenario, rng):
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

def detect(df):
    baseline = df.iloc[:BASELINE_END]
    columns = [
        "cpu_percent",
        "ram_percent",
        "temperature_c",
        "cpu_frequency_mhz",
    ]

    z = pd.DataFrame(index=df.index)

    for col in columns:
        mean = baseline[col].mean()
        std = baseline[col].std(ddof=1)
        if not np.isfinite(std) or std < 1e-9:
            std = 1.0
        z[col] = (df[col] - mean) / std

    cpu_threshold = df["cpu_percent"].to_numpy() > 60
    multi_signal = (z.abs() > 3).sum(axis=1).to_numpy() >= 2
    drift = (
        (z["temperature_c"].to_numpy() > 2)
        & (z["cpu_frequency_mhz"].to_numpy() < -2)
    )

    return {
        "CPU threshold": cpu_threshold,
        "Multi-signal z-score": multi_signal,
        "Temperature/frequency drift": drift,
    }

def calculate_metrics(pred, truth):
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)

    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    tn = int(np.sum(~pred & ~truth))

    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan

    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision)
        and np.isfinite(recall)
        and precision + recall > 0
        else np.nan
    )

    event_detected = (
        bool(np.any(pred[EVENT_START:EVENT_END]))
        if np.any(truth)
        else np.nan
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "F1": f1,
        "event_detected": event_detected,
    }

def main():
    scenarios = [
        "normal",
        "cpu_only_spike",
        "gradual_physical_drift",
        "multisignal_event",
    ]

    trial_rows = []

    for trial in range(TRIALS):
        for scenario_index, scenario in enumerate(scenarios):
            # Independent reproducible random stream for each trial/scenario.
            seed = 202600 + trial * 100 + scenario_index
            rng = np.random.default_rng(seed)

            df, truth = generate_data(scenario, rng)
            predictions = detect(df)

            for detector, pred in predictions.items():
                row = {
                    "trial": trial + 1,
                    "scenario": scenario,
                    "detector": detector,
                }
                row.update(calculate_metrics(pred, truth))
                trial_rows.append(row)

    trials = pd.DataFrame(trial_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    trials.to_csv(OUT / "validation_trial_results.csv", index=False)

    # Aggregate counts first, then calculate pooled sample-level metrics.
    summaries = []

    for (scenario, detector), group in trials.groupby(
        ["scenario", "detector"], sort=False
    ):
        tp = int(group["TP"].sum())
        fp = int(group["FP"].sum())
        fn = int(group["FN"].sum())
        tn = int(group["TN"].sum())

        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        fpr = fp / (fp + tn) if fp + tn else np.nan
        f1 = (
            2 * precision * recall / (precision + recall)
            if np.isfinite(precision)
            and np.isfinite(recall)
            and precision + recall > 0
            else np.nan
        )

        summaries.append({
            "scenario": scenario,
            "detector": detector,
            "trials": len(group),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "pooled_precision": precision,
            "pooled_recall": recall,
            "pooled_FPR": fpr,
            "pooled_F1": f1,
            "event_detection_rate": (
                group["event_detected"].mean()
                if scenario != "normal" else np.nan
            ),
            "recall_trial_std": group["recall"].std(),
        })

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "validation_summary.csv", index=False)

    print("\nPHYSOS v0.9 — STATISTICAL VALIDATION")
    print("=" * 100)
    print(summary.round(4).to_string(index=False))
    print(f"\nTrial-level data: {OUT / 'validation_trial_results.csv'}")
    print(f"Summary:          {OUT / 'validation_summary.csv'}")
    print("\nSynthetic benchmark only; not evidence of real-world prediction.")

if __name__ == "__main__":
    main()
