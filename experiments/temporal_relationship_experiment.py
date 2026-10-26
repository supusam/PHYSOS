
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"
TRIALS = 30
N = 600
BASELINE_END = 200
START, END = 300, 360
WINDOW = 20

def generate(scenario, seed):
    rng = np.random.default_rng(seed)

    cpu = 30 + rng.normal(0, 4, N)
    ram = 55 + rng.normal(0, 2.5, N)
    temp = 35 + rng.normal(0, 0.5, N)
    freq = 2400 + rng.normal(0, 45, N)
    truth = np.zeros(N, dtype=bool)

    if scenario != "normal":
        truth[START:END] = True

    if scenario == "cpu_only_spike":
        cpu[START:END] += rng.uniform(15, 35)

    elif scenario == "gradual_physical_drift":
        ramp = np.linspace(0, 1, END - START)
        temp[START:END] += rng.uniform(3, 7) * ramp
        freq[START:END] -= rng.uniform(150, 400) * ramp

    elif scenario == "multisignal_event":
        cpu[START:END] += rng.uniform(15, 25)
        ram[START:END] += rng.uniform(5, 10)
        temp[START:END] += np.linspace(
            0, rng.uniform(3, 6), END - START
        )
        freq[START:END] -= rng.uniform(150, 350)

    df = pd.DataFrame({
        "cpu_percent": np.clip(cpu, 0, 100),
        "ram_percent": np.clip(ram, 0, 100),
        "temperature_c": temp,
        "cpu_frequency_mhz": np.clip(freq, 0, None),
    })

    return df, truth

def detectors(df):
    base = df.iloc[:BASELINE_END]
    z = pd.DataFrame(index=df.index)

    for col in df.columns:
        sigma = base[col].std(ddof=1)
        if not np.isfinite(sigma) or sigma < 1e-9:
            sigma = 1.0
        z[col] = (df[col] - base[col].mean()) / sigma

    cpu_threshold = df["cpu_percent"].to_numpy() > 60

    multisignal = (
        (z.abs() > 3).sum(axis=1).to_numpy() >= 2
    )

    # New temporal feature:
    # compare each reading with the reading 20 samples earlier.
    temp_delta = (
        df["temperature_c"]
        - df["temperature_c"].shift(WINDOW)
    )
    freq_delta = (
        df["cpu_frequency_mhz"]
        - df["cpu_frequency_mhz"].shift(WINDOW)
    )

    # Both conditions must hold:
    # temperature increases by more than 1 degree,
    # frequency decreases by more than 50 MHz.
    raw_temporal = (
        (temp_delta > 1.0)
        & (freq_delta < -50.0)
    ).fillna(False)

    # Require the condition in at least 2 of the last 3 samples.
    temporal = (
        raw_temporal.astype(int)
        .rolling(3, min_periods=3)
        .sum()
        .ge(2)
        .fillna(False)
        .to_numpy()
    )

    return {
        "CPU threshold": cpu_threshold,
        "Multi-signal z-score": multisignal,
        "Temporal paired trend": temporal,
    }

def score(pred, truth):
    pred = np.asarray(pred, dtype=bool)
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    tn = int(np.sum(~pred & ~truth))

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "FPR": fp / (fp + tn) if fp + tn else np.nan,
        "event_detected": (
            bool(np.any(pred[START:END]))
            if np.any(truth) else np.nan
        ),
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
        for j, scenario in enumerate(scenarios):
            df, truth = generate(
                scenario, seed=203000 + trial * 10 + j
            )

            for name, pred in detectors(df).items():
                row = {
                    "trial": trial + 1,
                    "scenario": scenario,
                    "detector": name,
                }
                row.update(score(pred, truth))
                rows.append(row)

    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "temporal_trial_results.csv", index=False)

    summary = []

    for (scenario, detector), group in raw.groupby(
        ["scenario", "detector"], sort=False
    ):
        tp = int(group["TP"].sum())
        fp = int(group["FP"].sum())
        fn = int(group["FN"].sum())
        tn = int(group["TN"].sum())

        summary.append({
            "scenario": scenario,
            "detector": detector,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "pooled_precision": (
                tp / (tp + fp) if tp + fp else np.nan
            ),
            "pooled_recall": (
                tp / (tp + fn) if tp + fn else np.nan
            ),
            "pooled_FPR": (
                fp / (fp + tn) if fp + tn else np.nan
            ),
            "event_detection_rate": (
                group["event_detected"].mean()
                if scenario != "normal" else np.nan
            ),
        })

    result = pd.DataFrame(summary)
    result.to_csv(OUT / "temporal_summary.csv", index=False)

    print("\nPHYSOS v1.0 — TEMPORAL RELATIONSHIP EXPERIMENT")
    print("=" * 95)
    print(result.round(4).to_string(index=False))
    print(f"\nSummary saved to: {OUT / 'temporal_summary.csv'}")
    print("Synthetic benchmark only; no real hardware validation.")

if __name__ == "__main__":
    main()
