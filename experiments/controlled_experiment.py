
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = DATA / "experiment_results.csv"

RNG = np.random.default_rng(42)

def generate_dataset(n=600):
    """Synthetic benchmark with known injected anomaly intervals."""
    t = np.arange(n)

    cpu = 30 + RNG.normal(0, 3, n)
    ram = 55 + RNG.normal(0, 2, n)
    temp = 35 + RNG.normal(0, 0.4, n)
    frequency = 2400 + RNG.normal(0, 35, n)

    truth = np.zeros(n, dtype=int)

    # Inject a synthetic multi-signal event.
    # This is a test scenario, not a real hardware failure.
    start, end = 300, 360
    truth[start:end] = 1

    cpu[start:end] += 20
    ram[start:end] += 8
    temp[start:end] += np.linspace(0, 5, end - start)
    frequency[start:end] -= 250

    return pd.DataFrame({
        "time_step": t,
        "cpu_percent": np.clip(cpu, 0, 100),
        "ram_percent": np.clip(ram, 0, 100),
        "temperature_c": temp,
        "cpu_frequency_mhz": np.clip(frequency, 0, None),
        "ground_truth": truth,
    })

def evaluate(name, predicted, truth):
    predicted = np.asarray(predicted, dtype=bool)
    truth = np.asarray(truth, dtype=bool)

    tp = int(np.sum(predicted & truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    tn = int(np.sum(~predicted & ~truth))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "detector": name,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_positive_rate": round(fpr, 4),
        "F1": round(f1, 4),
    }

def main():
    df = generate_dataset()
    truth = df["ground_truth"].to_numpy()

    # Detector A: conventional fixed CPU threshold.
    threshold_prediction = df["cpu_percent"].to_numpy() > 60

    # Detector B: a multi-signal detector.
    # Standardize using the known pre-event baseline only.
    baseline = df.iloc[:200]

    signals = {
        "cpu": (df["cpu_percent"], baseline["cpu_percent"]),
        "ram": (df["ram_percent"], baseline["ram_percent"]),
        "temperature": (
            df["temperature_c"], baseline["temperature_c"]
        ),
        "frequency": (
            df["cpu_frequency_mhz"], baseline["cpu_frequency_mhz"]
        ),
    }

    z_scores = {}
    for name, (series, base) in signals.items():
        mean = base.mean()
        std = base.std(ddof=1)
        if not np.isfinite(std) or std < 1e-9:
            std = 1.0
        z_scores[name] = (series - mean) / std

    # An event is flagged when at least two signals are unusual.
    # Frequency changes in either direction count as unusual.
    unusual_count = (
        (z_scores["cpu"].abs() > 3)
        .astype(int)
        + (z_scores["ram"].abs() > 3).astype(int)
        + (z_scores["temperature"].abs() > 3).astype(int)
        + (z_scores["frequency"].abs() > 3).astype(int)
    )

    multi_signal_prediction = unusual_count >= 2

    results = pd.DataFrame([
        evaluate("CPU threshold > 60%", threshold_prediction, truth),
        evaluate("Multi-signal z-score", multi_signal_prediction, truth),
    ])

    DATA.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)

    print("\nPHYSOS v0.7 — SYNTHETIC CONTROLLED EXPERIMENT")
    print("=" * 58)
    print(results.to_string(index=False))
    print(f"\nResults saved to: {OUTPUT}")
    print("\nIMPORTANT: These are synthetic benchmark results.")
    print("They do not establish real-world predictive performance.")

if __name__ == "__main__":
    main()
