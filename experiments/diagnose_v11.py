
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
N = 600
START, END = 300, 360
BASELINE_END = 200
WINDOW = 20

def rolling_slope(series, window):
    x = np.arange(window, dtype=float)
    x -= x.mean()

    def calc(y):
        y = y - y.mean()
        return np.dot(x, y) / np.sum(x ** 2)

    return series.rolling(window, min_periods=window).apply(
        calc, raw=True
    )

def generate(seed=204002):
    rng = np.random.default_rng(seed)

    cpu = 30 + rng.normal(0, 4, N)
    ram = 55 + rng.normal(0, 2.5, N)
    temp = 35 + rng.normal(0, 0.5, N)
    freq = 2400 + rng.normal(0, 45, N)

    ramp = np.linspace(0, 1, END - START)
    temp[START:END] += 5 * ramp
    freq[START:END] -= 250 * ramp

    return pd.DataFrame({
        "cpu_percent": np.clip(cpu, 0, 100),
        "ram_percent": np.clip(ram, 0, 100),
        "temperature_c": temp,
        "cpu_frequency_mhz": np.clip(freq, 0, None),
    })

def main():
    df = generate()
    base = df.iloc[:BASELINE_END]

    temp_slope = rolling_slope(df["temperature_c"], WINDOW)
    freq_slope = rolling_slope(df["cpu_frequency_mhz"], WINDOW)

    temp_delta = df["temperature_c"].diff()
    freq_delta = df["cpu_frequency_mhz"].diff()

    corr = temp_delta.rolling(
        WINDOW, min_periods=WINDOW
    ).corr(freq_delta)

    print("\nPHYSOS v1.2 — TEMPORAL DIAGNOSTICS")
    print("=" * 72)

    print("\nBaseline statistics:")
    print(base.describe().round(3).to_string())

    print("\nEvent-window statistics (samples 300–359):")
    print(df.iloc[START:END].describe().round(3).to_string())

    print("\nTemporal feature statistics:")
    features = pd.DataFrame({
        "temperature_slope": temp_slope,
        "frequency_slope": freq_slope,
        "change_correlation": corr,
    })

    print(features.iloc[START:END].describe().round(4).to_string())

    print("\nThreshold pass counts inside the event:")
    event = features.iloc[START:END]

    checks = {
        "temperature slope > 0.02": event["temperature_slope"] > 0.02,
        "frequency slope < -2.0": event["frequency_slope"] < -2.0,
        "correlation < -0.30": event["change_correlation"] < -0.30,
    }

    for name, mask in checks.items():
        print(f"{name}: {int(mask.sum())} / {len(mask)}")

    combined = (
        checks["temperature slope > 0.02"]
        & checks["frequency slope < -2.0"]
        & checks["correlation < -0.30"]
    )

    print(f"All three conditions simultaneously: {int(combined.sum())} / {len(event)}")

    print("\nThis diagnostic uses one synthetic trial, not real hardware data.")

if __name__ == "__main__":
    main()
