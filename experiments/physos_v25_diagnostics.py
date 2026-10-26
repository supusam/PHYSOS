

"""
PHYSOS v2.5 — Experiment Diagnostics
Input:
    data/physos_v24_test_trials.csv

Outputs:
    data/physos_v25_diagnostics_summary.csv
    data/physos_v25_quality_report.txt

All source records must be labeled SIMULATED.
This script analyzes recorded results; it does not retrain detectors.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

INPUT_FILE = DATA_DIR / "physos_v24_test_trials.csv"
SUMMARY_FILE = DATA_DIR / "physos_v25_diagnostics_summary.csv"
REPORT_FILE = DATA_DIR / "physos_v25_quality_report.txt"

EXPECTED_SOURCE = "SIMULATED"

REQUIRED_COLUMNS = {
    "method",
    "TP",
    "FP",
    "FN",
    "TN",
    "precision",
    "recall",
    "event_detected",
    "delay_samples",
    "false_alarm_samples",
    "false_alarm_trial",
    "source",
    "scenario",
    "trial",
    "seed",
}


# --------------------------------------------------
# 2. HELPERS
# --------------------------------------------------

def parse_binary_column(series, column_name):
    """Convert a Boolean or 0/1 column to integers safely."""
    normalized = series.astype("string").str.strip().str.lower()

    mapping = {
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
    }

    converted = normalized.map(mapping)

    if converted.isna().any():
        bad_values = series[converted.isna()].unique().tolist()
        raise ValueError(
            f"Invalid values in '{column_name}': {bad_values}"
        )

    return converted.astype(int)


def wilson_interval(successes, total, z=1.959963984540054):
    """Calculate a two-sided 95% Wilson confidence interval."""
    if total <= 0:
        return np.nan, np.nan

    p = successes / total
    denominator = 1 + (z * z / total)

    center = (
        p + z * z / (2 * total)
    ) / denominator

    margin = (
        z
        * np.sqrt(
            p * (1 - p) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )

    return max(0.0, center - margin), min(1.0, center + margin)


def safe_mean(series):
    """Return a mean or NaN if no valid observations exist."""
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return np.nan

    return float(values.mean())


def format_number(value, digits=4):
    """Format a number for the text report."""
    if pd.isna(value):
        return "N/A"

    return f"{value:.{digits}f}"


# --------------------------------------------------
# 3. LOAD AND VALIDATE THE DATASET
# --------------------------------------------------

def main():
    print("=" * 70)
    print("PHYSOS v2.5 — EXPERIMENT DIAGNOSTICS")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_FILE}\n"
            "Run the v2.4 experiment first or check the file path."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    print(f"\nInput file: {INPUT_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    if df.empty:
        raise ValueError("The input dataset is empty.")

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    # --------------------------------------------------
    # 4. SOURCE PROVENANCE CHECK
    # --------------------------------------------------

    source_labels = (
        df["source"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    print("\nSource counts:")
    print(source_labels.value_counts(dropna=False).to_string())

    if source_labels.isna().any():
        raise ValueError(
            "Missing source labels found. Inspect the input CSV."
        )

    unexpected = sorted(
        set(source_labels.unique()) - {EXPECTED_SOURCE}
    )

    if unexpected:
        raise ValueError(
            f"Unexpected source labels: {unexpected}. "
            f"Expected only {EXPECTED_SOURCE}."
        )

    print("\nPASS: All records are labeled SIMULATED.")

    # Normalize the in-memory representation only.
    # The original input CSV is not modified.
    df["source"] = source_labels

    # --------------------------------------------------
    # 5. CHECK BASIC DATA QUALITY
    # --------------------------------------------------

    for column in ["method", "scenario", "trial", "seed"]:
        if df[column].isna().any():
            raise ValueError(
                f"Missing values found in required identifier: {column}"
            )

    for column in [
        "TP", "FP", "FN", "TN",
        "precision", "recall",
        "delay_samples", "false_alarm_samples",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[["TP", "FP", "FN", "TN"]].isna().any().any():
        raise ValueError(
            "TP, FP, FN, and TN must contain valid numeric values."
        )

    if (df[["TP", "FP", "FN", "TN"]] < 0).any().any():
        raise ValueError("Confusion-matrix counts cannot be negative.")

    if df["precision"].dropna().between(0, 1).eq(False).any():
        raise ValueError("Precision contains values outside [0, 1].")

    if df["recall"].dropna().between(0, 1).eq(False).any():
        raise ValueError("Recall contains values outside [0, 1].")

    df["event_detected"] = parse_binary_column(
        df["event_detected"], "event_detected"
    )

    df["false_alarm_trial"] = parse_binary_column(
        df["false_alarm_trial"], "false_alarm_trial"
    )

    if (
        df["false_alarm_samples"].dropna() < 0
    ).any():
        raise ValueError(
            "false_alarm_samples cannot contain negative values."
        )

    if (
        df["delay_samples"].dropna() < 0
    ).any():
        raise ValueError(
            "delay_samples cannot contain negative values."
        )

    duplicate_mask = df.duplicated(
        subset=["scenario", "method", "trial"],
        keep=False,
    )

    if duplicate_mask.any():
        print(
            "\nWARNING: Duplicate scenario/method/trial records found: "
            f"{int(duplicate_mask.sum())} rows."
        )
        print(
            "Check these records before interpreting the summary."
        )
    else:
        print(
            "PASS: No duplicate scenario/method/trial records."
        )

    print(f"Scenarios: {df['scenario'].nunique()}")
    print(f"Methods: {df['method'].nunique()}")
    print(f"Trials represented: {df['trial'].nunique()} unique trial IDs")

    print("\nScenarios found:")
    for scenario in sorted(df["scenario"].unique()):
        print(f"  - {scenario}")

    print("\nMethods found:")
    for method in sorted(df["method"].unique()):
        print(f"  - {method}")

    # --------------------------------------------------
    # 6. SUMMARIZE BY SCENARIO AND METHOD
    # --------------------------------------------------

    results = []

    for (scenario, method), group in df.groupby(
        ["scenario", "method"],
        sort=True,
        dropna=False,
    ):
        n_trials = len(group)
        detected = int(group["event_detected"].sum())

        detection_rate = (
            detected / n_trials if n_trials else np.nan
        )

        ci_low, ci_high = wilson_interval(detected, n_trials)

        results.append({
            "scenario": scenario,
            "method": method,
            "n_trials": n_trials,
            "events_detected": detected,
            "event_detection_rate": detection_rate,
            "event_detection_ci95_low": ci_low,
            "event_detection_ci95_high": ci_high,
            "mean_delay_samples_detected": safe_mean(
                group.loc[
                    group["event_detected"] == 1,
                    "delay_samples",
                ]
            ),
            "median_delay_samples_detected": (
                group.loc[
                    group["event_detected"] == 1,
                    "delay_samples",
                ].median()
            ),
            "mean_precision": safe_mean(group["precision"]),
            "mean_recall": safe_mean(group["recall"]),
            "false_alarm_trial_rate": safe_mean(
                group["false_alarm_trial"]
            ),
            "mean_false_alarm_samples": safe_mean(
                group["false_alarm_samples"]
            ),
            "source": EXPECTED_SOURCE,
        })

    summary = pd.DataFrame(results)

    summary.to_csv(SUMMARY_FILE, index=False)

    # --------------------------------------------------
    # 7. CREATE HUMAN-READABLE REPORT
    # --------------------------------------------------

    lines = [
        "PHYSOS v2.5 — DATA QUALITY AND DETECTOR DIAGNOSTICS",
        "=" * 70,
        "",
        f"Input: {INPUT_FILE}",
        f"Rows analyzed: {len(df)}",
        f"Scenarios: {df['scenario'].nunique()}",
        f"Methods: {df['method'].nunique()}",
        f"Source labels: {', '.join(sorted(source_labels.unique()))}",
        "",
        "IMPORTANT INTERPRETATION NOTES",
        "-" * 70,
        "1. All analyzed records are labeled SIMULATED.",
        "2. Event detection rate is the fraction of trials with "
        "event_detected == 1.",
        "3. Confidence intervals use the Wilson 95% interval for a "
        "binomial proportion.",
        "4. Delay is reported in samples, not seconds.",
        "5. Delay statistics include detected events only.",
        "6. False-alarm rates use the supplied false_alarm_trial field.",
        "7. These summaries do not establish performance on real hardware.",
        "8. Confidence intervals assume independent trial outcomes; "
        "shared seeds or correlated trials may weaken that assumption.",
        "",
        "RESULTS BY SCENARIO AND METHOD",
        "-" * 70,
    ]

    for _, row in summary.iterrows():
        lines.extend([
            "",
            f"Scenario: {row['scenario']}",
            f"Method: {row['method']}",
            f"Trials: {int(row['n_trials'])}",
            (
                "Event detection rate: "
                f"{format_number(row['event_detection_rate'])}"
            ),
            (
                "Wilson 95% CI: "
                f"[{format_number(row['event_detection_ci95_low'])}, "
                f"{format_number(row['event_detection_ci95_high'])}]"
            ),
            (
                "Mean delay (detected events, samples): "
                f"{format_number(row['mean_delay_samples_detected'], 2)}"
            ),
            (
                "Median delay (detected events, samples): "
                f"{format_number(row['median_delay_samples_detected'], 2)}"
            ),
            f"Mean precision: {format_number(row['mean_precision'])}",
            f"Mean recall: {format_number(row['mean_recall'])}",
            (
                "False-alarm trial rate: "
                f"{format_number(row['false_alarm_trial_rate'])}"
            ),
            (
                "Mean false-alarm samples: "
                f"{format_number(row['mean_false_alarm_samples'], 2)}"
            ),
        ])

    lines.extend([
        "",
        "=" * 70,
        "END OF REPORT",
    ])

    REPORT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # 8. PRINT KEY RESULTS
    # --------------------------------------------------

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 28)

    display_columns = [
        "scenario",
        "method",
        "n_trials",
        "event_detection_rate",
        "event_detection_ci95_low",
        "event_detection_ci95_high",
        "mean_delay_samples_detected",
        "false_alarm_trial_rate",
    ]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        summary[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nFiles created:")
    print(f"  Summary CSV: {SUMMARY_FILE}")
    print(f"  Text report: {REPORT_FILE}")

    print("\nDiagnostics completed successfully.")
    print("Data source remains SIMULATED; no real-hardware claim is made.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nDIAGNOSTICS FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
