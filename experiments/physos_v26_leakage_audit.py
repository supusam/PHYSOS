
"""
PHYSOS v2.6 — Leakage Audit and Failure Analysis

Input:
    data/physos_v24_test_trials.csv
    data/physos_v24_config.json          (if available)
    data/physos_v24_thresholds.json      (if available)

Outputs:
    data/physos_v26_audit_report.txt
    data/physos_v26_failure_analysis.csv

This script does not modify the v2.4 dataset, retrain models,
or claim to prove leakage-free evaluation.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

INPUT = DATA / "physos_v24_test_trials.csv"
CONFIG = DATA / "physos_v24_config.json"
THRESHOLDS = DATA / "physos_v24_thresholds.json"

REPORT = DATA / "physos_v26_audit_report.txt"
FAILURES = DATA / "physos_v26_failure_analysis.csv"

EXPECTED_SOURCE = "SIMULATED"

REQUIRED = {
    "method", "scenario", "trial", "seed", "source",
    "TP", "FP", "FN", "TN", "precision", "recall",
    "event_detected", "delay_samples",
    "false_alarm_samples", "false_alarm_trial",
}


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def binary_column(series, name):
    values = series.astype("string").str.strip().str.lower()
    mapping = {
        "1": 1, "0": 0,
        "true": 1, "false": 0,
        "yes": 1, "no": 0,
    }
    result = values.map(mapping)

    if result.isna().any():
        raise ValueError(
            f"Invalid or missing values in {name}: "
            f"{series[result.isna()].unique().tolist()}"
        )

    return result.astype(int)


def wilson(successes, total, z=1.959963984540054):
    if total == 0:
        return np.nan, np.nan

    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = (
        z * np.sqrt(
            p * (1 - p) / total
            + z * z / (4 * total * total)
        ) / denom
    )

    return max(0.0, center - margin), min(1.0, center + margin)


def read_json(path):
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: Could not read {path.name}: {exc}")
        return None


def find_seed_fields(obj, path="$"):
    """
    Find explicitly named seed fields in JSON metadata.
    This is a screening aid, not proof of train/test separation.
    """
    found = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            child_path = f"{path}.{key}"

            if "seed" in str(key).lower():
                found.append((child_path, value))

            found.extend(find_seed_fields(value, child_path))

    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            found.extend(
                find_seed_fields(value, f"{path}[{index}]")
            )

    return found


def flatten_numeric(value):
    """Extract integers from seed metadata when clearly numeric."""
    result = []

    if isinstance(value, bool):
        return result

    if isinstance(value, int):
        result.append(value)

    elif isinstance(value, list):
        for item in value:
            result.extend(flatten_numeric(item))

    return result


def main():
    DATA.mkdir(parents=True, exist_ok=True)

    section("PHYSOS v2.6 — LEAKAGE AUDIT")

    if not INPUT.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT}")

    df = pd.read_csv(INPUT)

    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Input dataset is empty.")

    print(f"Input file: {INPUT}")
    print(f"Rows: {len(df)}")
    print(f"Scenarios: {df['scenario'].nunique()}")
    print(f"Methods: {df['method'].nunique()}")

    # --------------------------------------------------
    # A. Source and structural validation
    # --------------------------------------------------

    section("A. DATA SOURCE AND STRUCTURE")

    sources = df["source"].astype("string").str.strip().str.upper()
    print("Source counts:")
    print(sources.value_counts(dropna=False).to_string())

    if sources.isna().any() or not sources.eq(EXPECTED_SOURCE).all():
        raise ValueError(
            "Source validation failed. Expected only SIMULATED."
        )

    df["source"] = sources
    print("PASS: All rows are labeled SIMULATED.")

    key = ["scenario", "method", "trial"]
    duplicate_count = int(df.duplicated(key).sum())

    print(f"Duplicate scenario/method/trial rows: {duplicate_count}")

    if duplicate_count:
        print("WARNING: Duplicate trial records need investigation.")
    else:
        print("PASS: No duplicate scenario/method/trial records.")

    for column in ["TP", "FP", "FN", "TN",
                   "precision", "recall",
                   "delay_samples", "false_alarm_samples"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[["TP", "FP", "FN", "TN"]].isna().any().any():
        raise ValueError("Invalid confusion-matrix values.")

    if (df[["TP", "FP", "FN", "TN"]] < 0).any().any():
        raise ValueError("Negative confusion-matrix counts found.")

    df["event_detected"] = binary_column(
        df["event_detected"], "event_detected"
    )
    df["false_alarm_trial"] = binary_column(
        df["false_alarm_trial"], "false_alarm_trial"
    )

    # Check whether stored precision/recall agree with confusion counts.
    tp = df["TP"]
    fp = df["FP"]
    fn = df["FN"]

    calculated_precision = (tp / (tp + fp)).where((tp + fp) != 0)
    calculated_recall = (tp / (tp + fn)).where((tp + fn) != 0)

    precision_error = (
        df["precision"].notna()
        & calculated_precision.notna()
        & ((df["precision"] - calculated_precision).abs() > 0.002)
    )

    recall_error = (
        df["recall"].notna()
        & calculated_recall.notna()
        & ((df["recall"] - calculated_recall).abs() > 0.002)
    )

    print(
        "Precision/count consistency mismatches:",
        int(precision_error.sum()),
    )
    print(
        "Recall/count consistency mismatches:",
        int(recall_error.sum()),
    )

    if precision_error.any() or recall_error.any():
        print(
            "WARNING: Some stored precision/recall values do not match "
            "the confusion counts. Inspect metric definitions."
        )

    # --------------------------------------------------
    # B. Seed and split audit
    # --------------------------------------------------

    section("B. SEED AND DATA-SPLIT AUDIT")

    test_seeds = set(
        pd.to_numeric(df["seed"], errors="coerce").dropna().astype(int)
    )

    print(f"Distinct seeds recorded in test CSV: {len(test_seeds)}")
    print(
        "Note: repeated seed values across scenarios may be intentional "
        "paired trials; they do not alone prove leakage."
    )

    config = read_json(CONFIG)
    thresholds = read_json(THRESHOLDS)

    if config is None:
        print("Config file not found/readable: split verification limited.")
    else:
        print(f"Config file loaded: {CONFIG.name}")

    if thresholds is None:
        print(
            "Threshold metadata file not found/readable: "
            "threshold provenance verification limited."
        )
    else:
        print(f"Threshold metadata loaded: {THRESHOLDS.name}")

    for label, obj in [
        ("CONFIG", config),
        ("THRESHOLDS", thresholds),
    ]:
        if obj is None:
            continue

        fields = find_seed_fields(obj)

        if not fields:
            print(
                f"{label}: no explicitly named seed fields found. "
                "Cannot verify split separation from this file."
            )
            continue

        print(f"{label}: seed-related metadata fields:")
        for field_path, value in fields:
            print(f"  {field_path} = {value}")

            # Compare only fields that explicitly look like test seeds.
            if "test" in field_path.lower() and "seed" in field_path.lower():
                numeric_seeds = set(flatten_numeric(value))
                overlap = numeric_seeds & test_seeds

                if numeric_seeds:
                    print(
                        "    Test-seed metadata overlap with CSV: "
                        f"{len(overlap)} matching seed(s)"
                    )

    print(
        "\nAUDIT LIMIT: This metadata scan cannot establish that "
        "training and test data are independent. Verify the actual "
        "generator, calibration, and model-training code."
    )

    # --------------------------------------------------
    # C. Failure analysis by scenario and method
    # --------------------------------------------------

    section("C. DETECTOR FAILURE ANALYSIS")

    summaries = []

    for (scenario, method), group in df.groupby(
        ["scenario", "method"], sort=True
    ):
        n = len(group)
        detected = int(group["event_detected"].sum())
        ci_low, ci_high = wilson(detected, n)

        delays = group.loc[
            group["event_detected"] == 1, "delay_samples"
        ].dropna()

        summaries.append({
            "scenario": scenario,
            "method": method,
            "n_trials": n,
            "event_detected_trials": detected,
            "missed_event_trials": n - detected,
            "detection_rate": detected / n if n else np.nan,
            "detection_ci95_low": ci_low,
            "detection_ci95_high": ci_high,
            "mean_delay_samples_detected": (
                delays.mean() if not delays.empty else np.nan
            ),
            "median_delay_samples_detected": (
                delays.median() if not delays.empty else np.nan
            ),
            "false_alarm_trial_rate": group[
                "false_alarm_trial"
            ].mean(),
            "mean_false_alarm_samples": group[
                "false_alarm_samples"
            ].mean(),
            "source": EXPECTED_SOURCE,
        })

    summary = pd.DataFrame(summaries)
    summary.to_csv(FAILURES, index=False)

    print(
        summary[
            [
                "scenario", "method", "n_trials",
                "detection_rate", "missed_event_trials",
                "mean_delay_samples_detected",
                "false_alarm_trial_rate",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    # --------------------------------------------------
    # D. Flag suspicious trade-offs
    # --------------------------------------------------

    section("D. AUTOMATIC REVIEW FLAGS")

    flags = []

    for _, row in summary.iterrows():
        if row["false_alarm_trial_rate"] > 0.05:
            flags.append(
                f"HIGH FALSE-ALARM RATE: {row['scenario']} / "
                f"{row['method']} = "
                f"{row['false_alarm_trial_rate']:.1%}"
            )

        if row["detection_rate"] < 0.50:
            flags.append(
                f"LOW DETECTION: {row['scenario']} / {row['method']} = "
                f"{row['detection_rate']:.1%}"
            )

        if (
            row["detection_rate"] >= 0.95
            and row["false_alarm_trial_rate"] > 0.05
        ):
            flags.append(
                f"DETECTION/FALSE-ALARM TRADE-OFF: "
                f"{row['scenario']} / {row['method']}"
            )

    if flags:
        for flag in flags:
            print("REVIEW:", flag)
    else:
        print("No automatic review flags triggered.")

    # --------------------------------------------------
    # E. Write report
    # --------------------------------------------------

    lines = [
        "PHYSOS v2.6 — LEAKAGE AUDIT AND FAILURE ANALYSIS",
        "=" * 72,
        f"Input: {INPUT}",
        f"Rows: {len(df)}",
        f"Scenarios: {df['scenario'].nunique()}",
        f"Methods: {df['method'].nunique()}",
        f"Source: {EXPECTED_SOURCE}",
        f"Duplicate scenario/method/trial rows: {duplicate_count}",
        f"Precision consistency mismatches: {int(precision_error.sum())}",
        f"Recall consistency mismatches: {int(recall_error.sum())}",
        "",
        "IMPORTANT:",
        "- This is a metadata and result-consistency audit.",
        "- It cannot prove the experiment is free of data leakage.",
        "- Verify split logic in the actual training and calibration code.",
        "- False-alarm rates depend on the original script's definitions.",
        "- All results are simulated, not real-hardware measurements.",
        "",
        "AUTOMATIC REVIEW FLAGS:",
        *([f"- {flag}" for flag in flags] or ["- None triggered."]),
        "",
        "Detailed scenario/method summary is in:",
        str(FAILURES),
    ]

    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("\nCreated:")
    print(f"  {FAILURES}")
    print(f"  {REPORT}")

    print("\nAudit completed. Review warnings before interpreting results.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nAUDIT FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
