
"""
PHYSOS v2.1 — Held-out Statistical Analysis

Reads the existing v2.0 held-out results.
Does not regenerate data, change thresholds, or modify the original results.

All conclusions apply to synthetic experiments only.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

INPUT = DATA / "physos_v20_heldout_trials.csv"

OUT_SUMMARY = DATA / "physos_v21_summary.csv"
OUT_COMPARISONS = DATA / "physos_v21_paired_comparisons.csv"
OUT_QUALITY = DATA / "physos_v21_quality_report.csv"

BOOTSTRAPS = 5000
BOOTSTRAP_SEED = 219021

METHODS = [
    "temperature_level",
    "frequency_level",
    "temperature_slope",
    "frequency_slope",
    "joint_slope",
    "joint_slope_persistent_2of3",
]

BASELINES = [
    "temperature_level",
    "frequency_level",
    "temperature_slope",
    "frequency_slope",
    "joint_slope_persistent_2of3",
]


def wilson_interval(successes, total, z=1.96):
    if total == 0:
        return np.nan, np.nan

    p = successes / total
    denominator = 1 + z * z / total

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

    return max(0, center - margin), min(1, center + margin)


def bootstrap_mean_ci(values, rng, n_boot=BOOTSTRAPS):
    """Trial-level percentile bootstrap confidence interval."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    if len(values) == 1:
        return float(values[0]), np.nan, np.nan

    indices = rng.integers(
        0, len(values), size=(n_boot, len(values))
    )

    estimates = values[indices].mean(axis=1)

    return (
        float(values.mean()),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def summarize(data):
    rows = []

    for (scenario, method), group in data.groupby(
        ["scenario", "method"], sort=True
    ):
        n = len(group)

        detected = group["event_detected"].dropna()
        successes = int(detected.sum())
        total_events = len(detected)

        if total_events:
            rate = successes / total_events
            wilson_low, wilson_high = wilson_interval(
                successes, total_events
            )
        else:
            rate = np.nan
            wilson_low = np.nan
            wilson_high = np.nan

        delay = group["detection_delay_samples"].dropna()

        false_samples = group["false_alarm_samples"].to_numpy()
        false_episodes = group["false_alarm_episodes"].to_numpy()

        false_trial_rate = float(
            np.mean(false_samples > 0)
        )

        rows.append({
            "scenario": scenario,
            "method": method,
            "trials": n,
            "event_detection_rate": rate,
            "event_detection_wilson95_low": wilson_low,
            "event_detection_wilson95_high": wilson_high,
            "detected_events": successes,
            "mean_delay_when_detected": (
                float(delay.mean()) if len(delay) else np.nan
            ),
            "median_delay_when_detected": (
                float(delay.median()) if len(delay) else np.nan
            ),
            "mean_sample_precision": group[
                "sample_precision"
            ].mean(),
            "mean_sample_recall": group[
                "sample_recall"
            ].mean(),
            "pooled_sample_precision": (
                group["tp"].sum()
                / max(
                    1,
                    group["tp"].sum() + group["fp"].sum()
                )
            ),
            "pooled_sample_recall": (
                group["tp"].sum()
                / max(
                    1,
                    group["tp"].sum() + group["fn"].sum()
                )
            ),
            "false_alarm_samples_per_trial": float(
                false_samples.mean()
            ),
            "false_alarm_episodes_per_trial": float(
                false_episodes.mean()
            ),
            "fraction_trials_with_false_alarms": false_trial_rate,
            "mean_total_alert_samples": group[
                "total_alert_samples"
            ].mean(),
            "source": "SIMULATED",
        })

    return pd.DataFrame(rows)


def paired_comparisons(data):
    """
    Paired bootstrap differences in event detection rate.

    Positive difference means joint_slope detected more events
    than the comparator on the same held-out trials.

    The bootstrap resamples trial indices, preserving pairing.
    """

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []

    for scenario, scenario_df in data.groupby("scenario", sort=True):
        pivot = scenario_df.pivot(
            index="trial",
            columns="method",
            values="event_detected",
        )

        if "joint_slope" not in pivot.columns:
            continue

        for baseline in BASELINES:
            if baseline not in pivot.columns:
                continue

            pair = pivot[
                ["joint_slope", baseline]
            ].dropna()

            if len(pair) == 0:
                continue

            joint = pair["joint_slope"].to_numpy(dtype=float)
            other = pair[baseline].to_numpy(dtype=float)

            differences = joint - other
            observed = float(differences.mean())

            if len(differences) > 1:
                indices = rng.integers(
                    0,
                    len(differences),
                    size=(BOOTSTRAPS, len(differences)),
                )
                boot = differences[indices].mean(axis=1)

                low, high = np.quantile(
                    boot, [0.025, 0.975]
                )
            else:
                low, high = np.nan, np.nan

            rows.append({
                "scenario": scenario,
                "comparison": f"joint_slope_vs_{baseline}",
                "paired_trials": len(pair),
                "joint_event_rate": float(joint.mean()),
                "baseline_event_rate": float(other.mean()),
                "paired_detection_rate_difference": observed,
                "bootstrap95_low": float(low),
                "bootstrap95_high": float(high),
                "interpretation": (
                    "Joint detector higher"
                    if low > 0
                    else (
                        "Joint detector lower"
                        if high < 0
                        else "Difference uncertain"
                    )
                ),
                "source": "SIMULATED",
            })

    return pd.DataFrame(rows)


def quality_report(data):
    required = [
        "scenario",
        "method",
        "trial",
        "seed",
        "event_detected",
        "detection_delay_samples",
        "tp",
        "fp",
        "fn",
        "tn",
        "false_alarm_samples",
        "false_alarm_episodes",
        "source",
    ]

    missing = [c for c in required if c not in data.columns]

    checks = []

    checks.append({
        "check": "required_columns_present",
        "passed": len(missing) == 0,
        "details": "Missing: " + ", ".join(missing)
        if missing else "All required columns found",
    })

    if missing:
        return pd.DataFrame(checks)

    checks.append({
        "check": "all_rows_marked_simulated",
        "passed": bool((data["source"] == "SIMULATED").all()),
        "details": f"Unique sources: {data['source'].unique().tolist()}",
    })

    checks.append({
        "check": "no_duplicate_trial_method_rows",
        "passed": not data.duplicated(
            ["scenario", "trial", "method"]
        ).any(),
        "details": "Each scenario/trial/method should occur once",
    })

    checks.append({
        "check": "expected_methods_present",
        "passed": set(METHODS).issubset(set(data["method"].unique())),
        "details": f"Methods found: {sorted(data['method'].unique())}",
    })

    trial_counts = data.groupby("scenario")["trial"].nunique()

    checks.append({
        "check": "100_trials_per_scenario",
        "passed": bool((trial_counts == 100).all()),
        "details": str(trial_counts.to_dict()),
    })

    expected_methods_per_trial = data.groupby(
        ["scenario", "trial"]
    )["method"].nunique()

    checks.append({
        "check": "all_methods_on_each_trial",
        "passed": bool(
            (expected_methods_per_trial == len(METHODS)).all()
        ),
        "details": (
            f"Expected {len(METHODS)} methods per trial"
        ),
    })

    return pd.DataFrame(checks)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT}\n"
            "Run experiments/physos_v20_validation.py first."
        )

    data = pd.read_csv(INPUT)

    report = quality_report(data)
    report.to_csv(OUT_QUALITY, index=False)

    print("=" * 76)
    print("PHYSOS v2.1 — DATA QUALITY CHECKS")
    print("=" * 76)
    print(report.to_string(index=False))

    if not report["passed"].all():
        raise ValueError(
            "Data quality checks failed. Inspect "
            "physos_v21_quality_report.csv before interpreting results."
        )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary = summarize(data)
    comparisons = paired_comparisons(data)

    summary.to_csv(OUT_SUMMARY, index=False)
    comparisons.to_csv(OUT_COMPARISONS, index=False)

    print("\n" + "=" * 76)
    print("PHYSOS v2.1 — HELD-OUT PERFORMANCE")
    print("=" * 76)

    columns = [
        "scenario",
        "method",
        "event_detection_rate",
        "event_detection_wilson95_low",
        "event_detection_wilson95_high",
        "mean_delay_when_detected",
        "pooled_sample_precision",
        "pooled_sample_recall",
        "false_alarm_samples_per_trial",
        "fraction_trials_with_false_alarms",
    ]

    print(summary[columns].round(4).to_string(index=False))

    print("\n" + "=" * 76)
    print("PAIRED EVENT-DETECTION COMPARISONS")
    print("Positive difference favors joint_slope.")
    print("=" * 76)

    if len(comparisons):
        print(
            comparisons[
                [
                    "scenario",
                    "comparison",
                    "paired_detection_rate_difference",
                    "bootstrap95_low",
                    "bootstrap95_high",
                    "interpretation",
                ]
            ].round(4).to_string(index=False)
        )

    print("\nFiles saved:")
    print(OUT_QUALITY)
    print(OUT_SUMMARY)
    print(OUT_COMPARISONS)

    print("\nScientific cautions:")
    print("- All evaluated signals are synthetic.")
    print("- Event-detection intervals use Wilson 95% intervals.")
    print("- Paired bootstrap intervals compare matched trial outcomes.")
    print("- Overlapping confidence intervals are not a formal equivalence test.")
    print("- A higher detection rate alone does not imply a better detector.")
    print("- Examine false alarms and delay alongside detection.")
    print("- Results do not establish real-hardware performance.")


if __name__ == "__main__":
    main()
