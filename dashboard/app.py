
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

st.set_page_config(
    page_title="PHYSOS Research Console",
    page_icon="🔬",
    layout="wide",
)

st.title("PHYSOS | Physical-Aware System Research")
st.caption(
    "Research prototype • Historical CSV data • "
    "Environmental simulation is not real sensor data"
)


def read_csv(filename):
    path = DATA / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return None


system = read_csv("system_telemetry.csv")
fused = read_csv("fused_telemetry.csv")
drift = read_csv("physical_drift.csv")

if system is None or system.empty:
    st.warning(
        "No system telemetry found. Run the PHYSOS telemetry collector first."
    )
    st.stop()

system["timestamp"] = pd.to_datetime(
    system["timestamp"], errors="coerce"
)
system = system.dropna(subset=["timestamp"]).sort_values("timestamp")

if system.empty:
    st.error("The system telemetry file has no valid timestamps.")
    st.stop()

latest = system.iloc[-1]

cpu_col = "cpu_percent"
ram_col = "ram_percent"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest CPU", f"{latest[cpu_col]:.1f}%")
c2.metric("Latest RAM", f"{latest[ram_col]:.1f}%")

frequency_col = (
    "cpu_frequency"
    if "cpu_frequency" in system.columns
    else "cpu_frequency_mhz"
)

if frequency_col in system.columns and pd.notna(latest[frequency_col]):
    c3.metric("CPU Frequency", f"{latest[frequency_col]:.0f} MHz")
else:
    c3.metric("CPU Frequency", "Unavailable")

c4.metric("System Samples", f"{len(system):,}")

st.divider()
st.subheader("System behavior over time")

metric_options = [
    col for col in [
        "cpu_percent",
        "ram_percent",
        "cpu_frequency",
        "cpu_frequency_mhz",
        "disk_percent",
    ]
    if col in system.columns
]

selected = st.multiselect(
    "Select signals",
    metric_options,
    default=[
        col for col in ["cpu_percent", "ram_percent"]
        if col in metric_options
    ],
)

if selected:
    fig = px.line(
        system,
        x="timestamp",
        y=selected,
        title="Computer telemetry",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select at least one signal to plot.")

st.divider()
st.subheader("Environmental sensor fusion")

if fused is None or fused.empty:
    st.info(
        "Fused data is not available yet. Run the sensor fusion engine."
    )
else:
    source_col = "source"

    if source_col in fused.columns:
        sources = fused[source_col].dropna().unique().tolist()
        st.caption("Data sources: " + ", ".join(map(str, sources)))

        if any(str(s).upper() == "SIMULATED" for s in sources):
            st.warning(
                "This dataset contains synthetic environmental measurements. "
                "Do not interpret them as actual temperature or humidity."
            )

    if "system_timestamp" in fused.columns:
        fused["system_timestamp"] = pd.to_datetime(
            fused["system_timestamp"], errors="coerce"
        )

    if "temperature_c" in fused.columns:
        environmental = fused.dropna(
            subset=["temperature_c"]
        ).copy()

        if not environmental.empty:
            fig = px.line(
                environmental,
                x="system_timestamp",
                y="temperature_c",
                title="Matched environmental temperature",
                labels={"temperature_c": "Temperature (°C)"},
            )
            st.plotly_chart(fig, use_container_width=True)

            if "cpu_percent" in environmental.columns:
                fig = px.scatter(
                    environmental,
                    x="temperature_c",
                    y="cpu_percent",
                    title="CPU usage versus temperature",
                    labels={
                        "temperature_c": "Temperature (°C)",
                        "cpu_percent": "CPU usage (%)",
                    },
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No matched temperature readings are available.")

st.divider()
st.subheader("Physical drift analysis")

if drift is None or drift.empty:
    st.info("Drift results are not available. Run the drift engine first.")
else:
    available = [
        col for col in [
            "timestamp",
            "drifting_signal_count",
            "multi_signal_drift",
            "persistent_drift",
            "drifting_signals",
        ]
        if col in drift.columns
    ]

    st.dataframe(
        drift[available].tail(100),
        use_container_width=True,
        hide_index=True,
    )

    if "drifting_signal_count" in drift.columns and "timestamp" in drift.columns:
        drift["timestamp"] = pd.to_datetime(
            drift["timestamp"], errors="coerce"
        )
        fig = px.line(
            drift,
            x="timestamp",
            y="drifting_signal_count",
            title="Number of signals crossing drift thresholds",
        )
        st.plotly_chart(fig, use_container_width=True)

st.caption(
    "PHYSOS is an experimental research prototype. "
    "Statistical anomalies and drift indicators are not confirmed failures "
    "or validated predictions of degradation."
)
