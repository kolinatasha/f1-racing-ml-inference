"""F1 Race Strategy Dashboard — Streamlit demo for the ML inference API."""

import os
import time

import requests
import streamlit as st

API_BASE = os.environ.get(
    "F1_API_BASE",
    "https://u59hxhp9l5.execute-api.us-east-1.amazonaws.com/prod",
)

TRACKS = ["Bahrain", "Saudi Arabia", "Australia", "Japan", "Azerbaijan", "Miami", "Silverstone", "Spa", "Monaco"]
COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
COMPOUND_COLOR = {
    "SOFT": "#e8002d",
    "MEDIUM": "#ffd600",
    "HARD": "#f0f0ec",
    "INTERMEDIATE": "#39b54a",
    "WET": "#0067ff",
}

st.set_page_config(
    page_title="F1 Strategy Dashboard",
    page_icon="🏎",
    layout="wide",
)

st.title("🏎 F1 Race Strategy Dashboard")
st.caption(
    "Live ML inference via AWS Lambda · "
    "[github.com/kolinatasha/f1-racing-ml-inference](https://github.com/kolinatasha/f1-racing-ml-inference)"
)

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Race Parameters")

    track = st.selectbox("Track", TRACKS, index=0)
    driver = st.text_input("Driver code", value="VER", max_chars=3).upper()
    compound = st.selectbox("Tire compound", COMPOUNDS, index=0)

    st.divider()

    current_lap = st.slider("Current lap", 1, 70, 20)
    total_laps = st.slider("Total race laps", 20, 78, 57)
    tire_age = st.slider("Tires age (laps)", 0, 40, 15)
    fuel_load = st.slider("Fuel load (kg)", 5, 110, max(5, 110 - current_lap * 2))

    st.divider()

    track_temp = st.slider("Track temp (°C)", 15, 60, 38)
    air_temp = st.slider("Air temp (°C)", 10, 45, 24)

    st.divider()

    gap_ahead = st.number_input("Gap to car ahead (s)", 0.0, 60.0, 2.5, step=0.1)
    gap_behind = st.number_input("Gap to car behind (s)", 0.0, 60.0, 8.0, step=0.1)
    position = st.number_input("Current position", 1, 20, 4)

    run = st.button("🔮 Predict", type="primary", use_container_width=True)


def call_api(endpoint: str, payload: dict) -> tuple[dict | None, float, int | None]:
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{API_BASE}/{endpoint}", json=payload, timeout=15)
        latency = (time.perf_counter() - t0) * 1000
        if r.ok:
            return r.json(), latency, r.status_code
        return None, latency, r.status_code
    except Exception as e:
        return None, (time.perf_counter() - t0) * 1000, None


def laptime_badge(laptime_str: str) -> str:
    return f"## ⏱ `{laptime_str}`"


# ── Persistent about / tutorial expander (always visible) ─────────────────────
with st.expander("📖 About this project & how to use it", expanded=not run):

    st.markdown("### What this project captures")
    st.markdown(
        """
Formula 1 races are won and lost in the **pit lane**. Each team's strategy engineer
sits on the pit wall with live telemetry, predicting three things simultaneously:

| Question | This dashboard |
|---|---|
| How fast will my car lap right now? | **Lap Time** panel |
| How many laps are left in these tires? | **Tire Degradation** panel |
| Should I pit this lap or stay out? | **Pit Strategy** panel |

This project builds that system end-to-end — from **raw telemetry to a live API** —
using the same data sources and decision framework real F1 teams use.

**What was built:**
- Collected **8,300 real laps** from 9 races (2023–2024) using the FastF1 library, which
  streams official FIA timing data
- Trained three **XGBoost models** (lap time regressor, degradation regressor, strategy scorer)
  on features derived from that telemetry: tire compound grip levels, track surface degradation
  factors, fuel burn estimates, positional gaps
- Validated against **holdout 2024 Bahrain data**: RMSE **1.198 s**, MAE **0.794 s** on 1,043 laps
- Deployed as a **serverless AWS stack** (API Gateway → Lambda → DynamoDB + S3) with
  215 ms p95 latency on warm containers
- Every call you make here hits that live AWS API in real time
        """
    )

    st.divider()

    st.markdown("### How to use this dashboard")
    st.markdown(
        """
**Step 1 — Pick a race scenario** *(sidebar)*

Choose a track, driver code (e.g. `VER`, `HAM`, `LEC`), and tire compound.
The compound color-codes the Lap Time panel border (🔴 Soft · 🟡 Medium · ⚪ Hard · 🟢 Inter · 🔵 Wet).

**Step 2 — Set the race situation** *(sidebar)*

| Parameter | What it means |
|---|---|
| Current lap | Which lap of the race you are on |
| Total race laps | Full race distance (e.g. 57 for Bahrain, 78 for Monaco) |
| Tires age (laps) | How many laps the current set has done |
| Fuel load (kg) | Starts ~100 kg, burns ~1.6 kg/lap — auto-estimated but editable |
| Track / air temp | Higher track temp → faster degradation |
| Gap to car ahead/behind | Used by the strategy model to evaluate undercut/overcut |

**Step 3 — Click 🔮 Predict**

Three API calls fire simultaneously. Each panel shows:
- The model's prediction
- The Lambda latency for that call (watch for **cold starts** — first call ~6 s, warm ~300 ms)
- The model version serving the prediction

**Step 4 — Try these comparisons to see the models in action**

- 🔴 → ⚪ **Soft vs Hard same lap**: Hard predicts a slower lap time, less degradation, longer remaining life
- Lap 5 → Lap 30 **same compound**: Lap time increases as tires wear, strategy shifts from "stay out" toward "pit now"
- Gap ahead **2 s → 0.5 s**: Tighter gap triggers earlier pit window (undercut opportunity)
- Gap behind **3 s → 20 s**: Big gap behind → model recommends staying out longer

**Step 5 — Run the Stint Simulation** *(bottom of results)*

Plots predicted lap time across 25 laps of tire age on a single set.
This shows the **degradation curve** — how long before the cliff — which is the
core input to a real strategy engineer's stop window calculation.
        """
    )

    st.divider()

    st.markdown("### Architecture")
    st.code(
        "Browser → Streamlit Cloud\n"
        "              ↓  HTTP POST (3 calls)\n"
        "         API Gateway  ←→  AWS X-Ray tracing\n"
        "              ↓\n"
        "         Lambda  (Python 3.11, 256 MB, 5 s timeout)\n"
        "           ├── DynamoDB  f1-features  (track/tire lookup)\n"
        "           └── S3  f1-ml-models  (XGBoost model.json, ~3 MB each)\n"
        "              ↓\n"
        "         CloudWatch  (latency alarms, error rate alarms)",
        language="text",
    )
    st.caption(
        "Models are cached in-process after first S3 load — cold start is ~6 s (S3 download), "
        "warm calls are ~215 ms p95. The `latency_ms` in each panel is measured inside Lambda "
        "and does not include API Gateway overhead."
    )

# ── Main content ──────────────────────────────────────────────────────────────
if not run:
    st.info("👈 Set parameters in the sidebar and click **🔮 Predict** to call the live AWS API.")

else:
    compound_color = COMPOUND_COLOR.get(compound, "#888")
    payload_base = {
        "driver": driver,
        "track": track.lower().replace(" ", "_"),
        "tire_compound": compound,
        "tire_age_laps": tire_age,
        "fuel_load_kg": fuel_load,
        "track_temp": track_temp,
        "air_temp": air_temp,
    }

    col1, col2, col3 = st.columns(3)

    # ── Lap time ──────────────────────────────────────────────────────────────
    with col1:
        with st.spinner("Calling /predict/laptime …"):
            data, latency, status = call_api("predict/laptime", payload_base)

        st.markdown(
            f"<div style='border-left: 4px solid {compound_color}; padding-left: 12px;'>"
            f"<h3>Lap Time</h3></div>",
            unsafe_allow_html=True,
        )

        if data:
            st.markdown(laptime_badge(data["predicted_laptime"]))
            lo, hi = data["confidence_interval"]
            st.caption(f"±0.3 s window: `{lo}` – `{hi}`")
            st.metric("Track", data.get("track", track))
            st.metric("Conditions", data.get("conditions", "—"))
            st.caption(f"Lambda latency: **{latency:.0f} ms** · model: `{data.get('model_version')}`")
        else:
            st.error(f"API error {status}")

    # ── Tire degradation ──────────────────────────────────────────────────────
    with col2:
        deg_payload = {
            "driver": driver,
            "track": track.lower().replace(" ", "_"),
            "tire_compound": compound,
            "laps_on_tire": tire_age,
            "track_temp": track_temp,
            "driver_style": "aggressive",
        }
        with st.spinner("Calling /predict/tire-degradation …"):
            data2, latency2, status2 = call_api("predict/tire-degradation", deg_payload)

        st.markdown("### Tire Degradation")

        if data2:
            deg = data2.get("current_degradation_percent", 0)
            remaining = data2.get("predicted_remaining_laps", 0)
            action = data2.get("recommended_action", "—").upper()
            cliff = data2.get("cliff_expected_lap", "—")
            delta = data2.get("laptime_delta_vs_fresh", 0)

            action_color = {"MONITOR": "normal", "PIT_SOON": "inverse", "CRITICAL": "off"}.get(action, "normal")

            st.metric("Degradation", f"{deg:.1f}%", delta=f"+{delta:.2f}s vs fresh")
            st.metric("Remaining life", f"{remaining} laps")
            st.metric("Recommended action", action)
            st.caption(f"Cliff at lap {cliff} · Lambda: **{latency2:.0f} ms** · model: `{data2.get('model_version')}`")
        else:
            st.error(f"API error {status2}")

    # ── Pit strategy ─────────────────────────────────────────────────────────
    with col3:
        strat_payload = {
            "driver": driver,
            "track": track.lower().replace(" ", "_"),
            "current_lap": current_lap,
            "total_laps": total_laps,
            "tire_age_laps": tire_age,
            "gap_ahead_seconds": gap_ahead,
            "gap_behind_seconds": gap_behind,
            "current_position": position,
        }
        with st.spinner("Calling /predict/pit-strategy …"):
            data3, latency3, status3 = call_api("predict/pit-strategy", strat_payload)

        st.markdown("### Pit Strategy")

        if data3:
            rec = data3.get("recommendation", "—")
            window = data3.get("optimal_pit_window", [])
            pred_pos = data3.get("predicted_position_after_pit", "—")
            tire_life = data3.get("estimated_tire_life_remaining", "—")

            REC_EMOJI = {
                "pit_now": "🚨 PIT NOW",
                "extend_5_laps": "⚠️ EXTEND 5 LAPS",
                "no_stop_window": "✅ STAY OUT",
            }
            st.markdown(f"## {REC_EMOJI.get(rec, rec)}")
            if window:
                st.metric("Optimal window", f"Lap {window[0]}–{window[1]}")
            st.metric("Position after pit", pred_pos)
            st.metric("Tire life remaining", f"{tire_life} laps")
            st.caption(f"Lambda: **{latency3:.0f} ms** · model: `{data3.get('model_version')}`")
        else:
            st.error(f"API error {status3}")

    # ── Stint simulation ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Stint Simulation — Lap Time Degradation")
    st.caption("How lap time evolves as tires age over a full stint (calls /predict/laptime for each lap)")

    if st.button("Run stint simulation (25 laps)", use_container_width=False):
        progress = st.progress(0, text="Simulating…")
        stint_times = []
        stint_laps = list(range(1, 26))

        for i, age in enumerate(stint_laps):
            p = {**payload_base, "tire_age_laps": age, "fuel_load_kg": max(5, fuel_load - i * 1.6)}
            d, _, _ = call_api("predict/laptime", p)
            if d:
                mins, secs = d["predicted_laptime"].split(":")
                stint_times.append(int(mins) * 60 + float(secs))
            else:
                stint_times.append(None)
            progress.progress((i + 1) / len(stint_laps), text=f"Lap {age}/25…")

        progress.empty()

        valid = [(l, t) for l, t in zip(stint_laps, stint_times) if t is not None]
        if valid:
            import pandas as pd
            df = pd.DataFrame(valid, columns=["Tire age (laps)", "Predicted lap time (s)"])
            st.line_chart(df.set_index("Tire age (laps)"))
            fastest = min(t for _, t in valid)
            slowest = max(t for _, t in valid)
            st.caption(
                f"Fastest: `{int(fastest//60)}:{fastest%60:06.3f}` (lap 1) · "
                f"Slowest: `{int(slowest//60)}:{slowest%60:06.3f}` (lap 25) · "
                f"Deg over stint: **+{slowest-fastest:.2f}s**"
            )
