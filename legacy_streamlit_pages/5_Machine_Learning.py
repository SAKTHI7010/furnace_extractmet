"""5 · Machine Learning — the hybrid endpoint model. Physics predicts; a GP
residual head learns the plant-specific gap ONLY when it can prove out-of-time
improvement (maturity gating). Shows the ML lift over physics and parity plots."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import engine as E          # noqa: E402
import ui                    # noqa: E402

st.set_page_config(page_title="Machine Learning · SmartMelt", page_icon="🧠",
                   layout="wide")
ui.inject_css()

cfg = E.get_config(st.session_state.get("plant_choice", "if_msme_12t"))

st.markdown("## 🧠 Hybrid Endpoint Model")
st.markdown('<span class="muted">Physics + a machine-learning residual that corrects '
            'it — and switches itself off until it can prove it beats physics on '
            'time-ordered data.</span>', unsafe_allow_html=True)

st.info("Generating heats runs the full physics simulator (~3–4 s each), so datasets "
        "are cached. Start small; increase once you've seen it work.", icon="⏱️")

with st.sidebar:
    st.markdown("### Dataset")
    n_heats = st.slider("Heats to simulate", 20, 80, 40, 5,
                        help="ML residual activates around 25+ heats")
    split = st.slider("Train fraction", 0.5, 0.85, 0.7, 0.05)
    seed = st.number_input("Seed", 0, 99, 0)
    go_btn = st.button("Generate & train", use_container_width=True, type="primary")


@st.cache_data(show_spinner="Simulating heats & fitting the hybrid model…")
def build(plant, n_heats, split, seed):
    cfg = E.get_config(plant)
    df = E.generate_dataset(cfg, n_heats=n_heats, seed=seed)
    ml = E.train_hybrid(cfg, df, split_frac=split)
    return df, ml


if go_btn or st.session_state.get("ml_ran"):
    st.session_state["ml_ran"] = True
    df, ml = build(st.session_state.get("plant_choice", "if_msme_12t"),
                   n_heats, split, seed)
    m = ml.metrics
    p = ml.pred_df

    # maturity + activation
    mats = {"insufficient": "bad", "coldstart": "warn",
            "deployable": "warn", "calibrated": "ok"}
    kind = mats.get(m["maturity"], "warn")
    st.markdown(
        f'Model maturity: {ui.pill(m["maturity"], kind)} &nbsp; '
        f'Temperature ML: {ui.pill("active" if m["ml_T_active"] else "gated off (physics-only)", "ok" if m["ml_T_active"] else "warn")} &nbsp; '
        f'Carbon ML: {ui.pill("active" if m["ml_C_active"] else "gated off (physics-only)", "ok" if m["ml_C_active"] else "warn")}',
        unsafe_allow_html=True)
    st.caption(f"{m['n_train']} training heats · {m['n_test']} test heats (time-ordered "
               f"split). The residual head only engages when it beats raw physics in "
               f"rolling-origin cross-validation.")

    # KPI strip: ML vs physics
    k = st.columns(4)
    with k[0]:
        ui.kpi("T hit ±15 °C", f'{m["T_hit_15C"]:.0f} %',
               f'physics {m["T_hit_15C_phys"]:.0f} %')
    with k[1]:
        ui.kpi("T MAE", f'{m["T_MAE_C"]:.1f} °C', 'hybrid')
    with k[2]:
        ui.kpi("C hit ±0.02 %", f'{m["C_hit_002"]:.0f} %',
               f'physics {m["C_hit_002_phys"]:.0f} %')
    with k[3]:
        ui.kpi("C MAE", f'{m["C_MAE"]:.3f} %', 'hybrid')

    c = st.columns(2)
    # temperature parity
    with c[0]:
        st.markdown("#### Temperature — predicted vs actual")
        fig = go.Figure()
        lo = float(np.nanmin([p["T_true_C"].min(), p["T_pred_C"].min()])) - 10
        hi = float(np.nanmax([p["T_true_C"].max(), p["T_pred_C"].max()])) + 10
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                 line=dict(color=ui.MUT, dash="dash"), showlegend=False))
        fig.add_trace(go.Scatter(x=p["T_true_C"], y=p["T_phys_C"], mode="markers",
                                 name="physics only",
                                 marker=dict(color=ui.SCRAP_COL, size=8, symbol="x")))
        fig.add_trace(go.Scatter(x=p["T_true_C"], y=p["T_pred_C"], mode="markers",
                                 name="hybrid", error_y=dict(
                                     type="data", array=2 * p["T_sigma"], visible=True,
                                     color="rgba(232,98,46,0.35)"),
                                 marker=dict(color=ui.MOLT, size=9)))
        ui.style_fig(fig, height=360, ylab="predicted °C")
        fig.update_xaxes(title_text="actual °C")
        st.plotly_chart(fig, use_container_width=True)

    # carbon parity
    with c[1]:
        st.markdown("#### Carbon — predicted vs actual")
        fig = go.Figure()
        lo = float(np.nanmin([p["C_true"].min(), p["C_pred"].min()])) - 0.02
        hi = float(np.nanmax([p["C_true"].max(), p["C_pred"].max()])) + 0.02
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                 line=dict(color=ui.MUT, dash="dash"), showlegend=False))
        fig.add_trace(go.Scatter(x=p["C_true"], y=p["C_phys"], mode="markers",
                                 name="physics only",
                                 marker=dict(color=ui.SCRAP_COL, size=8, symbol="x")))
        fig.add_trace(go.Scatter(x=p["C_true"], y=p["C_pred"], mode="markers",
                                 name="hybrid", marker=dict(color=ui.STEEL, size=9)))
        ui.style_fig(fig, height=360, ylab="predicted %C")
        fig.update_xaxes(title_text="actual %C")
        st.plotly_chart(fig, use_container_width=True)

    # per-heat error over the test set
    st.markdown("#### Test-set temperature error, heat by heat")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=p["heat"], y=p["T_pred_C"] - p["T_true_C"],
                         name="hybrid error", marker_color=ui.MOLT))
    fig.add_trace(go.Bar(x=p["heat"], y=p["T_phys_C"] - p["T_true_C"],
                         name="physics error", marker_color=ui.SCRAP_COL, opacity=0.6))
    fig.add_hrect(y0=-15, y1=15, fillcolor="rgba(51,193,122,0.10)", line_width=0)
    ui.style_fig(fig, height=300, ylab="pred − actual °C")
    fig.update_layout(barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Green band is the ±15 °C tolerance. When maturity is 'insufficient' the "
               "ML head stays off by design and the hybrid equals physics — which is the "
               "honest behaviour, not a bug.")
else:
    st.markdown("Set a dataset size in the sidebar and press **Generate & train**.")
