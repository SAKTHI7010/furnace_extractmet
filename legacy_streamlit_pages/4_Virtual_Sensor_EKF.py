"""4 · Virtual Sensor (EKF) — the Extended Kalman Filter tracks bath temperature
on a deliberately mismatched plant, assimilating only occasional immersion dips,
and converges the hidden furnace efficiency within a single heat."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import engine as E          # noqa: E402
import ui                    # noqa: E402

st.set_page_config(page_title="Virtual Sensor · SmartMelt", page_icon="🛰️",
                   layout="wide")
ui.inject_css()

cfg = E.get_config(st.session_state.get("plant_choice", "if_msme_12t"))
summary = E.config_summary(cfg)
heat_t = summary["Heat size (t)"]

st.markdown("## 🛰️ Virtual Temperature Sensor (EKF)")
st.markdown('<span class="muted">The estimator starts wrong on purpose — the true '
            'furnace is less efficient and loses more heat than the prior. Watch it '
            'lock on from a handful of immersion dips.</span>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Mismatched plant")
    true_eta = st.slider("True electrical efficiency", 0.80, 1.00, 0.90, 0.01,
                         help="prior is 1.00 — the EKF must find this")
    true_UA = st.slider("True wall-loss scale", 0.8, 1.8, 1.35, 0.05,
                        help="prior is 1.00")
    n_dips = st.slider("Immersion dips", 1, 6, 3, 1)
    power_kW = st.slider("Power (kW)", 2000, 8000,
                         int(summary.get("Rated power (kW)") or 5000), 100)


@st.cache_data(show_spinner="Running EKF over a mismatched heat…")
def run(plant, true_eta, true_UA, n_dips, power_kW):
    cfg = E.get_config(plant)
    dips = tuple(np.linspace(30, 78, n_dips))
    return E.run_ekf_demo(cfg, power_kW=power_kW, true_eta=true_eta,
                          true_UA_scale=true_UA, dip_times_min=dips, seed=1)


ek = run(st.session_state.get("plant_choice", "if_msme_12t"),
         true_eta, true_UA, n_dips, power_kW)
df = ek.df

k = st.columns(4)
with k[0]:
    ok = abs(ek.final_error_C) <= 15
    ui.kpi("Final error", f"{ek.final_error_C:+.1f} °C", "estimate − truth at tap")
with k[1]:
    eta_hat = ek.theta_path["eta_electrical"].iloc[-1]
    ui.kpi("η̂ electrical", f"{eta_hat:.3f}", f"true {true_eta:.2f}")
with k[2]:
    ui.kpi("σ_T start→end", f'{df["sigma_T"].iloc[0]:.0f}→{df["sigma_T"].iloc[-1]:.1f}',
           "°C · uncertainty collapse")
with k[3]:
    ui.kpi("Dips used", f"{len(ek.dip_df)}", "immersion measurements")

c = st.columns([1.4, 1])

# tracking with ±2σ band
with c[0]:
    st.markdown("#### Bath temperature — truth vs estimate")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["t_min"], y=df["T_est_C"] + 2 * df["sigma_T"],
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df["t_min"], y=df["T_est_C"] - 2 * df["sigma_T"],
                             fill="tonexty", fillcolor="rgba(232,98,46,0.18)",
                             line=dict(width=0), name="±2σ"))
    fig.add_trace(go.Scatter(x=df["t_min"], y=df["T_true_C"], name="true",
                             line=dict(color="#cfd6dd", width=2)))
    fig.add_trace(go.Scatter(x=df["t_min"], y=df["T_est_C"], name="EKF estimate",
                             line=dict(color=ui.MOLT, width=2)))
    if len(ek.dip_df):
        fig.add_trace(go.Scatter(x=ek.dip_df["t_min"], y=ek.dip_df["T_meas_C"],
                                 mode="markers", name="immersion dip",
                                 marker=dict(color=ui.STEEL, size=10, symbol="diamond")))
    ui.style_fig(fig, height=380, ylab="°C")
    st.plotly_chart(fig, use_container_width=True)

# theta convergence
with c[1]:
    st.markdown("#### Tracked parameters")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ek.theta_path["t_min"], y=ek.theta_path["eta_electrical"],
                             name="η electrical", line=dict(color=ui.MOLT)))
    fig.add_hline(y=true_eta, line=dict(color=ui.MOLT, dash="dash", width=1))
    if "UA_lining_scale" in ek.theta_path:
        fig.add_trace(go.Scatter(x=ek.theta_path["t_min"], y=ek.theta_path["UA_lining_scale"],
                                 name="UA wall scale", line=dict(color=ui.STEEL)))
        fig.add_hline(y=true_UA, line=dict(color=ui.STEEL, dash="dash", width=1))
    ui.style_fig(fig, height=380, ylab="value")
    st.plotly_chart(fig, use_container_width=True)

st.caption("Dashed lines are the true (hidden) values. The filter runs on a continuous "
           "noisy pyrometer plus intermittent immersion dips — exactly the sensor suite "
           "of a real melt shop. This is the temperature the operator console trusts "
           "between dips.")
