"""6 · Drift Monitor — population-stability (PSI) drift detection. Sets a
reference window on early heats, then flags when incoming scrap or practice
shifts, so the model widens its uncertainty instead of guessing."""
from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import engine as E          # noqa: E402
import ui                    # noqa: E402

st.set_page_config(page_title="Drift Monitor · SmartMelt", page_icon="📡",
                   layout="wide")
ui.inject_css()

cfg = E.get_config(st.session_state.get("plant_choice", "if_msme_12t"))

st.markdown("## 📡 Drift Monitor")
st.markdown('<span class="muted">When your scrap or practice changes, the model should '
            'know. PSI compares recent heats against a reference window and raises an '
            'alarm before accuracy quietly degrades.</span>', unsafe_allow_html=True)

st.info("This simulates a run where copper creeps up partway through (a scrap-quality "
        "regime change), so you can see the alarm fire. Datasets are cached.", icon="⏱️")

with st.sidebar:
    st.markdown("### Run")
    n_heats = st.slider("Heats", 30, 80, 50, 5)
    regime = st.slider("Regime change at heat", 15, 60, 32, 1,
                       help="copper step-change point")
    ref_frac = st.slider("Reference window fraction", 0.3, 0.7, 0.5, 0.05)
    go_btn = st.button("Generate & check", use_container_width=True, type="primary")


@st.cache_data(show_spinner="Simulating heats & checking drift…")
def build(plant, n_heats, regime, ref_frac):
    cfg = E.get_config(plant)
    df = E.generate_dataset(cfg, n_heats=n_heats, seed=0, regime_change_at=regime)
    dr = E.run_drift(cfg, df, ref_frac=ref_frac)
    return df, dr


if go_btn or st.session_state.get("drift_ran"):
    st.session_state["drift_ran"] = True
    df, dr = build(st.session_state.get("plant_choice", "if_msme_12t"),
                   n_heats, regime, ref_frac)

    k = st.columns(4)
    with k[0]:
        kind = "bad" if dr["alarm"] else "ok"
        st.markdown(f'#### Status')
        st.markdown(ui.pill("DRIFT ALARM" if dr["alarm"] else "stable", kind),
                    unsafe_allow_html=True)
    with k[1]:
        ui.kpi("Max PSI", f'{dr["psi_max"]:.2f}',
               "PSI>0.25 = shift · >0.5 = major")
    with k[2]:
        ui.kpi("Reference", f'{dr["n_ref"]} heats', "baseline window")
    with k[3]:
        ui.kpi("Recent", f'{dr["n_recent"]} heats', "checked window")

    if dr["reasons"]:
        st.markdown("**Reasons flagged:** " +
                    "  ".join(ui.pill(r, "warn") for r in dr["reasons"]),
                    unsafe_allow_html=True)

    # PSI per feature
    st.markdown("#### Population drift by feature (PSI)")
    psi = dr["psi_df"].head(14)
    colors = [ui.RED if v > 0.5 else ui.AMBER if v > 0.25 else ui.STEEL
              for v in psi["PSI"]]
    fig = go.Figure(go.Bar(x=psi["PSI"], y=psi["feature"], orientation="h",
                           marker_color=colors))
    fig.add_vline(x=0.25, line=dict(color=ui.AMBER, dash="dash", width=1))
    fig.add_vline(x=0.50, line=dict(color=ui.RED, dash="dash", width=1))
    ui.style_fig(fig, height=440, ylab="")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    # the drifting variable over the run
    st.markdown("#### The variable that moved")
    cu_col = "charge_Cu_pct" if "charge_Cu_pct" in df else df.columns[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[cu_col], mode="lines+markers",
                             name=cu_col, line=dict(color=ui.MOLT)))
    fig.add_vline(x=regime, line=dict(color=ui.RED, dash="dot"))
    fig.add_vrect(x0=0, x1=dr["n_ref"], fillcolor="rgba(79,168,216,0.08)", line_width=0,
                  annotation_text="reference", annotation_position="top left")
    ui.style_fig(fig, height=300, ylab=cu_col)
    fig.update_xaxes(title_text="heat number")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("PSI ≥ 0.25 signals a meaningful population shift; ≥ 0.5 is major. The "
               "monitor flags the copper step-change so the operator knows incoming "
               "scrap has changed and the model inflates its uncertainty accordingly.")
else:
    st.markdown("Set parameters in the sidebar and press **Generate & check**.")
