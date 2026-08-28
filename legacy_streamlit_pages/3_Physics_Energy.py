"""3 · Physics & Energy — where every kilowatt goes, and proof the model
conserves mass and energy. Heat-flow ledger, first-law closure, and an
energy-split waterfall from grid input to tapped steel."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import engine as E          # noqa: E402
import ui                    # noqa: E402

st.set_page_config(page_title="Physics & Energy · SmartMelt", page_icon="⚡",
                   layout="wide")
ui.inject_css()

cfg = E.get_config(st.session_state.get("plant_choice", "if_msme_12t"))
summary = E.config_summary(cfg)
heat_t = summary["Heat size (t)"]
floor = E.theoretical_floor_kWh_t(cfg)

st.markdown("## ⚡ Physics & Energy")
st.markdown('<span class="muted">Every kilowatt accounted for — heat flows, '
            'conservation audit, and the energy split from grid to tap.</span>',
            unsafe_allow_html=True)

with st.sidebar:
    charge_t = st.slider("Charge (t)", 4.0, float(max(6, heat_t * 1.1)), float(heat_t), 0.5)
    power_kW = st.slider("Power (kW)", 1000, 8000,
                         int(summary.get("Rated power (kW)") or 5200), 100)


@st.cache_data(show_spinner="Running physics…")
def run(plant, charge_t, power_kW):
    cfg = E.get_config(plant)
    specs = [E.AdditionSpec("Lime (92% CaO)", 10, 48),
             E.AdditionSpec("FeSi75", 45, 15),
             E.AdditionSpec("Mill scale (FeO)", 60, 150)]
    return E.run_heat(cfg, charge_t * 1000.0, E.DEFAULT_CHARGE_COMP, power_kW,
                      additions=E.build_additions(specs), dt=2.0)


res = run(st.session_state.get("plant_choice", "if_msme_12t"), charge_t, power_kW)
d = res.df
en = res.energy

# ── conservation KPIs ───────────────────────────────────────────────────────
k = st.columns(4)
with k[0]:
    ok = res.ledger_max_pct < 1.0
    ui.kpi("Element ledger", f"{res.ledger_max_pct:.2f} %", "mass closure, worst species")
with k[1]:
    clo = en.get("residual_pct", en.get("closure_pct", float("nan")))
    ui.kpi("First-law closure", f"{clo:+.1f} %", "energy in − out")
with k[2]:
    ui.kpi("Final SEC", f'{d["SEC_kWh_t"].iloc[-1]:.0f}', f"floor {floor:.0f} kWh/t")
with k[3]:
    eff = 100 * floor / max(d["SEC_kWh_t"].iloc[-1], 1)
    ui.kpi("vs floor", f"{eff:.0f} %", "of reversible minimum")

col = st.columns(2)

# ── heat-flow ledger over time ──────────────────────────────────────────────
with col[0]:
    st.markdown("#### Heat flows through the heat")
    fig = go.Figure()
    flows = [("Q_useful_kW", ui.MOLT, "useful (to charge)"),
             ("Q_wall_kW", ui.SLAG_TOP, "wall loss"),
             ("Q_rad_kW", ui.RED, "radiation loss"),
             ("Q_chem_kW", ui.GREEN, "chemistry"),
             ("Q_offgas_kW", ui.STEEL, "off-gas"),
             ("Q_cool_kW", ui.MUT, "coil cooling")]
    for key, c, nm in flows:
        if key in d:
            fig.add_trace(go.Scatter(x=d["t_min"], y=d[key], name=nm,
                                     line=dict(color=c, width=1.6)))
    ui.style_fig(fig, height=340, ylab="kW")
    st.plotly_chart(fig, use_container_width=True)

# ── energy split (where the grid kWh ended up) ──────────────────────────────
with col[1]:
    st.markdown("#### Energy split — grid input to tapped steel")
    # Use the engine's audit (authoritative kWh), not integrated flows.
    total = en.get("grid_kWh", d["E_kWh"].iloc[-1])
    conv = en.get("converter_loss_kWh", 0.0)
    coil = en.get("coil_water_loss_kWh", 0.0)
    wall = en.get("lining_loss_kWh", 0.0)
    rad = en.get("radiation_loss_kWh", 0.0)
    gas = en.get("offgas_loss_kWh", 0.0)
    useful = en.get("useful_melt_kWh", 0.0)

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "relative", "relative", "total"],
        x=["Grid input", "− converter", "− coil water", "− lining", "− radiation",
           "− off-gas", "To steel"],
        y=[total, -conv, -coil, -wall, -rad, -gas, None],
        connector=dict(line=dict(color="#3a444d")),
        decreasing=dict(marker=dict(color=ui.RED)),
        increasing=dict(marker=dict(color=ui.STEEL)),
        totals=dict(marker=dict(color=ui.MOLT)),
        text=[f"{v:.0f}" for v in [total, -conv, -coil, -wall, -rad, -gas, useful]],
        textposition="outside",
    ))
    ui.style_fig(fig, height=340, ylab="kWh")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"From the engine's energy audit. Useful melt energy is "
               f"{100*en.get('useful_fraction',0):.0f}% of grid input; the remainder is "
               f"converter, coil-water, lining, radiation and off-gas losses.")

# ── energy audit table ──────────────────────────────────────────────────────
st.markdown("#### Energy audit (kWh)")
audit_rows = {kk: v for kk, v in en.items() if isinstance(v, (int, float))}
import pandas as pd
audit_df = pd.DataFrame([{"quantity": kk, "value": round(v, 1)}
                         for kk, v in audit_rows.items()])
st.dataframe(audit_df, use_container_width=True, height=min(360, 40 + 28 * len(audit_df)))
st.caption("The audit is computed by the engine's own energy_audit(); the residual "
           "line is the first-law check reported on every heat.")
