"""2 · Process Trajectory — the full six-panel physics view of one heat:
temperatures, inventories & dissolution, metal chemistry, slag & basicity,
heat flows, and energy vs the theoretical floor."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import engine as E          # noqa: E402
import ui                    # noqa: E402

st.set_page_config(page_title="Process Trajectory · SmartMelt", page_icon="📈",
                   layout="wide")
ui.inject_css()

cfg = E.get_config(st.session_state.get("plant_choice", "if_msme_12t"))
summary = E.config_summary(cfg)
tap_aim = summary["Tap aim (°C)"]
heat_t = summary["Heat size (t)"]
floor = E.theoretical_floor_kWh_t(cfg)

st.markdown("## 📈 Process Trajectory")
st.markdown('<span class="muted">One reference heat, every state variable from the '
            'same audited physics run.</span>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Heat setup")
    charge_t = st.slider("Charge (t)", 4.0, float(max(6, heat_t * 1.1)), float(heat_t), 0.5)
    power_kW = st.slider("Power (kW)", 1000, 8000,
                         int(summary.get("Rated power (kW)") or 5200), 100)
    c_pct = st.slider("Charge carbon (%)", 0.1, 1.5, 0.6, 0.05)
    scale_kg = st.slider("Mill scale @ 60 min (kg)", 0, 300, 150, 10,
                         help="(FeO)+[C]→Fe+CO decarburiser/coolant")


@st.cache_data(show_spinner="Running physics…")
def run(plant, charge_t, power_kW, c_pct, scale_kg):
    cfg = E.get_config(plant)
    comp = dict(E.DEFAULT_CHARGE_COMP); comp["C"] = c_pct / 100.0
    specs = [E.AdditionSpec("Lime (92% CaO)", 10, 48),
             E.AdditionSpec("FeSi75", 45, 15)]
    if scale_kg > 0:
        specs.append(E.AdditionSpec("Mill scale (FeO)", 60, scale_kg))
    return E.run_heat(cfg, charge_t * 1000.0, comp, power_kW,
                      additions=E.build_additions(specs), dt=2.0)


res = run(st.session_state.get("plant_choice", "if_msme_12t"),
          charge_t, power_kW, c_pct, scale_kg)
d = res.df
t = d["t_min"]

# KPI strip
k = st.columns(5)
with k[0]:
    ui.kpi("Tap temp", f'{res.endpoint["T_C"]:.0f} °C', f'aim {tap_aim}')
with k[1]:
    ui.kpi("Carbon", f'{res.endpoint["pct_C"]:.3f} %', f'from {c_pct:.2f}')
with k[2]:
    ui.kpi("Tap time", f'{res.tap_min:.0f} min', 'to full melt')
with k[3]:
    ui.kpi("SEC", f'{d["SEC_kWh_t"].iloc[-1]:.0f}', f'floor {floor:.0f} kWh/t')
with k[4]:
    ui.kpi("Ledger", f'{res.ledger_max_pct:.2f} %', 'element closure')

fig = make_subplots(
    rows=2, cols=3, vertical_spacing=0.13, horizontal_spacing=0.08,
    subplot_titles=("Temperatures", "Inventories & dissolution", "Bath composition",
                    "Slag & basicity", "Heat flows", "Energy & specific energy"),
    specs=[[{"secondary_y": True}, {"secondary_y": True}, {}],
           [{"secondary_y": True}, {}, {"secondary_y": True}]],
)

# 1 temperatures
fig.add_trace(go.Scatter(x=t, y=d["T_bath_C"], name="bath", line=dict(color=ui.MOLT)), 1, 1)
fig.add_trace(go.Scatter(x=t, y=d["T_solid_C"], name="solid", line=dict(color=ui.SCRAP_COL)), 1, 1)
if "T_hotface_C" in d:
    fig.add_trace(go.Scatter(x=t, y=d["T_hotface_C"], name="hot face",
                             line=dict(color=ui.SLAG_TOP, dash="dot")), 1, 1)
fig.add_hline(y=tap_aim, line=dict(color=ui.GREEN, dash="dash", width=1), row=1, col=1)

# 2 inventories & dissolution
fig.add_trace(go.Scatter(x=t, y=d["M_solid_t"], name="solid t", line=dict(color=ui.SCRAP_COL)), 1, 2)
fig.add_trace(go.Scatter(x=t, y=d["M_liquid_t"], name="liquid t", line=dict(color=ui.MOLT)), 1, 2)
fig.add_trace(go.Scatter(x=t, y=d["undissolved_kg"], name="undissolved kg",
                         line=dict(color=ui.STEEL, width=1)), 1, 2, secondary_y=True)

# 3 metal composition
for el, c in [("C", ui.MOLT), ("Si", ui.STEEL), ("Mn", ui.GREEN), ("S", ui.SLAG_TOP)]:
    if f"pct_{el}" in d:
        fig.add_trace(go.Scatter(x=t, y=d[f"pct_{el}"], name=el, line=dict(color=c)), 1, 3)

# 4 slag & basicity
fig.add_trace(go.Scatter(x=t, y=d["slag_FeO_pct"], name="FeO %", line=dict(color=ui.MOLT)), 2, 1)
fig.add_trace(go.Scatter(x=t, y=d["B2"], name="B2", line=dict(color=ui.STEEL)), 2, 1, secondary_y=True)

# 5 heat flows
for key, c, nm in [("Q_wall_kW", ui.SLAG_TOP, "wall"), ("Q_rad_kW", ui.MOLT, "radiation"),
                   ("Q_bath_to_scrap_kW", ui.SCRAP_COL, "bath→scrap"),
                   ("Q_chem_kW", ui.GREEN, "chemistry")]:
    if key in d:
        fig.add_trace(go.Scatter(x=t, y=d[key], name=nm, line=dict(color=c)), 2, 2)

# 6 energy & SEC
fig.add_trace(go.Scatter(x=t, y=d["E_kWh"], name="energy kWh", line=dict(color=ui.SCRAP_COL)), 2, 3)
fig.add_trace(go.Scatter(x=t, y=d["SEC_kWh_t"], name="SEC kWh/t",
                         line=dict(color=ui.MOLT)), 2, 3, secondary_y=True)
fig.add_hline(y=floor, line=dict(color=ui.GREEN, dash="dash", width=1),
              row=2, col=3, secondary_y=True)

fig.update_layout(template="plotly_dark", height=640, showlegend=True,
                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f1418",
                  font=dict(family="DejaVu Sans", size=11, color="#c6ccd4"),
                  legend=dict(orientation="h", y=1.06, font=dict(size=10)),
                  margin=dict(l=45, r=25, t=60, b=35))
fig.update_xaxes(gridcolor="#20262c"); fig.update_yaxes(gridcolor="#20262c")
for r_ in (1, 2):
    fig.update_xaxes(title_text="min", row=r_, col=1)
    fig.update_xaxes(title_text="min", row=r_, col=2)
    fig.update_xaxes(title_text="min", row=r_, col=3)

if scale_kg > 0:
    for cc in (1, 2, 3):
        fig.add_vline(x=60, line=dict(color=ui.RED, dash="dot", width=0.8),
                      row=1, col=cc)

st.plotly_chart(fig, use_container_width=True)
st.caption(f"Reversible melting floor {floor:.0f} kWh/t shown dashed (practical IF "
           f"floor ≈ 500 kWh/t). Red line marks the mill-scale addition; watch carbon "
           f"fall and iron return as (FeO)+[C]→Fe+CO consumes it.")
