"""1 · Operator Console — live heat playback with the coloured furnace view,
streaming KPIs, tap advice and an event log. This is the shop-floor screen."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import engine as E          # noqa: E402
import ui                    # noqa: E402

st.set_page_config(page_title="Operator Console · SmartMelt", page_icon="🔥",
                   layout="wide")
ui.inject_css()

cfg = E.get_config(st.session_state.get("plant_choice", "if_msme_12t"))
summary = E.config_summary(cfg)
heat_t = summary["Heat size (t)"]
tap_aim = summary["Tap aim (°C)"]

st.markdown("## 🔥 Operator Console")
st.markdown('<span class="muted">Live melt state — the furnace, the numbers, and '
            'the next action. Advisory only.</span>', unsafe_allow_html=True)

# ── controls ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Heat setup")
    charge_t = st.slider("Charge (t)", 4.0, float(max(6, heat_t * 1.1)), float(heat_t), 0.5)
    power_kW = st.slider("Power — melt-in (kW)", 1000, 8000,
                         int(summary.get("Rated power (kW)") or 5200), 100)
    c_pct = st.slider("Charge carbon (%)", 0.1, 1.5, 0.6, 0.05)
    cu_pct = st.slider("Charge copper (%)", 0.05, 0.5, 0.20, 0.01)
    st.markdown("### Additions")
    add_lime = st.checkbox("Lime 48 kg @ 10 min", value=True)
    add_fesi = st.checkbox("FeSi75 15 kg @ 45 min", value=True)
    add_scale = st.checkbox("Mill scale 150 kg @ 60 min", value=True,
                            help="(FeO)+[C]→Fe+CO — decarburiser & coolant (E27c)")
    st.markdown("### Playback")
    speed = st.select_slider("Speed", options=[1, 5, 10, 30, 60], value=30,
                             help="simulated minutes per animation frame")


# ── build + run the heat (cached) ───────────────────────────────────────────
@st.cache_data(show_spinner="Simulating heat…")
def simulate(plant_name, charge_t, power_kW, c_pct, cu_pct,
             add_lime, add_fesi, add_scale):
    cfg = E.get_config(plant_name)
    comp = dict(E.DEFAULT_CHARGE_COMP)
    comp["C"] = c_pct / 100.0
    comp["Cu"] = cu_pct / 100.0
    specs = []
    if add_lime:
        specs.append(E.AdditionSpec("Lime (92% CaO)", 10, 48))
    if add_fesi:
        specs.append(E.AdditionSpec("FeSi75", 45, 15))
    if add_scale:
        specs.append(E.AdditionSpec("Mill scale (FeO)", 60, 150))
    r = E.run_heat(cfg, charge_t * 1000.0, comp, power_kW,
                   additions=E.build_additions(specs), dt=2.0)
    return r


res = simulate(st.session_state.get("plant_choice", "if_msme_12t"),
               charge_t, power_kW, c_pct, cu_pct, add_lime, add_fesi, add_scale)
df = res.df

# robust slag mass per step (sum of all slag_*_kg columns), used by the furnace view
_slag_cols = [c for c in df.columns if c.startswith("slag_") and c.endswith("_kg")]
df = df.copy()
df["slag_total_kg"] = df[_slag_cols].sum(axis=1) if _slag_cols else 20.0

# scrub position along the heat
if "op_frame" not in st.session_state:
    st.session_state.op_frame = len(df) - 1

top = st.columns([1.2, 1])
with top[0]:
    play = st.button("▶ Play heat", use_container_width=True)
with top[1]:
    if st.button("⏮ Reset to start", use_container_width=True):
        st.session_state.op_frame = 0

t_max = float(df["t_min"].iloc[-1])
scrub = st.slider("Heat time (min)", 0.0, t_max,
                  float(df["t_min"].iloc[st.session_state.op_frame]),
                  0.5, key="scrub_min")
# map scrub minutes → frame index
idx = int(np.searchsorted(df["t_min"].to_numpy(), scrub))
idx = max(0, min(idx, len(df) - 1))
st.session_state.op_frame = idx


def render_state(i: int, placeholders):
    row = df.iloc[i]
    fur_ph, kpi_ph, trend_ph = placeholders

    # ---- coloured furnace ----
    with fur_ph.container():
        svg = ui.furnace_svg(
            melted_pct=row["melted_pct"], T_bath_C=row["T_bath_C"],
            slag_kg=float(row.get("slag_total_kg", 20.0)),
            undissolved_kg=row["undissolved_kg"],
            heat_size_t=heat_t, tap_aim_C=tap_aim,
        )
        st.markdown(svg, unsafe_allow_html=True)
        st.markdown(ui.legend_row(), unsafe_allow_html=True)

    # ---- KPI grid ----
    with kpi_ph.container():
        ready = row["melted_pct"] > 99 and row["T_bath_C"] >= tap_aim - 5
        g1 = st.columns(4)
        with g1[0]:
            sub = "no pool yet" if row["melted_pct"] < 2 else f"aim {tap_aim:.0f}"
            ui.kpi("Bath temp °C", f'{row["T_bath_C"]:.0f}', sub)
        with g1[1]:
            ui.kpi("Carbon %", f'{row["pct_C"]:.3f}', f'Si {row["pct_Si"]:.3f} · Mn {row["pct_Mn"]:.3f}')
        with g1[2]:
            ui.kpi("Melted %", f'{row["melted_pct"]:.0f}', f'undissolved {row["undissolved_kg"]:.0f} kg')
        with g1[3]:
            ui.kpi("SEC kWh/t", f'{row["SEC_kWh_t"]:.0f}', f'{row["E_kWh"]:.0f} kWh total')
        g2 = st.columns(4)
        with g2[0]:
            ui.kpi("Power kW", f'{row.get("Q_useful_kW", np.nan):.0f}'
                   if np.isfinite(row.get("Q_useful_kW", np.nan)) else "—", "useful")
        with g2[1]:
            ui.kpi("Slag FeO %", f'{row["slag_FeO_pct"]:.1f}', f'B2 {row["B2"]:.2f}')
        with g2[2]:
            ui.kpi("P %", f'{row["pct_P"]:.4f}', f'S {row["pct_S"]:.4f}')
        with g2[3]:
            ui.kpi("Time min", f'{row["t_min"]:.0f}', f'tap ≈ {t_max:.0f}')
        # status
        if ready:
            st.markdown(ui.pill("READY TO TAP — on temperature & fully melted", "ok"),
                        unsafe_allow_html=True)
        elif row["melted_pct"] < 2:
            st.markdown(ui.pill("Charging / frozen heel — heating solid charge", "warn"),
                        unsafe_allow_html=True)
        else:
            gap = tap_aim - row["T_bath_C"]
            st.markdown(ui.pill(f"Melting — {gap:.0f} °C below tap aim", "warn"),
                        unsafe_allow_html=True)

    # ---- live trend up to now ----
    with trend_ph.container():
        d = df.iloc[: i + 1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d["t_min"], y=d["T_bath_C"], name="bath °C",
                                 line=dict(color=ui.MOLT, width=2)))
        fig.add_trace(go.Scatter(x=d["t_min"], y=d["T_solid_C"], name="solid °C",
                                 line=dict(color=ui.SCRAP_COL, width=1.5)))
        fig.add_hline(y=tap_aim, line=dict(color=ui.GREEN, dash="dash", width=1))
        fig.add_trace(go.Scatter(x=d["t_min"], y=d["melted_pct"], name="melted %",
                                 yaxis="y2", line=dict(color=ui.STEEL, width=1.5)))
        ui.style_fig(fig, height=280, ylab="°C", y2lab="melted %")
        fig.update_layout(yaxis2=dict(range=[0, 105]))
        # Render into the reused placeholder WITHOUT an explicit key. Streamlit
        # replaces the placeholder's content each call, and unkeyed charts are
        # auto-identified by position, so repeated frames never collide.
        trend_ph.plotly_chart(fig, use_container_width=True)


left, right = st.columns([1, 1.35])
fur_ph = left.empty()
kpi_ph = right.empty()
trend_ph = st.empty()

# animation loop
if play:
    step = max(1, int(len(df) * speed / (t_max * 6)))  # ~scale to speed
    for i in range(st.session_state.op_frame, len(df), step):
        render_state(i, (fur_ph, kpi_ph, trend_ph))
        st.session_state.op_frame = i
        time.sleep(0.05)
    render_state(len(df) - 1, (fur_ph, kpi_ph, trend_ph))
    st.session_state.op_frame = len(df) - 1
else:
    render_state(st.session_state.op_frame, (fur_ph, kpi_ph, trend_ph))

# ── endpoint + advice ───────────────────────────────────────────────────────
st.markdown("---")
ep = res.endpoint
adv = st.columns([1, 1, 1])
with adv[0]:
    hit_T = abs(ep["T_C"] - tap_aim) <= 15
    st.markdown("#### Endpoint")
    st.markdown(f"Tap **{ep['T_C']:.0f} °C** "
                + ui.pill("±15 °C hit" if hit_T else "off aim", "ok" if hit_T else "bad"),
                unsafe_allow_html=True)
    st.markdown(f"Carbon **{ep['pct_C']:.3f} %**")
    st.caption(f"Tap at {res.tap_min:.0f} min · SEC {df['SEC_kWh_t'].iloc[-1]:.0f} kWh/t")
with adv[1]:
    st.markdown("#### Conservation audit")
    ok = res.ledger_max_pct < 1.0
    st.markdown(ui.pill(f"element ledger {res.ledger_max_pct:.2f}%", "ok" if ok else "warn"),
                unsafe_allow_html=True)
    ec = res.energy
    clо = ec.get("residual_pct", ec.get("closure_pct", float("nan")))
    st.caption(f"first-law closure {clо:+.1f}% · undissolved {res.undissolved_kg:.0f} kg at tap")
with adv[2]:
    st.markdown("#### Operator note")
    if res.undissolved_kg > 5:
        st.markdown(ui.pill("late additions not fully dissolved — hold before tap", "warn"),
                    unsafe_allow_html=True)
    elif abs(ep["T_C"] - tap_aim) > 15:
        st.markdown(ui.pill("trim power to hit tap temperature", "warn"),
                    unsafe_allow_html=True)
    else:
        st.markdown(ui.pill("on aim — safe to tap", "ok"), unsafe_allow_html=True)

with st.expander("Heat data (per-step, exportable)"):
    show = df[["t_min", "T_bath_C", "melted_pct", "pct_C", "pct_Si", "pct_Mn",
               "slag_FeO_pct", "B2", "SEC_kWh_t", "undissolved_kg"]].round(3)
    st.dataframe(show, use_container_width=True, height=260)
    st.download_button("Download heat CSV", show.to_csv(index=False),
                       "smartmelt_heat.csv", "text/csv")
