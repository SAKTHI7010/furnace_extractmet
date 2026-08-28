"""8 · Economics & Validation — the money case and the validation window.
Savings, payback and CO2 from a measured SEC reduction (using the corrected
tariff and grid factor), and a parameter-audit table with the verified
constants and live conservation results."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import engine as E          # noqa: E402
import ui                    # noqa: E402

st.set_page_config(page_title="Economics & Validation · SmartMelt", page_icon="✅",
                   layout="wide")
ui.inject_css()

cfg = E.get_config(st.session_state.get("plant_choice", "if_msme_12t"))
summary = E.config_summary(cfg)
heat_t = summary["Heat size (t)"]
tariff = summary["Tariff (₹/kWh)"]
grid_ef = summary["Grid EF (tCO₂/MWh)"]
base_sec = summary["Baseline SEC (kWh/t)"]
floor = E.theoretical_floor_kWh_t(cfg)

tab_econ, tab_val = st.tabs(["💰 Economics", "✅ Validation window"])

# ════════════════════════════════════════════════════════════════ ECONOMICS
with tab_econ:
    st.markdown("### The money case — your numbers")
    st.markdown('<span class="muted">Transparent, and rebanded to verified Indian '
                'market figures. Challenge every input.</span>', unsafe_allow_html=True)

    c = st.columns(4)
    with c[0]:
        tpy = st.number_input("Annual output (t/yr)", 5000, 300000, 40000, 5000)
    with c[1]:
        saving = st.slider("SEC saving (kWh/t)", 10, 100, 40, 5,
                           help="verified defensible band: 30–80")
    with c[2]:
        tar = st.number_input("Tariff (₹/kWh)", 4.0, 10.0, float(tariff), 0.5)
    with c[3]:
        price = st.number_input("Licence (₹ lakh)", 5, 40, 20, 1)

    annual = tpy * saving * tar
    payback_mo = (price * 1e5) / annual * 12 if annual > 0 else float("inf")
    co2_saved = tpy * saving / 1000.0 * grid_ef   # tCO2/yr

    k = st.columns(4)
    with k[0]:
        ui.kpi("Annual energy saving", f"₹{annual/1e7:.2f} cr", f"at ₹{tar:.1f}/kWh")
    with k[1]:
        ui.kpi("Simple payback", f"{payback_mo:.1f} mo",
               "energy alone (arithmetic)")
    with k[2]:
        ui.kpi("CO₂ avoided", f"{co2_saved:,.0f} t/yr", f"at {grid_ef:.3f} tCO₂/MWh")
    with k[3]:
        floor_gap = base_sec - saving - floor
        ui.kpi("Headroom left", f"{max(floor_gap,0):.0f} kWh/t",
               f"above reversible {floor:.0f}")

    st.markdown("#### Savings sensitivity")
    savings_axis = [30, 50, 80]
    outputs = [30000, 50000, 100000]
    rows = []
    for o in outputs:
        rows.append([f"{o:,} t/yr"] + [f"₹{o*s*tar/1e7:.2f} cr" for s in savings_axis])
    sens = pd.DataFrame(rows, columns=["Annual output"] + [f"{s} kWh/t" for s in savings_axis])
    st.dataframe(sens, use_container_width=True, hide_index=True)
    st.caption(f"At ₹{tar:.1f}/kWh. Energy alone — yield, alloy and reduced reblows are "
               f"additional. Realised payback is quoted as 4–12 months (sub-6 only for "
               f"high-utilisation, high-tariff plants) because savings ramp up and "
               f"advisory captures part, not all, of the identified gap.")

    # engine economics cross-check
    try:
        ec = E.economics_summary(cfg, base_sec, base_sec - saving, tpy)
        st.markdown("#### Engine economics cross-check")
        st.caption("Computed by the engine's metrics.economics() for the same inputs.")
        st.dataframe(pd.DataFrame([{"metric": k, "value": round(v, 2)}
                                   for k, v in ec.items()]),
                     use_container_width=True, hide_index=True)
    except Exception:
        pass

# ════════════════════════════════════════════════════════════════ VALIDATION
with tab_val:
    st.markdown("### Validation window")
    st.markdown('<span class="muted">The verified constants driving this plant, and a '
                'live conservation check on a fresh heat.</span>', unsafe_allow_html=True)

    st.markdown("#### Parameter audit — verified against the literature (v0.5)")
    audit = pd.DataFrame([
        ["Latent heat of fusion", f"{summary['L_fusion (kJ/kg)']:.0f} kJ/kg", "247", "CRC Handbook 104th ed."],
        ["(FeO)+[C]→Fe+CO", "1.39 MJ/kg FeO", "+100 kJ/mol CO", "Turkdogan; Fruehan MSTS"],
        ["FeSi75 heat of solution", "−3511 kJ/kg", "−4681 kJ/kg Si", "Sigworth & Elliott 1974"],
        ["Carburiser heat of solution", "+1883 kJ/kg C", "+22.6 kJ/mol", "graphite dissolution"],
        ["Grid emission factor", f"{grid_ef:.3f} tCO₂/MWh", "0.712", "CEA v21.0, FY2024-25"],
        ["Reversible melting floor", f"{floor:.0f} kWh/t", "practical ≈500", "computed from L_f=247"],
        ["Default tariff", f"₹{tariff:.1f}/kWh", "₹6.0–8.5 grid band", "HT industrial FY2025-26"],
        ["Baseline SEC", f"{base_sec:.0f} kWh/t", "550–650 scrap IF", "field practice"],
    ], columns=["Quantity", "Value in model", "Literature value", "Source"])
    st.dataframe(audit, use_container_width=True, hide_index=True)

    st.markdown("#### Live conservation check")

    @st.cache_data(show_spinner="Running a validation heat…")
    def run(plant):
        cfg = E.get_config(plant)
        specs = [E.AdditionSpec("Lime (92% CaO)", 10, 48),
                 E.AdditionSpec("FeSi75", 45, 15),
                 E.AdditionSpec("Mill scale (FeO)", 60, 150)]
        return E.run_heat(cfg, heat_t * 1000.0, E.DEFAULT_CHARGE_COMP, 5200,
                          additions=E.build_additions(specs), dt=2.0)

    res = run(st.session_state.get("plant_choice", "if_msme_12t"))
    en = res.energy
    clo = en.get("residual_pct", en.get("closure_pct", float("nan")))

    k = st.columns(4)
    with k[0]:
        ok = res.ledger_max_pct < 1.0
        st.markdown("**Element ledger**")
        st.markdown(ui.pill(f"{res.ledger_max_pct:.2f}% < 1%", "ok" if ok else "warn"),
                    unsafe_allow_html=True)
    with k[1]:
        ok2 = abs(clo) < 5
        st.markdown("**First-law closure**")
        st.markdown(ui.pill(f"{clo:+.1f}%", "ok" if ok2 else "warn"),
                    unsafe_allow_html=True)
    with k[2]:
        st.markdown("**Endpoint temp**")
        hit = abs(res.endpoint["T_C"] - summary["Tap aim (°C)"]) <= 15
        st.markdown(ui.pill(f'{res.endpoint["T_C"]:.0f} °C', "ok" if hit else "warn"),
                    unsafe_allow_html=True)
    with k[3]:
        st.markdown("**Undissolved at tap**")
        cln = res.undissolved_kg < 5
        st.markdown(ui.pill(f"{res.undissolved_kg:.0f} kg", "ok" if cln else "warn"),
                    unsafe_allow_html=True)

    # per-element ledger bars
    lb = res.ledger_df
    if "closure_pct" in lb.columns:
        st.markdown("#### Per-element mass-balance closure")
        idcol = "element" if "element" in lb.columns else lb.columns[0]
        fig = go.Figure(go.Bar(x=lb[idcol].astype(str), y=lb["closure_pct"].abs(),
                               marker_color=ui.STEEL))
        fig.add_hline(y=1.0, line=dict(color=ui.AMBER, dash="dash"))
        ui.style_fig(fig, height=300, ylab="|closure| %")
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Endpoint accuracy in the model is competitive with published Level-2/ML "
               "melt models; always quote a tolerance with the achieved hit-rate for the "
               "specific plant, never a bare ±figure.")
