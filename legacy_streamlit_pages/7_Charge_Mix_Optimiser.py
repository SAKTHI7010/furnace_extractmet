"""7 · Charge-Mix Optimiser — the least-cost scrap blend that still hits the aim
chemistry and the tramp (copper) ceiling, plus the shadow price of that ceiling:
what one extra 0.01 %Cu of headroom would save per tonne."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import engine as E          # noqa: E402
import ui                    # noqa: E402

st.set_page_config(page_title="Charge-Mix Optimiser · SmartMelt", page_icon="⚖️",
                   layout="wide")
ui.inject_css()

cfg = E.get_config(st.session_state.get("plant_choice", "if_msme_12t"))
summary = E.config_summary(cfg)
heat_t = summary["Heat size (t)"]

st.markdown("## ⚖️ Charge-Mix Optimiser")
st.markdown('<span class="muted">The cheapest compliant blend — and what your copper '
            'ceiling is costing you, priced to the rupee.</span>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Targets")
    target_t = st.slider("Target liquid (t)", 4.0, float(max(6, heat_t * 1.1)),
                         float(heat_t), 0.5)
    c_lo = st.slider("Min carbon (%)", 0.0, 0.5, 0.10, 0.05,
                     help="a carbon floor forces some high-carbon feed into the mix")
    c_hi = st.slider("Max carbon (%)", 0.1, 1.0, 0.40, 0.05)
    cu_limit = st.slider("Copper ceiling (%)", 0.10, 0.50, 0.15, 0.01,
                         help="tighten this to see the optimiser blend in cleaner scrap")
    st.caption("Edit material prices & assays in the table, then solve.")

st.markdown("#### Available materials")
mats_default = E.default_materials()
edit_df = pd.DataFrame(mats_default).rename(columns={
    "price": "price ₹/kg", "yield_": "yield", "energy": "kWh/kg",
    "Fe": "Fe frac", "Cu": "Cu frac", "C": "C frac"})
edited = st.data_editor(edit_df, use_container_width=True, num_rows="fixed",
                        key="mats", height=240)

if st.button("Solve least-cost mix", type="primary", use_container_width=True):
    mats = []
    for _, r in edited.iterrows():
        mats.append(dict(name=r["name"], price=float(r["price ₹/kg"]),
                         Fe=float(r["Fe frac"]), Cu=float(r["Cu frac"]),
                         C=float(r["C frac"]), yield_=float(r["yield"]),
                         energy=float(r["kWh/kg"])))
    # optimiser expects aim/tramp in PERCENT (it divides by 100 internally)
    aim = {"C": (c_lo, c_hi)}
    res, shadow, rows = E.solve_charge_mix(cfg, mats, target_t, aim, cu_limit)

    if not getattr(res, "feasible", False):
        st.error(f"No feasible blend: {getattr(res, 'message', 'infeasible')}", icon="⚠️")
        st.caption("Try widening the carbon window, raising the copper ceiling, or "
                   "adding a cleaner scrap source in the table.")
    else:
        cpt = getattr(res, "cost_INR_per_t_liquid", 0)
        kk = st.columns(3)
        with kk[0]:
            ui.kpi("Least-cost blend", f"₹{cpt:,.0f}/t", "of liquid steel")
        with kk[1]:
            ui.kpi("Charge energy", f"{getattr(res,'energy_kWh',0):,.0f} kWh",
                   f"{getattr(res,'liquid_t',target_t):.1f} t liquid")
        with kk[2]:
            bath = getattr(res, "predicted_bath_pct", {})
            ui.kpi("Predicted Cu", f"{bath.get('Cu',0):.3f} %",
                   f"ceiling {cu_limit:.2f} %")

        st.markdown("#### Cheapest compliant blend")
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        bath = getattr(res, "predicted_bath_pct", {})
        if bath:
            st.markdown("#### Predicted bath chemistry")
            bath_df = pd.DataFrame([{"element": k, "wt %": round(float(v), 4)}
                                    for k, v in bath.items() if float(v) > 1e-6])
            st.dataframe(bath_df, use_container_width=True, hide_index=True)

        st.markdown("#### Copper ceiling — shadow price")
        cu_sh = shadow.get("Cu")
        if cu_sh and abs(cu_sh) > 1:
            per_pt = abs(float(cu_sh)) / 100.0   # shadow is per 1%; show per 0.01%
            st.markdown(
                ui.pill(f"Your {cu_limit:.2f}% Cu ceiling costs ≈ ₹{per_pt:,.0f} per tonne "
                        f"of liquid for every 0.01% you refuse to relax", "warn"),
                unsafe_allow_html=True)
            st.caption(f"Relaxing the ceiling to {cu_limit+0.01:.2f}% would cut cost by "
                       f"≈ ₹{per_pt:,.0f}/t. Is the customer spec really {cu_limit:.2f}%, "
                       f"or is that a habit worth revisiting?")
        else:
            st.markdown(ui.pill("copper ceiling not binding — it costs you nothing here", "ok"),
                        unsafe_allow_html=True)
            st.caption("At the optimum the blend sits comfortably under the copper "
                       "ceiling, so relaxing it would save nothing. Tighten the ceiling "
                       "in the sidebar to see the optimiser pay for cleaner scrap.")

    st.caption("The optimiser is the engine's ChargeMixOptimiser (a linear program). "
               "Tramp elements like copper are not removed by oxidising steelmaking, so "
               "the ceiling is a hard purchasing constraint — its shadow price tells you "
               "exactly how much cleaner scrap is worth paying for.")
else:
    st.markdown("Set your aim chemistry and copper ceiling, edit prices if needed, then "
                "**Solve**.")
