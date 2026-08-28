"""SmartMelt Studio — faithful Streamlit replica of run_gui.py.

Run:
    streamlit run streamlit_app.py

The browser screen intentionally preserves the native GUI's header, horizontal
tab order, compact dark layout, shared heat state and operator workflow.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "app" / "lib"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st


def _streamlit_version_tuple(value: str):
    out=[]
    for part in value.split(".")[:2]:
        digits="".join(ch for ch in part if ch.isdigit())
        out.append(int(digits or 0))
    return tuple((out+[0,0])[:2])


if _streamlit_version_tuple(st.__version__) < (1, 60):
    st.error("SmartMelt Smooth requires Streamlit 1.60 or newer. Run:  python -m pip install --upgrade -r requirements.txt")
    st.stop()

import engine as E
from app import exact_ui as U
from app import exact_tabs as T

st.set_page_config(page_title="SmartMelt Studio — melt optimisation (physics + ML)",
                   page_icon="🔥", layout="wide", initial_sidebar_state="collapsed")
U.inject_exact_css()
T.init_state()

# Header — same left-to-right order as gui/app.py.
h1,h2,h3,h4,h5 = st.columns([2.0,1.6,.45,1.35,2.6],gap="small")
h1.markdown('<div class="smartmelt-head-title">🔥 SmartMelt Studio</div>',unsafe_allow_html=True)
h2.markdown(f'<div class="smartmelt-head-meta">engine v{E.VERSION} · advisory-only</div>',unsafe_allow_html=True)
h3.markdown('<div class="smartmelt-head-meta" style="text-align:right">Plant:</div>',unsafe_allow_html=True)
h4.selectbox("Plant", list(E.available_configs()), key="sm_plant", label_visibility="collapsed", on_change=T.change_plant)
@st.fragment(run_every="800ms")
def _header_status_fragment():
    # Also collect a finished numerical worker when the operator has switched
    # to another tab; playback catches up from wall time when they return.
    T._poll_background()
    kind=st.session_state.sm_status_kind
    col={"ok":"#33d17a","warn":"#f0a83c","bad":"#e5484d"}.get(kind,"#33d17a")
    st.markdown(f'<div class="smartmelt-head-status" style="color:{col}">{st.session_state.sm_status}</div>',unsafe_allow_html=True)

with h5:
    _header_status_fragment()

names=["Operator Console","Process Trajectory","Physics & Energy","Virtual Sensor",
       "Machine Learning","Drift Monitor","Charge-Mix","Economics","Heat Log",
       "Settings","Validation","About / Details"]
renderers={
    "Operator Console":T.render_operator_console,
    "Process Trajectory":T.render_trajectory,
    "Physics & Energy":T.render_physics,
    "Virtual Sensor":T.render_ekf,
    "Machine Learning":T.render_ml,
    "Drift Monitor":T.render_drift,
    "Charge-Mix":T.render_charge_mix,
    "Economics":T.render_economics,
    "Heat Log":T.render_heat_log,
    "Settings":T.render_settings,
    "Validation":T.render_validation,
    "About / Details":T.render_about,
}

# Streamlit 1.60 tracked tabs preserve the native notebook appearance while
# allowing true lazy execution: only the visible tab runs. This removes the
# narrow 12-column button bar that clipped labels on smaller displays.
tabs=st.tabs(names,default="Operator Console",key="sm_native_tabs",on_change="rerun")
for tab,name in zip(tabs,names):
    if tab.open:
        with tab:
            renderers[name]()
        break
