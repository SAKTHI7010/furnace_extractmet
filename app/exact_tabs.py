"""Faithful Streamlit rendering of every tab in the native SmartMelt GUI.

The interface deliberately follows gui/app.py, gui/console_tab.py and gui/tabs.py:
same tab order, controls, shared heat, operator actions, plots and calculations.
No physics is duplicated; every computation calls app/lib/engine.py.
"""
from __future__ import annotations

import copy
import datetime as _dt
import io
import math
import time
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app import exact_ui as U
from app.background_jobs import start_simulation_job

# engine is placed on sys.path by streamlit_app.py
import engine as E

# palette copied from gui/theme.py
BG_DEEP = "#0a0d10"; BG_PANEL = "#12171b"; BG_RAISED = "#182027"; BG_INPUT = "#0e1317"
LINE = "#232c33"; TEXT = "#e9edf0"; TEXT_MUT = "#9aa4af"; TEXT_DIM = "#6b757f"
MOLTEN = "#ff6a34"; MOLTEN_HI = "#ffd166"; STEEL = "#4fa8d8"; GREEN = "#33d17a"
AMBER = "#f0a83c"; RED = "#e5484d"; SLAG_TOP = "#a08a5a"; SCRAP_COL = "#8792a0"


_POLL_LOCK = threading.Lock()

matplotlib.rcParams.update({
    "figure.facecolor": BG_PANEL, "axes.facecolor": "#0f1418",
    "axes.edgecolor": LINE, "axes.labelcolor": TEXT_MUT,
    "text.color": TEXT, "xtick.color": TEXT_MUT, "ytick.color": TEXT_MUT,
    "grid.color": "#20262c", "font.size": 8, "axes.titlesize": 9,
    "axes.titlecolor": TEXT, "figure.dpi": 110,
})


def _style_axes(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor("#0f1418")
    ax.grid(True, color="#20262c", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color(LINE)
    if title: ax.set_title(title, fontsize=9, color=TEXT, fontweight="bold")
    if xlabel: ax.set_xlabel(xlabel, fontsize=8, color=TEXT_MUT)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8, color=TEXT_MUT)
    ax.tick_params(labelsize=7)


def _legend(ax, **kw):
    return ax.legend(fontsize=7, labelcolor=TEXT, facecolor=BG_RAISED,
                     edgecolor=LINE, framealpha=.9, **kw)


def _plot(fig, height=None):
    st.pyplot(fig, use_container_width=True, clear_figure=False)
    plt.close(fig)


def cfg():
    return st.session_state.sm_cfg


def summary():
    return E.config_summary(cfg())


def set_status(text: str, kind: str = "ok"):
    st.session_state.sm_status = text
    st.session_state.sm_status_kind = kind


def log_event(event: str, detail: str = "", sim_min: float | None = None):
    st.session_state.sm_heat_log.append({
        "clock": _dt.datetime.now().strftime("%H:%M:%S"),
        "sim_min": f"{sim_min:.1f}" if sim_min is not None else "",
        "event": event,
        "detail": detail,
    })


def init_state():
    s = st.session_state
    configs = E.available_configs()
    default_plant = "if_msme_12t" if "if_msme_12t" in configs else next(iter(configs), "")
    if "sm_plant" not in s:
        s.sm_plant = default_plant
    if "sm_cfg" not in s:
        s.sm_cfg = E.get_config(s.sm_plant)
    defaults = {
        "sm_status": "ready", "sm_status_kind": "ok",
        "sm_heat_log": [],
        "sm_heat_spec": {
            "charge_t": 12.0, "power_kW": 5200.0,
            "charge_C_pct": 0.30, "charge_Cu_pct": 0.20,
            "schedule": [
                dict(material="Lime (92% CaO)", mass=48, time_min=8),
                dict(material="FeSi75", mass=15, time_min=42),
                dict(material="Carburiser", mass=12, time_min=48),
                dict(material="Mill scale (FeO)", mass=120, time_min=58),
            ],
        },
        "sm_spec_result": None, "sm_spec_key": None,
        "op_charge_t": 12.0, "op_power_kW": 5200.0,
        "op_C_pct": 0.30, "op_Cu_pct": 0.20,
        "op_frames": None, "op_states": None, "op_pools": None,
        "op_frame_i": 0, "op_running": False,
        "op_tapped": False, "op_complete": False, "op_speed": 10,
        "op_applied_adds": [], "op_add_log": [], "op_injected": [],
        "op_last_tick": time.time(), "op_play_anchor_wall": time.time(),
        "op_play_anchor_frame": 0, "op_furnace_prev": None, "op_end_text": "",
        "op_future": None, "op_pending": None,
        "traj_result": None, "physics_result": None,
        "ekf_result": None, "ml_result": None, "drift_result": None,
        "mix_result": None, "mix_shadow": {}, "mix_rows": [],
        "mix_manual_weights": {}, "economics_result": None,
        "validation_result": None,
    }
    for k, v in defaults.items():
        if k not in s:
            s[k] = copy.deepcopy(v)
    # Safe migration from earlier Streamlit builds that stored visual frames but
    # not exact state/pool checkpoints. A stale heat is cleared once rather than
    # failing on the first material addition.
    if s.op_frames is not None:
        bad_states = s.op_states is None or len(s.op_states) != len(s.op_frames)
        bad_pools = s.op_pools is None or len(s.op_pools) != len(s.op_frames)
        if bad_states or bad_pools:
            s.op_frames=None; s.op_states=None; s.op_pools=None
            s.op_frame_i=0; s.op_running=False; s.op_tapped=False; s.op_complete=False
            s.op_future=None; s.op_pending=None; s.op_furnace_prev=None
            set_status("ready — previous browser heat reset after performance upgrade","ok")


def change_plant():
    s = st.session_state
    s.sm_cfg = E.get_config(s.sm_plant)
    s.sm_spec_result = None; s.sm_spec_key = None
    s.traj_result = None; s.physics_result = None
    s.validation_result = None
    fut = getattr(s, "op_future", None)
    if fut is not None and not fut.done():
        fut.cancel()
    s.op_future = None; s.op_pending = None
    s.op_frames = None; s.op_states = None; s.op_pools = None
    s.op_frame_i = 0; s.op_running = False
    s.op_tapped = False; s.op_complete = False
    s.op_play_anchor_frame = 0; s.op_play_anchor_wall = time.time(); s.op_furnace_prev = None
    set_status(f"plant → {s.sm_plant}", "ok")


def _spec_key() -> tuple:
    h = st.session_state.sm_heat_spec
    return (st.session_state.sm_plant, h["charge_t"], h["power_kW"],
            h["charge_C_pct"], h["charge_Cu_pct"],
            tuple((a["material"], float(a["mass"]), round(float(a["time_min"]), 2))
                  for a in h["schedule"]))


def run_spec_heat(force=False):
    s = st.session_state
    key = _spec_key()
    if not force and s.sm_spec_result is not None and s.sm_spec_key == key:
        return s.sm_spec_result
    h = s.sm_heat_spec
    comp = dict(E.DEFAULT_CHARGE_COMP)
    comp["C"] = h["charge_C_pct"] / 100.0
    comp["Cu"] = h["charge_Cu_pct"] / 100.0
    specs = [E.AdditionSpec(a["material"], a["time_min"], a["mass"]) for a in h["schedule"]]
    res = E.run_heat(cfg(), h["charge_t"] * 1000.0, comp, h["power_kW"],
                     additions=E.build_additions(specs), dt=2.0)
    s.sm_spec_result = res; s.sm_spec_key = key
    return res


def _frames_df(frames: List[dict]) -> pd.DataFrame:
    return pd.DataFrame(frames) if frames else pd.DataFrame()


def _current_snap() -> dict | None:
    frames = st.session_state.op_frames
    if not frames:
        return None
    i = max(0, min(int(st.session_state.op_frame_i), len(frames)-1))
    return frames[i]


def _project_tap(frames: List[dict], i: int) -> float:
    s = frames[i]
    if i < 6 or s["melted_pct"] > 99:
        return float(s["T_bath_C"])
    recent = frames[max(0, i-5):i+1]
    dT = recent[-1]["T_bath_C"] - recent[0]["T_bath_C"]
    dt = max(recent[-1]["t_min"] - recent[0]["t_min"], .1)
    rate = dT / dt
    dm = recent[-1]["melted_pct"] - recent[0]["melted_pct"]
    if dm > .5:
        mins = (100 - s["melted_pct"]) / (dm/dt)
        return float(s["T_bath_C"] + rate * min(mins, 40))
    return float(s["T_bath_C"] + rate * 5)


def _blank_operator_plot():
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    _style_axes(ax, "Press START HEAT to begin", "Time (min)", "Temperature (°C)")
    ax.text(.5, .5, "Set charge, power and carbon, then START HEAT.\n"
                     "Add materials at any moment while it runs, then TAP HEAT.",
            ha="center", va="center", color=TEXT_MUT, fontsize=10, transform=ax.transAxes)
    return fig


def _operator_plot(frames: List[dict], i: int):
    d = pd.DataFrame(frames[:i+1])
    aim = getattr(cfg().plant, "tap_temperature_C", 1620)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.2, 4.3), sharex=True)
    fig.subplots_adjust(left=.10, right=.90, top=.93, bottom=.10, hspace=.38)
    _style_axes(a1, "Temperature & melt progress", ylabel="Temperature (°C)")
    a1.plot(d.t_min, d.T_bath_C, color=MOLTEN, lw=2, label="bath")
    a1.plot(d.t_min, d.T_solid_C, color=SCRAP_COL, lw=1.2, label="solid")
    a1.axhline(aim, color=GREEN, ls="--", lw=1, label=f"aim {aim:.0f}")
    a1b = a1.twinx(); a1b.plot(d.t_min, d.melted_pct, color=STEEL, lw=1.4)
    a1b.set_ylim(0,105); a1b.set_ylabel("Melted (%)", color=STEEL, fontsize=8)
    a1b.tick_params(axis="y", colors=STEEL, labelsize=7); _legend(a1, loc="center right")
    _style_axes(a2, "Bath chemistry", "Time (min)", "Content (wt %)")
    for el, c in [("pct_C",MOLTEN),("pct_Si",STEEL),("pct_Mn",GREEN),("pct_S",SLAG_TOP)]:
        if el in d: a2.plot(d.t_min, d[el], color=c, lw=1.3, label=el.replace("pct_",""))
    for a in st.session_state.op_applied_adds:
        a1.axvline(a["time_min"], color=AMBER, ls=":", lw=.8)
    _legend(a2, ncol=4)
    return fig


def _operator_trend_svg(frames: List[dict], i: int, width: int = 900, height: int = 340) -> str:
    """Crisp, low-latency SVG equivalent of the native two-axis live plot.

    Matplotlib PNG regeneration on every Streamlit rerun was the main source of
    sluggishness and visual softness. This renderer only emits SVG paths, so the
    browser repaints quickly and text/lines stay sharp at every zoom level.
    """
    if not frames:
        return f'''<div class="live-trend"><svg viewBox="0 0 {width} {height}" width="100%" height="{height}"
          xmlns="http://www.w3.org/2000/svg" style="background:#0f1418">
          <rect x="0" y="0" width="{width}" height="{height}" fill="#0f1418" stroke="#232c33"/>
          <text x="{width/2:.0f}" y="30" text-anchor="middle" fill="#e9edf0" font-size="13" font-weight="700"
                font-family="Segoe UI,DejaVu Sans,Arial">Press START HEAT to begin</text>
          <text x="{width/2:.0f}" y="{height/2:.0f}" text-anchor="middle" fill="#9aa4af" font-size="13"
                font-family="Segoe UI,DejaVu Sans,Arial">Set charge, power and carbon, then START HEAT.</text>
          <text x="{width/2:.0f}" y="{height/2+20:.0f}" text-anchor="middle" fill="#9aa4af" font-size="13"
                font-family="Segoe UI,DejaVu Sans,Arial">Add materials at any moment while it runs, then TAP HEAT.</text>
        </svg></div>'''

    d_full = frames[: max(1, min(i + 1, len(frames)))]
    # A 900 px chart cannot resolve thousands of two-second samples. Retain a
    # uniformly spaced set (including both endpoints) to keep the live payload
    # small and redraw latency low without changing the visible trajectory.
    if len(d_full) > 650:
        idx = np.linspace(0, len(d_full)-1, 650, dtype=int)
        d = [d_full[int(j)] for j in idx]
    else:
        d = d_full
    aim = float(getattr(cfg().plant, "tap_temperature_C", 1620))
    margin_l, margin_r, margin_t, margin_b = 62.0, 56.0, 25.0, 32.0
    gap = 36.0
    panel_h = (height - margin_t - margin_b - gap) / 2.0
    x0, x1 = margin_l, width - margin_r
    y1a, y1b = margin_t, margin_t + panel_h
    y2a, y2b = y1b + gap, y1b + gap + panel_h
    tvals = [float(r["t_min"]) for r in d]
    tmax = max(1.0, max(tvals))

    temp_vals = [float(r["T_bath_C"]) for r in d] + [float(r.get("T_solid_C", 30.0)) for r in d] + [aim]
    tlo = min(temp_vals)
    thi = max(temp_vals)
    pad = max(35.0, (thi - tlo) * 0.08)
    tlo = max(0.0, tlo - pad)
    thi = thi + pad
    if thi <= tlo: thi = tlo + 1.0

    chem_keys = [("pct_C", MOLTEN, "C"), ("pct_Si", STEEL, "Si"),
                 ("pct_Mn", GREEN, "Mn"), ("pct_S", SLAG_TOP, "S")]
    chem_max = max([float(r.get(k, 0.0)) for r in d for k,_,_ in chem_keys] + [0.1]) * 1.15

    def sx(t): return x0 + (float(t) / tmax) * (x1 - x0)
    def sy_temp(v): return y1b - (float(v) - tlo) / (thi - tlo) * (y1b - y1a)
    def sy_melt(v): return y1b - max(0.0, min(100.0, float(v))) / 100.0 * (y1b - y1a)
    def sy_chem(v): return y2b - max(0.0, float(v)) / chem_max * (y2b - y2a)
    def pts(key, sy): return " ".join(f"{sx(r['t_min']):.2f},{sy(r.get(key,0.0)):.2f}" for r in d)

    parts = [f'<div class="live-trend"><svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
             f'xmlns="http://www.w3.org/2000/svg" style="display:block;background:#0f1418">',
             f'<rect x="0" y="0" width="{width}" height="{height}" fill="#0f1418"/>']
    for ya,yb in ((y1a,y1b),(y2a,y2b)):
        parts.append(f'<rect x="{x0}" y="{ya}" width="{x1-x0}" height="{yb-ya}" fill="none" stroke="#232c33"/>')
        for j in range(5):
            yy=ya+j*(yb-ya)/4
            parts.append(f'<line x1="{x0}" y1="{yy:.2f}" x2="{x1}" y2="{yy:.2f}" stroke="#20262c" stroke-width="1"/>')
    for j in range(6):
        xx=x0+j*(x1-x0)/5
        parts.append(f'<line x1="{xx:.2f}" y1="{y1a}" x2="{xx:.2f}" y2="{y1b}" stroke="#20262c" stroke-width="1"/>')
        parts.append(f'<line x1="{xx:.2f}" y1="{y2a}" x2="{xx:.2f}" y2="{y2b}" stroke="#20262c" stroke-width="1"/>')
        parts.append(f'<text x="{xx:.2f}" y="{height-10}" text-anchor="middle" fill="#9aa4af" font-size="9" font-family="Segoe UI,DejaVu Sans,Arial">{tmax*j/5:.0f}</text>')
    parts += [
        f'<text x="{(x0+x1)/2:.2f}" y="14" text-anchor="middle" fill="#e9edf0" font-size="11" font-weight="700" font-family="Segoe UI,DejaVu Sans,Arial">Temperature &amp; melt progress</text>',
        f'<text x="{(x0+x1)/2:.2f}" y="{y2a-10:.2f}" text-anchor="middle" fill="#e9edf0" font-size="11" font-weight="700" font-family="Segoe UI,DejaVu Sans,Arial">Bath chemistry</text>',
        f'<text x="14" y="{(y1a+y1b)/2:.2f}" transform="rotate(-90 14 {(y1a+y1b)/2:.2f})" text-anchor="middle" fill="#9aa4af" font-size="9" font-family="Segoe UI,DejaVu Sans,Arial">Temperature (°C)</text>',
        f'<text x="14" y="{(y2a+y2b)/2:.2f}" transform="rotate(-90 14 {(y2a+y2b)/2:.2f})" text-anchor="middle" fill="#9aa4af" font-size="9" font-family="Segoe UI,DejaVu Sans,Arial">Content (wt %)</text>',
        f'<text x="{(x0+x1)/2:.2f}" y="{height-1}" text-anchor="middle" fill="#9aa4af" font-size="9" font-family="Segoe UI,DejaVu Sans,Arial">Time (min)</text>',
    ]
    for j in range(5):
        yy=y1b-j*(y1b-y1a)/4; val=tlo+j*(thi-tlo)/4
        parts.append(f'<text x="{x0-7}" y="{yy+3:.2f}" text-anchor="end" fill="#9aa4af" font-size="8" font-family="Segoe UI,DejaVu Sans,Arial">{val:.0f}</text>')
        yy2=y2b-j*(y2b-y2a)/4; val2=j*chem_max/4
        parts.append(f'<text x="{x0-7}" y="{yy2+3:.2f}" text-anchor="end" fill="#9aa4af" font-size="8" font-family="Segoe UI,DejaVu Sans,Arial">{val2:.2f}</text>')
    parts.append(f'<line x1="{x0}" y1="{sy_temp(aim):.2f}" x2="{x1}" y2="{sy_temp(aim):.2f}" stroke="{GREEN}" stroke-width="1" stroke-dasharray="6 4"/>')
    parts.append(f'<polyline points="{pts("T_bath_C",sy_temp)}" fill="none" stroke="{MOLTEN}" stroke-width="2" stroke-linejoin="round"/>')
    parts.append(f'<polyline points="{pts("T_solid_C",sy_temp)}" fill="none" stroke="{SCRAP_COL}" stroke-width="1.2" stroke-linejoin="round"/>')
    parts.append(f'<polyline points="{pts("melted_pct",sy_melt)}" fill="none" stroke="{STEEL}" stroke-width="1.4" stroke-linejoin="round"/>')
    for key,col,label in chem_keys:
        parts.append(f'<polyline points="{pts(key,sy_chem)}" fill="none" stroke="{col}" stroke-width="1.3" stroke-linejoin="round"/>')
    for a in st.session_state.op_applied_adds:
        xx=sx(a["time_min"])
        parts.append(f'<line x1="{xx:.2f}" y1="{y1a}" x2="{xx:.2f}" y2="{y2b}" stroke="{AMBER}" stroke-width="1" stroke-dasharray="2 3"/>')
    legends=[("bath",MOLTEN),("solid",SCRAP_COL),("melted %",STEEL),(f"aim {aim:.0f}",GREEN)]
    lx=x1-260
    for label,col in legends:
        parts.append(f'<line x1="{lx}" y1="{y1a+12}" x2="{lx+16}" y2="{y1a+12}" stroke="{col}" stroke-width="2"/>')
        parts.append(f'<text x="{lx+20}" y="{y1a+15}" fill="#e9edf0" font-size="8" font-family="Segoe UI,DejaVu Sans,Arial">{label}</text>')
        lx += 62
    lx=x1-160
    for _,col,label in chem_keys:
        parts.append(f'<line x1="{lx}" y1="{y2a+12}" x2="{lx+14}" y2="{y2a+12}" stroke="{col}" stroke-width="2"/>')
        parts.append(f'<text x="{lx+18}" y="{y2a+15}" fill="#e9edf0" font-size="8" font-family="Segoe UI,DejaVu Sans,Arial">{label}</text>')
        lx += 40
    parts.append('</svg></div>')
    return ''.join(parts)

def _kpi_grid(snap: dict | None, proj: float | None):
    aim = getattr(cfg().plant, "tap_temperature_C", 1620)
    power = st.session_state.op_power_kW
    if snap is None:
        values = [("Bath °C","—",""),("Carbon %","—",""),("Melted %","—",""),("SEC kWh/t","—",""),
                  ("Slag FeO %","—",""),("Basicity B2","—",""),("Silicon %","—",""),("Manganese %","—",""),
                  ("Power kW","—",""),("Total kWh","—",""),("Expected tap °C","—",f"aim {aim:.0f}"),("Actual bath °C","—","")]
    else:
        values = [
            ("Bath °C",f'{snap["T_bath_C"]:.0f}',f'aim {aim:.0f}'),
            ("Carbon %",f'{snap["pct_C"]:.3f}',""),
            ("Melted %",f'{snap["melted_pct"]:.0f}',f'{snap["M_liquid_t"]:.1f} t liq'),
            ("SEC kWh/t",f'{snap["SEC_kWh_t"]:.0f}',f'{snap["E_kWh"]:.0f} kWh'),
            ("Slag FeO %",f'{snap["slag_FeO_pct"]:.1f}',f'P {snap["pct_P"]:.4f}'),
            ("Basicity B2",f'{snap["B2"]:.2f}',"CaO/SiO2"),
            ("Silicon %",f'{snap["pct_Si"]:.3f}',""),
            ("Manganese %",f'{snap["pct_Mn"]:.3f}',f'S {snap["pct_S"]:.4f}'),
            ("Power kW",f'{power:.0f}',"off" if st.session_state.op_tapped else "grid"),
            ("Total kWh",f'{snap["E_kWh"]:.0f}',"cumulative"),
            ("Expected tap °C",f'{proj:.0f}',f'aim {aim:.0f}'),
            ("Actual bath °C",f'{snap["T_bath_C"]:.0f}',"measured"),
        ]
    for row in range(3):
        cols = st.columns(4)
        for j,c in enumerate(cols):
            with c:
                lab,val,sub = values[row*4+j]; U.kpi(lab,val,sub)


def _operator_status(snap):
    pending=getattr(st.session_state,"op_pending",None) or {}
    if pending.get("kind")=="inject" and getattr(st.session_state,"op_future",None) is not None:
        return f"addition accepted — updating from {pending.get('time_min',0):.1f} min", "warn"
    if snap is None:
        return "press START HEAT", "warn"
    if st.session_state.op_tapped:
        aim = getattr(cfg().plant, "tap_temperature_C", 1620)
        hit = abs(snap["T_bath_C"]-aim) <= 15
        return "TAPPED — " + ("on aim" if hit else f'{snap["T_bath_C"]-aim:+.0f}°C off aim'), "ok" if hit else "warn"
    if st.session_state.op_complete:
        return "heat complete — press TAP HEAT", "ok"
    aim = getattr(cfg().plant, "tap_temperature_C", 1620)
    if snap["melted_pct"] > 99 and snap["T_bath_C"] >= aim-5:
        return "READY TO TAP — on temperature & fully melted", "ok"
    if snap["melted_pct"] < 2:
        return "heating solid charge", "warn"
    return f'melting — {aim-snap["T_bath_C"]:.0f} °C below tap aim', "warn"


def _poll_background():
    """Commit a completed start/recompute job without blocking the browser."""
    with _POLL_LOCK:
        s=st.session_state
        fut=getattr(s,"op_future",None)
        if fut is None or not fut.done():
            return False
        pending=getattr(s,"op_pending",None) or {}
        s.op_future=None; s.op_pending=None
        try:
            bundle=fut.result()
        except Exception as exc:
            s.op_running=False if pending.get("kind")=="start" else s.op_running
            s.op_add_log.append(f"simulation failed: {exc}")
            set_status(f"simulation failed — {exc}","bad")
            return True
        frames=bundle["frames"]; states=bundle["states"]; pools=bundle["pools"]
        if pending.get("kind")=="start":
            s.op_frames=frames; s.op_states=states; s.op_pools=pools
            s.op_frame_i=0; s.op_running=True
            s.op_tapped=False; s.op_complete=False; s.op_speed=10
            s.op_play_anchor_wall=time.time(); s.op_play_anchor_frame=0
            s.op_furnace_prev=None
            set_status("heat running","ok")
        elif pending.get("kind")=="inject":
            # Keep every already-displayed frame and splice the newly calculated
            # continuation immediately after the exact injection checkpoint.
            cut=max(0,min(int(pending.get("cut_i",0)),len(s.op_frames)-1))
            current_i=int(s.op_frame_i)
            prefix_frames=list(s.op_frames[:cut+1])
            prefix_states=np.asarray(s.op_states[:cut+1]).copy()
            prefix_pools=list(s.op_pools[:cut+1])
            s.op_frames=prefix_frames+list(frames)
            s.op_states=np.concatenate([prefix_states,np.asarray(states)],axis=0) if len(states) else prefix_states
            s.op_pools=prefix_pools+list(pools)
            s.op_frame_i=min(current_i,len(s.op_frames)-1)
            s.op_play_anchor_frame=int(s.op_frame_i); s.op_play_anchor_wall=time.time()
            s.op_add_log.append(f"trajectory updated in {time.time()-pending.get('started_at',time.time()):.1f} s")
            set_status("heat running","ok")
        return True


def _start_heat():
    s = st.session_state
    if getattr(s,"op_future",None) is not None:
        return
    comp = dict(E.DEFAULT_CHARGE_COMP)
    comp["C"] = s.op_C_pct/100.0; comp["Cu"] = s.op_Cu_pct/100.0
    s.op_frames=None; s.op_states=None; s.op_pools=None
    s.op_frame_i=0; s.op_running=False
    s.op_tapped=False; s.op_complete=False; s.op_speed=10
    s.op_applied_adds=[]; s.op_injected=[]; s.op_add_log=[
        f"Heat started · {s.op_charge_t:.1f} t · {s.op_power_kW:.0f} kW"]
    s.op_last_tick=time.time(); s.op_play_anchor_wall=time.time(); s.op_play_anchor_frame=0
    s.op_furnace_prev=None; s.op_end_text=""
    captured=dict(charge_t=float(s.op_charge_t),power_kW=float(s.op_power_kW),
                  C_pct=float(s.op_C_pct),Cu_pct=float(s.op_Cu_pct))
    s.op_pending={"kind":"start","started_at":time.time(),**captured}
    s.op_future=start_simulation_job(cfg_obj=cfg(),charge_t=captured["charge_t"],
                                     comp=comp,power_kW=captured["power_kW"],additions=[])
    log_event("HEAT START",f"{captured['charge_t']:.1f} t, {captured['power_kW']:.0f} kW, C {captured['C_pct']:.2f}%",0.0)
    set_status("simulating heat","warn")


def _inject(material: str, mass: float):
    s = st.session_state
    if getattr(s,"op_future",None) is not None:
        s.op_add_log.append("wait for the current calculation to finish"); return
    if not s.op_running or s.op_tapped or not s.op_frames:
        s.op_add_log.append("start the heat first"); return
    info = E.ADDITION_LIBRARY.get(material)
    if info is None or mass <= 0: return
    cut_i=max(0,min(int(s.op_frame_i),len(s.op_frames)-1))
    cur = s.op_frames[cut_i]; tmin = float(cur["t_min"]); Tb = float(cur["T_bath_C"])
    add=E.make_addition_at(tmin*60.0,mass,info)
    s.op_applied_adds.append(dict(material=material,mass=mass,time_min=tmin))
    s.op_injected.append(add)
    s.op_add_log.append(f"{tmin:4.1f} min · +{mass:.0f} kg {material.split(' (')[0]} @ {Tb:.0f}°C")
    log_event("ADDITION",f"+{mass:.0f} kg {material} @ {Tb:.0f}°C",tmin)
    comp=dict(E.DEFAULT_CHARGE_COMP); comp["C"]=s.op_C_pct/100.0; comp["Cu"]=s.op_Cu_pct/100.0
    # Continue from the exact state and undissolved pool at this frame. Only the
    # new addition is pending; previous additions are already represented in the
    # checkpoint. This avoids a full minute-zero recalculation.
    from_state=np.asarray(s.op_states[cut_i],dtype=float).copy()
    from_pool=copy.deepcopy(s.op_pools[cut_i])
    s.op_pending={"kind":"inject","material":material,"mass":float(mass),
                  "time_min":tmin,"cut_i":cut_i,"started_at":time.time()}
    s.op_future=start_simulation_job(
        cfg_obj=cfg(),charge_t=float(s.op_charge_t),comp=comp,
        power_kW=float(s.op_power_kW),additions=[add],from_state=from_state,
        from_pool=from_pool,t0_s=tmin*60.0)
    set_status("addition accepted — updating future trajectory","warn")


def _tap_heat():
    s = st.session_state
    if not s.op_frames: return
    s.op_running=False; s.op_tapped=True
    s.op_play_anchor_frame = int(s.op_frame_i); s.op_play_anchor_wall = time.time()
    s.op_frames = s.op_frames[:s.op_frame_i+1]
    if s.op_states is not None: s.op_states = s.op_states[:s.op_frame_i+1]
    if s.op_pools is not None: s.op_pools = s.op_pools[:s.op_frame_i+1]
    snap=s.op_frames[-1]; aim=getattr(cfg().plant,"tap_temperature_C",1620)
    s.op_end_text=(f"Tapped at {snap['t_min']:.0f} min · {snap['T_bath_C']:.0f} °C · C {snap['pct_C']:.3f}% · "
                   f"SEC {snap['SEC_kWh_t']:.0f} kWh/t · slag FeO {snap['slag_FeO_pct']:.1f}% · B2 {snap['B2']:.2f}\n"
                   "Additions this heat: " + (", ".join(f"{a['mass']:.0f}kg {a['material'].split(' (')[0]}@{a['time_min']:.0f}min" for a in s.op_applied_adds) or "none"))
    s.sm_heat_spec={"charge_t":s.op_charge_t,"power_kW":s.op_power_kW,"charge_C_pct":s.op_C_pct,
                    "charge_Cu_pct":s.op_Cu_pct,"schedule":copy.deepcopy(s.op_applied_adds)}
    s.sm_spec_result=None; s.sm_spec_key=None
    log_event("TAP",f"{snap['T_bath_C']:.0f}°C, C {snap['pct_C']:.3f}%, SEC {snap['SEC_kWh_t']:.0f} kWh/t",snap["t_min"])
    set_status("heat tapped — trajectory frozen at final state","ok")


def _frames_per_tick(speed: int) -> int:
    return {0: 0, 1: 1, 10: 6, 60: 30}.get(int(speed), 6)


def _sync_playback(now: float | None = None) -> int:
    """Advance using the native GUI's 80 ms clock, independent of rerun speed."""
    s = st.session_state
    if not s.op_frames:
        s.op_frame_i = 0
        return 0
    if not s.op_running or s.op_tapped or s.op_speed == 0 or s.op_complete:
        return int(max(0, min(s.op_frame_i, len(s.op_frames)-1)))
    now = time.time() if now is None else float(now)
    anchor_wall = float(getattr(s, "op_play_anchor_wall", now))
    anchor_frame = int(getattr(s, "op_play_anchor_frame", s.op_frame_i))
    ticks = max(0, int((now - anchor_wall) / 0.080))
    i = min(anchor_frame + ticks * _frames_per_tick(s.op_speed), len(s.op_frames)-1)
    s.op_frame_i = i
    if i >= len(s.op_frames)-1:
        s.op_complete = True
    return i


def _set_play_speed(value: int):
    s = st.session_state
    _sync_playback()
    s.op_speed = int(value)
    s.op_play_anchor_frame = int(s.op_frame_i)
    s.op_play_anchor_wall = time.time()


def _render_operator_controls():
    s=st.session_state
    U.section_title("Heat setup (applied at Start)",small=True)
    s.op_charge_t=st.slider("Charge (t)",4.0,14.0,float(s.op_charge_t),.1,key="op_charge_widget")
    s.op_power_kW=st.slider("Power (kW)",1000,8000,int(s.op_power_kW),100,key="op_power_widget")
    s.op_C_pct=st.slider("Charge C (%)",.05,1.5,float(s.op_C_pct),.01,key="op_c_widget")
    s.op_Cu_pct=st.slider("Charge Cu (%)",.05,.5,float(s.op_Cu_pct),.01,key="op_cu_widget")
    pending=getattr(s,"op_future",None) is not None
    if pending:
        p=s.op_pending or {}; elapsed=max(0.0,time.time()-float(p.get("started_at",time.time())))
        label="Preparing heat trajectory" if p.get("kind")=="start" else f"Updating after {p.get('mass',0):.0f} kg {str(p.get('material','addition')).split(' (')[0]}"
        st.markdown(f'<div class="calc-banner"><span class="calc-dot"></span><b>{label}</b><span>{elapsed:.1f} s · live display remains active</span></div>',unsafe_allow_html=True)
    b1,b2=st.columns(2)
    with b1:
        start_label="▶ SIMULATING…" if pending and (s.op_pending or {}).get("kind")=="start" else "▶ START HEAT"
        if st.button(start_label,use_container_width=True,type="primary",key="start_heat",disabled=pending):
            _start_heat(); st.rerun(scope="fragment")
    with b2:
        if st.button("⏏ TAP HEAT",use_container_width=True,disabled=pending or not bool(s.op_frames) or s.op_tapped,key="tap_heat"):
            _sync_playback(); _tap_heat(); st.rerun(scope="fragment")
    speedcols=st.columns([.55,1,1,1,1])
    speedcols[0].markdown('<span class="thin-note">Speed:</span>',unsafe_allow_html=True)
    for col,(lab,val) in zip(speedcols[1:],[("⏸",0),("1×",1),("10×",10),("60×",60)]):
        with col:
            if st.button(lab,use_container_width=True,key=f"speed_{val}",type="primary" if s.op_speed==val else "secondary"):
                _set_play_speed(val); st.rerun(scope="fragment")
    U.section_title("Add material NOW (during heat)",small=True)
    cmat,cmass=st.columns([2.5,1])
    mat=cmat.selectbox("Material",list(E.ADDITION_LIBRARY),label_visibility="collapsed",key="op_mat")
    mass=cmass.number_input("kg",min_value=0.0,value=48.0,step=1.0,label_visibility="collapsed",key="op_mass")
    if st.button("＋ Add to bath now",use_container_width=True,disabled=pending or not s.op_running or s.op_tapped,key="op_add"):
        _sync_playback(); _inject(mat,mass); st.rerun(scope="fragment")
    q=st.columns(4)
    for col,(label,material,kg) in zip(q,[("Lime","Lime (92% CaO)",48),("FeSi75","FeSi75",15),("Carburiser","Carburiser",12),("Mill scale","Mill scale (FeO)",120)]):
        with col:
            if st.button(label,use_container_width=True,key=f"quick_{material}",disabled=pending):
                _sync_playback(); _inject(material,kg); st.rerun(scope="fragment")
    logtxt="\n".join(s.op_add_log[-5:])
    st.markdown(f'<div class="logbox">{logtxt or "&nbsp;"}</div>',unsafe_allow_html=True)


def _render_furnace_live():
    s=st.session_state; snap=_current_snap(); aim=getattr(cfg().plant,"tap_temperature_C",1620)
    if snap is None:
        cur=(0.0,30.0,0.0,0.0)
    else:
        cur=(float(snap["melted_pct"]),float(snap["T_bath_C"]),
             float(snap.get("slag_total_kg",0.0)),float(snap["undissolved_kg"]))
    pending=getattr(s,"op_pending",None) or {}
    if pending.get("kind")=="inject" and getattr(s,"op_future",None) is not None:
        cur=(cur[0],cur[1],cur[2],cur[3]+float(pending.get("mass",0.0)))
    prev=s.op_furnace_prev
    st.markdown(U.furnace_svg(*cur,s.op_charge_t,aim,height=240,previous=prev,animate_ms=360),unsafe_allow_html=True)
    s.op_furnace_prev=cur


def _render_operator_live_panel():
    s=st.session_state; snap=_current_snap(); i=int(s.op_frame_i); frames=s.op_frames or []
    if snap is None: clock="00:00"; proj=None
    else:
        clock=f'{int(snap["t_min"]):02d}:{int((snap["t_min"]%1)*60):02d}'
        proj=_project_tap(frames,i)
    hc1,hc2=st.columns([.16,.84])
    hc1.markdown(f'<div style="font:700 24px Consolas;color:{MOLTEN_HI};padding-top:2px">{clock}</div>',unsafe_allow_html=True)
    txt,kind=_operator_status(snap); hc2.markdown(U.pill(txt,kind),unsafe_allow_html=True)
    _kpi_grid(snap,proj)
    U.section_title("Advisory — live verdicts",small=True)
    adv=[("warn","—",""),]*6 if snap is None else E.build_advisories(snap,cfg(),projected_tap_C=proj)
    for r in range(2):
        cols=st.columns(3)
        for j,col in enumerate(cols):
            with col:
                level,title,msg=adv[r*3+j]; U.advisory_card(level,title,msg)
    st.markdown(_operator_trend_svg(frames,i),unsafe_allow_html=True)
    if s.op_end_text:
        st.markdown(f'<div class="thin-note" style="white-space:pre-wrap">{s.op_end_text}</div>',unsafe_allow_html=True)


@st.fragment(run_every="800ms")
def _operator_controls_fragment():
    _poll_background()
    _render_operator_controls()


@st.fragment(run_every="400ms")
def _operator_furnace_fragment():
    _poll_background()
    _sync_playback()
    _render_furnace_live()


@st.fragment(run_every="400ms")
def _operator_live_fragment():
    _poll_background()
    _sync_playback()
    _render_operator_live_panel()


def render_operator_console():
    """Stable two-column shell; only its three small fragments refresh."""
    left,right=st.columns([0.29,0.71],gap="small")
    with left:
        _operator_controls_fragment()
        U.section_title("Furnace state",small=True)
        _operator_furnace_fragment()
    with right:
        _operator_live_fragment()

# ────────────────────────────────────────────────────────────────────────────
# Process Trajectory
# ────────────────────────────────────────────────────────────────────────────
def _trajectory_figure_df(d: pd.DataFrame, additions: List[dict]):
    aim=getattr(cfg().plant,"tap_temperature_C",1620); floor=E.theoretical_floor_kWh_t(cfg()); t=d["t_min"]
    fig,axes=plt.subplots(2,3,figsize=(12,6.2)); (a1,a2,a3),(a4,a5,a6)=axes
    for a in axes.flat:_style_axes(a,xlabel="Time (min)")
    _style_axes(a1,"Temperatures","Time (min)","Temperature (°C)")
    a1.plot(t,d["T_bath_C"],color=MOLTEN,label="bath"); a1.plot(t,d["T_solid_C"],color=SCRAP_COL,label="solid charge")
    if "T_hotface_C" in d:a1.plot(t,d["T_hotface_C"],color=SLAG_TOP,ls=":",label="lining hot face")
    a1.axhline(aim,color=GREEN,ls="--",lw=1,label=f"tap aim {aim:.0f}"); _legend(a1)
    _style_axes(a2,"Inventories & dissolution","Time (min)","Metal mass (t)")
    a2.plot(t,d["M_solid_t"],color=SCRAP_COL,label="solid"); a2.plot(t,d["M_liquid_t"],color=MOLTEN,label="liquid")
    a2b=a2.twinx(); a2b.plot(t,d["undissolved_kg"],color=STEEL,lw=1); a2b.set_ylabel("Undissolved additions (kg)",color=STEEL,fontsize=8); a2b.tick_params(axis="y",colors=STEEL); _legend(a2)
    _style_axes(a3,"Bath composition","Time (min)","Element content (wt %)")
    for el,c in [("C",MOLTEN),("Si",STEEL),("Mn",GREEN),("S",SLAG_TOP)]:
        if f"pct_{el}" in d:a3.plot(t,d[f"pct_{el}"],color=c,label=el)
    _legend(a3,ncol=2)
    _style_axes(a4,"Slag chemistry & basicity","Time (min)","Slag FeO (wt %)")
    a4.plot(t,d["slag_FeO_pct"],color=MOLTEN,label="FeO"); a4b=a4.twinx(); a4b.plot(t,d["B2"],color=STEEL)
    a4b.set_ylabel("Basicity B2 (CaO/SiO₂)",color=STEEL,fontsize=8); a4b.tick_params(axis="y",colors=STEEL); _legend(a4)
    _style_axes(a5,"Heat-flow breakdown","Time (min)","Heat flow (kW)")
    for key,c,nm in [("Q_wall_kW",SLAG_TOP,"lining loss"),("Q_rad_kW",RED,"radiation"),("Q_bath_to_scrap_kW",SCRAP_COL,"bath→scrap"),("Q_chem_kW",GREEN,"chemical")]:
        if key in d:a5.plot(t,d[key],color=c,label=nm)
    _legend(a5)
    _style_axes(a6,"Energy & specific consumption","Time (min)","Cumulative energy (kWh)")
    a6.plot(t,d["E_kWh"],color=SCRAP_COL,label="cumulative kWh"); a6b=a6.twinx(); a6b.plot(t,d["SEC_kWh_t"],color=MOLTEN)
    a6b.axhline(floor,color=GREEN,ls="--",lw=1); a6b.set_ylabel("Specific energy (kWh/t)",color=MOLTEN,fontsize=8); a6b.tick_params(axis="y",colors=MOLTEN)
    for add in additions:
        for a in (a1,a2,a3):a.axvline(add["time_min"],color=AMBER,ls=":",lw=.7)
    fig.tight_layout(); return fig


def _trajectory_figure_snaps(frames: List[dict]):
    d=pd.DataFrame(frames); return _trajectory_figure_df(d,st.session_state.sm_heat_spec["schedule"])


def render_trajectory():
    s=st.session_state
    live=s.op_frames if (s.op_running or s.op_tapped) else None
    h=s.sm_heat_spec; adds=", ".join(f"{a['mass']:.0f}kg {a['material'].split(' (')[0]}@{a['time_min']:.0f}m" for a in h["schedule"]) or "no additions"
    c1,c2=st.columns([.84,.16]);
    if live:
        last=live[min(s.op_frame_i,len(live)-1)]; label=("● LIVE" if s.op_running else "■ TAPPED")+f" heat — {last['t_min']:.0f} min ({min(s.op_frame_i+1,len(live))} samples)"
    else:label=f"Operator's heat: {h['charge_t']:.1f} t · {h['power_kW']:.0f} kW · C {h['charge_C_pct']:.2f}% · additions: {adds}"
    c1.markdown(f'<span style="color:{STEEL};font-size:12px">{label}</span>',unsafe_allow_html=True)
    if c2.button("↻ Use operator's heat",use_container_width=True,key="traj_run"):
        with st.spinner("Running operator's heat…"):s.traj_result=run_spec_heat(force=True)
    if live:
        d=pd.DataFrame(live[:s.op_frame_i+1]); endpoint=d.iloc[-1]; ledger="live"; tapmin=endpoint.t_min
    else:
        if s.traj_result is None:
            with st.spinner("Running operator's heat…"):s.traj_result=run_spec_heat()
        r=s.traj_result; d=r.df; endpoint=pd.Series({"T_bath_C":r.endpoint["T_C"],"pct_C":r.endpoint["pct_C"]}); ledger=f"{r.ledger_max_pct:.2f}"; tapmin=r.tap_min
    aim=getattr(cfg().plant,"tap_temperature_C",1620); floor=E.theoretical_floor_kWh_t(cfg())
    cols=st.columns(5); vals=[("Tap °C",f'{endpoint["T_bath_C"]:.0f}',f"aim {aim}"),("Carbon %",f'{endpoint["pct_C"]:.3f}',""),("Tap min",f'{tapmin:.0f}',"live" if live else ""),("SEC kWh/t",f'{d["SEC_kWh_t"].iloc[-1]:.0f}',f"floor {floor:.0f}"),("Ledger %",ledger,"running" if live else "closure")]
    for col,(lab,val,sub) in zip(cols,vals):
        with col:U.kpi(lab,val,sub)
    _plot(_trajectory_figure_df(d,h["schedule"]))


# ────────────────────────────────────────────────────────────────────────────
# Physics & Energy
# ────────────────────────────────────────────────────────────────────────────
def _physics_figure(res):
    d=res.df; en=res.energy
    fig,((a1,a2),(a3,a4))=plt.subplots(2,2,figsize=(12,6.0))
    for a in (a1,a2,a3,a4):_style_axes(a)
    _style_axes(a1,"Heat-flow breakdown through the heat","Time (min)","Heat flow (kW)")
    for key,c,nm in [("Q_useful_kW",MOLTEN,"useful (to metal)"),("Q_wall_kW",SLAG_TOP,"lining loss"),("Q_rad_kW",RED,"radiation"),("Q_chem_kW",GREEN,"chemical"),("Q_offgas_kW",STEEL,"off-gas")]:
        if key in d:a1.plot(d.t_min,d[key],color=c,lw=1.4,label=nm)
    _legend(a1)
    _style_axes(a2,"Energy split — grid input to tapped steel",ylabel="Energy (kWh)")
    total=en.get("grid_kWh",d.E_kWh.iloc[-1]); parts=[("converter",en.get("converter_loss_kWh",0)),("coil water",en.get("coil_water_loss_kWh",0)),("lining",en.get("lining_loss_kWh",0)),("radiation",en.get("radiation_loss_kWh",0)),("off-gas",en.get("offgas_loss_kWh",0))]; useful=en.get("useful_melt_kWh",0)
    labels=["grid in"]+[p[0] for p in parts]+["to steel"]; vals=[total]+[-p[1] for p in parts]+[useful]; cum=0; bottoms=[]; heights=[]; colours=[]
    for i,v in enumerate(vals):
        if i==0:bottoms.append(0);heights.append(v);colours.append(MOLTEN);cum=v
        elif i==len(vals)-1:bottoms.append(0);heights.append(v);colours.append(STEEL)
        else:bottoms.append(cum+v);heights.append(-v);colours.append(RED);cum+=v
    a2.bar(range(len(vals)),heights,bottom=bottoms,color=colours,width=.6); a2.set_xticks(range(len(labels)));a2.set_xticklabels(labels,rotation=35,ha="right",fontsize=7)
    _style_axes(a3,"Element reaction rates","Time (min)","Rate (wt %/min)")
    plotted=False
    for key,c,nm in [("rate_C",MOLTEN,"C"),("rate_Si",STEEL,"Si"),("rate_Mn",GREEN,"Mn"),("rate_P",SLAG_TOP,"P")]:
        if key in d:a3.plot(d.t_min,d[key],color=c,lw=1.3,label=nm);plotted=True
    if not plotted:
        for el,c in [("C",MOLTEN),("Si",STEEL),("Mn",GREEN)]:
            if f"pct_{el}" in d:a3.plot(d.t_min,np.gradient(d[f"pct_{el}"],d.t_min),color=c,lw=1.3,label=f"d{el}/dt")
    a3.axhline(0,color=TEXT_DIM,lw=.6);_legend(a3,ncol=2)
    _style_axes(a4,"Cumulative energy: input vs useful","Time (min)","Cumulative energy (kWh)")
    a4.plot(d.t_min,d.E_kWh,color=MOLTEN,lw=1.6,label="grid input")
    if "Q_useful_kW" in d:
        dt_h=np.gradient(d.t_min)/60.;cu=np.cumsum(np.clip(d.Q_useful_kW.to_numpy(),0,None)*dt_h);a4.plot(d.t_min,cu,color=GREEN,lw=1.6,label="useful (to metal)");a4.fill_between(d.t_min,cu,d.E_kWh,color=RED,alpha=.12,label="losses")
    _legend(a4);fig.tight_layout();return fig


def render_physics():
    s=st.session_state; h=s.sm_heat_spec; adds=", ".join(f"{a['mass']:.0f}kg {a['material'].split(' (')[0]}@{a['time_min']:.0f}m" for a in h["schedule"]) or "no additions"
    c1,c2=st.columns([.84,.16]);c1.markdown(f'<span style="color:{STEEL};font-size:12px">Operator\'s heat: {h["charge_t"]:.1f} t · {h["power_kW"]:.0f} kW · additions: {adds}</span>',unsafe_allow_html=True)
    if c2.button("↻ Use operator's heat",use_container_width=True,key="phys_run"):
        with st.spinner("Running operator's heat…"):s.physics_result=run_spec_heat(force=True)
    if s.physics_result is None:
        with st.spinner("Running energy audit…"):s.physics_result=run_spec_heat()
    r=s.physics_result;en=r.energy;floor=E.theoretical_floor_kWh_t(cfg());uf=100*en.get("useful_fraction",0)
    cols=st.columns(4);vals=[("Element ledger %",f"{r.ledger_max_pct:.2f}","worst species"),("First-law closure %",f'{en.get("residual_pct",float("nan")):+.1f}',"in − out"),("Final SEC",f'{r.df.SEC_kWh_t.iloc[-1]:.0f}',f"floor {floor:.0f}"),("Useful fraction %",f"{uf:.0f}","of grid input")]
    for col,(lab,val,sub) in zip(cols,vals):
        with col:U.kpi(lab,val,sub)
    _plot(_physics_figure(r))


# ────────────────────────────────────────────────────────────────────────────
# EKF virtual sensor
# ────────────────────────────────────────────────────────────────────────────
def _ekf_figure(ek):
    d=ek.df;fig,(a1,a2)=plt.subplots(1,2,figsize=(12,5),gridspec_kw={"width_ratios":[1.5,1]});_style_axes(a1,"Bath temperature — truth vs EKF estimate","Time (min)","Temperature (°C)");_style_axes(a2,"Tracked parameters converging to truth","Time (min)","Parameter value")
    a1.fill_between(d.t_min,d.T_est_C-2*d.sigma_T,d.T_est_C+2*d.sigma_T,color=MOLTEN,alpha=.18,label="±2σ confidence");a1.plot(d.t_min,d.T_true_C,color="#cfd6dd",lw=2,label="true (hidden)");a1.plot(d.t_min,d.T_est_C,color=MOLTEN,lw=2,label="EKF estimate")
    if len(ek.dip_df):a1.scatter(ek.dip_df.t_min,ek.dip_df.T_meas_C,color=STEEL,s=55,marker="D",zorder=5,label="immersion dip")
    _legend(a1)
    a2.plot(ek.theta_path.t_min,ek.theta_path.eta_electrical,color=MOLTEN,label="η electrical")
    if "UA_lining_scale" in ek.theta_path:a2.plot(ek.theta_path.t_min,ek.theta_path.UA_lining_scale,color=STEEL,label="UA wall-loss scale")
    _legend(a2);fig.tight_layout();return fig


def render_ekf():
    s=st.session_state
    c=st.columns([1,1,1,.8]);eta=c[0].slider("True η electrical",.8,1.0,.90,.01,key="ekf_eta");ua=c[1].slider("True wall-loss scale",.8,1.8,1.35,.05,key="ekf_ua");nd=int(c[2].slider("Immersion dips",1,6,3,1,key="ekf_dips"))
    if c[3].button("Run live (~1 min)",use_container_width=True,key="ekf_run"):
        with st.spinner("Running EKF live…"):
            s.ekf_result=E.run_ekf_demo(cfg(),true_eta=eta,true_UA_scale=ua,dip_times_min=tuple(np.linspace(30,78,nd)),seed=1)
    st.markdown('<div class="thin-note">Default result is pre-computed and loads instantly. A live run recomputes the Kalman filter (finite-difference Jacobians over 34 states ≈ 1 min).</div>',unsafe_allow_html=True)
    if s.ekf_result is None:
        try:s.ekf_result=E.load_default_ekf()
        except Exception:s.ekf_result=None
    if s.ekf_result is None:
        st.warning("Packaged EKF cache could not be loaded. Use Run live.");return
    ek=s.ekf_result;eta_final=ek.theta_path.eta_electrical.iloc[-1]
    cols=st.columns(4);vals=[("Final error °C",f'{ek.final_error_C:+.1f}',"est − truth"),("η̂ electrical",f'{eta_final:.3f}',"converged"),("σ_T end °C",f'{ek.df.sigma_T.iloc[-1]:.1f}',"uncertainty"),("Dips used",str(len(ek.dip_df)),"measurements")]
    for col,(lab,val,sub) in zip(cols,vals):
        with col:U.kpi(lab,val,sub)
    _plot(_ekf_figure(ek))


# ────────────────────────────────────────────────────────────────────────────
# Machine learning and drift
# ────────────────────────────────────────────────────────────────────────────
def _load_dataset_safe():
    try:
        d=E.load_cached_dataset()
        if d is not None:return d
    except Exception:pass
    p=Path(__file__).resolve().parents[1]/"examples"/"heats_if_90.csv"
    return pd.read_csv(p) if p.exists() else None


def _ml_figure(ml):
    p=ml.pred_df;fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4.8));_style_axes(a1,"Temperature — predicted vs actual","actual °C","predicted °C");_style_axes(a2,"Test-set temperature error","test heat","pred − actual °C")
    lo=float(np.nanmin([p.T_true_C.min(),p.T_pred_C.min()]))-10;hi=float(np.nanmax([p.T_true_C.max(),p.T_pred_C.max()]))+10;a1.plot([lo,hi],[lo,hi],color=TEXT_MUT,ls="--");a1.scatter(p.T_true_C,p.T_phys_C,color=SCRAP_COL,marker="x",s=40,label="physics");a1.scatter(p.T_true_C,p.T_pred_C,color=MOLTEN,s=45,label="hybrid");_legend(a1)
    a2.bar(p.heat-.2,p.T_pred_C-p.T_true_C,width=.4,color=MOLTEN,label="hybrid");a2.bar(p.heat+.2,p.T_phys_C-p.T_true_C,width=.4,color=SCRAP_COL,label="physics");a2.axhspan(-15,15,color=GREEN,alpha=.1);_legend(a2);fig.tight_layout();return fig


def render_ml():
    s=st.session_state;c=st.columns([1,.8,1,.9]);split=c[0].slider("Train fraction",.5,.85,.70,.01,key="ml_split")
    train=c[1].button("Train on cached data",use_container_width=True,key="ml_cached");n=int(c[2].slider("Live heats",20,80,40,1,key="ml_n"));live=c[3].button("Generate live (slow)",use_container_width=True,key="ml_live")
    st.markdown('<div class="thin-note">The hybrid model = the SAME physics engine used on the Operator Console, plus a Gaussian-process residual head. Physics predicts, ML corrects, and gates itself off until it proves out-of-time improvement.</div>',unsafe_allow_html=True)
    if train or (s.ml_result is None and not live):
        d=_load_dataset_safe()
        if d is not None:
            with st.spinner("Training on cached data…"):s.ml_result=E.train_hybrid(cfg(),d,split_frac=split)
    if live:
        with st.spinner("Generating virtual heats and fitting ML…"):
            d=E.generate_dataset(cfg(),n_heats=n,seed=0);s.ml_result=E.train_hybrid(cfg(),d,split_frac=split)
    if s.ml_result is None:st.warning("No cached dataset found — use Generate live.");return
    ml=s.ml_result;m=ml.metrics;kind="ok" if m["ml_T_active"] else "warn";st.markdown(U.pill(f"maturity: {m['maturity']} · T-ML {'active' if m['ml_T_active'] else 'gated off'} · C-ML {'active' if m['ml_C_active'] else 'gated off'} ({m['n_train']} train / {m['n_test']} test)",kind),unsafe_allow_html=True)
    fmt=lambda x:f"{x:.0f}" if x==x else "—";cols=st.columns(4);vals=[("T hit ±15°C",fmt(m["T_hit_15C"])+"%",f"phys {fmt(m['T_hit_15C_phys'])}%"),("T MAE °C",f'{m["T_MAE_C"]:.1f}' if m["T_MAE_C"]==m["T_MAE_C"] else "—","hybrid"),("C hit ±0.02%",fmt(m["C_hit_002"])+"%",f"phys {fmt(m['C_hit_002_phys'])}%"),("C MAE %",f'{m["C_MAE"]:.3f}' if m["C_MAE"]==m["C_MAE"] else "—","hybrid")]
    for col,(lab,val,sub) in zip(cols,vals):
        with col:U.kpi(lab,val,sub)
    _plot(_ml_figure(ml))


def _drift_figure(df,dr,reg):
    fig,(a1,a2)=plt.subplots(1,2,figsize=(12,5),gridspec_kw={"width_ratios":[1.2,1]});_style_axes(a1,"Population drift by feature (PSI)");_style_axes(a2,"The variable that moved","heat number")
    psi=dr["psi_df"].head(12);cols=[RED if v>.5 else AMBER if v>.25 else STEEL for v in psi.PSI];a1.barh(psi.feature,psi.PSI,color=cols);a1.axvline(.25,color=AMBER,ls="--",lw=1);a1.axvline(.5,color=RED,ls="--",lw=1);a1.invert_yaxis();a1.tick_params(labelsize=7)
    cu="charge_Cu_pct" if "charge_Cu_pct" in df else df.columns[0];a2.plot(df.index,df[cu],color=MOLTEN,marker="o",ms=3);a2.axvline(reg,color=RED,ls=":",lw=1);a2.axvspan(0,dr["n_ref"],color=STEEL,alpha=.08);a2.set_ylabel(cu,fontsize=8);fig.tight_layout();return fig


def render_drift():
    s=st.session_state;c=st.columns([.8,1,1,.9]);cached=c[0].button("Check cached data",use_container_width=True,key="drift_cached");n=int(c[1].slider("Live heats",30,80,50,1,key="drift_n"));reg=int(c[2].slider("Regime change at heat",15,60,40,1,key="drift_reg"));live=c[3].button("Generate live (slow)",use_container_width=True,key="drift_live")
    st.markdown('<div class="thin-note">A pre-computed dataset checks instantly. Live generation runs the same physics simulator and introduces a copper regime change at the selected heat.</div>',unsafe_allow_html=True)
    if cached or (s.drift_result is None and not live):
        d=_load_dataset_safe()
        if d is not None:s.drift_result=(d,E.run_drift(cfg(),d,ref_frac=.5),40)
    if live:
        with st.spinner("Simulating heats and checking drift…"):
            d=E.generate_dataset(cfg(),n_heats=n,seed=0,regime_change_at=reg);s.drift_result=(d,E.run_drift(cfg(),d,ref_frac=.5),reg)
    if s.drift_result is None:st.warning("No cached dataset found — use Generate live.");return
    d,dr,reg_used=s.drift_result;st.markdown(U.pill("DRIFT ALARM — "+", ".join(dr["reasons"][:2]) if dr["alarm"] else "stable — no significant drift","bad" if dr["alarm"] else "ok"),unsafe_allow_html=True)
    cols=st.columns(3);vals=[("Max PSI",f'{dr["psi_max"]:.2f}',">0.25 shift · >0.5 major"),("Reference heats",str(dr["n_ref"]),"baseline"),("Recent heats",str(dr["n_recent"]),"checked")]
    for col,(lab,val,sub) in zip(cols,vals):
        with col:U.kpi(lab,val,sub)
    _plot(_drift_figure(d,dr,reg_used))

# ────────────────────────────────────────────────────────────────────────────
# Charge-Mix
# ────────────────────────────────────────────────────────────────────────────
def _material_df() -> pd.DataFrame:
    rows=[]
    for mm in E.default_materials():
        rows.append({"Material":mm["name"],"₹/kg":mm["price"],"Fe%":100*mm["Fe"],
                     "Cu%":100*mm["Cu"],"Sn%":100*mm.get("Sn",0),"C%":100*mm.get("C",0),
                     "kg (manual)":float(st.session_state.mix_manual_weights.get(mm["name"],0))})
    return pd.DataFrame(rows)


def _bath_df(bath: Dict[str,float]) -> pd.DataFrame:
    order=["C","Si","Mn","Cr","Cu","Sn","Fe"]
    return pd.DataFrame([{"Element":el,"wt %":bath[el]} for el in order if el in bath and bath[el]>1e-6])


def render_charge_mix():
    s=st.session_state;mats=E.default_materials()
    mode=st.radio("Mode:",["Optimise (least cost)","Manual (operator sets kg)"],horizontal=True,key="mix_mode")
    top=st.columns([1,1,1,1,1,.65]);target=top[0].slider("Target liquid (t)",4.0,14.0,12.0,.1,key="mix_target");clo=top[1].slider("Min C (%)",0.0,.5,.10,.01,key="mix_clo");chi=top[2].slider("Max C (%)",.1,1.0,.40,.01,key="mix_chi");cu=top[3].slider("Cu ceiling (%)",.08,.5,.20,.01,key="mix_cu");sn=top[4].slider("Sn ceiling (%)",.01,.10,.03,.001,key="mix_sn")
    solve=top[5].button("Solve" if mode.startswith("Optimise") else "Evaluate",use_container_width=True,key="mix_solve")
    left,right=st.columns([1.1,.9])
    with left:
        U.section_title("Scrap library — 17 streams (price ₹/kg · assays wt%)",small=True)
        mdf=_material_df()
        if mode.startswith("Manual"):
            edited=st.data_editor(mdf,use_container_width=True,height=445,hide_index=True,
                                  disabled=["Material","₹/kg","Fe%","Cu%","Sn%","C%"],key="mix_editor")
            s.mix_manual_weights={r["Material"]:float(r["kg (manual)"]) for _,r in edited.iterrows() if float(r["kg (manual)"])>0}
            st.markdown('<div class="thin-note">Manual mode: enter kg in the last column, then Evaluate blend.</div>',unsafe_allow_html=True)
        else:
            st.dataframe(mdf.drop(columns=["kg (manual)"]).round(4),use_container_width=True,height=445,hide_index=True)
            st.markdown('<div class="thin-note">Optimise mode: the solver picks the least-cost compliant blend.</div>',unsafe_allow_html=True)
    if solve or s.mix_result is None:
        try:
            if mode.startswith("Optimise"):
                with st.spinner("Optimising charge mix…"):
                    res,shadow,rows=E.solve_charge_mix(cfg(),mats,target,{"C":(clo,chi)},cu_limit=cu,tramp_limits={"Sn":sn})
                    s.mix_result=("optimise",res);s.mix_shadow=shadow;s.mix_rows=rows
            else:
                with st.spinner("Evaluating manual blend…"):
                    res=E.evaluate_manual_mix(cfg(),mats,s.mix_manual_weights);s.mix_result=("manual",res);s.mix_shadow={};s.mix_rows=[]
        except Exception as exc:
            st.error(f"Charge-mix calculation failed: {exc}")
    with right:
        U.section_title("Result — blend, bath chemistry, shadow price",small=True)
        mode_res,payload=s.mix_result if s.mix_result else (None,None)
        if mode_res=="optimise" and payload is not None:
            res=payload
            if not getattr(res,"feasible",False):
                st.markdown(U.pill("infeasible — widen C window or raise a ceiling","bad"),unsafe_allow_html=True);st.caption(getattr(res,"message",""));return
            bath=getattr(res,"predicted_bath_pct",{});st.markdown(U.pill("feasible — least-cost compliant blend","ok"),unsafe_allow_html=True)
            kc=st.columns(4);vals=[("Blend cost ₹/t",f'₹{res.cost_INR_per_t_liquid:,.0f}',"of liquid"),("Charge energy",f'{getattr(res,"energy_kWh",0):,.0f}',"kWh"),("Predicted Cu %",f'{bath.get("Cu",0):.3f}',f'≤ {cu:.2f}'),("Predicted C %",f'{bath.get("C",0):.3f}',f'{clo:.2f}–{chi:.2f}')]
            for col,(lab,val,sub) in zip(kc,vals):
                with col:U.kpi(lab,val,sub)
            st.dataframe(pd.DataFrame(s.mix_rows)[["Material","kg","% of charge"]].round(2) if s.mix_rows else pd.DataFrame(),use_container_width=True,height=215,hide_index=True)
            U.section_title("Predicted bath chemistry",small=True);st.dataframe(_bath_df(bath).round(4),use_container_width=True,height=180,hide_index=True)
            sh=s.mix_shadow.get("Cu")
            if sh and abs(sh)>1:
                per=abs(sh)/100.;st.info(f"Copper ceiling shadow price ≈ ₹{per:,.0f}/t liquid per 0.01% relaxed. Relaxing to {cu+0.01:.2f}% would save ≈ ₹{per:,.0f}/t.")
            else:st.info("Copper ceiling is not binding at this optimum — the cheapest blend already sits below it.")
        elif mode_res=="manual" and payload:
            m=payload
            if not m.get("feasible"):
                st.markdown(U.pill("no kg set — enter manual weights","warn"),unsafe_allow_html=True);return
            bath=m["predicted_bath_pct"];st.markdown(U.pill("manual blend evaluated — compare with optimiser","ok"),unsafe_allow_html=True)
            kc=st.columns(4);vals=[("Blend cost ₹/t",f'₹{m["cost_INR_per_t_liquid"]:,.0f}',f'{m["liquid_t"]:.1f} t liquid'),("Charge energy",f'{m["energy_kWh"]:,.0f}',"kWh"),("Predicted Cu %",f'{bath.get("Cu",0):.3f}',"tramp"),("Predicted C %",f'{bath.get("C",0):.3f}',"carbon")]
            for col,(lab,val,sub) in zip(kc,vals):
                with col:U.kpi(lab,val,sub)
            total=sum(s.mix_manual_weights.values());rr=[{"Material":n,"kg":kg,"% of charge":100*kg/total} for n,kg in sorted(s.mix_manual_weights.items(),key=lambda kv:-kv[1])]
            st.dataframe(pd.DataFrame(rr).round(2),use_container_width=True,height=215,hide_index=True);U.section_title("Predicted bath chemistry",small=True);st.dataframe(_bath_df(bath).round(4),use_container_width=True,height=180,hide_index=True)
            st.info(f"Operator blend: {total:,.0f} kg charged → {m['liquid_t']:.1f} t liquid at ₹{m['cost_INR_per_t_liquid']:,.0f}/t.")


# ────────────────────────────────────────────────────────────────────────────
# Economics
# ────────────────────────────────────────────────────────────────────────────
def render_economics():
    s=st.session_state;c=st.columns([1,1,1,.65]);tpy=c[0].slider("Annual output (t/yr)",5000,200000,40000,1000,key="eco_out");saving=c[1].slider("SEC saving (kWh/t)",10,100,40,1,key="eco_save");price=c[2].slider("Licence (₹ lakh)",5,40,20,1,key="eco_price");compute=c[3].button("Compute",use_container_width=True,key="eco_compute")
    sm=summary();tariff=sm["Tariff (₹/kWh)"];ef=sm["Grid EF (tCO₂/MWh)"];base=sm["Baseline SEC (kWh/t)"];floor=E.theoretical_floor_kWh_t(cfg());annual=tpy*saving*tariff;payback=(price*1e5)/annual*12 if annual>0 else float("inf");co2=tpy*saving/1000*ef
    cols=st.columns(4);vals=[("Annual saving",f'₹{annual/1e7:.2f} cr',f'at ₹{tariff:.1f}/kWh'),("Payback",f'{payback:.1f} mo',"energy alone"),("CO₂ avoided",f'{co2:,.0f} t/yr',f'at {ef:.3f}'),("Headroom left",f'{max(base-saving-floor,0):.0f} kWh/t',f'above {floor:.0f}')]
    for col,(lab,val,sub) in zip(cols,vals):
        with col:U.kpi(lab,val,sub)
    st.markdown(f'<div class="thin-note">At ₹{tariff:.1f}/kWh (mid-band Indian HT industrial). Energy alone — yield, alloy and reduced reblows are additional. Simple payback is arithmetic; realised payback is normally quoted as 4–12 months as savings ramp up.</div>',unsafe_allow_html=True)
    scen=[]
    for o in (30000,50000,100000):scen.append({"Annual output":f"{o:,} t/yr","30 kWh/t":f"₹{o*30*tariff/1e7:.2f} cr","50 kWh/t":f"₹{o*50*tariff/1e7:.2f} cr","80 kWh/t":f"₹{o*80*tariff/1e7:.2f} cr"})
    st.dataframe(pd.DataFrame(scen),use_container_width=True,hide_index=True,height=150)
    try:
        ec=E.economics_summary(cfg(),base,base-saving,tpy);edf=pd.DataFrame([{"Engine economics metric":k,"value":v} for k,v in ec.items()]);st.dataframe(edf,use_container_width=True,hide_index=True,height=300)
    except Exception as exc:st.caption(f"Detailed engine economics unavailable: {exc}")


# ────────────────────────────────────────────────────────────────────────────
# Heat log
# ────────────────────────────────────────────────────────────────────────────
def render_heat_log():
    s=st.session_state;c1,c2,c3=st.columns([.78,.11,.11]);c1.markdown('<div class="section-title">Heat log — audit trail</div>',unsafe_allow_html=True)
    if c2.button("Clear",use_container_width=True,key="log_clear"):s.sm_heat_log=[];st.rerun()
    csv=pd.DataFrame(s.sm_heat_log,columns=["clock","sim_min","event","detail"]).to_csv(index=False)
    c3.download_button("Export CSV",csv,f"smartmelt_heatlog_{_dt.datetime.now():%Y%m%d_%H%M%S}.csv","text/csv",use_container_width=True)
    st.markdown('<div class="thin-note">Every advisory shown, every action taken and every outcome lands here — the audit trail, ML training set and shared-savings evidence are the same table.</div>',unsafe_allow_html=True)
    df=pd.DataFrame(s.sm_heat_log,columns=["clock","sim_min","event","detail"])
    st.dataframe(df,use_container_width=True,height=560,hide_index=True,column_config={"clock":"Clock","sim_min":"Heat min","event":"Event","detail":"Detail"})


# ────────────────────────────────────────────────────────────────────────────
# Settings
# ────────────────────────────────────────────────────────────────────────────
def render_settings():
    s=st.session_state;U.section_title("Settings — plant & process configuration")
    st.markdown('<div class="thin-note">These set the aim and economic basis used by the advisory, endpoint checks and economics. Adjust to match your plant, then Apply.</div>',unsafe_allow_html=True)
    cc=cfg();left,right=st.columns([.55,.45])
    with left:
        with st.form("settings_form"):
            tap=st.number_input("Tap temperature aim (°C)",value=float(getattr(cc.plant,"tap_temperature_C",1620)))
            clo=st.number_input("Carbon aim — minimum (%)",value=float(getattr(cc.plant,"aim_C_lo_pct",.05)),format="%.3f")
            chi=st.number_input("Carbon aim — maximum (%)",value=float(getattr(cc.plant,"aim_C_hi_pct",.25)),format="%.3f")
            rated=st.number_input("Rated power (kW)",value=float(getattr(cc.electrical,"rated_power_kW",8000)))
            tariff=st.number_input("Electricity tariff (₹/kWh)",value=float(getattr(cc.economics,"tariff_INR_per_kWh",7.0)),format="%.3f")
            ef=st.number_input("Grid emission factor (tCO₂/MWh)",value=float(getattr(cc.economics,"grid_EF_tCO2_per_MWh",.712)),format="%.4f")
            baseline=st.number_input("Baseline SEC (kWh/t)",value=float(getattr(cc.economics,"baseline_SEC_kWh_per_t",600)))
            apply=st.form_submit_button("Apply settings")
        if apply:
            cc.plant.tap_temperature_C=tap
            if hasattr(cc.plant,"aim_C_lo_pct"):cc.plant.aim_C_lo_pct=clo;cc.plant.aim_C_hi_pct=chi
            cc.electrical.rated_power_kW=rated;cc.economics.tariff_INR_per_kWh=tariff;cc.economics.grid_EF_tCO2_per_MWh=ef;cc.economics.baseline_SEC_kWh_per_t=baseline
            s.sm_spec_result=None;s.sm_spec_key=None;s.physics_result=None;s.traj_result=None;log_event("SETTINGS","operator updated plant/process settings");set_status("settings applied","ok");st.success("applied — advisory & economics updated")
    with right:
        st.markdown(f"**Active plant configuration:** `{s.sm_plant}`")
        st.dataframe(pd.DataFrame([{"Setting":k,"Value":v} for k,v in E.config_summary(cc).items()]),use_container_width=True,hide_index=True,height=390)


# ────────────────────────────────────────────────────────────────────────────
# Validation
# ────────────────────────────────────────────────────────────────────────────
def _audit_df():
    sm=summary();floor=E.theoretical_floor_kWh_t(cfg())
    return pd.DataFrame([
        ("Latent heat of fusion",f"{sm['L_fusion (kJ/kg)']:.0f} kJ/kg","247","CRC Handbook 104th ed."),
        ("(FeO)+[C]→Fe+CO","1.39 MJ/kg FeO","+100 kJ/mol CO","Turkdogan; Fruehan MSTS"),
        ("FeSi75 heat of solution","−3511 kJ/kg","−4681 kJ/kg Si","Sigworth & Elliott 1974"),
        ("Carburiser heat of solution","+1883 kJ/kg C","+22.6 kJ/mol","graphite dissolution"),
        ("Grid emission factor",f"{sm['Grid EF (tCO₂/MWh)']:.3f} tCO₂/MWh","0.712","CEA v21.0, FY2024-25"),
        ("Reversible melting floor",f"{floor:.0f} kWh/t","practical ≈500","computed, L_f=247"),
        ("Default tariff",f"₹{sm['Tariff (₹/kWh)']:.1f}/kWh","₹6.0–8.5 grid","HT industrial FY25-26"),
        ("Baseline SEC",f"{sm['Baseline SEC (kWh/t)']:.0f} kWh/t","550–650 scrap IF","field practice"),
    ],columns=["Quantity","In model","Literature","Source"])


def _validation_fig(res):
    fig,ax=plt.subplots(figsize=(10,3.2));_style_axes(ax)
    lb=res.ledger_df
    if "closure_pct" in lb.columns:
        idc="element" if "element" in lb.columns else lb.columns[0];ax.bar(lb[idc].astype(str),lb.closure_pct.abs(),color=STEEL);ax.axhline(1,color=AMBER,ls="--");ax.set_ylabel("|closure| %");ax.set_title("Per-element mass-balance closure")
    fig.tight_layout();return fig


def render_validation():
    s=st.session_state;U.section_title("Validation — verified parameters & live conservation")
    U.section_title("Parameter audit — verified against the literature (v0.5)",small=True);st.dataframe(_audit_df(),use_container_width=True,hide_index=True,height=300)
    c1,c2=st.columns([.85,.15]);c1.markdown('<div class="section-title-sm">Live conservation check (fresh heat)</div>',unsafe_allow_html=True)
    rerun=c2.button("Re-run",use_container_width=True,key="validation_run")
    if rerun or s.validation_result is None:
        specs=[E.AdditionSpec("Lime (92% CaO)",10,48),E.AdditionSpec("FeSi75",45,15),E.AdditionSpec("Mill scale (FeO)",60,150)]
        with st.spinner("Running validation heat…"):s.validation_result=E.run_heat(cfg(),12000,dict(E.DEFAULT_CHARGE_COMP),5200,additions=E.build_additions(specs),dt=2.0)
    r=s.validation_result;aim=getattr(cfg().plant,"tap_temperature_C",1620);closure=r.energy.get("residual_pct",float("nan"));hit=abs(r.endpoint["T_C"]-aim)<=15
    cols=st.columns(4);pills=[(f"element ledger {r.ledger_max_pct:.2f}% < 1%","ok" if r.ledger_max_pct<1 else "warn"),(f"first-law {closure:+.1f}%","ok" if abs(closure)<5 else "warn"),(f"endpoint {r.endpoint['T_C']:.0f}°C","ok" if hit else "warn"),(f"undissolved {r.undissolved_kg:.0f} kg","ok" if r.undissolved_kg<5 else "warn")]
    for col,(txt,kind) in zip(cols,pills):col.markdown(U.pill(txt,kind),unsafe_allow_html=True)
    _plot(_validation_fig(r))


# ────────────────────────────────────────────────────────────────────────────
# About / Details
# ────────────────────────────────────────────────────────────────────────────
def render_about():
    st.markdown(f"# <span style='color:{MOLTEN}'>SmartMelt Studio</span>",unsafe_allow_html=True)
    st.markdown("Hybrid physics + machine-learning melt optimisation for induction, arc and basic-oxygen steelmaking. This browser application is a faithful Streamlit rendering of the full operator and manager console over the validated SmartMelt engine. It is advisory-only — it reads and computes, and never writes to a control system.")
    st.caption(f"Engine version {E.VERSION}. Plant identities are anonymised (Industry-X = MSME IF pilot, Industry-Y = integrated BOF).")
    st.markdown("## What each tab does")
    st.markdown("""
- **Operator Console** — start a heat, select playback speed, inject any flux, ferro-alloy or recarburiser at the current simulated time, watch the coloured furnace and streaming KPIs, and tap the heat.
- **Process Trajectory** — the same actual heat in six panels: temperatures, inventories, chemistry, slag/basicity, heat flows and energy.
- **Physics & Energy** — heat-flow ledger, first-law audit and energy split from grid input to tapped steel.
- **Virtual Sensor** — Extended Kalman Filter using intermittent immersion dips to estimate bath temperature and hidden furnace efficiency.
- **Machine Learning** — physics plus a gated residual ML head, with out-of-time performance against physics alone.
- **Drift Monitor** — PSI alarms when incoming scrap or practice changes.
- **Charge-Mix** — 17-stream least-cost optimiser and manual charge evaluation with Cu/Sn constraints.
- **Economics** — savings, payback and CO₂ using the active plant tariff and emission factor.
- **Heat Log** — session audit trail and CSV export.
- **Settings** — plant aims, power, tariff, grid factor and baseline SEC.
- **Validation** — verified-parameter audit plus live conservation test.
""")
    st.markdown("## The engine behind the GUI")
    st.code("""physics.py    first-principles furnace model (mass, energy, kinetics, refractory)
thermo.py     Wagner activities, equilibria, theoretical energy floor
ekf.py        Extended Kalman virtual temperature sensor
ml.py         hybrid GP-residual + GBM endpoint model, drift monitor
chargemix.py  least-cost charge LP with tramp shadow prices
mpc.py        receding-horizon power / tap-time advice
advisory.py   bilingual traffic-light operator guidance
simulator.py  virtual plant for rehearsal & ML data generation
metrics.py    hit-rates, PSI, economics
calibrate.py  per-plant calibration""")
    st.markdown("## Verified parameters (v0.5 literature pass)")
    st.code("""latent heat of fusion   272 → 247 kJ/kg           CRC Handbook 104th ed.
(FeO)+[C]→Fe+CO         1.89 → 1.39 MJ/kg FeO      Turkdogan; Fruehan MSTS
FeSi75 heat of solution −1150 → −3511 kJ/kg        Sigworth & Elliott 1974
carburiser              +2500 → +1883 kJ/kg C      graphite dissolution
grid emission factor    0.82 → 0.712 tCO₂/MWh      CEA v21.0, FY2024-25""")
    st.markdown("## How to run")
    st.code("streamlit run streamlit_app.py")
    st.caption("Figures are indicative until sized against a plant's audited baseline. This tool supports operators and managers; it does not replace metallurgical judgement or plant safety systems.")
