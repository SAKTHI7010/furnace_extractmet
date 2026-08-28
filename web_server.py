"""SmartMelt Studio Web Server.

Serves the frontend static files and exposes JSON APIs for all
mathematical model computations, simulations, EKF virtual sensor,
hybrid ML training, charge-mix LP, economics, settings, and validation.
"""
from __future__ import annotations

import sys
import os
import json
import time
import asyncio
import copy
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add app paths to sys.path
ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "app" / "lib"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import engine as E
from smartmelt.physics import make_addition

from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.requests import Request
from starlette.responses import JSONResponse, FileResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

# Global thread-safe session lock and state dictionary
_LOCK = asyncio.Lock()
session: Dict[str, Any] = {}

def init_session():
    configs = E.available_configs()
    default_plant = "if_msme_12t" if "if_msme_12t" in configs else next(iter(configs), "")
    
    session.clear()
    session.update({
        "sm_plant": default_plant,
        "sm_cfg": E.get_config(default_plant),
        "sm_status": "ready",
        "sm_status_kind": "ok",
        "sm_heat_log": [],
        "sm_heat_spec": {
            "charge_t": 12.0,
            "power_kW": 5200.0,
            "charge_C_pct": 0.30,
            "charge_Cu_pct": 0.20,
            "schedule": [
                {"material": "Lime (92% CaO)", "mass": 48.0, "time_min": 8.0},
                {"material": "FeSi75", "mass": 15.0, "time_min": 42.0},
                {"material": "Carburiser", "mass": 12.0, "time_min": 48.0},
                {"material": "Mill scale (FeO)", "mass": 120.0, "time_min": 58.0},
            ],
        },
        "sm_spec_result": None,
        "sm_spec_key": None,
        "op_charge_t": 12.0,
        "op_power_kW": 5200.0,
        "op_C_pct": 0.30,
        "op_Cu_pct": 0.20,
        "op_frames": None,
        "op_states": None,
        "op_pools": None,
        "op_frame_i": 0,
        "op_running": False,
        "op_tapped": False,
        "op_complete": False,
        "op_speed": 10,
        "op_applied_adds": [],
        "op_add_log": [],
        "op_injected": [],
        "op_last_tick": time.time(),
        "op_play_anchor_wall": time.time(),
        "op_play_anchor_frame": 0,
        "op_furnace_prev": None,
        "op_end_text": "",
        "traj_result": None,
        "physics_result": None,
        "ekf_result": None,
        "ml_result": None,
        "drift_result": None,
        "mix_result": None,
        "mix_shadow": {},
        "mix_rows": [],
        "mix_manual_weights": {},
        "validation_result": None,
    })

init_session()

# ── Shared spec-heat cache (keyed on plant+charge params) so tabs don't
# re-run the full 90-min simulation on every tab switch ──────────────────────
_SPEC_CACHE: Dict[str, Any] = {}

def _spec_cache_key(cfg, h: dict) -> str:
    import hashlib, json
    parts = {
        "plant": session.get("sm_plant", ""),
        "charge_t": h.get("charge_t", 12.0),
        "power_kW": h.get("power_kW", 5200.0),
        "C": h.get("charge_C_pct", 0.30),
        "Cu": h.get("charge_Cu_pct", 0.20),
        "schedule": json.dumps(h.get("schedule", []), sort_keys=True),
    }
    return hashlib.md5(json.dumps(parts, sort_keys=True).encode()).hexdigest()

async def _get_or_run_spec_heat(cfg, h: dict):
    """Return a cached HeatResult or compute one and cache it."""
    key = _spec_cache_key(cfg, h)
    if key in _SPEC_CACHE:
        return _SPEC_CACHE[key]
    def run_spec():
        comp = dict(E.DEFAULT_CHARGE_COMP)
        comp["C"] = h["charge_C_pct"] / 100.0
        comp["Cu"] = h["charge_Cu_pct"] / 100.0
        specs = [E.AdditionSpec(a["material"], a["time_min"], a["mass"]) for a in h.get("schedule", [])]
        return E.run_heat(cfg, h["charge_t"] * 1000.0, comp, h["power_kW"],
                          additions=E.build_additions(specs), dt=5.0)
    r = await asyncio.to_thread(run_spec)
    _SPEC_CACHE[key] = r
    return r

def log_event(event: str, detail: str = "", sim_min: float | None = None):
    session["sm_heat_log"].append({
        "clock": time.strftime("%H:%M:%S"),
        "sim_min": f"{sim_min:.1f}" if sim_min is not None else "",
        "event": event,
        "detail": detail,
    })

def make_serializable(obj: Any) -> Any:
    """Recursively convert NumPy/Pandas objects into standard JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if pd.isna(obj) if isinstance(obj, (float, int)) else False:
        return None
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return obj.replace({np.nan: None}).to_dict(orient="records")
    return obj

# ────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ────────────────────────────────────────────────────────────────────────────

async def get_index(request: Request):
    return FileResponse(ROOT / "static" / "index.html")

async def get_configs(request: Request):
    async with _LOCK:
        cfgs = list(E.available_configs().keys())
    return JSONResponse({"configs": cfgs, "active": session["sm_plant"]})

async def change_plant(request: Request):
    data = await request.json()
    plant_name = data.get("plant")
    async with _LOCK:
        configs = E.available_configs()
        if plant_name not in configs:
            return JSONResponse({"error": f"Unknown plant {plant_name}"}, status_code=400)
        
        session["sm_plant"] = plant_name
        session["sm_cfg"] = E.get_config(plant_name)
        session["sm_spec_result"] = None
        session["sm_spec_key"] = None
        session["traj_result"] = None
        session["physics_result"] = None
        session["validation_result"] = None
        session["op_frames"] = None
        session["op_states"] = None
        session["op_pools"] = None
        session["op_frame_i"] = 0
        session["op_running"] = False
        session["op_tapped"] = False
        session["op_complete"] = False
        session["op_applied_adds"] = []
        session["op_injected"] = []
        session["op_add_log"] = []
        session["op_end_text"] = ""
        session["sm_status"] = f"plant → {plant_name}"
        session["sm_status_kind"] = "ok"
        log_event("PLANT_CHANGE", f"Changed plant configuration to {plant_name}")
        _SPEC_CACHE.clear()
        
    return JSONResponse({
        "status": "success",
        "plant": plant_name,
        "config_summary": make_serializable(E.config_summary(session["sm_cfg"]))
    })

async def get_status(request: Request):
    async with _LOCK:
        frames = session["op_frames"]
        frame_i = session["op_frame_i"]
        n_frames = len(frames) if frames else 0
        cur_frame = frames[frame_i] if frames and frame_i < len(frames) else None

        res = {
            "plant": session["sm_plant"],
            "status": session["sm_status"],
            "status_kind": session["sm_status_kind"],
            "op_running": session["op_running"],
            "op_tapped": session["op_tapped"],
            "op_complete": session["op_complete"],
            "op_speed": session["op_speed"],
            "op_frame_i": frame_i,
            "n_frames": n_frames,
            "op_add_log": session["op_add_log"][-5:],
            # Return only current frame (not all frames) for fast polling
            "op_frames": frames,   # keep for JS compatibility but compress later
        }
    return JSONResponse(make_serializable(res))


async def operator_start(request: Request):
    data = await request.json()
    charge_t = float(data.get("charge_t", 12.0))
    power_kW = float(data.get("power_kW", 5200.0))
    C_pct    = float(data.get("C_pct", 0.30))
    Cu_pct   = float(data.get("Cu_pct", 0.20))

    # Default additions from the heat spec schedule
    default_schedule = [
        {"material": "Lime (92% CaO)", "mass": 48.0,  "time_min": 8.0},
        {"material": "FeSi75",          "mass": 15.0,  "time_min": 42.0},
        {"material": "Carburiser",      "mass": 12.0,  "time_min": 48.0},
        {"material": "Mill scale (FeO)", "mass": 120.0, "time_min": 58.0},
    ]

    comp = dict(E.DEFAULT_CHARGE_COMP)
    comp["C"]  = C_pct  / 100.0
    comp["Cu"] = Cu_pct / 100.0

    async with _LOCK:
        session["op_charge_t"] = charge_t
        session["op_power_kW"] = power_kW
        session["op_C_pct"]    = C_pct
        session["op_Cu_pct"]   = Cu_pct

        # Update heat spec to match so caches line up
        session["sm_heat_spec"] = {
            "charge_t": charge_t,
            "power_kW": power_kW,
            "charge_C_pct": C_pct,
            "charge_Cu_pct": Cu_pct,
            "schedule": default_schedule,
        }

        session["op_frames"]       = None
        session["op_states"]       = None
        session["op_pools"]        = None
        session["op_frame_i"]      = 0
        session["op_running"]      = False
        session["op_tapped"]       = False
        session["op_complete"]     = False
        session["op_applied_adds"] = []
        session["op_injected"]     = []
        session["op_add_log"]      = [f"Heat started · {charge_t:.1f} t · {power_kW:.0f} kW"]
        session["op_end_text"]     = ""
        session["sm_status"]       = "simulating heat"
        session["sm_status_kind"]  = "warn"
        cfg = session["sm_cfg"]

    # Also prime the spec cache so trajectory tab is instant afterwards
    specs = [E.AdditionSpec(a["material"], a["time_min"], a["mass"])
             for a in default_schedule]

    def run_sim():
        return E.simulate_frames_live(
            cfg, charge_t, comp,
            power_kW, additions=E.build_additions(specs), dt=5.0, cooperative=False
        )

    try:
        frames, states, pools = await asyncio.to_thread(run_sim)
        async with _LOCK:
            session["op_frames"]  = frames
            session["op_states"]  = states
            session["op_pools"]   = pools
            session["op_running"] = True
            session["sm_status"]  = "heat running"
            session["sm_status_kind"] = "ok"
            log_event("HEAT START",
                      f"{charge_t:.1f} t, {power_kW:.0f} kW, C {C_pct:.2f}%", 0.0)

        # Return only status + first frame, NOT all frames (too large)
        first_frame = frames[0] if frames else {}
        return JSONResponse(make_serializable({
            "status": "success",
            "n_frames": len(frames),
            "first_frame": first_frame,
            "add_log": session["op_add_log"]
        }))
    except Exception as e:
        traceback.print_exc()
        async with _LOCK:
            session["op_running"]      = False
            session["sm_status"]       = f"simulation failed: {str(e)}"
            session["sm_status_kind"]  = "bad"
        return JSONResponse({"error": str(e)}, status_code=500)


async def operator_inject(request: Request):
    data = await request.json()
    cut_i = int(data.get("cut_i", 0))
    material = data.get("material")
    mass = float(data.get("mass", 0.0))
    
    async with _LOCK:
        if not session["op_running"] or session["op_tapped"]:
            return JSONResponse({"error": "Heat is not active"}, status_code=400)
        
        frames = session["op_frames"]
        states = session["op_states"]
        pools = session["op_pools"]
        cfg = session["sm_cfg"]
        charge_t = session["op_charge_t"]
        power_kW = session["op_power_kW"]
        C_pct = session["op_C_pct"]
        Cu_pct = session["op_Cu_pct"]
        
        cut_i = max(0, min(cut_i, len(frames) - 1))
        cur_frame = frames[cut_i]
        tmin = float(cur_frame["t_min"])
        Tb = float(cur_frame["T_bath_C"])
        
        info = E.ADDITION_LIBRARY.get(material)
        if info is None or mass <= 0:
            return JSONResponse({"error": f"Invalid material {material} or mass {mass}"}, status_code=400)
            
        # Create addition object
        add = E.make_addition_at(tmin * 60.0, mass, info)
        session["op_applied_adds"].append({"material": material, "mass": mass, "time_min": tmin})
        session["op_injected"].append(add)
        log_msg = f"{tmin:4.1f} min · +{mass:.0f} kg {material.split(' (')[0]} @ {Tb:.0f}°C"
        session["op_add_log"].append(log_msg)
        log_event("ADDITION", f"+{mass:.0f} kg {material} @ {Tb:.0f}°C", tmin)
        
        comp = dict(E.DEFAULT_CHARGE_COMP)
        comp["C"] = C_pct / 100.0
        comp["Cu"] = Cu_pct / 100.0
        
        # Continuation inputs
        from_state = np.asarray(states[cut_i], dtype=float).copy()
        from_pool = copy.deepcopy(pools[cut_i])
        
    def run_continuation():
        return E.simulate_frames_live(
            cfg, charge_t, comp, power_kW, additions=[add], dt=5.0,
            from_state=from_state, from_pool=from_pool, t0_s=tmin*60.0, cooperative=False
        )
        
    try:
        new_frames, new_states, new_pools = await asyncio.to_thread(run_continuation)
        async with _LOCK:
            prefix_frames = list(session["op_frames"][:cut_i + 1])
            prefix_states = np.asarray(session["op_states"][:cut_i + 1]).copy()
            prefix_pools = list(session["op_pools"][:cut_i + 1])
            
            session["op_frames"] = prefix_frames + list(new_frames)
            session["op_states"] = np.concatenate([prefix_states, np.asarray(new_states)], axis=0) if len(new_states) else prefix_states
            session["op_pools"] = prefix_pools + list(new_pools)
            
        return JSONResponse(make_serializable({
            "status": "success",
            "frames": session["op_frames"],
            "add_log": session["op_add_log"]
        }))
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

async def operator_tap(request: Request):
    data = await request.json()
    cut_i = int(data.get("cut_i", 0))
    
    async with _LOCK:
        if not session["op_frames"]:
            return JSONResponse({"error": "No simulated heat exists"}, status_code=400)
            
        session["op_running"] = False
        session["op_tapped"] = True
        
        cut_i = max(0, min(cut_i, len(session["op_frames"]) - 1))
        session["op_frames"] = session["op_frames"][:cut_i + 1]
        if session["op_states"] is not None:
            session["op_states"] = session["op_states"][:cut_i + 1]
        if session["op_pools"] is not None:
            session["op_pools"] = session["op_pools"][:cut_i + 1]
            
        snap = session["op_frames"][-1]
        aim = getattr(session["sm_cfg"].plant, "tap_temperature_C", 1620)
        
        adds_str = ", ".join(f"{a['mass']:.0f}kg {a['material'].split(' (')[0]}@{a['time_min']:.0f}min" for a in session["op_applied_adds"]) or "none"
        end_text = (f"Tapped at {snap['t_min']:.0f} min · {snap['T_bath_C']:.0f} °C · C {snap['pct_C']:.3f}% · "
                    f"SEC {snap['SEC_kWh_t']:.0f} kWh/t · slag FeO {snap['slag_FeO_pct']:.1f}% · B2 {snap['B2']:.2f}\n"
                    f"Additions this heat: {adds_str}")
        session["op_end_text"] = end_text
        
        # Update heat spec schedule to match this run
        session["sm_heat_spec"] = {
            "charge_t": session["op_charge_t"],
            "power_kW": session["op_power_kW"],
            "charge_C_pct": session["op_C_pct"],
            "charge_Cu_pct": session["op_Cu_pct"],
            "schedule": copy.deepcopy(session["op_applied_adds"])
        }
        session["sm_spec_result"] = None
        session["sm_spec_key"] = None
        
        hit = abs(snap["T_bath_C"] - aim) <= 15
        session["sm_status"] = "TAPPED — " + ("on aim" if hit else f'{snap["T_bath_C"] - aim:+.0f}°C off aim')
        session["sm_status_kind"] = "ok" if hit else "warn"
        log_event("TAP", f"{snap['T_bath_C']:.0f}°C, C {snap['pct_C']:.3f}%, SEC {snap['SEC_kWh_t']:.0f} kWh/t", snap["t_min"])
        
    return JSONResponse(make_serializable({
        "status": "success",
        "end_text": end_text,
        "final_frame": snap
    }))

async def run_trajectory(request: Request):
    async with _LOCK:
        live = session["op_frames"]
        h = session["sm_heat_spec"]
        cfg = session["sm_cfg"]

    aim = getattr(cfg.plant, "tap_temperature_C", 1620)
    floor = E.theoretical_floor_kWh_t(cfg)

    if live:
        d = pd.DataFrame(live)
        endpoint_row = d.iloc[-1]
        ledger_str = "live"
        tapmin = float(endpoint_row.get("t_min", 0))
        tap_T = float(endpoint_row.get("T_bath_C", 0))
        tap_C = float(endpoint_row.get("pct_C", 0))
        tap_sec = float(endpoint_row.get("SEC_kWh_t", 0))
    else:
        r = await _get_or_run_spec_heat(cfg, h)
        d = r.df
        ledger_str = f"{r.ledger_max_pct:.2f}"
        tapmin = r.tap_min
        tap_T = r.endpoint["T_C"]
        tap_C = r.endpoint["pct_C"]
        tap_sec = float(d["SEC_kWh_t"].iloc[-1])

    kpis = {
        "tap_temp": tap_T,
        "tap_temp_aim": aim,
        "carbon": tap_C,
        "tap_min": tapmin,
        "sec": tap_sec,
        "sec_floor": floor,
        "ledger": ledger_str
    }

    # Ensure all expected columns exist with correct names
    # Legacy pages use M_liquid_t / M_solid_t; also expose m_*_kg aliases
    for old_col, new_col in [("M_liquid_t", "m_liquid_kg"), ("M_solid_t", "m_solid_kg")]:
        if old_col in d.columns and new_col not in d.columns:
            d = d.copy()
            d[new_col] = d[old_col] * 1000.0
        elif old_col not in d.columns and new_col in d.columns:
            d = d.copy()
            d[old_col] = d[new_col] / 1000.0

    for c in ["T_solid_C", "T_hotface_C", "M_solid_t", "M_liquid_t",
               "m_solid_kg", "m_liquid_kg", "m_undissolved_kg", "undissolved_kg",
               "pct_S", "Q_wall_kW", "Q_rad_kW", "Q_bath_to_scrap_kW",
               "Q_chem_kW", "Q_useful_kW", "Q_offgas_kW", "Q_cool_kW",
               "SEC_kWh_t", "B2", "slag_FeO_pct"]:
        if c not in d.columns:
            d = d.copy()
            d[c] = None

    # undissolved_kg alias
    if "undissolved_kg" in d.columns and "m_undissolved_kg" not in d.columns:
        d = d.copy()
        d["m_undissolved_kg"] = d["undissolved_kg"]

    return JSONResponse(make_serializable({
        "kpis": kpis,
        "data": d,
        "additions": h.get("schedule", [])
    }))

async def force_run_trajectory(request: Request):
    async with _LOCK:
        h = session["sm_heat_spec"]
        cfg = session["sm_cfg"]
        _SPEC_CACHE.pop(_spec_cache_key(cfg, h), None)   # force fresh run

    r = await _get_or_run_spec_heat(cfg, h)
    async with _LOCK:
        session["traj_result"] = r
        session["op_frames"] = None
        session["op_running"] = False
        session["op_tapped"] = False

    aim = getattr(cfg.plant, "tap_temperature_C", 1620)
    floor = E.theoretical_floor_kWh_t(cfg)

    kpis = {
        "tap_temp": r.endpoint["T_C"],
        "tap_temp_aim": aim,
        "carbon": r.endpoint["pct_C"],
        "tap_min": r.tap_min,
        "sec": float(r.df.SEC_kWh_t.iloc[-1]),
        "sec_floor": floor,
        "ledger": f"{r.ledger_max_pct:.2f}"
    }
    d = r.df.copy()
    for old_col, new_col in [("M_liquid_t", "m_liquid_kg"), ("M_solid_t", "m_solid_kg")]:
        if old_col in d.columns and new_col not in d.columns:
            d[new_col] = d[old_col] * 1000.0
        elif old_col not in d.columns and new_col in d.columns:
            d[old_col] = d[new_col] / 1000.0
    for c in ["T_solid_C", "T_hotface_C", "M_solid_t", "M_liquid_t",
               "m_solid_kg", "m_liquid_kg", "m_undissolved_kg", "undissolved_kg",
               "pct_S", "Q_wall_kW", "Q_rad_kW", "Q_bath_to_scrap_kW",
               "Q_chem_kW", "Q_useful_kW", "Q_offgas_kW", "Q_cool_kW",
               "SEC_kWh_t", "B2", "slag_FeO_pct"]:
        if c not in d.columns:
            d[c] = None
    if "undissolved_kg" in d.columns and "m_undissolved_kg" not in d.columns:
        d["m_undissolved_kg"] = d["undissolved_kg"]

    return JSONResponse(make_serializable({
        "kpis": kpis,
        "data": d,
        "additions": h.get("schedule", [])
    }))

async def run_physics(request: Request):
    async with _LOCK:
        h = session["sm_heat_spec"]
        cfg = session["sm_cfg"]

    r = await _get_or_run_spec_heat(cfg, h)
    en = r.energy
    floor = E.theoretical_floor_kWh_t(cfg)
    uf = 100 * en.get("useful_fraction", 0)

    kpis = {
        "ledger_max": r.ledger_max_pct,
        "first_law_closure": en.get("residual_pct", float("nan")),
        "final_sec": float(r.df.SEC_kWh_t.iloc[-1]),
        "sec_floor": floor,
        "useful_fraction": uf
    }

    # Waterfall
    total_in = en.get("grid_kWh", float(r.df.E_kWh.iloc[-1]))
    losses = {
        "converter": en.get("converter_loss_kWh", 0),
        "coil water": en.get("coil_water_loss_kWh", 0),
        "lining": en.get("lining_loss_kWh", 0),
        "radiation": en.get("radiation_loss_kWh", 0),
        "off-gas": en.get("offgas_loss_kWh", 0)
    }
    useful = en.get("useful_melt_kWh", 0)
    waterfall = {
        "labels": ["Grid Input"] + list(losses.keys()) + ["To Steel"],
        "values": [total_in] + [-v for v in losses.values()] + [useful]
    }

    d = r.df.copy()
    # Ensure heat-flow columns present (engine key -> df column)
    _flow_aliases = [
        ("Q_useful_kW", "Q_useful_kW"),
        ("Q_wall_kW", "Q_wall_kW"),
        ("Q_rad_kW", "Q_rad_kW"),
        ("Q_bath_to_scrap_kW", "Q_bath_to_scrap_kW"),
        ("Q_chem_kW", "Q_chem_kW"),
        ("Q_offgas_kW", "Q_offgas_kW"),
        ("Q_cool_kW", "Q_cool_kW"),
    ]
    for src, dst in _flow_aliases:
        if dst not in d.columns:
            d[dst] = None

    # Build energy audit rows suitable for a table
    energy_audit_table = [
        {"component": k.replace("_kWh", "").replace("_", " ").title(),
         "energy_kWh": round(float(v), 1)}
        for k, v in en.items()
        if isinstance(v, (int, float)) and not k.endswith("_pct")
    ]

    return JSONResponse(make_serializable({
        "kpis": kpis,
        "data": d,
        "energy_audit": en,
        "energy_audit_table": energy_audit_table,
        "waterfall": waterfall
    }))

async def run_ekf(request: Request):
    data = await request.json()
    true_eta = float(data.get("true_eta", 0.90))
    true_UA_scale = float(data.get("true_ua", 1.35))
    ndips = int(data.get("ndips", 3))
    
    async with _LOCK:
        cfg = session["sm_cfg"]
        
    def run_filter():
        dip_times = tuple(np.linspace(30.0, 78.0, ndips))
        return E.run_ekf_demo(cfg, true_eta=true_eta, true_UA_scale=true_UA_scale, dip_times_min=dip_times, seed=1, dt=5.0)
        
    try:
        ek = await asyncio.to_thread(run_filter)
        eta_final = ek.theta_path.eta_electrical.iloc[-1]
        
        kpis = {
            "final_error": ek.final_error_C,
            "eta_converged": eta_final,
            "sigma_end": ek.df.sigma_T.iloc[-1],
            "dips_count": len(ek.dip_df)
        }
        
        for c in ["sigma_T"]:
            if c not in ek.df.columns:
                ek.df[c] = None
        for c in ["t_min", "T_meas_C"]:
            if c not in ek.dip_df.columns:
                ek.dip_df[c] = None
        for c in ["UA_lining_scale", "eta_electrical"]:
            if c not in ek.theta_path.columns:
                ek.theta_path[c] = None

        return JSONResponse(make_serializable({
            "kpis": kpis,
            "df": ek.df,
            "dip_df": ek.dip_df,
            "theta_path": ek.theta_path
        }))
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

async def get_default_ekf(request: Request):
    def load_cache():
        return E.load_default_ekf()
        
    ek = await asyncio.to_thread(load_cache)
    if ek is None:
        # Fallback to a fast EKF computation
        async with _LOCK:
            cfg = session["sm_cfg"]
        try:
            ek = E.run_ekf_demo(cfg, true_eta=0.90, true_UA_scale=1.35, dip_times_min=(30.0, 54.0, 78.0), seed=1)
        except Exception as e:
            return JSONResponse({"error": f"EKF execution failed: {str(e)}"}, status_code=500)
            
    eta_final = ek.theta_path.eta_electrical.iloc[-1]
    kpis = {
        "final_error": ek.final_error_C,
        "eta_converged": eta_final,
        "sigma_end": ek.df.sigma_T.iloc[-1],
        "dips_count": len(ek.dip_df)
    }
    for c in ["sigma_T"]:
        if c not in ek.df.columns:
            ek.df[c] = None
    for c in ["t_min", "T_meas_C"]:
        if c not in ek.dip_df.columns:
            ek.dip_df[c] = None
    for c in ["UA_lining_scale", "eta_electrical"]:
        if c not in ek.theta_path.columns:
            ek.theta_path[c] = None

    return JSONResponse(make_serializable({
        "kpis": kpis,
        "df": ek.df,
        "dip_df": ek.dip_df,
        "theta_path": ek.theta_path
    }))

async def run_ml(request: Request):
    data = await request.json()
    split = float(data.get("split", 0.70))
    n_heats = int(data.get("n_heats", 40))
    live = bool(data.get("live", False))
    
    async with _LOCK:
        cfg = session["sm_cfg"]
        
    def train():
        if not live:
            d = E.load_cached_dataset()
            if d is None:
                d = E.generate_dataset(cfg, n_heats=40, seed=0)
        else:
            d = E.generate_dataset(cfg, n_heats=n_heats, seed=0)
        return E.train_hybrid(cfg, d, split_frac=split)
        
    try:
        ml = await asyncio.to_thread(train)
        m = ml.metrics
        return JSONResponse(make_serializable({
            "metrics": m,
            "pred_df": ml.pred_df
        }))
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

async def run_drift_endpoint(request: Request):
    data = await request.json()
    n_heats = int(data.get("n_heats", 50))
    reg = int(data.get("reg", 40))
    live = bool(data.get("live", False))
    
    async with _LOCK:
        cfg = session["sm_cfg"]
        
    def run_check():
        if not live:
            d = E.load_cached_dataset()
            if d is None:
                d = E.generate_dataset(cfg, n_heats=50, seed=0)
        else:
            d = E.generate_dataset(cfg, n_heats=n_heats, seed=0, regime_change_at=reg)
        dr = E.run_drift(cfg, d, ref_frac=.5)
        return d, dr
        
    try:
        d, dr = await asyncio.to_thread(run_check)
        # Select target variable to plot feature deviation (Cu %)
        cu_col = "charge_Cu_pct" if "charge_Cu_pct" in d else d.columns[0]
        tracking = {
            "x": list(d.index),
            "y": list(d[cu_col]),
            "feature": cu_col
        }
        
        psi_val = dr.get("psi_df")
        if isinstance(psi_val, pd.DataFrame):
            # The column is "PSI" uppercase in engine.py
            psi_list = [{"feature": str(row["feature"]), "psi": float(row["PSI"])} for idx, row in psi_val.iterrows()]
        elif isinstance(psi_val, pd.Series):
            psi_list = [{"feature": str(k), "psi": float(v)} for k, v in psi_val.items()]
        else:
            psi_list = make_serializable(psi_val)

        return JSONResponse(make_serializable({
            "alarm": dr["alarm"],
            "psi_max": dr["psi_max"],
            "n_ref": dr["n_ref"],
            "n_recent": dr["n_recent"],
            "reasons": dr.get("reasons", []),
            "psi_df": psi_list,
            "tracking": tracking,
            "regime_change_at": reg if live else 40
        }))
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

async def get_scrap_library(request: Request):
    return JSONResponse(E.default_materials())

async def run_chargemix(request: Request):
    data = await request.json()
    mode = data.get("mode", "optimise") # 'optimise' or 'manual'
    target = float(data.get("target", 12.0))
    clo = float(data.get("clo", 0.10))
    chi = float(data.get("chi", 0.40))
    cu = float(data.get("cu", 0.20))
    sn = float(data.get("sn", 0.03))
    
    # Custom scrap list with potentially modified prices or availabilities
    mats = data.get("mats", E.default_materials())
    manual_weights = data.get("manual_weights", {})
    
    async with _LOCK:
        cfg = session["sm_cfg"]
        
    def solve():
        if mode == "optimise":
            return E.solve_charge_mix(cfg, mats, target, {"C": (clo, chi)}, cu_limit=cu, tramp_limits={"Sn": sn})
        else:
            res = E.evaluate_manual_mix(cfg, mats, manual_weights)
            return res, {}, res.get("rows", [])
            
    try:
        res, shadow, rows = await asyncio.to_thread(solve)
        
        if mode == "optimise":
            feasible = getattr(res, "feasible", False)
            if not feasible:
                return JSONResponse({
                    "feasible": False,
                    "message": getattr(res, "message", "Optimization infeasible")
                })
            bath = getattr(res, "predicted_bath_pct", {})
            cost = res.cost_INR_per_t_liquid
            energy = getattr(res, "energy_kWh", 0.0)
            
            # Find shadow price info
            sh = shadow.get("Cu")
            shadow_msg = ""
            if sh and abs(sh) > 1:
                per = abs(sh) / 100.0
                shadow_msg = f"Copper ceiling shadow price ≈ ₹{per:,.0f}/t liquid per 0.01% relaxed. Relaxing to {cu+0.01:.2f}% would save ≈ ₹{per:,.0f}/t."
            else:
                shadow_msg = "Copper ceiling is not binding at this optimum — the cheapest blend already sits below it."
                
            return JSONResponse(make_serializable({
                "feasible": True,
                "cost_per_t": cost,
                "energy_kWh": energy,
                "predicted_bath": bath,
                "rows": rows,
                "shadow_msg": shadow_msg
            }))
        else:
            # Manual Evaluation mode
            feasible = res.get("feasible", False)
            if not feasible:
                return JSONResponse({
                    "feasible": False,
                    "message": res.get("message", "No manual weights provided")
                })
            
            return JSONResponse(make_serializable({
                "feasible": True,
                "cost_per_t": res["cost_INR_per_t_liquid"],
                "energy_kWh": res["energy_kWh"],
                "predicted_bath": res["predicted_bath_pct"],
                "rows": rows,
                "charge_kg": res["charge_kg"],
                "liquid_t": res["liquid_t"]
            }))
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

async def run_economics(request: Request):
    data = await request.json()
    tpy = float(data.get("tpy", 40000.0))
    saving = float(data.get("saving", 40.0))
    price_lakh = float(data.get("price_lakh", 20.0))
    
    async with _LOCK:
        cfg = session["sm_cfg"]
        
    sm = E.config_summary(cfg)
    tariff = sm["Tariff (₹/kWh)"]
    ef = sm["Grid EF (tCO₂/MWh)"]
    base = sm["Baseline SEC (kWh/t)"]
    floor = E.theoretical_floor_kWh_t(cfg)
    
    annual = tpy * saving * tariff
    payback = (price_lakh * 1e5) / annual * 12 if annual > 0 else float("inf")
    co2 = tpy * saving / 1000 * ef
    
    scenarios = []
    for o in (30000, 50000, 100000):
        scenarios.append({
            "output": f"{o:,} t/yr",
            "save_30": f"₹{o*30*tariff/1e7:.2f} cr",
            "save_50": f"₹{o*50*tariff/1e7:.2f} cr",
            "save_80": f"₹{o*80*tariff/1e7:.2f} cr"
        })
        
    def econ_summary():
        return E.economics_summary(cfg, base, base - saving, tonnes_per_year=tpy)
        
    try:
        detailed = await asyncio.to_thread(econ_summary)
        detailed_rows = [{"metric": k, "value": v} for k, v in detailed.items()]
    except Exception as e:
        detailed_rows = [{"metric": f"Unavailable: {str(e)}", "value": 0}]
        
    return JSONResponse(make_serializable({
        "tariff": tariff,
        "ef": ef,
        "baseline": base,
        "floor": floor,
        "annual_saving_cr": annual / 1e7,
        "payback_months": payback,
        "co2_avoided": co2,
        "headroom": max(base - saving - floor, 0),
        "scenarios": scenarios,
        "detailed": detailed_rows,
        "detailed_rows": detailed_rows
    }))

async def get_settings(request: Request):
    async with _LOCK:
        cfg = session["sm_cfg"]
        plant = session["sm_plant"]
        
    summary_data = E.config_summary(cfg)
    editable = {
        "tap_temperature_C": getattr(cfg.plant, "tap_temperature_C", 1620.0),
        "aim_C_lo_pct": getattr(cfg.plant, "aim_C_lo_pct", 0.05) if hasattr(cfg.plant, "aim_C_lo_pct") else 0.05,
        "aim_C_hi_pct": getattr(cfg.plant, "aim_C_hi_pct", 0.25) if hasattr(cfg.plant, "aim_C_hi_pct") else 0.25,
        "rated_power_kW": getattr(cfg.electrical, "rated_power_kW", 8000.0),
        "tariff_INR_per_kWh": getattr(cfg.economics, "tariff_INR_per_kWh", 7.0),
        "grid_EF_tCO2_per_MWh": getattr(cfg.economics, "grid_EF_tCO2_per_MWh", 0.712),
        "baseline_SEC_kWh_per_t": getattr(cfg.economics, "baseline_SEC_kWh_per_t", 600.0)
    }
    
    return JSONResponse(make_serializable({
        "active_plant": plant,
        "summary": summary_data,
        "editable": editable
    }))

async def apply_settings(request: Request):
    data = await request.json()
    tap = float(data.get("tap", 1620.0))
    clo = float(data.get("clo", 0.05))
    chi = float(data.get("chi", 0.25))
    rated = float(data.get("rated", 8000.0))
    tariff = float(data.get("tariff", 7.0))
    ef = float(data.get("ef", 0.712))
    baseline = float(data.get("baseline", 600.0))
    
    async with _LOCK:
        cfg = session["sm_cfg"]
        cfg.plant.tap_temperature_C = tap
        if hasattr(cfg.plant, "aim_C_lo_pct"):
            cfg.plant.aim_C_lo_pct = clo
            cfg.plant.aim_C_hi_pct = chi
        cfg.electrical.rated_power_kW = rated
        cfg.economics.tariff_INR_per_kWh = tariff
        cfg.economics.grid_EF_tCO2_per_MWh = ef
        cfg.economics.baseline_SEC_kWh_per_t = baseline
        
        session["sm_spec_result"] = None
        session["sm_spec_key"] = None
        session["physics_result"] = None
        session["traj_result"] = None
        _SPEC_CACHE.clear()

        log_event("SETTINGS", "Operator updated plant and process settings")
        session["sm_status"] = "settings applied"
        session["sm_status_kind"] = "ok"

    return JSONResponse({"status": "success"})

async def run_validation(request: Request):
    async with _LOCK:
        h = session["sm_heat_spec"]
        cfg = session["sm_cfg"]

    try:
        r = await _get_or_run_spec_heat(cfg, h)
        aim = getattr(cfg.plant, "tap_temperature_C", 1620)
        closure = r.energy.get("residual_pct", float("nan"))
        hit = abs(r.endpoint["T_C"] - aim) <= 15

        pills = [
            {"text": f"element ledger {r.ledger_max_pct:.2f}% < 1%",
             "kind": "ok" if r.ledger_max_pct < 1.0 else "warn"},
            {"text": f"first-law {closure:+.1f}%",
             "kind": "ok" if abs(closure) < 5.0 else "warn"},
            {"text": f"endpoint {r.endpoint['T_C']:.0f}°C",
             "kind": "ok" if hit else "warn"},
            {"text": f"undissolved {r.undissolved_kg:.0f} kg",
             "kind": "ok" if r.undissolved_kg < 5.0 else "warn"}
        ]

        # Element-wise ledger for bar chart
        ledger_rows = []
        if hasattr(r, 'ledger_df') and r.ledger_df is not None:
            ldf = r.ledger_df
            for _, row in ldf.iterrows():
                el = str(row.get("element", row.name if hasattr(row, 'name') else ""))
                cl = float(row.get("closure_pct", row.get("closure", 0.0)))
                ledger_rows.append({"element": el, "closure_pct": cl})

        # Audit rows (for the literature table)
        floor = E.theoretical_floor_kWh_t(cfg)
        audit_rows = [
            {"quantity": "Latent heat Fe",  "in_model": "247 kJ/kg",       "literature": "247 kJ/kg",     "source": "CRC 104th ed."},
            {"quantity": "Cp liquid Fe",    "in_model": "0.824 kJ/kg·K",  "literature": "~0.82 kJ/kg·K", "source": "Iida & Guthrie 1988"},
            {"quantity": "SEC floor",       "in_model": f"{floor:.0f} kWh/t", "literature": "381 kWh/t",  "source": "First-Law calc."},
            {"quantity": "Grid EF",         "in_model": "0.712 tCO₂/MWh", "literature": "0.712 tCO₂/MWh", "source": "CEA DB v21.0"}
        ]

        return JSONResponse(make_serializable({
            "pills": pills,
            "ledger_df": ledger_rows,
            "endpoint": r.endpoint,
            "energy": r.energy,
            "audit_rows": audit_rows,
            "element_ledger_pct": r.ledger_max_pct,
            "first_law_pct": closure,
            "endpoint_C": r.endpoint["T_C"],
            "undissolved_kg": r.undissolved_kg,
        }))
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

async def get_heat_log(request: Request):
    async with _LOCK:
        logs = list(session["sm_heat_log"])
    return JSONResponse(make_serializable({"logs": logs}))

async def clear_heat_log(request: Request):
    async with _LOCK:
        session["sm_heat_log"] = []
        log_event("LOG_CLEAR", "Heat log cleared by operator")
    return JSONResponse({"status": "success"})

async def run_advisory_endpoint(request: Request):
    """Return live advisory verdicts based on current operator frame snapshot."""
    async with _LOCK:
        cfg = session["sm_cfg"]
        frames = session.get("op_frames") or []
        frame_i = session.get("op_frame_i", 0)

    try:
        if frames:
            # Use the current playback frame's snapshot
            idx = min(frame_i, len(frames) - 1)
            snap = frames[idx]
        else:
            # Use whatever the client sent
            data = await request.json()
            snap = data.get("snapshot") or {}
            if not snap.get("T_bath_C"):
                # No data yet: return a welcome advisory
                return JSONResponse(make_serializable({
                    "advisories": [{
                        "level": "info",
                        "title": "SmartMelt Ready",
                        "message": "Start a heat to receive live metallurgical guidance."
                    }]
                }))

        verdicts = E.build_advisories(snap, cfg)
        advisories = [
            {"level": lvl, "title": title, "message": msg}
            for lvl, title, msg in verdicts
        ]
        return JSONResponse(make_serializable({"advisories": advisories}))
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# ────────────────────────────────────────────────────────────────────────────
# Routing setup
# ────────────────────────────────────────────────────────────────────────────

routes = [
    Route("/", get_index),
    Route("/api/configs", get_configs, methods=["GET", "POST"]),
    Route("/api/change_plant", change_plant, methods=["POST"]),
    Route("/api/status", get_status, methods=["GET", "POST"]),
    Route("/api/operator/start", operator_start, methods=["POST"]),
    Route("/api/operator/inject", operator_inject, methods=["POST"]),
    Route("/api/operator/tap", operator_tap, methods=["POST"]),
    Route("/api/trajectory/run", run_trajectory, methods=["POST"]),
    Route("/api/trajectory/force_run", force_run_trajectory, methods=["POST"]),
    Route("/api/physics/run", run_physics, methods=["POST"]),
    Route("/api/ekf/run", run_ekf, methods=["POST"]),
    Route("/api/ekf/default", get_default_ekf, methods=["GET", "POST"]),
    Route("/api/ml/train", run_ml, methods=["POST"]),
    Route("/api/drift/run", run_drift_endpoint, methods=["POST"]),
    Route("/api/chargemix/solve", run_chargemix, methods=["POST"]),
    Route("/api/chargemix/library", get_scrap_library, methods=["GET", "POST"]),
    Route("/api/economics/compute", run_economics, methods=["POST"]),
    Route("/api/settings/get", get_settings, methods=["GET", "POST"]),
    Route("/api/settings/apply", apply_settings, methods=["POST"]),
    Route("/api/validation/run", run_validation, methods=["POST"]),
    Route("/api/advisory/evaluate", run_advisory_endpoint, methods=["POST"]),
    Route("/api/log/get", get_heat_log, methods=["GET", "POST"]),
    Route("/api/log/clear", clear_heat_log, methods=["POST"]),
    Mount("/static", app=StaticFiles(directory=str(ROOT / "static")), name="static")
]

app = Starlette(
    routes=routes,
    middleware=[
        Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    ]
)

if __name__ == "__main__":
    import uvicorn
    print(f"Starting SmartMelt Studio at http://localhost:8000")
    uvicorn.run(app, host="localhost", port=8000, log_level="info")
