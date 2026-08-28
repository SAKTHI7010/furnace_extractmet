"""Regression test for exact, fast continuation after a live addition."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app" / "lib"))

import engine as E


def test_continuation_matches_full_recalculation_exactly():
    cfg = E.get_config("if_msme_12t")
    comp = dict(E.DEFAULT_CHARGE_COMP)
    comp["C"] = 0.003
    comp["Cu"] = 0.002

    base, states, pools = E.simulate_frames_live(
        cfg, 12.0, comp, 5200.0, additions=[], dt=2.0,
        t_end_min=36.0, cooperative=False)
    cut = min(range(len(base)), key=lambda j: abs(base[j]["t_min"] - 18.0))
    t_s = base[cut]["t_min"] * 60.0
    add = E.make_addition_at(t_s, 48.0, E.ADDITION_LIBRARY["Lime (92% CaO)"])

    full, _, _ = E.simulate_frames_live(
        cfg, 12.0, comp, 5200.0, additions=[add], dt=2.0,
        t_end_min=36.0, cooperative=False)
    future, _, _ = E.simulate_frames_live(
        cfg, 12.0, comp, 5200.0, additions=[add], dt=2.0,
        t_end_min=36.0, from_state=states[cut],
        from_pool=copy.deepcopy(pools[cut]), t0_s=t_s, cooperative=False)
    combined = base[:cut + 1] + future

    assert len(combined) == len(full)
    for key in ("T_bath_C", "melted_pct", "pct_C", "pct_Si",
                "slag_total_kg", "undissolved_kg", "E_kWh"):
        assert max(abs(float(a[key]) - float(b[key])) for a, b in zip(full, combined)) < 1e-10
