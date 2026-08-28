"""
Smoke tests: every assertion here is a *physics or contract guarantee*, not a
style check. If one fails after an edit, the edit broke a conservation law, a
config contract, or a module API that other layers rely on.

Run:  python -m pytest tests/ -q     (or just: python tests/test_smoke.py)
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smartmelt import load_config, FurnaceModel, HeatInputs          # noqa: E402
from smartmelt.config import PlantConfig                             # noqa: E402
from smartmelt.physics import make_addition                          # noqa: E402
from smartmelt.thermo import (KELVIN, theoretical_melt_energy_kWh_per_t)  # noqa: E402

HERE = os.path.dirname(__file__)
IF_YAML = os.path.join(HERE, "..", "configs", "if_msme_12t.yaml")
EAF_YAML = os.path.join(HERE, "..", "configs", "eaf_50t.yaml")

CHARGE = {"C": 0.0035, "Si": 0.0022, "Mn": 0.0035,
          "P": 0.00035, "S": 0.0003, "Cu": 0.002}


def _run(yaml_path, o2_Nm3h=0.0, dt=2.0):
    cfg = load_config(yaml_path)
    m = FurnaceModel(cfg)
    ch = cfg.plant.heat_size_t * 1000.0
    x0 = m.initial_state(ch, CHARGE, hot_heel_kg=0.08 * ch)
    o2 = (lambda t: 0.0) if o2_Nm3h == 0 else (lambda t: o2_Nm3h if t > 900 else 0.0)
    u = HeatInputs(lambda t: 0.92 * cfg.electrical.rated_power_kW, o2,
                   [make_addition(600, 0.004 * ch, {"CaO": 0.92, "SiO2": 0.04},
                                  "lime", into="slag"),
                    make_addition(2400, 0.0012 * ch, {"Si": 0.75, "Fe": 0.25},
                                  "FeSi75")])
    stop = (lambda t, x: (x[m.iTb] - KELVIN) >= cfg.plant.tap_temperature_C
            and x[m.iMs] < 0.002 * ch)
    traj = m.simulate(x0, u, 9000.0, dt=dt, stop_fn=stop)
    return cfg, m, u, traj, ch


def test_config_yaml_roundtrip():
    cfg = load_config(IF_YAML)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        cfg.save(fh.name)
        cfg2 = load_config(fh.name)
    assert cfg2.plant.heat_size_t == cfg.plant.heat_size_t
    assert len(cfg2.lining.layers) == len(cfg.lining.layers)
    assert cfg2.lining.layers[0].k_W_mK == cfg.lining.layers[0].k_W_mK
    os.unlink(fh.name)


def test_geometry_derives_areas():
    cfg = load_config(IF_YAML)
    m = FurnaceModel(cfg)
    g = cfg.geometry
    assert abs(m.A_wall - np.pi * g.D_inner_m * g.H_bath_m) < 1e-6
    assert m.A_top > 0


def test_theoretical_floor_sane():
    """Reversible minimum only. Literature band 360-400 kWh/t with L_f=247 kJ/kg;
    the ACHIEVABLE practical floor for a real IF is ~500 kWh/t (see README)."""
    cfg = load_config(IF_YAML)
    e = theoretical_melt_energy_kWh_per_t(cfg)
    assert 360.0 < e < 400.0, e


def test_if_heat_conservation_and_endpoint():
    cfg, m, u, traj, ch = _run(IF_YAML)
    ep = m.endpoint(traj)
    # sane endpoint
    assert 1600.0 < ep["T_C"] < 1800.0
    assert 0.05 < ep["pct_C"] < 0.6
    assert 10.0 < ep["tap_mass_t"] < 13.0
    assert 450.0 < ep["SEC_kWh_per_t"] < 750.0
    # (E61) element ledger closes
    eb = m.element_balance(traj, u, ch, CHARGE)
    assert abs(eb.closure_pct).max() < 1.0, eb.to_string()
    # (E62) first-law closure within documented bound
    ec = m.energy_closure(traj, dt=2.0)
    assert abs(ec["residual_pct"]) < 4.0, ec
    # hot face below the working-lining service limit
    assert ep["hot_face_C"] < cfg.lining.layers[0].T_limit_C + 5.0


def test_eaf_heat_conservation():
    cfg, m, u, traj, ch = _run(EAF_YAML, o2_Nm3h=1800.0)
    eb = m.element_balance(traj, u, ch, CHARGE)
    assert abs(eb.closure_pct).max() < 1.0
    ec = m.energy_closure(traj, dt=2.0)
    assert abs(ec["residual_pct"]) < 4.0


def test_dt_insensitivity():
    _, m1, _, t1, _ = _run(IF_YAML, dt=1.0)
    _, m5, _, t5, _ = _run(IF_YAML, dt=5.0)
    T1 = m1.endpoint(t1)["T_C"]
    T5 = m5.endpoint(t5)["T_C"]
    assert abs(T1 - T5) < 5.0


def test_dissolution_stalls_without_superheat():
    from smartmelt.physics import Addition
    cfg = load_config(IF_YAML)
    m = FurnaceModel(cfg)
    a = Addition(0.0, 100.0, {"C": 1.0}, tau_s=300.0, dT_ref_K=25.0)
    x = np.zeros(m.n_state)
    x[m.metal.index("Fe")] = 10000.0
    # hot bath: fast release
    x[m.iTb] = cfg.thermal.T_liquidus_C + 60.0 + KELVIN
    pool = [{"add": a, "m": 100.0}]
    m._release_dissolving(x.copy(), pool, 300.0)
    rel_hot = 100.0 - pool[0]["m"]
    # cold bath (2 K superheat): stalled
    x[m.iTb] = cfg.thermal.T_liquidus_C + 2.0 + KELVIN
    pool = [{"add": a, "m": 100.0}]
    m._release_dissolving(x.copy(), pool, 300.0)
    rel_cold = 100.0 - pool[0]["m"]
    assert rel_hot > 3.0 * rel_cold, (rel_hot, rel_cold)


def test_chargemix_feasible_and_cu_wall():
    from smartmelt.chargemix import ChargeMixOptimiser, Material
    mats = [
        Material("HMS", 33.5, {"C": .0025, "Si": .002, "Mn": .0045, "P": .0003,
                               "S": .00035, "Cu": .003}, metallic_yield=.94,
                 energy_kWh_per_kg=.60),
        Material("Busheling", 39.0, {"C": .001, "Si": .0005, "Mn": .0035,
                                     "P": .00012, "S": .00012, "Cu": .0008},
                 metallic_yield=.97, energy_kWh_per_kg=.55),
        Material("DRI", 31.5, {"C": .018, "P": .00045, "S": .00008, "Cu": .0001},
                 metallic_yield=.88, energy_kWh_per_kg=.75, available_kg=5000.0),
    ]
    opt = ChargeMixOptimiser(mats, tariff_INR_per_kWh=8.0, max_charge_kg=14000.0)
    res = opt.solve(12000.0, {"C": (0.15, 0.30)}, {"Cu": 0.25})
    assert res.feasible
    assert res.predicted_bath_pct["Cu"] <= 0.2501
    assert abs(res.liquid_t - 12.0) < 0.01


def test_ekf_builds_and_steps():
    from smartmelt.ekf import build_default_ekf
    cfg = load_config(IF_YAML)
    m = FurnaceModel(cfg)
    ch = 12000.0
    x0 = m.initial_state(ch, CHARGE, hot_heel_kg=0.08 * ch)
    u = HeatInputs(lambda t: 5000.0, lambda t: 0.0)
    ekf = build_default_ekf(m, u)
    nx = m.n_state
    P0 = np.eye(nx + 3) * 1.0
    P0[m.iTb, m.iTb] = 400.0
    ekf.init(x0, P0)
    ekf.predict(0.0, u, 10.0)
    y = np.array([1500.0, 0.0, 12.0])
    ekf.update(y, active=np.array([True, False, True]))
    assert np.isfinite(ekf.bath_temperature_C())
    assert 0.75 <= ekf.theta["eta_electrical"] <= 1.15


def test_no_plc_write_path():
    """Phase-1 advisory-only is structural: nothing imports a PLC/OPC library."""
    import pathlib
    root = pathlib.Path(HERE, "..", "smartmelt")
    banned = ("pycomm", "opcua", "opcda", "snap7", "pymodbus", "minimalmodbus")
    for f in root.glob("*.py"):
        text = f.read_text().lower()
        for b in banned:
            assert b not in text, f"{f.name} references {b}"




def test_reaction_enthalpies_hess_consistent():
    """(E27c) C_to_CO and Fe_to_FeO must differ by ~100 kJ/mol CO, since their
    difference IS the enthalpy of (FeO)+[C]->Fe+CO. Tuning either alone breaks
    the first law for every ore/mill-scale addition."""
    cfg = load_config(IF_YAML)
    en = cfg.enthalpy
    per_mol_C = en.C_to_CO * 0.012          # kJ/mol
    per_mol_Fe = en.Fe_to_FeO * 0.05585     # kJ/mol
    dH_reaction = per_mol_Fe - per_mol_C    # endothermic, kJ per mol CO
    assert 90.0 < dH_reaction < 110.0, f"(FeO)+[C] enthalpy = {dH_reaction:.1f} kJ/mol"
    per_kg_FeO = dH_reaction / 0.07185 / 1000.0
    assert 1.30 < per_kg_FeO < 1.50, f"{per_kg_FeO:.2f} MJ/kg FeO"


def test_mill_scale_decarburises_and_cools():
    """(E27c) (FeO)+[C] -> Fe+CO: mill scale must LOWER carbon, RAISE metallic
    Fe, and COOL the bath (net endothermic), with ledgers still closing."""
    cfg = load_config(IF_YAML)
    m = FurnaceModel(cfg)
    ch = 12000.0
    hiC = {**CHARGE, "C": 0.012}
    x0 = m.initial_state(ch, hiC, hot_heel_kg=0.08 * ch)
    lime = make_addition(600, 48, {"CaO": 0.92, "SiO2": 0.04}, "lime", into="slag")
    scale = make_addition(3300, 120, {"FeO": 0.97, "SiO2": 0.02},
                          "mill_scale", into="slag")
    out = {}
    for tag, adds in (("base", [lime]), ("scale", [lime, scale])):
        u = HeatInputs(lambda t: 5520.0, lambda t: 0.0, adds)
        tr = m.simulate(x0.copy(), u, 4800, dt=2.0)
        ep = m.endpoint(tr)
        eb = m.element_balance(tr, u, ch, hiC)
        out[tag] = (ep, tr.X[-1, m.metal.index("Fe")],
                    abs(eb.closure_pct).max())
    dC = out["scale"][0]["pct_C"] - out["base"][0]["pct_C"]
    dT = out["scale"][0]["T_C"] - out["base"][0]["T_C"]
    dFe = out["scale"][1] - out["base"][1]
    assert dC < -0.05, f"carbon not reduced by FeO: dC={dC:+.3f}"
    assert dC > -0.20, f"more C removed than stoichiometry allows: {dC:+.3f}"
    assert dFe > 50.0, f"Fe from FeO not credited to the bath: dFe={dFe:+.1f}"
    # Endothermic at the VERIFIED 1.39 MJ/kg FeO (=+100 kJ/mol CO). The old
    # 1.89 MJ/kg over-cooled by ~40 %; this window would catch that regression.
    assert -45.0 < dT < -18.0, f"FeO decarb cooling out of band: dT={dT:+.1f}"
    assert out["scale"][2] < 1.0, "element ledger broke under FeO reduction"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        print(f"{fn.__name__:45s}", end=" ", flush=True)
        fn()
        print("PASS")
    print(f"\n{len(fns)} tests passed.")
