# 🚀 QUICK START: How to Run the App

If you received this project as a ZIP file, follow these simple steps to start the application:

1. **Extract the ZIP file** to a folder on your computer.
2. Open the extracted folder (`Inductionfurnace-main`).
3. Double-click the **`run_web.bat`** file.
4. A command prompt will open, install any missing dependencies, and start the local server.
5. Your default web browser will automatically open the **SmartMelt Studio Web App** (at `http://localhost:8765`).

---

# SmartMelt Studio — Pixel-Matched Streamlit Release

For the exact `run_gui.py` browser replica, start with **`README_STREAMLIT_EXACT.md`**. The current release includes the corrected dynamic liquid/slag furnace view and non-blocking live operation.

# SmartMelt — hybrid physics + ML melt model

Reference implementation of the SmartMelt mathematical model (Extractmet Pvt. Ltd.):
a first-principles induction-furnace / EAF / BOF heat model with element-and-energy
conservation audited at every step, residual machine learning on top of the physics,
an EKF virtual sensor, charge-mix LP, receding-horizon power advice, and a bilingual
advisory layer. Pure Python (numpy / scipy / scikit-learn / pandas), edge-deployable.

**Read first:** `docs/SmartMelt_Mathematical_Model.md` — every equation (E0–E62) is
numbered there and appears as a comment at the matching line of code.

## Layout

    smartmelt/
      config.py      every plant-specific number: geometry, refractory layer stack,
                     electrical, sensors, kinetics, slag, economics  (YAML <-> dataclass)
      thermo.py      Wagner activities, equilibria, partitions, energy floor (E6-E20)
      physics.py     the Layer-1 model: mass/energy balances, dissolution kinetics,
                     multi-layer wall, oxygen ledger, audits E61/E62
      ekf.py         Layer-3 joint state+parameter estimator (E31-E33)
      ml.py          Layer-2 residual GP / quantile GBM / hybrid fusion / tramp
                     soft-sensor / PSI drift monitor (E0, E34)
      chargemix.py   least-cost charge LP with tramp walls + shadow prices (E40-E45)
      mpc.py         receding-horizon power/O2 advice on the physics (E50-E53)
      advisory.py    bilingual traffic-light recommendations with uncertainty gating
      simulator.py   virtual plant with realistic corruptions, for rehearsal + ML data
      metrics.py     honest accuracy: time-ordered splits, hit rates, savings attribution
      calibrate.py   staged per-plant fits + identifiability checks + standing-loss test
    configs/         if_msme_12t.yaml (Industry-X, SmartMelt Lite), eaf_50t.yaml (Pro) + generator
    tests/           conservation + contract smoke tests (run in CI)
    examples/        run_demo.py end-to-end walkthrough; generated heat datasets;
                     process_trajectory.png + demo plots;
                     operator_console_v3.html  <- THE operator GUI (single canonical copy)
    deck/            SmartMelt_Deck_v2_IndustryX.pptx + rendered PDF
    docs/            the mathematical model document

## Quickstart

    pip install numpy scipy scikit-learn pandas pyyaml matplotlib pyarrow
    python tests/test_smoke.py           # conservation + contract checks
    python examples/run_demo.py          # full end-to-end demo, writes PNGs + report

Minimal use:

```python
from smartmelt import load_config, FurnaceModel, HeatInputs
from smartmelt.physics import make_addition

cfg = load_config("configs/if_msme_12t.yaml")
m   = FurnaceModel(cfg)
x0  = m.initial_state(12000, {"C":0.0035,"Si":0.0022,"Mn":0.0035,
                              "P":0.00035,"S":0.0003,"Cu":0.002},
                      hot_heel_kg=960)
u   = HeatInputs(power_kW=lambda t: 5520.0,
                 oxygen_Nm3_per_h=lambda t: 0.0,
                 additions=[make_addition(600, 48, {"CaO":0.92,"SiO2":0.04},
                                          "lime", into="slag")])
traj = m.simulate(x0, u, 7200)
print(m.endpoint(traj))
print(m.element_balance(traj, u, 12000, {...}))   # E61 ledger
print(m.energy_closure(traj))                      # E62 first law
```

## Customising to a new client

Copy a reference YAML and edit top-down — no code changes:

1. `plant` — furnace type (IF/EAF/BOF), heat size, tap temperature, grade aims.
2. `geometry` — inner diameter, bath height, freeboard, lid coverage (from the GA
   drawing). Heat-transfer areas are derived (E5a).
3. `lining.layers` — the real refractory stack, one entry per layer with vendor
   k / rho / cp; `outer_bc: coil` (IF water jacket) or `shell` (EAF, convection+radiation).
4. `electrical` — rated power, converter efficiency, the operator's discrete tap levels.
5. `sensors` — what the shop actually has. This alone rebuilds the EKF observation
   model and is the Lite/Pro SKU switch.
6. `economics` — tariff, capex, opex, baseline SEC for payback reporting.

Then follow the commissioning order in the maths doc §16–17: standing-loss test,
staged calibration with identifiability checks, physics-only advisory first, ML
heads unlock automatically as logged heats accumulate (maturity tiers).

## Plots and process trajectory

`examples/process_trajectory.png` is the six-panel reference-heat trajectory
(temperatures, inventories with undissolved additions, bath composition, heat
flows, slag FeO/B2, energy and SEC against the 388 kWh/t floor) regenerated by
`run_demo.py`; `demo_1_heat.png` and `demo_2_ekf.png` show the demo heat and the
EKF tracking a mismatched plant. All plant references are anonymised: the MSME
IF pilot is Industry-X, the integrated BOF validation plant is Industry-Y.

## Which file is the GUI?

`examples/operator_console_v3.html` — open it in any browser, no install, works offline.
It is the only console in the package (the older `gui/` folder was removed in v0.4 to
stop two versions circulating). v3 carries: the TRAJECTORY screen with six charts,
the event log, and the (FeO)+[C] → Fe+CO chemistry (E27c). Verify your copy with:

    grep -c "E27c" examples/operator_console_v3.html      # -> non-zero
    cat BUILD_MANIFEST.md                                 # checksums + feature list

## Verified parameters and their sources (v0.5)

Every physical constant below was checked against the literature in a dedicated
verification pass; four were wrong in earlier revisions and are now corrected.

| Quantity | Value | Basis |
|---|---|---|
| Latent heat of fusion, Fe | **247 kJ/kg** (was 272) | 13.81 kJ/mol; CRC Handbook 104th ed. |
| (FeO)+[C] → Fe+CO | **+100 kJ/mol CO = +1.39 MJ/kg FeO** (was 1.89) | Turkdogan; Fruehan, *MSTS* 11th ed. |
| FeSi75 heat of solution | **−3 511 kJ/kg alloy** (was −1 150) | Sigworth & Elliott 1974; net −0.72 MJ/kg vs Bernhard et al. 2025 |
| Carburiser heat of solution | **+1 883 kJ/kg C** (was +2 500) | C(gr)=[C], ΔH +22.6 kJ/mol; total cold load ≈ +4.6 MJ/kg |
| Grid emission factor | **0.712 tCO₂/MWh** (was 0.82) | CEA CO₂ Baseline Database v21.0, FY2024-25 |
| Reversible melting minimum | **381 kWh/t** | E20 with L_f = 247; practical IF floor ≈ 500 kWh/t |
| Default tariff | **₹7.0/kWh** (was ₹8.0) | Indian HT industrial FY2025-26; ₹6.0–8.5 grid, ₹5.0–6.5 open access |
| Baseline SEC | **615 kWh/t** | Scrap-based Indian IF 550–650; DRI-heavy 650–800 |

Two constants are **coupled and must not be tuned independently**: `C_to_CO`
(11 100 kJ/kg C) and `Fe_to_FeO` (4 170 kJ/kg Fe). Their *difference* is the
enthalpy of (FeO)+[C] → Fe+CO, fixed by thermochemistry at ≈ +100 kJ/mol CO.
`test_reaction_enthalpies_hess_consistent` enforces this.

Dissolution presets follow a strict convention: `dH_dissolution_kJ_kg` is the
**intrinsic heat of solution only**, and the sensible heat of the cold addition
is a separate term using a per-addition effective cp. Conflating them was the
root cause of the previous FeSi75 and carburiser errors.

## Guarantees enforced by tests

Element ledger closes < 1 % per element per heat (E61); first-law closure within
the documented ~2–3 % bound (E62); endpoint insensitive to time step 1–5 s;
dissolution stalls without superheat (E7b); the Cu wall binds in the charge LP;
and there is no PLC write path anywhere in the package (Phase-1 advisory-only is
structural).
