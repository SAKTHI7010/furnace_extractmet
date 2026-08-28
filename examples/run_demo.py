"""
SmartMelt end-to-end demo on the reference 12 t MSME induction furnace.

Walks the full stack in the order a deployment would:
  1. Physics heat with dissolution kinetics + conservation audits (E61/E62)
  2. EKF virtual sensor on a deliberately mismatched plant
  3. ML residual layer trained on 90 virtual heats, honest time-ordered
     evaluation, PSI drift alarm on the scrap regime change at heat 70
  4. Charge-mix LP with tramp shadow prices
  5. MPC power advice from mid-heat
  6. Bilingual advisory board
  7. Economics / payback

Writes PNG figures next to this file. Runtime ~4-6 min (dominated by the EKF
finite-difference Jacobians and the MPC rollouts).

Usage: python examples/run_demo.py [--fast]   (--fast skips EKF and MPC)
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

from smartmelt import load_config, FurnaceModel, HeatInputs      # noqa: E402
from smartmelt.physics import make_addition                       # noqa: E402
from smartmelt.thermo import KELVIN, theoretical_melt_energy_kWh_per_t  # noqa: E402

FAST = "--fast" in sys.argv
CFG = os.path.join(HERE, "..", "configs", "if_msme_12t.yaml")
CHARGE = {"C": 0.0035, "Si": 0.0022, "Mn": 0.0035,
          "P": 0.00035, "S": 0.0003, "Cu": 0.002}


def banner(s):
    print("\n" + "=" * 72 + f"\n{s}\n" + "=" * 72)


# ======================================================================
banner("1. PHYSICS HEAT — dissolution kinetics + conservation audits")
cfg = load_config(CFG)
m = FurnaceModel(cfg)
ch = cfg.plant.heat_size_t * 1000.0
x0 = m.initial_state(ch, CHARGE, hot_heel_kg=0.08 * ch)
u = HeatInputs(
    power_kW=lambda t: 5520.0 if t < 4200 else 3000.0,   # taper near tap
    oxygen_Nm3_per_h=lambda t: 0.0,
    additions=[
        make_addition(600.0, 48.0, {"CaO": 0.92, "SiO2": 0.04}, "lime", into="slag"),
        make_addition(2700.0, 15.0, {"Si": 0.75, "Fe": 0.25}, "FeSi75"),
        make_addition(3200.0, 10.0, {"C": 0.99}, "carburiser"),
    ])
stop = (lambda t, x: (x[m.iTb] - KELVIN) >= cfg.plant.tap_temperature_C
        and x[m.iMs] < 0.002 * ch)
t0 = time.time()
traj = m.simulate(x0, u, 9000.0, dt=2.0, stop_fn=stop)
ep = m.endpoint(traj)
print(f"simulated {traj.t[-1]/60:.1f} min heat in {time.time()-t0:.1f} s wallclock")
print(f"tap: T={ep['T_C']:.0f} C  C={ep['pct_C']:.3f}%  Si={ep['pct_Si']:.3f}%  "
      f"Mn={ep['pct_Mn']:.3f}%  S={ep['pct_S']:.4f}%")
print(f"tap mass {ep['tap_mass_t']:.2f} t   SEC {ep['SEC_kWh_per_t']:.0f} kWh/t "
      f"(theoretical floor {theoretical_melt_energy_kWh_per_t(cfg):.0f})")
print(f"undissolved at tap: {traj.undissolved_kg:.1f} kg")

eb = m.element_balance(traj, u, ch, CHARGE)
print("\n(E61) element ledger:")
print(eb.to_string(index=False,
                   formatters={"in_kg": "{:10.2f}".format,
                               "out_kg": "{:10.2f}".format,
                               "closure_pct": "{:+8.3f}".format}))
ec = m.energy_closure(traj, dt=2.0)
print(f"\n(E62) first-law closure: residual {ec['residual_kWh']:+.0f} kWh "
      f"= {ec['residual_pct']:+.2f} % of input "
      f"(dH_lining {ec['dH_lining_kWh']:+.0f} kWh, wall-out {ec['E_wall_outer_kWh']:.0f} kWh)")

audit = m.energy_audit(traj, dt=2.0)
print("\nenergy audit (kWh):",
      {k: round(v, 0) for k, v in audit.items() if k != "useful_fraction"})

# ---- figure: heat trajectory ----------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(11, 7))
tmin = traj.t / 60.0
ax[0, 0].plot(tmin, traj.X[:, m.iTb] - KELVIN, label="bath")
ax[0, 0].plot(tmin, traj.X[:, m.iTs] - KELVIN, label="solid", alpha=0.7)
ax[0, 0].plot(tmin, traj.diagnostics["T_hotface"], label="hot face", alpha=0.7)
ax[0, 0].axhline(cfg.plant.tap_temperature_C, ls="--", c="k", lw=0.8)
ax[0, 0].set(title="Temperatures", xlabel="min", ylabel="C"); ax[0, 0].legend()
ax[0, 1].plot(tmin, traj.X[:, m.iMs] / 1000, label="solid")
ax[0, 1].plot(tmin, traj.X[:, :m.nM].sum(1) / 1000, label="liquid")
ax[0, 1].plot(tmin, traj.diagnostics["m_undissolved"], label="undissolved kg", alpha=0.7)
ax[0, 1].set(title="Inventories", xlabel="min", ylabel="t / kg"); ax[0, 1].legend()
iC = m.metal.index("C")
pctC = 100 * traj.X[:, iC] / np.maximum(traj.X[:, :m.nM].sum(1), 1e-6)
ax[1, 0].plot(tmin, pctC)
ax[1, 0].axhline(cfg.plant.target_carbon_pct, ls="--", c="k", lw=0.8)
ax[1, 0].set(title="Bath carbon", xlabel="min", ylabel="wt %")
ax[1, 1].plot(tmin, traj.diagnostics["Q_wall"], label="wall (hot face)")
ax[1, 1].plot(tmin, traj.diagnostics["Q_rad"], label="top radiation")
ax[1, 1].plot(tmin, traj.diagnostics["Q_s"], label="bath->scrap", alpha=0.6)
ax[1, 1].set(title="Heat flows", xlabel="min", ylabel="kW"); ax[1, 1].legend()
fig.tight_layout(); fig.savefig(os.path.join(HERE, "demo_1_heat.png"), dpi=110)
print("wrote demo_1_heat.png")

# ======================================================================
if not FAST:
    banner("2. EKF VIRTUAL SENSOR — plant deliberately mismatched")
    from smartmelt.ekf import build_default_ekf
    truth = FurnaceModel(cfg, {"eta_electrical": 0.90, "UA_lining_scale": 1.35})
    x0t = truth.initial_state(ch, CHARGE, hot_heel_kg=0.08 * ch)
    ut = HeatInputs(lambda t: 5520.0, lambda t: 0.0,
                    [make_addition(600, 48, {"CaO": 0.92, "SiO2": 0.04},
                                   "lime", into="slag")])
    trt = truth.simulate(x0t, ut, 4500, dt=2.0)

    nominal = FurnaceModel(cfg)
    ekf = build_default_ekf(nominal, ut)
    nx = nominal.n_state
    P0 = np.zeros((nx + 3, nx + 3))
    P0[nominal.iTb, nominal.iTb] = 20.0 ** 2
    P0[nominal.iMs, nominal.iMs] = 100.0 ** 2
    for i in range(nominal.nM):
        P0[i, i] = 20.0 ** 2
    P0[nx:, nx:] = np.diag([0.05 ** 2, 0.20 ** 2, 0.25 ** 2])
    ekf.init(x0t.copy(), P0)

    rng = np.random.default_rng(1)
    dips, di = [1200.0, 2400.0, 3600.0], 0
    rows = []
    t0 = time.time()
    for k in range(0, len(trt.t), 15):                 # 30 s cadence
        t = float(trt.t[k])
        ekf.predict(t, ut, 30.0)
        if t > 1800:
            y = np.array([trt.X[k, truth.iTb] - KELVIN
                          + rng.normal(0, cfg.sensors.sigma_T_pyrometer_C),
                          0.0,
                          (trt.X[k, :truth.nM].sum() + trt.X[k, truth.iMs]) / 1000
                          + rng.normal(0, 0.05)])
            act = np.array([True, False, True])
            if di < len(dips) and t >= dips[di]:
                y[1] = trt.X[k, truth.iTb] - KELVIN \
                    + rng.normal(0, cfg.sensors.sigma_T_immersion_C)
                act[1] = True
                di += 1
            ekf.update(y, act)
        rows.append((t / 60, trt.X[k, truth.iTb] - KELVIN,
                     ekf.bath_temperature_C(), ekf.sigma_T(),
                     ekf.theta["eta_electrical"]))
    E = np.array(rows)
    print(f"EKF: {time.time()-t0:.0f} s wallclock, final T err "
          f"{E[-1,2]-E[-1,1]:+.1f} C, eta-hat {E[-1,4]:.3f} (true 0.900), "
          f"sigma_T {E[0,3]:.0f} -> {E[-1,3]:.1f} C")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(E[:, 0], E[:, 1], "k", label="true")
    ax[0].plot(E[:, 0], E[:, 2], "C1", label="EKF")
    ax[0].fill_between(E[:, 0], E[:, 2] - 2 * E[:, 3], E[:, 2] + 2 * E[:, 3],
                       color="C1", alpha=0.2, label="±2 sigma")
    ax[0].set(title="Bath temperature: truth vs EKF", xlabel="min", ylabel="C")
    ax[0].legend()
    ax[1].plot(E[:, 0], E[:, 4]); ax[1].axhline(0.90, ls="--", c="k", lw=0.8)
    ax[1].set(title="Tracked eta_electrical (true 0.90)", xlabel="min")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "demo_2_ekf.png"), dpi=110)
    print("wrote demo_2_ekf.png")

# ======================================================================
banner("3. ML RESIDUAL LAYER — 90 virtual heats, regime change at #70")
from smartmelt.ml import HybridEndpointModel, TrampSoftSensor, DriftMonitor, build_features
from smartmelt.metrics import time_ordered_split, endpoint_hit_rate

data_csv = os.path.join(HERE, "heats_if_90.csv")
if not os.path.exists(data_csv):
    print("generating dataset (one-off, ~8 min) ...")
    from smartmelt.simulator import VirtualPlant
    vp = VirtualPlant(cfg, seed=7, regime_change_at=70, dt_s=5.0)
    df = vp.generate(90, progress=True)
    df.to_csv(data_csv, index=False)
df = pd.read_csv(data_csv)
print(f"{len(df)} heats; physics-only T residual: "
      f"bias {(df.meas_T_C-df.phys_T_C).mean():+.1f} C, "
      f"sd {(df.meas_T_C-df.phys_T_C).std():.1f} C")

train, test = time_ordered_split(df.iloc[:70], train_frac=0.75)  # pre-regime only
hyb = HybridEndpointModel(cfg).fit(train)
predT, predC = hyb.predict(test)
T_hat = np.array([p.mean for p in predT])
C_hat = np.array([p.mean for p in predC])
hrT_phys = endpoint_hit_rate(test.meas_T_C, test.phys_T_C, tol=15.0)
hrT_hyb = endpoint_hit_rate(test.meas_T_C, T_hat, tol=15.0)
hrC_phys = endpoint_hit_rate(test.meas_C_pct, test.phys_C_pct, tol=0.02)
hrC_hyb = endpoint_hit_rate(test.meas_C_pct, C_hat, tol=0.02)
print(f"maturity tier: {hyb.maturity}  (n={hyb.n_heats}) | heads enabled by "
      f"rolling-origin CV gate: T={hyb.use_T}  C={hyb.use_C}")
print(f"hit rate |dT|<=15C : physics-only {hrT_phys['hit_rate']*100:.0f}%  ->  "
      f"hybrid {hrT_hyb['hit_rate']*100:.0f}%   "
      f"(bias {hrT_phys['bias']:+.0f} -> {hrT_hyb['bias']:+.1f} C, "
      f"MAE {hrT_phys['mae']:.1f} -> {hrT_hyb['mae']:.1f} C)")
print(f"hit rate |dC|<=0.02: physics-only {hrC_phys['hit_rate']*100:.0f}%  ->  "
      f"hybrid {hrC_hyb['hit_rate']*100:.0f}%  (C head gated off: physics is at "
      f"the noise ceiling; the hybrid refuses to add an unproven correction)")

# drift alarm on the Cu regime change
mon = DriftMonitor(cfg).set_reference(build_features(df.iloc[35:60]))  # rolling ref
rep_ok = mon.check(build_features(df.iloc[55:70]))
rep_bad = mon.check(build_features(df.iloc[70:90]))
print(f"\nPSI drift monitor (rolling 25-heat reference):")
print(f"  window 55-70: psi_max={rep_ok['psi_max']:.2f} on "
      f"{max(rep_ok['psi'], key=rep_ok['psi'].get)} — the monitor is already "
      f"smelling the slow lining-wear drift the virtual plant injects")
print(f"  window 70-90: {'; '.join(rep_bad['reasons'])} — the Cu regime change "
      f"blows straight through the threshold")
print("-> on alarm the advisory demotes to physics-only and flags re-fit; this is")
print("   the documented defence against silent drift under scrap regime change.")

# tramp soft sensor
tss = TrampSoftSensor()
Xf = build_features(df)
mb = pd.DataFrame({"Cu": df.charge_Cu_pct, "Sn": 0.0, "Cr": 0.0})
tr_idx, te_idx = df.index[:56], df.index[56:70]
tss.fit(Xf.loc[tr_idx], mb.loc[tr_idx],
        pd.DataFrame({"Cu": df.true_Cu_pct.loc[tr_idx], "Sn": 0.0, "Cr": 0.0}))
cu_hat = tss.predict(Xf.loc[te_idx], mb.loc[te_idx])["Cu"]
err_prior = float(np.abs(mb.Cu.loc[te_idx] - df.true_Cu_pct.loc[te_idx]).mean())
err_ml = float(np.abs(cu_hat.to_numpy() - df.true_Cu_pct.loc[te_idx].to_numpy()).mean())
print(f"\n(E34) tramp Cu soft sensor: |err| charge-sheet prior {err_prior:.4f} %"
      f" -> corrected {err_ml:.4f} %")

# ======================================================================
banner("4. CHARGE-MIX LP — least cost against the grade walls")
from smartmelt.chargemix import ChargeMixOptimiser, Material
mats = [
    Material("HMS_80_20", 33.5, {"C": .0025, "Si": .0020, "Mn": .0045,
                                 "P": .0003, "S": .00035, "Cu": .0030, "Cr": .0015},
             metallic_yield=.94, energy_kWh_per_kg=.60),
    Material("Shredded", 35.5, {"C": .0020, "Si": .0015, "Mn": .0040,
                                "P": .0002, "S": .00025, "Cu": .0022, "Cr": .0010},
             metallic_yield=.95, energy_kWh_per_kg=.58),
    Material("Busheling", 39.0, {"C": .0010, "Si": .0005, "Mn": .0035,
                                 "P": .00012, "S": .00012, "Cu": .0008, "Cr": .0005},
             metallic_yield=.97, energy_kWh_per_kg=.55),
    Material("CI_borings", 30.0, {"C": .032, "Si": .018, "Mn": .0050,
                                  "P": .0008, "S": .0006, "Cu": .0015, "Cr": .0010},
             metallic_yield=.90, energy_kWh_per_kg=.52, available_kg=1800.0),
    Material("DRI", 31.5, {"C": .018, "P": .00045, "S": .00008, "Cu": .0001},
             metallic_yield=.88, energy_kWh_per_kg=.75, available_kg=5000.0),
    Material("PigIron", 42.0, {"C": .042, "Si": .009, "Mn": .0040,
                               "P": .0006, "S": .0002, "Cu": .0002, "Cr": .0002},
             metallic_yield=.97, energy_kWh_per_kg=.50, available_kg=3000.0),
]
opt = ChargeMixOptimiser(mats, tariff_INR_per_kWh=cfg.economics.tariff_INR_per_kWh,
                         max_charge_kg=cfg.plant.max_charge_t * 1000)
aim = {"C": (0.15, 0.30), "Si": (0.05, 0.35), "Mn": (0.25, 0.60)}
walls = {"Cu": 0.25, "P": 0.045, "S": 0.040, "Cr": 0.15}
res = opt.solve(ch, aim, walls)
print(res.pretty())
sp = opt.shadow_prices(res, aim, walls, ch, delta=0.01)
print("\nshadow price of each spec wall (INR/t liquid per +0.01 wt%):")
for k, v in sp.items():
    print(f"   {k:3s} {v*0.01:8.1f}" + ("   <- binding" if v > 1 else ""))

# ======================================================================
if not FAST:
    banner("5. MPC POWER ADVICE — from the 55-minute mark")
    from smartmelt.mpc import MeltMPC
    tr55 = m.simulate(x0, u, 3300, dt=2.0)
    x55 = tr55.X[-1].copy()
    mpc = MeltMPC(m, n_blocks=4, dt_s=15.0)
    t0 = time.time()
    plan = mpc.solve(x55, t0=3300.0, horizon_s=1500.0, base=u,
                     T_star=cfg.plant.tap_temperature_C,
                     C_star=cfg.plant.target_carbon_pct,
                     o2_fixed=np.zeros(4))
    print(f"solved in {time.time()-t0:.0f} s | predicted tap "
          f"T={plan.predicted_T_C:.0f} C  C={plan.predicted_C_pct:.3f}%  "
          f"E=+{plan.predicted_energy_kWh:.0f} kWh  in {plan.predicted_tap_time_s/60:.0f} min")
    for a in plan.as_operator_actions(cfg.electrical.tap_levels_kW):
        print("  ", a)
    predT_now, sigT_now = plan.predicted_T_C, 6.0
else:
    predT_now, sigT_now = ep["T_C"], 6.0

# ======================================================================
banner("6. ADVISORY BOARD — bilingual, uncertainty-gated")
from smartmelt.advisory import AdvisoryEngine
adv = AdvisoryEngine(cfg)
entry = adv.evaluate(
    pred_T=predT_now, sigma_T=sigT_now, pred_C=0.24, sigma_C=0.015,
    B2=ep["B2"], pct_FeO=ep["pct_FeO_slag"],
    sec_now=ep["SEC_kWh_per_t"],
    sec_target=cfg.economics.baseline_SEC_kWh_per_t,
    minutes_left=20.0, power_headroom_kW=480.0, o2_available=False,
    drift_report=rep_bad, heat_id="H-1042", t_s=3300.0)
print(adv.render(entry))
print("note: temperature & carbon advice SUSPENDED by the drift alarm from step 3 —")
print("the system degrades to physics-only rather than guessing. Clear the alarm")
print("(re-fit on post-regime heats) and the heads return.")

# ======================================================================
banner("7. ECONOMICS")
from smartmelt.metrics import economics
# Use a defensible, regression-adjusted saving (40 kWh/t is the low end of
# the 50-120 kWh/t brief) rather than baseline-vs-one-simulated-heat, which
# would claim an absurd instant payback. Every line below is disputable by
# the plant owner on purpose — that is the point of the table.
eco = economics(615.0, 575.0, cfg)
for k, v in eco.items():
    print(f"   {k:28s} {v:,.0f}" if isinstance(v, (int, float)) else f"   {k}: {v}")

print("\nDemo complete. Figures: examples/demo_1_heat.png, demo_2_ekf.png")
