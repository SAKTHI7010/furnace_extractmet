"""
simulator.py — the virtual plant.

You will not get 2,000 heats of clean IF data before you have to demo. So build
the plant you *think* you have, corrupt it in the ways real plants are corrupt,
and use it to (a) test the estimator, (b) test the drift monitor, (c) size the
data requirement, (d) rehearse the panel answers with real numbers.

Deliberate corruptions (each maps to a real failure mode in Sec. 9 of the brief):

  * theta_true != theta_nominal            -> plant-specific coil, lining
  * lining wear drifts over a campaign     -> non-stationary heat-loss term
  * scrap assay error, heat to heat        -> the bad-scrap-mix-data failure
  * operator power profile is sloppy       -> real control, not textbook
  * sensor noise + dropouts                -> pyrometer through slag/fume
  * a regime change at heat N              -> new scrap supplier, high Cu

Anything the ML head can learn from THIS simulator, it can learn from a plant.
Anything it cannot learn here, do not promise on a slide.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import PlantConfig
from .physics import FurnaceModel, HeatInputs, Addition
from .thermo import KELVIN


@dataclass
class HeatRecord:
    features: Dict[str, float]
    truth: Dict[str, float]
    trajectory: Optional[object] = None


class VirtualPlant:
    def __init__(self, cfg: PlantConfig, seed: int = 0,
                 regime_change_at: Optional[int] = None,
                 dt_s: float = 5.0):
        # dt=5 s for data generation: endpoint shift vs dt=1 s is <1 C and
        # <1 kWh/t (verified), and it makes 100-heat campaigns tractable.
        self.dt_s = dt_s
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.regime_change_at = regime_change_at
        self.heat_no = 0
        self.lining_age = 0

        # the plant's true (unknown to SmartMelt) parameters
        self.theta_true = dict(
            eta_electrical=float(self.rng.normal(0.96, 0.02)),
            UA_lining_scale=float(self.rng.normal(1.10, 0.08)),
            k_C_scale=float(self.rng.normal(1.00, 0.12)),
            h_solid_liquid_scale=float(self.rng.normal(1.00, 0.10)),
            gamma_FeO=float(self.rng.normal(cfg.slag.gamma_FeO, 0.12)),
        )

    # ------------------------------------------------------------------
    def _theta_now(self) -> Dict[str, float]:
        t = dict(self.theta_true)
        # lining wears through a campaign: heat loss rises ~25 % over 400 heats
        t["UA_lining_scale"] *= 1.0 + 0.25 * min(self.lining_age / 400.0, 1.0)
        return t

    def _charge(self) -> Dict[str, float]:
        r = self.rng
        high_cu = (self.regime_change_at is not None
                   and self.heat_no >= self.regime_change_at)
        comp = dict(
            C=float(np.clip(r.normal(0.35, 0.09), 0.05, 0.9)),
            Si=float(np.clip(r.normal(0.22, 0.06), 0.02, 0.6)),
            Mn=float(np.clip(r.normal(0.35, 0.08), 0.05, 0.9)),
            P=float(np.clip(r.normal(0.035, 0.010), 0.005, 0.09)),
            S=float(np.clip(r.normal(0.030, 0.008), 0.005, 0.07)),
            Cu=float(np.clip(r.normal(0.45 if high_cu else 0.20, 0.06), 0.02, 0.9)),
            Cr=float(np.clip(r.normal(0.08, 0.03), 0.0, 0.3)),
            Ni=float(np.clip(r.normal(0.06, 0.02), 0.0, 0.2)),
        )
        return {k: v / 100.0 for k, v in comp.items()}     # wt% -> fraction

    def _power_profile(self, P_rated: float):
        """Operator practice: full power, then a sloppy taper near the end."""
        r = self.rng
        t_full = r.uniform(1500, 2400)
        p_hi = P_rated * r.uniform(0.90, 1.0)
        p_lo = P_rated * r.uniform(0.35, 0.60)

        def f(t):
            if t < t_full:
                return p_hi * (1.0 + 0.02 * np.sin(t / 180.0))
            return p_lo
        return f

    # ------------------------------------------------------------------
    def run_heat(self, keep_trajectory=False) -> HeatRecord:
        cfg = self.cfg
        r = self.rng
        self.heat_no += 1
        self.lining_age += 1
        if self.lining_age > 450:                       # relining
            self.lining_age = 0

        charge_t = cfg.plant.heat_size_t * r.uniform(0.95, 1.03)
        charge_kg = charge_t * 1000.0
        comp_true = self._charge()
        heel = charge_kg * r.uniform(0.02, 0.08)

        # --- SmartMelt's *belief* about the charge: assay error -----------
        comp_believed = {k: v * (1.0 + r.normal(0, 0.12)) for k, v in comp_true.items()}

        true_model = FurnaceModel(cfg, self._theta_now())
        x0 = true_model.initial_state(charge_kg, comp_true, hot_heel_kg=heel,
                                      T_lining_C=cfg.thermal.T_ambient_C + r.uniform(150, 450))

        P = self._power_profile(cfg.electrical.rated_power_kW)
        o2 = (lambda t: 0.0) if cfg.plant.furnace_type == "IF" else \
             (lambda t: 900.0 if t > 900 else 0.0)

        adds = [Addition(600.0, charge_kg * 0.010, {"CaO": 0.92, "SiO2": 0.04},
                         into="slag", label="lime")]
        if r.random() < 0.5:
            adds.append(Addition(1800.0, charge_kg * r.uniform(0.0005, 0.0015),
                                 {"Si": 0.70, "Fe": 0.30}, into="metal", label="FeSi"))

        u = HeatInputs(P, o2, adds)
        T_tap = cfg.plant.tap_temperature_C + r.normal(0, 6)

        def stop(t, x):
            return x[true_model.iTb] - KELVIN >= T_tap and x[true_model.iMs] < 1.0

        traj = true_model.simulate(x0, u, cfg.numerics.max_heat_minutes * 60,
                                   dt=self.dt_s, stop_fn=stop)
        ep = true_model.endpoint(traj)

        # --- sensor noise -------------------------------------------------
        sen = cfg.sensors
        meas_T = ep["T_C"] + r.normal(0, sen.sigma_T_immersion_C)
        meas_C = ep["pct_C"] + r.normal(0, 0.004)
        meas_E = ep["energy_kWh"] * (1 + r.normal(0, 0.005))

        feats = dict(
            heat_no=self.heat_no,
            charge_mass_t=charge_t,
            scrap_frac=1.0, dri_frac=0.0, pig_frac=0.0, returns_frac=0.0,
            charge_C_pct=100 * comp_believed["C"],
            charge_Si_pct=100 * comp_believed["Si"],
            charge_Mn_pct=100 * comp_believed["Mn"],
            charge_Cu_pct=100 * comp_believed["Cu"],
            energy_kWh=meas_E,
            power_on_min=traj.t[-1] / 60.0,
            avg_power_kW=meas_E * 3600.0 / max(traj.t[-1], 1),
            O2_Nm3=0.0 if cfg.plant.furnace_type == "IF" else 900.0 * max(traj.t[-1] - 900, 0) / 3600.0,
            flux_CaO_kg=charge_kg * 0.010 * 0.92,
            hot_heel_t=heel / 1000.0,
            lining_age_heats=self.lining_age,
            tap_target_C=T_tap,
        )
        truth = dict(meas_T_C=meas_T, meas_C_pct=meas_C, meas_energy_kWh=meas_E,
                     true_T_C=ep["T_C"], true_C_pct=ep["pct_C"],
                     true_Cu_pct=100 * comp_true["Cu"] / 1.0 * 0 + ep["pct_Cu"],
                     SEC_kWh_per_t=ep["SEC_kWh_per_t"],
                     tap_to_tap_min=ep["tap_to_tap_min"],
                     B2=ep["B2"], pct_FeO=ep["pct_FeO_slag"])
        return HeatRecord(feats, truth, traj if keep_trajectory else None)

    # ------------------------------------------------------------------
    def generate(self, n_heats: int, nominal_model: Optional[FurnaceModel] = None,
                 progress=False) -> pd.DataFrame:
        """
        Generate n heats and attach the *nominal* physics prediction for each,
        so the ML layer has (phys_T_C, phys_C_pct) residual targets.
        """
        nominal = nominal_model or FurnaceModel(self.cfg)
        rows = []
        for i in range(n_heats):
            rec = self.run_heat()
            phys = self._nominal_prediction(nominal, rec)
            rows.append({**rec.features, **rec.truth, **phys})
            if progress and (i + 1) % 25 == 0:
                print(f"  ... {i+1}/{n_heats} heats", flush=True)
        return pd.DataFrame(rows)

    def _nominal_prediction(self, nominal: FurnaceModel, rec: HeatRecord) -> Dict[str, float]:
        """
        What the *shipped* physics model (theta = nominal, believed assay,
        no lining-age knowledge) would have predicted for this heat. The
        difference between this and the measurement is exactly what the GP learns.
        """
        f = rec.features
        charge_kg = f["charge_mass_t"] * 1000.0
        comp = {"C": f["charge_C_pct"] / 100, "Si": f["charge_Si_pct"] / 100,
                "Mn": f["charge_Mn_pct"] / 100, "Cu": f["charge_Cu_pct"] / 100,
                "P": 0.00035, "S": 0.0003}
        x0 = nominal.initial_state(charge_kg, comp, hot_heel_kg=f["hot_heel_t"] * 1000)
        P_avg = f["avg_power_kW"]
        u = HeatInputs(lambda t: P_avg,
                       (lambda t: 0.0) if self.cfg.plant.furnace_type == "IF"
                       else (lambda t: 900.0 if t > 900 else 0.0),
                       [Addition(600.0, f["flux_CaO_kg"] / 0.92,
                                 {"CaO": 0.92, "SiO2": 0.04}, into="slag")])
        traj = nominal.simulate(x0, u, f["power_on_min"] * 60.0, dt=self.dt_s)
        ep = nominal.endpoint(traj)
        return {"phys_T_C": ep["T_C"], "phys_C_pct": ep["pct_C"],
                "phys_SEC": ep["SEC_kWh_per_t"]}
