"""
calibrate.py — fitting theta to a plant, and knowing when you can't.

Do this ONCE per plant on the historical heats pulled during the pre-install
audit, then let the EKF track theta from there. Order matters:

  Step 1  Fit eta_electrical and UA_lining_scale on *energy* only.
          These two are the only parameters that a plant with no chemistry
          instrumentation can identify at all, and they are the ones that
          carry the kWh/t claim. Use heats with a wide spread of tap-to-tap
          time — long heats separate the standing loss (UA) from the
          throughput loss (eta).

  Step 2  Fit k_C_scale and gamma_FeO on *carbon* only, holding step 1 fixed.
          Needs a bath sample. Without one, leave them at nominal and inflate
          sigma_C — do not pretend.

  Step 3  Check identifiability BEFORE trusting the fit. `identifiability()`
          returns the correlation matrix of the estimate and the condition
          number of J^T J. If |rho(eta, UA)| > 0.95, your heats are all the
          same length and you have fitted one number, not two. Get longer and
          shorter heats, or fix UA from a cold-furnace standing-loss test.

That last step is the difference between a model and a curve-fit. It is also
the question a good panellist asks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from scipy.optimize import least_squares

from .physics import FurnaceModel, HeatInputs, Addition
from .config import PlantConfig
from .thermo import KELVIN


DEFAULT_BOUNDS = {
    "eta_electrical": (0.80, 1.10),
    "UA_lining_scale": (0.50, 2.50),
    "k_C_scale": (0.30, 3.00),
    "h_solid_liquid_scale": (0.50, 2.00),
    "gamma_FeO": (0.80, 3.00),
}


@dataclass
class CalibrationResult:
    theta: Dict[str, float]
    residual_rms: float
    n_heats: int
    correlation: pd.DataFrame
    condition_number: float
    warnings: List[str]

    def summary(self) -> str:
        s = [f"theta fitted on {self.n_heats} heats, residual RMS = {self.residual_rms:.3f}",
             f"cond(J^T J) = {self.condition_number:.1f}"]
        s += [f"  {k:<24}{v:.4f}" for k, v in self.theta.items()]
        s += ["WARNING: " + w for w in self.warnings]
        return "\n".join(s)


# --------------------------------------------------------------------------
def _simulate_heat(model: FurnaceModel, row: pd.Series) -> dict:
    charge_kg = row["charge_mass_t"] * 1000.0
    comp = {"C": row.get("charge_C_pct", 0.3) / 100,
            "Si": row.get("charge_Si_pct", 0.2) / 100,
            "Mn": row.get("charge_Mn_pct", 0.35) / 100,
            "Cu": row.get("charge_Cu_pct", 0.2) / 100,
            "P": 0.00035, "S": 0.0003}
    x0 = model.initial_state(charge_kg, comp,
                             hot_heel_kg=row.get("hot_heel_t", 0.03) * 1000)
    P = float(row["avg_power_kW"])
    o2 = float(row.get("O2_Nm3", 0.0))
    dur = float(row["power_on_min"]) * 60.0
    o2_rate = o2 * 3600.0 / max(dur, 1.0)
    u = HeatInputs(lambda t: P, lambda t: o2_rate,
                   [Addition(600.0, row.get("flux_CaO_kg", 0.0) / 0.92,
                             {"CaO": 0.92, "SiO2": 0.04}, into="slag")])
    traj = model.simulate(x0, u, dur, dt=3.0)
    return model.endpoint(traj)


def calibrate_physics(cfg: PlantConfig, heats: pd.DataFrame,
                      fit_keys: Sequence[str] = ("eta_electrical", "UA_lining_scale"),
                      targets: Sequence[str] = ("meas_T_C",),
                      weights: Optional[Dict[str, float]] = None,
                      theta0: Optional[Dict[str, float]] = None,
                      max_heats: int = 120, verbose: bool = True) -> CalibrationResult:
    """
    Weighted least squares:  min_theta  sum_h sum_y w_y (y_h - yhat_h(theta))^2
    Jacobian by finite differences on the *endpoint map*, not the ODE — small
    (n_theta columns), and it is what the physical residual actually depends on.
    """
    fit_keys = list(fit_keys)
    weights = weights or {"meas_T_C": 1.0 / 15.0, "meas_C_pct": 1.0 / 0.02,
                          "meas_energy_kWh": 1.0 / 200.0}
    df = heats.iloc[:max_heats].reset_index(drop=True)
    warnings: List[str] = []

    lo = np.array([DEFAULT_BOUNDS[k][0] for k in fit_keys])
    hi = np.array([DEFAULT_BOUNDS[k][1] for k in fit_keys])
    base = FurnaceModel(cfg)
    z0 = np.array([(theta0 or base.theta)[k] for k in fit_keys])

    key_map = {"meas_T_C": "T_C", "meas_C_pct": "pct_C", "meas_energy_kWh": "energy_kWh"}

    def residuals(z):
        model = FurnaceModel(cfg, dict(zip(fit_keys, z)))
        out = []
        for _, row in df.iterrows():
            ep = _simulate_heat(model, row)
            for tgt in targets:
                out.append(weights[tgt] * (row[tgt] - ep[key_map[tgt]]))
        return np.asarray(out)

    res = least_squares(residuals, np.clip(z0, lo, hi), bounds=(lo, hi),
                        diff_step=0.02, xtol=1e-6, ftol=1e-6,
                        verbose=2 if verbose else 0)

    theta = dict(zip(fit_keys, res.x))
    J = res.jac
    JTJ = J.T @ J
    cond = float(np.linalg.cond(JTJ)) if JTJ.size else 1.0
    try:
        cov = np.linalg.inv(JTJ)
        sd = np.sqrt(np.diag(cov))
        corr = cov / np.outer(sd, sd)
    except np.linalg.LinAlgError:
        corr = np.full((len(fit_keys),) * 2, np.nan)
        warnings.append("J^T J singular — parameters not identifiable from these heats")

    corr_df = pd.DataFrame(corr, index=fit_keys, columns=fit_keys)
    for i in range(len(fit_keys)):
        for j in range(i + 1, len(fit_keys)):
            if abs(corr[i, j]) > 0.95:
                warnings.append(
                    f"rho({fit_keys[i]},{fit_keys[j]}) = {corr[i,j]:.3f}: these two "
                    "are not separately identifiable from this data. Add heats with "
                    "different tap-to-tap times, or fix one from a standing-loss test.")
    if cond > 1e6:
        warnings.append(f"cond(J^T J) = {cond:.1e} — ill-conditioned fit")
    for k, v in theta.items():
        l, h = DEFAULT_BOUNDS[k]
        if abs(v - l) < 1e-3 or abs(v - h) < 1e-3:
            warnings.append(f"{k} pinned at its bound ({v:.3f}) — model structure "
                            "is absorbing an effect it does not represent")

    return CalibrationResult(theta, float(np.sqrt(np.mean(res.fun ** 2))),
                             len(df), corr_df, cond, warnings)


# --------------------------------------------------------------------------
def standing_loss_test(cfg: PlantConfig, hold_minutes: float = 30.0,
                       T_start_C: float = 1600.0) -> float:
    """
    The cheapest, highest-value experiment in the whole programme.

    Hold a full liquid bath at temperature with power off for ~20-30 min and log
    the temperature decay. dT/dt at t=0 gives UA directly:

        UA_total = M_l * cp_l * |dT/dt| / (T_b - T_amb)                   (E60)

    One heat's worth of lost production buys you an identifiable UA, which
    de-correlates eta_electrical in every subsequent fit. Ask the plant for it.
    """
    model = FurnaceModel(cfg)
    M = cfg.plant.heat_size_t * 1000.0
    x0 = model.initial_state(M, {"C": 0.002}, hot_heel_kg=M)
    x0[model.iTb] = T_start_C + KELVIN
    x0[model.iMs] = 0.0
    u = HeatInputs(lambda t: 0.0, lambda t: 0.0, [])
    traj = model.simulate(x0, u, hold_minutes * 60, dt=5.0)
    T = traj.X[:, model.iTb] - KELVIN
    return float((T[0] - T[-1]) / hold_minutes)     # deg C per minute
