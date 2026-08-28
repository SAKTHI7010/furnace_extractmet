"""
ml.py — Layer 2: data-driven correction on top of the physics prediction.

The single most important design decision in the whole system:

    y_measured  =  f_physics(x, u; theta)  +  g_ML(phi)  +  eps          (E31)

The ML head learns the *residual*, never the absolute end-point. Consequences:

  * If the ML head returns 0, you still have a metallurgically valid answer.
    That is your degraded mode, and it is why regime change cannot silently
    destroy the system (the documented failure mode in ref. 17 of the brief).
  * The residual is small and roughly stationary, so a GP with ~10^3 training
    points is enough. An absolute-endpoint learner would need 10^5.
  * Extrapolation is bounded: you can clip |g_ML| to, say, 40 C and 0.05 %C and
    lose almost nothing on in-distribution heats while capping the damage
    out-of-distribution.

Uncertainty: sigma^2 = sigma_GP^2 + sigma_physics^2 + sigma_sensor^2      (E32)
sigma_physics comes from propagating the EKF's theta covariance; sigma_GP is the
GP posterior variance. The advisory layer consumes sigma, not just the mean.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (Matern, RBF, WhiteKernel,
                                              ConstantKernel as C)
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
FEATURES = [
    "charge_mass_t", "scrap_frac", "dri_frac", "pig_frac", "returns_frac",
    "charge_C_pct", "charge_Si_pct", "charge_Mn_pct", "charge_Cu_pct",
    "energy_kWh", "power_on_min", "avg_power_kW", "O2_Nm3",
    "flux_CaO_kg", "hot_heel_t", "lining_age_heats", "tap_target_C",
    "phys_T_C", "phys_C_pct",            # <- physics prediction as a feature
]


def build_features(records: pd.DataFrame) -> pd.DataFrame:
    """Deterministic, order-stable feature frame. Missing columns -> 0."""
    df = pd.DataFrame(index=records.index)
    for f in FEATURES:
        df[f] = records[f] if f in records.columns else 0.0
    # A few physically motivated interactions
    df["spec_energy"] = df["energy_kWh"] / np.maximum(df["charge_mass_t"], 1e-3)
    df["O2_per_t"] = df["O2_Nm3"] / np.maximum(df["charge_mass_t"], 1e-3)
    df["lining_age_norm"] = df["lining_age_heats"] / 500.0
    return df


# --------------------------------------------------------------------------
# Residual GP
# --------------------------------------------------------------------------
class ResidualGPR:
    """
    GP on the physics residual. Matern-3/2 is the right default: the residual
    surface is continuous but not smooth (scrap-grade discontinuities).
    Returns (mean, sigma) — the sigma is what makes an *advisory* system honest.
    """

    def __init__(self, length_scale=1.0, noise=1e-2, kernel="matern32",
                 max_train=1500, clip=None, random_state=0):
        self.length_scale = length_scale
        self.noise = noise
        self.kernel_name = kernel
        self.max_train = max_train
        self.clip = clip                    # e.g. 40.0 for temperature (deg C)
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.gp: Optional[GaussianProcessRegressor] = None
        self.y_mean = 0.0

    def _kernel(self, d):
        if self.kernel_name == "rbf":
            base = RBF(length_scale=np.full(d, self.length_scale),
                       length_scale_bounds=(1e-2, 1e3))
        else:
            base = Matern(length_scale=np.full(d, self.length_scale), nu=1.5,
                          length_scale_bounds=(1e-2, 1e3))
        return C(1.0, (1e-3, 1e3)) * base + WhiteKernel(self.noise, (1e-6, 1e1))

    def fit(self, X: pd.DataFrame, residual: np.ndarray):
        Xs = self.scaler.fit_transform(np.asarray(X, float))
        y = np.asarray(residual, float)
        if len(y) > self.max_train:                      # subsample: GP is O(n^3)
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(y), self.max_train, replace=False)
            Xs, y = Xs[idx], y[idx]
        self.y_mean = float(y.mean())
        self.gp = GaussianProcessRegressor(
            kernel=self._kernel(Xs.shape[1]), normalize_y=False,
            alpha=1e-8, n_restarts_optimizer=2, random_state=self.random_state)
        self.gp.fit(Xs, y - self.y_mean)
        return self

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if self.gp is None:                              # cold start: no correction
            n = len(X)
            return np.zeros(n), np.full(n, np.inf)
        Xs = self.scaler.transform(np.asarray(X, float))
        dev, sd = self.gp.predict(Xs, return_std=True)
        # Clip the *deviation*, not the learned mean bias: the mean is the
        # best-established part of the residual (it is what n heats agree on);
        # the deviation is where extrapolation risk lives. Clipping the total
        # silently caps any systematic physics bias larger than `clip` — a
        # bug that cost 40 C of correction in testing.
        if self.clip is not None:
            dev = np.clip(dev, -self.clip, self.clip)
        return dev + self.y_mean, sd


# --------------------------------------------------------------------------
# Gradient boosting with quantiles
# --------------------------------------------------------------------------
class EndpointGBM:
    """
    Quantile GBM head. Cheap, robust, retrains in minutes on an edge box, and
    gives an empirical prediction interval that cross-checks the GP's sigma.
    Also used for the energy-consumption auxiliary head.
    """

    def __init__(self, quantiles=(0.1, 0.5, 0.9), max_iter=400, lr=0.05, seed=0):
        self.quantiles = tuple(quantiles)
        self.models: Dict[float, HistGradientBoostingRegressor] = {}
        self.kw = dict(max_iter=max_iter, learning_rate=lr, random_state=seed,
                       early_stopping=False)

    def fit(self, X, y):
        Xv = np.asarray(X, float)
        for q in self.quantiles:
            m = HistGradientBoostingRegressor(loss="quantile", quantile=q, **self.kw)
            m.fit(Xv, np.asarray(y, float))
            self.models[q] = m
        return self

    def predict(self, X) -> Dict[float, np.ndarray]:
        Xv = np.asarray(X, float)
        return {q: m.predict(Xv) for q, m in self.models.items()}

    def median_and_sigma(self, X):
        p = self.predict(X)
        lo, med, hi = p[self.quantiles[0]], p[0.5], p[self.quantiles[-1]]
        sigma = (hi - lo) / 2.563       # 10-90 interval of a normal
        return med, np.maximum(sigma, 1e-6)


# --------------------------------------------------------------------------
# Hybrid predictor
# --------------------------------------------------------------------------
@dataclass
@dataclass
class Prediction:
    mean: float
    sigma: float
    physics: float
    residual: float
    source: str


class HybridEndpointModel:
    """
    Fuses physics + GP residual (+ optional GBM cross-check) into a calibrated
    prediction. `fit` requires columns: phys_T_C, phys_C_pct, meas_T_C, meas_C_pct.
    """

    def __init__(self, cfg, clip_T=40.0, clip_C=0.05):
        mlc = cfg.ml
        self.cfg = cfg
        self.gp_T = ResidualGPR(mlc.gpr_length_scale, mlc.gpr_noise,
                                mlc.gpr_kernel, clip=clip_T)
        self.gp_C = ResidualGPR(mlc.gpr_length_scale, mlc.gpr_noise,
                                mlc.gpr_kernel, clip=clip_C)
        self.gbm_T = EndpointGBM(mlc.quantiles, mlc.gbm_max_iter, mlc.gbm_learning_rate)
        self.gbm_E = EndpointGBM((0.5,), mlc.gbm_max_iter, mlc.gbm_learning_rate)
        self.n_heats = 0
        self.maturity = "coldstart"

    def _maturity(self, n):
        m = self.cfg.ml
        if n >= m.min_heats_calibrated: return "calibrated"
        if n >= m.min_heats_deployable: return "deployable"
        if n >= m.min_heats_coldstart:  return "coldstart"
        return "insufficient"

    @staticmethod
    def _cv_gate(gp_factory, X, resid, min_gain=0.98):
        """
        Rolling-origin CV: does the GP correction beat zero-correction
        (i.e. raw physics) out-of-time? If not, the head is DISABLED.
        This is the enforcement of the hybrid promise: ML must *prove*
        improvement on time-ordered data before it touches the advisory.
        """
        n = len(resid)
        if n < 25:
            return False
        maes_gp, maes_phys = [], []
        for cut in (0.6, 0.8):
            k = int(n * cut)
            gp = gp_factory().fit(X.iloc[:k], resid[:k])
            mu, _ = gp.predict(X.iloc[k:])
            maes_gp.append(np.mean(np.abs(resid[k:] - mu)))
            maes_phys.append(np.mean(np.abs(resid[k:])))
        return float(np.mean(maes_gp)) < min_gain * float(np.mean(maes_phys))

    def fit(self, records: pd.DataFrame):
        X = build_features(records)
        mlc = self.cfg.ml
        rT = (records["meas_T_C"] - records["phys_T_C"]).to_numpy()
        rC = (records["meas_C_pct"] - records["phys_C_pct"]).to_numpy()
        mkT = lambda: ResidualGPR(mlc.gpr_length_scale, mlc.gpr_noise,
                                  mlc.gpr_kernel, clip=self.gp_T.clip)
        mkC = lambda: ResidualGPR(mlc.gpr_length_scale, mlc.gpr_noise,
                                  mlc.gpr_kernel, clip=self.gp_C.clip)
        self.use_T = self._cv_gate(mkT, X, rT)
        self.use_C = self._cv_gate(mkC, X, rC)
        if self.use_T:
            self.gp_T.fit(X, rT)
        if self.use_C:
            self.gp_C.fit(X, rC)
        self.gbm_T.fit(X, records["meas_T_C"])
        if "meas_energy_kWh" in records:
            self.gbm_E.fit(X, records["meas_energy_kWh"])
        self.n_heats = len(records)
        self.maturity = self._maturity(self.n_heats)
        return self

    def predict(self, record: pd.DataFrame, sigma_phys_T=6.0, sigma_phys_C=0.008):
        X = build_features(record)
        n = len(X)
        if getattr(self, "use_T", False):
            dT, sT = self.gp_T.predict(X)
        else:   # head gated off: physics-only with honestly inflated sigma
            dT, sT = np.zeros(n), np.full(n, 3 * sigma_phys_T)
        if getattr(self, "use_C", False):
            dC, sC = self.gp_C.predict(X)
        else:
            dC, sC = np.zeros(n), np.full(n, 3 * sigma_phys_C)
        if not np.isfinite(sT).all():
            dT, sT = np.zeros(n), np.full(n, 3 * sigma_phys_T)
        if not np.isfinite(sC).all():
            dC, sC = np.zeros(n), np.full(n, 3 * sigma_phys_C)
        T = record["phys_T_C"].to_numpy() + dT
        Cc = record["phys_C_pct"].to_numpy() + dC
        return (
            [Prediction(t, float(np.hypot(s, sigma_phys_T)), p, d, self.maturity)
             for t, s, p, d in zip(T, sT, record["phys_T_C"], dT)],
            [Prediction(c, float(np.hypot(s, sigma_phys_C)), p, d, self.maturity)
             for c, s, p, d in zip(Cc, sC, record["phys_C_pct"], dC)],
        )


# --------------------------------------------------------------------------
# Tramp-element soft sensor
# --------------------------------------------------------------------------
class TrampSoftSensor:
    """
    Cu, Sn, Cr are not removed by oxidising steelmaking: whatever enters with
    the scrap ends up in the steel. So the *prior* is an exact mass balance,
    and ML only has to learn the errors in the scrap-grade assay and the yield.

        [%Cu]_pred = MB(charge) + g(phi)                                   (E34)

    That is a far better-posed problem than learning [%Cu] from scratch, and it
    is what makes a soft sensor viable without a spectrometer on every heat.
    """

    def __init__(self, elements=("Cu", "Sn", "Cr"), alpha=1.0):
        self.elements = list(elements)
        self.models = {e: Ridge(alpha=alpha) for e in elements}
        self.scaler = StandardScaler()
        self.fitted = False

    @staticmethod
    def mass_balance(charge_masses: Dict[str, float],
                     assays: Dict[str, Dict[str, float]],
                     element: str, tap_mass_kg: float) -> float:
        num = sum(m * assays.get(mat, {}).get(element, 0.0)
                  for mat, m in charge_masses.items())
        return 100.0 * num / max(tap_mass_kg, 1e-6)

    def fit(self, X: pd.DataFrame, mb_prior: pd.DataFrame, measured: pd.DataFrame):
        Xs = self.scaler.fit_transform(np.asarray(X, float))
        for e in self.elements:
            if e in measured:
                self.models[e].fit(Xs, measured[e] - mb_prior[e])
        self.fitted = True
        return self

    def predict(self, X: pd.DataFrame, mb_prior: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        Xs = self.scaler.transform(np.asarray(X, float)) if self.fitted else None
        for e in self.elements:
            corr = self.models[e].predict(Xs) if self.fitted else 0.0
            out[e] = mb_prior[e] + corr
        return out


# --------------------------------------------------------------------------
# Drift monitor
# --------------------------------------------------------------------------
class DriftMonitor:
    """PSI on inputs + MAPE on the temperature head. Alarm -> widen or suspend."""

    # Features that move deterministically with time (campaign clocks) must
    # not feed PSI: any later window is "drifted" on a clock by construction.
    # Drift monitoring asks whether the INPUT DISTRIBUTION moved, not whether
    # time passed.
    EXCLUDE = ("lining_age_heats", "lining_age_norm", "heat_no")

    def __init__(self, cfg):
        self.psi_alarm = cfg.ml.psi_alarm
        self.mape_alarm = cfg.ml.mape_alarm_pct
        self.reference: Optional[pd.DataFrame] = None

    def set_reference(self, X: pd.DataFrame):
        self.reference = X.copy()
        return self

    MIN_WINDOW = 15      # PSI on fewer heats is noise, not evidence

    def check(self, X_recent: pd.DataFrame, y_true=None, y_pred=None) -> dict:
        from .metrics import psi, mape
        report = {"psi": {}, "psi_max": 0.0, "mape": None,
                  "alarm": False, "reasons": [],
                  "window_ok": len(X_recent) >= self.MIN_WINDOW}
        if self.reference is not None:
            for col in self.reference.columns:
                if col in self.EXCLUDE:
                    continue
                if col in X_recent:
                    p = psi(self.reference[col].to_numpy(), X_recent[col].to_numpy())
                    report["psi"][col] = p
            if report["psi"]:
                worst = max(report["psi"], key=report["psi"].get)
                report["psi_max"] = report["psi"][worst]
                if report["psi_max"] > self.psi_alarm and report["window_ok"]:
                    report["alarm"] = True
                    report["reasons"].append(f"PSI {worst}={report['psi_max']:.3f}")
        if y_true is not None and y_pred is not None:
            report["mape"] = mape(y_true, y_pred)
            if report["mape"] > self.mape_alarm:
                report["alarm"] = True
                report["reasons"].append(f"MAPE={report['mape']:.2f}%")
        return report
