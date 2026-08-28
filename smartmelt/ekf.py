"""
ekf.py — Layer 3a: state estimation.

The physics model is a *prediction*. The plant has a handful of noisy, sparse
sensors. The EKF fuses them, and — more importantly — estimates the small set
of plant-specific parameters theta online. This is what lets the same binary
run at Industry-X and at a second plant (Industry-Y) with different coils, different
lining age and different scrap.

Augmented state:      z = [ x ; theta ]
Process:              z_{k+1} = [ f_dt(x_k, theta_k, u_k) ; theta_k ] + w
Measurement:          y_k     = h(z_k) + v

Jacobians are central finite differences on the one-step map. With n ~ 25-30
states this costs ~2n RHS evaluations per second of plant time — still well
inside the edge budget at dt = 1 s.

Random-walk process noise on theta (Q_theta) is the tuning knob:
  too large -> theta chases sensor noise;  too small -> no adaptation.
Rule of thumb: sigma_theta per heat ~ 1-2 % of nominal.
"""
from __future__ import annotations

import numpy as np
from typing import Callable, List, Optional

from .physics import FurnaceModel, HeatInputs
from .thermo import KELVIN


class ExtendedKalmanFilter:
    def __init__(self, model: FurnaceModel, theta_keys: List[str],
                 h: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 R: np.ndarray, Q_x: np.ndarray, Q_theta: np.ndarray,
                 theta_bounds: Optional[dict] = None):
        self.m = model
        self.theta_keys = list(theta_keys)
        self.h = h
        self.R = np.atleast_2d(R)
        self.nx = model.n_state
        self.nt = len(theta_keys)
        self.Q = np.zeros((self.nx + self.nt, self.nx + self.nt))
        self.Q[:self.nx, :self.nx] = Q_x
        self.Q[self.nx:, self.nx:] = Q_theta
        self.bounds = theta_bounds or {}
        self.P: Optional[np.ndarray] = None
        self.z: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def init(self, x0: np.ndarray, P0: np.ndarray):
        theta0 = np.array([self.m.theta[k] for k in self.theta_keys])
        self.z = np.concatenate([x0, theta0])
        self.P = P0.copy()

    def _set_theta(self, theta_vec):
        for k, v in zip(self.theta_keys, theta_vec):
            lo, hi = self.bounds.get(k, (-np.inf, np.inf))
            self.m.theta[k] = float(np.clip(v, lo, hi))

    def _f(self, z, t, u: HeatInputs, dt):
        x, theta = z[:self.nx], z[self.nx:]
        saved = {k: self.m.theta[k] for k in self.theta_keys}
        self._set_theta(theta)
        xn = self.m.step(t, x.copy(), u, dt)
        self.m.theta.update(saved)
        return np.concatenate([xn, theta])

    def _jacobian(self, fun, z, eps_scale=1e-6):
        n = z.size
        f0 = fun(z)
        J = np.zeros((f0.size, n))
        for i in range(n):
            e = max(abs(z[i]), 1.0) * eps_scale
            zp = z.copy(); zp[i] += e
            zm = z.copy(); zm[i] -= e
            J[:, i] = (fun(zp) - fun(zm)) / (2 * e)
        return J, f0

    # ------------------------------------------------------------------
    def predict(self, t: float, u: HeatInputs, dt: float):
        F, z_pred = self._jacobian(lambda zz: self._f(zz, t, u, dt), self.z)
        self.z = z_pred
        self.P = F @ self.P @ F.T + self.Q * dt
        self._set_theta(self.z[self.nx:])

    def update(self, y: np.ndarray, active: Optional[np.ndarray] = None):
        """`active` masks out sensors that are offline this tick (E33)."""
        H, y_pred = self._jacobian(lambda zz: self.h(zz[:self.nx], zz[self.nx:]), self.z)
        if active is not None:
            H, y_pred, y = H[active], y_pred[active], y[active]
            R = self.R[np.ix_(active, active)]
        else:
            R = self.R
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.z = self.z + K @ (y - y_pred)
        I = np.eye(self.z.size)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T   # Joseph form
        self._set_theta(self.z[self.nx:])

    # ------------------------------------------------------------------
    @property
    def x(self):
        return self.z[:self.nx]

    @property
    def theta(self):
        return dict(zip(self.theta_keys, self.z[self.nx:]))

    def bath_temperature_C(self):
        return self.z[self.m.iTb] - KELVIN

    def sigma_T(self):
        return float(np.sqrt(self.P[self.m.iTb, self.m.iTb]))

    def pct_C(self):
        m = self.z[:self.m.nM]
        return 100.0 * m[self.m.metal.index("C")] / max(m.sum(), 1e-6)


# ----------------------------------------------------------------------
def build_default_ekf(model: FurnaceModel, u: HeatInputs) -> ExtendedKalmanFilter:
    """
    Observation model built from cfg.sensors — this is where the SKU shows up.
    SmartMelt Lite (IF): power meter + pyrometer + load cells.
    SmartMelt Pro (EAF/BOF): + off-gas CO/CO2 + immersion TC / sublance.
    """
    cfg = model.cfg
    sen = cfg.sensors
    obs, sig = [], []

    if sen.has_pyrometer:
        obs.append(("T_pyro", lambda x, th_: x[model.iTb] - KELVIN))
        sig.append(sen.sigma_T_pyrometer_C)
    if sen.has_immersion_tc:
        obs.append(("T_tc", lambda x, th_: x[model.iTb] - KELVIN))
        sig.append(sen.sigma_T_immersion_C)
    if sen.has_load_cells:
        obs.append(("mass_t", lambda x, th_: (x[:model.nM].sum() + x[model.iMs]) / 1000.0))
        sig.append(0.05)
    if sen.has_offgas_analyser:
        def co_pct(x, th_):
            co, co2 = x[model.iCO], x[model.iCO2]
            return 100.0 * co / max(co + co2 + 1e-6, 1e-6)
        obs.append(("CO_pct", co_pct))
        sig.append(sen.sigma_offgas_pct)

    names = [o[0] for o in obs]
    funcs = [o[1] for o in obs]

    def h(x, theta):
        return np.array([f(x, theta) for f in funcs])

    R = np.diag(np.square(sig))

    nx = model.n_state
    Q_x = np.zeros((nx, nx))
    Q_x[model.iTb, model.iTb] = 0.25 ** 2            # K^2 per second
    for i in range(model.nM):
        Q_x[i, i] = (1e-3 * 1.0) ** 2
    Q_x[model.iMs, model.iMs] = 0.5 ** 2

    theta_keys = ["eta_electrical", "UA_lining_scale", "k_C_scale"]
    Q_theta = np.diag([2e-7, 2e-6, 2e-6])            # random walk, per second

    P0 = np.zeros((nx + len(theta_keys), nx + len(theta_keys)))
    P0[model.iTb, model.iTb] = 20.0 ** 2
    P0[model.iMs, model.iMs] = 100.0 ** 2
    for i in range(model.nM):
        P0[i, i] = (0.02 * 1000.0) ** 2
    P0[nx:, nx:] = np.diag([0.03 ** 2, 0.15 ** 2, 0.20 ** 2])

    ekf = ExtendedKalmanFilter(
        model, theta_keys, h, R, Q_x, Q_theta,
        theta_bounds={"eta_electrical": (0.75, 1.15),
                      "UA_lining_scale": (0.4, 2.5),
                      "k_C_scale": (0.3, 3.0)})
    ekf.sensor_names = names
    return ekf
