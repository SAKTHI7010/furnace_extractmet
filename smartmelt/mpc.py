"""
mpc.py — Layer 3c: what the operator should actually do next.

Receding-horizon optimisation over the remaining heat:

  min_u  w_E * sum_k P_k dt  +  w_T (T_N - T*)^2  +  w_C (C_N - C*)^2
                             +  w_dP sum_k (P_k - P_{k-1})^2              (E50)
  s.t.   x_{k+1} = f_dt(x_k, u_k)                 (the physics model)
         P_k in the discrete tap set              (E51)
         T_hotface,k <= T_lim                     (refractory life)       (E52)
         T_k <= T_max                             (no burnt heat)

Single-shooting with SLSQP on a piecewise-constant control (n_blocks values).
Deliberately low-dimensional: 4-6 blocks over 20-40 min is all the resolution
an operator can act on anyway, and it keeps the solve inside a few hundred ms.

The projection onto the discrete tap set (E51) happens *after* the continuous
solve — proper MINLP here buys nothing an operator would notice.

Note on Phase 1: the output of this module is a RECOMMENDATION rendered on the
HMI. Nothing here writes to a PLC. The writeback path lives outside this
package by design, so that no code path from MPC to an OPC UA write can exist
without an explicit, reviewed integration layer.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from scipy.optimize import minimize

from .physics import FurnaceModel, HeatInputs
from .thermo import KELVIN


@dataclass
class MPCResult:
    power_blocks_kW: np.ndarray
    o2_blocks_Nm3h: np.ndarray
    block_edges_s: np.ndarray
    predicted_T_C: float
    predicted_C_pct: float
    predicted_energy_kWh: float
    predicted_tap_time_s: float
    cost: float
    success: bool

    def as_operator_actions(self, tap_levels: List[float]) -> List[str]:
        acts = []
        for i, (p, t0) in enumerate(zip(self.power_blocks_kW, self.block_edges_s[:-1])):
            lvl = int(np.argmin([abs(p - l) for l in tap_levels]))
            acts.append(f"t+{t0/60:5.1f} min : power tap {lvl} "
                        f"(~{tap_levels[lvl]:.0f} kW)")
        return acts


class MeltMPC:
    def __init__(self, model: FurnaceModel,
                 w_energy=1.0, w_T=40.0, w_C=2.0e5, w_dP=2.0e-4,
                 n_blocks=5, dt_s: Optional[float] = None):
        self.m = model
        self.w = dict(E=w_energy, T=w_T, C=w_C, dP=w_dP)
        self.n_blocks = n_blocks
        self.dt = dt_s or model.cfg.numerics.dt_mpc_s

    # ------------------------------------------------------------------
    def _rollout(self, x0, t0, horizon_s, P, O2, base: HeatInputs):
        edges = t0 + np.linspace(0, horizon_s, self.n_blocks + 1)

        def pw(vals):
            def f(t):
                k = int(np.clip(np.searchsorted(edges, t, "right") - 1,
                                0, self.n_blocks - 1))
                return float(vals[k])
            return f

        u = HeatInputs(power_kW=pw(P), oxygen_Nm3_per_h=pw(O2),
                       additions=[a for a in base.additions if a.time_s >= t0],
                       p_CO_atm=base.p_CO_atm)
        traj = self.m.simulate(x0, u, horizon_s, dt=self.dt)
        return traj, edges

    def _objective(self, z, x0, t0, horizon_s, base, T_star, C_star, o2_fixed):
        nb = self.n_blocks
        P = np.abs(z[:nb])
        O2 = o2_fixed if o2_fixed is not None else np.abs(z[nb:])
        traj, _ = self._rollout(x0, t0, horizon_s, P, O2, base)
        ep = self.m.endpoint(traj)

        E = traj.X[-1, self.m.iE] - x0[self.m.iE]
        J = self.w["E"] * E
        J += self.w["T"] * (ep["T_C"] - T_star) ** 2
        J += self.w["C"] * (ep["pct_C"] - C_star) ** 2
        J += self.w["dP"] * float(np.sum(np.diff(P) ** 2))

        # (E52) refractory & overheating barriers
        hot = traj.diagnostics["T_hotface"]
        lim = self.m.cfg.lining.hot_face_limit_C
        J += 5.0 * np.sum(np.maximum(hot[1:] - lim, 0.0) ** 2)
        T_series = traj.X[:, self.m.iTb] - KELVIN
        J += 5.0 * np.sum(np.maximum(T_series - (T_star + 60.0), 0.0) ** 2)

        # unmelted charge left at horizon end is a hard penalty
        J += 2.0 * max(traj.X[-1, self.m.iMs], 0.0) ** 2
        return J

    # ------------------------------------------------------------------
    def solve(self, x0: np.ndarray, t0: float, horizon_s: float,
              base: HeatInputs, T_star: float, C_star: float,
              P_max: Optional[float] = None,
              o2_fixed: Optional[np.ndarray] = None,
              P_init: Optional[np.ndarray] = None) -> MPCResult:
        el = self.m.cfg.electrical
        P_max = P_max or el.rated_power_kW
        nb = self.n_blocks

        P0 = P_init if P_init is not None else np.full(nb, 0.7 * P_max)
        if o2_fixed is None:
            o2_max = 2000.0 if self.m.cfg.plant.furnace_type != "IF" else 0.0
            z0 = np.concatenate([P0, np.full(nb, 0.3 * o2_max)])
            bnds = [(0, P_max)] * nb + [(0, o2_max)] * nb
        else:
            z0 = P0
            bnds = [(0, P_max)] * nb

        res = minimize(self._objective, z0,
                       args=(x0, t0, horizon_s, base, T_star, C_star, o2_fixed),
                       method="SLSQP", bounds=bnds,
                       options=dict(maxiter=40, ftol=1e-2, eps=max(P_max * 1e-3, 1.0)))

        z = res.x
        P = np.abs(z[:nb])
        O2 = o2_fixed if o2_fixed is not None else np.abs(z[nb:])

        # (E51) project power onto the discrete tap set the operator actually has
        taps = np.array(el.tap_levels_kW)
        P_proj = taps[np.argmin(np.abs(P[:, None] - taps[None, :]), axis=1)]

        traj, edges = self._rollout(x0, t0, horizon_s, P_proj, O2, base)
        ep = self.m.endpoint(traj)
        return MPCResult(P_proj, np.asarray(O2, float), edges,
                         ep["T_C"], ep["pct_C"],
                         traj.X[-1, self.m.iE] - x0[self.m.iE],
                         horizon_s, float(res.fun), bool(res.success))

    # ------------------------------------------------------------------
    def tap_time_advice(self, x0, t0, base: HeatInputs, T_star: float,
                        max_look_s: float = 1800.0) -> Dict[str, float]:
        """When does the current trajectory cross the tap temperature? (E53)"""
        traj = self.m.simulate(x0, base, max_look_s, dt=self.dt)
        T = traj.X[:, self.m.iTb] - KELVIN
        idx = np.argmax(T >= T_star)
        if T[idx] < T_star:
            return {"tap_in_s": float("inf"), "reachable": False}
        return {"tap_in_s": float(traj.t[idx]), "reachable": True,
                "energy_to_tap_kWh": float(traj.X[idx, self.m.iE] - x0[self.m.iE])}
