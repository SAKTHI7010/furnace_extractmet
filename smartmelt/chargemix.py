"""
chargemix.py — Layer 3b: least-cost charge under metallurgical constraints.

This is the module that pays for the box. Energy savings are 4-6 month payback;
charge-mix savings of Rs 200-500/t on a 30,000 t/yr plant are Rs 60-150 lakh/yr
and start on heat #1, because the LP does not need any heat history to be right.

Formulation (E40-E45). Decision variable x_j = kg of material j charged.

  min   sum_j (c_j + lambda_E * e_j) x_j                                  (E40)
  s.t.  sum_j y_j x_j            = M_liq                metallic yield     (E41)
        lo_i <= (sum_j w_ij r_ij x_j) / M_liq <= hi_i   aim elements       (E42)
        (sum_j w_ij x_j) / M_liq  <= tramp_i            residuals          (E43)
        L_j <= x_j <= U_j                               availability       (E44)
        sum_j x_j <= M_charge_max                       crucible volume    (E45)

  c_j  Rs/kg,  e_j  kWh/kg (specific melting energy, incl. DRI endothermy),
  y_j  metallic yield,  w_ij  mass fraction of element i,
  r_ij recovery of element i from material j into the bath.

Tramp elements (Cu, Sn, Ni, Mo, Cr in an oxidising practice) have r = 1 and no
sink: constraint (E43) is a hard wall, not a soft target. That single line is
what separates a metallurgically valid optimiser from a cost spreadsheet.

Extension to MILP: bundles (a "lot" of shredded scrap comes in 5 t bales), set
`integer_lot_kg` per material and the solver switches to scipy.optimize.milp.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from scipy.optimize import linprog, milp, LinearConstraint, Bounds


@dataclass
class Material:
    name: str
    price_INR_per_kg: float
    composition: Dict[str, float]              # mass fractions, e.g. {"C":0.035,"Cu":0.0025}
    metallic_yield: float = 0.94               # kg liquid per kg charged
    energy_kWh_per_kg: float = 0.40            # marginal melting energy
    recovery: Dict[str, float] = field(default_factory=dict)   # element -> fraction to bath
    available_kg: float = 1e9
    min_kg: float = 0.0
    integer_lot_kg: Optional[float] = None

    def rec(self, el: str) -> float:
        # Tramps are fully recovered; oxidisable elements default to partial.
        default = 1.0 if el in ("Cu", "Sn", "Ni", "Mo", "Cr") else 0.85
        return self.recovery.get(el, default)


@dataclass
class MixResult:
    feasible: bool
    masses_kg: Dict[str, float]
    cost_INR: float
    cost_INR_per_t_liquid: float
    energy_kWh: float
    predicted_bath_pct: Dict[str, float]
    liquid_t: float
    message: str = ""

    def pretty(self) -> str:
        lines = [f"{'material':<22}{'kg':>10}"]
        for k, v in sorted(self.masses_kg.items(), key=lambda kv: -kv[1]):
            if v > 1e-6:
                lines.append(f"{k:<22}{v:>10.1f}")
        lines.append(f"\ncost  Rs {self.cost_INR:,.0f}  "
                     f"({self.cost_INR_per_t_liquid:,.0f} Rs/t liquid)")
        lines.append("bath: " + "  ".join(
            f"{k}={v:.4f}%" for k, v in self.predicted_bath_pct.items()))
        return "\n".join(lines)


class ChargeMixOptimiser:
    def __init__(self, materials: List[Material],
                 tariff_INR_per_kWh: float = 8.0,
                 max_charge_kg: Optional[float] = None):
        self.mats = list(materials)
        self.tariff = tariff_INR_per_kWh
        self.max_charge_kg = max_charge_kg

    # ------------------------------------------------------------------
    def solve(self,
              target_liquid_kg: float,
              aim: Dict[str, Tuple[float, float]],       # element -> (lo%, hi%)
              tramp_limits: Optional[Dict[str, float]] = None,
              energy_budget_kWh: Optional[float] = None,
              use_integer_lots: bool = False) -> MixResult:
        n = len(self.mats)
        tramp_limits = tramp_limits or {}

        c = np.array([m.price_INR_per_kg + self.tariff * m.energy_kWh_per_kg
                      for m in self.mats])                          # (E40)

        A_eq = np.array([[m.metallic_yield for m in self.mats]])    # (E41)
        b_eq = np.array([target_liquid_kg])

        A_ub, b_ub = [], []
        for el, (lo, hi) in aim.items():                            # (E42)
            row = np.array([m.composition.get(el, 0.0) * m.rec(el) for m in self.mats])
            A_ub.append(row);  b_ub.append(hi / 100.0 * target_liquid_kg)
            A_ub.append(-row); b_ub.append(-lo / 100.0 * target_liquid_kg)
        for el, hi in tramp_limits.items():                         # (E43)
            row = np.array([m.composition.get(el, 0.0) * m.rec(el) for m in self.mats])
            A_ub.append(row);  b_ub.append(hi / 100.0 * target_liquid_kg)
        if energy_budget_kWh is not None:
            A_ub.append(np.array([m.energy_kWh_per_kg for m in self.mats]))
            b_ub.append(energy_budget_kWh)
        if self.max_charge_kg is not None:                          # (E45)
            A_ub.append(np.ones(n)); b_ub.append(self.max_charge_kg)

        A_ub = np.array(A_ub) if A_ub else None
        b_ub = np.array(b_ub) if b_ub else None
        lb = np.array([m.min_kg for m in self.mats])
        ub = np.array([m.available_kg for m in self.mats])

        if use_integer_lots and any(m.integer_lot_kg for m in self.mats):
            res = self._solve_milp(c, A_eq, b_eq, A_ub, b_ub, lb, ub)
        else:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=list(zip(lb, ub)), method="highs")

        if not getattr(res, "success", False):
            return MixResult(False, {}, 0, 0, 0, {}, 0,
                             message=f"infeasible: {getattr(res,'message','')}. "
                                     "Relax a tramp limit, add a clean-scrap "
                                     "source, or widen the aim window.")

        x = np.asarray(res.x, float)
        return self._report(x, target_liquid_kg, aim, tramp_limits)

    # ------------------------------------------------------------------
    def _solve_milp(self, c, A_eq, b_eq, A_ub, b_ub, lb, ub):
        cons = []
        if A_ub is not None:
            cons.append(LinearConstraint(A_ub, -np.inf, b_ub))
        cons.append(LinearConstraint(A_eq, b_eq, b_eq))
        integrality = np.array([1 if m.integer_lot_kg else 0 for m in self.mats])
        # rescale integer variables to lot counts
        scale = np.array([m.integer_lot_kg or 1.0 for m in self.mats])
        cons_scaled = []
        for con in cons:
            cons_scaled.append(LinearConstraint(con.A * scale, con.lb, con.ub))
        res = milp(c=c * scale, constraints=cons_scaled,
                   integrality=integrality,
                   bounds=Bounds(lb / scale, ub / scale))
        if res.success:
            res.x = res.x * scale
        return res

    def _report(self, x, target_liquid_kg, aim, tramp_limits) -> MixResult:
        masses = {m.name: float(xi) for m, xi in zip(self.mats, x)}
        cost = float(sum(m.price_INR_per_kg * xi for m, xi in zip(self.mats, x)))
        energy = float(sum(m.energy_kWh_per_kg * xi for m, xi in zip(self.mats, x)))
        liquid = float(sum(m.metallic_yield * xi for m, xi in zip(self.mats, x)))
        els = set(aim) | set(tramp_limits) | {"C", "Si", "Mn", "P", "S", "Cu"}
        bath = {el: 100.0 * sum(m.composition.get(el, 0.0) * m.rec(el) * xi
                                for m, xi in zip(self.mats, x)) / max(liquid, 1e-6)
                for el in sorted(els)}
        return MixResult(True, masses, cost, cost / max(liquid / 1000.0, 1e-6),
                         energy, bath, liquid / 1000.0)

    # ------------------------------------------------------------------
    def shadow_prices(self, base: MixResult, aim, tramp_limits,
                      target_liquid_kg, delta=0.001) -> Dict[str, float]:
        """
        Rs/t of relaxing each tramp constraint by `delta` wt-%. This is the
        number to show the plant owner: "your 0.25 % Cu ceiling costs you
        Rs 340/t; a 0.30 % ceiling would save Rs X — is your customer's spec
        really 0.25, or is that a habit?"
        """
        out = {}
        for el, hi in tramp_limits.items():
            tl = dict(tramp_limits); tl[el] = hi + delta
            r = self.solve(target_liquid_kg, aim, tl)
            if r.feasible:
                out[el] = (base.cost_INR_per_t_liquid - r.cost_INR_per_t_liquid) / delta
        return out
