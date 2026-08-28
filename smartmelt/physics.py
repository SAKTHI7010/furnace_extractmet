"""
physics.py — Layer 1: the coupled mass + energy + species + lining model.

State vector  x  (all SI-ish, masses in kg, temperatures in K):

    x[0 : nM]                 metal species masses      m_i        [kg]
    x[nM : nM+nS]             slag  species masses      m_j        [kg]
    x[iTb]                    bath temperature          T_b        [K]
    x[iMs]                    unmelted solid charge     m_s        [kg]
    x[iTs]                    solid charge temperature  T_s        [K]
    x[iLin : iLin+N]          lining node temperatures  T_w,k      [K]
    x[iE]                     cumulative grid energy               [kWh]
    x[iCO], x[iCO2]           cumulative gas produced              [kg]

Structure of the RHS (eq. numbers -> docs/SmartMelt_Mathematical_Model.md):

    E1   d m_i / dt   = a_dot_i - r_i        (metal species)
    E2   d m_j / dt   = b_dot_j + sum_i nu_ij r_i
    E3   M_l cp_l dT_b/dt = P_use + Q_chem + Q_pc - Q_s - Q_wall - Q_rad
                              - Q_gas + m_melt_dot cp_l (T_liq - T_b)
    E4   m_s cp_s dT_s/dt = Q_s                       (T_s < T_liq)
         d m_s / dt       = -Q_s / L_f                (T_s = T_liq)
    E5   rho cp dT_w/dt   = (1/r) d/dr ( k r dT_w/dr )   -> FV discretisation

Oxygen bookkeeping (the heart of the model, sec. 4.4 of the doc): the lance is
an oxygen *source*, slag FeO is an oxygen *buffer*. Rates of Si/Mn/P/C oxidation
are set by mass transfer to equilibrium; whatever oxygen those rates demand is
drawn from the lance first and the FeO buffer second. Surplus lance oxygen
becomes FeO. This single rule reproduces the classic BOF/EAF phenomenology:
silicon-first, the carbon plateau, the FeO rise at low carbon.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from . import thermo as th
from .thermo import MW, KELVIN, SIGMA_SB
from .config import PlantConfig


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
@dataclass
class Addition:
    """
    A discrete charge / alloy / flux addition WITH dissolution kinetics.

    Nothing dissolves instantly in a 1600 C bath. A lump of FeSi first freezes
    a steel shell around itself, the shell melts back, then the lump dissolves;
    lime sinters and dissolves via a calcium-silicate rim; carburiser dissolves
    only as fast as carbon diffuses through the boundary layer. We model each
    addition as first-order release with time constant tau (E7a):

        dm_undis/dt = -m_undis / tau_eff
        tau_eff = tau_s * max(1, dT_ref / max(T_b - T_liq, 2 K))        (E7b)

    i.e. dissolution STALLS when the bath has no superheat — which is exactly
    when operators over-add and then wonder where the alloy went (it turns up
    ten minutes later as an off-spec high). tau=0 recovers the impulse model.

    dH_dissolution_kJ_kg is the total heat sink per kg (sensible to bath T +
    fusion + heat of mixing; negative for exothermic dissolvers like FeSi75).
    """
    time_s: float
    mass_kg: float
    composition: Dict[str, float]      # mass fractions, keys = metal or slag species
    into: str = "metal"                # "metal" | "slag" | "solid"
    temperature_C: float = 30.0
    label: str = ""
    tau_s: float = 0.0                 # dissolution time constant; 0 = instant
    dT_ref_K: float = 25.0             # superheat scale in E7b
    dH_dissolution_kJ_kg: Optional[float] = None   # INTRINSIC heat of solution only
    cp_kJ_kgK: Optional[float] = None  # effective cp (sensible + fusion) of the
                                       # addition; None -> use the bath phase cp


# Typical dissolution time constants at ~30-50 K superheat, lump sizes as
# delivered. Tune per plant from addition->response lag in the logged data.
# Dissolution presets. IMPORTANT CONVENTION (corrected v0.5):
#   dH_dissolution_kJ_kg is the INTRINSIC HEAT OF SOLUTION only.
#   The sensible heat of raising the cold addition to bath temperature is a
#   SEPARATE term (E7c), computed with the per-addition cp_kJ_kgK below, which
#   is an EFFECTIVE cp lumping sensible heat + any fusion of the addition.
# Conflating the two was the root cause of the previous FeSi75 (-1150) and
# carburiser (+2500) errors, both of which were neither a clean heat of
# solution nor a clean total cold-charge load.
#
# Anchors (Sigworth & Elliott 1974; Turkdogan, Fundamentals of Steelmaking):
#   Si(l) = [Si]_1wt%   dH ~ -131.5 kJ/mol Si  -> -4681 kJ/kg Si
#                       -> FeSi75 ~ -3511 kJ/kg alloy (75 % Si)
#   C(gr) = [C]_1wt%    dH ~  +22.6 kJ/mol C   -> +1883 kJ/kg C
# Resulting NET effect on the bath for a cold charge (heat of solution +
# sensible) reproduces the industrially observed values:
#   FeSi75    net ~ -0.70 MJ/kg  (cf. +4.73 C per tonne FeSi75 in a 172 t ladle,
#                                 Bernhard et al., Metall. Mater. Trans. B 56
#                                 (2025) 2249, DOI 10.1007/s11663-024-03419-1)
#   carburiser total ~ +4.6 MJ/kg C
DISSOLUTION_PRESETS = {
    "lime":        dict(tau_s=240.0, dH_dissolution_kJ_kg=+150.0,  cp_kJ_kgK=1.00),
    "dolomite":    dict(tau_s=300.0, dH_dissolution_kJ_kg=+400.0,  cp_kJ_kgK=1.05),
    "FeSi75":      dict(tau_s=90.0,  dH_dissolution_kJ_kg=-3511.0, cp_kJ_kgK=1.78),
    "FeMn":        dict(tau_s=120.0, dH_dissolution_kJ_kg=-30.0,   cp_kJ_kgK=0.95),
    "SiMn":        dict(tau_s=110.0, dH_dissolution_kJ_kg=-1400.0, cp_kJ_kgK=1.20),
    "carburiser":  dict(tau_s=420.0, dH_dissolution_kJ_kg=+1883.0, cp_kJ_kgK=1.75),
    "DRI":         dict(tau_s=180.0, dH_dissolution_kJ_kg=+1100.0, cp_kJ_kgK=0.85),
    "mill_scale":  dict(tau_s=120.0, dH_dissolution_kJ_kg=+250.0,  cp_kJ_kgK=0.90),
    #   mill scale: only the SENSIBLE + melting load sits here. The reaction
    #   endotherm of (FeO)+[C]->Fe+CO is booked separately by the oxygen
    #   ledger (E27c) as the dissolved FeO is drawn down.
    "pig_iron":    dict(tau_s=150.0, dH_dissolution_kJ_kg=+30.0,   cp_kJ_kgK=0.85),
}


def make_addition(time_s, mass_kg, composition, kind, into="metal", **kw) -> Addition:
    """Addition with kinetics looked up from DISSOLUTION_PRESETS."""
    preset = DISSOLUTION_PRESETS.get(kind, {})
    return Addition(time_s, mass_kg, composition, into=into, label=kind,
                    **{**preset, **kw})


@dataclass
class HeatInputs:
    """Control trajectory u(t) for one heat."""
    power_kW: Callable[[float], float] = lambda t: 0.0
    oxygen_Nm3_per_h: Callable[[float], float] = lambda t: 0.0
    additions: List[Addition] = field(default_factory=list)
    p_CO_atm: float = 1.0

    def copy_with_power(self, f):
        return HeatInputs(f, self.oxygen_Nm3_per_h, list(self.additions), self.p_CO_atm)


@dataclass
class Trajectory:
    t: np.ndarray
    X: np.ndarray                       # (n_steps, n_state)
    diagnostics: Dict[str, np.ndarray]
    undissolved_kg: float = 0.0         # additions still solid at tap (a defect!)

    def series(self, key: str) -> np.ndarray:
        return self.diagnostics[key]


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class FurnaceModel:
    """
    Physics core. Deterministic, differentiable-by-finite-difference, and cheap:
    one RHS evaluation is a few tens of microseconds, so an RK4 second is ~4
    evaluations => comfortably inside the <=100 ms inference budget with room
    for the ML head.

    `theta` holds the online-identifiable parameters. Keep this list short —
    see sec. 6.2 (identifiability) in the doc. Anything you cannot excite with
    normal plant operation should stay a fixed config constant.
    """

    THETA_KEYS = ("eta_electrical", "UA_lining_scale", "k_C_scale",
                  "h_solid_liquid_scale", "gamma_FeO")

    def __init__(self, cfg: PlantConfig, theta: Optional[Dict[str, float]] = None):
        self.cfg = cfg
        self.metal = list(cfg.metal_species)
        self.slag = list(cfg.slag_species)
        self.nM, self.nS = len(self.metal), len(self.slag)
        self.N = cfg.lining.n_nodes

        self.iTb = self.nM + self.nS
        self.iMs = self.iTb + 1
        self.iTs = self.iMs + 1
        self.iLin = self.iTs + 1
        self.iE = self.iLin + self.N
        self.iCO = self.iE + 1
        self.iCO2 = self.iCO + 1
        self.n_state = self.iCO2 + 1

        self.theta = {"eta_electrical": 1.0, "UA_lining_scale": 1.0,
                      "k_C_scale": 1.0, "h_solid_liquid_scale": 1.0,
                      "gamma_FeO": cfg.slag.gamma_FeO}
        if theta:
            self.theta.update(theta)

        # composition of the unmelted solid charge (mass fractions); not a state
        # variable because it is known from the weighed charge sheet.
        self.solid_comp = np.zeros(self.nM)
        self.solid_comp[self.metal.index("Fe")] = 1.0

        # (E5a) heat-transfer areas from furnace design, unless overridden.
        g = cfg.geometry
        self.A_wall = cfg.thermal.A_wall_m2 if cfg.thermal.A_wall_m2 > 0 \
            else g.A_wall_wetted_m2()
        self.A_top = cfg.thermal.A_top_m2 if cfg.thermal.A_top_m2 > 0 \
            else g.A_top_open_m2()
        cfg.lining.r_inner_m = g.D_inner_m / 2.0

        self._build_lining_mesh()

    # ------------------------------------------------------------------
    # Lining mesh (E5)
    # ------------------------------------------------------------------
    def _build_lining_mesh(self):
        """
        (E5) Multi-layer, geometrically graded radial finite-volume mesh.

        Layers come from cfg.lining.layers (working lining -> backup -> grout
        or shell), each with its own k, rho, cp, so an IF silica ramming mass
        and an EAF MgO-C brick are just different YAML. Nodes are distributed
        across the TOTAL thickness with geometric grading (thin at the hot
        face — only millimetres of refractory swing with the bath on a heat
        timescale; a uniform mesh puts ~1.5 MJ/K on node 0 and lags the hot
        face by tens of minutes). Each node then inherits the properties of
        the layer its centre falls in; interface conduction uses the series
        (harmonic) resistance of the two adjacent half-cells, which is exact
        for piecewise-constant k.
        """
        lc = self.cfg.lining
        total = sum(l.thickness_m for l in lc.layers)
        thick = total * (1.0 - lc.wear_fraction)
        r0 = lc.r_inner_m + total * lc.wear_fraction
        w = lc.grading_ratio ** np.arange(self.N)
        w = w / w.sum() * thick
        edges = np.concatenate([[r0], r0 + np.cumsum(w)])
        self.r_c = 0.5 * (edges[:-1] + edges[1:])          # node centres
        self.dr = np.diff(edges)

        # assign layer properties by node-centre position
        bounds = np.cumsum([0.0] + [l.thickness_m for l in lc.layers]) \
            * (1.0 - lc.wear_fraction)
        depth = self.r_c - r0
        self.k_node = np.empty(self.N)
        rho_cp = np.empty(self.N)
        self.node_layer = []
        for i, d in enumerate(depth):
            j = int(np.searchsorted(bounds, d, side="right") - 1)
            j = min(max(j, 0), len(lc.layers) - 1)
            lay = lc.layers[j]
            self.k_node[i] = lay.k_W_mK
            rho_cp[i] = lay.rho_kg_m3 * lay.cp_kJ_kgK * 1e3   # J/m3K
            self.node_layer.append(lay.name)

        H = self.A_wall / (2 * np.pi * lc.r_inner_m)        # effective height
        self.A_face = 2 * np.pi * edges * H                 # face areas
        self.V_node = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2) * H
        self.C_node = rho_cp * self.V_node                  # J/K per node
        # inter-node conductances G[k] (W/K), series (harmonic) across interfaces
        self.G_node = np.array([
            self.A_face[k] / (self.dr[k - 1] / (2 * self.k_node[k - 1])
                              + self.dr[k] / (2 * self.k_node[k]))
            for k in range(1, self.N)])

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def initial_state(self, charge_kg: float, charge_comp: Dict[str, float],
                      T_charge_C: float = 30.0,
                      hot_heel_kg: float = 0.0,
                      T_lining_C: Optional[float] = None) -> np.ndarray:
        cfg = self.cfg
        x = np.zeros(self.n_state)

        comp = np.array([charge_comp.get(sp, 0.0) for sp in self.metal])
        i_fe = self.metal.index("Fe")
        if comp[i_fe] <= 0.0:
            comp[i_fe] = 1.0 - comp.sum()
        comp = np.maximum(comp, 0.0)
        comp /= comp.sum()
        self.solid_comp = comp

        heel = max(hot_heel_kg, 1e-3)
        x[:self.nM] = heel * comp

        slag_mass = cfg.slag.initial_slag_kg_per_t * (charge_kg / 1000.0)
        for j, sp in enumerate(self.slag):
            x[self.nM + j] = slag_mass * cfg.slag.initial_composition.get(sp, 0.0)

        x[self.iTb] = cfg.thermal.T_liquidus_C + 30.0 + KELVIN
        x[self.iMs] = charge_kg - heel
        x[self.iTs] = T_charge_C + KELVIN
        Tl = T_lining_C if T_lining_C is not None else cfg.thermal.T_ambient_C + 200.0
        x[self.iLin:self.iLin + self.N] = np.linspace(
            Tl + 400.0, cfg.lining.T_coolant_C + 40.0, self.N) + KELVIN
        return x

    def unpack(self, x: np.ndarray) -> dict:
        m_metal = np.maximum(x[:self.nM], 0.0)
        m_slag = np.maximum(x[self.nM:self.nM + self.nS], 0.0)
        M_l = max(m_metal.sum(), 1e-6)
        return dict(
            m_metal=m_metal, m_slag=m_slag, M_l=M_l,
            pct=dict(zip(self.metal, 100.0 * m_metal / M_l)),
            slag_kg=dict(zip(self.slag, m_slag)),
            T_b=x[self.iTb], m_s=max(x[self.iMs], 0.0), T_s=x[self.iTs],
            T_lin=x[self.iLin:self.iLin + self.N],
            E_kWh=x[self.iE],
        )

    # ------------------------------------------------------------------
    # Electrical (E21-E23)
    # ------------------------------------------------------------------
    def _useful_power_kW(self, P_grid: float, M_l: float, m_s: float) -> tuple:
        el = self.cfg.electrical
        P_coil = P_grid * el.eta_converter
        if self.cfg.plant.furnace_type == "IF":
            fill = np.clip((M_l + m_s) / (self.cfg.plant.max_charge_t * 1000.0), 0, 1.2)
            # E21: eta_coup(fill) = eta_max (1 - exp(-fill/fill_ref))
            eta_c = el.eta_coupling_max * (1.0 - np.exp(-fill / el.coupling_fill_ref))
        else:
            # E22: EAF arc efficiency rises as scrap shields the arc from the wall
            cover = np.clip(m_s / max(M_l + m_s, 1e-6), 0, 1)
            eta_c = el.eta_arc_bath + el.eta_arc_foamed_bonus * cover
        eta_c *= self.theta["eta_electrical"]
        P_use = P_coil * eta_c
        return P_use, P_coil - P_use, P_grid - P_coil   # useful, cooling loss, converter loss

    def _power_split(self, P_use: float, M_l: float, m_s: float) -> tuple:
        """
        (E23) Where does the useful power actually land?

        IF : the induced field couples to every electrically conductive mass in
             the crucible. Solid scrap is heated DIRECTLY, not via the pool.
             Share it by mass. Getting this wrong is the classic error — it
             makes the model stiff (all the melting heat has to pass through a
             tiny heel) and predicts melt-down times that are 3x too long.
        EAF: the arc heats the bath; a fraction radiates onto the scrap pile,
             scaled by how much scrap is still shielding the arc.
        """
        f_s = m_s / max(M_l + m_s, 1e-6)
        if self.cfg.plant.furnace_type == "IF":
            share = f_s
        else:
            share = 0.5 * f_s
        return P_use * (1.0 - share), P_use * share      # to liquid, to solid

    # ------------------------------------------------------------------
    # Chemistry (E24-E30)
    # ------------------------------------------------------------------
    def _oxidation_rates(self, s: dict, O2_Nm3_h: float, p_CO: float) -> dict:
        """
        Returns kg/s oxidation rates for C, Si, Mn, P and the net FeO change,
        after the lance/FeO oxygen ledger is balanced.
        """
        kin, sl_cfg = self.cfg.kinetics, self.cfg.slag
        T, pct, slag_kg = s["T_b"], s["pct"], s["slag_kg"]
        gamma = self.theta["gamma_FeO"]

        # (E24a) reaction switch: slag-metal chemistry needs a liquid bath and a
        # slag-metal interface. Both vanish at charge-in. Without this gate the
        # model runs BOF chemistry on a pile of cold scrap.
        T_sol = self.cfg.thermal.T_solidus_C + KELVIN
        f_liq = s["M_l"] / max(s["M_l"] + s["m_s"], 1e-6)
        switch = f_liq / (1.0 + np.exp(-(T - T_sol) / 15.0))
        rho_A = (kin.rho_metal_kg_m3 * kin.A_slag_metal_m2
                 * kin.stirring_multiplier * switch)

        # --- E24: mass-transfer-limited rates toward equilibrium ------------
        def mt(k, sp, pct_eq):
            drive = (pct.get(sp, 0.0) - pct_eq) / 100.0
            return max(rho_A * k * drive, 0.0)

        C_eq = th.pct_C_equilibrium(slag_kg, gamma, T, pct, p_CO)
        Si_eq = th.pct_Si_equilibrium(slag_kg, gamma, T, pct)
        Mn_eq = th.pct_Mn_equilibrium(slag_kg, gamma, T, pct)

        r_Si = mt(kin.k_Si, "Si", Si_eq)
        r_Mn = mt(kin.k_Mn, "Mn", Mn_eq)

        # E25: decarburisation, two-regime. Above C_crit the rate is limited by
        # oxygen supply; below it, by carbon transport to the reaction site.
        k_C = kin.k_C * self.theta["k_C_scale"]
        r_C_mt = mt(k_C, "C", C_eq)
        if pct.get("C", 0.0) > kin.C_critical_pct:
            r_C_mt *= 1.0 + 2.0 * (pct["C"] - kin.C_critical_pct)  # supply-rich regime

        # E26: phosphorus via Healy partition, relaxation kinetics
        L_P = th.L_P_healy(slag_kg, T)
        tot_slag = max(sum(slag_kg.values()), 1e-6)
        pct_P_slag = 100.0 * slag_kg.get("P2O5", 0.0) * (2 * MW["P"] / MW["P2O5"]) / tot_slag
        P_eq = pct_P_slag / max(L_P, 1e-6)
        r_P = max(rho_A * kin.k_P * (pct.get("P", 0.0) - P_eq) / 100.0, 0.0)

        # E29: desulphurisation, (CaO) + [S] = (CaS) + [O].
        # Sign is NOT forced positive: at high a_O the slag gives sulphur back.
        # That reversal is real, and a one-sided rate law hides it.
        Ls = th.L_S(slag_kg, gamma, T, pct)
        pct_S_slag = 100.0 * slag_kg.get("CaS", 0.0) * (MW["S"] / MW["CaS"]) / tot_slag
        S_eq = pct_S_slag / max(Ls, 1e-6)
        r_S = rho_A * kin.k_S * (pct.get("S", 0.0) - S_eq) / 100.0
        # reverse (slag -> metal) transfer cannot exceed the sulphur actually
        # held in the slag; drain it no faster than 1/60 s^-1.
        S_in_slag = slag_kg.get("CaS", 0.0) * MW["S"] / MW["CaS"]
        r_S = max(r_S, -S_in_slag / 60.0)

        # --- E27: oxygen ledger --------------------------------------------
        # mol of O ATOMS per second demanded by the four oxidation reactions.
        #   C  + O      -> CO      : 1 O per C
        #   Si + 2O     -> SiO2    : 2 O per Si
        #   Mn + O      -> MnO     : 1 O per Mn
        #   2P + 5O     -> P2O5    : 2.5 O per P
        # r_* are kg/s; MW is g/mol; hence the factor 1000.
        demand = 1000.0 * (r_C_mt / MW["C"] + 2.0 * r_Si / MW["Si"]
                           + r_Mn / MW["Mn"] + 2.5 * r_P / MW["P"])

        # E27a: oxygen supply = lance + air ingress. In an IF the lance term is
        # zero and air ingress is the ONLY oxygen source — it is what keeps a few
        # per cent FeO in the slag and lets the model reproduce IF melt loss.
        air = (self.cfg.offgas.air_ingress_Nm3_per_t_per_h
               * self.cfg.plant.heat_size_t * 0.21)              # Nm3 O2 / h
        supply = ((O2_Nm3_h * kin.eta_O2_utilisation + air)
                  / 3600.0 / 0.022414 * 2.0)                      # mol O/s

        FeO_kg = slag_kg.get("FeO", 0.0)
        scale = 1.0
        dFeO_ox = 0.0
        if supply >= demand:
            # E27b: only a fraction of surplus lance oxygen ends up as slag FeO.
            # The rest leaves as free O2 / burns CO in the freeboard. And the
            # slag saturates: above X_FeO ~ 0.45 the iron oxidises into an
            # emulsion that reduces straight back, so the net sink closes.
            surplus = supply - demand
            X = th.slag_mole_fractions(slag_kg)
            sat = float(np.clip(1.0 - X.get("FeO", 0.0) / 0.45, 0.0, 1.0))
            f_to_FeO = sl_cfg.surplus_O2_to_FeO * sat
            dFeO_ox = surplus * f_to_FeO * MW["FeO"] / 1000.0   # kg/s FeO formed
        else:
            deficit = demand - supply                        # mol O/s from FeO
            feo_avail_mol = FeO_kg / MW["FeO"] * 1000.0 / 1.0
            max_rate = feo_avail_mol / 60.0                  # drain no faster than 1/min
            if deficit > max_rate:
                scale = max(max_rate / max(deficit, 1e-9), 0.0)
                deficit = max_rate
            dFeO_ox = -deficit * MW["FeO"] / 1000.0

        r_C_mt *= scale
        r_Si *= scale
        r_Mn *= scale
        r_P *= scale

        return dict(r_C=r_C_mt, r_Si=r_Si, r_Mn=r_Mn, r_P=r_P, r_S=r_S,
                    dFeO_ox=dFeO_ox, C_eq=C_eq, L_P=L_P, L_S=Ls,
                    a_FeO=th.a_FeO(slag_kg, gamma))

    # ------------------------------------------------------------------
    # RHS
    # ------------------------------------------------------------------
    def rhs(self, t: float, x: np.ndarray, u: HeatInputs, diag: Optional[dict] = None):
        cfg = self.cfg
        thc, lc, og, en = cfg.thermal, cfg.lining, cfg.offgas, cfg.enthalpy
        s = self.unpack(x)
        dx = np.zeros_like(x)

        P_grid = float(u.power_kW(t))
        O2 = float(u.oxygen_Nm3_per_h(t))
        T_b, m_s, T_s, M_l = s["T_b"], s["m_s"], s["T_s"], s["M_l"]

        P_use, P_cool, P_conv = self._useful_power_kW(P_grid, M_l, m_s)

        # ---- chemistry -----------------------------------------------
        ox = self._oxidation_rates(s, O2, u.p_CO_atm)
        r_C, r_Si, r_Mn, r_P, r_S = (ox["r_C"], ox["r_Si"], ox["r_Mn"],
                                     ox["r_P"], ox["r_S"])

        # metal species
        for sp, r in (("C", r_C), ("Si", r_Si), ("Mn", r_Mn), ("P", r_P), ("S", r_S)):
            if sp in self.metal:
                dx[self.metal.index(sp)] -= r
        i_Fe = self.metal.index("Fe")
        dFe_to_FeO = ox["dFeO_ox"] * MW["Fe"] / MW["FeO"]
        dx[i_Fe] -= dFe_to_FeO

        # slag species
        def sl(sp): return self.nM + self.slag.index(sp) if sp in self.slag else None
        if sl("SiO2") is not None: dx[sl("SiO2")] += r_Si * MW["SiO2"] / MW["Si"]
        if sl("MnO") is not None:  dx[sl("MnO")] += r_Mn * MW["MnO"] / MW["Mn"]
        if sl("P2O5") is not None: dx[sl("P2O5")] += r_P * MW["P2O5"] / (2 * MW["P"])
        if sl("FeO") is not None:  dx[sl("FeO")] += ox["dFeO_ox"]
        if sl("CaS") is not None:
            dx[sl("CaS")] += r_S * MW["CaS"] / MW["S"]
            if sl("CaO") is not None:
                dx[sl("CaO")] -= r_S * MW["CaO"] / MW["S"]

        # gases
        m_CO = r_C * MW["CO"] / MW["C"]
        m_CO_burn = og.post_combustion_ratio * m_CO
        m_CO2 = m_CO_burn * MW["CO2"] / MW["CO"]
        dx[self.iCO] = m_CO - m_CO_burn
        dx[self.iCO2] = m_CO2

        # ---- heat sources (kW) ---------------------------------------
        # (E27c) The SIGN of the FeO term matters and is a classic mistake:
        # when oxygen comes from the lance/air, Fe -> FeO releases its
        # formation enthalpy (dFe_to_FeO > 0, exothermic). But when the ledger
        # runs a DEFICIT and the slag supplies the oxygen — ore / mill-scale
        # practice — the same enthalpy must be PAID BACK to decompose FeO
        # (dFe_to_FeO < 0). Net, by Hess's law:
        #     (FeO) + [C] -> Fe(l) + CO(g)
        #     dH = dH(C->CO) - (56/12) * dH(Fe->FeO) ~ -11.3 MJ per kg C,
        # i.e. ENDOTHERMIC (~ +101 kJ/mol CO at 1873 K) — which is exactly why
        # mill scale and ore are used as bath coolants. Booking decarb-by-FeO
        # as exothermic (the old max(.,0) clamp) violates the first law and
        # shows up as a phantom heat source whenever ore is charged.
        Q_chem = (r_C * en.C_to_CO + r_Si * en.Si_to_SiO2 + r_Mn * en.Mn_to_MnO
                  + r_P * en.P_to_P2O5 + dFe_to_FeO * en.Fe_to_FeO)
        Q_pc = m_CO_burn * en.CO_to_CO2_per_kg_CO * og.eta_pc_heat_return

        # ---- heat sinks ----------------------------------------------
        T_liq = thc.T_liquidus_C + KELVIN
        P_liq, P_solid = self._power_split(P_use, M_l, m_s)

        # (E28) melt<->scrap contact area needs BOTH phases present:
        #   A_eff = A_ref * (m_s/M_max)^(1/2) * f_liq^(1/3)
        # Zero at charge-in (no pool) and zero at melt-down (no scrap). The 1/2
        # exponent (not 2/3) is deliberate: shop scrap is plate- and turning-like,
        # so specific surface RISES as a piece thins. A 2/3 (sphere) exponent
        # makes the last 100 kg melt asymptotically and the bath run away — an
        # artefact that shows up as a spurious "overheat before tap".
        M_max = max(cfg.plant.max_charge_t * 1000.0, 1e-6)
        f_liq = M_l / max(M_l + m_s, 1e-6)
        A_s = thc.A_solid_ref_m2 * np.sqrt(m_s / M_max) * f_liq ** (1 / 3)
        h_sl = thc.h_solid_liquid_W_m2K * self.theta["h_solid_liquid_scale"]
        # (E28b) flux cap. h*dT with dT ~ 1400 K would give MW/m^2. In reality a
        # solidified shell forms on the cold scrap within milliseconds and the
        # shell, not the liquid film, sets the resistance. Capping the flux at
        # q_max is the cheap surrogate for a shell-growth sub-model.
        Q_s = min(h_sl * max(T_b - T_s, 0.0), thc.q_max_scrap_kW_m2 * 1e3) * A_s / 1000.0

        m_melt = 0.0
        if m_s > 1.0:
            Q_into_solid = Q_s + P_solid
            if T_s < T_liq - 1e-6:
                # E4a: sensible heating of the solid charge
                dx[self.iTs] = Q_into_solid / max(m_s * thc.cp_solid_kJ_kgK, 1e-3)
            else:
                # E4b: isothermal melting at T_liq
                dx[self.iTs] = 0.0
                m_melt = max(Q_into_solid, 0.0) / thc.L_fusion_kJ_kg   # kg/s
                dx[self.iMs] = -m_melt
                dx[:self.nM] += m_melt * self.solid_comp
        else:
            Q_s, P_solid, P_liq = 0.0, 0.0, P_use

        # lining (E5): the ENTIRE bath<->wall and scrap<->wall exchange is
        # handled by the implicit solve in step() (E5d) and debited from the
        # bath/scrap there, so both sides see EXACTLY the same joules — this
        # exact pairing is what makes the first-law audit (E62) close to <1 %.
        # rhs therefore carries no wall term at all.
        dx[self.iLin:self.iLin + self.N] = 0.0
        Q_wall = 0.0

        Q_rad = thc.emissivity_top * SIGMA_SB * self.A_top * \
            (T_b ** 4 - (thc.T_ambient_C + KELVIN) ** 4) / 1000.0
        m_gas = m_CO + m_CO2
        Q_gas = m_gas * og.cp_gas_kJ_kgK * max(T_b - (thc.T_ambient_C + KELVIN), 0.0)

        # ---- bath temperature (E3) -----------------------------------
        # Floor on the heat capacity: with a near-zero heel the ODE is stiff and
        # a 1 s explicit step will ring. The floor is numerical, not physical —
        # it only bites for M_l < 50 kg, i.e. before a pool exists.
        C_bath = max(M_l, 50.0) * thc.cp_liquid_kJ_kgK           # kJ/K
        Q_net = (P_liq + Q_chem + Q_pc - Q_s - Q_rad - Q_gas
                 + m_melt * thc.cp_liquid_kJ_kgK * (T_liq - T_b))
        if M_l < self.M_MIN_BATH_KG:
            # (E3c) no meaningful liquid pool exists (charge-in, or the heel has
            # frozen onto the cold scrap — "bolting", which is real practice,
            # not a failure). A bath temperature is undefined here; relax it to
            # the solid temperature so that when melting resumes, liquid appears
            # at the right enthalpy instead of at an integrator artefact.
            dx[self.iTb] = (max(T_s, thc.T_ambient_C + KELVIN) - T_b) / 30.0
        elif T_b <= T_liq and Q_net < 0.0:
            # (E3b) enthalpy method: a bath at the liquidus with a net heat
            # deficit FREEZES, it does not go sub-liquidus. Skipping this branch
            # is how melt models produce 400 C baths and nonsense chemistry.
            m_freeze = -Q_net / thc.L_fusion_kJ_kg               # kg/s
            m_freeze = min(m_freeze, (M_l - self.M_MIN_BATH_KG / 2) / 10.0)
            dx[self.iTb] = 0.0
            dx[self.iMs] += m_freeze
            dx[:self.nM] -= m_freeze * (s["m_metal"] / M_l)
            m_melt -= m_freeze
        else:
            dx[self.iTb] = Q_net / C_bath

        dx[self.iE] = P_grid / 3600.0                            # kWh/s

        if diag is not None:
            diag.update(P_use=P_use, P_liq=P_liq, P_solid=P_solid,
                        P_cool=P_cool, P_conv=P_conv,
                        Q_chem=Q_chem, Q_pc=Q_pc, Q_s=Q_s,
                        Q_rad=Q_rad, Q_gas=Q_gas, r_C=r_C, r_Si=r_Si,
                        r_Mn=r_Mn, r_P=r_P, r_S=r_S, C_eq=ox["C_eq"],
                        a_FeO=ox["a_FeO"], L_P=ox["L_P"], L_S=ox["L_S"],
                        m_melt=m_melt, T_hotface=s["T_lin"][0] - KELVIN,
                        f_liq=f_liq)
        return dx

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------
    def _apply_addition(self, x, add: Addition):
        thc = self.cfg.thermal
        if add.into == "solid":
            x[self.iMs] += add.mass_kg
            m_old = max(x[self.iMs] - add.mass_kg, 1e-6)
            x[self.iTs] = (x[self.iTs] * m_old +
                           (add.temperature_C + KELVIN) * add.mass_kg) / (m_old + add.mass_kg)
            return x
        if add.into == "metal":
            for sp, frac in add.composition.items():
                if sp in self.metal:
                    x[self.metal.index(sp)] += add.mass_kg * frac
            # sensible-heat penalty for cold alloy
            M_l = max(x[:self.nM].sum(), 1e-6)
            dQ = add.mass_kg * thc.cp_solid_kJ_kgK * (x[self.iTb] - (add.temperature_C + KELVIN))
            x[self.iTb] -= dQ / (M_l * thc.cp_liquid_kJ_kgK)
        else:  # slag / flux
            for sp, frac in add.composition.items():
                if sp in self.slag:
                    x[self.nM + self.slag.index(sp)] += add.mass_kg * frac
            M_l = max(x[:self.nM].sum(), 1e-6)
            dQ = add.mass_kg * self.cfg.thermal.cp_slag_kJ_kgK * \
                (x[self.iTb] - (add.temperature_C + KELVIN))
            x[self.iTb] -= dQ / (M_l * thc.cp_liquid_kJ_kgK)
        return x

    MAX_DT_TB_K = 5.0          # sub-step so |dT_b| per sub-step <= 5 K
    MAX_SUBSTEPS = 25
    M_MIN_BATH_KG = 20.0       # below this, no bath: T_b is slaved to T_s (E3c)

    def _rk4(self, t, x, u, dt, diag=None):
        k1 = self.rhs(t, x, u, diag)
        k2 = self.rhs(t + dt / 2, x + dt / 2 * k1, u)
        k3 = self.rhs(t + dt / 2, x + dt / 2 * k2, u)
        k4 = self.rhs(t + dt, x + dt * k3, u)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), k1

    # ------------------------------------------------------------------
    # (E5d) Implicit wall update — unconditionally stable, exact tridiagonal
    # ------------------------------------------------------------------
    def _update_lining_implicit(self, x: np.ndarray, dt: float) -> None:
        """
        Backward Euler on   C_i dT_i/dt = G_{i-1/2}(T_{i-1}-T_i)
                                        + G_{i+1/2}(T_{i+1}-T_i) + b_i
        with the bath/scrap side and the coil/shell side entering as Robin
        boundary conditions. Driving temperatures (T_b, T_s, coolant, ambient)
        are held over the sub-step (operator splitting); shell radiation is
        linearised about the current shell temperature, h_rad = eps*sigma*
        (T^2+Ta^2)(T+Ta), which is Newton's first iterate and is ample at
        these time steps. Solved directly (N<=12) — no stability limit, so the
        hot-face node can be as thin as the physics wants it to be.
        """
        lc, thc = self.cfg.lining, self.cfg.thermal
        N = self.N
        T = x[self.iLin:self.iLin + N].copy()
        M_l = max(x[:self.nM].sum(), 0.0)
        m_s = max(x[self.iMs], 0.0)
        f_l = M_l / max(M_l + m_s, 1e-6)
        UA = self.theta["UA_lining_scale"]

        # inner Robin BC: liquid + scrap contact in parallel
        h_in = (lc.h_inner_W_m2K * f_l + lc.h_solid_wall_W_m2K * (1 - f_l)) * UA
        G_in = h_in * self.A_face[0]
        w = lc.h_inner_W_m2K * f_l + lc.h_solid_wall_W_m2K * (1 - f_l)
        T_drive_in = (lc.h_inner_W_m2K * f_l * x[self.iTb]
                      + lc.h_solid_wall_W_m2K * (1 - f_l) * x[self.iTs]) / max(w, 1e-9)
        # outer Robin BC
        if lc.outer_bc == "coil":
            G_out = lc.h_outer_W_m2K * self.A_face[N]
            T_drive_out = lc.T_coolant_C + KELVIN
        else:
            Ta = thc.T_ambient_C + KELVIN
            h_rad = lc.shell_emissivity * SIGMA_SB * (T[-1] ** 2 + Ta ** 2) * (T[-1] + Ta)
            G_out = (lc.h_shell_conv_W_m2K + h_rad) * self.A_face[N]
            T_drive_out = Ta

        A = np.zeros((N, N))
        b = self.C_node / dt * T
        for i in range(N):
            A[i, i] = self.C_node[i] / dt
        A[0, 0] += G_in
        b[0] += G_in * T_drive_in
        for k, G in enumerate(self.G_node):          # between node k and k+1
            A[k, k] += G; A[k + 1, k + 1] += G
            A[k, k + 1] -= G; A[k + 1, k] -= G
        A[N - 1, N - 1] += G_out
        b[N - 1] += G_out * T_drive_out
        T_new = np.linalg.solve(A, b)
        x[self.iLin:self.iLin + N] = T_new

        # Exact energy exchanged at the hot face over dt (implicit flux), split
        # by phase weight and debited from bath / scrap so nothing is invented.
        q_face = G_in * (T_drive_in - T_new[0])                  # W (avg over dt)
        w_l = lc.h_inner_W_m2K * f_l / max(w, 1e-9)
        E_liq, E_sol = q_face * w_l * dt, q_face * (1 - w_l) * dt   # J
        thc2 = self.cfg.thermal
        M_lq = max(M_l, 50.0)
        x[self.iTb] -= (E_liq / 1000.0) / (M_lq * thc2.cp_liquid_kJ_kgK)
        if m_s > 1.0:
            x[self.iTs] -= (E_sol / 1000.0) / max(m_s * thc2.cp_solid_kJ_kgK, 1e-3)
        return q_face / 1000.0                                   # kW, for diag

    def step(self, t, x, u: HeatInputs, dt: float, diag=None) -> np.ndarray:
        """
        One control step of length dt, integrated with automatic sub-stepping.
        Charge-in (tiny liquid pool, huge heat fluxes) is stiff; melt-down and
        refining are not. Sub-stepping costs nothing when it is not needed and
        keeps a plain RK4 usable on an edge box — no implicit solver required.
        """
        d0 = {} if diag is None else diag
        k0 = self.rhs(t, x, u, d0)
        n_sub = int(np.clip(np.ceil(abs(k0[self.iTb]) * dt / self.MAX_DT_TB_K),
                            1, self.MAX_SUBSTEPS))
        h = dt / n_sub
        xn = x
        E_wall_kWs = 0.0
        acc: Dict[str, float] = {}
        for i in range(n_sub):
            di: dict = {}
            if self.cfg.numerics.integrator == "rk4":
                xn, _ = self._rk4(t + i * h, xn, u, h, di)
            else:                       # explicit Euler fallback (edge-cheap)
                xn = xn + h * self.rhs(t + i * h, xn, u, di)
            for k, v in di.items():
                acc[k] = acc.get(k, 0.0) + v / n_sub
            E_wall_kWs += self._update_lining_implicit(xn, h) * h
            xn[:self.nM + self.nS] = np.maximum(xn[:self.nM + self.nS], 0.0)
            xn[self.iMs] = max(xn[self.iMs], 0.0)
            # Temperature states must stay in a physically defensible band.
            # This is a numerical safety net, not a physics statement: if it
            # ever clips a healthy trajectory, MAX_DT_TB_K is set too loose,
            # not this bound.
            xn[self.iTb] = np.clip(xn[self.iTb], 250.0, 2500.0)
            xn[self.iTs] = np.clip(xn[self.iTs], 250.0, 2500.0)
            xn[self.iLin:self.iLin + self.N] = np.clip(
                xn[self.iLin:self.iLin + self.N], 250.0, 2500.0)
        d0.update(acc)                        # sub-step-averaged diagnostics
        d0["Q_wall"] = E_wall_kWs / dt        # average hot-face flux, kW
        return xn

    # ------------------------------------------------------------------
    # Dissolution kinetics (E7): operator splitting per control step
    # ------------------------------------------------------------------
    def _release_dissolving(self, x: np.ndarray, pool: list, dt: float) -> np.ndarray:
        """
        Transfer mass from the undissolved pool into the bath/slag over dt.

        Split from the ODE on purpose: the release is a stiff first-order decay
        whose EXACT solution over a step is m*(1-exp(-dt/tau_eff)), so handling
        it analytically between RK4 steps is both cheaper and more accurate
        than forcing the integrator to resolve it. Standard operator splitting;
        O(dt) splitting error is far below the tau uncertainty itself.
        """
        thc = self.cfg.thermal
        T_b = x[self.iTb]
        T_liq = thc.T_liquidus_C + KELVIN
        superheat = max(T_b - T_liq, 2.0)
        for item in pool:
            if item["m"] <= 1e-9:
                continue
            a: Addition = item["add"]
            tau_eff = max(a.tau_s, 1e-6) * max(1.0, a.dT_ref_K / superheat)  # E7b
            dm = item["m"] * (1.0 - np.exp(-dt / tau_eff))                   # E7a
            item["m"] -= dm
            x = self._transfer_mass(x, a, dm)
        return x

    def _transfer_mass(self, x: np.ndarray, add: Addition, dm: float) -> np.ndarray:
        """Move dm kg of an addition into the bath/slag with its heat sink (E7c)."""
        thc = self.cfg.thermal
        M_l = max(x[:self.nM].sum(), 1e-6)
        if add.into == "metal":
            for sp, frac in add.composition.items():
                if sp in self.metal:
                    x[self.metal.index(sp)] += dm * frac
            cp, dH_fus = thc.cp_solid_kJ_kgK, thc.L_fusion_kJ_kg
        else:
            for sp, frac in add.composition.items():
                if sp in self.slag:
                    x[self.nM + self.slag.index(sp)] += dm * frac
            cp, dH_fus = thc.cp_slag_kJ_kgK, 0.0
        if add.cp_kJ_kgK is not None:
            cp = add.cp_kJ_kgK          # effective cp incl. fusion of the addition
        if add.dH_dissolution_kJ_kg is not None:
            # (E7c) sensible heating of the cold addition to bath temperature
            # PLUS its intrinsic heat of solution — kept as separate physical
            # terms so neither can silently absorb the other.
            dQ = dm * (cp * (x[self.iTb] - (add.temperature_C + KELVIN))
                       + add.dH_dissolution_kJ_kg)
        else:
            dQ = dm * (cp * (x[self.iTb] - (add.temperature_C + KELVIN)) + dH_fus)
        x[self.iTb] -= dQ / (max(M_l, 50.0) * thc.cp_liquid_kJ_kgK)
        return x

    def simulate(self, x0: np.ndarray, u: HeatInputs, t_end_s: float,
                 dt: Optional[float] = None, stop_fn: Optional[Callable] = None) -> Trajectory:
        dt = dt or self.cfg.numerics.dt_s
        n = int(t_end_s / dt) + 1
        X = np.zeros((n, self.n_state))
        X[0] = x0.copy()
        keys = ["P_use", "P_liq", "P_solid", "P_cool", "P_conv", "Q_chem",
                "Q_pc", "Q_s", "Q_wall", "Q_rad", "Q_gas", "r_C", "r_Si",
                "r_Mn", "r_P", "r_S", "C_eq", "a_FeO", "L_P", "L_S",
                "m_melt", "T_hotface", "f_liq"]
        D = {k: np.zeros(n) for k in keys}
        pending = sorted(u.additions, key=lambda a: a.time_s)
        ai = 0
        pool: list = []                       # undissolved lumps in the bath
        D["m_undissolved"] = np.zeros(n)
        t = 0.0
        for i in range(1, n):
            while ai < len(pending) and pending[ai].time_s <= t:
                a = pending[ai]
                if a.into == "solid" or a.tau_s <= 0.0:
                    X[i - 1] = self._apply_addition(X[i - 1].copy(), a)
                else:
                    pool.append({"add": a, "m": a.mass_kg})
                ai += 1
            if pool:
                X[i - 1] = self._release_dissolving(X[i - 1].copy(), pool, dt)
            d = {}
            X[i] = self.step(t, X[i - 1], u, dt, d)
            for k in keys:
                D[k][i] = d.get(k, 0.0)
            D["m_undissolved"][i] = sum(p["m"] for p in pool)
            t += dt
            if stop_fn and stop_fn(t, X[i]):
                X, t_arr = X[:i + 1], np.arange(i + 1) * dt
                D = {k: v[:i + 1] for k, v in D.items()}
                traj = Trajectory(t_arr, X, D)
                self._attach_pool(traj, pool)
                return traj
        traj = Trajectory(np.arange(n) * dt, X, D)
        self._attach_pool(traj, pool)
        return traj

    @staticmethod
    def _attach_pool(traj, pool):
        traj.undissolved_kg = sum(p["m"] for p in pool)
        by_sp: Dict[str, float] = {}
        for p in pool:
            for sp, frac in p["add"].composition.items():
                by_sp[sp] = by_sp.get(sp, 0.0) + p["m"] * frac
        traj.undissolved_species = by_sp

    # ------------------------------------------------------------------
    # Convenience read-outs
    # ------------------------------------------------------------------
    def endpoint(self, traj: Trajectory) -> dict:
        x = traj.X[-1]
        s = self.unpack(x)
        tap_t = s["M_l"] / 1000.0
        return dict(
            T_C=s["T_b"] - KELVIN,
            pct_C=s["pct"].get("C", 0.0),
            pct_Si=s["pct"].get("Si", 0.0),
            pct_Mn=s["pct"].get("Mn", 0.0),
            pct_P=s["pct"].get("P", 0.0),
            pct_S=s["pct"].get("S", 0.0),
            pct_Cu=s["pct"].get("Cu", 0.0),
            tap_mass_t=tap_t,
            energy_kWh=x[self.iE],
            SEC_kWh_per_t=x[self.iE] / max(tap_t, 1e-6),
            tap_to_tap_min=traj.t[-1] / 60.0,
            B2=th.basicity_B2(s["slag_kg"]),
            pct_FeO_slag=100.0 * s["slag_kg"].get("FeO", 0) / max(sum(s["slag_kg"].values()), 1e-6),
            hot_face_C=s["T_lin"][0] - KELVIN,
        )

    def energy_audit(self, traj: Trajectory, dt: Optional[float] = None) -> dict:
        """Where the kWh went — the honest version of 'Figure 1' in the brief."""
        dt = dt or self.cfg.numerics.dt_s
        f = 1.0 / 3600.0 * dt
        d = traj.diagnostics
        grid = traj.X[-1, self.iE]
        out = {
            "grid_kWh": grid,
            "converter_loss_kWh": d["P_conv"].sum() * f,
            "coil_water_loss_kWh": d["P_cool"].sum() * f,
            "chemical_in_kWh": (d["Q_chem"].sum() + d["Q_pc"].sum()) * f,
            "lining_loss_kWh": d["Q_wall"].sum() * f,
            "radiation_loss_kWh": d["Q_rad"].sum() * f,
            "offgas_loss_kWh": d["Q_gas"].sum() * f,
        }
        s = self.unpack(traj.X[-1])
        thc = self.cfg.thermal
        useful = s["M_l"] * (thc.cp_solid_kJ_kgK * (thc.T_liquidus_C - thc.T_ambient_C)
                             + thc.L_fusion_kJ_kg
                             + thc.cp_liquid_kJ_kgK * (s["T_b"] - KELVIN - thc.T_liquidus_C)) / 3600.0
        out["useful_melt_kWh"] = useful
        out["useful_fraction"] = useful / max(grid, 1e-6)
        return out

    # ------------------------------------------------------------------
    # Conservation audits (E61, E62) — run these in CI and after any edit
    # ------------------------------------------------------------------
    ELEMENT_OF_OXIDE = {"FeO": ("Fe", MW["Fe"] / MW["FeO"]),
                        "SiO2": ("Si", MW["Si"] / MW["SiO2"]),
                        "MnO": ("Mn", MW["Mn"] / MW["MnO"]),
                        "P2O5": ("P", 2 * MW["P"] / MW["P2O5"]),
                        "CaS": ("S", MW["S"] / MW["CaS"])}

    def element_balance(self, traj: Trajectory, u: HeatInputs,
                        charge_kg: float, charge_comp: Dict[str, float],
                        hot_heel_kg: float = 0.0) -> "pd.DataFrame":
        """
        (E61) Element ledger over the whole heat, per element E:

            in(E)  = charge + heel + sum(additions)
            out(E) = metal + slag-bound + gas (C only, as CO/CO2)
            closure(E) = (in - out) / in

        |closure| should sit below ~0.5 % (integration error only). A large
        Fe deficit means the oxygen ledger is inventing or destroying iron; a
        C imbalance means the CO/CO2 counters disagree with the bath — both
        are bugs, not tuning matters. This is the audit a reviewer should ask
        to see before believing any endpoint number.
        """
        import pandas as pd
        x0, xf = traj.X[0], traj.X[-1]
        rows = []
        heel_total = charge_kg  # charge_kg includes heel by initial_state convention
        for el in self.metal:
            m_in = heel_total * charge_comp.get(el, 0.0)
            if el == "Fe":
                m_in = heel_total * max(1.0 - sum(charge_comp.values())
                                        + charge_comp.get("Fe", 0.0), 0.0) \
                    if charge_comp.get("Fe", 0.0) == 0.0 else heel_total * charge_comp["Fe"]
            # slag charged with the heat carries oxide-bound element in
            for ox, (e2, f2) in self.ELEMENT_OF_OXIDE.items():
                if e2 == el and ox in self.slag:
                    m_in += x0[self.nM + self.slag.index(ox)] * f2
            for a in u.additions:
                for sp, frac in a.composition.items():
                    if sp == el:
                        m_in += a.mass_kg * frac
                    elif sp in self.ELEMENT_OF_OXIDE and self.ELEMENT_OF_OXIDE[sp][0] == el:
                        m_in += a.mass_kg * frac * self.ELEMENT_OF_OXIDE[sp][1]
            # solid remainder at t0 counts as in (already in charge), fine.
            m_out = xf[self.metal.index(el)]
            m_out += xf[self.iMs] * self.solid_comp[self.metal.index(el)]
            und = getattr(traj, "undissolved_species", {})
            m_out += und.get(el, 0.0)
            for ox, (e2, f2) in self.ELEMENT_OF_OXIDE.items():
                if e2 == el:
                    m_out += und.get(ox, 0.0) * f2
            for ox, (e2, f2) in self.ELEMENT_OF_OXIDE.items():
                if e2 == el and ox in self.slag:
                    m_out += xf[self.nM + self.slag.index(ox)] * f2
            if el == "C":
                m_out += xf[self.iCO] * MW["C"] / MW["CO"] \
                    + xf[self.iCO2] * MW["C"] / MW["CO2"]
            rows.append(dict(element=el, in_kg=m_in, out_kg=m_out,
                             closure_pct=100.0 * (m_in - m_out) / max(m_in, 1e-9)))
        return pd.DataFrame(rows)

    def energy_closure(self, traj: Trajectory, dt: Optional[float] = None) -> dict:
        """
        (E62) First-law closure over the heat:

            E_grid + E_chem = dH_metal + dH_solid + dH_lining
                              + E_conv + E_cool + E_wall,out + E_rad + E_gas
                              + E_dissolution_sinks

        Reported as residual_kWh and residual_pct of grid input. The lining
        term uses the stored energy CHANGE of the wall nodes, and wall loss is
        counted at the OUTER boundary — counting q[0] as 'loss' double-books
        the energy parked in the refractory (it comes back next heat via the
        hot lining; that is why heat #1 on a cold furnace costs 8-12 % more).
        """
        dt = dt or self.cfg.numerics.dt_s
        f = dt / 3600.0
        d = traj.diagnostics
        thc = self.cfg.thermal
        x0, xf = traj.X[0], traj.X[-1]
        s0, sf = self.unpack(x0), self.unpack(xf)
        T_ref = thc.T_ambient_C + KELVIN

        def H_state(s):  # kWh, sensible+latent above ambient
            H_l = s["M_l"] * (thc.cp_solid_kJ_kgK * (thc.T_liquidus_C + KELVIN - T_ref)
                              + thc.L_fusion_kJ_kg
                              + thc.cp_liquid_kJ_kgK * (s["T_b"] - (thc.T_liquidus_C + KELVIN)))
            H_s = s["m_s"] * thc.cp_solid_kJ_kgK * (s["T_s"] - T_ref)
            H_sl = sum(s["slag_kg"].values()) * thc.cp_slag_kJ_kgK * (s["T_b"] - T_ref)
            return (H_l + H_s + H_sl) / 3600.0

        dH_lining = float(np.sum(self.C_node
                                 * (xf[self.iLin:self.iLin + self.N]
                                    - x0[self.iLin:self.iLin + self.N]))) / 3.6e6
        E_in = xf[self.iE] + (d["Q_chem"].sum() + d["Q_pc"].sum()) * f
        # Q_wall in diagnostics is the HOT-face flux. Energy that crossed the
        # hot face either left through the outer boundary or is stored in the
        # wall; splitting it this way avoids double-booking the stored part.
        E_wall_out = d["Q_wall"].sum() * f - dH_lining
        E_out = (H_state(sf) - H_state(s0)) + dH_lining + E_wall_out \
            + (d["P_conv"].sum() + d["P_cool"].sum() + d["Q_rad"].sum()
               + d["Q_gas"].sum()) * f
        resid = E_in - E_out
        return dict(E_in_kWh=E_in, E_out_kWh=E_out,
                    dH_bath_kWh=H_state(sf) - H_state(s0),
                    dH_lining_kWh=dH_lining, E_wall_outer_kWh=E_wall_out,
                    residual_kWh=resid,
                    residual_pct=100.0 * resid / max(E_in, 1e-9))
