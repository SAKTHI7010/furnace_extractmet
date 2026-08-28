"""
thermo.py — Layer-1 thermochemistry.

Everything here is a *closed-form* correlation, so it runs in microseconds on a
Jetson-class CPU core. Where a correlation is an approximation of a FactSage /
Gibbs-minimisation result, that is stated in the docstring. Swap any function
for a FactSage call (or a surrogate trained on FactSage) without touching the
rest of the code — that is the point of keeping this module pure.

Equation numbers (E1, E2, ...) refer to docs/SmartMelt_Mathematical_Model.md
"""
from __future__ import annotations

import numpy as np
from typing import Dict

R_GAS = 8.314462618          # J/mol/K
SIGMA_SB = 5.670374419e-8    # W/m^2/K^4
T_REF_K = 1873.15
KELVIN = 273.15

MW: Dict[str, float] = {
    "Fe": 55.845, "C": 12.011, "Si": 28.086, "Mn": 54.938, "P": 30.974,
    "S": 32.06, "Cu": 63.546, "Cr": 51.996, "Ni": 58.693, "Al": 26.982,
    "O": 15.999, "O2": 31.998, "N2": 28.014, "Ca": 40.078, "Mg": 24.305,
    "CO": 28.010, "CO2": 44.009,
    "FeO": 71.844, "SiO2": 60.084, "CaO": 56.077, "MgO": 40.304,
    "MnO": 70.937, "Al2O3": 101.961, "P2O5": 141.944, "CaF2": 78.075,
    "Cr2O3": 151.990, "CaS": 72.143,
}

# Oxygen atoms per formula unit — used for optical basicity (E12)
N_OXYGEN = {"FeO": 1, "SiO2": 2, "CaO": 1, "MgO": 1, "MnO": 1,
            "Al2O3": 3, "P2O5": 5, "CaF2": 0, "Cr2O3": 3, "CaS": 0}

# Single-cation optical basicity (Duffy & Ingram)
LAMBDA_OPT = {"CaO": 1.00, "MgO": 0.78, "FeO": 1.00, "MnO": 1.00,
              "SiO2": 0.48, "Al2O3": 0.61, "P2O5": 0.40, "CaF2": 0.67,
              "Cr2O3": 0.77, "CaS": 1.00}

# Wagner first-order interaction parameters e_i^j at 1873 K.
# Rows = solute i whose activity coefficient we want; cols = solute j.
E_1873: Dict[str, Dict[str, float]] = {
    "C":  {"C": 0.140, "O": -0.340, "Si": 0.080, "Mn": -0.012, "P": 0.051,
           "S": 0.046, "Cr": -0.024, "Ni": 0.012, "Cu": 0.016},
    "O":  {"C": -0.450, "O": -0.170, "Si": -0.131, "Mn": -0.021, "P": 0.070,
           "S": -0.133, "Cr": -0.040, "Ni": 0.006, "Cu": -0.013},
    "Si": {"C": 0.180, "O": -0.230, "Si": 0.110, "Mn": 0.002, "P": 0.110,
           "S": 0.056, "Cr": -0.0003, "Ni": 0.005, "Cu": 0.0},
    "Mn": {"C": -0.070, "O": -0.083, "Si": 0.0, "Mn": 0.0, "P": -0.0035,
           "S": -0.048, "Cr": 0.0039, "Ni": 0.0, "Cu": 0.0},
    "P":  {"C": 0.130, "O": 0.130, "Si": 0.120, "Mn": 0.0, "P": 0.062,
           "S": 0.028, "Cr": -0.030, "Ni": 0.0, "Cu": 0.0},
    "S":  {"C": 0.110, "O": -0.270, "Si": 0.063, "Mn": -0.026, "P": 0.029,
           "S": -0.028, "Cr": -0.011, "Ni": 0.0, "Cu": -0.0084},
}


# --------------------------------------------------------------------------
# Activity coefficients
# --------------------------------------------------------------------------
def wagner_log_f(i: str, pct: Dict[str, float], T_K: float) -> float:
    """
    (E10)  log10 f_i = sum_j e_i^j(T) * [%j],   e_i^j(T) = e_i^j(1873) * 1873/T

    1 wt-% standard state, Henrian. First-order truncation: adequate for
    steelmaking ranges (<2 wt-% solutes). Add second-order r_i^j here if you
    later work with high-Mn / high-Cr grades.
    """
    if i not in E_1873:
        return 0.0
    scale = T_REF_K / T_K
    return sum(e * scale * pct.get(j, 0.0) for j, e in E_1873[i].items())


def f_i(i: str, pct: Dict[str, float], T_K: float) -> float:
    return 10.0 ** wagner_log_f(i, pct, T_K)


# --------------------------------------------------------------------------
# Equilibrium constants (1 wt-% standard state for solutes, pure oxide for slag)
# --------------------------------------------------------------------------
def _Tclamp(T_K: float) -> float:
    """
    Equilibrium correlations below are fit over liquid-steel range only
    (~1500-1750 C). The ODE state can pass through unphysical or transient
    T (a cold heel before melt-down, a NaN-adjacent value during a bad
    sub-step) on its way to a valid trajectory; clamping here means a
    momentary excursion degrades accuracy for one sub-step instead of
    producing 10**300 and killing the integration.
    """
    return float(np.clip(T_K, 1400.0, 2300.0))


def logK_CO(T_K: float) -> float:
    """(E6)  [C] + [O] = CO(g)      log K = 1160/T + 2.003"""
    return 1160.0 / _Tclamp(T_K) + 2.003


def logK_FeO(T_K: float) -> float:
    """(E7)  Fe(l) + [O] = (FeO)    log K = 6320/T - 2.734"""
    return 6320.0 / _Tclamp(T_K) - 2.734


def logK_SiO2(T_K: float) -> float:
    """(E8)  [Si] + 2[O] = (SiO2)   log K = 30410/T - 11.59"""
    return 30410.0 / _Tclamp(T_K) - 11.59


def logK_MnO(T_K: float) -> float:
    """(E9)  [Mn] + [O] = (MnO)     log K = 12760/T - 5.62"""
    return 12760.0 / _Tclamp(T_K) - 5.62


# --------------------------------------------------------------------------
# Slag model
# --------------------------------------------------------------------------
def slag_mole_fractions(slag_kg: Dict[str, float]) -> Dict[str, float]:
    n = {k: v / MW[k] for k, v in slag_kg.items() if v > 0 and k in MW}
    tot = sum(n.values())
    if tot <= 0:
        return {k: 0.0 for k in slag_kg}
    return {k: n.get(k, 0.0) / tot for k in slag_kg}


def optical_basicity(slag_kg: Dict[str, float]) -> float:
    """(E12)  Lambda = sum(X_i n_i Lambda_i) / sum(X_i n_i)"""
    X = slag_mole_fractions(slag_kg)
    num = sum(X[k] * N_OXYGEN.get(k, 0) * LAMBDA_OPT.get(k, 0.6) for k in X)
    den = sum(X[k] * N_OXYGEN.get(k, 0) for k in X)
    return num / den if den > 1e-12 else 0.6


def basicity_B2(slag_kg: Dict[str, float]) -> float:
    """B2 = %CaO / %SiO2 (mass basis)."""
    sio2 = slag_kg.get("SiO2", 0.0)
    return slag_kg.get("CaO", 0.0) / sio2 if sio2 > 1e-9 else 10.0


def a_FeO(slag_kg: Dict[str, float], gamma_FeO: float) -> float:
    """
    (E11)  a_FeO = gamma_FeO * X_FeO

    gamma_FeO ~ 1.3-2.5 for basic steelmaking slags. Made a config parameter
    rather than hard-coded: fit it during plant calibration (Sec. 6 of the doc).
    A basicity dependence is applied as a mild correction.
    """
    X = slag_mole_fractions(slag_kg)
    B = np.clip(basicity_B2(slag_kg), 0.5, 4.0)
    gamma = gamma_FeO * (1.0 + 0.08 * (B - 1.6))
    return float(np.clip(gamma * X.get("FeO", 0.0), 1e-6, 1.0))


def h_O_from_slag(slag_kg: Dict[str, float], gamma_FeO: float, T_K: float) -> float:
    """
    (E13)  Dissolved-oxygen activity in equilibrium with the slag:
           Fe(l) + [O] = (FeO)  =>  h_O = a_FeO / K_FeO
    Returns h_O in wt-% (Henrian activity, f_O * %O).
    """
    return a_FeO(slag_kg, gamma_FeO) / (10.0 ** logK_FeO(T_K))


def pct_C_equilibrium(slag_kg, gamma_FeO, T_K, pct_metal, p_CO_atm=1.0) -> float:
    """
    (E14)  %C_eq = p_CO / (K_CO * f_C * h_O)
    The floor of the decarburisation driving force. Small a_FeO => high %C_eq
    => decarburisation stalls. This coupling is what makes slag practice show
    up in the carbon trajectory, and is why an IF cannot decarburise much.
    """
    hO = h_O_from_slag(slag_kg, gamma_FeO, T_K)
    K = 10.0 ** logK_CO(T_K)
    fC = f_i("C", pct_metal, T_K)
    return float(p_CO_atm / max(K * fC * hO, 1e-9))


def pct_Si_equilibrium(slag_kg, gamma_FeO, T_K, pct_metal) -> float:
    """(E15)  [Si] + 2[O] = (SiO2)  =>  %Si_eq = a_SiO2 / (K * f_Si * h_O^2)"""
    X = slag_mole_fractions(slag_kg)
    a_sio2 = max(X.get("SiO2", 0.0) * 0.6, 1e-6)   # gamma_SiO2 ~ 0.6 (basic slag)
    hO = h_O_from_slag(slag_kg, gamma_FeO, T_K)
    K = 10.0 ** logK_SiO2(T_K)
    fSi = f_i("Si", pct_metal, T_K)
    return float(a_sio2 / max(K * fSi * hO ** 2, 1e-12))


def pct_Mn_equilibrium(slag_kg, gamma_FeO, T_K, pct_metal) -> float:
    """(E16)  [Mn] + [O] = (MnO)"""
    X = slag_mole_fractions(slag_kg)
    a_mno = max(X.get("MnO", 0.0) * 1.0, 1e-6)
    hO = h_O_from_slag(slag_kg, gamma_FeO, T_K)
    K = 10.0 ** logK_MnO(T_K)
    fMn = f_i("Mn", pct_metal, T_K)
    return float(a_mno / max(K * fMn * hO, 1e-12))


def L_P_healy(slag_kg: Dict[str, float], T_K: float) -> float:
    """
    (E17) Healy's phosphorus partition:
      log( (%P)/[%P] ) = 22350/T - 16.0 + 0.08*(%CaO) + 2.5*log(%Fe_total)

    %Fe_total in slag counted as Fe from FeO. Valid ~1550-1700 C, basic slags.
    """
    tot = sum(slag_kg.values())
    if tot <= 0:
        return 1.0
    pct_CaO = 100.0 * slag_kg.get("CaO", 0.0) / tot
    feo = slag_kg.get("FeO", 0.0)
    pct_Fe_t = max(100.0 * feo * (MW["Fe"] / MW["FeO"]) / tot, 0.5)
    Tc = float(np.clip(T_K, 1650.0, 2100.0))
    log_LP = 22350.0 / Tc - 16.0 + 0.08 * pct_CaO + 2.5 * np.log10(pct_Fe_t)
    return float(np.clip(10.0 ** np.clip(log_LP, -3.0, 4.0), 1e-3, 1e4))


def sulphide_capacity(slag_kg: Dict[str, float], T_K: float) -> float:
    """
    (E18) Sosinsky-Sommerville optical-basicity model:
      log C_S = (22690 - 54640*Lambda)/T + 43.6*Lambda - 25.2
    T_K is clamped to a physically sane liquid-steel range before use — the
    correlation is only fit over ~1550-1700 C, and extrapolating it to a cold
    (sub-solidus) transient state produces a divide-by-a-tiny-number blow-up
    that has nothing to do with chemistry.
    """
    lam = optical_basicity(slag_kg)
    Tc = float(np.clip(T_K, 1650.0, 2100.0))
    log_Cs = (22690.0 - 54640.0 * lam) / Tc + 43.6 * lam - 25.2
    return float(10.0 ** np.clip(log_Cs, -12.0, 3.0))


def L_S(slag_kg, gamma_FeO, T_K, pct_metal) -> float:  # noqa: D401
    """
    (E19) log L_S = log C_S + log f_S + 935/T + 1.375 - log(a_O)
    Desulphurisation demands low a_O — the reason you cannot desulphurise and
    decarburise hard at the same time. Model reproduces that automatically.
    """
    Cs = sulphide_capacity(slag_kg, T_K)
    fS = f_i("S", pct_metal, T_K)
    aO = max(h_O_from_slag(slag_kg, gamma_FeO, T_K), 1e-6)
    log_ls = np.log10(Cs) + np.log10(fS) + 935.0 / _Tclamp(T_K) + 1.375 - np.log10(aO)
    return float(np.clip(10.0 ** log_ls, 1e-2, 1e4))


# --------------------------------------------------------------------------
# Enthalpy helpers
# --------------------------------------------------------------------------
def theoretical_melt_energy_kWh_per_t(cfg) -> float:
    """
    (E20) Thermodynamic floor for melting + superheating scrap:
      h = cp_s (T_liq - T_amb) + L_f + cp_l (T_tap - T_liq)     [kJ/kg]
    Divide by 3.6 to get kWh/t. Everything above this floor is loss.
    Typical: ~380-400 kWh/t for steel tapped at 1620 C.
    """
    th, pl = cfg.thermal, cfg.plant
    h = (th.cp_solid_kJ_kgK * (th.T_liquidus_C - th.T_ambient_C)
         + th.L_fusion_kJ_kg
         + th.cp_liquid_kJ_kgK * (pl.tap_temperature_C - th.T_liquidus_C))
    return h / 3.6
