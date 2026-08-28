"""
metrics.py — how you know it works, and how the plant owner knows it paid.

A note on the "+/-15 C, +/-0.02 %C" claim. That is only meaningful with three
qualifiers attached, and a panel will ask for all three:

  1. Is it 1-sigma, 2-sigma, or max error?        -> state it. Use `endpoint_hit_rate`.
  2. On what split?  In-sample residuals of a fitted model are not accuracy.
     -> use `grouped_cv_report` with a *time-ordered* split (train on heats
        1..N, test on N+1..M). Random k-fold leaks the lining campaign.
  3. Against what reference?  A drop-cell has its own +/-3-5 C.
     -> report sigma_model = sqrt(sigma_observed^2 - sigma_reference^2).

Anything else and you are quoting an artefact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Sequence


# --------------------------------------------------------------------------
def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. >0.10 investigate, >0.25 regime change."""
    ref, cur = np.asarray(reference, float), np.asarray(current, float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if ref.size < 2 or cur.size < 2:
        return 0.0
    # Small-sample safety: a 10-heat window against decile bins WILL leave
    # empty bins; with a hard eps-clip each empty bin contributes ~1.15 and
    # a perfectly stable feature "alarms" at PSI ~ 12. Two standard fixes:
    # adapt the bin count to the smaller sample, and Laplace-smooth counts.
    bins = int(np.clip(min(bins, cur.size // 4, ref.size // 4), 2, bins))
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    nb = edges.size - 1
    cp = np.histogram(ref, edges)[0].astype(float)
    cq = np.histogram(cur, edges)[0].astype(float)
    p = (cp + 0.5) / (cp.sum() + 0.5 * nb)
    q = (cq + 0.5) / (cq.sum() + 0.5 * nb)
    raw = float(np.sum((q - p) * np.log(q / p)))
    # Finite-sample bias correction: under NO drift, E[PSI] ~ (B-1)(1/n+1/m)
    # (chi-square approximation). At MSME window sizes (15-30 heats) this null
    # bias is 0.1-0.3 — the size of the alarm threshold itself — so raw PSI
    # "alarms" on perfectly stable inputs. Subtract the null expectation.
    null = (nb - 1) * (1.0 / ref.size + 1.0 / cur.size)
    return max(raw - null, 0.0)


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = np.abs(y_true) > 1e-9
    return float(100.0 * np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m])))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def endpoint_hit_rate(y_true, y_pred, tol: float,
                      sigma_reference: float = 0.0) -> Dict[str, float]:
    """
    Returns hit rate at |err| <= tol, plus a reference-corrected model sigma.
    sigma_model^2 = sigma_observed^2 - sigma_reference^2   (measurement is noisy too)
    """
    err = np.asarray(y_pred, float) - np.asarray(y_true, float)
    s_obs = float(np.std(err, ddof=1))
    s_model = float(np.sqrt(max(s_obs ** 2 - sigma_reference ** 2, 0.0)))
    return dict(hit_rate=float(np.mean(np.abs(err) <= tol)),
                bias=float(np.mean(err)), mae=float(np.mean(np.abs(err))),
                sigma_observed=s_obs, sigma_model=s_model,
                p95_abs_err=float(np.percentile(np.abs(err), 95)), n=int(err.size))


def time_ordered_split(df: pd.DataFrame, train_frac=0.7, order_col="heat_no"):
    """Never random-split heats: the lining campaign and scrap regime are serial."""
    d = df.sort_values(order_col) if order_col in df else df
    k = int(len(d) * train_frac)
    return d.iloc[:k].copy(), d.iloc[k:].copy()


# --------------------------------------------------------------------------
def energy_kpis(df: pd.DataFrame, cfg) -> Dict[str, float]:
    from .thermo import theoretical_melt_energy_kWh_per_t
    sec = df["SEC_kWh_per_t"] if "SEC_kWh_per_t" in df else \
        df["meas_energy_kWh"] / df["charge_mass_t"]
    floor = theoretical_melt_energy_kWh_per_t(cfg)
    return dict(sec_mean=float(sec.mean()), sec_p10=float(sec.quantile(0.10)),
                sec_p90=float(sec.quantile(0.90)), sec_std=float(sec.std()),
                thermodynamic_floor=floor,
                loss_above_floor=float(sec.mean()) - floor,
                # "best-of-fleet" target: the plant's own 10th percentile heat,
                # already achieved with the same equipment and the same scrap.
                achievable_target=float(sec.quantile(0.10)),
                recoverable_kWh_per_t=float(sec.mean() - sec.quantile(0.10)))


def economics(sec_before: float, sec_after: float, cfg,
              tonnes_per_year: Optional[float] = None) -> Dict[str, float]:
    """
    Deliberately conservative and fully explicit — every term is a line the
    plant owner can dispute. Do NOT quote a payback without showing this table.
    """
    ec = cfg.economics
    t_yr = tonnes_per_year or cfg.plant.heat_size_t * cfg.plant.heats_per_year
    saved = max(sec_before - sec_after, 0.0)
    kwh = saved * t_yr
    money = kwh * ec.tariff_INR_per_kWh
    co2 = kwh / 1000.0 * ec.grid_EF_tCO2_per_MWh
    carbon_value = co2 * ec.carbon_price_INR_per_tCO2
    net_annual = money + carbon_value - ec.opex_INR_per_year
    payback = ec.capex_INR / net_annual * 12.0 if net_annual > 0 else float("inf")
    return dict(tonnes_per_year=t_yr, kWh_saved_per_t=saved,
                kWh_saved_per_year=kwh, INR_saved_per_year=money,
                tCO2_avoided_per_year=co2, carbon_credit_INR=carbon_value,
                opex_INR=ec.opex_INR_per_year, capex_INR=ec.capex_INR,
                net_annual_INR=net_annual, payback_months=payback,
                five_year_NPV_INR_at_12pct=sum(
                    net_annual / (1.12 ** y) for y in range(1, 6)) - ec.capex_INR)


def savings_attribution(df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict[str, float]:
    """
    Regression-adjusted savings. The plant will change scrap and grade mix during
    the trial; a raw before/after SEC comparison will be challenged, and rightly.
    Fit SEC ~ charge mass + charge C + tap T on BEFORE, predict on AFTER, and
    attribute only the residual gap to SmartMelt.
    """
    from sklearn.linear_model import LinearRegression
    cols = [c for c in ("charge_mass_t", "charge_C_pct", "tap_target_C", "hot_heel_t")
            if c in df_before.columns and c in df_after.columns]
    y = df_before["SEC_kWh_per_t"] if "SEC_kWh_per_t" in df_before else None
    if y is None or not cols:
        raw = float(df_before["SEC_kWh_per_t"].mean() - df_after["SEC_kWh_per_t"].mean())
        return dict(raw_saving=raw, adjusted_saving=raw, confound_correction=0.0)
    lr = LinearRegression().fit(df_before[cols], y)
    expected = lr.predict(df_after[cols])
    actual = df_after["SEC_kWh_per_t"].to_numpy()
    adjusted = float(np.mean(expected - actual))
    raw = float(y.mean() - actual.mean())
    return dict(raw_saving=raw, adjusted_saving=adjusted,
                confound_correction=raw - adjusted)
