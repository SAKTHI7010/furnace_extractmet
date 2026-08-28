"""
advisory.py — the layer the operator actually sees.

Three rules, in order of importance:

  1. Never issue a confident recommendation the model cannot support.
     If sigma > sigma_suspend * tolerance, or the drift monitor has alarmed,
     the verdict degrades to AMBER-WIDE or SUSPENDED, and the reason is shown.
     A system that says "I don't know right now" survives a regime change;
     one that always answers does not.

  2. Every recommendation carries its reason, in the operator's language.
     "Add 5 kg FeSi" is an order. "Add 5 kg FeSi: predicted Si 0.30 % below aim,
     confidence high" is an argument. Operators comply with arguments.

  3. The counterfactual is logged. Recommendation, operator's actual action,
     and outcome, on every heat. That log is the training set, the audit trail,
     and the evidence for the shared-savings contract. Without it you have a
     dashboard, not a product.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Verdict(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"
    SUSPENDED = "SUSPENDED"


@dataclass
class Recommendation:
    variable: str
    verdict: Verdict
    action_en: str
    action_hi: str
    reason_en: str
    reason_hi: str
    confidence: str
    predicted: float
    sigma: float
    target: float

    def render(self, language="both") -> str:
        tag = {"GREEN": "[OK ]", "YELLOW": "[!  ]", "RED": "[!!!]",
               "SUSPENDED": "[ ? ]"}[self.verdict.value]
        parts = [f"{tag} {self.variable:<14}"]
        if language in ("en", "both"):
            parts.append(f"{self.action_en}  ({self.reason_en})")
        if language in ("hi", "both"):
            parts.append(f"      {self.action_hi}  ({self.reason_hi})")
        return "\n".join(parts)


@dataclass
class HeatLogEntry:
    heat_id: str
    t_s: float
    recommendations: List[Recommendation]
    operator_action: Optional[str] = None
    outcome: Optional[Dict[str, float]] = None
    drift: Optional[dict] = field(default=None)


class AdvisoryEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.a = cfg.advisory
        self.log: List[HeatLogEntry] = []

    # ------------------------------------------------------------------
    def _grade(self, err: float, sigma: float, green: float, yellow: float):
        if sigma > self.a.sigma_suspend_multiplier * yellow:
            return Verdict.SUSPENDED, "low"
        conf = "high" if sigma <= 0.5 * green else ("medium" if sigma <= green else "low")
        a = abs(err)
        if a <= green:  return Verdict.GREEN, conf
        if a <= yellow: return Verdict.YELLOW, conf
        return Verdict.RED, conf

    # ------------------------------------------------------------------
    def temperature(self, pred_T: float, sigma_T: float, target_T: float,
                    minutes_left: float, power_headroom_kW: float) -> Recommendation:
        err = pred_T - target_T
        v, conf = self._grade(err, sigma_T, self.a.T_tolerance_green_C,
                              self.a.T_tolerance_yellow_C)
        if v is Verdict.SUSPENDED:
            en, hi = "Hold current practice", "वर्तमान प्रैक्टिस जारी रखें"
            ren = f"model uncertainty +/-{sigma_T:.0f} C too high to advise"
            rhi = f"मॉडल की अनिश्चितता +/-{sigma_T:.0f} C — सलाह रोकी गई"
        elif err < -self.a.T_tolerance_green_C:
            dP = min(power_headroom_kW, 800.0)
            en = f"Raise power by ~{dP:.0f} kW"
            hi = f"पावर ~{dP:.0f} kW बढ़ाएँ"
            ren = f"predicted tap T {pred_T:.0f} C is {-err:.0f} C below aim"
            rhi = f"अनुमानित टैप ताप {pred_T:.0f} C, लक्ष्य से {-err:.0f} C कम"
        elif err > self.a.T_tolerance_green_C:
            en, hi = "Reduce power / tap earlier", "पावर घटाएँ / जल्दी टैप करें"
            ren = f"predicted tap T {pred_T:.0f} C is {err:.0f} C above aim — "\
                  f"lining and yield penalty"
            rhi = f"अनुमानित टैप ताप {pred_T:.0f} C, लक्ष्य से {err:.0f} C अधिक"
        else:
            en, hi = "On track", "सही दिशा में"
            ren = f"predicted {pred_T:.0f} +/-{sigma_T:.0f} C, tap in ~{minutes_left:.0f} min"
            rhi = f"अनुमानित {pred_T:.0f} +/-{sigma_T:.0f} C, ~{minutes_left:.0f} मिनट में टैप"
        return Recommendation("bath temp", v, en, hi, ren, rhi, conf,
                              pred_T, sigma_T, target_T)

    def carbon(self, pred_C: float, sigma_C: float, target_C: float,
               o2_available: bool) -> Recommendation:
        err = pred_C - target_C
        v, conf = self._grade(err, sigma_C, self.a.C_tolerance_green_pct,
                              self.a.C_tolerance_yellow_pct)
        if v is Verdict.SUSPENDED:
            en, hi = "Take a sample", "सैंपल लें"
            ren = f"carbon uncertainty +/-{sigma_C:.3f} % too high"
            rhi = f"कार्बन अनिश्चितता +/-{sigma_C:.3f} % — सैंपल आवश्यक"
        elif err > self.a.C_tolerance_green_pct:
            en = "Blow oxygen / extend refining" if o2_available else "Add ore / mill scale"
            hi = "ऑक्सीजन ब्लो करें" if o2_available else "मिल स्केल डालें"
            ren = f"predicted C {pred_C:.3f} % is {err:+.3f} % above aim"
            rhi = f"अनुमानित C {pred_C:.3f} %, लक्ष्य से {err:+.3f} % अधिक"
        elif err < -self.a.C_tolerance_green_pct:
            kg = abs(err) / 100.0 * self.cfg.plant.heat_size_t * 1000.0 / 0.85
            en = f"Recarburise: add ~{kg:.0f} kg carbon"
            hi = f"~{kg:.0f} kg कार्बन मिलाएँ"
            ren = f"predicted C {pred_C:.3f} % is {-err:.3f} % below aim"
            rhi = f"अनुमानित C {pred_C:.3f} %, लक्ष्य से {-err:.3f} % कम"
        else:
            en, hi = "Carbon on aim", "कार्बन लक्ष्य पर"
            ren = f"predicted {pred_C:.3f} +/-{sigma_C:.3f} %"
            rhi = f"अनुमानित {pred_C:.3f} +/-{sigma_C:.3f} %"
        return Recommendation("carbon", v, en, hi, ren, rhi, conf,
                              pred_C, sigma_C, target_C)

    def slag(self, B2: float, pct_FeO: float, heat_size_t: float) -> Recommendation:
        target = self.cfg.slag.target_basicity_B2
        err = B2 - target
        v = Verdict.GREEN if abs(err) < 0.2 else (
            Verdict.YELLOW if abs(err) < 0.5 else Verdict.RED)
        if err < -0.2:
            kg = abs(err) * 6.0 * heat_size_t
            en = f"Add ~{kg:.0f} kg lime (CaO)"
            hi = f"~{kg:.0f} kg चूना (CaO) डालें"
            ren = f"B2 = {B2:.2f} vs aim {target:.2f}; P and S will not partition"
            rhi = f"B2 = {B2:.2f}, लक्ष्य {target:.2f} — P/S नहीं निकलेगा"
        elif err > 0.2:
            en, hi = "Reduce lime; check slag fluidity", "चूना घटाएँ; स्लैग तरलता जाँचें"
            ren = f"B2 = {B2:.2f} above aim; slag stiff, heat loss rises"
            rhi = f"B2 = {B2:.2f} अधिक — स्लैग गाढ़ा, ऊष्मा हानि"
        else:
            en, hi = "Slag on aim", "स्लैग लक्ष्य पर"
            ren = f"B2 = {B2:.2f}, FeO = {pct_FeO:.1f} %"
            rhi = f"B2 = {B2:.2f}, FeO = {pct_FeO:.1f} %"
        return Recommendation("slag B2", v, en, hi, ren, rhi, "high", B2, 0.05, target)

    def energy(self, sec_now: float, sec_target: float) -> Recommendation:
        err = sec_now - sec_target
        v = Verdict.GREEN if err <= 15 else (Verdict.YELLOW if err <= 50 else Verdict.RED)
        en = "Specific energy on track" if v is Verdict.GREEN else \
            f"Specific energy {err:+.0f} kWh/t vs model optimum"
        hi = "विशिष्ट ऊर्जा ठीक" if v is Verdict.GREEN else \
            f"विशिष्ट ऊर्जा {err:+.0f} kWh/t अधिक"
        ren = f"actual {sec_now:.0f} vs achievable {sec_target:.0f} kWh/t"
        rhi = f"वास्तविक {sec_now:.0f}, संभव {sec_target:.0f} kWh/t"
        return Recommendation("energy", v, en, hi, ren, rhi, "high",
                              sec_now, 10.0, sec_target)

    # ------------------------------------------------------------------
    def evaluate(self, *, pred_T, sigma_T, pred_C, sigma_C, B2, pct_FeO,
                 sec_now, sec_target, minutes_left, power_headroom_kW,
                 o2_available, drift_report: Optional[dict] = None,
                 heat_id: str = "", t_s: float = 0.0) -> HeatLogEntry:
        recs = [
            self.temperature(pred_T, sigma_T, self.cfg.plant.tap_temperature_C,
                             minutes_left, power_headroom_kW),
            self.carbon(pred_C, sigma_C, self.cfg.plant.target_carbon_pct, o2_available),
            self.slag(B2, pct_FeO, self.cfg.plant.heat_size_t),
            self.energy(sec_now, sec_target),
        ]
        if drift_report and drift_report.get("alarm"):
            for r in recs[:2]:
                r.verdict = Verdict.SUSPENDED
                r.reason_en += " | drift alarm: " + "; ".join(drift_report["reasons"])
                r.reason_hi += " | ड्रिफ्ट अलार्म"
        entry = HeatLogEntry(heat_id, t_s, recs, drift=drift_report)
        self.log.append(entry)
        return entry

    def render(self, entry: HeatLogEntry) -> str:
        return "\n".join(r.render(self.a.language) for r in entry.recommendations)
