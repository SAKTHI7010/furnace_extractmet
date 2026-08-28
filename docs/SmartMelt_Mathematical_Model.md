# SmartMelt — The Mathematical Model

**Hybrid first-principles + machine-learning melt model for induction-furnace, EAF and BOF steelmaking**
Extractmet Pvt. Ltd. · Reference implementation: `smartmelt` Python package (v0.3)
Every equation below carries a number **(En)** that appears verbatim as a comment at the corresponding line of code, so the document and the implementation can be audited against each other.

---

## 1. Architecture and design philosophy

The model has three layers, matching the SmartMelt technical brief:

**Layer 1 — Physics (`physics.py`, `thermo.py`).** A coupled system of ordinary differential equations expressing elemental mass balances, a multi-zone energy balance, reaction kinetics driven toward thermodynamic equilibrium, dissolution kinetics for every addition, and a multi-layer refractory wall with conduction, convection and radiation. It runs faster than real time on an edge PC and is valid from heat #1 with zero plant data.

**Layer 2 — Machine learning (`ml.py`).** Learns only the **residual** between measurement and physics,

> y_measured = f_physics(u, x0; theta) + g_ML(features) + eps   **(E0)**

never the absolute endpoint. This single design decision provides: a valid degraded mode (if ML is suspended, physics still advises), bounded extrapolation (the GP prior reverts to physics away from data), small training-set requirements (the residual is small and smooth), and a defence against the silent-drift failure mode of pure-AI melt models under regime change.

**Layer 3 — Estimation and decision (`ekf.py`, `chargemix.py`, `mpc.py`, `advisory.py`).** An extended Kalman filter fuses sparse noisy sensors with the physics in real time and tracks slowly varying plant parameters theta; a linear program computes the least-cost charge mix against grade walls; a receding-horizon optimiser proposes power/oxygen/addition profiles; an advisory engine converts all of it into a bilingual traffic-light board and never writes to the PLC (Phase-1 advisory-only by design).

The rule that makes one engine serve many clients: **every plant-specific number lives in a YAML config** (`config.py` dataclasses); the physics and ML code contain no magic numbers.

---

## 2. Notation and state vector

| Symbol | Meaning | Units |
|---|---|---|
| m_i | mass of dissolved element i in {Fe, C, Si, Mn, P, S, Cr, Cu, Ni} in the bath | kg |
| M_l = sum(m_i) | liquid metal mass | kg |
| w_j | mass of slag species j in {FeO, SiO2, CaO, MgO, MnO, Al2O3, P2O5, CaS} | kg |
| T_b | bath temperature | K |
| m_s, T_s | unmelted solid charge mass and temperature | kg, K |
| T_w,k | refractory node temperatures, k = 1..N (hot face -> cold face) | K |
| E | cumulative electrical energy drawn from the grid | kWh |
| m_CO, m_CO2 | cumulative off-gas masses | kg |
| theta | slowly-varying plant parameters (eta_el, UA-scale, k_C-scale, ...) | - |
| [%i] | wt-% of element i in metal; (%j) wt-% of species j in slag | % |

The full state is
**x = [m_Fe .. m_Ni, w_FeO .. w_CaS, T_b, m_s, T_s, T_w,1..T_w,N, E, m_CO, m_CO2]^T**, dimension 9 + 8 + 3 + N + 3 (N = 8 by default).

All heat terms below are in kW, masses in kg, times in s unless stated.

---

## 3. Layer 1a — Mass balances (elemental, every time step)

For each dissolved element i the bath obeys

> dm_i/dt = mdot_melt*c_i,solid + sum_a mdot_a,diss*c_i,a − r_i*M_l + r_i,rev   **(E1)**

where mdot_melt is the scrap melting rate (E4b), mdot_a,diss the dissolution release of addition a (E7a), r_i the oxidation/refining flux to slag or gas (E24–E29), and r_i,rev any slag→metal reversion. For each slag species,

> dw_j/dt = +nu_j*r_i(j)*(MW_j/MW_i) + mdot_flux,j − mdot_reduction,j   **(E2)**

with nu the stoichiometric map (Si→SiO2, Mn→MnO, 2P→P2O5, S→CaS with equivalent CaO consumption, Fe↔FeO). Carbon leaves as gas and is integrated into the CO/CO2 counters, closing the carbon ledger.

**Element ledger audit (E61).** After every simulated heat the implementation computes, for each element,
in(E) = charge + heel + slag-bound + sum(additions), and out(E) = metal + solid remainder + undissolved lumps + slag-bound + gas(C), and reports closure = (in−out)/in. Closure is asserted below 1 % in CI; anything larger is a bug, not a tuning matter. Run `FurnaceModel.element_balance()` to reproduce.

---

## 4. Layer 1b — Energy balances (every time step)

### 4.1 Bath

> M_l c_p,l dT_b/dt = P_liq + Q_chem + Q_pc − Q_s − Q_wall − Q_rad − Q_gas + mdot_melt c_p,l (T_liq − T_b) − sum_a Q_diss,a   **(E3)**

with an enthalpy-method freezing branch: if T_b <= T_liq and the net heat is negative, temperature is held and mass freezes back to the solid inventory at mdot_freeze = −Q_net/L_f **(E3b)** — a bath at the liquidus with a heat deficit freezes, it does not go sub-liquidus. When M_l < 20 kg no meaningful pool exists (charge-in, or a heel that has "bolted" onto cold scrap — normal practice, not a failure) and T_b is relaxed to T_s **(E3c)** so liquid reappears at the correct enthalpy.

### 4.2 Solid charge and melting

> m_s c_p,s dT_s/dt = Q_s + P_solid − Q_wall,solid   (T_s < T_liq)   **(E4a)**
> mdot_melt = (Q_s + P_solid)/L_f,  dT_s/dt = 0     (T_s = T_liq)   **(E4b)**

Melting is isothermal at the liquidus; the melted mass enters the bath at T_liq carrying the weighed-charge composition, and the corresponding sensible term appears in (E3).

**Bath→scrap heat flux.** The wetted contact area needs both phases present:

> A_eff = A_ref*(m_s/M_max)^(1/2) * f_liq^(1/3)   **(E28)**
> Q_s = min( h_sl*(T_b − T_s), q_max )*A_eff       **(E28b)**

The 1/2 exponent (not 2/3) is deliberate: shop scrap is plate- and turning-like, so specific surface rises as pieces thin; a spherical 2/3 law makes the last 100 kg melt asymptotically and produces a spurious pre-tap overheat. The flux cap q_max (≈350 kW/m²) is the cheap surrogate for a solidified-shell sub-model: a shell freezes onto cold scrap within milliseconds and the shell, not the liquid film, sets the resistance.

### 4.3 First-law closure audit (E62)

Over the whole heat the implementation checks

> E_grid + E_chem = dH_bath+slag + dH_solid + dH_lining + E_conv + E_cool + E_wall,outer + E_rad + E_gas + residual

and reports the residual as a percentage of input. Wall loss is counted **at the outer boundary**; counting the hot-face flux as "loss" double-books the energy parked in the refractory (it returns next heat via the hot lining — which is also why heat #1 on a cold furnace costs 8–12 % more). Current reference runs close to ≈2.6 % (IF) and ≈2.0 % (EAF); the residual is dominated by the (E3c) regularisation and the freeze-branch enthalpy approximation at charge-in, both documented and bounded. Run `FurnaceModel.energy_closure()`.

---

## 5. Electrical input and where the power lands

### 5.1 Grid → useful power

> P_coil = eta_conv * P_grid   **(E21)**
> IF: eta_coup = eta_max*(1 − exp(−fill/f_ref));  EAF: eta = eta_arc + eta_foam*1(FeO, C in foaming window)   **(E22)**
> P_use = eta_theta * eta_coup * P_coil, with eta_theta tracked online by the EKF

### 5.2 Power split between pool and scrap — the classic IF error

> IF:  P_solid/P_use = m_s/(M_l+m_s);   EAF: P_solid/P_use = (1/2)*m_s/(M_l+m_s)   **(E23)**

In a coreless induction furnace the field couples to **every** conductive mass in the crucible: solid scrap is heated directly, not through the pool. Forcing all melting heat through a small heel makes the model stiff and predicts melt-down times ~3x too long. In an EAF the arc heats the bath and a scrap-shielding fraction radiates onto the pile.

---

## 6. Heat losses: conduction, convection, radiation — coupled

### 6.1 Geometry (E5a)

All heat-transfer areas derive from the furnace design in `GeometryConfig`:
A_wall,wetted = pi*D*H_bath, A_top,open = (pi*D²/4)*(1 − lid coverage), A_wall,total = pi*D*(H_bath + H_freeboard). Override numerically only if you have measured better values.

### 6.2 Multi-layer refractory wall (E5)

The wall is a radial finite-volume mesh over the layer stack defined in `LiningConfig.layers` (e.g. IF: silica dry-vibratable working lining → mica slip-plane → coil grout; EAF: MgO-C working → safety magnesite → insulating board → steel shell), each layer with its own k, rho, c_p and service-limit temperature. Nodes are geometrically graded, thin at the hot face — only millimetres of refractory swing with the bath on a heat timescale, and a uniform mesh lags the hot face by tens of minutes. Interface conduction uses the series (harmonic) resistance of adjacent half-cells, exact for piecewise-constant k:

> q_k = A_k (T_{k−1} − T_k) / [ dr_{k−1}/(2 k_{k−1}) + dr_k/(2 k_k) ]   **(E5)**

**Inner boundary (E5b).** The hot face only sees the phases that touch it; conductance is weighted by phase fraction and each phase drives with its own temperature: q_0 = A_0[h_l f_l (T_b−T_w1) + h_s(1−f_l)(T_s−T_w1)]. (Using the full wall area against a 600 kg heel produced a 9 MW phantom loss in development — the audit trail is in the repository history.)

**Outer boundary (E5c).** Selected per plant: `coil` (IF water jacket, Robin condition with h_out, T_coolant) or `shell` (EAF/BOF), where natural convection and radiation act **in parallel**:

> q_N = A_N [ h_conv (T_shell − T_amb) + eps_shell * sigma * (T_shell^4 − T_amb^4) ]

At 200–300 °C shell temperature, radiation is roughly half the total; dropping it under-reports wall loss by ~2x.

**Numerics (E5d).** The wall is advanced by an unconditionally stable implicit (backward-Euler) tridiagonal solve, operator-split from the bath, with shell radiation linearised as h_rad = eps*sigma*(T²+T_a²)(T+T_a). Crucially, the bath is debited **exactly** the joules the implicit solve absorbed at the hot face — this exact pairing is what makes the first-law audit close; an explicit flux on one side and an implicit one on the other leaks several percent (measured: 5.0 % → 2.6 % on the reference IF when the pairing was made exact).

### 6.3 Surface radiation and off-gas

> Q_rad = eps_top * sigma * A_top,open * (T_b^4 − T_amb^4)   (open-bath / lid-gap radiation)
> Q_gas = mdot_gas * c_p,g * (T_b − T_amb), with post-combustion return Q_pc = eta_pc * PCR * mdot_CO * dH_CO→CO2   **(E30)**

---

## 7. Dissolution kinetics of additions (E7)

Nothing dissolves instantly at 1600 °C: FeSi freezes a steel shell around itself, the shell melts back, then the lump dissolves; lime dissolves through a dicalcium-silicate rim; carburiser is boundary-layer limited. Each addition is released first-order:

> dm_undis/dt = − m_undis / tau_eff                                  **(E7a)**
> tau_eff = tau * max(1, dT_ref / max(T_b − T_liq, 2 K))             **(E7b)**

so dissolution **stalls when the bath has no superheat** — exactly the condition under which operators over-add and the alloy "reappears" ten minutes later as an off-spec high. Each released kilogram carries its heat sink (sensible to bath temperature + fusion + heat of mixing; negative for exothermic dissolvers like FeSi75):

> Q_diss = mdot_diss [ c_p (T_b − T_add) + dH_diss ]                 **(E7c)**

**Convention — heat of solution and sensible heat are SEPARATE terms.** `dH_dissolution_kJ_kg` carries the *intrinsic heat of solution only*; the sensible heat of raising a cold addition to bath temperature is computed separately using a per-addition effective cp (which lumps sensible heating and any fusion of the addition itself). Conflating the two is the single easiest way to get alloy thermal effects wrong by a factor of several, because the same physical addition has a large negative heat of solution and a large positive sensible load that partially cancel.

Anchors [Sigworth & Elliott, *Met. Sci.* 1974; Turkdogan]:

> Si(l) = [Si]₁wt%  ΔH ≈ −131.5 kJ/mol Si  →  −4 681 kJ/kg Si  →  **−3 511 kJ/kg FeSi75** (75 % Si)
> C(gr) = [C]₁wt%   ΔH ≈  +22.6 kJ/mol C   →  **+1 883 kJ/kg C**

With the sensible terms restored, the *net* effect on the bath for a cold charge reproduces industrial observation: FeSi75 ≈ **−0.72 MJ/kg** (compare the +4.73 °C per tonne of FeSi75 measured on a 172 t ladle by Bernhard et al., *Metall. Mater. Trans. B* **56** (2025) 2249, DOI 10.1007/s11663-024-03419-1), and carburiser ≈ **+4.6 MJ/kg C** total cold load. For pet coke or anthracite, scale the carbon term by fixed-carbon fraction and treat the ash as inert sensible load.

Implementation: exact exponential release between integrator steps (operator splitting — cheaper and more accurate than forcing the ODE solver through it). `DISSOLUTION_PRESETS` ships tau, the heat of solution and the effective cp for lime, dolomite, FeSi75, FeMn, SiMn, carburiser, DRI, mill scale and pig iron; tune tau per plant from the addition→response lag in logged data. Undissolved inventory at tap is reported (it is a quality defect) and enters the element ledger (E61).

---

## 8. Thermochemistry (`thermo.py`)

Closed-form and microsecond-fast, structured so a FactSage/ChemApp coupling can replace it function-by-function.

Activities of dissolved elements use the Wagner interaction-parameter formalism at 1873 K with a temperature scaling:

> log f_i = sum_j e_i^j [%j]   **(E10)**, h_i = f_i [%i]

Equilibrium constants from standard-state data: C+O→CO **(E6)**, Fe+O→FeO **(E7\*)**, Si+2O→SiO2 **(E8)**, Mn+O→MnO **(E9)**. Slag description: mole fractions, optical basicity Lambda **(E12)**, binary basicity B2, FeO activity a_FeO = gamma_FeO * X_FeO **(E11)** with gamma_FeO in [1.2, 2.0] tracked in theta; oxygen activity from the FeO/Fe equilibrium h_O **(E13)**.

Equilibrium set-points the kinetics relax toward: [%C]_eq **(E14)**, [%Si]_eq **(E15)**, [%Mn]_eq **(E16)**; phosphorus partition L_P by the Healy correlation **(E17)**; sulphide capacity C_S from optical basicity **(E18)** and sulphur partition L_S **(E19)**. The theoretical melting-energy floor is **(E20)**: with the latent heat of fusion of iron at **247 kJ/kg** (13.81 kJ/mol; CRC Handbook, 104th ed.) — *not* the 272 kJ/kg used in earlier revisions, which was ~10 % high — a plain-carbon charge from 35 °C to a 1620 °C tap gives **≈381 kWh/t**.

This is a **reversible thermodynamic minimum**, and must be labelled as such in any customer-facing document. It is not an attainable target: the practical floor for a real coreless induction furnace, after coil, radiation, wall and off-gas losses, is around **500 kWh/t**, against typical Indian scrap-based practice of 550–650 kWh/t (and 650–800 kWh/t on DRI-heavy charges). Quoting 381 kWh/t as a target rather than a bound would be an overclaim.

---

## 9. Reaction kinetics and the oxygen ledger (E24–E30)

All slag–metal reactions are first-order mass-transfer relaxations toward the equilibrium of §8:

> r_i = rho_m * A_sm * k_i * ([%i] − [%i]_eq)/100 * S   **(E24)**

with a reaction switch **(E24a)** S = f_liq * sigmoid((T−T_sol)/15 K): chemistry needs a liquid bath and a slag–metal interface, both absent at charge-in — without the gate the model runs BOF chemistry on a pile of cold scrap.

**Decarburisation (E25–E26).** Above the critical carbon C* the rate is oxygen-supply-limited; below C* it switches to mass-transfer control — this reproduces the classical decarburisation plateau and tail.

**Oxygen ledger (E27).** A single supply/demand rule reproduces the classic BOF/EAF phenomenology:

supply = 2*(V_O2*eta_O2 + V_air*0.21)/22.414 mol O/s **(E27a)** — in an IF the lance term is zero and **air ingress is the only oxygen source**, which is what maintains a few percent FeO in the slag and reproduces IF melt loss. Demand is the stoichiometric O for the kinetic rates, ranked Si → C → Mn → P by oxygen affinity at 1873 K. A deficit scales the rates and draws down FeO (the slag is the buffer); a surplus slags iron as FeO at a fraction f_FeO*(1 − X_FeO/0.45) **(E27b)** — the slag saturates near X_FeO ≈ 0.45, beyond which oxidised iron re-reduces from the emulsion.

**Decarburisation by iron oxide (E27c).** When the ledger runs a deficit and slag FeO supplies the oxygen — deliberate ore / mill-scale practice, or simply a carbon boil eating the slag — the net reaction is

> (FeO) + [C] → Fe(l) + CO(g),  dH ≈ **+100 kJ/mol CO** at 1873 K  ≡  **+1.39 MJ per kg FeO reduced**

i.e. **endothermic**, which is why mill scale and ore are bath coolants as well as decarburisers. Built from the two standard half-reactions on 1-wt% Henrian standard states [Turkdogan, *Fundamentals of Steelmaking*, 1996; Fruehan (ed.), *The Making, Shaping and Treating of Steel*, 11th ed., Ch. 2]:

> FeO(l) = Fe(l) + [O]   ΔG° ≈ 121 000 − 52.3·T  J/mol
> [C] + [O] = CO(g)      ΔG° ≈ −22 200 − 38.34·T  J/mol
> sum: ΔG° ≈ 99 800 − 90.6·T  →  ΔG(1873 K) ≈ −70 kJ/mol (spontaneous, as observed)

Note the value is per **kg of FeO** (0.07185 kg/mol), not per kg of Fe produced (0.05585 kg/mol) — the two differ by 29 % and confusing them is an easy and costly slip. The dissolved-species value (~100 kJ/mol) is correctly *lower* than the pure-solid textbook reaction FeO(s) + C(gr) (~+156 kJ/mol).

**Implementation constraint (do not tune independently).** The model books the reaction through two configured enthalpies whose *difference* must reproduce the value above:

> C_to_CO   11 100 kJ/kg C  × 0.012 kg/mol   = 133.2 kJ/mol (exothermic)
> Fe_to_FeO  4 170 kJ/kg Fe × 0.05585 kg/mol = 232.9 kJ/mol (exothermic)
> difference = **99.7 kJ/mol CO** = 1.39 MJ/kg FeO (endothermic)

Oxygen from the lance credits the Fe→FeO formation enthalpy; oxygen from the slag pays it back. A max(·,0) clamp on that term — a classic mistake — turns ore additions into a phantom heat source and violates the first law. The reduced iron is credited to the metal, the carbon leaves through the CO counter, and the element ledger (E61) closes across the event. `test_reaction_enthalpies_hess_consistent` asserts the constraint in CI.

Reference check on the 12 t IF: 120 kg of mill scale into a 1.19 %C bath gives ΔC = −0.130 % (stoichiometric ceiling −0.162 %), ΔFe = **+90.2 kg** (ceiling +90.4 — within 0.2 %), and ΔT = **−29.5 °C**, with the ledger at 0.86 %.

**Dephosphorisation (E17 + E24)** relaxes toward the Healy partition; it strengthens with basicity, FeO and falling temperature — exactly the lever set the advisory exposes. **Desulphurisation (E29)** (CaO)+[S]→(CaS)+[O] is **not** sign-clamped: at high oxygen activity the slag returns sulphur to the metal — a real reversal that a one-sided rate law hides — and reverse transfer is bounded by the sulphur actually held in the slag (mass conservation).

Reaction enthalpies (kJ per kg element oxidised at ~1873 K) enter Q_chem in (E3): C→CO 11.1 MJ, C→CO2 32.8 MJ, Si→SiO2 27.8 MJ, Mn→MnO 7.0 MJ, Fe→FeO 4.8 MJ, plus the post-combustion return (E30).

---

## 10. Numerical scheme

RK4 with automatic sub-stepping keyed to |dT_b/dt| (≤ 5 K per sub-step), positivity projection on masses, and the implicit wall solve (E5d) per sub-step. Charge-in is stiff; melt-down and refining are not — sub-stepping costs nothing when idle and keeps the integrator explicit and edge-cheap (no compiled code; a 77-minute IF heat simulates in ~4 s of pure Python at dt = 2 s). dt-insensitivity is verified from 1 s to 5 s (endpoint shifts < 1 °C, < 1 kWh/t). Temperature states carry a wide physical clamp as a numerical safety net that never engages on a healthy trajectory.

---

## 11. Layer 3a — State and parameter estimation (EKF, E31–E33)

Augmented state z = [x, theta], theta = {eta_el, UA-scale, k_C-scale} as a bounded random walk. Prediction integrates the full physics; the update uses whatever sensors the SKU provides — `SensorConfig` builds the observation model, and this is precisely where Lite and Pro differ:

> predict: z ← f(z), P ← F P F^T + Q;  update: K = P H^T (H P H^T + R)^{-1}, Joseph form   **(E31–E32)**
> offline sensors are masked per tick via an `active` vector   **(E33)**

Jacobians by central finite differences on the step map. Reference behaviour on a deliberately mismatched plant (eta_true = 0.90, UA_true = 1.35; one pyrometer + load cells + three immersion dips): sigma_T collapses 35 → ~4 °C, eta-hat converges to 0.91 within one heat, final bath-temperature error +1.6 °C, and the whole 75-minute heat filters in under a minute of compute at a 30 s cadence. The tracked theta doubles as condition monitoring: a drifting UA-scale **is** the lining-wear signal.

## 12. Layer 2 — Machine-learning residuals (E34)

Feature vector: charge masses and believed assay, energy and power-on time, O2, fluxes, hot-heel, lining age, tap target. Heads:

1. **ResidualGPR** — Matérn-3/2 Gaussian process on (measurement − physics) for T and C; the GP posterior variance is the honesty term in the advisory gate.
2. **EndpointGBM** — quantile gradient boosting (P10/P50/P90) as a nonparametric cross-check.
3. **HybridEndpointModel** — precision-weighted fusion of physics+GP and GBM with maturity tiers (cold-start < 200 heats: physics-dominant; deployable < 1000; calibrated ≥ 2000), matching the deployment brief's data policy.
4. **TrampSoftSensor (E34)** — Cu/Sn/Cr = mass-balance prior from the charge sheet + ridge correction from spectrometer history: tramps do not oxidise, so the prior alone is already good and the ML only corrects assay bias.
5. **DriftMonitor** — PSI on every feature (> 0.10 investigate, > 0.25 regime change) plus rolling MAPE; either alarm suspends ML advice and demotes the system to physics-only with widened sigma. This is the engineering answer to the silent-drift failure mode of pure-AI melt models.

Training discipline: **time-ordered splits only** — random k-fold leaks the lining campaign and flatters the claim.

## 13. Layer 3b — Charge-mix optimisation (E40–E45)

Least-cost LP (MILP with integer lots optional): minimise sum (price_m + tariff*energy_m) x_m **(E40)** subject to the liquid-yield equality sum y_m x_m = M_target **(E41)**, elemental windows through post-melt recovery lo ≤ sum rec_i c_{i,m} y_m x_m / M ≤ hi **(E42)**, hard tramp walls (recovery = 1, no metallurgical sink) **(E43)**, availability / minimum-lot bounds **(E44)** and an optional energy budget **(E45)**. `shadow_prices()` reports the rupee value of relaxing each tramp wall — the commercial number (reference run: the 0.25 % Cu ceiling binds and costs ₹173 per tonne liquid per 0.01 % of relaxation; "is your customer's spec really 0.25, or is that a habit?"). Works from heat #1 with zero history.

## 14. Layer 3c — Receding-horizon control advice (E50–E53)

Piecewise-constant power/O2 blocks over the remaining heat; SLSQP on

> J = w_E dE + w_T (T_tap−T*)² + w_C (C_tap−C*)² + w_dP sum(dP)² + barriers   **(E50, E52)**

with hot-face-limit and overheat barriers, a hard penalty on unmelted charge at horizon end, projection of the optimum onto the operator's discrete tap set **(E51)** and tap-time advice **(E53)**. Output is a recommendation list rendered as operator actions ("t+61 min: power tap 3"); there is **no PLC write path in the codebase** — Phase-1 advisory-only is enforced structurally, not by policy.

## 15. Advisory layer

Traffic-light grading (GREEN / YELLOW / RED / SUSPENDED) per quantity with uncertainty gating: advice is suspended when sigma exceeds config thresholds or the drift monitor alarms — the system says "I don't know" rather than guessing. All texts render in English and Hindi. Every recommendation, the operator's actual action and the outcome are logged (`HeatLogEntry`) — simultaneously the training set, the audit trail, and the evidence base for shared-savings billing.

## 16. Calibration, identifiability, and the honest accuracy claim

Per-plant calibration (`calibrate.py`) proceeds in a fixed order. First fit eta_el and UA on energy data alone — the only parameters identifiable without chemistry instrumentation, and the ones carrying the kWh/t claim; heats with a wide spread of tap-to-tap times are required to separate standing loss from throughput loss. Then fit k_C and gamma_FeO on bath-carbon samples with step one held; without samples, leave nominal and inflate sigma_C — do not pretend. Identifiability is checked before the fit is trusted: the correlation matrix of the estimate and cond(J^T J) are reported, and |rho(eta, UA)| > 0.95 means the heats are all the same length and one number has been fitted, not two. The cheapest cure is the **standing-loss test (E60)**: hold a full bath power-off for 20–30 minutes; UA = M c_p |dT/dt| / (T_b − T_amb) directly, de-correlating eta in every later fit. One heat of lost production buys an identifiable model.

Accuracy claims (±15 °C, ±0.02 %C) are only meaningful with three qualifiers, all implemented in `metrics.py`: state the statistic (hit-rate at tolerance, bias, sigma, P95 — `endpoint_hit_rate`); state the split (time-ordered only — `time_ordered_split`); and subtract the reference, sigma_model² = sigma_observed² − sigma_reference², because a drop-cell has its own ±3–5 °C.

Savings attribution uses regression adjustment (`savings_attribution`): fit SEC on pre-deployment covariates, predict the trial period, attribute only the residual gap — because the plant will change scrap and grade mix mid-trial and a raw before/after will be (rightly) challenged. `economics` exposes every term — tariff, tonnes, opex, capex, carbon value — as a disputable line item and reports payback and 5-year NPV at 12 %.

## 17. Customising to a new plant or client

Copy a reference YAML (`configs/if_msme_12t.yaml` or `configs/eaf_50t.yaml`) and work down the file. `plant`: furnace type, heat size, tap temperature, grade targets, heats per year. `geometry`: diameter, bath height, freeboard, lid coverage from the GA drawing — heat-transfer areas derive automatically (E5a). `lining.layers`: the actual refractory stack with vendor-datasheet k, rho, c_p, and `outer_bc` set to coil or shell. `electrical`: rated power, converter efficiency, discrete tap levels, and either the IF coupling curve or the EAF arc/foaming parameters. `sensors`: exactly what exists on the shop floor — this alone reconfigures the EKF observation model, and is the Lite/Pro SKU switch. `slag` and `kinetics`: target basicity/FeO and interface area, for which the shipped defaults are sane starting points. `economics`: tariff, capex, opex, baseline SEC. Then run the §16 audit fits, commission with physics-only advice, and let the maturity tiers open up ML as heats accumulate.

Furnace-type notes. **IF**: no lance — air ingress is the only oxygen source; power splits by mass (E23); coil outer boundary; mild slag chemistry. **EAF**: foaming bonus in (E22); shell outer boundary with convection+radiation; strong FeO chemistry; the off-gas analyser enables the CO observation in the EKF. **BOF**: set electrical power to zero and drive with the lance through (E27); the oxygen ledger and thermochemistry carry over unchanged.

## 18. Validation protocol and limitations

Shipped checks: element ledger < 1 % (E61); energy closure ≈ 2–2.6 % (E62) with the residual sources documented in §4.3; dt-insensitivity 1→5 s; EKF convergence on mismatched synthetic plants; and the virtual plant (`simulator.py`) with deliberate corruptions — parameter mismatch, lining-wear drift, 12 % assay error, sloppy operator power profiles, sensor noise and dropout, and a scrap-supplier regime change at a chosen heat — for rehearsing every failure mode before touching a furnace. Anything the ML cannot learn on that simulator should not be promised on a slide.

Known limitations, stated deliberately: a single lumped bath zone (no thermal stratification — acceptable for EM-stirred IFs, marginal for large quiet EAFs); slag as a well-mixed phase (no foaming-height dynamics beyond the efficiency bonus); first-order dissolution (no shell-growth sub-model — the flux cap and tau presets absorb this); the Wagner formalism at fixed interaction parameters (swap in a FactSage coupling for high-alloy grades); and the (E3c) no-bath regularisation, which trades ~2 % of first-law closure for unconditional robustness at charge-in. Each limitation is a config or module boundary, not a rewrite.

---

*Equation index: E0 hybrid decomposition · E1–E2 mass balances · E3(a–c) bath energy, freeze, no-bath · E4 solid heating/melting · E5(a–d) geometry, layered wall, boundaries, implicit solve · E6–E20 thermochemistry · E21–E23 electrical · E24–E30 kinetics, oxygen ledger and E27c FeO+[C] reduction · E7(a–c) dissolution · E28(b) melt area and flux cap · E31–E33 EKF · E34 tramp soft sensor · E40–E45 charge LP · E50–E53 MPC · E60 standing-loss test · E61 element ledger · E62 first-law closure.*
