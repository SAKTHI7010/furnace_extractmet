"""
config.py — every plant-, client- and furnace-specific number lives here.

Design rule: NO magic numbers inside the physics or ML code. If a quantity
could differ between two customers, it belongs in a YAML file that maps onto
these dataclasses. That is the whole basis of "same engine, different plant".
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import yaml


# --------------------------------------------------------------------------
# Sub-configs
# --------------------------------------------------------------------------
@dataclass
class PlantMeta:
    name: str = "Representative MSME IF plant"
    client_id: str = "DEMO"
    furnace_type: str = "IF"              # IF | EAF | BOF
    heat_size_t: float = 12.0             # nominal tap weight, tonnes
    max_charge_t: float = 14.0            # crucible capacity
    heats_per_year: int = 2500
    tap_temperature_C: float = 1620.0
    target_carbon_pct: float = 0.20
    grade: str = "IS 2062 E250"


@dataclass
class ElectricalConfig:
    rated_power_kW: float = 6000.0
    eta_converter: float = 0.94           # grid -> coil (SCR/IGBT losses)
    eta_coupling_max: float = 0.88        # coil -> charge at full fill (IF)
    coupling_fill_ref: float = 0.35       # fill fraction scale in eta(fill)
    eta_arc_bath: float = 0.72            # EAF: arc power fraction into bath
    eta_arc_foamed_bonus: float = 0.10    # EAF: extra when slag foaming active
    power_factor: float = 0.85
    tap_levels_kW: List[float] = field(
        default_factory=lambda: [0, 1500, 3000, 4200, 5100, 6000])


@dataclass
class ThermalConfig:
    cp_liquid_kJ_kgK: float = 0.82
    cp_solid_kJ_kgK: float = 0.70
    cp_slag_kJ_kgK: float = 1.20
    # Latent heat of fusion of iron: 247 kJ/kg (13.81 kJ/mol). Verified against
    # CRC Handbook 104th ed. and the iron melting-curve literature. The older
    # 272 kJ/kg used here previously was ~10 % high.
    L_fusion_kJ_kg: float = 247.0
    T_solidus_C: float = 1465.0
    T_liquidus_C: float = 1495.0
    T_ambient_C: float = 35.0
    h_solid_liquid_W_m2K: float = 1500.0  # bath -> unmelted scrap
    q_max_scrap_kW_m2: float = 350.0      # cap: shell-growth limited flux
    A_solid_ref_m2: float = 14.0          # scrap surface at full charge
    emissivity_top: float = 0.85
    # Set these <= 0 to have them DERIVED from GeometryConfig (E5a).
    A_top_m2: float = -1.0                # effective radiating (open-top) area
    A_wall_m2: float = -1.0               # wetted wall area


@dataclass
class GeometryConfig:
    """
    Furnace design. Heat-transfer areas are DERIVED from these dimensions
    (E5a) unless the ThermalConfig areas are explicitly overridden — so the
    same physics reads a 10 t crucible or a 100 t shell correctly by geometry
    alone. Fill in from the furnace GA drawing during the pre-install audit.
    """
    D_inner_m: float = 1.80               # working diameter at bath level
    H_bath_m: float = 1.50                # bath depth at nominal heat size
    H_freeboard_m: float = 0.40
    lid_coverage: float = 0.85            # fraction of top closed by lid/hood
    shape: str = "cylinder"               # cylinder | conical (EAF shell)

    def A_bath_top_m2(self) -> float:
        import math
        return math.pi * self.D_inner_m ** 2 / 4.0

    def A_top_open_m2(self) -> float:
        return self.A_bath_top_m2() * max(1.0 - self.lid_coverage, 0.0)

    def A_wall_wetted_m2(self) -> float:
        import math
        return math.pi * self.D_inner_m * self.H_bath_m

    def A_wall_total_m2(self) -> float:
        import math
        return math.pi * self.D_inner_m * (self.H_bath_m + self.H_freeboard_m)


@dataclass
class RefractoryLayer:
    """One radial layer of the wall, hot face outward."""
    name: str = "working"
    thickness_m: float = 0.10
    k_W_mK: float = 1.6                   # thermal conductivity
    rho_kg_m3: float = 2600.0
    cp_kJ_kgK: float = 1.05
    T_limit_C: float = 1700.0             # service limit of this material


@dataclass
class LiningConfig:
    """
    Multi-layer refractory wall. Default = a typical coreless-IF build:
      working lining (dry-vibratable silica ramming mass)
      -> backup (mica/ceramic-paper slip plane)
      -> grout/coil cement, then the water-cooled coil (as outer BC).
    For an EAF replace with magnesia-carbon brick -> insulating brick ->
    steel shell, and set outer BC to natural convection + radiation.
    """
    n_nodes: int = 8
    r_inner_m: float = 0.90
    layers: List[RefractoryLayer] = field(default_factory=lambda: [
        RefractoryLayer("working_silica", 0.11, 1.6, 2600.0, 1.05, 1700.0),
        RefractoryLayer("mica_slip", 0.004, 0.20, 700.0, 0.90, 900.0),
        RefractoryLayer("coil_grout", 0.030, 1.2, 2200.0, 1.00, 1100.0),
    ])
    h_inner_W_m2K: float = 1200.0         # melt -> hot face (forced conv, EM stir)
    h_solid_wall_W_m2K: float = 150.0     # cold scrap -> hot face (contact only)
    grading_ratio: float = 2.2            # node thickness growth, hot face -> cold
    # ---- outer boundary: EITHER water-cooled coil OR free shell -----------
    outer_bc: str = "coil"                # "coil" (IF) | "shell" (EAF/BOF)
    h_outer_W_m2K: float = 900.0          # coil: cold face -> jacket water
    T_coolant_C: float = 45.0
    shell_emissivity: float = 0.80        # shell: radiation to surroundings
    h_shell_conv_W_m2K: float = 12.0      # shell: natural convection
    wear_fraction: float = 0.0            # 0 = new lining, 0.5 = half eroded
    hot_face_limit_C: float = 1700.0

    # legacy single-layer accessors (kept so old YAMLs still load)
    @property
    def thickness_m(self) -> float:
        return sum(l.thickness_m for l in self.layers)


@dataclass
class OffgasConfig:
    post_combustion_ratio: float = 0.15   # fraction of CO burned in freeboard
    eta_pc_heat_return: float = 0.35      # of PC heat returned to bath
    T_offgas_C: float = 1150.0
    cp_gas_kJ_kgK: float = 1.15
    air_ingress_Nm3_per_t_per_h: float = 0.6


@dataclass
class KineticsConfig:
    """Mass-transfer coefficients, m/s, at the slag-metal interface."""
    A_slag_metal_m2: float = 8.0
    rho_metal_kg_m3: float = 7000.0
    k_C: float = 6.0e-4
    k_Si: float = 2.5e-3
    k_Mn: float = 1.5e-3
    k_P: float = 8.0e-4
    k_S: float = 6.0e-4
    C_critical_pct: float = 0.30          # below this, decarb is MT-limited
    eta_O2_utilisation: float = 0.90      # lance O2 actually reaching bath
    stirring_multiplier: float = 1.0      # bottom stirring / EM stirring boost


@dataclass
class SlagConfig:
    gamma_FeO: float = 1.6                # Raoultian activity coefficient
    target_basicity_B2: float = 1.6
    target_FeO_pct: float = 18.0          # EAF/BOF; ~2-5 for IF
    initial_slag_kg_per_t: float = 12.0
    surplus_O2_to_FeO: float = 0.35       # fraction of surplus lance O2 that slags
    initial_composition: Dict[str, float] = field(default_factory=lambda: {
        "CaO": 0.35, "SiO2": 0.30, "FeO": 0.15,
        "MgO": 0.08, "MnO": 0.05, "Al2O3": 0.07, "CaS": 0.0})


@dataclass
class ReactionEnthalpy:
    """kJ released per kg of ELEMENT oxidised, at ~1873 K (exothermic > 0)."""
    C_to_CO: float = 11_100.0
    C_to_CO2: float = 32_800.0
    Si_to_SiO2: float = 27_750.0
    Mn_to_MnO: float = 7_000.0
    # (E27c CONSTRAINT) These two must be mutually consistent, because their
    # DIFFERENCE is the enthalpy of (FeO)+[C] -> Fe(l)+CO(g), which is fixed by
    # thermochemistry at approximately +100 kJ per mol CO (endothermic) at
    # 1873 K [Turkdogan, Fundamentals of Steelmaking; Fruehan, MSTS 11th ed.]:
    #     C_to_CO  11_100 kJ/kg C  x 0.012  kg/mol = 133.2 kJ/mol  (exothermic)
    #     Fe_to_FeO 4_170 kJ/kg Fe x 0.05585 kg/mol = 232.9 kJ/mol (exothermic)
    #     difference = 99.7 kJ/mol CO  ->  1.39 MJ per kg FeO reduced (ENDO)
    # The previous 4_490 made the pair imply +118 kJ/mol, ~18 % too endothermic
    # on every ore/mill-scale addition. Do not tune these independently.
    Fe_to_FeO: float = 4_170.0
    P_to_P2O5: float = 23_600.0
    CO_to_CO2_per_kg_CO: float = 10_110.0  # post-combustion


@dataclass
class SensorConfig:
    """Which measurements exist. Drives the EKF observation model and the SKU."""
    has_power_meter: bool = True
    has_pyrometer: bool = True
    has_immersion_tc: bool = True
    has_load_cells: bool = True
    has_offgas_analyser: bool = False
    has_sublance: bool = False
    has_fibre_dts: bool = False
    sigma_T_pyrometer_C: float = 12.0
    sigma_T_immersion_C: float = 4.0
    sigma_power_kW: float = 25.0
    sigma_offgas_pct: float = 0.5
    sample_period_s: float = 1.0


@dataclass
class MLConfig:
    min_heats_coldstart: int = 200
    min_heats_deployable: int = 1000
    min_heats_calibrated: int = 2000
    gpr_kernel: str = "matern32"
    gpr_length_scale: float = 1.0
    gpr_noise: float = 1e-2
    gbm_max_iter: int = 400
    gbm_learning_rate: float = 0.05
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    psi_alarm: float = 0.25
    mape_alarm_pct: float = 2.0


@dataclass
class AdvisoryConfig:
    T_tolerance_green_C: float = 8.0
    T_tolerance_yellow_C: float = 20.0
    C_tolerance_green_pct: float = 0.010
    C_tolerance_yellow_pct: float = 0.025
    sigma_suspend_multiplier: float = 2.5   # widen/suspend advice above this
    language: str = "both"                  # en | hi | both


@dataclass
class EconomicsConfig:
    # Indian HT industrial power, FY2025-26. All-in grid rates run ~Rs 6.0-8.5/kWh
    # depending on state, demand charges and ToD; plants on open access or group
    # captive achieve ~Rs 5.0-6.5/kWh effective (CEEW: 20-30 % below grid).
    # Rs 7.0 is a defensible mid-band default; set per plant from the tariff order.
    tariff_INR_per_kWh: float = 7.0
    # CEA CO2 Baseline Database v21.0 (Nov 2025, FY2024-25): all-India weighted
    # average 0.7117 tCO2/MWh; Combined Margin 0.7383 (use CM for CDM/project
    # accounting). The previous 0.82 predates the grid's recent decarbonisation.
    grid_EF_tCO2_per_MWh: float = 0.712
    steel_price_INR_per_t: float = 50_000.0
    capex_INR: float = 20_00_000.0
    opex_INR_per_year: float = 7_50_000.0
    # Scrap-based Indian coreless IF: 550-650 kWh/t typical. DRI-heavy charges
    # run 650-800 kWh/t. Set from the plant's own audited baseline.
    baseline_SEC_kWh_per_t: float = 615.0
    carbon_price_INR_per_tCO2: float = 500.0


@dataclass
class NumericsConfig:
    dt_s: float = 1.0
    integrator: str = "rk4"               # rk4 | semi_implicit_euler
    dt_mpc_s: float = 15.0
    max_heat_minutes: float = 120.0


# --------------------------------------------------------------------------
# Master config
# --------------------------------------------------------------------------
@dataclass
class PlantConfig:
    plant: PlantMeta = field(default_factory=PlantMeta)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    electrical: ElectricalConfig = field(default_factory=ElectricalConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    lining: LiningConfig = field(default_factory=LiningConfig)
    offgas: OffgasConfig = field(default_factory=OffgasConfig)
    kinetics: KineticsConfig = field(default_factory=KineticsConfig)
    slag: SlagConfig = field(default_factory=SlagConfig)
    enthalpy: ReactionEnthalpy = field(default_factory=ReactionEnthalpy)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    advisory: AdvisoryConfig = field(default_factory=AdvisoryConfig)
    economics: EconomicsConfig = field(default_factory=EconomicsConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)

    metal_species: List[str] = field(default_factory=lambda: [
        "Fe", "C", "Si", "Mn", "P", "S", "Cr", "Cu", "Ni"])
    slag_species: List[str] = field(default_factory=lambda: [
        "FeO", "SiO2", "CaO", "MgO", "MnO", "Al2O3", "P2O5", "CaS"])

    # ---------------- serialisation ----------------
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    @staticmethod
    def from_dict(d: dict) -> "PlantConfig":
        sub = {
            "plant": PlantMeta, "geometry": GeometryConfig,
            "electrical": ElectricalConfig,
            "thermal": ThermalConfig, "lining": LiningConfig,
            "offgas": OffgasConfig, "kinetics": KineticsConfig,
            "slag": SlagConfig, "enthalpy": ReactionEnthalpy,
            "sensors": SensorConfig, "ml": MLConfig,
            "advisory": AdvisoryConfig, "economics": EconomicsConfig,
            "numerics": NumericsConfig,
        }
        import dataclasses as _dc
        kwargs = {}
        for key, cls in sub.items():
            payload = dict(d.get(key) or {})
            if key == "lining" and payload.get("layers"):
                payload["layers"] = [RefractoryLayer(**l) for l in payload["layers"]]
            # tolerate legacy / unknown keys so old client YAMLs keep loading
            valid = {f.name for f in _dc.fields(cls)}
            payload = {k: v for k, v in payload.items() if k in valid}
            kwargs[key] = cls(**payload) if payload else cls()
        for key in ("metal_species", "slag_species"):
            if key in d and d[key]:
                kwargs[key] = list(d[key])
        return PlantConfig(**kwargs)


def load_config(path: Optional[str] = None) -> PlantConfig:
    """Load a plant YAML. With no path, returns the built-in MSME IF default."""
    if path is None:
        return PlantConfig()
    with open(path) as fh:
        return PlantConfig.from_dict(yaml.safe_load(fh))
