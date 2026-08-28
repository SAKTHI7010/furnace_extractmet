"""Regenerate the reference plant configs. Run: python configs/make_configs.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smartmelt.config import *
from smartmelt.config import RefractoryLayer, GeometryConfig

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- SmartMelt Lite
cfg = PlantConfig()
cfg.plant = PlantMeta(name="Industry-X MSME induction furnace", client_id="IF_12T",
                      furnace_type="IF", heat_size_t=12.0, max_charge_t=14.0,
                      heats_per_year=2500, tap_temperature_C=1620.0,
                      target_carbon_pct=0.20, grade="IS 2062 E250")
cfg.geometry = GeometryConfig(D_inner_m=1.80, H_bath_m=1.50, H_freeboard_m=0.35,
                              lid_coverage=0.85)
cfg.electrical = ElectricalConfig(rated_power_kW=6000.0, eta_converter=0.94,
                                  eta_coupling_max=0.88, coupling_fill_ref=0.35,
                                  tap_levels_kW=[0, 1500, 3000, 4200, 5100, 6000])
cfg.thermal = ThermalConfig(A_solid_ref_m2=14.0, q_max_scrap_kW_m2=350.0)
cfg.lining = LiningConfig(outer_bc="coil", h_outer_W_m2K=900.0, T_coolant_C=45.0,
                          layers=[RefractoryLayer("working_silica", 0.11, 1.6, 2600.0, 1.05, 1700.0),
                                  RefractoryLayer("mica_slip", 0.004, 0.20, 700.0, 0.90, 900.0),
                                  RefractoryLayer("coil_grout", 0.030, 1.2, 2200.0, 1.00, 1100.0)])
cfg.sensors = SensorConfig(has_offgas_analyser=False, has_sublance=False,
                           has_immersion_tc=True, has_pyrometer=True)
cfg.kinetics = KineticsConfig(A_slag_metal_m2=2.5, k_S=2.0e-4, k_P=3.0e-4,
                              stirring_multiplier=0.8)
cfg.offgas = OffgasConfig(post_combustion_ratio=0.05, air_ingress_Nm3_per_t_per_h=0.6)
cfg.slag = SlagConfig(gamma_FeO=1.4, target_basicity_B2=1.2, target_FeO_pct=4.0,
                      initial_slag_kg_per_t=10.0, surplus_O2_to_FeO=0.60,
                      initial_composition={"CaO": 0.30, "SiO2": 0.40, "FeO": 0.10,
                                           "MgO": 0.08, "MnO": 0.05, "Al2O3": 0.07,
                                           "CaS": 0.0})
cfg.numerics = NumericsConfig(dt_s=2.0)
cfg.economics = EconomicsConfig(tariff_INR_per_kWh=7.0, capex_INR=800_000.0,
                                opex_INR_per_year=600_000.0,
                                baseline_SEC_kWh_per_t=615.0)
cfg.save(os.path.join(HERE, "if_msme_12t.yaml"))

# ---------------------------------------------------------------- SmartMelt Pro
e = PlantConfig()
e.plant = PlantMeta(name="Mid-tier EAF", client_id="EAF_50T", furnace_type="EAF",
                    heat_size_t=50.0, max_charge_t=55.0, heats_per_year=4000,
                    tap_temperature_C=1640.0, target_carbon_pct=0.06, grade="Billet")
e.electrical = ElectricalConfig(rated_power_kW=30000.0, eta_converter=0.97,
                                eta_arc_bath=0.72, eta_arc_foamed_bonus=0.12,
                                tap_levels_kW=[0, 8000, 15000, 21000, 26000, 30000])
e.geometry = GeometryConfig(D_inner_m=4.4, H_bath_m=1.1, H_freeboard_m=2.2,
                            lid_coverage=0.93)
e.thermal = ThermalConfig(A_solid_ref_m2=55.0, h_solid_liquid_W_m2K=1500.0,
                          q_max_scrap_kW_m2=400.0)
e.lining = LiningConfig(outer_bc="shell", shell_emissivity=0.80,
                        h_shell_conv_W_m2K=12.0, n_nodes=8,
                        layers=[RefractoryLayer("MgO_C_working", 0.30, 6.0, 2950.0, 1.05, 1800.0),
                                RefractoryLayer("safety_magnesite", 0.115, 4.5, 2900.0, 1.05, 1700.0),
                                RefractoryLayer("insulation_board", 0.012, 0.25, 900.0, 1.00, 1000.0),
                                RefractoryLayer("steel_shell", 0.030, 45.0, 7850.0, 0.49, 600.0)])
e.sensors = SensorConfig(has_offgas_analyser=True, has_immersion_tc=True,
                         has_pyrometer=True)
e.kinetics = KineticsConfig(A_slag_metal_m2=28.0, stirring_multiplier=1.2)
e.offgas = OffgasConfig(post_combustion_ratio=0.20, air_ingress_Nm3_per_t_per_h=0.0)
e.slag = SlagConfig(gamma_FeO=1.7, target_basicity_B2=1.8, target_FeO_pct=22.0,
                    initial_slag_kg_per_t=45.0, surplus_O2_to_FeO=0.30,
                    initial_composition={"CaO": 0.38, "SiO2": 0.18, "FeO": 0.22,
                                         "MgO": 0.09, "MnO": 0.05, "Al2O3": 0.08,
                                         "CaS": 0.0})
e.numerics = NumericsConfig(dt_s=2.0)
e.economics = EconomicsConfig(tariff_INR_per_kWh=7.5, capex_INR=2_200_000.0,
                              opex_INR_per_year=900_000.0, baseline_SEC_kWh_per_t=560.0)
e.save(os.path.join(HERE, "eaf_50t.yaml"))
print("wrote if_msme_12t.yaml and eaf_50t.yaml")
