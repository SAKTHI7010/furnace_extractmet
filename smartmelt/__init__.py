"""
SmartMelt — hybrid first-principles + machine-learning melt model.

Layer 1 : physics   (thermo.py, physics.py)
Layer 2 : ML        (ml.py)
Layer 3 : control   (ekf.py, mpc.py, chargemix.py, advisory.py)

Supporting: simulator.py (virtual plant), calibrate.py, metrics.py, config.py
"""

__version__ = "0.5.0"

from .config import PlantConfig, load_config
from .physics import FurnaceModel, HeatInputs, Addition
from .ml import HybridEndpointModel, ResidualGPR, EndpointGBM, DriftMonitor, TrampSoftSensor
from .ekf import ExtendedKalmanFilter, build_default_ekf
from .chargemix import ChargeMixOptimiser, Material
from .mpc import MeltMPC
from .advisory import AdvisoryEngine, Verdict
from .simulator import VirtualPlant
from .metrics import psi, mape, endpoint_hit_rate, energy_kpis, economics
from .calibrate import calibrate_physics

__all__ = [
    "PlantConfig", "load_config",
    "FurnaceModel", "HeatInputs", "Addition",
    "HybridEndpointModel", "ResidualGPR", "EndpointGBM", "DriftMonitor", "TrampSoftSensor",
    "ExtendedKalmanFilter", "build_default_ekf",
    "ChargeMixOptimiser", "Material",
    "MeltMPC",
    "AdvisoryEngine", "Verdict",
    "VirtualPlant",
    "psi", "mape", "endpoint_hit_rate", "energy_kpis", "economics",
    "calibrate_physics",
]
