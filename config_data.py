"""
Configuration data for PV measurement equipment, technologies, and reference devices.
This module provides structured data that can be easily extended.
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field


# ============================================================================
# 1. PV MODULE TECHNOLOGIES
# ============================================================================

@dataclass
class ModuleTechnology:
    """PV module technology specification."""
    name: str
    typical_efficiency: float  # %
    typical_temp_coeff: float  # %/°C
    typical_cell_count: int
    description: str
    typical_bifaciality: float = 0.0  # For bifacial modules


PV_TECHNOLOGIES = {
    "PERC": ModuleTechnology(
        name="PERC (Passivated Emitter and Rear Cell)",
        typical_efficiency=21.0,
        typical_temp_coeff=-0.35,
        typical_cell_count=60,
        description="Standard monofacial PERC technology",
        typical_bifaciality=0.0
    ),
    "TOPCon": ModuleTechnology(
        name="TOPCon (Tunnel Oxide Passivated Contact)",
        typical_efficiency=22.5,
        typical_temp_coeff=-0.30,
        typical_cell_count=60,
        description="Advanced n-type technology with better temperature performance",
        typical_bifaciality=0.75
    ),
    "HJT": ModuleTechnology(
        name="HJT (Heterojunction Technology)",
        typical_efficiency=23.0,
        typical_temp_coeff=-0.25,
        typical_cell_count=60,
        description="Premium technology with lowest temperature coefficient",
        typical_bifaciality=0.92
    ),
    "Perovskite": ModuleTechnology(
        name="Perovskite",
        typical_efficiency=18.0,
        typical_temp_coeff=-0.20,
        typical_cell_count=60,
        description="Emerging technology with excellent low-light performance",
        typical_bifaciality=0.0
    ),
    "Perovskite-Silicon Tandem": ModuleTechnology(
        name="Perovskite-Silicon Tandem",
        typical_efficiency=28.0,
        typical_temp_coeff=-0.22,
        typical_cell_count=60,
        description="Advanced tandem technology for high efficiency",
        typical_bifaciality=0.0
    ),
    "CIGS": ModuleTechnology(
        name="CIGS (Copper Indium Gallium Selenide)",
        typical_efficiency=17.0,
        typical_temp_coeff=-0.32,
        typical_cell_count=0,  # Thin film, no discrete cells
        description="Thin-film technology",
        typical_bifaciality=0.0
    ),
    "CdTe": ModuleTechnology(
        name="CdTe (Cadmium Telluride)",
        typical_efficiency=16.5,
        typical_temp_coeff=-0.25,
        typical_cell_count=0,
        description="Thin-film technology with good spectral response",
        typical_bifaciality=0.0
    ),
    "Custom": ModuleTechnology(
        name="Custom Technology",
        typical_efficiency=20.0,
        typical_temp_coeff=-0.35,
        typical_cell_count=60,
        description="User-defined custom technology",
        typical_bifaciality=0.0
    )
}


# ============================================================================
# 2. SUN SIMULATOR CONFIGURATIONS
# ============================================================================

@dataclass
class SimulatorConfig:
    """Sun simulator configuration."""
    manufacturer: str
    model: str
    lamp_type: str  # LED, Xenon, Metal Halide, Plasma
    classification: str  # AAA, AA+, A+, BBB, etc.
    typical_uniformity: float  # %
    typical_temporal_instability: float  # %
    typical_spectral_match: str  # A, B, C
    standard_distance_mm: float  # Distance from lamp to test plane
    description: str


SUN_SIMULATORS = {
    # Spire Solar (now Atonometrics)
    "Spire 5600SLP": SimulatorConfig(
        manufacturer="Spire Solar / Atonometrics",
        model="5600SLP",
        lamp_type="Xenon",
        classification="AAA",
        typical_uniformity=2.0,
        typical_temporal_instability=0.5,
        typical_spectral_match="A",
        standard_distance_mm=914.4,
        description="Large-area steady-state solar simulator"
    ),
    "Spire 4600": SimulatorConfig(
        manufacturer="Spire Solar / Atonometrics",
        model="4600",
        lamp_type="Xenon",
        classification="AAA",
        typical_uniformity=2.5,
        typical_temporal_instability=1.0,
        typical_spectral_match="A",
        standard_distance_mm=914.4,
        description="Previous generation solar simulator"
    ),

    # Eternalsun (Meyer Burger)
    "Eternalsun SLP-150": SimulatorConfig(
        manufacturer="Eternalsun / Meyer Burger",
        model="SLP-150",
        lamp_type="LED",
        classification="AAA+",
        typical_uniformity=1.5,
        typical_temporal_instability=0.3,
        typical_spectral_match="A",
        standard_distance_mm=500.0,
        description="Advanced LED-based simulator with superior uniformity"
    ),

    # Avalon (Wavelabs)
    "Avalon Nexun": SimulatorConfig(
        manufacturer="Wavelabs",
        model="Avalon Nexun",
        lamp_type="LED",
        classification="AAA",
        typical_uniformity=2.0,
        typical_temporal_instability=0.5,
        typical_spectral_match="A",
        standard_distance_mm=600.0,
        description="LED simulator for standard modules"
    ),
    "Avalon Perovskite": SimulatorConfig(
        manufacturer="Wavelabs",
        model="Avalon Perovskite",
        lamp_type="LED",
        classification="AAA",
        typical_uniformity=2.0,
        typical_temporal_instability=0.5,
        typical_spectral_match="A",
        standard_distance_mm=600.0,
        description="Optimized for perovskite modules"
    ),
    "Avalon SteadyState": SimulatorConfig(
        manufacturer="Wavelabs",
        model="Avalon SteadyState",
        lamp_type="LED",
        classification="AAA",
        typical_uniformity=1.8,
        typical_temporal_instability=0.3,
        typical_spectral_match="A",
        standard_distance_mm=600.0,
        description="Steady-state LED simulator"
    ),

    # Halm (now EETS - Eternal Energy Test Solutions)
    "Halm Flasher": SimulatorConfig(
        manufacturer="Halm / EETS",
        model="Standard Flasher",
        lamp_type="Xenon",
        classification="AAA",
        typical_uniformity=2.0,
        typical_temporal_instability=1.0,
        typical_spectral_match="A",
        standard_distance_mm=800.0,
        description="Industrial flasher system"
    ),

    # Pasan
    "Pasan HighLIGHT LED": SimulatorConfig(
        manufacturer="Pasan",
        model="HighLIGHT LED",
        lamp_type="LED",
        classification="AAA+",
        typical_uniformity=1.5,
        typical_temporal_instability=0.2,
        typical_spectral_match="A",
        standard_distance_mm=500.0,
        description="High-performance LED simulator"
    ),

    # ReRa Solutions
    "ReRa Tracer": SimulatorConfig(
        manufacturer="ReRa Solutions",
        model="Tracer",
        lamp_type="LED",
        classification="AAA",
        typical_uniformity=2.0,
        typical_temporal_instability=0.5,
        typical_spectral_match="A",
        standard_distance_mm=550.0,
        description="Compact LED simulator"
    ),

    # h.a.l.m (Lumartix)
    "Lumartix Flash": SimulatorConfig(
        manufacturer="Lumartix",
        model="Flash",
        lamp_type="Xenon",
        classification="AAA",
        typical_uniformity=2.0,
        typical_temporal_instability=1.0,
        typical_spectral_match="A",
        standard_distance_mm=750.0,
        description="Flash-based solar simulator"
    ),

    # Atlas Material Testing
    "Atlas SUNTEST": SimulatorConfig(
        manufacturer="Atlas Material Testing",
        model="SUNTEST",
        lamp_type="Xenon",
        classification="AA",
        typical_uniformity=3.0,
        typical_temporal_instability=2.0,
        typical_spectral_match="A",
        standard_distance_mm=700.0,
        description="For material testing and module characterization"
    ),

    # Custom option
    "Custom Simulator": SimulatorConfig(
        manufacturer="Custom",
        model="User Defined",
        lamp_type="Custom",
        classification="Custom",
        typical_uniformity=2.0,
        typical_temporal_instability=1.0,
        typical_spectral_match="A",
        standard_distance_mm=600.0,
        description="User-defined simulator configuration"
    )
}


# ============================================================================
# 3. REFERENCE DEVICE LABORATORIES
# ============================================================================

@dataclass
class ReferenceLab:
    """Reference device calibration laboratory."""
    name: str
    country: str
    lab_type: str  # Primary, Secondary, Accredited
    accreditation: str
    typical_uncertainty_wpvs: float  # % for WPVS cells
    typical_uncertainty_module: float  # % for reference modules
    description: str


REFERENCE_LABS = {
    # Primary Standards Labs
    "NREL": ReferenceLab(
        name="NREL (National Renewable Energy Laboratory)",
        country="USA",
        lab_type="Primary",
        accreditation="ISO 17025, NVLAP",
        typical_uncertainty_wpvs=0.5,
        typical_uncertainty_module=1.5,
        description="US primary PV calibration laboratory"
    ),
    "PTB": ReferenceLab(
        name="PTB (Physikalisch-Technische Bundesanstalt)",
        country="Germany",
        lab_type="Primary",
        accreditation="ISO 17025",
        typical_uncertainty_wpvs=0.4,
        typical_uncertainty_module=1.3,
        description="German national metrology institute"
    ),
    "AIST": ReferenceLab(
        name="AIST (National Institute of Advanced Industrial Science and Technology)",
        country="Japan",
        lab_type="Primary",
        accreditation="ISO 17025",
        typical_uncertainty_wpvs=0.45,
        typical_uncertainty_module=1.4,
        description="Japanese primary standards laboratory"
    ),
    "NIMS": ReferenceLab(
        name="NIMS (National Institute for Materials Science)",
        country="China",
        lab_type="Primary",
        accreditation="ISO 17025, CNAS",
        typical_uncertainty_wpvs=0.5,
        typical_uncertainty_module=1.5,
        description="Chinese national metrology institute"
    ),
    "ISFH": ReferenceLab(
        name="ISFH (Institute for Solar Energy Research Hamelin)",
        country="Germany",
        lab_type="Primary",
        accreditation="ISO 17025, DAkkS",
        typical_uncertainty_wpvs=0.45,
        typical_uncertainty_module=1.4,
        description="German solar research institute with calibration services"
    ),
    "Fraunhofer ISE": ReferenceLab(
        name="Fraunhofer ISE CalLab PV Cells",
        country="Germany",
        lab_type="Primary",
        accreditation="ISO 17025, DAkkS",
        typical_uncertainty_wpvs=0.4,
        typical_uncertainty_module=1.3,
        description="Fraunhofer Institute for Solar Energy Systems"
    ),

    # Secondary/Accredited Labs
    "TUV Rheinland": ReferenceLab(
        name="TÜV Rheinland PTL",
        country="Germany",
        lab_type="Accredited",
        accreditation="ISO 17025, DAkkS",
        typical_uncertainty_wpvs=0.8,
        typical_uncertainty_module=2.0,
        description="Global testing and certification laboratory"
    ),
    "TUV SUD": ReferenceLab(
        name="TÜV SÜD",
        country="Germany",
        lab_type="Accredited",
        accreditation="ISO 17025",
        typical_uncertainty_wpvs=0.8,
        typical_uncertainty_module=2.0,
        description="International testing and certification"
    ),
    "SUPSI": ReferenceLab(
        name="SUPSI PV Lab",
        country="Switzerland",
        lab_type="Accredited",
        accreditation="ISO 17025, SAS",
        typical_uncertainty_wpvs=0.7,
        typical_uncertainty_module=1.8,
        description="University of Applied Sciences and Arts of Southern Switzerland"
    ),
    "PI Berlin": ReferenceLab(
        name="PI Berlin (now part of Kiwa)",
        country="Germany",
        lab_type="Accredited",
        accreditation="ISO 17025, DAkkS",
        typical_uncertainty_wpvs=0.8,
        typical_uncertainty_module=2.0,
        description="Independent testing and engineering services"
    ),
    "DNV": ReferenceLab(
        name="DNV Energy Lab",
        country="Netherlands",
        lab_type="Accredited",
        accreditation="ISO 17025, RvA",
        typical_uncertainty_wpvs=0.8,
        typical_uncertainty_module=2.0,
        description="Global quality assurance and risk management"
    ),
    "RETC": ReferenceLab(
        name="RETC (Renewable Energy Test Center)",
        country="Taiwan",
        lab_type="Accredited",
        accreditation="ISO 17025, TAF",
        typical_uncertainty_wpvs=0.8,
        typical_uncertainty_module=2.0,
        description="Taiwan renewable energy testing center"
    ),
    "CSIRO": ReferenceLab(
        name="CSIRO Energy",
        country="Australia",
        lab_type="Accredited",
        accreditation="ISO 17025, NATA",
        typical_uncertainty_wpvs=0.75,
        typical_uncertainty_module=1.9,
        description="Australian national science research organization"
    ),
    "SERI": ReferenceLab(
        name="SERI (Solar Energy Research Institute of Singapore)",
        country="Singapore",
        lab_type="Accredited",
        accreditation="ISO 17025, SAC",
        typical_uncertainty_wpvs=0.8,
        typical_uncertainty_module=2.0,
        description="Singapore solar research and testing"
    ),
    "CEA-INES": ReferenceLab(
        name="CEA-INES",
        country="France",
        lab_type="Accredited",
        accreditation="ISO 17025, COFRAC",
        typical_uncertainty_wpvs=0.7,
        typical_uncertainty_module=1.8,
        description="French solar energy research institute"
    ),
    "NREL India": ReferenceLab(
        name="NABL Accredited Labs India",
        country="India",
        lab_type="Accredited",
        accreditation="ISO 17025, NABL",
        typical_uncertainty_wpvs=1.0,
        typical_uncertainty_module=2.5,
        description="Indian accredited PV testing laboratories"
    ),
    "Custom Lab": ReferenceLab(
        name="Custom Laboratory",
        country="Custom",
        lab_type="Custom",
        accreditation="Custom",
        typical_uncertainty_wpvs=1.0,
        typical_uncertainty_module=2.0,
        description="User-defined laboratory"
    )
}


# ============================================================================
# 4. MEASUREMENT PARAMETERS AND TYPES
# ============================================================================

MEASUREMENT_TYPES = {
    "STC": {
        "name": "Standard Test Conditions (STC)",
        "conditions": "1000 W/m², 25°C, AM1.5G spectrum",
        "parameters": ["Pmax", "Voc", "Isc", "Vmp", "Imp", "FF"],
        "standard": "IEC 61215, IEC 61730"
    },
    "NMOT": {
        "name": "Nominal Module Operating Temperature (NMOT)",
        "conditions": "800 W/m², NMOT temperature, AM1.5G spectrum",
        "parameters": ["Pmax", "Voc", "Isc", "Vmp", "Imp"],
        "standard": "IEC 61853-1"
    },
    "Low_Irradiance": {
        "name": "Low Irradiance Performance",
        "conditions": "200 W/m², 25°C, AM1.5G spectrum",
        "parameters": ["Pmax", "Voc", "Isc", "Efficiency"],
        "standard": "IEC 61853-1"
    },
    "Temperature_Coefficient": {
        "name": "Temperature Coefficients",
        "conditions": "Variable temperature at 1000 W/m²",
        "parameters": ["α_Isc", "β_Voc", "γ_Pmax"],
        "standard": "IEC 60891, IEC 61853-1"
    },
    "Energy_Rating": {
        "name": "Energy Rating (ER)",
        "conditions": "Various irradiance and temperature conditions",
        "parameters": ["ER_kWh/kWp"],
        "standard": "IEC 61853-3"
    },
    "Bifaciality": {
        "name": "Bifaciality Measurements",
        "conditions": "Front and rear illumination",
        "parameters": ["Bifaciality_Factor", "Bifacial_Gain"],
        "standard": "IEC TS 60904-1-2"
    },
    "IAM": {
        "name": "Incidence Angle Modifier",
        "conditions": "Variable angle of incidence",
        "parameters": ["IAM_curve"],
        "standard": "IEC 61853-2"
    },
    "Spectral_Response": {
        "name": "Spectral Response",
        "conditions": "Monochromatic illumination",
        "parameters": ["SR_curve", "EQE"],
        "standard": "IEC 60904-8"
    }
}


# ============================================================================
# 5. STANDARD SPECTRA
# ============================================================================

STANDARD_SPECTRA = {
    "AM1.5G": {
        "name": "AM1.5 Global",
        "standard": "IEC 60904-3",
        "description": "Standard terrestrial spectrum for flat-plate modules",
        "air_mass": 1.5,
        "integrated_irradiance": 1000.0
    },
    "AM1.5D": {
        "name": "AM1.5 Direct",
        "standard": "ASTM G173",
        "description": "Direct normal irradiance spectrum",
        "air_mass": 1.5,
        "integrated_irradiance": 900.0
    },
    "AM1.0": {
        "name": "AM1.0",
        "standard": "ASTM E490",
        "description": "Air mass 1.0 spectrum",
        "air_mass": 1.0,
        "integrated_irradiance": 1000.0
    },
    "AM0": {
        "name": "AM0 (Extraterrestrial)",
        "standard": "ASTM E490",
        "description": "Extraterrestrial solar spectrum for space applications",
        "air_mass": 0.0,
        "integrated_irradiance": 1367.0
    },
    "Custom": {
        "name": "Custom Spectrum",
        "standard": "User Defined",
        "description": "User-uploaded custom spectrum",
        "air_mass": None,
        "integrated_irradiance": 1000.0
    }
}


# ============================================================================
# 6. CURRENCY CONFIGURATIONS
# ============================================================================

CURRENCIES = {
    "USD": {"symbol": "$", "name": "US Dollar", "typical_rate_to_usd": 1.0},
    "EUR": {"symbol": "€", "name": "Euro", "typical_rate_to_usd": 1.1},
    "INR": {"symbol": "₹", "name": "Indian Rupee", "typical_rate_to_usd": 0.012},
    "CNY": {"symbol": "¥", "name": "Chinese Yuan", "typical_rate_to_usd": 0.14},
    "JPY": {"symbol": "¥", "name": "Japanese Yen", "typical_rate_to_usd": 0.0067},
    "GBP": {"symbol": "£", "name": "British Pound", "typical_rate_to_usd": 1.27},
    "CHF": {"symbol": "Fr", "name": "Swiss Franc", "typical_rate_to_usd": 1.12},
    "AUD": {"symbol": "A$", "name": "Australian Dollar", "typical_rate_to_usd": 0.65}
}


# ============================================================================
# 7. UNCERTAINTY FACTOR CATEGORIES (for Fishbone Diagram)
# ============================================================================

UNCERTAINTY_CATEGORIES = {
    "1": {
        "name": "Reference Device",
        "icon": "📏",
        "subcategories": {
            "1.1": {
                "name": "Calibration Uncertainty",
                "factors": {
                    "1.1.1": "WPVS Cell Calibration",
                    "1.1.2": "Reference Module Calibration",
                    "1.1.3": "Traceability Chain",
                    "1.1.4": "Calibration Drift"
                }
            },
            "1.2": {
                "name": "Reference Device Stability",
                "factors": {
                    "1.2.1": "Long-term Drift",
                    "1.2.2": "Temperature Dependence",
                    "1.2.3": "Degradation"
                }
            },
            "1.3": {
                "name": "Reference Device Positioning",
                "factors": {
                    "1.3.1": "Position in Test Plane",
                    "1.3.2": "Irradiance Non-uniformity Effect"
                }
            }
        }
    },
    "2": {
        "name": "Sun Simulator",
        "icon": "☀️",
        "subcategories": {
            "2.1": {
                "name": "Spatial Non-uniformity",
                "factors": {
                    "2.1.1": "Uniformity Classification",
                    "2.1.2": "Position-dependent Irradiance"
                }
            },
            "2.2": {
                "name": "Temporal Instability",
                "factors": {
                    "2.2.1": "Short-term Stability",
                    "2.2.2": "Flash Duration Variation",
                    "2.2.3": "Lamp Aging"
                }
            },
            "2.3": {
                "name": "Spectral Mismatch",
                "factors": {
                    "2.3.1": "Simulator Spectrum vs AM1.5G",
                    "2.3.2": "Module Spectral Response",
                    "2.3.3": "Reference Cell Spectral Response",
                    "2.3.4": "Mismatch Factor Calculation"
                }
            }
        }
    },
    "3": {
        "name": "Temperature Measurement",
        "icon": "🌡️",
        "subcategories": {
            "3.1": {
                "name": "Sensor Calibration",
                "factors": {
                    "3.1.1": "Thermocouple/RTD Uncertainty",
                    "3.1.2": "Calibration Accuracy"
                }
            },
            "3.2": {
                "name": "Temperature Uniformity",
                "factors": {
                    "3.2.1": "Module Temperature Gradient",
                    "3.2.2": "Sensor Placement"
                }
            },
            "3.3": {
                "name": "Temperature Correction",
                "factors": {
                    "3.3.1": "IEC 60891 Correction Procedure",
                    "3.3.2": "Temperature Coefficient Uncertainty"
                }
            }
        }
    },
    "4": {
        "name": "I-V Measurement",
        "icon": "📊",
        "subcategories": {
            "4.1": {
                "name": "Voltage Measurement",
                "factors": {
                    "4.1.1": "Voltmeter Calibration",
                    "4.1.2": "Contact Resistance",
                    "4.1.3": "Cable Resistance"
                }
            },
            "4.2": {
                "name": "Current Measurement",
                "factors": {
                    "4.2.1": "Ammeter/Shunt Calibration",
                    "4.2.2": "Current Range Selection"
                }
            },
            "4.3": {
                "name": "Data Acquisition",
                "factors": {
                    "4.3.1": "ADC Resolution",
                    "4.3.2": "Sampling Rate",
                    "4.3.3": "Curve Fitting Algorithm"
                }
            }
        }
    },
    "5": {
        "name": "Module Characteristics",
        "icon": "⚡",
        "subcategories": {
            "5.1": {
                "name": "Module Variability",
                "factors": {
                    "5.1.1": "Manufacturing Tolerance",
                    "5.1.2": "Binning Accuracy"
                }
            },
            "5.2": {
                "name": "Module Behavior",
                "factors": {
                    "5.2.1": "Hysteresis Effects",
                    "5.2.2": "Stabilization Time",
                    "5.2.3": "Light Soaking (for thin-film)"
                }
            }
        }
    },
    "6": {
        "name": "Environmental Conditions",
        "icon": "🌍",
        "subcategories": {
            "6.1": {
                "name": "Ambient Conditions",
                "factors": {
                    "6.1.1": "Ambient Temperature",
                    "6.1.2": "Humidity",
                    "6.1.3": "Air Pressure"
                }
            }
        }
    },
    "7": {
        "name": "Measurement Procedure",
        "icon": "📋",
        "subcategories": {
            "7.1": {
                "name": "Repeatability",
                "factors": {
                    "7.1.1": "Intra-laboratory Repeatability",
                    "7.1.2": "Same-day Measurements"
                }
            },
            "7.2": {
                "name": "Reproducibility",
                "factors": {
                    "7.2.1": "Inter-laboratory Reproducibility",
                    "7.2.2": "ILC/Round Robin Results"
                }
            },
            "7.3": {
                "name": "Operator Effects",
                "factors": {
                    "7.3.1": "Module Positioning",
                    "7.3.2": "Contact Application"
                }
            }
        }
    }
}


def get_technology_list() -> List[str]:
    """Get list of available PV technologies."""
    return list(PV_TECHNOLOGIES.keys())


def get_simulator_list() -> List[str]:
    """Get list of available sun simulators."""
    return list(SUN_SIMULATORS.keys())


def get_reference_lab_list() -> List[str]:
    """Get list of reference calibration laboratories."""
    return list(REFERENCE_LABS.keys())


def get_measurement_types() -> List[str]:
    """Get list of measurement types."""
    return list(MEASUREMENT_TYPES.keys())


def get_currency_list() -> List[str]:
    """Get list of supported currencies."""
    return list(CURRENCIES.keys())
