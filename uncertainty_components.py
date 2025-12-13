"""
Uncertainty Components Module

Comprehensive component classes for universal solar simulator uncertainty framework.
Supports all major simulator platforms (PASAN, Spire, Halm, Meyer Burger, Wavelabs, etc.)
with full bifacial measurement capability.

This module provides:
- UncertaintyComponent base class with GUM methodology
- Simulator-specific configurations
- Reference device components
- Temperature measurement components
- I-V measurement components
- Bifacial-specific components (rear irradiance, φ factors)
- Complete uncertainty budget builder

References:
- IEC TS 60904-1-2:2024: Bifacial PV measurement
- IEC 60904-9:2020: Solar simulator classification
- JCGM 100:2008: GUM methodology
- ISO 17025:2017: Laboratory competence

Author: Universal Solar Simulator Framework Team
Version: 2.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Callable, Any, Union
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod
import json
from datetime import datetime


# =============================================================================
# Enumerations
# =============================================================================

class UncertaintyType(Enum):
    """Type of uncertainty evaluation"""
    TYPE_A = "type_a"      # Statistical analysis of observations
    TYPE_B = "type_b"      # Other means (manufacturer spec, calibration cert, etc.)


class DistributionType(Enum):
    """Probability distribution types"""
    NORMAL = "normal"
    RECTANGULAR = "rectangular"
    TRIANGULAR = "triangular"
    U_SHAPED = "u_shaped"
    LOGNORMAL = "lognormal"
    CUSTOM = "custom"


class SimulatorType(Enum):
    """Solar simulator lamp types"""
    XENON = "xenon"
    LED = "led"
    HALOGEN = "halogen"
    HYBRID = "hybrid"       # LED + Xenon combination


class SimulatorClassification(Enum):
    """IEC 60904-9 simulator classification"""
    AAA_PLUS = "AAA+"       # Better than AAA
    AAA = "AAA"             # Best standard class
    ABA = "ABA"
    AAB = "AAB"
    ABB = "ABB"
    BAA = "BAA"
    BAB = "BAB"
    BBA = "BBA"
    BBB = "BBB"
    CUSTOM = "custom"


class ComponentCategory(Enum):
    """Uncertainty component categories"""
    REFERENCE_DEVICE = "reference_device"
    SIMULATOR_FRONT = "simulator_front"
    SIMULATOR_REAR = "simulator_rear"
    TEMPERATURE = "temperature"
    IV_MEASUREMENT = "iv_measurement"
    MODULE_CHARACTERISTICS = "module_characteristics"
    BIFACIALITY = "bifaciality"
    EQUIVALENT_IRRADIANCE = "equivalent_irradiance"
    SPECTRAL = "spectral"
    ENVIRONMENTAL = "environmental"
    REPEATABILITY = "repeatability"
    REPRODUCIBILITY = "reproducibility"
    PARASITIC = "parasitic"


# =============================================================================
# Base Classes
# =============================================================================

@dataclass
class UncertaintyComponent:
    """
    Base class for uncertainty components following GUM methodology.

    Attributes:
        name: Component identifier
        description: Detailed description
        category: Component category
        value: Measured or specified value
        uncertainty: Standard uncertainty (k=1)
        unit: Physical unit
        evaluation_type: Type A or Type B
        distribution: Probability distribution
        sensitivity_coefficient: Sensitivity coefficient c_i
        degrees_of_freedom: Degrees of freedom for this component
        enabled: Whether to include in calculations
        source: Source of uncertainty value (manufacturer, calibration, etc.)
        notes: Additional notes
    """
    name: str
    description: str = ""
    category: ComponentCategory = ComponentCategory.REFERENCE_DEVICE

    value: float = 0.0
    uncertainty: float = 0.0
    unit: str = "%"

    evaluation_type: UncertaintyType = UncertaintyType.TYPE_B
    distribution: DistributionType = DistributionType.NORMAL
    sensitivity_coefficient: float = 1.0
    degrees_of_freedom: float = float('inf')

    enabled: bool = True
    source: str = ""
    notes: str = ""

    # For hierarchical organization
    category_id: str = ""
    subcategory_id: str = ""
    factor_id: str = ""

    def __post_init__(self):
        """Validate and convert distribution if needed"""
        if isinstance(self.distribution, str):
            self.distribution = DistributionType(self.distribution.lower())
        if isinstance(self.evaluation_type, str):
            self.evaluation_type = UncertaintyType(self.evaluation_type.lower())
        if isinstance(self.category, str):
            self.category = ComponentCategory(self.category.lower())

    @property
    def standard_uncertainty(self) -> float:
        """Convert to standard uncertainty based on distribution"""
        if self.distribution == DistributionType.NORMAL:
            return self.uncertainty
        elif self.distribution == DistributionType.RECTANGULAR:
            return self.uncertainty / np.sqrt(3)
        elif self.distribution == DistributionType.TRIANGULAR:
            return self.uncertainty / np.sqrt(6)
        elif self.distribution == DistributionType.U_SHAPED:
            return self.uncertainty / np.sqrt(2)
        else:
            return self.uncertainty

    @property
    def variance_contribution(self) -> float:
        """Calculate variance contribution: (c_i × u_i)²"""
        return (self.sensitivity_coefficient * self.standard_uncertainty) ** 2

    @property
    def relative_uncertainty(self) -> float:
        """Relative uncertainty (if value is non-zero)"""
        if self.value != 0:
            return self.standard_uncertainty / abs(self.value)
        return 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "value": self.value,
            "uncertainty": self.uncertainty,
            "standard_uncertainty": self.standard_uncertainty,
            "unit": self.unit,
            "evaluation_type": self.evaluation_type.value,
            "distribution": self.distribution.value,
            "sensitivity_coefficient": self.sensitivity_coefficient,
            "degrees_of_freedom": self.degrees_of_freedom,
            "variance_contribution": self.variance_contribution,
            "enabled": self.enabled,
            "source": self.source,
            "notes": self.notes,
            "category_id": self.category_id,
            "subcategory_id": self.subcategory_id,
            "factor_id": self.factor_id,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "UncertaintyComponent":
        """Create from dictionary"""
        return cls(
            name=data.get("name", "Unknown"),
            description=data.get("description", ""),
            category=ComponentCategory(data.get("category", "reference_device")),
            value=data.get("value", 0.0),
            uncertainty=data.get("uncertainty", 0.0),
            unit=data.get("unit", "%"),
            evaluation_type=UncertaintyType(data.get("evaluation_type", "type_b")),
            distribution=DistributionType(data.get("distribution", "normal")),
            sensitivity_coefficient=data.get("sensitivity_coefficient", 1.0),
            degrees_of_freedom=data.get("degrees_of_freedom", float('inf')),
            enabled=data.get("enabled", True),
            source=data.get("source", ""),
            notes=data.get("notes", ""),
            category_id=data.get("category_id", ""),
            subcategory_id=data.get("subcategory_id", ""),
            factor_id=data.get("factor_id", ""),
        )


# =============================================================================
# Simulator Configurations
# =============================================================================

@dataclass
class SimulatorConfiguration:
    """
    Complete solar simulator configuration.

    Supports all major manufacturers with full specification.
    """
    manufacturer: str
    model: str
    lamp_type: SimulatorType = SimulatorType.LED
    classification: SimulatorClassification = SimulatorClassification.AAA

    # Front side specifications (always required)
    uniformity_front: float = 2.0       # % non-uniformity
    temporal_front: float = 0.5         # % temporal instability
    spectral_match_front: str = "A"     # A, B, or C

    # Irradiance range
    irradiance_min: float = 100.0       # W/m² minimum
    irradiance_max: float = 1200.0      # W/m² maximum

    # Rear side specifications (for bifacial-capable simulators)
    bifacial_capable: bool = False
    uniformity_rear: float = 3.0        # % typically worse than front
    temporal_rear: float = 1.0          # %
    spectral_match_rear: str = "A"
    rear_irradiance_min: float = 0.0    # W/m²
    rear_irradiance_max: float = 500.0  # W/m²
    rear_independent_control: bool = True

    # Physical specifications
    test_area_length_mm: float = 2200.0
    test_area_width_mm: float = 1200.0
    working_distance_mm: float = 500.0
    flash_duration_ms: Optional[float] = None  # For flash systems

    # Spectral characteristics
    spectral_tunability: bool = False
    wavelength_min_nm: float = 300.0
    wavelength_max_nm: float = 1200.0

    # Calibration
    calibration_date: Optional[str] = None
    calibration_lab: str = ""
    calibration_certificate: str = ""

    def get_class_components(self) -> Tuple[str, str, str]:
        """Extract uniformity, temporal, spectral class from classification"""
        class_str = self.classification.value.replace("+", "")
        if len(class_str) >= 3:
            return class_str[0], class_str[1], class_str[2]
        return "A", "A", "A"

    def get_uniformity_uncertainty(self, side: str = "front") -> float:
        """Get uniformity uncertainty as standard uncertainty (k=1)"""
        if side == "front":
            return self.uniformity_front / np.sqrt(3)  # Rectangular
        else:
            return self.uniformity_rear / np.sqrt(3)

    def get_temporal_uncertainty(self, side: str = "front") -> float:
        """Get temporal instability uncertainty as standard uncertainty (k=1)"""
        if side == "front":
            return self.temporal_front / np.sqrt(3)  # Rectangular
        else:
            return self.temporal_rear / np.sqrt(3)


# Pre-defined simulator configurations
SIMULATOR_DATABASE: Dict[str, SimulatorConfiguration] = {
    # PASAN Simulators
    "pasan_highlight_led": SimulatorConfiguration(
        manufacturer="PASAN",
        model="HighLIGHT LED",
        lamp_type=SimulatorType.LED,
        classification=SimulatorClassification.AAA_PLUS,
        uniformity_front=1.5,
        temporal_front=0.2,
        spectral_match_front="A",
        bifacial_capable=False,
    ),
    "pasan_highlight_bifacial": SimulatorConfiguration(
        manufacturer="PASAN",
        model="HighLIGHT BIFACIAL",
        lamp_type=SimulatorType.LED,
        classification=SimulatorClassification.AAA_PLUS,
        uniformity_front=1.5,
        temporal_front=0.2,
        spectral_match_front="A",
        bifacial_capable=True,
        uniformity_rear=2.5,
        temporal_rear=0.3,
        spectral_match_rear="A",
        rear_irradiance_max=600.0,
        rear_independent_control=True,
    ),
    "pasan_cetis_ctl": SimulatorConfiguration(
        manufacturer="PASAN",
        model="cetisPV-CTL",
        lamp_type=SimulatorType.LED,
        classification=SimulatorClassification.AAA,
        uniformity_front=2.0,
        temporal_front=0.3,
        spectral_match_front="A",
        bifacial_capable=False,
    ),

    # Spire/Atonometrics Simulators
    "spire_5600slp": SimulatorConfiguration(
        manufacturer="Spire/Atonometrics",
        model="5600SLP",
        lamp_type=SimulatorType.XENON,
        classification=SimulatorClassification.AAA,
        uniformity_front=2.0,
        temporal_front=0.5,
        spectral_match_front="A",
        flash_duration_ms=10.0,
        bifacial_capable=False,
    ),
    "spire_bifi1000": SimulatorConfiguration(
        manufacturer="Spire/Atonometrics",
        model="BiFi-1000",
        lamp_type=SimulatorType.HYBRID,
        classification=SimulatorClassification.AAA,
        uniformity_front=2.0,
        temporal_front=0.5,
        spectral_match_front="A",
        bifacial_capable=True,
        uniformity_rear=3.0,
        temporal_rear=0.8,
        spectral_match_rear="A",
        rear_irradiance_max=500.0,
    ),

    # Halm/EETS Simulators
    "halm_cetis_bi": SimulatorConfiguration(
        manufacturer="Halm/EETS",
        model="cetisPV-BI",
        lamp_type=SimulatorType.HYBRID,
        classification=SimulatorClassification.AAA_PLUS,
        uniformity_front=1.5,
        temporal_front=0.3,
        spectral_match_front="A",
        bifacial_capable=True,
        uniformity_rear=2.0,
        temporal_rear=0.5,
        spectral_match_rear="A",
        rear_irradiance_max=600.0,
        rear_independent_control=True,
    ),
    "halm_flasher3": SimulatorConfiguration(
        manufacturer="Halm/EETS",
        model="flasher III",
        lamp_type=SimulatorType.XENON,
        classification=SimulatorClassification.AAA,
        uniformity_front=2.0,
        temporal_front=1.0,
        spectral_match_front="A",
        flash_duration_ms=15.0,
        bifacial_capable=False,
    ),

    # Meyer Burger Simulators
    "meyer_burger_loana": SimulatorConfiguration(
        manufacturer="Meyer Burger",
        model="LOANA",
        lamp_type=SimulatorType.LED,
        classification=SimulatorClassification.AAA_PLUS,
        uniformity_front=1.5,
        temporal_front=0.2,
        spectral_match_front="A",
        spectral_tunability=True,
        bifacial_capable=False,
    ),
    "meyer_burger_pss30": SimulatorConfiguration(
        manufacturer="Meyer Burger",
        model="PSS-30",
        lamp_type=SimulatorType.LED,
        classification=SimulatorClassification.AAA,
        uniformity_front=2.0,
        temporal_front=0.5,
        spectral_match_front="A",
        bifacial_capable=True,
        uniformity_rear=3.0,
        temporal_rear=0.8,
        rear_independent_control=False,  # Sequential, not simultaneous
    ),

    # Wavelabs Simulators
    "wavelabs_avalon_nexun": SimulatorConfiguration(
        manufacturer="Wavelabs",
        model="Avalon Nexun",
        lamp_type=SimulatorType.LED,
        classification=SimulatorClassification.AAA,
        uniformity_front=2.0,
        temporal_front=0.5,
        spectral_match_front="A",
        spectral_tunability=True,
        bifacial_capable=False,
    ),
    "wavelabs_avalon_bifacial": SimulatorConfiguration(
        manufacturer="Wavelabs",
        model="Avalon Bifacial",
        lamp_type=SimulatorType.LED,
        classification=SimulatorClassification.AAA_PLUS,
        uniformity_front=1.8,
        temporal_front=0.3,
        spectral_match_front="A",
        bifacial_capable=True,
        uniformity_rear=2.5,
        temporal_rear=0.5,
        spectral_match_rear="A",
        rear_irradiance_max=500.0,
        rear_independent_control=True,
        spectral_tunability=True,
    ),
    "wavelabs_avalon_steadystate": SimulatorConfiguration(
        manufacturer="Wavelabs",
        model="Avalon SteadyState",
        lamp_type=SimulatorType.LED,
        classification=SimulatorClassification.AAA,
        uniformity_front=1.8,
        temporal_front=0.3,
        spectral_match_front="A",
        bifacial_capable=False,
    ),

    # Eternal Sun Simulators
    "eternalsun_slp150": SimulatorConfiguration(
        manufacturer="Eternal Sun",
        model="SLP-150",
        lamp_type=SimulatorType.LED,
        classification=SimulatorClassification.AAA_PLUS,
        uniformity_front=1.5,
        temporal_front=0.3,
        spectral_match_front="A",
        bifacial_capable=False,
    ),
    "eternalsun_slp_bifi": SimulatorConfiguration(
        manufacturer="Eternal Sun",
        model="SLP-BiFi",
        lamp_type=SimulatorType.LED,
        classification=SimulatorClassification.AAA_PLUS,
        uniformity_front=1.5,
        temporal_front=0.3,
        spectral_match_front="A",
        bifacial_capable=True,
        uniformity_rear=2.0,
        temporal_rear=0.5,
        spectral_match_rear="A",
        rear_irradiance_max=500.0,
        rear_independent_control=True,
    ),

    # ReRa Solutions
    "rera_tracer": SimulatorConfiguration(
        manufacturer="ReRa Solutions",
        model="Tracer",
        lamp_type=SimulatorType.LED,
        classification=SimulatorClassification.AAA,
        uniformity_front=2.0,
        temporal_front=0.5,
        spectral_match_front="A",
        bifacial_capable=False,
    ),

    # Generic/Custom
    "custom_aaa": SimulatorConfiguration(
        manufacturer="Custom",
        model="AAA Class Simulator",
        lamp_type=SimulatorType.LED,
        classification=SimulatorClassification.AAA,
        uniformity_front=2.0,
        temporal_front=2.0,
        spectral_match_front="A",
        bifacial_capable=True,
        uniformity_rear=2.0,
        temporal_rear=2.0,
        spectral_match_rear="A",
    ),
}


def get_simulator(key: str) -> Optional[SimulatorConfiguration]:
    """Get simulator configuration by key"""
    return SIMULATOR_DATABASE.get(key)


def list_simulators(bifacial_only: bool = False) -> List[str]:
    """List available simulator keys"""
    if bifacial_only:
        return [k for k, v in SIMULATOR_DATABASE.items() if v.bifacial_capable]
    return list(SIMULATOR_DATABASE.keys())


# =============================================================================
# Reference Device Configurations
# =============================================================================

@dataclass
class ReferenceDeviceConfiguration:
    """Reference cell/module configuration"""
    name: str
    device_type: str = "wpvs"           # wpvs, module, spectroradiometer
    technology: str = "c-Si"            # c-Si, mc-Si, thin-film, etc.

    # Calibration
    calibration_uncertainty: float = 1.0  # % (k=2)
    calibration_lab: str = ""
    calibration_date: Optional[str] = None
    traceability: str = ""              # NREL, PTB, etc.

    # Stability
    stability_annual: float = 0.5       # %/year drift
    temperature_coefficient: float = 0.05  # %/°C

    # Spectral response
    spectral_response_file: Optional[str] = None
    spectral_range_nm: Tuple[float, float] = (300.0, 1200.0)

    # Bifacial capability
    bifacial_calibrated: bool = False
    rear_calibration_uncertainty: float = 1.5  # % (k=2)


REFERENCE_DEVICE_DATABASE: Dict[str, ReferenceDeviceConfiguration] = {
    "nrel_wpvs": ReferenceDeviceConfiguration(
        name="NREL WPVS Reference Cell",
        device_type="wpvs",
        technology="c-Si",
        calibration_uncertainty=0.5,
        calibration_lab="NREL",
        traceability="NREL Primary Standard",
        stability_annual=0.3,
    ),
    "ptb_wpvs": ReferenceDeviceConfiguration(
        name="PTB WPVS Reference Cell",
        device_type="wpvs",
        technology="c-Si",
        calibration_uncertainty=0.4,
        calibration_lab="PTB",
        traceability="PTB Primary Standard",
        stability_annual=0.3,
    ),
    "ise_wpvs": ReferenceDeviceConfiguration(
        name="Fraunhofer ISE WPVS",
        device_type="wpvs",
        technology="c-Si",
        calibration_uncertainty=0.4,
        calibration_lab="Fraunhofer ISE",
        traceability="ISE Primary Standard",
        stability_annual=0.3,
    ),
    "generic_wpvs": ReferenceDeviceConfiguration(
        name="Generic WPVS Reference Cell",
        device_type="wpvs",
        technology="c-Si",
        calibration_uncertainty=1.0,
        stability_annual=0.5,
    ),
    "module_reference": ReferenceDeviceConfiguration(
        name="Module Reference Device",
        device_type="module",
        technology="c-Si",
        calibration_uncertainty=1.5,
        stability_annual=0.8,
    ),
    "bifacial_reference": ReferenceDeviceConfiguration(
        name="Bifacial Reference Cell",
        device_type="wpvs",
        technology="c-Si",
        calibration_uncertainty=1.0,
        bifacial_calibrated=True,
        rear_calibration_uncertainty=1.5,
        stability_annual=0.5,
    ),
}


# =============================================================================
# Uncertainty Budget Builder
# =============================================================================

class UncertaintyBudgetBuilder:
    """
    Builder class for constructing complete uncertainty budgets.

    Follows GUM methodology with support for:
    - All standard uncertainty categories
    - Bifacial-specific components
    - Multiple simulator platforms
    - Custom components
    """

    def __init__(
        self,
        measurement_type: str = "stc",
        is_bifacial: bool = False
    ):
        """
        Initialize budget builder.

        Args:
            measurement_type: Type of measurement (stc, nmot, bifacial, etc.)
            is_bifacial: Whether measurement includes bifacial components
        """
        self.measurement_type = measurement_type
        self.is_bifacial = is_bifacial
        self.components: Dict[str, UncertaintyComponent] = {}
        self.simulator: Optional[SimulatorConfiguration] = None
        self.reference: Optional[ReferenceDeviceConfiguration] = None

        # Initialize category structure
        self._init_category_structure()

    def _init_category_structure(self):
        """Initialize the hierarchical category structure"""
        self.category_structure = {
            "1": {
                "name": "Reference Device",
                "subcategories": {
                    "1.1": "Calibration Uncertainty",
                    "1.2": "Reference Device Stability",
                    "1.3": "Reference Device Positioning",
                }
            },
            "2": {
                "name": "Sun Simulator - Front Side",
                "subcategories": {
                    "2.1": "Spatial Non-uniformity (Front)",
                    "2.2": "Temporal Instability (Front)",
                    "2.3": "Spectral Mismatch (Front)",
                }
            },
            "3": {
                "name": "Sun Simulator - Rear Side",
                "subcategories": {
                    "3.1": "Spatial Non-uniformity (Rear)",
                    "3.2": "Temporal Instability (Rear)",
                    "3.3": "Spectral Mismatch (Rear)",
                    "3.4": "Rear Irradiance Uncertainty",
                }
            },
            "4": {
                "name": "Temperature Measurement",
                "subcategories": {
                    "4.1": "Sensor Calibration",
                    "4.2": "Temperature Uniformity",
                    "4.3": "Temperature Correction",
                }
            },
            "5": {
                "name": "I-V Measurement",
                "subcategories": {
                    "5.1": "Voltage Measurement",
                    "5.2": "Current Measurement",
                    "5.3": "Data Acquisition",
                }
            },
            "6": {
                "name": "Module Characteristics",
                "subcategories": {
                    "6.1": "Module Variability",
                    "6.2": "Module Behavior",
                    "6.3": "Bifacial-Specific",
                }
            },
            "7": {
                "name": "Bifaciality Factor",
                "subcategories": {
                    "7.1": "Bifaciality Factor Determination",
                    "7.2": "Bifaciality Factor Application",
                }
            },
            "8": {
                "name": "Equivalent Irradiance",
                "subcategories": {
                    "8.1": "G_eq Calculation",
                    "8.2": "Irradiance Correction",
                }
            },
            "9": {
                "name": "Environmental Conditions",
                "subcategories": {
                    "9.1": "Ambient Conditions",
                    "9.2": "Albedo Effects",
                }
            },
            "10": {
                "name": "Measurement Procedure",
                "subcategories": {
                    "10.1": "Repeatability",
                    "10.2": "Reproducibility",
                    "10.3": "Operator Effects",
                }
            },
            "11": {
                "name": "Parasitic Effects",
                "subcategories": {
                    "11.1": "Optical Crosstalk",
                    "11.2": "Electrical Effects",
                }
            },
        }

    def set_simulator(
        self,
        simulator: Union[str, SimulatorConfiguration]
    ) -> "UncertaintyBudgetBuilder":
        """Set the simulator configuration"""
        if isinstance(simulator, str):
            self.simulator = get_simulator(simulator)
            if self.simulator is None:
                raise ValueError(f"Unknown simulator: {simulator}")
        else:
            self.simulator = simulator

        # Auto-detect bifacial capability
        if self.simulator.bifacial_capable:
            self.is_bifacial = True

        return self

    def set_reference(
        self,
        reference: Union[str, ReferenceDeviceConfiguration]
    ) -> "UncertaintyBudgetBuilder":
        """Set the reference device configuration"""
        if isinstance(reference, str):
            self.reference = REFERENCE_DEVICE_DATABASE.get(reference)
            if self.reference is None:
                self.reference = REFERENCE_DEVICE_DATABASE["generic_wpvs"]
        else:
            self.reference = reference
        return self

    def add_component(
        self,
        component: UncertaintyComponent
    ) -> "UncertaintyBudgetBuilder":
        """Add a custom uncertainty component"""
        self.components[component.name] = component
        return self

    def add_reference_device_uncertainty(
        self,
        calibration_k2: float = 1.0,
        stability_annual: float = 0.5,
        positioning: float = 0.3
    ) -> "UncertaintyBudgetBuilder":
        """
        Add reference device uncertainty components.

        Args:
            calibration_k2: Calibration uncertainty (k=2, %)
            stability_annual: Annual drift (%)
            positioning: Positioning uncertainty (%)
        """
        # Use reference config if available
        if self.reference:
            calibration_k2 = self.reference.calibration_uncertainty
            stability_annual = self.reference.stability_annual

        self.components["ref_calibration"] = UncertaintyComponent(
            name="Reference Calibration",
            description="WPVS/Module calibration uncertainty",
            category=ComponentCategory.REFERENCE_DEVICE,
            uncertainty=calibration_k2 / 2.0,  # Convert k=2 to k=1
            unit="%",
            evaluation_type=UncertaintyType.TYPE_B,
            distribution=DistributionType.NORMAL,
            category_id="1",
            subcategory_id="1.1",
            factor_id="1.1.1",
            source=self.reference.calibration_lab if self.reference else "Calibration certificate",
        )

        self.components["ref_stability"] = UncertaintyComponent(
            name="Reference Stability",
            description="Long-term drift of reference device",
            category=ComponentCategory.REFERENCE_DEVICE,
            uncertainty=stability_annual,
            unit="%",
            evaluation_type=UncertaintyType.TYPE_B,
            distribution=DistributionType.RECTANGULAR,
            category_id="1",
            subcategory_id="1.2",
            factor_id="1.2.1",
        )

        self.components["ref_positioning"] = UncertaintyComponent(
            name="Reference Positioning",
            description="Position in test plane",
            category=ComponentCategory.REFERENCE_DEVICE,
            uncertainty=positioning,
            unit="%",
            evaluation_type=UncertaintyType.TYPE_B,
            distribution=DistributionType.RECTANGULAR,
            category_id="1",
            subcategory_id="1.3",
            factor_id="1.3.1",
        )

        return self

    def add_simulator_front_uncertainty(
        self,
        uniformity: Optional[float] = None,
        temporal: Optional[float] = None,
        spectral: float = 0.8
    ) -> "UncertaintyBudgetBuilder":
        """
        Add front-side simulator uncertainty components.

        Args:
            uniformity: Spatial non-uniformity (%) - uses simulator config if None
            temporal: Temporal instability (%) - uses simulator config if None
            spectral: Spectral mismatch uncertainty (%)
        """
        # Use simulator config if available
        if self.simulator:
            uniformity = uniformity or self.simulator.uniformity_front
            temporal = temporal or self.simulator.temporal_front

        self.components["sim_uniformity_front"] = UncertaintyComponent(
            name="Front Uniformity",
            description="Spatial non-uniformity over test area (front)",
            category=ComponentCategory.SIMULATOR_FRONT,
            uncertainty=uniformity or 2.0,
            unit="%",
            evaluation_type=UncertaintyType.TYPE_B,
            distribution=DistributionType.RECTANGULAR,
            category_id="2",
            subcategory_id="2.1",
            factor_id="2.1.1",
            source=f"{self.simulator.manufacturer} {self.simulator.model}" if self.simulator else "",
        )

        self.components["sim_temporal_front"] = UncertaintyComponent(
            name="Front Temporal",
            description="Temporal instability during measurement (front)",
            category=ComponentCategory.SIMULATOR_FRONT,
            uncertainty=temporal or 0.5,
            unit="%",
            evaluation_type=UncertaintyType.TYPE_B,
            distribution=DistributionType.RECTANGULAR,
            category_id="2",
            subcategory_id="2.2",
            factor_id="2.2.1",
        )

        self.components["sim_spectral_front"] = UncertaintyComponent(
            name="Front Spectral Mismatch",
            description="Spectral mismatch correction uncertainty (front)",
            category=ComponentCategory.SIMULATOR_FRONT,
            uncertainty=spectral,
            unit="%",
            evaluation_type=UncertaintyType.TYPE_B,
            distribution=DistributionType.NORMAL,
            category_id="2",
            subcategory_id="2.3",
            factor_id="2.3.1",
        )

        return self

    def add_simulator_rear_uncertainty(
        self,
        uniformity: Optional[float] = None,
        temporal: Optional[float] = None,
        spectral: float = 1.5,
        irradiance: float = 2.0,
        parasitic: float = 0.3
    ) -> "UncertaintyBudgetBuilder":
        """
        Add rear-side simulator uncertainty components (bifacial only).

        Args:
            uniformity: Rear spatial non-uniformity (%)
            temporal: Rear temporal instability (%)
            spectral: Rear spectral mismatch uncertainty (%)
            irradiance: Rear irradiance measurement uncertainty (%)
            parasitic: Parasitic front-to-rear light (%)
        """
        if not self.is_bifacial:
            return self

        # Use simulator config if available
        if self.simulator and self.simulator.bifacial_capable:
            uniformity = uniformity or self.simulator.uniformity_rear
            temporal = temporal or self.simulator.temporal_rear

        self.components["sim_uniformity_rear"] = UncertaintyComponent(
            name="Rear Uniformity",
            description="Spatial non-uniformity over test area (rear)",
            category=ComponentCategory.SIMULATOR_REAR,
            uncertainty=uniformity or 3.0,
            unit="%",
            evaluation_type=UncertaintyType.TYPE_B,
            distribution=DistributionType.RECTANGULAR,
            category_id="3",
            subcategory_id="3.1",
            factor_id="3.1.1",
        )

        self.components["sim_temporal_rear"] = UncertaintyComponent(
            name="Rear Temporal",
            description="Temporal instability during measurement (rear)",
            category=ComponentCategory.SIMULATOR_REAR,
            uncertainty=temporal or 1.0,
            unit="%",
            evaluation_type=UncertaintyType.TYPE_B,
            distribution=DistributionType.RECTANGULAR,
            category_id="3",
            subcategory_id="3.2",
            factor_id="3.2.1",
        )

        self.components["sim_spectral_rear"] = UncertaintyComponent(
            name="Rear Spectral Mismatch",
            description="Spectral mismatch correction uncertainty (rear)",
            category=ComponentCategory.SIMULATOR_REAR,
            uncertainty=spectral,
            unit="%",
            evaluation_type=UncertaintyType.TYPE_B,
            distribution=DistributionType.NORMAL,
            category_id="3",
            subcategory_id="3.3",
            factor_id="3.3.1",
        )

        self.components["sim_irradiance_rear"] = UncertaintyComponent(
            name="Rear Irradiance",
            description="Rear irradiance measurement uncertainty",
            category=ComponentCategory.SIMULATOR_REAR,
            uncertainty=irradiance,
            unit="%",
            evaluation_type=UncertaintyType.TYPE_B,
            distribution=DistributionType.NORMAL,
            category_id="3",
            subcategory_id="3.4",
            factor_id="3.4.1",
        )

        self.components["sim_parasitic_rear"] = UncertaintyComponent(
            name="Parasitic Light (Rear)",
            description="Parasitic front-to-rear reflection",
            category=ComponentCategory.PARASITIC,
            uncertainty=parasitic,
            unit="%",
            evaluation_type=UncertaintyType.TYPE_B,
            distribution=DistributionType.RECTANGULAR,
            category_id="11",
            subcategory_id="11.1",
            factor_id="11.1.1",
        )

        return self

    def add_temperature_uncertainty(
        self,
        sensor: float = 0.5,
        uniformity: float = 1.0,
        correction: float = 0.3,
        gamma_uncertainty: float = 0.05
    ) -> "UncertaintyBudgetBuilder":
        """
        Add temperature measurement uncertainty components.

        Args:
            sensor: Temperature sensor uncertainty (°C)
            uniformity: Module temperature gradient (°C)
            correction: Temperature correction procedure uncertainty (%)
            gamma_uncertainty: Uncertainty on γ coefficient (%/°C)
        """
        self.components["temp_sensor"] = UncertaintyComponent(
            name="Temperature Sensor",
            description="Thermocouple/RTD calibration uncertainty",
            category=ComponentCategory.TEMPERATURE,
            uncertainty=sensor,
            unit="°C",
            evaluation_type=UncertaintyType.TYPE_B,
            distribution=DistributionType.NORMAL,
            category_id="4",
            subcategory_id="4.1",
            factor_id="4.1.1",
        )

        self.components["temp_uniformity"] = UncertaintyComponent(
            name="Temperature Uniformity",
            description="Module temperature gradient",
            category=ComponentCategory.TEMPERATURE,
            uncertainty=uniformity,
            unit="°C",
            evaluation_type=UncertaintyType.TYPE_B,
            distribution=DistributionType.RECTANGULAR,
            category_id="4",
            subcategory_id="4.2",
            factor_id="4.2.1",
        )

        self.components["temp_correction"] = UncertaintyComponent(
            name="Temperature Correction",
            description="IEC 60891 procedure uncertainty",
            category=ComponentCategory.TEMPERATURE,
            uncertainty=correction,
            unit="%",
            evaluation_type=UncertaintyType.TYPE_B,
            distribution=DistributionType.NORMAL,
            category_id="4",
            subcategory_id="4.3",
            factor_id="4.3.1",
        )

        return self

    def add_iv_measurement_uncertainty(
        self,
        voltage: float = 0.2,
        current: float = 0.2,
        data_acquisition: float = 0.1
    ) -> "UncertaintyBudgetBuilder":
        """
        Add I-V measurement uncertainty components.

        Args:
            voltage: Voltage measurement uncertainty (%)
            current: Current measurement uncertainty (%)
            data_acquisition: DAQ and curve fitting uncertainty (%)
        """
        self.components["iv_voltage"] = UncertaintyComponent(
            name="Voltage Measurement",
            description="Voltmeter calibration + contact resistance",
            category=ComponentCategory.IV_MEASUREMENT,
            uncertainty=voltage,
            unit="%",
            evaluation_type=UncertaintyType.TYPE_B,
            distribution=DistributionType.NORMAL,
            category_id="5",
            subcategory_id="5.1",
            factor_id="5.1.1",
        )

        self.components["iv_current"] = UncertaintyComponent(
            name="Current Measurement",
            description="Ammeter/shunt calibration",
            category=ComponentCategory.IV_MEASUREMENT,
            uncertainty=current,
            unit="%",
            evaluation_type=UncertaintyType.TYPE_B,
            distribution=DistributionType.NORMAL,
            category_id="5",
            subcategory_id="5.2",
            factor_id="5.2.1",
        )

        self.components["iv_daq"] = UncertaintyComponent(
            name="Data Acquisition",
            description="ADC resolution, sampling, curve fitting",
            category=ComponentCategory.IV_MEASUREMENT,
            uncertainty=data_acquisition,
            unit="%",
            evaluation_type=UncertaintyType.TYPE_B,
            distribution=DistributionType.RECTANGULAR,
            category_id="5",
            subcategory_id="5.3",
            factor_id="5.3.1",
        )

        return self

    def add_bifaciality_uncertainty(
        self,
        phi_isc: float = 1.5,
        phi_pmax: float = 2.0,
        phi_voc: float = 0.5,
        phi_irradiance_dependence: float = 0.5,
        phi_temperature_dependence: float = 0.3
    ) -> "UncertaintyBudgetBuilder":
        """
        Add bifaciality factor uncertainty components.

        Args:
            phi_isc: Uncertainty on φ_Isc (%)
            phi_pmax: Uncertainty on φ_Pmax (%)
            phi_voc: Uncertainty on φ_Voc (%)
            phi_irradiance_dependence: φ variation with irradiance ratio (%)
            phi_temperature_dependence: φ variation with temperature (%)
        """
        if not self.is_bifacial:
            return self

        self.components["phi_isc"] = UncertaintyComponent(
            name="φ_Isc Uncertainty",
            description="Bifaciality factor for short-circuit current",
            category=ComponentCategory.BIFACIALITY,
            uncertainty=phi_isc,
            unit="%",
            evaluation_type=UncertaintyType.TYPE_B,
            distribution=DistributionType.NORMAL,
            category_id="7",
            subcategory_id="7.1",
            factor_id="7.1.1",
        )

        self.components["phi_pmax"] = UncertaintyComponent(
            name="φ_Pmax Uncertainty",
            description="Bifaciality factor for maximum power",
            category=ComponentCategory.BIFACIALITY,
            uncertainty=phi_pmax,
            unit="%",
            evaluation_type=UncertaintyType.TYPE_B,
            distribution=DistributionType.NORMAL,
            category_id="7",
            subcategory_id="7.1",
            factor_id="7.1.3",
        )

        self.components["phi_irradiance_dep"] = UncertaintyComponent(
            name="φ Irradiance Dependence",
            description="Bifaciality variation with rear/front ratio",
            category=ComponentCategory.BIFACIALITY,
            uncertainty=phi_irradiance_dependence,
            unit="%",
            evaluation_type=UncertaintyType.TYPE_B,
            distribution=DistributionType.RECTANGULAR,
            category_id="7",
            subcategory_id="7.2",
            factor_id="7.2.1",
        )

        return self

    def add_equivalent_irradiance_uncertainty(
        self,
        g_front_contribution: float = 2.0,
        g_rear_contribution: float = 3.0,
        phi_contribution: float = 1.5,
        correction: float = 0.5
    ) -> "UncertaintyBudgetBuilder":
        """
        Add equivalent irradiance uncertainty components.

        These are calculated based on the GUM propagation through G_eq = G_f + φ×G_r

        Args:
            g_front_contribution: Front irradiance contribution (%)
            g_rear_contribution: Rear irradiance contribution (%)
            phi_contribution: Bifaciality factor contribution (%)
            correction: Irradiance correction uncertainty (%)
        """
        if not self.is_bifacial:
            return self

        self.components["geq_calculation"] = UncertaintyComponent(
            name="G_eq Calculation",
            description="Combined uncertainty on equivalent irradiance",
            category=ComponentCategory.EQUIVALENT_IRRADIANCE,
            uncertainty=np.sqrt(g_front_contribution**2 + g_rear_contribution**2 + phi_contribution**2),
            unit="%",
            evaluation_type=UncertaintyType.TYPE_B,
            distribution=DistributionType.NORMAL,
            category_id="8",
            subcategory_id="8.1",
            factor_id="8.1.1",
        )

        self.components["geq_correction"] = UncertaintyComponent(
            name="Irradiance Correction",
            description="Correction to equivalent STC",
            category=ComponentCategory.EQUIVALENT_IRRADIANCE,
            uncertainty=correction,
            unit="%",
            evaluation_type=UncertaintyType.TYPE_B,
            distribution=DistributionType.NORMAL,
            category_id="8",
            subcategory_id="8.2",
            factor_id="8.2.1",
        )

        return self

    def add_repeatability(
        self,
        repeatability: float = 0.3,
        n_measurements: int = 5
    ) -> "UncertaintyBudgetBuilder":
        """
        Add repeatability (Type A) uncertainty.

        Args:
            repeatability: Standard deviation of repeated measurements (%)
            n_measurements: Number of measurements
        """
        self.components["repeatability"] = UncertaintyComponent(
            name="Repeatability",
            description="Standard deviation of repeated measurements",
            category=ComponentCategory.REPEATABILITY,
            uncertainty=repeatability / np.sqrt(n_measurements),
            unit="%",
            evaluation_type=UncertaintyType.TYPE_A,
            distribution=DistributionType.NORMAL,
            degrees_of_freedom=n_measurements - 1,
            category_id="10",
            subcategory_id="10.1",
            factor_id="10.1.1",
        )
        return self

    def add_reproducibility(
        self,
        reproducibility: float = 1.5
    ) -> "UncertaintyBudgetBuilder":
        """
        Add reproducibility uncertainty (inter-laboratory).

        Args:
            reproducibility: Inter-laboratory reproducibility (%)
        """
        self.components["reproducibility"] = UncertaintyComponent(
            name="Reproducibility",
            description="Inter-laboratory reproducibility (ILC/Round Robin)",
            category=ComponentCategory.REPRODUCIBILITY,
            uncertainty=reproducibility,
            unit="%",
            evaluation_type=UncertaintyType.TYPE_B,
            distribution=DistributionType.NORMAL,
            category_id="10",
            subcategory_id="10.2",
            factor_id="10.2.1",
        )
        return self

    def build(self) -> "UncertaintyBudget":
        """Build and return the complete uncertainty budget"""
        return UncertaintyBudget(
            components=self.components,
            measurement_type=self.measurement_type,
            is_bifacial=self.is_bifacial,
            simulator=self.simulator,
            reference=self.reference,
            category_structure=self.category_structure,
        )

    def build_standard_stc(self) -> "UncertaintyBudget":
        """Build a standard STC measurement budget"""
        return (
            self
            .add_reference_device_uncertainty()
            .add_simulator_front_uncertainty()
            .add_temperature_uncertainty()
            .add_iv_measurement_uncertainty()
            .add_repeatability()
            .add_reproducibility()
            .build()
        )

    def build_bifacial_stc(self) -> "UncertaintyBudget":
        """Build a complete bifacial STC measurement budget"""
        self.is_bifacial = True
        return (
            self
            .add_reference_device_uncertainty()
            .add_simulator_front_uncertainty()
            .add_simulator_rear_uncertainty()
            .add_temperature_uncertainty()
            .add_iv_measurement_uncertainty()
            .add_bifaciality_uncertainty()
            .add_equivalent_irradiance_uncertainty()
            .add_repeatability()
            .add_reproducibility()
            .build()
        )


# =============================================================================
# Uncertainty Budget Container
# =============================================================================

@dataclass
class UncertaintyBudget:
    """
    Complete uncertainty budget with all components and analysis methods.
    """
    components: Dict[str, UncertaintyComponent]
    measurement_type: str = "stc"
    is_bifacial: bool = False
    simulator: Optional[SimulatorConfiguration] = None
    reference: Optional[ReferenceDeviceConfiguration] = None
    category_structure: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Calculate derived quantities"""
        self._combined_uncertainty = None
        self._expanded_uncertainty = None
        self._contributions = None

    @property
    def enabled_components(self) -> Dict[str, UncertaintyComponent]:
        """Get only enabled components"""
        return {k: v for k, v in self.components.items() if v.enabled}

    @property
    def combined_standard_uncertainty(self) -> float:
        """Calculate combined standard uncertainty (k=1)"""
        if self._combined_uncertainty is None:
            variance_sum = sum(
                c.variance_contribution
                for c in self.enabled_components.values()
            )
            self._combined_uncertainty = np.sqrt(variance_sum)
        return self._combined_uncertainty

    @property
    def expanded_uncertainty(self) -> float:
        """Calculate expanded uncertainty (k=2, 95%)"""
        if self._expanded_uncertainty is None:
            self._expanded_uncertainty = 2.0 * self.combined_standard_uncertainty
        return self._expanded_uncertainty

    def get_expanded_uncertainty(self, coverage_factor: float = 2.0) -> float:
        """Get expanded uncertainty with specified k-factor"""
        return coverage_factor * self.combined_standard_uncertainty

    @property
    def contributions(self) -> Dict[str, float]:
        """Calculate variance contribution fractions for each component"""
        if self._contributions is None:
            total_var = self.combined_standard_uncertainty ** 2
            if total_var > 0:
                self._contributions = {
                    name: comp.variance_contribution / total_var
                    for name, comp in self.enabled_components.items()
                }
            else:
                self._contributions = {}
        return self._contributions

    def get_category_contributions(self) -> Dict[str, float]:
        """Get contributions by category"""
        category_vars: Dict[str, float] = {}

        for comp in self.enabled_components.values():
            cat = comp.category.value
            if cat not in category_vars:
                category_vars[cat] = 0.0
            category_vars[cat] += comp.variance_contribution

        total_var = sum(category_vars.values())
        if total_var > 0:
            return {k: v / total_var for k, v in category_vars.items()}
        return {}

    def get_dominant_contributors(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get the n largest contributors"""
        sorted_contrib = sorted(
            self.contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_contrib[:n]

    def effective_degrees_of_freedom(self) -> float:
        """Calculate effective degrees of freedom (Welch-Satterthwaite)"""
        u_c = self.combined_standard_uncertainty
        if u_c == 0:
            return float('inf')

        denominator = sum(
            (comp.variance_contribution ** 2) / comp.degrees_of_freedom
            for comp in self.enabled_components.values()
            if comp.degrees_of_freedom > 0 and comp.degrees_of_freedom != float('inf')
        )

        if denominator == 0:
            return float('inf')

        return (u_c ** 4) / denominator

    def to_dataframe(self):
        """Convert to pandas DataFrame for analysis"""
        try:
            import pandas as pd
            data = [comp.to_dict() for comp in self.components.values()]
            return pd.DataFrame(data)
        except ImportError:
            return None

    def to_dict(self) -> Dict:
        """Convert entire budget to dictionary"""
        return {
            "measurement_type": self.measurement_type,
            "is_bifacial": self.is_bifacial,
            "combined_uncertainty_k1": self.combined_standard_uncertainty,
            "expanded_uncertainty_k2": self.expanded_uncertainty,
            "effective_dof": self.effective_degrees_of_freedom(),
            "simulator": {
                "manufacturer": self.simulator.manufacturer if self.simulator else None,
                "model": self.simulator.model if self.simulator else None,
            },
            "reference": {
                "name": self.reference.name if self.reference else None,
            },
            "components": {
                name: comp.to_dict()
                for name, comp in self.components.items()
            },
            "category_contributions": self.get_category_contributions(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Export to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Generate text summary of uncertainty budget"""
        lines = [
            "=" * 60,
            "UNCERTAINTY BUDGET SUMMARY",
            "=" * 60,
            f"Measurement Type: {self.measurement_type.upper()}",
            f"Bifacial: {'Yes' if self.is_bifacial else 'No'}",
            "",
            f"Combined Standard Uncertainty (k=1): {self.combined_standard_uncertainty:.3f}%",
            f"Expanded Uncertainty (k=2, 95%): {self.expanded_uncertainty:.3f}%",
            f"Effective DOF: {self.effective_degrees_of_freedom():.1f}",
            "",
            "Top Contributors:",
        ]

        for name, fraction in self.get_dominant_contributors(5):
            lines.append(f"  {name}: {fraction*100:.1f}%")

        if self.simulator:
            lines.extend([
                "",
                f"Simulator: {self.simulator.manufacturer} {self.simulator.model}",
            ])

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Build a bifacial uncertainty budget with PASAN simulator
    builder = UncertaintyBudgetBuilder(measurement_type="bifacial_stc", is_bifacial=True)

    budget = (
        builder
        .set_simulator("pasan_highlight_bifacial")
        .set_reference("nrel_wpvs")
        .build_bifacial_stc()
    )

    # Print summary
    print(budget.summary())

    # Show detailed contributions
    print("\nDetailed Component Contributions:")
    print("-" * 60)
    for name, fraction in sorted(budget.contributions.items(), key=lambda x: -x[1]):
        comp = budget.components[name]
        print(f"{name:30s}: {fraction*100:6.2f}%  (u={comp.standard_uncertainty:.3f}%)")

    # Category breakdown
    print("\nCategory Contributions:")
    print("-" * 60)
    for cat, fraction in sorted(budget.get_category_contributions().items(), key=lambda x: -x[1]):
        print(f"{cat:30s}: {fraction*100:6.2f}%")

    # List available simulators
    print("\n" + "=" * 60)
    print("AVAILABLE SIMULATORS")
    print("=" * 60)
    for key in list_simulators():
        sim = get_simulator(key)
        bifi = "✓" if sim.bifacial_capable else " "
        print(f"[{bifi}] {key:30s} - {sim.manufacturer} {sim.model}")

    print("\nBifacial-capable simulators:")
    for key in list_simulators(bifacial_only=True):
        sim = get_simulator(key)
        print(f"  - {sim.manufacturer} {sim.model}")
