"""
Enhanced PV Measurement Uncertainty Calculator
Comprehensive uncertainty analysis for solar PV IV measurements including all
relevant factors from reference devices, equipment, procedures, and environmental conditions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json


@dataclass
class UncertaintyFactor:
    """Individual uncertainty factor with metadata."""
    category_id: str
    subcategory_id: str
    factor_id: str
    name: str
    value: float = 0.0
    standard_uncertainty: float = 0.0
    distribution: str = "normal"  # normal, uniform, triangular
    sensitivity_coefficient: float = 1.0
    unit: str = ""
    enabled: bool = True
    notes: str = ""

    @property
    def variance_contribution(self) -> float:
        """Calculate variance contribution."""
        if not self.enabled:
            return 0.0
        return (self.sensitivity_coefficient * self.standard_uncertainty) ** 2


class PVUncertaintyBudget:
    """
    Comprehensive uncertainty budget for PV measurements.
    Organized according to fishbone diagram categories.
    """

    def __init__(self, measurement_type: str = "STC"):
        """
        Initialize uncertainty budget.

        Args:
            measurement_type: Type of measurement (STC, NMOT, Low_Irradiance, etc.)
        """
        self.measurement_type = measurement_type
        self.factors: List[UncertaintyFactor] = []
        self.result_value: float = 0.0
        self.result_parameter: str = "Pmax"  # Pmax, Voc, Isc, etc.

    def add_factor(self, factor: UncertaintyFactor) -> None:
        """Add an uncertainty factor to the budget."""
        self.factors.append(factor)

    def add_reference_device_uncertainty(
        self,
        calibration_uncertainty: float,
        drift_uncertainty: float = 0.0,
        positioning_uncertainty: float = 0.0,
        distribution: str = "normal"
    ) -> None:
        """
        Add reference device uncertainty components.

        Args:
            calibration_uncertainty: Calibration uncertainty (%)
            drift_uncertainty: Long-term drift uncertainty (%)
            positioning_uncertainty: Position-dependent uncertainty (%)
            distribution: Probability distribution
        """
        # 1.1.1 WPVS/Reference Cell Calibration
        self.add_factor(UncertaintyFactor(
            category_id="1",
            subcategory_id="1.1",
            factor_id="1.1.1",
            name="Reference Device Calibration",
            value=calibration_uncertainty,
            standard_uncertainty=calibration_uncertainty,
            distribution=distribution,
            sensitivity_coefficient=1.0,
            unit="%",
            notes="Calibration uncertainty from reference laboratory"
        ))

        # 1.2.1 Long-term Drift
        if drift_uncertainty > 0:
            self.add_factor(UncertaintyFactor(
                category_id="1",
                subcategory_id="1.2",
                factor_id="1.2.1",
                name="Reference Device Drift",
                value=drift_uncertainty,
                standard_uncertainty=drift_uncertainty,
                distribution="rectangular",  # Assume uniform distribution for drift
                sensitivity_coefficient=1.0,
                unit="%",
                notes="Long-term stability of reference device"
            ))

        # 1.3.1 Positioning Effect
        if positioning_uncertainty > 0:
            self.add_factor(UncertaintyFactor(
                category_id="1",
                subcategory_id="1.3",
                factor_id="1.3.1",
                name="Reference Device Positioning",
                value=positioning_uncertainty,
                standard_uncertainty=positioning_uncertainty / np.sqrt(3),  # Rectangular distribution
                distribution="rectangular",
                sensitivity_coefficient=1.0,
                unit="%",
                notes="Position difference between reference and test module"
            ))

    def add_sun_simulator_uncertainty(
        self,
        uniformity: float,
        temporal_instability: float,
        spectral_mismatch: float = 0.0
    ) -> None:
        """
        Add sun simulator uncertainty components.

        Args:
            uniformity: Spatial non-uniformity (%)
            temporal_instability: Temporal instability (%)
            spectral_mismatch: Spectral mismatch factor uncertainty (%)
        """
        # 2.1.1 Spatial Non-uniformity
        self.add_factor(UncertaintyFactor(
            category_id="2",
            subcategory_id="2.1",
            factor_id="2.1.1",
            name="Spatial Non-uniformity",
            value=uniformity,
            standard_uncertainty=uniformity / np.sqrt(3),  # Rectangular distribution
            distribution="rectangular",
            sensitivity_coefficient=1.0,
            unit="%",
            notes="Irradiance uniformity classification"
        ))

        # 2.2.1 Temporal Instability
        self.add_factor(UncertaintyFactor(
            category_id="2",
            subcategory_id="2.2",
            factor_id="2.2.1",
            name="Temporal Instability",
            value=temporal_instability,
            standard_uncertainty=temporal_instability / np.sqrt(3),
            distribution="rectangular",
            sensitivity_coefficient=1.0,
            unit="%",
            notes="Flash-to-flash or temporal stability"
        ))

        # 2.3.4 Spectral Mismatch
        if spectral_mismatch > 0:
            self.add_factor(UncertaintyFactor(
                category_id="2",
                subcategory_id="2.3",
                factor_id="2.3.4",
                name="Spectral Mismatch",
                value=spectral_mismatch,
                standard_uncertainty=spectral_mismatch,
                distribution="normal",
                sensitivity_coefficient=1.0,
                unit="%",
                notes="Mismatch between simulator, reference, and test device"
            ))

    def add_temperature_uncertainty(
        self,
        sensor_uncertainty: float,
        uniformity_uncertainty: float,
        temp_coefficient_uncertainty: float = 0.0,
        delta_temperature: float = 0.0
    ) -> None:
        """
        Add temperature measurement and correction uncertainties.

        Args:
            sensor_uncertainty: Temperature sensor uncertainty (°C)
            uniformity_uncertainty: Module temperature uniformity (°C)
            temp_coefficient_uncertainty: Temperature coefficient uncertainty (%/°C)
            delta_temperature: Temperature difference from STC (°C)
        """
        # 3.1.1 Sensor Calibration
        self.add_factor(UncertaintyFactor(
            category_id="3",
            subcategory_id="3.1",
            factor_id="3.1.1",
            name="Temperature Sensor Calibration",
            value=sensor_uncertainty,
            standard_uncertainty=sensor_uncertainty,
            distribution="normal",
            sensitivity_coefficient=abs(delta_temperature) * 0.004 if delta_temperature != 0 else 0.1,  # Typical ~0.4%/°C
            unit="°C",
            notes="Thermocouple/RTD calibration uncertainty"
        ))

        # 3.2.1 Temperature Uniformity
        if uniformity_uncertainty > 0:
            self.add_factor(UncertaintyFactor(
                category_id="3",
                subcategory_id="3.2",
                factor_id="3.2.1",
                name="Temperature Uniformity",
                value=uniformity_uncertainty,
                standard_uncertainty=uniformity_uncertainty / np.sqrt(3),
                distribution="rectangular",
                sensitivity_coefficient=abs(delta_temperature) * 0.004 if delta_temperature != 0 else 0.1,
                unit="°C",
                notes="Module temperature gradient"
            ))

        # 3.3.2 Temperature Coefficient Uncertainty
        if temp_coefficient_uncertainty > 0 and delta_temperature != 0:
            self.add_factor(UncertaintyFactor(
                category_id="3",
                subcategory_id="3.3",
                factor_id="3.3.2",
                name="Temperature Coefficient Uncertainty",
                value=temp_coefficient_uncertainty,
                standard_uncertainty=temp_coefficient_uncertainty,
                distribution="normal",
                sensitivity_coefficient=abs(delta_temperature),
                unit="%/°C",
                notes="Uncertainty in temperature coefficient value"
            ))

    def add_iv_measurement_uncertainty(
        self,
        voltage_uncertainty: float,
        current_uncertainty: float,
        curve_fitting_uncertainty: float = 0.0
    ) -> None:
        """
        Add I-V curve measurement uncertainties.

        Args:
            voltage_uncertainty: Voltage measurement uncertainty (%)
            current_uncertainty: Current measurement uncertainty (%)
            curve_fitting_uncertainty: Curve fitting/interpolation uncertainty (%)
        """
        # 4.1.1 Voltage Measurement
        self.add_factor(UncertaintyFactor(
            category_id="4",
            subcategory_id="4.1",
            factor_id="4.1.1",
            name="Voltage Measurement",
            value=voltage_uncertainty,
            standard_uncertainty=voltage_uncertainty,
            distribution="normal",
            sensitivity_coefficient=0.5,  # For Pmax, partial derivative
            unit="%",
            notes="Voltmeter calibration and contact resistance"
        ))

        # 4.2.1 Current Measurement
        self.add_factor(UncertaintyFactor(
            category_id="4",
            subcategory_id="4.2",
            factor_id="4.2.1",
            name="Current Measurement",
            value=current_uncertainty,
            standard_uncertainty=current_uncertainty,
            distribution="normal",
            sensitivity_coefficient=0.5,  # For Pmax
            unit="%",
            notes="Ammeter/shunt calibration uncertainty"
        ))

        # 4.3.3 Curve Fitting
        if curve_fitting_uncertainty > 0:
            self.add_factor(UncertaintyFactor(
                category_id="4",
                subcategory_id="4.3",
                factor_id="4.3.3",
                name="Curve Fitting Algorithm",
                value=curve_fitting_uncertainty,
                standard_uncertainty=curve_fitting_uncertainty / np.sqrt(3),
                distribution="rectangular",
                sensitivity_coefficient=1.0,
                unit="%",
                notes="MPP determination and curve fitting"
            ))

    def add_module_characteristics_uncertainty(
        self,
        hysteresis_uncertainty: float = 0.0,
        stabilization_uncertainty: float = 0.0
    ) -> None:
        """
        Add module-specific uncertainty factors.

        Args:
            hysteresis_uncertainty: Hysteresis effect (%)
            stabilization_uncertainty: Pre-conditioning/stabilization (%)
        """
        # 5.2.1 Hysteresis
        if hysteresis_uncertainty > 0:
            self.add_factor(UncertaintyFactor(
                category_id="5",
                subcategory_id="5.2",
                factor_id="5.2.1",
                name="Hysteresis Effects",
                value=hysteresis_uncertainty,
                standard_uncertainty=hysteresis_uncertainty / np.sqrt(3),
                distribution="rectangular",
                sensitivity_coefficient=1.0,
                unit="%",
                notes="Forward/reverse scan differences"
            ))

        # 5.2.2 Stabilization
        if stabilization_uncertainty > 0:
            self.add_factor(UncertaintyFactor(
                category_id="5",
                subcategory_id="5.2",
                factor_id="5.2.2",
                name="Module Stabilization",
                value=stabilization_uncertainty,
                standard_uncertainty=stabilization_uncertainty / np.sqrt(3),
                distribution="rectangular",
                sensitivity_coefficient=1.0,
                unit="%",
                notes="Pre-conditioning and stabilization time"
            ))

    def add_repeatability_reproducibility(
        self,
        repeatability: float = 0.0,
        reproducibility: float = 0.0
    ) -> None:
        """
        Add repeatability and reproducibility uncertainties.

        Args:
            repeatability: Within-lab repeatability (%)
            reproducibility: Between-lab reproducibility (%)
        """
        # 7.1.1 Repeatability
        if repeatability > 0:
            self.add_factor(UncertaintyFactor(
                category_id="7",
                subcategory_id="7.1",
                factor_id="7.1.1",
                name="Repeatability",
                value=repeatability,
                standard_uncertainty=repeatability,
                distribution="normal",
                sensitivity_coefficient=1.0,
                unit="%",
                notes="Intra-laboratory measurement repeatability"
            ))

        # 7.2.1 Reproducibility
        if reproducibility > 0:
            self.add_factor(UncertaintyFactor(
                category_id="7",
                subcategory_id="7.2",
                factor_id="7.2.1",
                name="Reproducibility",
                value=reproducibility,
                standard_uncertainty=reproducibility,
                distribution="normal",
                sensitivity_coefficient=1.0,
                unit="%",
                notes="Inter-laboratory measurement reproducibility (ILC/RR)"
            ))

    def calculate_combined_uncertainty(self) -> Tuple[float, Dict]:
        """
        Calculate combined standard uncertainty using GUM methodology.

        Returns:
            Tuple of (combined_uncertainty_%, uncertainty_budget_dict)
        """
        # Filter enabled factors
        enabled_factors = [f for f in self.factors if f.enabled]

        if not enabled_factors:
            return 0.0, {"components": [], "combined_standard_uncertainty": 0.0}

        # Calculate total variance
        total_variance = sum(f.variance_contribution for f in enabled_factors)
        combined_uncertainty = np.sqrt(total_variance)

        # Create detailed budget
        budget = {
            "measurement_type": self.measurement_type,
            "result_parameter": self.result_parameter,
            "result_value": self.result_value,
            "combined_standard_uncertainty": combined_uncertainty,
            "expanded_uncertainty_k2": combined_uncertainty * 2.0,
            "relative_uncertainty_percent": (combined_uncertainty / self.result_value * 100)
            if self.result_value > 0 else 0.0,
            "components": []
        }

        # Add component details
        for factor in enabled_factors:
            variance_contrib = factor.variance_contribution
            percentage_contrib = (variance_contrib / total_variance * 100) if total_variance > 0 else 0

            budget["components"].append({
                "category_id": factor.category_id,
                "subcategory_id": factor.subcategory_id,
                "factor_id": factor.factor_id,
                "name": factor.name,
                "value": factor.value,
                "standard_uncertainty": factor.standard_uncertainty,
                "distribution": factor.distribution,
                "sensitivity_coefficient": factor.sensitivity_coefficient,
                "variance_contribution": variance_contrib,
                "percentage_contribution": percentage_contrib,
                "unit": factor.unit,
                "notes": factor.notes
            })

        # Sort by contribution (descending)
        budget["components"].sort(key=lambda x: x["percentage_contribution"], reverse=True)

        return combined_uncertainty, budget

    def export_to_dict(self) -> Dict:
        """Export complete uncertainty budget to dictionary."""
        _, budget = self.calculate_combined_uncertainty()
        return budget

    def export_to_json(self, filepath: str) -> None:
        """Export uncertainty budget to JSON file."""
        budget = self.export_to_dict()
        with open(filepath, 'w') as f:
            json.dump(budget, f, indent=2)

    def get_factors_by_category(self, category_id: str) -> List[UncertaintyFactor]:
        """Get all factors in a specific category."""
        return [f for f in self.factors if f.category_id == category_id and f.enabled]

    def get_category_contribution(self, category_id: str) -> float:
        """Calculate total contribution from a category."""
        factors = self.get_factors_by_category(category_id)
        return np.sqrt(sum(f.variance_contribution for f in factors))


class STCMeasurementUncertainty:
    """
    Complete STC measurement uncertainty calculator.
    Combines all uncertainty sources for STC power measurements.
    """

    @staticmethod
    def calculate(
        # Measurement result
        pmax_measured: float,  # W
        # Reference device
        ref_calibration_unc: float = 1.5,  # %
        ref_drift_unc: float = 0.5,  # %
        ref_positioning_unc: float = 0.3,  # %
        # Sun simulator
        simulator_uniformity: float = 2.0,  # %
        simulator_temporal: float = 0.5,  # %
        spectral_mismatch_unc: float = 1.0,  # %
        # Temperature
        temp_sensor_unc: float = 0.5,  # °C
        temp_uniformity_unc: float = 1.0,  # °C
        temp_coeff_unc: float = 0.05,  # %/°C
        delta_temperature: float = 0.0,  # °C from 25°C
        # I-V measurement
        voltage_unc: float = 0.2,  # %
        current_unc: float = 0.2,  # %
        curve_fitting_unc: float = 0.1,  # %
        # Module characteristics
        hysteresis_unc: float = 0.3,  # %
        stabilization_unc: float = 0.2,  # %
        # Repeatability/Reproducibility
        repeatability: float = 0.5,  # %
        reproducibility: float = 1.5,  # %
    ) -> Dict:
        """
        Calculate complete STC measurement uncertainty.

        Returns:
            Dictionary with uncertainty budget and results
        """
        budget = PVUncertaintyBudget(measurement_type="STC")
        budget.result_value = pmax_measured
        budget.result_parameter = "Pmax"

        # Add all uncertainty components
        budget.add_reference_device_uncertainty(
            ref_calibration_unc, ref_drift_unc, ref_positioning_unc
        )

        budget.add_sun_simulator_uncertainty(
            simulator_uniformity, simulator_temporal, spectral_mismatch_unc
        )

        budget.add_temperature_uncertainty(
            temp_sensor_unc, temp_uniformity_unc, temp_coeff_unc, delta_temperature
        )

        budget.add_iv_measurement_uncertainty(
            voltage_unc, current_unc, curve_fitting_unc
        )

        budget.add_module_characteristics_uncertainty(
            hysteresis_unc, stabilization_unc
        )

        budget.add_repeatability_reproducibility(
            repeatability, reproducibility
        )

        # Calculate combined uncertainty
        combined_unc, result = budget.calculate_combined_uncertainty()

        # Add absolute values
        result["pmax_measured"] = pmax_measured
        result["pmax_uncertainty_absolute"] = pmax_measured * combined_unc / 100
        result["pmax_confidence_interval_95"] = (
            pmax_measured - result["pmax_uncertainty_absolute"] * 2,
            pmax_measured + result["pmax_uncertainty_absolute"] * 2
        )

        return result


def create_default_stc_budget(pmax_measured: float = 300.0) -> PVUncertaintyBudget:
    """
    Create a default STC uncertainty budget with typical values.

    Args:
        pmax_measured: Measured Pmax value in watts

    Returns:
        Pre-configured PVUncertaintyBudget
    """
    budget = PVUncertaintyBudget(measurement_type="STC")
    budget.result_value = pmax_measured
    budget.result_parameter = "Pmax"

    # Typical values for a well-equipped commercial test lab
    budget.add_reference_device_uncertainty(
        calibration_uncertainty=1.5,  # Secondary reference module
        drift_uncertainty=0.3,
        positioning_uncertainty=0.2
    )

    budget.add_sun_simulator_uncertainty(
        uniformity=2.0,  # Class AAA
        temporal_instability=0.5,
        spectral_mismatch=0.8
    )

    budget.add_temperature_uncertainty(
        sensor_uncertainty=0.5,
        uniformity_uncertainty=1.0,
        temp_coefficient_uncertainty=0.05,
        delta_temperature=0.0  # At 25°C
    )

    budget.add_iv_measurement_uncertainty(
        voltage_uncertainty=0.2,
        current_uncertainty=0.2,
        curve_fitting_uncertainty=0.1
    )

    budget.add_module_characteristics_uncertainty(
        hysteresis_uncertainty=0.3,
        stabilization_uncertainty=0.2
    )

    budget.add_repeatability_reproducibility(
        repeatability=0.5,
        reproducibility=1.5
    )

    return budget
