"""
PV Uncertainty Calculator
Implements GUM (Guide to the Expression of Uncertainty in Measurement) methodology
for calculating measurement uncertainty in PV systems.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class UncertaintyComponent:
    """Represents a single uncertainty component."""
    name: str
    value: float
    standard_uncertainty: float
    distribution: str = "normal"  # normal, uniform, triangular
    sensitivity_coefficient: float = 1.0

    @property
    def variance_contribution(self) -> float:
        """Calculate variance contribution of this component."""
        return (self.sensitivity_coefficient * self.standard_uncertainty) ** 2


class PVUncertaintyCalculator:
    """
    Calculator for PV measurement uncertainties following GUM methodology.
    """

    def __init__(self):
        self.components: List[UncertaintyComponent] = []

    def add_component(
        self,
        name: str,
        value: float,
        uncertainty: float,
        uncertainty_type: str = "standard",
        distribution: str = "normal",
        sensitivity_coefficient: float = 1.0
    ) -> None:
        """
        Add an uncertainty component.

        Args:
            name: Component name
            value: Measured value
            uncertainty: Uncertainty value
            uncertainty_type: 'standard' or 'expanded' (k=2)
            distribution: 'normal', 'uniform', or 'triangular'
            sensitivity_coefficient: Sensitivity coefficient for this component
        """
        # Convert to standard uncertainty if expanded
        if uncertainty_type == "expanded":
            standard_uncertainty = uncertainty / 2.0  # Assuming k=2
        else:
            standard_uncertainty = uncertainty

        # Convert based on distribution
        if distribution == "uniform":
            # For uniform distribution: u = a/sqrt(3)
            standard_uncertainty = uncertainty / np.sqrt(3)
        elif distribution == "triangular":
            # For triangular distribution: u = a/sqrt(6)
            standard_uncertainty = uncertainty / np.sqrt(6)

        component = UncertaintyComponent(
            name=name,
            value=value,
            standard_uncertainty=standard_uncertainty,
            distribution=distribution,
            sensitivity_coefficient=sensitivity_coefficient
        )
        self.components.append(component)

    def calculate_combined_uncertainty(self) -> Tuple[float, Dict]:
        """
        Calculate combined standard uncertainty using root sum of squares.

        Returns:
            Tuple of (combined_uncertainty, uncertainty_budget_dict)
        """
        if not self.components:
            return 0.0, {}

        # Calculate variance contributions
        total_variance = sum(comp.variance_contribution for comp in self.components)
        combined_uncertainty = np.sqrt(total_variance)

        # Create uncertainty budget
        budget = {
            "components": [],
            "combined_standard_uncertainty": combined_uncertainty,
            "expanded_uncertainty_k2": combined_uncertainty * 2.0,
            "total_variance": total_variance
        }

        for comp in self.components:
            variance_contrib = comp.variance_contribution
            percentage_contrib = (variance_contrib / total_variance * 100) if total_variance > 0 else 0

            budget["components"].append({
                "name": comp.name,
                "value": comp.value,
                "standard_uncertainty": comp.standard_uncertainty,
                "sensitivity_coefficient": comp.sensitivity_coefficient,
                "variance_contribution": variance_contrib,
                "percentage_contribution": percentage_contrib,
                "distribution": comp.distribution
            })

        # Sort by contribution (descending)
        budget["components"].sort(key=lambda x: x["percentage_contribution"], reverse=True)

        return combined_uncertainty, budget

    def calculate_relative_uncertainty(self, measured_value: float) -> float:
        """
        Calculate relative combined uncertainty as percentage.

        Args:
            measured_value: The measured value

        Returns:
            Relative uncertainty in percent
        """
        combined_uncertainty, _ = self.calculate_combined_uncertainty()
        if measured_value == 0:
            return 0.0
        return (combined_uncertainty / abs(measured_value)) * 100

    def clear_components(self) -> None:
        """Clear all uncertainty components."""
        self.components = []


class PVPowerUncertainty:
    """
    Specialized calculator for PV power measurement uncertainty.
    """

    @staticmethod
    def calculate_power_uncertainty(
        irradiance: float,
        irradiance_uncertainty: float,
        temperature: float,
        temperature_uncertainty: float,
        power: float,
        power_meter_uncertainty: float,
        module_efficiency: float = 0.20,
        efficiency_uncertainty: float = 0.01,
        temp_coefficient: float = -0.004,  # %/°C
        reference_temp: float = 25.0
    ) -> Dict:
        """
        Calculate combined uncertainty for PV power measurement.

        Args:
            irradiance: Measured irradiance (W/m²)
            irradiance_uncertainty: Irradiance uncertainty (W/m²)
            temperature: Module temperature (°C)
            temperature_uncertainty: Temperature uncertainty (°C)
            power: Measured power (W)
            power_meter_uncertainty: Power meter uncertainty (W)
            module_efficiency: Module efficiency (fraction)
            efficiency_uncertainty: Efficiency uncertainty (fraction)
            temp_coefficient: Temperature coefficient (%/°C)
            reference_temp: Reference temperature (°C)

        Returns:
            Dictionary with uncertainty analysis results
        """
        calc = PVUncertaintyCalculator()

        # Irradiance contribution
        # Sensitivity: dP/dG ≈ P/G (assuming linear relationship)
        if irradiance > 0:
            irrad_sensitivity = power / irradiance
        else:
            irrad_sensitivity = 0

        calc.add_component(
            name="Irradiance Measurement",
            value=irradiance,
            uncertainty=irradiance_uncertainty,
            uncertainty_type="standard",
            distribution="normal",
            sensitivity_coefficient=irrad_sensitivity
        )

        # Temperature contribution
        # Sensitivity: dP/dT = P * temp_coefficient
        temp_sensitivity = power * temp_coefficient / 100  # Convert %/°C to fraction/°C

        calc.add_component(
            name="Temperature Measurement",
            value=temperature,
            uncertainty=temperature_uncertainty,
            uncertainty_type="standard",
            distribution="normal",
            sensitivity_coefficient=abs(temp_sensitivity)
        )

        # Power meter contribution
        calc.add_component(
            name="Power Meter",
            value=power,
            uncertainty=power_meter_uncertainty,
            uncertainty_type="standard",
            distribution="normal",
            sensitivity_coefficient=1.0
        )

        # Module efficiency contribution (if varying)
        if efficiency_uncertainty > 0:
            efficiency_sensitivity = power / module_efficiency if module_efficiency > 0 else 0
            calc.add_component(
                name="Module Efficiency",
                value=module_efficiency,
                uncertainty=efficiency_uncertainty,
                uncertainty_type="standard",
                distribution="normal",
                sensitivity_coefficient=efficiency_sensitivity
            )

        combined_unc, budget = calc.calculate_combined_uncertainty()

        # Calculate relative uncertainty
        relative_unc = (combined_unc / power * 100) if power > 0 else 0

        result = {
            "power": power,
            "combined_uncertainty": combined_unc,
            "expanded_uncertainty_k2": combined_unc * 2.0,
            "relative_uncertainty_percent": relative_unc,
            "confidence_interval_95": (
                power - combined_unc * 2.0,
                power + combined_unc * 2.0
            ),
            "budget": budget
        }

        return result


class PVPerformanceRatioUncertainty:
    """
    Calculator for Performance Ratio (PR) uncertainty in PV systems.
    """

    @staticmethod
    def calculate_pr_uncertainty(
        measured_energy: float,
        measured_energy_uncertainty: float,
        irradiation: float,
        irradiation_uncertainty: float,
        installed_capacity: float,
        capacity_uncertainty: float,
        performance_ratio: float
    ) -> Dict:
        """
        Calculate uncertainty in Performance Ratio.

        PR = E_measured / (H * P_installed)

        Args:
            measured_energy: Measured energy output (kWh)
            measured_energy_uncertainty: Energy measurement uncertainty (kWh)
            irradiation: Total irradiation (kWh/m²)
            irradiation_uncertainty: Irradiation uncertainty (kWh/m²)
            installed_capacity: Installed DC capacity (kWp)
            capacity_uncertainty: Capacity uncertainty (kWp)
            performance_ratio: Calculated PR (fraction)

        Returns:
            Dictionary with PR uncertainty analysis
        """
        calc = PVUncertaintyCalculator()

        # Energy measurement contribution
        # Sensitivity: dPR/dE = 1 / (H * P)
        denominator = irradiation * installed_capacity
        if denominator > 0:
            energy_sensitivity = 1.0 / denominator
        else:
            energy_sensitivity = 0

        calc.add_component(
            name="Energy Measurement",
            value=measured_energy,
            uncertainty=measured_energy_uncertainty,
            uncertainty_type="standard",
            distribution="normal",
            sensitivity_coefficient=energy_sensitivity
        )

        # Irradiation contribution
        # Sensitivity: dPR/dH = -E / (H² * P)
        if denominator > 0 and irradiation > 0:
            irrad_sensitivity = -measured_energy / (irradiation**2 * installed_capacity)
        else:
            irrad_sensitivity = 0

        calc.add_component(
            name="Irradiation",
            value=irradiation,
            uncertainty=irradiation_uncertainty,
            uncertainty_type="standard",
            distribution="normal",
            sensitivity_coefficient=abs(irrad_sensitivity)
        )

        # Installed capacity contribution
        # Sensitivity: dPR/dP = -E / (H * P²)
        if denominator > 0 and installed_capacity > 0:
            capacity_sensitivity = -measured_energy / (irradiation * installed_capacity**2)
        else:
            capacity_sensitivity = 0

        calc.add_component(
            name="Installed Capacity",
            value=installed_capacity,
            uncertainty=capacity_uncertainty,
            uncertainty_type="standard",
            distribution="normal",
            sensitivity_coefficient=abs(capacity_sensitivity)
        )

        combined_unc, budget = calc.calculate_combined_uncertainty()

        # Calculate relative uncertainty
        relative_unc = (combined_unc / performance_ratio * 100) if performance_ratio > 0 else 0

        result = {
            "performance_ratio": performance_ratio,
            "combined_uncertainty": combined_unc,
            "expanded_uncertainty_k2": combined_unc * 2.0,
            "relative_uncertainty_percent": relative_unc,
            "confidence_interval_95": (
                performance_ratio - combined_unc * 2.0,
                performance_ratio + combined_unc * 2.0
            ),
            "budget": budget
        }

        return result
