"""
Financial Impact Calculator for PV Measurement Uncertainty
Calculates the monetary implications of measurement uncertainty on module sales,
warranty claims, and project financing.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from config_data import CURRENCIES


@dataclass
class FinancialScenario:
    """Financial analysis scenario configuration."""
    scenario_type: str  # "fresh_module", "warranty_claim", "project_financing"
    module_power: float  # W
    power_uncertainty: float  # W
    module_price_per_watt: float  # Currency/W
    currency: str = "USD"
    plant_size_mw: float = 1.0  # For project scenarios
    operating_years: int = 25
    discount_rate: float = 0.05  # 5% per year
    degradation_rate: float = 0.005  # 0.5% per year
    energy_price_per_kwh: float = 0.10  # Currency/kWh


class FinancialImpactCalculator:
    """
    Calculate financial implications of PV measurement uncertainty.
    """

    @staticmethod
    def calculate_module_price_impact(
        module_power: float,
        power_uncertainty: float,
        price_per_watt: float,
        currency: str = "USD",
        confidence_level: float = 95.0
    ) -> Dict:
        """
        Calculate price impact for fresh module sales.

        Args:
            module_power: Nameplate power rating (W)
            power_uncertainty: Combined standard uncertainty (W)
            price_per_watt: Module price (Currency/W)
            currency: Currency code
            confidence_level: Confidence level (68, 95, or 99)

        Returns:
            Dictionary with price impact analysis
        """
        # Determine k-factor for confidence level
        k_factors = {68.0: 1.0, 95.0: 2.0, 99.0: 3.0}
        k = k_factors.get(confidence_level, 2.0)

        # Calculate power bounds
        power_lower = module_power - k * power_uncertainty
        power_upper = module_power + k * power_uncertainty
        power_range = 2 * k * power_uncertainty

        # Calculate price impacts
        module_price = module_power * price_per_watt
        price_uncertainty = power_uncertainty * price_per_watt * k
        price_lower = power_lower * price_per_watt
        price_upper = power_upper * price_per_watt

        # Relative impacts
        relative_uncertainty_percent = (power_uncertainty / module_power) * 100
        price_risk_percent = (price_uncertainty / module_price) * 100

        currency_symbol = CURRENCIES.get(currency, {}).get("symbol", "$")

        result = {
            "module_power_nominal": module_power,
            "power_uncertainty": power_uncertainty,
            "power_lower_bound": power_lower,
            "power_upper_bound": power_upper,
            "power_range": power_range,
            "relative_uncertainty_percent": relative_uncertainty_percent,
            "module_price_nominal": module_price,
            "price_per_watt": price_per_watt,
            "price_uncertainty": price_uncertainty,
            "price_lower_bound": price_lower,
            "price_upper_bound": price_upper,
            "price_risk_percent": price_risk_percent,
            "currency": currency,
            "currency_symbol": currency_symbol,
            "confidence_level": confidence_level,
            "interpretation": {
                "seller_risk": price_uncertainty,  # Potential loss if actual < nominal
                "buyer_risk": price_uncertainty,   # Potential loss if actual > nominal
                "measurement_value": price_uncertainty * 2  # Total range of price uncertainty
            }
        }

        return result

    @staticmethod
    def calculate_warranty_claim_impact(
        module_power_measured: float,
        nameplate_power: float,
        power_uncertainty: float,
        warranty_threshold: float,  # Fraction, e.g., 0.90 for 90% of nameplate
        module_price_per_watt: float,
        currency: str = "USD"
    ) -> Dict:
        """
        Calculate impact on warranty/insurance claims.

        Args:
            module_power_measured: Measured power (W)
            nameplate_power: Original nameplate rating (W)
            power_uncertainty: Measurement uncertainty (W)
            warranty_threshold: Warranty threshold (fraction of nameplate)
            module_price_per_watt: Replacement/compensation price (Currency/W)
            currency: Currency code

        Returns:
            Dictionary with warranty claim analysis
        """
        warranty_power = nameplate_power * warranty_threshold
        power_deficit = nameplate_power - module_power_measured

        # Calculate probability of valid claim considering uncertainty
        # Using normal distribution assumption
        z_score = (module_power_measured - warranty_power) / power_uncertainty if power_uncertainty > 0 else 0

        # Probability that true power is below warranty threshold
        from scipy import stats
        prob_below_threshold = stats.norm.cdf(-z_score) if power_uncertainty > 0 else 0.0

        # Confidence bounds
        power_lower_95 = module_power_measured - 2 * power_uncertainty
        power_upper_95 = module_power_measured + 2 * power_uncertainty

        # Financial implications
        deficit_claim_value = power_deficit * module_price_per_watt
        max_claim_uncertainty = 2 * power_uncertainty * module_price_per_watt

        # Decision analysis
        claim_valid_certain = power_upper_95 < warranty_power
        claim_invalid_certain = power_lower_95 > warranty_power
        claim_uncertain = not (claim_valid_certain or claim_invalid_certain)

        currency_symbol = CURRENCIES.get(currency, {}).get("symbol", "$")

        result = {
            "nameplate_power": nameplate_power,
            "measured_power": module_power_measured,
            "power_uncertainty": power_uncertainty,
            "warranty_threshold_percent": warranty_threshold * 100,
            "warranty_power_threshold": warranty_power,
            "power_deficit": power_deficit,
            "deficit_percent": (power_deficit / nameplate_power) * 100,
            "power_lower_95": power_lower_95,
            "power_upper_95": power_upper_95,
            "probability_below_threshold": prob_below_threshold,
            "claim_valid_certain": claim_valid_certain,
            "claim_invalid_certain": claim_invalid_certain,
            "claim_uncertain": claim_uncertain,
            "claim_value_nominal": deficit_claim_value,
            "claim_uncertainty": max_claim_uncertainty,
            "currency": currency,
            "currency_symbol": currency_symbol,
            "recommendation": (
                "VALID CLAIM - Power clearly below warranty threshold"
                if claim_valid_certain
                else "INVALID CLAIM - Power clearly above warranty threshold"
                if claim_invalid_certain
                else "UNCERTAIN - Measurement uncertainty overlaps warranty threshold. Additional testing recommended."
            )
        }

        return result

    @staticmethod
    def calculate_project_npv_impact(
        plant_size_mw: float,
        power_uncertainty_percent: float,
        module_price_per_watt: float,
        energy_price_per_kwh: float,
        operating_years: int = 25,
        discount_rate: float = 0.05,
        degradation_rate: float = 0.005,
        capacity_factor: float = 0.20,
        currency: str = "USD"
    ) -> Dict:
        """
        Calculate NPV impact for project financing scenarios.

        Args:
            plant_size_mw: Plant size (MW DC)
            power_uncertainty_percent: Power measurement uncertainty (%)
            module_price_per_watt: Module cost (Currency/W)
            energy_price_per_kwh: PPA price (Currency/kWh)
            operating_years: Project lifetime (years)
            discount_rate: Discount rate (fraction per year)
            degradation_rate: Annual degradation (fraction per year)
            capacity_factor: Annual capacity factor (fraction)
            currency: Currency code

        Returns:
            Dictionary with NPV and financial analysis
        """
        plant_size_w = plant_size_mw * 1e6
        power_uncertainty_w = plant_size_w * (power_uncertainty_percent / 100)

        # Calculate annual energy production (kWh/year)
        hours_per_year = 8760
        year_1_energy_kwh = plant_size_w * capacity_factor * hours_per_year / 1000

        # Calculate NPV for nominal case
        npv_nominal = 0.0
        for year in range(1, operating_years + 1):
            # Energy degradation
            degradation_factor = (1 - degradation_rate) ** (year - 1)
            annual_energy = year_1_energy_kwh * degradation_factor

            # Revenue and discount
            annual_revenue = annual_energy * energy_price_per_kwh
            discount_factor = (1 + discount_rate) ** year
            npv_nominal += annual_revenue / discount_factor

        # Initial investment
        initial_investment = plant_size_w * module_price_per_watt

        # NPV accounting for uncertainty
        energy_uncertainty_percent = power_uncertainty_percent  # Proportional for DC-AC
        energy_uncertainty_kwh = year_1_energy_kwh * (energy_uncertainty_percent / 100)

        # Lower bound NPV (conservative)
        npv_lower = 0.0
        year_1_energy_lower = year_1_energy_kwh - 2 * energy_uncertainty_kwh
        for year in range(1, operating_years + 1):
            degradation_factor = (1 - degradation_rate) ** (year - 1)
            annual_energy = year_1_energy_lower * degradation_factor
            annual_revenue = annual_energy * energy_price_per_kwh
            discount_factor = (1 + discount_rate) ** year
            npv_lower += annual_revenue / discount_factor

        # Upper bound NPV (optimistic)
        npv_upper = 0.0
        year_1_energy_upper = year_1_energy_kwh + 2 * energy_uncertainty_kwh
        for year in range(1, operating_years + 1):
            degradation_factor = (1 - degradation_rate) ** (year - 1)
            annual_energy = year_1_energy_upper * degradation_factor
            annual_revenue = annual_energy * energy_price_per_kwh
            discount_factor = (1 + discount_rate) ** year
            npv_upper += annual_revenue / discount_factor

        # Calculate ROI
        roi_nominal = ((npv_nominal - initial_investment) / initial_investment) * 100
        roi_lower = ((npv_lower - initial_investment) / initial_investment) * 100
        roi_upper = ((npv_upper - initial_investment) / initial_investment) * 100

        # Payback period (simplified - without discounting)
        simple_payback_nominal = initial_investment / (year_1_energy_kwh * energy_price_per_kwh)
        simple_payback_lower = initial_investment / (year_1_energy_lower * energy_price_per_kwh)
        simple_payback_upper = initial_investment / (year_1_energy_upper * energy_price_per_kwh)

        # NPV impact range
        npv_range = npv_upper - npv_lower
        npv_risk_percent = (npv_range / (2 * npv_nominal)) * 100 if npv_nominal > 0 else 0

        currency_symbol = CURRENCIES.get(currency, {}).get("symbol", "$")

        result = {
            "plant_size_mw": plant_size_mw,
            "plant_size_w": plant_size_w,
            "power_uncertainty_percent": power_uncertainty_percent,
            "power_uncertainty_w": power_uncertainty_w,
            "year_1_energy_kwh_nominal": year_1_energy_kwh,
            "year_1_energy_kwh_lower": year_1_energy_lower,
            "year_1_energy_kwh_upper": year_1_energy_upper,
            "energy_uncertainty_kwh": energy_uncertainty_kwh,
            "initial_investment": initial_investment,
            "npv_nominal": npv_nominal,
            "npv_lower_95": npv_lower,
            "npv_upper_95": npv_upper,
            "npv_range": npv_range,
            "npv_risk_percent": npv_risk_percent,
            "roi_nominal_percent": roi_nominal,
            "roi_lower_percent": roi_lower,
            "roi_upper_percent": roi_upper,
            "payback_period_nominal_years": simple_payback_nominal,
            "payback_period_lower_years": simple_payback_lower,
            "payback_period_upper_years": simple_payback_upper,
            "operating_years": operating_years,
            "discount_rate": discount_rate,
            "degradation_rate": degradation_rate,
            "capacity_factor": capacity_factor,
            "energy_price_per_kwh": energy_price_per_kwh,
            "currency": currency,
            "currency_symbol": currency_symbol,
            "financial_risk_assessment": {
                "npv_at_risk": npv_range / 2,  # ±uncertainty
                "revenue_at_risk_year_1": energy_uncertainty_kwh * energy_price_per_kwh * 2,
                "cumulative_revenue_risk": npv_range,
                "risk_as_percent_of_investment": (npv_range / initial_investment) * 100
            }
        }

        return result


def calculate_cost_per_watt_benchmark(
    technology: str,
    region: str = "global"
) -> Dict[str, float]:
    """
    Provide benchmark pricing for different PV technologies.

    Args:
        technology: PV technology type
        region: Geographic region

    Returns:
        Dictionary with pricing benchmarks in different currencies
    """
    # 2024 approximate benchmarks (USD/W)
    technology_prices = {
        "PERC": 0.20,
        "TOPCon": 0.22,
        "HJT": 0.28,
        "Perovskite": 0.25,
        "Perovskite-Silicon Tandem": 0.35,
        "CIGS": 0.30,
        "CdTe": 0.25,
        "Custom": 0.22
    }

    base_price_usd = technology_prices.get(technology, 0.22)

    # Regional adjustments
    regional_factors = {
        "global": 1.0,
        "china": 0.85,
        "europe": 1.15,
        "usa": 1.10,
        "india": 0.90,
        "middle_east": 1.05
    }

    factor = regional_factors.get(region.lower(), 1.0)
    adjusted_price_usd = base_price_usd * factor

    # Convert to different currencies
    prices = {}
    for currency, data in CURRENCIES.items():
        rate = data["typical_rate_to_usd"]
        prices[currency] = adjusted_price_usd / rate

    return {
        "technology": technology,
        "region": region,
        "base_price_usd": base_price_usd,
        "adjusted_price_usd": adjusted_price_usd,
        "regional_factor": factor,
        "prices_by_currency": prices,
        "note": "Approximate 2024 module prices - for guidance only"
    }
