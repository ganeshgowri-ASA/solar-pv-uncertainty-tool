"""
Monte Carlo Simulation for PV Uncertainty Propagation
Implements Monte Carlo methods for uncertainty analysis as per GUM Supplement 1.
"""

import numpy as np
from typing import Dict, Callable, List, Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class MonteCarloInput:
    """Represents an input variable for Monte Carlo simulation."""
    name: str
    mean: float
    std_uncertainty: float
    distribution: str = "normal"  # normal, uniform, triangular, lognormal

    def sample(self, size: int, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate random samples from the distribution.

        Args:
            size: Number of samples to generate
            random_state: Random seed for reproducibility

        Returns:
            Array of samples
        """
        if random_state is not None:
            np.random.seed(random_state)

        if self.distribution == "normal":
            return np.random.normal(self.mean, self.std_uncertainty, size)

        elif self.distribution == "uniform":
            # For uniform: std = width/sqrt(12), so width = std*sqrt(12)
            half_width = self.std_uncertainty * np.sqrt(3)
            return np.random.uniform(
                self.mean - half_width,
                self.mean + half_width,
                size
            )

        elif self.distribution == "triangular":
            # For triangular: std = width/sqrt(24), so width = std*sqrt(24)
            half_width = self.std_uncertainty * np.sqrt(6)
            return np.random.triangular(
                self.mean - half_width,
                self.mean,
                self.mean + half_width,
                size
            )

        elif self.distribution == "lognormal":
            # Parameters for lognormal distribution
            # Assuming std_uncertainty is the standard deviation of the log-normal variable
            mu = np.log(self.mean**2 / np.sqrt(self.mean**2 + self.std_uncertainty**2))
            sigma = np.sqrt(np.log(1 + (self.std_uncertainty / self.mean)**2))
            return np.random.lognormal(mu, sigma, size)

        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


class MonteCarloSimulator:
    """
    Monte Carlo simulator for uncertainty propagation in PV measurements.
    """

    def __init__(self, n_samples: int = 100000, random_state: Optional[int] = None):
        """
        Initialize Monte Carlo simulator.

        Args:
            n_samples: Number of Monte Carlo samples
            random_state: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.random_state = random_state
        self.inputs: List[MonteCarloInput] = []

    def add_input(
        self,
        name: str,
        mean: float,
        std_uncertainty: float,
        distribution: str = "normal"
    ) -> None:
        """
        Add an input variable for simulation.

        Args:
            name: Variable name
            mean: Mean value
            std_uncertainty: Standard uncertainty
            distribution: Distribution type
        """
        self.inputs.append(
            MonteCarloInput(name, mean, std_uncertainty, distribution)
        )

    def run_simulation(
        self,
        model_function: Callable,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict:
        """
        Run Monte Carlo simulation.

        Args:
            model_function: Function that takes input samples and returns output
                           Should accept kwargs with input variable names
            progress_callback: Optional callback for progress updates (current, total)

        Returns:
            Dictionary with simulation results and statistics
        """
        if not self.inputs:
            raise ValueError("No input variables defined")

        # Generate samples for all inputs
        samples = {}
        for input_var in self.inputs:
            samples[input_var.name] = input_var.sample(
                self.n_samples,
                self.random_state
            )

        # Run model for all samples
        output_samples = np.zeros(self.n_samples)

        # Process in batches for progress reporting
        batch_size = max(1, self.n_samples // 100)

        for i in range(self.n_samples):
            # Prepare input dictionary for this sample
            sample_inputs = {name: samples[name][i] for name in samples}

            # Evaluate model
            try:
                output_samples[i] = model_function(**sample_inputs)
            except Exception as e:
                warnings.warn(f"Error in sample {i}: {str(e)}")
                output_samples[i] = np.nan

            # Progress callback
            if progress_callback and (i % batch_size == 0 or i == self.n_samples - 1):
                progress_callback(i + 1, self.n_samples)

        # Remove any NaN values
        valid_samples = output_samples[~np.isnan(output_samples)]

        if len(valid_samples) < self.n_samples * 0.95:
            warnings.warn(
                f"Only {len(valid_samples)}/{self.n_samples} valid samples. "
                "Check model function for errors."
            )

        # Calculate statistics
        results = self._calculate_statistics(valid_samples, samples)

        return results

    def _calculate_statistics(
        self,
        output_samples: np.ndarray,
        input_samples: Dict[str, np.ndarray]
    ) -> Dict:
        """
        Calculate statistics from Monte Carlo samples.

        Args:
            output_samples: Array of output samples
            input_samples: Dictionary of input sample arrays

        Returns:
            Dictionary with statistical results
        """
        # Output statistics
        mean = np.mean(output_samples)
        std = np.std(output_samples, ddof=1)  # Sample standard deviation
        median = np.median(output_samples)

        # Percentiles for confidence intervals
        percentiles = [0.5, 2.5, 5, 16, 50, 84, 95, 97.5, 99.5]
        percentile_values = np.percentile(output_samples, percentiles)

        # Coverage intervals
        ci_68 = (percentile_values[3], percentile_values[5])  # 16th to 84th percentile
        ci_95 = (percentile_values[1], percentile_values[7])  # 2.5th to 97.5th percentile
        ci_99 = (percentile_values[0], percentile_values[8])  # 0.5th to 99.5th percentile

        # Sensitivity analysis using correlation coefficients
        sensitivities = {}
        for name, samples in input_samples.items():
            # Pearson correlation coefficient
            if len(output_samples) == len(samples):
                valid_indices = ~np.isnan(samples)
                if np.sum(valid_indices) > 0:
                    corr = np.corrcoef(
                        samples[valid_indices],
                        output_samples[valid_indices]
                    )[0, 1]
                    sensitivities[name] = {
                        "correlation": corr,
                        "importance": abs(corr)
                    }

        # Sort sensitivities by importance
        sorted_sensitivities = dict(
            sorted(
                sensitivities.items(),
                key=lambda x: x[1]["importance"],
                reverse=True
            )
        )

        results = {
            "n_samples": len(output_samples),
            "mean": mean,
            "median": median,
            "std_uncertainty": std,
            "expanded_uncertainty_k2": std * 2.0,
            "relative_uncertainty_percent": (std / abs(mean) * 100) if mean != 0 else 0,
            "percentiles": dict(zip(percentiles, percentile_values)),
            "confidence_intervals": {
                "68_percent": ci_68,
                "95_percent": ci_95,
                "99_percent": ci_99
            },
            "sensitivities": sorted_sensitivities,
            "samples": output_samples.tolist(),
            "skewness": self._calculate_skewness(output_samples),
            "kurtosis": self._calculate_kurtosis(output_samples)
        }

        return results

    @staticmethod
    def _calculate_skewness(data: np.ndarray) -> float:
        """Calculate skewness of distribution."""
        n = len(data)
        if n < 3:
            return 0.0

        mean = np.mean(data)
        std = np.std(data, ddof=1)

        if std == 0:
            return 0.0

        m3 = np.sum((data - mean)**3) / n
        skewness = m3 / (std**3)

        return skewness

    @staticmethod
    def _calculate_kurtosis(data: np.ndarray) -> float:
        """Calculate excess kurtosis of distribution."""
        n = len(data)
        if n < 4:
            return 0.0

        mean = np.mean(data)
        std = np.std(data, ddof=1)

        if std == 0:
            return 0.0

        m4 = np.sum((data - mean)**4) / n
        kurtosis = m4 / (std**4) - 3  # Excess kurtosis

        return kurtosis


class PVMonteCarlo:
    """
    Specialized Monte Carlo simulator for PV power and performance metrics.
    """

    @staticmethod
    def simulate_power_uncertainty(
        irradiance_mean: float,
        irradiance_std: float,
        temperature_mean: float,
        temperature_std: float,
        power_meter_std: float,
        module_efficiency: float = 0.20,
        efficiency_std: float = 0.01,
        temp_coefficient: float = -0.004,
        reference_temp: float = 25.0,
        n_samples: int = 100000,
        random_state: Optional[int] = None
    ) -> Dict:
        """
        Monte Carlo simulation for PV power uncertainty.

        Args:
            irradiance_mean: Mean irradiance (W/m²)
            irradiance_std: Irradiance standard uncertainty (W/m²)
            temperature_mean: Mean temperature (°C)
            temperature_std: Temperature standard uncertainty (°C)
            power_meter_std: Power meter standard uncertainty (W)
            module_efficiency: Module efficiency (fraction)
            efficiency_std: Efficiency standard uncertainty (fraction)
            temp_coefficient: Temperature coefficient (%/°C)
            reference_temp: Reference temperature (°C)
            n_samples: Number of Monte Carlo samples
            random_state: Random seed

        Returns:
            Dictionary with simulation results
        """
        simulator = MonteCarloSimulator(n_samples, random_state)

        # Add inputs
        simulator.add_input("irradiance", irradiance_mean, irradiance_std, "normal")
        simulator.add_input("temperature", temperature_mean, temperature_std, "normal")
        simulator.add_input("efficiency", module_efficiency, efficiency_std, "normal")

        # Define power model
        def power_model(irradiance, temperature, efficiency):
            """Calculate power with temperature correction."""
            # Simplified PV power model
            # P = G * η * A * (1 + γ * (T - T_ref))
            # Assuming area normalized (A=1)
            temp_factor = 1 + temp_coefficient * (temperature - reference_temp)
            power_ideal = irradiance * efficiency * temp_factor

            # Add measurement uncertainty (not correlated with other inputs)
            power_with_meter_error = power_ideal + np.random.normal(0, power_meter_std)

            return power_with_meter_error

        # Run simulation
        results = simulator.run_simulation(power_model)

        return results

    @staticmethod
    def simulate_pr_uncertainty(
        energy_mean: float,
        energy_std: float,
        irradiation_mean: float,
        irradiation_std: float,
        capacity_mean: float,
        capacity_std: float,
        n_samples: int = 100000,
        random_state: Optional[int] = None
    ) -> Dict:
        """
        Monte Carlo simulation for Performance Ratio uncertainty.

        Args:
            energy_mean: Mean energy (kWh)
            energy_std: Energy standard uncertainty (kWh)
            irradiation_mean: Mean irradiation (kWh/m²)
            irradiation_std: Irradiation standard uncertainty (kWh/m²)
            capacity_mean: Mean installed capacity (kWp)
            capacity_std: Capacity standard uncertainty (kWp)
            n_samples: Number of Monte Carlo samples
            random_state: Random seed

        Returns:
            Dictionary with simulation results
        """
        simulator = MonteCarloSimulator(n_samples, random_state)

        # Add inputs
        simulator.add_input("energy", energy_mean, energy_std, "normal")
        simulator.add_input("irradiation", irradiation_mean, irradiation_std, "normal")
        simulator.add_input("capacity", capacity_mean, capacity_std, "normal")

        # Define PR model
        def pr_model(energy, irradiation, capacity):
            """Calculate Performance Ratio."""
            denominator = irradiation * capacity
            if denominator <= 0:
                return np.nan
            return energy / denominator

        # Run simulation
        results = simulator.run_simulation(pr_model)

        return results
