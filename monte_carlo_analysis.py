"""
Monte Carlo Uncertainty Analysis Module

Enhanced Monte Carlo simulation for bifacial PV uncertainty analysis
following JCGM 101:2008 (GUM Supplement 1).

This module provides:
- Monte Carlo propagation for complex measurement models
- Bifacial-specific models (equivalent irradiance, bifaciality factors)
- Correlation handling between input quantities
- Convergence monitoring and adaptive sampling
- Sensitivity analysis via correlation/regression methods
- Full statistical output with coverage intervals

References:
- JCGM 101:2008: GUM Supplement 1 - Monte Carlo method
- JCGM 102:2011: GUM Supplement 2 - Multivariate models
- IEC TS 60904-1-2:2024: Bifacial PV measurement
- NREL Bifacial PV uncertainty research

Author: Universal Solar Simulator Framework Team
Version: 2.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Callable, Union
from enum import Enum
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, pearsonr
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time


# =============================================================================
# Enumerations and Type Definitions
# =============================================================================

class DistributionType(Enum):
    """Probability distributions for Monte Carlo sampling"""
    NORMAL = "normal"
    RECTANGULAR = "rectangular"      # Uniform
    TRIANGULAR = "triangular"
    U_SHAPED = "u_shaped"           # Arc-sine
    LOGNORMAL = "lognormal"
    TRUNCATED_NORMAL = "truncated_normal"
    BETA = "beta"
    GAMMA = "gamma"
    CUSTOM = "custom"


class ConvergenceCriterion(Enum):
    """Convergence monitoring criteria"""
    STANDARD_DEVIATION = "std"       # Stability of standard deviation
    PERCENTILE = "percentile"        # Stability of percentiles
    COVERAGE_INTERVAL = "coverage"   # Stability of coverage interval


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MCInputVariable:
    """
    Monte Carlo input variable with distribution specification.

    Attributes:
        name: Variable identifier
        mean: Central value / expected value
        uncertainty: Standard uncertainty (k=1)
        distribution: Probability distribution
        unit: Physical unit
        bounds: Optional (min, max) bounds for truncation
        correlation: Correlation with other variables {name: correlation}
        custom_samples: Pre-generated samples (for custom distributions)
    """
    name: str
    mean: float
    uncertainty: float
    distribution: DistributionType = DistributionType.NORMAL
    unit: str = ""
    bounds: Optional[Tuple[float, float]] = None
    correlation: Dict[str, float] = field(default_factory=dict)
    custom_samples: Optional[np.ndarray] = None

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Generate n random samples from the distribution.

        Args:
            n: Number of samples
            rng: Random number generator (uses default if None)

        Returns:
            Array of n samples
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.custom_samples is not None:
            # Use custom samples with random selection if needed
            indices = rng.integers(0, len(self.custom_samples), size=n)
            return self.custom_samples[indices]

        if self.distribution == DistributionType.NORMAL:
            samples = rng.normal(self.mean, self.uncertainty, n)

        elif self.distribution == DistributionType.RECTANGULAR:
            # Uniform: half-width = uncertainty * sqrt(3)
            half_width = self.uncertainty * np.sqrt(3)
            samples = rng.uniform(self.mean - half_width, self.mean + half_width, n)

        elif self.distribution == DistributionType.TRIANGULAR:
            # Triangular: half-width = uncertainty * sqrt(6)
            half_width = self.uncertainty * np.sqrt(6)
            samples = rng.triangular(
                self.mean - half_width,
                self.mean,
                self.mean + half_width,
                n
            )

        elif self.distribution == DistributionType.U_SHAPED:
            # Arc-sine (U-shaped): half-width = uncertainty * sqrt(2)
            half_width = self.uncertainty * np.sqrt(2)
            # Use beta(0.5, 0.5) scaled
            samples = (rng.beta(0.5, 0.5, n) - 0.5) * 2 * half_width + self.mean

        elif self.distribution == DistributionType.LOGNORMAL:
            # Lognormal: use method of moments
            cv = self.uncertainty / self.mean if self.mean > 0 else 0.1
            sigma_ln = np.sqrt(np.log(1 + cv**2))
            mu_ln = np.log(self.mean) - 0.5 * sigma_ln**2
            samples = rng.lognormal(mu_ln, sigma_ln, n)

        elif self.distribution == DistributionType.TRUNCATED_NORMAL:
            # Truncated normal within bounds
            if self.bounds is None:
                samples = rng.normal(self.mean, self.uncertainty, n)
            else:
                a, b = (self.bounds[0] - self.mean) / self.uncertainty, \
                       (self.bounds[1] - self.mean) / self.uncertainty
                samples = stats.truncnorm.rvs(a, b, loc=self.mean,
                                              scale=self.uncertainty, size=n,
                                              random_state=rng)

        else:
            # Default to normal
            samples = rng.normal(self.mean, self.uncertainty, n)

        # Apply bounds if specified
        if self.bounds is not None:
            samples = np.clip(samples, self.bounds[0], self.bounds[1])

        return samples


@dataclass
class MCResult:
    """
    Monte Carlo simulation result.

    Contains full statistical analysis of the output distribution.
    """
    # Output samples (optional - can be memory intensive)
    samples: Optional[np.ndarray] = None

    # Central tendency
    mean: float = 0.0
    median: float = 0.0
    mode: float = 0.0

    # Dispersion
    std: float = 0.0
    variance: float = 0.0
    mad: float = 0.0                 # Median absolute deviation

    # Shape
    skewness: float = 0.0
    kurtosis: float = 0.0

    # Percentiles
    percentile_2_5: float = 0.0
    percentile_5: float = 0.0
    percentile_16: float = 0.0       # -1 sigma
    percentile_50: float = 0.0       # Median
    percentile_84: float = 0.0       # +1 sigma
    percentile_95: float = 0.0
    percentile_97_5: float = 0.0

    # Coverage intervals
    coverage_68: Tuple[float, float] = (0.0, 0.0)
    coverage_95: Tuple[float, float] = (0.0, 0.0)
    coverage_99: Tuple[float, float] = (0.0, 0.0)

    # Simulation metadata
    n_samples: int = 0
    n_effective: int = 0             # After any filtering
    converged: bool = False
    convergence_metric: float = 0.0
    execution_time_s: float = 0.0

    # Sensitivity analysis
    sensitivity_indices: Dict[str, float] = field(default_factory=dict)
    correlation_coefficients: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_samples(
        cls,
        samples: np.ndarray,
        store_samples: bool = False,
        execution_time: float = 0.0
    ) -> "MCResult":
        """
        Create MCResult from output samples.

        Args:
            samples: Array of output samples
            store_samples: Whether to store full samples array
            execution_time: Simulation execution time in seconds

        Returns:
            MCResult with computed statistics
        """
        samples = samples[~np.isnan(samples)]  # Remove NaNs
        n = len(samples)

        if n == 0:
            return cls(n_samples=0)

        # Compute percentiles efficiently
        percentiles = np.percentile(samples, [2.5, 5, 16, 50, 84, 95, 97.5])

        # Coverage intervals (shortest intervals containing p% of samples)
        sorted_samples = np.sort(samples)

        def shortest_coverage_interval(p: float) -> Tuple[float, float]:
            """Find shortest interval containing p% of samples"""
            alpha = 1 - p
            n_in_interval = int(np.ceil(n * p))
            widths = sorted_samples[n_in_interval:] - sorted_samples[:-n_in_interval]
            if len(widths) == 0:
                return (sorted_samples[0], sorted_samples[-1])
            min_idx = np.argmin(widths)
            return (sorted_samples[min_idx], sorted_samples[min_idx + n_in_interval])

        # Mode estimation (kernel density)
        try:
            kde = stats.gaussian_kde(samples)
            x_grid = np.linspace(samples.min(), samples.max(), 1000)
            mode = x_grid[np.argmax(kde(x_grid))]
        except Exception:
            mode = percentiles[3]  # Use median as fallback

        return cls(
            samples=samples if store_samples else None,
            mean=np.mean(samples),
            median=percentiles[3],
            mode=mode,
            std=np.std(samples, ddof=1),
            variance=np.var(samples, ddof=1),
            mad=np.median(np.abs(samples - percentiles[3])),
            skewness=stats.skew(samples),
            kurtosis=stats.kurtosis(samples),
            percentile_2_5=percentiles[0],
            percentile_5=percentiles[1],
            percentile_16=percentiles[2],
            percentile_50=percentiles[3],
            percentile_84=percentiles[4],
            percentile_95=percentiles[5],
            percentile_97_5=percentiles[6],
            coverage_68=shortest_coverage_interval(0.68),
            coverage_95=shortest_coverage_interval(0.95),
            coverage_99=shortest_coverage_interval(0.99),
            n_samples=n,
            n_effective=n,
            execution_time_s=execution_time,
        )

    @property
    def expanded_uncertainty_k2(self) -> float:
        """Expanded uncertainty at k=2 (half-width of 95% coverage)"""
        return (self.coverage_95[1] - self.coverage_95[0]) / 2

    @property
    def relative_uncertainty(self) -> float:
        """Relative standard uncertainty"""
        return self.std / self.mean if self.mean != 0 else 0.0

    def summary(self) -> str:
        """Generate text summary"""
        lines = [
            "=" * 50,
            "MONTE CARLO SIMULATION RESULTS",
            "=" * 50,
            f"Samples: {self.n_samples:,}",
            f"Execution time: {self.execution_time_s:.2f} s",
            "",
            "Central Values:",
            f"  Mean:   {self.mean:.6f}",
            f"  Median: {self.median:.6f}",
            f"  Mode:   {self.mode:.6f}",
            "",
            "Dispersion:",
            f"  Std Dev: {self.std:.6f}",
            f"  Relative: {self.relative_uncertainty*100:.2f}%",
            "",
            "Shape:",
            f"  Skewness: {self.skewness:.3f}",
            f"  Kurtosis: {self.kurtosis:.3f}",
            "",
            "Coverage Intervals:",
            f"  68%: [{self.coverage_68[0]:.4f}, {self.coverage_68[1]:.4f}]",
            f"  95%: [{self.coverage_95[0]:.4f}, {self.coverage_95[1]:.4f}]",
            f"  99%: [{self.coverage_99[0]:.4f}, {self.coverage_99[1]:.4f}]",
            "=" * 50,
        ]

        if self.sensitivity_indices:
            lines.append("\nSensitivity Indices:")
            for name, idx in sorted(self.sensitivity_indices.items(),
                                   key=lambda x: -x[1]):
                lines.append(f"  {name}: {idx:.3f}")

        return "\n".join(lines)


@dataclass
class BifacialMCInputs:
    """
    Structured input variables for bifacial Monte Carlo simulation.
    """
    # Irradiance
    g_front: MCInputVariable = field(default_factory=lambda: MCInputVariable(
        name="G_front", mean=1000.0, uncertainty=20.0,
        distribution=DistributionType.NORMAL, unit="W/m²"
    ))
    g_rear: MCInputVariable = field(default_factory=lambda: MCInputVariable(
        name="G_rear", mean=150.0, uncertainty=5.0,
        distribution=DistributionType.NORMAL, unit="W/m²"
    ))

    # Bifaciality
    phi_isc: MCInputVariable = field(default_factory=lambda: MCInputVariable(
        name="φ_Isc", mean=0.75, uncertainty=0.015,
        distribution=DistributionType.NORMAL, unit=""
    ))
    phi_pmax: MCInputVariable = field(default_factory=lambda: MCInputVariable(
        name="φ_Pmax", mean=0.70, uncertainty=0.02,
        distribution=DistributionType.NORMAL, unit=""
    ))

    # Module parameters
    pmax_front: MCInputVariable = field(default_factory=lambda: MCInputVariable(
        name="Pmax_front", mean=400.0, uncertainty=4.0,
        distribution=DistributionType.NORMAL, unit="W"
    ))
    isc_front: MCInputVariable = field(default_factory=lambda: MCInputVariable(
        name="Isc_front", mean=10.0, uncertainty=0.1,
        distribution=DistributionType.NORMAL, unit="A"
    ))
    voc_front: MCInputVariable = field(default_factory=lambda: MCInputVariable(
        name="Voc_front", mean=45.0, uncertainty=0.2,
        distribution=DistributionType.NORMAL, unit="V"
    ))

    # Temperature
    temperature: MCInputVariable = field(default_factory=lambda: MCInputVariable(
        name="T_module", mean=25.0, uncertainty=1.0,
        distribution=DistributionType.NORMAL, unit="°C"
    ))
    gamma: MCInputVariable = field(default_factory=lambda: MCInputVariable(
        name="γ_Pmax", mean=-0.35, uncertainty=0.02,
        distribution=DistributionType.NORMAL, unit="%/°C"
    ))

    # Spectral
    spectral_mismatch_front: MCInputVariable = field(default_factory=lambda: MCInputVariable(
        name="M_front", mean=1.0, uncertainty=0.01,
        distribution=DistributionType.NORMAL, unit=""
    ))
    spectral_mismatch_rear: MCInputVariable = field(default_factory=lambda: MCInputVariable(
        name="M_rear", mean=1.0, uncertainty=0.015,
        distribution=DistributionType.NORMAL, unit=""
    ))

    def to_list(self) -> List[MCInputVariable]:
        """Convert to list of input variables"""
        return [
            self.g_front, self.g_rear, self.phi_isc, self.phi_pmax,
            self.pmax_front, self.isc_front, self.voc_front,
            self.temperature, self.gamma,
            self.spectral_mismatch_front, self.spectral_mismatch_rear
        ]


# =============================================================================
# Monte Carlo Simulator Classes
# =============================================================================

class MonteCarloSimulator:
    """
    General-purpose Monte Carlo uncertainty propagation simulator.

    Implements JCGM 101:2008 methodology with:
    - Multiple distribution types
    - Correlated inputs
    - Convergence monitoring
    - Sensitivity analysis
    """

    def __init__(
        self,
        n_samples: int = 100000,
        seed: Optional[int] = None,
        parallel: bool = False,
        n_workers: int = 4
    ):
        """
        Initialize Monte Carlo simulator.

        Args:
            n_samples: Number of Monte Carlo samples
            seed: Random seed for reproducibility
            parallel: Enable parallel execution
            n_workers: Number of parallel workers
        """
        self.n_samples = n_samples
        self.seed = seed
        self.parallel = parallel
        self.n_workers = n_workers
        self.rng = np.random.default_rng(seed)

        self.inputs: Dict[str, MCInputVariable] = {}
        self.input_samples: Dict[str, np.ndarray] = {}
        self.output_samples: Optional[np.ndarray] = None
        self.result: Optional[MCResult] = None

    def add_input(self, variable: MCInputVariable) -> "MonteCarloSimulator":
        """Add an input variable"""
        self.inputs[variable.name] = variable
        return self

    def add_inputs(self, variables: List[MCInputVariable]) -> "MonteCarloSimulator":
        """Add multiple input variables"""
        for var in variables:
            self.inputs[var.name] = var
        return self

    def _generate_correlated_samples(self) -> Dict[str, np.ndarray]:
        """
        Generate samples with correlations using Cholesky decomposition.

        Returns:
            Dictionary of correlated samples for each input
        """
        n = self.n_samples
        var_names = list(self.inputs.keys())
        n_vars = len(var_names)

        if n_vars == 0:
            return {}

        # Build correlation matrix
        corr_matrix = np.eye(n_vars)
        for i, name_i in enumerate(var_names):
            for j, name_j in enumerate(var_names):
                if i != j and name_j in self.inputs[name_i].correlation:
                    rho = self.inputs[name_i].correlation[name_j]
                    corr_matrix[i, j] = rho
                    corr_matrix[j, i] = rho

        # Check if positive definite
        try:
            # Cholesky decomposition
            L = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            # Fall back to independent samples
            warnings.warn("Correlation matrix not positive definite, using independent samples")
            return {name: var.sample(n, self.rng) for name, var in self.inputs.items()}

        # Generate independent standard normal samples
        z = self.rng.standard_normal((n, n_vars))

        # Apply correlation structure
        corr_z = z @ L.T

        # Transform to desired marginal distributions
        samples = {}
        for i, name in enumerate(var_names):
            var = self.inputs[name]
            # Convert correlated standard normal to uniform via CDF
            u = stats.norm.cdf(corr_z[:, i])

            # Transform uniform to desired distribution via inverse CDF
            if var.distribution == DistributionType.NORMAL:
                samples[name] = var.mean + var.uncertainty * corr_z[:, i]

            elif var.distribution == DistributionType.RECTANGULAR:
                half_width = var.uncertainty * np.sqrt(3)
                samples[name] = var.mean + (2 * u - 1) * half_width

            elif var.distribution == DistributionType.TRIANGULAR:
                half_width = var.uncertainty * np.sqrt(6)
                samples[name] = stats.triang.ppf(u, 0.5,
                                                 loc=var.mean - half_width,
                                                 scale=2 * half_width)

            elif var.distribution == DistributionType.LOGNORMAL:
                cv = var.uncertainty / var.mean if var.mean > 0 else 0.1
                sigma_ln = np.sqrt(np.log(1 + cv**2))
                mu_ln = np.log(var.mean) - 0.5 * sigma_ln**2
                samples[name] = stats.lognorm.ppf(u, sigma_ln, scale=np.exp(mu_ln))

            else:
                # Default to normal
                samples[name] = var.mean + var.uncertainty * corr_z[:, i]

            # Apply bounds
            if var.bounds is not None:
                samples[name] = np.clip(samples[name], var.bounds[0], var.bounds[1])

        return samples

    def run(
        self,
        model: Callable[..., np.ndarray],
        store_samples: bool = False,
        convergence_check: bool = True,
        convergence_tolerance: float = 0.01
    ) -> MCResult:
        """
        Run Monte Carlo simulation.

        Args:
            model: Function that takes input sample arrays and returns output array
            store_samples: Whether to store output samples in result
            convergence_check: Enable convergence monitoring
            convergence_tolerance: Relative tolerance for convergence

        Returns:
            MCResult with full statistical analysis
        """
        start_time = time.time()

        # Generate input samples
        has_correlations = any(
            len(var.correlation) > 0 for var in self.inputs.values()
        )

        if has_correlations:
            self.input_samples = self._generate_correlated_samples()
        else:
            self.input_samples = {
                name: var.sample(self.n_samples, self.rng)
                for name, var in self.inputs.items()
            }

        # Run model
        self.output_samples = model(**self.input_samples)

        # Calculate result
        execution_time = time.time() - start_time
        self.result = MCResult.from_samples(
            self.output_samples,
            store_samples=store_samples,
            execution_time=execution_time
        )

        # Sensitivity analysis
        self._calculate_sensitivity()

        # Convergence check
        if convergence_check:
            self.result.converged, self.result.convergence_metric = \
                self._check_convergence(convergence_tolerance)

        return self.result

    def _calculate_sensitivity(self):
        """Calculate sensitivity indices using correlation method"""
        if self.result is None or self.output_samples is None:
            return

        for name, samples in self.input_samples.items():
            # Spearman correlation (rank-based, more robust)
            rho, _ = spearmanr(samples, self.output_samples)
            self.result.correlation_coefficients[name] = rho

            # Sensitivity index (squared correlation)
            self.result.sensitivity_indices[name] = rho ** 2

    def _check_convergence(
        self,
        tolerance: float,
        n_batches: int = 10
    ) -> Tuple[bool, float]:
        """
        Check convergence by comparing batch statistics.

        Args:
            tolerance: Relative tolerance
            n_batches: Number of batches to compare

        Returns:
            (converged, metric) tuple
        """
        if self.output_samples is None:
            return False, 1.0

        batch_size = len(self.output_samples) // n_batches
        if batch_size < 100:
            return True, 0.0

        batch_stds = []
        for i in range(n_batches):
            batch = self.output_samples[i * batch_size:(i + 1) * batch_size]
            batch_stds.append(np.std(batch))

        # Check stability of standard deviation
        relative_variation = np.std(batch_stds) / np.mean(batch_stds)
        converged = relative_variation < tolerance

        return converged, relative_variation


class BifacialMonteCarloSimulator(MonteCarloSimulator):
    """
    Specialized Monte Carlo simulator for bifacial PV measurements.

    Includes pre-defined models for:
    - Equivalent irradiance
    - Bifacial power
    - Bifaciality factors
    - Bifacial gain
    """

    def __init__(
        self,
        n_samples: int = 100000,
        seed: Optional[int] = None
    ):
        super().__init__(n_samples=n_samples, seed=seed)
        self.bifacial_inputs: Optional[BifacialMCInputs] = None

    def set_bifacial_inputs(
        self,
        inputs: BifacialMCInputs
    ) -> "BifacialMonteCarloSimulator":
        """Set bifacial measurement inputs"""
        self.bifacial_inputs = inputs
        for var in inputs.to_list():
            self.add_input(var)
        return self

    def simulate_equivalent_irradiance(
        self,
        store_samples: bool = False
    ) -> MCResult:
        """
        Simulate equivalent irradiance: G_eq = G_front + φ × G_rear

        Returns:
            MCResult for equivalent irradiance
        """
        def model(G_front, G_rear, **kwargs):
            phi = kwargs.get('φ_Isc', np.full_like(G_front, 0.75))
            return G_front + phi * G_rear

        return self.run(model, store_samples=store_samples)

    def simulate_bifacial_power_stc(
        self,
        reference_power: float = 400.0,
        store_samples: bool = False
    ) -> MCResult:
        """
        Simulate bifacial power corrected to STC.

        Model:
        P_STC = P_meas × (G_STC / G_eq) × [1 + γ(T_STC - T)] × M_f × M_r

        Args:
            reference_power: Reference power at STC (W)
            store_samples: Whether to store samples

        Returns:
            MCResult for STC power
        """
        g_stc = 1000.0
        t_stc = 25.0

        def model(**kwargs):
            g_front = kwargs['G_front']
            g_rear = kwargs['G_rear']
            phi = kwargs.get('φ_Isc', np.full_like(g_front, 0.75))
            temperature = kwargs['T_module']
            gamma = kwargs['γ_Pmax']
            m_front = kwargs['M_front']
            m_rear = kwargs['M_rear']

            # Equivalent irradiance
            g_eq = g_front + phi * g_rear

            # Power at measurement conditions (simplified model)
            # Assume power scales with equivalent irradiance
            p_meas = reference_power * (g_eq / g_stc)

            # Temperature correction
            temp_factor = 1 + (gamma / 100) * (t_stc - temperature)

            # Spectral correction (simplified - average of front and rear)
            m_effective = (g_front * m_front + phi * g_rear * m_rear) / g_eq

            # Corrected power
            p_stc = p_meas * (g_stc / g_eq) * temp_factor * m_effective

            return p_stc

        return self.run(model, store_samples=store_samples)

    def simulate_bifaciality_factor(
        self,
        parameter: Literal["isc", "pmax", "voc"] = "isc",
        store_samples: bool = False
    ) -> MCResult:
        """
        Simulate bifaciality factor: φ = X_rear / X_front

        Args:
            parameter: Which parameter (isc, pmax, voc)
            store_samples: Whether to store samples

        Returns:
            MCResult for bifaciality factor
        """
        def model(**kwargs):
            if parameter == "isc":
                front = kwargs['Isc_front']
                phi = kwargs['φ_Isc']
                rear = front * phi
            elif parameter == "pmax":
                front = kwargs['Pmax_front']
                phi = kwargs['φ_Pmax']
                rear = front * phi
            else:
                front = kwargs['Voc_front']
                phi = kwargs.get('φ_Voc', kwargs.get('φ_Isc', 0.75) * 0.98)
                rear = front * phi

            return rear / front

        return self.run(model, store_samples=store_samples)

    def simulate_bifacial_gain(
        self,
        front_power: float = 400.0,
        rear_irradiance_ratio: float = 0.15,
        store_samples: bool = False
    ) -> MCResult:
        """
        Simulate bifacial gain: BG = (P_bi - P_mono) / P_mono

        Args:
            front_power: Front-only power at STC (W)
            rear_irradiance_ratio: G_rear / G_front ratio
            store_samples: Whether to store samples

        Returns:
            MCResult for bifacial gain
        """
        def model(**kwargs):
            phi_pmax = kwargs['φ_Pmax']
            g_front = kwargs['G_front']
            g_rear = kwargs['G_rear']

            # Effective irradiance ratio
            ratio = g_rear / g_front

            # Simple bifacial gain model
            # BG ≈ φ_Pmax × (G_rear / G_front)
            bifacial_gain = phi_pmax * ratio

            return bifacial_gain

        return self.run(model, store_samples=store_samples)

    def full_bifacial_analysis(
        self,
        reference_power: float = 400.0,
        store_all_samples: bool = False
    ) -> Dict[str, MCResult]:
        """
        Run complete bifacial uncertainty analysis.

        Returns:
            Dictionary with results for all quantities
        """
        results = {}

        # Reset RNG for reproducibility
        self.rng = np.random.default_rng(self.seed)

        # Equivalent irradiance
        results['g_equivalent'] = self.simulate_equivalent_irradiance(store_all_samples)

        # Reset and run power simulation
        self.rng = np.random.default_rng(self.seed)
        results['power_stc'] = self.simulate_bifacial_power_stc(
            reference_power, store_all_samples
        )

        # Bifaciality factors
        for param in ['isc', 'pmax']:
            self.rng = np.random.default_rng(self.seed)
            results[f'phi_{param}'] = self.simulate_bifaciality_factor(
                param, store_all_samples
            )

        # Bifacial gain
        self.rng = np.random.default_rng(self.seed)
        results['bifacial_gain'] = self.simulate_bifacial_gain(
            reference_power, store_samples=store_all_samples
        )

        return results


# =============================================================================
# Adaptive Monte Carlo
# =============================================================================

class AdaptiveMonteCarloSimulator(MonteCarloSimulator):
    """
    Adaptive Monte Carlo with automatic sample size determination.

    Per JCGM 101:2008 Section 7.9
    """

    def __init__(
        self,
        initial_samples: int = 10000,
        max_samples: int = 1000000,
        target_stability: float = 0.01,
        seed: Optional[int] = None
    ):
        """
        Initialize adaptive simulator.

        Args:
            initial_samples: Initial number of samples
            max_samples: Maximum samples before stopping
            target_stability: Target stability for coverage interval (0.01 = 1%)
            seed: Random seed
        """
        super().__init__(n_samples=initial_samples, seed=seed)
        self.max_samples = max_samples
        self.target_stability = target_stability
        self.sample_history: List[int] = []
        self.stability_history: List[float] = []

    def run_adaptive(
        self,
        model: Callable[..., np.ndarray],
        batch_size: int = 10000,
        max_iterations: int = 100,
        store_samples: bool = False
    ) -> MCResult:
        """
        Run adaptive Monte Carlo simulation.

        Increases sample size until coverage interval is stable.

        Args:
            model: Model function
            batch_size: Samples to add per iteration
            max_iterations: Maximum iterations
            store_samples: Store final samples

        Returns:
            MCResult when converged
        """
        start_time = time.time()

        # Initial run
        result = self.run(model, store_samples=True)
        all_samples = list(self.output_samples)
        total_samples = len(all_samples)

        self.sample_history.append(total_samples)
        self.stability_history.append(1.0)

        prev_coverage_95 = result.coverage_95

        for iteration in range(max_iterations):
            if total_samples >= self.max_samples:
                break

            # Generate more samples
            self.n_samples = batch_size

            # Generate new input samples
            has_correlations = any(
                len(var.correlation) > 0 for var in self.inputs.values()
            )
            if has_correlations:
                new_input_samples = self._generate_correlated_samples()
            else:
                new_input_samples = {
                    name: var.sample(batch_size, self.rng)
                    for name, var in self.inputs.items()
                }

            # Run model on new samples
            new_output_samples = model(**new_input_samples)
            all_samples.extend(new_output_samples.tolist())
            total_samples = len(all_samples)

            # Calculate new coverage interval
            samples_array = np.array(all_samples)
            new_result = MCResult.from_samples(samples_array, store_samples=False)
            new_coverage_95 = new_result.coverage_95

            # Calculate stability metric
            lower_change = abs(new_coverage_95[0] - prev_coverage_95[0])
            upper_change = abs(new_coverage_95[1] - prev_coverage_95[1])
            width = new_coverage_95[1] - new_coverage_95[0]

            if width > 0:
                stability = max(lower_change, upper_change) / width
            else:
                stability = 0.0

            self.sample_history.append(total_samples)
            self.stability_history.append(stability)

            # Check convergence
            if stability < self.target_stability:
                break

            prev_coverage_95 = new_coverage_95

        # Final result
        execution_time = time.time() - start_time
        samples_array = np.array(all_samples)

        self.output_samples = samples_array
        self.n_samples = total_samples

        result = MCResult.from_samples(
            samples_array,
            store_samples=store_samples,
            execution_time=execution_time
        )
        result.converged = self.stability_history[-1] < self.target_stability
        result.convergence_metric = self.stability_history[-1]

        # Sensitivity analysis
        self._calculate_sensitivity()
        result.sensitivity_indices = self.result.sensitivity_indices if self.result else {}
        result.correlation_coefficients = self.result.correlation_coefficients if self.result else {}

        self.result = result
        return result


# =============================================================================
# Comparison Functions
# =============================================================================

def compare_gum_monte_carlo(
    gum_uncertainty: float,
    mc_result: MCResult,
    significance_level: float = 0.05
) -> Dict[str, any]:
    """
    Compare GUM and Monte Carlo uncertainty results.

    Per JCGM 101:2008 Section 8.

    Args:
        gum_uncertainty: GUM combined standard uncertainty
        mc_result: Monte Carlo result
        significance_level: Significance level for comparison

    Returns:
        Comparison results dictionary
    """
    # GUM expanded uncertainty (k=2)
    gum_expanded = 2.0 * gum_uncertainty
    gum_interval = (-gum_expanded, gum_expanded)  # Symmetric

    # MC coverage interval
    mc_interval = mc_result.coverage_95

    # Check if MC is within GUM bounds
    mc_width = mc_interval[1] - mc_interval[0]
    gum_width = 2 * gum_expanded

    # Numerical tolerance (per JCGM 101:2008)
    # δ = 0.5 × n_dig × 10^(-n_dig) × u where n_dig is significant digits
    n_dig = 2
    tolerance = 0.5 * n_dig * 10**(-n_dig) * gum_uncertainty

    compatible = abs(mc_result.std - gum_uncertainty) < tolerance

    return {
        "gum_std_uncertainty": gum_uncertainty,
        "mc_std_uncertainty": mc_result.std,
        "gum_expanded_k2": gum_expanded,
        "mc_expanded_k2": mc_result.expanded_uncertainty_k2,
        "gum_interval": gum_interval,
        "mc_interval": mc_interval,
        "gum_width": gum_width,
        "mc_width": mc_width,
        "difference_std": abs(mc_result.std - gum_uncertainty),
        "difference_rel": abs(mc_result.std - gum_uncertainty) / gum_uncertainty * 100,
        "tolerance": tolerance,
        "compatible": compatible,
        "mc_skewness": mc_result.skewness,
        "mc_kurtosis": mc_result.kurtosis,
        "recommendation": "GUM adequate" if compatible else "Use Monte Carlo result"
    }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MONTE CARLO BIFACIAL UNCERTAINTY ANALYSIS")
    print("=" * 60)

    # Create bifacial inputs
    inputs = BifacialMCInputs()
    inputs.g_front = MCInputVariable("G_front", 900.0, 18.0, DistributionType.NORMAL, "W/m²")
    inputs.g_rear = MCInputVariable("G_rear", 135.0, 5.0, DistributionType.NORMAL, "W/m²")
    inputs.phi_isc = MCInputVariable("φ_Isc", 0.75, 0.015, DistributionType.NORMAL, "")
    inputs.phi_pmax = MCInputVariable("φ_Pmax", 0.70, 0.02, DistributionType.NORMAL, "")
    inputs.temperature = MCInputVariable("T_module", 25.0, 1.0, DistributionType.NORMAL, "°C")
    inputs.gamma = MCInputVariable("γ_Pmax", -0.35, 0.02, DistributionType.NORMAL, "%/°C")
    inputs.spectral_mismatch_front = MCInputVariable("M_front", 1.0, 0.01, DistributionType.NORMAL, "")
    inputs.spectral_mismatch_rear = MCInputVariable("M_rear", 1.0, 0.015, DistributionType.NORMAL, "")
    inputs.pmax_front = MCInputVariable("Pmax_front", 400.0, 4.0, DistributionType.NORMAL, "W")
    inputs.isc_front = MCInputVariable("Isc_front", 10.0, 0.1, DistributionType.NORMAL, "A")
    inputs.voc_front = MCInputVariable("Voc_front", 45.0, 0.2, DistributionType.NORMAL, "V")

    # Create simulator
    sim = BifacialMonteCarloSimulator(n_samples=100000, seed=42)
    sim.set_bifacial_inputs(inputs)

    # Run full analysis
    print("\nRunning Monte Carlo simulations...")
    results = sim.full_bifacial_analysis(reference_power=400.0)

    # Print results
    print("\n" + "=" * 60)
    print("EQUIVALENT IRRADIANCE")
    print("=" * 60)
    print(results['g_equivalent'].summary())

    print("\n" + "=" * 60)
    print("STC POWER")
    print("=" * 60)
    print(results['power_stc'].summary())

    print("\n" + "=" * 60)
    print("BIFACIAL GAIN")
    print("=" * 60)
    print(results['bifacial_gain'].summary())

    # Compare with GUM
    print("\n" + "=" * 60)
    print("GUM vs MONTE CARLO COMPARISON")
    print("=" * 60)

    # Approximate GUM uncertainty for power
    gum_power_uncertainty = 400.0 * 0.025  # 2.5% relative

    comparison = compare_gum_monte_carlo(gum_power_uncertainty, results['power_stc'])
    print(f"GUM std uncertainty: {comparison['gum_std_uncertainty']:.3f} W")
    print(f"MC std uncertainty:  {comparison['mc_std_uncertainty']:.3f} W")
    print(f"Difference: {comparison['difference_rel']:.2f}%")
    print(f"Compatible: {comparison['compatible']}")
    print(f"Recommendation: {comparison['recommendation']}")

    # Adaptive simulation example
    print("\n" + "=" * 60)
    print("ADAPTIVE MONTE CARLO")
    print("=" * 60)

    adaptive_sim = AdaptiveMonteCarloSimulator(
        initial_samples=5000,
        max_samples=200000,
        target_stability=0.005,
        seed=42
    )

    for var in inputs.to_list():
        adaptive_sim.add_input(var)

    def simple_model(**kwargs):
        g_front = kwargs['G_front']
        g_rear = kwargs['G_rear']
        phi = kwargs['φ_Isc']
        return g_front + phi * g_rear

    adaptive_result = adaptive_sim.run_adaptive(
        simple_model,
        batch_size=5000,
        max_iterations=50
    )

    print(f"Final samples: {adaptive_result.n_samples:,}")
    print(f"Converged: {adaptive_result.converged}")
    print(f"Final stability: {adaptive_result.convergence_metric:.4f}")
    print(f"95% coverage: [{adaptive_result.coverage_95[0]:.2f}, {adaptive_result.coverage_95[1]:.2f}]")
