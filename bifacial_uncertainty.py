"""
Bifacial PV Module Uncertainty Calculator

Comprehensive implementation of bifacial measurement uncertainty per IEC TS 60904-1-2:2024
with full GUM (JCGM 100:2008) methodology support.

This module provides:
- Bifaciality factor calculation with uncertainty
- Equivalent irradiance models
- Rear-side irradiance uncertainty
- Spectral albedo effects
- Parasitic reflection corrections
- Bifacial gain uncertainty

References:
- IEC TS 60904-1-2:2024: Measurement of bifacial PV devices
- IEC 60904-1:2020: PV device characterization
- IEC 60904-9:2020: Solar simulator classification
- JCGM 100:2008: Guide to Expression of Uncertainty in Measurement
- NREL Bifacial PV Uncertainty Research

Author: Universal Solar Simulator Framework Team
Version: 2.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union
from enum import Enum
import numpy as np
from scipy import stats
import warnings


# =============================================================================
# Enumerations and Type Definitions
# =============================================================================

class BifacialMode(Enum):
    """Measurement modes per IEC TS 60904-1-2:2024"""
    SINGLE_FRONT = "single_front"          # Front-only illumination
    SINGLE_REAR = "single_rear"            # Rear-only illumination
    DOUBLE_SIDED = "double_sided"          # Simultaneous front+rear
    EQUIVALENT_STC = "equivalent_stc"      # Adjusted for G_eq = 1000 W/m²


class CellTechnology(Enum):
    """PV cell technologies with bifacial capability"""
    PERC = "perc"
    TOPCON = "topcon"
    HJT = "hjt"                            # Heterojunction
    PEROVSKITE_SI = "perovskite_si"        # Tandem
    CIGS = "cigs"
    CDTE = "cdte"
    CUSTOM = "custom"


class AlbedoType(Enum):
    """Common ground albedo surfaces"""
    WHITE_SAND = "white_sand"
    GREEN_GRASS = "green_grass"
    CONCRETE = "concrete"
    SNOW = "snow"
    DARK_SOIL = "dark_soil"
    WHITE_ROOF = "white_roof"
    CUSTOM = "custom"


class DistributionType(Enum):
    """Probability distributions for uncertainty components"""
    NORMAL = "normal"
    RECTANGULAR = "rectangular"       # Uniform
    TRIANGULAR = "triangular"
    U_SHAPED = "u_shaped"            # Arc-sine
    LOGNORMAL = "lognormal"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SpectralResponse:
    """Spectral response data for a PV device"""
    wavelength_nm: np.ndarray        # Wavelength array (nm)
    response: np.ndarray             # Relative spectral response (0-1)
    response_type: str = "relative"  # "relative" or "absolute" (A/W)

    def interpolate(self, wavelength: float) -> float:
        """Interpolate response at given wavelength"""
        return np.interp(wavelength, self.wavelength_nm, self.response)

    def integrate(self, spectrum: np.ndarray, wavelength: np.ndarray) -> float:
        """Integrate response with given spectrum"""
        sr_interp = np.interp(wavelength, self.wavelength_nm, self.response)
        return np.trapz(spectrum * sr_interp, wavelength)


@dataclass
class AlbedoSpectrum:
    """Spectral albedo reflectance data"""
    surface_type: AlbedoType
    wavelength_nm: np.ndarray
    reflectance: np.ndarray          # 0-1 reflectance values
    average_reflectance: float
    uncertainty: float               # Relative uncertainty on reflectance

    @classmethod
    def from_type(cls, albedo_type: AlbedoType) -> "AlbedoSpectrum":
        """Create standard albedo spectrum from type"""
        # Standard wavelength range (300-1200 nm)
        wavelength = np.linspace(300, 1200, 91)

        albedo_params = {
            AlbedoType.WHITE_SAND: (0.40, 0.05, "flat"),
            AlbedoType.GREEN_GRASS: (0.20, 0.10, "red_edge"),
            AlbedoType.CONCRETE: (0.25, 0.08, "slightly_blue"),
            AlbedoType.SNOW: (0.85, 0.03, "high_uv"),
            AlbedoType.DARK_SOIL: (0.10, 0.15, "brown"),
            AlbedoType.WHITE_ROOF: (0.75, 0.05, "high_nir"),
        }

        avg_refl, uncertainty, shape = albedo_params.get(
            albedo_type, (0.25, 0.10, "flat")
        )

        # Generate spectral shape
        if shape == "flat":
            reflectance = np.ones_like(wavelength) * avg_refl
        elif shape == "red_edge":
            # Vegetation red edge around 700 nm
            reflectance = avg_refl * (1 + 0.5 * np.tanh((wavelength - 700) / 50))
        elif shape == "slightly_blue":
            reflectance = avg_refl * (1 + 0.1 * (1 - wavelength / 1200))
        elif shape == "high_uv":
            reflectance = avg_refl * np.ones_like(wavelength)
            reflectance[wavelength < 400] *= 1.1
        elif shape == "brown":
            reflectance = avg_refl * (0.8 + 0.4 * wavelength / 1200)
        elif shape == "high_nir":
            reflectance = avg_refl * (0.9 + 0.2 * wavelength / 1200)
        else:
            reflectance = np.ones_like(wavelength) * avg_refl

        return cls(
            surface_type=albedo_type,
            wavelength_nm=wavelength,
            reflectance=reflectance,
            average_reflectance=avg_refl,
            uncertainty=uncertainty
        )


@dataclass
class BifacialityFactors:
    """Bifaciality factors with uncertainties"""
    phi_isc: float                   # I_sc,rear / I_sc,front
    phi_voc: float                   # V_oc,rear / V_oc,front
    phi_pmax: float                  # P_max,rear / P_max,front
    phi_ff: float                    # FF_rear / FF_front

    u_phi_isc: float                 # Uncertainty on φ_Isc (absolute)
    u_phi_voc: float                 # Uncertainty on φ_Voc
    u_phi_pmax: float                # Uncertainty on φ_Pmax
    u_phi_ff: float                  # Uncertainty on φ_FF

    @property
    def u_phi_isc_rel(self) -> float:
        """Relative uncertainty on φ_Isc"""
        return self.u_phi_isc / self.phi_isc if self.phi_isc > 0 else 0.0

    @property
    def u_phi_pmax_rel(self) -> float:
        """Relative uncertainty on φ_Pmax"""
        return self.u_phi_pmax / self.phi_pmax if self.phi_pmax > 0 else 0.0


@dataclass
class BifacialModule:
    """Complete bifacial PV module specification"""
    # Identification
    manufacturer: str = ""
    model: str = ""
    serial_number: str = ""
    cell_technology: CellTechnology = CellTechnology.TOPCON

    # Physical dimensions
    length_mm: float = 2000.0
    width_mm: float = 1000.0
    cell_count: int = 72
    cell_rows: int = 6
    cell_columns: int = 12

    # Front side parameters (STC)
    isc_front: float = 10.0          # A
    voc_front: float = 45.0          # V
    pmax_front: float = 400.0        # W
    vmp_front: float = 38.0          # V
    imp_front: float = 10.5          # A

    # Rear side parameters (STC)
    isc_rear: float = 7.5            # A (default φ_Isc = 0.75)
    voc_rear: float = 44.0           # V
    pmax_rear: float = 280.0         # W
    vmp_rear: float = 37.0           # V
    imp_rear: float = 7.6            # A

    # Temperature coefficients
    alpha_isc: float = 0.05          # %/°C
    beta_voc: float = -0.28          # %/°C
    gamma_pmax: float = -0.34        # %/°C

    # Spectral response (optional)
    spectral_response_front: Optional[SpectralResponse] = None
    spectral_response_rear: Optional[SpectralResponse] = None

    @property
    def ff_front(self) -> float:
        """Fill factor - front side"""
        return self.pmax_front / (self.isc_front * self.voc_front)

    @property
    def ff_rear(self) -> float:
        """Fill factor - rear side"""
        return self.pmax_rear / (self.isc_rear * self.voc_rear)

    @property
    def phi_isc(self) -> float:
        """Bifaciality factor for Isc"""
        return self.isc_rear / self.isc_front

    @property
    def phi_voc(self) -> float:
        """Bifaciality factor for Voc"""
        return self.voc_rear / self.voc_front

    @property
    def phi_pmax(self) -> float:
        """Bifaciality factor for Pmax"""
        return self.pmax_rear / self.pmax_front

    @property
    def phi_ff(self) -> float:
        """Bifaciality factor for FF"""
        return self.ff_rear / self.ff_front

    @property
    def area_m2(self) -> float:
        """Module area in square meters"""
        return (self.length_mm * self.width_mm) / 1e6


@dataclass
class IrradianceConditions:
    """Irradiance measurement conditions"""
    g_front: float = 1000.0          # W/m² front irradiance
    g_rear: float = 0.0              # W/m² rear irradiance
    g_parasitic_front: float = 0.0   # W/m² parasitic on front
    g_parasitic_rear: float = 0.0    # W/m² parasitic on rear

    u_g_front: float = 20.0          # W/m² uncertainty
    u_g_rear: float = 0.0            # W/m² uncertainty
    u_g_parasitic: float = 0.0       # W/m² parasitic uncertainty

    @property
    def rear_ratio(self) -> float:
        """G_rear / G_front ratio"""
        return self.g_rear / self.g_front if self.g_front > 0 else 0.0


@dataclass
class TemperatureConditions:
    """Temperature measurement conditions"""
    t_module: float = 25.0           # °C module temperature
    t_ambient: float = 25.0          # °C ambient temperature
    t_reference: float = 25.0        # °C reference (STC)

    u_t_module: float = 1.0          # °C uncertainty
    u_t_gradient: float = 0.5        # °C front-rear gradient


@dataclass
class EquivalentIrradianceResult:
    """Result of equivalent irradiance calculation"""
    g_equivalent: float              # W/m² equivalent irradiance
    u_g_equivalent: float            # W/m² uncertainty
    u_g_equivalent_rel: float        # Relative uncertainty

    # Contribution breakdown
    contribution_g_front: float      # Variance contribution from G_front
    contribution_g_rear: float       # Variance contribution from G_rear
    contribution_phi: float          # Variance contribution from φ


@dataclass
class BifacialGainResult:
    """Result of bifacial gain calculation"""
    bifacial_gain: float             # Fractional gain (e.g., 0.10 = 10%)
    bifacial_gain_percent: float     # Percentage (e.g., 10%)
    u_bifacial_gain: float           # Absolute uncertainty
    u_bifacial_gain_rel: float       # Relative uncertainty

    p_bifacial: float                # W bifacial power
    p_monofacial: float              # W monofacial power


@dataclass
class PowerUncertaintyResult:
    """Complete power uncertainty result"""
    # Measured/corrected power
    power: float                     # W
    power_stc: float                 # W corrected to STC

    # Uncertainties
    u_combined: float                # W combined standard uncertainty
    u_expanded: float                # W expanded uncertainty (k=2)
    coverage_factor: float           # k
    confidence_level: float          # % (e.g., 95.45)

    # Relative uncertainties
    u_combined_rel: float            # Relative combined (%)
    u_expanded_rel: float            # Relative expanded (%)

    # Degrees of freedom
    dof_effective: float             # Effective degrees of freedom

    # Component contributions (variance fractions)
    contributions: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# Main Calculator Classes
# =============================================================================

class BifacialUncertaintyCalculator:
    """
    Main calculator for bifacial PV module uncertainty analysis.

    Implements IEC TS 60904-1-2:2024 measurement methodology with
    complete GUM uncertainty propagation.
    """

    def __init__(
        self,
        module: Optional[BifacialModule] = None,
        correlation_coefficient: float = 0.5
    ):
        """
        Initialize bifacial uncertainty calculator.

        Args:
            module: BifacialModule specification (can be set later)
            correlation_coefficient: Correlation between front/rear measurements
                                   (0.0 = independent, 1.0 = fully correlated)
        """
        self.module = module or BifacialModule()
        self.correlation_coefficient = correlation_coefficient
        self._uncertainty_components: Dict[str, Dict] = {}

    def calculate_bifaciality_factors(
        self,
        front_isc: float,
        front_voc: float,
        front_pmax: float,
        front_ff: float,
        rear_isc: float,
        rear_voc: float,
        rear_pmax: float,
        rear_ff: float,
        u_front_rel: float = 0.02,    # 2% relative uncertainty
        u_rear_rel: float = 0.03,     # 3% relative uncertainty
        include_correlation: bool = True
    ) -> BifacialityFactors:
        """
        Calculate bifaciality factors from front and rear measurements.

        Per IEC TS 60904-1-2:2024 Section 7:
        φ_X = X_rear / X_front

        Args:
            front_isc, front_voc, front_pmax, front_ff: Front parameters
            rear_isc, rear_voc, rear_pmax, rear_ff: Rear parameters
            u_front_rel: Relative uncertainty on front measurements (k=1)
            u_rear_rel: Relative uncertainty on rear measurements (k=1)
            include_correlation: Include correlation effects

        Returns:
            BifacialityFactors with calculated values and uncertainties
        """
        # Calculate bifaciality factors
        phi_isc = rear_isc / front_isc
        phi_voc = rear_voc / front_voc
        phi_pmax = rear_pmax / front_pmax
        phi_ff = rear_ff / front_ff

        # Calculate uncertainties using GUM
        # u(φ)/φ = sqrt[(u_r/X_r)² + (u_f/X_f)² - 2ρ(u_r/X_r)(u_f/X_f)]
        rho = self.correlation_coefficient if include_correlation else 0.0

        def calc_phi_uncertainty(phi: float, u_f_rel: float, u_r_rel: float) -> float:
            variance = u_r_rel**2 + u_f_rel**2 - 2 * rho * u_r_rel * u_f_rel
            return phi * np.sqrt(max(0, variance))

        u_phi_isc = calc_phi_uncertainty(phi_isc, u_front_rel, u_rear_rel)
        u_phi_voc = calc_phi_uncertainty(phi_voc, u_front_rel * 0.5, u_rear_rel * 0.5)
        u_phi_pmax = calc_phi_uncertainty(phi_pmax, u_front_rel, u_rear_rel)
        u_phi_ff = calc_phi_uncertainty(phi_ff, u_front_rel * 0.7, u_rear_rel * 0.7)

        return BifacialityFactors(
            phi_isc=phi_isc,
            phi_voc=phi_voc,
            phi_pmax=phi_pmax,
            phi_ff=phi_ff,
            u_phi_isc=u_phi_isc,
            u_phi_voc=u_phi_voc,
            u_phi_pmax=u_phi_pmax,
            u_phi_ff=u_phi_ff
        )

    def calculate_equivalent_irradiance(
        self,
        g_front: float,
        g_rear: float,
        phi: float,
        u_g_front: float,
        u_g_rear: float,
        u_phi: float,
        phi_type: Literal["isc", "pmax", "voc"] = "isc"
    ) -> EquivalentIrradianceResult:
        """
        Calculate equivalent irradiance per IEC TS 60904-1-2:2024.

        G_eq = G_front + φ × G_rear

        Args:
            g_front: Front irradiance (W/m²)
            g_rear: Rear irradiance (W/m²)
            phi: Bifaciality factor
            u_g_front: Front irradiance uncertainty (W/m²)
            u_g_rear: Rear irradiance uncertainty (W/m²)
            u_phi: Bifaciality factor uncertainty (absolute)
            phi_type: Which bifaciality factor is used

        Returns:
            EquivalentIrradianceResult with G_eq and uncertainty
        """
        # Calculate equivalent irradiance
        g_eq = g_front + phi * g_rear

        # Sensitivity coefficients
        c_g_front = 1.0
        c_g_rear = phi
        c_phi = g_rear

        # Variance contributions
        var_g_front = (c_g_front * u_g_front) ** 2
        var_g_rear = (c_g_rear * u_g_rear) ** 2
        var_phi = (c_phi * u_phi) ** 2

        # Combined variance
        var_total = var_g_front + var_g_rear + var_phi
        u_g_eq = np.sqrt(var_total)

        # Relative contributions
        total_var = var_total if var_total > 0 else 1.0

        return EquivalentIrradianceResult(
            g_equivalent=g_eq,
            u_g_equivalent=u_g_eq,
            u_g_equivalent_rel=u_g_eq / g_eq if g_eq > 0 else 0.0,
            contribution_g_front=var_g_front / total_var,
            contribution_g_rear=var_g_rear / total_var,
            contribution_phi=var_phi / total_var
        )

    def calculate_required_irradiances(
        self,
        g_eq_target: float,
        rear_ratio: float,
        phi: float
    ) -> Tuple[float, float]:
        """
        Calculate required front and rear irradiances for target G_eq.

        For equivalent STC measurement:
        G_front = G_eq,target / (1 + φ × R)
        G_rear = R × G_front

        Args:
            g_eq_target: Target equivalent irradiance (typically 1000 W/m²)
            rear_ratio: Desired G_rear/G_front ratio
            phi: Bifaciality factor

        Returns:
            (G_front, G_rear) required irradiances
        """
        g_front = g_eq_target / (1 + phi * rear_ratio)
        g_rear = rear_ratio * g_front

        return g_front, g_rear

    def calculate_rear_irradiance_uncertainty(
        self,
        g_rear: float,
        u_calibration: float = 0.015,     # 1.5% calibration
        u_uniformity: float = 0.03,       # 3% uniformity
        u_temporal: float = 0.005,        # 0.5% temporal
        u_positioning: float = 0.01,      # 1% positioning
        u_parasitic: float = 0.003,       # 0.3% parasitic
        distribution: DistributionType = DistributionType.RECTANGULAR
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate rear-side irradiance uncertainty.

        Args:
            g_rear: Rear irradiance (W/m²)
            u_calibration: Relative calibration uncertainty
            u_uniformity: Relative uniformity uncertainty
            u_temporal: Relative temporal uncertainty
            u_positioning: Relative positioning uncertainty
            u_parasitic: Relative parasitic light uncertainty
            distribution: Assumed distribution for Type B components

        Returns:
            (u_total, contributions_dict) - Total uncertainty and breakdown
        """
        # Convert to standard uncertainties based on distribution
        divisor = {
            DistributionType.NORMAL: 1.0,
            DistributionType.RECTANGULAR: np.sqrt(3),
            DistributionType.TRIANGULAR: np.sqrt(6),
            DistributionType.U_SHAPED: np.sqrt(2),
        }.get(distribution, np.sqrt(3))

        # Standard uncertainties
        u_cal_std = u_calibration / 1.0       # Already k=1
        u_unif_std = u_uniformity / divisor
        u_temp_std = u_temporal / divisor
        u_pos_std = u_positioning / divisor
        u_para_std = u_parasitic / divisor

        # Combine in quadrature
        u_total_rel = np.sqrt(
            u_cal_std**2 + u_unif_std**2 + u_temp_std**2 +
            u_pos_std**2 + u_para_std**2
        )

        contributions = {
            "calibration": (u_cal_std / u_total_rel)**2 if u_total_rel > 0 else 0,
            "uniformity": (u_unif_std / u_total_rel)**2 if u_total_rel > 0 else 0,
            "temporal": (u_temp_std / u_total_rel)**2 if u_total_rel > 0 else 0,
            "positioning": (u_pos_std / u_total_rel)**2 if u_total_rel > 0 else 0,
            "parasitic": (u_para_std / u_total_rel)**2 if u_total_rel > 0 else 0,
        }

        return g_rear * u_total_rel, contributions

    def calculate_spectral_albedo_uncertainty(
        self,
        albedo: AlbedoSpectrum,
        sr_module: Optional[SpectralResponse] = None,
        sr_reference: Optional[SpectralResponse] = None,
        am15g_spectrum: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        Calculate spectral mismatch factor for rear side with albedo.

        M_rear = [∫E×ρ×SR_DUT dλ × ∫E×SR_ref dλ] / [∫E×ρ×SR_ref dλ × ∫E×SR_DUT dλ]

        Args:
            albedo: AlbedoSpectrum object
            sr_module: Module spectral response (optional)
            sr_reference: Reference device spectral response (optional)
            am15g_spectrum: AM1.5G spectrum (optional, uses default if None)

        Returns:
            (M_rear, u_M_rear) - Spectral mismatch factor and uncertainty
        """
        # If no spectral data, return unity with typical uncertainty
        if sr_module is None or sr_reference is None:
            return 1.0, 0.015  # 1.5% uncertainty assumed

        wavelength = albedo.wavelength_nm

        # Default AM1.5G spectrum approximation
        if am15g_spectrum is None:
            # Simplified AM1.5G shape (peak around 500nm)
            am15g_spectrum = np.exp(-((wavelength - 500) / 300)**2)
            am15g_spectrum /= np.max(am15g_spectrum)

        # Albedo-modified spectrum
        e_albedo = am15g_spectrum * albedo.reflectance

        # Calculate integrals
        sr_dut = np.array([sr_module.interpolate(w) for w in wavelength])
        sr_ref = np.array([sr_reference.interpolate(w) for w in wavelength])

        # Numerator terms
        int_e_albedo_dut = np.trapz(e_albedo * sr_dut, wavelength)
        int_e_ref = np.trapz(am15g_spectrum * sr_ref, wavelength)

        # Denominator terms
        int_e_albedo_ref = np.trapz(e_albedo * sr_ref, wavelength)
        int_e_dut = np.trapz(am15g_spectrum * sr_dut, wavelength)

        # Mismatch factor
        if int_e_albedo_ref > 0 and int_e_dut > 0:
            m_rear = (int_e_albedo_dut * int_e_ref) / (int_e_albedo_ref * int_e_dut)
        else:
            m_rear = 1.0

        # Uncertainty (combination of albedo and SR uncertainties)
        u_m_rear = np.sqrt(albedo.uncertainty**2 + 0.01**2)  # Add 1% SR uncertainty

        return m_rear, u_m_rear

    def calculate_parasitic_reflection(
        self,
        g_front: float,
        fixture_reflectance: float = 0.05,
        view_factor: float = 0.1,
        u_reflectance_rel: float = 0.5
    ) -> Tuple[float, float]:
        """
        Calculate parasitic light reaching rear from fixture reflections.

        G_parasitic = G_front × ρ_fixture × F_view

        Args:
            g_front: Front irradiance (W/m²)
            fixture_reflectance: Fixture surface reflectance (0-1)
            view_factor: Geometric view factor
            u_reflectance_rel: Relative uncertainty on reflectance (0-1)

        Returns:
            (G_parasitic, u_G_parasitic) - Parasitic irradiance and uncertainty
        """
        g_parasitic = g_front * fixture_reflectance * view_factor

        # Uncertainty (dominantly from reflectance knowledge)
        u_g_parasitic = g_parasitic * u_reflectance_rel

        return g_parasitic, u_g_parasitic

    def calculate_bifacial_gain(
        self,
        p_bifacial: float,
        p_monofacial: float,
        u_p_bifacial: float,
        u_p_monofacial: float,
        correlation: float = 0.0
    ) -> BifacialGainResult:
        """
        Calculate bifacial power gain and its uncertainty.

        BG = (P_bifacial - P_mono) / P_mono = P_bifacial/P_mono - 1

        Args:
            p_bifacial: Power under dual illumination (W)
            p_monofacial: Power under front-only illumination (W)
            u_p_bifacial: Uncertainty on bifacial power (W)
            u_p_monofacial: Uncertainty on monofacial power (W)
            correlation: Correlation between measurements

        Returns:
            BifacialGainResult with gain and uncertainty
        """
        if p_monofacial <= 0:
            raise ValueError("Monofacial power must be positive")

        bg = (p_bifacial - p_monofacial) / p_monofacial

        # Sensitivity coefficients
        c_bi = 1.0 / p_monofacial
        c_mono = -p_bifacial / (p_monofacial ** 2)

        # Combined variance with correlation
        var_bg = (
            (c_bi * u_p_bifacial)**2 +
            (c_mono * u_p_monofacial)**2 +
            2 * correlation * c_bi * c_mono * u_p_bifacial * u_p_monofacial
        )
        u_bg = np.sqrt(max(0, var_bg))

        return BifacialGainResult(
            bifacial_gain=bg,
            bifacial_gain_percent=bg * 100,
            u_bifacial_gain=u_bg,
            u_bifacial_gain_rel=u_bg / bg if bg > 0 else 0,
            p_bifacial=p_bifacial,
            p_monofacial=p_monofacial
        )

    def calculate_combined_power_uncertainty(
        self,
        mode: BifacialMode,
        irradiance: IrradianceConditions,
        temperature: TemperatureConditions,
        power_measured: float,
        u_components: Dict[str, float],
        coverage_factor: float = 2.0
    ) -> PowerUncertaintyResult:
        """
        Calculate complete power uncertainty for bifacial measurement.

        Implements full GUM propagation for all uncertainty components.

        Args:
            mode: Measurement mode (single/double/equivalent)
            irradiance: Irradiance conditions and uncertainties
            temperature: Temperature conditions and uncertainties
            power_measured: Measured power (W)
            u_components: Dictionary of uncertainty components (relative, k=1)
                Required keys depend on mode:
                - "reference_cal": Reference device calibration
                - "reference_stability": Reference stability
                - "uniformity_front": Front spatial uniformity
                - "uniformity_rear": Rear spatial uniformity (bifacial)
                - "temporal_front": Front temporal instability
                - "temporal_rear": Rear temporal instability (bifacial)
                - "spectral_front": Front spectral mismatch
                - "spectral_rear": Rear spectral mismatch (bifacial)
                - "temperature_sensor": Temperature sensor
                - "temperature_correction": Temperature correction
                - "iv_measurement": I-V measurement
                - "phi_uncertainty": Bifaciality factor (bifacial)
                - "repeatability": Measurement repeatability
                - "reproducibility": Inter-lab reproducibility
            coverage_factor: k-factor for expanded uncertainty

        Returns:
            PowerUncertaintyResult with complete uncertainty analysis
        """
        # Default component values
        defaults = {
            "reference_cal": 0.015,
            "reference_stability": 0.005,
            "uniformity_front": 0.02,
            "uniformity_rear": 0.03,
            "temporal_front": 0.005,
            "temporal_rear": 0.008,
            "spectral_front": 0.01,
            "spectral_rear": 0.015,
            "temperature_sensor": 0.003,
            "temperature_correction": 0.004,
            "iv_measurement": 0.003,
            "phi_uncertainty": 0.02,
            "repeatability": 0.005,
            "reproducibility": 0.015,
            "g_front": 0.02,
            "g_rear": 0.03,
        }

        # Merge with provided values
        for key, value in defaults.items():
            if key not in u_components:
                u_components[key] = value

        # Calculate variance contributions based on mode
        var_components: Dict[str, float] = {}

        # Reference device (always)
        var_components["reference"] = (
            u_components["reference_cal"]**2 +
            u_components["reference_stability"]**2
        )

        # Front illumination (always)
        var_components["simulator_front"] = (
            u_components["uniformity_front"]**2 +
            u_components["temporal_front"]**2 +
            u_components["spectral_front"]**2
        )

        # Rear illumination (bifacial modes only)
        if mode in [BifacialMode.DOUBLE_SIDED, BifacialMode.EQUIVALENT_STC]:
            var_components["simulator_rear"] = (
                u_components["uniformity_rear"]**2 +
                u_components["temporal_rear"]**2 +
                u_components["spectral_rear"]**2
            )

            # Equivalent irradiance uncertainty
            if irradiance.g_rear > 0 and self.module:
                g_eq = irradiance.g_front + self.module.phi_isc * irradiance.g_rear
                # Weighted contribution based on irradiance ratio
                weight_rear = (self.module.phi_isc * irradiance.g_rear / g_eq)**2
                var_components["equivalent_irradiance"] = (
                    weight_rear * u_components["g_rear"]**2 +
                    (irradiance.g_rear / g_eq)**2 * u_components["phi_uncertainty"]**2
                )

        # Temperature (always)
        var_components["temperature"] = (
            u_components["temperature_sensor"]**2 +
            u_components["temperature_correction"]**2
        )

        # I-V measurement (always)
        var_components["iv_measurement"] = u_components["iv_measurement"]**2

        # Repeatability/reproducibility (always)
        var_components["repeatability"] = u_components["repeatability"]**2
        var_components["reproducibility"] = u_components["reproducibility"]**2

        # Calculate combined variance
        total_variance = sum(var_components.values())
        u_combined_rel = np.sqrt(total_variance)

        # Absolute uncertainties
        u_combined = power_measured * u_combined_rel
        u_expanded = u_combined * coverage_factor

        # Calculate effective degrees of freedom (simplified)
        # Assume each Type B component has ν → ∞, Type A has limited ν
        dof_effective = 50  # Typical value for mostly Type B

        # Confidence level from coverage factor
        if coverage_factor == 2.0:
            confidence = 95.45
        elif coverage_factor == 2.58:
            confidence = 99.0
        elif coverage_factor == 1.0:
            confidence = 68.27
        else:
            # Approximate from t-distribution
            confidence = 100 * (2 * stats.norm.cdf(coverage_factor) - 1)

        # Contribution fractions
        contributions = {
            name: var / total_variance if total_variance > 0 else 0
            for name, var in var_components.items()
        }

        return PowerUncertaintyResult(
            power=power_measured,
            power_stc=power_measured,  # Simplified (assume already corrected)
            u_combined=u_combined,
            u_expanded=u_expanded,
            coverage_factor=coverage_factor,
            confidence_level=confidence,
            u_combined_rel=u_combined_rel * 100,
            u_expanded_rel=u_combined_rel * coverage_factor * 100,
            dof_effective=dof_effective,
            contributions=contributions
        )


class BifacialMeasurementAnalyzer:
    """
    High-level analyzer for bifacial PV measurements.

    Provides convenience methods for common analysis workflows.
    """

    def __init__(self, calculator: Optional[BifacialUncertaintyCalculator] = None):
        """
        Initialize analyzer with optional calculator.

        Args:
            calculator: BifacialUncertaintyCalculator instance
        """
        self.calculator = calculator or BifacialUncertaintyCalculator()

    def analyze_single_sided_measurement(
        self,
        module: BifacialModule,
        side: Literal["front", "rear"],
        power_measured: float,
        irradiance: float = 1000.0,
        temperature: float = 25.0,
        u_irradiance: float = 0.02,
        u_temperature: float = 1.0,
        simulator_class: str = "AAA"
    ) -> PowerUncertaintyResult:
        """
        Analyze single-sided (front-only or rear-only) measurement.

        Args:
            module: BifacialModule specification
            side: "front" or "rear"
            power_measured: Measured power (W)
            irradiance: Irradiance level (W/m²)
            temperature: Module temperature (°C)
            u_irradiance: Relative irradiance uncertainty
            u_temperature: Temperature uncertainty (°C)
            simulator_class: Simulator classification ("AAA", "AAA+", etc.)

        Returns:
            PowerUncertaintyResult for the measurement
        """
        self.calculator.module = module

        # Set irradiance conditions
        if side == "front":
            irr = IrradianceConditions(
                g_front=irradiance,
                g_rear=0.0,
                u_g_front=irradiance * u_irradiance
            )
            mode = BifacialMode.SINGLE_FRONT
        else:
            irr = IrradianceConditions(
                g_front=0.0,
                g_rear=irradiance,
                u_g_rear=irradiance * u_irradiance
            )
            mode = BifacialMode.SINGLE_REAR

        temp = TemperatureConditions(
            t_module=temperature,
            u_t_module=u_temperature
        )

        # Set uncertainties based on simulator class
        class_uncertainties = {
            "AAA+": {"uniformity_front": 0.015, "temporal_front": 0.003},
            "AAA": {"uniformity_front": 0.02, "temporal_front": 0.005},
            "ABA": {"uniformity_front": 0.05, "temporal_front": 0.005},
            "BAA": {"uniformity_front": 0.02, "temporal_front": 0.05},
        }
        u_components = class_uncertainties.get(simulator_class, class_uncertainties["AAA"])

        return self.calculator.calculate_combined_power_uncertainty(
            mode=mode,
            irradiance=irr,
            temperature=temp,
            power_measured=power_measured,
            u_components=u_components
        )

    def analyze_bifacial_stc_measurement(
        self,
        module: BifacialModule,
        g_front: float,
        g_rear: float,
        power_measured: float,
        temperature: float = 25.0,
        use_equivalent_stc: bool = True
    ) -> Dict:
        """
        Complete analysis of bifacial STC measurement.

        Args:
            module: BifacialModule specification
            g_front: Front irradiance (W/m²)
            g_rear: Rear irradiance (W/m²)
            power_measured: Measured bifacial power (W)
            temperature: Module temperature (°C)
            use_equivalent_stc: Use equivalent irradiance method

        Returns:
            Dictionary with complete analysis results
        """
        self.calculator.module = module

        # Calculate equivalent irradiance
        eq_result = self.calculator.calculate_equivalent_irradiance(
            g_front=g_front,
            g_rear=g_rear,
            phi=module.phi_isc,
            u_g_front=g_front * 0.02,
            u_g_rear=g_rear * 0.03,
            u_phi=module.phi_isc * 0.02
        )

        # Calculate power uncertainty
        mode = BifacialMode.EQUIVALENT_STC if use_equivalent_stc else BifacialMode.DOUBLE_SIDED

        irr = IrradianceConditions(
            g_front=g_front,
            g_rear=g_rear,
            u_g_front=g_front * 0.02,
            u_g_rear=g_rear * 0.03
        )

        power_result = self.calculator.calculate_combined_power_uncertainty(
            mode=mode,
            irradiance=irr,
            temperature=TemperatureConditions(t_module=temperature),
            power_measured=power_measured,
            u_components={}
        )

        # Calculate bifacial gain
        p_mono_estimated = module.pmax_front * (g_front / 1000)
        gain_result = self.calculator.calculate_bifacial_gain(
            p_bifacial=power_measured,
            p_monofacial=p_mono_estimated,
            u_p_bifacial=power_result.u_combined,
            u_p_monofacial=p_mono_estimated * 0.025
        )

        return {
            "equivalent_irradiance": eq_result,
            "power_uncertainty": power_result,
            "bifacial_gain": gain_result,
            "module": module,
            "conditions": {
                "g_front": g_front,
                "g_rear": g_rear,
                "g_equivalent": eq_result.g_equivalent,
                "temperature": temperature
            }
        }

    def compare_measurement_modes(
        self,
        module: BifacialModule,
        power_front_only: float,
        power_rear_only: float,
        power_dual: float,
        g_front: float = 1000.0,
        g_rear: float = 150.0
    ) -> Dict:
        """
        Compare different bifacial measurement approaches.

        Args:
            module: BifacialModule specification
            power_front_only: Power from front-only measurement (W)
            power_rear_only: Power from rear-only measurement (W)
            power_dual: Power from simultaneous dual illumination (W)
            g_front: Front irradiance used (W/m²)
            g_rear: Rear irradiance used (W/m²)

        Returns:
            Comparison dictionary with all results
        """
        # Analyze each mode
        front_result = self.analyze_single_sided_measurement(
            module, "front", power_front_only, g_front
        )
        rear_result = self.analyze_single_sided_measurement(
            module, "rear", power_rear_only, g_front
        )
        dual_result = self.analyze_bifacial_stc_measurement(
            module, g_front, g_rear, power_dual
        )

        # Calculate bifaciality from measurements
        measured_phi_pmax = power_rear_only / power_front_only

        return {
            "single_sided_front": {
                "power": power_front_only,
                "uncertainty": front_result.u_expanded_rel,
            },
            "single_sided_rear": {
                "power": power_rear_only,
                "uncertainty": rear_result.u_expanded_rel,
            },
            "dual_illumination": dual_result,
            "measured_bifaciality": {
                "phi_pmax": measured_phi_pmax,
                "vs_spec": measured_phi_pmax / module.phi_pmax - 1
            }
        }


# =============================================================================
# Utility Functions
# =============================================================================

def convert_distribution_to_standard(
    value: float,
    distribution: DistributionType,
    half_width: bool = True
) -> float:
    """
    Convert distribution bounds to standard uncertainty.

    Args:
        value: Half-width (rectangular) or characteristic value
        distribution: Distribution type
        half_width: If True, value is half-width; if False, it's full range

    Returns:
        Standard uncertainty
    """
    if not half_width:
        value = value / 2

    divisors = {
        DistributionType.NORMAL: 1.0,      # Already standard deviation
        DistributionType.RECTANGULAR: np.sqrt(3),
        DistributionType.TRIANGULAR: np.sqrt(6),
        DistributionType.U_SHAPED: np.sqrt(2),
        DistributionType.LOGNORMAL: 1.0,   # Requires special handling
    }

    return value / divisors.get(distribution, np.sqrt(3))


def calculate_effective_dof(
    uncertainties: List[float],
    dofs: List[float]
) -> float:
    """
    Calculate effective degrees of freedom using Welch-Satterthwaite.

    ν_eff = u_c^4 / Σ(u_i^4 / ν_i)

    Args:
        uncertainties: List of standard uncertainties
        dofs: List of degrees of freedom for each component

    Returns:
        Effective degrees of freedom
    """
    u_c = np.sqrt(sum(u**2 for u in uncertainties))
    if u_c == 0:
        return float('inf')

    denominator = sum(
        (u**4 / nu) if nu > 0 else 0
        for u, nu in zip(uncertainties, dofs)
    )

    if denominator == 0:
        return float('inf')

    return u_c**4 / denominator


def coverage_factor_from_dof(
    dof: float,
    confidence: float = 0.95
) -> float:
    """
    Calculate coverage factor from effective degrees of freedom.

    Args:
        dof: Effective degrees of freedom
        confidence: Confidence level (0-1)

    Returns:
        Coverage factor k
    """
    if dof > 100:
        # Use normal distribution for large DOF
        return stats.norm.ppf((1 + confidence) / 2)
    else:
        # Use t-distribution
        return stats.t.ppf((1 + confidence) / 2, dof)


# =============================================================================
# Module-Level Convenience Functions
# =============================================================================

def quick_bifacial_uncertainty(
    pmax_front: float,
    pmax_rear: float,
    phi_isc: float = 0.75,
    u_front_rel: float = 0.025,
    u_rear_rel: float = 0.035
) -> Dict[str, float]:
    """
    Quick calculation of bifacial module uncertainties.

    Args:
        pmax_front: Front power at STC (W)
        pmax_rear: Rear power at STC (W)
        phi_isc: Bifaciality factor for Isc
        u_front_rel: Relative uncertainty on front power
        u_rear_rel: Relative uncertainty on rear power

    Returns:
        Dictionary with key uncertainty values
    """
    calc = BifacialUncertaintyCalculator()

    # Calculate phi_pmax
    phi_pmax = pmax_rear / pmax_front
    u_phi_pmax = phi_pmax * np.sqrt(u_front_rel**2 + u_rear_rel**2)

    # Equivalent irradiance at typical conditions
    g_front, g_rear = 900.0, 135.0  # Typical outdoor ratio
    eq_result = calc.calculate_equivalent_irradiance(
        g_front=g_front,
        g_rear=g_rear,
        phi=phi_isc,
        u_g_front=g_front * 0.02,
        u_g_rear=g_rear * 0.03,
        u_phi=phi_isc * 0.02
    )

    return {
        "phi_pmax": phi_pmax,
        "u_phi_pmax": u_phi_pmax,
        "u_phi_pmax_rel": u_phi_pmax / phi_pmax * 100,
        "g_equivalent": eq_result.g_equivalent,
        "u_g_equivalent": eq_result.u_g_equivalent,
        "u_g_equivalent_rel": eq_result.u_g_equivalent_rel * 100,
        "u_pmax_front": pmax_front * u_front_rel,
        "u_pmax_rear": pmax_rear * u_rear_rel,
    }


# Example usage and testing
if __name__ == "__main__":
    # Create a sample HJT bifacial module
    module = BifacialModule(
        manufacturer="Example Solar",
        model="HJT-400-BF",
        cell_technology=CellTechnology.HJT,
        pmax_front=400.0,
        pmax_rear=350.0,
        isc_front=10.5,
        isc_rear=9.7,
        voc_front=45.0,
        voc_rear=44.8
    )

    print("=" * 60)
    print("Bifacial Module Properties")
    print("=" * 60)
    print(f"Module: {module.manufacturer} {module.model}")
    print(f"Pmax Front: {module.pmax_front:.1f} W")
    print(f"Pmax Rear: {module.pmax_rear:.1f} W")
    print(f"φ_Isc: {module.phi_isc:.3f}")
    print(f"φ_Pmax: {module.phi_pmax:.3f}")

    # Create calculator
    calc = BifacialUncertaintyCalculator(module=module)

    # Calculate bifaciality factors
    print("\n" + "=" * 60)
    print("Bifaciality Factor Calculation")
    print("=" * 60)

    factors = calc.calculate_bifaciality_factors(
        front_isc=10.5, front_voc=45.0, front_pmax=400.0, front_ff=0.84,
        rear_isc=9.7, rear_voc=44.8, rear_pmax=350.0, rear_ff=0.83
    )

    print(f"φ_Isc: {factors.phi_isc:.4f} ± {factors.u_phi_isc:.4f} "
          f"({factors.u_phi_isc_rel*100:.2f}%)")
    print(f"φ_Pmax: {factors.phi_pmax:.4f} ± {factors.u_phi_pmax:.4f} "
          f"({factors.u_phi_pmax_rel*100:.2f}%)")

    # Calculate equivalent irradiance
    print("\n" + "=" * 60)
    print("Equivalent Irradiance Calculation")
    print("=" * 60)

    eq_result = calc.calculate_equivalent_irradiance(
        g_front=900.0,
        g_rear=135.0,
        phi=factors.phi_isc,
        u_g_front=18.0,
        u_g_rear=5.0,
        u_phi=factors.u_phi_isc
    )

    print(f"G_eq: {eq_result.g_equivalent:.1f} ± {eq_result.u_g_equivalent:.1f} W/m²")
    print(f"Relative uncertainty: {eq_result.u_g_equivalent_rel*100:.2f}%")
    print(f"Contribution from G_front: {eq_result.contribution_g_front*100:.1f}%")
    print(f"Contribution from G_rear: {eq_result.contribution_g_rear*100:.1f}%")
    print(f"Contribution from φ: {eq_result.contribution_phi*100:.1f}%")

    # Calculate bifacial gain
    print("\n" + "=" * 60)
    print("Bifacial Gain Calculation")
    print("=" * 60)

    gain_result = calc.calculate_bifacial_gain(
        p_bifacial=440.0,
        p_monofacial=400.0,
        u_p_bifacial=11.0,
        u_p_monofacial=10.0
    )

    print(f"Bifacial Gain: {gain_result.bifacial_gain_percent:.1f}%")
    print(f"Uncertainty: ±{gain_result.u_bifacial_gain*100:.2f}%")

    # Complete power uncertainty
    print("\n" + "=" * 60)
    print("Combined Power Uncertainty")
    print("=" * 60)

    power_result = calc.calculate_combined_power_uncertainty(
        mode=BifacialMode.DOUBLE_SIDED,
        irradiance=IrradianceConditions(g_front=900, g_rear=135),
        temperature=TemperatureConditions(t_module=25.0),
        power_measured=440.0,
        u_components={}
    )

    print(f"Power: {power_result.power:.1f} W")
    print(f"Combined uncertainty (k=1): ±{power_result.u_combined:.2f} W "
          f"({power_result.u_combined_rel:.2f}%)")
    print(f"Expanded uncertainty (k=2): ±{power_result.u_expanded:.2f} W "
          f"({power_result.u_expanded_rel:.2f}%)")
    print(f"Confidence level: {power_result.confidence_level:.1f}%")

    print("\nDominant uncertainty contributions:")
    sorted_contrib = sorted(power_result.contributions.items(),
                           key=lambda x: x[1], reverse=True)
    for name, fraction in sorted_contrib[:5]:
        print(f"  {name}: {fraction*100:.1f}%")
