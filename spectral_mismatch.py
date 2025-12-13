"""
Spectral Mismatch Calculation Module
Implements IEC 60904-7 methodology for calculating spectral mismatch factor M
and its uncertainty contribution to PV measurements.

The spectral mismatch factor M corrects for:
1. Difference between simulator spectrum and reference spectrum (AM1.5G)
2. Difference between reference device SR and test device SR

M = [integral(E_ref * SR_ref) / integral(E_ref * SR_test)] *
    [integral(E_sim * SR_test) / integral(E_sim * SR_ref)]

Where:
- E_ref: Reference spectrum (AM1.5G per IEC 60904-3)
- E_sim: Simulator spectrum (measured or from manufacturer)
- SR_ref: Spectral response of reference device (WPVS cell)
- SR_test: Spectral response of test device
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from scipy import interpolate
from scipy.integrate import trapezoid, simpson
import io
import json


@dataclass
class SpectralData:
    """Container for spectral data (irradiance or spectral response)."""
    wavelength: np.ndarray  # nm
    values: np.ndarray  # W/m²/nm for irradiance, A/W for SR
    uncertainty: Optional[np.ndarray] = None  # Optional uncertainty at each wavelength
    name: str = ""
    source: str = ""
    unit: str = ""

    def __post_init__(self):
        """Validate data consistency."""
        if len(self.wavelength) != len(self.values):
            raise ValueError("Wavelength and values arrays must have same length")
        if self.uncertainty is not None and len(self.uncertainty) != len(self.wavelength):
            raise ValueError("Uncertainty array must match wavelength length")

    @property
    def wavelength_range(self) -> Tuple[float, float]:
        """Return wavelength range."""
        return (float(np.min(self.wavelength)), float(np.max(self.wavelength)))

    def integrated_value(self) -> float:
        """Calculate integrated value using trapezoidal rule."""
        return float(trapezoid(self.values, self.wavelength))

    def normalize(self, target_integral: float = 1000.0) -> 'SpectralData':
        """Normalize spectrum to target integrated value."""
        current_integral = self.integrated_value()
        if current_integral == 0:
            raise ValueError("Cannot normalize spectrum with zero integral")

        scale_factor = target_integral / current_integral
        return SpectralData(
            wavelength=self.wavelength.copy(),
            values=self.values * scale_factor,
            uncertainty=self.uncertainty * scale_factor if self.uncertainty is not None else None,
            name=self.name + " (normalized)",
            source=self.source,
            unit=self.unit
        )


class SpectralDataParser:
    """
    Parser for various spectral data file formats.
    Supports WPVS data, simulator spectra, and spectral response curves.
    """

    @staticmethod
    def parse_csv(
        file_content: Union[str, bytes, io.StringIO],
        wavelength_col: str = None,
        value_col: str = None,
        uncertainty_col: str = None,
        skip_rows: int = 0,
        delimiter: str = ','
    ) -> SpectralData:
        """
        Parse CSV file containing spectral data.

        Args:
            file_content: File content as string, bytes, or StringIO
            wavelength_col: Name of wavelength column (auto-detect if None)
            value_col: Name of value column (auto-detect if None)
            uncertainty_col: Name of uncertainty column (optional)
            skip_rows: Number of header rows to skip
            delimiter: Column delimiter

        Returns:
            SpectralData object
        """
        # Convert to StringIO if needed
        if isinstance(file_content, bytes):
            file_content = file_content.decode('utf-8')
        if isinstance(file_content, str):
            file_content = io.StringIO(file_content)

        # Read CSV
        df = pd.read_csv(file_content, skiprows=skip_rows, delimiter=delimiter)

        # Auto-detect columns if not specified
        if wavelength_col is None:
            wavelength_patterns = ['wavelength', 'lambda', 'wl', 'nm', 'wave']
            for col in df.columns:
                if any(p in col.lower() for p in wavelength_patterns):
                    wavelength_col = col
                    break
            if wavelength_col is None:
                wavelength_col = df.columns[0]  # Default to first column

        if value_col is None:
            value_patterns = ['irradiance', 'spectral', 'sr', 'response', 'value', 'intensity']
            for col in df.columns:
                if col != wavelength_col and any(p in col.lower() for p in value_patterns):
                    value_col = col
                    break
            if value_col is None:
                # Default to second column
                value_col = [c for c in df.columns if c != wavelength_col][0]

        wavelength = df[wavelength_col].values.astype(float)
        values = df[value_col].values.astype(float)

        uncertainty = None
        if uncertainty_col and uncertainty_col in df.columns:
            uncertainty = df[uncertainty_col].values.astype(float)

        return SpectralData(
            wavelength=wavelength,
            values=values,
            uncertainty=uncertainty,
            name=value_col,
            source="CSV import"
        )

    @staticmethod
    def parse_two_column(
        file_content: Union[str, bytes],
        skip_header_lines: int = 0
    ) -> SpectralData:
        """
        Parse simple two-column format (wavelength, value).
        Common format for NREL/ASTM reference spectra.
        """
        if isinstance(file_content, bytes):
            file_content = file_content.decode('utf-8')

        lines = file_content.strip().split('\n')[skip_header_lines:]
        wavelength = []
        values = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    wavelength.append(float(parts[0]))
                    values.append(float(parts[1]))
                except ValueError:
                    continue

        return SpectralData(
            wavelength=np.array(wavelength),
            values=np.array(values),
            name="Imported spectrum",
            source="Two-column import"
        )


class AM15GSpectrum:
    """
    IEC 60904-3 AM1.5G Reference Spectrum (1000 W/m² integrated).
    This is the standard reference spectrum for terrestrial PV testing.
    """

    # Simplified AM1.5G spectrum at key wavelengths (nm, W/m²/nm)
    # Full spectrum from 280-4000nm, simplified version here
    _WAVELENGTH = np.array([
        280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500,
        520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720, 740,
        760, 780, 800, 820, 840, 860, 880, 900, 920, 940, 960, 980,
        1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450,
        1500, 1550, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300,
        2400, 2500, 3000, 3500, 4000
    ])

    # Spectral irradiance values (W/m²/nm) - approximated from IEC 60904-3
    _IRRADIANCE = np.array([
        0.0001, 0.005, 0.12, 0.32, 0.48, 0.72, 1.05, 1.28, 1.45, 1.58,
        1.70, 1.65, 1.68, 1.62, 1.58, 1.55, 1.52, 1.47, 1.42, 1.38,
        1.32, 1.28, 1.22, 1.18, 1.05, 1.15, 1.08, 1.02, 0.98, 0.94,
        0.90, 0.85, 0.80, 0.72, 0.68, 0.65, 0.62, 0.54, 0.47, 0.40,
        0.34, 0.29, 0.25, 0.21, 0.08, 0.15, 0.13, 0.11, 0.09, 0.07,
        0.05, 0.03, 0.02, 0.015, 0.01, 0.008, 0.006, 0.004, 0.002,
        0.001, 0.0005
    ])

    @classmethod
    def get_spectrum(cls) -> SpectralData:
        """Get AM1.5G reference spectrum."""
        return SpectralData(
            wavelength=cls._WAVELENGTH.copy(),
            values=cls._IRRADIANCE.copy(),
            name="AM1.5G Reference Spectrum",
            source="IEC 60904-3",
            unit="W/m²/nm"
        )

    @classmethod
    def get_high_resolution(cls, wavelength_range: Tuple[float, float] = (280, 1200),
                            step: float = 5.0) -> SpectralData:
        """
        Get interpolated high-resolution AM1.5G spectrum.

        Args:
            wavelength_range: Min and max wavelength (nm)
            step: Wavelength step (nm)
        """
        wl_new = np.arange(wavelength_range[0], wavelength_range[1] + step, step)

        # Interpolate
        f = interpolate.interp1d(cls._WAVELENGTH, cls._IRRADIANCE,
                                  kind='linear', bounds_error=False, fill_value=0)
        values_new = f(wl_new)

        return SpectralData(
            wavelength=wl_new,
            values=values_new,
            name="AM1.5G Reference Spectrum (high-res)",
            source="IEC 60904-3 (interpolated)",
            unit="W/m²/nm"
        )


@dataclass
class WPVSReferenceDevice:
    """
    World Photovoltaic Scale (WPVS) Reference Device data.
    Contains spectral response and calibration information.
    """
    device_id: str
    calibration_lab: str
    calibration_date: str
    calibration_uncertainty: float  # % (k=2)
    spectral_response: SpectralData
    short_circuit_current_stc: float  # A at STC
    area_cm2: float
    technology: str = "c-Si"
    temperature_coefficient: float = 0.0  # %/°C

    @property
    def calibration_standard_uncertainty(self) -> float:
        """Standard uncertainty (k=1)."""
        return self.calibration_uncertainty / 2.0


@dataclass
class SimulatorSpectrum:
    """Solar simulator spectrum with metadata."""
    manufacturer: str
    model: str
    lamp_type: str
    spectrum: SpectralData
    classification: str = "AAA"  # IEC 60904-9 classification
    measurement_date: str = ""

    # Class limits per IEC 60904-9:2020
    SPECTRAL_MATCH_LIMITS = {
        'A+': (0.875, 1.125),
        'A': (0.75, 1.25),
        'B': (0.60, 1.40),
        'C': (0.40, 2.00)
    }


class SpectralMismatchCalculator:
    """
    Calculate spectral mismatch factor M per IEC 60904-7.

    M = [integral(E_ref * SR_ref) / integral(E_ref * SR_test)] *
        [integral(E_sim * SR_test) / integral(E_sim * SR_ref)]

    If M = 1.0, there is no spectral mismatch.
    If M != 1.0, measured Isc needs correction: Isc_corrected = Isc_measured * M
    """

    def __init__(
        self,
        reference_spectrum: SpectralData = None,
        simulator_spectrum: SpectralData = None,
        reference_device_sr: SpectralData = None,
        test_device_sr: SpectralData = None
    ):
        """
        Initialize calculator with spectral data.

        Args:
            reference_spectrum: Reference spectrum (default: AM1.5G)
            simulator_spectrum: Measured simulator spectrum
            reference_device_sr: WPVS/reference cell spectral response
            test_device_sr: Test device spectral response
        """
        self.reference_spectrum = reference_spectrum or AM15GSpectrum.get_spectrum()
        self.simulator_spectrum = simulator_spectrum
        self.reference_device_sr = reference_device_sr
        self.test_device_sr = test_device_sr

        # Common wavelength grid for integration
        self._common_wavelength = None

    def set_reference_spectrum(self, spectrum: SpectralData) -> None:
        """Set reference spectrum (e.g., AM1.5G)."""
        self.reference_spectrum = spectrum
        self._common_wavelength = None

    def set_simulator_spectrum(self, spectrum: SpectralData) -> None:
        """Set simulator spectrum."""
        self.simulator_spectrum = spectrum
        self._common_wavelength = None

    def set_reference_device_sr(self, sr: SpectralData) -> None:
        """Set reference device spectral response."""
        self.reference_device_sr = sr
        self._common_wavelength = None

    def set_test_device_sr(self, sr: SpectralData) -> None:
        """Set test device spectral response."""
        self.test_device_sr = sr
        self._common_wavelength = None

    def _determine_common_wavelength_range(self) -> Tuple[float, float]:
        """
        Determine common wavelength range for all spectra.
        Uses intersection of all wavelength ranges.
        """
        ranges = []
        for data in [self.reference_spectrum, self.simulator_spectrum,
                     self.reference_device_sr, self.test_device_sr]:
            if data is not None:
                ranges.append(data.wavelength_range)

        if not ranges:
            return (300, 1200)  # Default range for c-Si

        wl_min = max(r[0] for r in ranges)
        wl_max = min(r[1] for r in ranges)

        return (wl_min, wl_max)

    def _interpolate_to_common_grid(self, step: float = 5.0) -> Tuple[np.ndarray, ...]:
        """
        Interpolate all spectra to common wavelength grid.

        Args:
            step: Wavelength step in nm

        Returns:
            Tuple of (wavelength, E_ref, E_sim, SR_ref, SR_test)
        """
        wl_min, wl_max = self._determine_common_wavelength_range()
        self._common_wavelength = np.arange(wl_min, wl_max + step, step)

        interpolated = []
        for data in [self.reference_spectrum, self.simulator_spectrum,
                     self.reference_device_sr, self.test_device_sr]:
            if data is not None:
                f = interpolate.interp1d(
                    data.wavelength, data.values,
                    kind='linear', bounds_error=False, fill_value=0
                )
                interpolated.append(f(self._common_wavelength))
            else:
                interpolated.append(None)

        return (self._common_wavelength,) + tuple(interpolated)

    def calculate_mismatch_factor(self) -> Dict:
        """
        Calculate spectral mismatch factor M.

        Returns:
            Dictionary with M factor and diagnostic information
        """
        # Validate required data
        if self.simulator_spectrum is None:
            raise ValueError("Simulator spectrum is required")
        if self.reference_device_sr is None:
            raise ValueError("Reference device spectral response is required")
        if self.test_device_sr is None:
            raise ValueError("Test device spectral response is required")

        # Interpolate to common grid
        wl, E_ref, E_sim, SR_ref, SR_test = self._interpolate_to_common_grid()

        # Calculate integrals using Simpson's rule (more accurate than trapezoid)
        # Numerator integrals
        int_Eref_SRref = simpson(E_ref * SR_ref, x=wl)
        int_Esim_SRtest = simpson(E_sim * SR_test, x=wl)

        # Denominator integrals
        int_Eref_SRtest = simpson(E_ref * SR_test, x=wl)
        int_Esim_SRref = simpson(E_sim * SR_ref, x=wl)

        # Check for zero denominators
        if int_Eref_SRtest == 0 or int_Esim_SRref == 0:
            raise ValueError("Zero integral in denominator - check spectral data overlap")

        # Calculate M factor
        M = (int_Eref_SRref / int_Eref_SRtest) * (int_Esim_SRtest / int_Esim_SRref)

        # Calculate intermediate ratios for diagnostics
        ratio_ref = int_Eref_SRref / int_Eref_SRtest  # Effect of SR difference under ref spectrum
        ratio_sim = int_Esim_SRtest / int_Esim_SRref  # Effect of SR difference under simulator

        result = {
            'M_factor': float(M),
            'M_deviation_percent': float((M - 1.0) * 100),
            'ratio_reference_spectrum': float(ratio_ref),
            'ratio_simulator_spectrum': float(ratio_sim),
            'integrals': {
                'E_ref_SR_ref': float(int_Eref_SRref),
                'E_ref_SR_test': float(int_Eref_SRtest),
                'E_sim_SR_ref': float(int_Esim_SRref),
                'E_sim_SR_test': float(int_Esim_SRtest)
            },
            'wavelength_range': {
                'min': float(wl[0]),
                'max': float(wl[-1]),
                'points': len(wl)
            }
        }

        return result

    def calculate_uncertainty(
        self,
        sr_ref_uncertainty: float = 0.01,  # Relative uncertainty in SR_ref
        sr_test_uncertainty: float = 0.02,  # Relative uncertainty in SR_test
        spectrum_ref_uncertainty: float = 0.005,  # Relative uncertainty in E_ref
        spectrum_sim_uncertainty: float = 0.02,  # Relative uncertainty in E_sim
        correlation_sr: float = 0.5,  # Correlation between SR measurements
        correlation_spectrum: float = 0.3  # Correlation between spectrum measurements
    ) -> Dict:
        """
        Calculate uncertainty in spectral mismatch factor using sensitivity analysis.

        Uses partial derivatives and uncertainty propagation per GUM.

        Args:
            sr_ref_uncertainty: Relative standard uncertainty in reference SR
            sr_test_uncertainty: Relative standard uncertainty in test SR
            spectrum_ref_uncertainty: Relative standard uncertainty in reference spectrum
            spectrum_sim_uncertainty: Relative standard uncertainty in simulator spectrum
            correlation_sr: Correlation coefficient between SR measurements
            correlation_spectrum: Correlation coefficient between spectrum measurements

        Returns:
            Dictionary with uncertainty analysis results
        """
        # First calculate M factor
        m_result = self.calculate_mismatch_factor()
        M = m_result['M_factor']

        # Sensitivity coefficients (partial derivatives of ln(M) w.r.t. inputs)
        # For M = (A*D)/(B*C), dM/dx has sensitivity ~1 for each integral

        # Simplified uncertainty propagation assuming relative uncertainties
        # u(M)/M = sqrt(sum of squared relative uncertainty contributions)

        # Each integral contributes its relative uncertainty
        # With partial correlation considered

        u_rel_squared = (
            sr_ref_uncertainty**2 +  # From SR_ref in numerator
            sr_ref_uncertainty**2 +  # From SR_ref in denominator
            sr_test_uncertainty**2 +  # From SR_test in denominator
            sr_test_uncertainty**2 +  # From SR_test in numerator
            spectrum_ref_uncertainty**2 +  # From E_ref in numerator
            spectrum_ref_uncertainty**2 +  # From E_ref in denominator
            spectrum_sim_uncertainty**2 +  # From E_sim in numerator
            spectrum_sim_uncertainty**2    # From E_sim in denominator
        )

        # Subtract correlated terms (reduces uncertainty)
        u_rel_squared -= 2 * correlation_sr * sr_ref_uncertainty**2  # SR_ref appears in both
        u_rel_squared -= 2 * correlation_sr * sr_test_uncertainty**2  # SR_test appears in both
        u_rel_squared -= 2 * correlation_spectrum * spectrum_ref_uncertainty**2  # E_ref appears in both
        u_rel_squared -= 2 * correlation_spectrum * spectrum_sim_uncertainty**2  # E_sim appears in both

        u_rel_M = np.sqrt(max(0, u_rel_squared))
        u_abs_M = M * u_rel_M

        result = {
            'M_factor': M,
            'standard_uncertainty': float(u_abs_M),
            'relative_uncertainty': float(u_rel_M),
            'relative_uncertainty_percent': float(u_rel_M * 100),
            'expanded_uncertainty_k2': float(u_abs_M * 2),
            'coverage_factor': 2.0,
            'confidence_level': 0.95,
            'M_range_95': (float(M - 2*u_abs_M), float(M + 2*u_abs_M)),
            'input_uncertainties': {
                'sr_ref': sr_ref_uncertainty,
                'sr_test': sr_test_uncertainty,
                'spectrum_ref': spectrum_ref_uncertainty,
                'spectrum_sim': spectrum_sim_uncertainty
            },
            'correlations': {
                'sr': correlation_sr,
                'spectrum': correlation_spectrum
            }
        }

        return result

    def get_isc_correction(
        self,
        isc_measured: float,
        isc_uncertainty: float = 0.0
    ) -> Dict:
        """
        Calculate corrected short-circuit current.

        Isc_corrected = Isc_measured * M

        Args:
            isc_measured: Measured short-circuit current (A)
            isc_uncertainty: Uncertainty in measured Isc (A)

        Returns:
            Dictionary with corrected Isc and uncertainty
        """
        m_result = self.calculate_mismatch_factor()
        M = m_result['M_factor']

        isc_corrected = isc_measured * M

        # Uncertainty propagation
        # u(Isc_corr)/Isc_corr = sqrt((u(Isc)/Isc)^2 + (u(M)/M)^2)
        m_unc = self.calculate_uncertainty()

        rel_unc_isc = isc_uncertainty / isc_measured if isc_measured > 0 else 0
        rel_unc_M = m_unc['relative_uncertainty']

        rel_unc_corrected = np.sqrt(rel_unc_isc**2 + rel_unc_M**2)
        abs_unc_corrected = isc_corrected * rel_unc_corrected

        return {
            'isc_measured': isc_measured,
            'isc_corrected': isc_corrected,
            'M_factor': M,
            'correction_percent': (M - 1.0) * 100,
            'uncertainty': abs_unc_corrected,
            'relative_uncertainty_percent': rel_unc_corrected * 100,
            'expanded_uncertainty_k2': abs_unc_corrected * 2
        }


# ============================================================================
# WAVELABS SIMULATOR PRESETS
# ============================================================================

class WavelabsSimulatorPresets:
    """
    Preset spectra for Wavelabs solar simulators.
    Based on typical LED-based simulator characteristics.
    """

    @staticmethod
    def get_sinus_220_spectrum() -> SpectralData:
        """
        Wavelabs SINUS-220 LED simulator spectrum.
        Class AAA per IEC 60904-9.
        """
        # Typical LED simulator spectrum (approximation)
        wavelength = np.arange(350, 1200, 10)

        # Multi-peak LED spectrum characteristic
        values = np.zeros_like(wavelength, dtype=float)

        # Blue LED peak (~450nm)
        values += 0.8 * np.exp(-((wavelength - 450)**2) / (2 * 30**2))
        # Green LED contribution (~530nm)
        values += 0.9 * np.exp(-((wavelength - 530)**2) / (2 * 40**2))
        # Phosphor broadband (~550-650nm)
        values += 1.2 * np.exp(-((wavelength - 600)**2) / (2 * 80**2))
        # Red LED (~630nm)
        values += 0.85 * np.exp(-((wavelength - 660)**2) / (2 * 35**2))
        # NIR LED (~850nm)
        values += 0.5 * np.exp(-((wavelength - 850)**2) / (2 * 50**2))
        # NIR LED (~950nm)
        values += 0.3 * np.exp(-((wavelength - 950)**2) / (2 * 50**2))

        # Normalize to approximately match AM1.5G integrated irradiance
        values = values / trapezoid(values, wavelength) * 1000

        return SpectralData(
            wavelength=wavelength,
            values=values,
            name="Wavelabs SINUS-220",
            source="Wavelabs LED Simulator (estimated)",
            unit="W/m²/nm"
        )

    @staticmethod
    def get_avalon_nexun_spectrum() -> SpectralData:
        """
        Wavelabs Avalon Nexun LED simulator spectrum.
        Optimized for multi-junction cells.
        """
        wavelength = np.arange(350, 1200, 10)
        values = np.zeros_like(wavelength, dtype=float)

        # More uniform spectrum for Class AAA
        values += 0.7 * np.exp(-((wavelength - 400)**2) / (2 * 25**2))
        values += 0.9 * np.exp(-((wavelength - 500)**2) / (2 * 60**2))
        values += 1.1 * np.exp(-((wavelength - 620)**2) / (2 * 70**2))
        values += 0.8 * np.exp(-((wavelength - 750)**2) / (2 * 60**2))
        values += 0.5 * np.exp(-((wavelength - 900)**2) / (2 * 80**2))
        values += 0.25 * np.exp(-((wavelength - 1050)**2) / (2 * 60**2))

        values = values / trapezoid(values, wavelength) * 1000

        return SpectralData(
            wavelength=wavelength,
            values=values,
            name="Wavelabs Avalon Nexun",
            source="Wavelabs LED Simulator (estimated)",
            unit="W/m²/nm"
        )

    @staticmethod
    def get_class_aaa_xenon_spectrum() -> SpectralData:
        """
        Generic Class AAA Xenon simulator spectrum.
        Closer to AM1.5G than LED simulators.
        """
        # Use AM1.5G as base with slight modifications for xenon characteristics
        am15g = AM15GSpectrum.get_high_resolution((300, 1200), step=10)

        # Add xenon emission lines (simplified)
        wavelength = am15g.wavelength
        values = am15g.values.copy()

        # Xenon has stronger UV and some emission lines
        uv_boost = np.exp(-((wavelength - 350)**2) / (2 * 50**2)) * 0.1
        values = values * (1 + uv_boost)

        # Slight NIR enhancement
        nir_mod = 1 + 0.05 * np.exp(-((wavelength - 900)**2) / (2 * 100**2))
        values = values * nir_mod

        # Renormalize
        values = values / trapezoid(values, wavelength) * 1000

        return SpectralData(
            wavelength=wavelength,
            values=values,
            name="Class AAA Xenon Simulator",
            source="Generic Xenon (estimated)",
            unit="W/m²/nm"
        )


# ============================================================================
# TYPICAL SPECTRAL RESPONSE CURVES
# ============================================================================

class TypicalSpectralResponse:
    """
    Typical spectral response curves for different PV technologies.
    Used when actual SR data is not available.
    """

    @staticmethod
    def get_csi_mono() -> SpectralData:
        """Typical c-Si monocrystalline spectral response."""
        wavelength = np.arange(300, 1200, 10)

        # c-Si SR curve characteristics
        # Rising edge from ~350nm, peak around 900nm, cutoff at ~1100nm (bandgap)
        sr = np.zeros_like(wavelength, dtype=float)

        # Main response curve
        for i, wl in enumerate(wavelength):
            if wl < 350:
                sr[i] = 0
            elif wl < 500:
                sr[i] = 0.4 * (wl - 350) / 150  # Rising edge
            elif wl < 900:
                sr[i] = 0.4 + 0.2 * (wl - 500) / 400  # Gradual increase
            elif wl < 1050:
                sr[i] = 0.6 - 0.1 * (wl - 900) / 150  # Slight decrease
            elif wl < 1150:
                sr[i] = 0.5 * (1150 - wl) / 100  # Cutoff
            else:
                sr[i] = 0

        return SpectralData(
            wavelength=wavelength,
            values=sr,
            name="c-Si Monocrystalline SR",
            source="Typical (estimated)",
            unit="A/W"
        )

    @staticmethod
    def get_csi_wpvs() -> SpectralData:
        """
        Typical WPVS reference cell spectral response.
        Based on encapsulated c-Si reference cells.
        """
        wavelength = np.arange(300, 1200, 10)
        sr = np.zeros_like(wavelength, dtype=float)

        # WPVS cells have slightly different response due to encapsulation
        for i, wl in enumerate(wavelength):
            if wl < 350:
                sr[i] = 0
            elif wl < 450:
                sr[i] = 0.35 * (wl - 350) / 100  # UV region (limited by glass)
            elif wl < 850:
                sr[i] = 0.35 + 0.25 * (wl - 450) / 400
            elif wl < 1000:
                sr[i] = 0.6 - 0.05 * (wl - 850) / 150
            elif wl < 1150:
                sr[i] = 0.55 * (1150 - wl) / 150
            else:
                sr[i] = 0

        return SpectralData(
            wavelength=wavelength,
            values=sr,
            name="WPVS Reference Cell SR",
            source="Typical (estimated)",
            unit="A/W"
        )

    @staticmethod
    def get_hjt() -> SpectralData:
        """Typical HJT (Heterojunction) spectral response."""
        wavelength = np.arange(300, 1200, 10)
        sr = np.zeros_like(wavelength, dtype=float)

        # HJT has excellent blue response due to thin a-Si layers
        for i, wl in enumerate(wavelength):
            if wl < 320:
                sr[i] = 0
            elif wl < 400:
                sr[i] = 0.45 * (wl - 320) / 80  # Better UV response
            elif wl < 850:
                sr[i] = 0.45 + 0.15 * (wl - 400) / 450
            elif wl < 1050:
                sr[i] = 0.6 - 0.1 * (wl - 850) / 200
            elif wl < 1180:
                sr[i] = 0.5 * (1180 - wl) / 130
            else:
                sr[i] = 0

        return SpectralData(
            wavelength=wavelength,
            values=sr,
            name="HJT Cell SR",
            source="Typical (estimated)",
            unit="A/W"
        )

    @staticmethod
    def get_topcon() -> SpectralData:
        """Typical TOPCon spectral response."""
        wavelength = np.arange(300, 1200, 10)
        sr = np.zeros_like(wavelength, dtype=float)

        # TOPCon has very good response, similar to high-efficiency c-Si
        for i, wl in enumerate(wavelength):
            if wl < 340:
                sr[i] = 0
            elif wl < 450:
                sr[i] = 0.42 * (wl - 340) / 110
            elif wl < 900:
                sr[i] = 0.42 + 0.21 * (wl - 450) / 450
            elif wl < 1080:
                sr[i] = 0.63 - 0.08 * (wl - 900) / 180
            elif wl < 1180:
                sr[i] = 0.55 * (1180 - wl) / 100
            else:
                sr[i] = 0

        return SpectralData(
            wavelength=wavelength,
            values=sr,
            name="TOPCon Cell SR",
            source="Typical (estimated)",
            unit="A/W"
        )

    @staticmethod
    def get_perovskite() -> SpectralData:
        """Typical Perovskite spectral response."""
        wavelength = np.arange(300, 900, 10)
        sr = np.zeros_like(wavelength, dtype=float)

        # Perovskite has wider bandgap (~1.5-1.6 eV), cutoff around 800nm
        for i, wl in enumerate(wavelength):
            if wl < 350:
                sr[i] = 0
            elif wl < 450:
                sr[i] = 0.5 * (wl - 350) / 100  # Good blue response
            elif wl < 700:
                sr[i] = 0.5 + 0.1 * (wl - 450) / 250
            elif wl < 800:
                sr[i] = 0.6 * (800 - wl) / 100  # Sharper cutoff
            else:
                sr[i] = 0

        return SpectralData(
            wavelength=wavelength,
            values=sr,
            name="Perovskite Cell SR",
            source="Typical (estimated)",
            unit="A/W"
        )

    @staticmethod
    def get_cdte() -> SpectralData:
        """Typical CdTe thin-film spectral response."""
        wavelength = np.arange(300, 900, 10)
        sr = np.zeros_like(wavelength, dtype=float)

        # CdTe has ~1.45 eV bandgap, cutoff around 850nm
        for i, wl in enumerate(wavelength):
            if wl < 350:
                sr[i] = 0
            elif wl < 500:
                sr[i] = 0.35 * (wl - 350) / 150
            elif wl < 750:
                sr[i] = 0.35 + 0.15 * (wl - 500) / 250
            elif wl < 850:
                sr[i] = 0.5 * (850 - wl) / 100
            else:
                sr[i] = 0

        return SpectralData(
            wavelength=wavelength,
            values=sr,
            name="CdTe Cell SR",
            source="Typical (estimated)",
            unit="A/W"
        )


# ============================================================================
# DATABASE INTERFACE FOR RAILWAY
# ============================================================================

@dataclass
class SpectralMismatchRecord:
    """Record for storing spectral mismatch calculation in database."""
    calculation_id: str
    timestamp: str
    M_factor: float
    M_uncertainty: float
    M_deviation_percent: float
    reference_spectrum: str
    simulator_spectrum: str
    reference_device: str
    test_device: str
    wavelength_range_min: float
    wavelength_range_max: float
    notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for database storage."""
        return {
            'calculation_id': self.calculation_id,
            'timestamp': self.timestamp,
            'm_factor': self.M_factor,
            'm_uncertainty': self.M_uncertainty,
            'm_deviation_percent': self.M_deviation_percent,
            'reference_spectrum': self.reference_spectrum,
            'simulator_spectrum': self.simulator_spectrum,
            'reference_device': self.reference_device,
            'test_device': self.test_device,
            'wavelength_range_min': self.wavelength_range_min,
            'wavelength_range_max': self.wavelength_range_max,
            'notes': self.notes
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SpectralMismatchRecord':
        """Create from dictionary."""
        return cls(
            calculation_id=data.get('calculation_id', ''),
            timestamp=data.get('timestamp', ''),
            M_factor=data.get('m_factor', 1.0),
            M_uncertainty=data.get('m_uncertainty', 0.0),
            M_deviation_percent=data.get('m_deviation_percent', 0.0),
            reference_spectrum=data.get('reference_spectrum', ''),
            simulator_spectrum=data.get('simulator_spectrum', ''),
            reference_device=data.get('reference_device', ''),
            test_device=data.get('test_device', ''),
            wavelength_range_min=data.get('wavelength_range_min', 300.0),
            wavelength_range_max=data.get('wavelength_range_max', 1200.0),
            notes=data.get('notes', '')
        )


class SpectralMismatchDB:
    """
    Database interface for storing spectral mismatch calculations.
    Supports Railway PostgreSQL via psycopg2.
    """

    def __init__(self, connection_string: str = None):
        """
        Initialize database connection.

        Args:
            connection_string: PostgreSQL connection string
                              (uses RAILWAY_DATABASE_URL env var if not provided)
        """
        self.connection_string = connection_string
        self._connection = None

    def connect(self) -> bool:
        """Establish database connection."""
        import os

        if self.connection_string is None:
            self.connection_string = os.environ.get('RAILWAY_DATABASE_URL') or \
                                     os.environ.get('DATABASE_URL')

        if not self.connection_string:
            return False

        try:
            import psycopg2
            self._connection = psycopg2.connect(self.connection_string)
            return True
        except Exception:
            return False

    def create_table(self) -> bool:
        """Create spectral_data table if it doesn't exist."""
        if not self._connection:
            return False

        create_sql = """
        CREATE TABLE IF NOT EXISTS spectral_data (
            id SERIAL PRIMARY KEY,
            calculation_id VARCHAR(100) UNIQUE NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            m_factor DECIMAL(10, 6) NOT NULL,
            m_uncertainty DECIMAL(10, 6),
            m_deviation_percent DECIMAL(10, 4),
            reference_spectrum VARCHAR(200),
            simulator_spectrum VARCHAR(200),
            reference_device VARCHAR(200),
            test_device VARCHAR(200),
            wavelength_range_min DECIMAL(10, 2),
            wavelength_range_max DECIMAL(10, 2),
            notes TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_spectral_data_timestamp
            ON spectral_data(timestamp);
        CREATE INDEX IF NOT EXISTS idx_spectral_data_m_factor
            ON spectral_data(m_factor);
        """

        try:
            cursor = self._connection.cursor()
            cursor.execute(create_sql)
            self._connection.commit()
            cursor.close()
            return True
        except Exception:
            return False

    def save_record(self, record: SpectralMismatchRecord) -> bool:
        """Save spectral mismatch calculation to database."""
        if not self._connection:
            return False

        insert_sql = """
        INSERT INTO spectral_data (
            calculation_id, timestamp, m_factor, m_uncertainty,
            m_deviation_percent, reference_spectrum, simulator_spectrum,
            reference_device, test_device, wavelength_range_min,
            wavelength_range_max, notes
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (calculation_id) DO UPDATE SET
            m_factor = EXCLUDED.m_factor,
            m_uncertainty = EXCLUDED.m_uncertainty,
            m_deviation_percent = EXCLUDED.m_deviation_percent,
            updated_at = NOW();
        """

        try:
            cursor = self._connection.cursor()
            cursor.execute(insert_sql, (
                record.calculation_id,
                record.timestamp,
                record.M_factor,
                record.M_uncertainty,
                record.M_deviation_percent,
                record.reference_spectrum,
                record.simulator_spectrum,
                record.reference_device,
                record.test_device,
                record.wavelength_range_min,
                record.wavelength_range_max,
                record.notes
            ))
            self._connection.commit()
            cursor.close()
            return True
        except Exception:
            return False

    def get_recent_records(self, limit: int = 10) -> List[SpectralMismatchRecord]:
        """Retrieve recent spectral mismatch calculations."""
        if not self._connection:
            return []

        select_sql = """
        SELECT calculation_id, timestamp, m_factor, m_uncertainty,
               m_deviation_percent, reference_spectrum, simulator_spectrum,
               reference_device, test_device, wavelength_range_min,
               wavelength_range_max, notes
        FROM spectral_data
        ORDER BY timestamp DESC
        LIMIT %s;
        """

        try:
            cursor = self._connection.cursor()
            cursor.execute(select_sql, (limit,))
            rows = cursor.fetchall()
            cursor.close()

            records = []
            for row in rows:
                records.append(SpectralMismatchRecord(
                    calculation_id=row[0],
                    timestamp=str(row[1]),
                    M_factor=float(row[2]),
                    M_uncertainty=float(row[3]) if row[3] else 0.0,
                    M_deviation_percent=float(row[4]) if row[4] else 0.0,
                    reference_spectrum=row[5] or '',
                    simulator_spectrum=row[6] or '',
                    reference_device=row[7] or '',
                    test_device=row[8] or '',
                    wavelength_range_min=float(row[9]) if row[9] else 300.0,
                    wavelength_range_max=float(row[10]) if row[10] else 1200.0,
                    notes=row[11] or ''
                ))
            return records
        except Exception:
            return []

    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None


# ============================================================================
# INTEGRATION WITH UNCERTAINTY BUDGET
# ============================================================================

def create_spectral_mismatch_uncertainty_factor(
    m_result: Dict,
    m_uncertainty_result: Dict
) -> Dict:
    """
    Create uncertainty factor for integration with PVUncertaintyBudget.

    Args:
        m_result: Result from SpectralMismatchCalculator.calculate_mismatch_factor()
        m_uncertainty_result: Result from SpectralMismatchCalculator.calculate_uncertainty()

    Returns:
        Dictionary compatible with UncertaintyFactor in pv_uncertainty_enhanced.py
    """
    M = m_result['M_factor']
    u_M = m_uncertainty_result['standard_uncertainty']

    # Convert M factor to percentage uncertainty contribution
    # The M factor affects Isc, so its uncertainty propagates to Pmax
    # u_relative = u(M) / M
    u_rel_percent = m_uncertainty_result['relative_uncertainty_percent']

    return {
        'category_id': '2',
        'subcategory_id': '2.3',
        'factor_id': '2.3.4',
        'name': 'Spectral Mismatch Factor (M)',
        'value': M,
        'standard_uncertainty': u_rel_percent,  # % contribution to uncertainty
        'distribution': 'normal',
        'sensitivity_coefficient': 1.0,
        'unit': '%',
        'notes': f"M = {M:.4f} (deviation: {m_result['M_deviation_percent']:.2f}%)"
    }


def integrate_with_uncertainty_budget(
    budget,  # PVUncertaintyBudget from pv_uncertainty_enhanced
    m_result: Dict,
    m_uncertainty_result: Dict
) -> None:
    """
    Add spectral mismatch factor to existing uncertainty budget.

    Args:
        budget: PVUncertaintyBudget instance
        m_result: Result from calculate_mismatch_factor()
        m_uncertainty_result: Result from calculate_uncertainty()
    """
    from pv_uncertainty_enhanced import UncertaintyFactor

    factor_dict = create_spectral_mismatch_uncertainty_factor(m_result, m_uncertainty_result)

    factor = UncertaintyFactor(
        category_id=factor_dict['category_id'],
        subcategory_id=factor_dict['subcategory_id'],
        factor_id=factor_dict['factor_id'],
        name=factor_dict['name'],
        value=factor_dict['value'],
        standard_uncertainty=factor_dict['standard_uncertainty'],
        distribution=factor_dict['distribution'],
        sensitivity_coefficient=factor_dict['sensitivity_coefficient'],
        unit=factor_dict['unit'],
        notes=factor_dict['notes']
    )

    budget.add_factor(factor)
