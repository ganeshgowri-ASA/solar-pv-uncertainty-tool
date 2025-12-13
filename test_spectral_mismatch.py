"""
Unit tests for spectral mismatch calculation module.
Tests IEC 60904-7 compliant spectral mismatch factor calculations.
"""

import unittest
import numpy as np
from spectral_mismatch import (
    SpectralData,
    SpectralDataParser,
    SpectralMismatchCalculator,
    AM15GSpectrum,
    WavelabsSimulatorPresets,
    TypicalSpectralResponse,
    SpectralMismatchRecord,
    create_spectral_mismatch_uncertainty_factor
)


class TestSpectralData(unittest.TestCase):
    """Tests for SpectralData dataclass."""

    def test_spectral_data_creation(self):
        """Test basic SpectralData creation."""
        wavelength = np.array([300, 400, 500, 600, 700])
        values = np.array([0.1, 0.5, 0.8, 0.7, 0.4])

        sd = SpectralData(
            wavelength=wavelength,
            values=values,
            name="Test Spectrum",
            unit="W/m²/nm"
        )

        self.assertEqual(len(sd.wavelength), 5)
        self.assertEqual(sd.name, "Test Spectrum")
        self.assertEqual(sd.wavelength_range, (300, 700))

    def test_spectral_data_validation(self):
        """Test SpectralData validates array lengths."""
        wavelength = np.array([300, 400, 500])
        values = np.array([0.1, 0.5])  # Wrong length

        with self.assertRaises(ValueError):
            SpectralData(wavelength=wavelength, values=values)

    def test_integrated_value(self):
        """Test numerical integration of spectral data."""
        # Simple linear spectrum for easy verification
        wavelength = np.array([0, 100])
        values = np.array([1.0, 1.0])

        sd = SpectralData(wavelength=wavelength, values=values)
        integral = sd.integrated_value()

        # For constant value of 1 over 100nm, integral should be 100
        self.assertAlmostEqual(integral, 100.0, places=5)

    def test_normalize_spectrum(self):
        """Test spectrum normalization."""
        wavelength = np.array([0, 100, 200])
        values = np.array([1.0, 1.0, 1.0])

        sd = SpectralData(wavelength=wavelength, values=values)
        normalized = sd.normalize(target_integral=1000.0)

        self.assertAlmostEqual(normalized.integrated_value(), 1000.0, places=3)


class TestSpectralDataParser(unittest.TestCase):
    """Tests for spectral data parsing."""

    def test_parse_two_column(self):
        """Test parsing of simple two-column format."""
        content = """300 0.1
400 0.5
500 0.8
600 0.7
700 0.4"""

        sd = SpectralDataParser.parse_two_column(content)

        self.assertEqual(len(sd.wavelength), 5)
        self.assertAlmostEqual(sd.wavelength[0], 300.0)
        self.assertAlmostEqual(sd.values[2], 0.8)

    def test_parse_csv(self):
        """Test CSV parsing with auto-detection."""
        content = """wavelength,irradiance
300,0.1
400,0.5
500,0.8
600,0.7
700,0.4"""

        sd = SpectralDataParser.parse_csv(content)

        self.assertEqual(len(sd.wavelength), 5)
        self.assertAlmostEqual(sd.values[1], 0.5)


class TestAM15GSpectrum(unittest.TestCase):
    """Tests for AM1.5G reference spectrum."""

    def test_get_spectrum(self):
        """Test AM1.5G spectrum retrieval."""
        spectrum = AM15GSpectrum.get_spectrum()

        self.assertIsInstance(spectrum, SpectralData)
        self.assertEqual(spectrum.source, "IEC 60904-3")
        self.assertGreater(len(spectrum.wavelength), 50)

        # Check wavelength range is reasonable
        wl_min, wl_max = spectrum.wavelength_range
        self.assertLess(wl_min, 400)  # Should include UV
        self.assertGreater(wl_max, 2000)  # Should extend to IR

    def test_get_high_resolution(self):
        """Test high-resolution interpolated spectrum."""
        spectrum = AM15GSpectrum.get_high_resolution(
            wavelength_range=(350, 1100),
            step=5.0
        )

        self.assertIsInstance(spectrum, SpectralData)
        # Should have (1100-350)/5 + 1 = 151 points
        self.assertEqual(len(spectrum.wavelength), 151)

        # Check step size
        step = spectrum.wavelength[1] - spectrum.wavelength[0]
        self.assertAlmostEqual(step, 5.0)


class TestTypicalSpectralResponse(unittest.TestCase):
    """Tests for typical spectral response curves."""

    def test_csi_mono_sr(self):
        """Test c-Si monocrystalline SR curve."""
        sr = TypicalSpectralResponse.get_csi_mono()

        self.assertIsInstance(sr, SpectralData)
        self.assertEqual(sr.unit, "A/W")

        # c-Si should have response from ~350nm to ~1150nm
        wl_min, wl_max = sr.wavelength_range
        self.assertLessEqual(wl_min, 350)
        self.assertGreaterEqual(wl_max, 1100)

        # Peak response should be positive
        self.assertGreater(sr.values.max(), 0.3)

    def test_wpvs_sr(self):
        """Test WPVS reference cell SR curve."""
        sr = TypicalSpectralResponse.get_csi_wpvs()

        self.assertIsInstance(sr, SpectralData)
        self.assertGreater(sr.values.max(), 0.3)

    def test_all_technologies(self):
        """Test SR curves for all technologies."""
        technologies = [
            TypicalSpectralResponse.get_csi_mono,
            TypicalSpectralResponse.get_csi_wpvs,
            TypicalSpectralResponse.get_hjt,
            TypicalSpectralResponse.get_topcon,
            TypicalSpectralResponse.get_perovskite,
            TypicalSpectralResponse.get_cdte,
        ]

        for tech_func in technologies:
            sr = tech_func()
            self.assertIsInstance(sr, SpectralData)
            self.assertGreater(len(sr.wavelength), 10)
            self.assertGreater(sr.values.max(), 0)


class TestWavelabsPresets(unittest.TestCase):
    """Tests for Wavelabs simulator presets."""

    def test_sinus_220_spectrum(self):
        """Test SINUS-220 LED simulator spectrum."""
        spectrum = WavelabsSimulatorPresets.get_sinus_220_spectrum()

        self.assertIsInstance(spectrum, SpectralData)
        self.assertEqual(spectrum.unit, "W/m²/nm")
        self.assertIn("SINUS-220", spectrum.name)

        # Should have integrated irradiance near 1000 W/m²
        integral = spectrum.integrated_value()
        self.assertAlmostEqual(integral, 1000.0, delta=50)

    def test_avalon_nexun_spectrum(self):
        """Test Avalon Nexun spectrum."""
        spectrum = WavelabsSimulatorPresets.get_avalon_nexun_spectrum()

        self.assertIsInstance(spectrum, SpectralData)
        self.assertAlmostEqual(spectrum.integrated_value(), 1000.0, delta=50)

    def test_xenon_spectrum(self):
        """Test Class AAA Xenon spectrum."""
        spectrum = WavelabsSimulatorPresets.get_class_aaa_xenon_spectrum()

        self.assertIsInstance(spectrum, SpectralData)
        self.assertAlmostEqual(spectrum.integrated_value(), 1000.0, delta=50)


class TestSpectralMismatchCalculator(unittest.TestCase):
    """Tests for spectral mismatch factor calculation."""

    def setUp(self):
        """Set up test calculator with standard data."""
        self.ref_spectrum = AM15GSpectrum.get_high_resolution((350, 1100), step=10)
        self.sim_spectrum = WavelabsSimulatorPresets.get_class_aaa_xenon_spectrum()
        self.ref_sr = TypicalSpectralResponse.get_csi_wpvs()
        self.test_sr = TypicalSpectralResponse.get_csi_mono()

    def test_calculator_initialization(self):
        """Test calculator initialization."""
        calc = SpectralMismatchCalculator(
            reference_spectrum=self.ref_spectrum,
            simulator_spectrum=self.sim_spectrum,
            reference_device_sr=self.ref_sr,
            test_device_sr=self.test_sr
        )

        self.assertIsNotNone(calc.reference_spectrum)
        self.assertIsNotNone(calc.simulator_spectrum)

    def test_mismatch_factor_calculation(self):
        """Test M factor calculation."""
        calc = SpectralMismatchCalculator(
            reference_spectrum=self.ref_spectrum,
            simulator_spectrum=self.sim_spectrum,
            reference_device_sr=self.ref_sr,
            test_device_sr=self.test_sr
        )

        result = calc.calculate_mismatch_factor()

        # Check result structure
        self.assertIn('M_factor', result)
        self.assertIn('M_deviation_percent', result)
        self.assertIn('integrals', result)
        self.assertIn('wavelength_range', result)

        # M factor should be close to 1 for similar devices
        M = result['M_factor']
        self.assertGreater(M, 0.8)
        self.assertLess(M, 1.2)

    def test_identical_devices_m_factor(self):
        """Test M factor is 1.0 when reference and test are identical."""
        # Use same SR for both reference and test
        same_sr = TypicalSpectralResponse.get_csi_wpvs()

        # Use same spectrum for reference and simulator
        same_spectrum = AM15GSpectrum.get_high_resolution((350, 1100), step=10)

        calc = SpectralMismatchCalculator(
            reference_spectrum=same_spectrum,
            simulator_spectrum=same_spectrum,
            reference_device_sr=same_sr,
            test_device_sr=same_sr
        )

        result = calc.calculate_mismatch_factor()

        # M should be exactly 1.0
        self.assertAlmostEqual(result['M_factor'], 1.0, places=6)
        self.assertAlmostEqual(result['M_deviation_percent'], 0.0, places=4)

    def test_missing_data_raises_error(self):
        """Test that missing spectral data raises error."""
        calc = SpectralMismatchCalculator()

        with self.assertRaises(ValueError):
            calc.calculate_mismatch_factor()

    def test_uncertainty_calculation(self):
        """Test uncertainty propagation."""
        calc = SpectralMismatchCalculator(
            reference_spectrum=self.ref_spectrum,
            simulator_spectrum=self.sim_spectrum,
            reference_device_sr=self.ref_sr,
            test_device_sr=self.test_sr
        )

        result = calc.calculate_uncertainty(
            sr_ref_uncertainty=0.01,
            sr_test_uncertainty=0.02,
            spectrum_ref_uncertainty=0.005,
            spectrum_sim_uncertainty=0.02
        )

        # Check result structure
        self.assertIn('M_factor', result)
        self.assertIn('standard_uncertainty', result)
        self.assertIn('relative_uncertainty_percent', result)
        self.assertIn('expanded_uncertainty_k2', result)

        # Uncertainty should be positive and reasonable
        self.assertGreater(result['standard_uncertainty'], 0)
        self.assertLess(result['relative_uncertainty_percent'], 10)  # Should be < 10%

        # Expanded uncertainty should be 2x standard
        self.assertAlmostEqual(
            result['expanded_uncertainty_k2'],
            result['standard_uncertainty'] * 2,
            places=10
        )

    def test_isc_correction(self):
        """Test Isc correction calculation."""
        calc = SpectralMismatchCalculator(
            reference_spectrum=self.ref_spectrum,
            simulator_spectrum=self.sim_spectrum,
            reference_device_sr=self.ref_sr,
            test_device_sr=self.test_sr
        )

        isc_measured = 10.0  # A
        isc_uncertainty = 0.1  # A

        result = calc.get_isc_correction(isc_measured, isc_uncertainty)

        # Check result structure
        self.assertIn('isc_measured', result)
        self.assertIn('isc_corrected', result)
        self.assertIn('M_factor', result)
        self.assertIn('correction_percent', result)
        self.assertIn('uncertainty', result)

        # Corrected Isc should be close to measured
        self.assertGreater(result['isc_corrected'], 8.0)
        self.assertLess(result['isc_corrected'], 12.0)


class TestIntegrationWithUncertaintyBudget(unittest.TestCase):
    """Tests for integration with PVUncertaintyBudget."""

    def test_create_uncertainty_factor(self):
        """Test creation of uncertainty factor for budget."""
        # Create mock results
        m_result = {
            'M_factor': 0.995,
            'M_deviation_percent': -0.5
        }

        m_unc_result = {
            'standard_uncertainty': 0.02,
            'relative_uncertainty': 0.02,
            'relative_uncertainty_percent': 2.0
        }

        factor = create_spectral_mismatch_uncertainty_factor(m_result, m_unc_result)

        self.assertEqual(factor['category_id'], '2')
        self.assertEqual(factor['subcategory_id'], '2.3')
        self.assertEqual(factor['factor_id'], '2.3.4')
        self.assertIn('Spectral Mismatch', factor['name'])
        self.assertAlmostEqual(factor['standard_uncertainty'], 2.0)


class TestSpectralMismatchRecord(unittest.TestCase):
    """Tests for database record class."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        record = SpectralMismatchRecord(
            calculation_id="test123",
            timestamp="2024-01-01T12:00:00",
            M_factor=0.995,
            M_uncertainty=0.02,
            M_deviation_percent=-0.5,
            reference_spectrum="AM1.5G",
            simulator_spectrum="Wavelabs SINUS-220",
            reference_device="WPVS",
            test_device="c-Si Mono",
            wavelength_range_min=350.0,
            wavelength_range_max=1100.0
        )

        d = record.to_dict()

        self.assertEqual(d['calculation_id'], "test123")
        self.assertAlmostEqual(d['m_factor'], 0.995)
        self.assertEqual(d['reference_spectrum'], "AM1.5G")

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            'calculation_id': "test456",
            'timestamp': "2024-01-02T12:00:00",
            'm_factor': 1.005,
            'm_uncertainty': 0.015,
            'm_deviation_percent': 0.5,
            'reference_spectrum': "AM1.5G",
            'simulator_spectrum': "Xenon",
            'reference_device': "WPVS",
            'test_device': "HJT",
            'wavelength_range_min': 300.0,
            'wavelength_range_max': 1200.0
        }

        record = SpectralMismatchRecord.from_dict(d)

        self.assertEqual(record.calculation_id, "test456")
        self.assertAlmostEqual(record.M_factor, 1.005)
        self.assertEqual(record.test_device, "HJT")


class TestDifferentTechnologies(unittest.TestCase):
    """Tests for M factor calculations with different PV technologies."""

    def test_perovskite_mismatch(self):
        """Test M factor for perovskite module."""
        calc = SpectralMismatchCalculator(
            reference_spectrum=AM15GSpectrum.get_high_resolution((350, 900), step=10),
            simulator_spectrum=WavelabsSimulatorPresets.get_sinus_220_spectrum(),
            reference_device_sr=TypicalSpectralResponse.get_csi_wpvs(),
            test_device_sr=TypicalSpectralResponse.get_perovskite()
        )

        result = calc.calculate_mismatch_factor()

        # Perovskite has different spectral response - M should differ from 1
        self.assertIsNotNone(result['M_factor'])
        self.assertGreater(result['M_factor'], 0.7)
        self.assertLess(result['M_factor'], 1.3)

    def test_hjt_mismatch(self):
        """Test M factor for HJT module."""
        calc = SpectralMismatchCalculator(
            reference_spectrum=AM15GSpectrum.get_high_resolution((350, 1100), step=10),
            simulator_spectrum=WavelabsSimulatorPresets.get_avalon_nexun_spectrum(),
            reference_device_sr=TypicalSpectralResponse.get_csi_wpvs(),
            test_device_sr=TypicalSpectralResponse.get_hjt()
        )

        result = calc.calculate_mismatch_factor()

        # HJT is similar to c-Si, M should be close to 1
        self.assertGreater(result['M_factor'], 0.95)
        self.assertLess(result['M_factor'], 1.05)


if __name__ == '__main__':
    unittest.main()
