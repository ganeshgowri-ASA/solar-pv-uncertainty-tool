"""
Comprehensive Test Scenarios for Universal Solar Simulator Uncertainty Framework

This module provides:
- Unit tests for all core calculations
- Integration tests for complete workflows
- Reference test cases from IEC standards
- Monofacial backward compatibility tests
- Bifacial-specific validation tests
- Monte Carlo validation tests
- Standards compliance tests

Test Categories:
1. Bifaciality factor calculations
2. Equivalent irradiance calculations
3. Uncertainty propagation (GUM)
4. Monte Carlo simulation validation
5. IEC TS 60904-1-2:2024 compliance
6. Simulator classification
7. Backward compatibility with monofacial

Author: Universal Solar Simulator Framework Team
Version: 2.0.0
"""

import unittest
import numpy as np
from typing import Dict, List, Tuple
import sys
import os

# Import modules to test
from bifacial_uncertainty import (
    BifacialModule,
    BifacialUncertaintyCalculator,
    BifacialMeasurementAnalyzer,
    BifacialMode,
    CellTechnology,
    IrradianceConditions,
    TemperatureConditions,
    AlbedoType,
    AlbedoSpectrum,
    quick_bifacial_uncertainty,
    convert_distribution_to_standard,
    DistributionType,
)

from uncertainty_components import (
    UncertaintyComponent,
    UncertaintyBudgetBuilder,
    UncertaintyBudget,
    SimulatorConfiguration,
    SimulatorType,
    SimulatorClassification,
    ComponentCategory,
    get_simulator,
    list_simulators,
    SIMULATOR_DATABASE,
)

from monte_carlo_analysis import (
    MCInputVariable,
    MCResult,
    BifacialMCInputs,
    MonteCarloSimulator,
    BifacialMonteCarloSimulator,
    AdaptiveMonteCarloSimulator,
    compare_gum_monte_carlo,
    DistributionType as MCDistribution,
)

from standards_compliance import (
    IEC_60904_9_Checker,
    IEC_TS_60904_1_2_Checker,
    GUM_Compliance_Checker,
    ISO_17025_Checker,
    ComplianceStatus,
    MeasurementType,
    check_bifacial_stc_compliance,
)


# =============================================================================
# Test Reference Data
# =============================================================================

class ReferenceData:
    """Reference test data and expected values"""

    # HJT bifacial module reference
    HJT_MODULE = {
        "manufacturer": "Test Solar",
        "model": "HJT-400-BF",
        "pmax_front": 400.0,
        "pmax_rear": 360.0,
        "isc_front": 10.5,
        "isc_rear": 9.66,
        "voc_front": 45.0,
        "voc_rear": 44.8,
        "phi_isc_expected": 0.92,
        "phi_pmax_expected": 0.90,
    }

    # TOPCon bifacial module reference
    TOPCON_MODULE = {
        "manufacturer": "Test Solar",
        "model": "TOPCon-380-BF",
        "pmax_front": 380.0,
        "pmax_rear": 266.0,
        "isc_front": 10.0,
        "isc_rear": 7.5,
        "voc_front": 44.5,
        "voc_rear": 44.2,
        "phi_isc_expected": 0.75,
        "phi_pmax_expected": 0.70,
    }

    # Equivalent irradiance test cases
    EQUIVALENT_IRRADIANCE_CASES = [
        # (G_front, G_rear, phi, expected_G_eq)
        (1000.0, 0.0, 0.75, 1000.0),
        (1000.0, 100.0, 0.75, 1075.0),
        (1000.0, 150.0, 0.75, 1112.5),
        (900.0, 135.0, 0.75, 1001.25),
        (800.0, 200.0, 0.80, 960.0),
    ]

    # Simulator classification test cases
    SIMULATOR_CLASSIFICATION_CASES = [
        # (uniformity, temporal, expected_u_class, expected_t_class)
        (1.5, 0.3, "A", "A"),
        (2.0, 2.0, "A", "A"),
        (2.5, 1.0, "B", "A"),
        (4.0, 4.0, "B", "B"),
        (8.0, 8.0, "C", "C"),
        (12.0, 12.0, "Fail", "Fail"),
    ]


# =============================================================================
# Unit Tests - Bifaciality Calculations
# =============================================================================

class TestBifacialityFactors(unittest.TestCase):
    """Test bifaciality factor calculations"""

    def setUp(self):
        """Set up test fixtures"""
        self.calculator = BifacialUncertaintyCalculator()

    def test_phi_isc_calculation(self):
        """Test φ_Isc calculation"""
        result = self.calculator.calculate_bifaciality_factors(
            front_isc=10.0, front_voc=45.0, front_pmax=400.0, front_ff=0.85,
            rear_isc=7.5, rear_voc=44.8, rear_pmax=280.0, rear_ff=0.84
        )
        self.assertAlmostEqual(result.phi_isc, 0.75, places=3)

    def test_phi_pmax_calculation(self):
        """Test φ_Pmax calculation"""
        result = self.calculator.calculate_bifaciality_factors(
            front_isc=10.0, front_voc=45.0, front_pmax=400.0, front_ff=0.85,
            rear_isc=7.5, rear_voc=44.8, rear_pmax=280.0, rear_ff=0.84
        )
        self.assertAlmostEqual(result.phi_pmax, 0.70, places=3)

    def test_phi_voc_calculation(self):
        """Test φ_Voc calculation (typically ~0.98-1.0)"""
        result = self.calculator.calculate_bifaciality_factors(
            front_isc=10.0, front_voc=45.0, front_pmax=400.0, front_ff=0.85,
            rear_isc=7.5, rear_voc=44.8, rear_pmax=280.0, rear_ff=0.84
        )
        self.assertAlmostEqual(result.phi_voc, 44.8/45.0, places=3)

    def test_phi_uncertainty_propagation(self):
        """Test uncertainty propagation in φ calculation"""
        result = self.calculator.calculate_bifaciality_factors(
            front_isc=10.0, front_voc=45.0, front_pmax=400.0, front_ff=0.85,
            rear_isc=7.5, rear_voc=44.8, rear_pmax=280.0, rear_ff=0.84,
            u_front_rel=0.02,
            u_rear_rel=0.03
        )
        # Uncertainty should be non-zero and reasonable
        self.assertGreater(result.u_phi_isc, 0)
        self.assertLess(result.u_phi_isc_rel, 0.1)  # Should be < 10%

    def test_hjt_module_reference(self):
        """Test against HJT module reference data"""
        ref = ReferenceData.HJT_MODULE
        result = self.calculator.calculate_bifaciality_factors(
            front_isc=ref["isc_front"],
            front_voc=45.0,
            front_pmax=ref["pmax_front"],
            front_ff=0.85,
            rear_isc=ref["isc_rear"],
            rear_voc=44.8,
            rear_pmax=ref["pmax_rear"],
            rear_ff=0.84
        )
        self.assertAlmostEqual(result.phi_isc, ref["phi_isc_expected"], places=2)

    def test_correlation_effect(self):
        """Test that correlation affects uncertainty"""
        # With correlation
        result_corr = self.calculator.calculate_bifaciality_factors(
            front_isc=10.0, front_voc=45.0, front_pmax=400.0, front_ff=0.85,
            rear_isc=7.5, rear_voc=44.8, rear_pmax=280.0, rear_ff=0.84,
            include_correlation=True
        )

        # Without correlation (set coefficient to 0)
        self.calculator.correlation_coefficient = 0.0
        result_nocorr = self.calculator.calculate_bifaciality_factors(
            front_isc=10.0, front_voc=45.0, front_pmax=400.0, front_ff=0.85,
            rear_isc=7.5, rear_voc=44.8, rear_pmax=280.0, rear_ff=0.84,
            include_correlation=False
        )

        # Uncertainty should differ
        self.assertNotAlmostEqual(
            result_corr.u_phi_isc,
            result_nocorr.u_phi_isc,
            places=5
        )


# =============================================================================
# Unit Tests - Equivalent Irradiance
# =============================================================================

class TestEquivalentIrradiance(unittest.TestCase):
    """Test equivalent irradiance calculations"""

    def setUp(self):
        self.calculator = BifacialUncertaintyCalculator()

    def test_equivalent_irradiance_formula(self):
        """Test G_eq = G_front + φ × G_rear"""
        for g_front, g_rear, phi, expected in ReferenceData.EQUIVALENT_IRRADIANCE_CASES:
            result = self.calculator.calculate_equivalent_irradiance(
                g_front=g_front,
                g_rear=g_rear,
                phi=phi,
                u_g_front=g_front * 0.02,
                u_g_rear=g_rear * 0.03 if g_rear > 0 else 0,
                u_phi=phi * 0.02
            )
            self.assertAlmostEqual(result.g_equivalent, expected, places=2,
                                   msg=f"Failed for G_front={g_front}, G_rear={g_rear}")

    def test_equivalent_irradiance_uncertainty(self):
        """Test uncertainty propagation in G_eq"""
        result = self.calculator.calculate_equivalent_irradiance(
            g_front=900.0,
            g_rear=135.0,
            phi=0.75,
            u_g_front=18.0,
            u_g_rear=5.0,
            u_phi=0.015
        )
        # Uncertainty should be positive
        self.assertGreater(result.u_g_equivalent, 0)
        # Relative uncertainty should be reasonable (< 5%)
        self.assertLess(result.u_g_equivalent_rel, 0.05)

    def test_contribution_fractions_sum_to_one(self):
        """Test that contribution fractions sum to 1.0"""
        result = self.calculator.calculate_equivalent_irradiance(
            g_front=900.0,
            g_rear=135.0,
            phi=0.75,
            u_g_front=18.0,
            u_g_rear=5.0,
            u_phi=0.015
        )
        total = (result.contribution_g_front +
                result.contribution_g_rear +
                result.contribution_phi)
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_rear_zero_gives_front_only(self):
        """Test that G_rear=0 gives G_eq = G_front"""
        result = self.calculator.calculate_equivalent_irradiance(
            g_front=1000.0,
            g_rear=0.0,
            phi=0.75,
            u_g_front=20.0,
            u_g_rear=0.0,
            u_phi=0.015
        )
        self.assertAlmostEqual(result.g_equivalent, 1000.0, places=2)

    def test_required_irradiances_calculation(self):
        """Test calculation of required irradiances for target G_eq"""
        g_front, g_rear = self.calculator.calculate_required_irradiances(
            g_eq_target=1000.0,
            rear_ratio=0.15,
            phi=0.75
        )
        # Verify G_eq = G_front + φ × G_rear = 1000
        g_eq = g_front + 0.75 * g_rear
        self.assertAlmostEqual(g_eq, 1000.0, places=2)
        # Verify ratio
        self.assertAlmostEqual(g_rear / g_front, 0.15, places=3)


# =============================================================================
# Unit Tests - Uncertainty Components
# =============================================================================

class TestUncertaintyComponents(unittest.TestCase):
    """Test uncertainty component classes"""

    def test_standard_uncertainty_normal(self):
        """Test standard uncertainty for normal distribution"""
        comp = UncertaintyComponent(
            name="test",
            uncertainty=1.0,
            distribution=DistributionType.NORMAL
        )
        self.assertAlmostEqual(comp.standard_uncertainty, 1.0, places=5)

    def test_standard_uncertainty_rectangular(self):
        """Test standard uncertainty for rectangular distribution"""
        comp = UncertaintyComponent(
            name="test",
            uncertainty=1.0,
            distribution=DistributionType.RECTANGULAR
        )
        expected = 1.0 / np.sqrt(3)
        self.assertAlmostEqual(comp.standard_uncertainty, expected, places=5)

    def test_standard_uncertainty_triangular(self):
        """Test standard uncertainty for triangular distribution"""
        comp = UncertaintyComponent(
            name="test",
            uncertainty=1.0,
            distribution=DistributionType.TRIANGULAR
        )
        expected = 1.0 / np.sqrt(6)
        self.assertAlmostEqual(comp.standard_uncertainty, expected, places=5)

    def test_variance_contribution(self):
        """Test variance contribution calculation"""
        comp = UncertaintyComponent(
            name="test",
            uncertainty=2.0,
            sensitivity_coefficient=0.5,
            distribution=DistributionType.NORMAL
        )
        expected = (0.5 * 2.0) ** 2
        self.assertAlmostEqual(comp.variance_contribution, expected, places=5)

    def test_component_serialization(self):
        """Test component to/from dict"""
        comp = UncertaintyComponent(
            name="test_component",
            description="Test description",
            uncertainty=1.5,
            unit="%"
        )
        data = comp.to_dict()
        restored = UncertaintyComponent.from_dict(data)
        self.assertEqual(restored.name, comp.name)
        self.assertAlmostEqual(restored.uncertainty, comp.uncertainty, places=5)


class TestUncertaintyBudgetBuilder(unittest.TestCase):
    """Test uncertainty budget builder"""

    def test_standard_stc_budget(self):
        """Test building standard STC budget"""
        builder = UncertaintyBudgetBuilder(measurement_type="stc")
        budget = builder.build_standard_stc()

        self.assertGreater(len(budget.components), 5)
        self.assertGreater(budget.combined_standard_uncertainty, 0)
        self.assertGreater(budget.expanded_uncertainty, 0)

    def test_bifacial_stc_budget(self):
        """Test building bifacial STC budget"""
        builder = UncertaintyBudgetBuilder(measurement_type="bifacial_stc")
        budget = builder.build_bifacial_stc()

        # Should have more components than standard
        self.assertGreater(len(budget.components), 10)
        self.assertTrue(budget.is_bifacial)

    def test_simulator_integration(self):
        """Test simulator configuration integration"""
        builder = UncertaintyBudgetBuilder()
        builder.set_simulator("pasan_highlight_bifacial")

        self.assertIsNotNone(builder.simulator)
        self.assertEqual(builder.simulator.manufacturer, "PASAN")
        self.assertTrue(builder.simulator.bifacial_capable)

    def test_category_contributions(self):
        """Test category contribution calculation"""
        builder = UncertaintyBudgetBuilder()
        budget = builder.build_standard_stc()

        contributions = budget.get_category_contributions()
        self.assertGreater(len(contributions), 0)
        # Contributions should sum to ~1.0
        total = sum(contributions.values())
        self.assertAlmostEqual(total, 1.0, places=3)


class TestSimulatorDatabase(unittest.TestCase):
    """Test simulator database"""

    def test_pasan_simulators_exist(self):
        """Test PASAN simulators are in database"""
        pasan_sims = [k for k in SIMULATOR_DATABASE.keys() if "pasan" in k]
        self.assertGreater(len(pasan_sims), 0)

    def test_bifacial_capable_list(self):
        """Test bifacial-capable simulator list"""
        bifacial_sims = list_simulators(bifacial_only=True)
        self.assertGreater(len(bifacial_sims), 0)

        for key in bifacial_sims:
            sim = get_simulator(key)
            self.assertTrue(sim.bifacial_capable)

    def test_all_simulators_valid(self):
        """Test all simulators have valid configurations"""
        for key in list_simulators():
            sim = get_simulator(key)
            self.assertIsNotNone(sim)
            self.assertGreater(sim.uniformity_front, 0)
            self.assertGreater(sim.temporal_front, 0)


# =============================================================================
# Unit Tests - Monte Carlo
# =============================================================================

class TestMonteCarloSimulation(unittest.TestCase):
    """Test Monte Carlo simulation"""

    def test_basic_simulation(self):
        """Test basic MC simulation runs"""
        sim = MonteCarloSimulator(n_samples=10000, seed=42)
        sim.add_input(MCInputVariable("x", 100.0, 5.0))
        sim.add_input(MCInputVariable("y", 50.0, 2.0))

        def model(x, y):
            return x + y

        result = sim.run(model)
        self.assertAlmostEqual(result.mean, 150.0, delta=1.0)
        # Standard uncertainty should be sqrt(5^2 + 2^2) ≈ 5.39
        self.assertAlmostEqual(result.std, np.sqrt(25 + 4), delta=0.5)

    def test_bifacial_equivalent_irradiance(self):
        """Test bifacial equivalent irradiance MC simulation"""
        inputs = BifacialMCInputs()
        inputs.g_front = MCInputVariable("G_front", 900.0, 18.0)
        inputs.g_rear = MCInputVariable("G_rear", 135.0, 5.0)
        inputs.phi_isc = MCInputVariable("φ_Isc", 0.75, 0.015)

        sim = BifacialMonteCarloSimulator(n_samples=50000, seed=42)
        sim.set_bifacial_inputs(inputs)

        result = sim.simulate_equivalent_irradiance()

        # Expected G_eq = 900 + 0.75 × 135 = 1001.25
        self.assertAlmostEqual(result.mean, 1001.25, delta=5.0)
        self.assertGreater(result.std, 0)

    def test_coverage_interval_contains_mean(self):
        """Test that coverage intervals contain the mean"""
        sim = MonteCarloSimulator(n_samples=50000, seed=42)
        sim.add_input(MCInputVariable("x", 100.0, 10.0))

        result = sim.run(lambda x: x)

        # 95% interval should contain mean
        self.assertGreater(result.mean, result.coverage_95[0])
        self.assertLess(result.mean, result.coverage_95[1])

    def test_sensitivity_indices(self):
        """Test sensitivity analysis"""
        sim = MonteCarloSimulator(n_samples=50000, seed=42)
        sim.add_input(MCInputVariable("x", 100.0, 10.0))  # High uncertainty
        sim.add_input(MCInputVariable("y", 50.0, 1.0))    # Low uncertainty

        def model(x, y):
            return x + y

        result = sim.run(model)

        # x should have higher sensitivity than y
        self.assertGreater(
            result.sensitivity_indices.get("x", 0),
            result.sensitivity_indices.get("y", 0)
        )

    def test_gum_mc_comparison(self):
        """Test GUM vs Monte Carlo comparison function"""
        gum_u = 10.0

        sim = MonteCarloSimulator(n_samples=100000, seed=42)
        sim.add_input(MCInputVariable("x", 100.0, 10.0))
        mc_result = sim.run(lambda x: x)

        comparison = compare_gum_monte_carlo(gum_u, mc_result)

        # Should be compatible for normal distribution
        self.assertIn("compatible", comparison)


# =============================================================================
# Unit Tests - Standards Compliance
# =============================================================================

class TestSimulatorClassification(unittest.TestCase):
    """Test IEC 60904-9 simulator classification"""

    def test_classification_limits(self):
        """Test classification against known limits"""
        checker = IEC_60904_9_Checker()

        for unif, temp, exp_u, exp_t in ReferenceData.SIMULATOR_CLASSIFICATION_CASES:
            result = checker.classify_simulator(
                uniformity_front=unif,
                temporal_front=temp,
                spectral_ratios={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0}
            )
            if exp_u != "Fail":
                self.assertEqual(result.uniformity_class, exp_u,
                                msg=f"Uniformity {unif}% should be class {exp_u}")

    def test_aaa_plus_classification(self):
        """Test AAA+ classification (better than AAA)"""
        checker = IEC_60904_9_Checker()
        result = checker.classify_simulator(
            uniformity_front=1.0,  # Better than 2%
            temporal_front=0.3,   # Better than 2%
            spectral_ratios={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0}
        )
        self.assertEqual(result.overall_class, "AAA")
        self.assertTrue(result.compliant)


class TestBifacialCompliance(unittest.TestCase):
    """Test IEC TS 60904-1-2:2024 compliance"""

    def test_front_irradiance_compliance(self):
        """Test front irradiance check"""
        checker = IEC_TS_60904_1_2_Checker()

        # Within tolerance
        check = checker.check_front_irradiance(1000.0, 1000.0)
        self.assertEqual(check.status, ComplianceStatus.PASS)

        # Outside tolerance
        check = checker.check_front_irradiance(950.0, 1000.0)
        self.assertEqual(check.status, ComplianceStatus.FAIL)

    def test_dark_side_requirement(self):
        """Test dark side irradiance requirement"""
        checker = IEC_TS_60904_1_2_Checker()

        # Below limit
        check = checker.check_rear_irradiance_dark(0.05)
        self.assertEqual(check.status, ComplianceStatus.PASS)

        # Above limit
        check = checker.check_rear_irradiance_dark(0.5)
        self.assertEqual(check.status, ComplianceStatus.FAIL)

    def test_temperature_compliance(self):
        """Test temperature check"""
        checker = IEC_TS_60904_1_2_Checker()

        # Within tolerance
        check = checker.check_temperature(25.5, 25.0)
        self.assertEqual(check.status, ComplianceStatus.PASS)

        # Outside tolerance
        check = checker.check_temperature(27.0, 25.0)
        self.assertEqual(check.status, ComplianceStatus.FAIL)

    def test_bifaciality_factor_range(self):
        """Test bifaciality factor range check"""
        checker = IEC_TS_60904_1_2_Checker()

        # Normal range
        check = checker.check_bifaciality_factor_range(0.75, "Pmax")
        self.assertEqual(check.status, ComplianceStatus.PASS)

        # Too low
        check = checker.check_bifaciality_factor_range(0.2, "Pmax")
        self.assertEqual(check.status, ComplianceStatus.FAIL)

    def test_full_compliance_check(self):
        """Test full bifacial compliance check"""
        checker = IEC_TS_60904_1_2_Checker()
        report = checker.run_full_bifacial_compliance(
            g_front=1000.0,
            g_rear=150.0,
            temperature=25.0,
            phi_isc=0.75,
            phi_pmax=0.70,
            phi_voc=0.99
        )

        self.assertGreater(report.total_checks, 0)
        self.assertGreater(report.compliance_percentage, 0)


class TestGUMCompliance(unittest.TestCase):
    """Test GUM methodology compliance"""

    def test_uncertainty_sources_check(self):
        """Test uncertainty sources completeness"""
        checker = GUM_Compliance_Checker()

        # Complete sources
        check = checker.check_uncertainty_sources_complete([
            "reference_calibration",
            "reference_stability",
            "uniformity",
            "temporal_instability",
            "spectral_mismatch",
            "temperature_sensor",
            "temperature_correction",
            "voltage_measurement",
            "current_measurement",
            "repeatability"
        ])
        self.assertEqual(check.status, ComplianceStatus.PASS)

    def test_coverage_factor_check(self):
        """Test coverage factor appropriateness"""
        checker = GUM_Compliance_Checker()

        # k=2 with large DOF
        check = checker.check_coverage_factor(2.0, 50)
        self.assertEqual(check.status, ComplianceStatus.PASS)


# =============================================================================
# Integration Tests
# =============================================================================

class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with monofacial calculations"""

    def test_monofacial_stc_calculation(self):
        """Test standard monofacial STC calculation still works"""
        builder = UncertaintyBudgetBuilder(measurement_type="stc", is_bifacial=False)
        budget = builder.build_standard_stc()

        # Should not include bifacial components
        bifacial_comps = [c for c in budget.components.keys()
                        if "rear" in c.lower() or "phi" in c.lower()]
        self.assertEqual(len(bifacial_comps), 0)

    def test_monofacial_uncertainty_values(self):
        """Test monofacial uncertainty values are reasonable"""
        builder = UncertaintyBudgetBuilder(measurement_type="stc", is_bifacial=False)
        budget = builder.build_standard_stc()

        # Combined uncertainty should be in typical range (1.5-3%)
        self.assertGreater(budget.combined_standard_uncertainty, 1.0)
        self.assertLess(budget.combined_standard_uncertainty, 5.0)

    def test_pasan_monofacial_configuration(self):
        """Test PASAN configuration for monofacial measurement"""
        builder = UncertaintyBudgetBuilder()
        builder.set_simulator("pasan_highlight_led")

        self.assertFalse(builder.simulator.bifacial_capable)
        self.assertEqual(builder.simulator.classification, SimulatorClassification.AAA_PLUS)


class TestCompleteWorkflow(unittest.TestCase):
    """Test complete measurement workflows"""

    def test_bifacial_measurement_workflow(self):
        """Test complete bifacial measurement workflow"""
        # 1. Create module
        module = BifacialModule(
            manufacturer="Test",
            model="BF-400",
            pmax_front=400.0,
            pmax_rear=280.0,
            isc_front=10.0,
            isc_rear=7.5
        )

        # 2. Create analyzer
        analyzer = BifacialMeasurementAnalyzer()

        # 3. Run analysis
        result = analyzer.analyze_bifacial_stc_measurement(
            module=module,
            g_front=900.0,
            g_rear=135.0,
            power_measured=440.0,
            temperature=25.0
        )

        # 4. Verify results
        self.assertIn("equivalent_irradiance", result)
        self.assertIn("power_uncertainty", result)
        self.assertIn("bifacial_gain", result)

    def test_monte_carlo_gum_agreement(self):
        """Test that Monte Carlo and GUM give similar results"""
        # Build uncertainty budget (GUM)
        builder = UncertaintyBudgetBuilder()
        budget = builder.set_simulator("pasan_highlight_bifacial").build_standard_stc()
        gum_u = budget.combined_standard_uncertainty

        # Run Monte Carlo with same inputs
        sim = MonteCarloSimulator(n_samples=100000, seed=42)
        sim.add_input(MCInputVariable("power", 100.0, gum_u))

        result = sim.run(lambda power: power)

        # Results should be within 10%
        rel_diff = abs(result.std - gum_u) / gum_u
        self.assertLess(rel_diff, 0.1)


# =============================================================================
# Reference Test Cases from IEC Standards
# =============================================================================

class TestIECReferenceCases(unittest.TestCase):
    """Test cases derived from IEC standard examples"""

    def test_equivalent_irradiance_example(self):
        """
        Reference: IEC TS 60904-1-2:2024 Section 6

        Example calculation for equivalent irradiance
        """
        calculator = BifacialUncertaintyCalculator()

        # Example: φ = 0.70, R = 0.15 (rear/front ratio)
        phi = 0.70
        g_front = 1000.0 / (1 + phi * 0.15)  # ≈ 905 W/m²
        g_rear = 0.15 * g_front              # ≈ 135.7 W/m²

        result = calculator.calculate_equivalent_irradiance(
            g_front=g_front,
            g_rear=g_rear,
            phi=phi,
            u_g_front=g_front * 0.02,
            u_g_rear=g_rear * 0.03,
            u_phi=phi * 0.02
        )

        # G_eq should equal 1000 W/m²
        self.assertAlmostEqual(result.g_equivalent, 1000.0, delta=1.0)

    def test_bifaciality_factor_example(self):
        """
        Reference: IEC TS 60904-1-2:2024 Section 7

        Example bifaciality factor calculation
        """
        calculator = BifacialUncertaintyCalculator()

        # Example: HJT cell with high bifaciality
        result = calculator.calculate_bifaciality_factors(
            front_isc=10.0,
            front_voc=0.72,
            front_pmax=7.2,
            front_ff=1.0,  # Normalized
            rear_isc=9.2,   # φ_Isc = 0.92
            rear_voc=0.715,
            rear_pmax=6.48,
            rear_ff=1.0
        )

        self.assertAlmostEqual(result.phi_isc, 0.92, places=2)


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance(unittest.TestCase):
    """Performance and stress tests"""

    def test_large_monte_carlo(self):
        """Test large Monte Carlo simulation completes in reasonable time"""
        import time

        sim = MonteCarloSimulator(n_samples=1000000, seed=42)
        sim.add_input(MCInputVariable("x", 100.0, 10.0))
        sim.add_input(MCInputVariable("y", 50.0, 5.0))
        sim.add_input(MCInputVariable("z", 25.0, 2.5))

        start = time.time()
        result = sim.run(lambda x, y, z: x + y + z)
        elapsed = time.time() - start

        # Should complete in < 10 seconds
        self.assertLess(elapsed, 10.0)
        self.assertEqual(result.n_samples, 1000000)

    def test_budget_builder_performance(self):
        """Test budget builder with many components"""
        import time

        builder = UncertaintyBudgetBuilder()

        start = time.time()
        for i in range(100):
            builder.add_component(UncertaintyComponent(
                name=f"test_component_{i}",
                uncertainty=1.0
            ))
        budget = builder.build()
        elapsed = time.time() - start

        # Should complete in < 1 second
        self.assertLess(elapsed, 1.0)
        self.assertEqual(len(budget.components), 100)


# =============================================================================
# Validation Tests
# =============================================================================

class TestValidation(unittest.TestCase):
    """Input validation tests"""

    def test_negative_irradiance_handling(self):
        """Test handling of negative irradiance values"""
        calculator = BifacialUncertaintyCalculator()

        # Should handle gracefully or raise appropriate error
        with self.assertRaises((ValueError, ZeroDivisionError)):
            calculator.calculate_bifaciality_factors(
                front_isc=-10.0,  # Invalid
                front_voc=45.0,
                front_pmax=400.0,
                front_ff=0.85,
                rear_isc=7.5,
                rear_voc=44.8,
                rear_pmax=280.0,
                rear_ff=0.84
            )

    def test_zero_uncertainty_handling(self):
        """Test handling of zero uncertainty"""
        comp = UncertaintyComponent(
            name="test",
            uncertainty=0.0,
            distribution=DistributionType.NORMAL
        )
        self.assertEqual(comp.standard_uncertainty, 0.0)
        self.assertEqual(comp.variance_contribution, 0.0)


# =============================================================================
# Test Runner
# =============================================================================

def run_tests(verbosity: int = 2) -> bool:
    """
    Run all tests and return success status.

    Args:
        verbosity: Test output verbosity (0, 1, or 2)

    Returns:
        True if all tests passed
    """
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestBifacialityFactors,
        TestEquivalentIrradiance,
        TestUncertaintyComponents,
        TestUncertaintyBudgetBuilder,
        TestSimulatorDatabase,
        TestMonteCarloSimulation,
        TestSimulatorClassification,
        TestBifacialCompliance,
        TestGUMCompliance,
        TestBackwardCompatibility,
        TestCompleteWorkflow,
        TestIECReferenceCases,
        TestPerformance,
        TestValidation,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 70)

    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    # Run with verbosity=2 for detailed output
    success = run_tests(verbosity=2)
    sys.exit(0 if success else 1)
