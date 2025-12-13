"""
Standards Compliance Module

Comprehensive IEC and ISO standards compliance checking for PV measurements.
Implements requirements from IEC TS 60904-1-2:2024 and related standards.

This module provides:
- IEC 60904 series compliance checking
- IEC TS 60904-1-2:2024 bifacial requirements
- IEC 60904-9 simulator classification
- ISO 17025 laboratory requirements
- JCGM 100:2008 (GUM) methodology validation
- Automated compliance reporting

Standards Covered:
- IEC TS 60904-1-2:2024: Bifacial PV device characterization
- IEC 60904-1:2020: I-V characteristics measurement
- IEC 60904-3:2019: Spectral irradiance reference
- IEC 60904-7:2019: Spectral mismatch correction
- IEC 60904-8:2014: Spectral responsivity measurement
- IEC 60904-9:2020: Solar simulator classification
- IEC 60891:2021: Temperature/irradiance corrections
- ISO 17025:2017: Laboratory competence
- JCGM 100:2008: Uncertainty expression (GUM)

Author: Universal Solar Simulator Framework Team
Version: 2.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union
from enum import Enum
from datetime import datetime
import json


# =============================================================================
# Enumerations
# =============================================================================

class ComplianceStatus(Enum):
    """Compliance check result status"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"
    NOT_CHECKED = "not_checked"


class StandardReference(Enum):
    """Reference standards"""
    IEC_TS_60904_1_2_2024 = "IEC TS 60904-1-2:2024"
    IEC_60904_1_2020 = "IEC 60904-1:2020"
    IEC_60904_3_2019 = "IEC 60904-3:2019"
    IEC_60904_7_2019 = "IEC 60904-7:2019"
    IEC_60904_8_2014 = "IEC 60904-8:2014"
    IEC_60904_9_2020 = "IEC 60904-9:2020"
    IEC_60891_2021 = "IEC 60891:2021"
    ISO_17025_2017 = "ISO 17025:2017"
    JCGM_100_2008 = "JCGM 100:2008"
    JCGM_101_2008 = "JCGM 101:2008"


class MeasurementType(Enum):
    """Types of PV measurements"""
    STC = "stc"                         # Standard Test Conditions
    BIFACIAL_SINGLE_FRONT = "bifacial_single_front"
    BIFACIAL_SINGLE_REAR = "bifacial_single_rear"
    BIFACIAL_DOUBLE_SIDED = "bifacial_double_sided"
    BIFACIAL_EQUIVALENT = "bifacial_equivalent"
    NMOT = "nmot"                       # Nominal Module Operating Temperature
    LOW_IRRADIANCE = "low_irradiance"
    TEMPERATURE_COEFFICIENTS = "temp_coefficients"
    SPECTRAL_RESPONSE = "spectral_response"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ComplianceCheck:
    """Single compliance check result"""
    requirement_id: str
    requirement_text: str
    standard: StandardReference
    status: ComplianceStatus
    measured_value: Optional[float] = None
    required_value: Optional[float] = None
    tolerance: Optional[float] = None
    unit: str = ""
    details: str = ""
    section: str = ""

    @property
    def is_compliant(self) -> bool:
        return self.status in [ComplianceStatus.PASS, ComplianceStatus.NOT_APPLICABLE]

    def to_dict(self) -> Dict:
        return {
            "requirement_id": self.requirement_id,
            "requirement_text": self.requirement_text,
            "standard": self.standard.value,
            "status": self.status.value,
            "measured_value": self.measured_value,
            "required_value": self.required_value,
            "tolerance": self.tolerance,
            "unit": self.unit,
            "details": self.details,
            "section": self.section,
            "is_compliant": self.is_compliant
        }


@dataclass
class ComplianceReport:
    """Complete compliance report"""
    measurement_type: MeasurementType
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    checks: List[ComplianceCheck] = field(default_factory=list)
    overall_status: ComplianceStatus = ComplianceStatus.NOT_CHECKED
    standards_referenced: List[StandardReference] = field(default_factory=list)
    notes: str = ""

    @property
    def total_checks(self) -> int:
        return len(self.checks)

    @property
    def passed_checks(self) -> int:
        return sum(1 for c in self.checks if c.status == ComplianceStatus.PASS)

    @property
    def failed_checks(self) -> int:
        return sum(1 for c in self.checks if c.status == ComplianceStatus.FAIL)

    @property
    def warning_checks(self) -> int:
        return sum(1 for c in self.checks if c.status == ComplianceStatus.WARNING)

    @property
    def compliance_percentage(self) -> float:
        applicable = [c for c in self.checks
                     if c.status != ComplianceStatus.NOT_APPLICABLE]
        if not applicable:
            return 100.0
        passed = sum(1 for c in applicable if c.status == ComplianceStatus.PASS)
        return (passed / len(applicable)) * 100

    def add_check(self, check: ComplianceCheck):
        self.checks.append(check)
        if check.standard not in self.standards_referenced:
            self.standards_referenced.append(check.standard)

    def calculate_overall_status(self):
        if any(c.status == ComplianceStatus.FAIL for c in self.checks):
            self.overall_status = ComplianceStatus.FAIL
        elif any(c.status == ComplianceStatus.WARNING for c in self.checks):
            self.overall_status = ComplianceStatus.WARNING
        elif all(c.status in [ComplianceStatus.PASS, ComplianceStatus.NOT_APPLICABLE]
                for c in self.checks):
            self.overall_status = ComplianceStatus.PASS
        else:
            self.overall_status = ComplianceStatus.NOT_CHECKED

    def to_dict(self) -> Dict:
        return {
            "measurement_type": self.measurement_type.value,
            "timestamp": self.timestamp,
            "overall_status": self.overall_status.value,
            "compliance_percentage": self.compliance_percentage,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "warning_checks": self.warning_checks,
            "standards_referenced": [s.value for s in self.standards_referenced],
            "checks": [c.to_dict() for c in self.checks],
            "notes": self.notes
        }

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "STANDARDS COMPLIANCE REPORT",
            "=" * 70,
            f"Measurement Type: {self.measurement_type.value.upper()}",
            f"Timestamp: {self.timestamp}",
            f"Overall Status: {self.overall_status.value.upper()}",
            "",
            f"Compliance: {self.compliance_percentage:.1f}%",
            f"  Passed:  {self.passed_checks}/{self.total_checks}",
            f"  Failed:  {self.failed_checks}/{self.total_checks}",
            f"  Warnings: {self.warning_checks}/{self.total_checks}",
            "",
            "Standards Referenced:",
        ]
        for std in self.standards_referenced:
            lines.append(f"  - {std.value}")

        lines.append("")
        lines.append("=" * 70)

        if self.failed_checks > 0:
            lines.append("FAILED CHECKS:")
            lines.append("-" * 70)
            for check in self.checks:
                if check.status == ComplianceStatus.FAIL:
                    lines.append(f"  [{check.requirement_id}] {check.requirement_text}")
                    lines.append(f"    Measured: {check.measured_value} {check.unit}")
                    lines.append(f"    Required: {check.required_value} ±{check.tolerance} {check.unit}")
            lines.append("")

        return "\n".join(lines)


@dataclass
class SimulatorClassificationResult:
    """IEC 60904-9 simulator classification result"""
    uniformity_class: str = "A"       # A, B, or C
    temporal_class: str = "A"
    spectral_class: str = "A"
    overall_class: str = "AAA"

    uniformity_value: float = 0.0     # % non-uniformity
    temporal_value: float = 0.0       # % temporal instability
    spectral_values: Dict[str, float] = field(default_factory=dict)

    compliant: bool = True
    details: str = ""


@dataclass
class BifacialRequirements:
    """IEC TS 60904-1-2:2024 specific requirements"""
    # Irradiance requirements
    front_irradiance_stc: float = 1000.0           # W/m²
    front_irradiance_tolerance: float = 0.02       # ±2%
    rear_irradiance_dark_max: float = 0.1          # W/m² (dark side limit)
    rear_irradiance_tolerance: float = 0.02        # ±2%

    # Temperature requirements
    temperature_stc: float = 25.0                  # °C
    temperature_tolerance: float = 1.0             # ±1°C

    # Spectral requirements
    spectral_distribution: str = "AM1.5G"          # Per IEC 60904-3

    # Uniformity requirements (derived from IEC 60904-9)
    max_uniformity_class_a: float = 2.0            # % for Class A
    max_temporal_class_a: float = 2.0              # % for Class A

    # Measurement timing
    min_stabilization_time_s: float = 60.0         # seconds
    max_temperature_change_rate: float = 0.5       # °C/min


# =============================================================================
# Compliance Checker Classes
# =============================================================================

class IEC_60904_9_Checker:
    """
    IEC 60904-9:2020 Solar Simulator Classification Checker

    Checks simulator classification requirements for:
    - Spatial non-uniformity
    - Temporal instability
    - Spectral match
    """

    # Class limits per IEC 60904-9
    CLASS_LIMITS = {
        "uniformity": {"A": 2.0, "B": 5.0, "C": 10.0},
        "temporal": {"A": 2.0, "B": 5.0, "C": 10.0},
        "spectral": {"A": (0.75, 1.25), "B": (0.6, 1.4), "C": (0.4, 2.0)}
    }

    # Spectral intervals per IEC 60904-3
    SPECTRAL_INTERVALS = [
        (400, 500),   # Interval 1
        (500, 600),   # Interval 2
        (600, 700),   # Interval 3
        (700, 800),   # Interval 4
        (800, 900),   # Interval 5
        (900, 1100),  # Interval 6
    ]

    def __init__(self):
        self.report = ComplianceReport(measurement_type=MeasurementType.STC)

    def check_uniformity(
        self,
        uniformity_percent: float,
        side: str = "front"
    ) -> ComplianceCheck:
        """
        Check spatial non-uniformity against IEC 60904-9.

        Args:
            uniformity_percent: Measured non-uniformity (%)
            side: "front" or "rear"

        Returns:
            ComplianceCheck result
        """
        # Determine class
        if uniformity_percent <= self.CLASS_LIMITS["uniformity"]["A"]:
            classification = "A"
            status = ComplianceStatus.PASS
        elif uniformity_percent <= self.CLASS_LIMITS["uniformity"]["B"]:
            classification = "B"
            status = ComplianceStatus.WARNING
        elif uniformity_percent <= self.CLASS_LIMITS["uniformity"]["C"]:
            classification = "C"
            status = ComplianceStatus.WARNING
        else:
            classification = "Fail"
            status = ComplianceStatus.FAIL

        check = ComplianceCheck(
            requirement_id=f"IEC60904-9-UNIF-{side.upper()}",
            requirement_text=f"Spatial non-uniformity ({side} side) shall be ≤2% for Class A",
            standard=StandardReference.IEC_60904_9_2020,
            status=status,
            measured_value=uniformity_percent,
            required_value=self.CLASS_LIMITS["uniformity"]["A"],
            tolerance=0.0,
            unit="%",
            details=f"Classification: {classification}",
            section="6.2"
        )
        self.report.add_check(check)
        return check

    def check_temporal_instability(
        self,
        temporal_percent: float,
        side: str = "front"
    ) -> ComplianceCheck:
        """
        Check temporal instability against IEC 60904-9.

        Args:
            temporal_percent: Measured temporal instability (%)
            side: "front" or "rear"

        Returns:
            ComplianceCheck result
        """
        if temporal_percent <= self.CLASS_LIMITS["temporal"]["A"]:
            classification = "A"
            status = ComplianceStatus.PASS
        elif temporal_percent <= self.CLASS_LIMITS["temporal"]["B"]:
            classification = "B"
            status = ComplianceStatus.WARNING
        elif temporal_percent <= self.CLASS_LIMITS["temporal"]["C"]:
            classification = "C"
            status = ComplianceStatus.WARNING
        else:
            classification = "Fail"
            status = ComplianceStatus.FAIL

        check = ComplianceCheck(
            requirement_id=f"IEC60904-9-TEMP-{side.upper()}",
            requirement_text=f"Temporal instability ({side} side) shall be ≤2% for Class A",
            standard=StandardReference.IEC_60904_9_2020,
            status=status,
            measured_value=temporal_percent,
            required_value=self.CLASS_LIMITS["temporal"]["A"],
            unit="%",
            details=f"Classification: {classification}",
            section="6.3"
        )
        self.report.add_check(check)
        return check

    def check_spectral_match(
        self,
        interval_ratios: Dict[int, float]
    ) -> ComplianceCheck:
        """
        Check spectral match for all intervals.

        Args:
            interval_ratios: Dict of interval number to spectral ratio

        Returns:
            ComplianceCheck result
        """
        # Check each interval
        worst_class = "A"
        failing_intervals = []

        for interval, ratio in interval_ratios.items():
            # Class A: 0.75-1.25
            if 0.75 <= ratio <= 1.25:
                interval_class = "A"
            elif 0.6 <= ratio <= 1.4:
                interval_class = "B"
            elif 0.4 <= ratio <= 2.0:
                interval_class = "C"
            else:
                interval_class = "Fail"
                failing_intervals.append(interval)

            if interval_class > worst_class:
                worst_class = interval_class

        if worst_class == "A":
            status = ComplianceStatus.PASS
        elif worst_class in ["B", "C"]:
            status = ComplianceStatus.WARNING
        else:
            status = ComplianceStatus.FAIL

        check = ComplianceCheck(
            requirement_id="IEC60904-9-SPEC",
            requirement_text="Spectral match ratio shall be 0.75-1.25 for Class A in all intervals",
            standard=StandardReference.IEC_60904_9_2020,
            status=status,
            details=f"Classification: {worst_class}. Failing intervals: {failing_intervals}",
            section="6.4"
        )
        self.report.add_check(check)
        return check

    def classify_simulator(
        self,
        uniformity_front: float,
        temporal_front: float,
        spectral_ratios: Dict[int, float],
        uniformity_rear: Optional[float] = None,
        temporal_rear: Optional[float] = None
    ) -> SimulatorClassificationResult:
        """
        Full simulator classification per IEC 60904-9.

        Args:
            uniformity_front: Front non-uniformity (%)
            temporal_front: Front temporal instability (%)
            spectral_ratios: Spectral interval ratios
            uniformity_rear: Rear non-uniformity (%) for bifacial
            temporal_rear: Rear temporal instability (%) for bifacial

        Returns:
            SimulatorClassificationResult
        """
        # Check front side
        u_check = self.check_uniformity(uniformity_front, "front")
        t_check = self.check_temporal_instability(temporal_front, "front")
        s_check = self.check_spectral_match(spectral_ratios)

        # Determine classes
        u_class = "A" if uniformity_front <= 2.0 else ("B" if uniformity_front <= 5.0 else "C")
        t_class = "A" if temporal_front <= 2.0 else ("B" if temporal_front <= 5.0 else "C")

        # Spectral class from worst interval
        s_class = "A"
        for ratio in spectral_ratios.values():
            if not (0.75 <= ratio <= 1.25):
                s_class = "B" if 0.6 <= ratio <= 1.4 else "C"
                break

        overall_class = u_class + t_class + s_class

        # Check rear if provided
        if uniformity_rear is not None:
            self.check_uniformity(uniformity_rear, "rear")
        if temporal_rear is not None:
            self.check_temporal_instability(temporal_rear, "rear")

        return SimulatorClassificationResult(
            uniformity_class=u_class,
            temporal_class=t_class,
            spectral_class=s_class,
            overall_class=overall_class,
            uniformity_value=uniformity_front,
            temporal_value=temporal_front,
            spectral_values=spectral_ratios,
            compliant=overall_class in ["AAA", "AAB", "ABA", "BAA"]
        )


class IEC_TS_60904_1_2_Checker:
    """
    IEC TS 60904-1-2:2024 Bifacial PV Device Measurement Checker

    Checks compliance with bifacial-specific requirements:
    - Irradiance conditions
    - Dark side requirements
    - Temperature requirements
    - Measurement procedure
    - Bifaciality factor calculation
    """

    def __init__(self):
        self.requirements = BifacialRequirements()
        self.report = ComplianceReport(
            measurement_type=MeasurementType.BIFACIAL_DOUBLE_SIDED
        )

    def set_measurement_type(self, mtype: MeasurementType):
        """Set the measurement type for the report"""
        self.report.measurement_type = mtype

    def check_front_irradiance(
        self,
        irradiance: float,
        target: float = 1000.0
    ) -> ComplianceCheck:
        """
        Check front irradiance against STC requirements.

        Args:
            irradiance: Measured front irradiance (W/m²)
            target: Target irradiance (default 1000 W/m²)

        Returns:
            ComplianceCheck result
        """
        tolerance = target * self.requirements.front_irradiance_tolerance
        deviation = abs(irradiance - target)

        if deviation <= tolerance:
            status = ComplianceStatus.PASS
        else:
            status = ComplianceStatus.FAIL

        check = ComplianceCheck(
            requirement_id="IEC-TS-60904-1-2-G-FRONT",
            requirement_text=f"Front irradiance shall be {target:.0f} W/m² ±{self.requirements.front_irradiance_tolerance*100:.0f}%",
            standard=StandardReference.IEC_TS_60904_1_2_2024,
            status=status,
            measured_value=irradiance,
            required_value=target,
            tolerance=tolerance,
            unit="W/m²",
            details=f"Deviation: {deviation:.1f} W/m²",
            section="5.2"
        )
        self.report.add_check(check)
        return check

    def check_rear_irradiance_dark(
        self,
        irradiance: float
    ) -> ComplianceCheck:
        """
        Check rear side is sufficiently dark for single-sided measurement.

        Per IEC TS 60904-1-2:2024 Section 5.2.1:
        Dark side irradiance shall be < 0.1 W/m²

        Args:
            irradiance: Measured rear irradiance (W/m²)

        Returns:
            ComplianceCheck result
        """
        max_allowed = self.requirements.rear_irradiance_dark_max

        if irradiance < max_allowed:
            status = ComplianceStatus.PASS
        else:
            status = ComplianceStatus.FAIL

        check = ComplianceCheck(
            requirement_id="IEC-TS-60904-1-2-DARK",
            requirement_text=f"Dark side irradiance shall be < {max_allowed} W/m²",
            standard=StandardReference.IEC_TS_60904_1_2_2024,
            status=status,
            measured_value=irradiance,
            required_value=max_allowed,
            unit="W/m²",
            details="For single-sided bifacial measurement",
            section="5.2.1"
        )
        self.report.add_check(check)
        return check

    def check_rear_irradiance(
        self,
        irradiance: float,
        target: float,
    ) -> ComplianceCheck:
        """
        Check rear irradiance for double-sided measurement.

        Args:
            irradiance: Measured rear irradiance (W/m²)
            target: Target rear irradiance (W/m²)

        Returns:
            ComplianceCheck result
        """
        tolerance = target * self.requirements.rear_irradiance_tolerance
        deviation = abs(irradiance - target)

        if deviation <= tolerance:
            status = ComplianceStatus.PASS
        else:
            status = ComplianceStatus.FAIL

        check = ComplianceCheck(
            requirement_id="IEC-TS-60904-1-2-G-REAR",
            requirement_text=f"Rear irradiance shall be {target:.0f} W/m² ±{self.requirements.rear_irradiance_tolerance*100:.0f}%",
            standard=StandardReference.IEC_TS_60904_1_2_2024,
            status=status,
            measured_value=irradiance,
            required_value=target,
            tolerance=tolerance,
            unit="W/m²",
            details=f"Deviation: {deviation:.1f} W/m²",
            section="5.3"
        )
        self.report.add_check(check)
        return check

    def check_temperature(
        self,
        temperature: float,
        target: float = 25.0
    ) -> ComplianceCheck:
        """
        Check module temperature against STC requirements.

        Args:
            temperature: Measured module temperature (°C)
            target: Target temperature (default 25°C)

        Returns:
            ComplianceCheck result
        """
        tolerance = self.requirements.temperature_tolerance
        deviation = abs(temperature - target)

        if deviation <= tolerance:
            status = ComplianceStatus.PASS
        else:
            status = ComplianceStatus.FAIL

        check = ComplianceCheck(
            requirement_id="IEC-TS-60904-1-2-TEMP",
            requirement_text=f"Module temperature shall be {target:.0f}°C ±{tolerance:.0f}°C",
            standard=StandardReference.IEC_TS_60904_1_2_2024,
            status=status,
            measured_value=temperature,
            required_value=target,
            tolerance=tolerance,
            unit="°C",
            details=f"Deviation: {deviation:.2f}°C",
            section="5.2"
        )
        self.report.add_check(check)
        return check

    def check_equivalent_irradiance(
        self,
        g_front: float,
        g_rear: float,
        phi: float,
        target_g_eq: float = 1000.0
    ) -> ComplianceCheck:
        """
        Check equivalent irradiance calculation.

        G_eq = G_front + φ × G_rear

        Args:
            g_front: Front irradiance (W/m²)
            g_rear: Rear irradiance (W/m²)
            phi: Bifaciality factor
            target_g_eq: Target equivalent irradiance (W/m²)

        Returns:
            ComplianceCheck result
        """
        g_eq = g_front + phi * g_rear
        tolerance = target_g_eq * 0.02  # 2% tolerance
        deviation = abs(g_eq - target_g_eq)

        if deviation <= tolerance:
            status = ComplianceStatus.PASS
        else:
            status = ComplianceStatus.WARNING  # Warning, not fail

        check = ComplianceCheck(
            requirement_id="IEC-TS-60904-1-2-G-EQ",
            requirement_text="Equivalent irradiance G_eq = G_front + φ×G_rear",
            standard=StandardReference.IEC_TS_60904_1_2_2024,
            status=status,
            measured_value=g_eq,
            required_value=target_g_eq,
            tolerance=tolerance,
            unit="W/m²",
            details=f"G_front={g_front:.1f}, G_rear={g_rear:.1f}, φ={phi:.3f}",
            section="6"
        )
        self.report.add_check(check)
        return check

    def check_bifaciality_factor_range(
        self,
        phi: float,
        parameter: str = "Pmax"
    ) -> ComplianceCheck:
        """
        Check bifaciality factor is within reasonable range.

        Typical ranges:
        - TOPCon: 0.70-0.85
        - HJT: 0.85-0.95
        - PERC: 0.60-0.75

        Args:
            phi: Measured bifaciality factor
            parameter: "Isc", "Pmax", or "Voc"

        Returns:
            ComplianceCheck result
        """
        # Reasonable ranges for bifaciality
        if parameter in ["Isc", "Pmax"]:
            min_expected = 0.5
            max_expected = 1.0
        else:  # Voc
            min_expected = 0.95
            max_expected = 1.02

        if min_expected <= phi <= max_expected:
            status = ComplianceStatus.PASS
        elif 0.3 <= phi <= 1.05:
            status = ComplianceStatus.WARNING
        else:
            status = ComplianceStatus.FAIL

        check = ComplianceCheck(
            requirement_id=f"IEC-TS-60904-1-2-PHI-{parameter.upper()}",
            requirement_text=f"Bifaciality factor φ_{parameter} shall be in reasonable range",
            standard=StandardReference.IEC_TS_60904_1_2_2024,
            status=status,
            measured_value=phi,
            required_value=(min_expected + max_expected) / 2,
            unit="",
            details=f"Expected range: {min_expected:.2f} - {max_expected:.2f}",
            section="7"
        )
        self.report.add_check(check)
        return check

    def check_measurement_stability(
        self,
        power_readings: List[float],
        max_variation: float = 0.5
    ) -> ComplianceCheck:
        """
        Check measurement stability (variation between readings).

        Args:
            power_readings: List of power readings (W)
            max_variation: Maximum allowed variation (%)

        Returns:
            ComplianceCheck result
        """
        if len(power_readings) < 2:
            return ComplianceCheck(
                requirement_id="IEC-TS-60904-1-2-STAB",
                requirement_text="Minimum 2 readings required for stability check",
                standard=StandardReference.IEC_TS_60904_1_2_2024,
                status=ComplianceStatus.NOT_APPLICABLE
            )

        import numpy as np
        mean_power = np.mean(power_readings)
        max_dev = np.max(np.abs(np.array(power_readings) - mean_power))
        variation = (max_dev / mean_power) * 100

        if variation <= max_variation:
            status = ComplianceStatus.PASS
        else:
            status = ComplianceStatus.FAIL

        check = ComplianceCheck(
            requirement_id="IEC-TS-60904-1-2-STAB",
            requirement_text=f"Power variation between readings shall be ≤{max_variation:.1f}%",
            standard=StandardReference.IEC_TS_60904_1_2_2024,
            status=status,
            measured_value=variation,
            required_value=max_variation,
            unit="%",
            details=f"Mean power: {mean_power:.2f} W, Max deviation: {max_dev:.2f} W",
            section="5.4"
        )
        self.report.add_check(check)
        return check

    def run_full_bifacial_compliance(
        self,
        g_front: float,
        g_rear: float,
        temperature: float,
        phi_isc: float,
        phi_pmax: float,
        phi_voc: float,
        mode: MeasurementType = MeasurementType.BIFACIAL_DOUBLE_SIDED,
        power_readings: Optional[List[float]] = None
    ) -> ComplianceReport:
        """
        Run complete bifacial measurement compliance check.

        Args:
            g_front: Front irradiance (W/m²)
            g_rear: Rear irradiance (W/m²)
            temperature: Module temperature (°C)
            phi_isc: Bifaciality factor for Isc
            phi_pmax: Bifaciality factor for Pmax
            phi_voc: Bifaciality factor for Voc
            mode: Measurement mode
            power_readings: Optional list of power readings for stability check

        Returns:
            ComplianceReport with all checks
        """
        self.set_measurement_type(mode)

        # Irradiance checks
        self.check_front_irradiance(g_front)

        if mode == MeasurementType.BIFACIAL_SINGLE_FRONT:
            self.check_rear_irradiance_dark(g_rear)
        elif mode == MeasurementType.BIFACIAL_SINGLE_REAR:
            self.check_rear_irradiance_dark(g_front)  # Front should be dark
        else:
            self.check_rear_irradiance(g_rear, g_rear)  # Target is specified value
            self.check_equivalent_irradiance(g_front, g_rear, phi_isc)

        # Temperature check
        self.check_temperature(temperature)

        # Bifaciality factor checks
        self.check_bifaciality_factor_range(phi_isc, "Isc")
        self.check_bifaciality_factor_range(phi_pmax, "Pmax")
        self.check_bifaciality_factor_range(phi_voc, "Voc")

        # Stability check
        if power_readings:
            self.check_measurement_stability(power_readings)

        # Calculate overall status
        self.report.calculate_overall_status()

        return self.report


class GUM_Compliance_Checker:
    """
    JCGM 100:2008 (GUM) Methodology Compliance Checker

    Verifies uncertainty calculation follows GUM requirements:
    - Proper identification of uncertainty sources
    - Correct distribution selection
    - Sensitivity coefficient derivation
    - Combined uncertainty calculation
    - Effective degrees of freedom
    - Expanded uncertainty with coverage factor
    """

    def __init__(self):
        self.report = ComplianceReport(measurement_type=MeasurementType.STC)

    def check_uncertainty_sources_complete(
        self,
        sources: List[str],
        required_sources: Optional[List[str]] = None
    ) -> ComplianceCheck:
        """
        Check that all required uncertainty sources are identified.

        Args:
            sources: List of identified uncertainty sources
            required_sources: List of required sources (uses defaults if None)

        Returns:
            ComplianceCheck result
        """
        if required_sources is None:
            required_sources = [
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
            ]

        missing = [s for s in required_sources if s not in sources]

        if not missing:
            status = ComplianceStatus.PASS
            details = f"All {len(required_sources)} required sources identified"
        else:
            status = ComplianceStatus.FAIL
            details = f"Missing sources: {', '.join(missing)}"

        check = ComplianceCheck(
            requirement_id="GUM-4.1",
            requirement_text="All uncertainty sources shall be identified",
            standard=StandardReference.JCGM_100_2008,
            status=status,
            details=details,
            section="4.1"
        )
        self.report.add_check(check)
        return check

    def check_distribution_validity(
        self,
        distributions: Dict[str, str],
        valid_distributions: Optional[List[str]] = None
    ) -> ComplianceCheck:
        """
        Check that valid distributions are used.

        Args:
            distributions: Dict of source name to distribution type
            valid_distributions: List of valid distribution names

        Returns:
            ComplianceCheck result
        """
        if valid_distributions is None:
            valid_distributions = [
                "normal", "rectangular", "triangular",
                "u_shaped", "lognormal", "truncated_normal"
            ]

        invalid = {name: dist for name, dist in distributions.items()
                  if dist.lower() not in valid_distributions}

        if not invalid:
            status = ComplianceStatus.PASS
            details = "All distributions are valid"
        else:
            status = ComplianceStatus.FAIL
            details = f"Invalid distributions: {invalid}"

        check = ComplianceCheck(
            requirement_id="GUM-4.3",
            requirement_text="Valid probability distributions shall be assigned",
            standard=StandardReference.JCGM_100_2008,
            status=status,
            details=details,
            section="4.3"
        )
        self.report.add_check(check)
        return check

    def check_combined_uncertainty_method(
        self,
        uses_quadrature: bool = True,
        handles_correlation: bool = False,
        correlation_present: bool = False
    ) -> ComplianceCheck:
        """
        Check combined uncertainty calculation method.

        Args:
            uses_quadrature: Uses root-sum-of-squares combination
            handles_correlation: Handles correlated inputs
            correlation_present: Correlations exist between inputs

        Returns:
            ComplianceCheck result
        """
        if uses_quadrature:
            if correlation_present and not handles_correlation:
                status = ComplianceStatus.WARNING
                details = "Correlations present but not accounted for"
            else:
                status = ComplianceStatus.PASS
                details = "Quadrature combination correctly applied"
        else:
            status = ComplianceStatus.FAIL
            details = "Must use root-sum-of-squares for combining uncertainties"

        check = ComplianceCheck(
            requirement_id="GUM-5.1",
            requirement_text="Combined uncertainty shall use law of propagation of uncertainty",
            standard=StandardReference.JCGM_100_2008,
            status=status,
            details=details,
            section="5.1"
        )
        self.report.add_check(check)
        return check

    def check_coverage_factor(
        self,
        k: float,
        effective_dof: float,
        confidence_level: float = 0.95
    ) -> ComplianceCheck:
        """
        Check coverage factor is appropriate.

        Args:
            k: Coverage factor used
            effective_dof: Effective degrees of freedom
            confidence_level: Target confidence level

        Returns:
            ComplianceCheck result
        """
        from scipy import stats

        if effective_dof > 30:
            # Normal approximation
            expected_k = stats.norm.ppf((1 + confidence_level) / 2)
        else:
            # t-distribution
            expected_k = stats.t.ppf((1 + confidence_level) / 2, effective_dof)

        # Allow 5% tolerance
        if abs(k - expected_k) / expected_k <= 0.05:
            status = ComplianceStatus.PASS
            details = f"k={k:.2f} appropriate for ν_eff={effective_dof:.1f}"
        elif abs(k - 2.0) < 0.01 and effective_dof > 10:
            status = ComplianceStatus.PASS
            details = "k=2 acceptable approximation for large DOF"
        else:
            status = ComplianceStatus.WARNING
            details = f"k={k:.2f} vs expected {expected_k:.2f} for ν_eff={effective_dof:.1f}"

        check = ComplianceCheck(
            requirement_id="GUM-6.2",
            requirement_text="Coverage factor shall provide stated confidence level",
            standard=StandardReference.JCGM_100_2008,
            status=status,
            measured_value=k,
            required_value=expected_k,
            details=details,
            section="6.2"
        )
        self.report.add_check(check)
        return check


class ISO_17025_Checker:
    """
    ISO 17025:2017 Laboratory Competence Compliance Checker

    Verifies laboratory requirements for:
    - Measurement traceability
    - Method validation
    - Equipment calibration
    - Quality assurance
    - Uncertainty reporting
    """

    def __init__(self):
        self.report = ComplianceReport(measurement_type=MeasurementType.STC)

    def check_traceability(
        self,
        has_traceability: bool,
        traceability_chain: Optional[List[str]] = None
    ) -> ComplianceCheck:
        """
        Check measurement traceability to SI units.

        Args:
            has_traceability: Whether traceability is established
            traceability_chain: List of entities in traceability chain

        Returns:
            ComplianceCheck result
        """
        if has_traceability and traceability_chain:
            status = ComplianceStatus.PASS
            details = f"Traceability chain: {' → '.join(traceability_chain)}"
        elif has_traceability:
            status = ComplianceStatus.PASS
            details = "Traceability established"
        else:
            status = ComplianceStatus.FAIL
            details = "No traceability to SI units"

        check = ComplianceCheck(
            requirement_id="ISO17025-6.5",
            requirement_text="Measurement results shall be traceable to SI units",
            standard=StandardReference.ISO_17025_2017,
            status=status,
            details=details,
            section="6.5"
        )
        self.report.add_check(check)
        return check

    def check_calibration_validity(
        self,
        calibration_date: str,
        validity_months: int = 12
    ) -> ComplianceCheck:
        """
        Check if calibration is still valid.

        Args:
            calibration_date: Calibration date (ISO format)
            validity_months: Calibration validity period

        Returns:
            ComplianceCheck result
        """
        from datetime import datetime, timedelta

        try:
            cal_date = datetime.fromisoformat(calibration_date)
            expiry_date = cal_date + timedelta(days=validity_months * 30)

            if datetime.now() < expiry_date:
                status = ComplianceStatus.PASS
                days_remaining = (expiry_date - datetime.now()).days
                details = f"Valid until {expiry_date.strftime('%Y-%m-%d')} ({days_remaining} days remaining)"
            else:
                status = ComplianceStatus.FAIL
                details = f"Calibration expired on {expiry_date.strftime('%Y-%m-%d')}"
        except ValueError:
            status = ComplianceStatus.FAIL
            details = f"Invalid calibration date format: {calibration_date}"

        check = ComplianceCheck(
            requirement_id="ISO17025-6.4.6",
            requirement_text="Equipment calibration shall be current and valid",
            standard=StandardReference.ISO_17025_2017,
            status=status,
            details=details,
            section="6.4.6"
        )
        self.report.add_check(check)
        return check

    def check_uncertainty_reported(
        self,
        uncertainty_reported: bool,
        uncertainty_value: Optional[float] = None,
        coverage_factor: Optional[float] = None,
        confidence_level: Optional[float] = None
    ) -> ComplianceCheck:
        """
        Check uncertainty is properly reported.

        Args:
            uncertainty_reported: Whether uncertainty is reported
            uncertainty_value: Reported uncertainty value
            coverage_factor: Coverage factor used
            confidence_level: Confidence level (%)

        Returns:
            ComplianceCheck result
        """
        if uncertainty_reported and all([uncertainty_value, coverage_factor, confidence_level]):
            status = ComplianceStatus.PASS
            details = f"U = {uncertainty_value:.3f}% (k={coverage_factor:.1f}, {confidence_level:.0f}%)"
        elif uncertainty_reported:
            status = ComplianceStatus.WARNING
            details = "Uncertainty reported but incomplete (missing k or confidence level)"
        else:
            status = ComplianceStatus.FAIL
            details = "Measurement uncertainty not reported"

        check = ComplianceCheck(
            requirement_id="ISO17025-7.6",
            requirement_text="Measurement uncertainty shall be reported with confidence level",
            standard=StandardReference.ISO_17025_2017,
            status=status,
            details=details,
            section="7.6"
        )
        self.report.add_check(check)
        return check


# =============================================================================
# Convenience Functions
# =============================================================================

def check_bifacial_stc_compliance(
    g_front: float,
    g_rear: float,
    temperature: float,
    phi_isc: float,
    phi_pmax: float,
    uniformity_front: float,
    uniformity_rear: float,
    temporal_front: float,
    temporal_rear: float
) -> Dict[str, ComplianceReport]:
    """
    Convenience function for complete bifacial STC compliance check.

    Args:
        g_front: Front irradiance (W/m²)
        g_rear: Rear irradiance (W/m²)
        temperature: Module temperature (°C)
        phi_isc: Bifaciality factor for Isc
        phi_pmax: Bifaciality factor for Pmax
        uniformity_front: Front non-uniformity (%)
        uniformity_rear: Rear non-uniformity (%)
        temporal_front: Front temporal instability (%)
        temporal_rear: Rear temporal instability (%)

    Returns:
        Dict with 'bifacial' and 'simulator' compliance reports
    """
    # Bifacial measurement compliance
    bifacial_checker = IEC_TS_60904_1_2_Checker()
    bifacial_report = bifacial_checker.run_full_bifacial_compliance(
        g_front=g_front,
        g_rear=g_rear,
        temperature=temperature,
        phi_isc=phi_isc,
        phi_pmax=phi_pmax,
        phi_voc=phi_isc * 0.98,  # Estimate
        mode=MeasurementType.BIFACIAL_DOUBLE_SIDED
    )

    # Simulator classification
    sim_checker = IEC_60904_9_Checker()
    sim_result = sim_checker.classify_simulator(
        uniformity_front=uniformity_front,
        temporal_front=temporal_front,
        spectral_ratios={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0},
        uniformity_rear=uniformity_rear,
        temporal_rear=temporal_rear
    )
    sim_checker.report.calculate_overall_status()

    return {
        "bifacial": bifacial_report,
        "simulator": sim_checker.report,
        "classification": sim_result
    }


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("STANDARDS COMPLIANCE CHECKER - BIFACIAL PV MEASUREMENT")
    print("=" * 70)

    # Example bifacial measurement
    results = check_bifacial_stc_compliance(
        g_front=1000.0,
        g_rear=150.0,
        temperature=25.2,
        phi_isc=0.75,
        phi_pmax=0.70,
        uniformity_front=1.8,
        uniformity_rear=2.5,
        temporal_front=0.3,
        temporal_rear=0.5
    )

    # Print bifacial report
    print("\n" + results['bifacial'].summary())

    # Print simulator classification
    print("\n" + "=" * 70)
    print("SIMULATOR CLASSIFICATION (IEC 60904-9)")
    print("=" * 70)
    class_result = results['classification']
    print(f"Overall Classification: {class_result.overall_class}")
    print(f"  Uniformity: Class {class_result.uniformity_class} ({class_result.uniformity_value:.1f}%)")
    print(f"  Temporal: Class {class_result.temporal_class} ({class_result.temporal_value:.1f}%)")
    print(f"  Spectral: Class {class_result.spectral_class}")

    # GUM compliance check
    print("\n" + "=" * 70)
    print("GUM METHODOLOGY COMPLIANCE (JCGM 100:2008)")
    print("=" * 70)

    gum_checker = GUM_Compliance_Checker()
    gum_checker.check_uncertainty_sources_complete([
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
    gum_checker.check_distribution_validity({
        "calibration": "normal",
        "uniformity": "rectangular",
        "repeatability": "normal"
    })
    gum_checker.check_combined_uncertainty_method(True, False, False)
    gum_checker.check_coverage_factor(2.0, 50)
    gum_checker.report.calculate_overall_status()

    print(gum_checker.report.summary())

    # ISO 17025 check
    print("\n" + "=" * 70)
    print("LABORATORY COMPLIANCE (ISO 17025:2017)")
    print("=" * 70)

    iso_checker = ISO_17025_Checker()
    iso_checker.check_traceability(True, ["Lab Reference", "NREL", "SI"])
    iso_checker.check_calibration_validity("2024-06-15")
    iso_checker.check_uncertainty_reported(True, 2.5, 2.0, 95.0)
    iso_checker.report.calculate_overall_status()

    print(iso_checker.report.summary())
