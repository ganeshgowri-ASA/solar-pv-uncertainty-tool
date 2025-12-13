#!/usr/bin/env python3
"""
Seed Data Script for Solar PV Uncertainty Tool.

This script populates the PostgreSQL database with initial reference data:
- Reference calibration laboratories (NREL, PTB, Fraunhofer ISE, etc.)
- Sun simulators (Spire, Eternalsun, Avalon, Pasan, etc.)
- Standard spectra (AM1.5G, AM1.5D, AM0, etc.)
- Demo user account

Usage:
    python scripts/seed_data.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.database import (
    Base, create_db_engine, create_session_factory, init_db,
    User, ReferenceDevice, Simulator, SpectralData,
    UserRole, LabType, LampType
)


# ============================================================================
# REFERENCE DEVICES / CALIBRATION LABORATORIES
# ============================================================================

REFERENCE_DEVICES_DATA = [
    # Primary Standards Labs
    {
        "lab_code": "NREL",
        "lab_name": "NREL (National Renewable Energy Laboratory)",
        "country": "USA",
        "lab_type": LabType.PRIMARY,
        "accreditation": "ISO 17025, NVLAP",
        "typical_uncertainty_wpvs": 0.5,
        "typical_uncertainty_module": 1.5,
        "website": "https://www.nrel.gov",
        "description": "US primary PV calibration laboratory"
    },
    {
        "lab_code": "PTB",
        "lab_name": "PTB (Physikalisch-Technische Bundesanstalt)",
        "country": "Germany",
        "lab_type": LabType.PRIMARY,
        "accreditation": "ISO 17025",
        "typical_uncertainty_wpvs": 0.4,
        "typical_uncertainty_module": 1.3,
        "website": "https://www.ptb.de",
        "description": "German national metrology institute"
    },
    {
        "lab_code": "AIST",
        "lab_name": "AIST (National Institute of Advanced Industrial Science and Technology)",
        "country": "Japan",
        "lab_type": LabType.PRIMARY,
        "accreditation": "ISO 17025",
        "typical_uncertainty_wpvs": 0.45,
        "typical_uncertainty_module": 1.4,
        "website": "https://www.aist.go.jp",
        "description": "Japanese primary standards laboratory"
    },
    {
        "lab_code": "NIMS",
        "lab_name": "NIMS (National Institute for Materials Science)",
        "country": "China",
        "lab_type": LabType.PRIMARY,
        "accreditation": "ISO 17025, CNAS",
        "typical_uncertainty_wpvs": 0.5,
        "typical_uncertainty_module": 1.5,
        "description": "Chinese national metrology institute"
    },
    {
        "lab_code": "ISFH",
        "lab_name": "ISFH (Institute for Solar Energy Research Hamelin)",
        "country": "Germany",
        "lab_type": LabType.PRIMARY,
        "accreditation": "ISO 17025, DAkkS",
        "typical_uncertainty_wpvs": 0.45,
        "typical_uncertainty_module": 1.4,
        "website": "https://www.isfh.de",
        "description": "German solar research institute with calibration services"
    },
    {
        "lab_code": "FRAUNHOFER_ISE",
        "lab_name": "Fraunhofer ISE CalLab PV Cells",
        "country": "Germany",
        "lab_type": LabType.PRIMARY,
        "accreditation": "ISO 17025, DAkkS",
        "typical_uncertainty_wpvs": 0.4,
        "typical_uncertainty_module": 1.3,
        "website": "https://www.ise.fraunhofer.de",
        "description": "Fraunhofer Institute for Solar Energy Systems"
    },
    # Secondary/Accredited Labs
    {
        "lab_code": "TUV_RHEINLAND",
        "lab_name": "TUV Rheinland PTL",
        "country": "Germany",
        "lab_type": LabType.ACCREDITED,
        "accreditation": "ISO 17025, DAkkS",
        "typical_uncertainty_wpvs": 0.8,
        "typical_uncertainty_module": 2.0,
        "website": "https://www.tuv.com",
        "description": "Global testing and certification laboratory"
    },
    {
        "lab_code": "TUV_SUD",
        "lab_name": "TUV SUD",
        "country": "Germany",
        "lab_type": LabType.ACCREDITED,
        "accreditation": "ISO 17025",
        "typical_uncertainty_wpvs": 0.8,
        "typical_uncertainty_module": 2.0,
        "website": "https://www.tuvsud.com",
        "description": "International testing and certification"
    },
    {
        "lab_code": "SUPSI",
        "lab_name": "SUPSI PV Lab",
        "country": "Switzerland",
        "lab_type": LabType.ACCREDITED,
        "accreditation": "ISO 17025, SAS",
        "typical_uncertainty_wpvs": 0.7,
        "typical_uncertainty_module": 1.8,
        "website": "https://www.supsi.ch",
        "description": "University of Applied Sciences and Arts of Southern Switzerland"
    },
    {
        "lab_code": "PI_BERLIN",
        "lab_name": "PI Berlin (now part of Kiwa)",
        "country": "Germany",
        "lab_type": LabType.ACCREDITED,
        "accreditation": "ISO 17025, DAkkS",
        "typical_uncertainty_wpvs": 0.8,
        "typical_uncertainty_module": 2.0,
        "description": "Independent testing and engineering services"
    },
    {
        "lab_code": "DNV",
        "lab_name": "DNV Energy Lab",
        "country": "Netherlands",
        "lab_type": LabType.ACCREDITED,
        "accreditation": "ISO 17025, RvA",
        "typical_uncertainty_wpvs": 0.8,
        "typical_uncertainty_module": 2.0,
        "website": "https://www.dnv.com",
        "description": "Global quality assurance and risk management"
    },
    {
        "lab_code": "CSIRO",
        "lab_name": "CSIRO Energy",
        "country": "Australia",
        "lab_type": LabType.ACCREDITED,
        "accreditation": "ISO 17025, NATA",
        "typical_uncertainty_wpvs": 0.75,
        "typical_uncertainty_module": 1.9,
        "website": "https://www.csiro.au",
        "description": "Australian national science research organization"
    },
    {
        "lab_code": "CEA_INES",
        "lab_name": "CEA-INES",
        "country": "France",
        "lab_type": LabType.ACCREDITED,
        "accreditation": "ISO 17025, COFRAC",
        "typical_uncertainty_wpvs": 0.7,
        "typical_uncertainty_module": 1.8,
        "website": "https://www.ines-solaire.org",
        "description": "French solar energy research institute"
    },
    {
        "lab_code": "CUSTOM",
        "lab_name": "Custom Laboratory",
        "country": "Custom",
        "lab_type": LabType.ACCREDITED,
        "accreditation": "Custom",
        "typical_uncertainty_wpvs": 1.0,
        "typical_uncertainty_module": 2.0,
        "description": "User-defined laboratory"
    }
]


# ============================================================================
# SUN SIMULATORS
# ============================================================================

SIMULATORS_DATA = [
    # Spire Solar (now Atonometrics)
    {
        "simulator_code": "SPIRE_5600SLP",
        "manufacturer": "Spire Solar / Atonometrics",
        "model": "5600SLP",
        "lamp_type": LampType.XENON,
        "classification": "AAA",
        "typical_uniformity": 2.0,
        "typical_temporal_instability": 0.5,
        "typical_spectral_match": "A",
        "standard_distance_mm": 914.4,
        "description": "Large-area steady-state solar simulator"
    },
    {
        "simulator_code": "SPIRE_4600",
        "manufacturer": "Spire Solar / Atonometrics",
        "model": "4600",
        "lamp_type": LampType.XENON,
        "classification": "AAA",
        "typical_uniformity": 2.5,
        "typical_temporal_instability": 1.0,
        "typical_spectral_match": "A",
        "standard_distance_mm": 914.4,
        "description": "Previous generation solar simulator"
    },
    # Eternalsun (Meyer Burger)
    {
        "simulator_code": "ETERNALSUN_SLP150",
        "manufacturer": "Eternalsun / Meyer Burger",
        "model": "SLP-150",
        "lamp_type": LampType.LED,
        "classification": "AAA+",
        "typical_uniformity": 1.5,
        "typical_temporal_instability": 0.3,
        "typical_spectral_match": "A",
        "standard_distance_mm": 500.0,
        "description": "Advanced LED-based simulator with superior uniformity"
    },
    # Avalon (Wavelabs)
    {
        "simulator_code": "AVALON_NEXUN",
        "manufacturer": "Wavelabs",
        "model": "Avalon Nexun",
        "lamp_type": LampType.LED,
        "classification": "AAA",
        "typical_uniformity": 2.0,
        "typical_temporal_instability": 0.5,
        "typical_spectral_match": "A",
        "standard_distance_mm": 600.0,
        "description": "LED simulator for standard modules"
    },
    {
        "simulator_code": "AVALON_PEROVSKITE",
        "manufacturer": "Wavelabs",
        "model": "Avalon Perovskite",
        "lamp_type": LampType.LED,
        "classification": "AAA",
        "typical_uniformity": 2.0,
        "typical_temporal_instability": 0.5,
        "typical_spectral_match": "A",
        "standard_distance_mm": 600.0,
        "description": "Optimized for perovskite modules"
    },
    {
        "simulator_code": "AVALON_STEADYSTATE",
        "manufacturer": "Wavelabs",
        "model": "Avalon SteadyState",
        "lamp_type": LampType.LED,
        "classification": "AAA",
        "typical_uniformity": 1.8,
        "typical_temporal_instability": 0.3,
        "typical_spectral_match": "A",
        "standard_distance_mm": 600.0,
        "description": "Steady-state LED simulator"
    },
    # Halm (now EETS)
    {
        "simulator_code": "HALM_FLASHER",
        "manufacturer": "Halm / EETS",
        "model": "Standard Flasher",
        "lamp_type": LampType.XENON,
        "classification": "AAA",
        "typical_uniformity": 2.0,
        "typical_temporal_instability": 1.0,
        "typical_spectral_match": "A",
        "standard_distance_mm": 800.0,
        "description": "Industrial flasher system"
    },
    # Pasan
    {
        "simulator_code": "PASAN_HIGHLIGHT_LED",
        "manufacturer": "Pasan",
        "model": "HighLIGHT LED",
        "lamp_type": LampType.LED,
        "classification": "AAA+",
        "typical_uniformity": 1.5,
        "typical_temporal_instability": 0.2,
        "typical_spectral_match": "A",
        "standard_distance_mm": 500.0,
        "description": "High-performance LED simulator"
    },
    # ReRa Solutions
    {
        "simulator_code": "RERA_TRACER",
        "manufacturer": "ReRa Solutions",
        "model": "Tracer",
        "lamp_type": LampType.LED,
        "classification": "AAA",
        "typical_uniformity": 2.0,
        "typical_temporal_instability": 0.5,
        "typical_spectral_match": "A",
        "standard_distance_mm": 550.0,
        "description": "Compact LED simulator"
    },
    # Lumartix
    {
        "simulator_code": "LUMARTIX_FLASH",
        "manufacturer": "Lumartix",
        "model": "Flash",
        "lamp_type": LampType.XENON,
        "classification": "AAA",
        "typical_uniformity": 2.0,
        "typical_temporal_instability": 1.0,
        "typical_spectral_match": "A",
        "standard_distance_mm": 750.0,
        "description": "Flash-based solar simulator"
    },
    # Atlas Material Testing
    {
        "simulator_code": "ATLAS_SUNTEST",
        "manufacturer": "Atlas Material Testing",
        "model": "SUNTEST",
        "lamp_type": LampType.XENON,
        "classification": "AA",
        "typical_uniformity": 3.0,
        "typical_temporal_instability": 2.0,
        "typical_spectral_match": "A",
        "standard_distance_mm": 700.0,
        "description": "For material testing and module characterization"
    },
    # Custom
    {
        "simulator_code": "CUSTOM",
        "manufacturer": "Custom",
        "model": "User Defined",
        "lamp_type": LampType.CUSTOM,
        "classification": "Custom",
        "typical_uniformity": 2.0,
        "typical_temporal_instability": 1.0,
        "typical_spectral_match": "A",
        "standard_distance_mm": 600.0,
        "description": "User-defined simulator configuration"
    }
]


# ============================================================================
# SPECTRAL DATA
# ============================================================================

SPECTRAL_DATA = [
    {
        "spectrum_code": "AM1.5G",
        "spectrum_name": "AM1.5 Global",
        "standard": "IEC 60904-3",
        "air_mass": 1.5,
        "integrated_irradiance": 1000.0,
        "wavelength_range_nm": "280-4000",
        "description": "Standard terrestrial spectrum for flat-plate modules",
        "is_standard": True
    },
    {
        "spectrum_code": "AM1.5D",
        "spectrum_name": "AM1.5 Direct",
        "standard": "ASTM G173",
        "air_mass": 1.5,
        "integrated_irradiance": 900.0,
        "wavelength_range_nm": "280-4000",
        "description": "Direct normal irradiance spectrum",
        "is_standard": True
    },
    {
        "spectrum_code": "AM1.0",
        "spectrum_name": "AM1.0",
        "standard": "ASTM E490",
        "air_mass": 1.0,
        "integrated_irradiance": 1000.0,
        "wavelength_range_nm": "280-4000",
        "description": "Air mass 1.0 spectrum",
        "is_standard": True
    },
    {
        "spectrum_code": "AM0",
        "spectrum_name": "AM0 (Extraterrestrial)",
        "standard": "ASTM E490",
        "air_mass": 0.0,
        "integrated_irradiance": 1367.0,
        "wavelength_range_nm": "280-4000",
        "description": "Extraterrestrial solar spectrum for space applications",
        "is_standard": True
    },
    {
        "spectrum_code": "CUSTOM",
        "spectrum_name": "Custom Spectrum",
        "standard": "User Defined",
        "air_mass": None,
        "integrated_irradiance": 1000.0,
        "wavelength_range_nm": "Custom",
        "description": "User-uploaded custom spectrum",
        "is_standard": False
    }
]


# ============================================================================
# DEMO USER
# ============================================================================

DEMO_USER = {
    "email": "demo@solarpv-uncertainty.com",
    "username": "demo_user",
    "password_hash": "pbkdf2:sha256:260000$demo$placeholder",  # Placeholder - should be properly hashed
    "role": UserRole.ENGINEER,
    "full_name": "Demo User",
    "organization": "Solar PV Uncertainty Tool",
    "country": "Global",
    "preferred_currency": "USD",
    "preferred_units": "SI"
}


# ============================================================================
# SEEDING FUNCTIONS
# ============================================================================

def seed_reference_devices(session):
    """Seed reference devices / calibration laboratories."""
    print("Seeding reference devices...")
    count = 0
    for data in REFERENCE_DEVICES_DATA:
        existing = session.query(ReferenceDevice).filter_by(lab_code=data["lab_code"]).first()
        if not existing:
            device = ReferenceDevice(**data)
            session.add(device)
            count += 1
    session.commit()
    print(f"  Added {count} reference devices")
    return count


def seed_simulators(session):
    """Seed sun simulators."""
    print("Seeding sun simulators...")
    count = 0
    for data in SIMULATORS_DATA:
        existing = session.query(Simulator).filter_by(simulator_code=data["simulator_code"]).first()
        if not existing:
            simulator = Simulator(**data)
            session.add(simulator)
            count += 1
    session.commit()
    print(f"  Added {count} simulators")
    return count


def seed_spectral_data(session):
    """Seed standard spectra."""
    print("Seeding spectral data...")
    count = 0
    for data in SPECTRAL_DATA:
        existing = session.query(SpectralData).filter_by(spectrum_code=data["spectrum_code"]).first()
        if not existing:
            spectrum = SpectralData(**data)
            session.add(spectrum)
            count += 1
    session.commit()
    print(f"  Added {count} spectral datasets")
    return count


def seed_demo_user(session):
    """Seed demo user account."""
    print("Seeding demo user...")
    existing = session.query(User).filter_by(username=DEMO_USER["username"]).first()
    if not existing:
        user = User(**DEMO_USER)
        session.add(user)
        session.commit()
        print("  Added demo user")
        return 1
    print("  Demo user already exists")
    return 0


def run_all_seeds(session):
    """Run all seed functions."""
    print("\n" + "=" * 60)
    print("SOLAR PV UNCERTAINTY TOOL - DATABASE SEEDING")
    print("=" * 60 + "\n")

    total = 0
    total += seed_reference_devices(session)
    total += seed_simulators(session)
    total += seed_spectral_data(session)
    total += seed_demo_user(session)

    print("\n" + "-" * 60)
    print(f"SEEDING COMPLETE: {total} total records added")
    print("-" * 60 + "\n")

    return total


def verify_seed_data(session):
    """Verify all seed data was loaded correctly."""
    print("\n" + "=" * 60)
    print("VERIFYING SEED DATA")
    print("=" * 60 + "\n")

    # Count records in each table
    users_count = session.query(User).count()
    ref_devices_count = session.query(ReferenceDevice).count()
    simulators_count = session.query(Simulator).count()
    spectral_count = session.query(SpectralData).count()

    print(f"  Users:             {users_count}")
    print(f"  Reference Devices: {ref_devices_count}")
    print(f"  Simulators:        {simulators_count}")
    print(f"  Spectral Data:     {spectral_count}")

    # Verify expected counts
    expected = {
        "users": 1,
        "reference_devices": len(REFERENCE_DEVICES_DATA),
        "simulators": len(SIMULATORS_DATA),
        "spectral_data": len(SPECTRAL_DATA)
    }

    all_verified = True
    print("\n  Verification:")
    if users_count >= expected["users"]:
        print(f"    Users: OK")
    else:
        print(f"    Users: FAILED (expected {expected['users']}, got {users_count})")
        all_verified = False

    if ref_devices_count >= expected["reference_devices"]:
        print(f"    Reference Devices: OK")
    else:
        print(f"    Reference Devices: FAILED")
        all_verified = False

    if simulators_count >= expected["simulators"]:
        print(f"    Simulators: OK")
    else:
        print(f"    Simulators: FAILED")
        all_verified = False

    if spectral_count >= expected["spectral_data"]:
        print(f"    Spectral Data: OK")
    else:
        print(f"    Spectral Data: FAILED")
        all_verified = False

    if all_verified:
        print("\n  All seed data verified successfully!")
    else:
        print("\n  WARNING: Some seed data verification failed!")

    return all_verified


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point for seeding database."""
    print("\nInitializing database connection...")

    try:
        # Create engine and session
        engine = create_db_engine(echo=False)
        Session = create_session_factory(engine)
        session = Session()

        # Initialize database (create tables if not exist)
        print("Creating database tables...")
        init_db(engine)

        # Run all seeds
        run_all_seeds(session)

        # Verify data
        verify_seed_data(session)

        # Close session
        session.close()

        print("\nDatabase seeding completed successfully!")
        return 0

    except Exception as e:
        print(f"\nERROR: Database seeding failed!")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
