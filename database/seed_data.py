"""
Seed data for PV Measurement Uncertainty Tool database
Pre-populates reference laboratories, standard spectra, and typical equipment
"""

from datetime import datetime
from database.models import (
    Organization, User, ReferenceDevice, SunSimulator,
    UserRole, PVTechnology
)
from database.connection import session_scope


# =============================================================================
# REFERENCE LABORATORIES DATA
# =============================================================================

REFERENCE_LABS_DATA = [
    # Primary Standards Labs
    {
        "name": "NREL (National Renewable Energy Laboratory)",
        "address": "Golden, Colorado, USA",
        "accreditation_number": "NVLAP Lab Code 200932-0",
        "accreditation_body": "NVLAP / ISO 17025",
        "website": "https://www.nrel.gov",
        "typical_wpvs_uncertainty": 0.5,
        "typical_module_uncertainty": 1.5,
        "lab_type": "Primary"
    },
    {
        "name": "PTB (Physikalisch-Technische Bundesanstalt)",
        "address": "Braunschweig, Germany",
        "accreditation_number": "DAkkS D-K-15063-01-00",
        "accreditation_body": "DAkkS / ISO 17025",
        "website": "https://www.ptb.de",
        "typical_wpvs_uncertainty": 0.4,
        "typical_module_uncertainty": 1.3,
        "lab_type": "Primary"
    },
    {
        "name": "Fraunhofer ISE CalLab PV Cells",
        "address": "Freiburg, Germany",
        "accreditation_number": "DAkkS D-K-15201-01-00",
        "accreditation_body": "DAkkS / ISO 17025",
        "website": "https://www.ise.fraunhofer.de",
        "typical_wpvs_uncertainty": 0.4,
        "typical_module_uncertainty": 1.3,
        "lab_type": "Primary"
    },
    {
        "name": "AIST (National Institute of Advanced Industrial Science and Technology)",
        "address": "Tsukuba, Japan",
        "accreditation_number": "JNLA",
        "accreditation_body": "ISO 17025",
        "website": "https://www.aist.go.jp",
        "typical_wpvs_uncertainty": 0.45,
        "typical_module_uncertainty": 1.4,
        "lab_type": "Primary"
    },
    {
        "name": "ISFH (Institute for Solar Energy Research Hamelin)",
        "address": "Emmerthal, Germany",
        "accreditation_number": "DAkkS D-K-20427-01-00",
        "accreditation_body": "DAkkS / ISO 17025",
        "website": "https://www.isfh.de",
        "typical_wpvs_uncertainty": 0.45,
        "typical_module_uncertainty": 1.4,
        "lab_type": "Primary"
    },
    # Secondary/Accredited Labs
    {
        "name": "TÜV Rheinland PTL",
        "address": "Cologne, Germany",
        "accreditation_number": "DAkkS D-PL-11242-01-01",
        "accreditation_body": "DAkkS / ISO 17025",
        "website": "https://www.tuv.com",
        "typical_wpvs_uncertainty": 0.8,
        "typical_module_uncertainty": 2.0,
        "lab_type": "Accredited"
    },
    {
        "name": "SUPSI PV Lab",
        "address": "Mendrisio, Switzerland",
        "accreditation_number": "SAS STS 0542",
        "accreditation_body": "SAS / ISO 17025",
        "website": "https://www.supsi.ch",
        "typical_wpvs_uncertainty": 0.7,
        "typical_module_uncertainty": 1.8,
        "lab_type": "Accredited"
    },
    {
        "name": "DNV Energy Lab",
        "address": "Arnhem, Netherlands",
        "accreditation_number": "RvA L524",
        "accreditation_body": "RvA / ISO 17025",
        "website": "https://www.dnv.com",
        "typical_wpvs_uncertainty": 0.8,
        "typical_module_uncertainty": 2.0,
        "lab_type": "Accredited"
    },
    {
        "name": "CEA-INES",
        "address": "Le Bourget-du-Lac, France",
        "accreditation_number": "COFRAC 1-1631",
        "accreditation_body": "COFRAC / ISO 17025",
        "website": "https://www.ines-solaire.org",
        "typical_wpvs_uncertainty": 0.7,
        "typical_module_uncertainty": 1.8,
        "lab_type": "Accredited"
    },
]


# =============================================================================
# SUN SIMULATORS DATA
# =============================================================================

SUN_SIMULATORS_DATA = [
    # Spire Solar (now Atonometrics)
    {
        "manufacturer": "Spire Solar / Atonometrics",
        "model": "5600SLP",
        "lamp_type": "Xenon",
        "classification": "AAA",
        "typical_uniformity_pct": 2.0,
        "typical_temporal_instability_pct": 0.5,
        "typical_spectral_match": "A",
        "standard_distance_mm": 914.4,
    },
    {
        "manufacturer": "Spire Solar / Atonometrics",
        "model": "4600",
        "lamp_type": "Xenon",
        "classification": "AAA",
        "typical_uniformity_pct": 2.5,
        "typical_temporal_instability_pct": 1.0,
        "typical_spectral_match": "A",
        "standard_distance_mm": 914.4,
    },
    # Eternalsun (Meyer Burger)
    {
        "manufacturer": "Eternalsun / Meyer Burger",
        "model": "SLP-150",
        "lamp_type": "LED",
        "classification": "AAA+",
        "typical_uniformity_pct": 1.5,
        "typical_temporal_instability_pct": 0.3,
        "typical_spectral_match": "A",
        "standard_distance_mm": 500.0,
    },
    # Wavelabs Avalon
    {
        "manufacturer": "Wavelabs",
        "model": "Avalon Nexun",
        "lamp_type": "LED",
        "classification": "AAA",
        "typical_uniformity_pct": 2.0,
        "typical_temporal_instability_pct": 0.5,
        "typical_spectral_match": "A",
        "standard_distance_mm": 600.0,
    },
    {
        "manufacturer": "Wavelabs",
        "model": "Avalon SteadyState",
        "lamp_type": "LED",
        "classification": "AAA",
        "typical_uniformity_pct": 1.8,
        "typical_temporal_instability_pct": 0.3,
        "typical_spectral_match": "A",
        "standard_distance_mm": 600.0,
    },
    # Pasan
    {
        "manufacturer": "Pasan",
        "model": "HighLIGHT LED",
        "lamp_type": "LED",
        "classification": "AAA+",
        "typical_uniformity_pct": 1.5,
        "typical_temporal_instability_pct": 0.2,
        "typical_spectral_match": "A",
        "standard_distance_mm": 500.0,
    },
    # Halm
    {
        "manufacturer": "Halm / EETS",
        "model": "Standard Flasher",
        "lamp_type": "Xenon",
        "classification": "AAA",
        "typical_uniformity_pct": 2.0,
        "typical_temporal_instability_pct": 1.0,
        "typical_spectral_match": "A",
        "standard_distance_mm": 800.0,
    },
    # ReRa Solutions
    {
        "manufacturer": "ReRa Solutions",
        "model": "Tracer",
        "lamp_type": "LED",
        "classification": "AAA",
        "typical_uniformity_pct": 2.0,
        "typical_temporal_instability_pct": 0.5,
        "typical_spectral_match": "A",
        "standard_distance_mm": 550.0,
    },
]


# =============================================================================
# SEEDING FUNCTIONS
# =============================================================================

def seed_demo_organization():
    """Create a demo organization with admin user."""
    with session_scope() as session:
        # Check if demo org already exists
        existing = session.query(Organization).filter_by(
            name="Demo PV Testing Laboratory"
        ).first()

        if existing:
            print("Demo organization already exists.")
            return existing.id

        # Create demo organization
        org = Organization(
            name="Demo PV Testing Laboratory",
            address="123 Solar Way, Sunshine City, SC 12345",
            accreditation_number="ISO-17025-DEMO-001",
            accreditation_body="Demo Accreditation Body",
            document_format_prefix="PV-DEMO",
            record_prefix="DEMO-REC",
            website="https://demo.pvlab.example.com",
            contact_email="demo@pvlab.example.com"
        )
        session.add(org)
        session.flush()  # Get the ID

        # Create demo admin user
        from hashlib import sha256
        demo_password_hash = sha256(b"demo123").hexdigest()

        admin = User(
            organization_id=org.id,
            email="admin@demo.pvlab.com",
            password_hash=demo_password_hash,
            first_name="Demo",
            last_name="Admin",
            title="Laboratory Manager",
            role=UserRole.ADMIN,
            is_active=True,
            is_verified=True
        )
        session.add(admin)

        # Create demo engineer
        engineer = User(
            organization_id=org.id,
            email="engineer@demo.pvlab.com",
            password_hash=demo_password_hash,
            first_name="Demo",
            last_name="Engineer",
            title="Test Engineer",
            role=UserRole.ENGINEER,
            is_active=True,
            is_verified=True
        )
        session.add(engineer)

        print(f"Created demo organization (ID: {org.id}) with admin and engineer users.")
        return org.id


def seed_sun_simulators(organization_id: int = None):
    """Seed sun simulator equipment database."""
    with session_scope() as session:
        count = 0
        for sim_data in SUN_SIMULATORS_DATA:
            existing = session.query(SunSimulator).filter_by(
                manufacturer=sim_data['manufacturer'],
                model=sim_data['model']
            ).first()

            if not existing:
                sim = SunSimulator(
                    organization_id=organization_id,
                    **sim_data,
                    is_active=True
                )
                session.add(sim)
                count += 1

        print(f"Seeded {count} sun simulators.")


def seed_all(create_demo: bool = True):
    """
    Run all seed functions.

    Args:
        create_demo: Whether to create demo organization and users
    """
    print("Starting database seeding...")

    org_id = None
    if create_demo:
        org_id = seed_demo_organization()

    seed_sun_simulators(org_id)

    print("Database seeding complete!")


# CLI interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Seed database with initial data')
    parser.add_argument('--demo', action='store_true', help='Create demo organization and users')
    parser.add_argument('--all', action='store_true', help='Seed all data')

    args = parser.parse_args()

    if args.all:
        seed_all(create_demo=args.demo)
    elif args.demo:
        seed_demo_organization()
    else:
        print("Use --demo to create demo data or --all to seed everything.")
