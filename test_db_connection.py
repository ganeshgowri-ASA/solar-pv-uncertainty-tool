#!/usr/bin/env python3
"""
Database Connection Test Script
Tests Railway PostgreSQL connectivity for the PV Uncertainty Tool
"""

import os
import sys

def test_connection():
    """Test database connection and report status."""
    print("=" * 60)
    print("PV Uncertainty Tool - Database Connection Test")
    print("=" * 60)

    # Check for DATABASE_URL in environment
    database_url = os.environ.get('DATABASE_URL')

    if database_url:
        # Mask password for display
        from urllib.parse import urlparse
        parsed = urlparse(database_url)
        masked_url = f"{parsed.scheme}://{parsed.username}:****@{parsed.hostname}:{parsed.port}{parsed.path}"
        print(f"\nDATABASE_URL found: {masked_url}")
    else:
        print("\nDATABASE_URL not found in environment")
        print("Checking for individual POSTGRES_* variables...")

        host = os.environ.get('POSTGRES_HOST', 'not set')
        port = os.environ.get('POSTGRES_PORT', 'not set')
        db = os.environ.get('POSTGRES_DB', 'not set')
        user = os.environ.get('POSTGRES_USER', 'not set')

        print(f"  POSTGRES_HOST: {host}")
        print(f"  POSTGRES_PORT: {port}")
        print(f"  POSTGRES_DB: {db}")
        print(f"  POSTGRES_USER: {user}")
        print(f"  POSTGRES_PASSWORD: {'set' if os.environ.get('POSTGRES_PASSWORD') else 'not set'}")

    print("\n" + "-" * 60)
    print("Testing connection...")
    print("-" * 60)

    try:
        from database.connection import check_connection, get_connection_info, init_database

        # Get connection info
        info = get_connection_info()
        print(f"\nConnection Details:")
        print(f"  Host: {info.get('host', 'N/A')}")
        print(f"  Port: {info.get('port', 'N/A')}")
        print(f"  Database: {info.get('database', 'N/A')}")
        print(f"  User: {info.get('user', 'N/A')}")

        if info.get('connected'):
            print("\n[SUCCESS] Database connection successful!")

            # Offer to initialize schema
            if len(sys.argv) > 1 and sys.argv[1] == '--init':
                print("\n" + "-" * 60)
                print("Initializing database schema...")
                print("-" * 60)
                init_database(drop_existing=False)
                print("[SUCCESS] Schema created successfully!")
            else:
                print("\nTip: Run with --init flag to create database schema:")
                print("  python test_db_connection.py --init")

            return True
        else:
            print("\n[FAILED] Database connection failed!")
            return False

    except ImportError as e:
        print(f"\n[ERROR] Import error: {e}")
        print("Make sure you're running from the project root directory")
        return False
    except Exception as e:
        print(f"\n[ERROR] Connection test failed: {e}")
        return False


def test_crud_operations():
    """Test basic CRUD operations."""
    print("\n" + "=" * 60)
    print("Testing CRUD Operations")
    print("=" * 60)

    try:
        from database.connection import session_scope
        from database.models import Organization, SunSimulator

        with session_scope() as session:
            # Test Read - Count organizations
            org_count = session.query(Organization).count()
            print(f"\nOrganizations in database: {org_count}")

            # Test Read - Count sun simulators
            sim_count = session.query(SunSimulator).count()
            print(f"Sun simulators in database: {sim_count}")

            # List some data if available
            if org_count > 0:
                orgs = session.query(Organization).limit(3).all()
                print("\nSample organizations:")
                for org in orgs:
                    print(f"  - {org.name}")

            if sim_count > 0:
                sims = session.query(SunSimulator).limit(3).all()
                print("\nSample sun simulators:")
                for sim in sims:
                    print(f"  - {sim.manufacturer} {sim.model}")

        print("\n[SUCCESS] CRUD operations working!")
        return True

    except Exception as e:
        print(f"\n[ERROR] CRUD test failed: {e}")
        return False


def seed_demo_data():
    """Seed database with demo data."""
    print("\n" + "=" * 60)
    print("Seeding Demo Data")
    print("=" * 60)

    try:
        from database.seed_data import seed_all
        seed_all()
        print("\n[SUCCESS] Demo data seeded!")
        return True
    except Exception as e:
        print(f"\n[ERROR] Seeding failed: {e}")
        return False


if __name__ == '__main__':
    # Run connection test
    connected = test_connection()

    if connected:
        # Run CRUD test
        test_crud_operations()

        # Seed data if requested
        if len(sys.argv) > 1 and sys.argv[1] == '--seed':
            seed_demo_data()
        elif '--seed' not in sys.argv:
            print("\nTip: Run with --seed flag to add demo data:")
            print("  python test_db_connection.py --seed")

    print("\n" + "=" * 60)
    sys.exit(0 if connected else 1)
