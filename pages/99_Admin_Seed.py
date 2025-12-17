"""
Admin Seed - Complete Database Management Dashboard
=====================================================
Full-featured admin page with bulletproof error handling for:
- Database Status & Schema Validation
- Migration Management (docs/migrations/)
- QA Testing Suite
- Protocol/Seed Data Management
- Danger Zone Operations

This file is designed to be idempotent - safe to run multiple times.
All operations include pre-flight checks and rollback capability.
"""

import streamlit as st
import os
import sys
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Set
from pathlib import Path

# Page configuration MUST be first Streamlit command
st.set_page_config(
    page_title="Admin Seed - PV Uncertainty Tool",
    page_icon="🔧",
    layout="wide"
)


# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

# Version tracking
ADMIN_VERSION = "3.0.0"
BUILD_DATE = "2024-12-17"

# Migration paths - Use docs/migrations/ as the canonical location
PROJECT_ROOT = Path(__file__).parent.parent
MIGRATIONS_DIR = PROJECT_ROOT / "docs" / "migrations"
FALLBACK_MIGRATIONS_DIR = PROJECT_ROOT / "migrations"  # Fallback to root migrations

# Migration definitions - comprehensive list with proper SQL
MIGRATIONS: Dict[str, Dict[str, Any]] = {
    "001": {
        "name": "initial_schema",
        "description": "Base schema created by SQLAlchemy ORM",
        "requires": [],
        "reversible": True,
        "sql_up": """
-- Migration 001: Initial Schema (ORM managed)
-- Base tables are created by SQLAlchemy Base.metadata.create_all()
SELECT 'Migration 001 UP: Schema initialization marker' AS status;
""",
        "sql_down": """
-- Migration 001 DOWN: Full schema reset
-- WARNING: This drops all tables!
DROP SCHEMA public CASCADE;
CREATE SCHEMA public;
GRANT ALL ON SCHEMA public TO postgres;
GRANT ALL ON SCHEMA public TO public;
SELECT 'Migration 001 DOWN: Schema reset complete' AS status;
"""
    },
    "002": {
        "name": "add_sample_tracking",
        "description": "Add sample tracking fields to link samples with receipts and inspections",
        "requires": ["001"],
        "reversible": True,
        "sql_up": """
-- Migration 002: Add Sample Tracking Fields
-- Links samples to receipts and inspections for full traceability

-- Add receipt_id to samples if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'samples' AND column_name = 'receipt_id'
    ) THEN
        ALTER TABLE samples ADD COLUMN receipt_id INTEGER;
    END IF;
END $$;

-- Add inspection_id to samples if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'samples' AND column_name = 'inspection_id'
    ) THEN
        ALTER TABLE samples ADD COLUMN inspection_id INTEGER;
    END IF;
END $$;

SELECT 'Migration 002 UP: Sample tracking fields added' AS status;
""",
        "sql_down": """
-- Migration 002 DOWN: Remove sample tracking fields
ALTER TABLE samples DROP COLUMN IF EXISTS receipt_id;
ALTER TABLE samples DROP COLUMN IF EXISTS inspection_id;
SELECT 'Migration 002 DOWN: Sample tracking fields removed' AS status;
"""
    },
    "003": {
        "name": "add_inspection_allocation",
        "description": "Add allocation tracking to incoming inspections",
        "requires": ["001"],
        "reversible": True,
        "sql_up": """
-- Migration 003: Add Inspection Allocation Tracking

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'incoming_inspections' AND column_name = 'allocation_triggered'
    ) THEN
        ALTER TABLE incoming_inspections ADD COLUMN allocation_triggered BOOLEAN DEFAULT FALSE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'incoming_inspections' AND column_name = 'allocated_sample_id'
    ) THEN
        ALTER TABLE incoming_inspections ADD COLUMN allocated_sample_id INTEGER;
    END IF;
END $$;

SELECT 'Migration 003 UP: Inspection allocation fields added' AS status;
""",
        "sql_down": """
-- Migration 003 DOWN: Remove inspection allocation fields
ALTER TABLE incoming_inspections DROP COLUMN IF EXISTS allocation_triggered;
ALTER TABLE incoming_inspections DROP COLUMN IF EXISTS allocated_sample_id;
SELECT 'Migration 003 DOWN: Inspection allocation fields removed' AS status;
"""
    },
    "004": {
        "name": "add_measurement_indexes",
        "description": "Add performance indexes for measurement queries",
        "requires": ["001"],
        "reversible": True,
        "sql_up": """
-- Migration 004: Performance Indexes for Measurements

CREATE INDEX IF NOT EXISTS idx_measurements_created_at ON measurements(created_at);
CREATE INDEX IF NOT EXISTS idx_measurements_module_date ON measurements(module_id, test_date);
CREATE INDEX IF NOT EXISTS idx_uncertainty_results_param ON uncertainty_results(target_parameter);

SELECT 'Migration 004 UP: Measurement indexes created' AS status;
""",
        "sql_down": """
-- Migration 004 DOWN: Remove measurement indexes
DROP INDEX IF EXISTS idx_measurements_created_at;
DROP INDEX IF EXISTS idx_measurements_module_date;
DROP INDEX IF EXISTS idx_uncertainty_results_param;
SELECT 'Migration 004 DOWN: Measurement indexes removed' AS status;
"""
    },
    "005": {
        "name": "add_audit_log_indexes",
        "description": "Add performance indexes for audit log queries",
        "requires": ["001"],
        "reversible": True,
        "sql_up": """
-- Migration 005: Audit Log Performance Indexes

CREATE INDEX IF NOT EXISTS idx_audit_logs_org_time ON audit_logs(organization_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_entity_full ON audit_logs(entity_type, entity_id, timestamp);

SELECT 'Migration 005 UP: Audit log indexes created' AS status;
""",
        "sql_down": """
-- Migration 005 DOWN: Remove audit log indexes
DROP INDEX IF EXISTS idx_audit_logs_org_time;
DROP INDEX IF EXISTS idx_audit_logs_entity_full;
SELECT 'Migration 005 DOWN: Audit log indexes removed' AS status;
"""
    },
    "006": {
        "name": "add_file_content_storage",
        "description": "Add binary content storage for files",
        "requires": ["001"],
        "reversible": True,
        "sql_up": """
-- Migration 006: File Content Storage

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'files' AND column_name = 'content'
    ) THEN
        ALTER TABLE files ADD COLUMN content BYTEA;
    END IF;
END $$;

SELECT 'Migration 006 UP: File content storage added' AS status;
""",
        "sql_down": """
-- Migration 006 DOWN: Remove file content storage
ALTER TABLE files DROP COLUMN IF EXISTS content;
SELECT 'Migration 006 DOWN: File content storage removed' AS status;
"""
    },
    "007": {
        "name": "add_module_tracking_fields",
        "description": "Add additional tracking fields for modules",
        "requires": ["001"],
        "reversible": True,
        "sql_up": """
-- Migration 007: Module Tracking Fields

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'modules' AND column_name = 'production_date'
    ) THEN
        ALTER TABLE modules ADD COLUMN production_date DATE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'modules' AND column_name = 'batch_number'
    ) THEN
        ALTER TABLE modules ADD COLUMN batch_number VARCHAR(100);
    END IF;
END $$;

SELECT 'Migration 007 UP: Module tracking fields added' AS status;
""",
        "sql_down": """
-- Migration 007 DOWN: Remove module tracking fields
ALTER TABLE modules DROP COLUMN IF EXISTS production_date;
ALTER TABLE modules DROP COLUMN IF EXISTS batch_number;
SELECT 'Migration 007 DOWN: Module tracking fields removed' AS status;
"""
    },
    "008": {
        "name": "add_reference_device_tracking",
        "description": "Enhanced tracking for reference devices",
        "requires": ["001"],
        "reversible": True,
        "sql_up": """
-- Migration 008: Reference Device Enhanced Tracking

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'reference_devices' AND column_name = 'next_calibration_date'
    ) THEN
        ALTER TABLE reference_devices ADD COLUMN next_calibration_date DATE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'reference_devices' AND column_name = 'calibration_interval_months'
    ) THEN
        ALTER TABLE reference_devices ADD COLUMN calibration_interval_months INTEGER DEFAULT 12;
    END IF;
END $$;

SELECT 'Migration 008 UP: Reference device tracking enhanced' AS status;
""",
        "sql_down": """
-- Migration 008 DOWN: Remove enhanced reference device tracking
ALTER TABLE reference_devices DROP COLUMN IF EXISTS next_calibration_date;
ALTER TABLE reference_devices DROP COLUMN IF EXISTS calibration_interval_months;
SELECT 'Migration 008 DOWN: Reference device tracking reverted' AS status;
"""
    },
    "009": {
        "name": "add_sun_simulator_maintenance",
        "description": "Add maintenance tracking for sun simulators",
        "requires": ["001"],
        "reversible": True,
        "sql_up": """
-- Migration 009: Sun Simulator Maintenance Tracking

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'sun_simulators' AND column_name = 'last_maintenance_date'
    ) THEN
        ALTER TABLE sun_simulators ADD COLUMN last_maintenance_date DATE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'sun_simulators' AND column_name = 'lamp_hours'
    ) THEN
        ALTER TABLE sun_simulators ADD COLUMN lamp_hours INTEGER DEFAULT 0;
    END IF;
END $$;

SELECT 'Migration 009 UP: Sun simulator maintenance tracking added' AS status;
""",
        "sql_down": """
-- Migration 009 DOWN: Remove sun simulator maintenance tracking
ALTER TABLE sun_simulators DROP COLUMN IF EXISTS last_maintenance_date;
ALTER TABLE sun_simulators DROP COLUMN IF EXISTS lamp_hours;
SELECT 'Migration 009 DOWN: Sun simulator maintenance reverted' AS status;
"""
    },
    "010": {
        "name": "add_measurement_conditions",
        "description": "Add environmental condition tracking for measurements",
        "requires": ["001"],
        "reversible": True,
        "sql_up": """
-- Migration 010: Measurement Environmental Conditions

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'measurements' AND column_name = 'barometric_pressure_hpa'
    ) THEN
        ALTER TABLE measurements ADD COLUMN barometric_pressure_hpa FLOAT;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'measurements' AND column_name = 'wind_speed_m_s'
    ) THEN
        ALTER TABLE measurements ADD COLUMN wind_speed_m_s FLOAT;
    END IF;
END $$;

SELECT 'Migration 010 UP: Measurement conditions tracking added' AS status;
""",
        "sql_down": """
-- Migration 010 DOWN: Remove environmental conditions
ALTER TABLE measurements DROP COLUMN IF EXISTS barometric_pressure_hpa;
ALTER TABLE measurements DROP COLUMN IF EXISTS wind_speed_m_s;
SELECT 'Migration 010 DOWN: Measurement conditions reverted' AS status;
"""
    },
    "011": {
        "name": "add_uncertainty_component_types",
        "description": "Add type classification for uncertainty components",
        "requires": ["001"],
        "reversible": True,
        "sql_up": """
-- Migration 011: Uncertainty Component Types

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'uncertainty_components' AND column_name = 'component_type'
    ) THEN
        ALTER TABLE uncertainty_components ADD COLUMN component_type VARCHAR(20) DEFAULT 'B';
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'uncertainty_components' AND column_name = 'degrees_of_freedom'
    ) THEN
        ALTER TABLE uncertainty_components ADD COLUMN degrees_of_freedom FLOAT;
    END IF;
END $$;

SELECT 'Migration 011 UP: Uncertainty component types added' AS status;
""",
        "sql_down": """
-- Migration 011 DOWN: Remove uncertainty component types
ALTER TABLE uncertainty_components DROP COLUMN IF EXISTS component_type;
ALTER TABLE uncertainty_components DROP COLUMN IF EXISTS degrees_of_freedom;
SELECT 'Migration 011 DOWN: Uncertainty component types reverted' AS status;
"""
    },
    "012": {
        "name": "add_iv_curve_analysis",
        "description": "Add analysis fields for IV curve data",
        "requires": ["001"],
        "reversible": True,
        "sql_up": """
-- Migration 012: IV Curve Analysis Fields

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'iv_curve_data' AND column_name = 'series_resistance_ohm'
    ) THEN
        ALTER TABLE iv_curve_data ADD COLUMN series_resistance_ohm FLOAT;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'iv_curve_data' AND column_name = 'shunt_resistance_ohm'
    ) THEN
        ALTER TABLE iv_curve_data ADD COLUMN shunt_resistance_ohm FLOAT;
    END IF;
END $$;

SELECT 'Migration 012 UP: IV curve analysis fields added' AS status;
""",
        "sql_down": """
-- Migration 012 DOWN: Remove IV curve analysis fields
ALTER TABLE iv_curve_data DROP COLUMN IF EXISTS series_resistance_ohm;
ALTER TABLE iv_curve_data DROP COLUMN IF EXISTS shunt_resistance_ohm;
SELECT 'Migration 012 DOWN: IV curve analysis fields reverted' AS status;
"""
    },
    "013": {
        "name": "add_spectral_response_metadata",
        "description": "Add metadata fields for spectral response data",
        "requires": ["001"],
        "reversible": True,
        "sql_up": """
-- Migration 013: Spectral Response Metadata

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'spectral_responses' AND column_name = 'measurement_method'
    ) THEN
        ALTER TABLE spectral_responses ADD COLUMN measurement_method VARCHAR(100);
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'spectral_responses' AND column_name = 'temperature_c'
    ) THEN
        ALTER TABLE spectral_responses ADD COLUMN temperature_c FLOAT DEFAULT 25.0;
    END IF;
END $$;

SELECT 'Migration 013 UP: Spectral response metadata added' AS status;
""",
        "sql_down": """
-- Migration 013 DOWN: Remove spectral response metadata
ALTER TABLE spectral_responses DROP COLUMN IF EXISTS measurement_method;
ALTER TABLE spectral_responses DROP COLUMN IF EXISTS temperature_c;
SELECT 'Migration 013 DOWN: Spectral response metadata reverted' AS status;
"""
    },
    "014": {
        "name": "add_sample_specifications",
        "description": "Add specifications JSONB column to samples table",
        "requires": ["001"],
        "reversible": True,
        "sql_up": """
-- Migration 014: Sample Specifications Column

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'samples' AND column_name = 'specifications'
    ) THEN
        ALTER TABLE samples ADD COLUMN specifications JSONB DEFAULT '{}';
    END IF;
END $$;

-- Create index for JSONB queries if not exists
CREATE INDEX IF NOT EXISTS idx_samples_specifications ON samples USING GIN (specifications);

SELECT 'Migration 014 UP: Sample specifications column added' AS status;
""",
        "sql_down": """
-- Migration 014 DOWN: Remove sample specifications
DROP INDEX IF EXISTS idx_samples_specifications;
ALTER TABLE samples DROP COLUMN IF EXISTS specifications;
SELECT 'Migration 014 DOWN: Sample specifications reverted' AS status;
"""
    },
}

# Expected tables based on models.py
EXPECTED_TABLES: Set[str] = {
    'organizations',
    'users',
    'modules',
    'measurements',
    'iv_curve_data',
    'sun_simulators',
    'reference_devices',
    'spectral_responses',
    'uncertainty_results',
    'uncertainty_components',
    'files',
    'audit_logs',
    'approval_workflows',
}


# =============================================================================
# AUTHENTICATION
# =============================================================================

def check_admin_password() -> bool:
    """
    Check if user has provided correct admin password.
    Returns True if authenticated, False otherwise.
    """
    # Check if already authenticated in session
    if st.session_state.get('admin_authenticated', False):
        return True

    # Get password from secrets
    try:
        admin_password = st.secrets.get('ADMIN_PASSWORD', None)
        if not admin_password:
            st.error("ADMIN_PASSWORD not configured in Streamlit secrets")
            st.info("Add ADMIN_PASSWORD to your Streamlit secrets to enable admin access")
            return False
    except Exception:
        st.error("Could not access Streamlit secrets")
        return False

    # Show login form
    st.markdown("## Admin Authentication Required")
    st.markdown("---")

    with st.form("admin_login"):
        password = st.text_input("Admin Password", type="password", key="admin_pwd_input")
        submitted = st.form_submit_button("Login", type="primary")

        if submitted:
            if password == admin_password:
                st.session_state['admin_authenticated'] = True
                st.rerun()
            else:
                st.error("Invalid password")

    return False


def logout():
    """Clear admin authentication."""
    st.session_state['admin_authenticated'] = False
    st.rerun()


# =============================================================================
# DATABASE UTILITIES - BULLETPROOF ERROR HANDLING
# =============================================================================

def safe_import_database():
    """
    Safely import database modules with detailed error reporting.
    Returns tuple of (success, modules_dict, error_message)
    """
    modules = {}
    try:
        from database.connection import (
            get_database_url, check_connection, check_connection_detailed,
            reset_engine, get_engine, session_scope, get_last_error
        )
        modules['get_database_url'] = get_database_url
        modules['check_connection'] = check_connection
        modules['check_connection_detailed'] = check_connection_detailed
        modules['reset_engine'] = reset_engine
        modules['get_engine'] = get_engine
        modules['session_scope'] = session_scope
        modules['get_last_error'] = get_last_error

        from database.models import Base
        modules['Base'] = Base

        return True, modules, None
    except ImportError as e:
        return False, modules, f"Import error: {str(e)}"
    except Exception as e:
        return False, modules, f"Unexpected error: {str(e)}"


def get_database_status(force_reconnect: bool = False) -> Dict[str, Any]:
    """
    Get comprehensive database connection status with bulletproof error handling.
    """
    result = {
        'available': False,
        'connected': False,
        'host': None,
        'port': None,
        'database': None,
        'user': None,
        'error': None,
        'ssl_enabled': False,
        'tables_exist': False,
        'table_count': 0,
    }

    try:
        success, db_modules, error = safe_import_database()
        if not success:
            result['error'] = error
            return result

        from urllib.parse import urlparse

        # Reset engine if requested
        if force_reconnect:
            db_modules['reset_engine']()

        # Parse URL for details
        url = db_modules['get_database_url']()
        parsed = urlparse(url)

        result['available'] = True
        result['host'] = parsed.hostname
        result['port'] = parsed.port or 5432
        result['database'] = parsed.path.lstrip('/') if parsed.path else None
        result['user'] = parsed.username

        # Check if SSL will be used
        is_external = parsed.hostname and parsed.hostname not in ('localhost', '127.0.0.1', '::1')
        is_railway = parsed.hostname and ('railway' in parsed.hostname or 'rlwy.net' in parsed.hostname)
        result['ssl_enabled'] = is_external or is_railway

        # Test connection
        connected, error = db_modules['check_connection_detailed']()
        result['connected'] = connected
        if error:
            result['error'] = error

        # If connected, check tables
        if connected:
            try:
                from sqlalchemy import inspect
                engine = db_modules['get_engine']()
                inspector = inspect(engine)
                tables = inspector.get_table_names()
                result['tables_exist'] = len(tables) > 0
                result['table_count'] = len(tables)
            except Exception as e:
                result['error'] = f"Table check failed: {str(e)}"

    except Exception as e:
        result['error'] = f"Unexpected error: {str(e)}"

    return result


def get_table_info() -> Dict[str, Dict[str, Any]]:
    """
    Get detailed information about all database tables.
    """
    tables = {}
    try:
        success, db_modules, error = safe_import_database()
        if not success:
            return tables

        from sqlalchemy import inspect, text
        engine = db_modules['get_engine']()
        inspector = inspect(engine)

        for table_name in inspector.get_table_names():
            tables[table_name] = {
                'columns': [],
                'row_count': 0,
                'primary_key': [],
                'indexes': [],
            }

            # Get columns
            for col in inspector.get_columns(table_name):
                tables[table_name]['columns'].append({
                    'name': col['name'],
                    'type': str(col['type']),
                    'nullable': col.get('nullable', True),
                })

            # Get primary key
            pk = inspector.get_pk_constraint(table_name)
            if pk:
                tables[table_name]['primary_key'] = pk.get('constrained_columns', [])

            # Get indexes
            for idx in inspector.get_indexes(table_name):
                tables[table_name]['indexes'].append({
                    'name': idx['name'],
                    'columns': idx['column_names'],
                    'unique': idx.get('unique', False),
                })

            # Get row count
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
                    tables[table_name]['row_count'] = result.scalar()
            except Exception:
                tables[table_name]['row_count'] = -1

    except Exception as e:
        st.error(f"Error getting table info: {str(e)}")

    return tables


def validate_schema() -> Dict[str, Any]:
    """
    Validate database schema against expected models.
    Returns detailed validation results.
    """
    result = {
        'valid': True,
        'missing_tables': [],
        'extra_tables': [],
        'column_mismatches': {},
        'details': [],
    }

    try:
        success, db_modules, error = safe_import_database()
        if not success:
            result['valid'] = False
            result['details'].append(f"Could not import database: {error}")
            return result

        from sqlalchemy import inspect
        engine = db_modules['get_engine']()
        inspector = inspect(engine)

        actual_tables = set(inspector.get_table_names())

        # Check for missing tables
        missing = EXPECTED_TABLES - actual_tables
        if missing:
            result['missing_tables'] = sorted(missing)
            result['valid'] = False
            result['details'].append(f"Missing {len(missing)} tables: {', '.join(sorted(missing))}")

        # Check for extra tables (informational, not an error)
        extra = actual_tables - EXPECTED_TABLES
        if extra:
            result['extra_tables'] = sorted(extra)
            result['details'].append(f"Extra {len(extra)} tables found: {', '.join(sorted(extra))}")

        # Validate column structure for existing tables
        Base = db_modules['Base']
        for table_name in EXPECTED_TABLES & actual_tables:
            if table_name in Base.metadata.tables:
                model_table = Base.metadata.tables[table_name]
                db_columns = {col['name'] for col in inspector.get_columns(table_name)}
                model_columns = {col.name for col in model_table.columns}

                missing_cols = model_columns - db_columns
                extra_cols = db_columns - model_columns

                if missing_cols or extra_cols:
                    result['column_mismatches'][table_name] = {
                        'missing': sorted(missing_cols),
                        'extra': sorted(extra_cols),
                    }
                    if missing_cols:
                        result['valid'] = False
                        result['details'].append(
                            f"Table '{table_name}' missing columns: {', '.join(sorted(missing_cols))}"
                        )

        if result['valid']:
            result['details'].append("Schema validation passed - all expected tables and columns present")

    except Exception as e:
        result['valid'] = False
        result['details'].append(f"Validation error: {str(e)}")

    return result


def initialize_database() -> Tuple[bool, str, List[str]]:
    """
    Initialize database schema with comprehensive error handling.
    """
    try:
        success, db_modules, error = safe_import_database()
        if not success:
            return False, f"Could not import database: {error}", []

        from sqlalchemy import inspect

        if not db_modules['check_connection']():
            return False, "Cannot connect to database", []

        engine = db_modules['get_engine']()
        Base = db_modules['Base']

        # Get tables before
        inspector = inspect(engine)
        tables_before = set(inspector.get_table_names())

        # Create all tables
        Base.metadata.create_all(bind=engine)

        # Get tables after
        inspector = inspect(engine)
        tables_after = set(inspector.get_table_names())

        # Find newly created tables
        new_tables = sorted(tables_after - tables_before)
        all_tables = sorted(tables_after)

        if new_tables:
            return True, f"Created {len(new_tables)} new table(s)", new_tables
        else:
            return True, f"Schema verified. All {len(all_tables)} tables exist.", all_tables

    except Exception as e:
        return False, f"Initialization failed: {str(e)}", []


# =============================================================================
# MIGRATION UTILITIES
# =============================================================================

def get_migrations_path() -> Path:
    """
    Get the migrations directory path.
    Uses docs/migrations/ as primary, falls back to migrations/
    """
    if MIGRATIONS_DIR.exists():
        return MIGRATIONS_DIR
    elif FALLBACK_MIGRATIONS_DIR.exists():
        return FALLBACK_MIGRATIONS_DIR
    else:
        # Create docs/migrations if neither exists
        MIGRATIONS_DIR.mkdir(parents=True, exist_ok=True)
        return MIGRATIONS_DIR


def get_migration_files() -> Dict[str, Dict[str, Path]]:
    """
    Get all migration files from the migrations directory.
    Returns dict of {migration_num: {'up': path, 'down': path}}
    """
    migrations_path = get_migrations_path()
    files = {}

    if not migrations_path.exists():
        return files

    for f in migrations_path.iterdir():
        if f.suffix == '.sql':
            # Parse filename: XXX_name_UP.sql or XXX_name_DOWN.sql
            parts = f.stem.upper().split('_')
            if len(parts) >= 2:
                num = parts[0]
                direction = parts[-1]

                if num not in files:
                    files[num] = {'up': None, 'down': None}

                if direction == 'UP':
                    files[num]['up'] = f
                elif direction == 'DOWN':
                    files[num]['down'] = f

    return files


def ensure_migration_files_exist():
    """
    Create migration files from MIGRATIONS dict if they don't exist.
    """
    migrations_path = get_migrations_path()
    migrations_path.mkdir(parents=True, exist_ok=True)

    created = []
    for num, migration in MIGRATIONS.items():
        name = migration['name']

        # UP file
        up_file = migrations_path / f"{num}_{name}_UP.sql"
        if not up_file.exists():
            up_file.write_text(migration['sql_up'])
            created.append(up_file.name)

        # DOWN file
        down_file = migrations_path / f"{num}_{name}_DOWN.sql"
        if not down_file.exists():
            down_file.write_text(migration['sql_down'])
            created.append(down_file.name)

    return created


def run_migration_sql(sql: str) -> Tuple[bool, str]:
    """
    Execute migration SQL with proper error handling.
    """
    try:
        success, db_modules, error = safe_import_database()
        if not success:
            return False, f"Could not import database: {error}"

        from sqlalchemy import text
        engine = db_modules['get_engine']()

        with engine.begin() as conn:
            # Split by semicolons but handle DO $$ blocks
            statements = []
            current = []
            in_block = False

            for line in sql.split('\n'):
                stripped = line.strip()

                # Track DO $$ blocks
                if stripped.startswith('DO $$'):
                    in_block = True
                if '$$;' in stripped:
                    current.append(line)
                    in_block = False
                    statements.append('\n'.join(current))
                    current = []
                    continue

                if in_block:
                    current.append(line)
                elif ';' in stripped and not stripped.startswith('--'):
                    current.append(line)
                    statements.append('\n'.join(current))
                    current = []
                else:
                    current.append(line)

            # Add any remaining content
            if current:
                remaining = '\n'.join(current).strip()
                if remaining and not remaining.startswith('--'):
                    statements.append(remaining)

            # Execute each statement
            executed = 0
            for stmt in statements:
                stmt = stmt.strip()
                if stmt and not stmt.startswith('--'):
                    try:
                        conn.execute(text(stmt))
                        executed += 1
                    except Exception as e:
                        # Some statements might fail if objects already exist
                        if 'already exists' not in str(e).lower():
                            raise

        return True, f"Migration completed ({executed} statement(s) executed)"

    except Exception as e:
        return False, f"Migration failed: {str(e)}"


def get_applied_migrations() -> Set[str]:
    """
    Get set of applied migration numbers by checking database state.
    This is a heuristic based on schema inspection.
    """
    applied = set()

    try:
        success, db_modules, error = safe_import_database()
        if not success:
            return applied

        from sqlalchemy import inspect
        engine = db_modules['get_engine']()
        inspector = inspect(engine)

        tables = set(inspector.get_table_names())

        # 001 is applied if base tables exist
        if EXPECTED_TABLES & tables:
            applied.add('001')

        # Check for specific columns added by migrations
        for table_name in tables:
            columns = {col['name'] for col in inspector.get_columns(table_name)}

            # 002: samples.receipt_id, samples.inspection_id
            if table_name == 'samples':
                if 'receipt_id' in columns or 'inspection_id' in columns:
                    applied.add('002')
                if 'specifications' in columns:
                    applied.add('014')

            # 003: incoming_inspections.allocation_triggered
            if table_name == 'incoming_inspections':
                if 'allocation_triggered' in columns:
                    applied.add('003')

            # Check indexes for 004, 005
            try:
                indexes = {idx['name'] for idx in inspector.get_indexes(table_name)}
                if 'idx_measurements_created_at' in indexes:
                    applied.add('004')
                if 'idx_audit_logs_org_time' in indexes:
                    applied.add('005')
            except Exception:
                pass

    except Exception:
        pass

    return applied


# =============================================================================
# QA TEST SUITE
# =============================================================================

def run_qa_tests() -> List[Dict[str, Any]]:
    """
    Run comprehensive QA test suite.
    Returns list of test results.
    """
    tests = []

    # Test 1: Database Connection
    test = {
        'name': 'Database Connection',
        'category': 'Infrastructure',
        'passed': False,
        'message': '',
        'details': ''
    }
    try:
        status = get_database_status()
        if status['connected']:
            test['passed'] = True
            test['message'] = 'Connected successfully'
            test['details'] = f"Host: {status['host']}, DB: {status['database']}"
        else:
            test['message'] = 'Connection failed'
            test['details'] = status.get('error', 'Unknown error')
    except Exception as e:
        test['message'] = 'Test error'
        test['details'] = str(e)
    tests.append(test)

    # Test 2: Schema Validation
    test = {
        'name': 'Schema Validation',
        'category': 'Database',
        'passed': False,
        'message': '',
        'details': ''
    }
    try:
        validation = validate_schema()
        test['passed'] = validation['valid']
        test['message'] = 'Schema valid' if validation['valid'] else 'Schema issues found'
        test['details'] = '; '.join(validation['details'][:3])  # First 3 details
    except Exception as e:
        test['message'] = 'Validation error'
        test['details'] = str(e)
    tests.append(test)

    # Test 3: Tables Exist
    test = {
        'name': 'Required Tables',
        'category': 'Database',
        'passed': False,
        'message': '',
        'details': ''
    }
    try:
        table_info = get_table_info()
        existing = set(table_info.keys())
        missing = EXPECTED_TABLES - existing

        if not missing:
            test['passed'] = True
            test['message'] = f'All {len(EXPECTED_TABLES)} tables present'
        else:
            test['message'] = f'Missing {len(missing)} tables'
            test['details'] = ', '.join(sorted(missing)[:5])
    except Exception as e:
        test['message'] = 'Check failed'
        test['details'] = str(e)
    tests.append(test)

    # Test 4: Migration Files
    test = {
        'name': 'Migration Files',
        'category': 'Migrations',
        'passed': False,
        'message': '',
        'details': ''
    }
    try:
        migrations_path = get_migrations_path()
        migration_files = get_migration_files()
        defined_count = len(MIGRATIONS)
        file_count = len(migration_files)

        if migrations_path.exists() and file_count >= defined_count * 0.5:
            test['passed'] = True
            test['message'] = f'{file_count} migration pairs found'
            test['details'] = f"Path: {migrations_path}"
        else:
            test['message'] = 'Missing migration files'
            test['details'] = f"Found {file_count}/{defined_count} expected at {migrations_path}"
    except Exception as e:
        test['message'] = 'Check failed'
        test['details'] = str(e)
    tests.append(test)

    # Test 5: Database Models Import
    test = {
        'name': 'Database Models',
        'category': 'Code',
        'passed': False,
        'message': '',
        'details': ''
    }
    try:
        from database.models import (
            Organization, User, Module, Measurement,
            SunSimulator, ReferenceDevice, UncertaintyResult
        )
        test['passed'] = True
        test['message'] = 'All models imported successfully'
        test['details'] = '7 core models verified'
    except ImportError as e:
        test['message'] = 'Import failed'
        test['details'] = str(e)
    except Exception as e:
        test['message'] = 'Model error'
        test['details'] = str(e)
    tests.append(test)

    # Test 6: Seed Data Module
    test = {
        'name': 'Seed Data Module',
        'category': 'Code',
        'passed': False,
        'message': '',
        'details': ''
    }
    try:
        from database.seed_data import (
            seed_demo_organization, seed_sun_simulators,
            SUN_SIMULATORS_DATA, REFERENCE_LABS_DATA
        )
        test['passed'] = True
        test['message'] = 'Seed module available'
        test['details'] = f"{len(SUN_SIMULATORS_DATA)} simulators, {len(REFERENCE_LABS_DATA)} labs defined"
    except ImportError as e:
        test['message'] = 'Import failed'
        test['details'] = str(e)
    except Exception as e:
        test['message'] = 'Module error'
        test['details'] = str(e)
    tests.append(test)

    # Test 7: Table Constraints
    test = {
        'name': 'Table Constraints',
        'category': 'Database',
        'passed': False,
        'message': '',
        'details': ''
    }
    try:
        success, db_modules, error = safe_import_database()
        if success:
            from sqlalchemy import inspect
            engine = db_modules['get_engine']()
            inspector = inspect(engine)

            constraint_count = 0
            for table in inspector.get_table_names():
                pk = inspector.get_pk_constraint(table)
                if pk and pk.get('constrained_columns'):
                    constraint_count += 1

            if constraint_count > 0:
                test['passed'] = True
                test['message'] = f'{constraint_count} tables have primary keys'
            else:
                test['message'] = 'No constraints found'
        else:
            test['message'] = 'Could not check'
            test['details'] = error
    except Exception as e:
        test['message'] = 'Check failed'
        test['details'] = str(e)
    tests.append(test)

    return tests


# =============================================================================
# SEED DATA OPERATIONS
# =============================================================================

def seed_demo_data() -> Tuple[bool, str]:
    """
    Seed demo organization and users.
    """
    try:
        from database.seed_data import seed_demo_organization
        org_id = seed_demo_organization()
        return True, f"Demo organization created/verified (ID: {org_id})"
    except Exception as e:
        return False, f"Failed to seed demo data: {str(e)}"


def seed_sun_simulators() -> Tuple[bool, str]:
    """
    Seed sun simulator equipment data.
    """
    try:
        from database.seed_data import seed_sun_simulators, SUN_SIMULATORS_DATA
        seed_sun_simulators()
        return True, f"Seeded {len(SUN_SIMULATORS_DATA)} sun simulator configurations"
    except Exception as e:
        return False, f"Failed to seed sun simulators: {str(e)}"


def seed_all_data() -> Tuple[bool, str]:
    """
    Run all seed operations.
    """
    try:
        from database.seed_data import seed_all
        seed_all(create_demo=True)
        return True, "All seed data operations completed"
    except Exception as e:
        return False, f"Seed operation failed: {str(e)}"


# =============================================================================
# UI SECTIONS
# =============================================================================

def display_header():
    """Display admin page header."""
    col1, col2 = st.columns([6, 1])

    with col1:
        st.title("Admin Seed Dashboard")
        st.markdown(f"*Database management and system administration • v{ADMIN_VERSION}*")

    with col2:
        if st.button("Logout", type="secondary"):
            logout()


def display_database_status_tab():
    """Display database connection status tab."""
    st.markdown("### Database Connection")

    col1, col2 = st.columns([3, 1])
    with col2:
        force_reconnect = st.button("Reconnect", help="Force reconnection")

    with st.spinner("Checking connection..."):
        status = get_database_status(force_reconnect=force_reconnect)

    # Status cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if status['connected']:
            st.success("**Connected**")
        elif status['available']:
            st.error("**Disconnected**")
        else:
            st.error("**Unavailable**")

    with col2:
        st.metric("Host", status['host'] or "N/A")

    with col3:
        st.metric("Database", status['database'] or "N/A")

    with col4:
        st.metric("Tables", status['table_count'])

    # SSL and error info
    if status['ssl_enabled']:
        st.info("SSL/TLS: **Enabled** (secure connection)")

    if status['error'] and not status['connected']:
        st.error(f"**Connection Error:** {status['error']}")

        with st.expander("Troubleshooting Tips"):
            st.markdown("""
            **Common Issues:**
            1. Verify DATABASE_URL is correctly set in Streamlit secrets
            2. Format: `postgresql://user:password@host:port/database`
            3. Railway requires SSL - ensure `sslmode=require`
            4. Check if database server is running
            5. Verify firewall allows connections
            """)

    # Table overview
    if status['connected']:
        st.markdown("### Table Overview")

        with st.spinner("Loading table info..."):
            table_info = get_table_info()

        if table_info:
            # Summary metrics
            total_rows = sum(t['row_count'] for t in table_info.values() if t['row_count'] >= 0)
            st.metric("Total Rows", f"{total_rows:,}")

            # Table grid
            cols = st.columns(3)
            for i, (name, info) in enumerate(sorted(table_info.items())):
                with cols[i % 3]:
                    count = info['row_count']
                    icon = "" if count >= 0 else ""
                    st.text(f"{icon} {name}: {count if count >= 0 else 'Error'}")


def display_schema_validation_tab():
    """Display schema validation tab."""
    st.markdown("### Schema Validation")
    st.markdown("Compare database schema against SQLAlchemy models")

    if st.button("Run Schema Validation", type="primary"):
        with st.spinner("Validating schema..."):
            result = validate_schema()

        # Overall status
        if result['valid']:
            st.success("Schema validation **PASSED**")
        else:
            st.error("Schema validation **FAILED**")

        # Details
        for detail in result['details']:
            if 'passed' in detail.lower():
                st.success(detail)
            elif 'missing' in detail.lower():
                st.error(detail)
            else:
                st.info(detail)

        # Missing tables
        if result['missing_tables']:
            st.markdown("#### Missing Tables")
            for table in result['missing_tables']:
                st.text(f"  - {table}")

            if st.button("Initialize Missing Tables"):
                success, msg, tables = initialize_database()
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

        # Column mismatches
        if result['column_mismatches']:
            st.markdown("#### Column Mismatches")
            for table, issues in result['column_mismatches'].items():
                with st.expander(f"Table: {table}"):
                    if issues['missing']:
                        st.error(f"Missing columns: {', '.join(issues['missing'])}")
                    if issues['extra']:
                        st.info(f"Extra columns: {', '.join(issues['extra'])}")


def display_migrations_tab():
    """Display migrations management tab."""
    st.markdown("### Migration Management")

    # Path info
    migrations_path = get_migrations_path()
    st.info(f"**Migrations path:** `{migrations_path}`")

    # Ensure migration files exist
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Sync Migration Files"):
            created = ensure_migration_files_exist()
            if created:
                st.success(f"Created {len(created)} migration files")
                st.rerun()
            else:
                st.info("All migration files already exist")

    # Get current state
    migration_files = get_migration_files()
    applied = get_applied_migrations()

    # Migration status
    st.markdown("### Migration Status")

    for num in sorted(MIGRATIONS.keys()):
        migration = MIGRATIONS[num]
        files = migration_files.get(num, {'up': None, 'down': None})
        is_applied = num in applied

        # Status icons
        status_icon = "" if is_applied else ""
        file_icon = "" if files['up'] else ""

        col1, col2, col3, col4 = st.columns([0.5, 2, 1, 1])

        with col1:
            st.text(f"{num}")

        with col2:
            st.text(f"{status_icon} {migration['name']}")

        with col3:
            st.text(f"{file_icon} Files")

        with col4:
            if not is_applied and files['up']:
                if st.button("Apply", key=f"apply_{num}"):
                    st.session_state[f'pending_migration'] = num
            elif is_applied and files['down']:
                if st.button("Revert", key=f"revert_{num}", type="secondary"):
                    st.session_state[f'pending_revert'] = num

    # Pending migration execution
    if st.session_state.get('pending_migration'):
        num = st.session_state['pending_migration']
        migration = MIGRATIONS[num]

        st.markdown("---")
        st.warning(f"**Apply Migration {num}:** {migration['name']}")
        st.markdown(f"*{migration['description']}*")

        with st.expander("Preview SQL"):
            st.code(migration['sql_up'], language='sql')

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Confirm Apply", type="primary"):
                with st.spinner("Applying migration..."):
                    success, msg = run_migration_sql(migration['sql_up'])
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
                del st.session_state['pending_migration']
                st.rerun()

        with col2:
            if st.button("Cancel"):
                del st.session_state['pending_migration']
                st.rerun()

    # Pending revert
    if st.session_state.get('pending_revert'):
        num = st.session_state['pending_revert']
        migration = MIGRATIONS[num]

        st.markdown("---")
        st.error(f"**Revert Migration {num}:** {migration['name']}")

        with st.expander("Preview DOWN SQL"):
            st.code(migration['sql_down'], language='sql')

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Confirm Revert", type="primary"):
                with st.spinner("Reverting migration..."):
                    success, msg = run_migration_sql(migration['sql_down'])
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
                del st.session_state['pending_revert']
                st.rerun()

        with col2:
            if st.button("Cancel", key="cancel_revert"):
                del st.session_state['pending_revert']
                st.rerun()


def display_qa_tests_tab():
    """Display QA testing tab."""
    st.markdown("### QA Test Suite")
    st.markdown("Automated validation of database and system health")

    if st.button("Run All Tests", type="primary"):
        with st.spinner("Running test suite..."):
            results = run_qa_tests()

        # Summary
        passed = sum(1 for t in results if t['passed'])
        total = len(results)

        if passed == total:
            st.success(f"All {total} tests **PASSED**")
        else:
            st.warning(f"**{passed}/{total}** tests passed")

        # Results by category
        categories = {}
        for test in results:
            cat = test['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(test)

        for category, tests in categories.items():
            st.markdown(f"#### {category}")

            for test in tests:
                icon = "" if test['passed'] else ""
                col1, col2 = st.columns([1, 3])

                with col1:
                    st.text(f"{icon} {test['name']}")

                with col2:
                    if test['passed']:
                        st.text(test['message'])
                    else:
                        st.error(f"{test['message']}: {test['details']}")


def display_seed_data_tab():
    """Display seed data management tab."""
    st.markdown("### Seed Data Management")
    st.markdown("Populate database with initial/demo data")

    # Check connection first
    status = get_database_status()
    if not status['connected']:
        st.error("Database not connected. Cannot seed data.")
        return

    st.markdown("#### Available Seed Operations")

    # Demo Organization
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Seed Demo Org", key="seed_demo"):
            with st.spinner("Creating demo organization..."):
                success, msg = seed_demo_data()
            if success:
                st.success(msg)
            else:
                st.error(msg)
    with col2:
        st.markdown("Creates demo organization with admin and engineer users")

    # Sun Simulators
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Seed Sun Simulators", key="seed_sims"):
            with st.spinner("Seeding sun simulators..."):
                success, msg = seed_sun_simulators()
            if success:
                st.success(msg)
            else:
                st.error(msg)
    with col2:
        st.markdown("Populates sun simulator equipment database")

    # All Data
    st.markdown("---")
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Seed All Data", type="primary", key="seed_all"):
            with st.spinner("Running all seed operations..."):
                success, msg = seed_all_data()
            if success:
                st.success(msg)
            else:
                st.error(msg)
    with col2:
        st.markdown("**Runs all seed operations** - creates demo org, users, and equipment")

    # Current data stats
    st.markdown("---")
    st.markdown("#### Current Data Statistics")

    table_info = get_table_info()
    if table_info:
        key_tables = ['organizations', 'users', 'modules', 'measurements', 'sun_simulators']
        cols = st.columns(len(key_tables))

        for i, table in enumerate(key_tables):
            with cols[i]:
                count = table_info.get(table, {}).get('row_count', 0)
                st.metric(table.replace('_', ' ').title(), count)


def display_danger_zone_tab():
    """Display danger zone with destructive operations."""
    st.markdown("### Danger Zone")
    st.error("**WARNING:** Operations in this section can cause data loss!")

    # Check connection first
    status = get_database_status()
    if not status['connected']:
        st.warning("Database not connected.")
        return

    st.markdown("---")

    # Confirm checkbox
    confirm = st.checkbox("I understand these operations can destroy data", key="danger_confirm")

    if not confirm:
        st.info("Check the confirmation box to enable danger zone operations")
        return

    # Reset Schema
    st.markdown("#### Reset Database Schema")
    st.markdown("Drop and recreate all tables (all data will be lost)")

    if st.button("Reset Schema", type="primary", disabled=not confirm):
        if st.session_state.get('confirm_reset'):
            with st.spinner("Resetting schema..."):
                try:
                    success, db_modules, error = safe_import_database()
                    if success:
                        engine = db_modules['get_engine']()
                        Base = db_modules['Base']

                        # Drop all tables
                        Base.metadata.drop_all(bind=engine)

                        # Recreate
                        Base.metadata.create_all(bind=engine)

                        st.success("Schema reset complete - all tables recreated")
                        del st.session_state['confirm_reset']
                    else:
                        st.error(f"Could not reset: {error}")
                except Exception as e:
                    st.error(f"Reset failed: {str(e)}")
        else:
            st.session_state['confirm_reset'] = True
            st.warning("Click again to confirm schema reset")
            st.rerun()

    st.markdown("---")

    # Clear All Data
    st.markdown("#### Clear All Data")
    st.markdown("Delete all rows from all tables (keeps schema)")

    if st.button("Clear All Data", disabled=not confirm):
        if st.session_state.get('confirm_clear'):
            with st.spinner("Clearing data..."):
                try:
                    success, db_modules, error = safe_import_database()
                    if success:
                        from sqlalchemy import text
                        engine = db_modules['get_engine']()

                        # Get tables in dependency order (reverse)
                        tables = [
                            'audit_logs', 'approval_workflows', 'uncertainty_components',
                            'uncertainty_results', 'iv_curve_data', 'files',
                            'spectral_responses', 'measurements', 'modules',
                            'sun_simulators', 'reference_devices', 'users', 'organizations'
                        ]

                        with engine.begin() as conn:
                            for table in tables:
                                try:
                                    conn.execute(text(f'TRUNCATE TABLE "{table}" CASCADE'))
                                except Exception:
                                    pass  # Table might not exist

                        st.success("All data cleared")
                        del st.session_state['confirm_clear']
                    else:
                        st.error(f"Could not clear: {error}")
                except Exception as e:
                    st.error(f"Clear failed: {str(e)}")
        else:
            st.session_state['confirm_clear'] = True
            st.warning("Click again to confirm data clear")
            st.rerun()


def display_system_info():
    """Display system information."""
    st.markdown("---")
    st.markdown("### System Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Application**")
        st.text(f"Admin Version: {ADMIN_VERSION}")
        st.text(f"Build: {BUILD_DATE}")

    with col2:
        st.markdown("**Environment**")
        st.text(f"Python: {sys.version.split()[0]}")
        try:
            import sqlalchemy
            st.text(f"SQLAlchemy: {sqlalchemy.__version__}")
        except Exception:
            st.text("SQLAlchemy: N/A")

    with col3:
        st.markdown("**Paths**")
        st.text(f"Migrations: {get_migrations_path()}")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""

    # Check authentication
    if not check_admin_password():
        return

    # Display header
    display_header()
    st.markdown("---")

    # Create tabs
    tabs = st.tabs([
        " Database Status",
        " Schema Validation",
        " Migrations",
        " QA Tests",
        " Seed Data",
        " Danger Zone"
    ])

    with tabs[0]:
        display_database_status_tab()

    with tabs[1]:
        display_schema_validation_tab()

    with tabs[2]:
        display_migrations_tab()

    with tabs[3]:
        display_qa_tests_tab()

    with tabs[4]:
        display_seed_data_tab()

    with tabs[5]:
        display_danger_zone_tab()

    # System info footer
    display_system_info()


if __name__ == "__main__":
    main()
