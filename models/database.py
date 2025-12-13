"""
PostgreSQL Database Models for Solar PV Uncertainty Tool.

This module defines the SQLAlchemy ORM models for the production database.
Tables: users, reference_devices, simulators, spectral_data, measurement_files, uncertainty_results
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text, DateTime,
    Boolean, ForeignKey, JSON, Enum as SQLEnum, Index, UniqueConstraint, text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import JSONB
import enum

# Database Base
Base = declarative_base()


# ============================================================================
# ENUMS
# ============================================================================

class UserRole(enum.Enum):
    """User role types."""
    ADMIN = "admin"
    ENGINEER = "engineer"
    VIEWER = "viewer"


class LabType(enum.Enum):
    """Reference laboratory types."""
    PRIMARY = "Primary"
    SECONDARY = "Secondary"
    ACCREDITED = "Accredited"


class LampType(enum.Enum):
    """Sun simulator lamp types."""
    LED = "LED"
    XENON = "Xenon"
    METAL_HALIDE = "Metal Halide"
    PLASMA = "Plasma"
    CUSTOM = "Custom"


class FileType(enum.Enum):
    """Measurement file types."""
    IV_CURVE = "iv_curve"
    CALIBRATION_CERT = "calibration_cert"
    DATASHEET = "datasheet"
    TEST_REPORT = "test_report"
    SPECTRUM = "spectrum"
    CONTROL_CHART = "control_chart"


class UncertaintyStatus(enum.Enum):
    """Uncertainty calculation status."""
    DRAFT = "draft"
    CALCULATED = "calculated"
    VALIDATED = "validated"
    ARCHIVED = "archived"


# ============================================================================
# TABLE 1: USERS
# ============================================================================

class User(Base):
    """
    Application users table.
    Stores user authentication and profile information.
    """
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(SQLEnum(UserRole), default=UserRole.ENGINEER)

    # Profile info
    full_name = Column(String(255))
    organization = Column(String(255))
    country = Column(String(100))

    # Settings
    preferred_currency = Column(String(3), default='USD')
    preferred_units = Column(String(20), default='SI')

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    is_active = Column(Boolean, default=True)

    # Relationships
    measurement_files = relationship("MeasurementFile", back_populates="user")
    uncertainty_results = relationship("UncertaintyResult", back_populates="user")

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', role={self.role.value})>"


# ============================================================================
# TABLE 2: REFERENCE DEVICES
# ============================================================================

class ReferenceDevice(Base):
    """
    Reference device calibration laboratories and devices table.
    Stores information about calibration labs (NREL, PTB, Fraunhofer ISE, etc.)
    and their associated reference cells/modules.
    """
    __tablename__ = 'reference_devices'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Laboratory info
    lab_code = Column(String(50), unique=True, nullable=False, index=True)
    lab_name = Column(String(255), nullable=False)
    country = Column(String(100))
    lab_type = Column(SQLEnum(LabType), nullable=False)
    accreditation = Column(String(255))

    # Uncertainty specifications
    typical_uncertainty_wpvs = Column(Float)  # % for WPVS cells
    typical_uncertainty_module = Column(Float)  # % for reference modules

    # Contact and metadata
    website = Column(String(500))
    contact_email = Column(String(255))
    description = Column(Text)

    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    uncertainty_results = relationship("UncertaintyResult", back_populates="reference_device")

    def __repr__(self):
        return f"<ReferenceDevice(id={self.id}, lab_code='{self.lab_code}', lab_type={self.lab_type.value})>"


# ============================================================================
# TABLE 3: SIMULATORS
# ============================================================================

class Simulator(Base):
    """
    Sun simulator configurations table.
    Stores manufacturer/model specifications for solar simulators.
    """
    __tablename__ = 'simulators'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Identification
    simulator_code = Column(String(100), unique=True, nullable=False, index=True)
    manufacturer = Column(String(255), nullable=False)
    model = Column(String(255), nullable=False)

    # Technical specifications
    lamp_type = Column(SQLEnum(LampType), nullable=False)
    classification = Column(String(20))  # AAA, AA+, A+, BBB, etc.

    # Performance metrics
    typical_uniformity = Column(Float)  # %
    typical_temporal_instability = Column(Float)  # %
    typical_spectral_match = Column(String(10))  # A, B, C
    standard_distance_mm = Column(Float)  # Distance from lamp to test plane

    # Additional specs
    test_area_mm2 = Column(Float)
    flash_duration_ms = Column(Float)
    description = Column(Text)

    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    uncertainty_results = relationship("UncertaintyResult", back_populates="simulator")

    def __repr__(self):
        return f"<Simulator(id={self.id}, code='{self.simulator_code}', classification='{self.classification}')>"


# ============================================================================
# TABLE 4: SPECTRAL DATA
# ============================================================================

class SpectralData(Base):
    """
    Standard spectra and spectral mismatch data table.
    Stores AM1.5G, AM1.5D, AM0, and custom spectrum information.
    """
    __tablename__ = 'spectral_data'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Identification
    spectrum_code = Column(String(50), unique=True, nullable=False, index=True)
    spectrum_name = Column(String(255), nullable=False)

    # Specifications
    standard = Column(String(100))  # IEC 60904-3, ASTM G173, etc.
    air_mass = Column(Float)
    integrated_irradiance = Column(Float)  # W/m²

    # Data
    wavelength_range_nm = Column(String(50))  # e.g., "280-4000"
    data_points = Column(Integer)
    spectrum_data = Column(JSONB)  # Store wavelength/irradiance pairs

    # Metadata
    description = Column(Text)
    source_url = Column(String(500))
    is_standard = Column(Boolean, default=True)  # True for IEC/ASTM, False for custom

    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    uncertainty_results = relationship("UncertaintyResult", back_populates="spectrum")

    def __repr__(self):
        return f"<SpectralData(id={self.id}, code='{self.spectrum_code}', standard='{self.standard}')>"


# ============================================================================
# TABLE 5: MEASUREMENT FILES
# ============================================================================

class MeasurementFile(Base):
    """
    Uploaded measurement files table.
    Tracks all uploaded files (I-V curves, certificates, datasheets, etc.)
    """
    __tablename__ = 'measurement_files'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # File identification
    filename = Column(String(500), nullable=False)
    original_filename = Column(String(500), nullable=False)
    file_type = Column(SQLEnum(FileType), nullable=False)
    mime_type = Column(String(100))
    file_size_bytes = Column(Integer)

    # Storage
    storage_path = Column(String(1000))  # Cloud storage path or local path
    storage_provider = Column(String(50), default='local')  # local, s3, gcs, etc.
    checksum = Column(String(64))  # SHA-256 hash

    # Extracted data
    extracted_data = Column(JSONB)  # Parsed data from file
    extraction_status = Column(String(50), default='pending')  # pending, success, failed
    extraction_error = Column(Text)

    # Relationships
    user_id = Column(Integer, ForeignKey('users.id'), index=True)
    user = relationship("User", back_populates="measurement_files")

    # Metadata
    description = Column(Text)
    tags = Column(JSONB)  # Array of tags for searching

    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)

    # Indexes
    __table_args__ = (
        Index('ix_measurement_files_user_type', 'user_id', 'file_type'),
    )

    def __repr__(self):
        return f"<MeasurementFile(id={self.id}, filename='{self.filename}', type={self.file_type.value})>"


# ============================================================================
# TABLE 6: UNCERTAINTY RESULTS
# ============================================================================

class UncertaintyResult(Base):
    """
    Uncertainty calculation results table.
    Stores complete uncertainty budgets and calculation results.
    """
    __tablename__ = 'uncertainty_results'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Calculation identification
    calculation_name = Column(String(255), nullable=False)
    calculation_reference = Column(String(100), unique=True, index=True)
    status = Column(SQLEnum(UncertaintyStatus), default=UncertaintyStatus.DRAFT)

    # Module information
    module_manufacturer = Column(String(255))
    module_model = Column(String(255))
    module_serial = Column(String(100))
    technology = Column(String(50))  # PERC, TOPCon, HJT, etc.

    # Measurement conditions
    measurement_type = Column(String(50))  # STC, NMOT, Low_Irradiance, etc.
    irradiance = Column(Float)  # W/m²
    temperature = Column(Float)  # °C

    # Measured values
    pmax_measured = Column(Float)  # W
    voc_measured = Column(Float)  # V
    isc_measured = Column(Float)  # A
    vmp_measured = Column(Float)  # V
    imp_measured = Column(Float)  # A
    ff_measured = Column(Float)  # Fill factor

    # Uncertainty results
    combined_std_uncertainty = Column(Float)  # W
    expanded_uncertainty_k2 = Column(Float)  # W (k=2)
    relative_uncertainty_pct = Column(Float)  # %
    coverage_factor = Column(Float, default=2.0)

    # Confidence intervals
    confidence_68_lower = Column(Float)
    confidence_68_upper = Column(Float)
    confidence_95_lower = Column(Float)
    confidence_95_upper = Column(Float)
    confidence_99_lower = Column(Float)
    confidence_99_upper = Column(Float)

    # Uncertainty budget (detailed breakdown)
    uncertainty_budget = Column(JSONB)  # Full GUM-style budget
    category_contributions = Column(JSONB)  # Contributions by category

    # Financial impact
    financial_impact = Column(JSONB)  # Price adjustments, NPV impact, etc.

    # Foreign keys
    user_id = Column(Integer, ForeignKey('users.id'), index=True)
    reference_device_id = Column(Integer, ForeignKey('reference_devices.id'))
    simulator_id = Column(Integer, ForeignKey('simulators.id'))
    spectrum_id = Column(Integer, ForeignKey('spectral_data.id'))

    # Relationships
    user = relationship("User", back_populates="uncertainty_results")
    reference_device = relationship("ReferenceDevice", back_populates="uncertainty_results")
    simulator = relationship("Simulator", back_populates="uncertainty_results")
    spectrum = relationship("SpectralData", back_populates="uncertainty_results")

    # Metadata
    notes = Column(Text)
    report_generated = Column(Boolean, default=False)

    # Timestamps
    calculated_at = Column(DateTime, default=datetime.utcnow)
    validated_at = Column(DateTime)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index('ix_uncertainty_results_user_status', 'user_id', 'status'),
        Index('ix_uncertainty_results_technology', 'technology'),
    )

    def __repr__(self):
        return f"<UncertaintyResult(id={self.id}, ref='{self.calculation_reference}', status={self.status.value})>"


# ============================================================================
# DATABASE CONNECTION AND SESSION MANAGEMENT
# ============================================================================

def get_database_url() -> str:
    """Get database URL from environment variable."""
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")

    # Railway uses postgres:// but SQLAlchemy needs postgresql://
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)

    return database_url


def create_db_engine(echo: bool = False):
    """Create database engine with connection pooling."""
    database_url = get_database_url()

    engine = create_engine(
        database_url,
        echo=echo,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800
    )
    return engine


def create_session_factory(engine):
    """Create session factory for database operations."""
    Session = sessionmaker(bind=engine)
    return Session


def init_db(engine):
    """Initialize database - create all tables."""
    Base.metadata.create_all(engine)
    print("Database tables created successfully!")


def drop_all_tables(engine):
    """Drop all tables (use with caution!)."""
    Base.metadata.drop_all(engine)
    print("All database tables dropped!")


def test_connection():
    """Test database connection."""
    try:
        engine = create_db_engine()
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            print("Database connection successful!")
            return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_all_table_names() -> List[str]:
    """Get list of all table names."""
    return [
        'users',
        'reference_devices',
        'simulators',
        'spectral_data',
        'measurement_files',
        'uncertainty_results'
    ]


if __name__ == "__main__":
    # Test database connection
    print("Testing database connection...")
    test_connection()
