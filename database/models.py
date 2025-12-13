"""
SQLAlchemy models for PV Measurement Uncertainty Tool
Comprehensive schema for Railway PostgreSQL database
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, JSON,
    ForeignKey, Enum, Index, CheckConstraint, UniqueConstraint, LargeBinary
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()


# =============================================================================
# ENUMS
# =============================================================================

class UserRole(enum.Enum):
    """User roles for access control."""
    ADMIN = "admin"
    ENGINEER = "engineer"
    REVIEWER = "reviewer"
    VIEWER = "viewer"


class ApprovalStatus(enum.Enum):
    """Approval workflow status."""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    REVISION_REQUESTED = "revision_requested"


class FileType(enum.Enum):
    """Types of uploaded files."""
    DATASHEET = "datasheet"
    CALIBRATION_CERT = "calibration_certificate"
    CLASSIFICATION_CERT = "classification_certificate"
    TEST_REPORT = "test_report"
    IV_CURVE_DATA = "iv_curve_data"
    SPECTRAL_RESPONSE = "spectral_response"
    PAN_FILE = "pan_file"
    OTHER = "other"


class MeasurementType(enum.Enum):
    """Types of PV measurements."""
    STC = "stc"
    NMOT = "nmot"
    LOW_IRRADIANCE = "low_irradiance"
    TEMPERATURE_COEFFICIENT = "temperature_coefficient"
    ENERGY_RATING = "energy_rating"
    BIFACIALITY = "bifaciality"
    IAM = "iam"
    SPECTRAL_RESPONSE = "spectral_response"


class PVTechnology(enum.Enum):
    """PV module technologies."""
    PERC = "PERC"
    TOPCON = "TOPCon"
    HJT = "HJT"
    PEROVSKITE = "Perovskite"
    TANDEM = "Perovskite-Silicon Tandem"
    CIGS = "CIGS"
    CDTE = "CdTe"
    CUSTOM = "Custom"


class AuditAction(enum.Enum):
    """Audit log action types."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    VIEW = "view"
    APPROVE = "approve"
    REJECT = "reject"
    SUBMIT = "submit"
    LOGIN = "login"
    LOGOUT = "logout"
    EXPORT = "export"


# =============================================================================
# USER & ORGANIZATION MODELS
# =============================================================================

class Organization(Base):
    """Laboratory or organization entity."""
    __tablename__ = 'organizations'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    address = Column(Text)
    accreditation_number = Column(String(100))
    accreditation_body = Column(String(100))
    logo_path = Column(String(500))
    website = Column(String(255))
    contact_email = Column(String(255))
    contact_phone = Column(String(50))

    # ISO 17025 document control
    document_format_prefix = Column(String(50), default='PV-UNC')
    record_prefix = Column(String(50), default='REC')

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Relationships
    users = relationship("User", back_populates="organization")
    measurements = relationship("Measurement", back_populates="organization")
    files = relationship("File", back_populates="organization")

    def __repr__(self):
        return f"<Organization(id={self.id}, name='{self.name}')>"


class User(Base):
    """User account for authentication and authorization."""
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer, ForeignKey('organizations.id'))

    # Authentication
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)

    # Profile
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    title = Column(String(100))  # e.g., "Test Engineer", "Technical Manager"
    phone = Column(String(50))

    # Authorization
    role = Column(Enum(UserRole), default=UserRole.VIEWER, nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)

    # Signature info (for reports)
    signature_image_path = Column(String(500))

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)

    # Relationships
    organization = relationship("Organization", back_populates="users")
    measurements_prepared = relationship("Measurement", foreign_keys="Measurement.preparer_id", back_populates="preparer")
    measurements_reviewed = relationship("Measurement", foreign_keys="Measurement.reviewer_id", back_populates="reviewer")
    measurements_approved = relationship("Measurement", foreign_keys="Measurement.approver_id", back_populates="approver")
    audit_logs = relationship("AuditLog", back_populates="user")
    files_uploaded = relationship("File", back_populates="uploaded_by_user")

    __table_args__ = (
        Index('idx_users_email', 'email'),
        Index('idx_users_org_role', 'organization_id', 'role'),
    )

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"

    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', role={self.role.value})>"


# =============================================================================
# MODULE MODELS
# =============================================================================

class Module(Base):
    """PV module under test."""
    __tablename__ = 'modules'

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer, ForeignKey('organizations.id'))

    # Identification
    serial_number = Column(String(100), index=True)
    model_name = Column(String(255))
    manufacturer = Column(String(255))

    # Technology
    technology = Column(Enum(PVTechnology), nullable=False)
    cell_type = Column(String(100))  # Detailed cell type

    # Physical specifications
    cells_series = Column(Integer)
    cells_parallel = Column(Integer, default=1)
    module_area_m2 = Column(Float)

    # Nameplate specifications (from datasheet)
    pmax_nameplate_w = Column(Float)
    voc_nameplate_v = Column(Float)
    isc_nameplate_a = Column(Float)
    vmp_nameplate_v = Column(Float)
    imp_nameplate_a = Column(Float)

    # Temperature coefficients
    gamma_pmax_pct_per_c = Column(Float)  # %/°C
    beta_voc_pct_per_c = Column(Float)  # %/°C
    alpha_isc_pct_per_c = Column(Float)  # %/°C

    # Bifacial properties
    is_bifacial = Column(Boolean, default=False)
    bifaciality_factor = Column(Float)

    # Metadata
    datasheet_file_id = Column(Integer, ForeignKey('files.id'))
    notes = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    measurements = relationship("Measurement", back_populates="module")
    datasheet = relationship("File", foreign_keys=[datasheet_file_id])

    __table_args__ = (
        Index('idx_modules_serial', 'serial_number'),
        Index('idx_modules_manufacturer', 'manufacturer'),
    )

    def __repr__(self):
        return f"<Module(id={self.id}, serial='{self.serial_number}', model='{self.model_name}')>"


# =============================================================================
# REFERENCE DEVICE MODELS
# =============================================================================

class ReferenceDevice(Base):
    """WPVS reference cells and reference modules."""
    __tablename__ = 'reference_devices'

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer, ForeignKey('organizations.id'))

    # Identification
    serial_number = Column(String(100), nullable=False, index=True)
    device_type = Column(String(50), nullable=False)  # 'WPVS_CELL', 'REFERENCE_MODULE'
    manufacturer = Column(String(255))
    model = Column(String(255))

    # Calibration info
    calibration_lab = Column(String(255))
    calibration_lab_accreditation = Column(String(255))
    calibration_date = Column(DateTime)
    calibration_expiry = Column(DateTime)

    # Calibration values
    isc_calibrated_a = Column(Float)  # Calibrated Isc at STC
    pmax_calibrated_w = Column(Float)  # For reference modules
    calibration_uncertainty_pct = Column(Float)  # Expanded uncertainty (k=2)

    # Drift tracking
    last_verification_date = Column(DateTime)
    estimated_drift_pct = Column(Float, default=0.0)

    # Spectral response data
    has_spectral_response = Column(Boolean, default=False)

    # Files
    calibration_cert_id = Column(Integer, ForeignKey('files.id'))

    # Status
    is_active = Column(Boolean, default=True)
    notes = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    spectral_responses = relationship("SpectralResponse", back_populates="reference_device")
    calibration_cert = relationship("File", foreign_keys=[calibration_cert_id])
    measurements = relationship("Measurement", back_populates="reference_device")

    __table_args__ = (
        Index('idx_ref_device_serial', 'serial_number'),
        Index('idx_ref_device_expiry', 'calibration_expiry'),
    )

    def __repr__(self):
        return f"<ReferenceDevice(id={self.id}, type='{self.device_type}', serial='{self.serial_number}')>"


class SpectralResponse(Base):
    """Spectral response data for reference devices and modules."""
    __tablename__ = 'spectral_responses'

    id = Column(Integer, primary_key=True, autoincrement=True)
    reference_device_id = Column(Integer, ForeignKey('reference_devices.id'))
    module_id = Column(Integer, ForeignKey('modules.id'))

    # Source
    source_lab = Column(String(255))  # e.g., NREL, PTB, Fraunhofer ISE
    measurement_date = Column(DateTime)
    measurement_number = Column(Integer, default=1)  # For multiple measurements

    # Data (JSON array of {wavelength_nm: float, sr_a_per_w: float})
    wavelength_data = Column(JSON)  # [{wavelength: 300, sr: 0.001}, ...]

    # Wavelength range
    min_wavelength_nm = Column(Float)
    max_wavelength_nm = Column(Float)
    wavelength_step_nm = Column(Float)

    # Quality indicators
    uncertainty_pct = Column(Float)
    notes = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    reference_device = relationship("ReferenceDevice", back_populates="spectral_responses")
    module = relationship("Module")

    def __repr__(self):
        return f"<SpectralResponse(id={self.id}, device_id={self.reference_device_id})>"


# =============================================================================
# SUN SIMULATOR MODEL
# =============================================================================

class SunSimulator(Base):
    """Sun simulator equipment configuration."""
    __tablename__ = 'sun_simulators'

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer, ForeignKey('organizations.id'))

    # Equipment identification
    manufacturer = Column(String(255), nullable=False)
    model = Column(String(255), nullable=False)
    serial_number = Column(String(100))

    # Lamp type and specifications
    lamp_type = Column(String(50))  # LED, Xenon, Metal Halide, Plasma
    classification = Column(String(20))  # AAA, AA+, A+, BBB, etc.

    # Typical performance (can be overridden per measurement)
    typical_uniformity_pct = Column(Float)
    typical_temporal_instability_pct = Column(Float)
    typical_spectral_match = Column(String(10))  # A, B, C

    # Geometry
    standard_distance_mm = Column(Float)
    test_area_m2 = Column(Float)

    # Spectral data
    has_spectral_data = Column(Boolean, default=False)
    spectrum_data = Column(JSON)  # Wavelength vs intensity data

    # Classification certificate
    classification_cert_id = Column(Integer, ForeignKey('files.id'))
    last_classification_date = Column(DateTime)

    # Status
    is_active = Column(Boolean, default=True)
    notes = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    classification_cert = relationship("File", foreign_keys=[classification_cert_id])
    measurements = relationship("Measurement", back_populates="sun_simulator")

    def __repr__(self):
        return f"<SunSimulator(id={self.id}, model='{self.model}', class='{self.classification}')>"


# =============================================================================
# MEASUREMENT MODELS
# =============================================================================

class Measurement(Base):
    """Individual PV measurement session."""
    __tablename__ = 'measurements'

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer, ForeignKey('organizations.id'))

    # References
    module_id = Column(Integer, ForeignKey('modules.id'), nullable=False)
    reference_device_id = Column(Integer, ForeignKey('reference_devices.id'))
    sun_simulator_id = Column(Integer, ForeignKey('sun_simulators.id'))

    # Measurement identification
    measurement_number = Column(String(50), unique=True, index=True)
    measurement_type = Column(Enum(MeasurementType), default=MeasurementType.STC)
    test_date = Column(DateTime, nullable=False)

    # Test conditions
    irradiance_target_w_m2 = Column(Float, default=1000.0)
    temperature_target_c = Column(Float, default=25.0)
    spectrum = Column(String(20), default='AM1.5G')

    # Actual conditions during measurement
    actual_irradiance_w_m2 = Column(Float)
    actual_temperature_c = Column(Float)
    ambient_temperature_c = Column(Float)
    relative_humidity_pct = Column(Float)

    # Measured parameters (STC corrected)
    voc_v = Column(Float)
    isc_a = Column(Float)
    vmp_v = Column(Float)
    imp_a = Column(Float)
    pmax_w = Column(Float)
    fill_factor = Column(Float)

    # Efficiency
    efficiency_pct = Column(Float)

    # Repeatability (if multiple flashes)
    number_of_flashes = Column(Integer, default=1)
    pmax_std_dev_pct = Column(Float)  # Standard deviation from multiple measurements

    # Workflow status
    approval_status = Column(Enum(ApprovalStatus), default=ApprovalStatus.DRAFT)
    preparer_id = Column(Integer, ForeignKey('users.id'))
    reviewer_id = Column(Integer, ForeignKey('users.id'))
    approver_id = Column(Integer, ForeignKey('users.id'))
    submitted_at = Column(DateTime)
    reviewed_at = Column(DateTime)
    approved_at = Column(DateTime)

    # Notes and metadata
    notes = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    organization = relationship("Organization", back_populates="measurements")
    module = relationship("Module", back_populates="measurements")
    reference_device = relationship("ReferenceDevice", back_populates="measurements")
    sun_simulator = relationship("SunSimulator", back_populates="measurements")
    preparer = relationship("User", foreign_keys=[preparer_id], back_populates="measurements_prepared")
    reviewer = relationship("User", foreign_keys=[reviewer_id], back_populates="measurements_reviewed")
    approver = relationship("User", foreign_keys=[approver_id], back_populates="measurements_approved")
    iv_curves = relationship("IVCurveData", back_populates="measurement")
    uncertainty_results = relationship("UncertaintyResult", back_populates="measurement")

    __table_args__ = (
        Index('idx_measurements_date', 'test_date'),
        Index('idx_measurements_status', 'approval_status'),
        Index('idx_measurements_module', 'module_id'),
    )

    def __repr__(self):
        return f"<Measurement(id={self.id}, number='{self.measurement_number}', type={self.measurement_type.value})>"


class IVCurveData(Base):
    """Raw I-V curve data points."""
    __tablename__ = 'iv_curve_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    measurement_id = Column(Integer, ForeignKey('measurements.id'), nullable=False)

    # Flash/measurement number within the session
    flash_number = Column(Integer, default=1)

    # I-V curve data (JSON array of {v: float, i: float})
    voltage_current_data = Column(JSON)  # [{v: 0, i: 10.5}, {v: 0.5, i: 10.4}, ...]

    # Number of data points
    num_points = Column(Integer)

    # Measurement conditions for this specific curve
    irradiance_w_m2 = Column(Float)
    temperature_c = Column(Float)
    timestamp = Column(DateTime)

    # Scan direction
    scan_direction = Column(String(20))  # 'forward' (Isc->Voc) or 'reverse' (Voc->Isc)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    measurement = relationship("Measurement", back_populates="iv_curves")

    __table_args__ = (
        Index('idx_iv_curve_measurement', 'measurement_id'),
    )

    def __repr__(self):
        return f"<IVCurveData(id={self.id}, measurement_id={self.measurement_id}, flash={self.flash_number})>"


# =============================================================================
# UNCERTAINTY RESULTS MODELS
# =============================================================================

class UncertaintyResult(Base):
    """Calculated uncertainty analysis results."""
    __tablename__ = 'uncertainty_results'

    id = Column(Integer, primary_key=True, autoincrement=True)
    measurement_id = Column(Integer, ForeignKey('measurements.id'), nullable=False)

    # Analysis metadata
    analysis_date = Column(DateTime, default=datetime.utcnow)
    analysis_version = Column(String(50), default='1.0')  # Tool version
    calculation_method = Column(String(50), default='GUM')  # GUM, Monte Carlo

    # Target parameter
    target_parameter = Column(String(50), default='Pmax')  # Pmax, Voc, Isc, etc.
    measured_value = Column(Float)
    measured_unit = Column(String(20), default='W')

    # Combined uncertainty
    combined_standard_uncertainty_pct = Column(Float)
    expanded_uncertainty_k2_pct = Column(Float)
    absolute_uncertainty = Column(Float)

    # Confidence interval
    confidence_level_pct = Column(Float, default=95.0)
    ci_lower = Column(Float)
    ci_upper = Column(Float)

    # Coverage factor
    coverage_factor = Column(Float, default=2.0)
    effective_degrees_of_freedom = Column(Float)

    # Monte Carlo results (if applicable)
    monte_carlo_samples = Column(Integer)
    mc_mean = Column(Float)
    mc_std_dev = Column(Float)
    mc_percentile_2_5 = Column(Float)
    mc_percentile_97_5 = Column(Float)

    # Full budget as JSON
    full_budget_json = Column(JSON)

    # Report generated
    report_generated = Column(Boolean, default=False)
    report_file_id = Column(Integer, ForeignKey('files.id'))

    created_at = Column(DateTime, default=datetime.utcnow)
    created_by_id = Column(Integer, ForeignKey('users.id'))

    # Relationships
    measurement = relationship("Measurement", back_populates="uncertainty_results")
    components = relationship("UncertaintyComponent", back_populates="uncertainty_result")
    report_file = relationship("File", foreign_keys=[report_file_id])

    __table_args__ = (
        Index('idx_uncertainty_measurement', 'measurement_id'),
        Index('idx_uncertainty_date', 'analysis_date'),
    )

    def __repr__(self):
        return f"<UncertaintyResult(id={self.id}, measurement={self.measurement_id}, unc={self.expanded_uncertainty_k2_pct:.2f}%)>"


class UncertaintyComponent(Base):
    """Individual uncertainty budget components."""
    __tablename__ = 'uncertainty_components'

    id = Column(Integer, primary_key=True, autoincrement=True)
    uncertainty_result_id = Column(Integer, ForeignKey('uncertainty_results.id'), nullable=False)

    # Component identification (matches fishbone structure)
    category_id = Column(String(10))  # '1', '2', '3', etc.
    subcategory_id = Column(String(10))  # '1.1', '1.2', etc.
    factor_id = Column(String(20))  # '1.1.1', '1.1.2', etc.
    name = Column(String(255), nullable=False)

    # Values
    input_value = Column(Float)  # Original input value
    standard_uncertainty = Column(Float)
    distribution = Column(String(50))  # normal, rectangular, triangular
    sensitivity_coefficient = Column(Float, default=1.0)
    unit = Column(String(20))

    # Contribution
    variance_contribution = Column(Float)
    percentage_contribution = Column(Float)

    # Notes
    notes = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    uncertainty_result = relationship("UncertaintyResult", back_populates="components")

    __table_args__ = (
        Index('idx_unc_component_result', 'uncertainty_result_id'),
        Index('idx_unc_component_category', 'category_id'),
    )

    def __repr__(self):
        return f"<UncertaintyComponent(id={self.id}, factor='{self.factor_id}', contrib={self.percentage_contribution:.1f}%)>"


# =============================================================================
# FILE MANAGEMENT MODEL
# =============================================================================

class File(Base):
    """File storage and management."""
    __tablename__ = 'files'

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer, ForeignKey('organizations.id'))

    # File identification
    original_filename = Column(String(500), nullable=False)
    stored_filename = Column(String(500), nullable=False)  # UUID-based name
    file_type = Column(Enum(FileType), nullable=False)
    mime_type = Column(String(100))
    file_size_bytes = Column(Integer)

    # Storage location
    storage_path = Column(String(1000))  # Path or S3 key
    storage_backend = Column(String(50), default='local')  # local, s3, railway

    # File hash for integrity
    sha256_hash = Column(String(64))

    # Metadata
    description = Column(Text)
    uploaded_by = Column(Integer, ForeignKey('users.id'))

    # Related entities (optional - can link to specific records)
    related_module_id = Column(Integer, ForeignKey('modules.id'))
    related_measurement_id = Column(Integer, ForeignKey('measurements.id'))
    related_reference_device_id = Column(Integer, ForeignKey('reference_devices.id'))

    # Approval workflow for files
    approval_status = Column(Enum(ApprovalStatus), default=ApprovalStatus.DRAFT)
    approved_by = Column(Integer, ForeignKey('users.id'))
    approved_at = Column(DateTime)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Soft delete
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime)

    # Relationships
    organization = relationship("Organization", back_populates="files")
    uploaded_by_user = relationship("User", foreign_keys=[uploaded_by], back_populates="files_uploaded")

    __table_args__ = (
        Index('idx_files_org', 'organization_id'),
        Index('idx_files_type', 'file_type'),
        Index('idx_files_status', 'approval_status'),
    )

    def __repr__(self):
        return f"<File(id={self.id}, name='{self.original_filename}', type={self.file_type.value})>"


# =============================================================================
# AUDIT LOG MODEL
# =============================================================================

class AuditLog(Base):
    """Comprehensive audit trail for QMS compliance."""
    __tablename__ = 'audit_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    organization_id = Column(Integer, ForeignKey('organizations.id'))
    user_id = Column(Integer, ForeignKey('users.id'))

    # Action details
    action = Column(Enum(AuditAction), nullable=False)
    entity_type = Column(String(100), nullable=False)  # 'Measurement', 'File', 'User', etc.
    entity_id = Column(Integer)

    # Change tracking
    old_values = Column(JSON)  # Previous state
    new_values = Column(JSON)  # New state

    # Context
    ip_address = Column(String(45))  # IPv4 or IPv6
    user_agent = Column(String(500))
    session_id = Column(String(100))

    # Additional metadata
    description = Column(Text)

    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    user = relationship("User", back_populates="audit_logs")

    __table_args__ = (
        Index('idx_audit_timestamp', 'timestamp'),
        Index('idx_audit_entity', 'entity_type', 'entity_id'),
        Index('idx_audit_user', 'user_id'),
        Index('idx_audit_action', 'action'),
    )

    def __repr__(self):
        return f"<AuditLog(id={self.id}, action={self.action.value}, entity={self.entity_type}:{self.entity_id})>"


# =============================================================================
# APPROVAL WORKFLOW MODEL
# =============================================================================

class ApprovalWorkflow(Base):
    """Tracks approval workflow history."""
    __tablename__ = 'approval_workflows'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Entity being approved
    entity_type = Column(String(100), nullable=False)  # 'Measurement', 'File', etc.
    entity_id = Column(Integer, nullable=False)

    # Workflow step
    from_status = Column(Enum(ApprovalStatus))
    to_status = Column(Enum(ApprovalStatus), nullable=False)

    # Actor
    acted_by_id = Column(Integer, ForeignKey('users.id'), nullable=False)

    # Comments/feedback
    comments = Column(Text)

    timestamp = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_workflow_entity', 'entity_type', 'entity_id'),
        Index('idx_workflow_timestamp', 'timestamp'),
    )

    def __repr__(self):
        return f"<ApprovalWorkflow(id={self.id}, entity={self.entity_type}:{self.entity_id}, status={self.to_status.value})>"
