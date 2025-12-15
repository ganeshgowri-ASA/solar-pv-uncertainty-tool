-- Migration: 001_initial_schema.sql
-- Description: Initial PostgreSQL schema for Solar PV Uncertainty Tool
-- Date: 2025-12-15
-- Compatible with: PostgreSQL 14+, SQLAlchemy 2.0

-- ============================================
-- UP MIGRATION
-- ============================================

-- Enable UUID extension for unique identifiers
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- ENUM TYPES
-- ============================================

CREATE TYPE user_role AS ENUM ('admin', 'technician', 'reviewer', 'client', 'readonly');
CREATE TYPE request_status AS ENUM ('draft', 'submitted', 'in_progress', 'pending_review', 'completed', 'cancelled');
CREATE TYPE sample_status AS ENUM ('pending', 'received', 'in_testing', 'tested', 'reported', 'returned', 'disposed');
CREATE TYPE measurement_type AS ENUM ('STC', 'NMOT', 'LI', 'LID', 'PID', 'EL', 'IR', 'visual');
CREATE TYPE calibration_status AS ENUM ('valid', 'expiring_soon', 'expired', 'needs_recalibration');
CREATE TYPE report_status AS ENUM ('draft', 'pending_review', 'approved', 'issued', 'superseded', 'cancelled');

-- ============================================
-- CORE TABLES
-- ============================================

-- Organizations/Customers table
CREATE TABLE IF NOT EXISTS organizations (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    short_name VARCHAR(50),
    address TEXT,
    city VARCHAR(100),
    state VARCHAR(100),
    country VARCHAR(100) DEFAULT 'USA',
    postal_code VARCHAR(20),
    phone VARCHAR(50),
    email VARCHAR(255),
    website VARCHAR(255),
    tax_id VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255),
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    role user_role DEFAULT 'readonly',
    organization_id INTEGER REFERENCES organizations(id) ON DELETE SET NULL,
    phone VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    last_login TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Equipment/Instruments table
CREATE TABLE IF NOT EXISTS equipment (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    equipment_type VARCHAR(100) NOT NULL,  -- 'simulator', 'reference_cell', 'temperature_sensor', etc.
    manufacturer VARCHAR(255),
    model VARCHAR(255),
    serial_number VARCHAR(100),
    asset_tag VARCHAR(50),
    location VARCHAR(255),
    calibration_status calibration_status DEFAULT 'valid',
    last_calibration_date DATE,
    next_calibration_date DATE,
    calibration_interval_days INTEGER DEFAULT 365,
    calibration_lab VARCHAR(255),
    calibration_certificate_number VARCHAR(100),
    uncertainty_value DECIMAL(10, 6),
    uncertainty_unit VARCHAR(20),
    specifications JSONB,  -- Store equipment specs as JSON
    is_active BOOLEAN DEFAULT TRUE,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Sample Receipts table (for tracking incoming sample shipments)
CREATE TABLE IF NOT EXISTS sample_receipts (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    receipt_number VARCHAR(50) UNIQUE NOT NULL,
    organization_id INTEGER REFERENCES organizations(id) ON DELETE SET NULL,
    received_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    receipt_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    carrier VARCHAR(100),
    tracking_number VARCHAR(100),
    expected_sample_count INTEGER DEFAULT 0,
    actual_sample_count INTEGER DEFAULT 0,
    condition_on_arrival TEXT,
    packaging_intact BOOLEAN DEFAULT TRUE,
    temperature_controlled BOOLEAN DEFAULT FALSE,
    arrival_temperature DECIMAL(5, 2),
    photos_taken BOOLEAN DEFAULT FALSE,
    photo_paths TEXT[],
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Service Requests table
CREATE TABLE IF NOT EXISTS service_requests (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    request_number VARCHAR(50) UNIQUE NOT NULL,
    organization_id INTEGER REFERENCES organizations(id) ON DELETE SET NULL,
    contact_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    assigned_to INTEGER REFERENCES users(id) ON DELETE SET NULL,
    receipt_id INTEGER REFERENCES sample_receipts(id) ON DELETE SET NULL,
    status request_status DEFAULT 'draft',
    priority INTEGER DEFAULT 3,  -- 1=urgent, 2=high, 3=normal, 4=low
    title VARCHAR(255),
    description TEXT,
    test_requirements TEXT,
    expected_sample_quantity INTEGER DEFAULT 0,
    actual_sample_quantity INTEGER DEFAULT 0,
    quantity_verified BOOLEAN DEFAULT FALSE,
    requested_tests TEXT[],  -- Array of test types
    turnaround_days INTEGER DEFAULT 10,
    rush_order BOOLEAN DEFAULT FALSE,
    quote_number VARCHAR(50),
    quote_amount DECIMAL(12, 2),
    po_number VARCHAR(100),
    submitted_date TIMESTAMP WITH TIME ZONE,
    due_date DATE,
    completed_date TIMESTAMP WITH TIME ZONE,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Samples table (individual PV modules/cells)
CREATE TABLE IF NOT EXISTS samples (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    sample_id VARCHAR(100) UNIQUE NOT NULL,  -- Lab-assigned ID
    service_request_id INTEGER REFERENCES service_requests(id) ON DELETE CASCADE,
    receipt_id INTEGER REFERENCES sample_receipts(id) ON DELETE SET NULL,
    client_reference VARCHAR(255),  -- Client's own reference number
    status sample_status DEFAULT 'pending',

    -- Module/Cell Information
    technology VARCHAR(100),  -- PERC, TOPCon, HJT, etc.
    manufacturer VARCHAR(255),
    model VARCHAR(255),
    serial_number VARCHAR(100),
    cell_type VARCHAR(50),  -- mono, poly, thin-film
    cell_count INTEGER,

    -- Physical Specifications
    rated_power DECIMAL(10, 2),
    rated_power_unit VARCHAR(10) DEFAULT 'W',
    voc DECIMAL(10, 4),
    isc DECIMAL(10, 4),
    vmp DECIMAL(10, 4),
    imp DECIMAL(10, 4),
    fill_factor DECIMAL(6, 4),
    efficiency DECIMAL(6, 4),
    area DECIMAL(10, 4),
    area_unit VARCHAR(10) DEFAULT 'm2',

    -- Temperature Coefficients
    temp_coeff_pmax DECIMAL(8, 5),  -- %/°C
    temp_coeff_voc DECIMAL(8, 5),
    temp_coeff_isc DECIMAL(8, 5),

    -- Condition
    condition_on_receipt TEXT,
    visual_defects TEXT,
    photos TEXT[],

    -- Tracking
    location VARCHAR(100),
    received_date TIMESTAMP WITH TIME ZONE,
    tested_date TIMESTAMP WITH TIME ZONE,
    returned_date TIMESTAMP WITH TIME ZONE,
    disposal_date TIMESTAMP WITH TIME ZONE,

    notes TEXT,
    metadata JSONB,  -- Additional flexible data
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Measurements table
CREATE TABLE IF NOT EXISTS measurements (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    sample_id INTEGER REFERENCES samples(id) ON DELETE CASCADE,
    measurement_type measurement_type NOT NULL,
    performed_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    equipment_id INTEGER REFERENCES equipment(id) ON DELETE SET NULL,

    -- Test Conditions
    measurement_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    irradiance DECIMAL(10, 4),
    irradiance_unit VARCHAR(20) DEFAULT 'W/m2',
    temperature DECIMAL(8, 4),
    temperature_unit VARCHAR(10) DEFAULT 'C',
    spectral_class VARCHAR(10),  -- A+, A, B, C

    -- IV Curve Results (at test conditions)
    measured_pmax DECIMAL(12, 6),
    measured_voc DECIMAL(10, 6),
    measured_isc DECIMAL(10, 6),
    measured_vmp DECIMAL(10, 6),
    measured_imp DECIMAL(10, 6),
    measured_ff DECIMAL(8, 6),

    -- STC Corrected Results
    pmax_stc DECIMAL(12, 6),
    voc_stc DECIMAL(10, 6),
    isc_stc DECIMAL(10, 6),
    vmp_stc DECIMAL(10, 6),
    imp_stc DECIMAL(10, 6),

    -- Uncertainty Results
    combined_uncertainty DECIMAL(10, 6),
    expanded_uncertainty DECIMAL(10, 6),
    coverage_factor DECIMAL(4, 2) DEFAULT 2.0,
    relative_uncertainty_percent DECIMAL(8, 4),

    -- Uncertainty Budget (stored as JSON)
    uncertainty_budget JSONB,

    -- IV Curve Data
    iv_curve_data JSONB,  -- Voltage, Current arrays

    -- Quality Flags
    is_valid BOOLEAN DEFAULT TRUE,
    validation_notes TEXT,
    reviewed_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    reviewed_date TIMESTAMP WITH TIME ZONE,

    notes TEXT,
    raw_data_path VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Reports table
CREATE TABLE IF NOT EXISTS reports (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    report_number VARCHAR(50) UNIQUE NOT NULL,
    service_request_id INTEGER REFERENCES service_requests(id) ON DELETE SET NULL,
    sample_id INTEGER REFERENCES samples(id) ON DELETE SET NULL,
    status report_status DEFAULT 'draft',

    -- Report Details
    report_type VARCHAR(50),  -- 'calibration', 'test', 'uncertainty'
    title VARCHAR(255),
    version INTEGER DEFAULT 1,

    -- Personnel
    prepared_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    reviewed_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    approved_by INTEGER REFERENCES users(id) ON DELETE SET NULL,

    -- Dates
    prepared_date TIMESTAMP WITH TIME ZONE,
    reviewed_date TIMESTAMP WITH TIME ZONE,
    approved_date TIMESTAMP WITH TIME ZONE,
    issued_date TIMESTAMP WITH TIME ZONE,

    -- Content
    summary TEXT,
    conclusions TEXT,
    report_data JSONB,  -- Full report data

    -- File References
    pdf_path VARCHAR(500),
    excel_path VARCHAR(500),

    -- Supersession
    supersedes_report_id INTEGER REFERENCES reports(id) ON DELETE SET NULL,
    superseded_by_report_id INTEGER REFERENCES reports(id) ON DELETE SET NULL,

    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Audit Log table
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(50) NOT NULL,  -- 'create', 'update', 'delete', 'login', etc.
    table_name VARCHAR(100),
    record_id INTEGER,
    record_uuid UUID,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Migration tracking table
CREATE TABLE IF NOT EXISTS schema_migrations (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(255),
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    checksum VARCHAR(64)
);

-- ============================================
-- INDEXES
-- ============================================

-- Organizations
CREATE INDEX IF NOT EXISTS idx_organizations_name ON organizations(name);
CREATE INDEX IF NOT EXISTS idx_organizations_active ON organizations(is_active);

-- Users
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_organization ON users(organization_id);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);

-- Equipment
CREATE INDEX IF NOT EXISTS idx_equipment_type ON equipment(equipment_type);
CREATE INDEX IF NOT EXISTS idx_equipment_calibration_status ON equipment(calibration_status);
CREATE INDEX IF NOT EXISTS idx_equipment_next_calibration ON equipment(next_calibration_date);

-- Sample Receipts
CREATE INDEX IF NOT EXISTS idx_sample_receipts_number ON sample_receipts(receipt_number);
CREATE INDEX IF NOT EXISTS idx_sample_receipts_organization ON sample_receipts(organization_id);
CREATE INDEX IF NOT EXISTS idx_sample_receipts_date ON sample_receipts(receipt_date);

-- Service Requests
CREATE INDEX IF NOT EXISTS idx_service_requests_number ON service_requests(request_number);
CREATE INDEX IF NOT EXISTS idx_service_requests_organization ON service_requests(organization_id);
CREATE INDEX IF NOT EXISTS idx_service_requests_status ON service_requests(status);
CREATE INDEX IF NOT EXISTS idx_service_requests_receipt ON service_requests(receipt_id);
CREATE INDEX IF NOT EXISTS idx_service_requests_assigned ON service_requests(assigned_to);

-- Samples
CREATE INDEX IF NOT EXISTS idx_samples_sample_id ON samples(sample_id);
CREATE INDEX IF NOT EXISTS idx_samples_service_request ON samples(service_request_id);
CREATE INDEX IF NOT EXISTS idx_samples_status ON samples(status);
CREATE INDEX IF NOT EXISTS idx_samples_technology ON samples(technology);

-- Measurements
CREATE INDEX IF NOT EXISTS idx_measurements_sample ON measurements(sample_id);
CREATE INDEX IF NOT EXISTS idx_measurements_type ON measurements(measurement_type);
CREATE INDEX IF NOT EXISTS idx_measurements_date ON measurements(measurement_date);
CREATE INDEX IF NOT EXISTS idx_measurements_equipment ON measurements(equipment_id);

-- Reports
CREATE INDEX IF NOT EXISTS idx_reports_number ON reports(report_number);
CREATE INDEX IF NOT EXISTS idx_reports_service_request ON reports(service_request_id);
CREATE INDEX IF NOT EXISTS idx_reports_sample ON reports(sample_id);
CREATE INDEX IF NOT EXISTS idx_reports_status ON reports(status);

-- Audit Log
CREATE INDEX IF NOT EXISTS idx_audit_log_user ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_table ON audit_log(table_name);
CREATE INDEX IF NOT EXISTS idx_audit_log_created ON audit_log(created_at);

-- ============================================
-- FUNCTIONS AND TRIGGERS
-- ============================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at trigger to all tables
DO $$
DECLARE
    t text;
BEGIN
    FOR t IN
        SELECT table_name
        FROM information_schema.columns
        WHERE column_name = 'updated_at'
        AND table_schema = 'public'
    LOOP
        EXECUTE format('
            DROP TRIGGER IF EXISTS update_%I_updated_at ON %I;
            CREATE TRIGGER update_%I_updated_at
            BEFORE UPDATE ON %I
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        ', t, t, t, t);
    END LOOP;
END $$;

-- Function to generate request numbers
CREATE OR REPLACE FUNCTION generate_request_number()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.request_number IS NULL THEN
        NEW.request_number := 'SR-' || TO_CHAR(CURRENT_DATE, 'YYYYMM') || '-' ||
                              LPAD(nextval('service_requests_id_seq')::TEXT, 5, '0');
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_request_number
BEFORE INSERT ON service_requests
FOR EACH ROW
EXECUTE FUNCTION generate_request_number();

-- Function to generate report numbers
CREATE OR REPLACE FUNCTION generate_report_number()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.report_number IS NULL THEN
        NEW.report_number := 'RPT-' || TO_CHAR(CURRENT_DATE, 'YYYYMM') || '-' ||
                             LPAD(nextval('reports_id_seq')::TEXT, 5, '0');
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_report_number
BEFORE INSERT ON reports
FOR EACH ROW
EXECUTE FUNCTION generate_report_number();

-- Record initial migration
INSERT INTO schema_migrations (version, name, checksum)
VALUES ('001', 'initial_schema', md5('001_initial_schema.sql'))
ON CONFLICT (version) DO NOTHING;


-- ============================================
-- DOWN MIGRATION
-- ============================================
-- To rollback this migration, run the following:

/*
-- DROP TRIGGERS
DROP TRIGGER IF EXISTS set_report_number ON reports;
DROP TRIGGER IF EXISTS set_request_number ON service_requests;

-- DROP FUNCTIONS
DROP FUNCTION IF EXISTS generate_report_number();
DROP FUNCTION IF EXISTS generate_request_number();
DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE;

-- DROP TABLES (in reverse dependency order)
DROP TABLE IF EXISTS audit_log CASCADE;
DROP TABLE IF EXISTS reports CASCADE;
DROP TABLE IF EXISTS measurements CASCADE;
DROP TABLE IF EXISTS samples CASCADE;
DROP TABLE IF EXISTS service_requests CASCADE;
DROP TABLE IF EXISTS sample_receipts CASCADE;
DROP TABLE IF EXISTS equipment CASCADE;
DROP TABLE IF EXISTS users CASCADE;
DROP TABLE IF EXISTS organizations CASCADE;
DROP TABLE IF EXISTS schema_migrations CASCADE;

-- DROP TYPES
DROP TYPE IF EXISTS report_status;
DROP TYPE IF EXISTS calibration_status;
DROP TYPE IF EXISTS measurement_type;
DROP TYPE IF EXISTS sample_status;
DROP TYPE IF EXISTS request_status;
DROP TYPE IF EXISTS user_role;

-- DROP EXTENSION
DROP EXTENSION IF EXISTS "uuid-ossp";
*/
