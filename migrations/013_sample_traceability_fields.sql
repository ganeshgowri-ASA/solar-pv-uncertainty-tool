-- Migration: 013_sample_traceability_fields.sql
-- Description: Add sample traceability fields and incoming_inspections table
-- Date: 2025-12-16
--
-- Purpose: Complete the sample chain of custody and traceability system
--
-- Relationships established:
--   sample_receipts -> incoming_inspections (via receipt_id)
--   sample_receipts -> samples (via receipt_id) [already exists]
--   incoming_inspections -> samples (via inspection_id)
--   samples -> service_requests (via service_request_id) [already exists]

-- ============================================
-- UP MIGRATION
-- ============================================

-- Create inspection_result ENUM type
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'inspection_result') THEN
        CREATE TYPE inspection_result AS ENUM (
            'pass',
            'pass_with_observations',
            'fail',
            'pending_review'
        );
    END IF;
END$$;

-- Create inspection_type ENUM type
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'inspection_type') THEN
        CREATE TYPE inspection_type AS ENUM (
            'incoming',
            'pre_test',
            'post_test',
            'final',
            'damage_assessment'
        );
    END IF;
END$$;

-- ============================================
-- TABLE: incoming_inspections
-- ============================================
-- Tracks visual and physical inspections performed on received samples
-- Links to sample_receipts for complete chain of custody

CREATE TABLE IF NOT EXISTS incoming_inspections (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    inspection_number VARCHAR(50) UNIQUE NOT NULL,

    -- Foreign Keys
    receipt_id INTEGER REFERENCES sample_receipts(id) ON DELETE SET NULL,
    sample_id INTEGER REFERENCES samples(id) ON DELETE CASCADE,
    inspected_by INTEGER REFERENCES users(id) ON DELETE SET NULL,

    -- Inspection Details
    inspection_type inspection_type DEFAULT 'incoming',
    inspection_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    result inspection_result DEFAULT 'pending_review',

    -- Physical Inspection
    packaging_condition TEXT,
    packaging_photos TEXT[],
    label_legible BOOLEAN DEFAULT TRUE,
    label_matches_docs BOOLEAN DEFAULT TRUE,
    serial_number_verified BOOLEAN DEFAULT FALSE,
    serial_number_found VARCHAR(100),

    -- Visual Inspection (Module)
    frame_condition TEXT,
    glass_condition TEXT,
    backsheet_condition TEXT,
    junction_box_condition TEXT,
    connector_condition TEXT,
    cell_visible_defects TEXT,

    -- Defect Tracking
    defects_found BOOLEAN DEFAULT FALSE,
    defect_count INTEGER DEFAULT 0,
    defect_types TEXT[],  -- Array of defect categories
    defect_locations TEXT[],  -- Array of defect locations
    defect_photos TEXT[],
    defect_severity VARCHAR(20),  -- 'minor', 'major', 'critical'

    -- Measurements at Inspection
    measured_dimensions JSONB,  -- {length, width, thickness}
    measured_weight DECIMAL(10, 3),
    weight_unit VARCHAR(10) DEFAULT 'kg',

    -- Environmental Conditions
    inspection_temperature DECIMAL(5, 2),
    inspection_humidity DECIMAL(5, 2),

    -- Documentation
    photos TEXT[],
    documents TEXT[],
    checklist_completed BOOLEAN DEFAULT FALSE,
    checklist_data JSONB,

    -- Review & Approval
    reviewed_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    reviewed_date TIMESTAMP WITH TIME ZONE,
    review_notes TEXT,

    -- Disposition
    disposition VARCHAR(50),  -- 'proceed_testing', 'hold', 'reject', 'return'
    disposition_reason TEXT,
    disposition_date TIMESTAMP WITH TIME ZONE,

    notes TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- ADD MISSING COLUMNS TO samples TABLE
-- ============================================

-- Add custody_history column for chain of custody tracking
ALTER TABLE samples
ADD COLUMN IF NOT EXISTS custody_history JSONB DEFAULT '[]'::jsonb;

-- Add inspection_id foreign key to link samples to their incoming inspection
ALTER TABLE samples
ADD COLUMN IF NOT EXISTS inspection_id INTEGER REFERENCES incoming_inspections(id) ON DELETE SET NULL;

-- Add specifications column for detailed sample specs
ALTER TABLE samples
ADD COLUMN IF NOT EXISTS specifications JSONB;

-- Add project_id if not exists (for project-based sample tracking)
ALTER TABLE samples
ADD COLUMN IF NOT EXISTS project_id INTEGER;

-- ============================================
-- INDEXES
-- ============================================

-- incoming_inspections indexes
CREATE INDEX IF NOT EXISTS idx_incoming_inspections_receipt
ON incoming_inspections(receipt_id);

CREATE INDEX IF NOT EXISTS idx_incoming_inspections_sample
ON incoming_inspections(sample_id);

CREATE INDEX IF NOT EXISTS idx_incoming_inspections_date
ON incoming_inspections(inspection_date);

CREATE INDEX IF NOT EXISTS idx_incoming_inspections_result
ON incoming_inspections(result);

CREATE INDEX IF NOT EXISTS idx_incoming_inspections_number
ON incoming_inspections(inspection_number);

-- samples additional indexes
CREATE INDEX IF NOT EXISTS idx_samples_inspection
ON samples(inspection_id);

CREATE INDEX IF NOT EXISTS idx_samples_project
ON samples(project_id);

-- GIN index for JSONB columns (for efficient JSON queries)
CREATE INDEX IF NOT EXISTS idx_samples_custody_history_gin
ON samples USING GIN (custody_history);

CREATE INDEX IF NOT EXISTS idx_samples_specifications_gin
ON samples USING GIN (specifications);

CREATE INDEX IF NOT EXISTS idx_incoming_inspections_checklist_gin
ON incoming_inspections USING GIN (checklist_data);

-- ============================================
-- TRIGGERS
-- ============================================

-- Auto-generate inspection number
CREATE OR REPLACE FUNCTION generate_inspection_number()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.inspection_number IS NULL THEN
        NEW.inspection_number := 'INS-' || TO_CHAR(CURRENT_DATE, 'YYYYMM') || '-' ||
                                 LPAD(nextval('incoming_inspections_id_seq')::TEXT, 5, '0');
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS set_inspection_number ON incoming_inspections;
CREATE TRIGGER set_inspection_number
BEFORE INSERT ON incoming_inspections
FOR EACH ROW
EXECUTE FUNCTION generate_inspection_number();

-- Apply updated_at trigger to incoming_inspections
DROP TRIGGER IF EXISTS update_incoming_inspections_updated_at ON incoming_inspections;
CREATE TRIGGER update_incoming_inspections_updated_at
BEFORE UPDATE ON incoming_inspections
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- COMMENTS
-- ============================================

COMMENT ON TABLE incoming_inspections IS 'Tracks incoming inspection records for received samples';
COMMENT ON COLUMN incoming_inspections.receipt_id IS 'Links inspection to the sample receipt for traceability';
COMMENT ON COLUMN incoming_inspections.custody_history IS 'JSON array tracking chain of custody events';

COMMENT ON COLUMN samples.custody_history IS 'JSON array of custody events: [{timestamp, action, user, location, notes}]';
COMMENT ON COLUMN samples.inspection_id IS 'Reference to the incoming inspection record';
COMMENT ON COLUMN samples.specifications IS 'Detailed specifications as JSON: {electrical, mechanical, thermal, etc.}';
COMMENT ON COLUMN samples.project_id IS 'Optional project association for batch testing';

-- ============================================
-- RECORD MIGRATION
-- ============================================

INSERT INTO schema_migrations (version, name, checksum)
VALUES ('013', 'sample_traceability_fields', md5('013_sample_traceability_fields.sql'))
ON CONFLICT (version) DO UPDATE SET
    name = EXCLUDED.name,
    checksum = EXCLUDED.checksum,
    applied_at = CURRENT_TIMESTAMP;


-- ============================================
-- DOWN MIGRATION
-- ============================================
-- To rollback this migration, run the following:

/*
-- Remove indexes
DROP INDEX IF EXISTS idx_incoming_inspections_checklist_gin;
DROP INDEX IF EXISTS idx_samples_specifications_gin;
DROP INDEX IF EXISTS idx_samples_custody_history_gin;
DROP INDEX IF EXISTS idx_samples_project;
DROP INDEX IF EXISTS idx_samples_inspection;
DROP INDEX IF EXISTS idx_incoming_inspections_number;
DROP INDEX IF EXISTS idx_incoming_inspections_result;
DROP INDEX IF EXISTS idx_incoming_inspections_date;
DROP INDEX IF EXISTS idx_incoming_inspections_sample;
DROP INDEX IF EXISTS idx_incoming_inspections_receipt;

-- Remove triggers
DROP TRIGGER IF EXISTS update_incoming_inspections_updated_at ON incoming_inspections;
DROP TRIGGER IF EXISTS set_inspection_number ON incoming_inspections;

-- Remove function
DROP FUNCTION IF EXISTS generate_inspection_number();

-- Remove columns from samples
ALTER TABLE samples DROP COLUMN IF EXISTS project_id;
ALTER TABLE samples DROP COLUMN IF EXISTS specifications;
ALTER TABLE samples DROP COLUMN IF EXISTS inspection_id;
ALTER TABLE samples DROP COLUMN IF EXISTS custody_history;

-- Drop incoming_inspections table
DROP TABLE IF EXISTS incoming_inspections CASCADE;

-- Drop enum types
DROP TYPE IF EXISTS inspection_type;
DROP TYPE IF EXISTS inspection_result;

-- Remove migration record
DELETE FROM schema_migrations WHERE version = '013';
*/
