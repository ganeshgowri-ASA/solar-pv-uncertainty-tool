-- Migration 013: Add Sample Traceability Fields
-- Creates the samples table for lab sample tracking and chain of custody

-- Create enum type for sample status
DO $$ BEGIN
    CREATE TYPE samplestatus AS ENUM (
        'received',
        'in_storage',
        'in_testing',
        'testing_complete',
        'returned',
        'disposed'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Create samples table
CREATE TABLE IF NOT EXISTS samples (
    id SERIAL PRIMARY KEY,
    organization_id INTEGER REFERENCES organizations(id),

    -- Sample identification - receipt_id is the primary tracking number
    receipt_id VARCHAR(100) UNIQUE NOT NULL,
    internal_id VARCHAR(100),
    client_reference VARCHAR(255),

    -- Link to module under test
    module_id INTEGER REFERENCES modules(id),

    -- Receipt information
    receipt_date TIMESTAMP NOT NULL,
    received_by_id INTEGER REFERENCES users(id),
    client_name VARCHAR(255),
    client_contact VARCHAR(255),

    -- Sample condition on receipt
    condition_on_receipt TEXT,
    packaging_intact BOOLEAN DEFAULT TRUE,
    visual_damage_noted BOOLEAN DEFAULT FALSE,
    damage_description TEXT,

    -- Storage information
    storage_location VARCHAR(255),
    storage_conditions VARCHAR(255),

    -- Status tracking
    status samplestatus DEFAULT 'received',

    -- Chain of custody (JSON array)
    custody_log JSONB,

    -- Testing allocation
    allocated_tests JSONB,
    priority VARCHAR(20) DEFAULT 'normal',

    -- Return/disposition
    return_date TIMESTAMP,
    returned_to VARCHAR(255),
    return_tracking VARCHAR(255),
    disposition_notes TEXT,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_samples_receipt_id ON samples(receipt_id);
CREATE INDEX IF NOT EXISTS idx_samples_internal_id ON samples(internal_id);
CREATE INDEX IF NOT EXISTS idx_samples_status ON samples(status);
CREATE INDEX IF NOT EXISTS idx_samples_receipt_date ON samples(receipt_date);
CREATE INDEX IF NOT EXISTS idx_samples_org ON samples(organization_id);
CREATE INDEX IF NOT EXISTS idx_samples_module ON samples(module_id);

-- Add comment for documentation
COMMENT ON TABLE samples IS 'Sample tracking for lab traceability and chain of custody per ISO 17025';
COMMENT ON COLUMN samples.receipt_id IS 'Primary tracking number assigned on sample receipt';
COMMENT ON COLUMN samples.custody_log IS 'JSON array of custody transfers: [{date, from_user, to_user, reason}]';
COMMENT ON COLUMN samples.allocated_tests IS 'JSON array of test types allocated to this sample';

SELECT 'Migration 013 UP completed - samples table created' AS status;
