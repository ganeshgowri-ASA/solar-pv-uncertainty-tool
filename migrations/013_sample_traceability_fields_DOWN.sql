-- Migration 013 DOWN: Remove Sample Traceability Fields
-- Removes the samples table and related objects

-- Drop indexes first
DROP INDEX IF EXISTS idx_samples_receipt_id;
DROP INDEX IF EXISTS idx_samples_internal_id;
DROP INDEX IF EXISTS idx_samples_status;
DROP INDEX IF EXISTS idx_samples_receipt_date;
DROP INDEX IF EXISTS idx_samples_org;
DROP INDEX IF EXISTS idx_samples_module;

-- Drop the samples table
DROP TABLE IF EXISTS samples CASCADE;

-- Drop the enum type
DROP TYPE IF EXISTS samplestatus;

SELECT 'Migration 013 DOWN completed - samples table removed' AS status;
