-- Migration 003 DOWN: Remove inspection allocation fields
ALTER TABLE incoming_inspections DROP COLUMN IF EXISTS allocation_triggered;
ALTER TABLE incoming_inspections DROP COLUMN IF EXISTS allocated_sample_id;
SELECT 'Migration 003 DOWN: Inspection allocation fields removed' AS status;
