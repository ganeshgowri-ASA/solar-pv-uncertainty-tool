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
