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
