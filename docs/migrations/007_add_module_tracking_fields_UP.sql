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
