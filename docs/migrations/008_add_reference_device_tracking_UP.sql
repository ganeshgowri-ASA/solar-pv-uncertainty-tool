-- Migration 008: Reference Device Enhanced Tracking

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'reference_devices' AND column_name = 'next_calibration_date'
    ) THEN
        ALTER TABLE reference_devices ADD COLUMN next_calibration_date DATE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'reference_devices' AND column_name = 'calibration_interval_months'
    ) THEN
        ALTER TABLE reference_devices ADD COLUMN calibration_interval_months INTEGER DEFAULT 12;
    END IF;
END $$;

SELECT 'Migration 008 UP: Reference device tracking enhanced' AS status;
