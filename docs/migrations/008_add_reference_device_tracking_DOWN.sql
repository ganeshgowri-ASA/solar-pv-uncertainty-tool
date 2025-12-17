-- Migration 008 DOWN: Remove enhanced reference device tracking
ALTER TABLE reference_devices DROP COLUMN IF EXISTS next_calibration_date;
ALTER TABLE reference_devices DROP COLUMN IF EXISTS calibration_interval_months;
SELECT 'Migration 008 DOWN: Reference device tracking reverted' AS status;
