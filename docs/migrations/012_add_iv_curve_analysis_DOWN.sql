-- Migration 012 DOWN: Remove IV curve analysis fields
ALTER TABLE iv_curve_data DROP COLUMN IF EXISTS series_resistance_ohm;
ALTER TABLE iv_curve_data DROP COLUMN IF EXISTS shunt_resistance_ohm;
SELECT 'Migration 012 DOWN: IV curve analysis fields reverted' AS status;
