-- Migration 009 DOWN: Remove sun simulator maintenance tracking
ALTER TABLE sun_simulators DROP COLUMN IF EXISTS last_maintenance_date;
ALTER TABLE sun_simulators DROP COLUMN IF EXISTS lamp_hours;
SELECT 'Migration 009 DOWN: Sun simulator maintenance reverted' AS status;
