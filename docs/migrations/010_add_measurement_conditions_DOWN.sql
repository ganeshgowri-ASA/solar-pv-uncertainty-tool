-- Migration 010 DOWN: Remove environmental conditions
ALTER TABLE measurements DROP COLUMN IF EXISTS barometric_pressure_hpa;
ALTER TABLE measurements DROP COLUMN IF EXISTS wind_speed_m_s;
SELECT 'Migration 010 DOWN: Measurement conditions reverted' AS status;
