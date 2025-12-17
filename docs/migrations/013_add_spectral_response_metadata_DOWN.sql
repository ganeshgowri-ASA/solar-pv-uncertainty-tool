-- Migration 013 DOWN: Remove spectral response metadata
ALTER TABLE spectral_responses DROP COLUMN IF EXISTS measurement_method;
ALTER TABLE spectral_responses DROP COLUMN IF EXISTS temperature_c;
SELECT 'Migration 013 DOWN: Spectral response metadata reverted' AS status;
