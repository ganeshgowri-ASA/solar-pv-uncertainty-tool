-- Migration 014 DOWN: Remove sample specifications
DROP INDEX IF EXISTS idx_samples_specifications;
ALTER TABLE samples DROP COLUMN IF EXISTS specifications;
SELECT 'Migration 014 DOWN: Sample specifications reverted' AS status;
