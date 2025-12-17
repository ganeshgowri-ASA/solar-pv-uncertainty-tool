-- Migration 002 DOWN: Remove sample tracking fields
ALTER TABLE samples DROP COLUMN IF EXISTS receipt_id;
ALTER TABLE samples DROP COLUMN IF EXISTS inspection_id;
SELECT 'Migration 002 DOWN: Sample tracking fields removed' AS status;
