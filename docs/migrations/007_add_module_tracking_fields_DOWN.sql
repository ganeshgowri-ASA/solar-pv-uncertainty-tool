-- Migration 007 DOWN: Remove module tracking fields
ALTER TABLE modules DROP COLUMN IF EXISTS production_date;
ALTER TABLE modules DROP COLUMN IF EXISTS batch_number;
SELECT 'Migration 007 DOWN: Module tracking fields removed' AS status;
