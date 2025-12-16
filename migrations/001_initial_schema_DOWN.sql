-- Migration 001: Initial Schema - ROLLBACK
-- Use this to undo migration 001

-- Example: Remove custom indexes
-- DROP INDEX IF EXISTS idx_measurements_date;

-- Example: Remove custom constraints
-- ALTER TABLE measurements DROP CONSTRAINT IF EXISTS chk_positive_pmax;

-- Warning: This does NOT drop the base tables
-- To fully reset, use: Base.metadata.drop_all()

SELECT 'Migration 001 DOWN completed' AS status;
