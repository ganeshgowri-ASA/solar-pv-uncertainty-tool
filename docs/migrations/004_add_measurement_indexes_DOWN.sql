-- Migration 004 DOWN: Remove measurement indexes
DROP INDEX IF EXISTS idx_measurements_created_at;
DROP INDEX IF EXISTS idx_measurements_module_date;
DROP INDEX IF EXISTS idx_uncertainty_results_param;
SELECT 'Migration 004 DOWN: Measurement indexes removed' AS status;
