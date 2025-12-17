-- Migration 004: Performance Indexes for Measurements

CREATE INDEX IF NOT EXISTS idx_measurements_created_at ON measurements(created_at);
CREATE INDEX IF NOT EXISTS idx_measurements_module_date ON measurements(module_id, test_date);
CREATE INDEX IF NOT EXISTS idx_uncertainty_results_param ON uncertainty_results(target_parameter);

SELECT 'Migration 004 UP: Measurement indexes created' AS status;
