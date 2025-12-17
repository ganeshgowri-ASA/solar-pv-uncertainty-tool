-- Migration 011 DOWN: Remove uncertainty component types
ALTER TABLE uncertainty_components DROP COLUMN IF EXISTS component_type;
ALTER TABLE uncertainty_components DROP COLUMN IF EXISTS degrees_of_freedom;
SELECT 'Migration 011 DOWN: Uncertainty component types reverted' AS status;
