-- Migration 011: Uncertainty Component Types

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'uncertainty_components' AND column_name = 'component_type'
    ) THEN
        ALTER TABLE uncertainty_components ADD COLUMN component_type VARCHAR(20) DEFAULT 'B';
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'uncertainty_components' AND column_name = 'degrees_of_freedom'
    ) THEN
        ALTER TABLE uncertainty_components ADD COLUMN degrees_of_freedom FLOAT;
    END IF;
END $$;

SELECT 'Migration 011 UP: Uncertainty component types added' AS status;
