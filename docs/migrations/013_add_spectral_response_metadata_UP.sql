-- Migration 013: Spectral Response Metadata

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'spectral_responses' AND column_name = 'measurement_method'
    ) THEN
        ALTER TABLE spectral_responses ADD COLUMN measurement_method VARCHAR(100);
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'spectral_responses' AND column_name = 'temperature_c'
    ) THEN
        ALTER TABLE spectral_responses ADD COLUMN temperature_c FLOAT DEFAULT 25.0;
    END IF;
END $$;

SELECT 'Migration 013 UP: Spectral response metadata added' AS status;
