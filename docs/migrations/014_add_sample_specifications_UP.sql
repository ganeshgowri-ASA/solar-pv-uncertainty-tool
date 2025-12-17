-- Migration 014: Sample Specifications Column

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'samples' AND column_name = 'specifications'
    ) THEN
        ALTER TABLE samples ADD COLUMN specifications JSONB DEFAULT '{}';
    END IF;
END $$;

-- Create index for JSONB queries if not exists
CREATE INDEX IF NOT EXISTS idx_samples_specifications ON samples USING GIN (specifications);

SELECT 'Migration 014 UP: Sample specifications column added' AS status;
