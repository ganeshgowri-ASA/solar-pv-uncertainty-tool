-- Migration 006: File Content Storage

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'files' AND column_name = 'content'
    ) THEN
        ALTER TABLE files ADD COLUMN content BYTEA;
    END IF;
END $$;

SELECT 'Migration 006 UP: File content storage added' AS status;
