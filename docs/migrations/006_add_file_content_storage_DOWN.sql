-- Migration 006 DOWN: Remove file content storage
ALTER TABLE files DROP COLUMN IF EXISTS content;
SELECT 'Migration 006 DOWN: File content storage removed' AS status;
