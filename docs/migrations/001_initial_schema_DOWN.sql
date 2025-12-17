-- Migration 001 DOWN: Full schema reset
-- WARNING: This drops all tables!
DROP SCHEMA public CASCADE;
CREATE SCHEMA public;
GRANT ALL ON SCHEMA public TO postgres;
GRANT ALL ON SCHEMA public TO public;
SELECT 'Migration 001 DOWN: Schema reset complete' AS status;
