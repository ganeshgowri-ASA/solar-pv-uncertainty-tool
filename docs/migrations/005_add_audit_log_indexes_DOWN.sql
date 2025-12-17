-- Migration 005 DOWN: Remove audit log indexes
DROP INDEX IF EXISTS idx_audit_logs_org_time;
DROP INDEX IF EXISTS idx_audit_logs_entity_full;
SELECT 'Migration 005 DOWN: Audit log indexes removed' AS status;
