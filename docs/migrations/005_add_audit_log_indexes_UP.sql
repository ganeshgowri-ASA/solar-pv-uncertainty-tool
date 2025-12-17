-- Migration 005: Audit Log Performance Indexes

CREATE INDEX IF NOT EXISTS idx_audit_logs_org_time ON audit_logs(organization_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_entity_full ON audit_logs(entity_type, entity_id, timestamp);

SELECT 'Migration 005 UP: Audit log indexes created' AS status;
