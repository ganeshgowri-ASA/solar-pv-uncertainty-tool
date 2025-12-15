-- Migration 001: Initial Schema
-- This migration is managed by SQLAlchemy ORM
-- Only use this file for additional schema changes not handled by models.py

-- Example: Add custom indexes
-- CREATE INDEX IF NOT EXISTS idx_measurements_date ON measurements(created_at);

-- Example: Add custom constraints
-- ALTER TABLE measurements ADD CONSTRAINT chk_positive_pmax CHECK (pmax_measured_w > 0);

-- Note: Base tables are created automatically by SQLAlchemy Base.metadata.create_all()
-- This file is for:
--   1. Custom indexes not defined in models
--   2. Custom constraints
--   3. Data migrations
--   4. Schema alterations not suitable for ORM

SELECT 'Migration 001 UP completed' AS status;
