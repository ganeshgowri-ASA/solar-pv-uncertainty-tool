-- Migration 009: Sun Simulator Maintenance Tracking

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'sun_simulators' AND column_name = 'last_maintenance_date'
    ) THEN
        ALTER TABLE sun_simulators ADD COLUMN last_maintenance_date DATE;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'sun_simulators' AND column_name = 'lamp_hours'
    ) THEN
        ALTER TABLE sun_simulators ADD COLUMN lamp_hours INTEGER DEFAULT 0;
    END IF;
END $$;

SELECT 'Migration 009 UP: Sun simulator maintenance tracking added' AS status;
