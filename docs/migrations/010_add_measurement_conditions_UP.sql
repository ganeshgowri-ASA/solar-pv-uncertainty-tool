-- Migration 010: Measurement Environmental Conditions

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'measurements' AND column_name = 'barometric_pressure_hpa'
    ) THEN
        ALTER TABLE measurements ADD COLUMN barometric_pressure_hpa FLOAT;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'measurements' AND column_name = 'wind_speed_m_s'
    ) THEN
        ALTER TABLE measurements ADD COLUMN wind_speed_m_s FLOAT;
    END IF;
END $$;

SELECT 'Migration 010 UP: Measurement conditions tracking added' AS status;
