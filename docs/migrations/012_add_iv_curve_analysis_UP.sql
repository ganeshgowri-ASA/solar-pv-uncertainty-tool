-- Migration 012: IV Curve Analysis Fields

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'iv_curve_data' AND column_name = 'series_resistance_ohm'
    ) THEN
        ALTER TABLE iv_curve_data ADD COLUMN series_resistance_ohm FLOAT;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'iv_curve_data' AND column_name = 'shunt_resistance_ohm'
    ) THEN
        ALTER TABLE iv_curve_data ADD COLUMN shunt_resistance_ohm FLOAT;
    END IF;
END $$;

SELECT 'Migration 012 UP: IV curve analysis fields added' AS status;
