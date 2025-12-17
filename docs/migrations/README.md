# Database Migrations

This directory contains SQL migration files for the PV Uncertainty Tool database.

## Naming Convention

```
XXX_<name>_UP.sql   - Forward migration (applies changes)
XXX_<name>_DOWN.sql - Rollback migration (reverts changes)
```

Where `XXX` is a 3-digit migration number (001, 002, etc.)

## Available Migrations

| # | Name | Description |
|---|------|-------------|
| 001 | initial_schema | Base schema created by SQLAlchemy ORM |
| 002 | add_sample_tracking | Links samples to receipts and inspections |
| 003 | add_inspection_allocation | Allocation tracking for incoming inspections |
| 004 | add_measurement_indexes | Performance indexes for measurement queries |
| 005 | add_audit_log_indexes | Performance indexes for audit log queries |
| 006 | add_file_content_storage | Binary content storage for files |
| 007 | add_module_tracking_fields | Production date and batch tracking |
| 008 | add_reference_device_tracking | Enhanced calibration tracking |
| 009 | add_sun_simulator_maintenance | Maintenance and lamp hour tracking |
| 010 | add_measurement_conditions | Environmental conditions (pressure, wind) |
| 011 | add_uncertainty_component_types | Type A/B classification for components |
| 012 | add_iv_curve_analysis | Series/shunt resistance analysis |
| 013 | add_spectral_response_metadata | Measurement method and temperature |
| 014 | add_sample_specifications | JSONB specifications column |

## Running Migrations

Migrations can be run through the Admin Seed dashboard:

1. Navigate to the **Admin Seed** page (99_Admin_Seed.py)
2. Go to the **Migrations** tab
3. Click **Sync Migration Files** to ensure all files exist
4. Apply migrations by clicking **Apply** next to pending migrations

## Design Principles

1. **Idempotent**: All migrations use `IF NOT EXISTS` checks
2. **Reversible**: Each UP migration has a corresponding DOWN
3. **Safe**: Pre-flight checks prevent partial application
4. **Traceable**: Status messages for audit trail

## Manual Application

To manually apply a migration:

```sql
-- Run the UP migration
\i docs/migrations/014_add_sample_specifications_UP.sql
```

To revert:

```sql
-- Run the DOWN migration
\i docs/migrations/014_add_sample_specifications_DOWN.sql
```
