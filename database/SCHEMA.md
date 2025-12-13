# Database Schema Documentation

## Overview

This document describes the PostgreSQL database schema for the PV Measurement Uncertainty Tool. The schema is designed to support:

- Multi-tenant organizations
- Role-based access control (RBAC)
- Complete measurement data archiving
- Uncertainty calculation results storage
- Reference device management with spectral response data
- File management with approval workflows
- Comprehensive audit logging for ISO 17025 compliance

## Entity Relationship Diagram

```
┌──────────────────┐       ┌──────────────────┐
│  organizations   │◄──────│      users       │
└──────────────────┘       └──────────────────┘
         │                          │
         │                          │
         ▼                          ▼
┌──────────────────┐       ┌──────────────────┐
│     modules      │       │   audit_logs     │
└──────────────────┘       └──────────────────┘
         │
         │
         ▼
┌──────────────────┐       ┌──────────────────┐
│   measurements   │◄──────│ reference_devices│
└──────────────────┘       └──────────────────┘
    │         │                    │
    │         │                    │
    ▼         ▼                    ▼
┌─────────┐ ┌─────────────────┐ ┌─────────────────┐
│iv_curves│ │uncertainty_     │ │spectral_        │
│         │ │results          │ │responses        │
└─────────┘ └─────────────────┘ └─────────────────┘
                    │
                    │
                    ▼
            ┌─────────────────┐
            │uncertainty_     │
            │components       │
            └─────────────────┘

┌──────────────────┐       ┌──────────────────┐
│  sun_simulators  │       │      files       │
└──────────────────┘       └──────────────────┘

┌──────────────────┐
│approval_workflows│
└──────────────────┘
```

## Tables

### Core Tables

#### `organizations`
Multi-tenant organization management for laboratories.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| name | VARCHAR(255) | Organization name |
| address | TEXT | Physical address |
| accreditation_number | VARCHAR(100) | ISO 17025 accreditation |
| accreditation_body | VARCHAR(100) | Accrediting body |
| logo_path | VARCHAR(500) | Logo file path |
| document_format_prefix | VARCHAR(50) | Document numbering prefix |
| is_active | BOOLEAN | Active status |

#### `users`
User authentication and authorization.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| organization_id | INTEGER | FK to organizations |
| email | VARCHAR(255) | Unique email (login) |
| password_hash | VARCHAR(255) | Bcrypt password hash |
| first_name | VARCHAR(100) | First name |
| last_name | VARCHAR(100) | Last name |
| title | VARCHAR(100) | Job title |
| role | ENUM | ADMIN, ENGINEER, REVIEWER, VIEWER |
| is_active | BOOLEAN | Account active |
| is_verified | BOOLEAN | Email verified |

### Module & Measurement Tables

#### `modules`
PV modules under test.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| serial_number | VARCHAR(100) | Module serial |
| model_name | VARCHAR(255) | Model designation |
| manufacturer | VARCHAR(255) | Manufacturer |
| technology | ENUM | PERC, TOPCon, HJT, etc. |
| cells_series | INTEGER | Cells in series |
| cells_parallel | INTEGER | Cells in parallel |
| module_area_m2 | FLOAT | Module area |
| pmax_nameplate_w | FLOAT | Nameplate power |
| gamma_pmax_pct_per_c | FLOAT | Power temp coefficient |
| is_bifacial | BOOLEAN | Bifacial module flag |
| bifaciality_factor | FLOAT | Bifaciality factor |

#### `measurements`
Individual measurement sessions.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| module_id | INTEGER | FK to modules |
| reference_device_id | INTEGER | FK to reference_devices |
| sun_simulator_id | INTEGER | FK to sun_simulators |
| measurement_number | VARCHAR(50) | Unique measurement ID |
| measurement_type | ENUM | STC, NMOT, LOW_IRRADIANCE, etc. |
| test_date | DATETIME | Date of measurement |
| voc_v | FLOAT | Open circuit voltage |
| isc_a | FLOAT | Short circuit current |
| vmp_v | FLOAT | Voltage at MPP |
| imp_a | FLOAT | Current at MPP |
| pmax_w | FLOAT | Maximum power |
| fill_factor | FLOAT | Fill factor |
| approval_status | ENUM | DRAFT, SUBMITTED, APPROVED |
| preparer_id | INTEGER | FK to users |
| reviewer_id | INTEGER | FK to users |
| approver_id | INTEGER | FK to users |

#### `iv_curve_data`
Raw I-V curve data points.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| measurement_id | INTEGER | FK to measurements |
| flash_number | INTEGER | Flash number in session |
| voltage_current_data | JSON | Array of {v, i} pairs |
| num_points | INTEGER | Number of data points |
| scan_direction | VARCHAR(20) | forward/reverse |

### Reference Device Tables

#### `reference_devices`
WPVS cells and reference modules.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| serial_number | VARCHAR(100) | Device serial |
| device_type | VARCHAR(50) | WPVS_CELL, REFERENCE_MODULE |
| calibration_lab | VARCHAR(255) | Calibrating laboratory |
| calibration_date | DATETIME | Calibration date |
| calibration_expiry | DATETIME | Expiry date |
| isc_calibrated_a | FLOAT | Calibrated Isc |
| calibration_uncertainty_pct | FLOAT | Expanded uncertainty (k=2) |
| has_spectral_response | BOOLEAN | SR data available |

#### `spectral_responses`
Spectral response data for reference devices.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| reference_device_id | INTEGER | FK to reference_devices |
| source_lab | VARCHAR(255) | NREL, PTB, etc. |
| measurement_date | DATETIME | SR measurement date |
| wavelength_data | JSON | [{wavelength, sr}, ...] |
| min_wavelength_nm | FLOAT | Minimum wavelength |
| max_wavelength_nm | FLOAT | Maximum wavelength |

### Uncertainty Tables

#### `uncertainty_results`
Calculated uncertainty analysis results.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| measurement_id | INTEGER | FK to measurements |
| analysis_date | DATETIME | Analysis timestamp |
| calculation_method | VARCHAR(50) | GUM, Monte Carlo |
| target_parameter | VARCHAR(50) | Pmax, Voc, etc. |
| measured_value | FLOAT | Measured value |
| combined_standard_uncertainty_pct | FLOAT | Combined u(%) |
| expanded_uncertainty_k2_pct | FLOAT | U (k=2) |
| ci_lower | FLOAT | 95% CI lower bound |
| ci_upper | FLOAT | 95% CI upper bound |
| full_budget_json | JSON | Complete budget data |

#### `uncertainty_components`
Individual uncertainty budget components.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| uncertainty_result_id | INTEGER | FK to uncertainty_results |
| category_id | VARCHAR(10) | Fishbone category |
| factor_id | VARCHAR(20) | Factor identifier |
| name | VARCHAR(255) | Factor name |
| standard_uncertainty | FLOAT | u (standard) |
| distribution | VARCHAR(50) | normal, rectangular, etc. |
| sensitivity_coefficient | FLOAT | Sensitivity |
| percentage_contribution | FLOAT | % contribution |

### Support Tables

#### `sun_simulators`
Sun simulator equipment database.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| manufacturer | VARCHAR(255) | Manufacturer |
| model | VARCHAR(255) | Model |
| lamp_type | VARCHAR(50) | LED, Xenon, etc. |
| classification | VARCHAR(20) | AAA, AA+, etc. |
| typical_uniformity_pct | FLOAT | Non-uniformity |
| typical_temporal_instability_pct | FLOAT | Temporal instability |

#### `files`
File storage and management.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| original_filename | VARCHAR(500) | Original name |
| stored_filename | VARCHAR(500) | UUID-based storage name |
| file_type | ENUM | DATASHEET, CALIBRATION_CERT, etc. |
| storage_path | VARCHAR(1000) | Storage location |
| sha256_hash | VARCHAR(64) | File hash for integrity |
| approval_status | ENUM | DRAFT, APPROVED, etc. |

#### `audit_logs`
Comprehensive audit trail.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| user_id | INTEGER | FK to users |
| action | ENUM | CREATE, UPDATE, DELETE, etc. |
| entity_type | VARCHAR(100) | Affected entity type |
| entity_id | INTEGER | Affected entity ID |
| old_values | JSON | Previous state |
| new_values | JSON | New state |
| ip_address | VARCHAR(45) | Client IP |
| timestamp | DATETIME | Action timestamp |

## Indexes

The schema includes indexes for:
- User email lookup (`idx_users_email`)
- Measurement queries by date, status, module
- Audit log queries by timestamp, entity, user
- File queries by organization, type, status

## Enumerations

### UserRole
- `ADMIN` - Full system access
- `ENGINEER` - Create/edit measurements
- `REVIEWER` - Review and approve
- `VIEWER` - Read-only access

### ApprovalStatus
- `DRAFT` - Initial state
- `SUBMITTED` - Pending review
- `UNDER_REVIEW` - Being reviewed
- `APPROVED` - Approved
- `REJECTED` - Rejected
- `REVISION_REQUESTED` - Needs changes

### MeasurementType
- `STC` - Standard Test Conditions
- `NMOT` - Nominal Module Operating Temperature
- `LOW_IRRADIANCE` - Low irradiance testing
- `TEMPERATURE_COEFFICIENT` - Temp coefficient measurement
- `ENERGY_RATING` - IEC 61853-3 energy rating
- `BIFACIALITY` - Bifacial measurements
- `IAM` - Incidence Angle Modifier
- `SPECTRAL_RESPONSE` - Spectral response measurement

### FileType
- `DATASHEET` - Module datasheet
- `CALIBRATION_CERT` - Calibration certificate
- `CLASSIFICATION_CERT` - Simulator classification
- `TEST_REPORT` - Test report
- `IV_CURVE_DATA` - Raw I-V data
- `SPECTRAL_RESPONSE` - SR data file
- `PAN_FILE` - PVsyst .PAN file

## Railway PostgreSQL Setup

### Environment Variables
```bash
# Railway provides these automatically
DATABASE_URL=postgresql://user:password@host:port/database

# Or set individually
POSTGRES_HOST=your-railway-host
POSTGRES_PORT=5432
POSTGRES_DB=pv_uncertainty
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your-password
```

### Initialize Database
```bash
# Create schema
python -m database.connection --init

# Seed demo data
python -m database.seed_data --all --demo

# Check connection
python -m database.connection --check
```

### Migrations (Alembic)
```bash
# Initialize Alembic
alembic init migrations

# Create migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head
```
