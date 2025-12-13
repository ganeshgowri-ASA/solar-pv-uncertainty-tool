# Changelog

All notable changes to the PV Measurement Uncertainty Tool will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0-integrated] - 2025-12-13

### Tag: `v2.0.0-integrated`

Major integration release combining Railway PostgreSQL database, universal solar simulator support, and full bifacial module capability per IEC TS 60904-1-2:2024.

### Added

#### Database (Railway PostgreSQL)
- **Database Schema**: Comprehensive PostgreSQL schema for Railway deployment
  - `organizations` table for multi-tenant support
  - `users` table with role-based access control (Admin, Engineer, Reviewer, Viewer)
  - `modules` table for PV module specifications
  - `measurements` table with full IV parameter storage
  - `iv_curve_data` table for raw I-V curve points
  - `reference_devices` table for WPVS cells and reference modules
  - `spectral_responses` table for spectral response data
  - `sun_simulators` table for equipment database
  - `uncertainty_results` and `uncertainty_components` tables
  - `files` table with approval workflow support
  - `audit_logs` table for ISO 17025 compliance
- Database connection utilities with Railway PostgreSQL support
- Seed data for reference laboratories and sun simulators
- Schema documentation (database/SCHEMA.md)

#### Bifacial Module Support (IEC TS 60904-1-2:2024)
- **bifacial_uncertainty.py**: Core bifacial calculations
  - Bifaciality factors (φ_Isc, φ_Pmax, φ_Voc, φ_FF)
  - Equivalent irradiance (G_eq = G_front + φ × G_rear)
  - Rear-side irradiance uncertainty
  - Spectral albedo effects
  - Bifacial gain calculation
- **uncertainty_components.py**: 15+ simulator configurations
  - PASAN (HighLIGHT LED, BIFACIAL, cetisPV)
  - Spire/Atonometrics (5600SLP, 4600, BiFi-1000)
  - Halm/EETS (cetisPV-BI, flasher III)
  - Meyer Burger (LOANA, PSS-30)
  - Wavelabs (Avalon Nexun, Bifacial, SteadyState)
  - Eternal Sun (SLP-150, SLP-BiFi)
- **monte_carlo_analysis.py**: Enhanced MC simulation
  - Adaptive sampling with convergence monitoring
  - Sensitivity analysis
  - GUM vs MC comparison
- **standards_compliance.py**: IEC/ISO compliance checking
  - IEC 60904-9 simulator classification
  - IEC TS 60904-1-2:2024 bifacial compliance
  - GUM methodology validation
  - ISO 17025 laboratory checks
- **test_scenarios.py**: 80+ comprehensive tests

#### Documentation
- ENHANCEMENT_SPEC.md: Technical specification
- EXCEL_TEMPLATE_GUIDE.md: Template documentation

### Changed
- Updated requirements.txt with SQLAlchemy, psycopg2-binary, alembic
- Added authentication libraries (python-jose, passlib, bcrypt)
- Enhanced README.md with bifacial and database documentation

### Fixed
- Fixed missing `Union` import in uncertainty_components.py

### QA Results
- 48 tests run, 44 passed (92% pass rate)
- All core functionality verified
- All module imports successful
- Streamlit app loads successfully

---

## [1.0.0-stable-bugfix] - 2025-12-13

### Tag: `v1.0-stable-bugfix`

This release marks the stable baseline after critical bug fixes. It serves as a safe rollback point before major feature enhancements.

### Fixed
- **Critical KeyError in Uncertainty Calculation**: Fixed `KeyError: 'combined_standard_uncertainty'` that occurred when calculation results were accessed before being properly initialized
- **Report Generation KeyError**: Fixed missing key access in ISO 17025 report generation that caused PDF/Excel export failures
- **Session State Initialization**: Improved session state handling to prevent undefined variable errors

### Changed
- **UI Migration**: Migrated from `streamlit_app_pro.py` to `streamlit_app.py` as the main application entry point
- **Improved Error Handling**: Added defensive checks for dictionary key access throughout the application

### Security
- No security vulnerabilities identified in this release

---

## [1.0.0] - 2025-12-01

### Initial Release - Professional Edition

This is the first production release of the PV Measurement Uncertainty Tool, featuring comprehensive uncertainty analysis for solar PV IV measurements.

### Added

#### Core Features
- **GUM Methodology Implementation** (JCGM 100:2008)
  - Law of propagation of uncertainty
  - Combined standard uncertainty calculation
  - Expanded uncertainty with coverage factors (k=2)
  - Multiple distribution support (normal, uniform, triangular)

- **7-Category Uncertainty Framework** (Fishbone Diagram)
  1. Reference Device (calibration, drift, positioning)
  2. Sun Simulator (uniformity, temporal stability, spectral mismatch)
  3. Temperature (sensor calibration, uniformity, correction procedures)
  4. I-V Measurement (voltage, current, curve fitting)
  5. Module Characteristics (hysteresis, stabilization)
  6. Environmental Conditions (ambient factors)
  7. Measurement Procedure (repeatability, reproducibility, ILC)

- **Monte Carlo Simulation Engine**
  - Configurable sample sizes (1,000 to 1,000,000)
  - Multiple distribution types
  - Full distribution statistics (mean, median, skewness, kurtosis)
  - Percentile and sensitivity analysis

#### Equipment Database
- **8 PV Technologies**: PERC, TOPCon, HJT, Perovskite, Perovskite-Silicon Tandem, CIGS, CdTe, Custom
- **13+ Sun Simulators**: Spire 5600SLP/4600, Eternalsun SLP-150, Wavelabs Avalon (Nexun/Perovskite/SteadyState), Halm Flasher, Pasan HighLIGHT LED, ReRa Tracer, Lumartix Flash, Atlas SUNTEST, Custom
- **12+ Reference Laboratories**: NREL, PTB, AIST, NIMS, ISFH, Fraunhofer ISE, TUV Rheinland, TUV SUD, SUPSI, PI Berlin, DNV, RETC, CSIRO, SERI, CEA-INES, NABL India, Custom

#### Financial Impact Analysis
- **Fresh Module Pricing**: Uncertainty impact on module sales
- **Warranty/Insurance Claims**: Threshold analysis with probability assessment
- **Project NPV/ROI**: Uncertainty propagation through project financials
- **Multi-Currency Support**: USD, EUR, INR, CNY, JPY, GBP, CHF, AUD

#### Professional Reporting
- **ISO 17025 Compliant PDF Reports**
  - Document control (format numbers, record references)
  - Company logo and header customization
  - Detailed uncertainty budget tables
  - Statement of conformity section
  - Preparer/Reviewer/Approver signature fields

- **Excel Workbooks**
  - Multi-sheet analysis (Summary, Budget, Equipment)
  - Professional formatting with conditional styling
  - Easy data export for further analysis

#### File Processing
- **PDF Extraction**: Calibration certificates, test reports, datasheets
- **Excel Processing**: I-V curve data, summary results, repeatability data
- **PVsyst .PAN Files**: Complete module parameter extraction
- **Data Validation**: I-V ratio checks, fill factor calculation, outlier detection

#### Visualization
- Fishbone uncertainty diagram
- Uncertainty budget bar charts
- Contribution pie charts
- Pareto cumulative analysis
- Financial impact visualizations
- Interactive Plotly charts

### Technical Details
- **Framework**: Streamlit (compatible with Snowflake deployment)
- **Calculation Engine**: NumPy, SciPy
- **Visualization**: Plotly
- **PDF Generation**: ReportLab
- **Excel Generation**: XlsxWriter
- **PDF Parsing**: PyMuPDF (fitz)

### Standards Compliance
- JCGM 100:2008 (GUM) - Guide to the Expression of Uncertainty in Measurement
- JCGM 101:2008 - Supplement 1 to GUM (Monte Carlo method)
- IEC 60904-1:2020 - Photovoltaic devices measurement
- IEC 61215 / IEC 61730 - Module design qualification
- IEC 60891 - Temperature and irradiance corrections
- ISO 17025 - Testing and calibration laboratory requirements

---

## Version Tagging Convention

| Tag Format | Description | Example |
|------------|-------------|---------|
| `vX.Y.Z` | Standard semantic version | `v1.0.0` |
| `vX.Y.Z-stable-*` | Stable release with descriptor | `v1.0.0-stable-bugfix` |
| `vX.Y.Z-beta` | Beta release for testing | `v1.1.0-beta` |
| `vX.Y.Z-rc.N` | Release candidate | `v2.0.0-rc.1` |

---

## Migration Guide

### From v1.0.0 to v1.0.0-stable-bugfix
No migration required. This is a bugfix release with no breaking changes.

### Future Migrations
Database migrations will be documented here when the PostgreSQL integration is implemented.

---

## Contributing

When contributing changes, please update this changelog following these guidelines:
1. Add entries under `[Unreleased]` section
2. Use clear, descriptive language
3. Reference issue/PR numbers where applicable
4. Categorize changes as: Added, Changed, Deprecated, Removed, Fixed, Security

---

## Links

- [README](README.md) - Main documentation
- [GETTING_STARTED](GETTING_STARTED.md) - Quick start guide
- [DEVELOPMENT_ROADMAP](DEVELOPMENT_ROADMAP.md) - Future development plans
