# PV Uncertainty Tool - Development Roadmap

## Project Vision

Transform the basic PV uncertainty tool into a **professional-grade, industry-standard measurement uncertainty analysis platform** for solar PV IV measurements, serving third-party laboratories, manufacturers, researchers, developers, and financial stakeholders.

---

## ✅ PHASE 1: COMPLETED - Core Infrastructure (DONE)

### 1.1 Configuration System ✅
**File:** `config_data.py` (700+ lines)

**Implemented:**
- ✅ **8 PV Technologies:** PERC, TOPCon, HJT, Perovskite, Perovskite-Silicon Tandem, CIGS, CdTe, Custom
- ✅ **13+ Sun Simulators:** Spire (5600SLP, 4600), Eternalsun, Avalon (Nexun, Perovskite, SteadyState), Pasan, ReRa, Lumartix, Atlas, Custom
- ✅ **12+ Reference Labs:** NREL, PTB, AIST, NIMS China, ISFH, Fraunhofer ISE, TÜV Rheinland/SÜD, SUPSI, PI Berlin, DNV, RETC, SERI, CEA-INES, Custom
- ✅ **8 Measurement Types:** STC, NMOT, Low Irradiance, Temperature Coefficients, Energy Rating, Bifaciality, IAM, Spectral Response
- ✅ **5 Standard Spectra:** AM1.5G, AM1.5D, AM1.0, AM0, Custom
- ✅ **8 Currencies:** USD, EUR, INR, CNY, JPY, GBP, CHF, AUD
- ✅ **Hierarchical Uncertainty Categories:** 7 main categories, 15+ subcategories, 40+ individual factors

**Uncertainty Category Structure:**
1. Reference Device (calibration, drift, positioning)
2. Sun Simulator (uniformity, temporal, spectral mismatch)
3. Temperature Measurement (sensor, uniformity, correction)
4. I-V Measurement (voltage, current, curve fitting)
5. Module Characteristics (hysteresis, stabilization)
6. Environmental Conditions (ambient)
7. Measurement Procedure (repeatability, reproducibility, operator)

### 1.2 Enhanced Uncertainty Calculator ✅
**File:** `pv_uncertainty_enhanced.py` (550+ lines)

**Implemented:**
- ✅ `UncertaintyFactor` dataclass with full metadata
- ✅ `PVUncertaintyBudget` class with GUM methodology
- ✅ Category/subcategory/factor ID system (e.g., 1.1.1, 2.3.4)
- ✅ Reference device uncertainty components
- ✅ Sun simulator uncertainty (uniformity, temporal instability, spectral mismatch)
- ✅ Temperature measurement and IEC 60891 correction uncertainties
- ✅ I-V measurement uncertainties (voltage, current, curve fitting)
- ✅ Module behavior (hysteresis, stabilization)
- ✅ Repeatability & Reproducibility (R&R, ILC/Round Robin)
- ✅ Distribution types (normal, rectangular, triangular)
- ✅ Sensitivity coefficients
- ✅ Variance contribution calculation
- ✅ JSON export functionality
- ✅ Category-wise contribution analysis
- ✅ Complete STC measurement calculator

### 1.3 Financial Impact Calculator ✅
**File:** `financial_impact.py` (350+ lines)

**Implemented:**
- ✅ Module price impact calculator (fresh modules)
- ✅ Warranty/insurance claim analysis
- ✅ Project NPV calculator with uncertainty
- ✅ ROI calculations (nominal, lower, upper bounds)
- ✅ Payback period analysis
- ✅ Multi-currency support (8 currencies)
- ✅ Technology-specific price benchmarks
- ✅ Risk assessment metrics
- ✅ Confidence level-based pricing (68%, 95%, 99%)
- ✅ Degradation modeling
- ✅ Discount rate calculations

### 1.4 File Upload & Data Extraction ✅
**File:** `file_utilities.py` (400+ lines)

**Implemented:**
- ✅ PDF text extraction (PyPDF2)
- ✅ Calibration certificate data extraction
- ✅ Excel I-V curve reader
- ✅ Excel summary data extractor
- ✅ Module datasheet parser
- ✅ PVsyst .PAN file parser
- ✅ I-V ratio validation (Vmp/Voc, Imp/Isc)
- ✅ Temperature coefficient validation
- ✅ Fill factor calculator
- ✅ Repeatability data processing
- ✅ Automatic column detection for I-V data

### 1.5 Dependencies Updated ✅
**File:** `requirements.txt`

**Added:**
- ✅ PyPDF2 >= 3.0.0
- ✅ Pillow >= 10.0.0
- ✅ openpyxl >= 3.1.0
- ✅ reportlab >= 4.0.0
- ✅ xlsxwriter >= 3.1.0

---

## 🚧 PHASE 2: IN PROGRESS - Enhanced UI & Integration

### 2.1 Enhanced Streamlit Application (Priority: CRITICAL)
**File:** `streamlit_app_enhanced.py` (NEW)

**To Implement:**

#### Section 1: Module Configuration
- [ ] Technology dropdown (PERC, TOPCon, HJT, etc.)
- [ ] Custom cell configuration (series × parallel)
- [ ] Datasheet upload button
- [ ] Auto-extract from datasheet
- [ ] Manual parameter input fallback
- [ ] Technology-specific defaults
- [ ] Temperature coefficient inputs
- [ ] Bifaciality parameters (for HJT/TOPCon)

#### Section 2: Sun Simulator Configuration
- [ ] Simulator dropdown (by manufacturer/model)
- [ ] Custom simulator option
- [ ] Classification upload (PDF/Word)
- [ ] Spectrum file upload (Excel/Image)
- [ ] AAA/AA+/A+/BBB classification
- [ ] Manual uniformity/temporal/spectral inputs
- [ ] Lamp-to-module distance input
- [ ] Equipment manual upload option

#### Section 3: Measurement Conditions
- [ ] Measurement type selector (STC, NMOT, LI, etc.)
- [ ] Dynamic form based on measurement type
- [ ] Standard spectrum selector (AM1.5G, AM1, AM0, etc.)
- [ ] Custom spectrum upload
- [ ] Concentrated PV spectrum options
- [ ] Environmental condition inputs

#### Section 4: Measurement Data
- [ ] Test report upload (PDF/image/screenshot)
- [ ] Raw I-V data upload (Excel/CSV)
- [ ] Summary file upload
- [ ] Manual data entry fields
- [ ] Auto-populate from uploads
- [ ] Data authenticity checks (Vmp/Voc, Imp/Isc ratios)
- [ ] Fill factor auto-calculation
- [ ] Warning messages for out-of-range values

#### Section 5: Reference Device
- [ ] Reference type selector (WPVS cell / Reference module)
- [ ] Calibration lab dropdown
- [ ] Calibration certificate upload
- [ ] Auto-extract calibration data
- [ ] Reference parameters (Isc, Voc, Pmax, uncertainty)
- [ ] Control chart upload (3-month drift data)
- [ ] Drift analysis visualization
- [ ] Position in test plane

#### Section 6: Repeatability & Reproducibility
- [ ] Repeatability data upload
- [ ] ILC/Round Robin data input
- [ ] Statistical analysis (mean, std, range, CV%)
- [ ] Control chart visualization
- [ ] Reproducibility uncertainty calculation

#### Section 7: Uncertainty Factors (Dynamic)
- [ ] **Interactive Fishbone Diagram** (CRITICAL)
  - [ ] SVG/Canvas-based visualization
  - [ ] 7 main branches with icons
  - [ ] Expandable subcategories
  - [ ] Tree-style sub-branches (slanted, readable)
  - [ ] Category numbering (1, 1.1, 1.1.1)
  - [ ] Click to enable/disable factors
  - [ ] Visual indication of enabled/disabled
  - [ ] Auto-update based on measurement type

- [ ] **Dynamic Input Matrix**
  - [ ] Factors appear based on measurement type
  - [ ] Dropdown for distribution type
  - [ ] Uncertainty value inputs
  - [ ] Sensitivity coefficient inputs (with help)
  - [ ] Unit displays
  - [ ] Notes/comments field
  - [ ] Enable/disable checkboxes

- [ ] **Help System**
  - [ ] "?" icons for each term
  - [ ] Popup explanations
  - [ ] Sensitivity coefficient guidance
  - [ ] Distribution type guide (normal, uniform, triangular)
  - [ ] GUM methodology overview

- [ ] **Calculate Button**
  - [ ] Prominent "Calculate Uncertainty" button
  - [ ] Mandatory field validation
  - [ ] Missing field warnings at top
  - [ ] Progress indicator

#### Section 8: Results & Uncertainty Budget
- [ ] Combined standard uncertainty display
- [ ] Expanded uncertainty (k=2)
- [ ] Relative uncertainty (%)
- [ ] Confidence intervals (68%, 95%, 99%)
- [ ] **Uncertainty Budget Table**
  - [ ] Category/subcategory/factor numbering
  - [ ] Factor names
  - [ ] Standard uncertainties
  - [ ] Sensitivity coefficients
  - [ ] Variance contributions
  - [ ] Percentage contributions
  - [ ] Sortable by contribution

- [ ] **Visualizations**
  - [ ] Uncertainty budget bar chart (by category)
  - [ ] Contribution pie chart
  - [ ] Waterfall chart
  - [ ] Pareto chart (cumulative contribution)

#### Section 9: Financial Impact Analysis
- [ ] Scenario selector
  - [ ] Fresh module nameplate pricing
  - [ ] Warranty/insurance claim
  - [ ] Project financing

- [ ] **Currency Selection**
  - [ ] Currency dropdown (USD, EUR, INR, etc.)
  - [ ] Live/manual exchange rates

- [ ] **Fresh Module Scenario**
  - [ ] Price/Watt input (with technology benchmark)
  - [ ] Price impact calculation
  - [ ] Seller/buyer risk analysis
  - [ ] Price uncertainty visualization

- [ ] **Warranty Claim Scenario**
  - [ ] Nameplate power input
  - [ ] Warranty threshold (%)
  - [ ] Claim validity probability
  - [ ] Recommendation (Valid/Invalid/Uncertain)
  - [ ] Claim value calculation

- [ ] **Project Financing Scenario**
  - [ ] Plant size (MW)
  - [ ] Operating years
  - [ ] Discount rate
  - [ ] Degradation rate
  - [ ] Energy price (Currency/kWh)
  - [ ] NPV calculation (nominal, lower, upper)
  - [ ] ROI analysis
  - [ ] Payback period
  - [ ] Financial risk metrics

- [ ] **"Calculate" Button for Financial**
- [ ] **Visualizations**
  - [ ] NPV confidence interval chart
  - [ ] ROI range visualization
  - [ ] Revenue at risk chart
  - [ ] Payback period comparison

#### Section 10: Professional Reporting
- [ ] **Report Configuration**
  - [ ] Company name input
  - [ ] Company logo upload
  - [ ] Document format number
  - [ ] Record reference number
  - [ ] Preparer name/signature
  - [ ] Reviewer name/signature
  - [ ] Approver name/signature
  - [ ] Date fields

- [ ] **ISO 17025 Format**
  - [ ] Standard header template
  - [ ] Logo placement
  - [ ] Document control fields
  - [ ] Traceability section
  - [ ] Uncertainty statement
  - [ ] Conformity assessment

- [ ] **PDF Report Generation**
  - [ ] Professional layout
  - [ ] High-resolution graphs
  - [ ] Uncertainty budget table
  - [ ] Financial analysis (if applicable)
  - [ ] Measurement conditions summary
  - [ ] Equipment list
  - [ ] Reference device information
  - [ ] Signatures section

- [ ] **Excel Report Generation**
  - [ ] Summary sheet
  - [ ] Uncertainty budget sheet
  - [ ] Financial analysis sheet
  - [ ] Raw data sheet
  - [ ] Embedded charts
  - [ ] Formatted tables

- [ ] **Download Buttons**
  - [ ] PDF Report download
  - [ ] Excel Report download
  - [ ] JSON data export

#### Admin Mode
- [ ] **Admin Login/Toggle**
  - [ ] Password protection
  - [ ] Admin mode indicator

- [ ] **Factor Management**
  - [ ] Add new uncertainty factors
  - [ ] Edit existing factors
  - [ ] Enable/disable factors
  - [ ] Reorder factors
  - [ ] Edit formulas

- [ ] **Equipment Database**
  - [ ] Add new simulators
  - [ ] Edit simulator specs
  - [ ] Add reference labs
  - [ ] Update typical uncertainties

- [ ] **UI Customization**
  - [ ] Show/hide sections
  - [ ] Reorder sections
  - [ ] Edit help text
  - [ ] Customize validation rules

### 2.2 Section Numbering & Navigation
- [ ] Numbered sections/subsections (1, 1.1, 1.1.1)
- [ ] Sidebar navigation with section links
- [ ] Breadcrumb navigation
- [ ] Progress tracker
- [ ] Jump to section functionality
- [ ] Section completion indicators

### 2.3 Measurement Type-Specific Features

**STC:**
- [ ] All standard uncertainty factors
- [ ] IEC 61215 compliance markers

**Low Irradiance:**
- [ ] 200 W/m² specific factors
- [ ] Low-light spectral considerations

**NMOT (P@NMOT):**
- [ ] NMOT temperature input
- [ ] 800 W/m² conditions

**Temperature Coefficients:**
- [ ] Multi-temperature data input
- [ ] Linear regression analysis
- [ ] IEC 60891 procedure selector

**Energy Rating:**
- [ ] Matrix of conditions (3×3 or 4×4)
- [ ] Weighted uncertainty analysis

**Bifaciality:**
- [ ] Front/rear illumination data
- [ ] Bifaciality factor calculation
- [ ] Bifacial gain analysis

**IAM:**
- [ ] Angle of incidence data series
- [ ] Curve fitting uncertainty
- [ ] IAM coefficient extraction

**Spectral Response:**
- [ ] Wavelength-dependent data
- [ ] Spectral mismatch calculation
- [ ] EQE integration

---

## 📋 PHASE 3: Advanced Features

### 3.1 PVsyst Integration
- [ ] .PAN file upload and parsing
- [ ] Energy yield analysis
- [ ] Loss factor extraction
- [ ] Historical yield simulations
- [ ] Monthly energy prediction
- [ ] Uncertainty propagation to yield

### 3.2 Advanced Visualizations
- [ ] Interactive fishbone diagram (Plotly/D3.js)
- [ ] 3D uncertainty surface plots
- [ ] Correlation matrix heatmap
- [ ] Time-series drift plots
- [ ] Control charts
- [ ] Measurement distribution overlays

### 3.3 Database Backend (Optional, for production)
- [ ] User management
- [ ] Measurement history
- [ ] Equipment calibration tracking
- [ ] Reference device drift database
- [ ] Report archive
- [ ] Audit trail

### 3.4 API Development (Optional)
- [ ] REST API for programmatic access
- [ ] Batch processing endpoint
- [ ] Integration with LIMS systems
- [ ] Automated report generation

---

## 🎯 Priority Implementation Order

### IMMEDIATE (Week 1-2):
1. ✅ ~~Core infrastructure modules~~ (DONE)
2. 🚧 Basic enhanced UI skeleton
3. 🚧 Module & Simulator configuration sections
4. 🚧 Measurement data input (STC)
5. 🚧 Simple uncertainty budget calculation

### HIGH PRIORITY (Week 3-4):
6. Fishbone diagram visualization
7. Dynamic uncertainty factor inputs
8. Results & uncertainty budget display
9. Financial impact (fresh module scenario)
10. Basic PDF reporting

### MEDIUM PRIORITY (Week 5-6):
11. File upload/extraction integration
12. Warranty claim scenario
13. Project financing scenario
14. Excel reporting
15. Help system implementation

### LOWER PRIORITY (Week 7+):
16. Admin mode
17. Advanced measurement types (all 8)
18. PVsyst integration
19. Advanced visualizations
20. Database backend

---

## 📊 Current Status Summary

### Completed (Foundation - 40%):
- ✅ **Configuration System:** Equipment, technologies, labs, currencies
- ✅ **Uncertainty Calculator:** GUM methodology, all factor categories
- ✅ **Financial Calculator:** NPV, ROI, warranty, pricing
- ✅ **File Utilities:** PDF, Excel, PAN file parsers
- ✅ **Data Validation:** I-V ratios, temperature coefficients

### In Progress (UI - 10%):
- 🚧 Enhanced Streamlit application structure
- 🚧 Section organization

### To Do (Integration & Features - 50%):
- ⏳ Complete UI implementation
- ⏳ Fishbone diagram
- ⏳ Dynamic forms
- ⏳ Reporting system
- ⏳ Admin mode
- ⏳ Advanced measurement types
- ⏳ PVsyst integration

---

## 🔧 Technical Debt & Improvements

### Code Quality:
- [ ] Add comprehensive docstrings
- [ ] Type hints for all functions
- [ ] Unit tests for calculators
- [ ] Integration tests for UI
- [ ] Error handling improvements
- [ ] Logging system
- [ ] Performance profiling

### Documentation:
- [ ] API documentation
- [ ] User manual
- [ ] Administrator guide
- [ ] Validation documentation
- [ ] Example workflows
- [ ] Video tutorials

### Deployment:
- [ ] Docker containerization
- [ ] Snowflake optimization
- [ ] CI/CD pipeline
- [ ] Automated testing
- [ ] Performance monitoring
- [ ] Error tracking

---

## 📝 Notes for Implementation

### Fishbone Diagram Challenges:
- Streamlit has limited native support for complex SVG/Canvas diagrams
- Consider using:
  - Plotly with custom shapes
  - Graphviz for static diagram generation
  - HTML/JavaScript embedded component
  - Or simplified tree view as fallback

### Dynamic Form Complexity:
- 40+ potential uncertainty factors
- Need smart show/hide logic based on measurement type
- Session state management for form persistence
- Validation before calculation

### File Upload Scalability:
- Handle various file formats (PDF, Excel, CSV, images)
- OCR might be needed for image-based certificates
- Size limits for Snowflake deployment
- Temporary file management

### Financial Calculations:
- Real-time currency conversion (API or manual)
- Regional price benchmarks (need updates)
- NPV sensitivity analysis
- Monte Carlo for financial uncertainty

### Reporting Complexity:
- ISO 17025 format requires specific layout
- High-resolution embedded charts
- Digital signature integration (future)
- PDF/A archival format compliance

---

## 🎓 Testing & Validation Plan

### Test Cases:
1. **STC Measurement:** 300W module, AAA simulator, accredited lab
2. **Warranty Claim:** 10-year old module, borderline performance
3. **Project Financing:** 100 MW plant, 25-year lifetime
4. **Multiple Technologies:** HJT, PERC, Perovskite
5. **Various Labs:** Primary standards vs. commercial labs
6. **Edge Cases:** Zero uncertainty, extreme values

### Validation:
- Compare with manual GUM calculations
- Cross-check with commercial uncertainty software
- Validate against published ILC/round robin results
- Financial model verification with real project data

---

## 📞 Support & Maintenance

### Community:
- User feedback integration
- Bug reporting system
- Feature request tracking
- Community forum (optional)

### Updates:
- Quarterly equipment database updates
- Annual price benchmark reviews
- Standards compliance updates (IEC, JCGM)
- New technology additions (as they emerge)

---

**Last Updated:** 2025-10-22
**Version:** 2.0-dev
**Status:** Phase 1 Complete, Phase 2 In Progress
