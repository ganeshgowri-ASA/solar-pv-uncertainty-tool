# ☀️ PV Measurement Uncertainty Tool - Professional Edition

A **comprehensive, production-ready platform** for calculating measurement uncertainty in photovoltaic (PV) IV measurements using internationally recognized standards. Designed for third-party testing laboratories, manufacturers, researchers, and financial stakeholders.

## 🎯 Overview

This tool implements **GUM (JCGM 100:2008)** methodology, **ISO 17025** compliant reporting, and **financial impact analysis** for comprehensive uncertainty quantification in solar PV measurements. It features an extensive equipment database, automated data extraction from certificates, and professional report generation.

### ⭐ Key Features

#### **Comprehensive Uncertainty Analysis**
- **7 Main Uncertainty Categories** with 40+ individual factors:
  - Reference Device (calibration, drift, positioning)
  - Sun Simulator (uniformity, temporal, spectral mismatch)
  - Temperature Measurement & Correction (IEC 60891)
  - I-V Measurement (voltage, current, curve fitting)
  - Module Characteristics (hysteresis, stabilization)
  - Environmental Conditions
  - Measurement Procedure (R&R, ILC/Round Robin)

#### **Industry Equipment Database**
- **8 PV Technologies**: PERC, TOPCon, HJT, Perovskite, Perovskite-Silicon Tandem, CIGS, CdTe, Custom
- **13+ Sun Simulators**: Spire, Eternalsun, Wavelabs Avalon, Pasan, ReRa, Lumartix, Atlas, and more
- **12+ Reference Labs**: NREL, PTB, AIST, NIMS, Fraunhofer ISE, TÜV, SUPSI, PI Berlin, DNV, RETC, and more
- **5 Standard Spectra**: AM1.5G, AM1.5D, AM1.0, AM0, Custom

#### **Financial Impact Analysis**
- **Multi-Scenario Analysis**:
  - Fresh module pricing impact
  - Warranty/insurance claim assessment
  - Project NPV/ROI with uncertainty propagation
- **Multi-Currency Support**: USD, EUR, INR, CNY, JPY, GBP, CHF, AUD
- **Technology-Specific Benchmarks**: 2024 pricing data

#### **Professional Reporting (ISO 17025)** ✅ FULLY INTEGRATED
- PDF reports with company logo and signatures
- Excel workbooks with multiple sheets (Summary, Budget, Equipment)
- Document control (format numbers, record references)
- Preparer/Reviewer/Approver signature fields
- Compliant with ISO 17025 format requirements
- One-click generation and download
- Customizable report options and content

#### **File Upload & Auto-Extraction**
- **PDF**: Calibration certificates, test reports, datasheets
- **Excel**: I-V curve data, summary results, repeatability data
- **PVsyst .PAN Files**: Complete module parameter extraction
- **Automatic Data Validation**: I-V ratio checks, fill factor calculation

#### **Interactive Visualizations**
- Fishbone uncertainty diagram
- Uncertainty budget bar charts
- Contribution pie charts
- Pareto cumulative analysis
- Financial impact visualizations

## 🚀 Quick Start

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd solar-pv-uncertainty-tool
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Professional Edition:**
   ```bash
   streamlit run streamlit_app_pro.py
   ```

   Or the original simplified version:
   ```bash
   streamlit run streamlit_app.py
   ```

The app will open in your browser at `http://localhost:8501`

**📖 First time user?** See **[GETTING_STARTED.md](GETTING_STARTED.md)** for a 5-minute tutorial!

### Using with Snowflake

This tool is compatible with Snowflake Streamlit apps. Deploy it to Snowflake by:

1. Upload all Python files to your Snowflake stage
2. Create a Streamlit app pointing to `streamlit_app.py`
3. The tool will automatically use Snowflake's Python environment

## User Guide

### 1. Power Measurement Uncertainty

Calculate uncertainty for instantaneous PV power measurements.

**Inputs:**
- Irradiance (W/m²) and its uncertainty
- Module temperature (°C) and its uncertainty
- Measured power (W) and power meter uncertainty
- Module efficiency and its uncertainty
- Temperature coefficient (%/°C)

**Outputs:**
- Combined standard uncertainty
- Expanded uncertainty (k=2, 95% confidence)
- Relative uncertainty (%)
- Uncertainty budget showing contribution of each component
- Interactive visualizations

**Example Use Case:** Determining the uncertainty in a flash test measurement of a PV module.

### 2. Performance Ratio Uncertainty

Calculate uncertainty for Performance Ratio (PR) calculations.

**Formula:** PR = E_measured / (H × P_installed)

**Inputs:**
- Measured energy output (kWh)
- Total irradiation (kWh/m²)
- Installed capacity (kWp)
- Uncertainties for each parameter

**Outputs:**
- PR value with combined uncertainty
- Uncertainty budget
- Confidence intervals

**Example Use Case:** Annual performance reporting with uncertainty bounds.

### 3. Custom Uncertainty Analysis

Build your own uncertainty budget with user-defined components.

**Features:**
- Add multiple uncertainty components
- Define distribution types (normal, uniform, triangular)
- Specify sensitivity coefficients
- Calculate combined uncertainty using GUM method
- Export detailed budget

**Example Use Case:** Analyzing complex measurement chains or custom PV metrics.

### 4. Monte Carlo Simulation

Run Monte Carlo simulations for non-linear uncertainty propagation.

**Features:**
- Configurable number of samples (1,000 to 1,000,000)
- Multiple distribution types
- Full distribution statistics (mean, median, skewness, kurtosis)
- Percentile analysis
- Sensitivity analysis via correlation

**Example Use Case:** Validating GUM results or analyzing non-Gaussian distributions.

### 5. Batch Data Analysis

Process time-series PV measurement data.

**Features:**
- CSV file upload
- Data validation and quality checks
- Batch uncertainty calculations
- Time-series visualization with uncertainty bands
- Statistical summaries
- Export results

**Example Use Case:** Analyzing hourly or daily PV production data with uncertainties.

## Technical Details

### Calculation Methods

#### GUM Methodology

The tool implements the law of propagation of uncertainty:

```
u_c²(y) = Σ(∂f/∂x_i)² · u²(x_i) + 2·Σ Σ(∂f/∂x_i)·(∂f/∂x_j)·u(x_i,x_j)
```

Where:
- `u_c(y)` is the combined standard uncertainty
- `∂f/∂x_i` are sensitivity coefficients
- `u(x_i)` are standard uncertainties of input quantities
- Correlation terms are included when relevant

#### Monte Carlo Method

Implements GUM Supplement 1 approach:
1. Sample from input probability distributions
2. Propagate through measurement model
3. Analyze output distribution
4. Calculate coverage intervals

### Supported Distributions

- **Normal (Gaussian)**: Most common for calibrated instruments
- **Uniform (Rectangular)**: For instruments with manufacturer's specification
- **Triangular**: When limits and most likely value are known
- **Log-Normal**: For strictly positive quantities

### Uncertainty Budget Components

The tool automatically calculates:
- Standard uncertainty for each component
- Sensitivity coefficients
- Variance contributions
- Percentage contributions
- Correlation effects (when specified)

## Architecture Overview

The PV Measurement Uncertainty Tool is a modular, production-ready platform built with Streamlit. The architecture follows a clean separation of concerns:

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PRESENTATION LAYER                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     streamlit_app.py                                 │   │
│  │  - 8 Main Sections (Module, Simulator, Reference, Data, Uncertainty,│   │
│  │    Results, Financial, Reporting)                                    │   │
│  │  - Session State Management                                          │   │
│  │  - Interactive UI with Custom CSS                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BUSINESS LOGIC LAYER                               │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐  │
│  │ pv_uncertainty_      │  │ uncertainty_         │  │ monte_carlo.py   │  │
│  │ enhanced.py          │  │ calculator.py        │  │                  │  │
│  │ - PVUncertaintyBudget│  │ - Basic GUM methods  │  │ - Monte Carlo    │  │
│  │ - 7-Category Fishbone│  │ - Power uncertainty  │  │   simulations    │  │
│  │ - STC Calculations   │  │ - PR uncertainty     │  │ - Distribution   │  │
│  └──────────────────────┘  └──────────────────────┘  └──────────────────┘  │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐  │
│  │ financial_impact.py  │  │ report_generator.py  │  │ visualizations.py│  │
│  │ - Module pricing     │  │ - ISO 17025 PDF      │  │ - Plotly charts  │  │
│  │ - Warranty claims    │  │ - Excel workbooks    │  │ - Fishbone       │  │
│  │ - Project NPV/ROI    │  │ - Signature sections │  │ - Pareto/Pie     │  │
│  └──────────────────────┘  └──────────────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             DATA LAYER                                       │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐  │
│  │ config_data.py       │  │ file_utilities.py    │  │ data_handler.py  │  │
│  │ - PV_TECHNOLOGIES    │  │ - PDFExtractor       │  │ - DataValidator  │  │
│  │ - SUN_SIMULATORS     │  │ - ExcelExtractor     │  │ - CSV processing │  │
│  │ - REFERENCE_LABS     │  │ - PVsystPANParser    │  │ - I/O operations │  │
│  │ - CURRENCIES         │  │ - DatasheetExtractor │  │                  │  │
│  └──────────────────────┘  └──────────────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module Structure

```
solar-pv-uncertainty-tool/
│
├── streamlit_app.py              # Main Professional Edition application
├── streamlit_app_legacy.py       # Legacy simplified version
│
├── CORE CALCULATION MODULES
├── uncertainty_calculator.py     # Basic GUM methodology (PVUncertaintyCalculator)
├── pv_uncertainty_enhanced.py    # Enhanced 7-category uncertainty budget
├── monte_carlo.py                # Monte Carlo simulation engine
│
├── DATA & CONFIGURATION
├── config_data.py                # Equipment database & reference data
│                                 # - 8 PV technologies (PERC, TOPCon, HJT, etc.)
│                                 # - 13+ sun simulators (Spire, Wavelabs, etc.)
│                                 # - 12+ reference labs (NREL, PTB, etc.)
│                                 # - 8 currencies with exchange rates
│                                 # - 7 uncertainty category definitions
├── data_handler.py               # Data validation and I/O utilities
├── file_utilities.py             # File extraction (PDF, Excel, PVsyst .PAN)
│
├── OUTPUT GENERATION
├── report_generator.py           # ISO 17025 compliant PDF/Excel reports
├── visualizations.py             # Plotly visualizations
├── financial_impact.py           # Financial impact analysis
│
├── DOCUMENTATION
├── README.md                     # This documentation
├── GETTING_STARTED.md            # Quick start guide
├── DEVELOPMENT_ROADMAP.md        # Future development plans
├── CHANGELOG.md                  # Version history
│
├── CONFIGURATION FILES
├── requirements.txt              # Python dependencies
├── .devcontainer/                # DevContainer configuration
└── .github/workflows/            # GitHub Actions workflows
```

### Key Components

#### 1. Uncertainty Calculation Engine (`pv_uncertainty_enhanced.py`)
The core calculation engine implements GUM (JCGM 100:2008) methodology with a comprehensive 7-category fishbone structure:

| Category | Description | Key Factors |
|----------|-------------|-------------|
| 1. Reference Device | Calibration chain | Calibration, drift, positioning |
| 2. Sun Simulator | Equipment performance | Uniformity, temporal stability, spectral match |
| 3. Temperature | Thermal effects | Sensor calibration, uniformity, coefficients |
| 4. I-V Measurement | Electrical measurement | Voltage, current, curve fitting |
| 5. Module Behavior | Device characteristics | Hysteresis, stabilization |
| 6. Environment | Ambient conditions | Temperature, humidity, pressure |
| 7. Procedure | Measurement process | Repeatability, reproducibility (R&R, ILC) |

#### 2. Equipment Database (`config_data.py`)
Pre-configured database with typical specifications for common PV testing equipment, enabling quick setup while allowing customization.

#### 3. Financial Impact Analysis (`financial_impact.py`)
Three analysis modes:
- **Fresh Module Pricing**: Uncertainty impact on module sales price
- **Warranty Claims**: Statistical analysis for warranty threshold assessment
- **Project NPV/ROI**: Uncertainty propagation through project financials

#### 4. Report Generator (`report_generator.py`)
ISO 17025 compliant output with:
- PDF reports with document control, signatures, and professional formatting
- Excel workbooks with multi-sheet analysis
- Customizable headers, logos, and metadata

## API Reference

### UncertaintyCalculator

```python
from uncertainty_calculator import PVUncertaintyCalculator

calc = PVUncertaintyCalculator()
calc.add_component(
    name="Irradiance",
    value=1000.0,
    uncertainty=20.0,
    uncertainty_type="standard",
    distribution="normal",
    sensitivity_coefficient=0.2
)
combined_unc, budget = calc.calculate_combined_uncertainty()
```

### Monte Carlo Simulator

```python
from monte_carlo import PVMonteCarlo

result = PVMonteCarlo.simulate_power_uncertainty(
    irradiance_mean=1000.0,
    irradiance_std=20.0,
    temperature_mean=45.0,
    temperature_std=1.0,
    power_meter_std=2.0,
    n_samples=100000
)
```

## Standards and References

This tool implements methods from:

- **JCGM 100:2008** - Evaluation of measurement data — Guide to the expression of uncertainty in measurement (GUM)
- **JCGM 101:2008** - Evaluation of measurement data — Supplement 1 to the GUM — Propagation of distributions using a Monte Carlo method
- **IEC 61724-1:2021** - Photovoltaic system performance — Part 1: Monitoring
- **ISO/IEC Guide 98-3:2008** - Uncertainty of measurement

## Example Applications

### 1. Module Flash Test Uncertainty

Quantify uncertainty in PV module power rating based on:
- Pyranometer calibration uncertainty
- Spectrum mismatch
- Temperature sensor accuracy
- Power meter precision
- Non-uniformity of irradiance

### 2. Array Performance Monitoring

Calculate PR uncertainty including:
- Energy meter accuracy
- Plane-of-array irradiance sensor
- Rated capacity tolerance
- Data acquisition uncertainty
- Calculation algorithm effects

### 3. Warranty Claim Verification

Demonstrate whether measured performance is within specification considering:
- Measurement uncertainties
- Environmental corrections
- Degradation models
- Statistical confidence levels

## Performance Optimization

The tool is optimized for:
- Fast GUM calculations (< 1 second)
- Efficient Monte Carlo sampling (100k samples in ~2-5 seconds)
- Large batch processing (1000s of records)
- Minimal memory footprint for Snowflake deployment

## Troubleshooting

### Common Issues

**Issue**: Sensitivity coefficients are zero or incorrect
- **Solution**: Check that measured values are non-zero and sensitivity calculation logic is appropriate for your model

**Issue**: Monte Carlo results differ from GUM
- **Solution**: This is normal for non-linear models or non-Gaussian distributions. Use more samples or review input distributions

**Issue**: Batch processing is slow
- **Solution**: Reduce number of records, simplify uncertainty model, or use vectorized calculations

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

Apache License 2.0 - See LICENSE file for details

## Support

For issues, questions, or suggestions:
- Create an issue on GitHub
- Review the documentation
- Check the troubleshooting section

## Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

**v1.0-stable-bugfix** (Current - December 2025)
- Fixed critical KeyError bugs in uncertainty calculation and reporting
- Migrated to Professional Edition UI
- Comprehensive 7-category uncertainty budget (40+ individual factors)
- ISO 17025 compliant PDF/Excel report generation
- Financial impact analysis (fresh module, warranty, project NPV)
- Equipment database (13+ simulators, 12+ reference labs, 8 technologies)
- File extraction from PDFs, Excel, PVsyst .PAN files
- Multi-currency support (USD, EUR, INR, CNY, JPY, GBP, CHF, AUD)

**v1.0.0** (Initial Release)
- Initial release with basic GUM methodology
- Monte Carlo simulation engine
- Five analysis modes
- Interactive visualizations
- Batch processing capabilities
- CSV export functionality
- Streamlit/Snowflake compatible

## Acknowledgments

Developed following international metrological standards and best practices from:
- BIPM (International Bureau of Weights and Measures)
- IEC (International Electrotechnical Commission)
- ISO (International Organization for Standardization)
- NREL (National Renewable Energy Laboratory) PV measurement guidelines

---

**Author**: Solar Energy Uncertainty Analysis Team
**Contact**: [Your Contact Information]
**Last Updated**: 2025
