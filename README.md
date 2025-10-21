# ☀️ PV Measurement Uncertainty Tool

A professional-grade tool for calculating measurement uncertainty in photovoltaic (PV) systems using internationally recognized standards.

## Overview

This tool implements **GUM (Guide to the Expression of Uncertainty in Measurement)** methodology and **Monte Carlo simulation** for comprehensive uncertainty analysis of PV measurements. It's designed for solar energy professionals, researchers, and engineers who need to quantify and report measurement uncertainties in accordance with international standards.

### Key Features

- **GUM Methodology**: Calculate combined uncertainty following JCGM 100:2008
- **Monte Carlo Simulation**: Uncertainty propagation following JCGM 101:2008 (GUM Supplement 1)
- **PV-Specific Calculations**:
  - Power measurement uncertainty
  - Performance Ratio (PR) uncertainty
  - Custom uncertainty budgets
- **Interactive Visualizations**:
  - Uncertainty budget charts
  - Distribution histograms
  - Sensitivity analysis
  - Confidence interval plots
- **Batch Processing**: Analyze time-series data from CSV files
- **Export Capabilities**: Download results in CSV and TXT formats
- **Streamlit/Snowflake Ready**: Optimized for cloud deployment

## Quick Start

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

3. Run the application:
   ```bash
   streamlit run streamlit_app.py
   ```

The app will open in your browser at `http://localhost:8501`

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

## Module Structure

```
solar-pv-uncertainty-tool/
├── streamlit_app.py           # Main Streamlit application
├── uncertainty_calculator.py  # GUM methodology implementation
├── monte_carlo.py             # Monte Carlo simulation engine
├── visualizations.py          # Plotly visualization functions
├── data_handler.py            # Data validation and I/O
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

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

**v1.0.0** (Current)
- Initial release
- GUM methodology implementation
- Monte Carlo simulation
- Five analysis modes
- Interactive visualizations
- Batch processing
- CSV export
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
