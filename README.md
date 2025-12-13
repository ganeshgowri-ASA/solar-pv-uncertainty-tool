# Universal Solar Simulator Uncertainty Framework

## Professional Edition v2.0 - Bifacial PV Support

A **comprehensive, production-ready platform** for calculating measurement uncertainty in photovoltaic (PV) IV measurements using internationally recognized standards. Now featuring **full bifacial module support per IEC TS 60904-1-2:2024**.

---

## Overview

This tool implements **GUM (JCGM 100:2008)** methodology, **ISO 17025** compliant reporting, and **financial impact analysis** for comprehensive uncertainty quantification in solar PV measurements. Version 2.0 introduces **universal solar simulator support** and complete **bifacial measurement capabilities**.

### What's New in v2.0

- **Universal Simulator Support**: PASAN, Spire, Halm, Meyer Burger, Wavelabs, Eternal Sun, and custom
- **Bifacial Module Support**: Full IEC TS 60904-1-2:2024 compliance
- **Enhanced Uncertainty Budget**: 11 categories with 60+ individual factors
- **Equivalent Irradiance Calculation**: G_eq = G_front + phi x G_rear
- **Bifaciality Factor Analysis**: phi_Isc, phi_Pmax, phi_Voc with uncertainties
- **Spectral Albedo Effects**: Ground reflectance uncertainty
- **Advanced Monte Carlo**: Adaptive sampling with convergence monitoring
- **Standards Compliance Checker**: Automated IEC/ISO compliance verification

---

## Key Features

### Comprehensive Uncertainty Analysis

**11 Main Uncertainty Categories** with 60+ individual factors:

| Category | Subcategories | Bifacial-Specific |
|----------|---------------|-------------------|
| 1. Reference Device | Calibration, Stability, Positioning | Rear positioning |
| 2. Simulator - Front | Uniformity, Temporal, Spectral | - |
| 3. Simulator - Rear | Uniformity, Temporal, Spectral, Irradiance | Yes |
| 4. Temperature | Sensor, Uniformity, Correction | Front-rear gradient |
| 5. I-V Measurement | Voltage, Current, DAQ | - |
| 6. Module Characteristics | Variability, Behavior | Edge effects |
| 7. Bifaciality Factor | phi determination, application | Yes |
| 8. Equivalent Irradiance | G_eq calculation, correction | Yes |
| 9. Environmental | Ambient, Albedo | Spectral albedo |
| 10. Measurement Procedure | Repeatability, Reproducibility | - |
| 11. Parasitic Effects | Optical crosstalk, Electrical | Yes |

### Universal Simulator Database

**15+ Solar Simulators** from major manufacturers:

| Manufacturer | Models | Type | Classification | Bifacial |
|--------------|--------|------|----------------|----------|
| PASAN | HighLIGHT LED, BIFACIAL, cetisPV | LED | AAA+ | Yes |
| Spire/Atonometrics | 5600SLP, 4600, BiFi-1000 | Xenon/Hybrid | AAA | Yes |
| Halm/EETS | cetisPV-BI, flasher III | Hybrid | AAA+ | Yes |
| Meyer Burger | LOANA, PSS-30 | LED | AAA+ | Yes |
| Wavelabs | Avalon Nexun, Bifacial, SteadyState | LED | AAA | Yes |
| Eternal Sun | SLP-150, SLP-BiFi | LED | AAA+ | Yes |
| Custom | User-defined | Any | Configurable | Yes |

### Bifacial Measurement Modes

Per IEC TS 60904-1-2:2024:

1. **Single-Sided Front**: Front illumination, rear dark (<0.1 W/m²)
2. **Single-Sided Rear**: Rear illumination, front dark
3. **Double-Sided Simultaneous**: Concurrent front and rear illumination
4. **Equivalent Irradiance**: Adjusted for G_eq = 1000 W/m² at STC

### Financial Impact Analysis

- **Multi-Scenario Analysis**: Module pricing, warranty claims, project NPV
- **Multi-Currency Support**: USD, EUR, INR, CNY, JPY, GBP, CHF, AUD
- **Bifacial Gain Impact**: Financial effect of bifacial energy gain

### Professional Reporting (ISO 17025)

- PDF/Excel reports with complete uncertainty budgets
- Bifacial-specific result sections
- Compliance status for all applicable standards
- Preparer/Reviewer/Approver signature fields

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd solar-pv-uncertainty-tool

# Install dependencies
pip install -r requirements.txt

# Run the Professional Edition
streamlit run streamlit_app_pro.py
```

The app will open in your browser at `http://localhost:8501`

### Basic Usage - Bifacial Module

```python
from bifacial_uncertainty import BifacialModule, BifacialUncertaintyCalculator

# Create a bifacial module
module = BifacialModule(
    manufacturer="Example Solar",
    model="TOPCon-400-BF",
    pmax_front=400.0,
    pmax_rear=280.0,
    isc_front=10.5,
    isc_rear=7.87
)

# Calculate bifaciality factors
calc = BifacialUncertaintyCalculator(module=module)
factors = calc.calculate_bifaciality_factors(
    front_isc=10.5, front_voc=45.0, front_pmax=400.0, front_ff=0.85,
    rear_isc=7.87, rear_voc=44.8, rear_pmax=280.0, rear_ff=0.84
)

print(f"phi_Isc: {factors.phi_isc:.3f} +/- {factors.u_phi_isc:.4f}")
print(f"phi_Pmax: {factors.phi_pmax:.3f} +/- {factors.u_phi_pmax:.4f}")
```

### Build Uncertainty Budget

```python
from uncertainty_components import UncertaintyBudgetBuilder

# Build bifacial STC uncertainty budget
builder = UncertaintyBudgetBuilder(is_bifacial=True)
budget = (
    builder
    .set_simulator("pasan_highlight_bifacial")
    .set_reference("nrel_wpvs")
    .build_bifacial_stc()
)

print(f"Combined uncertainty (k=1): {budget.combined_standard_uncertainty:.2f}%")
print(f"Expanded uncertainty (k=2): {budget.expanded_uncertainty:.2f}%")
```

### Run Monte Carlo Analysis

```python
from monte_carlo_analysis import BifacialMonteCarloSimulator, BifacialMCInputs

# Configure inputs
inputs = BifacialMCInputs()
inputs.g_front.mean = 900.0
inputs.g_rear.mean = 135.0
inputs.phi_isc.mean = 0.75

# Run simulation
sim = BifacialMonteCarloSimulator(n_samples=100000, seed=42)
sim.set_bifacial_inputs(inputs)
result = sim.simulate_equivalent_irradiance()

print(f"G_eq: {result.mean:.1f} +/- {result.std:.1f} W/m²")
print(f"95% coverage: [{result.coverage_95[0]:.1f}, {result.coverage_95[1]:.1f}]")
```

### Check Standards Compliance

```python
from standards_compliance import check_bifacial_stc_compliance

results = check_bifacial_stc_compliance(
    g_front=1000.0,
    g_rear=150.0,
    temperature=25.2,
    phi_isc=0.75,
    phi_pmax=0.70,
    uniformity_front=1.8,
    uniformity_rear=2.5,
    temporal_front=0.3,
    temporal_rear=0.5
)

print(results['bifacial'].summary())
print(f"Simulator class: {results['classification'].overall_class}")
```

---

## Module Structure

```
solar-pv-uncertainty-tool/
├── Core Application
│   ├── streamlit_app_pro.py      # Professional Streamlit UI
│   ├── streamlit_app.py          # Original simplified UI
│   └── config_data.py            # Equipment database
│
├── Uncertainty Calculation (Original)
│   ├── uncertainty_calculator.py  # GUM methodology
│   ├── pv_uncertainty_enhanced.py # Enhanced calculator
│   └── monte_carlo.py             # Basic MC simulation
│
├── Bifacial Enhancement (NEW v2.0)
│   ├── bifacial_uncertainty.py    # Bifacial calculations
│   ├── uncertainty_components.py  # Component classes
│   ├── monte_carlo_analysis.py    # Enhanced MC simulation
│   └── standards_compliance.py    # IEC standards checking
│
├── Support Modules
│   ├── report_generator.py        # ISO 17025 reports
│   ├── financial_impact.py        # Financial analysis
│   ├── visualizations.py          # Plotly charts
│   ├── file_utilities.py          # File I/O
│   └── data_handler.py            # Data validation
│
├── Testing
│   └── test_scenarios.py          # Comprehensive tests
│
├── Documentation
│   ├── README.md                  # This file
│   ├── ENHANCEMENT_SPEC.md        # Technical specification
│   ├── EXCEL_TEMPLATE_GUIDE.md    # Template documentation
│   ├── GETTING_STARTED.md         # User tutorial
│   └── DEVELOPMENT_ROADMAP.md     # Development plan
│
└── requirements.txt               # Dependencies
```

---

## Technical Details

### Bifaciality Factor Calculation

Per IEC TS 60904-1-2:2024:

```
phi_Isc = Isc_rear / Isc_front
phi_Voc = Voc_rear / Voc_front
phi_Pmax = Pmax_rear / Pmax_front
phi_FF = FF_rear / FF_front
```

**Uncertainty propagation** (with correlation):
```
u(phi)/phi = sqrt[(u_rear/X_rear)² + (u_front/X_front)² - 2*rho*(u_rear/X_rear)*(u_front/X_front)]
```

### Equivalent Irradiance

```
G_eq = G_front + phi × G_rear
```

**For target G_eq = 1000 W/m² with ratio R = G_rear/G_front:**
```
G_front = 1000 / (1 + phi × R)
G_rear = R × G_front
```

**Uncertainty propagation:**
```
u²(G_eq) = u²(G_front) + phi² × u²(G_rear) + G_rear² × u²(phi)
```

### Bifacial Gain

```
BG = (P_bifacial - P_mono) / P_mono
```

Typical values: 5-25% depending on installation conditions.

---

## Supported Standards

| Standard | Version | Description |
|----------|---------|-------------|
| IEC TS 60904-1-2 | 2024 | Bifacial PV device characterization |
| IEC 60904-1 | 2020 | I-V characteristics measurement |
| IEC 60904-3 | 2019 | Spectral irradiance reference |
| IEC 60904-7 | 2019 | Spectral mismatch correction |
| IEC 60904-9 | 2020 | Solar simulator classification |
| IEC 60891 | 2021 | Temperature/irradiance corrections |
| JCGM 100 | 2008 | GUM uncertainty methodology |
| JCGM 101 | 2008 | Monte Carlo method |
| ISO 17025 | 2017 | Laboratory competence |

---

## Typical Uncertainty Values

### Monofacial Measurements

| Parameter | Combined (k=1) | Expanded (k=2) |
|-----------|----------------|----------------|
| Pmax STC | 1.8-2.5% | 3.6-5.0% |
| Isc | 1.5-2.0% | 3.0-4.0% |
| Voc | 0.5-1.0% | 1.0-2.0% |
| FF | 1.0-1.5% | 2.0-3.0% |

### Bifacial Measurements

| Parameter | Combined (k=1) | Expanded (k=2) |
|-----------|----------------|----------------|
| Pmax front | 1.8-2.5% | 3.6-5.0% |
| Pmax rear | 2.5-3.5% | 5.0-7.0% |
| phi_Isc | 2.0-3.0% | 4.0-6.0% |
| phi_Pmax | 2.5-3.5% | 5.0-7.0% |
| G_equivalent | 2.0-2.8% | 4.0-5.6% |
| Bifacial gain | 10-20% relative | - |

---

## Running Tests

```bash
# Run all tests
python test_scenarios.py

# Run with verbose output
python -m pytest test_scenarios.py -v

# Run specific test class
python -m pytest test_scenarios.py::TestBifacialityFactors -v
```

---

## Performance

- GUM calculations: < 100 ms
- Monte Carlo (100k samples): 2-5 seconds
- Monte Carlo (1M samples): < 10 seconds
- Budget construction: < 500 ms
- Compliance checks: < 100 ms

---

## Backward Compatibility

All existing monofacial calculations remain fully functional:

```python
# Existing code continues to work
from uncertainty_components import UncertaintyBudgetBuilder

builder = UncertaintyBudgetBuilder(is_bifacial=False)
budget = builder.build_standard_stc()  # Monofacial budget
```

The original `uncertainty_calculator.py` and `monte_carlo.py` modules are unchanged.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

---

## License

Apache License 2.0 - See LICENSE file for details

---

## Version History

### v2.0.0 (Current)
- Universal solar simulator support
- Full bifacial module support per IEC TS 60904-1-2:2024
- Enhanced 11-category uncertainty budget
- Equivalent irradiance calculations
- Bifaciality factor analysis with uncertainty
- Advanced Monte Carlo with adaptive sampling
- Standards compliance checker
- Comprehensive test suite

### v1.0.0
- Initial release
- GUM methodology implementation
- Monte Carlo simulation
- ISO 17025 reporting
- Financial impact analysis

---

## Acknowledgments

- BIPM (International Bureau of Weights and Measures)
- IEC (International Electrotechnical Commission)
- ISO (International Organization for Standardization)
- NREL (National Renewable Energy Laboratory)
- Fraunhofer ISE bifacial PV research

---

**Author**: Universal Solar Simulator Framework Team
**Last Updated**: 2025
