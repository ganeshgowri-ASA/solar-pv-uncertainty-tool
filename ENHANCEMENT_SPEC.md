# Universal Solar Simulator Uncertainty Framework
## Technical Specification v2.0

### IEC TS 60904-1-2:2024 Bifacial Module Support

---

## Table of Contents

1. [Overview](#1-overview)
2. [Universal Simulator Support](#2-universal-simulator-support)
3. [Bifacial Measurement Methodology](#3-bifacial-measurement-methodology)
4. [Enhanced Uncertainty Budget](#4-enhanced-uncertainty-budget)
5. [GUM Methodology Implementation](#5-gum-methodology-implementation)
6. [IEC TS 60904-1-2:2024 Compliance](#6-iec-ts-60904-1-22024-compliance)
7. [API Specification](#7-api-specification)
8. [Validation Requirements](#8-validation-requirements)

---

## 1. Overview

### 1.1 Purpose

This specification defines the enhancement of the PV Measurement Uncertainty Tool from a PASAN-specific implementation to a **Universal Solar Simulator Uncertainty Framework** with comprehensive bifacial module support per IEC TS 60904-1-2:2024.

### 1.2 Scope

- Universal simulator support (PASAN, Spire, Halm, Meyer Burger, Wavelabs, and custom)
- Complete bifacial measurement modes (single-sided, double-sided simultaneous, equivalent irradiance)
- Enhanced uncertainty budget with rear-side and bifaciality-specific components
- Full GUM (JCGM 100:2008) methodology compliance
- IEC TS 60904-1-2:2024 bifacial characterization standard compliance
- Backward compatibility with existing monofacial calculations

### 1.3 Key References

| Standard | Title | Application |
|----------|-------|-------------|
| IEC TS 60904-1-2:2024 | PV devices - Part 1-2: Measurement of current-voltage characteristics of bifacial PV devices | Primary bifacial standard |
| IEC 60904-1:2020 | PV devices - Part 1: Measurement of photovoltaic current-voltage characteristics | Base I-V measurement |
| IEC 60904-3:2019 | PV devices - Part 3: Measurement principles for terrestrial PV solar devices with reference spectral irradiance data | Spectral requirements |
| IEC 60904-7:2019 | PV devices - Part 7: Computation of spectral mismatch correction | Spectral mismatch |
| IEC 60904-8:2014 | PV devices - Part 8: Measurement of spectral responsivity | Spectral response |
| IEC 60904-9:2020 | PV devices - Part 9: Classification and requirements of solar simulators | Simulator classification |
| IEC 60891:2021 | PV devices - Procedures for temperature and irradiance corrections | Corrections procedure |
| JCGM 100:2008 | GUM - Guide to Expression of Uncertainty in Measurement | Uncertainty methodology |
| JCGM 101:2008 | GUM Supplement 1 - Propagation of distributions using Monte Carlo | MC methodology |

---

## 2. Universal Simulator Support

### 2.1 Supported Simulator Platforms

#### 2.1.1 PASAN (h.a.l.m. elektronik GmbH)

| Model | Type | Classification | Bifacial Capability | Key Specifications |
|-------|------|----------------|---------------------|-------------------|
| HighLIGHT LED | LED | AAA+ | Single-sided | 1.5% uniformity, 0.2% temporal |
| HighLIGHT BIFACIAL | LED | AAA+ | Dual-sided simultaneous | Independent front/rear control |
| cetisPV-CTL | LED | AAA | Single-sided | Inline integration |
| cetisPV-CELL | LED | AAA+ | Cell-level | High-resolution uniformity |

#### 2.1.2 Spire/Atonometrics

| Model | Type | Classification | Bifacial Capability | Key Specifications |
|-------|------|----------------|---------------------|-------------------|
| 5600SLP | Xenon | AAA | Single-sided | 2.0% uniformity, 0.5% temporal |
| 4600 | Xenon | AAA | Single-sided | Modular platform |
| BiFi-1000 | Xenon + LED | AAA | Dual-sided simultaneous | Integrated rear illumination |

#### 2.1.3 Halm/EETS

| Model | Type | Classification | Bifacial Capability | Key Specifications |
|-------|------|----------------|---------------------|-------------------|
| cetisPV-BI | Xenon/LED | AAA+ | Dual-sided | Configurable rear source |
| flasher III | Xenon | AAA | Single-sided | High throughput |
| cetisPV-LL | LED | AAA | Single-sided | Long-pulse LED |

#### 2.1.4 Meyer Burger

| Model | Type | Classification | Bifacial Capability | Key Specifications |
|-------|------|----------------|---------------------|-------------------|
| LOANA | LED | AAA+ | Single-sided | High-efficiency cells |
| PSS-30 | LED | AAA | Dual-sided sequential | Production line |

#### 2.1.5 Wavelabs

| Model | Type | Classification | Bifacial Capability | Key Specifications |
|-------|------|----------------|---------------------|-------------------|
| Avalon Nexun | LED | AAA | Single-sided | Spectral tunability |
| Avalon Bifacial | LED | AAA+ | Dual-sided | Front/rear independent |
| Avalon SteadyState | LED | AAA | Single-sided | Continuous illumination |

#### 2.1.6 Eternal Sun

| Model | Type | Classification | Bifacial Capability | Key Specifications |
|-------|------|----------------|---------------------|-------------------|
| SLP-150 | LED | AAA+ | Single-sided | 1.5% uniformity |
| SLP-BiFi | LED | AAA+ | Dual-sided | Simultaneous front/rear |

#### 2.1.7 Custom Simulator Configuration

```python
class CustomSimulatorConfig:
    """User-defined simulator configuration"""
    manufacturer: str
    model: str
    lamp_type: Literal["LED", "Xenon", "Halogen", "Hybrid"]
    classification: Literal["AAA+", "AAA", "ABA", "BAA", "Custom"]

    # Front side specifications
    front_uniformity: float          # % (k=1)
    front_temporal_instability: float # % (k=1)
    front_spectral_match: str        # A, B, C, or Custom
    front_irradiance_range: Tuple[float, float]  # W/m²

    # Rear side specifications (if bifacial capable)
    bifacial_capable: bool
    rear_uniformity: Optional[float]
    rear_temporal_instability: Optional[float]
    rear_spectral_match: Optional[str]
    rear_irradiance_range: Optional[Tuple[float, float]]
    rear_independent_control: bool   # Can set rear G independently?

    # Spectral configuration
    spectral_tunability: bool
    wavelength_range: Tuple[float, float]  # nm
    spectral_resolution: Optional[float]   # nm

    # Physical configuration
    test_area: Tuple[float, float]   # mm × mm
    working_distance: float          # mm
    flash_duration: Optional[float]  # ms (for flash systems)
```

### 2.2 Simulator Selection Logic

```python
def select_simulator(
    manufacturer: str,
    model: str,
    measurement_type: str,
    module_type: str
) -> SimulatorProfile:
    """
    Select appropriate simulator configuration based on:
    - Manufacturer and model from database
    - Measurement type (STC, bifacial, spectral, etc.)
    - Module technology (affects spectral requirements)

    Returns complete simulator profile with uncertainties.
    """
```

---

## 3. Bifacial Measurement Methodology

### 3.1 Measurement Modes per IEC TS 60904-1-2:2024

#### 3.1.1 Single-Sided Illumination (Method A)

**Purpose**: Determine front and rear side performance independently.

**Procedure**:
1. Front-side measurement at G_front = 1000 W/m², G_rear = 0 (dark)
2. Rear-side measurement at G_rear = 1000 W/m², G_front = 0 (dark)
3. Dark side must be <0.1 W/m² irradiance (per IEC TS 60904-1-2:2024 §5.2.1)

**Measured Parameters**:
- Front: I_sc,front, V_oc,front, P_max,front, FF_front
- Rear: I_sc,rear, V_oc,rear, P_max,rear, FF_rear

**Bifaciality Factors**:
```
φ_Isc = I_sc,rear / I_sc,front
φ_Voc = V_oc,rear / V_oc,front
φ_Pmax = P_max,rear / P_max,front
φ_FF = FF_rear / FF_front
```

**Uncertainty Sources (Single-Sided)**:
- Front/rear uniformity independence
- Light leakage to dark side
- Temperature equilibration between measurements
- Reference device spectral response differences

#### 3.1.2 Double-Sided Simultaneous Illumination (Method B)

**Purpose**: Measure actual bifacial performance under dual illumination.

**Procedure**:
1. Configure front irradiance: G_front (typically 1000 W/m²)
2. Configure rear irradiance: G_rear (typically 50-400 W/m² for φ = 5-40%)
3. Simultaneous illumination measurement
4. Calculate equivalent irradiance

**Equivalent Irradiance Model**:
```
G_eq = G_front + φ × G_rear

Where:
  G_eq = Equivalent irradiance (W/m²)
  G_front = Front irradiance (W/m²)
  G_rear = Rear irradiance (W/m²)
  φ = Bifaciality factor (typically φ_Isc for current-based)
```

**Alternative Equivalent Irradiance Models**:
```python
# IEC TS 60904-1-2:2024 recommended (Isc-based)
G_eq_isc = G_front + φ_Isc × G_rear

# Power-based equivalent irradiance
G_eq_pmax = G_front + φ_Pmax × G_rear

# Weighted average model (research-based)
G_eq_weighted = G_front + (w_isc × φ_Isc + w_pmax × φ_Pmax) × G_rear
```

**Measurement Sequence Options**:
- Fixed front, variable rear (rear irradiance sweep)
- Fixed ratio mode (constant G_rear/G_front)
- Equivalent STC mode (adjust G_front + G_rear to maintain G_eq = 1000 W/m²)

#### 3.1.3 Equivalent Irradiance Method (Method C)

**Purpose**: Report bifacial module performance at equivalent STC.

**Procedure**:
1. Define target equivalent irradiance: G_eq,target = 1000 W/m²
2. Select rear irradiance ratio: R = G_rear/G_front (e.g., 0.1, 0.2)
3. Calculate required irradiances:
   ```
   G_front = G_eq,target / (1 + φ × R)
   G_rear = R × G_front
   ```
4. Measure I-V curve at calculated conditions
5. No irradiance correction needed (measurement is at equivalent STC)

**Example for φ = 0.70, R = 0.15**:
```
G_front = 1000 / (1 + 0.70 × 0.15) = 1000 / 1.105 = 905.0 W/m²
G_rear = 0.15 × 905.0 = 135.7 W/m²
G_eq = 905.0 + 0.70 × 135.7 = 1000.0 W/m² ✓
```

### 3.2 Bifacial Reference Device Requirements

#### 3.2.1 Reference Cell Options

| Reference Type | Front Use | Rear Use | Calibration Required |
|----------------|-----------|----------|---------------------|
| WPVS monofacial | ✓ | ✓ (with spectral correction) | Separate front/rear |
| WPVS bifacial | ✓ | ✓ | Single calibration |
| Module reference | ✓ | Limited | Technology-matched |
| Spectroradiometer | ✓ | ✓ | Wavelength calibration |

#### 3.2.2 Spectral Considerations for Rear Side

The rear-side spectral distribution differs from front due to:
- Albedo spectral reflectance (ground/mounting structure)
- Module glass absorption/reflection
- Cell edge effects

**Rear Reference Spectral Mismatch Correction**:
```
M_rear = [∫E_sim,rear(λ)×SR_DUT(λ)dλ × ∫E_ref(λ)×SR_ref(λ)dλ] /
         [∫E_sim,rear(λ)×SR_ref(λ)dλ × ∫E_ref(λ)×SR_DUT(λ)dλ]
```

### 3.3 Bifacial Gain Calculations

#### 3.3.1 Bifacial Power Gain

```
BG_Pmax = (P_max,bifacial - P_max,front) / P_max,front × 100%

Where:
  P_max,bifacial = Power under dual illumination
  P_max,front = Power under front-only illumination at same G_front
```

#### 3.3.2 Energy-Based Bifacial Gain

```
BG_energy = ∫(P_bifacial(t) - P_mono(t))dt / ∫P_mono(t)dt × 100%
```

#### 3.3.3 Rear Irradiance Gain Coefficient

```
g_rear = (P_max(G_front, G_rear) - P_max(G_front, 0)) / (G_rear × A_module)
```

---

## 4. Enhanced Uncertainty Budget

### 4.1 Complete Bifacial Uncertainty Framework

#### 4.1.1 Category Structure (Extended)

```
1. Reference Device Uncertainty
   1.1 Calibration Uncertainty
       1.1.1 WPVS/Module calibration uncertainty
       1.1.2 Calibration laboratory traceability
       1.1.3 Reference device spectral response uncertainty
   1.2 Reference Device Stability
       1.2.1 Long-term drift
       1.2.2 Temperature dependence
       1.2.3 Non-linearity with irradiance
   1.3 Reference Device Positioning
       1.3.1 Position in test plane (front)
       1.3.2 Position in test plane (rear) [BIFACIAL]
       1.3.3 Angular alignment

2. Sun Simulator - Front Side
   2.1 Spatial Non-uniformity (Front)
       2.1.1 Uniformity classification
       2.1.2 Position-dependent irradiance variation
   2.2 Temporal Instability (Front)
       2.2.1 Short-term stability
       2.2.2 Flash-to-flash variation
       2.2.3 Lamp aging effects
   2.3 Spectral Mismatch (Front)
       2.3.1 Simulator spectrum vs AM1.5G
       2.3.2 Spectral mismatch correction uncertainty
       2.3.3 Module spectral response uncertainty

3. Sun Simulator - Rear Side [BIFACIAL]
   3.1 Spatial Non-uniformity (Rear)
       3.1.1 Rear uniformity classification
       3.1.2 Rear position-dependent variation
       3.1.3 Front-rear uniformity correlation
   3.2 Temporal Instability (Rear)
       3.2.1 Rear short-term stability
       3.2.2 Front-rear synchronization
   3.3 Spectral Mismatch (Rear)
       3.3.1 Rear simulator spectrum
       3.3.2 Rear spectral mismatch correction
   3.4 Rear Irradiance Uncertainty
       3.4.1 Rear irradiance measurement
       3.4.2 Rear irradiance control accuracy
       3.4.3 Parasitic front-to-rear reflection

4. Temperature Measurement
   4.1 Sensor Calibration
       4.1.1 Thermocouple/RTD uncertainty
       4.1.2 Calibration accuracy
   4.2 Temperature Uniformity
       4.2.1 Module temperature gradient (front)
       4.2.2 Module temperature gradient (rear) [BIFACIAL]
       4.2.3 Front-rear temperature difference [BIFACIAL]
   4.3 Temperature Correction
       4.3.1 IEC 60891 procedure uncertainty
       4.3.2 Temperature coefficient uncertainty

5. I-V Measurement
   5.1 Voltage Measurement
       5.1.1 Voltmeter calibration
       5.1.2 Contact resistance
       5.1.3 Cable resistance
   5.2 Current Measurement
       5.2.1 Ammeter/shunt calibration
       5.2.2 Current range selection
   5.3 Data Acquisition
       5.3.1 ADC resolution
       5.3.2 Sampling rate adequacy
       5.3.3 Curve fitting algorithm

6. Module Characteristics
   6.1 Module Variability
       6.1.1 Manufacturing tolerance
       6.1.2 Binning accuracy
   6.2 Module Behavior
       6.2.1 Hysteresis effects
       6.2.2 Stabilization time
       6.2.3 Light soaking requirements
   6.3 Bifacial-Specific [BIFACIAL]
       6.3.1 Cell-to-cell bifaciality variation
       6.3.2 Module edge effects
       6.3.3 Junction box shadowing

7. Bifaciality Factor Uncertainty [BIFACIAL]
   7.1 Bifaciality Factor Determination
       7.1.1 φ_Isc uncertainty
       7.1.2 φ_Voc uncertainty
       7.1.3 φ_Pmax uncertainty
       7.1.4 φ_FF uncertainty
   7.2 Bifaciality Factor Application
       7.2.1 Irradiance ratio dependency
       7.2.2 Temperature dependency of φ
       7.2.3 Spectral dependency of φ

8. Equivalent Irradiance Uncertainty [BIFACIAL]
   8.1 G_eq Calculation
       8.1.1 Front irradiance uncertainty contribution
       8.1.2 Rear irradiance uncertainty contribution
       8.1.3 Bifaciality factor uncertainty propagation
   8.2 Irradiance Correction
       8.2.1 Equivalent STC correction uncertainty
       8.2.2 Non-linear response effects

9. Environmental Conditions
   9.1 Ambient Conditions
       9.1.1 Ambient temperature
       9.1.2 Relative humidity
       9.1.3 Atmospheric pressure
   9.2 Albedo Effects [BIFACIAL]
       9.2.1 Test area albedo variation
       9.2.2 Spectral albedo uncertainty
       9.2.3 Angular albedo dependency

10. Measurement Procedure
    10.1 Repeatability
         10.1.1 Intra-laboratory repeatability
         10.1.2 Same-day measurement variation
    10.2 Reproducibility
         10.2.1 Inter-laboratory reproducibility
         10.2.2 Round robin results
    10.3 Operator Effects
         10.3.1 Module positioning (front)
         10.3.2 Module positioning (rear) [BIFACIAL]
         10.3.3 Contact application

11. Parasitic Effects [BIFACIAL]
    11.1 Optical Crosstalk
         11.1.1 Front-to-rear light leakage
         11.1.2 Rear-to-front reflection
         11.1.3 Frame/mounting reflections
    11.2 Electrical Effects
         11.2.1 Induced currents
         11.2.2 Ground loop effects
```

### 4.2 Bifacial-Specific Uncertainty Components

#### 4.2.1 Rear Side Irradiance Uncertainty

**Sources**:
1. Rear irradiance sensor calibration
2. Rear irradiance uniformity
3. Rear irradiance temporal stability
4. Rear reference device positioning
5. Parasitic front illumination

**Mathematical Model**:
```
u²(G_rear) = u²(G_rear,cal) + u²(G_rear,unif) + u²(G_rear,temp) +
             u²(G_rear,pos) + u²(G_rear,parasitic)
```

**Typical Values** (k=1):
| Component | Typical Value | Distribution |
|-----------|---------------|--------------|
| Calibration | 1.0-2.0% | Normal |
| Uniformity | 2.0-5.0% | Rectangular |
| Temporal | 0.3-1.0% | Rectangular |
| Positioning | 0.5-1.5% | Rectangular |
| Parasitic | 0.1-0.5% | Rectangular |

#### 4.2.2 Bifaciality Factor Uncertainty

**For φ_Isc**:
```
φ_Isc = I_sc,rear / I_sc,front

u(φ_Isc)/φ_Isc = √[(u(I_sc,rear)/I_sc,rear)² + (u(I_sc,front)/I_sc,front)²]
```

**For φ_Pmax**:
```
φ_Pmax = P_max,rear / P_max,front

u(φ_Pmax)/φ_Pmax = √[(u(P_max,rear)/P_max,rear)² + (u(P_max,front)/P_max,front)²]
```

**Correlation Considerations**:
- Front and rear measurements share some systematic uncertainties
- Correlation coefficient ρ affects combined uncertainty
- For same-device sequential measurements: ρ ≈ 0.3-0.7

**Correlated Uncertainty Model**:
```
u²(φ) = φ² × [(u_rear/X_rear)² + (u_front/X_front)² - 2ρ(u_rear/X_rear)(u_front/X_front)]
```

#### 4.2.3 Equivalent Irradiance Uncertainty

**Model**: G_eq = G_front + φ × G_rear

**Sensitivity Coefficients**:
```
c_Gfront = ∂G_eq/∂G_front = 1
c_Grear = ∂G_eq/∂G_rear = φ
c_φ = ∂G_eq/∂φ = G_rear
```

**Combined Uncertainty**:
```
u²(G_eq) = c²_Gfront × u²(G_front) + c²_Grear × u²(G_rear) + c²_φ × u²(φ)
         = u²(G_front) + φ² × u²(G_rear) + G²_rear × u²(φ)
```

**Relative Form**:
```
[u(G_eq)/G_eq]² = [G_front/G_eq]² × [u(G_front)/G_front]² +
                  [φ×G_rear/G_eq]² × [u(G_rear)/G_rear]² +
                  [G_rear/G_eq]² × [u(φ)]²
```

#### 4.2.4 Spectral Albedo Effects

**Albedo Spectral Reflectance Model**:
```
G_rear,eff(λ) = G_ground(λ) × ρ_albedo(λ) × F_view

Where:
  G_ground(λ) = Spectral irradiance reaching ground
  ρ_albedo(λ) = Spectral albedo reflectance
  F_view = View factor from module to ground
```

**Common Albedo Spectra** (Weighted Average Reflectance):
| Surface | ρ_avg | Spectral Shape | u(ρ) |
|---------|-------|----------------|------|
| White sand | 0.35-0.45 | Flat | 5% |
| Green grass | 0.15-0.25 | Red edge at 700nm | 10% |
| Concrete | 0.20-0.30 | Slightly blue | 8% |
| Snow | 0.80-0.95 | High UV | 3% |
| Dark soil | 0.05-0.15 | Brown/red | 15% |
| White roof coating | 0.70-0.85 | High NIR | 5% |

**Spectral Mismatch for Rear (Albedo-Modified)**:
```
M_rear,albedo = [∫E_ref(λ)×ρ(λ)×SR_DUT(λ)dλ × ∫E_sim(λ)×SR_ref(λ)dλ] /
                [∫E_ref(λ)×ρ(λ)×SR_ref(λ)dλ × ∫E_sim(λ)×SR_DUT(λ)dλ]
```

#### 4.2.5 Parasitic Reflections

**Sources of Parasitic Light**:
1. Test fixture reflections
2. Back-sheet of adjacent modules
3. Room/enclosure surfaces
4. Measurement equipment

**Quantification**:
```
G_parasitic = G_front × ρ_fixture × F_parasitic

Where:
  ρ_fixture = Fixture reflectance (typically 0.02-0.10)
  F_parasitic = Geometric view factor
```

**Uncertainty Contribution**:
```
u(G_rear,parasitic) = 0.5 × G_parasitic  (assume 50% uncertainty on parasitic)
```

#### 4.2.6 Bifacial Gain Uncertainty

**Bifacial Gain Model**:
```
BG = (P_bifacial - P_mono) / P_mono = P_bifacial/P_mono - 1
```

**Uncertainty**:
```
u(BG) = √[(∂BG/∂P_bi)² × u²(P_bi) + (∂BG/∂P_mono)² × u²(P_mono)]
      = √[(1/P_mono)² × u²(P_bi) + (P_bi/P²_mono)² × u²(P_mono)]
```

**Simplified Form** (assuming equal relative uncertainties):
```
u(BG) ≈ (1 + BG) × √2 × u_rel(P)
```

---

## 5. GUM Methodology Implementation

### 5.1 Measurement Model

#### 5.1.1 Monofacial Power Measurement

```
P_max,STC = P_max,meas × (G_STC/G_meas) × [1 + γ(T_STC - T_meas)] × M
```

#### 5.1.2 Bifacial Power Measurement (Equivalent Irradiance)

```
P_max,STC,bi = P_max,meas × (G_eq,STC/G_eq,meas) × [1 + γ(T_STC - T_meas)] × M_front × M_rear,eff

Where:
  G_eq,STC = 1000 W/m² (equivalent)
  G_eq,meas = G_front,meas + φ × G_rear,meas
  M_rear,eff = Effective rear spectral mismatch (albedo-dependent)
```

### 5.2 Sensitivity Coefficient Derivation

#### 5.2.1 Bifacial Power Model

Full model:
```
P = f(I_sc,front, I_sc,rear, V_oc, FF, G_front, G_rear, φ, T, γ, M_f, M_r)
```

Sensitivity coefficients:
```
c_Gfront = ∂P/∂G_front = P/G_eq × 1
c_Grear = ∂P/∂G_rear = P/G_eq × φ
c_φ = ∂P/∂φ = P/G_eq × G_rear
c_T = ∂P/∂T = P × γ
c_γ = ∂P/∂γ = P × (T_STC - T_meas)
c_M = ∂P/∂M = P/M
```

### 5.3 Uncertainty Propagation

#### 5.3.1 Law of Propagation of Uncertainty (GUM)

```
u²_c(y) = Σᵢ (∂f/∂xᵢ)² × u²(xᵢ) + 2 × Σᵢ Σⱼ>ᵢ (∂f/∂xᵢ)(∂f/∂xⱼ) × u(xᵢ,xⱼ)
```

#### 5.3.2 Combined Standard Uncertainty (Bifacial Power)

```
u²_c(P_bi) = c²_Gf × u²(G_f) + c²_Gr × u²(G_r) + c²_φ × u²(φ) +
             c²_T × u²(T) + c²_γ × u²(γ) + c²_Mf × u²(M_f) + c²_Mr × u²(M_r) +
             c²_ref × u²(ref) + c²_IV × u²(IV) + c²_rep × u²(rep) +
             2 × [covariance terms for correlated inputs]
```

### 5.4 Expanded Uncertainty

```
U = k × u_c(y)

Coverage factors:
  k = 2.0 for 95.45% confidence (normal distribution)
  k = 2.58 for 99% confidence
  k from t-distribution if ν_eff < 30 (Welch-Satterthwaite)
```

### 5.5 Effective Degrees of Freedom

```
ν_eff = u⁴_c(y) / Σᵢ[(cᵢ × uᵢ)⁴ / νᵢ]
```

---

## 6. IEC TS 60904-1-2:2024 Compliance

### 6.1 Measurement Conditions per Standard

#### 6.1.1 Standard Test Conditions (Bifacial)

| Parameter | Requirement | Tolerance |
|-----------|-------------|-----------|
| Front Irradiance | 1000 W/m² | ±2% |
| Rear Irradiance | 0 W/m² (single-sided) or specified | ±2% |
| Cell Temperature | 25°C | ±1°C |
| Spectral Distribution | IEC 60904-3 AM1.5G | Class A/B/C |
| Dark Side Irradiance | <0.1 W/m² (single-sided) | Verified |

#### 6.1.2 Bifacial Test Conditions (BTC)

| Parameter | Symbol | Value | Notes |
|-----------|--------|-------|-------|
| Equivalent Irradiance | G_eq | 1000 W/m² | G_f + φ×G_r |
| Rear Irradiance Ratio | R | 0.1 typical | Configurable |
| Cell Temperature | T_c | 25°C | Same as STC |

### 6.2 Simulator Requirements

#### 6.2.1 Classification per IEC 60904-9

| Class | Non-uniformity | Temporal Instability | Spectral Match |
|-------|----------------|---------------------|----------------|
| A | ≤2% | ≤2% | 0.75-1.25 each interval |
| B | ≤5% | ≤5% | 0.6-1.4 each interval |
| C | ≤10% | ≤10% | 0.4-2.0 each interval |

#### 6.2.2 Bifacial Simulator Additional Requirements

| Requirement | Specification |
|-------------|---------------|
| Rear uniformity | Same classification as front |
| Front-rear isolation | <0.1 W/m² crosstalk at dark side |
| Rear irradiance control | ±2% of setpoint |
| Rear spectral match | Class A, B, or characterized |

### 6.3 Measurement Procedure Compliance

#### 6.3.1 Single-Sided Measurement Procedure

```
1. Mount module with absorbing background (ρ < 0.05)
2. Stabilize temperature to 25°C ±1°C
3. Verify rear irradiance < 0.1 W/m²
4. Measure front-illuminated I-V curve
5. Record: I_sc, V_oc, P_max, FF, T_module
6. Rotate/remount with front to dark side
7. Repeat steps 2-5 for rear measurement
8. Calculate bifaciality factors
```

#### 6.3.2 Double-Sided Measurement Procedure

```
1. Configure front and rear irradiance settings
2. Verify irradiance levels with calibrated reference
3. Stabilize module temperature
4. Measure I-V curve under simultaneous illumination
5. Record: I_sc, V_oc, P_max, FF, G_front, G_rear, T
6. Calculate equivalent irradiance
7. Correct to equivalent STC if required
```

### 6.4 Reporting Requirements

#### 6.4.1 Required Report Elements (per IEC TS 60904-1-2:2024)

```
Module Identification:
- Manufacturer, model, serial number
- Cell technology, dimensions
- Bifacial construction details

Test Conditions:
- Front irradiance (W/m²) ± uncertainty
- Rear irradiance (W/m²) ± uncertainty (if applicable)
- Module temperature (°C) ± uncertainty
- Spectral condition (reference to IEC 60904-3)
- Simulator classification (front and rear)

Results:
- I_sc,front, V_oc,front, P_max,front, FF_front (with uncertainties)
- I_sc,rear, V_oc,rear, P_max,rear, FF_rear (with uncertainties)
- φ_Isc, φ_Voc, φ_Pmax, φ_FF (with uncertainties)
- Bifacial power at specified BTC (with uncertainty)

Uncertainty Statement:
- Combined standard uncertainty (k=1)
- Expanded uncertainty (k=2, 95% confidence)
- Coverage factor and method (GUM or Monte Carlo)
- Reference to JCGM 100:2008
```

---

## 7. API Specification

### 7.1 Core Classes

#### 7.1.1 BifacialModule

```python
@dataclass
class BifacialModule:
    """Represents a bifacial PV module with full characterization."""

    # Identification
    manufacturer: str
    model: str
    serial_number: str
    cell_technology: CellTechnology  # PERC, TOPCon, HJT, etc.

    # Physical dimensions
    length_mm: float
    width_mm: float
    cell_count: int
    cell_rows: int
    cell_columns: int

    # Front side parameters (STC)
    isc_front: float          # A
    voc_front: float          # V
    pmax_front: float         # W
    vmp_front: float          # V
    imp_front: float          # A
    ff_front: float           # ratio

    # Rear side parameters (STC)
    isc_rear: float           # A
    voc_rear: float           # V
    pmax_rear: float          # W
    vmp_rear: float           # V
    imp_rear: float           # A
    ff_rear: float            # ratio

    # Bifaciality factors
    phi_isc: float            # I_sc,rear / I_sc,front
    phi_voc: float            # V_oc,rear / V_oc,front
    phi_pmax: float           # P_max,rear / P_max,front
    phi_ff: float             # FF_rear / FF_front

    # Temperature coefficients
    alpha_isc: float          # %/°C
    beta_voc: float           # %/°C
    gamma_pmax: float         # %/°C

    # Spectral response (optional)
    spectral_response_front: Optional[SpectralResponse]
    spectral_response_rear: Optional[SpectralResponse]
```

#### 7.1.2 BifacialMeasurement

```python
@dataclass
class BifacialMeasurement:
    """Single bifacial measurement with complete uncertainty budget."""

    # Measurement mode
    mode: BifacialMode  # SINGLE_FRONT, SINGLE_REAR, DOUBLE_SIDED, EQUIVALENT

    # Irradiance conditions
    g_front: float            # W/m²
    g_rear: float             # W/m²
    g_equivalent: float       # W/m² (calculated)

    # Temperature
    t_module: float           # °C
    t_ambient: float          # °C

    # I-V parameters
    isc: float                # A
    voc: float                # V
    pmax: float               # W
    vmp: float                # V
    imp: float                # A
    ff: float                 # ratio

    # Full I-V curve (optional)
    iv_curve: Optional[IVCurve]

    # Uncertainties
    uncertainty_budget: BifacialUncertaintyBudget

    # Corrections applied
    temperature_corrected: bool
    irradiance_corrected: bool
    spectral_corrected: bool
```

#### 7.1.3 BifacialUncertaintyBudget

```python
@dataclass
class BifacialUncertaintyBudget:
    """Complete uncertainty budget for bifacial measurement."""

    # Reference device
    u_ref_cal: UncertaintyComponent
    u_ref_stability: UncertaintyComponent
    u_ref_position_front: UncertaintyComponent
    u_ref_position_rear: UncertaintyComponent

    # Simulator - Front
    u_uniformity_front: UncertaintyComponent
    u_temporal_front: UncertaintyComponent
    u_spectral_front: UncertaintyComponent

    # Simulator - Rear
    u_uniformity_rear: UncertaintyComponent
    u_temporal_rear: UncertaintyComponent
    u_spectral_rear: UncertaintyComponent
    u_irradiance_rear: UncertaintyComponent
    u_parasitic_rear: UncertaintyComponent

    # Temperature
    u_temp_sensor: UncertaintyComponent
    u_temp_uniformity: UncertaintyComponent
    u_temp_correction: UncertaintyComponent

    # I-V measurement
    u_voltage: UncertaintyComponent
    u_current: UncertaintyComponent
    u_data_acquisition: UncertaintyComponent

    # Bifaciality
    u_phi_isc: UncertaintyComponent
    u_phi_pmax: UncertaintyComponent
    u_phi_voc: UncertaintyComponent

    # Equivalent irradiance
    u_g_equivalent: UncertaintyComponent

    # Spectral albedo
    u_albedo_spectral: UncertaintyComponent

    # Repeatability/Reproducibility
    u_repeatability: UncertaintyComponent
    u_reproducibility: UncertaintyComponent

    # Methods
    def calculate_combined_uncertainty(self) -> CombinedUncertainty
    def get_dominant_contributors(self, n: int = 5) -> List[UncertaintyComponent]
    def export_to_dataframe(self) -> pd.DataFrame
```

### 7.2 Core Functions

#### 7.2.1 Bifaciality Factor Calculation

```python
def calculate_bifaciality_factors(
    front_measurement: BifacialMeasurement,
    rear_measurement: BifacialMeasurement,
    include_correlation: bool = True
) -> BifacialityFactors:
    """
    Calculate bifaciality factors from front and rear measurements.

    Parameters:
        front_measurement: Front-only illumination measurement
        rear_measurement: Rear-only illumination measurement
        include_correlation: Include correlated uncertainty components

    Returns:
        BifacialityFactors with φ_Isc, φ_Voc, φ_Pmax, φ_FF and uncertainties
    """
```

#### 7.2.2 Equivalent Irradiance Calculation

```python
def calculate_equivalent_irradiance(
    g_front: float,
    g_rear: float,
    phi: float,
    u_g_front: float,
    u_g_rear: float,
    u_phi: float,
    phi_type: Literal["isc", "pmax", "voc"] = "isc"
) -> Tuple[float, float]:
    """
    Calculate equivalent irradiance with uncertainty.

    Parameters:
        g_front: Front irradiance (W/m²)
        g_rear: Rear irradiance (W/m²)
        phi: Bifaciality factor
        u_g_front: Front irradiance uncertainty (W/m²)
        u_g_rear: Rear irradiance uncertainty (W/m²)
        u_phi: Bifaciality factor uncertainty (absolute)
        phi_type: Which bifaciality factor is used

    Returns:
        (G_eq, u_G_eq) - Equivalent irradiance and its uncertainty
    """
```

#### 7.2.3 Bifacial Power Uncertainty

```python
def calculate_bifacial_power_uncertainty(
    module: BifacialModule,
    measurement: BifacialMeasurement,
    uncertainty_budget: BifacialUncertaintyBudget,
    method: Literal["GUM", "Monte_Carlo"] = "GUM",
    coverage_factor: float = 2.0,
    n_samples: int = 100000
) -> PowerUncertaintyResult:
    """
    Calculate complete power uncertainty for bifacial module.

    Parameters:
        module: Bifacial module specification
        measurement: Measurement data and conditions
        uncertainty_budget: Complete uncertainty components
        method: Calculation method (GUM or Monte Carlo)
        coverage_factor: k-factor for expanded uncertainty
        n_samples: Number of MC samples (if method="Monte_Carlo")

    Returns:
        PowerUncertaintyResult with combined and expanded uncertainties
    """
```

---

## 8. Validation Requirements

### 8.1 Unit Test Requirements

| Test Category | Minimum Coverage |
|---------------|------------------|
| Bifaciality factor calculation | 100% |
| Equivalent irradiance | 100% |
| Uncertainty propagation | 100% |
| GUM sensitivity coefficients | 100% |
| Monte Carlo convergence | 95% |
| IEC compliance checks | 100% |

### 8.2 Integration Test Scenarios

#### 8.2.1 Monofacial Backward Compatibility

```python
def test_monofacial_compatibility():
    """Verify existing monofacial calculations unchanged."""
    # Use existing test cases
    # Results must match within 0.01%
```

#### 8.2.2 Bifacial Reference Cases

```python
# Reference case 1: HJT module with high bifaciality
module_hjt = BifacialModule(
    phi_isc=0.92,
    phi_pmax=0.85,
    pmax_front=400.0
)
# Expected uncertainty contributions must match published values

# Reference case 2: TOPCon module with medium bifaciality
module_topcon = BifacialModule(
    phi_isc=0.75,
    phi_pmax=0.70,
    pmax_front=380.0
)
```

### 8.3 Validation Against Published Data

| Source | Validation Criteria |
|--------|---------------------|
| NREL bifacial uncertainty | Within ±0.1% |
| IEC TS 60904-1-2 examples | Exact match |
| Round-robin results | Within reported U |

---

## Appendix A: Mathematical Derivations

### A.1 Equivalent Irradiance Uncertainty Derivation

Starting from:
```
G_eq = G_front + φ × G_rear
```

Applying GUM:
```
u²(G_eq) = (∂G_eq/∂G_f)² × u²(G_f) + (∂G_eq/∂G_r)² × u²(G_r) + (∂G_eq/∂φ)² × u²(φ)
```

Partial derivatives:
```
∂G_eq/∂G_f = 1
∂G_eq/∂G_r = φ
∂G_eq/∂φ = G_r
```

Therefore:
```
u²(G_eq) = u²(G_f) + φ² × u²(G_r) + G_r² × u²(φ)
```

Relative form:
```
[u(G_eq)/G_eq]² = [1/G_eq]² × u²(G_f) + [φ/G_eq]² × u²(G_r) + [G_r/G_eq]² × u²(φ)
```

### A.2 Bifaciality Factor Uncertainty with Correlation

For correlated measurements:
```
u²(φ) = φ² × {[u(X_r)/X_r]² + [u(X_f)/X_f]² - 2ρ × [u(X_r)/X_r] × [u(X_f)/X_f]}
```

Where ρ is the correlation coefficient between front and rear measurements.

### A.3 Temperature Coefficient Uncertainty Impact

```
∂P/∂γ = P_STC × (T_meas - T_STC)
u(P)_γ = P_STC × (T_meas - T_STC) × u(γ)
```

---

## Appendix B: Default Uncertainty Values

### B.1 Typical Laboratory Values

| Component | Type A | Type B | Typical Total (k=1) |
|-----------|--------|--------|---------------------|
| Reference calibration | - | 1.0% | 1.0% |
| Reference drift | 0.3% | - | 0.3% |
| Front uniformity | - | 1.5% | 1.5% |
| Rear uniformity | - | 2.5% | 2.5% |
| Temporal instability | - | 0.3% | 0.3% |
| Spectral mismatch (front) | - | 0.8% | 0.8% |
| Spectral mismatch (rear) | - | 1.5% | 1.5% |
| Temperature sensor | - | 0.5°C | 0.5°C |
| Temperature correction | - | 0.3% | 0.3% |
| I-V measurement | - | 0.3% | 0.3% |
| Bifaciality factor | 1.0% | 0.5% | 1.1% |
| Repeatability | 0.3% | - | 0.3% |

### B.2 Total Uncertainty Estimates

| Measurement Type | Combined (k=1) | Expanded (k=2) |
|------------------|----------------|----------------|
| Front STC power | 1.8-2.5% | 3.6-5.0% |
| Rear STC power | 2.5-3.5% | 5.0-7.0% |
| Bifaciality factor | 2.0-3.0% | 4.0-6.0% |
| Equivalent STC power | 2.2-3.0% | 4.4-6.0% |

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 2.0 | 2024-01 | Universal Solar Framework Team | Initial bifacial specification |

---

*This specification is designed for implementation in the Universal Solar Simulator Uncertainty Framework and complies with IEC TS 60904-1-2:2024 and JCGM 100:2008 (GUM).*
