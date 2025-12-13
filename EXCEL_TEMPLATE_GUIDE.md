# Excel Template Guide for Bifacial PV Uncertainty Analysis

## Universal Solar Simulator Uncertainty Framework v2.0

This guide describes the enhanced Excel template structure for bifacial PV module uncertainty analysis per IEC TS 60904-1-2:2024.

---

## Table of Contents

1. [Template Overview](#1-template-overview)
2. [Sheet Structure](#2-sheet-structure)
3. [Input Data Format](#3-input-data-format)
4. [Bifacial-Specific Inputs](#4-bifacial-specific-inputs)
5. [Uncertainty Budget Sheet](#5-uncertainty-budget-sheet)
6. [Results and Reporting](#6-results-and-reporting)
7. [Data Validation Rules](#7-data-validation-rules)
8. [Examples](#8-examples)

---

## 1. Template Overview

### 1.1 Purpose

The Excel template provides a structured format for:
- Importing measurement data from various solar simulators
- Defining uncertainty components for all sources
- Calculating combined and expanded uncertainties
- Generating ISO 17025 compliant reports
- Supporting both monofacial and bifacial measurements

### 1.2 Supported Simulators

| Manufacturer | Models | Bifacial Support |
|--------------|--------|------------------|
| PASAN | HighLIGHT LED, HighLIGHT BIFACIAL, cetisPV-CTL | Yes |
| Spire/Atonometrics | 5600SLP, 4600, BiFi-1000 | Yes |
| Halm/EETS | cetisPV-BI, flasher III | Yes |
| Meyer Burger | LOANA, PSS-30 | Yes |
| Wavelabs | Avalon Nexun, Avalon Bifacial | Yes |
| Eternal Sun | SLP-150, SLP-BiFi | Yes |
| Custom | User-defined | Yes |

### 1.3 Measurement Types

- STC (Standard Test Conditions)
- Bifacial Single-Sided (Front)
- Bifacial Single-Sided (Rear)
- Bifacial Double-Sided Simultaneous
- Bifacial Equivalent Irradiance
- NMOT (Nominal Module Operating Temperature)
- Low Irradiance Performance
- Temperature Coefficients

---

## 2. Sheet Structure

### 2.1 Required Sheets

| Sheet Name | Description | Required |
|------------|-------------|----------|
| `Module_Info` | Module identification and specifications | Yes |
| `Measurement_Data` | I-V measurement results | Yes |
| `Uncertainty_Budget` | Complete uncertainty components | Yes |
| `Simulator_Config` | Simulator specifications | Yes |
| `Reference_Device` | Reference cell calibration | Yes |
| `Bifacial_Data` | Bifacial-specific measurements | For bifacial |
| `Results` | Calculated results summary | Generated |
| `Report` | ISO 17025 report content | Generated |

### 2.2 Sheet Dependencies

```
Module_Info ──┬──> Measurement_Data ──┬──> Results
              │                       │
Simulator_Config ──┘                  │
              │                       │
Reference_Device ─> Uncertainty_Budget ┘
              │
Bifacial_Data ─┘ (if bifacial)
```

---

## 3. Input Data Format

### 3.1 Module_Info Sheet

| Row | Column A | Column B | Column C | Description |
|-----|----------|----------|----------|-------------|
| 1 | `Module_ID` | | | Section header |
| 2 | Manufacturer | [Text] | | Module manufacturer |
| 3 | Model | [Text] | | Model number |
| 4 | Serial_Number | [Text] | | Unique serial |
| 5 | | | | |
| 6 | `Physical_Specs` | | | Section header |
| 7 | Length_mm | [Number] | mm | Module length |
| 8 | Width_mm | [Number] | mm | Module width |
| 9 | Cell_Count | [Integer] | | Number of cells |
| 10 | Cell_Technology | [Dropdown] | | PERC/TOPCon/HJT/etc. |
| 11 | Is_Bifacial | [Yes/No] | | Bifacial module? |
| 12 | | | | |
| 13 | `Datasheet_Values` | | | Section header |
| 14 | Pmax_W | [Number] | W | Nameplate power |
| 15 | Voc_V | [Number] | V | Open-circuit voltage |
| 16 | Isc_A | [Number] | A | Short-circuit current |
| 17 | Vmp_V | [Number] | V | Maximum power voltage |
| 18 | Imp_A | [Number] | A | Maximum power current |
| 19 | | | | |
| 20 | `Temp_Coefficients` | | | Section header |
| 21 | Alpha_Isc | [Number] | %/°C | Isc temp coefficient |
| 22 | Beta_Voc | [Number] | %/°C | Voc temp coefficient |
| 23 | Gamma_Pmax | [Number] | %/°C | Pmax temp coefficient |

### 3.2 Measurement_Data Sheet

| Column | Header | Format | Description |
|--------|--------|--------|-------------|
| A | Measurement_ID | Text | Unique measurement identifier |
| B | Timestamp | DateTime | Measurement date/time |
| C | Side | Front/Rear | Illuminated side |
| D | G_Front_Wm2 | Number | Front irradiance (W/m²) |
| E | G_Rear_Wm2 | Number | Rear irradiance (W/m²) |
| F | T_Module_C | Number | Module temperature (°C) |
| G | T_Ambient_C | Number | Ambient temperature (°C) |
| H | Isc_A | Number | Measured Isc |
| I | Voc_V | Number | Measured Voc |
| J | Pmax_W | Number | Measured Pmax |
| K | Vmp_V | Number | Measured Vmp |
| L | Imp_A | Number | Measured Imp |
| M | FF | Number | Fill factor (0-1) |
| N | Spectral_Mismatch | Number | M factor (typically 0.98-1.02) |
| O | Notes | Text | Measurement notes |

---

## 4. Bifacial-Specific Inputs

### 4.1 Bifacial_Data Sheet

| Row | Column A | Column B | Column C | Column D | Description |
|-----|----------|----------|----------|----------|-------------|
| 1 | `Bifacial_Mode` | | | | Section header |
| 2 | Mode | [Dropdown] | | | Single_Front/Single_Rear/Double_Sided/Equivalent |
| 3 | | | | | |
| 4 | `Front_Side_STC` | | | | Section header |
| 5 | Isc_Front | [Number] | A | | Front Isc at STC |
| 6 | Voc_Front | [Number] | V | | Front Voc at STC |
| 7 | Pmax_Front | [Number] | W | | Front Pmax at STC |
| 8 | FF_Front | [Number] | | | Front fill factor |
| 9 | | | | | |
| 10 | `Rear_Side_STC` | | | | Section header |
| 11 | Isc_Rear | [Number] | A | | Rear Isc at STC |
| 12 | Voc_Rear | [Number] | V | | Rear Voc at STC |
| 13 | Pmax_Rear | [Number] | W | | Rear Pmax at STC |
| 14 | FF_Rear | [Number] | | | Rear fill factor |
| 15 | | | | | |
| 16 | `Bifaciality_Factors` | | | | Section header |
| 17 | Phi_Isc | [Calculated] | | | Isc_Rear / Isc_Front |
| 18 | Phi_Voc | [Calculated] | | | Voc_Rear / Voc_Front |
| 19 | Phi_Pmax | [Calculated] | | | Pmax_Rear / Pmax_Front |
| 20 | Phi_FF | [Calculated] | | | FF_Rear / FF_Front |
| 21 | | | | | |
| 22 | `Equivalent_Irradiance` | | | | Section header |
| 23 | G_Front_Target | [Number] | W/m² | | Front irradiance setpoint |
| 24 | G_Rear_Target | [Number] | W/m² | | Rear irradiance setpoint |
| 25 | Rear_Ratio_R | [Number] | | | G_Rear / G_Front |
| 26 | G_Equivalent | [Calculated] | W/m² | | G_Front + φ × G_Rear |
| 27 | | | | | |
| 28 | `Uncertainties` | | | | Section header |
| 29 | u_Phi_Isc | [Number] | % | k=1 | Uncertainty on φ_Isc |
| 30 | u_Phi_Pmax | [Number] | % | k=1 | Uncertainty on φ_Pmax |
| 31 | u_G_Rear | [Number] | % | k=1 | Rear irradiance uncertainty |

### 4.2 Bifaciality Factor Calculation Formulas

In Excel, use these formulas:

```excel
' Cell B17 (Phi_Isc):
=IF(B5>0, B11/B5, 0)

' Cell B18 (Phi_Voc):
=IF(B6>0, B12/B6, 0)

' Cell B19 (Phi_Pmax):
=IF(B7>0, B13/B7, 0)

' Cell B26 (G_Equivalent):
=B23 + B17 * B24

' Cell B25 (Rear_Ratio):
=IF(B23>0, B24/B23, 0)
```

---

## 5. Uncertainty Budget Sheet

### 5.1 Structure

| Column | Header | Description |
|--------|--------|-------------|
| A | Category_ID | Hierarchical ID (e.g., 1.1.1) |
| B | Category | Main category name |
| C | Subcategory | Subcategory name |
| D | Factor | Uncertainty factor name |
| E | Value | Specified/measured value |
| F | Uncertainty | Uncertainty value |
| G | Unit | Physical unit |
| H | Distribution | Normal/Rectangular/Triangular |
| I | Divisor | Distribution divisor |
| J | Standard_U | Standard uncertainty (calculated) |
| K | Sensitivity | Sensitivity coefficient |
| L | Variance_Contrib | (K×J)² |
| M | Enabled | TRUE/FALSE |
| N | Type | Type_A/Type_B |
| O | DOF | Degrees of freedom |
| P | Source | Source of value |

### 5.2 Complete Category Structure

```
1. Reference Device
   1.1 Calibration Uncertainty
       1.1.1 WPVS/Module calibration
       1.1.2 Calibration laboratory traceability
   1.2 Reference Device Stability
       1.2.1 Long-term drift
       1.2.2 Temperature dependence
   1.3 Reference Device Positioning
       1.3.1 Position in test plane (front)
       1.3.2 Position in test plane (rear) [BIFACIAL]

2. Sun Simulator - Front Side
   2.1 Spatial Non-uniformity (Front)
       2.1.1 Uniformity classification
       2.1.2 Position-dependent variation
   2.2 Temporal Instability (Front)
       2.2.1 Short-term stability
       2.2.2 Flash-to-flash variation
   2.3 Spectral Mismatch (Front)
       2.3.1 Simulator vs AM1.5G spectrum
       2.3.2 Spectral mismatch correction

3. Sun Simulator - Rear Side [BIFACIAL]
   3.1 Spatial Non-uniformity (Rear)
       3.1.1 Rear uniformity
       3.1.2 Front-rear correlation
   3.2 Temporal Instability (Rear)
       3.2.1 Rear short-term stability
       3.2.2 Front-rear synchronization
   3.3 Spectral Mismatch (Rear)
       3.3.1 Rear simulator spectrum
       3.3.2 Rear spectral correction
   3.4 Rear Irradiance Uncertainty
       3.4.1 Rear irradiance measurement
       3.4.2 Parasitic front-to-rear light

4. Temperature Measurement
   4.1 Sensor Calibration
       4.1.1 Thermocouple/RTD uncertainty
   4.2 Temperature Uniformity
       4.2.1 Module temperature gradient
       4.2.2 Front-rear gradient [BIFACIAL]
   4.3 Temperature Correction
       4.3.1 IEC 60891 procedure
       4.3.2 Temperature coefficient uncertainty

5. I-V Measurement
   5.1 Voltage Measurement
       5.1.1 Voltmeter calibration
       5.1.2 Contact resistance
   5.2 Current Measurement
       5.2.1 Ammeter/shunt calibration
   5.3 Data Acquisition
       5.3.1 ADC resolution
       5.3.2 Curve fitting

6. Module Characteristics
   6.1 Module Variability
       6.1.1 Manufacturing tolerance
   6.2 Module Behavior
       6.2.1 Hysteresis effects
       6.2.2 Stabilization time

7. Bifaciality Factor [BIFACIAL]
   7.1 Bifaciality Factor Determination
       7.1.1 φ_Isc uncertainty
       7.1.2 φ_Pmax uncertainty
   7.2 Bifaciality Factor Application
       7.2.1 Irradiance ratio dependency
       7.2.2 Temperature dependency

8. Equivalent Irradiance [BIFACIAL]
   8.1 G_eq Calculation
       8.1.1 Combined G_eq uncertainty
   8.2 Irradiance Correction
       8.2.1 STC correction uncertainty

9. Environmental Conditions
   9.1 Ambient Conditions
       9.1.1 Ambient temperature
       9.1.2 Humidity
   9.2 Albedo Effects [BIFACIAL]
       9.2.1 Spectral albedo

10. Measurement Procedure
    10.1 Repeatability
         10.1.1 Intra-laboratory repeatability
    10.2 Reproducibility
         10.2.1 Inter-laboratory reproducibility
```

### 5.3 Standard Uncertainty Formulas

```excel
' Column J (Standard_U) - depends on distribution in Column H:
=IF(H2="Normal", F2,
 IF(H2="Rectangular", F2/SQRT(3),
 IF(H2="Triangular", F2/SQRT(6),
 IF(H2="U_Shaped", F2/SQRT(2), F2))))

' Column L (Variance_Contrib):
=IF(M2=TRUE, (K2*J2)^2, 0)

' Combined Standard Uncertainty (summary cell):
=SQRT(SUMIF(M:M, TRUE, L:L))

' Expanded Uncertainty (k=2):
=2 * [Combined_U_Cell]
```

---

## 6. Results and Reporting

### 6.1 Results Sheet Structure

| Section | Content |
|---------|---------|
| Module Summary | ID, manufacturer, model, technology |
| Test Conditions | Irradiance, temperature, spectral |
| Measured Values | Isc, Voc, Pmax, FF, efficiency |
| Bifacial Results | φ factors, G_eq, bifacial gain |
| Uncertainty Summary | Combined (k=1), Expanded (k=2) |
| Dominant Contributors | Top 5 uncertainty sources |
| Compliance Status | IEC compliance checks |

### 6.2 Report Content Elements

Per ISO 17025:2017 requirements:

1. **Header Information**
   - Laboratory name and accreditation
   - Report number and date
   - Client information

2. **Module Information**
   - Complete module identification
   - Cell technology and construction

3. **Test Equipment**
   - Simulator model and classification
   - Reference device calibration
   - I-V tracer specifications

4. **Measurement Conditions**
   - Irradiance (front and rear for bifacial)
   - Temperature
   - Spectral distribution

5. **Results with Uncertainty**
   - All electrical parameters
   - Bifaciality factors (if applicable)
   - Expanded uncertainty (k=2, 95%)

6. **Uncertainty Budget**
   - Complete component list
   - Contribution percentages
   - Visualization (bar chart)

7. **Compliance Statement**
   - Standards referenced
   - Compliance status

8. **Signatures**
   - Preparer, Reviewer, Approver

---

## 7. Data Validation Rules

### 7.1 Physical Value Limits

| Parameter | Min | Max | Typical |
|-----------|-----|-----|---------|
| Pmax (W) | 50 | 800 | 300-500 |
| Voc (V) | 20 | 60 | 35-50 |
| Isc (A) | 5 | 20 | 8-12 |
| FF | 0.60 | 0.90 | 0.75-0.85 |
| Efficiency (%) | 10 | 30 | 18-24 |
| φ_Isc | 0.50 | 1.00 | 0.70-0.95 |
| φ_Pmax | 0.50 | 0.95 | 0.65-0.90 |
| G_front (W/m²) | 800 | 1100 | 1000 |
| G_rear (W/m²) | 0 | 500 | 100-200 |
| Temperature (°C) | 20 | 30 | 25 |

### 7.2 Excel Validation Formulas

```excel
' Irradiance validation (Data Validation - Custom):
=AND(D2>=800, D2<=1100)

' Temperature validation:
=AND(F2>=20, F2<=30)

' Bifaciality factor validation:
=AND(B17>=0.5, B17<=1.0)

' Fill Factor validation:
=AND(M2>=0.6, M2<=0.9)

' I-V ratio validation (Vmp/Voc):
=AND(K2/I2>=0.7, K2/I2<=0.9)

' I-V ratio validation (Imp/Isc):
=AND(L2/H2>=0.85, L2/H2<=0.99)
```

### 7.3 Conditional Formatting

```excel
' Highlight out-of-range values (red):
Formula: =OR(D2<800, D2>1100)

' Highlight warnings (yellow):
Formula: =AND(D2>=950, D2<=1050)

' Highlight compliant (green):
Formula: =AND(D2>=980, D2<=1020)
```

---

## 8. Examples

### 8.1 Example: TOPCon Bifacial Module

**Module_Info:**
```
Manufacturer: Example Solar
Model: TOPCon-400-BF
Serial: ESM-2024-001234
Cell_Technology: TOPCon
Is_Bifacial: Yes
Pmax: 400 W
```

**Bifacial_Data:**
```
Mode: Double_Sided
Isc_Front: 10.5 A
Voc_Front: 45.0 V
Pmax_Front: 400.0 W
Isc_Rear: 7.87 A
Voc_Rear: 44.8 V
Pmax_Rear: 280.0 W
Phi_Isc: 0.75
Phi_Pmax: 0.70
G_Front: 900 W/m²
G_Rear: 135 W/m²
G_Equivalent: 1001.25 W/m²
```

**Key Uncertainties:**
```
Reference Calibration: 1.0% (k=2)
Front Uniformity: 1.5% (rectangular)
Rear Uniformity: 2.5% (rectangular)
φ_Isc: 1.5%
φ_Pmax: 2.0%
Repeatability: 0.3%
```

**Results:**
```
Combined Uncertainty (k=1): 2.3%
Expanded Uncertainty (k=2): 4.6%
Pmax_STC: 400.0 ± 18.4 W
Bifacial Gain (@ R=0.15): 10.5 ± 0.8%
```

### 8.2 Example: Quick Uncertainty Check

For quick validation, use these typical combined uncertainties:

| Measurement Type | Typical u_c (k=1) | Typical U (k=2) |
|------------------|-------------------|-----------------|
| Monofacial STC | 1.8-2.5% | 3.6-5.0% |
| Bifacial Front-Only | 2.0-2.8% | 4.0-5.6% |
| Bifacial Rear-Only | 2.5-3.5% | 5.0-7.0% |
| Bifacial Double-Sided | 2.5-3.2% | 5.0-6.4% |
| Bifaciality Factor | 2.0-3.0% | 4.0-6.0% |

---

## Appendix A: Template Download

The complete Excel template is available in the repository:
- `templates/bifacial_uncertainty_template.xlsx`
- `templates/monofacial_uncertainty_template.xlsx`

## Appendix B: Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | 2024-01 | Added bifacial support per IEC TS 60904-1-2:2024 |
| 1.0 | 2023-06 | Initial monofacial template |

---

*This template guide is part of the Universal Solar Simulator Uncertainty Framework v2.0*
