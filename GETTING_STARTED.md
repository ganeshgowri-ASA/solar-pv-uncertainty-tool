# 🚀 Getting Started with PV Uncertainty Tool - Professional Edition

Welcome to the **PV Measurement Uncertainty Tool - Professional Edition**! This guide will help you get started quickly.

---

## 📦 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd solar-pv-uncertainty-tool

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_app_pro.py
```

The application will open in your browser at `http://localhost:8501`

---

## 🎯 Your First Uncertainty Analysis (5 Minutes)

Follow this quick tutorial to perform your first STC measurement uncertainty analysis:

### Step 1: Module Configuration (Tab 1)

1. **Select PV Technology:** Choose from PERC, TOPCon, HJT, Perovskite, etc.
2. **Enter Nameplate Power:** e.g., 400 W
3. **Temperature Coefficients:** Use the pre-filled typical values or enter your own

**Example:**
- Technology: TOPCon
- Nameplate Power: 450 W
- γ_Pmax: -0.30 %/°C

### Step 2: Sun Simulator Configuration (Tab 2)

1. **Select Simulator:** Choose your equipment (e.g., "Spire 5600SLP")
2. **Review Parameters:** Check pre-filled uniformity and temporal stability
3. **Adjust if Needed:** Enter actual values from your classification certificate

**Example:**
- Simulator: Spire 5600SLP
- Uniformity: 2.0%
- Temporal: 0.5%

### Step 3: Reference Device (Tab 3)

1. **Select Reference Type:** WPVS Cell or Reference Module
2. **Choose Calibration Lab:** Select from list (e.g., "NREL", "PTB", "Fraunhofer ISE")
3. **Enter Calibration Data:** Uncertainty from calibration certificate

**Example:**
- Type: Reference Module
- Lab: Fraunhofer ISE CalLab
- Calibration Uncertainty: 1.3%

### Step 4: Measurement Data (Tab 4)

1. **Enter Measured Values:**
   - Voc: e.g., 48.5 V
   - Isc: e.g., 11.2 A
   - Vmp: e.g., 40.8 V
   - Imp: e.g., 11.0 A

2. **Review Auto-Calculated:**
   - Pmax = Vmp × Imp
   - Fill Factor
   - I-V Ratio Validation

**Example:**
- Voc: 48.5 V
- Isc: 11.2 A
- Vmp: 40.8 V
- Imp: 11.0 A
- → Pmax: 448.8 W ✅

### Step 5: Uncertainty Analysis (Tab 5)

1. **Review Fishbone Diagram:** See the 7 main uncertainty categories
2. **Adjust Uncertainty Factors:** Fine-tune values if needed (defaults are provided)
3. **Click "Calculate Combined Uncertainty"**

**Wait for calculation... ⏱️**

### Step 6: View Results (Tab 6)

**You'll see:**
- Combined Standard Uncertainty (k=1)
- Expanded Uncertainty (k=2) for 95% confidence
- Absolute Uncertainty in Watts
- 95% Confidence Interval
- Detailed Uncertainty Budget Table
- Visualization Charts (Bar, Pie, Pareto)

**Example Result:**
```
Pmax = 448.8 W ± 10.2 W (k=2)
95% CI: [438.6 W, 459.0 W]
Relative Uncertainty: 2.27%
```

### Step 7: Financial Impact (Tab 7) - OPTIONAL

**Scenario 1: Fresh Module Pricing**
1. Enter price per watt (e.g., $0.22/W)
2. Calculate price impact
3. See financial risk in dollars

**Scenario 2: Warranty Claim**
1. Enter original nameplate (e.g., 450 W)
2. Set warranty threshold (e.g., 90%)
3. Get claim validity assessment

**Scenario 3: Project Financing**
1. Enter plant size (e.g., 100 MW)
2. Energy price (e.g., $0.10/kWh)
3. Get NPV/ROI with uncertainty

### Step 8: Generate Professional Reports (Tab 8)

**Now Available!** Professional ISO 17025 compliant PDF and Excel reports

1. **Configure Report Details:**
   - Enter laboratory name and address
   - Add accreditation number (if applicable)
   - Specify report and document numbers
   - Enter client name
   - Set test and report dates

2. **Upload Company Logo (Optional):**
   - Upload your lab's logo (PNG/JPG)
   - Recommended size: 150x50 pixels

3. **Enter Signatures:**
   - Prepared by: Test Engineer name and title
   - Reviewed by: Senior Engineer name and title
   - Approved by: Technical Manager name and title

4. **Configure Report Options:**
   - Choose to include financial analysis
   - Include/exclude uncertainty charts
   - Include/exclude detailed equipment info
   - Add additional comments or notes

5. **Generate Reports:**
   - Click "Generate PDF Report" for ISO 17025 PDF
   - Click "Generate Excel Report" for detailed Excel workbook
   - Download reports directly from the interface

**Example:**
- Laboratory: "Solar Testing Lab Inc."
- Report Number: "UNC-20251022-1430"
- Client: "Solar Module Manufacturing Co."
- Prepared by: "John Doe, Test Engineer"
- → Generate both PDF and Excel reports with one click!

---

## 💡 Tips for Best Results

### 1. **Use Actual Equipment Data**
- Upload your simulator classification certificate (PDF)
- Upload reference device calibration certificate (PDF)
- Use actual uncertainty values from your lab

### 2. **Data Validation**
- Pay attention to warning messages
- Ensure Vmp/Voc ratio is between 0.70-0.90
- Ensure Imp/Isc ratio is between 0.85-0.99

### 3. **Understand Uncertainty Contributors**
- Check the Pareto chart in Tab 6
- Top 3-5 contributors typically account for 80%+ of total uncertainty
- Focus improvement efforts on these factors

### 4. **Save Your Work**
- Download uncertainty budget as CSV
- Download financial analysis results
- Keep configuration for future measurements

---

## 📊 Understanding Your Results

### Combined Standard Uncertainty
- This is the "k=1" uncertainty
- Represents ~68% confidence level
- Calculated using GUM methodology (root sum of squares)

### Expanded Uncertainty (k=2)
- Standard uncertainty × 2
- Represents ~95% confidence level
- **This is what you report to clients**

### Confidence Interval
- The range where true value likely lies
- At 95% confidence: approximately 19 out of 20 measurements will fall within this range

### Uncertainty Budget
- Shows contribution of each factor
- Percentage contribution helps prioritize improvements
- Sensitivity coefficient shows impact of each factor

---

## 🎓 Common Scenarios

### Scenario A: Commercial PV Testing Lab

**Your Setup:**
- AAA-class LED simulator (Pasan HighLIGHT LED)
- Secondary reference module from TÜV Rheinland
- Typical testing: 60-cell modules, 300-550 W

**Expected Uncertainty:** 2-3% (expanded, k=2)

**Main Contributors:**
1. Reference module calibration (~1.5-2%)
2. Simulator spatial uniformity (~2%)
3. Spectral mismatch (~0.8%)

### Scenario B: Research Laboratory

**Your Setup:**
- AAA+ class steady-state simulator (Wavelabs)
- Primary reference cell from PTB
- High-precision instrumentation

**Expected Uncertainty:** 1.5-2.5% (expanded, k=2)

**Main Contributors:**
1. Spectral mismatch (especially for new technologies)
2. Temperature measurement and correction
3. Module hysteresis effects

### Scenario C: Manufacturer QC Line

**Your Setup:**
- Flash-based simulator (multiple units)
- Secondary reference modules
- High throughput requirements

**Expected Uncertainty:** 2.5-4% (expanded, k=2)

**Main Contributors:**
1. Temporal instability (flash-to-flash)
2. Reference module drift (frequent recalibration needed)
3. Repeatability (operator effects)

---

## 🔧 Troubleshooting

### Issue: "Please complete Section 5 first"
**Solution:** Go to Tab 5 (Uncertainty) and click "Calculate Combined Uncertainty"

### Issue: I-V Ratio Warnings
**Solution:**
- Double-check measured values
- Ensure module was properly stabilized
- Check for series resistance issues
- Verify temperature is correct

### Issue: Very High Uncertainty (>5%)
**Possible Causes:**
- Check reference device calibration date (drift?)
- Review simulator classification (recent calibration?)
- Verify all uncertainty values are realistic
- Check if reproducibility/ILC data is too conservative

### Issue: Very Low Uncertainty (<1%)
**Warning:** This is unusually low for PV measurements
- Verify you're not underestimating uncertainties
- Include all relevant factors
- Check that Type B uncertainties are included

---

## 📚 Learning Resources

### Standards & Guidelines
- **JCGM 100:2008 (GUM)** - Guide to Expression of Uncertainty in Measurement
- **JCGM 101:2008** - GUM Supplement 1 (Monte Carlo)
- **IEC 60904-1:2020** - PV Current-Voltage Characteristics
- **IEC 61215** - Terrestrial PV Modules - Design Qualification
- **IEC 61724-1:2021** - PV System Performance Monitoring

### Key Concepts

**Type A Uncertainty:** Statistical (from repeated measurements)
- Repeatability
- Reproducibility (ILC/Round Robin)

**Type B Uncertainty:** Other sources (not statistical)
- Calibration certificates
- Manufacturer specifications
- Educated assumptions

**Sensitivity Coefficient:** How much output changes per unit input change
- For power: ∂P/∂G ≈ P/G (power vs. irradiance)
- For temperature: ∂P/∂T = P × γ (temperature coefficient)

**Distribution Types:**
- **Normal:** Calibrated instruments, repeated measurements
- **Rectangular (Uniform):** Manufacturer specs with limits
- **Triangular:** Limits with most likely value in middle

---

## 🎯 Next Steps

### After Your First Analysis

1. **Compare with Historical Data**
   - Do results match previous measurements?
   - Is uncertainty consistent with your lab's capability?

2. **Validate Against ILC Results**
   - Compare with Inter-Laboratory Comparison data
   - Adjust reproducibility uncertainty if needed

3. **Optimize Your Lab**
   - Identify top uncertainty contributors
   - Plan improvements (better reference, simulator upgrade, etc.)
   - Track uncertainty reduction over time

4. **Integrate into Workflow**
   - Use for all commercial testing
   - Include in test reports
   - Train staff on interpretation

5. **Explore Advanced Features**
   - Try different measurement types (Low Irradiance, Temperature Coefficients)
   - Run financial impact scenarios
   - Compare different technologies

---

## 💰 Financial Impact Analysis - Quick Guide

### Fresh Module Sales
**Question:** How does measurement uncertainty affect module pricing?

**Example:**
- Module: 450 W
- Uncertainty: ±10 W (k=2)
- Price: $0.22/W

**Result:**
- Nominal Price: $99.00
- Price Uncertainty: ±$2.20
- Buyer/Seller Risk: $2.20 each

**Interpretation:** Measurement uncertainty creates a $4.40 price range

### Warranty Claims
**Question:** Is the module performing below warranty threshold?

**Example:**
- Nameplate: 450 W (when new)
- Measured: 395 W (after 10 years)
- Uncertainty: ±10 W
- Warranty: 90% (405 W threshold)

**Result:**
- 95% CI: [385 W, 405 W]
- Decision: UNCERTAIN (overlaps threshold)
- Recommendation: Additional testing

### Project NPV
**Question:** How does measurement uncertainty affect project returns?

**Example:**
- Plant: 100 MW
- Energy Price: $0.10/kWh
- Uncertainty: 2.5%
- Operating: 25 years

**Result:**
- NPV Uncertainty: ±$X million
- ROI Range: Y% to Z%
- Revenue at Risk: $X/year

---

## 🤝 Support & Feedback

### Getting Help
- Review this guide and README.md
- Check DEVELOPMENT_ROADMAP.md for planned features
- Review code documentation in each module

### Reporting Issues
1. Check if issue is already known
2. Provide detailed description
3. Include: module type, simulator, uncertainty values
4. Share screenshots if helpful

### Feature Requests
We're continuously improving! Priority areas:
- Additional measurement types (Bifaciality, IAM, etc.)
- Advanced visualizations
- Database integration
- API access

---

## ✅ Checklist: Ready for Production Use

Before using in commercial testing:

- [ ] Verified simulator classification is current (<12 months)
- [ ] Reference device calibration is current (<24 months)
- [ ] All uncertainty values verified against equipment certificates
- [ ] Compared results with previous manual calculations
- [ ] Tested with known good modules
- [ ] Staff trained on interpretation
- [ ] Documentation reviewed and understood
- [ ] ISO 17025 reporting format approved by technical manager
- [ ] Backup procedures in place for critical data

---

## 🎉 You're Ready!

You now have a professional-grade PV uncertainty analysis tool. Start with simple STC measurements and gradually explore advanced features.

**Key Takeaways:**
1. Default values are provided, but use actual equipment data
2. Validate I-V ratios to catch measurement errors
3. Top 3-5 contributors drive most uncertainty
4. Report expanded uncertainty (k=2) for 95% confidence
5. Use financial analysis to communicate risk to stakeholders

**Happy Testing!** ☀️

---

*Last Updated: 2025-10-22*
*Version: 1.0 Professional Edition*
