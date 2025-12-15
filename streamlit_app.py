"""
PV Measurement Uncertainty Tool - Professional Edition
Complete uncertainty analysis platform for solar PV IV measurements
Implements GUM methodology, ISO 17025 reporting, and financial impact analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
import io

# Import custom modules
from config_data import (
    PV_TECHNOLOGIES, SUN_SIMULATORS, REFERENCE_LABS,
    MEASUREMENT_TYPES, CURRENCIES, UNCERTAINTY_CATEGORIES,
    get_technology_list, get_simulator_list, get_reference_lab_list
)
from pv_uncertainty_enhanced import (
    PVUncertaintyBudget, UncertaintyFactor, STCMeasurementUncertainty,
    create_default_stc_budget
)
from financial_impact import (
    FinancialImpactCalculator, calculate_cost_per_watt_benchmark
)
from file_utilities import (
    DataValidator, PDFExtractor, ExcelExtractor,
    DatasheetExtractor, PVsystPANParser
)
from report_generator import ISO17025ReportGenerator

# Database integration (optional - fails gracefully if not configured)
try:
    from database.streamlit_integration import display_db_status_sidebar, get_db_status
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="PV Uncertainty Tool - Professional Edition",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #3b82f6;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1e40af;
        padding: 0.8rem 0;
        border-left: 4px solid #3b82f6;
        padding-left: 1rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        background-color: #f0f9ff;
    }
    .subsection-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1e40af;
        padding: 0.5rem 0;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f8fafc;
        padding: 1.2rem;
        border-radius: 0.5rem;
        border: 1px solid #cbd5e1;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #dcfce7;
        border: 1px solid #86efac;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fef3c7;
        border: 1px solid #fcd34d;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .error-box {
        background-color: #fee2e2;
        border: 1px solid #fca5a5;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #dbeafe;
        border: 1px solid #93c5fd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .help-icon {
        display: inline-block;
        background-color: #3b82f6;
        color: white;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        text-align: center;
        font-size: 14px;
        font-weight: bold;
        cursor: help;
        margin-left: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: #f1f5f9;
        border-radius: 5px 5px 0 0;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize all session state variables."""
    if 'module_config' not in st.session_state:
        st.session_state.module_config = {}
    if 'simulator_config' not in st.session_state:
        st.session_state.simulator_config = {}
    if 'reference_config' not in st.session_state:
        st.session_state.reference_config = {}
    if 'measurement_data' not in st.session_state:
        st.session_state.measurement_data = {}
    if 'uncertainty_factors' not in st.session_state:
        st.session_state.uncertainty_factors = {}
    if 'calculation_results' not in st.session_state:
        st.session_state.calculation_results = None
    if 'financial_results' not in st.session_state:
        st.session_state.financial_results = None
    if 'report_config' not in st.session_state:
        st.session_state.report_config = {}


def show_help(help_text: str):
    """Display help text in an expander."""
    with st.expander("ℹ️ Help", expanded=False):
        st.info(help_text)


def section_1_module_configuration():
    """Section 1: Module Configuration"""
    st.markdown('<div class="section-header">1️⃣ Section 1: Module Configuration</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Technology selection
        st.markdown("### 1.1 PV Technology")
        technology = st.selectbox(
            "Select PV Technology",
            options=get_technology_list(),
            help="Choose the photovoltaic technology type"
        )

        if technology:
            tech_info = PV_TECHNOLOGIES[technology]
            st.info(f"**{tech_info.name}**: {tech_info.description}")

            # Display typical specifications
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Typical Efficiency", f"{tech_info.typical_efficiency}%")
            with col_b:
                st.metric("Typical Temp Coeff", f"{tech_info.typical_temp_coeff}%/°C")
            with col_c:
                st.metric("Typical Cells", tech_info.typical_cell_count if tech_info.typical_cell_count > 0 else "N/A")

    with col2:
        st.markdown("### 1.2 Datasheet Upload")
        uploaded_datasheet = st.file_uploader(
            "Upload Module Datasheet (PDF)",
            type=['pdf'],
            help="Optional: Upload datasheet for automatic parameter extraction"
        )

        if uploaded_datasheet:
            with st.spinner("Extracting datasheet parameters..."):
                extractor = PDFExtractor()
                text = extractor.extract_text_from_pdf(uploaded_datasheet)
                ds_extractor = DatasheetExtractor()
                extracted = ds_extractor.parse_module_datasheet(text)

                if any(extracted.values()):
                    st.success("✅ Parameters extracted successfully!")
                    st.session_state.datasheet_extracted = extracted

    # Module specifications
    st.markdown("### 1.3 Module Specifications")

    use_custom_cells = st.checkbox("Custom Cell Configuration", value=False)

    col1, col2, col3 = st.columns(3)

    with col1:
        if use_custom_cells:
            cells_series = st.number_input("Cells in Series", min_value=1, value=60, step=1)
            cells_parallel = st.number_input("Cells in Parallel", min_value=1, value=1, step=1)
            total_cells = cells_series * cells_parallel
            st.info(f"Total Cells: {total_cells}")
        else:
            total_cells = tech_info.typical_cell_count

    with col2:
        pmax_nameplate = st.number_input(
            "Nameplate Power Pmax (W)",
            min_value=0.0,
            value=float(st.session_state.get('datasheet_extracted', {}).get('pmax_stc', 400.0)),
            step=10.0,
            help="Rated power at STC"
        )

    with col3:
        module_area = st.number_input(
            "Module Area (m²)",
            min_value=0.1,
            value=2.0,
            step=0.1,
            help="Total module area"
        )

    # Temperature Coefficients
    st.markdown("### 1.4 Temperature Coefficients")

    col1, col2, col3 = st.columns(3)

    with col1:
        gamma_pmax = st.number_input(
            "γ_Pmax (%/°C)",
            min_value=-1.0,
            max_value=0.0,
            value=float(tech_info.typical_temp_coeff),
            step=0.01,
            format="%.3f",
            help="Power temperature coefficient (typically negative)"
        )

    with col2:
        beta_voc = st.number_input(
            "β_Voc (%/°C)",
            min_value=-0.5,
            max_value=-0.2,
            value=-0.30,
            step=0.01,
            format="%.3f",
            help="Voc temperature coefficient"
        )

    with col3:
        alpha_isc = st.number_input(
            "α_Isc (%/°C)",
            min_value=0.0,
            max_value=0.1,
            value=0.05,
            step=0.01,
            format="%.3f",
            help="Isc temperature coefficient (positive)"
        )

    # Bifaciality (if applicable)
    if tech_info.typical_bifaciality > 0:
        st.markdown("### 1.5 Bifaciality Parameters")
        col1, col2 = st.columns(2)
        with col1:
            bifaciality_factor = st.number_input(
                "Bifaciality Factor",
                min_value=0.0,
                max_value=1.0,
                value=float(tech_info.typical_bifaciality),
                step=0.01,
                help="Ratio of rear to front efficiency"
            )
        with col2:
            st.info(f"Technology default: {tech_info.typical_bifaciality:.2f}")
    else:
        bifaciality_factor = 0.0

    # Save to session state
    st.session_state.module_config = {
        'technology': technology,
        'cells_series': cells_series if use_custom_cells else tech_info.typical_cell_count,
        'cells_parallel': cells_parallel if use_custom_cells else 1,
        'pmax_nameplate': pmax_nameplate,
        'module_area': module_area,
        'gamma_pmax': gamma_pmax,
        'beta_voc': beta_voc,
        'alpha_isc': alpha_isc,
        'bifaciality_factor': bifaciality_factor
    }


def section_2_sun_simulator():
    """Section 2: Sun Simulator Configuration"""
    st.markdown('<div class="section-header">2️⃣ Section 2: Sun Simulator Configuration</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 2.1 Simulator Selection")
        simulator_name = st.selectbox(
            "Select Sun Simulator",
            options=get_simulator_list(),
            help="Choose your solar simulator equipment"
        )

        if simulator_name:
            sim_info = SUN_SIMULATORS[simulator_name]

            # Display simulator information
            st.info(f"""
            **{sim_info.manufacturer} - {sim_info.model}**
            📋 {sim_info.description}
            💡 Lamp Type: {sim_info.lamp_type}
            ⭐ Classification: {sim_info.classification}
            📏 Standard Distance: {sim_info.standard_distance_mm} mm
            """)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Typical Uniformity", f"{sim_info.typical_uniformity}%")
            with col_b:
                st.metric("Temporal Stability", f"{sim_info.typical_temporal_instability}%")
            with col_c:
                st.metric("Spectral Match", sim_info.typical_spectral_match)

    with col2:
        st.markdown("### 2.2 Classification Certificate")
        uploaded_cert = st.file_uploader(
            "Upload Classification Certificate (PDF)",
            type=['pdf', 'docx'],
            help="Optional: Upload simulator classification certificate",
            key="simulator_cert"
        )

        if uploaded_cert:
            st.success("✅ Certificate uploaded")

    # Detailed simulator parameters
    st.markdown("### 2.3 Simulator Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        uniformity = st.number_input(
            "Spatial Non-uniformity (%)",
            min_value=0.0,
            max_value=10.0,
            value=float(sim_info.typical_uniformity),
            step=0.1,
            help="Classification A: ±2%, B: ±5%, C: ±10%"
        )

    with col2:
        temporal = st.number_input(
            "Temporal Instability (%)",
            min_value=0.0,
            max_value=10.0,
            value=float(sim_info.typical_temporal_instability),
            step=0.1,
            help="Classification A: ±2%, B: ±5%, C: ±10%"
        )

    with col3:
        spectral_class = st.selectbox(
            "Spectral Match Class",
            options=['A', 'B', 'C'],
            index=0 if sim_info.typical_spectral_match == 'A' else 1,
            help="A: 0.75-1.25, B: 0.6-1.4, C: 0.4-2.0"
        )

    # Lamp to module distance
    distance_mm = st.number_input(
        "Lamp to Module Test Plane Distance (mm)",
        min_value=100.0,
        max_value=2000.0,
        value=float(sim_info.standard_distance_mm),
        step=10.0,
        help="Distance affects irradiance uniformity"
    )

    # Save to session state
    st.session_state.simulator_config = {
        'simulator_name': simulator_name,
        'manufacturer': sim_info.manufacturer,
        'model': sim_info.model,
        'lamp_type': sim_info.lamp_type,
        'classification': sim_info.classification,
        'uniformity': uniformity,
        'temporal_instability': temporal,
        'spectral_match_class': spectral_class,
        'distance_mm': distance_mm
    }


def section_3_reference_device():
    """Section 3: Reference Device & Calibration"""
    st.markdown('<div class="section-header">3️⃣ Section 3: Reference Device & Calibration</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 3.1 Reference Device Type")
        ref_type = st.radio(
            "Reference Device",
            options=["WPVS Reference Cell", "Reference Module"],
            horizontal=True,
            help="Select the type of reference device used"
        )

        st.markdown("### 3.2 Calibration Laboratory")
        lab_name = st.selectbox(
            "Calibration Laboratory",
            options=get_reference_lab_list(),
            help="Laboratory that calibrated the reference device"
        )

        if lab_name:
            lab_info = REFERENCE_LABS[lab_name]
            st.info(f"""
            **{lab_info.name}**
            🌍 {lab_info.country} | 🏛️ {lab_info.lab_type}
            📜 Accreditation: {lab_info.accreditation}
            📋 {lab_info.description}
            """)

            # Typical uncertainties
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("WPVS Cell Uncertainty", f"{lab_info.typical_uncertainty_wpvs}%")
            with col_b:
                st.metric("Module Uncertainty", f"{lab_info.typical_uncertainty_module}%")

    with col2:
        st.markdown("### 3.3 Calibration Certificate")
        uploaded_cal = st.file_uploader(
            "Upload Calibration Certificate (PDF)",
            type=['pdf'],
            help="Upload reference device calibration certificate",
            key="calibration_cert"
        )

        if uploaded_cal:
            with st.spinner("Extracting calibration data..."):
                extractor = PDFExtractor()
                text = extractor.extract_text_from_pdf(uploaded_cal)
                cal_data = extractor.extract_calibration_data(text)

                if any(cal_data.values()):
                    st.success("✅ Calibration data extracted!")
                    st.session_state.calibration_extracted = cal_data

    # Reference device parameters
    st.markdown("### 3.4 Reference Device Parameters")

    col1, col2, col3 = st.columns(3)

    # Get default values
    default_uncertainty = lab_info.typical_uncertainty_wpvs if ref_type == "WPVS Reference Cell" else lab_info.typical_uncertainty_module

    with col1:
        ref_isc = st.number_input(
            f"Reference Isc (A)",
            min_value=0.0,
            value=float(st.session_state.get('calibration_extracted', {}).get('isc_ref', 8.5)),
            step=0.01,
            help="Calibrated short-circuit current"
        )

    with col2:
        if ref_type == "Reference Module":
            ref_pmax = st.number_input(
                "Reference Pmax (W)",
                min_value=0.0,
                value=float(st.session_state.get('calibration_extracted', {}).get('pmax_ref', 300.0)),
                step=1.0,
                help="Calibrated maximum power"
            )
        else:
            ref_pmax = None

    with col3:
        ref_uncertainty = st.number_input(
            "Calibration Uncertainty (%)",
            min_value=0.0,
            max_value=10.0,
            value=float(st.session_state.get('calibration_extracted', {}).get('uncertainty', default_uncertainty)),
            step=0.1,
            help="Expanded uncertainty (k=2) from certificate"
        )

    # Additional uncertainty factors
    st.markdown("### 3.5 Additional Uncertainties")

    col1, col2 = st.columns(2)

    with col1:
        ref_drift = st.number_input(
            "Reference Device Drift (%)",
            min_value=0.0,
            max_value=5.0,
            value=0.3,
            step=0.1,
            help="Long-term drift since last calibration"
        )

    with col2:
        ref_positioning = st.number_input(
            "Positioning Uncertainty (%)",
            min_value=0.0,
            max_value=2.0,
            value=0.2,
            step=0.1,
            help="Uncertainty due to position difference in test plane"
        )

    # Save to session state
    st.session_state.reference_config = {
        'ref_type': ref_type,
        'lab_name': lab_name,
        'ref_isc': ref_isc,
        'ref_pmax': ref_pmax,
        'calibration_uncertainty': ref_uncertainty,
        'drift_uncertainty': ref_drift,
        'positioning_uncertainty': ref_positioning
    }


def section_4_measurement_data():
    """Section 4: STC Measurement Data Input"""
    st.markdown('<div class="section-header">4️⃣ Section 4: STC Measurement Data</div>',
                unsafe_allow_html=True)

    st.info("📋 Enter the measured I-V parameters at Standard Test Conditions (STC): 1000 W/m², 25°C, AM1.5G spectrum")

    # File upload option
    st.markdown("### 4.1 Data Import (Optional)")

    col1, col2, col3 = st.columns(3)

    with col1:
        test_report = st.file_uploader(
            "STC Test Report (PDF/Image)",
            type=['pdf', 'jpg', 'png'],
            help="Upload test report for documentation",
            key="test_report"
        )

    with col2:
        iv_data_file = st.file_uploader(
            "Raw I-V Data (Excel/CSV)",
            type=['xlsx', 'xls', 'csv'],
            help="Upload I-V curve data file",
            key="iv_data"
        )

        if iv_data_file:
            extractor = ExcelExtractor()
            iv_df = extractor.read_iv_curve_data(iv_data_file)
            if iv_df is not None:
                st.success(f"✅ Loaded {len(iv_df)} I-V points")
                st.session_state.iv_curve_data = iv_df

    with col3:
        summary_file = st.file_uploader(
            "Summary File (Excel)",
            type=['xlsx', 'xls'],
            help="Upload measurement summary",
            key="summary_file"
        )

        if summary_file:
            extractor = ExcelExtractor()
            summary_data = extractor.extract_summary_data(summary_file)
            if summary_data:
                st.success("✅ Parameters extracted")
                st.session_state.summary_extracted = summary_data

    # Manual data entry
    st.markdown("### 4.2 Measured STC Parameters")

    col1, col2, col3, col4 = st.columns(4)

    # Get extracted values if available
    extracted = st.session_state.get('summary_extracted', {})

    with col1:
        voc_meas = st.number_input(
            "Voc (V)",
            min_value=0.0,
            value=float(extracted.get('Voc', 45.0)),
            step=0.1,
            help="Open circuit voltage"
        )

    with col2:
        isc_meas = st.number_input(
            "Isc (A)",
            min_value=0.0,
            value=float(extracted.get('Isc', 10.0)),
            step=0.1,
            help="Short circuit current"
        )

    with col3:
        vmp_meas = st.number_input(
            "Vmp (V)",
            min_value=0.0,
            value=float(extracted.get('Vmp', 37.5)),
            step=0.1,
            help="Voltage at maximum power point"
        )

    with col4:
        imp_meas = st.number_input(
            "Imp (A)",
            min_value=0.0,
            value=float(extracted.get('Imp', 9.5)),
            step=0.1,
            help="Current at maximum power point"
        )

    # Calculate derived parameters
    pmax_meas = vmp_meas * imp_meas
    ff_meas = DataValidator.calculate_fill_factor(vmp_meas, voc_meas, imp_meas, isc_meas)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Calculated Pmax", f"{pmax_meas:.2f} W")

    with col2:
        st.metric("Fill Factor", f"{ff_meas:.4f}" if ff_meas > 0 else "N/A")

    # Data authenticity checks
    st.markdown("### 4.3 Data Validation")

    validation = DataValidator.validate_iv_ratios(vmp_meas, voc_meas, imp_meas, isc_meas)

    if validation['valid']:
        st.success("✅ I-V data passed validation checks")
    else:
        st.warning("⚠️ Data validation warnings detected:")
        for warning in validation['warnings']:
            st.markdown(f"- {warning}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Vmp/Voc Ratio", f"{validation['ratios']['vmp_voc']:.3f}")
        if 0.70 <= validation['ratios']['vmp_voc'] <= 0.90:
            st.caption("✅ Within normal range [0.70-0.90]")
        else:
            st.caption("⚠️ Outside normal range")

    with col2:
        st.metric("Imp/Isc Ratio", f"{validation['ratios']['imp_isc']:.3f}")
        if 0.85 <= validation['ratios']['imp_isc'] <= 0.99:
            st.caption("✅ Within normal range [0.85-0.99]")
        else:
            st.caption("⚠️ Outside normal range")

    # Measurement conditions
    st.markdown("### 4.4 Actual Measurement Conditions")

    col1, col2, col3 = st.columns(3)

    with col1:
        actual_irradiance = st.number_input(
            "Measured Irradiance (W/m²)",
            min_value=900.0,
            max_value=1100.0,
            value=1000.0,
            step=1.0,
            help="Actual irradiance during measurement"
        )

    with col2:
        actual_temp = st.number_input(
            "Measured Module Temperature (°C)",
            min_value=0.0,
            max_value=50.0,
            value=25.0,
            step=0.1,
            help="Actual module temperature"
        )

    with col3:
        delta_temp = actual_temp - 25.0
        st.metric("ΔT from STC", f"{delta_temp:+.1f} °C")

    # Save to session state
    st.session_state.measurement_data = {
        'voc': voc_meas,
        'isc': isc_meas,
        'vmp': vmp_meas,
        'imp': imp_meas,
        'pmax': pmax_meas,
        'fill_factor': ff_meas,
        'actual_irradiance': actual_irradiance,
        'actual_temperature': actual_temp,
        'delta_temperature': delta_temp,
        'validation': validation
    }


def create_fishbone_diagram_plotly(budget_data: Dict) -> go.Figure:
    """Create a simplified fishbone diagram using Plotly."""

    fig = go.Figure()

    # Main spine
    fig.add_trace(go.Scatter(
        x=[0, 10],
        y=[0, 0],
        mode='lines',
        line=dict(color='black', width=3),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Categories with positions
    categories = [
        ("1", "Reference\nDevice", -3, 2),
        ("2", "Sun\nSimulator", -2, -2),
        ("3", "Temperature", -1, 2),
        ("4", "I-V\nMeasurement", 1, -2),
        ("5", "Module\nBehavior", 2, 2),
        ("6", "Environment", 3, -2),
        ("7", "Procedure", 4, 2)
    ]

    for cat_id, name, x_offset, y_offset in categories:
        # Branch line
        fig.add_trace(go.Scatter(
            x=[5 + x_offset, 5 + x_offset + np.sign(y_offset)*0.5],
            y=[0, y_offset],
            mode='lines',
            line=dict(color='#3b82f6', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Category label
        fig.add_annotation(
            x=5 + x_offset + np.sign(y_offset)*0.7,
            y=y_offset,
            text=f"<b>{cat_id}. {name}</b>",
            showarrow=False,
            font=dict(size=10, color='#1e40af'),
            bgcolor='#dbeafe',
            bordercolor='#3b82f6',
            borderwidth=1,
            borderpad=4
        )

    # Result box
    fig.add_annotation(
        x=10,
        y=0,
        text="<b>Combined<br>Uncertainty</b>",
        showarrow=False,
        font=dict(size=12, color='white'),
        bgcolor='#dc2626',
        bordercolor='#991b1b',
        borderwidth=2,
        borderpad=8
    )

    fig.update_layout(
        title="Uncertainty Fishbone Diagram",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 11]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3, 3]),
        plot_bgcolor='white',
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def section_5_uncertainty_factors():
    """Section 5: Uncertainty Factors & Budget"""
    st.markdown('<div class="section-header">5️⃣ Section 5: Uncertainty Analysis</div>',
                unsafe_allow_html=True)

    st.info("📊 Review and adjust uncertainty factors. Default values are provided based on typical equipment and procedures.")

    # Create uncertainty budget
    pmax_measured = st.session_state.measurement_data.get('pmax', 300.0)

    # Get configuration values
    ref_config = st.session_state.reference_config
    sim_config = st.session_state.simulator_config
    meas_data = st.session_state.measurement_data

    # Show fishbone diagram
    st.markdown("### 5.1 Uncertainty Fishbone Diagram")
    fishbone_fig = create_fishbone_diagram_plotly({})
    st.plotly_chart(fishbone_fig, use_container_width=True)

    show_help("""
    **Fishbone Diagram** shows the seven main categories of uncertainty sources:
    1. **Reference Device**: Calibration, drift, positioning
    2. **Sun Simulator**: Uniformity, temporal stability, spectral mismatch
    3. **Temperature**: Sensor calibration, uniformity, correction procedure
    4. **I-V Measurement**: Voltage/current meter calibration, curve fitting
    5. **Module Behavior**: Hysteresis, stabilization effects
    6. **Environment**: Ambient conditions
    7. **Procedure**: Repeatability, reproducibility (R&R, ILC)
    """)

    # Uncertainty factor inputs organized by category
    st.markdown("### 5.2 Uncertainty Factor Values")

    tabs = st.tabs([
        "1️⃣ Reference Device",
        "2️⃣ Sun Simulator",
        "3️⃣ Temperature",
        "4️⃣ I-V Measurement",
        "5️⃣ Module & Procedure"
    ])

    uncertainty_values = {}

    with tabs[0]:
        st.markdown("#### 1. Reference Device Uncertainties")
        col1, col2 = st.columns(2)
        with col1:
            uncertainty_values['ref_calibration'] = st.number_input(
                "1.1 Calibration Uncertainty (%)",
                min_value=0.0,
                max_value=10.0,
                value=ref_config.get('calibration_uncertainty', 1.5),
                step=0.1,
                key="unc_ref_cal"
            )
            uncertainty_values['ref_drift'] = st.number_input(
                "1.2 Reference Drift (%)",
                min_value=0.0,
                max_value=5.0,
                value=ref_config.get('drift_uncertainty', 0.3),
                step=0.1,
                key="unc_ref_drift"
            )
        with col2:
            uncertainty_values['ref_positioning'] = st.number_input(
                "1.3 Positioning (%)",
                min_value=0.0,
                max_value=2.0,
                value=ref_config.get('positioning_uncertainty', 0.2),
                step=0.1,
                key="unc_ref_pos"
            )

    with tabs[1]:
        st.markdown("#### 2. Sun Simulator Uncertainties")
        col1, col2, col3 = st.columns(3)
        with col1:
            uncertainty_values['sim_uniformity'] = st.number_input(
                "2.1 Spatial Non-uniformity (%)",
                min_value=0.0,
                max_value=10.0,
                value=sim_config.get('uniformity', 2.0),
                step=0.1,
                key="unc_sim_unif"
            )
        with col2:
            uncertainty_values['sim_temporal'] = st.number_input(
                "2.2 Temporal Instability (%)",
                min_value=0.0,
                max_value=10.0,
                value=sim_config.get('temporal_instability', 0.5),
                step=0.1,
                key="unc_sim_temp"
            )
        with col3:
            uncertainty_values['spectral_mismatch'] = st.number_input(
                "2.3 Spectral Mismatch (%)",
                min_value=0.0,
                max_value=5.0,
                value=0.8,
                step=0.1,
                key="unc_spectral"
            )

    with tabs[2]:
        st.markdown("#### 3. Temperature Measurement Uncertainties")
        col1, col2, col3 = st.columns(3)
        with col1:
            uncertainty_values['temp_sensor'] = st.number_input(
                "3.1 Sensor Calibration (°C)",
                min_value=0.0,
                max_value=5.0,
                value=0.5,
                step=0.1,
                key="unc_temp_sensor"
            )
        with col2:
            uncertainty_values['temp_uniformity'] = st.number_input(
                "3.2 Module Uniformity (°C)",
                min_value=0.0,
                max_value=5.0,
                value=1.0,
                step=0.1,
                key="unc_temp_unif"
            )
        with col3:
            uncertainty_values['temp_coeff_unc'] = st.number_input(
                "3.3 Temp Coefficient Unc (%/°C)",
                min_value=0.0,
                max_value=0.2,
                value=0.05,
                step=0.01,
                format="%.3f",
                key="unc_temp_coeff"
            )

    with tabs[3]:
        st.markdown("#### 4. I-V Measurement Uncertainties")
        col1, col2, col3 = st.columns(3)
        with col1:
            uncertainty_values['voltage_unc'] = st.number_input(
                "4.1 Voltage Measurement (%)",
                min_value=0.0,
                max_value=2.0,
                value=0.2,
                step=0.05,
                key="unc_voltage"
            )
        with col2:
            uncertainty_values['current_unc'] = st.number_input(
                "4.2 Current Measurement (%)",
                min_value=0.0,
                max_value=2.0,
                value=0.2,
                step=0.05,
                key="unc_current"
            )
        with col3:
            uncertainty_values['curve_fitting'] = st.number_input(
                "4.3 Curve Fitting (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                key="unc_curve"
            )

    with tabs[4]:
        st.markdown("#### 5. Module Behavior & Procedure")
        col1, col2, col3 = st.columns(3)
        with col1:
            uncertainty_values['hysteresis'] = st.number_input(
                "5.1 Hysteresis (%)",
                min_value=0.0,
                max_value=2.0,
                value=0.3,
                step=0.1,
                key="unc_hysteresis"
            )
        with col2:
            uncertainty_values['repeatability'] = st.number_input(
                "7.1 Repeatability (%)",
                min_value=0.0,
                max_value=5.0,
                value=0.5,
                step=0.1,
                key="unc_repeat"
            )
        with col3:
            uncertainty_values['reproducibility'] = st.number_input(
                "7.2 Reproducibility/ILC (%)",
                min_value=0.0,
                max_value=5.0,
                value=1.5,
                step=0.1,
                key="unc_repro"
            )

    st.session_state.uncertainty_factors = uncertainty_values

    # Calculate button
    st.markdown("---")
    if st.button("🔢 Calculate Combined Uncertainty", type="primary", use_container_width=True):
        with st.spinner("Calculating uncertainty budget..."):
            result = STCMeasurementUncertainty.calculate(
                pmax_measured=pmax_measured,
                ref_calibration_unc=uncertainty_values['ref_calibration'],
                ref_drift_unc=uncertainty_values['ref_drift'],
                ref_positioning_unc=uncertainty_values['ref_positioning'],
                simulator_uniformity=uncertainty_values['sim_uniformity'],
                simulator_temporal=uncertainty_values['sim_temporal'],
                spectral_mismatch_unc=uncertainty_values['spectral_mismatch'],
                temp_sensor_unc=uncertainty_values['temp_sensor'],
                temp_uniformity_unc=uncertainty_values['temp_uniformity'],
                temp_coeff_unc=uncertainty_values['temp_coeff_unc'],
                delta_temperature=meas_data.get('delta_temperature', 0.0),
                voltage_unc=uncertainty_values['voltage_unc'],
                current_unc=uncertainty_values['current_unc'],
                curve_fitting_unc=uncertainty_values['curve_fitting'],
                hysteresis_unc=uncertainty_values['hysteresis'],
                repeatability=uncertainty_values['repeatability'],
                reproducibility=uncertainty_values['reproducibility']
            )

            st.session_state.calculation_results = result
            st.success("✅ Uncertainty calculation complete!")
            st.rerun()


def section_6_results():
    """Section 6: Results & Uncertainty Budget"""
    st.markdown('<div class="section-header">6️⃣ Section 6: Results & Uncertainty Budget</div>',
                unsafe_allow_html=True)

    if st.session_state.calculation_results is None:
        st.warning("⚠️ Please complete Section 5 and calculate uncertainty first.")
        return

    result = st.session_state.calculation_results
    pmax = result['pmax_measured']
    combined_unc = result['combined_standard_uncertainty']
    expanded_unc = result['expanded_uncertainty_k2']
    relative_unc = result['relative_uncertainty_percent']

    # Main results
    st.markdown("### 6.1 Summary Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Measured Pmax",
            f"{pmax:.2f} W",
            help="Maximum power at STC"
        )

    with col2:
        st.metric(
            "Combined Uncertainty",
            f"{combined_unc:.2f}%",
            help="Combined standard uncertainty (k=1)"
        )

    with col3:
        st.metric(
            "Expanded Uncertainty (k=2)",
            f"{expanded_unc:.2f}%",
            help="95% confidence level"
        )

    with col4:
        st.metric(
            "Absolute Uncertainty",
            f"±{result['pmax_uncertainty_absolute']:.2f} W",
            help="At 95% confidence"
        )

    # Confidence interval
    st.markdown("### 6.2 Measurement Result Statement")
    ci_lower, ci_upper = result['pmax_confidence_interval_95']

    st.success(f"""
    **Measurement Result (ISO 17025 format):**

    The maximum power of the tested PV module at Standard Test Conditions (1000 W/m², 25°C, AM1.5G spectrum) is:

    **Pmax = {pmax:.2f} W ± {result['pmax_uncertainty_absolute']:.2f} W (k=2)**

    The reported expanded uncertainty is based on a combined standard uncertainty multiplied by a coverage factor k=2,
    providing a level of confidence of approximately 95%.

    **95% Confidence Interval: [{ci_lower:.2f} W, {ci_upper:.2f} W]**
    """)

    # Uncertainty budget table
    st.markdown("### 6.3 Detailed Uncertainty Budget")

    budget_df = pd.DataFrame(result['components'])
    budget_df = budget_df[[
        'factor_id', 'name', 'standard_uncertainty', 'distribution',
        'sensitivity_coefficient', 'percentage_contribution'
    ]].copy()

    budget_df.columns = [
        'ID', 'Uncertainty Source', 'Std Unc', 'Distribution',
        'Sensitivity', 'Contribution (%)'
    ]

    # Format for display
    budget_df['Std Unc'] = budget_df['Std Unc'].round(4)
    budget_df['Sensitivity'] = budget_df['Sensitivity'].round(3)
    budget_df['Contribution (%)'] = budget_df['Contribution (%)'].round(2)

    st.dataframe(
        budget_df,
        use_container_width=True,
        hide_index=True
    )

    # Download budget
    csv = budget_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Uncertainty Budget (CSV)",
        data=csv,
        file_name=f"uncertainty_budget_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

    # Visualizations
    st.markdown("### 6.4 Uncertainty Budget Visualization")

    tab1, tab2, tab3 = st.tabs(["📊 Contribution Chart", "🥧 Pie Chart", "📈 Pareto Chart"])

    with tab1:
        # Bar chart of contributions
        top_10 = budget_df.nlargest(10, 'Contribution (%)')

        fig = px.bar(
            top_10,
            x='Contribution (%)',
            y='Uncertainty Source',
            orientation='h',
            title="Top 10 Uncertainty Contributors",
            color='Contribution (%)',
            color_continuous_scale='Blues_r'
        )
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Pie chart
        top_8 = budget_df.nlargest(8, 'Contribution (%)')
        others_sum = budget_df.nsmallest(len(budget_df) - 8, 'Contribution (%)')[['Contribution (%)']].sum()['Contribution (%)']

        pie_data = pd.concat([
            top_8[['Uncertainty Source', 'Contribution (%)']],
            pd.DataFrame({'Uncertainty Source': ['Others'], 'Contribution (%)': [others_sum]})
        ])

        fig = px.pie(
            pie_data,
            names='Uncertainty Source',
            values='Contribution (%)',
            title="Uncertainty Budget Distribution"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Pareto chart
        sorted_budget = budget_df.sort_values('Contribution (%)', ascending=False).copy()
        sorted_budget['Cumulative (%)'] = sorted_budget['Contribution (%)'].cumsum()

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(
                x=sorted_budget['Uncertainty Source'][:15],
                y=sorted_budget['Contribution (%)'][:15],
                name='Individual Contribution',
                marker_color='#3b82f6'
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=sorted_budget['Uncertainty Source'][:15],
                y=sorted_budget['Cumulative (%)'][:15],
                name='Cumulative',
                mode='lines+markers',
                marker=dict(color='#dc2626', size=8),
                line=dict(color='#dc2626', width=2)
            ),
            secondary_y=True
        )

        fig.update_xaxes(title_text="Uncertainty Source", tickangle=-45)
        fig.update_yaxes(title_text="Contribution (%)", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative (%)", secondary_y=True)
        fig.update_layout(
            title="Pareto Chart - Cumulative Contribution",
            height=500,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)


def section_7_financial_impact():
    """Section 7: Financial Impact Analysis"""
    st.markdown('<div class="section-header">7️⃣ Section 7: Financial Impact Analysis</div>',
                unsafe_allow_html=True)

    if st.session_state.calculation_results is None:
        st.warning("⚠️ Please complete uncertainty calculation first (Section 5).")
        return

    result = st.session_state.calculation_results
    pmax = result['pmax_measured']
    pmax_unc_absolute = result['pmax_uncertainty_absolute']

    st.info("💰 Analyze the financial implications of measurement uncertainty in different scenarios")

    # Scenario selection
    scenario = st.radio(
        "Select Financial Analysis Scenario",
        options=[
            "Fresh Module Pricing",
            "Warranty/Insurance Claim",
            "Project Financing (NPV/ROI)"
        ],
        horizontal=True
    )

    # Currency selection
    col1, col2 = st.columns([1, 3])
    with col1:
        currency = st.selectbox(
            "Currency",
            options=list(CURRENCIES.keys()),
            index=0,
            format_func=lambda x: f"{CURRENCIES[x]['symbol']} {x} - {CURRENCIES[x]['name']}"
        )

    with col2:
        # Technology benchmark
        tech = st.session_state.module_config.get('technology', 'PERC')
        benchmark = calculate_cost_per_watt_benchmark(tech, region='global')
        benchmark_price = benchmark['prices_by_currency'][currency]
        st.info(f"💡 2024 Benchmark for {tech}: {CURRENCIES[currency]['symbol']}{benchmark_price:.3f}/W")

    if scenario == "Fresh Module Pricing":
        st.markdown("### Fresh Module Nameplate Pricing Analysis")

        col1, col2 = st.columns(2)

        with col1:
            price_per_watt = st.number_input(
                f"Module Price ({CURRENCIES[currency]['symbol']}/W)",
                min_value=0.0,
                value=float(benchmark_price),
                step=0.01,
                format="%.3f",
                help="Selling price per watt"
            )

        with col2:
            confidence_level = st.selectbox(
                "Confidence Level",
                options=[68.0, 95.0, 99.0],
                index=1,
                format_func=lambda x: f"{x}% ({int(x/95)} sigma)"
            )

        if st.button("Calculate Price Impact", type="primary"):
            calc = FinancialImpactCalculator()
            financial_result = calc.calculate_module_price_impact(
                module_power=pmax,
                power_uncertainty=pmax_unc_absolute,
                price_per_watt=price_per_watt,
                currency=currency,
                confidence_level=confidence_level
            )

            st.session_state.financial_results = financial_result

            # Display results
            st.success("✅ Price impact calculated")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Nominal Module Price",
                    f"{financial_result['currency_symbol']}{financial_result['module_price_nominal']:.2f}"
                )

            with col2:
                st.metric(
                    "Price Uncertainty",
                    f"±{financial_result['currency_symbol']}{financial_result['price_uncertainty']:.2f}",
                    delta=f"{financial_result['price_risk_percent']:.2f}%"
                )

            with col3:
                st.metric(
                    "Price Range",
                    f"{financial_result['currency_symbol']}{financial_result['price_lower_bound']:.2f} - {financial_result['currency_symbol']}{financial_result['price_upper_bound']:.2f}"
                )

            # Visualization
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=['Lower Bound', 'Nominal', 'Upper Bound'],
                y=[
                    financial_result['price_lower_bound'],
                    financial_result['module_price_nominal'],
                    financial_result['price_upper_bound']
                ],
                marker_color=['#dc2626', '#3b82f6', '#16a34a'],
                text=[
                    f"{financial_result['currency_symbol']}{financial_result['price_lower_bound']:.2f}",
                    f"{financial_result['currency_symbol']}{financial_result['module_price_nominal']:.2f}",
                    f"{financial_result['currency_symbol']}{financial_result['price_upper_bound']:.2f}"
                ],
                textposition='outside'
            ))

            fig.update_layout(
                title=f"Module Price with Uncertainty ({confidence_level}% Confidence)",
                yaxis_title=f"Price ({financial_result['currency_symbol']})",
                height=400,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

            # Risk interpretation
            st.markdown("#### Financial Risk Interpretation")
            st.info(f"""
            **Seller Risk:** Up to {financial_result['currency_symbol']}{financial_result['interpretation']['seller_risk']:.2f}
            if actual power is lower than nameplate

            **Buyer Risk:** Up to {financial_result['currency_symbol']}{financial_result['interpretation']['buyer_risk']:.2f}
            if actual power is higher than nameplate

            **Total Measurement Uncertainty Impact:** {financial_result['currency_symbol']}{financial_result['interpretation']['measurement_value']:.2f}
            (price range due to measurement uncertainty)
            """)

    elif scenario == "Warranty/Insurance Claim":
        st.markdown("### Warranty/Insurance Claim Analysis")

        col1, col2 = st.columns(2)

        with col1:
            nameplate_power = st.number_input(
                "Original Nameplate Power (W)",
                min_value=0.0,
                value=st.session_state.module_config.get('pmax_nameplate', 400.0),
                step=10.0,
                help="Original rated power when new"
            )

        with col2:
            warranty_threshold = st.number_input(
                "Warranty Threshold (%)",
                min_value=70.0,
                max_value=100.0,
                value=90.0,
                step=1.0,
                help="Minimum guaranteed power as % of nameplate"
            ) / 100

        price_per_watt_claim = st.number_input(
            f"Replacement/Compensation Price ({CURRENCIES[currency]['symbol']}/W)",
            min_value=0.0,
            value=float(benchmark_price),
            step=0.01,
            format="%.3f"
        )

        if st.button("Analyze Warranty Claim", type="primary"):
            calc = FinancialImpactCalculator()
            warranty_result = calc.calculate_warranty_claim_impact(
                module_power_measured=pmax,
                nameplate_power=nameplate_power,
                power_uncertainty=pmax_unc_absolute,
                warranty_threshold=warranty_threshold,
                module_price_per_watt=price_per_watt_claim,
                currency=currency
            )

            st.session_state.financial_results = warranty_result

            # Display results
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Nameplate Power", f"{warranty_result['nameplate_power']:.2f} W")
                st.metric("Measured Power", f"{warranty_result['measured_power']:.2f} W")

            with col2:
                st.metric("Warranty Threshold", f"{warranty_result['warranty_power_threshold']:.2f} W")
                st.metric("Power Deficit", f"{warranty_result['power_deficit']:.2f} W ({warranty_result['deficit_percent']:.1f}%)")

            with col3:
                st.metric("Claim Probability", f"{warranty_result['probability_below_threshold']*100:.1f}%")
                st.metric("Potential Claim Value", f"{warranty_result['currency_symbol']}{warranty_result['claim_value_nominal']:.2f}")

            # Decision
            st.markdown("#### Claim Validity Assessment")

            if warranty_result['claim_valid_certain']:
                st.error(f"**VALID CLAIM** ✅ {warranty_result['recommendation']}")
            elif warranty_result['claim_invalid_certain']:
                st.success(f"**INVALID CLAIM** ❌ {warranty_result['recommendation']}")
            else:
                st.warning(f"**UNCERTAIN** ⚠️ {warranty_result['recommendation']}")

            # Visualization
            fig = go.Figure()

            # Add measured value with uncertainty bars
            fig.add_trace(go.Scatter(
                x=[warranty_result['measured_power']],
                y=[1],
                mode='markers',
                marker=dict(size=15, color='blue'),
                name='Measured',
                error_x=dict(
                    type='data',
                    array=[pmax_unc_absolute * 2],
                    visible=True,
                    color='blue',
                    thickness=3
                )
            ))

            # Add warranty threshold line
            fig.add_vline(
                x=warranty_result['warranty_power_threshold'],
                line=dict(color='red', width=2, dash='dash'),
                annotation_text=f"Warranty Threshold ({warranty_threshold*100}%)",
                annotation_position="top"
            )

            # Add nameplate power
            fig.add_vline(
                x=warranty_result['nameplate_power'],
                line=dict(color='green', width=2, dash='dot'),
                annotation_text="Original Nameplate",
                annotation_position="top"
            )

            fig.update_layout(
                title="Warranty Claim Visualization",
                xaxis_title="Power (W)",
                yaxis=dict(showticklabels=False, range=[0.5, 1.5]),
                height=300,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

    elif scenario == "Project Financing (NPV/ROI)":
        st.markdown("### Project Financing Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:
            plant_size_mw = st.number_input(
                "Plant Size (MW DC)",
                min_value=0.1,
                max_value=1000.0,
                value=1.0,
                step=0.1,
                help="Total DC capacity"
            )

        with col2:
            energy_price = st.number_input(
                f"Energy Price ({CURRENCIES[currency]['symbol']}/kWh)",
                min_value=0.0,
                value=0.10,
                step=0.01,
                format="%.3f",
                help="PPA or feed-in tariff price"
            )

        with col3:
            operating_years = st.number_input(
                "Operating Years",
                min_value=1,
                max_value=50,
                value=25,
                step=1
            )

        # Advanced parameters
        with st.expander("Advanced Financial Parameters"):
            col1, col2, col3 = st.columns(3)

            with col1:
                discount_rate = st.number_input(
                    "Discount Rate (%/year)",
                    min_value=0.0,
                    max_value=20.0,
                    value=5.0,
                    step=0.5
                ) / 100

            with col2:
                degradation_rate = st.number_input(
                    "Degradation Rate (%/year)",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.5,
                    step=0.1
                ) / 100

            with col3:
                capacity_factor = st.number_input(
                    "Capacity Factor",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.20,
                    step=0.01,
                    format="%.2f",
                    help="Annual average capacity factor"
                )

        if st.button("Calculate Project NPV/ROI", type="primary"):
            calc = FinancialImpactCalculator()

            # Get relative uncertainty
            relative_unc_pct = result['relative_uncertainty_percent']

            npv_result = calc.calculate_project_npv_impact(
                plant_size_mw=plant_size_mw,
                power_uncertainty_percent=relative_unc_pct,
                module_price_per_watt=price_per_watt if 'price_per_watt' in locals() else benchmark_price,
                energy_price_per_kwh=energy_price,
                operating_years=operating_years,
                discount_rate=discount_rate,
                degradation_rate=degradation_rate,
                capacity_factor=capacity_factor,
                currency=currency
            )

            st.session_state.financial_results = npv_result

            # Display results
            st.success("✅ Project financial analysis complete")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "NPV (Nominal)",
                    f"{npv_result['currency_symbol']}{npv_result['npv_nominal']/1e6:.2f}M"
                )

            with col2:
                st.metric(
                    "NPV Uncertainty",
                    f"±{npv_result['currency_symbol']}{npv_result['financial_risk_assessment']['npv_at_risk']/1e6:.2f}M",
                    delta=f"±{npv_result['npv_risk_percent']:.1f}%"
                )

            with col3:
                st.metric(
                    "ROI (Nominal)",
                    f"{npv_result['roi_nominal_percent']:.1f}%"
                )

            with col4:
                st.metric(
                    "Payback Period",
                    f"{npv_result['payback_period_nominal_years']:.1f} years"
                )

            # NPV range visualization
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=['Lower (95%)', 'Nominal', 'Upper (95%)'],
                y=[
                    npv_result['npv_lower_95'] / 1e6,
                    npv_result['npv_nominal'] / 1e6,
                    npv_result['npv_upper_95'] / 1e6
                ],
                marker_color=['#dc2626', '#3b82f6', '#16a34a'],
                text=[
                    f"{npv_result['currency_symbol']}{npv_result['npv_lower_95']/1e6:.2f}M",
                    f"{npv_result['currency_symbol']}{npv_result['npv_nominal']/1e6:.2f}M",
                    f"{npv_result['currency_symbol']}{npv_result['npv_upper_95']/1e6:.2f}M"
                ],
                textposition='outside'
            ))

            fig.update_layout(
                title="Net Present Value with Uncertainty",
                yaxis_title=f"NPV ({npv_result['currency_symbol']} Million)",
                height=400,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

            # Risk assessment
            st.markdown("#### Financial Risk Assessment")

            col1, col2 = st.columns(2)

            with col1:
                st.info(f"""
                **Revenue at Risk (Year 1):**
                {npv_result['currency_symbol']}{npv_result['financial_risk_assessment']['revenue_at_risk_year_1']:,.2f}

                **Cumulative Revenue Risk ({operating_years} years):**
                {npv_result['currency_symbol']}{npv_result['financial_risk_assessment']['cumulative_revenue_risk']/1e6:.2f}M
                """)

            with col2:
                st.info(f"""
                **ROI Range:**
                {npv_result['roi_lower_percent']:.1f}% to {npv_result['roi_upper_percent']:.1f}%

                **Payback Period Range:**
                {npv_result['payback_period_lower_years']:.1f} to {npv_result['payback_period_upper_years']:.1f} years
                """)


def section_8_professional_reporting():
    """Section 8: Professional ISO 17025 Reporting"""
    st.markdown('<div class="section-header">8️⃣ Section 8: Professional Reporting</div>',
                unsafe_allow_html=True)

    # Check if calculation results exist
    if st.session_state.calculation_results is None:
        st.warning("⚠️ Please complete uncertainty calculation first (Section 5).")
        st.info("Navigate to Section 5 (Uncertainty) and click 'Calculate Combined Uncertainty' to proceed.")
        return

    st.info("📄 Generate professional ISO 17025 compliant reports in PDF and Excel formats")

    # Report configuration section
    st.markdown("### 📋 Report Configuration")

    # Initialize report_config in session state if not exists
    if 'report_config' not in st.session_state or st.session_state.report_config is None:
        st.session_state.report_config = {}

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Laboratory Information")

        company_name = st.text_input(
            "Laboratory Name",
            value=st.session_state.report_config.get('company_name', 'PV Testing Laboratory'),
            help="Official name of the testing laboratory"
        )

        lab_address = st.text_area(
            "Laboratory Address",
            value=st.session_state.report_config.get('lab_address', ''),
            help="Complete laboratory address",
            height=100
        )

        lab_accreditation = st.text_input(
            "Accreditation Number",
            value=st.session_state.report_config.get('lab_accreditation', ''),
            help="ISO 17025 accreditation number (if applicable)"
        )

        document_format = st.text_input(
            "Document Format Number",
            value=st.session_state.report_config.get('document_format', 'PV-UNC-001'),
            help="Internal document format identifier"
        )

    with col2:
        st.markdown("#### Report Details")

        report_number = st.text_input(
            "Report Number",
            value=st.session_state.report_config.get('report_number', f"UNC-{datetime.now().strftime('%Y%m%d-%H%M')}"),
            help="Unique report identifier"
        )

        client_name = st.text_input(
            "Client Name",
            value=st.session_state.report_config.get('client_name', ''),
            help="Name of client requesting the test"
        )

        test_date = st.date_input(
            "Test Date",
            value=datetime.now().date(),
            help="Date when measurements were performed"
        )

        report_date = st.date_input(
            "Report Date",
            value=datetime.now().date(),
            help="Date when report is issued"
        )

    # Logo upload
    st.markdown("#### Company Logo (Optional)")
    logo_file = st.file_uploader(
        "Upload Laboratory Logo",
        type=['png', 'jpg', 'jpeg'],
        help="Recommended size: 150x50 pixels or similar aspect ratio"
    )

    logo_path = None
    if logo_file is not None:
        # Save uploaded logo temporarily
        logo_bytes = logo_file.read()
        logo_path = f"/tmp/logo_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        with open(logo_path, 'wb') as f:
            f.write(logo_bytes)
        st.success(f"✅ Logo uploaded: {logo_file.name}")

    # Signature section
    st.markdown("### ✍️ Signature Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Prepared By")
        preparer_name = st.text_input(
            "Name",
            value=st.session_state.report_config.get('preparer_name', ''),
            key="preparer_name"
        )
        preparer_title = st.text_input(
            "Title",
            value=st.session_state.report_config.get('preparer_title', 'Test Engineer'),
            key="preparer_title"
        )

    with col2:
        st.markdown("#### Reviewed By")
        reviewer_name = st.text_input(
            "Name",
            value=st.session_state.report_config.get('reviewer_name', ''),
            key="reviewer_name"
        )
        reviewer_title = st.text_input(
            "Title",
            value=st.session_state.report_config.get('reviewer_title', 'Senior Engineer'),
            key="reviewer_title"
        )

    with col3:
        st.markdown("#### Approved By")
        approver_name = st.text_input(
            "Name",
            value=st.session_state.report_config.get('approver_name', ''),
            key="approver_name"
        )
        approver_title = st.text_input(
            "Title",
            value=st.session_state.report_config.get('approver_title', 'Technical Manager'),
            key="approver_title"
        )

    # Additional report options
    with st.expander("📝 Additional Report Options"):
        include_financial = st.checkbox(
            "Include Financial Impact Analysis",
            value=True,
            help="Include Section 7 financial analysis in the report"
        )

        include_charts = st.checkbox(
            "Include Uncertainty Charts",
            value=True,
            help="Include bar charts, pie charts, and Pareto diagram"
        )

        include_equipment_details = st.checkbox(
            "Include Detailed Equipment Information",
            value=True,
            help="Include complete simulator and reference device specifications"
        )

        comments = st.text_area(
            "Additional Comments/Notes",
            value=st.session_state.report_config.get('comments', ''),
            help="Any additional information to include in the report",
            height=100
        )

    # Update session state with all report config
    st.session_state.report_config.update({
        'company_name': company_name,
        'lab_address': lab_address,
        'lab_accreditation': lab_accreditation,
        'document_format': document_format,
        'report_number': report_number,
        'client_name': client_name,
        'test_date': test_date.strftime('%Y-%m-%d'),
        'report_date': report_date.strftime('%Y-%m-%d'),
        'preparer_name': preparer_name,
        'preparer_title': preparer_title,
        'reviewer_name': reviewer_name,
        'reviewer_title': reviewer_title,
        'approver_name': approver_name,
        'approver_title': approver_title,
        'include_financial': include_financial,
        'include_charts': include_charts,
        'include_equipment_details': include_equipment_details,
        'comments': comments
    })

    # Report generation section
    st.markdown("---")
    st.markdown("### 📥 Generate Reports")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### PDF Report")
        st.markdown("ISO 17025 compliant PDF report with:")
        st.markdown("""
        - Module and equipment information
        - Measurement results with uncertainty
        - Detailed uncertainty budget
        - Signature section
        - Professional formatting
        """)

        if st.button("📄 Generate PDF Report", type="primary", use_container_width=True):
            try:
                with st.spinner("Generating PDF report..."):
                    # Create report generator
                    generator = ISO17025ReportGenerator(
                        company_name=company_name,
                        document_format=document_format,
                        lab_address=lab_address,
                        lab_accreditation=lab_accreditation,
                        lab_logo_path=logo_path
                    )

                    # Generate PDF
                    pdf_bytes = generator.generate_pdf_report(
                        uncertainty_result=st.session_state.calculation_results,
                        module_config=st.session_state.module_config or {},
                        simulator_config=st.session_state.simulator_config or {},
                        reference_config=st.session_state.reference_config or {},
                        measurement_data=st.session_state.measurement_data or {},
                        report_config=st.session_state.report_config
                    )

                    st.success("✅ PDF report generated successfully!")

                    # Download button
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"PV_Uncertainty_Report_{report_number}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"❌ Error generating PDF report: {str(e)}")
                st.exception(e)

    with col2:
        st.markdown("#### Excel Report")
        st.markdown("Comprehensive Excel workbook with:")
        st.markdown("""
        - Summary sheet with key results
        - Detailed uncertainty budget
        - Equipment specifications
        - Financial analysis (if included)
        - Easy to customize
        """)

        if st.button("📊 Generate Excel Report", type="primary", use_container_width=True):
            try:
                with st.spinner("Generating Excel report..."):
                    # Create report generator
                    generator = ISO17025ReportGenerator(
                        company_name=company_name,
                        document_format=document_format,
                        lab_address=lab_address,
                        lab_accreditation=lab_accreditation,
                        lab_logo_path=logo_path
                    )

                    # Generate Excel
                    excel_bytes = generator.generate_excel_report(
                        uncertainty_result=st.session_state.calculation_results,
                        module_config=st.session_state.module_config or {},
                        simulator_config=st.session_state.simulator_config or {},
                        reference_config=st.session_state.reference_config or {},
                        measurement_data=st.session_state.measurement_data or {},
                        report_config=st.session_state.report_config
                    )

                    st.success("✅ Excel report generated successfully!")

                    # Download button
                    st.download_button(
                        label="⬇️ Download Excel Report",
                        data=excel_bytes,
                        file_name=f"PV_Uncertainty_Report_{report_number}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"❌ Error generating Excel report: {str(e)}")
                st.exception(e)

    # Preview section
    st.markdown("---")
    st.markdown("### 👁️ Report Preview")

    with st.expander("Preview Report Content", expanded=False):
        st.markdown("#### Report Header")
        st.json({
            "Laboratory": company_name,
            "Report Number": report_number,
            "Client": client_name,
            "Test Date": test_date.strftime('%Y-%m-%d'),
            "Report Date": report_date.strftime('%Y-%m-%d')
        })

        if st.session_state.module_config:
            st.markdown("#### Module Configuration")
            st.json(st.session_state.module_config)

        if st.session_state.calculation_results:
            st.markdown("#### Key Results")
            result = st.session_state.calculation_results
            st.json({
                "Pmax": f"{result['pmax_measured']:.2f} W",
                "Uncertainty (k=2)": f"±{result['expanded_uncertainty_k2']:.2f}%",
                "Absolute Uncertainty": f"±{result['pmax_uncertainty_absolute']:.2f} W",
                "95% CI": f"[{result['pmax_confidence_interval_95'][0]:.2f}, {result['pmax_confidence_interval_95'][1]:.2f}] W"
            })


def main():
    """Main application function."""

    # Initialize session state
    initialize_session_state()

    # Header
    st.markdown('<div class="main-header">☀️ PV Measurement Uncertainty Tool - Professional Edition</div>',
                unsafe_allow_html=True)

    st.markdown("""
    **Complete uncertainty analysis platform for solar PV IV measurements**
    Implements GUM methodology (JCGM 100:2008) | ISO 17025 compliant reporting | Financial impact analysis
    """)

    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50/3b82f6/ffffff?text=PV+Uncertainty", use_container_width=True)

        st.markdown("### 📋 Navigation")
        st.markdown("""
        1️⃣ Module Configuration
        2️⃣ Sun Simulator
        3️⃣ Reference Device
        4️⃣ Measurement Data
        5️⃣ Uncertainty Analysis
        6️⃣ Results & Budget
        7️⃣ Financial Impact
        8️⃣ Professional Reporting
        """)

        st.markdown("---")
        st.markdown("### ⚙️ Current Configuration")

        if st.session_state.module_config:
            st.caption(f"**Technology:** {st.session_state.module_config.get('technology', 'N/A')}")
        if st.session_state.simulator_config:
            st.caption(f"**Simulator:** {st.session_state.simulator_config.get('model', 'N/A')}")
        if st.session_state.measurement_data:
            st.caption(f"**Pmax:** {st.session_state.measurement_data.get('pmax', 0):.2f} W")
        if st.session_state.calculation_results:
            st.caption(f"**Uncertainty:** ±{st.session_state.calculation_results.get('expanded_uncertainty_k2', 0):.2f}%")

        st.markdown("---")
        st.markdown("### 📖 Help & Info")
        if st.button("📚 User Guide", use_container_width=True):
            st.info("User guide will open here")
        if st.button("🔬 Standards Info", use_container_width=True):
            st.info("Standards information will open here")

        # Database status (Railway PostgreSQL)
        if DB_AVAILABLE:
            display_db_status_sidebar()

    # Main content tabs
    tabs = st.tabs([
        "1️⃣ Module",
        "2️⃣ Simulator",
        "3️⃣ Reference",
        "4️⃣ Data",
        "5️⃣ Uncertainty",
        "6️⃣ Results",
        "7️⃣ Financial",
        "8️⃣ Report"
    ])

    with tabs[0]:
        section_1_module_configuration()

    with tabs[1]:
        section_2_sun_simulator()

    with tabs[2]:
        section_3_reference_device()

    with tabs[3]:
        section_4_measurement_data()

    with tabs[4]:
        section_5_uncertainty_factors()

    with tabs[5]:
        section_6_results()

    with tabs[6]:
        section_7_financial_impact()

    with tabs[7]:
        section_8_professional_reporting()


if __name__ == "__main__":
    main()
