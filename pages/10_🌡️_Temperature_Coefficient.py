"""
Temperature Coefficient Uncertainty Analysis
Provides GUM-based uncertainty analysis for PV module temperature coefficients
(Alpha - Isc, Beta - Voc, Gamma - Pmax)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Page configuration
st.set_page_config(
    page_title="Temperature Coefficient Analysis - PV Uncertainty Tool",
    page_icon="🌡️",
    layout="wide"
)


# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================

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
    .coefficient-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #7dd3fc;
        margin: 0.5rem 0;
    }
    .result-highlight {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #22c55e;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA CLASSES FOR UNCERTAINTY CALCULATION
# =============================================================================

@dataclass
class TemperatureCoefficient:
    """Temperature coefficient with uncertainty."""
    name: str
    symbol: str
    value: float  # %/°C
    uncertainty: float  # Standard uncertainty in %/°C
    distribution: str = "normal"  # normal, rectangular, triangular
    unit: str = "%/°C"

    @property
    def expanded_uncertainty_k2(self) -> float:
        """Expanded uncertainty with k=2 (95% confidence)."""
        return self.uncertainty * 2.0

    @property
    def relative_uncertainty_percent(self) -> float:
        """Relative uncertainty as percentage of value."""
        if abs(self.value) < 1e-10:
            return 0.0
        return abs(self.uncertainty / self.value) * 100


@dataclass
class UncertaintySource:
    """Individual uncertainty source for temperature coefficient measurement."""
    name: str
    value: float
    standard_uncertainty: float
    distribution: str
    sensitivity_coefficient: float = 1.0
    notes: str = ""

    @property
    def variance_contribution(self) -> float:
        """Calculate variance contribution: (c_i * u_i)^2"""
        return (self.sensitivity_coefficient * self.standard_uncertainty) ** 2


# =============================================================================
# GUM UNCERTAINTY CALCULATION
# =============================================================================

class TempCoeffUncertaintyCalculator:
    """
    GUM-based uncertainty calculator for temperature coefficients.

    Follows IEC 60891 and IEC 61215-2 methodologies for temperature
    coefficient determination.
    """

    def __init__(self):
        self.sources: List[UncertaintySource] = []

    def add_source(self, source: UncertaintySource):
        """Add an uncertainty source."""
        self.sources.append(source)

    def clear_sources(self):
        """Clear all uncertainty sources."""
        self.sources = []

    def calculate_combined_uncertainty(self) -> Tuple[float, Dict]:
        """
        Calculate combined standard uncertainty using GUM methodology.

        u_c = sqrt(sum(c_i * u_i)^2)

        Returns:
            Tuple of (combined_uncertainty, detailed_budget_dict)
        """
        if not self.sources:
            return 0.0, {"components": [], "combined_standard_uncertainty": 0.0}

        # Calculate total variance
        total_variance = sum(s.variance_contribution for s in self.sources)
        combined_uncertainty = np.sqrt(total_variance)

        # Create detailed budget
        budget = {
            "combined_standard_uncertainty": combined_uncertainty,
            "expanded_uncertainty_k2": combined_uncertainty * 2.0,
            "components": []
        }

        # Add component details
        for source in self.sources:
            variance_contrib = source.variance_contribution
            percentage_contrib = (variance_contrib / total_variance * 100) if total_variance > 0 else 0

            budget["components"].append({
                "name": source.name,
                "value": source.value,
                "standard_uncertainty": source.standard_uncertainty,
                "distribution": source.distribution,
                "sensitivity_coefficient": source.sensitivity_coefficient,
                "variance_contribution": variance_contrib,
                "percentage_contribution": percentage_contrib,
                "notes": source.notes
            })

        # Sort by contribution (descending)
        budget["components"].sort(key=lambda x: x["percentage_contribution"], reverse=True)

        return combined_uncertainty, budget


def convert_to_standard_uncertainty(value: float, distribution: str) -> float:
    """
    Convert uncertainty value to standard uncertainty based on distribution.

    Args:
        value: The uncertainty value (half-width for rectangular/triangular)
        distribution: Distribution type

    Returns:
        Standard uncertainty
    """
    if distribution == "normal":
        return value  # Assume already standard uncertainty
    elif distribution == "rectangular":
        return value / np.sqrt(3)
    elif distribution == "triangular":
        return value / np.sqrt(6)
    elif distribution == "u-shaped":
        return value / np.sqrt(2)
    else:
        return value


def calculate_temp_coeff_uncertainty(
    coeff_value: float,
    measurement_sources: List[Dict]
) -> Tuple[float, Dict]:
    """
    Calculate uncertainty for a temperature coefficient.

    Args:
        coeff_value: The measured coefficient value (%/°C)
        measurement_sources: List of uncertainty source dictionaries

    Returns:
        Tuple of (combined_std_uncertainty, budget_dict)
    """
    calculator = TempCoeffUncertaintyCalculator()

    for src in measurement_sources:
        std_unc = convert_to_standard_uncertainty(
            src['uncertainty'],
            src.get('distribution', 'normal')
        )

        calculator.add_source(UncertaintySource(
            name=src['name'],
            value=src.get('value', 0.0),
            standard_uncertainty=std_unc,
            distribution=src.get('distribution', 'normal'),
            sensitivity_coefficient=src.get('sensitivity', 1.0),
            notes=src.get('notes', '')
        ))

    return calculator.calculate_combined_uncertainty()


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_uncertainty_budget_chart(budget: Dict, title: str = "Uncertainty Budget") -> go.Figure:
    """Create horizontal bar chart showing uncertainty contributions."""
    if "components" not in budget or not budget["components"]:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5,
                          showarrow=False, font=dict(size=16))
        return fig

    components = budget["components"]
    names = [comp["name"] for comp in components]
    contributions = [comp["percentage_contribution"] for comp in components]

    # Color scale
    colors = px.colors.sequential.Blues_r[:len(names)]
    if len(names) > len(colors):
        colors = colors * (len(names) // len(colors) + 1)
    colors = colors[:len(names)]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=names,
        x=contributions,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgb(8,48,107)', width=1.5)
        ),
        text=[f"{c:.1f}%" for c in contributions],
        textposition='auto',
        hovertemplate=(
            "<b>%{y}</b><br>" +
            "Contribution: %{x:.2f}%<br>" +
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Percentage Contribution (%)",
        yaxis_title="Uncertainty Source",
        height=max(350, len(names) * 45),
        template="plotly_white",
        showlegend=False,
        margin=dict(l=200, r=50, t=60, b=50)
    )

    return fig


def create_comparison_chart(coefficients: List[TemperatureCoefficient]) -> go.Figure:
    """Create comparison chart for all temperature coefficients."""
    if not coefficients:
        fig = go.Figure()
        fig.add_annotation(text="No coefficients to display", x=0.5, y=0.5,
                          showarrow=False, font=dict(size=16))
        return fig

    names = [f"{c.symbol} ({c.name})" for c in coefficients]
    values = [c.value for c in coefficients]
    uncertainties = [c.expanded_uncertainty_k2 for c in coefficients]

    # Color based on sign
    colors = ['#ef4444' if v < 0 else '#22c55e' for v in values]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=names,
        y=values,
        error_y=dict(
            type='data',
            array=uncertainties,
            visible=True,
            color='black',
            thickness=2,
            width=8
        ),
        marker=dict(
            color=colors,
            line=dict(color='black', width=1.5)
        ),
        text=[f"{v:.4f} ± {u:.4f}" for v, u in zip(values, uncertainties)],
        textposition='outside',
        hovertemplate=(
            "<b>%{x}</b><br>" +
            "Value: %{y:.4f} %/°C<br>" +
            "Expanded Uncertainty (k=2): ±%{error_y.array:.4f} %/°C<br>" +
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        title="Temperature Coefficients Comparison (with Expanded Uncertainty k=2)",
        yaxis_title="Temperature Coefficient (%/°C)",
        template="plotly_white",
        showlegend=False,
        height=450,
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray')
    )

    return fig


def create_pie_chart(budget: Dict, title: str = "Uncertainty Distribution") -> go.Figure:
    """Create pie chart showing uncertainty breakdown."""
    if "components" not in budget or not budget["components"]:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5,
                          showarrow=False, font=dict(size=16))
        return fig

    components = budget["components"]
    names = [comp["name"] for comp in components]
    contributions = [comp["percentage_contribution"] for comp in components]

    fig = go.Figure(data=[go.Pie(
        labels=names,
        values=contributions,
        hole=0.35,
        marker=dict(
            colors=px.colors.sequential.Blues_r,
            line=dict(color='white', width=2)
        ),
        textinfo='label+percent',
        textposition='outside',
        hovertemplate=(
            "<b>%{label}</b><br>" +
            "Contribution: %{value:.2f}%<br>" +
            "<extra></extra>"
        )
    )])

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=400,
        showlegend=True,
        legend=dict(x=1.02, y=0.5)
    )

    return fig


def create_sensitivity_chart(
    coeff_value: float,
    coeff_uncertainty: float,
    temp_range: Tuple[float, float] = (-20, 80)
) -> go.Figure:
    """Create chart showing power/voltage/current deviation across temperature range."""
    temps = np.linspace(temp_range[0], temp_range[1], 100)
    stc_temp = 25.0

    # Calculate deviation from STC
    delta_t = temps - stc_temp
    deviation = coeff_value * delta_t  # % deviation

    # Uncertainty bands
    upper = (coeff_value + coeff_uncertainty) * delta_t
    lower = (coeff_value - coeff_uncertainty) * delta_t

    fig = go.Figure()

    # Uncertainty band
    fig.add_trace(go.Scatter(
        x=np.concatenate([temps, temps[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Uncertainty Band (k=1)',
        hoverinfo='skip'
    ))

    # Main line
    fig.add_trace(go.Scatter(
        x=temps,
        y=deviation,
        mode='lines',
        name='Deviation from STC',
        line=dict(color='#1e40af', width=2),
        hovertemplate=(
            "Temperature: %{x:.1f}°C<br>" +
            "Deviation: %{y:.2f}%<br>" +
            "<extra></extra>"
        )
    ))

    # STC reference line
    fig.add_vline(x=25, line=dict(color='green', dash='dash', width=1.5),
                  annotation_text="STC (25°C)", annotation_position="top")

    # Zero line
    fig.add_hline(y=0, line=dict(color='gray', width=1))

    fig.update_layout(
        title="Parameter Deviation from STC vs Temperature",
        xaxis_title="Temperature (°C)",
        yaxis_title="Deviation from STC (%)",
        template="plotly_white",
        height=400,
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )

    return fig


def create_combined_dashboard(
    alpha: TemperatureCoefficient,
    beta: TemperatureCoefficient,
    gamma: TemperatureCoefficient,
    budgets: Dict[str, Dict]
) -> go.Figure:
    """Create combined dashboard with all coefficients."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Temperature Coefficients Comparison",
            "Alpha (Isc) Uncertainty Budget",
            "Beta (Voc) Uncertainty Budget",
            "Gamma (Pmax) Uncertainty Budget"
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )

    # Comparison chart (top-left)
    coeffs = [alpha, beta, gamma]
    names = [f"{c.symbol}" for c in coeffs]
    values = [c.value for c in coeffs]
    colors = ['#ef4444' if v < 0 else '#22c55e' for v in values]

    fig.add_trace(
        go.Bar(
            x=names,
            y=values,
            marker=dict(color=colors),
            showlegend=False,
            text=[f"{v:.4f}" for v in values],
            textposition='outside'
        ),
        row=1, col=1
    )

    # Individual budgets
    for idx, (key, budget) in enumerate(budgets.items()):
        if "components" in budget and budget["components"]:
            components = budget["components"][:5]  # Top 5 contributors
            comp_names = [c["name"][:20] for c in components]
            contrib = [c["percentage_contribution"] for c in components]

            row = 1 if idx == 0 else 2
            col = 2 if idx == 0 else (1 if idx == 1 else 2)

            fig.add_trace(
                go.Bar(
                    y=comp_names,
                    x=contrib,
                    orientation='h',
                    marker=dict(color=px.colors.sequential.Blues_r[:len(comp_names)]),
                    showlegend=False
                ),
                row=row, col=col
            )

    fig.update_layout(
        title_text="Temperature Coefficient Analysis Dashboard",
        template="plotly_white",
        height=700,
        showlegend=False
    )

    return fig


# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def save_to_database(
    coefficients: Dict[str, TemperatureCoefficient],
    budgets: Dict[str, Dict],
    module_info: Dict
) -> Tuple[bool, str]:
    """
    Save temperature coefficient analysis results to database.

    Args:
        coefficients: Dictionary of TemperatureCoefficient objects
        budgets: Uncertainty budgets for each coefficient
        module_info: Module identification info

    Returns:
        Tuple of (success, message)
    """
    try:
        from database.connection import session_scope, check_connection
        from database.models import (
            UncertaintyResult, UncertaintyComponent, MeasurementType
        )

        if not check_connection():
            return False, "Database not connected"

        with session_scope() as session:
            # Create result for each coefficient
            for coeff_name, coeff in coefficients.items():
                budget = budgets.get(coeff_name, {})

                # Create uncertainty result
                result = UncertaintyResult(
                    analysis_date=datetime.utcnow(),
                    analysis_version="1.0",
                    calculation_method="GUM",
                    target_parameter=f"Temperature Coefficient {coeff.symbol}",
                    measured_value=coeff.value,
                    measured_unit=coeff.unit,
                    combined_standard_uncertainty_pct=coeff.uncertainty,
                    expanded_uncertainty_k2_pct=coeff.expanded_uncertainty_k2,
                    absolute_uncertainty=coeff.uncertainty,
                    confidence_level_pct=95.0,
                    coverage_factor=2.0,
                    full_budget_json=budget
                )

                session.add(result)
                session.flush()  # Get the ID

                # Add components
                if "components" in budget:
                    for idx, comp in enumerate(budget["components"]):
                        component = UncertaintyComponent(
                            uncertainty_result_id=result.id,
                            category_id="TC",
                            subcategory_id=f"TC.{coeff_name}",
                            factor_id=f"TC.{coeff_name}.{idx+1}",
                            name=comp["name"],
                            input_value=comp.get("value", 0.0),
                            standard_uncertainty=comp["standard_uncertainty"],
                            distribution=comp["distribution"],
                            sensitivity_coefficient=comp.get("sensitivity_coefficient", 1.0),
                            variance_contribution=comp["variance_contribution"],
                            percentage_contribution=comp["percentage_contribution"],
                            notes=comp.get("notes", "")
                        )
                        session.add(component)

            session.commit()
            return True, "Results saved successfully to database"

    except ImportError:
        return False, "Database module not available"
    except Exception as e:
        return False, f"Database error: {str(e)}"


def get_database_status() -> Dict:
    """Get database connection status."""
    try:
        from database.connection import check_connection, get_database_url
        from urllib.parse import urlparse

        url = get_database_url()
        parsed = urlparse(url)

        return {
            'available': True,
            'connected': check_connection(),
            'host': parsed.hostname,
            'database': parsed.path.lstrip('/') if parsed.path else None
        }
    except Exception:
        return {
            'available': False,
            'connected': False,
            'host': None,
            'database': None
        }


# =============================================================================
# UI COMPONENTS
# =============================================================================

def display_header():
    """Display page header."""
    st.markdown(
        '<div class="main-header">🌡️ Temperature Coefficient Uncertainty Analysis</div>',
        unsafe_allow_html=True
    )
    st.markdown("""
    Comprehensive GUM-based uncertainty analysis for PV module temperature coefficients
    following IEC 60891 and IEC 61215-2 methodologies.
    """)


def display_info_section():
    """Display information about temperature coefficients."""
    with st.expander("ℹ️ About Temperature Coefficients", expanded=False):
        st.markdown("""
        ### Temperature Coefficients for PV Modules

        Temperature coefficients describe how a PV module's electrical parameters change with temperature:

        | Coefficient | Symbol | Parameter | Typical Range | Description |
        |------------|--------|-----------|---------------|-------------|
        | **Alpha** | α | Isc | +0.03 to +0.06 %/°C | Short-circuit current increases slightly with temperature |
        | **Beta** | β | Voc | -0.25 to -0.35 %/°C | Open-circuit voltage decreases significantly with temperature |
        | **Gamma** | γ | Pmax | -0.30 to -0.50 %/°C | Maximum power decreases with temperature (dominated by Voc) |

        ### Uncertainty Sources

        Key uncertainty sources in temperature coefficient measurements include:

        1. **Temperature Measurement**: Sensor calibration, thermal contact, uniformity
        2. **Irradiance Setting**: Simulator stability, reference device calibration
        3. **I-V Measurement**: Voltage and current measurement accuracy
        4. **Curve Fitting**: Regression analysis uncertainty
        5. **Environmental Control**: Temperature stabilization, ambient conditions

        ### Standards Reference
        - **IEC 60891**: Procedures for temperature and irradiance corrections
        - **IEC 61215-2**: Design qualification and type approval testing
        """)


def display_coefficient_input(
    name: str,
    symbol: str,
    default_value: float,
    default_unc: float,
    key_prefix: str
) -> Tuple[float, float, str, List[Dict]]:
    """
    Display input section for a single temperature coefficient.

    Returns:
        Tuple of (value, uncertainty, distribution, uncertainty_sources)
    """
    st.markdown(f'<div class="subsection-header">{symbol} - {name} Temperature Coefficient</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        value = st.number_input(
            f"{symbol} Value (%/°C)",
            min_value=-1.0,
            max_value=1.0,
            value=default_value,
            step=0.001,
            format="%.4f",
            key=f"{key_prefix}_value",
            help=f"Measured {name} temperature coefficient"
        )

    with col2:
        total_unc = st.number_input(
            f"{symbol} Combined Uncertainty (%/°C)",
            min_value=0.0,
            max_value=0.5,
            value=default_unc,
            step=0.001,
            format="%.4f",
            key=f"{key_prefix}_unc",
            help="Combined standard uncertainty (or enter individual sources below)"
        )

    with col3:
        distribution = st.selectbox(
            "Distribution",
            options=["normal", "rectangular", "triangular"],
            index=0,
            key=f"{key_prefix}_dist",
            help="Probability distribution of uncertainty"
        )

    # Detailed uncertainty sources
    uncertainty_sources = []

    with st.expander(f"📊 Detailed Uncertainty Sources for {symbol}", expanded=False):
        st.markdown("Enter individual uncertainty sources (optional - overrides combined uncertainty above)")

        use_detailed = st.checkbox(
            "Use detailed uncertainty breakdown",
            key=f"{key_prefix}_use_detailed",
            value=False
        )

        if use_detailed:
            # Temperature measurement uncertainty
            col1, col2, col3 = st.columns(3)
            with col1:
                temp_unc = st.number_input(
                    "Temperature Sensor (°C)",
                    min_value=0.0, max_value=5.0,
                    value=0.5, step=0.1,
                    key=f"{key_prefix}_temp_unc",
                    help="Temperature measurement uncertainty"
                )
            with col2:
                temp_sens = st.number_input(
                    "Sensitivity (°C⁻¹)",
                    min_value=0.0, max_value=0.1,
                    value=0.01, step=0.001, format="%.4f",
                    key=f"{key_prefix}_temp_sens",
                    help="Sensitivity of coefficient to temperature error"
                )
            with col3:
                temp_dist = st.selectbox(
                    "Distribution",
                    ["normal", "rectangular", "triangular"],
                    key=f"{key_prefix}_temp_dist"
                )

            if temp_unc > 0:
                uncertainty_sources.append({
                    'name': 'Temperature Measurement',
                    'value': temp_unc,
                    'uncertainty': temp_unc * temp_sens,
                    'distribution': temp_dist,
                    'sensitivity': 1.0,
                    'notes': 'Temperature sensor calibration and thermal contact'
                })

            # Irradiance uncertainty
            col1, col2, col3 = st.columns(3)
            with col1:
                irr_unc = st.number_input(
                    "Irradiance Setting (%)",
                    min_value=0.0, max_value=5.0,
                    value=1.0, step=0.1,
                    key=f"{key_prefix}_irr_unc",
                    help="Irradiance measurement/setting uncertainty"
                )
            with col2:
                irr_sens = st.number_input(
                    "Sensitivity (%/%)",
                    min_value=0.0, max_value=1.0,
                    value=0.01, step=0.001, format="%.4f",
                    key=f"{key_prefix}_irr_sens",
                    help="Effect of irradiance error on coefficient"
                )
            with col3:
                irr_dist = st.selectbox(
                    "Distribution",
                    ["normal", "rectangular", "triangular"],
                    key=f"{key_prefix}_irr_dist"
                )

            if irr_unc > 0:
                uncertainty_sources.append({
                    'name': 'Irradiance Setting',
                    'value': irr_unc,
                    'uncertainty': irr_unc * irr_sens / 100,
                    'distribution': irr_dist,
                    'sensitivity': 1.0,
                    'notes': 'Sun simulator irradiance stability and calibration'
                })

            # I-V measurement uncertainty
            col1, col2, col3 = st.columns(3)
            with col1:
                iv_unc = st.number_input(
                    "I-V Measurement (%)",
                    min_value=0.0, max_value=2.0,
                    value=0.2, step=0.05,
                    key=f"{key_prefix}_iv_unc",
                    help="Voltage/current measurement uncertainty"
                )
            with col2:
                iv_sens = st.number_input(
                    "Sensitivity",
                    min_value=0.0, max_value=1.0,
                    value=0.5, step=0.1,
                    key=f"{key_prefix}_iv_sens",
                    help="Contribution to coefficient uncertainty"
                )
            with col3:
                iv_dist = st.selectbox(
                    "Distribution",
                    ["normal", "rectangular", "triangular"],
                    key=f"{key_prefix}_iv_dist"
                )

            if iv_unc > 0:
                uncertainty_sources.append({
                    'name': 'I-V Measurement',
                    'value': iv_unc,
                    'uncertainty': iv_unc * iv_sens / 100,
                    'distribution': iv_dist,
                    'sensitivity': 1.0,
                    'notes': 'Voltage and current measurement accuracy'
                })

            # Curve fitting uncertainty
            col1, col2, col3 = st.columns(3)
            with col1:
                fit_unc = st.number_input(
                    "Curve Fitting (%/°C)",
                    min_value=0.0, max_value=0.1,
                    value=0.005, step=0.001, format="%.4f",
                    key=f"{key_prefix}_fit_unc",
                    help="Regression analysis uncertainty"
                )
            with col2:
                fit_r2 = st.number_input(
                    "R² Value",
                    min_value=0.9, max_value=1.0,
                    value=0.998, step=0.001, format="%.4f",
                    key=f"{key_prefix}_fit_r2",
                    help="Coefficient of determination"
                )
            with col3:
                fit_dist = st.selectbox(
                    "Distribution",
                    ["normal", "rectangular", "triangular"],
                    key=f"{key_prefix}_fit_dist"
                )

            if fit_unc > 0:
                uncertainty_sources.append({
                    'name': 'Curve Fitting/Regression',
                    'value': fit_r2,
                    'uncertainty': fit_unc,
                    'distribution': fit_dist,
                    'sensitivity': 1.0,
                    'notes': f'Linear regression uncertainty (R² = {fit_r2:.4f})'
                })

            # Repeatability
            col1, col2 = st.columns(2)
            with col1:
                repeat_unc = st.number_input(
                    "Repeatability (%/°C)",
                    min_value=0.0, max_value=0.1,
                    value=0.003, step=0.001, format="%.4f",
                    key=f"{key_prefix}_repeat_unc",
                    help="Standard deviation from repeated measurements"
                )
            with col2:
                repeat_n = st.number_input(
                    "Number of Measurements",
                    min_value=1, max_value=100,
                    value=5, step=1,
                    key=f"{key_prefix}_repeat_n",
                    help="Number of independent measurements"
                )

            if repeat_unc > 0:
                uncertainty_sources.append({
                    'name': 'Repeatability',
                    'value': repeat_n,
                    'uncertainty': repeat_unc / np.sqrt(repeat_n),
                    'distribution': 'normal',
                    'sensitivity': 1.0,
                    'notes': f'Standard deviation from {repeat_n} measurements'
                })

    # If not using detailed sources, create a single source from the combined uncertainty
    if not uncertainty_sources:
        uncertainty_sources.append({
            'name': 'Combined Uncertainty',
            'value': total_unc,
            'uncertainty': total_unc,
            'distribution': distribution,
            'sensitivity': 1.0,
            'notes': 'User-provided combined standard uncertainty'
        })

    return value, total_unc, distribution, uncertainty_sources


def display_results(
    coefficients: Dict[str, TemperatureCoefficient],
    budgets: Dict[str, Dict]
):
    """Display calculation results."""
    st.markdown('<div class="section-header">📊 Analysis Results</div>',
                unsafe_allow_html=True)

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    for idx, (key, coeff) in enumerate(coefficients.items()):
        with [col1, col2, col3][idx]:
            st.markdown(f"""
            <div class="coefficient-card">
                <h4>{coeff.symbol} - {coeff.name}</h4>
                <p><strong>Value:</strong> {coeff.value:.4f} {coeff.unit}</p>
                <p><strong>Standard Uncertainty:</strong> ±{coeff.uncertainty:.4f} {coeff.unit}</p>
                <p><strong>Expanded Uncertainty (k=2):</strong> ±{coeff.expanded_uncertainty_k2:.4f} {coeff.unit}</p>
                <p><strong>Relative Uncertainty:</strong> {coeff.relative_uncertainty_percent:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ISO 17025 format results
    st.markdown('<div class="subsection-header">ISO 17025 Format Results</div>',
                unsafe_allow_html=True)

    results_text = ""
    for key, coeff in coefficients.items():
        results_text += f"""
**{coeff.name} Temperature Coefficient ({coeff.symbol}):**
{coeff.symbol} = {coeff.value:.4f} {coeff.unit} ± {coeff.expanded_uncertainty_k2:.4f} {coeff.unit} (k=2, 95% confidence)

"""

    st.success(results_text)

    # Comparison chart
    st.markdown('<div class="subsection-header">Coefficients Comparison</div>',
                unsafe_allow_html=True)

    fig_comparison = create_comparison_chart(list(coefficients.values()))
    st.plotly_chart(fig_comparison, use_container_width=True)

    # Individual budget charts
    st.markdown('<div class="subsection-header">Uncertainty Budgets</div>',
                unsafe_allow_html=True)

    tabs = st.tabs([f"{coeff.symbol} Budget" for coeff in coefficients.values()])

    for idx, (tab, (key, coeff)) in enumerate(zip(tabs, coefficients.items())):
        with tab:
            budget = budgets.get(key, {})

            col1, col2 = st.columns(2)

            with col1:
                fig_bar = create_uncertainty_budget_chart(
                    budget,
                    f"{coeff.symbol} Uncertainty Budget"
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with col2:
                fig_pie = create_pie_chart(
                    budget,
                    f"{coeff.symbol} Contribution Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            # Sensitivity/deviation chart
            fig_sens = create_sensitivity_chart(
                coeff.value,
                coeff.uncertainty,
                temp_range=(-20, 80)
            )
            st.plotly_chart(fig_sens, use_container_width=True)

            # Budget table
            if "components" in budget and budget["components"]:
                st.markdown("**Detailed Budget Table:**")
                df = pd.DataFrame(budget["components"])
                df = df[['name', 'standard_uncertainty', 'distribution',
                        'sensitivity_coefficient', 'percentage_contribution', 'notes']]
                df.columns = ['Source', 'Std. Uncertainty', 'Distribution',
                             'Sensitivity', 'Contribution (%)', 'Notes']
                st.dataframe(df, use_container_width=True, hide_index=True)


def display_export_section(
    coefficients: Dict[str, TemperatureCoefficient],
    budgets: Dict[str, Dict]
):
    """Display export and save options."""
    st.markdown('<div class="section-header">💾 Export & Save</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        # CSV export
        export_data = []
        for key, coeff in coefficients.items():
            budget = budgets.get(key, {})
            export_data.append({
                'Coefficient': coeff.name,
                'Symbol': coeff.symbol,
                'Value (%/°C)': coeff.value,
                'Standard Uncertainty': coeff.uncertainty,
                'Expanded Uncertainty (k=2)': coeff.expanded_uncertainty_k2,
                'Relative Uncertainty (%)': coeff.relative_uncertainty_percent,
                'Distribution': coeff.distribution
            })

        df_export = pd.DataFrame(export_data)
        csv = df_export.to_csv(index=False)

        st.download_button(
            label="📥 Download Summary (CSV)",
            data=csv,
            file_name=f"temp_coeff_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    with col2:
        # JSON export (full budget)
        import json

        export_json = {
            'analysis_date': datetime.now().isoformat(),
            'analysis_type': 'Temperature Coefficient Uncertainty',
            'methodology': 'GUM (Guide to the Expression of Uncertainty in Measurement)',
            'coefficients': {},
            'budgets': budgets
        }

        for key, coeff in coefficients.items():
            export_json['coefficients'][key] = {
                'name': coeff.name,
                'symbol': coeff.symbol,
                'value': coeff.value,
                'unit': coeff.unit,
                'standard_uncertainty': coeff.uncertainty,
                'expanded_uncertainty_k2': coeff.expanded_uncertainty_k2,
                'relative_uncertainty_percent': coeff.relative_uncertainty_percent,
                'distribution': coeff.distribution
            }

        json_str = json.dumps(export_json, indent=2)

        st.download_button(
            label="📥 Download Full Report (JSON)",
            data=json_str,
            file_name=f"temp_coeff_full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    with col3:
        # Database save
        db_status = get_database_status()

        if db_status['connected']:
            st.success("Database: Connected")

            if st.button("💾 Save to Database", type="primary"):
                success, message = save_to_database(
                    coefficients,
                    budgets,
                    {}  # Module info placeholder
                )

                if success:
                    st.success(message)
                else:
                    st.error(message)
        else:
            st.warning("Database: Not connected")
            st.info("Configure database in Admin page to enable saving")


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def initialize_session_state():
    """Initialize session state variables."""
    if 'temp_coeff_calculated' not in st.session_state:
        st.session_state.temp_coeff_calculated = False
    if 'temp_coeff_results' not in st.session_state:
        st.session_state.temp_coeff_results = None
    if 'temp_coeff_budgets' not in st.session_state:
        st.session_state.temp_coeff_budgets = None


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    initialize_session_state()

    # Header
    display_header()
    st.markdown("---")

    # Info section
    display_info_section()

    # Input section
    st.markdown('<div class="section-header">📝 Input Parameters</div>',
                unsafe_allow_html=True)

    # Module identification (optional)
    with st.expander("🔖 Module Identification (Optional)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            module_id = st.text_input("Module Serial Number", value="",
                                       help="Optional module identifier")
            manufacturer = st.text_input("Manufacturer", value="",
                                         help="Module manufacturer")
        with col2:
            model = st.text_input("Model", value="",
                                  help="Module model name")
            technology = st.selectbox(
                "Technology",
                ["PERC", "TOPCon", "HJT", "CIGS", "CdTe", "Perovskite", "Custom"],
                help="PV technology type"
            )

    # Technology-based defaults
    tech_defaults = {
        "PERC": {"alpha": 0.05, "beta": -0.30, "gamma": -0.37},
        "TOPCon": {"alpha": 0.04, "beta": -0.27, "gamma": -0.34},
        "HJT": {"alpha": 0.03, "beta": -0.23, "gamma": -0.26},
        "CIGS": {"alpha": 0.01, "beta": -0.30, "gamma": -0.36},
        "CdTe": {"alpha": 0.04, "beta": -0.25, "gamma": -0.32},
        "Perovskite": {"alpha": 0.02, "beta": -0.15, "gamma": -0.20},
        "Custom": {"alpha": 0.05, "beta": -0.30, "gamma": -0.40}
    }

    defaults = tech_defaults.get(technology, tech_defaults["Custom"])

    st.markdown("---")

    # Alpha (Isc) input
    alpha_value, alpha_unc, alpha_dist, alpha_sources = display_coefficient_input(
        "Isc (Short-circuit Current)",
        "α",
        defaults["alpha"],
        0.01,
        "alpha"
    )

    st.markdown("---")

    # Beta (Voc) input
    beta_value, beta_unc, beta_dist, beta_sources = display_coefficient_input(
        "Voc (Open-circuit Voltage)",
        "β",
        defaults["beta"],
        0.02,
        "beta"
    )

    st.markdown("---")

    # Gamma (Pmax) input
    gamma_value, gamma_unc, gamma_dist, gamma_sources = display_coefficient_input(
        "Pmax (Maximum Power)",
        "γ",
        defaults["gamma"],
        0.03,
        "gamma"
    )

    st.markdown("---")

    # Calculate button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        calculate_clicked = st.button(
            "🧮 Calculate Uncertainty",
            type="primary",
            use_container_width=True
        )

    if calculate_clicked:
        # Perform calculations
        with st.spinner("Calculating uncertainties..."):
            # Calculate uncertainties for each coefficient
            alpha_combined_unc, alpha_budget = calculate_temp_coeff_uncertainty(
                alpha_value, alpha_sources
            )
            beta_combined_unc, beta_budget = calculate_temp_coeff_uncertainty(
                beta_value, beta_sources
            )
            gamma_combined_unc, gamma_budget = calculate_temp_coeff_uncertainty(
                gamma_value, gamma_sources
            )

            # Create coefficient objects
            coefficients = {
                'alpha': TemperatureCoefficient(
                    name="Isc",
                    symbol="α",
                    value=alpha_value,
                    uncertainty=alpha_combined_unc,
                    distribution=alpha_dist
                ),
                'beta': TemperatureCoefficient(
                    name="Voc",
                    symbol="β",
                    value=beta_value,
                    uncertainty=beta_combined_unc,
                    distribution=beta_dist
                ),
                'gamma': TemperatureCoefficient(
                    name="Pmax",
                    symbol="γ",
                    value=gamma_value,
                    uncertainty=gamma_combined_unc,
                    distribution=gamma_dist
                )
            }

            budgets = {
                'alpha': alpha_budget,
                'beta': beta_budget,
                'gamma': gamma_budget
            }

            # Store in session state
            st.session_state.temp_coeff_calculated = True
            st.session_state.temp_coeff_results = coefficients
            st.session_state.temp_coeff_budgets = budgets

    # Display results if calculated
    if st.session_state.temp_coeff_calculated:
        coefficients = st.session_state.temp_coeff_results
        budgets = st.session_state.temp_coeff_budgets

        display_results(coefficients, budgets)
        st.markdown("---")
        display_export_section(coefficients, budgets)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 0.9rem;">
        Temperature Coefficient Uncertainty Analysis Tool |
        Based on GUM methodology (JCGM 100:2008) |
        Standards: IEC 60891, IEC 61215-2
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
