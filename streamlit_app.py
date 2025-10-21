"""
PV Uncertainty Tool - Streamlit Application
Professional tool for calculating measurement uncertainty in PV systems.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional
import io

# Import custom modules
from uncertainty_calculator import (
    PVUncertaintyCalculator,
    PVPowerUncertainty,
    PVPerformanceRatioUncertainty
)
from monte_carlo import MonteCarloSimulator, PVMonteCarlo
from visualizations import UncertaintyVisualizer
from data_handler import (
    PVDataValidator,
    PVDataHandler,
    ResultsFormatter
)


# Page configuration
st.set_page_config(
    page_title="PV Uncertainty Tool",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    """Main application function."""

    # Header
    st.markdown('<p class="main-header">☀️ PV Measurement Uncertainty Tool</p>', unsafe_allow_html=True)

    st.markdown("""
    **Professional uncertainty analysis for photovoltaic systems**

    This tool implements:
    - GUM (Guide to the Expression of Uncertainty in Measurement) methodology
    - Monte Carlo simulation for uncertainty propagation
    - Comprehensive sensitivity analysis
    - Visual uncertainty budget analysis
    """)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    analysis_type = st.sidebar.radio(
        "Select Analysis Type:",
        [
            "Power Measurement Uncertainty",
            "Performance Ratio Uncertainty",
            "Custom Uncertainty Analysis",
            "Monte Carlo Simulation",
            "Batch Data Analysis"
        ]
    )

    # Main content based on selection
    if analysis_type == "Power Measurement Uncertainty":
        power_uncertainty_analysis()

    elif analysis_type == "Performance Ratio Uncertainty":
        performance_ratio_analysis()

    elif analysis_type == "Custom Uncertainty Analysis":
        custom_uncertainty_analysis()

    elif analysis_type == "Monte Carlo Simulation":
        monte_carlo_analysis()

    elif analysis_type == "Batch Data Analysis":
        batch_data_analysis()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    **Version:** 1.0.0
    **Method:** GUM & Monte Carlo
    **Developed for:** Streamlit/Snowflake

    ### References
    - JCGM 100:2008 (GUM)
    - JCGM 101:2008 (GUM Supplement 1)
    - IEC 61724-1:2021 (PV Monitoring)
    """)


def power_uncertainty_analysis():
    """Power measurement uncertainty analysis interface."""

    st.header("Power Measurement Uncertainty Analysis")

    st.markdown("""
    Calculate combined uncertainty for PV power measurements considering:
    - Irradiance measurement uncertainty
    - Temperature measurement uncertainty
    - Power meter uncertainty
    - Module efficiency uncertainty
    """)

    # Create two columns for inputs
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Measurement Values")

        irradiance = st.number_input(
            "Irradiance (W/m²)",
            min_value=0.0,
            max_value=2000.0,
            value=1000.0,
            step=10.0,
            help="Measured irradiance on module plane"
        )

        temperature = st.number_input(
            "Module Temperature (°C)",
            min_value=-40.0,
            max_value=100.0,
            value=45.0,
            step=1.0,
            help="Measured module temperature"
        )

        power = st.number_input(
            "Measured Power (W)",
            min_value=0.0,
            max_value=100000.0,
            value=200.0,
            step=10.0,
            help="Measured DC power output"
        )

        module_efficiency = st.number_input(
            "Module Efficiency",
            min_value=0.0,
            max_value=1.0,
            value=0.20,
            step=0.01,
            format="%.3f",
            help="Module efficiency (fraction, e.g., 0.20 for 20%)"
        )

    with col2:
        st.subheader("Uncertainty Values (Standard)")

        irradiance_unc = st.number_input(
            "Irradiance Uncertainty (W/m²)",
            min_value=0.0,
            max_value=200.0,
            value=20.0,
            step=1.0,
            help="Standard uncertainty of irradiance measurement"
        )

        temperature_unc = st.number_input(
            "Temperature Uncertainty (°C)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Standard uncertainty of temperature measurement"
        )

        power_meter_unc = st.number_input(
            "Power Meter Uncertainty (W)",
            min_value=0.0,
            max_value=100.0,
            value=2.0,
            step=0.1,
            help="Standard uncertainty of power meter"
        )

        efficiency_unc = st.number_input(
            "Efficiency Uncertainty",
            min_value=0.0,
            max_value=0.1,
            value=0.01,
            step=0.001,
            format="%.4f",
            help="Standard uncertainty in module efficiency"
        )

    # Advanced parameters
    with st.expander("Advanced Parameters"):
        temp_coefficient = st.number_input(
            "Temperature Coefficient (%/°C)",
            min_value=-1.0,
            max_value=0.0,
            value=-0.4,
            step=0.01,
            format="%.3f",
            help="Module power temperature coefficient"
        ) / 100  # Convert to fraction

        reference_temp = st.number_input(
            "Reference Temperature (°C)",
            min_value=0.0,
            max_value=50.0,
            value=25.0,
            step=1.0,
            help="STC reference temperature"
        )

    # Calculate button
    if st.button("Calculate Uncertainty", type="primary"):
        # Validate inputs
        validator = PVDataValidator()
        validation = validator.validate_uncertainty_inputs(
            irradiance, irradiance_unc,
            temperature, temperature_unc,
            power, power_meter_unc
        )

        # Display validation results
        if not validation.is_valid:
            for error in validation.errors:
                st.error(f"❌ {error}")
            return

        if validation.warnings:
            for warning in validation.warnings:
                st.warning(f"⚠️ {warning}")

        # Calculate uncertainty
        with st.spinner("Calculating uncertainty..."):
            result = PVPowerUncertainty.calculate_power_uncertainty(
                irradiance=irradiance,
                irradiance_uncertainty=irradiance_unc,
                temperature=temperature,
                temperature_uncertainty=temperature_unc,
                power=power,
                power_meter_uncertainty=power_meter_unc,
                module_efficiency=module_efficiency,
                efficiency_uncertainty=efficiency_unc,
                temp_coefficient=temp_coefficient,
                reference_temp=reference_temp
            )

        # Display results
        display_uncertainty_results(result, "Power (W)")


def performance_ratio_analysis():
    """Performance Ratio uncertainty analysis interface."""

    st.header("Performance Ratio Uncertainty Analysis")

    st.markdown("""
    Calculate uncertainty for Performance Ratio (PR):

    **PR = E_measured / (H × P_installed)**

    Where:
    - E_measured: Measured energy output
    - H: Total irradiation on array plane
    - P_installed: Installed DC capacity
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Measurement Values")

        measured_energy = st.number_input(
            "Measured Energy (kWh)",
            min_value=0.0,
            value=1000.0,
            step=10.0,
            help="Total energy output during measurement period"
        )

        irradiation = st.number_input(
            "Total Irradiation (kWh/m²)",
            min_value=0.0,
            value=5.5,
            step=0.1,
            help="Total in-plane irradiation during measurement period"
        )

        installed_capacity = st.number_input(
            "Installed Capacity (kWp)",
            min_value=0.0,
            value=250.0,
            step=10.0,
            help="Installed DC capacity at STC"
        )

    with col2:
        st.subheader("Uncertainty Values (Standard)")

        energy_unc = st.number_input(
            "Energy Uncertainty (kWh)",
            min_value=0.0,
            value=10.0,
            step=1.0,
            help="Standard uncertainty in energy measurement"
        )

        irradiation_unc = st.number_input(
            "Irradiation Uncertainty (kWh/m²)",
            min_value=0.0,
            value=0.2,
            step=0.01,
            help="Standard uncertainty in irradiation measurement"
        )

        capacity_unc = st.number_input(
            "Capacity Uncertainty (kWp)",
            min_value=0.0,
            value=5.0,
            step=0.5,
            help="Standard uncertainty in installed capacity"
        )

    if st.button("Calculate PR Uncertainty", type="primary"):
        # Calculate PR
        if irradiation > 0 and installed_capacity > 0:
            pr = measured_energy / (irradiation * installed_capacity)
        else:
            st.error("Irradiation and Installed Capacity must be greater than 0")
            return

        # Calculate uncertainty
        with st.spinner("Calculating PR uncertainty..."):
            result = PVPerformanceRatioUncertainty.calculate_pr_uncertainty(
                measured_energy=measured_energy,
                measured_energy_uncertainty=energy_unc,
                irradiation=irradiation,
                irradiation_uncertainty=irradiation_unc,
                installed_capacity=installed_capacity,
                capacity_uncertainty=capacity_unc,
                performance_ratio=pr
            )

        # Display results
        display_uncertainty_results(result, "Performance Ratio")


def custom_uncertainty_analysis():
    """Custom uncertainty analysis with user-defined components."""

    st.header("Custom Uncertainty Analysis")

    st.markdown("""
    Define your own uncertainty components and calculate combined uncertainty using GUM methodology.
    """)

    # Initialize session state for components
    if 'components' not in st.session_state:
        st.session_state.components = []

    # Add component form
    with st.expander("Add Uncertainty Component", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            comp_name = st.text_input("Component Name", value="Component 1")
            comp_value = st.number_input("Value", value=100.0)

        with col2:
            comp_uncertainty = st.number_input("Uncertainty", value=5.0, min_value=0.0)
            comp_distribution = st.selectbox(
                "Distribution",
                ["normal", "uniform", "triangular"]
            )

        with col3:
            comp_sensitivity = st.number_input(
                "Sensitivity Coefficient",
                value=1.0,
                help="Partial derivative ∂f/∂x"
            )
            unc_type = st.selectbox("Uncertainty Type", ["standard", "expanded"])

        if st.button("Add Component"):
            st.session_state.components.append({
                'name': comp_name,
                'value': comp_value,
                'uncertainty': comp_uncertainty,
                'uncertainty_type': unc_type,
                'distribution': comp_distribution,
                'sensitivity_coefficient': comp_sensitivity
            })
            st.success(f"✓ Added: {comp_name}")

    # Display current components
    if st.session_state.components:
        st.subheader("Current Components")

        components_df = pd.DataFrame(st.session_state.components)
        st.dataframe(components_df, use_container_width=True)

        if st.button("Clear All Components"):
            st.session_state.components = []
            st.rerun()

        # Calculate combined uncertainty
        if st.button("Calculate Combined Uncertainty", type="primary"):
            calc = PVUncertaintyCalculator()

            for comp in st.session_state.components:
                calc.add_component(**comp)

            combined_unc, budget = calc.calculate_combined_uncertainty()

            # Display results
            st.success("✓ Calculation Complete")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Combined Standard Uncertainty",
                    f"{combined_unc:.4f}"
                )

            with col2:
                st.metric(
                    "Expanded Uncertainty (k=2)",
                    f"{combined_unc * 2:.4f}"
                )

            with col3:
                # Calculate total measured value
                total_value = sum(c['value'] * c['sensitivity_coefficient']
                                for c in st.session_state.components)
                if total_value != 0:
                    rel_unc = (combined_unc / abs(total_value)) * 100
                    st.metric(
                        "Relative Uncertainty",
                        f"{rel_unc:.2f}%"
                    )

            # Visualizations
            st.subheader("Uncertainty Budget")

            vis = UncertaintyVisualizer()

            col1, col2 = st.columns(2)

            with col1:
                fig_budget = vis.create_uncertainty_budget_chart(budget)
                st.plotly_chart(fig_budget, use_container_width=True)

            with col2:
                fig_pie = vis.create_uncertainty_breakdown_pie(budget)
                st.plotly_chart(fig_pie, use_container_width=True)

            # Export option
            st.download_button(
                label="Download Results (CSV)",
                data=pd.DataFrame(budget['components']).to_csv(index=False),
                file_name="uncertainty_budget.csv",
                mime="text/csv"
            )


def monte_carlo_analysis():
    """Monte Carlo simulation interface."""

    st.header("Monte Carlo Simulation")

    st.markdown("""
    Run Monte Carlo simulation for uncertainty propagation following GUM Supplement 1.
    """)

    # Analysis selection
    mc_type = st.radio(
        "Select Analysis:",
        ["Power Uncertainty", "Performance Ratio Uncertainty"]
    )

    # Simulation parameters
    with st.expander("Simulation Settings"):
        n_samples = st.number_input(
            "Number of Samples",
            min_value=1000,
            max_value=1000000,
            value=100000,
            step=10000,
            help="More samples = more accurate but slower"
        )

        random_seed = st.number_input(
            "Random Seed (for reproducibility)",
            min_value=0,
            value=42,
            help="Set seed for reproducible results"
        )

        use_seed = st.checkbox("Use Random Seed", value=True)

    if mc_type == "Power Uncertainty":
        monte_carlo_power(n_samples, random_seed if use_seed else None)
    else:
        monte_carlo_pr(n_samples, random_seed if use_seed else None)


def monte_carlo_power(n_samples: int, random_seed: Optional[int]):
    """Monte Carlo simulation for power uncertainty."""

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Parameters (Mean Values)")
        irradiance_mean = st.number_input("Irradiance (W/m²)", value=1000.0)
        temperature_mean = st.number_input("Temperature (°C)", value=45.0)
        module_efficiency = st.number_input("Module Efficiency", value=0.20, format="%.3f")

    with col2:
        st.subheader("Standard Uncertainties")
        irradiance_std = st.number_input("Irradiance Std (W/m²)", value=20.0)
        temperature_std = st.number_input("Temperature Std (°C)", value=1.0)
        power_meter_std = st.number_input("Power Meter Std (W)", value=2.0)
        efficiency_std = st.number_input("Efficiency Std", value=0.01, format="%.4f")

    if st.button("Run Monte Carlo Simulation", type="primary"):
        with st.spinner(f"Running {n_samples:,} Monte Carlo samples..."):
            result = PVMonteCarlo.simulate_power_uncertainty(
                irradiance_mean=irradiance_mean,
                irradiance_std=irradiance_std,
                temperature_mean=temperature_mean,
                temperature_std=temperature_std,
                power_meter_std=power_meter_std,
                module_efficiency=module_efficiency,
                efficiency_std=efficiency_std,
                n_samples=n_samples,
                random_state=random_seed
            )

        display_monte_carlo_results(result, "Power (W)")


def monte_carlo_pr(n_samples: int, random_seed: Optional[int]):
    """Monte Carlo simulation for PR uncertainty."""

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Parameters (Mean Values)")
        energy_mean = st.number_input("Energy (kWh)", value=1000.0)
        irradiation_mean = st.number_input("Irradiation (kWh/m²)", value=5.5)
        capacity_mean = st.number_input("Capacity (kWp)", value=250.0)

    with col2:
        st.subheader("Standard Uncertainties")
        energy_std = st.number_input("Energy Std (kWh)", value=10.0)
        irradiation_std = st.number_input("Irradiation Std (kWh/m²)", value=0.2)
        capacity_std = st.number_input("Capacity Std (kWp)", value=5.0)

    if st.button("Run Monte Carlo Simulation", type="primary"):
        with st.spinner(f"Running {n_samples:,} Monte Carlo samples..."):
            result = PVMonteCarlo.simulate_pr_uncertainty(
                energy_mean=energy_mean,
                energy_std=energy_std,
                irradiation_mean=irradiation_mean,
                irradiation_std=irradiation_std,
                capacity_mean=capacity_mean,
                capacity_std=capacity_std,
                n_samples=n_samples,
                random_state=random_seed
            )

        display_monte_carlo_results(result, "Performance Ratio")


def batch_data_analysis():
    """Batch data analysis from CSV upload."""

    st.header("Batch Data Analysis")

    st.markdown("""
    Upload CSV file with time-series PV measurement data for batch uncertainty analysis.

    **Required columns:** `irradiance`, `temperature`, `power`
    **Optional column:** `timestamp`
    """)

    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=['csv'],
        help="CSV file with PV measurement data"
    )

    # Option to use sample data
    if st.button("Use Sample Data"):
        df = PVDataHandler.create_sample_data()
        st.session_state.batch_data = df
        st.success("✓ Loaded 100 sample measurements")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.batch_data = df
        st.success(f"✓ Loaded {len(df)} records")

    # Process data if available
    if 'batch_data' in st.session_state:
        df = st.session_state.batch_data

        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        # Data validation
        validator = PVDataValidator()
        validation = validator.validate_power_data(df)

        if validation.warnings:
            with st.expander("⚠️ Data Validation Warnings"):
                for warning in validation.warnings:
                    st.warning(warning)

        if not validation.is_valid:
            st.error("❌ Data validation failed:")
            for error in validation.errors:
                st.error(error)
            return

        # Statistics
        st.subheader("Data Statistics")

        col1, col2, col3 = st.columns(3)

        for col, col_name in zip([col1, col2, col3],
                                 ['irradiance', 'temperature', 'power']):
            if col_name in df.columns:
                stats = PVDataHandler.calculate_statistics(df, col_name)
                with col:
                    st.metric(f"{col_name.title()} Mean", f"{stats['mean']:.2f}")
                    st.caption(f"Std: {stats['std']:.2f} | Range: [{stats['min']:.1f}, {stats['max']:.1f}]")

        # Analysis options
        st.subheader("Batch Analysis Settings")

        col1, col2 = st.columns(2)

        with col1:
            irradiance_unc_pct = st.number_input(
                "Irradiance Uncertainty (%)",
                min_value=0.0,
                max_value=20.0,
                value=2.0,
                step=0.1
            )

            temperature_unc = st.number_input(
                "Temperature Uncertainty (°C)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1
            )

        with col2:
            power_meter_unc_pct = st.number_input(
                "Power Meter Uncertainty (%)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1
            )

        if st.button("Run Batch Analysis", type="primary"):
            with st.spinner("Analyzing batch data..."):
                results_list = []

                for idx, row in df.iterrows():
                    irrad = row['irradiance']
                    temp = row['temperature']
                    pwr = row['power']

                    # Calculate uncertainties
                    irrad_unc = irrad * irradiance_unc_pct / 100
                    pwr_unc = pwr * power_meter_unc_pct / 100

                    result = PVPowerUncertainty.calculate_power_uncertainty(
                        irradiance=irrad,
                        irradiance_uncertainty=irrad_unc,
                        temperature=temp,
                        temperature_uncertainty=temperature_unc,
                        power=pwr,
                        power_meter_uncertainty=pwr_unc
                    )

                    results_list.append({
                        'index': idx,
                        'power': pwr,
                        'uncertainty': result['combined_uncertainty'],
                        'relative_uncertainty_pct': result['relative_uncertainty_percent']
                    })

                results_df = pd.DataFrame(results_list)

                # Display summary
                st.success("✓ Batch analysis complete!")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Mean Uncertainty",
                        f"{results_df['uncertainty'].mean():.2f} W"
                    )

                with col2:
                    st.metric(
                        "Mean Relative Uncertainty",
                        f"{results_df['relative_uncertainty_pct'].mean():.2f}%"
                    )

                with col3:
                    st.metric(
                        "Max Relative Uncertainty",
                        f"{results_df['relative_uncertainty_pct'].max():.2f}%"
                    )

                # Plot results
                st.subheader("Results Visualization")

                import plotly.graph_objects as go
                from plotly.subplots import make_subplots

                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Power with Uncertainty Bands",
                                  "Relative Uncertainty Over Time"),
                    vertical_spacing=0.12
                )

                # Power with error bands
                fig.add_trace(
                    go.Scatter(
                        x=results_df.index,
                        y=results_df['power'],
                        mode='lines',
                        name='Power',
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=results_df.index,
                        y=results_df['power'] + results_df['uncertainty'] * 2,
                        mode='lines',
                        name='Upper (k=2)',
                        line=dict(width=0),
                        showlegend=False
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=results_df.index,
                        y=results_df['power'] - results_df['uncertainty'] * 2,
                        mode='lines',
                        name='Lower (k=2)',
                        fill='tonexty',
                        fillcolor='rgba(0,100,255,0.2)',
                        line=dict(width=0),
                        showlegend=False
                    ),
                    row=1, col=1
                )

                # Relative uncertainty
                fig.add_trace(
                    go.Scatter(
                        x=results_df.index,
                        y=results_df['relative_uncertainty_pct'],
                        mode='lines',
                        name='Relative Uncertainty',
                        line=dict(color='red')
                    ),
                    row=2, col=1
                )

                fig.update_xaxes(title_text="Sample Index", row=2, col=1)
                fig.update_yaxes(title_text="Power (W)", row=1, col=1)
                fig.update_yaxes(title_text="Relative Uncertainty (%)", row=2, col=1)

                fig.update_layout(
                    height=700,
                    template="plotly_white",
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)

                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results (CSV)",
                    data=csv,
                    file_name="batch_uncertainty_results.csv",
                    mime="text/csv"
                )


def display_uncertainty_results(result: Dict, value_label: str):
    """Display uncertainty analysis results."""

    st.success("✓ Calculation Complete")

    # Main metrics
    col1, col2, col3, col4 = st.columns(4)

    value_key = 'power' if 'power' in result else 'performance_ratio'

    with col1:
        st.metric(
            value_label,
            f"{result[value_key]:.4f}"
        )

    with col2:
        st.metric(
            "Combined Uncertainty",
            f"{result['combined_uncertainty']:.4f}"
        )

    with col3:
        st.metric(
            "Expanded Unc. (k=2)",
            f"{result['expanded_uncertainty_k2']:.4f}"
        )

    with col4:
        st.metric(
            "Relative Uncertainty",
            f"{result['relative_uncertainty_percent']:.2f}%"
        )

    # Confidence interval
    st.subheader("95% Confidence Interval")
    ci = result['confidence_interval_95']
    st.info(f"**{ci[0]:.4f}** to **{ci[1]:.4f}**")

    # Visualizations
    st.subheader("Uncertainty Budget Analysis")

    vis = UncertaintyVisualizer()
    budget = result['budget']

    col1, col2 = st.columns(2)

    with col1:
        fig_budget = vis.create_uncertainty_budget_chart(budget)
        st.plotly_chart(fig_budget, use_container_width=True)

    with col2:
        fig_pie = vis.create_uncertainty_breakdown_pie(budget)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Detailed budget table
    with st.expander("View Detailed Uncertainty Budget"):
        budget_df = pd.DataFrame(budget['components'])
        st.dataframe(budget_df, use_container_width=True)

    # Export options
    col1, col2 = st.columns(2)

    with col1:
        formatter = ResultsFormatter()
        text_result = formatter.format_uncertainty_result(result)
        st.download_button(
            label="Download Results (TXT)",
            data=text_result,
            file_name="uncertainty_results.txt",
            mime="text/plain"
        )

    with col2:
        handler = PVDataHandler()
        csv_result = handler.export_results_to_csv(result)
        st.download_button(
            label="Download Results (CSV)",
            data=csv_result,
            file_name="uncertainty_results.csv",
            mime="text/csv"
        )


def display_monte_carlo_results(result: Dict, value_label: str):
    """Display Monte Carlo simulation results."""

    st.success(f"✓ Simulation Complete ({result['n_samples']:,} samples)")

    # Main metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Mean", f"{result['mean']:.4f}")

    with col2:
        st.metric("Median", f"{result['median']:.4f}")

    with col3:
        st.metric("Std Uncertainty", f"{result['std_uncertainty']:.4f}")

    with col4:
        st.metric("Relative Unc.", f"{result['relative_uncertainty_percent']:.2f}%")

    # Distribution statistics
    with st.expander("Distribution Statistics"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Skewness", f"{result['skewness']:.4f}")
        with col2:
            st.metric("Kurtosis", f"{result['kurtosis']:.4f}")

    # Visualizations
    st.subheader("Distribution Analysis")

    vis = UncertaintyVisualizer()

    # Histogram
    fig_hist = vis.create_monte_carlo_histogram(
        samples=result['samples'],
        mean=result['mean'],
        std=result['std_uncertainty'],
        confidence_intervals=result['confidence_intervals'],
        title=f"{value_label} - Monte Carlo Distribution"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Confidence intervals
    fig_ci = vis.create_confidence_interval_plot(
        value=result['mean'],
        confidence_intervals=result['confidence_intervals'],
        label=value_label
    )
    st.plotly_chart(fig_ci, use_container_width=True)

    # Sensitivity analysis
    if 'sensitivities' in result and result['sensitivities']:
        st.subheader("Sensitivity Analysis")
        fig_sens = vis.create_sensitivity_chart(result['sensitivities'])
        st.plotly_chart(fig_sens, use_container_width=True)

    # Percentiles
    with st.expander("View Percentiles"):
        percentiles_df = pd.DataFrame([
            {"Percentile": f"{k}%", "Value": f"{v:.4f}"}
            for k, v in result['percentiles'].items()
        ])
        st.dataframe(percentiles_df, use_container_width=True)


if __name__ == "__main__":
    main()
