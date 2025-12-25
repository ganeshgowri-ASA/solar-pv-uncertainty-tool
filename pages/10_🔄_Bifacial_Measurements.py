"""
Bifacial PV Measurements Page
Calculates bifaciality coefficients from front and rear measurements
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Bifacial PV Measurements",
    page_icon="🔄",
    layout="wide"
)

st.title("🔄 Bifacial PV Measurements")
st.markdown("Calculate bifaciality coefficients from front and rear side measurements")
st.markdown("---")


# =============================================================================
# INPUT SECTION
# =============================================================================

st.header("📥 Input Parameters")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Front Side (STC)")
    front_irradiance = st.number_input(
        "Front Irradiance (W/m²)",
        min_value=0.0,
        max_value=2000.0,
        value=1000.0,
        step=10.0,
        key="front_irradiance"
    )
    front_isc = st.number_input(
        "Front Isc (A)",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        step=0.1,
        key="front_isc"
    )
    front_voc = st.number_input(
        "Front Voc (V)",
        min_value=0.0,
        max_value=100.0,
        value=45.0,
        step=0.1,
        key="front_voc"
    )
    front_pmax = st.number_input(
        "Front Pmax (W)",
        min_value=0.0,
        max_value=1000.0,
        value=400.0,
        step=1.0,
        key="front_pmax"
    )

with col2:
    st.subheader("Rear Side (STC)")
    rear_irradiance = st.number_input(
        "Rear Irradiance (W/m²)",
        min_value=0.0,
        max_value=2000.0,
        value=1000.0,
        step=10.0,
        key="rear_irradiance"
    )
    rear_isc = st.number_input(
        "Rear Isc (A)",
        min_value=0.0,
        max_value=50.0,
        value=7.5,
        step=0.1,
        key="rear_isc"
    )
    rear_voc = st.number_input(
        "Rear Voc (V)",
        min_value=0.0,
        max_value=100.0,
        value=44.0,
        step=0.1,
        key="rear_voc"
    )
    rear_pmax = st.number_input(
        "Rear Pmax (W)",
        min_value=0.0,
        max_value=1000.0,
        value=280.0,
        step=1.0,
        key="rear_pmax"
    )

st.markdown("---")


# =============================================================================
# CALCULATION SECTION
# =============================================================================

st.header("📊 Bifaciality Coefficients")

# Calculate bifaciality factors (avoid division by zero)
phi_isc = rear_isc / front_isc if front_isc > 0 else 0.0
phi_voc = rear_voc / front_voc if front_voc > 0 else 0.0
phi_pmax = rear_pmax / front_pmax if front_pmax > 0 else 0.0

# Calculate fill factors
front_ff = (front_pmax / (front_isc * front_voc)) if (front_isc * front_voc) > 0 else 0.0
rear_ff = (rear_pmax / (rear_isc * rear_voc)) if (rear_isc * rear_voc) > 0 else 0.0
phi_ff = rear_ff / front_ff if front_ff > 0 else 0.0

# Calculate bifacial gain (typical rear irradiance ratio)
typical_rear_ratio = 0.15  # 15% typical rear irradiance
bifacial_gain = phi_pmax * typical_rear_ratio * 100  # in percent


# =============================================================================
# DISPLAY SECTION
# =============================================================================

# Results in columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="φ Isc (Bifaciality)",
        value=f"{phi_isc:.3f}",
        delta=f"{(phi_isc - 0.70) * 100:.1f}% vs 0.70 baseline"
    )

with col2:
    st.metric(
        label="φ Voc",
        value=f"{phi_voc:.3f}",
        delta=f"{(phi_voc - 0.95) * 100:.1f}% vs 0.95 baseline"
    )

with col3:
    st.metric(
        label="φ Pmax",
        value=f"{phi_pmax:.3f}",
        delta=f"{(phi_pmax - 0.70) * 100:.1f}% vs 0.70 baseline"
    )

with col4:
    st.metric(
        label="φ FF",
        value=f"{phi_ff:.3f}",
        delta=None
    )

st.markdown("---")

# Results table
st.subheader("Summary Table")

results_data = {
    "Parameter": ["Isc (A)", "Voc (V)", "Pmax (W)", "Fill Factor", "Irradiance (W/m²)"],
    "Front": [f"{front_isc:.2f}", f"{front_voc:.2f}", f"{front_pmax:.1f}",
              f"{front_ff:.3f}", f"{front_irradiance:.0f}"],
    "Rear": [f"{rear_isc:.2f}", f"{rear_voc:.2f}", f"{rear_pmax:.1f}",
             f"{rear_ff:.3f}", f"{rear_irradiance:.0f}"],
    "Bifaciality (φ)": [f"{phi_isc:.3f}", f"{phi_voc:.3f}", f"{phi_pmax:.3f}",
                        f"{phi_ff:.3f}", "-"]
}

results_df = pd.DataFrame(results_data)
st.dataframe(results_df, use_container_width=True, hide_index=True)

# Bifacial gain info
st.info(f"**Estimated Bifacial Gain:** {bifacial_gain:.1f}% (assuming {typical_rear_ratio*100:.0f}% rear irradiance ratio)")

st.markdown("---")


# =============================================================================
# CHART SECTION
# =============================================================================

st.subheader("📈 Bifacial Gain Visualization")

# Create bar chart comparing front vs rear
fig = go.Figure()

# Front values (normalized)
fig.add_trace(go.Bar(
    name='Front Side',
    x=['Isc', 'Voc', 'Pmax', 'FF'],
    y=[1.0, 1.0, 1.0, 1.0],
    marker_color='#1f77b4',
    text=['100%', '100%', '100%', '100%'],
    textposition='outside'
))

# Rear values (as fraction of front = bifaciality)
fig.add_trace(go.Bar(
    name='Rear Side (φ)',
    x=['Isc', 'Voc', 'Pmax', 'FF'],
    y=[phi_isc, phi_voc, phi_pmax, phi_ff],
    marker_color='#ff7f0e',
    text=[f"{phi_isc*100:.1f}%", f"{phi_voc*100:.1f}%",
          f"{phi_pmax*100:.1f}%", f"{phi_ff*100:.1f}%"],
    textposition='outside'
))

fig.update_layout(
    title="Bifaciality Coefficients Comparison",
    xaxis_title="Parameter",
    yaxis_title="Relative Value (Front = 1.0)",
    barmode='group',
    yaxis=dict(range=[0, 1.2]),
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    height=400
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")


# =============================================================================
# SAVE SECTION
# =============================================================================

st.header("💾 Save Measurement")

# Module info for saving
col1, col2, col3 = st.columns(3)

with col1:
    module_serial = st.text_input("Module Serial Number", value="", key="module_serial")

with col2:
    module_model = st.text_input("Module Model", value="", key="module_model")

with col3:
    notes = st.text_area("Notes", value="", height=68, key="notes")

# Save button
if st.button("💾 Save to Database", type="primary", use_container_width=True):
    try:
        from database.connection import session_scope, check_connection
        from database.models import Measurement, MeasurementType, Module

        if not check_connection():
            st.error("Database not connected. Please configure database in Admin page.")
        else:
            with session_scope() as session:
                # Create or find module
                module = None
                if module_serial:
                    module = session.query(Module).filter_by(serial_number=module_serial).first()
                    if not module:
                        st.warning(f"Module with serial '{module_serial}' not found. Saving measurement without module reference.")

                # Generate measurement number
                measurement_number = f"BIF-{datetime.now().strftime('%Y%m%d%H%M%S')}"

                # Create measurement record
                measurement = Measurement(
                    measurement_number=measurement_number,
                    measurement_type=MeasurementType.BIFACIALITY,
                    test_date=datetime.now(),
                    module_id=module.id if module else None,
                    # Front side values
                    isc_a=front_isc,
                    voc_v=front_voc,
                    pmax_w=front_pmax,
                    fill_factor=front_ff,
                    actual_irradiance_w_m2=front_irradiance,
                    # Store bifaciality in notes (JSON-like)
                    notes=f"Bifacial Measurement\n"
                          f"phi_Isc: {phi_isc:.4f}\n"
                          f"phi_Voc: {phi_voc:.4f}\n"
                          f"phi_Pmax: {phi_pmax:.4f}\n"
                          f"phi_FF: {phi_ff:.4f}\n"
                          f"Rear Isc: {rear_isc} A\n"
                          f"Rear Voc: {rear_voc} V\n"
                          f"Rear Pmax: {rear_pmax} W\n"
                          f"Rear Irradiance: {rear_irradiance} W/m²\n"
                          f"---\n{notes}"
                )

                session.add(measurement)
                session.commit()

                st.success(f"Measurement saved successfully! (ID: {measurement_number})")
                st.balloons()

    except ImportError:
        st.error("Database modules not available. Please ensure database is configured.")
    except Exception as e:
        st.error(f"Error saving measurement: {str(e)}")


# =============================================================================
# HELP SECTION
# =============================================================================

with st.expander("ℹ️ About Bifaciality Coefficients"):
    st.markdown("""
    ### Bifaciality Factor (φ)

    The bifaciality factor is defined as the ratio of rear-side to front-side performance:

    - **φ_Isc** = I_sc,rear / I_sc,front
    - **φ_Voc** = V_oc,rear / V_oc,front
    - **φ_Pmax** = P_max,rear / P_max,front
    - **φ_FF** = FF_rear / FF_front

    ### Typical Values

    | Technology | φ_Pmax Range |
    |------------|--------------|
    | PERC | 0.65 - 0.75 |
    | TOPCon | 0.75 - 0.85 |
    | HJT | 0.85 - 0.95 |

    ### Bifacial Gain

    The bifacial gain represents the additional energy yield from rear-side illumination:

    **BG = φ × (G_rear / G_front) × 100%**

    Typical outdoor conditions see 10-20% rear irradiance, resulting in 7-15% energy gain.

    ### Reference Standards

    - IEC TS 60904-1-2:2024 - Measurement of bifacial PV devices
    - IEC 60904-1:2020 - PV device characterization
    """)
