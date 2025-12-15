"""
Streamlit Database Integration
Provides database status display and initialization for the Streamlit app
"""

import streamlit as st
from typing import Optional, Dict, Any
from datetime import datetime


def get_db_status() -> Dict[str, Any]:
    """
    Get database connection status with caching for performance.

    Returns:
        Dictionary with connection status and details
    """
    try:
        from database.connection import get_connection_info, check_connection
        info = get_connection_info()
        return {
            'available': True,
            'connected': info.get('connected', False),
            'host': info.get('host', 'Unknown'),
            'database': info.get('database', 'Unknown'),
            'error': None
        }
    except Exception as e:
        return {
            'available': False,
            'connected': False,
            'host': None,
            'database': None,
            'error': str(e)
        }


def init_db_schema() -> tuple[bool, str]:
    """
    Initialize database schema safely.

    Returns:
        Tuple of (success, message)
    """
    try:
        from database.connection import init_database, check_connection

        if not check_connection():
            return False, "Cannot connect to database"

        init_database(drop_existing=False)
        return True, "Database schema initialized successfully"
    except Exception as e:
        return False, f"Schema initialization failed: {str(e)}"


def display_db_status_sidebar():
    """
    Display database connection status in the Streamlit sidebar.
    Call this function in your main app to show database status.
    """
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Database Status")

        status = get_db_status()

        if not status['available']:
            st.error("Database module not available")
            with st.expander("Error Details"):
                st.code(status.get('error', 'Unknown error'))
        elif status['connected']:
            st.success("Connected")
            with st.expander("Connection Details"):
                st.text(f"Host: {status['host']}")
                st.text(f"Database: {status['database']}")
        else:
            st.warning("Not Connected")
            with st.expander("Configuration"):
                st.text(f"Host: {status['host'] or 'Not configured'}")
                st.text(f"Database: {status['database'] or 'Not configured'}")
                st.caption("Check DATABASE_URL in Streamlit secrets")


def display_db_admin_panel():
    """
    Display database administration panel for schema management.
    Only show this in development/admin mode.
    """
    st.markdown("### Database Administration")

    status = get_db_status()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Connection Status**")
        if status['connected']:
            st.success("Connected to PostgreSQL")
        else:
            st.error("Not Connected")

    with col2:
        st.markdown("**Database Info**")
        st.text(f"Host: {status.get('host', 'N/A')}")
        st.text(f"Database: {status.get('database', 'N/A')}")

    st.markdown("---")

    # Schema initialization
    st.markdown("**Schema Management**")

    if status['connected']:
        if st.button("Initialize Schema", type="primary"):
            with st.spinner("Creating database schema..."):
                success, message = init_db_schema()
                if success:
                    st.success(message)
                else:
                    st.error(message)

        # Show table counts
        try:
            from database.connection import session_scope
            from database.models import (
                Organization, User, Module, Measurement,
                SunSimulator, ReferenceDevice, UncertaintyResult
            )

            with session_scope() as session:
                st.markdown("**Table Row Counts**")
                counts = {
                    "Organizations": session.query(Organization).count(),
                    "Users": session.query(User).count(),
                    "Modules": session.query(Module).count(),
                    "Measurements": session.query(Measurement).count(),
                    "Sun Simulators": session.query(SunSimulator).count(),
                    "Reference Devices": session.query(ReferenceDevice).count(),
                    "Uncertainty Results": session.query(UncertaintyResult).count(),
                }

                for table, count in counts.items():
                    st.text(f"  {table}: {count}")

        except Exception as e:
            st.warning(f"Could not query tables: {e}")
            st.info("Tables may not be initialized yet. Click 'Initialize Schema' to create them.")
    else:
        st.warning("Connect to database first to manage schema")
        st.info("Set DATABASE_URL in Streamlit secrets (Settings > Secrets)")


def save_uncertainty_result(
    measurement_data: Dict[str, Any],
    uncertainty_result: Dict[str, Any],
    components: list
) -> Optional[int]:
    """
    Save uncertainty calculation results to the database.

    Args:
        measurement_data: Dictionary with measurement parameters
        uncertainty_result: Dictionary with calculated uncertainty values
        components: List of uncertainty component dictionaries

    Returns:
        ID of saved result, or None if save failed
    """
    status = get_db_status()
    if not status['connected']:
        return None

    try:
        from database.connection import session_scope
        from database.models import UncertaintyResult, UncertaintyComponent

        with session_scope() as session:
            # Create uncertainty result record
            result = UncertaintyResult(
                measurement_id=measurement_data.get('measurement_id'),
                analysis_date=datetime.utcnow(),
                analysis_version='2.0',
                calculation_method=uncertainty_result.get('method', 'GUM'),
                target_parameter=uncertainty_result.get('parameter', 'Pmax'),
                measured_value=uncertainty_result.get('measured_value'),
                measured_unit=uncertainty_result.get('unit', 'W'),
                combined_standard_uncertainty_pct=uncertainty_result.get('combined_uncertainty'),
                expanded_uncertainty_k2_pct=uncertainty_result.get('expanded_uncertainty'),
                confidence_level_pct=95.0,
                coverage_factor=2.0,
                full_budget_json=uncertainty_result.get('full_budget'),
            )
            session.add(result)
            session.flush()  # Get the ID

            # Add components
            for comp in components:
                component = UncertaintyComponent(
                    uncertainty_result_id=result.id,
                    category_id=comp.get('category_id'),
                    subcategory_id=comp.get('subcategory_id'),
                    factor_id=comp.get('factor_id'),
                    name=comp.get('name'),
                    standard_uncertainty=comp.get('standard_uncertainty'),
                    distribution=comp.get('distribution'),
                    sensitivity_coefficient=comp.get('sensitivity_coefficient', 1.0),
                    variance_contribution=comp.get('variance'),
                    percentage_contribution=comp.get('percentage'),
                )
                session.add(component)

            return result.id

    except Exception as e:
        st.error(f"Failed to save results: {e}")
        return None


def get_saved_results(limit: int = 10) -> list:
    """
    Retrieve recent uncertainty calculation results from the database.

    Args:
        limit: Maximum number of results to return

    Returns:
        List of result dictionaries
    """
    status = get_db_status()
    if not status['connected']:
        return []

    try:
        from database.connection import session_scope
        from database.models import UncertaintyResult

        with session_scope() as session:
            results = session.query(UncertaintyResult)\
                .order_by(UncertaintyResult.analysis_date.desc())\
                .limit(limit)\
                .all()

            return [
                {
                    'id': r.id,
                    'date': r.analysis_date,
                    'parameter': r.target_parameter,
                    'value': r.measured_value,
                    'uncertainty': r.expanded_uncertainty_k2_pct,
                    'method': r.calculation_method,
                }
                for r in results
            ]

    except Exception as e:
        return []
