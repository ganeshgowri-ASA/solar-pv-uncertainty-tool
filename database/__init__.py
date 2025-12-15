"""
Database module for PV Measurement Uncertainty Tool
Provides SQLAlchemy models and database utilities for Railway PostgreSQL
"""

from database.models import (
    Base,
    User,
    Organization,
    Module,
    Measurement,
    IVCurveData,
    ReferenceDevice,
    SpectralResponse,
    SunSimulator,
    UncertaintyResult,
    UncertaintyComponent,
    File,
    AuditLog,
    ApprovalWorkflow
)
from database.connection import (
    get_database_url,
    get_engine,
    get_session,
    session_scope,
    init_database,
    check_connection,
    get_connection_info
)
from database.streamlit_integration import (
    get_db_status,
    init_db_schema,
    display_db_status_sidebar,
    display_db_admin_panel,
    save_uncertainty_result,
    get_saved_results
)

__all__ = [
    # Models
    'Base',
    'User',
    'Organization',
    'Module',
    'Measurement',
    'IVCurveData',
    'ReferenceDevice',
    'SpectralResponse',
    'SunSimulator',
    'UncertaintyResult',
    'UncertaintyComponent',
    'File',
    'AuditLog',
    'ApprovalWorkflow',
    # Connection utilities
    'get_database_url',
    'get_engine',
    'get_session',
    'session_scope',
    'init_database',
    'check_connection',
    'get_connection_info',
    # Streamlit integration
    'get_db_status',
    'init_db_schema',
    'display_db_status_sidebar',
    'display_db_admin_panel',
    'save_uncertainty_result',
    'get_saved_results'
]
