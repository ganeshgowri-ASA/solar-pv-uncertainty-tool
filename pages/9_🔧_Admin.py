"""
Admin Dashboard - Database Management
Provides database initialization, health checks, and migration tools
"""

import streamlit as st
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# Page configuration
st.set_page_config(
    page_title="Admin Dashboard - PV Uncertainty Tool",
    page_icon="🔧",
    layout="wide"
)


# =============================================================================
# AUTHENTICATION
# =============================================================================

def check_admin_password() -> bool:
    """
    Check if user has provided correct admin password.
    Returns True if authenticated, False otherwise.
    """
    # Check if already authenticated in session
    if st.session_state.get('admin_authenticated', False):
        return True

    # Get password from secrets
    try:
        admin_password = st.secrets.get('ADMIN_PASSWORD', None)
        if not admin_password:
            st.error("ADMIN_PASSWORD not configured in Streamlit secrets")
            st.info("Add ADMIN_PASSWORD to your Streamlit secrets to enable admin access")
            return False
    except Exception:
        st.error("Could not access Streamlit secrets")
        return False

    # Show login form
    st.markdown("## Admin Authentication Required")
    st.markdown("---")

    with st.form("admin_login"):
        password = st.text_input("Admin Password", type="password", key="admin_pwd_input")
        submitted = st.form_submit_button("Login", type="primary")

        if submitted:
            if password == admin_password:
                st.session_state['admin_authenticated'] = True
                st.rerun()
            else:
                st.error("Invalid password")

    return False


def logout():
    """Clear admin authentication."""
    st.session_state['admin_authenticated'] = False
    st.rerun()


# =============================================================================
# DATABASE UTILITIES
# =============================================================================

def get_database_status() -> Dict[str, Any]:
    """
    Get comprehensive database connection status.
    """
    result = {
        'available': False,
        'connected': False,
        'host': None,
        'port': None,
        'database': None,
        'user': None,
        'error': None
    }

    try:
        from database.connection import get_connection_info, check_connection, get_database_url
        from urllib.parse import urlparse

        # Parse URL for details
        url = get_database_url()
        parsed = urlparse(url)

        result['available'] = True
        result['host'] = parsed.hostname
        result['port'] = parsed.port or 5432
        result['database'] = parsed.path.lstrip('/') if parsed.path else None
        result['user'] = parsed.username
        result['connected'] = check_connection()

    except Exception as e:
        result['error'] = str(e)

    return result


def get_table_counts() -> Dict[str, int]:
    """
    Get row counts for all tables.
    """
    counts = {}
    try:
        from database.connection import session_scope
        from database.models import (
            Organization, User, Module, Measurement, IVCurveData,
            SunSimulator, ReferenceDevice, SpectralResponse,
            UncertaintyResult, UncertaintyComponent, File, AuditLog,
            ApprovalWorkflow
        )

        models = [
            ('organizations', Organization),
            ('users', User),
            ('modules', Module),
            ('measurements', Measurement),
            ('iv_curve_data', IVCurveData),
            ('sun_simulators', SunSimulator),
            ('reference_devices', ReferenceDevice),
            ('spectral_responses', SpectralResponse),
            ('uncertainty_results', UncertaintyResult),
            ('uncertainty_components', UncertaintyComponent),
            ('files', File),
            ('audit_logs', AuditLog),
            ('approval_workflows', ApprovalWorkflow),
        ]

        with session_scope() as session:
            for table_name, model in models:
                try:
                    counts[table_name] = session.query(model).count()
                except Exception:
                    counts[table_name] = -1  # Error indicator

    except Exception as e:
        st.error(f"Error querying tables: {e}")

    return counts


def get_all_tables() -> List[str]:
    """
    Get list of all tables in the database schema.
    """
    try:
        from database.models import Base
        return sorted(Base.metadata.tables.keys())
    except Exception:
        return []


def initialize_database() -> Tuple[bool, str, List[str]]:
    """
    Initialize database schema.
    Returns (success, message, created_tables).
    """
    try:
        from database.connection import get_engine, check_connection
        from database.models import Base
        from sqlalchemy import inspect

        if not check_connection():
            return False, "Cannot connect to database", []

        engine = get_engine()

        # Get tables before
        inspector = inspect(engine)
        tables_before = set(inspector.get_table_names())

        # Create all tables
        Base.metadata.create_all(bind=engine)

        # Get tables after
        inspector = inspect(engine)
        tables_after = set(inspector.get_table_names())

        # Find newly created tables
        new_tables = sorted(tables_after - tables_before)
        all_tables = sorted(tables_after)

        if new_tables:
            return True, f"Created {len(new_tables)} new table(s)", new_tables
        else:
            return True, f"Schema verified. All {len(all_tables)} tables exist.", all_tables

    except Exception as e:
        return False, f"Initialization failed: {str(e)}", []


def get_available_migrations() -> List[Dict[str, str]]:
    """
    Get list of available SQL migration files.
    """
    migrations = []
    migrations_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'migrations')

    if not os.path.exists(migrations_dir):
        return migrations

    for filename in sorted(os.listdir(migrations_dir)):
        if filename.endswith('.sql'):
            filepath = os.path.join(migrations_dir, filename)
            # Parse migration info
            parts = filename.replace('.sql', '').split('_')
            migration_num = parts[0] if parts else '000'
            direction = 'UP' if '_UP' in filename.upper() else ('DOWN' if '_DOWN' in filename.upper() else 'UNKNOWN')

            migrations.append({
                'filename': filename,
                'filepath': filepath,
                'number': migration_num,
                'direction': direction,
                'name': '_'.join(parts[1:-1]) if len(parts) > 2 else filename
            })

    return migrations


def run_migration(filepath: str) -> Tuple[bool, str]:
    """
    Execute a SQL migration file.
    """
    try:
        from database.connection import get_engine
        from sqlalchemy import text

        # Read migration file
        with open(filepath, 'r') as f:
            sql_content = f.read()

        if not sql_content.strip():
            return False, "Migration file is empty"

        engine = get_engine()

        with engine.begin() as conn:
            # Execute each statement separately
            statements = [s.strip() for s in sql_content.split(';') if s.strip()]
            for stmt in statements:
                conn.execute(text(stmt))

        return True, f"Migration executed successfully ({len(statements)} statement(s))"

    except Exception as e:
        return False, f"Migration failed: {str(e)}"


# =============================================================================
# UI SECTIONS
# =============================================================================

def display_header():
    """Display admin page header."""
    col1, col2 = st.columns([6, 1])

    with col1:
        st.title("🔧 Admin Dashboard")
        st.markdown("Database management and system administration")

    with col2:
        if st.button("Logout", type="secondary"):
            logout()


def display_connection_status():
    """Display database connection status section."""
    st.markdown("## Database Connection Status")

    status = get_database_status()

    # Status indicator
    col1, col2, col3 = st.columns(3)

    with col1:
        if status['connected']:
            st.success("**Status:** Connected")
        elif status['available']:
            st.warning("**Status:** Not Connected")
        else:
            st.error("**Status:** Module Unavailable")

    with col2:
        st.metric("Host", status['host'] or "Not configured")

    with col3:
        st.metric("Database", status['database'] or "Not configured")

    # Configuration details
    with st.expander("Configuration Details", expanded=False):
        st.markdown("**Connection Parameters:**")
        st.code(f"""
Host:     {status['host'] or 'Not set'}
Port:     {status['port'] or 'Not set'}
Database: {status['database'] or 'Not set'}
User:     {status['user'] or 'Not set'}
Password: ********
        """, language="text")

        if status['error']:
            st.error(f"Error: {status['error']}")


def display_database_initialization():
    """Display database initialization section."""
    st.markdown("## Database Initialization")
    st.markdown("---")

    status = get_database_status()

    if not status['connected']:
        st.warning("Database not connected. Cannot initialize schema.")
        st.info("Configure DATABASE_URL in Streamlit secrets first.")
        return

    # Warning message
    st.warning("""
    **Warning:** This will create all database tables. Only run this once when setting up a new database!

    If tables already exist, they will NOT be modified or dropped. This is safe to run multiple times.
    """)

    # Show expected tables
    expected_tables = get_all_tables()
    with st.expander("Tables to be created", expanded=False):
        for table in expected_tables:
            st.text(f"  - {table}")

    col1, col2 = st.columns([1, 3])

    with col1:
        init_button = st.button(
            "🚀 Initialize Database Schema",
            type="primary",
            use_container_width=True
        )

    if init_button:
        with st.spinner("Initializing database schema..."):
            success, message, tables = initialize_database()

        if success:
            st.success(f"✅ {message}")

            if tables:
                st.markdown("**Tables:**")
                for table in tables:
                    st.text(f"  ✓ {table}")
        else:
            st.error(f"❌ {message}")


def display_health_check():
    """Display database health check section."""
    st.markdown("## Database Health Check")
    st.markdown("---")

    col1, col2 = st.columns([1, 3])

    with col1:
        check_button = st.button(
            "🔍 Run Health Check",
            type="secondary",
            use_container_width=True
        )

    if check_button or st.session_state.get('show_health_check', False):
        st.session_state['show_health_check'] = True

        with st.spinner("Checking database health..."):
            status = get_database_status()

        # Connection test
        st.markdown("### Connection Test")
        if status['connected']:
            st.success("✅ Database connection successful")
        else:
            st.error("❌ Database connection failed")
            if status['error']:
                st.code(status['error'])
            return

        # Table counts
        st.markdown("### Table Statistics")

        with st.spinner("Querying tables..."):
            counts = get_table_counts()

        if counts:
            total_rows = sum(v for v in counts.values() if v >= 0)
            total_tables = len([v for v in counts.values() if v >= 0])

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Tables", total_tables)
            with col2:
                st.metric("Total Rows", total_rows)

            st.markdown("**Row counts by table:**")

            # Display as grid
            cols = st.columns(3)
            for i, (table, count) in enumerate(sorted(counts.items())):
                with cols[i % 3]:
                    if count >= 0:
                        st.text(f"📊 {table}: {count}")
                    else:
                        st.text(f"❌ {table}: Error")
        else:
            st.warning("No tables found. Run 'Initialize Database Schema' first.")


def display_migration_runner():
    """Display migration runner section."""
    st.markdown("## Migration Runner")
    st.markdown("---")

    status = get_database_status()

    if not status['connected']:
        st.warning("Database not connected. Cannot run migrations.")
        return

    # Get available migrations
    migrations = get_available_migrations()

    if not migrations:
        st.info("""
        No migration files found in the `migrations/` folder.

        **To create a migration:**
        1. Create a file like `migrations/001_initial_schema_UP.sql`
        2. Add your SQL statements
        3. Refresh this page

        **Naming convention:**
        - UP migrations: `XXX_description_UP.sql`
        - DOWN migrations: `XXX_description_DOWN.sql`
        """)
        return

    # Show available migrations
    st.markdown("### Available Migrations")

    # Group by migration number
    migration_groups = {}
    for m in migrations:
        num = m['number']
        if num not in migration_groups:
            migration_groups[num] = []
        migration_groups[num].append(m)

    for num in sorted(migration_groups.keys()):
        group = migration_groups[num]
        with st.expander(f"Migration {num}: {group[0]['name']}", expanded=False):
            for m in group:
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.text(m['filename'])

                with col2:
                    direction_color = "🟢" if m['direction'] == 'UP' else "🔴"
                    st.text(f"{direction_color} {m['direction']}")

                with col3:
                    # Preview button
                    if st.button(f"Preview", key=f"preview_{m['filename']}"):
                        st.session_state['preview_migration'] = m['filepath']

                    # Run button with confirmation
                    if st.button(f"Run", key=f"run_{m['filename']}", type="primary"):
                        st.session_state['pending_migration'] = m['filepath']
                        st.session_state['pending_migration_name'] = m['filename']

    # Preview panel
    if st.session_state.get('preview_migration'):
        st.markdown("### Migration Preview")
        try:
            with open(st.session_state['preview_migration'], 'r') as f:
                content = f.read()
            st.code(content, language="sql")
        except Exception as e:
            st.error(f"Could not read file: {e}")

        if st.button("Close Preview"):
            del st.session_state['preview_migration']
            st.rerun()

    # Confirmation dialog
    if st.session_state.get('pending_migration'):
        st.markdown("### Confirm Migration")
        st.warning(f"""
        **You are about to run:** `{st.session_state.get('pending_migration_name')}`

        This action cannot be undone automatically. Make sure you have a DOWN migration ready!
        """)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Confirm & Execute", type="primary"):
                filepath = st.session_state['pending_migration']
                with st.spinner("Executing migration..."):
                    success, message = run_migration(filepath)

                if success:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")

                del st.session_state['pending_migration']
                del st.session_state['pending_migration_name']

        with col2:
            if st.button("❌ Cancel"):
                del st.session_state['pending_migration']
                del st.session_state['pending_migration_name']
                st.rerun()


def display_quick_actions():
    """Display quick action buttons."""
    st.markdown("## Quick Actions")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()

    with col2:
        if st.button("📊 View Table Stats", use_container_width=True):
            st.session_state['show_health_check'] = True
            st.rerun()

    with col3:
        if st.button("🧹 Clear Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key != 'admin_authenticated':
                    del st.session_state[key]
            st.rerun()


def display_system_info():
    """Display system information."""
    st.markdown("## System Information")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Application**")
        st.text("Version: 3.0.0-production")
        st.text(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    with col2:
        st.markdown("**Environment**")
        st.text(f"Python: {os.sys.version.split()[0]}")
        try:
            import sqlalchemy
            st.text(f"SQLAlchemy: {sqlalchemy.__version__}")
        except Exception:
            st.text("SQLAlchemy: Not available")


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application entry point."""

    # Check authentication
    if not check_admin_password():
        return

    # Display header with logout button
    display_header()
    st.markdown("---")

    # Quick actions bar
    display_quick_actions()
    st.markdown("---")

    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📡 Connection Status",
        "🚀 Initialize Database",
        "💓 Health Check",
        "🔄 Run Migrations"
    ])

    with tab1:
        display_connection_status()

    with tab2:
        display_database_initialization()

    with tab3:
        display_health_check()

    with tab4:
        display_migration_runner()

    st.markdown("---")
    display_system_info()


if __name__ == "__main__":
    main()
