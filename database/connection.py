"""
Database connection utilities for Railway PostgreSQL
Provides connection management and session handling
"""

import os
from typing import Optional, Tuple
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from database.models import Base


# Default configuration (can be overridden by environment variables)
DEFAULT_CONFIG = {
    'POSTGRES_HOST': 'localhost',
    'POSTGRES_PORT': '5432',
    'POSTGRES_DB': 'pv_uncertainty',
    'POSTGRES_USER': 'postgres',
    'POSTGRES_PASSWORD': '',
    'DATABASE_URL': None,  # Full URL takes precedence
}

# Store the last connection error for debugging
_last_connection_error: Optional[str] = None


def _get_streamlit_secrets():
    """
    Try to get database URL from Streamlit secrets.
    Returns None if not running in Streamlit or secrets not configured.
    """
    try:
        import streamlit as st
        # Check if DATABASE_URL is in secrets
        if hasattr(st, 'secrets') and 'DATABASE_URL' in st.secrets:
            return st.secrets['DATABASE_URL']
        # Also check for nested postgres section
        if hasattr(st, 'secrets') and 'postgres' in st.secrets:
            pg = st.secrets['postgres']
            password = pg.get('password', '')
            if password:
                return f"postgresql://{pg['user']}:{password}@{pg['host']}:{pg['port']}/{pg['database']}"
            else:
                return f"postgresql://{pg['user']}@{pg['host']}:{pg['port']}/{pg['database']}"
    except (ImportError, AttributeError, KeyError):
        pass
    return None


def get_database_url() -> str:
    """
    Get database URL from multiple sources.

    Priority:
    1. Streamlit secrets (for Streamlit Cloud deployment)
    2. DATABASE_URL environment variable (Railway sets this automatically)
    3. Constructed from individual POSTGRES_* environment variables

    Returns:
        PostgreSQL connection URL
    """
    # Check Streamlit secrets first (for Streamlit Cloud)
    database_url = _get_streamlit_secrets()

    # Check for Railway-style DATABASE_URL in environment
    if not database_url:
        database_url = os.environ.get('DATABASE_URL')

    if database_url:
        # Railway uses postgres:// but SQLAlchemy 2.0 requires postgresql://
        if database_url.startswith('postgres://'):
            database_url = database_url.replace('postgres://', 'postgresql://', 1)
        return database_url

    # Construct from individual variables
    host = os.environ.get('POSTGRES_HOST', DEFAULT_CONFIG['POSTGRES_HOST'])
    port = os.environ.get('POSTGRES_PORT', DEFAULT_CONFIG['POSTGRES_PORT'])
    database = os.environ.get('POSTGRES_DB', DEFAULT_CONFIG['POSTGRES_DB'])
    user = os.environ.get('POSTGRES_USER', DEFAULT_CONFIG['POSTGRES_USER'])
    password = os.environ.get('POSTGRES_PASSWORD', DEFAULT_CONFIG['POSTGRES_PASSWORD'])

    if password:
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"
    else:
        return f"postgresql://{user}@{host}:{port}/{database}"


def get_engine(database_url: Optional[str] = None, echo: bool = False):
    """
    Create and return a SQLAlchemy engine.

    Args:
        database_url: Optional explicit database URL
        echo: Whether to log SQL statements

    Returns:
        SQLAlchemy Engine instance
    """
    url = database_url or get_database_url()

    # Determine if SSL is needed (Railway and other cloud PostgreSQL require SSL)
    # Check if this is an external/cloud connection (not localhost)
    from urllib.parse import urlparse
    parsed = urlparse(url)
    is_external = parsed.hostname and parsed.hostname not in ('localhost', '127.0.0.1', '::1')

    # Railway uses *.railway.app or *.rlwy.net domains
    is_railway = parsed.hostname and ('railway' in parsed.hostname or 'rlwy.net' in parsed.hostname)

    # Build connection arguments
    connect_args = {}
    if is_external or is_railway:
        # Railway PostgreSQL requires SSL for external connections
        # Use sslmode=require for secure connection
        connect_args['sslmode'] = 'require'
        # Set connection timeout for faster failure detection
        connect_args['connect_timeout'] = 10

    engine = create_engine(
        url,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_pre_ping=True,  # Check connection health before use
        echo=echo,
        connect_args=connect_args
    )

    return engine


# Global engine and session factory (lazy initialization)
_engine = None
_SessionLocal = None


def _get_engine():
    """Get or create the global engine."""
    global _engine
    if _engine is None:
        _engine = get_engine()
    return _engine


def _get_session_factory():
    """Get or create the global session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=_get_engine()
        )
    return _SessionLocal


def reset_engine():
    """
    Reset the global engine and session factory.
    Call this after changing database configuration to force reconnection.
    """
    global _engine, _SessionLocal, _last_connection_error
    if _engine is not None:
        try:
            _engine.dispose()
        except Exception:
            pass
    _engine = None
    _SessionLocal = None
    _last_connection_error = None


def get_last_error() -> Optional[str]:
    """Get the last connection error message."""
    return _last_connection_error


def get_session() -> Session:
    """
    Create a new database session.

    Returns:
        SQLAlchemy Session instance

    Usage:
        session = get_session()
        try:
            # Do database operations
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    """
    SessionLocal = _get_session_factory()
    return SessionLocal()


@contextmanager
def session_scope():
    """
    Provide a transactional scope around a series of operations.

    Usage:
        with session_scope() as session:
            user = session.query(User).filter_by(email='test@test.com').first()
            user.name = 'New Name'
            # Automatically commits on success, rolls back on error
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_database(drop_existing: bool = False):
    """
    Initialize the database schema.

    Args:
        drop_existing: If True, drops all existing tables first

    Warning:
        Setting drop_existing=True will DELETE all data!
    """
    engine = _get_engine()

    if drop_existing:
        Base.metadata.drop_all(bind=engine)
        print("Dropped all existing tables.")

    Base.metadata.create_all(bind=engine)
    print("Database schema created successfully.")


def check_connection() -> bool:
    """
    Test the database connection.

    Returns:
        True if connection is successful, False otherwise
    """
    global _last_connection_error
    try:
        engine = _get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        _last_connection_error = None
        return True
    except Exception as e:
        error_msg = str(e)
        _last_connection_error = error_msg
        print(f"Database connection failed: {error_msg}")
        return False


def check_connection_detailed() -> Tuple[bool, Optional[str]]:
    """
    Test database connection and return detailed status.

    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    global _last_connection_error
    try:
        # Reset engine to ensure fresh connection attempt
        reset_engine()
        engine = _get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        _last_connection_error = None
        return True, None
    except Exception as e:
        error_msg = str(e)
        _last_connection_error = error_msg
        return False, error_msg


def get_connection_info() -> dict:
    """
    Get information about the database connection.

    Returns:
        Dictionary with connection details (sanitized)
    """
    url = get_database_url()

    # Parse URL to extract info (without password)
    from urllib.parse import urlparse
    parsed = urlparse(url)

    return {
        'host': parsed.hostname,
        'port': parsed.port,
        'database': parsed.path.lstrip('/'),
        'user': parsed.username,
        'connected': check_connection()
    }


# Dependency injection for FastAPI/Streamlit
def get_db():
    """
    Dependency generator for database sessions.

    Usage in FastAPI:
        @app.get("/users")
        def get_users(db: Session = Depends(get_db)):
            return db.query(User).all()

    Usage in Streamlit:
        def get_data():
            db = next(get_db())
            try:
                return db.query(Measurement).all()
            finally:
                db.close()
    """
    db = get_session()
    try:
        yield db
    finally:
        db.close()


# CLI utilities
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Database management utilities')
    parser.add_argument('--init', action='store_true', help='Initialize database schema')
    parser.add_argument('--drop', action='store_true', help='Drop existing tables before init')
    parser.add_argument('--check', action='store_true', help='Check database connection')
    parser.add_argument('--info', action='store_true', help='Show connection info')

    args = parser.parse_args()

    if args.info:
        info = get_connection_info()
        print("Database Connection Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

    if args.check:
        if check_connection():
            print("Database connection: OK")
        else:
            print("Database connection: FAILED")
            exit(1)

    if args.init:
        init_database(drop_existing=args.drop)
