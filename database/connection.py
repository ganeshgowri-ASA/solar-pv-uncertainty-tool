"""
Database connection utilities for Railway PostgreSQL
Provides connection management and session handling
"""

import os
from typing import Optional
from contextlib import contextmanager
from sqlalchemy import create_engine
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


def get_database_url() -> str:
    """
    Get database URL from environment variables.

    Priority:
    1. DATABASE_URL (Railway sets this automatically)
    2. Constructed from individual POSTGRES_* variables

    Returns:
        PostgreSQL connection URL
    """
    # Check for Railway-style DATABASE_URL first
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

    engine = create_engine(
        url,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_pre_ping=True,  # Check connection health before use
        echo=echo
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
    try:
        engine = _get_engine()
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False


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
