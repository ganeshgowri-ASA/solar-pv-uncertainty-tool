"""
Database utilities for Solar PV Uncertainty Tool
Handles PostgreSQL connection, migrations, and CRUD operations
Compatible with SQLAlchemy 2.0 and Railway PostgreSQL
"""

import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
import hashlib

# SQLAlchemy 2.0 imports
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy.pool import QueuePool


@dataclass
class MigrationResult:
    """Result of a migration operation."""
    success: bool
    message: str
    version: Optional[str] = None
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None


@dataclass
class ConnectionStatus:
    """Database connection status."""
    connected: bool
    database_name: Optional[str] = None
    host: Optional[str] = None
    version: Optional[str] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None


class DatabaseManager:
    """
    Manages PostgreSQL database connections and migrations.
    Uses SQLAlchemy 2.0 with text() wrapper for raw SQL.
    """

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database manager.

        Args:
            database_url: PostgreSQL connection string. If not provided,
                         reads from DATABASE_URL environment variable.
        """
        self.database_url = database_url or os.environ.get('DATABASE_URL')
        self._engine: Optional[Engine] = None
        self.migrations_dir = Path(__file__).parent / 'migrations'

    @property
    def engine(self) -> Optional[Engine]:
        """Get or create SQLAlchemy engine."""
        if self._engine is None and self.database_url:
            try:
                # Handle Railway's postgres:// vs postgresql:// URL format
                url = self.database_url
                if url.startswith('postgres://'):
                    url = url.replace('postgres://', 'postgresql://', 1)

                self._engine = create_engine(
                    url,
                    poolclass=QueuePool,
                    pool_size=5,
                    max_overflow=10,
                    pool_timeout=30,
                    pool_recycle=1800,
                    echo=False
                )
            except Exception as e:
                print(f"Failed to create engine: {e}")
                return None
        return self._engine

    def test_connection(self) -> ConnectionStatus:
        """
        Test database connection and return status.

        Returns:
            ConnectionStatus with connection details
        """
        if not self.database_url:
            return ConnectionStatus(
                connected=False,
                error="DATABASE_URL not configured"
            )

        start_time = datetime.now()

        try:
            if self.engine is None:
                return ConnectionStatus(
                    connected=False,
                    error="Failed to create database engine"
                )

            with self.engine.connect() as conn:
                # Test query with SQLAlchemy 2.0 text() wrapper
                result = conn.execute(text("SELECT version(), current_database()"))
                row = result.fetchone()

                latency = (datetime.now() - start_time).total_seconds() * 1000

                # Parse host from URL (safely)
                host = "Unknown"
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(self.database_url)
                    host = parsed.hostname or "Unknown"
                except:
                    pass

                return ConnectionStatus(
                    connected=True,
                    database_name=row[1] if row else None,
                    host=host,
                    version=row[0].split(',')[0] if row else None,
                    latency_ms=round(latency, 2)
                )

        except OperationalError as e:
            return ConnectionStatus(
                connected=False,
                error=f"Connection failed: {str(e)}"
            )
        except Exception as e:
            return ConnectionStatus(
                connected=False,
                error=f"Unexpected error: {str(e)}"
            )

    def get_applied_migrations(self) -> List[Dict[str, Any]]:
        """
        Get list of applied migrations from database.

        Returns:
            List of migration records
        """
        if self.engine is None:
            return []

        try:
            with self.engine.connect() as conn:
                # Check if schema_migrations table exists
                result = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = 'schema_migrations'
                    )
                """))
                exists = result.scalar()

                if not exists:
                    return []

                # Get applied migrations
                result = conn.execute(text("""
                    SELECT version, name, applied_at, checksum
                    FROM schema_migrations
                    ORDER BY version ASC
                """))

                migrations = []
                for row in result:
                    migrations.append({
                        'version': row[0],
                        'name': row[1],
                        'applied_at': row[2],
                        'checksum': row[3]
                    })

                return migrations

        except Exception as e:
            print(f"Error getting migrations: {e}")
            return []

    def get_pending_migrations(self) -> List[Dict[str, str]]:
        """
        Get list of pending (unapplied) migrations.

        Returns:
            List of pending migration files
        """
        if not self.migrations_dir.exists():
            return []

        applied = {m['version'] for m in self.get_applied_migrations()}
        pending = []

        # Find all UP migration files
        for file_path in sorted(self.migrations_dir.glob('*_UP.sql')):
            # Extract version from filename (e.g., "001" from "001_initial_schema_UP.sql")
            version = file_path.stem.split('_')[0]

            if version not in applied:
                pending.append({
                    'version': version,
                    'name': file_path.stem.replace('_UP', ''),
                    'file_path': str(file_path)
                })

        # Also check for files without _UP suffix (like 001_initial_schema.sql)
        for file_path in sorted(self.migrations_dir.glob('[0-9]*.sql')):
            if '_UP' in file_path.stem or '_DOWN' in file_path.stem:
                continue

            version = file_path.stem.split('_')[0]

            if version not in applied and not any(p['version'] == version for p in pending):
                pending.append({
                    'version': version,
                    'name': file_path.stem,
                    'file_path': str(file_path)
                })

        return sorted(pending, key=lambda x: x['version'])

    def _calculate_checksum(self, content: str) -> str:
        """Calculate MD5 checksum of migration content."""
        return hashlib.md5(content.encode()).hexdigest()

    def run_migration(self, migration: Dict[str, str]) -> MigrationResult:
        """
        Run a single migration file.

        Args:
            migration: Dict with version, name, file_path

        Returns:
            MigrationResult with success status
        """
        if self.engine is None:
            return MigrationResult(
                success=False,
                message="Database not connected",
                error="No database engine available"
            )

        file_path = Path(migration['file_path'])
        if not file_path.exists():
            return MigrationResult(
                success=False,
                message=f"Migration file not found: {file_path}",
                error="File not found"
            )

        start_time = datetime.now()

        try:
            # Read migration SQL
            sql_content = file_path.read_text()
            checksum = self._calculate_checksum(sql_content)

            # Filter out DOWN migration section (commented block)
            # Split on the DOWN MIGRATION comment and take only the UP part
            if '-- DOWN MIGRATION' in sql_content:
                sql_content = sql_content.split('-- DOWN MIGRATION')[0]
            elif '-- ============================================\n-- DOWN' in sql_content:
                sql_content = sql_content.split('-- ============================================\n-- DOWN')[0]

            with self.engine.begin() as conn:
                # Execute migration SQL
                # Split by semicolons and execute each statement
                statements = [s.strip() for s in sql_content.split(';') if s.strip()]

                for statement in statements:
                    # Skip empty statements and comments-only statements
                    clean_stmt = '\n'.join(
                        line for line in statement.split('\n')
                        if line.strip() and not line.strip().startswith('--')
                    )
                    if clean_stmt:
                        conn.execute(text(statement))

                # Record migration (if not already recorded by the migration itself)
                conn.execute(text("""
                    INSERT INTO schema_migrations (version, name, checksum)
                    VALUES (:version, :name, :checksum)
                    ON CONFLICT (version) DO UPDATE SET
                        name = EXCLUDED.name,
                        checksum = EXCLUDED.checksum,
                        applied_at = CURRENT_TIMESTAMP
                """), {
                    'version': migration['version'],
                    'name': migration['name'],
                    'checksum': checksum
                })

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return MigrationResult(
                success=True,
                message=f"Migration {migration['version']} applied successfully",
                version=migration['version'],
                execution_time_ms=round(execution_time, 2)
            )

        except SQLAlchemyError as e:
            return MigrationResult(
                success=False,
                message=f"Migration {migration['version']} failed",
                version=migration['version'],
                error=str(e)
            )
        except Exception as e:
            return MigrationResult(
                success=False,
                message=f"Unexpected error in migration {migration['version']}",
                version=migration['version'],
                error=str(e)
            )

    def run_all_pending_migrations(self) -> List[MigrationResult]:
        """
        Run all pending migrations in order.

        Returns:
            List of MigrationResult for each migration
        """
        results = []
        pending = self.get_pending_migrations()

        if not pending:
            results.append(MigrationResult(
                success=True,
                message="No pending migrations to run"
            ))
            return results

        for migration in pending:
            result = self.run_migration(migration)
            results.append(result)

            # Stop on first failure
            if not result.success:
                break

        return results

    def get_table_list(self) -> List[str]:
        """Get list of all tables in the database."""
        if self.engine is None:
            return []

        try:
            inspector = inspect(self.engine)
            return inspector.get_table_names()
        except Exception:
            return []

    def get_table_row_counts(self) -> Dict[str, int]:
        """Get row counts for all tables."""
        if self.engine is None:
            return {}

        counts = {}
        tables = self.get_table_list()

        try:
            with self.engine.connect() as conn:
                for table in tables:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    counts[table] = result.scalar() or 0
        except Exception:
            pass

        return counts


# Singleton instance for app-wide use
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get or create the singleton DatabaseManager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def init_database_from_secrets(secrets: Dict[str, Any]) -> DatabaseManager:
    """
    Initialize database manager from Streamlit secrets.

    Args:
        secrets: Streamlit secrets dict containing DATABASE_URL

    Returns:
        Configured DatabaseManager
    """
    global _db_manager

    database_url = secrets.get('DATABASE_URL') or secrets.get('database', {}).get('url')

    if database_url:
        _db_manager = DatabaseManager(database_url)
    else:
        _db_manager = DatabaseManager()

    return _db_manager
