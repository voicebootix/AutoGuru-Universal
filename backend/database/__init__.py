"""
Database module for AutoGuru Universal

This module provides PostgreSQL database connectivity and management
that works universally across all business niches.
"""

from backend.database.connection import (
    PostgreSQLConnectionManager,
    get_db_manager,
    close_db_manager,
    DatabaseConnectionError,
    DatabasePoolError,
    DatabaseQueryError,
    DatabaseTransactionError,
)

__all__ = [
    "PostgreSQLConnectionManager",
    "get_db_manager",
    "close_db_manager",
    "DatabaseConnectionError",
    "DatabasePoolError",
    "DatabaseQueryError",
    "DatabaseTransactionError",
]
