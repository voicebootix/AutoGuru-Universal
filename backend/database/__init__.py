"""
Database module for AutoGuru Universal

This module provides PostgreSQL database connectivity and management
that works universally across all business niches.
"""

from backend.database.connection import (
    get_db_session,
    get_db_context,
    init_database,
    close_database,
    check_database_connection,
    Base,
)

__all__ = [
    "get_db_session",
    "get_db_context",
    "init_database",
    "close_database",
    "check_database_connection",
    "Base",
]
