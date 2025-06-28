"""
Configuration module for AutoGuru Universal

This module provides centralized configuration management
that works universally across all business niches.
"""

from backend.config.database import (
    DatabaseSettings,
    get_database_settings,
    db_settings,
)

__all__ = [
    "DatabaseSettings",
    "get_database_settings",
    "db_settings",
]