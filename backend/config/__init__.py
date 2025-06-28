"""
AutoGuru Universal - Configuration Package

This package manages all application settings and configuration.
Supports multiple environments and works universally for any business niche.
"""

from backend.config.settings import (
    Settings,
    settings,
    get_settings,
    get_environment_config,
    Environment,
    LogLevel,
    DatabaseConfig,
    AIServiceConfig,
    SocialMediaConfig,
    SecurityConfig,
    LoggingConfig,
    RateLimitConfig,
)

__all__ = [
    "Settings",
    "settings",
    "get_settings",
    "get_environment_config",
    "Environment",
    "LogLevel",
    "DatabaseConfig",
    "AIServiceConfig",
    "SocialMediaConfig",
    "SecurityConfig",
    "LoggingConfig",
    "RateLimitConfig",
]