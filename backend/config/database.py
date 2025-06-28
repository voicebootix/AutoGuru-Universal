"""
Database Configuration for AutoGuru Universal

This module provides environment-based database configuration that works
across all business niches. Uses pydantic-settings for validation and
environment variable management.
"""

import os
from typing import Optional, Dict, Any
from pydantic import Field, validator, PostgresDsn
from pydantic_settings import BaseSettings
from functools import lru_cache


class DatabaseSettings(BaseSettings):
    """
    Database configuration settings with environment variable support.
    Works universally across all business niches without hardcoded logic.
    """
    
    # PostgreSQL Connection Settings
    postgres_host: str = Field(
        default="localhost",
        env="POSTGRES_HOST",
        description="PostgreSQL host address"
    )
    postgres_port: int = Field(
        default=5432,
        env="POSTGRES_PORT",
        description="PostgreSQL port number"
    )
    postgres_user: str = Field(
        default="autoguru",
        env="POSTGRES_USER",
        description="PostgreSQL username"
    )
    postgres_password: str = Field(
        ...,
        env="POSTGRES_PASSWORD",
        description="PostgreSQL password (required)"
    )
    postgres_db: str = Field(
        default="autoguru_universal",
        env="POSTGRES_DB",
        description="PostgreSQL database name"
    )
    
    # Connection Pool Settings
    pool_min_size: int = Field(
        default=10,
        env="DB_POOL_MIN_SIZE",
        description="Minimum number of connections in pool"
    )
    pool_max_size: int = Field(
        default=20,
        env="DB_POOL_MAX_SIZE",
        description="Maximum number of connections in pool"
    )
    pool_max_queries: int = Field(
        default=50000,
        env="DB_POOL_MAX_QUERIES",
        description="Maximum queries per connection before recycling"
    )
    pool_max_inactive_connection_lifetime: float = Field(
        default=300.0,
        env="DB_POOL_MAX_INACTIVE_LIFETIME",
        description="Maximum seconds a connection can be idle"
    )
    
    # Retry Configuration
    retry_attempts: int = Field(
        default=3,
        env="DB_RETRY_ATTEMPTS",
        description="Number of retry attempts for failed operations"
    )
    retry_delay: float = Field(
        default=1.0,
        env="DB_RETRY_DELAY",
        description="Initial delay between retries in seconds"
    )
    retry_max_delay: float = Field(
        default=10.0,
        env="DB_RETRY_MAX_DELAY",
        description="Maximum delay between retries in seconds"
    )
    
    # Query Configuration
    query_timeout: int = Field(
        default=30,
        env="DB_QUERY_TIMEOUT",
        description="Query timeout in seconds"
    )
    statement_cache_size: int = Field(
        default=20,
        env="DB_STATEMENT_CACHE_SIZE",
        description="Number of prepared statements to cache"
    )
    
    # SSL Configuration
    ssl_mode: Optional[str] = Field(
        default=None,
        env="DB_SSL_MODE",
        description="SSL mode (disable, require, verify-ca, verify-full)"
    )
    ssl_cert_path: Optional[str] = Field(
        default=None,
        env="DB_SSL_CERT_PATH",
        description="Path to SSL certificate file"
    )
    ssl_key_path: Optional[str] = Field(
        default=None,
        env="DB_SSL_KEY_PATH",
        description="Path to SSL key file"
    )
    ssl_ca_path: Optional[str] = Field(
        default=None,
        env="DB_SSL_CA_PATH",
        description="Path to SSL CA certificate file"
    )
    
    # Environment Configuration
    environment: str = Field(
        default="development",
        env="ENVIRONMENT",
        description="Current environment (development, staging, production)"
    )
    
    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @validator("pool_max_size")
    def validate_pool_size(cls, v, values):
        """Ensure max pool size is greater than min pool size"""
        min_size = values.get("pool_min_size", 10)
        if v < min_size:
            raise ValueError(f"pool_max_size ({v}) must be >= pool_min_size ({min_size})")
        return v
    
    @validator("postgres_password")
    def validate_password(cls, v):
        """Ensure password is provided and secure"""
        if not v:
            raise ValueError("PostgreSQL password is required")
        if len(v) < 8 and os.getenv("ENVIRONMENT") == "production":
            raise ValueError("Password must be at least 8 characters in production")
        return v
    
    @property
    def database_url(self) -> str:
        """Generate PostgreSQL connection URL"""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    @property
    def asyncpg_url(self) -> str:
        """Generate asyncpg-specific connection URL"""
        return self.database_url.replace("postgresql://", "postgres://")
    
    @property
    def ssl_context(self) -> Optional[Dict[str, Any]]:
        """Generate SSL context configuration for asyncpg"""
        if not self.ssl_mode:
            return None
        
        ssl_config = {"ssl": self.ssl_mode}
        
        if self.ssl_cert_path:
            ssl_config["ssl_cert"] = self.ssl_cert_path
        if self.ssl_key_path:
            ssl_config["ssl_key"] = self.ssl_key_path
        if self.ssl_ca_path:
            ssl_config["ssl_ca"] = self.ssl_ca_path
            
        return ssl_config
    
    def get_pool_config(self) -> Dict[str, Any]:
        """Get connection pool configuration dictionary"""
        config = {
            "min_size": self.pool_min_size,
            "max_size": self.pool_max_size,
            "max_queries": self.pool_max_queries,
            "max_inactive_connection_lifetime": self.pool_max_inactive_connection_lifetime,
            "command_timeout": self.query_timeout,
            "statement_cache_size": self.statement_cache_size,
        }
        
        # Add SSL configuration if present
        if self.ssl_context:
            config.update(self.ssl_context)
            
        return config
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment.lower() == "development"


@lru_cache()
def get_database_settings() -> DatabaseSettings:
    """
    Get cached database settings instance.
    
    Returns:
        DatabaseSettings instance with configuration from environment
    """
    return DatabaseSettings()


# Export settings instance for easy access
db_settings = get_database_settings()