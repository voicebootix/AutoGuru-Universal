"""
AutoGuru Universal - Application Settings Management

This module handles all configuration settings for the AutoGuru Universal platform.
It uses Pydantic settings for validation and supports multiple environments.
All sensitive data is encrypted and API keys are managed securely.
"""

import os
from typing import Dict, List, Optional, Any
from enum import Enum
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field, validator, SecretStr, HttpUrl, PostgresDsn, RedisDsn
from cryptography.fernet import Fernet


class Environment(str, Enum):
    """Application environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging level configurations"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class RateLimitConfig(BaseSettings):
    """Rate limiting configuration for API endpoints and platform integrations"""
    
    # Global rate limits
    requests_per_minute: int = Field(60, description="Global requests per minute")
    requests_per_hour: int = Field(3600, description="Global requests per hour")
    
    # Platform-specific rate limits (universal defaults that work for any business)
    twitter_requests_per_15min: int = Field(300, description="Twitter API rate limit")
    linkedin_requests_per_day: int = Field(100, description="LinkedIn API rate limit")
    facebook_requests_per_hour: int = Field(200, description="Facebook API rate limit")
    instagram_requests_per_hour: int = Field(200, description="Instagram API rate limit")
    youtube_requests_per_day: int = Field(10000, description="YouTube API quota")
    tiktok_requests_per_hour: int = Field(100, description="TikTok API rate limit")
    
    # AI service rate limits
    openai_requests_per_minute: int = Field(60, description="OpenAI API rate limit")
    anthropic_requests_per_minute: int = Field(50, description="Anthropic API rate limit")
    
    class Config:
        env_prefix = "RATE_LIMIT_"


class DatabaseConfig(BaseSettings):
    """Database configuration settings"""
    
    # PostgreSQL settings
    postgres_user: str = Field(..., description="PostgreSQL username")
    postgres_password: SecretStr = Field(..., description="PostgreSQL password")
    postgres_host: str = Field("localhost", description="PostgreSQL host")
    postgres_port: int = Field(5432, description="PostgreSQL port")
    postgres_db: str = Field("autoguru_universal", description="PostgreSQL database name")
    
    # Connection pool settings
    pool_size: int = Field(20, description="Database connection pool size")
    max_overflow: int = Field(10, description="Maximum overflow connections")
    pool_timeout: int = Field(30, description="Pool timeout in seconds")
    pool_recycle: int = Field(3600, description="Pool recycle time in seconds")
    
    # Redis settings
    redis_host: str = Field("localhost", description="Redis host")
    redis_port: int = Field(6379, description="Redis port")
    redis_password: Optional[SecretStr] = Field(None, description="Redis password")
    redis_db: int = Field(0, description="Redis database number")
    redis_decode_responses: bool = Field(True, description="Decode Redis responses")
    
    @property
    def postgres_dsn(self) -> PostgresDsn:
        """Construct PostgreSQL DSN"""
        return PostgresDsn.build(
            scheme="postgresql+asyncpg",
            user=self.postgres_user,
            password=self.postgres_password.get_secret_value(),
            host=self.postgres_host,
            port=str(self.postgres_port),
            path=f"/{self.postgres_db}",
        )
    
    @property
    def redis_dsn(self) -> RedisDsn:
        """Construct Redis DSN"""
        if self.redis_password:
            return RedisDsn.build(
                scheme="redis",
                user=None,
                password=self.redis_password.get_secret_value(),
                host=self.redis_host,
                port=str(self.redis_port),
                path=f"/{self.redis_db}",
            )
        return RedisDsn.build(
            scheme="redis",
            host=self.redis_host,
            port=str(self.redis_port),
            path=f"/{self.redis_db}",
        )
    
    class Config:
        env_prefix = "DB_"


class AIServiceConfig(BaseSettings):
    """AI service configuration for universal business support"""
    
    # OpenAI configuration
    openai_api_key: SecretStr = Field(..., description="OpenAI API key")
    openai_model: str = Field("gpt-4-turbo-preview", description="Default OpenAI model")
    openai_temperature: float = Field(0.7, description="OpenAI temperature setting")
    openai_max_tokens: int = Field(2000, description="Maximum tokens for OpenAI responses")
    
    # Anthropic configuration
    anthropic_api_key: SecretStr = Field(..., description="Anthropic API key")
    anthropic_model: str = Field("claude-3-opus-20240229", description="Default Anthropic model")
    anthropic_max_tokens: int = Field(2000, description="Maximum tokens for Anthropic responses")
    
    # Universal AI strategy settings
    enable_niche_detection: bool = Field(True, description="Enable automatic business niche detection")
    enable_content_adaptation: bool = Field(True, description="Enable AI-driven content adaptation")
    enable_strategy_optimization: bool = Field(True, description="Enable AI strategy optimization")
    
    class Config:
        env_prefix = "AI_"


class SocialMediaConfig(BaseSettings):
    """Social media platform API configurations"""
    
    # Twitter/X configuration
    twitter_api_key: Optional[SecretStr] = Field(None, description="Twitter API key")
    twitter_api_secret: Optional[SecretStr] = Field(None, description="Twitter API secret")
    twitter_access_token: Optional[SecretStr] = Field(None, description="Twitter access token")
    twitter_access_token_secret: Optional[SecretStr] = Field(None, description="Twitter access token secret")
    twitter_bearer_token: Optional[SecretStr] = Field(None, description="Twitter bearer token")
    
    # LinkedIn configuration
    linkedin_client_id: Optional[SecretStr] = Field(None, description="LinkedIn client ID")
    linkedin_client_secret: Optional[SecretStr] = Field(None, description="LinkedIn client secret")
    linkedin_redirect_uri: Optional[HttpUrl] = Field(None, description="LinkedIn OAuth redirect URI")
    
    # Facebook/Instagram configuration
    facebook_app_id: Optional[SecretStr] = Field(None, description="Facebook app ID")
    facebook_app_secret: Optional[SecretStr] = Field(None, description="Facebook app secret")
    facebook_access_token: Optional[SecretStr] = Field(None, description="Facebook access token")
    
    # YouTube configuration
    youtube_api_key: Optional[SecretStr] = Field(None, description="YouTube API key")
    youtube_client_id: Optional[SecretStr] = Field(None, description="YouTube client ID")
    youtube_client_secret: Optional[SecretStr] = Field(None, description="YouTube client secret")
    
    # TikTok configuration
    tiktok_api_key: Optional[SecretStr] = Field(None, description="TikTok API key")
    tiktok_api_secret: Optional[SecretStr] = Field(None, description="TikTok API secret")
    
    class Config:
        env_prefix = "SOCIAL_"


class SecurityConfig(BaseSettings):
    """Security configuration settings"""
    
    # Encryption settings
    encryption_key: SecretStr = Field(..., description="Master encryption key for sensitive data")
    jwt_secret_key: SecretStr = Field(..., description="JWT secret key")
    jwt_algorithm: str = Field("HS256", description="JWT algorithm")
    jwt_expiration_hours: int = Field(24, description="JWT token expiration in hours")
    
    # CORS settings
    allowed_origins: List[str] = Field(["http://localhost:3000"], description="Allowed CORS origins")
    allowed_methods: List[str] = Field(["GET", "POST", "PUT", "DELETE"], description="Allowed HTTP methods")
    allowed_headers: List[str] = Field(["*"], description="Allowed headers")
    
    # Security headers
    enable_security_headers: bool = Field(True, description="Enable security headers")
    enable_rate_limiting: bool = Field(True, description="Enable rate limiting")
    enable_request_validation: bool = Field(True, description="Enable request validation")
    
    @property
    def fernet_key(self) -> Fernet:
        """Get Fernet encryption instance"""
        return Fernet(self.encryption_key.get_secret_value().encode())
    
    class Config:
        env_prefix = "SECURITY_"
    
    @validator("encryption_key")
    def validate_encryption_key(cls, v: SecretStr) -> SecretStr:
        """Validate encryption key format"""
        key = v.get_secret_value()
        try:
            Fernet(key.encode())
        except Exception:
            raise ValueError("Invalid encryption key format. Use Fernet.generate_key() to create a valid key.")
        return v


class LoggingConfig(BaseSettings):
    """Logging configuration settings"""
    
    # General logging
    log_level: LogLevel = Field(LogLevel.INFO, description="Default logging level")
    log_format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    log_json_format: bool = Field(True, description="Use JSON format for logs")
    
    # File logging
    enable_file_logging: bool = Field(True, description="Enable file logging")
    log_file_path: Path = Field(Path("logs/autoguru.log"), description="Log file path")
    log_file_rotation: str = Field("midnight", description="Log file rotation schedule")
    log_file_retention_days: int = Field(30, description="Log file retention in days")
    log_file_max_bytes: int = Field(10485760, description="Maximum log file size (10MB)")
    
    # Per-module logging levels
    module_log_levels: Dict[str, LogLevel] = Field(
        default_factory=dict,
        description="Module-specific log levels"
    )
    
    class Config:
        env_prefix = "LOG_"


class Settings(BaseSettings):
    """Main settings class that combines all configuration sections"""
    
    # Environment
    environment: Environment = Field(Environment.DEVELOPMENT, description="Application environment")
    debug: bool = Field(False, description="Debug mode")
    
    # Application metadata
    app_name: str = Field("AutoGuru Universal", description="Application name")
    app_version: str = Field("1.0.0", description="Application version")
    app_description: str = Field(
        "Universal social media automation platform that works for any business niche",
        description="Application description"
    )
    
    # API settings
    api_v1_prefix: str = Field("/api/v1", description="API v1 prefix")
    api_docs_enabled: bool = Field(True, description="Enable API documentation")
    
    # Worker settings
    celery_broker_url: Optional[str] = Field(None, description="Celery broker URL")
    celery_result_backend: Optional[str] = Field(None, description="Celery result backend")
    worker_concurrency: int = Field(4, description="Worker concurrency")
    
    # Feature flags (universal features that work for any business)
    enable_auto_scheduling: bool = Field(True, description="Enable automatic content scheduling")
    enable_content_generation: bool = Field(True, description="Enable AI content generation")
    enable_analytics: bool = Field(True, description="Enable analytics tracking")
    enable_auto_engagement: bool = Field(True, description="Enable automatic engagement")
    enable_competitor_analysis: bool = Field(True, description="Enable competitor analysis")
    enable_trend_detection: bool = Field(True, description="Enable trend detection")
    
    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    ai_service: AIServiceConfig = Field(default_factory=AIServiceConfig)
    social_media: SocialMediaConfig = Field(default_factory=SocialMediaConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @validator("environment")
    def validate_environment(cls, v: Environment) -> Environment:
        """Validate and set environment-specific defaults"""
        if v == Environment.PRODUCTION:
            # Production-specific validations
            pass
        return v
    
    @validator("celery_broker_url", "celery_result_backend", pre=True)
    def build_celery_urls(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Build Celery URLs from Redis configuration if not explicitly set"""
        if v is None and "database" in values:
            db_config = values["database"]
            if isinstance(db_config, DatabaseConfig):
                return str(db_config.redis_dsn)
        return v
    
    def get_platform_config(self, platform: str) -> Dict[str, Any]:
        """Get configuration for a specific social media platform"""
        platform_configs = {
            "twitter": {
                "api_key": self.social_media.twitter_api_key,
                "api_secret": self.social_media.twitter_api_secret,
                "access_token": self.social_media.twitter_access_token,
                "access_token_secret": self.social_media.twitter_access_token_secret,
                "bearer_token": self.social_media.twitter_bearer_token,
                "rate_limit": self.rate_limit.twitter_requests_per_15min,
            },
            "linkedin": {
                "client_id": self.social_media.linkedin_client_id,
                "client_secret": self.social_media.linkedin_client_secret,
                "redirect_uri": self.social_media.linkedin_redirect_uri,
                "rate_limit": self.rate_limit.linkedin_requests_per_day,
            },
            "facebook": {
                "app_id": self.social_media.facebook_app_id,
                "app_secret": self.social_media.facebook_app_secret,
                "access_token": self.social_media.facebook_access_token,
                "rate_limit": self.rate_limit.facebook_requests_per_hour,
            },
            "instagram": {
                "app_id": self.social_media.facebook_app_id,  # Uses Facebook API
                "app_secret": self.social_media.facebook_app_secret,
                "access_token": self.social_media.facebook_access_token,
                "rate_limit": self.rate_limit.instagram_requests_per_hour,
            },
            "youtube": {
                "api_key": self.social_media.youtube_api_key,
                "client_id": self.social_media.youtube_client_id,
                "client_secret": self.social_media.youtube_client_secret,
                "rate_limit": self.rate_limit.youtube_requests_per_day,
            },
            "tiktok": {
                "api_key": self.social_media.tiktok_api_key,
                "api_secret": self.social_media.tiktok_api_secret,
                "rate_limit": self.rate_limit.tiktok_requests_per_hour,
            },
        }
        
        return platform_configs.get(platform.lower(), {})
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data using Fernet encryption"""
        return self.security.fernet_key.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data using Fernet encryption"""
        return self.security.fernet_key.decrypt(encrypted_data.encode()).decode()


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    This ensures we only create one instance of settings throughout the application.
    """
    return Settings()


# Create settings instance
settings = get_settings()


# Environment-specific configuration examples
def get_environment_config() -> Dict[str, Any]:
    """Get environment-specific configuration overrides"""
    env_configs = {
        Environment.DEVELOPMENT: {
            "debug": True,
            "log_level": LogLevel.DEBUG,
            "api_docs_enabled": True,
            "allowed_origins": ["http://localhost:3000", "http://localhost:8000"],
        },
        Environment.STAGING: {
            "debug": False,
            "log_level": LogLevel.INFO,
            "api_docs_enabled": True,
            "allowed_origins": ["https://staging.autoguru.com"],
        },
        Environment.PRODUCTION: {
            "debug": False,
            "log_level": LogLevel.WARNING,
            "api_docs_enabled": False,
            "allowed_origins": ["https://autoguru.com", "https://app.autoguru.com"],
        },
        Environment.TESTING: {
            "debug": True,
            "log_level": LogLevel.DEBUG,
            "api_docs_enabled": True,
            "allowed_origins": ["*"],
        },
    }
    
    return env_configs.get(settings.environment, {})