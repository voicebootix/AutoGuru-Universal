"""
AutoGuru Universal - Production Configuration

Production-specific settings for Render deployment.
"""

import os
from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

class DatabaseConfig(BaseModel):
    """Database configuration for production"""
    database_url: str = Field(..., alias="DATABASE_URL", description="Database connection URL")
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Max overflow connections")
    echo: bool = Field(default=False, description="Enable SQL echo")
    
    class Config:
        extra = "allow"

class RedisConfig(BaseModel):
    """Redis configuration for production"""
    url: str = Field(default="redis://localhost:6379", alias="REDIS_URL", description="Redis connection URL")
    max_connections: int = Field(default=20, description="Max Redis connections")
    
    class Config:
        extra = "allow"

class CeleryConfig(BaseModel):
    """Celery configuration for production"""
    broker_url: str = Field(default="redis://localhost:6379", alias="CELERY_BROKER_URL", description="Celery broker URL")
    result_backend: str = Field(default="redis://localhost:6379", alias="CELERY_RESULT_BACKEND", description="Celery result backend")
    task_serializer: str = Field(default="json", description="Task serializer")
    accept_content: List[str] = Field(default=["json"], description="Accepted content types")
    result_serializer: str = Field(default="json", description="Result serializer")
    timezone: str = Field(default="UTC", description="Timezone")
    enable_utc: bool = Field(default=True, description="Enable UTC")
    
    class Config:
        extra = "allow"

class LoggingConfig(BaseModel):
    """Logging configuration for production"""
    level: LogLevel = Field(default=LogLevel.INFO, description="Log level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    enable_file_logging: bool = Field(default=False, description="Enable file logging")
    log_file_path: str = Field(default="logs/autoguru.log", description="Log file path")
    
    class Config:
        extra = "allow"

class SecurityConfig(BaseModel):
    """Security configuration for production"""
    secret_key: str = Field(..., alias="SECRET_KEY", description="Secret key for encryption")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiry")
    cors_origins: List[str] = Field(default=["*"], description="CORS origins")
    
    class Config:
        extra = "allow"

class ProductionSettings(BaseModel):
    """Production settings for AutoGuru Universal"""
    
    # Environment
    environment: str = Field(default="production", description="Environment name")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Database
    database: DatabaseConfig = Field(..., description="Database configuration")
    
    # Redis
    redis: RedisConfig = Field(..., description="Redis configuration")
    
    # Celery
    celery: CeleryConfig = Field(..., description="Celery configuration")
    
    # Logging
    logging: LoggingConfig = Field(..., description="Logging configuration")
    
    # Security
    security: SecurityConfig = Field(..., description="Security configuration")
    
    # API Configuration
    api_prefix: str = Field(default="/api/v1", description="API prefix")
    title: str = Field(default="AutoGuru Universal", description="API title")
    description: str = Field(
        default="Universal social media automation for ANY business niche",
        description="API description"
    )
    version: str = Field(default="1.0.0", description="API version")
    
    # AI Services
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API key")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, description="Rate limit requests per minute")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")
    
    # Content Generation
    max_content_length: int = Field(default=10000, description="Maximum content length")
    max_analysis_tokens: int = Field(default=500, description="Maximum analysis tokens")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"  # Allow extra fields to prevent validation errors

def get_production_settings() -> ProductionSettings:
    """Get production settings from environment variables"""
    
    # Database URL from Render
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is required")
    
    # Secret key
    secret_key = os.getenv("SECRET_KEY")
    if not secret_key:
        raise ValueError("SECRET_KEY environment variable is required")
    
    # CORS origins
    cors_origins_str = os.getenv("CORS_ORIGINS", "*")
    cors_origins = cors_origins_str.split(",") if cors_origins_str != "*" else ["*"]
    
    # Redis URL
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Celery URLs
    celery_broker_url = os.getenv("CELERY_BROKER_URL", redis_url)
    celery_result_backend = os.getenv("CELERY_RESULT_BACKEND", redis_url)
    
    # Log level
    log_level = os.getenv("LOG_LEVEL", "INFO")
    
    return ProductionSettings(
        environment=os.getenv("ENVIRONMENT", "production"),
        debug=os.getenv("DEBUG", "false").lower() == "true",
        
        database=DatabaseConfig(
            database_url=database_url,
            pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
            echo=os.getenv("DB_ECHO", "false").lower() == "true"
        ),
        
        redis=RedisConfig(
            url=redis_url,
            max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "20"))
        ),
        
        celery=CeleryConfig(
            broker_url=celery_broker_url,
            result_backend=celery_result_backend,
            task_serializer=os.getenv("CELERY_TASK_SERIALIZER", "json"),
            accept_content=os.getenv("CELERY_ACCEPT_CONTENT", "json").split(","),
            result_serializer=os.getenv("CELERY_RESULT_SERIALIZER", "json"),
            timezone=os.getenv("CELERY_TIMEZONE", "UTC"),
            enable_utc=os.getenv("CELERY_ENABLE_UTC", "true").lower() == "true"
        ),
        
        logging=LoggingConfig(
            level=LogLevel(log_level),
            format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            enable_file_logging=os.getenv("LOG_FILE_ENABLED", "false").lower() == "true",
            log_file_path=os.getenv("LOG_FILE_PATH", "logs/autoguru.log")
        ),
        
        security=SecurityConfig(
            secret_key=secret_key,
            algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
            access_token_expire_minutes=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")),
            cors_origins=cors_origins
        ),
        
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        
        rate_limit_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
        rate_limit_window=int(os.getenv("RATE_LIMIT_WINDOW", "60")),
        
        max_content_length=int(os.getenv("MAX_CONTENT_LENGTH", "10000")),
        max_analysis_tokens=int(os.getenv("MAX_ANALYSIS_TOKENS", "500"))
    )

# Global settings instance
production_settings = get_production_settings() 