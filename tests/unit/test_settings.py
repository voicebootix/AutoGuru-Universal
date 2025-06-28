"""
Unit tests for AutoGuru Universal settings management.

Tests all configuration classes, validation logic, and environment handling.
Ensures settings work universally for any business niche.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from typing import Dict, Any

from backend.config.settings import (
    Settings,
    Environment,
    LogLevel,
    DatabaseConfig,
    AIServiceConfig,
    SocialMediaConfig,
    SecurityConfig,
    LoggingConfig,
    RateLimitConfig,
    get_settings,
    get_environment_config
)


class TestEnvironmentEnum:
    """Test Environment enum functionality"""
    
    def test_environment_values(self):
        """Test that all environment values are correctly defined"""
        assert Environment.DEVELOPMENT == "development"
        assert Environment.STAGING == "staging"
        assert Environment.PRODUCTION == "production"
        assert Environment.TESTING == "testing"


class TestLogLevelEnum:
    """Test LogLevel enum functionality"""
    
    def test_log_level_values(self):
        """Test that all log levels are correctly defined"""
        assert LogLevel.DEBUG == "DEBUG"
        assert LogLevel.INFO == "INFO"
        assert LogLevel.WARNING == "WARNING"
        assert LogLevel.ERROR == "ERROR"
        assert LogLevel.CRITICAL == "CRITICAL"


class TestRateLimitConfig:
    """Test rate limiting configuration"""
    
    def test_default_rate_limits(self):
        """Test default rate limit values"""
        config = RateLimitConfig()
        
        # Global limits
        assert config.requests_per_minute == 60
        assert config.requests_per_hour == 3600
        
        # Platform-specific limits
        assert config.twitter_requests_per_15min == 300
        assert config.linkedin_requests_per_day == 100
        assert config.facebook_requests_per_hour == 200
        assert config.instagram_requests_per_hour == 200
        assert config.youtube_requests_per_day == 10000
        assert config.tiktok_requests_per_hour == 100
        
        # AI service limits
        assert config.openai_requests_per_minute == 60
        assert config.anthropic_requests_per_minute == 50
    
    def test_env_prefix(self):
        """Test environment variable prefix"""
        with patch.dict(os.environ, {"RATE_LIMIT_REQUESTS_PER_MINUTE": "120"}):
            config = RateLimitConfig()
            assert config.requests_per_minute == 120


class TestDatabaseConfig:
    """Test database configuration"""
    
    def test_postgres_dsn_construction(self):
        """Test PostgreSQL DSN construction"""
        with patch.dict(os.environ, {
            "DB_POSTGRES_USER": "testuser",
            "DB_POSTGRES_PASSWORD": "testpass",
            "DB_POSTGRES_HOST": "testhost",
            "DB_POSTGRES_PORT": "5433",
            "DB_POSTGRES_DB": "testdb"
        }):
            config = DatabaseConfig()
            dsn = str(config.postgres_dsn)
            
            assert "postgresql+asyncpg://" in dsn
            assert "testuser" in dsn
            assert "testpass" in dsn
            assert "testhost" in dsn
            assert "5433" in dsn
            assert "testdb" in dsn
    
    def test_redis_dsn_without_password(self):
        """Test Redis DSN construction without password"""
        config = DatabaseConfig(
            postgres_user="user",
            postgres_password="pass"
        )
        dsn = str(config.redis_dsn)
        
        assert "redis://" in dsn
        assert "localhost" in dsn
        assert "6379" in dsn
        assert "/0" in dsn
    
    def test_redis_dsn_with_password(self):
        """Test Redis DSN construction with password"""
        config = DatabaseConfig(
            postgres_user="user",
            postgres_password="dbpass",
            redis_password="redispass"
        )
        dsn = str(config.redis_dsn)
        
        assert "redis://" in dsn
        assert "redispass" in dsn
    
    def test_connection_pool_settings(self):
        """Test database connection pool settings"""
        config = DatabaseConfig(
            postgres_user="user",
            postgres_password="pass"
        )
        
        assert config.pool_size == 20
        assert config.max_overflow == 10
        assert config.pool_timeout == 30
        assert config.pool_recycle == 3600


class TestAIServiceConfig:
    """Test AI service configuration"""
    
    def test_openai_defaults(self):
        """Test OpenAI default settings"""
        config = AIServiceConfig(
            openai_api_key="test-key",
            anthropic_api_key="test-key"
        )
        
        assert config.openai_model == "gpt-4-turbo-preview"
        assert config.openai_temperature == 0.7
        assert config.openai_max_tokens == 2000
    
    def test_anthropic_defaults(self):
        """Test Anthropic default settings"""
        config = AIServiceConfig(
            openai_api_key="test-key",
            anthropic_api_key="test-key"
        )
        
        assert config.anthropic_model == "claude-3-opus-20240229"
        assert config.anthropic_max_tokens == 2000
    
    def test_universal_features(self):
        """Test universal AI features are enabled by default"""
        config = AIServiceConfig(
            openai_api_key="test-key",
            anthropic_api_key="test-key"
        )
        
        assert config.enable_niche_detection is True
        assert config.enable_content_adaptation is True
        assert config.enable_strategy_optimization is True


class TestSocialMediaConfig:
    """Test social media platform configuration"""
    
    def test_optional_platform_configs(self):
        """Test that all platform configurations are optional"""
        config = SocialMediaConfig()
        
        # All should be None by default
        assert config.twitter_api_key is None
        assert config.linkedin_client_id is None
        assert config.facebook_app_id is None
        assert config.youtube_api_key is None
        assert config.tiktok_api_key is None
    
    def test_platform_config_from_env(self):
        """Test loading platform config from environment variables"""
        with patch.dict(os.environ, {
            "SOCIAL_TWITTER_API_KEY": "twitter-key",
            "SOCIAL_LINKEDIN_CLIENT_ID": "linkedin-id",
            "SOCIAL_FACEBOOK_APP_ID": "facebook-id"
        }):
            config = SocialMediaConfig()
            
            assert config.twitter_api_key.get_secret_value() == "twitter-key"
            assert config.linkedin_client_id.get_secret_value() == "linkedin-id"
            assert config.facebook_app_id.get_secret_value() == "facebook-id"


class TestSecurityConfig:
    """Test security configuration"""
    
    def test_encryption_key_validation(self):
        """Test encryption key validation"""
        # Generate a valid Fernet key
        from cryptography.fernet import Fernet
        valid_key = Fernet.generate_key().decode()
        
        config = SecurityConfig(
            encryption_key=valid_key,
            jwt_secret_key="jwt-secret"
        )
        
        assert config.encryption_key.get_secret_value() == valid_key
    
    def test_invalid_encryption_key(self):
        """Test invalid encryption key raises error"""
        with pytest.raises(ValueError, match="Invalid encryption key format"):
            SecurityConfig(
                encryption_key="invalid-key",
                jwt_secret_key="jwt-secret"
            )
    
    def test_fernet_key_property(self):
        """Test Fernet key property returns valid instance"""
        from cryptography.fernet import Fernet
        valid_key = Fernet.generate_key().decode()
        
        config = SecurityConfig(
            encryption_key=valid_key,
            jwt_secret_key="jwt-secret"
        )
        
        assert isinstance(config.fernet_key, Fernet)
    
    def test_default_security_settings(self):
        """Test default security settings"""
        from cryptography.fernet import Fernet
        valid_key = Fernet.generate_key().decode()
        
        config = SecurityConfig(
            encryption_key=valid_key,
            jwt_secret_key="jwt-secret"
        )
        
        assert config.jwt_algorithm == "HS256"
        assert config.jwt_expiration_hours == 24
        assert config.allowed_origins == ["http://localhost:3000"]
        assert config.allowed_methods == ["GET", "POST", "PUT", "DELETE"]
        assert config.enable_security_headers is True
        assert config.enable_rate_limiting is True
        assert config.enable_request_validation is True


class TestLoggingConfig:
    """Test logging configuration"""
    
    def test_default_logging_settings(self):
        """Test default logging settings"""
        config = LoggingConfig()
        
        assert config.log_level == LogLevel.INFO
        assert config.log_json_format is True
        assert config.enable_file_logging is True
        assert config.log_file_path == Path("logs/autoguru.log")
        assert config.log_file_rotation == "midnight"
        assert config.log_file_retention_days == 30
        assert config.log_file_max_bytes == 10485760  # 10MB
    
    def test_module_log_levels(self):
        """Test module-specific log levels"""
        config = LoggingConfig(
            module_log_levels={
                "backend.core": LogLevel.DEBUG,
                "backend.api": LogLevel.WARNING
            }
        )
        
        assert config.module_log_levels["backend.core"] == LogLevel.DEBUG
        assert config.module_log_levels["backend.api"] == LogLevel.WARNING


class TestSettings:
    """Test main Settings class"""
    
    @pytest.fixture
    def valid_env_vars(self):
        """Fixture providing valid environment variables"""
        from cryptography.fernet import Fernet
        return {
            "ENVIRONMENT": "testing",
            "DEBUG": "true",
            "DB_POSTGRES_USER": "testuser",
            "DB_POSTGRES_PASSWORD": "testpass",
            "AI_OPENAI_API_KEY": "openai-test-key",
            "AI_ANTHROPIC_API_KEY": "anthropic-test-key",
            "SECURITY_ENCRYPTION_KEY": Fernet.generate_key().decode(),
            "SECURITY_JWT_SECRET_KEY": "jwt-test-secret"
        }
    
    def test_settings_initialization(self, valid_env_vars):
        """Test Settings initialization with environment variables"""
        with patch.dict(os.environ, valid_env_vars):
            settings = Settings()
            
            assert settings.environment == Environment.TESTING
            assert settings.debug is True
            assert settings.app_name == "AutoGuru Universal"
            assert settings.app_version == "1.0.0"
    
    def test_universal_features_enabled(self, valid_env_vars):
        """Test that universal features are enabled by default"""
        with patch.dict(os.environ, valid_env_vars):
            settings = Settings()
            
            # All universal features should be enabled
            assert settings.enable_auto_scheduling is True
            assert settings.enable_content_generation is True
            assert settings.enable_analytics is True
            assert settings.enable_auto_engagement is True
            assert settings.enable_competitor_analysis is True
            assert settings.enable_trend_detection is True
    
    def test_get_platform_config(self, valid_env_vars):
        """Test getting platform-specific configuration"""
        env_vars = {**valid_env_vars}
        env_vars.update({
            "SOCIAL_TWITTER_API_KEY": "twitter-key",
            "SOCIAL_TWITTER_API_SECRET": "twitter-secret",
            "SOCIAL_LINKEDIN_CLIENT_ID": "linkedin-id",
            "SOCIAL_FACEBOOK_APP_ID": "facebook-id"
        })
        
        with patch.dict(os.environ, env_vars):
            settings = Settings()
            
            # Test Twitter config
            twitter_config = settings.get_platform_config("twitter")
            assert twitter_config["api_key"].get_secret_value() == "twitter-key"
            assert twitter_config["rate_limit"] == 300
            
            # Test LinkedIn config
            linkedin_config = settings.get_platform_config("linkedin")
            assert linkedin_config["client_id"].get_secret_value() == "linkedin-id"
            assert linkedin_config["rate_limit"] == 100
            
            # Test Instagram uses Facebook config
            instagram_config = settings.get_platform_config("instagram")
            assert instagram_config["app_id"].get_secret_value() == "facebook-id"
            assert instagram_config["rate_limit"] == 200
            
            # Test unknown platform
            unknown_config = settings.get_platform_config("unknown")
            assert unknown_config == {}
    
    def test_encrypt_decrypt_data(self, valid_env_vars):
        """Test data encryption and decryption"""
        with patch.dict(os.environ, valid_env_vars):
            settings = Settings()
            
            # Test data
            original_data = "sensitive business data for any niche"
            
            # Encrypt
            encrypted = settings.encrypt_sensitive_data(original_data)
            assert encrypted != original_data
            assert isinstance(encrypted, str)
            
            # Decrypt
            decrypted = settings.decrypt_sensitive_data(encrypted)
            assert decrypted == original_data
    
    def test_celery_url_from_redis(self, valid_env_vars):
        """Test Celery URLs are built from Redis config when not set"""
        with patch.dict(os.environ, valid_env_vars):
            settings = Settings()
            
            # Should use Redis DSN when Celery URLs not explicitly set
            assert settings.celery_broker_url is not None
            assert "redis://" in settings.celery_broker_url
            assert settings.celery_result_backend is not None
            assert "redis://" in settings.celery_result_backend
    
    def test_environment_validation(self, valid_env_vars):
        """Test environment validation"""
        # Test production environment
        env_vars = {**valid_env_vars}
        env_vars["ENVIRONMENT"] = "production"
        
        with patch.dict(os.environ, env_vars):
            settings = Settings()
            assert settings.environment == Environment.PRODUCTION


class TestGetSettings:
    """Test get_settings function"""
    
    def test_settings_singleton(self):
        """Test that get_settings returns the same instance"""
        with patch.dict(os.environ, {
            "DB_POSTGRES_USER": "user",
            "DB_POSTGRES_PASSWORD": "pass",
            "AI_OPENAI_API_KEY": "key1",
            "AI_ANTHROPIC_API_KEY": "key2",
            "SECURITY_ENCRYPTION_KEY": "RkVSTkVUX0tFWV9GT1JfVEVTVElORw==",
            "SECURITY_JWT_SECRET_KEY": "secret"
        }):
            settings1 = get_settings()
            settings2 = get_settings()
            
            assert settings1 is settings2


class TestGetEnvironmentConfig:
    """Test get_environment_config function"""
    
    def test_development_config(self):
        """Test development environment configuration"""
        with patch("backend.config.settings.settings.environment", Environment.DEVELOPMENT):
            config = get_environment_config()
            
            assert config["debug"] is True
            assert config["log_level"] == LogLevel.DEBUG
            assert config["api_docs_enabled"] is True
            assert "http://localhost:3000" in config["allowed_origins"]
            assert "http://localhost:8000" in config["allowed_origins"]
    
    def test_staging_config(self):
        """Test staging environment configuration"""
        with patch("backend.config.settings.settings.environment", Environment.STAGING):
            config = get_environment_config()
            
            assert config["debug"] is False
            assert config["log_level"] == LogLevel.INFO
            assert config["api_docs_enabled"] is True
            assert "https://staging.autoguru.com" in config["allowed_origins"]
    
    def test_production_config(self):
        """Test production environment configuration"""
        with patch("backend.config.settings.settings.environment", Environment.PRODUCTION):
            config = get_environment_config()
            
            assert config["debug"] is False
            assert config["log_level"] == LogLevel.WARNING
            assert config["api_docs_enabled"] is False
            assert "https://autoguru.com" in config["allowed_origins"]
            assert "https://app.autoguru.com" in config["allowed_origins"]
    
    def test_testing_config(self):
        """Test testing environment configuration"""
        with patch("backend.config.settings.settings.environment", Environment.TESTING):
            config = get_environment_config()
            
            assert config["debug"] is True
            assert config["log_level"] == LogLevel.DEBUG
            assert config["api_docs_enabled"] is True
            assert "*" in config["allowed_origins"]


class TestUniversalBusinessSupport:
    """Test that settings support any business niche universally"""
    
    def test_no_hardcoded_business_logic(self, valid_env_vars):
        """Test that settings don't contain hardcoded business logic"""
        with patch.dict(os.environ, valid_env_vars):
            settings = Settings()
            
            # Check that app description is generic
            assert "any business niche" in settings.app_description.lower()
            
            # Check that all features are generic and universal
            feature_flags = [
                settings.enable_auto_scheduling,
                settings.enable_content_generation,
                settings.enable_analytics,
                settings.enable_auto_engagement,
                settings.enable_competitor_analysis,
                settings.enable_trend_detection
            ]
            
            # All features should be enabled by default
            assert all(feature_flags)
            
            # AI settings should support niche detection
            assert settings.ai_service.enable_niche_detection is True
            assert settings.ai_service.enable_content_adaptation is True
            assert settings.ai_service.enable_strategy_optimization is True
    
    def test_platform_agnostic_rate_limits(self):
        """Test that rate limits work for any business type"""
        config = RateLimitConfig()
        
        # Rate limits should be conservative enough for any business
        assert config.requests_per_minute <= 60  # Safe for most APIs
        assert config.twitter_requests_per_15min <= 300  # Twitter's standard limit
        assert config.linkedin_requests_per_day <= 100  # Conservative for professional use
        
        # These limits work for fitness coaches, consultants, artists, etc.
        # They don't assume any specific posting frequency or business model