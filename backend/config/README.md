# AutoGuru Universal - Configuration Management

This directory contains the configuration management system for AutoGuru Universal, a comprehensive social media automation platform that works universally for any business niche.

## Overview

The settings management system uses Pydantic for validation and supports multiple environments (development, staging, production, testing). All sensitive data is encrypted using Fernet encryption.

## Files

- `settings.py` - Main settings module with all configuration classes
- `__init__.py` - Package initialization with exports

## Configuration Classes

### Main Settings
- `Settings` - Main configuration class that combines all sub-configurations
- `get_settings()` - Function to get cached settings instance

### Sub-configurations
- `DatabaseConfig` - PostgreSQL and Redis settings
- `AIServiceConfig` - OpenAI and Anthropic API configurations
- `SocialMediaConfig` - Platform-specific API credentials (Twitter, LinkedIn, Facebook, etc.)
- `SecurityConfig` - Encryption, JWT, and CORS settings
- `LoggingConfig` - Logging levels and file configuration
- `RateLimitConfig` - API rate limiting for platforms and services

## Environment Variables

All settings can be configured via environment variables with specific prefixes:
- General: No prefix (e.g., `ENVIRONMENT`, `DEBUG`)
- Database: `DB_` prefix
- AI Services: `AI_` prefix
- Social Media: `SOCIAL_` prefix
- Security: `SECURITY_` prefix
- Logging: `LOG_` prefix
- Rate Limits: `RATE_LIMIT_` prefix

## Setup Instructions

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Generate an encryption key:**
   ```bash
   python backend/utils/generate_encryption_key.py
   ```

3. **Fill in your configuration:**
   - Add your database credentials
   - Add AI service API keys (OpenAI and Anthropic)
   - Add the generated encryption key
   - Add social media platform credentials as needed

4. **Required Environment Variables:**
   - `DB_POSTGRES_USER`
   - `DB_POSTGRES_PASSWORD`
   - `AI_OPENAI_API_KEY`
   - `AI_ANTHROPIC_API_KEY`
   - `SECURITY_ENCRYPTION_KEY`
   - `SECURITY_JWT_SECRET_KEY`

## Usage Example

```python
from backend.config import settings

# Access configuration
print(settings.app_name)  # "AutoGuru Universal"
print(settings.environment)  # Environment.DEVELOPMENT

# Get platform-specific config
twitter_config = settings.get_platform_config("twitter")

# Encrypt sensitive data
encrypted = settings.encrypt_sensitive_data("sensitive_info")
decrypted = settings.decrypt_sensitive_data(encrypted)

# Access sub-configurations
db_url = settings.database.postgres_dsn
openai_key = settings.ai_service.openai_api_key.get_secret_value()
```

## Environment-Specific Settings

The system automatically adjusts settings based on the environment:

- **Development**: Debug enabled, verbose logging, API docs enabled
- **Staging**: Production-like with API docs
- **Production**: Optimized settings, minimal logging, no API docs
- **Testing**: Debug enabled for test runs

## Universal Design

This configuration system is designed to work for ANY business niche:
- No hardcoded business logic
- Flexible rate limits that work for all use cases
- AI-driven strategy detection and adaptation
- Platform-agnostic design

## Security Notes

- Never commit `.env` files to version control
- Keep encryption keys secure and backed up
- Use strong, unique passwords for all services
- Rotate API keys regularly
- All sensitive data is encrypted at rest