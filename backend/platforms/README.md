# Platform Publishers Module

This module contains the base platform publisher class and platform-specific implementations for AutoGuru Universal's social media automation platform.

## Overview

The `BasePlatformPublisher` is an abstract base class that provides a universal interface for all social media platform integrations. It works seamlessly across any business niche without hardcoded business logic, using AI to adapt content dynamically.

## Key Features

- **Universal Design**: Works for ANY business type (education, fitness, consulting, e-commerce, etc.)
- **Secure Authentication**: OAuth token management with automatic refresh and encryption
- **Smart Rate Limiting**: Token bucket algorithm with per-business tracking
- **Content Adaptation**: AI-driven content optimization for each platform
- **Media Optimization**: Automatic media resizing and format conversion
- **Analytics Integration**: Standardized analytics across all platforms
- **Error Handling**: Comprehensive retry logic with exponential backoff

## Architecture

```
BasePlatformPublisher (Abstract)
├── InstagramPublisher
├── LinkedInPublisher
├── TwitterPublisher
├── FacebookPublisher
├── TikTokPublisher
└── YouTubePublisher
```

## Usage Example

### Creating a Platform-Specific Publisher

```python
from backend.platforms.instagram_publisher import InstagramPublisher
from backend.utils.encryption import EncryptionManager
from backend.database.connection import PostgreSQLConnectionManager

# Initialize dependencies
encryption_manager = EncryptionManager()
db_manager = PostgreSQLConnectionManager()
await db_manager.initialize()

# Create Instagram publisher
async with InstagramPublisher(
    encryption_manager=encryption_manager,
    db_manager=db_manager
) as publisher:
    # Authenticate
    credentials = {
        "username": "business_account",
        "password": "secure_password"
    }
    auth_result = await publisher.manage_oauth_tokens(
        business_id="business_123",
        credentials=credentials
    )
    
    # Create content
    content = PlatformContent(
        platform=Platform.INSTAGRAM,
        content_text="Check out our latest update!",
        content_format=ContentFormat.IMAGE,
        media_requirements={"path": "/path/to/image.jpg"},
        hashtags=["#business", "#update"],
        call_to_action="Link in bio",
        character_count=28
    )
    
    # Publish with retry logic
    result = await publisher.publish_with_retry(
        content=content,
        business_id="business_123"
    )
    
    if result.status == PublishStatus.SUCCESS:
        print(f"Published: {result.url}")
```

## Implementing a New Platform Publisher

To add support for a new platform, create a class that inherits from `BasePlatformPublisher`:

```python
from backend.platforms.base_publisher import BasePlatformPublisher

class NewPlatformPublisher(BasePlatformPublisher):
    def __init__(self, **kwargs):
        super().__init__(Platform.NEW_PLATFORM, **kwargs)
    
    async def authenticate(self, credentials: Dict[str, Any]) -> AuthResult:
        # Implement platform-specific authentication
        pass
    
    async def publish_content(self, content: PlatformContent) -> PublishResult:
        # Implement content publishing
        pass
    
    async def schedule_content(
        self, 
        content: PlatformContent, 
        publish_time: datetime
    ) -> ScheduleResult:
        # Implement content scheduling
        pass
    
    async def get_analytics(self, content_id: str) -> PlatformAnalytics:
        # Implement analytics retrieval
        pass
    
    async def validate_content(self, content: PlatformContent) -> ValidationResult:
        # Implement platform-specific validation
        pass
    
    async def get_rate_limits(self) -> RateLimitInfo:
        # Implement rate limit checking
        pass
```

## Key Classes and Data Models

### AuthResult
Contains authentication status, tokens, and user information.

### PublishResult
Contains publishing status, content IDs, URLs, and initial metrics.

### ScheduleResult
Contains scheduling status and platform-specific schedule information.

### PlatformAnalytics
Standardized analytics data including impressions, reach, engagement, and demographics.

### ValidationResult
Content validation results with errors, warnings, and suggestions.

### RateLimitInfo
Current rate limit status and recommendations.

## Shared Functionality

The base class provides these methods that work universally:

### Content Adaptation
```python
adapted_content = await publisher.adapt_content_for_platform(
    content="Your original content here",
    platform=Platform.TWITTER,
    business_niche=BusinessNicheType.FITNESS_WELLNESS
)
```

### Media Optimization
```python
optimized = await publisher.optimize_media_for_platform(
    media=MediaContent(
        file_path="/path/to/media.jpg",
        media_type="image",
        original_size_bytes=5242880
    ),
    platform=Platform.INSTAGRAM
)
```

### Hashtag Generation
```python
hashtags = await publisher.generate_platform_hashtags(
    base_hashtags=["fitness", "health", "wellness"],
    platform=Platform.INSTAGRAM
)
```

### Optimal Posting Time
```python
best_time = await publisher.calculate_optimal_posting_time(
    business_niche=BusinessNicheType.BUSINESS_CONSULTING,
    platform=Platform.LINKEDIN
)
```

## Platform Limits

Each platform has specific limits that are automatically enforced:

| Platform | Text Limit | Hashtags | Media Files | Video Duration |
|----------|------------|----------|-------------|----------------|
| Twitter | 280 chars | 5 | 4 | 140 seconds |
| Instagram | 2,200 chars | 30 | 10 | 60 seconds |
| LinkedIn | 3,000 chars | 5 | 9 | 600 seconds |
| Facebook | 63,206 chars | 10 | 10 | 240 seconds |
| TikTok | 2,200 chars | 10 | 1 | 180 seconds |

## Error Handling

The module includes custom exceptions for different error scenarios:

- `AuthenticationError`: OAuth or credential issues
- `RateLimitError`: API rate limit exceeded
- `ContentValidationError`: Content doesn't meet platform requirements
- `PublishingError`: General publishing failures

All methods include retry logic with exponential backoff for transient failures.

## Security

- All credentials are encrypted using AES-256 encryption
- OAuth tokens are stored securely and refreshed automatically
- API keys are never logged or exposed
- Each business has isolated rate limit tracking

## Testing

Run the test suite:

```bash
pytest tests/unit/test_base_publisher.py -v
```

## Configuration

Platform-specific settings are managed in `backend/config/settings.py`:

```python
settings = get_settings()

# Get platform configuration
twitter_config = settings.get_platform_config("twitter")
```

## Best Practices

1. **Always use context managers** for proper cleanup:
   ```python
   async with PublisherClass() as publisher:
       # Use publisher
   ```

2. **Handle rate limits gracefully** by checking before publishing:
   ```python
   rate_info = await publisher.get_rate_limits()
   if rate_info.requests_remaining < 10:
       # Wait or schedule for later
   ```

3. **Validate content before publishing**:
   ```python
   validation = await publisher.validate_content(content)
   if not validation.is_valid:
       # Fix errors before publishing
   ```

4. **Use retry logic** for reliability:
   ```python
   result = await publisher.publish_with_retry(content, business_id)
   ```

5. **Monitor analytics** for optimization:
   ```python
   analytics = await publisher.get_analytics(content_id)
   if analytics.engagement < threshold:
       # Adjust strategy
   ```

## Future Enhancements

- Real-time webhook support for instant analytics
- Advanced AI content generation per platform
- Cross-platform content synchronization
- A/B testing framework
- Automated engagement responses
- Competitor analysis integration