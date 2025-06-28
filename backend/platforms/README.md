# Platform Publishers

This module contains platform-specific publishers for AutoGuru Universal's social media automation.

## Overview

The platform publishers provide a unified interface for posting content to various social media platforms while adapting to ANY business niche automatically. Each publisher inherits from `BasePlatformPublisher` and implements platform-specific functionality.

## Instagram Publisher

The Instagram publisher (`instagram_publisher.py`) provides complete automation for Instagram, supporting:

### Features

- **Content Types**:
  - Feed posts (images, videos, carousels)
  - Stories with interactive elements
  - Reels with trending audio
  - IGTV (through video posts)

- **OAuth Integration**: Secure token management with encryption
- **Universal Business Support**: Works for fitness, consulting, creative, education, e-commerce, local services, technology, and non-profits
- **Algorithm Optimization**: AI-driven content optimization
- **Rate Limiting**: Compliant with Instagram API limits
- **Analytics**: Comprehensive post and account insights

### Usage Example

```python
from backend.platforms.instagram_publisher import InstagramPublisher
from backend.models.content_models import PlatformContent, BusinessNicheType

# Initialize publisher
publisher = InstagramPublisher(business_id="business_123")

# Authenticate with OAuth token
credentials = {
    'access_token': 'YOUR_INSTAGRAM_ACCESS_TOKEN'
}
await publisher.authenticate(credentials)

# Create content (works for ANY business type)
content = PlatformContent(
    platform=Platform.INSTAGRAM,
    content_text="Your amazing content here",
    content_format=ContentFormat.IMAGE,
    media_requirements={'media_url': 'https://example.com/image.jpg'},
    hashtags=['#yourbusiness'],
    call_to_action="Learn more!"
)

# Post to feed with optimized hashtags
result = await publisher.post_to_feed(content, ['#additional', '#hashtags'])

# Post a reel
video_content = VideoContent(
    video_asset=MediaAsset(
        type=MediaType.VIDEO,
        url="https://example.com/video.mp4"
    ),
    title="Amazing Reel",
    description="Check this out!",
    hashtags=['#reels', '#trending']
)
reel_result = await publisher.post_reel(video_content, "trending_audio_id")

# Post a story with engagement features
story_content = StoryContent(
    media_asset=MediaAsset(
        type=MediaType.IMAGE,
        url="https://example.com/story.jpg"
    ),
    text_overlay="Swipe up!",
    interactive_elements=[
        {'type': 'poll', 'question': 'Do you like this?', 'options': ['Yes', 'No']}
    ]
)
story_result = await publisher.post_story(story_content, ['poll'])

# Optimize content for algorithm
optimized = await publisher.optimize_for_algorithm(
    content,
    BusinessNicheType.FITNESS_WELLNESS  # Works for ANY niche
)

# Get analytics
analytics = await publisher.get_analytics(
    post_id="123456789",
    include_demographics=True
)
```

### Authentication

Instagram uses OAuth 2.0. You need:
1. Instagram Business Account
2. Facebook App with Instagram Basic Display or Instagram Graph API
3. Access Token (long-lived tokens last 60 days)

The publisher automatically:
- Encrypts and stores tokens securely
- Checks token expiration
- Handles rate limiting

### Content Requirements

- **Images**: JPG/PNG, max 8MB
- **Videos**: MP4/MOV, max 100MB (4GB for IGTV)
- **Dimensions**:
  - Feed Square: 1080x1080
  - Feed Landscape: 1080x566
  - Feed Portrait: 1080x1350
  - Stories/Reels: 1080x1920
- **Caption**: Max 2200 characters
- **Hashtags**: Max 30 per post

### Universal Business Support

The Instagram publisher adapts automatically to different business types:

```python
# Fitness Business
fitness_content = PlatformContent(
    content_text="30-minute HIIT workout to boost your metabolism! 💪",
    platform_features={'business_niche': 'fitness_wellness'}
)

# Business Consulting
consulting_content = PlatformContent(
    content_text="5 strategies to scale your business in 2024 📈",
    platform_features={'business_niche': 'business_consulting'}
)

# Creative/Artist
creative_content = PlatformContent(
    content_text="New artwork inspired by urban landscapes 🎨",
    platform_features={'business_niche': 'creative'}
)
```

The publisher will:
- Optimize hashtags for each niche
- Suggest relevant mentions
- Determine best posting times
- Adapt content format recommendations

### Error Handling

All methods return structured results with error information:

```python
result = await publisher.post_to_feed(content, hashtags)

if not result.is_success:
    print(f"Error: {result.error_message}")
    print(f"Error code: {result.error_code}")
    
    if result.status == PublishStatus.RATE_LIMITED:
        print(f"Rate limit info: {result.rate_limit_info}")
```

### Rate Limiting

The publisher implements intelligent rate limiting:
- 25 API calls per minute (burst)
- 200 API calls per hour
- Automatic waiting when limits are reached
- No manual intervention needed

### Future Platforms

Coming soon:
- `twitter_publisher.py` - Twitter/X automation
- `linkedin_publisher.py` - LinkedIn automation  
- `facebook_publisher.py` - Facebook automation
- `tiktok_publisher.py` - TikTok automation
- `youtube_publisher.py` - YouTube automation

Each will follow the same universal pattern, working for ANY business type automatically.

## Development

When adding new platforms:
1. Inherit from `BasePlatformPublisher`
2. Implement all abstract methods
3. Support all business niches universally
4. Include comprehensive error handling
5. Add rate limiting
6. Encrypt all credentials
7. Write tests for all functionality

Remember: NO hardcoded business logic! Use AI to adapt to different niches.