"""
Unit tests for BasePlatformPublisher

Tests the abstract base class functionality and demonstrates
how platform-specific publishers should inherit from it.
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

from backend.platforms.base_publisher import (
    BasePlatformPublisher,
    AuthResult,
    AuthStatus,
    PublishResult,
    PublishStatus,
    ScheduleResult,
    PlatformAnalytics,
    ValidationResult,
    ValidationStatus,
    RateLimitInfo,
    MediaContent,
    OptimizedMedia,
    RateLimitError
)
from backend.models.content_models import (
    Platform,
    PlatformContent,
    BusinessNicheType,
    ContentFormat
)


class MockInstagramPublisher(BasePlatformPublisher):
    """Mock Instagram publisher for testing"""
    
    def __init__(self, **kwargs):
        super().__init__(Platform.INSTAGRAM, **kwargs)
    
    async def authenticate(self, credentials: Dict[str, Any]) -> AuthResult:
        """Mock authentication"""
        if credentials.get("username") == "valid_user":
            return AuthResult(
                status=AuthStatus.AUTHENTICATED,
                access_token="mock_access_token",
                refresh_token="mock_refresh_token",
                expires_at=datetime.utcnow() + timedelta(hours=1),
                scope=["read", "write"],
                user_info={"id": "12345", "username": "valid_user"}
            )
        return AuthResult(
            status=AuthStatus.INVALID,
            error_message="Invalid credentials"
        )
    
    async def publish_content(self, content: PlatformContent) -> PublishResult:
        """Mock content publishing"""
        return PublishResult(
            status=PublishStatus.SUCCESS,
            content_id="test_content_123",
            platform_post_id="ig_post_456",
            published_at=datetime.utcnow(),
            url="https://instagram.com/p/mock_post",
            metrics={"impressions": 0, "reach": 0}
        )
    
    async def schedule_content(
        self, 
        content: PlatformContent, 
        publish_time: datetime
    ) -> ScheduleResult:
        """Mock content scheduling"""
        return ScheduleResult(
            status=PublishStatus.SCHEDULED,
            schedule_id="schedule_789",
            scheduled_for=publish_time,
            platform_schedule_id="ig_schedule_101"
        )
    
    async def get_analytics(self, content_id: str) -> PlatformAnalytics:
        """Mock analytics retrieval"""
        return PlatformAnalytics(
            content_id=content_id,
            platform=Platform.INSTAGRAM,
            impressions=1000,
            reach=800,
            engagement=150,
            likes=100,
            comments=30,
            shares=20,
            performance_score=0.85
        )
    
    async def validate_content(self, content: PlatformContent) -> ValidationResult:
        """Mock content validation"""
        errors = []
        warnings = []
        suggestions = []
        
        # Check character limit
        if len(content.content_text) > 2200:
            errors.append("Content exceeds Instagram's 2200 character limit")
        
        # Check hashtag count
        if len(content.hashtags) > 30:
            errors.append("Too many hashtags (max: 30)")
        elif len(content.hashtags) < 5:
            warnings.append("Consider using 5-10 hashtags for better reach")
        
        # Check for engagement elements
        if "?" not in content.content_text:
            suggestions.append("Consider adding a question to boost engagement")
        
        return ValidationResult(
            status=ValidationStatus.VALID if not errors else ValidationStatus.INVALID,
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    async def get_rate_limits(self) -> RateLimitInfo:
        """Mock rate limit information"""
        return RateLimitInfo(
            platform=Platform.INSTAGRAM,
            requests_remaining=180,
            requests_limit=200,
            reset_time=datetime.utcnow() + timedelta(hours=1),
            current_usage_percent=10.0,
            recommended_delay_seconds=0
        )


class TestBasePlatformPublisher:
    """Test cases for BasePlatformPublisher"""
    
    @pytest.fixture
    async def publisher(self):
        """Create mock Instagram publisher"""
        publisher = MockInstagramPublisher()
        yield publisher
        await publisher._http_client.aclose()
    
    @pytest.mark.asyncio
    async def test_authentication_success(self, publisher):
        """Test successful authentication"""
        credentials = {"username": "valid_user", "password": "valid_pass"}
        result = await publisher.authenticate(credentials)
        
        assert result.status == AuthStatus.AUTHENTICATED
        assert result.access_token == "mock_access_token"
        assert result.user_info["username"] == "valid_user"
    
    @pytest.mark.asyncio
    async def test_authentication_failure(self, publisher):
        """Test failed authentication"""
        credentials = {"username": "invalid_user", "password": "invalid_pass"}
        result = await publisher.authenticate(credentials)
        
        assert result.status == AuthStatus.INVALID
        assert result.error_message == "Invalid credentials"
    
    @pytest.mark.asyncio
    async def test_content_adaptation(self, publisher):
        """Test content adaptation for different platforms"""
        original_content = "This is a test post for AutoGuru Universal. " * 20  # Long content
        
        # Test Twitter adaptation (280 char limit)
        twitter_content = await publisher.adapt_content_for_platform(
            original_content,
            Platform.TWITTER,
            BusinessNicheType.BUSINESS_CONSULTING
        )
        assert len(twitter_content) <= 280
        
        # Test Instagram adaptation
        instagram_content = await publisher.adapt_content_for_platform(
            original_content,
            Platform.INSTAGRAM,
            BusinessNicheType.FITNESS_WELLNESS
        )
        assert len(instagram_content) <= 2200
    
    @pytest.mark.asyncio
    async def test_hashtag_optimization(self, publisher):
        """Test platform-specific hashtag optimization"""
        base_hashtags = [f"hashtag{i}" for i in range(50)]  # 50 hashtags
        
        # Instagram allows up to 30 hashtags
        instagram_tags = await publisher.generate_platform_hashtags(
            base_hashtags,
            Platform.INSTAGRAM
        )
        assert len(instagram_tags) <= 30
        assert all(tag.startswith("#") for tag in instagram_tags)
        
        # LinkedIn prefers fewer hashtags
        linkedin_tags = await publisher.generate_platform_hashtags(
            base_hashtags,
            Platform.LINKEDIN
        )
        assert len(linkedin_tags) <= 5
    
    @pytest.mark.asyncio
    async def test_optimal_posting_time(self, publisher):
        """Test optimal posting time calculation"""
        optimal_time = await publisher.calculate_optimal_posting_time(
            BusinessNicheType.BUSINESS_CONSULTING,
            Platform.LINKEDIN
        )
        
        assert isinstance(optimal_time, datetime)
        assert optimal_time > datetime.utcnow()
        
        # Check if time is in optimal hours for LinkedIn
        optimal_hours = [7, 10, 12, 17]
        assert optimal_time.hour in optimal_hours
    
    @pytest.mark.asyncio
    async def test_media_optimization(self, publisher):
        """Test media optimization for platforms"""
        media = MediaContent(
            file_path="/tmp/test_image.jpg",
            media_type="image",
            original_size_bytes=5242880,  # 5MB
            metadata={"width": 3000, "height": 2000}
        )
        
        optimized = await publisher.optimize_media_for_platform(
            media,
            Platform.INSTAGRAM
        )
        
        assert optimized.media_type == "image"
        assert optimized.dimensions == (1080, 1080)  # Instagram square format
        assert "Resized to (1080, 1080)" in optimized.optimizations_applied
    
    @pytest.mark.asyncio
    async def test_content_validation(self, publisher):
        """Test content validation"""
        # Valid content
        valid_content = PlatformContent(
            platform=Platform.INSTAGRAM,
            content_text="Check out our latest update! #automation #business",
            content_format=ContentFormat.IMAGE,
            media_requirements={"dimensions": (1080, 1080)},
            hashtags=["#automation", "#business"],
            call_to_action="Learn more in bio",
            character_count=48
        )
        
        result = await publisher.validate_content(valid_content)
        assert result.is_valid
        assert result.status == ValidationStatus.VALID
        assert len(result.warnings) == 1  # Should warn about low hashtag count
        
        # Invalid content (too long)
        invalid_content = PlatformContent(
            platform=Platform.INSTAGRAM,
            content_text="x" * 2500,  # Exceeds limit
            content_format=ContentFormat.IMAGE,
            media_requirements={},
            hashtags=[],
            call_to_action="",
            character_count=2500
        )
        
        result = await publisher.validate_content(invalid_content)
        assert not result.is_valid
        assert result.status == ValidationStatus.INVALID
        assert len(result.errors) > 0
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, publisher):
        """Test rate limiting functionality"""
        rate_info = await publisher.get_rate_limits()
        
        assert rate_info.platform == Platform.INSTAGRAM
        assert rate_info.requests_remaining < rate_info.requests_limit
        assert rate_info.current_usage_percent >= 0
        assert rate_info.reset_time > datetime.utcnow()
    
    @pytest.mark.asyncio
    async def test_oauth_token_management(self, publisher):
        """Test OAuth token management with caching"""
        business_id = "test_business_123"
        credentials = {"username": "valid_user", "password": "valid_pass"}
        
        # First authentication
        result1 = await publisher.manage_oauth_tokens(business_id, credentials)
        assert result1.status == AuthStatus.AUTHENTICATED
        
        # Second call should use cache
        result2 = await publisher.manage_oauth_tokens(business_id, credentials)
        assert result2.status == AuthStatus.AUTHENTICATED
        assert result2.access_token == result1.access_token
    
    @pytest.mark.asyncio
    async def test_publish_with_retry(self, publisher):
        """Test publishing with retry logic"""
        content = PlatformContent(
            platform=Platform.INSTAGRAM,
            content_text="Test post for retry logic",
            content_format=ContentFormat.IMAGE,
            media_requirements={},
            hashtags=["#test"],
            call_to_action="Test CTA",
            character_count=24
        )
        
        result = await publisher.publish_with_retry(content, "test_business_123")
        assert result.status == PublishStatus.SUCCESS
        assert result.platform_post_id is not None
        assert result.url is not None
    
    @pytest.mark.asyncio
    async def test_platform_specific_formatting(self, publisher):
        """Test platform-specific content formatting"""
        base_content = "This is a test post. It has multiple sentences. We want to see how it formats."
        
        # Test Twitter formatting
        twitter_formatted = publisher._format_for_twitter(base_content)
        assert len(twitter_formatted.split('\n\n')) <= 3  # Should be concise
        
        # Test LinkedIn formatting
        linkedin_formatted = publisher._format_for_linkedin(
            base_content,
            BusinessNicheType.BUSINESS_CONSULTING
        )
        assert linkedin_formatted.startswith("💡")  # Should add professional marker
        
        # Test Instagram formatting
        instagram_formatted = publisher._format_for_instagram(base_content)
        # Should have line breaks for readability
        
        # Test TikTok formatting
        tiktok_formatted = publisher._format_for_tiktok(base_content)
        assert tiktok_formatted.startswith("Wait for it...")  # Should add hook
    
    def test_platform_limits(self, publisher):
        """Test platform limit configurations"""
        limits = publisher._get_platform_limits()
        
        # Verify limits for each platform
        assert limits[Platform.TWITTER]["text_limit"] == 280
        assert limits[Platform.INSTAGRAM]["hashtag_limit"] == 30
        assert limits[Platform.LINKEDIN]["text_limit"] == 3000
        assert limits[Platform.TIKTOK]["video_duration"] == 180
    
    def test_media_specifications(self, publisher):
        """Test platform media specifications"""
        specs = publisher._get_platform_media_specs()
        
        # Verify Instagram specs
        assert specs[Platform.INSTAGRAM]["image_dimensions"] == (1080, 1080)
        assert specs[Platform.INSTAGRAM]["max_file_size"] == 8388608
        
        # Verify LinkedIn specs
        assert specs[Platform.LINKEDIN]["video_dimensions"] == (1920, 1080)
        assert specs[Platform.LINKEDIN]["max_video_duration"] == 600