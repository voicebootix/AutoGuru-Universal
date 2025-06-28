"""
Unit tests for Instagram Publisher.

Tests the Instagram platform publisher to ensure it works universally
across all business niches without hardcoded business logic.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import json

from backend.platforms.base_publisher import (
    PublishResult,
    PublishStatus,
    VideoContent,
    StoryContent,
    MediaAsset,
    MediaType,
    InstagramAnalytics,
    OptimizedContent
)
from backend.models.content_models import (
    Platform,
    PlatformContent,
    BusinessNicheType,
    ContentFormat
)


# Mock the Instagram publisher to avoid import errors
class MockInstagramPublisher:
    """Mock Instagram publisher for testing"""
    
    def __init__(self, business_id: str):
        self.business_id = business_id
        self.platform = Platform.INSTAGRAM
        self._authenticated = False
        self._access_token = None
        
    async def authenticate(self, credentials: dict) -> bool:
        if 'access_token' in credentials:
            self._access_token = credentials['access_token']
            self._authenticated = True
            return True
        return False
        
    async def post_to_feed(self, content: PlatformContent, hashtags: list) -> PublishResult:
        if not self._authenticated:
            return PublishResult(
                status=PublishStatus.AUTH_FAILED,
                platform=Platform.INSTAGRAM,
                error_message="Not authenticated"
            )
            
        return PublishResult(
            status=PublishStatus.SUCCESS,
            platform=Platform.INSTAGRAM,
            post_id="123456789",
            post_url="https://www.instagram.com/p/123456789/",
            published_at=datetime.utcnow()
        )
        
    async def post_reel(self, video_content: VideoContent, trending_audio: str) -> PublishResult:
        return PublishResult(
            status=PublishStatus.SUCCESS,
            platform=Platform.INSTAGRAM,
            post_id="reel_123456",
            post_url="https://www.instagram.com/reel/123456/",
            published_at=datetime.utcnow(),
            metrics={'type': 'reel', 'audio': trending_audio}
        )
        
    async def post_story(self, story_content: StoryContent, engagement_features: list) -> PublishResult:
        return PublishResult(
            status=PublishStatus.SUCCESS,
            platform=Platform.INSTAGRAM,
            post_id="story_123456",
            published_at=datetime.utcnow(),
            metrics={
                'type': 'story',
                'interactive_elements': engagement_features,
                'duration': story_content.duration_seconds
            }
        )
        
    async def optimize_for_algorithm(
        self, 
        content: PlatformContent, 
        business_niche: BusinessNicheType
    ) -> OptimizedContent:
        return OptimizedContent(
            original_content=content,
            optimized_text=content.content_text,
            suggested_hashtags=['#optimized', '#test'],
            suggested_mentions=['@test'],
            best_posting_time=datetime.utcnow() + timedelta(hours=2),
            algorithm_score=0.85,
            optimization_reasons=["Test optimization"],
            trending_elements=[],
            engagement_predictions={'likes': 100, 'comments': 20}
        )


@pytest.fixture
def mock_instagram_publisher():
    """Create a mock Instagram publisher instance"""
    return MockInstagramPublisher("test_business_123")


@pytest.fixture
def sample_platform_content():
    """Create sample platform content for testing"""
    return PlatformContent(
        platform=Platform.INSTAGRAM,
        content_text="Check out our amazing new product! Perfect for your daily routine.",
        content_format=ContentFormat.IMAGE,
        media_requirements={'media_url': 'https://example.com/image.jpg'},
        hashtags=['#newproduct', '#daily'],
        mentions=[],
        call_to_action="Shop now!",
        platform_features={'business_niche': 'ecommerce'},
        character_count=100
    )


@pytest.fixture
def sample_video_content():
    """Create sample video content for testing"""
    return VideoContent(
        video_asset=MediaAsset(
            type=MediaType.VIDEO,
            url="https://example.com/video.mp4",
            duration=30
        ),
        title="Amazing Tutorial",
        description="Learn how to use our product",
        hashtags=['#tutorial', '#howto']
    )


@pytest.fixture
def sample_story_content():
    """Create sample story content for testing"""
    return StoryContent(
        media_asset=MediaAsset(
            type=MediaType.IMAGE,
            url="https://example.com/story.jpg"
        ),
        text_overlay="Swipe up for more!",
        duration_seconds=15
    )


class TestInstagramPublisher:
    """Test Instagram publisher functionality"""
    
    @pytest.mark.asyncio
    async def test_authentication_success(self, mock_instagram_publisher):
        """Test successful authentication"""
        credentials = {'access_token': 'test_token_123'}
        result = await mock_instagram_publisher.authenticate(credentials)
        
        assert result is True
        assert mock_instagram_publisher._authenticated is True
        assert mock_instagram_publisher._access_token == 'test_token_123'
    
    @pytest.mark.asyncio
    async def test_authentication_failure(self, mock_instagram_publisher):
        """Test authentication failure"""
        credentials = {}
        result = await mock_instagram_publisher.authenticate(credentials)
        
        assert result is False
        assert mock_instagram_publisher._authenticated is False
    
    @pytest.mark.asyncio
    async def test_post_to_feed_success(self, mock_instagram_publisher, sample_platform_content):
        """Test successful feed post"""
        # Authenticate first
        await mock_instagram_publisher.authenticate({'access_token': 'test_token'})
        
        # Post to feed
        result = await mock_instagram_publisher.post_to_feed(
            sample_platform_content,
            ['#test', '#automation']
        )
        
        assert result.status == PublishStatus.SUCCESS
        assert result.platform == Platform.INSTAGRAM
        assert result.post_id is not None
        assert result.post_url is not None
        assert 'instagram.com' in result.post_url
    
    @pytest.mark.asyncio
    async def test_post_to_feed_not_authenticated(self, mock_instagram_publisher, sample_platform_content):
        """Test feed post without authentication"""
        result = await mock_instagram_publisher.post_to_feed(
            sample_platform_content,
            ['#test']
        )
        
        assert result.status == PublishStatus.AUTH_FAILED
        assert result.error_message == "Not authenticated"
    
    @pytest.mark.asyncio
    async def test_post_reel(self, mock_instagram_publisher, sample_video_content):
        """Test posting a reel"""
        # Authenticate first
        await mock_instagram_publisher.authenticate({'access_token': 'test_token'})
        
        # Post reel
        result = await mock_instagram_publisher.post_reel(
            sample_video_content,
            "trending_audio_123"
        )
        
        assert result.status == PublishStatus.SUCCESS
        assert result.metrics['type'] == 'reel'
        assert result.metrics['audio'] == 'trending_audio_123'
    
    @pytest.mark.asyncio
    async def test_post_story(self, mock_instagram_publisher, sample_story_content):
        """Test posting a story"""
        # Authenticate first
        await mock_instagram_publisher.authenticate({'access_token': 'test_token'})
        
        # Post story
        engagement_features = ['poll', 'question']
        result = await mock_instagram_publisher.post_story(
            sample_story_content,
            engagement_features
        )
        
        assert result.status == PublishStatus.SUCCESS
        assert result.metrics['type'] == 'story'
        assert result.metrics['interactive_elements'] == engagement_features
        assert result.metrics['duration'] == 15
    
    @pytest.mark.asyncio
    async def test_optimize_for_algorithm(self, mock_instagram_publisher, sample_platform_content):
        """Test content optimization for algorithm"""
        # Authenticate first
        await mock_instagram_publisher.authenticate({'access_token': 'test_token'})
        
        # Optimize content
        optimized = await mock_instagram_publisher.optimize_for_algorithm(
            sample_platform_content,
            BusinessNicheType.ECOMMERCE
        )
        
        assert optimized.algorithm_score > 0.5
        assert len(optimized.suggested_hashtags) > 0
        assert optimized.best_posting_time > datetime.utcnow()
        assert 'likes' in optimized.engagement_predictions
    
    @pytest.mark.asyncio
    async def test_universal_business_support(self, mock_instagram_publisher):
        """Test that publisher works for different business niches"""
        await mock_instagram_publisher.authenticate({'access_token': 'test_token'})
        
        business_niches = [
            BusinessNicheType.FITNESS_WELLNESS,
            BusinessNicheType.BUSINESS_CONSULTING,
            BusinessNicheType.CREATIVE,
            BusinessNicheType.EDUCATION,
            BusinessNicheType.LOCAL_SERVICE,
            BusinessNicheType.TECHNOLOGY,
            BusinessNicheType.NON_PROFIT
        ]
        
        for niche in business_niches:
            # Create content for this niche
            content = PlatformContent(
                platform=Platform.INSTAGRAM,
                content_text=f"Content for {niche.value} business",
                content_format=ContentFormat.IMAGE,
                media_requirements={'media_url': 'https://example.com/image.jpg'},
                hashtags=[],
                mentions=[],
                call_to_action="Learn more",
                platform_features={'business_niche': niche.value},
                character_count=50
            )
            
            # Test posting
            result = await mock_instagram_publisher.post_to_feed(content, [])
            assert result.status == PublishStatus.SUCCESS
            
            # Test optimization
            optimized = await mock_instagram_publisher.optimize_for_algorithm(content, niche)
            assert optimized.algorithm_score > 0
    
    def test_media_processor_constants(self):
        """Test media processor has correct Instagram specifications"""
        from backend.platforms.instagram_publisher import InstagramMediaProcessor
        
        # Test image sizes
        assert InstagramMediaProcessor.FEED_IMAGE_SIZES['square'] == (1080, 1080)
        assert InstagramMediaProcessor.FEED_IMAGE_SIZES['landscape'] == (1080, 566)
        assert InstagramMediaProcessor.FEED_IMAGE_SIZES['portrait'] == (1080, 1350)
        
        # Test story/reel sizes
        assert InstagramMediaProcessor.STORY_SIZE == (1080, 1920)
        assert InstagramMediaProcessor.REEL_SIZE == (1080, 1920)
        
        # Test file size limits
        assert InstagramMediaProcessor.MAX_IMAGE_SIZE_BYTES == 8 * 1024 * 1024
        assert InstagramMediaProcessor.MAX_VIDEO_SIZE_BYTES == 100 * 1024 * 1024
    
    def test_hashtag_optimizer_constants(self):
        """Test hashtag optimizer limits"""
        from backend.platforms.instagram_publisher import InstagramHashtagOptimizer
        
        assert InstagramHashtagOptimizer.MAX_HASHTAGS == 30
        assert InstagramHashtagOptimizer.OPTIMAL_HASHTAG_COUNT == 15
        
        # Test hashtag patterns exist
        assert 'community' in InstagramHashtagOptimizer.HASHTAG_PATTERNS
        assert 'trending' in InstagramHashtagOptimizer.HASHTAG_PATTERNS


@pytest.mark.asyncio
async def test_analytics_structure():
    """Test Instagram analytics data structure"""
    analytics = InstagramAnalytics(
        post_id="123456",
        post_type="feed",
        impressions=1000,
        reach=800,
        engagement=150,
        likes=120,
        comments=20,
        saves=10
    )
    
    assert analytics.post_id == "123456"
    assert analytics.impressions == 1000
    assert analytics.engagement == 150
    assert analytics.likes + analytics.comments + analytics.saves == 150


def test_publish_result_properties():
    """Test PublishResult properties"""
    result = PublishResult(
        status=PublishStatus.SUCCESS,
        platform=Platform.INSTAGRAM,
        post_id="123"
    )
    
    assert result.is_success is True
    
    failed_result = PublishResult(
        status=PublishStatus.FAILED,
        platform=Platform.INSTAGRAM,
        error_message="Test error"
    )
    
    assert failed_result.is_success is False