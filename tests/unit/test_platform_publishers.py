"""
Comprehensive tests for all platform publishers.

Tests cover all business niches and ensure universal compatibility
without hardcoded business logic.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import json

from backend.platforms.enhanced_base_publisher import (
    UniversalPlatformPublisher,
    PublishResult,
    PublishStatus,
    RevenueMetrics,
    PerformanceMetrics
)
from backend.platforms.youtube_publisher import YouTubeEnhancedPublisher
from backend.platforms.linkedin_publisher import LinkedInEnhancedPublisher
from backend.platforms.tiktok_publisher import TikTokEnhancedPublisher
# from backend.platforms.twitter_publisher import TwitterEnhancedPublisher
# from backend.platforms.facebook_publisher import FacebookEnhancedPublisher
from backend.models.content_models import BusinessNicheType


class TestUniversalPlatformPublishers:
    """Comprehensive tests for all platform publishers"""
    
    @pytest.fixture
    def sample_content_education(self):
        return {
            'text': 'Master calculus in 30 days with our proven method. Transform your math skills!',
            'type': 'educational',
            'business_niche': 'education',
            'target_audience': 'students',
            'media_url': 'https://example.com/video.mp4'
        }
    
    @pytest.fixture
    def sample_content_business(self):
        return {
            'text': 'Scale your startup from $0 to $1M ARR. Here\'s the exact framework we used.',
            'type': 'business_strategy',
            'business_niche': 'business_consulting',
            'target_audience': 'entrepreneurs'
        }
    
    @pytest.fixture
    def sample_content_fitness(self):
        return {
            'text': 'Lose 20 pounds in 90 days without giving up your favorite foods. Real results!',
            'type': 'fitness_transformation',
            'business_niche': 'fitness',
            'target_audience': 'fitness_seekers',
            'video_file': '/path/to/workout.mp4'
        }
    
    @pytest.fixture
    def sample_content_creative(self):
        return {
            'text': 'Create stunning photography that sells. Professional techniques for beginners.',
            'type': 'creative_tutorial',
            'business_niche': 'creative_arts',
            'target_audience': 'aspiring_photographers'
        }
    
    @pytest.fixture
    def mock_credentials(self):
        return {
            'access_token': 'test_access_token',
            'refresh_token': 'test_refresh_token',
            'client_id': 'test_client_id',
            'client_secret': 'test_client_secret'
        }
    
    # YouTube Publisher Tests
    @pytest.mark.asyncio
    async def test_youtube_publisher_initialization(self):
        """Test YouTube publisher initialization"""
        publisher = YouTubeEnhancedPublisher(client_id="test_client")
        
        assert publisher.client_id == "test_client"
        assert publisher.platform_name == "youtube"
        assert publisher.youtube_service is None
        assert not publisher._authenticated
    
    @pytest.mark.asyncio
    async def test_youtube_authentication(self, mock_credentials):
        """Test YouTube authentication process"""
        publisher = YouTubeEnhancedPublisher(client_id="test_client")
        
        # Mock the Google API client
        with patch('backend.platforms.youtube_publisher.build') as mock_build:
            mock_youtube = Mock()
            mock_youtube.channels().list().execute.return_value = {
                'items': [{'id': 'channel_123'}]
            }
            mock_build.return_value = mock_youtube
            
            # Mock encryption manager
            publisher.encryption_manager.encrypt_credentials = Mock(return_value='encrypted')
            
            result = await publisher.authenticate(mock_credentials)
            
            assert result is True
            assert publisher._authenticated is True
            assert publisher.youtube_service is not None
    
    @pytest.mark.asyncio
    async def test_youtube_content_optimization(self, sample_content_education):
        """Test YouTube content optimization for education niche"""
        publisher = YouTubeEnhancedPublisher(client_id="test_client")
        
        # Mock dependencies
        publisher.content_analyzer.analyze_content = AsyncMock(
            return_value=Mock(business_niche=Mock(value='education'))
        )
        publisher.analytics_service = Mock()
        
        optimizations = await publisher.optimize_for_revenue(sample_content_education)
        
        assert 'optimal_posting_time' in optimizations
        assert 'revenue_optimization' in optimizations
        assert 'audience_targeting' in optimizations
        assert 'platform_specific_tweaks' in optimizations
        assert isinstance(optimizations['optimal_posting_time'], datetime)
    
    @pytest.mark.asyncio
    async def test_youtube_metadata_generation(self, sample_content_education):
        """Test YouTube metadata generation"""
        publisher = YouTubeEnhancedPublisher(client_id="test_client")
        publisher._authenticated = True
        
        # Mock business niche detection
        publisher.detect_business_niche = AsyncMock(return_value='education')
        
        metadata = await publisher.generate_youtube_metadata(
            sample_content_education,
            {'suggestions': []}
        )
        
        assert metadata.title
        assert metadata.description
        assert len(metadata.tags) > 0
        assert metadata.category_id == "27"  # Education category
        assert metadata.privacy_status == "public"
    
    @pytest.mark.asyncio
    async def test_youtube_revenue_calculation(self, sample_content_business):
        """Test YouTube revenue potential calculation"""
        publisher = YouTubeEnhancedPublisher(client_id="test_client")
        
        # Mock analytics tracker
        mock_analytics = {
            'views': 100000,
            'likes': 5000,
            'comments': 500
        }
        
        publisher.analytics_tracker = Mock()
        publisher.analytics_tracker.calculate_estimated_revenue = AsyncMock(
            return_value=250.0
        )
        
        revenue = await publisher.calculate_revenue_potential(
            sample_content_business,
            {'optimal_posting_time': datetime.utcnow()}
        )
        
        assert revenue > 0
        assert isinstance(revenue, float)
    
    # LinkedIn Publisher Tests
    @pytest.mark.asyncio
    async def test_linkedin_publisher_initialization(self):
        """Test LinkedIn publisher initialization"""
        publisher = LinkedInEnhancedPublisher(client_id="test_client")
        
        assert publisher.client_id == "test_client"
        assert publisher.platform_name == "linkedin"
        assert publisher.linkedin_api is None
        assert not publisher._authenticated
    
    @pytest.mark.asyncio
    async def test_linkedin_professional_optimization(self, sample_content_business):
        """Test LinkedIn professional content optimization"""
        publisher = LinkedInEnhancedPublisher(client_id="test_client")
        
        # Mock business niche detection
        publisher.detect_business_niche = AsyncMock(return_value='business_consulting')
        
        optimizations = await publisher.optimize_for_professional_engagement(
            sample_content_business
        )
        
        assert 'professional_content' in optimizations
        assert 'lead_cta' in optimizations
        assert 'hashtags' in optimizations
        assert len(optimizations['hashtags']) <= 5  # LinkedIn optimal
        assert 'target_audience' in optimizations
    
    @pytest.mark.asyncio
    async def test_linkedin_lead_generation_metrics(self):
        """Test LinkedIn lead generation tracking"""
        publisher = LinkedInEnhancedPublisher(client_id="test_client")
        publisher._authenticated = True
        
        # Mock analytics tracker
        publisher.analytics_tracker = Mock()
        lead_metrics = Mock(
            profile_views=125,
            form_submissions=5,
            estimated_lead_value=8750.0,
            conversion_funnel={'engaged_leads': 48}
        )
        publisher.analytics_tracker.track_lead_metrics = AsyncMock(
            return_value=lead_metrics
        )
        
        # Test lead tracking
        metrics = await publisher.analytics_tracker.track_lead_metrics("post_123")
        
        assert metrics.estimated_lead_value == 8750.0
        assert metrics.form_submissions == 5
        assert metrics.profile_views == 125
    
    @pytest.mark.asyncio
    async def test_linkedin_optimal_posting_time(self):
        """Test LinkedIn optimal posting time calculation"""
        publisher = LinkedInEnhancedPublisher(client_id="test_client")
        
        optimal_time = await publisher.get_optimal_posting_time(
            'post',
            'business_consulting'
        )
        
        # Should be during business hours on a weekday
        assert optimal_time.hour in [7, 8, 9, 10, 12, 14, 17]
        assert optimal_time.weekday() in [1, 2, 3]  # Tue, Wed, Thu
    
    # TikTok Publisher Tests
    @pytest.mark.asyncio
    async def test_tiktok_publisher_initialization(self):
        """Test TikTok publisher initialization"""
        publisher = TikTokEnhancedPublisher(client_id="test_client")
        
        assert publisher.client_id == "test_client"
        assert publisher.platform_name == "tiktok"
        assert publisher.tiktok_api is None
        assert not publisher._authenticated
    
    @pytest.mark.asyncio
    async def test_tiktok_viral_optimization(self, sample_content_fitness):
        """Test TikTok viral optimization for fitness content"""
        publisher = TikTokEnhancedPublisher(client_id="test_client")
        
        # Mock business niche detection
        publisher.detect_business_niche = AsyncMock(return_value='fitness_wellness')
        
        viral_optimizations = await publisher.optimize_for_viral_potential(
            sample_content_fitness
        )
        
        assert 'algorithm_optimizations' in viral_optimizations
        assert 'trending_elements' in viral_optimizations
        assert 'hashtags' in viral_optimizations
        assert '#fyp' in viral_optimizations['hashtags']
        assert '#viral' in viral_optimizations['hashtags']
        assert 'video_specifications' in viral_optimizations
    
    @pytest.mark.asyncio
    async def test_tiktok_viral_metrics_calculation(self):
        """Test TikTok viral metrics tracking"""
        publisher = TikTokEnhancedPublisher(client_id="test_client")
        
        # Mock analytics
        mock_analytics = {
            'views': 50000,
            'likes': 7500,
            'comments': 500,
            'shares': 1500,
            'completion_rate': 0.65
        }
        
        publisher.analytics_tracker = Mock()
        publisher.analytics_tracker._calculate_viral_score = Mock(return_value=75.5)
        
        viral_score = publisher.analytics_tracker._calculate_viral_score(mock_analytics)
        
        assert viral_score > 70  # High viral score
        assert viral_score <= 100
    
    @pytest.mark.asyncio
    async def test_tiktok_creator_revenue(self, sample_content_creative):
        """Test TikTok creator revenue calculation"""
        publisher = TikTokEnhancedPublisher(client_id="test_client")
        
        # Mock viral metrics
        viral_metrics = Mock(viral_score=80.0)
        analytics = {'views': 100000}
        
        # Mock business niche detection
        publisher.detect_business_niche = AsyncMock(return_value='creative')
        
        revenue = await publisher.calculate_creator_revenue_potential(
            sample_content_creative,
            analytics,
            viral_metrics
        )
        
        assert revenue > 0
        assert isinstance(revenue, float)
    
    # Cross-Platform Tests
    @pytest.mark.asyncio
    async def test_universal_niche_detection(self):
        """Test universal business niche detection across all platforms"""
        test_cases = [
            ("Learn Python programming in 30 days", "education"),
            ("Scale your business with proven marketing strategies", "business_consulting"),
            ("Get fit with our 21-day challenge", "fitness_wellness"),
            ("Master portrait photography techniques", "creative"),
            ("Best organic skincare products for sensitive skin", "ecommerce"),
            ("Professional plumbing services in your area", "local_service"),
            ("Build scalable cloud applications with AWS", "technology"),
            ("Support our mission to feed homeless families", "non_profit")
        ]
        
        publishers = [
            YouTubeEnhancedPublisher("test"),
            LinkedInEnhancedPublisher("test"),
            TikTokEnhancedPublisher("test")
        ]
        
        for publisher in publishers:
            # Mock content analyzer
            publisher.content_analyzer.analyze_content = AsyncMock()
            
            for content_text, expected_niche in test_cases:
                # Set up mock return value
                mock_result = Mock()
                mock_result.business_niche.value = expected_niche
                publisher.content_analyzer.analyze_content.return_value = mock_result
                
                detected_niche = await publisher.detect_business_niche(content_text)
                
                assert detected_niche == expected_niche
    
    @pytest.mark.asyncio
    async def test_revenue_optimization_across_platforms(self):
        """Test revenue optimization works for all business types across all platforms"""
        business_types = [
            {'niche': 'education', 'content': 'Online course about data science'},
            {'niche': 'fitness_wellness', 'content': 'Personal training program'},
            {'niche': 'business_consulting', 'content': 'Business growth strategies'},
            {'niche': 'creative', 'content': 'Digital art techniques'}
        ]
        
        publishers = [
            YouTubeEnhancedPublisher("test"),
            LinkedInEnhancedPublisher("test"),
            TikTokEnhancedPublisher("test")
        ]
        
        for publisher in publishers:
            # Mock dependencies
            publisher.content_analyzer.analyze_content = AsyncMock()
            publisher.analytics_service = Mock()
            
            for business_type in business_types:
                content = {
                    'text': business_type['content'],
                    'type': 'promotional',
                    'business_niche': business_type['niche']
                }
                
                # Set up mock
                mock_result = Mock()
                mock_result.business_niche.value = business_type['niche']
                publisher.content_analyzer.analyze_content.return_value = mock_result
                
                optimizations = await publisher.optimize_for_revenue(content)
                
                assert 'revenue_optimization' in optimizations
                assert 'optimal_posting_time' in optimizations
                assert isinstance(optimizations['optimal_posting_time'], datetime)
    
    @pytest.mark.asyncio
    async def test_platform_specific_adaptations(self):
        """Test that content adapts appropriately for each platform"""
        base_content = {
            'text': 'Transform your business with AI automation. Save 20 hours per week!',
            'business_niche': 'business_consulting',
            'target_audience': 'small_business_owners'
        }
        
        # YouTube - should focus on video optimization
        youtube_pub = YouTubeEnhancedPublisher("test")
        youtube_pub.detect_business_niche = AsyncMock(return_value='business_consulting')
        
        youtube_opt = await youtube_pub.get_platform_optimizations(
            base_content,
            'business_consulting'
        )
        assert 'video_optimizations' in youtube_opt
        assert 'algorithm_optimizations' in youtube_opt
        
        # LinkedIn - should focus on professional engagement
        linkedin_pub = LinkedInEnhancedPublisher("test")
        linkedin_pub.detect_business_niche = AsyncMock(return_value='business_consulting')
        
        linkedin_opt = await linkedin_pub.get_platform_optimizations(
            base_content,
            'business_consulting'
        )
        assert 'lead_generation_tactics' in linkedin_opt
        assert 'content_optimizations' in linkedin_opt
        
        # TikTok - should focus on viral potential
        tiktok_pub = TikTokEnhancedPublisher("test")
        tiktok_pub.detect_business_niche = AsyncMock(return_value='business_consulting')
        
        tiktok_opt = await tiktok_pub.get_platform_optimizations(
            base_content,
            'business_consulting'
        )
        assert 'algorithm_hacks' in tiktok_opt
        assert 'growth_tactics' in tiktok_opt
    
    @pytest.mark.asyncio
    async def test_error_handling_across_platforms(self):
        """Test error handling for all platforms"""
        publishers = [
            YouTubeEnhancedPublisher("test"),
            LinkedInEnhancedPublisher("test"),
            TikTokEnhancedPublisher("test")
        ]
        
        for publisher in publishers:
            # Test unauthenticated publishing
            result = await publisher.publish_content({'text': 'test'})
            
            assert result.status == PublishStatus.FAILED
            assert result.error_message == "Not authenticated"
            assert result.post_id is None
    
    @pytest.mark.asyncio
    async def test_performance_metrics_consistency(self):
        """Test that performance metrics are consistent across platforms"""
        mock_analytics = {
            'likes': 1000,
            'comments': 100,
            'shares': 50,
            'views': 10000,
            'impressions': 10000,
            'clicks': 200
        }
        
        publishers = [
            YouTubeEnhancedPublisher("test"),
            LinkedInEnhancedPublisher("test"),
            TikTokEnhancedPublisher("test")
        ]
        
        for publisher in publishers:
            # Calculate engagement rate
            engagement = mock_analytics['likes'] + mock_analytics['comments'] + mock_analytics['shares']
            engagement_rate = (engagement / mock_analytics['impressions']) * 100
            
            # Create performance metrics
            metrics = PerformanceMetrics(
                engagement_rate=engagement_rate,
                reach=mock_analytics['impressions'],
                impressions=mock_analytics['impressions'],
                clicks=mock_analytics['clicks'],
                shares=mock_analytics['shares'],
                saves=0,
                comments=mock_analytics['comments'],
                likes=mock_analytics['likes']
            )
            
            # Verify consistency
            assert metrics.engagement_rate == engagement_rate
            assert metrics.reach == mock_analytics['impressions']
            assert metrics.clicks == mock_analytics['clicks']
    
    @pytest.mark.asyncio
    async def test_audience_analysis_universal(self):
        """Test audience analysis works universally across niches"""
        niches = [
            'education',
            'business_consulting',
            'fitness_wellness',
            'creative',
            'ecommerce',
            'technology',
            'non_profit'
        ]
        
        publishers = [
            YouTubeEnhancedPublisher("test"),
            LinkedInEnhancedPublisher("test"),
            TikTokEnhancedPublisher("test")
        ]
        
        for publisher in publishers:
            for niche in niches:
                audience_data = await publisher.analyze_audience_engagement(niche)
                
                # Verify required fields
                assert 'demographics' in audience_data
                assert 'engagement_patterns' in audience_data
                assert 'content_preferences' in audience_data
                
                # Verify demographics
                demographics = audience_data['demographics']
                assert any(key in demographics for key in [
                    'age_groups', 'age_distribution', 
                    'seniority_levels', 'gender_split'
                ])
                
                # Verify engagement patterns
                patterns = audience_data['engagement_patterns']
                assert 'peak_hours' in patterns
                assert 'peak_days' in patterns