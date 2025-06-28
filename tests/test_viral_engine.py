"""
Tests for the Viral Content Engine module.

This module tests the ViralContentEngine class and its methods to ensure
they work correctly for all business niches and platforms.
"""

import asyncio
import json
import pytest
from datetime import datetime, timezone as tz
from unittest.mock import AsyncMock, MagicMock, patch, call

from backend.core.viral_engine import (
    ViralContentEngine,
    Content,
    BusinessPersona,
    PLATFORM_CONSTRAINTS,
    VIRAL_PATTERNS
)
from backend.models.content_models import (
    BusinessNicheType,
    Platform,
    ContentFormat,
    AudienceProfile,
    BrandVoice,
    ContentTheme,
    PlatformContent,
    Demographics,
    Psychographics,
    ContentPreferences,
    CommunicationStyle,
    ToneType
)


class TestViralContentEngine:
    """Test cases for ViralContentEngine class"""
    
    @pytest.fixture
    def sample_audience_profile(self):
        """Create a sample audience profile for testing"""
        return AudienceProfile(
            demographics=Demographics(
                age_range="25-35",
                gender_distribution={"male": 40.0, "female": 60.0},
                location_focus=["United States"],
                income_level="medium",
                education_level="college",
                occupation_categories=["professionals"]
            ),
            psychographics=Psychographics(
                values=["health", "success"],
                lifestyle="Active professional",
                personality_traits=["ambitious", "health-conscious"],
                attitudes=["growth-minded"],
                motivations=["self-improvement"],
                challenges=["time management"]
            ),
            pain_points=["lack of time", "stress"],
            interests=["fitness", "productivity"],
            preferred_platforms=[Platform.INSTAGRAM, Platform.LINKEDIN],
            content_preferences=ContentPreferences(
                preferred_formats=[ContentFormat.VIDEO, ContentFormat.CAROUSEL],
                optimal_length={"video": "30-60s", "carousel": "5-7 slides"},
                best_posting_times=["7AM", "12PM", "6PM"],
                engagement_triggers=["questions", "tips"],
                content_themes=["transformation", "tips"]
            ),
            buyer_journey_stage="consideration",
            influence_factors=["social proof", "results"]
        )
    
    @pytest.fixture
    def sample_brand_voice(self):
        """Create a sample brand voice for testing"""
        return BrandVoice(
            tone=ToneType.FRIENDLY,
            secondary_tones=[ToneType.INSPIRATIONAL],
            personality_traits=["supportive", "knowledgeable", "authentic"],
            communication_style=CommunicationStyle(
                vocabulary_level="simple",
                sentence_structure="varied",
                engagement_style="storytelling",
                formality_level="casual",
                emoji_usage="moderate"
            ),
            unique_phrases=["Let's grow together"],
            storytelling_approach="Personal stories with lessons",
            humor_style="Light and relatable",
            cultural_sensitivity=["inclusive language"],
            do_not_use=["jargon", "negative language"]
        )
    
    @pytest.fixture
    def sample_content_theme(self):
        """Create a sample content theme for testing"""
        return ContentTheme(
            theme_name="Morning Routines",
            description="Productive morning routine tips",
            relevance_score=0.9,
            subtopics=["exercise", "meditation", "planning"],
            content_pillars=["education", "inspiration"],
            keywords=["morningroutine", "productivity", "success"],
            audience_interest_level="high",
            competitive_advantage="Science-backed approaches"
        )
    
    @pytest.fixture
    def sample_persona(self, sample_audience_profile, sample_brand_voice, sample_content_theme):
        """Create a sample business persona for testing"""
        return BusinessPersona(
            audience_profile=sample_audience_profile,
            brand_voice=sample_brand_voice,
            business_niche=BusinessNicheType.FITNESS_WELLNESS,
            content_themes=[sample_content_theme]
        )
    
    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response for testing"""
        return AsyncMock(return_value="Test viral content response")
    
    @pytest.fixture
    def engine_with_mock_llm(self, mock_llm_response):
        """Create engine with mocked LLM"""
        with patch('backend.core.viral_engine.openai'):
            engine = ViralContentEngine(openai_api_key="test-key")
            engine._call_llm = mock_llm_response
            return engine
    
    @pytest.mark.asyncio
    async def test_init_with_openai(self):
        """Test engine initialization with OpenAI API key"""
        with patch('backend.core.viral_engine.openai.AsyncOpenAI') as mock_openai:
            engine = ViralContentEngine(openai_api_key="test-key")
            assert engine.openai_client is not None
            assert engine.default_llm == "openai"
            mock_openai.assert_called_once_with(api_key="test-key")
    
    @pytest.mark.asyncio
    async def test_init_with_anthropic(self):
        """Test engine initialization with Anthropic API key"""
        with patch('backend.core.viral_engine.anthropic.AsyncAnthropic') as mock_anthropic:
            engine = ViralContentEngine(anthropic_api_key="test-key", default_llm="anthropic")
            assert engine.anthropic_client is not None
            assert engine.default_llm == "anthropic"
            mock_anthropic.assert_called_once_with(api_key="test-key")
    
    def test_init_without_api_keys(self):
        """Test engine initialization fails without API keys"""
        with pytest.raises(ValueError, match="At least one LLM API key must be provided"):
            ViralContentEngine()
    
    def test_generate_content_id(self, engine_with_mock_llm):
        """Test unique content ID generation"""
        content = "Test content"
        platform = Platform.INSTAGRAM
        
        id1 = engine_with_mock_llm._generate_content_id(content, platform)
        id2 = engine_with_mock_llm._generate_content_id(content, platform)
        
        assert len(id1) == 12
        assert len(id2) == 12
        assert id1 != id2  # Different timestamps should create different IDs
    
    def test_validate_content_length(self, engine_with_mock_llm):
        """Test content length validation for different platforms"""
        # Test Instagram
        valid_content = "a" * 2000
        invalid_content = "a" * 3000
        
        assert engine_with_mock_llm._validate_content_length(valid_content, Platform.INSTAGRAM) == True
        assert engine_with_mock_llm._validate_content_length(invalid_content, Platform.INSTAGRAM) == False
        
        # Test Twitter
        valid_tweet = "a" * 280
        invalid_tweet = "a" * 300
        
        assert engine_with_mock_llm._validate_content_length(valid_tweet, Platform.TWITTER) == True
        assert engine_with_mock_llm._validate_content_length(invalid_tweet, Platform.TWITTER) == False
    
    def test_extract_hashtags(self, engine_with_mock_llm):
        """Test hashtag extraction from content"""
        content = "Check out our #fitness routine! #health #wellness2024"
        hashtags = engine_with_mock_llm._extract_hashtags(content)
        
        assert len(hashtags) == 3
        assert "#fitness" in hashtags
        assert "#health" in hashtags
        assert "#wellness2024" in hashtags
    
    def test_clean_content(self, engine_with_mock_llm):
        """Test content cleaning and normalization"""
        dirty_content = "  This   is   messy   content  ,with  bad  spacing  .  "
        clean = engine_with_mock_llm._clean_content(dirty_content)
        
        assert clean == "This is messy content,with bad spacing."
    
    @pytest.mark.asyncio
    async def test_generate_viral_content_success(self, engine_with_mock_llm, sample_persona):
        """Test successful viral content generation"""
        platforms = [Platform.INSTAGRAM, Platform.LINKEDIN]
        source_content = "5 minute morning workout"
        
        # Mock the LLM responses
        base_content = "🔥 Transform your mornings in just 5 minutes!"
        engine_with_mock_llm._call_llm = AsyncMock(side_effect=[
            base_content,  # Base content generation
            json.dumps({  # Instagram optimization
                "content_text": "🔥 Transform your mornings!",
                "format": "video",
                "call_to_action": "Save this for tomorrow!",
                "media_requirements": {"type": "video", "dimensions": "9:16"},
                "platform_features": ["reels", "music"]
            }),
            "[]",  # Instagram hashtags
            json.dumps({  # LinkedIn optimization
                "content_text": "Boost productivity with this morning routine",
                "format": "article",
                "call_to_action": "Share your morning routine",
                "media_requirements": {"type": "article"},
                "platform_features": ["article"]
            }),
            "[]"  # LinkedIn hashtags
        ])
        
        result = await engine_with_mock_llm.generate_viral_content(
            source_content, sample_persona, platforms
        )
        
        assert len(result) == 2
        assert Platform.INSTAGRAM in result
        assert Platform.LINKEDIN in result
        assert isinstance(result[Platform.INSTAGRAM], PlatformContent)
        assert isinstance(result[Platform.LINKEDIN], PlatformContent)
    
    @pytest.mark.asyncio
    async def test_generate_viral_content_empty_source(self, engine_with_mock_llm, sample_persona):
        """Test viral content generation with empty source content"""
        with pytest.raises(ValueError, match="Source content cannot be empty"):
            await engine_with_mock_llm.generate_viral_content("", sample_persona, [Platform.INSTAGRAM])
    
    @pytest.mark.asyncio
    async def test_generate_viral_content_no_platforms(self, engine_with_mock_llm, sample_persona):
        """Test viral content generation with no platforms"""
        with pytest.raises(ValueError, match="At least one platform must be specified"):
            await engine_with_mock_llm.generate_viral_content("Test content", sample_persona, [])
    
    @pytest.mark.asyncio
    async def test_optimize_for_platform(self, engine_with_mock_llm):
        """Test platform-specific content optimization"""
        content = "Check out this amazing fitness tip!"
        platform = Platform.INSTAGRAM
        niche = BusinessNicheType.FITNESS_WELLNESS
        
        # Mock LLM responses
        engine_with_mock_llm._call_llm = AsyncMock(side_effect=[
            json.dumps({
                "hashtags": [
                    {"tag": "fitness", "reach": "high", "relevance": "high"},
                    {"tag": "health", "reach": "high", "relevance": "high"}
                ],
                "primary_tags": ["fitness", "health"]
            }),
            json.dumps({
                "content_text": "💪 Check out this amazing fitness tip!",
                "format": "carousel",
                "call_to_action": "Save for your next workout!",
                "media_requirements": {
                    "type": "carousel",
                    "dimensions": "1:1",
                    "slides": 5
                },
                "platform_features": ["carousel", "music"],
                "posting_notes": "Post during peak hours"
            })
        ])
        
        result = await engine_with_mock_llm.optimize_for_platform(content, platform, niche)
        
        assert isinstance(result, PlatformContent)
        assert result.platform == platform
        assert result.content_format == ContentFormat.CAROUSEL
        assert "#fitness" in result.hashtags
        assert result.call_to_action == "Save for your next workout!"
    
    def test_get_platform_best_practices(self, engine_with_mock_llm):
        """Test retrieval of platform best practices"""
        instagram_practices = engine_with_mock_llm._get_platform_best_practices(Platform.INSTAGRAM)
        linkedin_practices = engine_with_mock_llm._get_platform_best_practices(Platform.LINKEDIN)
        
        assert "line breaks" in instagram_practices
        assert "professional tone" in linkedin_practices.lower()
    
    def test_truncate_content(self, engine_with_mock_llm):
        """Test intelligent content truncation"""
        long_content = "This is a very long content " * 100
        hashtags = ["#fitness", "#health", "#wellness"]
        max_length = 280  # Twitter limit
        
        truncated = engine_with_mock_llm._truncate_content(long_content, max_length, hashtags)
        
        assert len(truncated) <= max_length
        assert "..." in truncated
        assert "#fitness" in truncated
    
    @pytest.mark.asyncio
    async def test_generate_content_series(self, engine_with_mock_llm, sample_persona):
        """Test content series generation"""
        theme = "30-Day Transformation"
        count = 3
        
        # Mock LLM response
        series_response = {
            "series_title": "30-Day Body Transformation",
            "pieces": [
                {
                    "number": 1,
                    "title": "Day 1: Setting Goals",
                    "content": "Start your transformation journey",
                    "format": "video",
                    "angle": "Goal setting",
                    "cta": "Comment your goals below!"
                },
                {
                    "number": 2,
                    "title": "Day 2: First Workout",
                    "content": "Your first workout routine",
                    "format": "carousel",
                    "angle": "Workout introduction",
                    "cta": "Save this workout!"
                },
                {
                    "number": 3,
                    "title": "Day 3: Nutrition Basics",
                    "content": "Fuel your transformation",
                    "format": "image",
                    "angle": "Nutrition tips",
                    "cta": "Share your meal prep!"
                }
            ]
        }
        
        engine_with_mock_llm._call_llm = AsyncMock(return_value=json.dumps(series_response))
        
        result = await engine_with_mock_llm.generate_content_series(theme, sample_persona, count)
        
        assert len(result) == 3
        assert all(isinstance(content, Content) for content in result)
        assert result[0].metadata["series_title"] == "30-Day Body Transformation"
        assert result[0].format == ContentFormat.VIDEO
        assert result[1].format == ContentFormat.CAROUSEL
        assert result[2].format == ContentFormat.IMAGE
    
    @pytest.mark.asyncio
    async def test_generate_content_series_invalid_count(self, engine_with_mock_llm, sample_persona):
        """Test content series generation with invalid count"""
        with pytest.raises(ValueError, match="Count must be between 1 and 10"):
            await engine_with_mock_llm.generate_content_series("Theme", sample_persona, 0)
        
        with pytest.raises(ValueError, match="Count must be between 1 and 10"):
            await engine_with_mock_llm.generate_content_series("Theme", sample_persona, 11)
    
    @pytest.mark.asyncio
    async def test_create_trending_content(self, engine_with_mock_llm):
        """Test trending content creation"""
        niche = BusinessNicheType.FITNESS_WELLNESS
        trending_topics = ["New Year Goals", "Home Workouts"]
        
        # Mock LLM response
        trending_response = {
            "contents": [
                {
                    "topic": "New Year Goals",
                    "content": "New Year, New You! Start your fitness journey",
                    "connection": "Perfect time for fitness transformations",
                    "format": "video",
                    "viral_elements": ["timeliness", "motivation"]
                },
                {
                    "topic": "Home Workouts",
                    "content": "No gym? No problem! Home workout guide",
                    "connection": "Accessible fitness for everyone",
                    "format": "carousel",
                    "viral_elements": ["accessibility", "practicality"]
                }
            ]
        }
        
        engine_with_mock_llm._call_llm = AsyncMock(side_effect=[
            json.dumps(trending_response),
            json.dumps({"hashtags": [], "primary_tags": []}),  # Hashtags for content 1
            json.dumps({"hashtags": [], "primary_tags": []})   # Hashtags for content 2
        ])
        
        result = await engine_with_mock_llm.create_trending_content(niche, trending_topics)
        
        assert len(result) == 2
        assert all(isinstance(content, Content) for content in result)
        assert result[0].viral_score == 0.85  # High score for trending
        assert "#newyeargoals" in result[0].hashtags
        assert "#homeworkouts" in result[1].hashtags
    
    @pytest.mark.asyncio
    async def test_create_trending_content_no_topics(self, engine_with_mock_llm):
        """Test trending content creation with no topics"""
        with pytest.raises(ValueError, match="At least one trending topic must be provided"):
            await engine_with_mock_llm.create_trending_content(BusinessNicheType.FITNESS_WELLNESS, [])
    
    @pytest.mark.asyncio
    async def test_optimize_hashtags(self, engine_with_mock_llm):
        """Test hashtag optimization"""
        content = "Amazing fitness transformation tips"
        platform = Platform.INSTAGRAM
        niche = BusinessNicheType.FITNESS_WELLNESS
        
        # Mock LLM response
        hashtag_response = {
            "hashtags": [
                {"tag": "fitness", "reach": "high", "relevance": "high"},
                {"tag": "transformation", "reach": "medium", "relevance": "high"},
                {"tag": "fitnesstips", "reach": "medium", "relevance": "high"},
                {"tag": "healthylifestyle", "reach": "high", "relevance": "medium"},
                {"tag": "workout", "reach": "high", "relevance": "medium"}
            ],
            "primary_tags": ["fitness", "transformation", "fitnesstips"]
        }
        
        engine_with_mock_llm._call_llm = AsyncMock(return_value=json.dumps(hashtag_response))
        
        result = await engine_with_mock_llm.optimize_hashtags(content, platform, niche)
        
        assert len(result) <= PLATFORM_CONSTRAINTS[platform]["hashtag_max"]
        assert all(tag.startswith("#") for tag in result)
        assert "#fitness" in result
        assert "#transformation" in result
    
    @pytest.mark.asyncio
    async def test_optimize_hashtags_caching(self, engine_with_mock_llm):
        """Test hashtag optimization caching"""
        content = "Test content for caching"
        platform = Platform.INSTAGRAM
        niche = BusinessNicheType.FITNESS_WELLNESS
        
        # Mock LLM response
        hashtag_response = {
            "hashtags": [{"tag": "test", "reach": "low", "relevance": "high"}],
            "primary_tags": ["test"]
        }
        
        engine_with_mock_llm._call_llm = AsyncMock(return_value=json.dumps(hashtag_response))
        
        # First call
        result1 = await engine_with_mock_llm.optimize_hashtags(content, platform, niche)
        
        # Second call (should use cache)
        result2 = await engine_with_mock_llm.optimize_hashtags(content, platform, niche)
        
        assert result1 == result2
        # LLM should only be called once due to caching
        engine_with_mock_llm._call_llm.assert_called_once()
    
    def test_get_hashtag_strategy(self, engine_with_mock_llm):
        """Test platform-specific hashtag strategies"""
        instagram_strategy = engine_with_mock_llm._get_hashtag_strategy(Platform.INSTAGRAM)
        linkedin_strategy = engine_with_mock_llm._get_hashtag_strategy(Platform.LINKEDIN)
        
        assert "30 hashtags" in instagram_strategy
        assert "3-5" in linkedin_strategy
    
    def test_get_fallback_hashtags(self, engine_with_mock_llm):
        """Test fallback hashtag generation"""
        fitness_tags = engine_with_mock_llm._get_fallback_hashtags(
            BusinessNicheType.FITNESS_WELLNESS, Platform.INSTAGRAM
        )
        business_tags = engine_with_mock_llm._get_fallback_hashtags(
            BusinessNicheType.BUSINESS_CONSULTING, Platform.LINKEDIN
        )
        
        assert "#fitness" in fitness_tags
        assert "#business" in business_tags
    
    @pytest.mark.asyncio
    async def test_calculate_optimal_timing(self, engine_with_mock_llm):
        """Test optimal timing calculation"""
        niche = BusinessNicheType.FITNESS_WELLNESS
        platform = Platform.INSTAGRAM
        
        result = await engine_with_mock_llm.calculate_optimal_timing(niche, platform)
        
        assert len(result) > 0
        assert all(isinstance(time, datetime) for time in result)
        assert all(time.tzinfo is not None for time in result)  # Times should have timezone
    
    @pytest.mark.asyncio
    async def test_calculate_optimal_timing_caching(self, engine_with_mock_llm):
        """Test optimal timing caching"""
        niche = BusinessNicheType.BUSINESS_CONSULTING
        platform = Platform.LINKEDIN
        
        # First call
        result1 = await engine_with_mock_llm.calculate_optimal_timing(niche, platform)
        
        # Second call (should use cache)
        result2 = await engine_with_mock_llm.calculate_optimal_timing(niche, platform)
        
        assert result1 == result2
    
    def test_batch_calculate_viral_scores(self, engine_with_mock_llm):
        """Test batch viral score calculation"""
        contents = [
            Content(
                id="1",
                text="Amazing transformation! Check this out?",
                format=ContentFormat.VIDEO,
                platform=Platform.INSTAGRAM,
                hashtags=[],
                media_requirements={},
                viral_score=0.0,
                created_at=datetime.now(tz.utc),
                metadata={}
            ),
            Content(
                id="2",
                text="Simple tips for success",
                format=ContentFormat.CAROUSEL,
                platform=Platform.INSTAGRAM,
                hashtags=[],
                media_requirements={},
                viral_score=0.0,
                created_at=datetime.now(tz.utc),
                metadata={}
            )
        ]
        
        scores = asyncio.run(engine_with_mock_llm._batch_calculate_viral_scores(
            contents, BusinessNicheType.FITNESS_WELLNESS
        ))
        
        assert len(scores) == 2
        assert scores[0] > 0.5  # Should have bonus for emotion word, question, and video
        assert scores[1] > 0.5  # Should have bonus for carousel
        assert all(0 <= score <= 1 for score in scores)
    
    @pytest.mark.asyncio
    async def test_fallback_optimization(self, engine_with_mock_llm):
        """Test fallback optimization when main optimization fails"""
        content = "Test content for fallback"
        platform = Platform.TWITTER
        
        result = await engine_with_mock_llm._fallback_optimization(content, platform)
        
        assert isinstance(result, PlatformContent)
        assert result.platform == platform
        assert result.content_text == content
        assert result.content_format == ContentFormat.TEXT
        assert result.call_to_action == "Learn more"
    
    @pytest.mark.asyncio
    async def test_call_llm_openai(self):
        """Test LLM call with OpenAI"""
        with patch('backend.core.viral_engine.openai.AsyncOpenAI') as mock_openai_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="OpenAI response"))]
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai_class.return_value = mock_client
            
            engine = ViralContentEngine(openai_api_key="test-key")
            result = await engine._call_llm("Test prompt")
            
            assert result == "OpenAI response"
            mock_client.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_call_llm_anthropic(self):
        """Test LLM call with Anthropic"""
        with patch('backend.core.viral_engine.anthropic.AsyncAnthropic') as mock_anthropic_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Claude response")]
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_anthropic_class.return_value = mock_client
            
            engine = ViralContentEngine(anthropic_api_key="test-key", default_llm="anthropic")
            result = await engine._call_llm("Test prompt")
            
            assert result == "Claude response"
            mock_client.messages.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_business_persona_to_dict(self, sample_persona):
        """Test BusinessPersona to_dict method"""
        persona_dict = sample_persona.to_dict()
        
        assert "audience" in persona_dict
        assert "voice" in persona_dict
        assert "niche" in persona_dict
        assert "themes" in persona_dict
        assert persona_dict["niche"] == "fitness_wellness"
        assert len(persona_dict["themes"]) <= 3
    
    def test_platform_constraints_completeness(self):
        """Test that platform constraints are defined for all platforms"""
        # Check that we have constraints for major platforms
        expected_platforms = [
            Platform.INSTAGRAM,
            Platform.LINKEDIN,
            Platform.TIKTOK,
            Platform.TWITTER,
            Platform.FACEBOOK,
            Platform.YOUTUBE
        ]
        
        for platform in expected_platforms:
            assert platform in PLATFORM_CONSTRAINTS
            constraints = PLATFORM_CONSTRAINTS[platform]
            
            # Check required constraint fields
            assert "caption_max" in constraints or "post_max" in constraints or "tweet_max" in constraints
            assert "hashtag_max" in constraints
            assert "formats" in constraints
            assert len(constraints["formats"]) > 0
    
    def test_viral_patterns_for_niches(self):
        """Test that viral patterns exist for key niches"""
        key_niches = [
            BusinessNicheType.FITNESS_WELLNESS,
            BusinessNicheType.BUSINESS_CONSULTING,
            BusinessNicheType.CREATIVE,
            BusinessNicheType.EDUCATION
        ]
        
        for niche in key_niches:
            assert niche in VIRAL_PATTERNS
            patterns = VIRAL_PATTERNS[niche]
            
            # Check required pattern fields
            assert "hooks" in patterns
            assert "formats" in patterns
            assert "emotions" in patterns
            assert len(patterns["hooks"]) > 0
            assert len(patterns["formats"]) > 0
            assert len(patterns["emotions"]) > 0


@pytest.mark.asyncio
async def test_integration_full_workflow():
    """Integration test for complete viral content generation workflow"""
    with patch('backend.core.viral_engine.openai.AsyncOpenAI') as mock_openai_class:
        # Set up mock OpenAI client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        
        # Create engine
        engine = ViralContentEngine(openai_api_key="test-key")
        
        # Set up mock responses for the full workflow
        responses = [
            # Base viral content
            MagicMock(choices=[MagicMock(message=MagicMock(
                content="🔥 Transform your life with this simple morning routine!"
            ))]),
            # Instagram hashtags
            MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps({
                "hashtags": [
                    {"tag": "morningroutine", "reach": "high", "relevance": "high"},
                    {"tag": "productivity", "reach": "high", "relevance": "high"},
                    {"tag": "success", "reach": "high", "relevance": "medium"}
                ],
                "primary_tags": ["morningroutine", "productivity"]
            })))]),
            # Instagram optimization
            MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps({
                "content_text": "🔥 Transform your life with this simple morning routine!",
                "format": "carousel",
                "call_to_action": "Save this for tomorrow morning!",
                "media_requirements": {
                    "type": "carousel",
                    "dimensions": "1:1",
                    "slides": 7
                },
                "platform_features": ["carousel"],
                "posting_notes": "Post at 7AM for best engagement"
            })))]),
            # LinkedIn hashtags
            MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps({
                "hashtags": [
                    {"tag": "productivity", "reach": "high", "relevance": "high"},
                    {"tag": "leadership", "reach": "high", "relevance": "medium"}
                ],
                "primary_tags": ["productivity", "leadership"]
            })))]),
            # LinkedIn optimization
            MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps({
                "content_text": "Transform your professional life with this morning routine",
                "format": "article",
                "call_to_action": "What's your morning routine? Share below!",
                "media_requirements": {
                    "type": "article"
                },
                "platform_features": ["article"],
                "posting_notes": "Best for Tuesday-Thursday posting"
            })))])
        ]
        
        mock_client.chat.completions.create = AsyncMock(side_effect=responses)
        
        # Create test persona
        audience = AudienceProfile(
            demographics=Demographics(
                age_range="25-35",
                gender_distribution={"male": 50.0, "female": 50.0},
                location_focus=["United States"],
                income_level="medium",
                education_level="college",
                occupation_categories=["professionals"]
            ),
            psychographics=Psychographics(
                values=["success", "growth"],
                lifestyle="Busy professional",
                personality_traits=["ambitious", "organized"],
                attitudes=["positive"],
                motivations=["achievement"],
                challenges=["time management"]
            ),
            pain_points=["productivity", "work-life balance"],
            interests=["self-improvement", "business"],
            preferred_platforms=[Platform.INSTAGRAM, Platform.LINKEDIN],
            content_preferences=ContentPreferences(
                preferred_formats=[ContentFormat.CAROUSEL, ContentFormat.ARTICLE],
                optimal_length={"carousel": "7 slides", "article": "1000 words"},
                best_posting_times=["7AM", "12PM"],
                engagement_triggers=["questions", "actionable tips"],
                content_themes=["productivity", "success"]
            ),
            buyer_journey_stage="awareness",
            influence_factors=["expert advice", "peer recommendations"]
        )
        
        voice = BrandVoice(
            tone=ToneType.PROFESSIONAL,
            secondary_tones=[ToneType.INSPIRATIONAL],
            personality_traits=["knowledgeable", "approachable", "motivating"],
            communication_style=CommunicationStyle(
                vocabulary_level="moderate",
                sentence_structure="varied",
                engagement_style="educational",
                formality_level="formal",
                emoji_usage="minimal"
            ),
            unique_phrases=["Elevate your potential"],
            storytelling_approach="Data-driven insights with real examples",
            humor_style=None,
            cultural_sensitivity=["inclusive language"],
            do_not_use=["slang", "controversial topics"]
        )
        
        theme = ContentTheme(
            theme_name="Productivity Hacks",
            description="Evidence-based productivity strategies",
            relevance_score=0.95,
            subtopics=["morning routines", "time blocking", "focus techniques"],
            content_pillars=["education", "inspiration", "practical tips"],
            keywords=["productivity", "efficiency", "success"],
            audience_interest_level="high",
            competitive_advantage="Research-backed methods"
        )
        
        persona = BusinessPersona(
            audience_profile=audience,
            brand_voice=voice,
            business_niche=BusinessNicheType.BUSINESS_CONSULTING,
            content_themes=[theme]
        )
        
        # Run the workflow
        result = await engine.generate_viral_content(
            "Morning routine for peak productivity",
            persona,
            [Platform.INSTAGRAM, Platform.LINKEDIN]
        )
        
        # Verify results
        assert len(result) == 2
        assert Platform.INSTAGRAM in result
        assert Platform.LINKEDIN in result
        
        # Check Instagram content
        ig_content = result[Platform.INSTAGRAM]
        assert ig_content.platform == Platform.INSTAGRAM
        assert ig_content.content_format == ContentFormat.CAROUSEL
        assert "#morningroutine" in ig_content.hashtags
        assert ig_content.call_to_action == "Save this for tomorrow morning!"
        
        # Check LinkedIn content
        li_content = result[Platform.LINKEDIN]
        assert li_content.platform == Platform.LINKEDIN
        assert li_content.content_format == ContentFormat.ARTICLE
        assert "#productivity" in li_content.hashtags
        assert "What's your morning routine?" in li_content.call_to_action
        
        # Verify API calls
        assert mock_client.chat.completions.create.call_count == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])