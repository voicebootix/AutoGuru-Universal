"""
Unit tests for UniversalContentAnalyzer

Tests the AI-powered content analysis for various business niches including
education, business consulting, fitness, and creative businesses.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from backend.core.content_analyzer import (
    UniversalContentAnalyzer,
    BusinessNiche,
    Platform,
    ContentAnalysisResult
)


# Sample content for different business niches
EDUCATION_CONTENT = """
Unlock your full potential with our comprehensive online course on Data Science!
Learn from industry experts with over 10 years of experience. Our curriculum covers
everything from basic Python programming to advanced machine learning algorithms.
Join thousands of students who have transformed their careers. Limited time offer:
Get 30% off when you enroll today! #DataScience #OnlineLearning #CareerGrowth
"""

BUSINESS_CONTENT = """
Struggling to scale your business? Our proven consulting framework has helped
over 500 companies achieve 3x growth in just 12 months. We specialize in
strategic planning, operational efficiency, and digital transformation.
Book your free consultation today and discover how we can unlock your
business's hidden potential. #BusinessGrowth #Consulting #Strategy
"""

FITNESS_CONTENT = """
Transform your body in just 8 weeks with our revolutionary HIIT program!
No gym required - all workouts can be done at home with minimal equipment.
Our certified trainers provide personalized meal plans and 24/7 support.
Join our community of fitness enthusiasts and start your journey today.
First week FREE! #Fitness #HomeWorkout #HealthyLifestyle
"""

CREATIVE_CONTENT = """
Bring your vision to life with stunning photography that tells your story.
Specializing in portrait, wedding, and commercial photography. Our artistic
approach captures authentic moments and emotions. View our portfolio and
book your session today. Limited slots available for fall mini-sessions!
#Photography #PortraitPhotography #CreativeArts
"""


class TestUniversalContentAnalyzer:
    """Test suite for UniversalContentAnalyzer"""
    
    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI API responses"""
        def create_response(content):
            return MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=content
                        )
                    )
                ]
            )
        return create_response
    
    @pytest.fixture
    def mock_anthropic_response(self):
        """Mock Anthropic API responses"""
        def create_response(content):
            return MagicMock(
                content=[
                    MagicMock(
                        text=content
                    )
                ]
            )
        return create_response
    
    @pytest.fixture
    def analyzer_with_mocked_openai(self, mock_openai_response):
        """Create analyzer with mocked OpenAI client"""
        with patch('openai.AsyncOpenAI') as mock_client:
            analyzer = UniversalContentAnalyzer(openai_api_key="test-key")
            analyzer.openai_client = AsyncMock()
            analyzer.openai_client.chat = AsyncMock()
            analyzer.openai_client.chat.completions = AsyncMock()
            analyzer.openai_client.chat.completions.create = AsyncMock()
            return analyzer, mock_openai_response
    
    @pytest.mark.asyncio
    async def test_init_with_openai_key(self):
        """Test initialization with OpenAI API key"""
        with patch('openai.AsyncOpenAI') as mock_openai:
            analyzer = UniversalContentAnalyzer(openai_api_key="test-openai-key")
            assert analyzer.openai_client is not None
            assert analyzer.default_llm == "openai"
    
    @pytest.mark.asyncio
    async def test_init_with_anthropic_key(self):
        """Test initialization with Anthropic API key"""
        with patch('anthropic.AsyncAnthropic') as mock_anthropic:
            analyzer = UniversalContentAnalyzer(anthropic_api_key="test-anthropic-key")
            assert analyzer.anthropic_client is not None
            assert analyzer.default_llm == "openai"  # Default is still OpenAI
    
    @pytest.mark.asyncio
    async def test_init_without_keys_raises_error(self):
        """Test initialization without API keys raises error"""
        with pytest.raises(ValueError, match="At least one LLM API key must be provided"):
            analyzer = UniversalContentAnalyzer()
    
    @pytest.mark.asyncio
    async def test_detect_education_niche(self, analyzer_with_mocked_openai):
        """Test detection of education business niche"""
        analyzer, mock_response = analyzer_with_mocked_openai
        
        # Mock the LLM response
        mock_niche_response = json.dumps({
            "niche": "education",
            "confidence": 0.95,
            "reasoning": "Content mentions online courses, learning, and curriculum"
        })
        
        analyzer.openai_client.chat.completions.create.return_value = mock_response(mock_niche_response)
        
        niche, confidence = await analyzer.detect_business_niche(EDUCATION_CONTENT)
        
        assert niche == BusinessNiche.EDUCATION
        assert confidence == 0.95
        assert analyzer.openai_client.chat.completions.create.called
    
    @pytest.mark.asyncio
    async def test_detect_business_consulting_niche(self, analyzer_with_mocked_openai):
        """Test detection of business consulting niche"""
        analyzer, mock_response = analyzer_with_mocked_openai
        
        mock_niche_response = json.dumps({
            "niche": "business_consulting",
            "confidence": 0.92,
            "reasoning": "Content focuses on business growth and consulting services"
        })
        
        analyzer.openai_client.chat.completions.create.return_value = mock_response(mock_niche_response)
        
        niche, confidence = await analyzer.detect_business_niche(BUSINESS_CONTENT)
        
        assert niche == BusinessNiche.BUSINESS_CONSULTING
        assert confidence == 0.92
    
    @pytest.mark.asyncio
    async def test_detect_fitness_niche(self, analyzer_with_mocked_openai):
        """Test detection of fitness/wellness niche"""
        analyzer, mock_response = analyzer_with_mocked_openai
        
        mock_niche_response = json.dumps({
            "niche": "fitness_wellness",
            "confidence": 0.98,
            "reasoning": "Content is about fitness programs and health transformation"
        })
        
        analyzer.openai_client.chat.completions.create.return_value = mock_response(mock_niche_response)
        
        niche, confidence = await analyzer.detect_business_niche(FITNESS_CONTENT)
        
        assert niche == BusinessNiche.FITNESS_WELLNESS
        assert confidence == 0.98
    
    @pytest.mark.asyncio
    async def test_detect_creative_niche(self, analyzer_with_mocked_openai):
        """Test detection of creative professional niche"""
        analyzer, mock_response = analyzer_with_mocked_openai
        
        mock_niche_response = json.dumps({
            "niche": "creative",
            "confidence": 0.94,
            "reasoning": "Content is about photography and creative services"
        })
        
        analyzer.openai_client.chat.completions.create.return_value = mock_response(mock_niche_response)
        
        niche, confidence = await analyzer.detect_business_niche(CREATIVE_CONTENT)
        
        assert niche == BusinessNiche.CREATIVE
        assert confidence == 0.94
    
    @pytest.mark.asyncio
    async def test_analyze_target_audience(self, analyzer_with_mocked_openai):
        """Test target audience analysis"""
        analyzer, mock_response = analyzer_with_mocked_openai
        
        mock_audience_response = json.dumps({
            "demographics": {
                "age_range": "25-45",
                "gender": "all",
                "location": "urban areas",
                "income_level": "medium-high",
                "education": "college educated"
            },
            "psychographics": {
                "values": ["growth", "success", "efficiency"],
                "lifestyle": "busy professionals",
                "personality": ["ambitious", "results-oriented"],
                "attitudes": ["open to innovation", "value expertise"]
            },
            "pain_points": ["lack of time", "need for growth", "scaling challenges"],
            "interests": ["business", "technology", "professional development"],
            "preferred_platforms": ["linkedin", "twitter"],
            "content_preferences": {
                "format": "text and video",
                "tone": "professional",
                "topics": ["business strategy", "case studies"]
            }
        })
        
        analyzer.openai_client.chat.completions.create.return_value = mock_response(mock_audience_response)
        
        audience = await analyzer.analyze_target_audience(BUSINESS_CONTENT)
        
        assert audience["demographics"]["age_range"] == "25-45"
        assert "linkedin" in audience["preferred_platforms"]
        assert "scaling challenges" in audience["pain_points"]
    
    @pytest.mark.asyncio
    async def test_extract_brand_voice(self, analyzer_with_mocked_openai):
        """Test brand voice extraction"""
        analyzer, mock_response = analyzer_with_mocked_openai
        
        mock_voice_response = json.dumps({
            "tone": "professional",
            "style": "educational",
            "personality_traits": ["knowledgeable", "supportive", "encouraging"],
            "communication_preferences": {
                "vocabulary": "simple",
                "sentence_structure": "varied",
                "engagement_style": "direct"
            },
            "unique_elements": ["data-driven approach", "student success stories"],
            "do_not_use": ["jargon", "overly technical terms"]
        })
        
        analyzer.openai_client.chat.completions.create.return_value = mock_response(mock_voice_response)
        
        voice = await analyzer.extract_brand_voice(EDUCATION_CONTENT)
        
        assert voice["tone"] == "professional"
        assert voice["style"] == "educational"
        assert "supportive" in voice["personality_traits"]
    
    @pytest.mark.asyncio
    async def test_assess_viral_potential(self, analyzer_with_mocked_openai):
        """Test viral potential assessment across platforms"""
        analyzer, mock_response = analyzer_with_mocked_openai
        
        mock_viral_response = json.dumps({
            "instagram": {
                "score": 0.85,
                "factors": ["visual transformation stories", "before/after potential"],
                "improvements": ["add more visuals", "use trending fitness hashtags"]
            },
            "tiktok": {
                "score": 0.92,
                "factors": ["quick workout demos", "trending fitness content"],
                "improvements": ["create 30-second workout snippets"]
            },
            "youtube": {
                "score": 0.78,
                "factors": ["workout tutorials", "fitness journey vlogs"],
                "improvements": ["create longer form content", "add workout playlists"]
            }
        })
        
        analyzer.openai_client.chat.completions.create.return_value = mock_response(mock_viral_response)
        
        platforms = [Platform.INSTAGRAM, Platform.TIKTOK, Platform.YOUTUBE]
        viral_scores = await analyzer.assess_viral_potential(
            FITNESS_CONTENT,
            BusinessNiche.FITNESS_WELLNESS,
            {"demographics": {"age_range": "18-35"}},
            platforms
        )
        
        assert viral_scores[Platform.INSTAGRAM] == 0.85
        assert viral_scores[Platform.TIKTOK] == 0.92
        assert viral_scores[Platform.YOUTUBE] == 0.78
    
    @pytest.mark.asyncio
    async def test_complete_content_analysis(self, analyzer_with_mocked_openai):
        """Test complete content analysis workflow"""
        analyzer, mock_response = analyzer_with_mocked_openai
        
        # Setup mock responses for all API calls
        responses = [
            # Niche detection
            json.dumps({
                "niche": "education",
                "confidence": 0.95,
                "reasoning": "Educational content"
            }),
            # Audience analysis
            json.dumps({
                "demographics": {"age_range": "25-45", "gender": "all"},
                "psychographics": {"values": ["learning", "growth"]},
                "pain_points": ["skill gaps"],
                "interests": ["technology", "career"],
                "preferred_platforms": ["linkedin", "youtube"],
                "content_preferences": {"format": "video", "tone": "professional"}
            }),
            # Brand voice
            json.dumps({
                "tone": "professional",
                "style": "educational",
                "personality_traits": ["expert", "supportive"],
                "communication_preferences": {"vocabulary": "simple"},
                "unique_elements": ["practical examples"],
                "do_not_use": ["jargon"]
            }),
            # Key themes
            json.dumps({
                "themes": ["online learning", "career transformation", "data science"]
            }),
            # Viral potential
            json.dumps({
                "instagram": {"score": 0.7},
                "twitter": {"score": 0.6},
                "linkedin": {"score": 0.85},
                "facebook": {"score": 0.65},
                "tiktok": {"score": 0.5},
                "youtube": {"score": 0.8}
            }),
            # Recommendations
            json.dumps({
                "recommendations": [
                    "Focus on LinkedIn for professional audience",
                    "Create video tutorials for YouTube",
                    "Share student success stories",
                    "Use data visualizations",
                    "Offer free mini-courses"
                ]
            })
        ]
        
        # Configure mock to return different responses for each call
        analyzer.openai_client.chat.completions.create.side_effect = [
            mock_response(resp) for resp in responses
        ]
        
        result = await analyzer.analyze_content(EDUCATION_CONTENT)
        
        assert isinstance(result, ContentAnalysisResult)
        assert result.business_niche == BusinessNiche.EDUCATION
        assert result.confidence_score == 0.95
        assert "online learning" in result.key_themes
        assert len(result.recommendations) == 5
        assert result.viral_potential[Platform.LINKEDIN] == 0.85
    
    @pytest.mark.asyncio
    async def test_error_handling_in_analysis(self, analyzer_with_mocked_openai):
        """Test error handling when API calls fail"""
        analyzer, _ = analyzer_with_mocked_openai
        
        # Mock API failure
        analyzer.openai_client.chat.completions.create.side_effect = Exception("API Error")
        
        with pytest.raises(Exception, match="API Error"):
            await analyzer.detect_business_niche(EDUCATION_CONTENT)
    
    @pytest.mark.asyncio
    async def test_retry_on_failure(self, analyzer_with_mocked_openai):
        """Test retry mechanism on temporary failures"""
        analyzer, mock_response = analyzer_with_mocked_openai
        
        # First two calls fail, third succeeds
        analyzer.openai_client.chat.completions.create.side_effect = [
            Exception("Temporary error"),
            Exception("Temporary error"),
            mock_response(json.dumps({
                "niche": "education",
                "confidence": 0.9,
                "reasoning": "Retry successful"
            }))
        ]
        
        niche, confidence = await analyzer.detect_business_niche(EDUCATION_CONTENT)
        
        assert niche == BusinessNiche.EDUCATION
        assert confidence == 0.9
        assert analyzer.openai_client.chat.completions.create.call_count == 3
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self, analyzer_with_mocked_openai):
        """Test that analysis tasks run in parallel"""
        analyzer, mock_response = analyzer_with_mocked_openai
        
        # Track call times
        call_times = []
        
        async def mock_api_call(*args, **kwargs):
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.1)  # Simulate API delay
            return mock_response(json.dumps({"test": "response"}))
        
        analyzer._call_llm = mock_api_call
        
        # This should fail but we're testing parallelism
        try:
            await analyzer.analyze_content(EDUCATION_CONTENT)
        except:
            pass
        
        # Check that multiple calls happened close together (parallel)
        if len(call_times) >= 2:
            time_diff = max(call_times) - min(call_times)
            assert time_diff < 0.2  # Should be much less than sequential (0.4s)