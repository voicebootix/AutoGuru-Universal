"""
Unit tests for the Universal Persona Factory

Tests the AI-powered persona generation that works for ANY business niche.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from backend.core.persona_factory import (
    UniversalPersonaFactory,
    PersonaArchetype,
    PersonaStyle,
    BasePersona,
    BusinessPersona,
    EducatorPersona,
    WellnessPersona,
    CreativePersona,
    PlatformPersona
)
from backend.models.content_models import (
    BusinessNicheType,
    Platform,
    ContentAnalysis,
    BusinessNiche,
    AudienceProfile,
    BrandVoice,
    ContentTheme,
    Demographics,
    Psychographics,
    ContentPreferences,
    CommunicationStyle,
    ToneType,
    ContentFormat
)


@pytest.fixture
def mock_content_analysis():
    """Create a mock ContentAnalysis object for testing"""
    # Create nested objects
    demographics = Demographics(
        age_range="25-35",
        gender_distribution={"male": 40.0, "female": 55.0, "other": 5.0},
        location_focus=["United States", "Canada"],
        income_level="medium",
        education_level="college",
        occupation_categories=["professionals", "entrepreneurs"]
    )
    
    psychographics = Psychographics(
        values=["health", "success", "balance"],
        lifestyle="Active and career-focused",
        personality_traits=["ambitious", "health-conscious"],
        attitudes=["growth-minded", "tech-savvy"],
        motivations=["self-improvement", "efficiency"],
        challenges=["time management", "work-life balance"]
    )
    
    content_preferences = ContentPreferences(
        preferred_formats=[ContentFormat.VIDEO, ContentFormat.CAROUSEL],
        optimal_length={"video": "30-60 seconds", "carousel": "5-7 slides"},
        best_posting_times=["7-9 AM", "12-1 PM", "5-7 PM"],
        engagement_triggers=["questions", "polls", "behind-the-scenes"],
        content_themes=["tips", "transformation", "motivation"]
    )
    
    audience_profile = AudienceProfile(
        demographics=demographics,
        psychographics=psychographics,
        pain_points=["lack of time", "information overload"],
        interests=["fitness", "productivity", "technology"],
        preferred_platforms=[Platform.INSTAGRAM, Platform.LINKEDIN],
        content_preferences=content_preferences,
        buyer_journey_stage="consideration",
        influence_factors=["social proof", "expert endorsement"]
    )
    
    business_niche = BusinessNiche(
        niche_type=BusinessNicheType.FITNESS_WELLNESS,
        confidence_score=0.92,
        sub_niches=["personal_training", "nutrition_coaching"],
        reasoning="Content focuses on workout routines and meal planning",
        keywords=["fitness", "health", "workout", "nutrition"]
    )
    
    communication_style = CommunicationStyle(
        vocabulary_level="simple",
        sentence_structure="varied",
        engagement_style="storytelling",
        formality_level="casual",
        emoji_usage="moderate"
    )
    
    brand_voice = BrandVoice(
        tone=ToneType.INSPIRATIONAL,
        secondary_tones=[ToneType.FRIENDLY, ToneType.EDUCATIONAL],
        personality_traits=["motivating", "supportive", "knowledgeable"],
        communication_style=communication_style,
        unique_phrases=["Let's crush it!", "Your journey starts now"],
        storytelling_approach="Personal transformation stories",
        humor_style="Light and relatable",
        cultural_sensitivity=["inclusive language", "diverse representation"],
        do_not_use=["shame", "guilt", "pressure"]
    )
    
    content_theme = ContentTheme(
        theme_name="Fitness Transformation",
        description="Journey-based fitness content",
        relevance_score=0.95,
        subtopics=["workouts", "nutrition", "mindset"],
        content_pillars=["education", "motivation", "community"],
        keywords=["fitness", "transformation", "health"],
        audience_interest_level="high",
        competitive_advantage="Real client success stories"
    )
    
    # Create mock ContentAnalysis
    content_analysis = Mock(spec=ContentAnalysis)
    content_analysis.business_niche = business_niche
    content_analysis.audience_profile = audience_profile
    content_analysis.brand_voice = brand_voice
    content_analysis.content_themes = [content_theme]
    
    return content_analysis


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response for persona generation"""
    return {
        "name": "Coach Alex",
        "archetype": "motivator",
        "style": "inspirational",
        "traits": ["energetic", "supportive", "knowledgeable", "authentic", "results-driven"],
        "values": ["health", "personal growth", "community", "authenticity"],
        "voice_characteristics": {
            "tone_modifiers": ["encouraging", "upbeat"],
            "language_patterns": ["action verbs", "inclusive language"],
            "signature_phrases": ["Let's crush it!", "You've got this!"],
            "conversation_starters": ["How's your energy today?", "Ready for a challenge?"]
        },
        "content_preferences": {
            "favorite_topics": ["workout tips", "nutrition hacks", "mindset"],
            "content_formats": ["video", "carousel", "stories"],
            "storytelling_style": "Personal transformation narratives",
            "humor_level": "light"
        },
        "business_focus": "Holistic fitness transformation",
        "expertise_areas": ["strength training", "nutrition", "mindset coaching"],
        "unique_perspective": "Fitness is a journey of self-discovery, not just physical change",
        "engagement_approach": "Building a supportive community through shared challenges"
    }


@pytest.fixture
def mock_platform_adaptation():
    """Mock platform-specific adaptation response"""
    return {
        "platform_voice": {
            "tone_adjustment": "More visual and inspirational",
            "language_style": "Short, punchy captions with emojis",
            "emoji_usage": "moderate",
            "hashtag_strategy": "Mix of branded and trending fitness hashtags",
            "mention_strategy": "Tag clients in transformation posts"
        },
        "content_strategy": {
            "primary_content_types": ["reels", "carousel posts"],
            "posting_frequency": "1-2 times daily",
            "optimal_length": "15-30 second reels, 5-7 slide carousels",
            "visual_style": "Bright, energetic, before/after focused",
            "engagement_hooks": ["Questions in captions", "Polls in stories", "Challenge invitations"]
        },
        "engagement_tactics": [
            "Reply to all comments within 2 hours",
            "Use stories for behind-the-scenes content",
            "Host weekly Q&A sessions",
            "Create shareable quote graphics"
        ],
        "platform_specific_traits": [
            "Visual storyteller",
            "Hashtag strategist",
            "Community builder"
        ]
    }


class TestUniversalPersonaFactory:
    """Test suite for UniversalPersonaFactory"""
    
    @pytest.mark.asyncio
    async def test_factory_initialization_with_openai(self):
        """Test factory initialization with OpenAI client"""
        with patch('backend.config.settings.get_settings') as mock_settings:
            # Mock settings with OpenAI API key
            mock_settings.return_value.ai_service.openai_api_key.get_secret_value.return_value = "test-key"
            mock_settings.return_value.ai_service.anthropic_api_key = None
            
            with patch('openai.AsyncOpenAI'):
                factory = UniversalPersonaFactory()
                assert factory.openai_client is not None
                assert factory.default_llm == "openai"
    
    @pytest.mark.asyncio
    async def test_factory_initialization_with_anthropic(self):
        """Test factory initialization with Anthropic client"""
        with patch('backend.config.settings.get_settings') as mock_settings:
            # Mock settings with Anthropic API key
            mock_settings.return_value.ai_service.openai_api_key = None
            mock_settings.return_value.ai_service.anthropic_api_key.get_secret_value.return_value = "test-key"
            
            with patch('anthropic.AsyncAnthropic'):
                factory = UniversalPersonaFactory()
                assert factory.anthropic_client is not None
                assert factory.default_llm == "anthropic"
    
    @pytest.mark.asyncio
    async def test_factory_initialization_no_keys(self):
        """Test factory initialization fails without API keys"""
        with patch('backend.config.settings.get_settings') as mock_settings:
            # Mock settings with no API keys
            mock_settings.return_value.ai_service.openai_api_key = None
            mock_settings.return_value.ai_service.anthropic_api_key = None
            
            with pytest.raises(ValueError, match="At least one AI service"):
                UniversalPersonaFactory()
    
    @pytest.mark.asyncio
    async def test_generate_persona_fitness(self, mock_content_analysis, mock_openai_response):
        """Test persona generation for fitness business"""
        with patch('backend.config.settings.get_settings') as mock_settings:
            mock_settings.return_value.ai_service.openai_api_key.get_secret_value.return_value = "test-key"
            mock_settings.return_value.ai_service.anthropic_api_key = None
            mock_settings.return_value.ai_service.openai_model = "gpt-4"
            mock_settings.return_value.ai_service.openai_max_tokens = 2000
            
            with patch('openai.AsyncOpenAI') as mock_openai:
                # Mock the OpenAI response
                mock_client = AsyncMock()
                mock_openai.return_value = mock_client
                mock_response = Mock()
                mock_response.choices = [Mock(message=Mock(content=json.dumps(mock_openai_response)))]
                mock_client.chat.completions.create.return_value = mock_response
                
                factory = UniversalPersonaFactory()
                persona = await factory.generate_persona(mock_content_analysis)
                
                # Verify persona attributes
                assert isinstance(persona, BusinessPersona)
                assert persona.name == "Coach Alex"
                assert persona.archetype == PersonaArchetype.MOTIVATOR
                assert persona.style == PersonaStyle.INSPIRATIONAL
                assert len(persona.traits) <= 7
                assert len(persona.values) <= 5
                assert persona.business_focus == "Holistic fitness transformation"
                assert "strength training" in persona.expertise_areas
    
    @pytest.mark.asyncio
    async def test_adapt_persona_for_instagram(self, mock_platform_adaptation):
        """Test adapting persona for Instagram platform"""
        # Create a test persona
        base_persona = BusinessPersona(
            name="Coach Alex",
            archetype=PersonaArchetype.MOTIVATOR,
            style=PersonaStyle.INSPIRATIONAL,
            traits=["energetic", "supportive"],
            values=["health", "growth"],
            voice_characteristics={"tone": "upbeat"},
            content_preferences={"formats": ["video"]},
            business_focus="Fitness transformation",
            expertise_areas=["fitness", "nutrition"],
            unique_perspective="Holistic approach",
            engagement_approach="Community building"
        )
        
        with patch('backend.config.settings.get_settings') as mock_settings:
            mock_settings.return_value.ai_service.openai_api_key.get_secret_value.return_value = "test-key"
            mock_settings.return_value.ai_service.anthropic_api_key = None
            mock_settings.return_value.ai_service.openai_model = "gpt-4"
            mock_settings.return_value.ai_service.openai_max_tokens = 2000
            
            with patch('openai.AsyncOpenAI') as mock_openai:
                # Mock the OpenAI response
                mock_client = AsyncMock()
                mock_openai.return_value = mock_client
                mock_response = Mock()
                mock_response.choices = [Mock(message=Mock(content=json.dumps(mock_platform_adaptation)))]
                mock_client.chat.completions.create.return_value = mock_response
                
                factory = UniversalPersonaFactory()
                platform_persona = await factory.adapt_persona_for_platform(
                    base_persona, Platform.INSTAGRAM
                )
                
                # Verify platform adaptation
                assert isinstance(platform_persona, PlatformPersona)
                assert platform_persona.platform == Platform.INSTAGRAM
                assert platform_persona.platform_voice["emoji_usage"] == "moderate"
                assert "reels" in platform_persona.content_strategy["primary_content_types"]
                assert len(platform_persona.engagement_tactics) > 0
    
    @pytest.mark.asyncio
    async def test_create_educator_persona(self):
        """Test creating an educator persona"""
        educator_response = {
            "name": "Professor Sarah",
            "style": "educational",
            "traits": ["patient", "knowledgeable", "encouraging", "organized", "innovative"],
            "values": ["learning", "growth", "curiosity", "excellence"],
            "voice_characteristics": {
                "explanation_style": "Step-by-step with real-world examples",
                "example_usage": "Practical, relatable scenarios",
                "question_style": "Thought-provoking and open-ended",
                "encouragement_style": "Celebrating small wins and progress"
            },
            "content_preferences": {
                "lesson_structure": "Introduction, concept, example, practice",
                "visual_aids": "Infographics and diagrams",
                "interaction_level": "High engagement with Q&A",
                "assessment_style": "Self-reflection and practical application"
            },
            "business_focus": "Data science education",
            "expertise_areas": ["Python", "machine learning", "data visualization"],
            "unique_perspective": "Making complex concepts accessible to beginners",
            "engagement_approach": "Interactive learning with hands-on projects",
            "teaching_philosophy": "Learn by doing, fail forward",
            "learning_frameworks": ["Bloom's Taxonomy", "Active Learning", "Scaffolding"]
        }
        
        with patch('backend.config.settings.get_settings') as mock_settings:
            mock_settings.return_value.ai_service.openai_api_key.get_secret_value.return_value = "test-key"
            mock_settings.return_value.ai_service.anthropic_api_key = None
            mock_settings.return_value.ai_service.openai_model = "gpt-4"
            mock_settings.return_value.ai_service.openai_max_tokens = 2000
            
            with patch('openai.AsyncOpenAI') as mock_openai:
                # Mock the OpenAI response
                mock_client = AsyncMock()
                mock_openai.return_value = mock_client
                mock_response = Mock()
                mock_response.choices = [Mock(message=Mock(content=json.dumps(educator_response)))]
                mock_client.chat.completions.create.return_value = mock_response
                
                factory = UniversalPersonaFactory()
                educator = await factory.create_educator_persona(
                    "Data Science", "Interactive and practical"
                )
                
                # Verify educator persona
                assert isinstance(educator, EducatorPersona)
                assert educator.name == "Professor Sarah"
                assert educator.archetype == PersonaArchetype.EDUCATOR
                assert educator.teaching_philosophy == "Learn by doing, fail forward"
                assert "Bloom's Taxonomy" in educator.learning_frameworks
    
    @pytest.mark.asyncio
    async def test_create_wellness_persona(self):
        """Test creating a wellness persona"""
        wellness_response = {
            "name": "Wellness Warrior Maya",
            "style": "supportive",
            "traits": ["empathetic", "motivating", "holistic", "authentic", "mindful"],
            "values": ["balance", "self-care", "growth", "community"],
            "voice_characteristics": {
                "motivation_style": "Gentle encouragement with accountability",
                "empathy_expression": "Acknowledging struggles and celebrating progress",
                "progress_celebration": "Every step counts approach",
                "challenge_approach": "Gradual, sustainable challenges"
            },
            "content_preferences": {
                "transformation_stories": "Real client journeys with ups and downs",
                "scientific_backing": "Evidence-based but accessible",
                "practical_tips": "5-minute wellness wins",
                "community_building": "Support circles and buddy systems"
            },
            "business_focus": "Holistic wellness coaching",
            "expertise_areas": ["mindfulness", "nutrition", "movement"],
            "unique_perspective": "Wellness is a personal journey, not a destination",
            "engagement_approach": "Creating safe spaces for vulnerability",
            "wellness_philosophy": "Small steps lead to lasting change",
            "transformation_approach": "Inside-out transformation focusing on mindset"
        }
        
        with patch('backend.config.settings.get_settings') as mock_settings:
            mock_settings.return_value.ai_service.openai_api_key.get_secret_value.return_value = "test-key"
            mock_settings.return_value.ai_service.anthropic_api_key = None
            mock_settings.return_value.ai_service.openai_model = "gpt-4"
            mock_settings.return_value.ai_service.openai_max_tokens = 2000
            
            with patch('openai.AsyncOpenAI') as mock_openai:
                # Mock the OpenAI response
                mock_client = AsyncMock()
                mock_openai.return_value = mock_client
                mock_response = Mock()
                mock_response.choices = [Mock(message=Mock(content=json.dumps(wellness_response)))]
                mock_client.chat.completions.create.return_value = mock_response
                
                factory = UniversalPersonaFactory()
                wellness = await factory.create_wellness_persona(
                    "Holistic health", "Gentle and sustainable"
                )
                
                # Verify wellness persona
                assert isinstance(wellness, WellnessPersona)
                assert wellness.name == "Wellness Warrior Maya"
                assert wellness.archetype == PersonaArchetype.MOTIVATOR
                assert wellness.wellness_philosophy == "Small steps lead to lasting change"
                assert wellness.transformation_approach == "Inside-out transformation focusing on mindset"
    
    @pytest.mark.asyncio
    async def test_create_creative_persona(self):
        """Test creating a creative persona"""
        creative_response = {
            "name": "Visual Storyteller Luna",
            "style": "creative",
            "traits": ["imaginative", "expressive", "observant", "passionate", "innovative"],
            "values": ["authenticity", "beauty", "emotion", "storytelling"],
            "voice_characteristics": {
                "creative_expression": "Poetic descriptions of visual moments",
                "inspiration_sharing": "Behind every image is a story",
                "process_description": "From vision to creation journey",
                "artistic_critique": "Constructive and inspiring feedback"
            },
            "content_preferences": {
                "visual_storytelling": "Narrative-driven imagery",
                "behind_scenes": "Raw creative process footage",
                "collaboration_style": "Co-creation with community",
                "trend_interaction": "Putting unique spin on trends"
            },
            "business_focus": "Photography and visual storytelling",
            "expertise_areas": ["portrait photography", "brand storytelling", "photo editing"],
            "unique_perspective": "Every moment has a story waiting to be captured",
            "engagement_approach": "Inviting followers into the creative process",
            "creative_philosophy": "Authenticity over perfection",
            "artistic_influences": ["Annie Leibovitz", "Street photography", "Film noir"]
        }
        
        with patch('backend.config.settings.get_settings') as mock_settings:
            mock_settings.return_value.ai_service.openai_api_key.get_secret_value.return_value = "test-key"
            mock_settings.return_value.ai_service.anthropic_api_key = None
            mock_settings.return_value.ai_service.openai_model = "gpt-4"
            mock_settings.return_value.ai_service.openai_max_tokens = 2000
            
            with patch('openai.AsyncOpenAI') as mock_openai:
                # Mock the OpenAI response
                mock_client = AsyncMock()
                mock_openai.return_value = mock_client
                mock_response = Mock()
                mock_response.choices = [Mock(message=Mock(content=json.dumps(creative_response)))]
                mock_client.chat.completions.create.return_value = mock_response
                
                factory = UniversalPersonaFactory()
                creative = await factory.create_creative_persona(
                    "Photography", "Documentary and emotional"
                )
                
                # Verify creative persona
                assert isinstance(creative, CreativePersona)
                assert creative.name == "Visual Storyteller Luna"
                assert creative.archetype == PersonaArchetype.INNOVATOR
                assert creative.creative_philosophy == "Authenticity over perfection"
                assert "Annie Leibovitz" in creative.artistic_influences
    
    @pytest.mark.asyncio
    async def test_generate_multi_platform_personas(self, mock_content_analysis):
        """Test generating personas for multiple platforms"""
        base_persona_response = {
            "name": "Coach Alex",
            "archetype": "motivator",
            "style": "inspirational",
            "traits": ["energetic", "supportive", "knowledgeable"],
            "values": ["health", "growth", "community"],
            "voice_characteristics": {"tone": "upbeat"},
            "content_preferences": {"formats": ["video", "text"]},
            "business_focus": "Fitness transformation",
            "expertise_areas": ["fitness", "nutrition"],
            "unique_perspective": "Holistic approach",
            "engagement_approach": "Community building"
        }
        
        platform_responses = {
            Platform.INSTAGRAM: {
                "platform_voice": {"emoji_usage": "moderate"},
                "content_strategy": {"primary_content_types": ["reels", "stories"]},
                "engagement_tactics": ["Reply to comments", "Host Q&As"],
                "platform_specific_traits": ["Visual storyteller"]
            },
            Platform.LINKEDIN: {
                "platform_voice": {"emoji_usage": "minimal"},
                "content_strategy": {"primary_content_types": ["articles", "posts"]},
                "engagement_tactics": ["Share industry insights", "Network actively"],
                "platform_specific_traits": ["Thought leader"]
            }
        }
        
        with patch('backend.config.settings.get_settings') as mock_settings:
            mock_settings.return_value.ai_service.openai_api_key.get_secret_value.return_value = "test-key"
            mock_settings.return_value.ai_service.anthropic_api_key = None
            mock_settings.return_value.ai_service.openai_model = "gpt-4"
            mock_settings.return_value.ai_service.openai_max_tokens = 2000
            
            with patch('openai.AsyncOpenAI') as mock_openai:
                # Mock the OpenAI client
                mock_client = AsyncMock()
                mock_openai.return_value = mock_client
                
                # Set up response sequence
                responses = [
                    Mock(choices=[Mock(message=Mock(content=json.dumps(base_persona_response)))]),
                    Mock(choices=[Mock(message=Mock(content=json.dumps(platform_responses[Platform.INSTAGRAM])))]),
                    Mock(choices=[Mock(message=Mock(content=json.dumps(platform_responses[Platform.LINKEDIN])))])
                ]
                mock_client.chat.completions.create.side_effect = responses
                
                factory = UniversalPersonaFactory()
                platforms = [Platform.INSTAGRAM, Platform.LINKEDIN]
                result = await factory.generate_multi_platform_personas(
                    mock_content_analysis, platforms
                )
                
                # Verify results
                assert len(result) == 2
                assert Platform.INSTAGRAM in result
                assert Platform.LINKEDIN in result
                assert result[Platform.INSTAGRAM].platform_voice["emoji_usage"] == "moderate"
                assert result[Platform.LINKEDIN].platform_voice["emoji_usage"] == "minimal"
    
    def test_validate_archetype(self):
        """Test archetype validation"""
        with patch('backend.config.settings.get_settings'):
            with patch('openai.AsyncOpenAI'):
                factory = UniversalPersonaFactory()
                
                # Test valid archetypes
                assert factory._validate_archetype("educator") == PersonaArchetype.EDUCATOR
                assert factory._validate_archetype("MOTIVATOR") == PersonaArchetype.MOTIVATOR
                assert factory._validate_archetype("problem_solver") == PersonaArchetype.PROBLEM_SOLVER
                
                # Test invalid archetype defaults to EDUCATOR
                assert factory._validate_archetype("invalid") == PersonaArchetype.EDUCATOR
    
    def test_validate_style(self):
        """Test style validation"""
        with patch('backend.config.settings.get_settings'):
            with patch('openai.AsyncOpenAI'):
                factory = UniversalPersonaFactory()
                
                # Test valid styles
                assert factory._validate_style("professional") == PersonaStyle.PROFESSIONAL
                assert factory._validate_style("CREATIVE") == PersonaStyle.CREATIVE
                assert factory._validate_style("inspirational") == PersonaStyle.INSPIRATIONAL
                
                # Test invalid style defaults to PROFESSIONAL
                assert factory._validate_style("invalid") == PersonaStyle.PROFESSIONAL
    
    def test_persona_to_dict_conversion(self):
        """Test persona objects convert to dictionary correctly"""
        # Test BusinessPersona
        business_persona = BusinessPersona(
            name="Test Business",
            archetype=PersonaArchetype.AUTHORITY,
            style=PersonaStyle.PROFESSIONAL,
            traits=["trait1", "trait2"],
            values=["value1", "value2"],
            voice_characteristics={"tone": "professional"},
            content_preferences={"format": "article"},
            business_focus="Consulting",
            expertise_areas=["strategy", "operations"],
            unique_perspective="Data-driven insights",
            engagement_approach="Thought leadership"
        )
        
        persona_dict = business_persona.to_dict()
        assert persona_dict["name"] == "Test Business"
        assert persona_dict["archetype"] == "authority"
        assert persona_dict["business_focus"] == "Consulting"
        assert "created_at" in persona_dict
        
        # Test PlatformPersona
        platform_persona = PlatformPersona(
            base_persona=business_persona,
            platform=Platform.LINKEDIN,
            platform_voice={"tone": "professional"},
            content_strategy={"types": ["articles"]},
            engagement_tactics=["networking"],
            platform_specific_traits=["thought leader"]
        )
        
        platform_dict = platform_persona.to_dict()
        assert platform_dict["platform"] == "linkedin"
        assert platform_dict["base_persona"]["name"] == "Test Business"
        assert platform_dict["engagement_tactics"] == ["networking"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])