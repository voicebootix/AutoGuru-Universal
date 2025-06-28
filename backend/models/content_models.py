"""
Content models for AutoGuru Universal content analyzer.

This module provides Pydantic models for content analysis data structures
that work universally across any business niche. All models include
comprehensive validation and are designed to be AI-driven without
hardcoded business logic.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Annotated
from enum import Enum
from pydantic import BaseModel, Field, validator


class BusinessNicheType(str, Enum):
    """Supported business niches - dynamically determined by AI"""
    EDUCATION = "education"
    BUSINESS_CONSULTING = "business_consulting"
    FITNESS_WELLNESS = "fitness_wellness"
    CREATIVE = "creative"
    ECOMMERCE = "ecommerce"
    LOCAL_SERVICE = "local_service"
    TECHNOLOGY = "technology"
    NON_PROFIT = "non_profit"
    OTHER = "other"


class Platform(str, Enum):
    """Social media platforms supported"""
    INSTAGRAM = "instagram"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    FACEBOOK = "facebook"
    TIKTOK = "tiktok"
    YOUTUBE = "youtube"
    PINTEREST = "pinterest"
    REDDIT = "reddit"


class ContentFormat(str, Enum):
    """Content format types"""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    CAROUSEL = "carousel"
    STORY = "story"
    LIVE = "live"
    POLL = "poll"
    ARTICLE = "article"


class ToneType(str, Enum):
    """Brand voice tone types"""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"
    INSPIRATIONAL = "inspirational"
    EDUCATIONAL = "educational"
    HUMOROUS = "humorous"
    CONVERSATIONAL = "conversational"


class BusinessNiche(BaseModel):
    """
    Business niche detection model.
    
    Represents the detected business type with confidence scoring
    and reasoning from AI analysis.
    """
    niche_type: BusinessNicheType = Field(
        ...,
        description="The detected business niche category"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of the niche detection (0.0-1.0)"
    )
    sub_niches: List[str] = Field(
        default_factory=list,
        description="More specific sub-categories within the niche"
    )
    reasoning: str = Field(
        ...,
        description="AI reasoning for the niche classification"
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Key industry-specific keywords detected"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "niche_type": "fitness_wellness",
                "confidence_score": 0.92,
                "sub_niches": ["personal_training", "nutrition_coaching"],
                "reasoning": "Content focuses on workout routines and meal planning",
                "keywords": ["fitness", "health", "workout", "nutrition"]
            }
        }


class Demographics(BaseModel):
    """Demographic information for audience profiling"""
    age_range: str = Field(
        ...,
        description="Target age range (e.g., '25-35')"
    )
    gender_distribution: Dict[str, float] = Field(
        ...,
        description="Gender distribution percentages"
    )
    location_focus: List[str] = Field(
        default_factory=list,
        description="Geographic locations (countries, regions, cities)"
    )
    income_level: str = Field(
        ...,
        description="Income bracket (low/medium/high/premium)"
    )
    education_level: str = Field(
        ...,
        description="Primary education level of audience"
    )
    occupation_categories: List[str] = Field(
        default_factory=list,
        description="Common occupations in target audience"
    )
    
    @validator('gender_distribution')
    def validate_gender_distribution(cls, v):
        """Ensure gender distribution adds up to 100%"""
        total = sum(v.values())
        if not (99.0 <= total <= 101.0):  # Allow small floating point errors
            raise ValueError(f"Gender distribution must sum to 100%, got {total}%")
        return v


class Psychographics(BaseModel):
    """Psychographic information for audience profiling"""
    values: List[str] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="Core values of the target audience"
    )
    lifestyle: str = Field(
        ...,
        description="Lifestyle description"
    )
    personality_traits: List[str] = Field(
        ...,
        min_items=1,
        max_items=8,
        description="Key personality traits"
    )
    attitudes: List[str] = Field(
        default_factory=list,
        description="Important attitudes and beliefs"
    )
    motivations: List[str] = Field(
        default_factory=list,
        description="Primary motivations and drivers"
    )
    challenges: List[str] = Field(
        default_factory=list,
        description="Common challenges faced"
    )


class ContentPreferences(BaseModel):
    """Content consumption preferences"""
    preferred_formats: List[ContentFormat] = Field(
        ...,
        min_items=1,
        description="Preferred content formats"
    )
    optimal_length: Dict[str, str] = Field(
        ...,
        description="Optimal content length by format"
    )
    best_posting_times: List[str] = Field(
        default_factory=list,
        description="Optimal posting times"
    )
    engagement_triggers: List[str] = Field(
        default_factory=list,
        description="Elements that trigger engagement"
    )
    content_themes: List[str] = Field(
        default_factory=list,
        description="Preferred content themes"
    )


class AudienceProfile(BaseModel):
    """
    Comprehensive target audience profile.
    
    AI-generated audience analysis that works universally
    across any business niche.
    """
    demographics: Demographics = Field(
        ...,
        description="Demographic characteristics"
    )
    psychographics: Psychographics = Field(
        ...,
        description="Psychographic characteristics"
    )
    pain_points: List[str] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="Primary pain points and problems"
    )
    interests: List[str] = Field(
        ...,
        min_items=1,
        max_items=15,
        description="Key interests and hobbies"
    )
    preferred_platforms: List[Platform] = Field(
        ...,
        min_items=1,
        description="Preferred social media platforms"
    )
    content_preferences: ContentPreferences = Field(
        ...,
        description="Content consumption preferences"
    )
    buyer_journey_stage: str = Field(
        ...,
        description="Typical stage in buyer journey"
    )
    influence_factors: List[str] = Field(
        default_factory=list,
        description="Key factors that influence decisions"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "demographics": {
                    "age_range": "25-35",
                    "gender_distribution": {"male": 40.0, "female": 55.0, "other": 5.0},
                    "location_focus": ["United States", "Canada"],
                    "income_level": "medium",
                    "education_level": "college",
                    "occupation_categories": ["professionals", "entrepreneurs"]
                },
                "psychographics": {
                    "values": ["health", "success", "balance"],
                    "lifestyle": "Active and career-focused",
                    "personality_traits": ["ambitious", "health-conscious"],
                    "attitudes": ["growth-minded", "tech-savvy"],
                    "motivations": ["self-improvement", "efficiency"],
                    "challenges": ["time management", "work-life balance"]
                },
                "pain_points": ["lack of time", "information overload"],
                "interests": ["fitness", "productivity", "technology"],
                "preferred_platforms": ["instagram", "linkedin"],
                "content_preferences": {
                    "preferred_formats": ["video", "carousel"],
                    "optimal_length": {"video": "30-60 seconds", "carousel": "5-7 slides"},
                    "best_posting_times": ["7-9 AM", "12-1 PM", "5-7 PM"],
                    "engagement_triggers": ["questions", "polls", "behind-the-scenes"],
                    "content_themes": ["tips", "transformation", "motivation"]
                },
                "buyer_journey_stage": "consideration",
                "influence_factors": ["social proof", "expert endorsement"]
            }
        }


class CommunicationStyle(BaseModel):
    """Communication style preferences"""
    vocabulary_level: str = Field(
        ...,
        description="Vocabulary complexity (simple/moderate/technical)"
    )
    sentence_structure: str = Field(
        ...,
        description="Sentence structure preference (short/varied/complex)"
    )
    engagement_style: str = Field(
        ...,
        description="Engagement approach (direct/storytelling/educational)"
    )
    formality_level: str = Field(
        ...,
        description="Formality level (very formal/formal/casual/very casual)"
    )
    emoji_usage: str = Field(
        ...,
        description="Emoji usage level (none/minimal/moderate/heavy)"
    )


class BrandVoice(BaseModel):
    """
    Brand voice and communication style model.
    
    Defines how the brand communicates across all content,
    ensuring consistency while adapting to different platforms.
    """
    tone: ToneType = Field(
        ...,
        description="Primary tone of communication"
    )
    secondary_tones: List[ToneType] = Field(
        default_factory=list,
        max_items=3,
        description="Secondary tones that complement the primary"
    )
    personality_traits: List[str] = Field(
        ...,
        min_items=3,
        max_items=7,
        description="Brand personality traits"
    )
    communication_style: CommunicationStyle = Field(
        ...,
        description="Detailed communication preferences"
    )
    unique_phrases: List[str] = Field(
        default_factory=list,
        description="Signature phrases or expressions"
    )
    storytelling_approach: str = Field(
        ...,
        description="How the brand tells stories"
    )
    humor_style: Optional[str] = Field(
        None,
        description="Type of humor used (if any)"
    )
    cultural_sensitivity: List[str] = Field(
        default_factory=list,
        description="Cultural considerations"
    )
    do_not_use: List[str] = Field(
        default_factory=list,
        description="Words, phrases, or topics to avoid"
    )
    
    @validator('personality_traits')
    def validate_personality_traits(cls, v):
        """Ensure personality traits are unique"""
        if len(v) != len(set(v)):
            raise ValueError("Personality traits must be unique")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "tone": "friendly",
                "secondary_tones": ["educational", "inspirational"],
                "personality_traits": ["approachable", "knowledgeable", "supportive", "authentic"],
                "communication_style": {
                    "vocabulary_level": "simple",
                    "sentence_structure": "varied",
                    "engagement_style": "storytelling",
                    "formality_level": "casual",
                    "emoji_usage": "moderate"
                },
                "unique_phrases": ["Let's grow together", "Your success is our mission"],
                "storytelling_approach": "Personal anecdotes with practical lessons",
                "humor_style": "Light and relatable",
                "cultural_sensitivity": ["inclusive language", "diverse representation"],
                "do_not_use": ["jargon", "negative language", "pressure tactics"]
            }
        }


class ContentTheme(BaseModel):
    """
    Content theme and topic model.
    
    Represents key themes and topics identified in content
    or recommended for content strategy.
    """
    theme_name: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Name of the content theme"
    )
    description: str = Field(
        ...,
        description="Detailed description of the theme"
    )
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score for the business niche (0.0-1.0)"
    )
    subtopics: List[str] = Field(
        default_factory=list,
        max_items=10,
        description="Related subtopics within this theme"
    )
    content_pillars: List[str] = Field(
        default_factory=list,
        description="Content pillars this theme supports"
    )
    keywords: List[str] = Field(
        ...,
        min_items=3,
        max_items=20,
        description="SEO and hashtag keywords"
    )
    audience_interest_level: str = Field(
        ...,
        description="Audience interest level (low/medium/high/viral)"
    )
    competitive_advantage: Optional[str] = Field(
        None,
        description="Unique angle or competitive advantage"
    )
    seasonal_relevance: Optional[Dict[str, float]] = Field(
        None,
        description="Seasonal relevance scores by month/season"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "theme_name": "Productivity Hacks",
                "description": "Time-saving tips and efficiency strategies for busy professionals",
                "relevance_score": 0.85,
                "subtopics": ["morning routines", "tool reviews", "workflow optimization"],
                "content_pillars": ["education", "inspiration", "practical tips"],
                "keywords": ["productivity", "timemanagement", "efficiency", "lifehacks"],
                "audience_interest_level": "high",
                "competitive_advantage": "Real-world tested strategies with measurable results",
                "seasonal_relevance": {"Q1": 0.9, "Q2": 0.7, "Q3": 0.6, "Q4": 0.8}
            }
        }


class ViralFactors(BaseModel):
    """Factors contributing to viral potential"""
    emotional_triggers: List[str] = Field(
        ...,
        description="Emotions the content triggers"
    )
    shareability_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How likely users are to share (0.0-1.0)"
    )
    trend_alignment: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Alignment with current trends (0.0-1.0)"
    )
    uniqueness_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Content uniqueness and originality (0.0-1.0)"
    )
    timing_relevance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Timing and relevance score (0.0-1.0)"
    )


class ViralScore(BaseModel):
    """
    Viral potential score for a specific platform.
    
    AI-calculated viral potential with detailed factors
    and improvement suggestions.
    """
    platform: Platform = Field(
        ...,
        description="The social media platform"
    )
    overall_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall viral potential score (0.0-1.0)"
    )
    viral_factors: ViralFactors = Field(
        ...,
        description="Detailed viral factors analysis"
    )
    platform_specific_factors: Dict[str, Any] = Field(
        ...,
        description="Platform-specific viral factors"
    )
    improvement_suggestions: List[str] = Field(
        ...,
        min_items=1,
        max_items=5,
        description="Specific suggestions to increase viral potential"
    )
    best_format: ContentFormat = Field(
        ...,
        description="Optimal content format for this platform"
    )
    optimal_length: str = Field(
        ...,
        description="Optimal content length for virality"
    )
    hashtag_recommendations: List[str] = Field(
        default_factory=list,
        max_items=30,
        description="Recommended hashtags for visibility"
    )
    predicted_reach: str = Field(
        ...,
        description="Predicted reach category (low/medium/high/viral)"
    )
    
    @validator('hashtag_recommendations')
    def validate_hashtags(cls, v):
        """Ensure hashtags are properly formatted"""
        return [tag if tag.startswith('#') else f'#{tag}' for tag in v]
    
    class Config:
        schema_extra = {
            "example": {
                "platform": "instagram",
                "overall_score": 0.78,
                "viral_factors": {
                    "emotional_triggers": ["inspiration", "relatability", "humor"],
                    "shareability_score": 0.82,
                    "trend_alignment": 0.75,
                    "uniqueness_score": 0.69,
                    "timing_relevance": 0.85
                },
                "platform_specific_factors": {
                    "visual_appeal": 0.9,
                    "story_potential": 0.8,
                    "reel_compatibility": 0.85,
                    "carousel_effectiveness": 0.7
                },
                "improvement_suggestions": [
                    "Add trending audio to increase reach",
                    "Include a strong call-to-action",
                    "Use more vibrant colors in visuals"
                ],
                "best_format": "video",
                "optimal_length": "15-30 seconds",
                "hashtag_recommendations": ["#productivity", "#lifehacks", "#motivation"],
                "predicted_reach": "high"
            }
        }


class PlatformContent(BaseModel):
    """
    Platform-optimized content model.
    
    Contains platform-specific content adaptations while
    maintaining brand consistency.
    """
    platform: Platform = Field(
        ...,
        description="Target platform"
    )
    content_text: str = Field(
        ...,
        description="Platform-optimized content text"
    )
    content_format: ContentFormat = Field(
        ...,
        description="Recommended content format"
    )
    media_requirements: Dict[str, Any] = Field(
        ...,
        description="Media specifications (dimensions, duration, etc.)"
    )
    hashtags: List[str] = Field(
        default_factory=list,
        max_items=30,
        description="Platform-optimized hashtags"
    )
    mentions: List[str] = Field(
        default_factory=list,
        description="Accounts to mention or tag"
    )
    call_to_action: str = Field(
        ...,
        description="Platform-specific call to action"
    )
    posting_time: Optional[datetime] = Field(
        None,
        description="Optimal posting time"
    )
    platform_features: List[str] = Field(
        default_factory=list,
        description="Platform features to utilize (polls, stickers, etc.)"
    )
    character_count: int = Field(
        ...,
        ge=0,
        description="Character count of the content"
    )
    accessibility_text: Optional[str] = Field(
        None,
        description="Alt text or accessibility description"
    )
    
    @validator('character_count')
    def validate_character_count(cls, v, values):
        """Validate character count against platform limits"""
        platform = values.get('platform')
        limits = {
            Platform.TWITTER: 280,
            Platform.INSTAGRAM: 2200,
            Platform.FACEBOOK: 63206,
            Platform.LINKEDIN: 3000,
            Platform.TIKTOK: 2200,
            Platform.PINTEREST: 500
        }
        
        if platform in limits and v > limits[platform]:
            raise ValueError(f"Character count {v} exceeds {platform} limit of {limits[platform]}")
        return v
    
    @validator('hashtags')
    def validate_hashtags(cls, v):
        """Ensure hashtags are properly formatted"""
        return [tag if tag.startswith('#') else f'#{tag}' for tag in v]
    
    class Config:
        schema_extra = {
            "example": {
                "platform": "instagram",
                "content_text": "🚀 Transform your morning routine with these 5 productivity hacks!",
                "content_format": "carousel",
                "media_requirements": {
                    "dimensions": "1080x1080",
                    "slides": 7,
                    "format": "jpg",
                    "color_scheme": "brand_colors"
                },
                "hashtags": ["#productivity", "#morningroutine", "#lifehacks"],
                "mentions": ["@productivityguru"],
                "call_to_action": "Save this post and try one hack tomorrow! ⬇️",
                "posting_time": "2024-01-15T07:30:00Z",
                "platform_features": ["location_tag", "product_tags"],
                "character_count": 125,
                "accessibility_text": "Carousel showing 5 morning productivity tips"
            }
        }


class ContentMetadata(BaseModel):
    """Metadata for content analysis"""
    analysis_id: str = Field(
        ...,
        description="Unique identifier for this analysis"
    )
    analyzed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of analysis"
    )
    content_source: str = Field(
        ...,
        description="Source of the content (user_input, file, url, etc.)"
    )
    content_length: int = Field(
        ...,
        ge=0,
        description="Length of analyzed content in characters"
    )
    language: str = Field(
        default="en",
        description="Detected language code"
    )
    llm_provider: str = Field(
        ...,
        description="LLM provider used for analysis"
    )
    llm_model: str = Field(
        ...,
        description="Specific model used"
    )
    processing_time_ms: int = Field(
        ...,
        ge=0,
        description="Processing time in milliseconds"
    )
    confidence_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Confidence scores for various analysis components"
    )


class ContentAnalysis(BaseModel):
    """
    Main content analysis result model.
    
    Comprehensive analysis output that combines all aspects
    of content analysis for universal business automation.
    """
    business_niche: BusinessNiche = Field(
        ...,
        description="Detected business niche with confidence"
    )
    audience_profile: AudienceProfile = Field(
        ...,
        description="Comprehensive target audience analysis"
    )
    brand_voice: BrandVoice = Field(
        ...,
        description="Brand voice and communication style"
    )
    content_themes: List[ContentTheme] = Field(
        ...,
        min_items=1,
        max_items=10,
        description="Key content themes identified or recommended"
    )
    viral_scores: List[ViralScore] = Field(
        ...,
        min_items=1,
        description="Viral potential scores per platform"
    )
    platform_content: List[PlatformContent] = Field(
        default_factory=list,
        description="Platform-specific content adaptations"
    )
    overall_recommendations: List[str] = Field(
        ...,
        min_items=3,
        max_items=10,
        description="Strategic recommendations for content success"
    )
    competitive_insights: Optional[Dict[str, Any]] = Field(
        None,
        description="Competitive landscape insights"
    )
    performance_predictions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Predicted performance metrics"
    )
    metadata: ContentMetadata = Field(
        ...,
        description="Analysis metadata and metrics"
    )
    
    @validator('viral_scores')
    def validate_unique_platforms(cls, v):
        """Ensure each platform appears only once in viral scores"""
        platforms = [score.platform for score in v]
        if len(platforms) != len(set(platforms)):
            raise ValueError("Each platform should appear only once in viral scores")
        return v
    
    @validator('content_themes')
    def sort_themes_by_relevance(cls, v):
        """Sort themes by relevance score in descending order"""
        return sorted(v, key=lambda x: x.relevance_score, reverse=True)
    
    def get_top_platforms(self, n: int = 3) -> List[Platform]:
        """Get top N platforms by viral score"""
        sorted_scores = sorted(
            self.viral_scores,
            key=lambda x: x.overall_score,
            reverse=True
        )
        return [score.platform for score in sorted_scores[:n]]
    
    def get_primary_theme(self) -> ContentTheme:
        """Get the most relevant content theme"""
        return self.content_themes[0] if self.content_themes else None
    
    def get_platform_content(self, platform: Platform) -> Optional[PlatformContent]:
        """Get platform-specific content for a given platform"""
        for content in self.platform_content:
            if content.platform == platform:
                return content
        return None
    
    class Config:
        schema_extra = {
            "example": {
                "business_niche": {
                    "niche_type": "fitness_wellness",
                    "confidence_score": 0.92,
                    "sub_niches": ["personal_training", "nutrition"],
                    "reasoning": "Content focuses on fitness and nutrition",
                    "keywords": ["fitness", "health", "workout"]
                },
                "audience_profile": {
                    "demographics": {
                        "age_range": "25-45",
                        "gender_distribution": {"male": 35.0, "female": 60.0, "other": 5.0},
                        "location_focus": ["United States"],
                        "income_level": "medium",
                        "education_level": "college",
                        "occupation_categories": ["professionals"]
                    },
                    "psychographics": {
                        "values": ["health", "achievement"],
                        "lifestyle": "Active and health-conscious",
                        "personality_traits": ["motivated", "disciplined"],
                        "attitudes": ["growth-minded"],
                        "motivations": ["self-improvement"],
                        "challenges": ["time constraints"]
                    },
                    "pain_points": ["lack of time", "motivation"],
                    "interests": ["fitness", "nutrition", "wellness"],
                    "preferred_platforms": ["instagram", "youtube"],
                    "content_preferences": {
                        "preferred_formats": ["video", "image"],
                        "optimal_length": {"video": "60 seconds"},
                        "best_posting_times": ["6-8 AM", "5-7 PM"],
                        "engagement_triggers": ["challenges", "transformations"],
                        "content_themes": ["workouts", "meal prep"]
                    },
                    "buyer_journey_stage": "consideration",
                    "influence_factors": ["results", "testimonials"]
                },
                "brand_voice": {
                    "tone": "inspirational",
                    "secondary_tones": ["friendly", "educational"],
                    "personality_traits": ["supportive", "knowledgeable", "motivating"],
                    "communication_style": {
                        "vocabulary_level": "simple",
                        "sentence_structure": "short",
                        "engagement_style": "direct",
                        "formality_level": "casual",
                        "emoji_usage": "moderate"
                    },
                    "unique_phrases": ["You've got this!", "Every rep counts"],
                    "storytelling_approach": "Personal transformation stories",
                    "humor_style": "Light and encouraging",
                    "cultural_sensitivity": ["body positivity", "inclusive fitness"],
                    "do_not_use": ["shame", "extreme", "quick fix"]
                },
                "content_themes": [
                    {
                        "theme_name": "Home Workouts",
                        "description": "Equipment-free exercises for busy people",
                        "relevance_score": 0.95,
                        "subtopics": ["HIIT", "bodyweight", "stretching"],
                        "content_pillars": ["education", "motivation"],
                        "keywords": ["homeworkout", "noequipment", "fitness"],
                        "audience_interest_level": "high",
                        "competitive_advantage": "Quick, effective routines"
                    }
                ],
                "viral_scores": [
                    {
                        "platform": "instagram",
                        "overall_score": 0.82,
                        "viral_factors": {
                            "emotional_triggers": ["inspiration", "achievement"],
                            "shareability_score": 0.78,
                            "trend_alignment": 0.85,
                            "uniqueness_score": 0.72,
                            "timing_relevance": 0.88
                        },
                        "platform_specific_factors": {
                            "visual_appeal": 0.9,
                            "reel_potential": 0.85
                        },
                        "improvement_suggestions": ["Add trending audio"],
                        "best_format": "video",
                        "optimal_length": "30 seconds",
                        "hashtag_recommendations": ["#homeworkout", "#fitness"],
                        "predicted_reach": "high"
                    }
                ],
                "platform_content": [],
                "overall_recommendations": [
                    "Focus on short-form video content",
                    "Share transformation stories",
                    "Create workout challenges"
                ],
                "competitive_insights": {
                    "market_saturation": "medium",
                    "differentiation_opportunities": ["personalization", "community"]
                },
                "performance_predictions": {
                    "engagement_rate": "5-7%",
                    "growth_potential": "high"
                },
                "metadata": {
                    "analysis_id": "ana_123456",
                    "analyzed_at": "2024-01-15T10:30:00Z",
                    "content_source": "user_input",
                    "content_length": 500,
                    "language": "en",
                    "llm_provider": "openai",
                    "llm_model": "gpt-4",
                    "processing_time_ms": 2500,
                    "confidence_metrics": {
                        "niche_detection": 0.92,
                        "audience_analysis": 0.88
                    }
                }
            }
        }


# Export all models
__all__ = [
    'BusinessNicheType',
    'Platform',
    'ContentFormat',
    'ToneType',
    'BusinessNiche',
    'Demographics',
    'Psychographics',
    'ContentPreferences',
    'AudienceProfile',
    'CommunicationStyle',
    'BrandVoice',
    'ContentTheme',
    'ViralFactors',
    'ViralScore',
    'PlatformContent',
    'ContentMetadata',
    'ContentAnalysis'
]