"""
Universal Persona Factory for AutoGuru Universal

This module provides AI-powered persona generation that works for ANY business niche
automatically. It creates authentic, platform-optimized personas based on content
analysis without any hardcoded business logic.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import openai
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.models.content_models import (
    BusinessNicheType,
    Platform,
    ContentAnalysis,
    BrandVoice,
    AudienceProfile,
    ToneType
)
from backend.config.settings import get_settings

# Configure logging
logger = logging.getLogger(__name__)


class PersonaArchetype(str, Enum):
    """Universal persona archetypes that work across all businesses"""
    EDUCATOR = "educator"
    AUTHORITY = "authority"
    FRIEND = "friend"
    INNOVATOR = "innovator"
    MOTIVATOR = "motivator"
    CURATOR = "curator"
    STORYTELLER = "storyteller"
    PROBLEM_SOLVER = "problem_solver"


class PersonaStyle(str, Enum):
    """Communication styles for personas"""
    PROFESSIONAL = "professional"
    CONVERSATIONAL = "conversational"
    INSPIRATIONAL = "inspirational"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    SUPPORTIVE = "supportive"
    ENTERTAINING = "entertaining"
    EDUCATIONAL = "educational"


class BasePersona:
    """Base class for all persona types"""
    
    def __init__(
        self,
        name: str,
        archetype: PersonaArchetype,
        style: PersonaStyle,
        traits: List[str],
        values: List[str],
        voice_characteristics: Dict[str, Any],
        content_preferences: Dict[str, Any]
    ):
        self.name = name
        self.archetype = archetype
        self.style = style
        self.traits = traits
        self.values = values
        self.voice_characteristics = voice_characteristics
        self.content_preferences = content_preferences
        self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert persona to dictionary"""
        return {
            "name": self.name,
            "archetype": self.archetype.value,
            "style": self.style.value,
            "traits": self.traits,
            "values": self.values,
            "voice_characteristics": self.voice_characteristics,
            "content_preferences": self.content_preferences,
            "created_at": self.created_at.isoformat()
        }


class BusinessPersona(BasePersona):
    """Universal business persona that adapts to any niche"""
    
    def __init__(
        self,
        name: str,
        archetype: PersonaArchetype,
        style: PersonaStyle,
        traits: List[str],
        values: List[str],
        voice_characteristics: Dict[str, Any],
        content_preferences: Dict[str, Any],
        business_focus: str,
        expertise_areas: List[str],
        unique_perspective: str,
        engagement_approach: str
    ):
        super().__init__(name, archetype, style, traits, values, voice_characteristics, content_preferences)
        self.business_focus = business_focus
        self.expertise_areas = expertise_areas
        self.unique_perspective = unique_perspective
        self.engagement_approach = engagement_approach
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert business persona to dictionary"""
        base_dict = super().to_dict()
        base_dict.update({
            "business_focus": self.business_focus,
            "expertise_areas": self.expertise_areas,
            "unique_perspective": self.unique_perspective,
            "engagement_approach": self.engagement_approach
        })
        return base_dict


class EducatorPersona(BusinessPersona):
    """Educator persona for knowledge-sharing businesses"""
    
    def __init__(
        self,
        name: str,
        style: PersonaStyle,
        traits: List[str],
        values: List[str],
        voice_characteristics: Dict[str, Any],
        content_preferences: Dict[str, Any],
        business_focus: str,
        expertise_areas: List[str],
        unique_perspective: str,
        engagement_approach: str,
        teaching_philosophy: str,
        learning_frameworks: List[str]
    ):
        super().__init__(
            name=name,
            archetype=PersonaArchetype.EDUCATOR,
            style=style,
            traits=traits,
            values=values,
            voice_characteristics=voice_characteristics,
            content_preferences=content_preferences,
            business_focus=business_focus,
            expertise_areas=expertise_areas,
            unique_perspective=unique_perspective,
            engagement_approach=engagement_approach
        )
        self.teaching_philosophy = teaching_philosophy
        self.learning_frameworks = learning_frameworks
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert educator persona to dictionary"""
        base_dict = super().to_dict()
        base_dict.update({
            "teaching_philosophy": self.teaching_philosophy,
            "learning_frameworks": self.learning_frameworks
        })
        return base_dict


class WellnessPersona(BusinessPersona):
    """Wellness persona for health and fitness businesses"""
    
    def __init__(
        self,
        name: str,
        style: PersonaStyle,
        traits: List[str],
        values: List[str],
        voice_characteristics: Dict[str, Any],
        content_preferences: Dict[str, Any],
        business_focus: str,
        expertise_areas: List[str],
        unique_perspective: str,
        engagement_approach: str,
        wellness_philosophy: str,
        transformation_approach: str
    ):
        super().__init__(
            name=name,
            archetype=PersonaArchetype.MOTIVATOR,
            style=style,
            traits=traits,
            values=values,
            voice_characteristics=voice_characteristics,
            content_preferences=content_preferences,
            business_focus=business_focus,
            expertise_areas=expertise_areas,
            unique_perspective=unique_perspective,
            engagement_approach=engagement_approach
        )
        self.wellness_philosophy = wellness_philosophy
        self.transformation_approach = transformation_approach
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert wellness persona to dictionary"""
        base_dict = super().to_dict()
        base_dict.update({
            "wellness_philosophy": self.wellness_philosophy,
            "transformation_approach": self.transformation_approach
        })
        return base_dict


class CreativePersona(BusinessPersona):
    """Creative persona for artistic and design businesses"""
    
    def __init__(
        self,
        name: str,
        style: PersonaStyle,
        traits: List[str],
        values: List[str],
        voice_characteristics: Dict[str, Any],
        content_preferences: Dict[str, Any],
        business_focus: str,
        expertise_areas: List[str],
        unique_perspective: str,
        engagement_approach: str,
        creative_philosophy: str,
        artistic_influences: List[str]
    ):
        super().__init__(
            name=name,
            archetype=PersonaArchetype.INNOVATOR,
            style=style,
            traits=traits,
            values=values,
            voice_characteristics=voice_characteristics,
            content_preferences=content_preferences,
            business_focus=business_focus,
            expertise_areas=expertise_areas,
            unique_perspective=unique_perspective,
            engagement_approach=engagement_approach
        )
        self.creative_philosophy = creative_philosophy
        self.artistic_influences = artistic_influences
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert creative persona to dictionary"""
        base_dict = super().to_dict()
        base_dict.update({
            "creative_philosophy": self.creative_philosophy,
            "artistic_influences": self.artistic_influences
        })
        return base_dict


class PlatformPersona:
    """Platform-specific persona adaptation"""
    
    def __init__(
        self,
        base_persona: BusinessPersona,
        platform: Platform,
        platform_voice: Dict[str, Any],
        content_strategy: Dict[str, Any],
        engagement_tactics: List[str],
        platform_specific_traits: List[str]
    ):
        self.base_persona = base_persona
        self.platform = platform
        self.platform_voice = platform_voice
        self.content_strategy = content_strategy
        self.engagement_tactics = engagement_tactics
        self.platform_specific_traits = platform_specific_traits
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert platform persona to dictionary"""
        return {
            "base_persona": self.base_persona.to_dict(),
            "platform": self.platform.value,
            "platform_voice": self.platform_voice,
            "content_strategy": self.content_strategy,
            "engagement_tactics": self.engagement_tactics,
            "platform_specific_traits": self.platform_specific_traits
        }


class UniversalPersonaFactory:
    """
    AI-powered persona factory that generates authentic personas for ANY business niche.
    No hardcoded business logic - everything is determined by AI analysis.
    """
    
    def __init__(self):
        """Initialize the UniversalPersonaFactory with AI clients"""
        settings = get_settings()
        
        self.openai_client = None
        self.anthropic_client = None
        
        # Initialize AI clients
        try:
            if settings.ai_service.openai_api_key:
                self.openai_client = openai.AsyncOpenAI(
                    api_key=settings.ai_service.openai_api_key.get_secret_value()
                )
                logger.info("OpenAI client initialized for persona generation")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
        
        try:
            if settings.ai_service.anthropic_api_key:
                self.anthropic_client = anthropic.AsyncAnthropic(
                    api_key=settings.ai_service.anthropic_api_key.get_secret_value()
                )
                logger.info("Anthropic client initialized for persona generation")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {str(e)}")
        
        if not self.openai_client and not self.anthropic_client:
            raise ValueError("At least one AI service (OpenAI or Anthropic) must be configured")
        
        self.default_llm = settings.ai_service.openai_api_key and "openai" or "anthropic"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_persona(self, content_analysis: ContentAnalysis) -> BusinessPersona:
        """
        Generate a universal business persona based on content analysis.
        
        Args:
            content_analysis: The analyzed content data
            
        Returns:
            BusinessPersona: AI-generated persona that works universally
        """
        try:
            # Extract key information from content analysis
            niche = content_analysis.business_niche
            audience = content_analysis.audience_profile
            brand_voice = content_analysis.brand_voice
            themes = content_analysis.content_themes
            
            prompt = f"""
            Generate a comprehensive business persona based on this content analysis:
            
            Business Niche: {niche.niche_type.value}
            Sub-niches: {', '.join(niche.sub_niches)}
            
            Target Audience:
            - Demographics: {json.dumps(audience.demographics.dict())}
            - Psychographics: {json.dumps(audience.psychographics.dict())}
            - Pain Points: {', '.join(audience.pain_points[:5])}
            - Interests: {', '.join(audience.interests[:5])}
            
            Brand Voice:
            - Tone: {brand_voice.tone.value}
            - Personality Traits: {', '.join(brand_voice.personality_traits)}
            - Communication Style: {json.dumps(brand_voice.communication_style.dict())}
            
            Content Themes: {', '.join([theme.theme_name for theme in themes[:5]])}
            
            Create a persona that:
            1. Authentically represents this business
            2. Resonates with the target audience
            3. Maintains consistency across all content
            4. Adapts to different platforms while staying true to core identity
            
            Respond in JSON format:
            {{
                "name": "persona name that reflects the business",
                "archetype": "educator/authority/friend/innovator/motivator/curator/storyteller/problem_solver",
                "style": "professional/conversational/inspirational/analytical/creative/supportive/entertaining/educational",
                "traits": ["trait1", "trait2", "trait3", "trait4", "trait5"],
                "values": ["value1", "value2", "value3", "value4"],
                "voice_characteristics": {{
                    "tone_modifiers": ["modifier1", "modifier2"],
                    "language_patterns": ["pattern1", "pattern2"],
                    "signature_phrases": ["phrase1", "phrase2"],
                    "conversation_starters": ["starter1", "starter2"]
                }},
                "content_preferences": {{
                    "favorite_topics": ["topic1", "topic2", "topic3"],
                    "content_formats": ["format1", "format2"],
                    "storytelling_style": "description",
                    "humor_level": "none/light/moderate/heavy"
                }},
                "business_focus": "main focus area",
                "expertise_areas": ["area1", "area2", "area3"],
                "unique_perspective": "what makes this persona unique",
                "engagement_approach": "how the persona engages with audience"
            }}
            """
            
            response = await self._call_llm(prompt, temperature=0.7)
            persona_data = json.loads(response)
            
            # Validate and map archetype
            archetype = self._validate_archetype(persona_data.get("archetype", "educator"))
            style = self._validate_style(persona_data.get("style", "professional"))
            
            # Create the business persona
            persona = BusinessPersona(
                name=persona_data["name"],
                archetype=archetype,
                style=style,
                traits=persona_data["traits"][:7],  # Limit to 7 traits
                values=persona_data["values"][:5],  # Limit to 5 values
                voice_characteristics=persona_data["voice_characteristics"],
                content_preferences=persona_data["content_preferences"],
                business_focus=persona_data["business_focus"],
                expertise_areas=persona_data["expertise_areas"],
                unique_perspective=persona_data["unique_perspective"],
                engagement_approach=persona_data["engagement_approach"]
            )
            
            logger.info(f"Generated persona: {persona.name} ({persona.archetype.value})")
            return persona
            
        except Exception as e:
            logger.error(f"Failed to generate persona: {str(e)}", exc_info=True)
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def adapt_persona_for_platform(
        self,
        persona: BusinessPersona,
        platform: Platform
    ) -> PlatformPersona:
        """
        Adapt a business persona for a specific social media platform.
        
        Args:
            persona: The base business persona
            platform: The target platform
            
        Returns:
            PlatformPersona: Platform-optimized persona
        """
        try:
            prompt = f"""
            Adapt this business persona for {platform.value}:
            
            Persona: {persona.name}
            Archetype: {persona.archetype.value}
            Style: {persona.style.value}
            Traits: {', '.join(persona.traits)}
            Business Focus: {persona.business_focus}
            Engagement Approach: {persona.engagement_approach}
            
            Platform-specific considerations for {platform.value}:
            - Instagram: Visual storytelling, hashtags, stories, reels
            - Twitter: Concise thoughts, threads, real-time engagement
            - LinkedIn: Professional insights, thought leadership, networking
            - Facebook: Community building, longer posts, groups
            - TikTok: Entertainment, trends, short-form video
            - YouTube: Long-form content, education, entertainment
            - Pinterest: Visual inspiration, tutorials, collections
            - Reddit: Community discussion, authenticity, value-adding
            
            Create platform-specific adaptations in JSON format:
            {{
                "platform_voice": {{
                    "tone_adjustment": "how to adjust tone for this platform",
                    "language_style": "platform-appropriate language",
                    "emoji_usage": "none/minimal/moderate/heavy",
                    "hashtag_strategy": "approach to hashtags",
                    "mention_strategy": "how to use mentions/tags"
                }},
                "content_strategy": {{
                    "primary_content_types": ["type1", "type2"],
                    "posting_frequency": "recommended frequency",
                    "optimal_length": "content length guidelines",
                    "visual_style": "visual approach for platform",
                    "engagement_hooks": ["hook1", "hook2", "hook3"]
                }},
                "engagement_tactics": [
                    "tactic1",
                    "tactic2",
                    "tactic3",
                    "tactic4"
                ],
                "platform_specific_traits": [
                    "trait1",
                    "trait2",
                    "trait3"
                ]
            }}
            """
            
            response = await self._call_llm(prompt, temperature=0.6)
            platform_data = json.loads(response)
            
            platform_persona = PlatformPersona(
                base_persona=persona,
                platform=platform,
                platform_voice=platform_data["platform_voice"],
                content_strategy=platform_data["content_strategy"],
                engagement_tactics=platform_data["engagement_tactics"],
                platform_specific_traits=platform_data["platform_specific_traits"]
            )
            
            logger.info(f"Adapted persona for {platform.value}: {persona.name}")
            return platform_persona
            
        except Exception as e:
            logger.error(f"Failed to adapt persona for {platform.value}: {str(e)}", exc_info=True)
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def create_educator_persona(
        self,
        subject_area: str,
        teaching_style: str
    ) -> EducatorPersona:
        """
        Create an educator persona for knowledge-based businesses.
        
        Args:
            subject_area: The area of expertise
            teaching_style: The approach to teaching
            
        Returns:
            EducatorPersona: AI-generated educator persona
        """
        try:
            prompt = f"""
            Create an educator persona for this subject area and teaching style:
            
            Subject Area: {subject_area}
            Teaching Style: {teaching_style}
            
            Generate a comprehensive educator persona in JSON format:
            {{
                "name": "memorable educator name",
                "style": "professional/conversational/inspirational/analytical/creative/supportive/entertaining/educational",
                "traits": ["trait1", "trait2", "trait3", "trait4", "trait5"],
                "values": ["value1", "value2", "value3", "value4"],
                "voice_characteristics": {{
                    "explanation_style": "how they explain concepts",
                    "example_usage": "how they use examples",
                    "question_style": "how they ask questions",
                    "encouragement_style": "how they encourage learners"
                }},
                "content_preferences": {{
                    "lesson_structure": "preferred lesson format",
                    "visual_aids": "use of visuals",
                    "interaction_level": "student interaction approach",
                    "assessment_style": "how they assess understanding"
                }},
                "business_focus": "main educational focus",
                "expertise_areas": ["area1", "area2", "area3"],
                "unique_perspective": "what makes this educator unique",
                "engagement_approach": "how they engage with students",
                "teaching_philosophy": "core teaching philosophy",
                "learning_frameworks": ["framework1", "framework2", "framework3"]
            }}
            """
            
            response = await self._call_llm(prompt, temperature=0.7)
            educator_data = json.loads(response)
            
            style = self._validate_style(educator_data.get("style", "educational"))
            
            educator = EducatorPersona(
                name=educator_data["name"],
                style=style,
                traits=educator_data["traits"],
                values=educator_data["values"],
                voice_characteristics=educator_data["voice_characteristics"],
                content_preferences=educator_data["content_preferences"],
                business_focus=educator_data["business_focus"],
                expertise_areas=educator_data["expertise_areas"],
                unique_perspective=educator_data["unique_perspective"],
                engagement_approach=educator_data["engagement_approach"],
                teaching_philosophy=educator_data["teaching_philosophy"],
                learning_frameworks=educator_data["learning_frameworks"]
            )
            
            logger.info(f"Created educator persona: {educator.name}")
            return educator
            
        except Exception as e:
            logger.error(f"Failed to create educator persona: {str(e)}", exc_info=True)
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def create_business_persona(
        self,
        industry: str,
        leadership_style: str
    ) -> BusinessPersona:
        """
        Create a business/consulting persona.
        
        Args:
            industry: The industry focus
            leadership_style: The leadership approach
            
        Returns:
            BusinessPersona: AI-generated business persona
        """
        try:
            prompt = f"""
            Create a business/consulting persona for this industry and leadership style:
            
            Industry: {industry}
            Leadership Style: {leadership_style}
            
            Generate a comprehensive business persona in JSON format:
            {{
                "name": "professional persona name",
                "archetype": "authority/innovator/problem_solver/curator",
                "style": "professional/analytical/inspirational/supportive",
                "traits": ["trait1", "trait2", "trait3", "trait4", "trait5"],
                "values": ["value1", "value2", "value3", "value4"],
                "voice_characteristics": {{
                    "expertise_display": "how they show expertise",
                    "advice_style": "how they give advice",
                    "case_study_approach": "how they present case studies",
                    "thought_leadership": "thought leadership style"
                }},
                "content_preferences": {{
                    "content_depth": "surface/medium/deep",
                    "data_usage": "how they use data and statistics",
                    "storytelling_approach": "business storytelling style",
                    "actionable_insights": "how they provide actionable advice"
                }},
                "business_focus": "main business focus",
                "expertise_areas": ["area1", "area2", "area3"],
                "unique_perspective": "unique business perspective",
                "engagement_approach": "professional engagement approach"
            }}
            """
            
            response = await self._call_llm(prompt, temperature=0.6)
            business_data = json.loads(response)
            
            archetype = self._validate_archetype(business_data.get("archetype", "authority"))
            style = self._validate_style(business_data.get("style", "professional"))
            
            business_persona = BusinessPersona(
                name=business_data["name"],
                archetype=archetype,
                style=style,
                traits=business_data["traits"],
                values=business_data["values"],
                voice_characteristics=business_data["voice_characteristics"],
                content_preferences=business_data["content_preferences"],
                business_focus=business_data["business_focus"],
                expertise_areas=business_data["expertise_areas"],
                unique_perspective=business_data["unique_perspective"],
                engagement_approach=business_data["engagement_approach"]
            )
            
            logger.info(f"Created business persona: {business_persona.name}")
            return business_persona
            
        except Exception as e:
            logger.error(f"Failed to create business persona: {str(e)}", exc_info=True)
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def create_wellness_persona(
        self,
        wellness_type: str,
        approach: str
    ) -> WellnessPersona:
        """
        Create a wellness/fitness persona.
        
        Args:
            wellness_type: Type of wellness focus (fitness, nutrition, mental health, etc.)
            approach: The wellness approach
            
        Returns:
            WellnessPersona: AI-generated wellness persona
        """
        try:
            prompt = f"""
            Create a wellness persona for this type and approach:
            
            Wellness Type: {wellness_type}
            Approach: {approach}
            
            Generate a comprehensive wellness persona in JSON format:
            {{
                "name": "inspiring wellness persona name",
                "style": "inspirational/supportive/energetic/educational",
                "traits": ["trait1", "trait2", "trait3", "trait4", "trait5"],
                "values": ["value1", "value2", "value3", "value4"],
                "voice_characteristics": {{
                    "motivation_style": "how they motivate",
                    "empathy_expression": "how they show empathy",
                    "progress_celebration": "how they celebrate progress",
                    "challenge_approach": "how they present challenges"
                }},
                "content_preferences": {{
                    "transformation_stories": "use of success stories",
                    "scientific_backing": "use of research/science",
                    "practical_tips": "actionable advice style",
                    "community_building": "community engagement approach"
                }},
                "business_focus": "main wellness focus",
                "expertise_areas": ["area1", "area2", "area3"],
                "unique_perspective": "unique wellness perspective",
                "engagement_approach": "how they engage with community",
                "wellness_philosophy": "core wellness philosophy",
                "transformation_approach": "approach to client transformation"
            }}
            """
            
            response = await self._call_llm(prompt, temperature=0.7)
            wellness_data = json.loads(response)
            
            style = self._validate_style(wellness_data.get("style", "inspirational"))
            
            wellness = WellnessPersona(
                name=wellness_data["name"],
                style=style,
                traits=wellness_data["traits"],
                values=wellness_data["values"],
                voice_characteristics=wellness_data["voice_characteristics"],
                content_preferences=wellness_data["content_preferences"],
                business_focus=wellness_data["business_focus"],
                expertise_areas=wellness_data["expertise_areas"],
                unique_perspective=wellness_data["unique_perspective"],
                engagement_approach=wellness_data["engagement_approach"],
                wellness_philosophy=wellness_data["wellness_philosophy"],
                transformation_approach=wellness_data["transformation_approach"]
            )
            
            logger.info(f"Created wellness persona: {wellness.name}")
            return wellness
            
        except Exception as e:
            logger.error(f"Failed to create wellness persona: {str(e)}", exc_info=True)
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def create_creative_persona(
        self,
        creative_field: str,
        style: str
    ) -> CreativePersona:
        """
        Create a creative/artistic persona.
        
        Args:
            creative_field: The creative field (art, design, photography, etc.)
            style: The creative style
            
        Returns:
            CreativePersona: AI-generated creative persona
        """
        try:
            prompt = f"""
            Create a creative persona for this field and style:
            
            Creative Field: {creative_field}
            Style: {style}
            
            Generate a comprehensive creative persona in JSON format:
            {{
                "name": "memorable creative persona name",
                "style": "creative/inspirational/analytical/entertaining",
                "traits": ["trait1", "trait2", "trait3", "trait4", "trait5"],
                "values": ["value1", "value2", "value3", "value4"],
                "voice_characteristics": {{
                    "creative_expression": "how they express creativity",
                    "inspiration_sharing": "how they share inspiration",
                    "process_description": "how they describe their process",
                    "artistic_critique": "how they discuss work"
                }},
                "content_preferences": {{
                    "visual_storytelling": "visual content approach",
                    "behind_scenes": "sharing creative process",
                    "collaboration_style": "working with others",
                    "trend_interaction": "relationship with trends"
                }},
                "business_focus": "main creative focus",
                "expertise_areas": ["area1", "area2", "area3"],
                "unique_perspective": "unique creative perspective",
                "engagement_approach": "creative community engagement",
                "creative_philosophy": "core creative philosophy",
                "artistic_influences": ["influence1", "influence2", "influence3"]
            }}
            """
            
            response = await self._call_llm(prompt, temperature=0.8)
            creative_data = json.loads(response)
            
            persona_style = self._validate_style(creative_data.get("style", "creative"))
            
            creative = CreativePersona(
                name=creative_data["name"],
                style=persona_style,
                traits=creative_data["traits"],
                values=creative_data["values"],
                voice_characteristics=creative_data["voice_characteristics"],
                content_preferences=creative_data["content_preferences"],
                business_focus=creative_data["business_focus"],
                expertise_areas=creative_data["expertise_areas"],
                unique_perspective=creative_data["unique_perspective"],
                engagement_approach=creative_data["engagement_approach"],
                creative_philosophy=creative_data["creative_philosophy"],
                artistic_influences=creative_data["artistic_influences"]
            )
            
            logger.info(f"Created creative persona: {creative.name}")
            return creative
            
        except Exception as e:
            logger.error(f"Failed to create creative persona: {str(e)}", exc_info=True)
            raise
    
    async def _call_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Call the appropriate LLM service.
        
        Args:
            prompt: The prompt to send
            temperature: Temperature setting for response
            
        Returns:
            str: The LLM response
        """
        settings = get_settings()
        
        try:
            if self.default_llm == "openai" and self.openai_client:
                response = await self.openai_client.chat.completions.create(
                    model=settings.ai_service.openai_model,
                    messages=[
                        {"role": "system", "content": "You are an expert at creating authentic, engaging personas for businesses."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=settings.ai_service.openai_max_tokens
                )
                return response.choices[0].message.content
            
            elif self.anthropic_client:
                response = await self.anthropic_client.messages.create(
                    model=settings.ai_service.anthropic_model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=settings.ai_service.anthropic_max_tokens,
                    system="You are an expert at creating authentic, engaging personas for businesses."
                )
                return response.content[0].text
            
            else:
                raise ValueError("No LLM client available")
                
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            raise
    
    def _validate_archetype(self, archetype_str: str) -> PersonaArchetype:
        """Validate and convert archetype string to enum"""
        archetype_map = {
            "educator": PersonaArchetype.EDUCATOR,
            "authority": PersonaArchetype.AUTHORITY,
            "friend": PersonaArchetype.FRIEND,
            "innovator": PersonaArchetype.INNOVATOR,
            "motivator": PersonaArchetype.MOTIVATOR,
            "curator": PersonaArchetype.CURATOR,
            "storyteller": PersonaArchetype.STORYTELLER,
            "problem_solver": PersonaArchetype.PROBLEM_SOLVER
        }
        
        archetype_lower = archetype_str.lower().replace("_", "")
        for key, value in archetype_map.items():
            if key.replace("_", "") == archetype_lower:
                return value
        
        logger.warning(f"Unknown archetype '{archetype_str}', defaulting to EDUCATOR")
        return PersonaArchetype.EDUCATOR
    
    def _validate_style(self, style_str: str) -> PersonaStyle:
        """Validate and convert style string to enum"""
        style_map = {
            "professional": PersonaStyle.PROFESSIONAL,
            "conversational": PersonaStyle.CONVERSATIONAL,
            "inspirational": PersonaStyle.INSPIRATIONAL,
            "analytical": PersonaStyle.ANALYTICAL,
            "creative": PersonaStyle.CREATIVE,
            "supportive": PersonaStyle.SUPPORTIVE,
            "entertaining": PersonaStyle.ENTERTAINING,
            "educational": PersonaStyle.EDUCATIONAL
        }
        
        style_lower = style_str.lower()
        if style_lower in style_map:
            return style_map[style_lower]
        
        logger.warning(f"Unknown style '{style_str}', defaulting to PROFESSIONAL")
        return PersonaStyle.PROFESSIONAL
    
    async def generate_multi_platform_personas(
        self,
        content_analysis: ContentAnalysis,
        platforms: Optional[List[Platform]] = None
    ) -> Dict[Platform, PlatformPersona]:
        """
        Generate platform-specific personas for multiple platforms.
        
        Args:
            content_analysis: The analyzed content data
            platforms: List of platforms (defaults to all if not specified)
            
        Returns:
            Dictionary mapping platforms to their adapted personas
        """
        try:
            # Generate base persona first
            base_persona = await self.generate_persona(content_analysis)
            
            # Use all platforms if not specified
            if not platforms:
                platforms = list(Platform)
            
            # Generate platform adaptations in parallel
            tasks = [
                self.adapt_persona_for_platform(base_persona, platform)
                for platform in platforms
            ]
            
            platform_personas = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Build result dictionary, handling any errors
            result = {}
            for i, platform in enumerate(platforms):
                if isinstance(platform_personas[i], Exception):
                    logger.error(f"Failed to adapt persona for {platform.value}: {str(platform_personas[i])}")
                else:
                    result[platform] = platform_personas[i]
            
            logger.info(f"Generated personas for {len(result)} platforms")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate multi-platform personas: {str(e)}", exc_info=True)
            raise