"""
Viral Content Engine for AutoGuru Universal

This module provides AI-powered viral content generation that works for ANY business niche
automatically. It generates platform-optimized content with maximum viral potential using
advanced AI strategies without hardcoded business logic.
"""

import asyncio
import json
import logging
import random
import re
from datetime import datetime, timedelta, timezone as tz
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from collections import defaultdict
import hashlib

import openai
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

# Import models from content_models
from backend.models.content_models import (
    BusinessNicheType,
    Platform,
    ContentFormat,
    AudienceProfile,
    BrandVoice,
    ContentTheme,
    PlatformContent,
    ContentAnalysis,
    ViralScore,
    ViralFactors
)

# Configure logging
logger = logging.getLogger(__name__)

# Platform-specific constraints
PLATFORM_CONSTRAINTS = {
    Platform.INSTAGRAM: {
        "caption_max": 2200,
        "hashtag_max": 30,
        "optimal_caption_length": 125,
        "formats": [ContentFormat.IMAGE, ContentFormat.VIDEO, ContentFormat.CAROUSEL, ContentFormat.STORY],
        "video_duration": {"min": 3, "max": 90},
        "carousel_max": 10
    },
    Platform.LINKEDIN: {
        "post_max": 3000,
        "hashtag_max": 5,
        "optimal_post_length": 1300,
        "formats": [ContentFormat.TEXT, ContentFormat.IMAGE, ContentFormat.VIDEO, ContentFormat.ARTICLE],
        "video_duration": {"min": 3, "max": 600}
    },
    Platform.TIKTOK: {
        "caption_max": 150,
        "hashtag_max": 100,  # TikTok allows many hashtags
        "optimal_caption_length": 80,
        "formats": [ContentFormat.VIDEO],
        "video_duration": {"min": 15, "max": 180}
    },
    Platform.TWITTER: {
        "tweet_max": 280,
        "thread_max": 25,
        "hashtag_max": 2,
        "optimal_tweet_length": 120,
        "formats": [ContentFormat.TEXT, ContentFormat.IMAGE, ContentFormat.VIDEO],
        "video_duration": {"min": 0.5, "max": 140}
    },
    Platform.FACEBOOK: {
        "post_max": 63206,
        "hashtag_max": 10,
        "optimal_post_length": 400,
        "formats": [ContentFormat.TEXT, ContentFormat.IMAGE, ContentFormat.VIDEO, ContentFormat.LIVE],
        "video_duration": {"min": 1, "max": 240}
    },
    Platform.YOUTUBE: {
        "title_max": 100,
        "description_max": 5000,
        "tags_max": 500,  # Total characters for all tags
        "optimal_description_length": 125,
        "formats": [ContentFormat.VIDEO],
        "video_duration": {"min": 60, "max": 43200}  # Up to 12 hours
    },
    Platform.PINTEREST: {
        "pin_description_max": 500,
        "board_description_max": 500,
        "hashtag_max": 20,
        "optimal_description_length": 200,
        "formats": [ContentFormat.IMAGE, ContentFormat.VIDEO],
        "video_duration": {"min": 4, "max": 60}
    }
}

# Viral content patterns by niche
VIRAL_PATTERNS = {
    BusinessNicheType.FITNESS_WELLNESS: {
        "hooks": ["Transform your body in", "The #1 mistake", "Science-backed", "30-day challenge"],
        "formats": ["before/after", "quick tips", "workout demos", "myth busting"],
        "emotions": ["inspiration", "motivation", "empowerment", "surprise"]
    },
    BusinessNicheType.BUSINESS_CONSULTING: {
        "hooks": ["How I made $", "The framework that", "Stop doing this", "CEOs use this"],
        "formats": ["case studies", "frameworks", "mistakes to avoid", "success stories"],
        "emotions": ["curiosity", "ambition", "fear of missing out", "confidence"]
    },
    BusinessNicheType.CREATIVE: {
        "hooks": ["Behind the scenes", "From concept to", "My creative process", "Client reaction"],
        "formats": ["process videos", "transformations", "tutorials", "showcases"],
        "emotions": ["awe", "inspiration", "curiosity", "appreciation"]
    },
    BusinessNicheType.EDUCATION: {
        "hooks": ["Learn this in 5 minutes", "The secret to", "Why schools don't teach", "Master class"],
        "formats": ["quick lessons", "myth busting", "step-by-step", "comparisons"],
        "emotions": ["curiosity", "empowerment", "surprise", "satisfaction"]
    }
}


@dataclass
class Content:
    """Represents a piece of content with metadata"""
    id: str
    text: str
    format: ContentFormat
    platform: Platform
    hashtags: List[str]
    media_requirements: Dict[str, Any]
    viral_score: float
    created_at: datetime
    metadata: Dict[str, Any]


@dataclass
class BusinessPersona:
    """Business persona combining audience profile and brand voice"""
    audience_profile: AudienceProfile
    brand_voice: BrandVoice
    business_niche: BusinessNicheType
    content_themes: List[ContentTheme]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls"""
        return {
            "audience": self.audience_profile.dict(),
            "voice": self.brand_voice.dict(),
            "niche": self.business_niche.value,
            "themes": [theme.dict() for theme in self.content_themes[:3]]  # Top 3 themes
        }


class ViralContentEngine:
    """
    AI-powered viral content generation engine that works for ANY business niche.
    Generates platform-optimized content with maximum viral potential.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        default_llm: str = "openai"
    ):
        """
        Initialize the ViralContentEngine.
        
        Args:
            openai_api_key: OpenAI API key for GPT models
            anthropic_api_key: Anthropic API key for Claude models
            default_llm: Default LLM provider to use ("openai" or "anthropic")
        """
        self.openai_client = None
        self.anthropic_client = None
        self.default_llm = default_llm
        
        if openai_api_key:
            self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
            
        if anthropic_api_key:
            self.anthropic_client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
            
        if not self.openai_client and not self.anthropic_client:
            raise ValueError("At least one LLM API key must be provided")
        
        self._trending_cache = {}
        self._hashtag_cache = defaultdict(list)
        self._timing_cache = {}
        
    def _generate_content_id(self, content: str, platform: Platform) -> str:
        """Generate unique content ID"""
        hash_input = f"{content}{platform.value}{datetime.utcnow().isoformat()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _validate_content_length(self, content: str, platform: Platform) -> bool:
        """Validate content length for platform"""
        constraints = PLATFORM_CONSTRAINTS.get(platform, {})
        max_length = constraints.get("caption_max", float('inf'))
        return len(content) <= max_length
    
    def _extract_hashtags(self, content: str) -> List[str]:
        """Extract hashtags from content"""
        hashtags = re.findall(r'#\w+', content)
        return [tag.lower() for tag in hashtags]
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content"""
        # Remove extra whitespace
        content = ' '.join(content.split())
        # Fix common formatting issues
        content = content.replace(' ,', ',').replace(' .', '.')
        return content.strip()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_viral_content(
        self,
        source_content: str,
        persona: BusinessPersona,
        platforms: List[Platform]
    ) -> Dict[Platform, PlatformContent]:
        """
        Generate viral content optimized for multiple platforms.
        
        Args:
            source_content: Source content or topic to create viral content from
            persona: Business persona including audience and brand voice
            platforms: Target platforms for content generation
            
        Returns:
            Dictionary mapping platforms to optimized content
        """
        if not source_content:
            raise ValueError("Source content cannot be empty")
        
        if not platforms:
            raise ValueError("At least one platform must be specified")
        
        try:
            # Generate base viral content
            base_content = await self._generate_base_viral_content(
                source_content, persona
            )
            
            # Optimize for each platform in parallel
            optimization_tasks = [
                self.optimize_for_platform(base_content, platform, persona.business_niche)
                for platform in platforms
            ]
            
            optimized_contents = await asyncio.gather(*optimization_tasks, return_exceptions=True)
            
            # Handle results and exceptions
            result = {}
            for i, platform in enumerate(platforms):
                if isinstance(optimized_contents[i], Exception):
                    logger.error(f"Failed to optimize for {platform.value}: {str(optimized_contents[i])}")
                    # Fallback to basic optimization
                    result[platform] = await self._fallback_optimization(base_content, platform)
                else:
                    result[platform] = optimized_contents[i]
            
            return result
            
        except Exception as e:
            logger.error(f"Viral content generation failed: {str(e)}")
            raise
    
    async def _generate_base_viral_content(
        self,
        source_content: str,
        persona: BusinessPersona
    ) -> str:
        """Generate base viral content before platform optimization"""
        
        # Get viral patterns for the niche
        viral_patterns = VIRAL_PATTERNS.get(
            persona.business_niche,
            VIRAL_PATTERNS[BusinessNicheType.BUSINESS_CONSULTING]  # Default patterns
        )
        
        prompt = f"""
        Create viral social media content based on this source material.
        
        Source Content: {source_content[:1500]}
        
        Business Context:
        - Niche: {persona.business_niche.value}
        - Target Audience: {json.dumps(persona.audience_profile.demographics.dict())}
        - Brand Voice: {persona.brand_voice.tone.value}
        - Key Themes: {[theme.theme_name for theme in persona.content_themes[:3]]}
        
        Viral Content Requirements:
        1. Start with a compelling hook that creates curiosity or emotion
        2. Deliver immediate value in the first few seconds/lines
        3. Use the brand's unique voice and personality
        4. Include a clear call-to-action
        5. Make it highly shareable and save-worthy
        
        Proven Viral Patterns for this niche:
        - Hooks: {viral_patterns['hooks']}
        - Formats: {viral_patterns['formats']}
        - Emotions: {viral_patterns['emotions']}
        
        Create content that would work well across multiple platforms.
        Focus on the core message and value proposition.
        
        Return ONLY the viral content text, no explanations.
        """
        
        try:
            response = await self._call_llm(prompt, temperature=0.8)
            return self._clean_content(response)
        except Exception as e:
            logger.error(f"Base viral content generation failed: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def optimize_for_platform(
        self,
        content: str,
        platform: Platform,
        niche: BusinessNicheType
    ) -> PlatformContent:
        """
        Optimize content for a specific platform.
        
        Args:
            content: Content to optimize
            platform: Target platform
            niche: Business niche for context
            
        Returns:
            Platform-optimized content
        """
        constraints = PLATFORM_CONSTRAINTS.get(platform, {})
        
        # Get optimal hashtags for the platform
        hashtags = await self.optimize_hashtags(content, platform, niche)
        
        # Get platform-specific optimization
        prompt = f"""
        Optimize this content specifically for {platform.value}.
        
        Original Content: {content}
        
        Platform Constraints:
        - Maximum length: {constraints.get('caption_max', 'No limit')}
        - Optimal length: {constraints.get('optimal_caption_length', 'Flexible')}
        - Hashtag limit: {constraints.get('hashtag_max', 30)}
        - Supported formats: {[f.value for f in constraints.get('formats', [])]}
        
        Platform Best Practices for {platform.value}:
        {self._get_platform_best_practices(platform)}
        
        Requirements:
        1. Adapt the content to fit platform culture and user behavior
        2. Optimize length for maximum engagement
        3. Include platform-specific features (e.g., @mentions, emojis)
        4. Create a strong call-to-action appropriate for the platform
        5. Ensure the content would stop users from scrolling
        
        Respond in JSON format:
        {{
            "content_text": "optimized content with proper formatting",
            "format": "recommended format (image/video/carousel/etc)",
            "call_to_action": "specific CTA for this platform",
            "media_requirements": {{
                "type": "image/video/etc",
                "dimensions": "optimal dimensions",
                "duration": "for video content",
                "additional_specs": {{}}
            }},
            "platform_features": ["features to use"],
            "posting_notes": "any special instructions"
        }}
        """
        
        try:
            response = await self._call_llm(prompt, temperature=0.7)
            result = json.loads(response)
            
            # Create PlatformContent object
            content_text = result['content_text']
            
            # Add hashtags to content if not already included
            hashtag_text = ' '.join(hashtags)
            if not any(tag in content_text for tag in hashtags):
                if platform in [Platform.INSTAGRAM, Platform.TIKTOK]:
                    content_text = f"{content_text}\n\n{hashtag_text}"
                else:
                    content_text = f"{content_text} {hashtag_text}"
            
            # Ensure content fits platform limits
            if len(content_text) > constraints.get('caption_max', float('inf')):
                content_text = self._truncate_content(content_text, constraints['caption_max'], hashtags)
            
            return PlatformContent(
                platform=platform,
                content_text=content_text,
                content_format=ContentFormat(result['format']),
                media_requirements=result['media_requirements'],
                hashtags=hashtags,
                mentions=[],  # To be populated based on relationships
                call_to_action=result['call_to_action'],
                posting_time=None,  # Will be set by timing optimization
                platform_features=result.get('platform_features', []),
                character_count=len(content_text),
                accessibility_text=None  # To be added based on media
            )
            
        except Exception as e:
            logger.error(f"Platform optimization failed for {platform.value}: {str(e)}")
            raise
    
    def _get_platform_best_practices(self, platform: Platform) -> str:
        """Get platform-specific best practices"""
        practices = {
            Platform.INSTAGRAM: """
            - Use line breaks for readability
            - Start with the most important message
            - Use emojis strategically
            - Include a question to boost engagement
            - Save the hashtags for the end or first comment
            """,
            Platform.LINKEDIN: """
            - Professional tone but personable
            - Share insights and lessons learned
            - Use bullet points or numbered lists
            - Include industry-relevant keywords
            - End with a thought-provoking question
            """,
            Platform.TIKTOK: """
            - Hook viewers in the first 3 seconds
            - Use trending sounds and effects notation
            - Keep text overlay suggestions brief
            - Include a clear CTA (follow, share, save)
            - Reference current trends when relevant
            """,
            Platform.TWITTER: """
            - Be concise and punchy
            - Use thread format for longer content
            - Include relevant handles for reach
            - Make it retweetable (quotable)
            - Add compelling stats or facts
            """,
            Platform.FACEBOOK: """
            - Tell a story or share an experience
            - Use conversational tone
            - Include a clear CTA
            - Make it shareable for groups
            - Consider adding a question for comments
            """,
            Platform.YOUTUBE: """
            - Compelling title with keywords
            - Hook in the description preview
            - Include timestamps for sections
            - Add relevant tags
            - Include links and resources
            """
        }
        return practices.get(platform, "Follow general best practices for engagement")
    
    def _truncate_content(self, content: str, max_length: int, hashtags: List[str]) -> str:
        """Intelligently truncate content while preserving hashtags"""
        hashtag_text = ' '.join(hashtags)
        hashtag_length = len(hashtag_text) + 3  # Adding space and ...
        
        if len(content) <= max_length:
            return content
        
        # Remove hashtags from content first
        content_without_tags = content
        for tag in hashtags:
            content_without_tags = content_without_tags.replace(tag, '')
        
        # Calculate available space
        available_length = max_length - hashtag_length
        
        # Truncate and add hashtags
        if available_length > 0:
            truncated = content_without_tags[:available_length-3] + "..."
            return f"{truncated} {hashtag_text}"
        else:
            # If hashtags are too long, include only most important ones
            essential_tags = hashtags[:5]
            return f"{content[:max_length-len(' '.join(essential_tags))-3]}... {' '.join(essential_tags)}"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_content_series(
        self,
        theme: str,
        persona: BusinessPersona,
        count: int = 5
    ) -> List[Content]:
        """
        Generate a series of related content pieces around a theme.
        
        Args:
            theme: Central theme for the content series
            persona: Business persona for context
            count: Number of content pieces to generate
            
        Returns:
            List of related content pieces
        """
        if count < 1 or count > 10:
            raise ValueError("Count must be between 1 and 10")
        
        prompt = f"""
        Create a content series of {count} related posts about: {theme}
        
        Business Context:
        - Niche: {persona.business_niche.value}
        - Audience: {json.dumps(persona.audience_profile.demographics.dict())}
        - Brand Voice: {persona.brand_voice.tone.value}
        
        Requirements:
        1. Each piece should stand alone but connect to the theme
        2. Vary the content angles and formats
        3. Build momentum throughout the series
        4. Include different types of value (educational, inspirational, practical)
        5. Each should have viral potential
        
        Create {count} unique content pieces.
        
        Respond in JSON format:
        {{
            "series_title": "overall series name",
            "pieces": [
                {{
                    "number": 1,
                    "title": "piece title",
                    "content": "full content text",
                    "format": "image/video/carousel",
                    "angle": "unique angle for this piece",
                    "cta": "specific call to action"
                }}
            ]
        }}
        """
        
        try:
            response = await self._call_llm(prompt, temperature=0.8)
            series_data = json.loads(response)
            
            content_list = []
            for i, piece in enumerate(series_data['pieces'][:count]):
                content = Content(
                    id=self._generate_content_id(piece['content'], Platform.INSTAGRAM),
                    text=piece['content'],
                    format=ContentFormat(piece['format']),
                    platform=Platform.INSTAGRAM,  # Default, can be adapted
                    hashtags=[],  # Will be generated separately
                    media_requirements={
                        "type": piece['format'],
                        "title": piece['title'],
                        "series_position": f"{i+1}/{count}"
                    },
                    viral_score=0.0,  # Will be calculated
                    created_at=datetime.utcnow(),
                    metadata={
                        "series_title": series_data['series_title'],
                        "angle": piece['angle'],
                        "cta": piece['cta'],
                        "theme": theme
                    }
                )
                content_list.append(content)
            
            # Calculate viral scores for each piece
            viral_scores = await self._batch_calculate_viral_scores(
                content_list, persona.business_niche
            )
            
            for content, score in zip(content_list, viral_scores):
                content.viral_score = score
            
            return content_list
            
        except Exception as e:
            logger.error(f"Content series generation failed: {str(e)}")
            raise
    
    async def _batch_calculate_viral_scores(
        self,
        contents: List[Content],
        niche: BusinessNicheType
    ) -> List[float]:
        """Calculate viral scores for multiple content pieces"""
        scores = []
        for content in contents:
            # Simplified viral score calculation
            score = 0.5  # Base score
            
            # Boost for emotion triggers
            emotion_words = ['amazing', 'shocking', 'incredible', 'transform', 'secret']
            for word in emotion_words:
                if word.lower() in content.text.lower():
                    score += 0.05
            
            # Boost for questions
            if '?' in content.text:
                score += 0.1
            
            # Boost for numbered lists
            if any(char.isdigit() for char in content.text[:50]):
                score += 0.05
            
            # Platform-specific boosts
            if content.format == ContentFormat.VIDEO:
                score += 0.1
            elif content.format == ContentFormat.CAROUSEL:
                score += 0.05
            
            scores.append(min(score, 1.0))
        
        return scores
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def create_trending_content(
        self,
        niche: BusinessNicheType,
        trending_topics: List[str]
    ) -> List[Content]:
        """
        Create content based on trending topics.
        
        Args:
            niche: Business niche for context
            trending_topics: List of current trending topics
            
        Returns:
            List of trending-optimized content
        """
        if not trending_topics:
            raise ValueError("At least one trending topic must be provided")
        
        # Limit to top 5 trending topics
        trending_topics = trending_topics[:5]
        
        prompt = f"""
        Create viral content that leverages these trending topics for a {niche.value} business.
        
        Trending Topics: {trending_topics}
        
        For each trending topic, create content that:
        1. Naturally connects the trend to the business niche
        2. Provides unique value or perspective
        3. Feels authentic, not forced
        4. Has high viral potential
        5. Includes a clear business benefit
        
        Create one piece of content per trending topic.
        
        Respond in JSON format:
        {{
            "contents": [
                {{
                    "topic": "trending topic used",
                    "content": "full content text",
                    "connection": "how it connects to the niche",
                    "format": "recommended format",
                    "viral_elements": ["element1", "element2"]
                }}
            ]
        }}
        """
        
        try:
            response = await self._call_llm(prompt, temperature=0.8)
            trending_data = json.loads(response)
            
            content_list = []
            for item in trending_data['contents']:
                content = Content(
                    id=self._generate_content_id(item['content'], Platform.INSTAGRAM),
                    text=item['content'],
                    format=ContentFormat(item['format']),
                    platform=Platform.INSTAGRAM,  # Default
                    hashtags=[],  # Will be generated
                    media_requirements={
                        "type": item['format'],
                        "trending_topic": item['topic']
                    },
                    viral_score=0.85,  # High score for trending content
                    created_at=datetime.utcnow(),
                    metadata={
                        "trending_topic": item['topic'],
                        "connection": item['connection'],
                        "viral_elements": item['viral_elements']
                    }
                )
                content_list.append(content)
            
            # Generate hashtags for each piece
            for content in content_list:
                topic = content.metadata['trending_topic']
                base_hashtags = await self.optimize_hashtags(
                    content.text, content.platform, niche
                )
                # Add trending topic as hashtag
                trending_tag = f"#{topic.replace(' ', '').lower()}"
                content.hashtags = [trending_tag] + base_hashtags[:10]
            
            return content_list
            
        except Exception as e:
            logger.error(f"Trending content creation failed: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def optimize_hashtags(
        self,
        content: str,
        platform: Platform,
        niche: BusinessNicheType
    ) -> List[str]:
        """
        Generate optimized hashtags for content.
        
        Args:
            content: Content to generate hashtags for
            platform: Target platform
            niche: Business niche for context
            
        Returns:
            List of optimized hashtags
        """
        # Check cache first
        cache_key = f"{platform.value}_{niche.value}_{content[:50]}"
        if cache_key in self._hashtag_cache:
            return self._hashtag_cache[cache_key]
        
        constraints = PLATFORM_CONSTRAINTS.get(platform, {})
        max_hashtags = constraints.get('hashtag_max', 30)
        
        prompt = f"""
        Generate optimal hashtags for this content on {platform.value}.
        
        Content: {content[:500]}
        Business Niche: {niche.value}
        Platform Limit: {max_hashtags} hashtags
        
        Requirements:
        1. Mix of broad reach and niche-specific hashtags
        2. Include trending hashtags when relevant
        3. Balance popular and less competitive tags
        4. Platform-appropriate formatting
        5. Remove spaces and special characters
        
        Hashtag Strategy for {platform.value}:
        {self._get_hashtag_strategy(platform)}
        
        Respond in JSON format:
        {{
            "hashtags": [
                {{
                    "tag": "hashtag",
                    "reach": "high/medium/low",
                    "relevance": "high/medium/low"
                }}
            ],
            "primary_tags": ["top 5 most important tags"]
        }}
        """
        
        try:
            response = await self._call_llm(prompt, temperature=0.6)
            hashtag_data = json.loads(response)
            
            # Extract hashtags and ensure proper formatting
            hashtags = []
            for item in hashtag_data['hashtags'][:max_hashtags]:
                tag = item['tag'].lower()
                if not tag.startswith('#'):
                    tag = f"#{tag}"
                # Remove spaces and special characters
                tag = re.sub(r'[^#\w]', '', tag)
                if tag and tag != '#':
                    hashtags.append(tag)
            
            # Ensure primary tags are included
            for primary in hashtag_data.get('primary_tags', [])[:5]:
                primary_tag = f"#{primary.lower().replace(' ', '')}"
                if primary_tag not in hashtags:
                    hashtags.insert(0, primary_tag)
            
            # Limit to platform maximum
            hashtags = hashtags[:max_hashtags]
            
            # Cache the result
            self._hashtag_cache[cache_key] = hashtags
            
            return hashtags
            
        except Exception as e:
            logger.error(f"Hashtag optimization failed: {str(e)}")
            # Return fallback hashtags
            return self._get_fallback_hashtags(niche, platform)[:max_hashtags]
    
    def _get_hashtag_strategy(self, platform: Platform) -> str:
        """Get platform-specific hashtag strategies"""
        strategies = {
            Platform.INSTAGRAM: """
            - Use all 30 hashtags for maximum reach
            - Mix hashtag sizes: 5-10 large (1M+), 10-15 medium (100K-1M), 5-10 small (<100K)
            - Include branded and community hashtags
            - Hide hashtags in first comment if needed
            """,
            Platform.LINKEDIN: """
            - Use 3-5 highly relevant professional hashtags
            - Focus on industry-specific terms
            - Include skill-based hashtags
            - Avoid overuse - quality over quantity
            """,
            Platform.TIKTOK: """
            - Mix trending and niche hashtags
            - Include challenge hashtags when relevant
            - Use discovery-focused tags
            - Include 5-8 in caption, more in comments
            """,
            Platform.TWITTER: """
            - Use 1-2 highly relevant hashtags only
            - Focus on trending or event hashtags
            - Don't sacrifice message for hashtags
            - Place at end of tweet
            """
        }
        return strategies.get(platform, "Use relevant hashtags appropriate for the platform")
    
    def _get_fallback_hashtags(self, niche: BusinessNicheType, platform: Platform) -> List[str]:
        """Get fallback hashtags by niche"""
        niche_hashtags = {
            BusinessNicheType.FITNESS_WELLNESS: [
                "#fitness", "#health", "#wellness", "#workout", "#fitnessmotivation",
                "#healthylifestyle", "#gym", "#training", "#nutrition", "#fitfam"
            ],
            BusinessNicheType.BUSINESS_CONSULTING: [
                "#business", "#entrepreneur", "#consulting", "#businessgrowth", "#leadership",
                "#businesstips", "#success", "#businessowner", "#strategy", "#businessadvice"
            ],
            BusinessNicheType.CREATIVE: [
                "#creative", "#design", "#art", "#creativity", "#designer",
                "#artwork", "#creative", "#artist", "#graphicdesign", "#creativeprocess"
            ],
            BusinessNicheType.EDUCATION: [
                "#education", "#learning", "#teaching", "#edtech", "#onlinelearning",
                "#studytips", "#educational", "#knowledge", "#students", "#teachers"
            ]
        }
        return niche_hashtags.get(niche, ["#business", "#entrepreneur", "#success"])
    
    async def calculate_optimal_timing(
        self,
        niche: BusinessNicheType,
        platform: Platform,
        timezone_str: str = "UTC"
    ) -> List[datetime]:
        """
        Calculate optimal posting times.
        
        Args:
            niche: Business niche for audience behavior
            platform: Target platform
            timezone_str: Timezone string for the times (e.g., "UTC", "US/Eastern")
            
        Returns:
            List of optimal posting times
        """
        # Check cache
        cache_key = f"{niche.value}_{platform.value}_{timezone_str}"
        if cache_key in self._timing_cache:
            return self._timing_cache[cache_key]
        
        # Platform-specific optimal times (in UTC)
        optimal_times = {
            Platform.INSTAGRAM: {
                BusinessNicheType.FITNESS_WELLNESS: [(6, 0), (12, 0), (17, 30), (19, 0)],
                BusinessNicheType.BUSINESS_CONSULTING: [(7, 0), (12, 30), (17, 0), (19, 0)],
                BusinessNicheType.CREATIVE: [(9, 0), (14, 0), (18, 0), (21, 0)],
                BusinessNicheType.EDUCATION: [(8, 0), (15, 0), (19, 0), (21, 0)],
                "default": [(9, 0), (13, 0), (17, 0), (20, 0)]
            },
            Platform.LINKEDIN: {
                BusinessNicheType.BUSINESS_CONSULTING: [(7, 30), (12, 0), (17, 30)],
                "default": [(8, 0), (12, 0), (17, 0)]
            },
            Platform.TIKTOK: {
                BusinessNicheType.FITNESS_WELLNESS: [(6, 0), (10, 0), (19, 0), (22, 0)],
                BusinessNicheType.CREATIVE: [(9, 0), (15, 0), (20, 0), (23, 0)],
                "default": [(7, 0), (14, 0), (19, 0), (21, 0)]
            }
        }
        
        # Get platform times
        platform_times = optimal_times.get(platform, optimal_times[Platform.INSTAGRAM])
        niche_times = platform_times.get(niche, platform_times.get("default", [(9, 0), (17, 0)]))
        
        # Convert to datetime objects
        # For now, we'll use UTC times and note that timezone conversion should be handled separately
        today = datetime.now(tz.utc)
        posting_times = []
        
        for hour, minute in niche_times:
            time = today.replace(hour=hour, minute=minute, second=0, microsecond=0)
            posting_times.append(time)
        
        # Cache the result
        self._timing_cache[cache_key] = posting_times
        
        return posting_times
    
    async def _fallback_optimization(self, content: str, platform: Platform) -> PlatformContent:
        """Fallback optimization when main optimization fails"""
        constraints = PLATFORM_CONSTRAINTS.get(platform, {})
        max_length = constraints.get('caption_max', 2000)
        
        # Truncate content if needed
        if len(content) > max_length:
            content = content[:max_length-3] + "..."
        
        return PlatformContent(
            platform=platform,
            content_text=content,
            content_format=ContentFormat.TEXT,
            media_requirements={"type": "text"},
            hashtags=[],
            mentions=[],
            call_to_action="Learn more",
            posting_time=None,
            platform_features=[],
            character_count=len(content),
            accessibility_text=None
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _call_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """Call the configured LLM"""
        try:
            if self.default_llm == "openai" and self.openai_client:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[
                        {"role": "system", "content": "You are a viral content expert who creates engaging social media content."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=2000
                )
                return response.choices[0].message.content
                
            elif self.default_llm == "anthropic" and self.anthropic_client:
                response = await self.anthropic_client.messages.create(
                    model="claude-3-opus-20240229",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=2000
                )
                return response.content[0].text
                
            else:
                raise ValueError(f"No LLM client available for {self.default_llm}")
                
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            raise


# Example usage and testing
async def example_usage():
    """Example of how to use the ViralContentEngine"""
    
    # Initialize the engine
    engine = ViralContentEngine(
        openai_api_key="your-api-key",
        default_llm="openai"
    )
    
    # Create a sample persona
    from backend.models.content_models import (
        Demographics, Psychographics, ContentPreferences,
        CommunicationStyle, ToneType
    )
    
    sample_audience = AudienceProfile(
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
    
    sample_voice = BrandVoice(
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
    
    sample_theme = ContentTheme(
        theme_name="Morning Routines",
        description="Productive morning routine tips",
        relevance_score=0.9,
        subtopics=["exercise", "meditation", "planning"],
        content_pillars=["education", "inspiration"],
        keywords=["morningroutine", "productivity", "success"],
        audience_interest_level="high",
        competitive_advantage="Science-backed approaches"
    )
    
    persona = BusinessPersona(
        audience_profile=sample_audience,
        brand_voice=sample_voice,
        business_niche=BusinessNicheType.FITNESS_WELLNESS,
        content_themes=[sample_theme]
    )
    
    # Generate viral content
    source = "5 minute morning workout that boosts energy and productivity"
    platforms = [Platform.INSTAGRAM, Platform.LINKEDIN, Platform.TIKTOK]
    
    viral_content = await engine.generate_viral_content(source, persona, platforms)
    
    # Generate content series
    series = await engine.generate_content_series(
        "30-Day Transformation Challenge",
        persona,
        count=5
    )
    
    # Create trending content
    trending_topics = ["New Year Goals", "Work From Home", "Self Care"]
    trending_content = await engine.create_trending_content(
        BusinessNicheType.FITNESS_WELLNESS,
        trending_topics
    )
    
    return viral_content, series, trending_content


if __name__ == "__main__":
    # For testing purposes
    asyncio.run(example_usage())