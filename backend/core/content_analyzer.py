"""
Universal Content Analyzer for AutoGuru Universal

This module provides AI-powered content analysis that works for ANY business niche
automatically. It uses LLMs to detect business niches, analyze content, and determine
optimal strategies without hardcoded business logic.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import openai
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logger = logging.getLogger(__name__)


class BusinessNiche(Enum):
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


class Platform(Enum):
    """Social media platforms supported"""
    INSTAGRAM = "instagram"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    FACEBOOK = "facebook"
    TIKTOK = "tiktok"
    YOUTUBE = "youtube"


@dataclass
class ContentAnalysisResult:
    """Results from content analysis"""
    business_niche: BusinessNiche
    confidence_score: float
    target_audience: Dict[str, Any]
    brand_voice: Dict[str, str]
    viral_potential: Dict[Platform, float]
    key_themes: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]


@dataclass
class TargetAudience:
    """Target audience analysis"""
    demographics: Dict[str, Any]
    psychographics: Dict[str, Any]
    pain_points: List[str]
    interests: List[str]
    preferred_platforms: List[Platform]


@dataclass
class BrandVoice:
    """Brand voice characteristics"""
    tone: str
    style: str
    personality_traits: List[str]
    communication_preferences: Dict[str, str]
    do_not_use: List[str]


class UniversalContentAnalyzer:
    """
    AI-powered content analyzer that works for ANY business niche automatically.
    Uses LLMs to analyze content and determine optimal strategies without hardcoded logic.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        default_llm: str = "openai"
    ):
        """
        Initialize the UniversalContentAnalyzer.
        
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
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def analyze_content(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        platforms: Optional[List[Platform]] = None
    ) -> ContentAnalysisResult:
        """
        Main analysis function that orchestrates all content analysis tasks.
        
        Args:
            content: The content to analyze (text, post, article, etc.)
            context: Additional context about the business/content
            platforms: Target platforms for viral potential assessment
            
        Returns:
            ContentAnalysisResult with comprehensive analysis
        """
        try:
            # Run all analyses in parallel for better performance
            tasks = [
                self.detect_business_niche(content, context),
                self.analyze_target_audience(content, context),
                self.extract_brand_voice(content, context),
                self._extract_key_themes(content, context)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions from parallel tasks
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Task {i} failed: {str(result)}")
                    raise result
            
            # Safe unpacking after exception check
            niche_result = results[0]
            audience_result = results[1]
            voice_result = results[2]
            themes = results[3]
            
            # Assess viral potential for specified platforms
            if not platforms:
                platforms = list(Platform)
            
            viral_potential = await self.assess_viral_potential(
                content, niche_result[0], audience_result, platforms
            )
            
            # Generate recommendations based on all analyses
            recommendations = await self._generate_recommendations(
                content, niche_result[0], audience_result, voice_result, viral_potential
            )
            
            return ContentAnalysisResult(
                business_niche=niche_result[0],
                confidence_score=niche_result[1],
                target_audience=audience_result,
                brand_voice=voice_result,
                viral_potential=viral_potential,
                key_themes=themes,
                recommendations=recommendations,
                metadata={
                    "content_length": len(content),
                    "analysis_timestamp": asyncio.get_event_loop().time(),
                    "llm_provider": self.default_llm
                }
            )
            
        except Exception as e:
            logger.error(f"Content analysis failed: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def detect_business_niche(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[BusinessNiche, float]:
        """
        AI-powered business niche detection from content.
        
        Args:
            content: Content to analyze
            context: Additional business context
            
        Returns:
            Tuple of (BusinessNiche, confidence_score)
        """
        prompt = f"""
        Analyze the following content and determine the business niche it belongs to.
        
        Content: {content[:2000]}  # Limit content length for API
        
        {f"Additional Context: {json.dumps(context)}" if context else ""}
        
        Classify into one of these niches:
        - education (courses, tutoring, coaching)
        - business_consulting (business advice, consulting)
        - fitness_wellness (fitness, health, wellness)
        - creative (art, design, photography)
        - ecommerce (retail, products)
        - local_service (local businesses, services)
        - technology (tech, SaaS, software)
        - non_profit (charitable organizations)
        - other (if doesn't fit above categories)
        
        Respond in JSON format:
        {{
            "niche": "selected_niche",
            "confidence": 0.0-1.0,
            "reasoning": "brief explanation"
        }}
        """
        
        try:
            response = await self._call_llm(prompt, temperature=0.3)
            result = json.loads(response)
            
            niche_map = {
                "education": BusinessNiche.EDUCATION,
                "business_consulting": BusinessNiche.BUSINESS_CONSULTING,
                "fitness_wellness": BusinessNiche.FITNESS_WELLNESS,
                "creative": BusinessNiche.CREATIVE,
                "ecommerce": BusinessNiche.ECOMMERCE,
                "local_service": BusinessNiche.LOCAL_SERVICE,
                "technology": BusinessNiche.TECHNOLOGY,
                "non_profit": BusinessNiche.NON_PROFIT,
                "other": BusinessNiche.OTHER
            }
            
            niche = niche_map.get(result["niche"], BusinessNiche.OTHER)
            confidence = float(result["confidence"])
            
            logger.info(f"Detected niche: {niche.value} with confidence: {confidence}")
            return niche, confidence
            
        except Exception as e:
            logger.error(f"Niche detection failed: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def analyze_target_audience(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Determine optimal target audience using AI analysis.
        
        Args:
            content: Content to analyze
            context: Additional business context
            
        Returns:
            Dictionary with detailed audience analysis
        """
        prompt = f"""
        Analyze the content and determine the optimal target audience.
        
        Content: {content[:2000]}
        
        {f"Context: {json.dumps(context)}" if context else ""}
        
        Provide a comprehensive audience analysis in JSON format:
        {{
            "demographics": {{
                "age_range": "18-25, 25-35, etc",
                "gender": "primary gender or 'all'",
                "location": "geographic focus",
                "income_level": "low/medium/high",
                "education": "education level"
            }},
            "psychographics": {{
                "values": ["list of core values"],
                "lifestyle": "lifestyle description",
                "personality": ["personality traits"],
                "attitudes": ["key attitudes"]
            }},
            "pain_points": ["list of main pain points"],
            "interests": ["list of interests"],
            "preferred_platforms": ["instagram", "linkedin", etc],
            "content_preferences": {{
                "format": "video/image/text",
                "tone": "professional/casual/etc",
                "topics": ["preferred topics"]
            }}
        }}
        """
        
        try:
            response = await self._call_llm(prompt, temperature=0.5)
            audience_data = json.loads(response)
            
            logger.info("Target audience analysis completed")
            return audience_data
            
        except Exception as e:
            logger.error(f"Audience analysis failed: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def extract_brand_voice(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Identify communication style and brand voice from content.
        
        Args:
            content: Content to analyze
            context: Additional business context
            
        Returns:
            Dictionary with brand voice characteristics
        """
        prompt = f"""
        Analyze the content and extract the brand voice characteristics.
        
        Content: {content[:2000]}
        
        {f"Context: {json.dumps(context)}" if context else ""}
        
        Provide brand voice analysis in JSON format:
        {{
            "tone": "professional/casual/friendly/authoritative/etc",
            "style": "conversational/formal/educational/promotional/etc",
            "personality_traits": ["trait1", "trait2", "trait3"],
            "communication_preferences": {{
                "vocabulary": "simple/technical/mixed",
                "sentence_structure": "short/long/varied",
                "engagement_style": "direct/indirect/storytelling"
            }},
            "unique_elements": ["element1", "element2"],
            "do_not_use": ["words/phrases to avoid"]
        }}
        """
        
        try:
            response = await self._call_llm(prompt, temperature=0.4)
            voice_data = json.loads(response)
            
            logger.info("Brand voice extraction completed")
            return voice_data
            
        except Exception as e:
            logger.error(f"Brand voice extraction failed: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def assess_viral_potential(
        self,
        content: str,
        business_niche: BusinessNiche,
        target_audience: Dict[str, Any],
        platforms: List[Platform]
    ) -> Dict[Platform, float]:
        """
        Predict viral potential per platform using AI analysis.
        
        Args:
            content: Content to analyze
            business_niche: Detected business niche
            target_audience: Target audience analysis
            platforms: Platforms to assess
            
        Returns:
            Dictionary mapping platforms to viral potential scores (0-1)
        """
        platform_names = [p.value for p in platforms]
        
        prompt = f"""
        Assess the viral potential of this content across different platforms.
        
        Content: {content[:1500]}
        Business Niche: {business_niche.value}
        Target Audience: {json.dumps(target_audience)[:500]}
        
        For each platform in {platform_names}, provide:
        1. Viral potential score (0.0-1.0)
        2. Key factors affecting virality
        
        Consider platform-specific factors:
        - Instagram: Visual appeal, hashtag potential, story-worthy
        - Twitter: Shareability, thread potential, trending topics
        - LinkedIn: Professional value, thought leadership
        - Facebook: Community engagement, share-worthiness
        - TikTok: Trend alignment, entertainment value
        - YouTube: Search potential, watch time, thumbnail appeal
        
        Respond in JSON format:
        {{
            "platform_name": {{
                "score": 0.0-1.0,
                "factors": ["factor1", "factor2"],
                "improvements": ["suggestion1", "suggestion2"]
            }}
        }}
        """
        
        try:
            response = await self._call_llm(prompt, temperature=0.6)
            viral_data = json.loads(response)
            
            # Convert to Platform enum keys
            result = {}
            for platform in platforms:
                if platform.value in viral_data:
                    result[platform] = float(viral_data[platform.value]["score"])
                else:
                    result[platform] = 0.5  # Default score if not analyzed
            
            logger.info(f"Viral potential assessed for {len(platforms)} platforms")
            return result
            
        except Exception as e:
            logger.error(f"Viral potential assessment failed: {str(e)}")
            raise
    
    async def _extract_key_themes(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Extract key themes from content"""
        prompt = f"""
        Extract 3-5 key themes from this content.
        
        Content: {content[:2000]}
        
        Return as JSON: {{"themes": ["theme1", "theme2", ...]}}
        """
        
        try:
            response = await self._call_llm(prompt, temperature=0.4)
            result = json.loads(response)
            return result.get("themes", [])
        except Exception as e:
            logger.error(f"Theme extraction failed: {str(e)}")
            return []
    
    async def _generate_recommendations(
        self,
        content: str,
        niche: BusinessNiche,
        audience: Dict[str, Any],
        voice: Dict[str, str],
        viral_potential: Dict[Platform, float]
    ) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        prompt = f"""
        Based on the content analysis, provide 5-7 actionable recommendations.
        
        Analysis Summary:
        - Business Niche: {niche.value}
        - Target Audience: {json.dumps(audience)[:300]}
        - Brand Voice: {json.dumps(voice)[:200]}
        - Top Platforms: {sorted(viral_potential.items(), key=lambda x: x[1], reverse=True)[:3]}
        
        Provide specific, actionable recommendations for improving content performance.
        
        Return as JSON: {{"recommendations": ["rec1", "rec2", ...]}}
        """
        
        try:
            response = await self._call_llm(prompt, temperature=0.7)
            result = json.loads(response)
            return result.get("recommendations", [])
        except Exception as e:
            logger.error(f"Recommendation generation failed: {str(e)}")
            return []
    
    async def _call_llm(self, prompt: str, temperature: float = 0.5) -> str:
        """
        Call the configured LLM with proper error handling.
        
        Args:
            prompt: The prompt to send
            temperature: Temperature for response generation
            
        Returns:
            LLM response as string
        """
        try:
            if self.default_llm == "openai" and self.openai_client:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[
                        {"role": "system", "content": "You are an expert content analyst for social media. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content
                
            elif self.default_llm == "anthropic" and self.anthropic_client:
                response = await self.anthropic_client.messages.create(
                    model="claude-3-opus-20240229",
                    messages=[
                        {"role": "user", "content": f"You are an expert content analyst. Always respond with valid JSON.\n\n{prompt}"}
                    ],
                    temperature=temperature,
                    max_tokens=2000
                )
                return response.content[0].text
                
            else:
                # Fallback to whichever client is available
                if self.openai_client:
                    self.default_llm = "openai"
                    return await self._call_llm(prompt, temperature)
                elif self.anthropic_client:
                    self.default_llm = "anthropic"
                    return await self._call_llm(prompt, temperature)
                else:
                    raise ValueError("No LLM client available")
                    
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            raise