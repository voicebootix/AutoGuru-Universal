"""
AI Creative Service

Central AI service for content creation and analysis.
"""

import os
import logging
from typing import Dict, List, Any, Optional
import json
import asyncio
from datetime import datetime

# AI imports
try:
    import openai
    import anthropic
except ImportError:
    openai = anthropic = None

logger = logging.getLogger(__name__)


class AICreativeService:
    """AI service for creative content generation and analysis"""
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self._initialize_clients()
        
    def _initialize_clients(self):
        """Initialize AI clients"""
        if openai and os.getenv('OPENAI_API_KEY'):
            self.openai_client = openai
            openai.api_key = os.getenv('OPENAI_API_KEY')
            
        if anthropic and os.getenv('ANTHROPIC_API_KEY'):
            self.anthropic_client = anthropic.Anthropic(
                api_key=os.getenv('ANTHROPIC_API_KEY')
            )
            
    async def detect_business_niche(self, content_text: str, visual_elements: List[str], brand_context: Dict[str, Any]) -> str:
        """Detect business niche from content context"""
        # Analyze content to determine business niche
        niche_indicators = {
            'education': ['learning', 'course', 'student', 'teach', 'knowledge', 'skill'],
            'fitness': ['workout', 'exercise', 'health', 'gym', 'training', 'nutrition'],
            'business_consulting': ['strategy', 'growth', 'consulting', 'business', 'leadership'],
            'creative_arts': ['design', 'art', 'creative', 'aesthetic', 'visual'],
            'finance': ['investment', 'money', 'finance', 'trading', 'wealth'],
            'health_wellness': ['wellness', 'mindfulness', 'health', 'balance', 'healing']
        }
        
        # Simple keyword matching (in production, use ML model)
        content_lower = content_text.lower()
        niche_scores = {}
        
        for niche, keywords in niche_indicators.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            score += sum(0.5 for element in visual_elements if any(keyword in element.lower() for keyword in keywords))
            niche_scores[niche] = score
            
        # Return highest scoring niche
        if niche_scores:
            return max(niche_scores.items(), key=lambda x: x[1])[0]
        return 'business_consulting'  # Default
        
    async def generate_creative_brief(self, business_niche: str, target_audience: Dict[str, Any], objectives: List[str]) -> str:
        """Generate AI-optimized creative brief"""
        prompt = f"""
        Create a creative brief for a {business_niche} business targeting:
        - Age: {target_audience.get('age_range', '25-45')}
        - Interests: {', '.join(target_audience.get('interests', []))}
        
        Objectives: {', '.join(objectives)}
        
        Provide a concise, actionable creative brief.
        """
        
        if self.openai_client:
            try:
                response = await asyncio.to_thread(
                    self.openai_client.chat.completions.create,
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"OpenAI creative brief generation failed: {str(e)}")
                
        # Fallback
        return f"Create engaging {business_niche} content for {target_audience.get('age_range', 'general')} audience"
        
    async def identify_viral_elements(self, platform: str, business_niche: str, current_trends: Dict[str, Any]) -> Dict[str, Any]:
        """Identify viral elements for content"""
        viral_elements = {
            'hooks': await self.generate_viral_hooks(platform, business_niche),
            'emotional_triggers': await self.identify_emotional_triggers(business_niche),
            'trending_elements': current_trends,
            'engagement_tactics': await self.suggest_engagement_tactics(platform, business_niche)
        }
        
        return viral_elements
        
    async def generate_viral_hooks(self, platform: str, business_niche: str) -> List[str]:
        """Generate viral hooks for content"""
        platform_hooks = {
            'instagram': ['Did you know...', 'Save this for later!', 'Tag someone who needs this'],
            'tiktok': ['Wait for it...', 'POV:', 'Nobody talks about this...'],
            'youtube': ['You won\'t believe...', 'The truth about...', 'X things you didn\'t know'],
            'facebook': ['Share if you agree', 'This changed my life', 'Must read!']
        }
        
        niche_hooks = {
            'education': ['Learn this in 30 seconds', '5 skills that will change your life'],
            'fitness': ['Transform your body with...', 'The workout that actually works'],
            'business_consulting': ['The strategy Fortune 500 companies use', 'Grow your business by 10x']
        }
        
        hooks = platform_hooks.get(platform, []) + niche_hooks.get(business_niche, [])
        return hooks[:5]  # Return top 5 hooks
        
    async def identify_emotional_triggers(self, business_niche: str) -> List[str]:
        """Identify emotional triggers for business niche"""
        triggers = {
            'education': ['achievement', 'curiosity', 'growth', 'empowerment'],
            'fitness': ['transformation', 'confidence', 'energy', 'determination'],
            'business_consulting': ['success', 'ambition', 'leadership', 'innovation'],
            'creative_arts': ['inspiration', 'creativity', 'expression', 'beauty'],
            'finance': ['security', 'freedom', 'wealth', 'control'],
            'health_wellness': ['peace', 'balance', 'vitality', 'healing']
        }
        
        return triggers.get(business_niche, ['motivation', 'success', 'growth'])
        
    async def suggest_engagement_tactics(self, platform: str, business_niche: str) -> List[str]:
        """Suggest engagement tactics"""
        tactics = []
        
        # Platform-specific tactics
        if platform == 'instagram':
            tactics.extend(['Use carousel posts', 'Add interactive stickers', 'Create saveable content'])
        elif platform == 'tiktok':
            tactics.extend(['Use trending sounds', 'Create duet-able content', 'Jump on challenges'])
        elif platform == 'youtube':
            tactics.extend(['Add end screens', 'Create playlists', 'Use timestamps'])
            
        # Niche-specific tactics
        if business_niche == 'education':
            tactics.extend(['Create how-to guides', 'Share quick tips', 'Make checklists'])
        elif business_niche == 'fitness':
            tactics.extend(['Show before/after', 'Create workout challenges', 'Share progress updates'])
            
        return tactics[:5]
        
    async def generate_ad_copy(self, approach: str, business_niche: str, target_audience: Dict[str, Any], 
                               creative_brief: str, psychological_triggers: List[str], 
                               platform_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ad copy using AI"""
        prompt = f"""
        Create {approach} ad copy for a {business_niche} business.
        Target audience: {json.dumps(target_audience)}
        Brief: {creative_brief}
        Psychological triggers to use: {', '.join(psychological_triggers)}
        Platform: {list(platform_requirements.keys())[0] if platform_requirements else 'general'}
        
        Return JSON with: headline, body, cta
        """
        
        if self.openai_client:
            try:
                response = await asyncio.to_thread(
                    self.openai_client.chat.completions.create,
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    response_format={"type": "json_object"}
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                logger.error(f"Ad copy generation failed: {str(e)}")
                
        # Fallback copy
        return {
            'headline': f"Transform Your {business_niche.title()} Today",
            'body': f"Discover the proven {approach} approach that helps you achieve your goals faster.",
            'cta': "Start Now",
            'predicted_conversion_score': 0.7
        }
        
    async def generate_video_script(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate video script based on parameters"""
        business_niche = parameters['business_niche']
        video_length = parameters.get('video_length', 60)
        
        script = {
            'title': f"{business_niche.title()} Success Guide",
            'duration': video_length,
            'scenes': []
        }
        
        # Generate scene structure based on video length
        if video_length <= 30:
            # Short format (TikTok, Reels)
            script['scenes'] = [
                {
                    'type': 'text_overlay',
                    'duration': 3,
                    'content': 'Hook statement',
                    'dialogue': await self.generate_hook_dialogue(business_niche)
                },
                {
                    'type': 'talking_head',
                    'duration': 20,
                    'content': 'Main content',
                    'dialogue': await self.generate_main_content(business_niche, 'short')
                },
                {
                    'type': 'text_overlay',
                    'duration': 7,
                    'content': 'Call to action',
                    'dialogue': await self.generate_cta_dialogue(business_niche)
                }
            ]
        else:
            # Longer format
            script['scenes'] = [
                {
                    'type': 'animation',
                    'duration': 5,
                    'content': 'Intro animation',
                    'dialogue': ''
                },
                {
                    'type': 'talking_head',
                    'duration': 10,
                    'content': 'Introduction',
                    'dialogue': await self.generate_intro_dialogue(business_niche)
                },
                {
                    'type': 'screen_recording',
                    'duration': video_length - 25,
                    'content': 'Main demonstration',
                    'dialogue': await self.generate_main_content(business_niche, 'long')
                },
                {
                    'type': 'text_overlay',
                    'duration': 10,
                    'content': 'Summary and CTA',
                    'dialogue': await self.generate_summary_cta(business_niche)
                }
            ]
            
        script['mood'] = await self.determine_video_mood(business_niche)
        
        return script
        
    async def generate_hook_dialogue(self, business_niche: str) -> str:
        """Generate hook dialogue for video"""
        hooks = {
            'education': "Want to learn 10x faster? Here's how...",
            'fitness': "Transform your body in just 4 weeks!",
            'business_consulting': "The secret strategy that grew my business 300%",
            'creative_arts': "Turn your creativity into income",
            'finance': "Build wealth with this simple strategy",
            'health_wellness': "Feel amazing every single day"
        }
        
        return hooks.get(business_niche, "Discover the secret to success")
        
    async def generate_main_content(self, business_niche: str, length: str) -> str:
        """Generate main content for video"""
        if length == 'short':
            return f"Here are 3 quick tips for {business_niche} success: First, focus on consistency. Second, track your progress. Third, never stop learning."
        else:
            return f"In this video, we'll explore the fundamental principles of {business_niche} excellence. We'll cover proven strategies, common mistakes to avoid, and actionable steps you can take today."
            
    async def generate_intro_dialogue(self, business_niche: str) -> str:
        """Generate introduction dialogue"""
        return f"Welcome! Today we're diving deep into {business_niche} strategies that actually work."
        
    async def generate_cta_dialogue(self, business_niche: str) -> str:
        """Generate call-to-action dialogue"""
        return f"Ready to transform your {business_niche} journey? Click the link below to get started!"
        
    async def generate_summary_cta(self, business_niche: str) -> str:
        """Generate summary and CTA"""
        return f"To recap, we've covered the essential {business_niche} strategies. Now it's time to take action. Start implementing these tips today!"
        
    async def determine_video_mood(self, business_niche: str) -> str:
        """Determine video mood based on niche"""
        moods = {
            'education': 'inspiring',
            'fitness': 'energetic',
            'business_consulting': 'professional',
            'creative_arts': 'creative',
            'finance': 'confident',
            'health_wellness': 'calming'
        }
        
        return moods.get(business_niche, 'positive')
        
    async def generate_voiceover(self, text: str, voice: str, speed: float, emotion: str, emphasis_words: List[str]) -> Any:
        """Generate AI voiceover (placeholder)"""
        # This would integrate with TTS services like ElevenLabs
        logger.info(f"Generating voiceover: {text[:50]}... with voice {voice}")
        return None  # Placeholder
        
    async def generate_ai_avatar_video(self, script_text: str, avatar_style: str, background: str, duration: int) -> Any:
        """Generate AI avatar video (placeholder)"""
        # This would integrate with avatar generation services
        logger.info(f"Generating AI avatar video: {avatar_style} style, {duration}s duration")
        return None  # Placeholder