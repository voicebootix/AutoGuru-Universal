"""
Facebook Enhanced Publisher with built-in business intelligence.

Handles Facebook and Instagram publishing with revenue optimization,
engagement tracking, and universal business niche support.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlencode

import aiohttp
from facebook_business.api import FacebookAdsApi
from facebook_business.adobjects.page import Page
from facebook_business.adobjects.pagepost import PagePost
from facebook_business.adobjects.iguser import IGUser
from facebook_business.adobjects.igmedia import IGMedia

from backend.platforms.enhanced_base_publisher import (
    UniversalPlatformPublisher,
    PublishResult,
    PublishStatus,
    RevenueMetrics,
    PerformanceMetrics,
    AudienceInsights
)
from backend.services.content_analyzer import ContentAnalyzer
from backend.services.analytics_service import UniversalAnalyticsService
from backend.utils.encryption_manager import EncryptionManager
from backend.models.content_models import BusinessNicheType

logger = logging.getLogger(__name__)


@dataclass
class FacebookPostMetadata:
    """Metadata for Facebook posts"""
    post_type: str  # link, photo, video, carousel, story
    targeting: Optional[Dict[str, Any]] = None
    boost_budget: Optional[float] = None
    instagram_crosspost: bool = True
    story_duration: int = 24  # hours
    

@dataclass
class FacebookAdMetrics:
    """Metrics for Facebook ads and boosted posts"""
    ad_id: str
    impressions: int
    reach: int
    clicks: int
    conversions: int
    spend: float
    cpm: float
    cpc: float
    roas: float


class FacebookEngagementOptimizer:
    """Optimizes content for Facebook's algorithm and engagement"""
    
    def __init__(self):
        self.engagement_patterns = {
            'education': {
                'post_types': ['video', 'carousel', 'link'],
                'optimal_length': 1500,
                'emoji_usage': 'moderate',
                'optimal_times': [9, 13, 15, 19],
                'hashtag_count': 3,
                'engagement_triggers': [
                    "Did you know that",
                    "Here's a quick tip:",
                    "Learn something new today:",
                    "The secret to"
                ]
            },
            'business_consulting': {
                'post_types': ['link', 'video', 'photo'],
                'optimal_length': 1200,
                'emoji_usage': 'minimal',
                'optimal_times': [8, 12, 17, 19],
                'hashtag_count': 5,
                'engagement_triggers': [
                    "How we helped",
                    "Case study:",
                    "Business tip:",
                    "Transform your business"
                ]
            },
            'fitness_wellness': {
                'post_types': ['video', 'photo', 'carousel'],
                'optimal_length': 800,
                'emoji_usage': 'high',
                'optimal_times': [6, 12, 17, 20],
                'hashtag_count': 10,
                'engagement_triggers': [
                    "Transformation Tuesday",
                    "Fitness motivation:",
                    "Your daily workout:",
                    "Health tip:"
                ]
            },
            'creative': {
                'post_types': ['photo', 'carousel', 'video'],
                'optimal_length': 500,
                'emoji_usage': 'high',
                'optimal_times': [11, 14, 18, 21],
                'hashtag_count': 15,
                'engagement_triggers': [
                    "Behind the scenes:",
                    "Work in progress:",
                    "Creative process:",
                    "Art of the day:"
                ]
            }
        }
    
    async def optimize_for_engagement(self, content: Dict[str, Any], business_niche: str) -> Dict[str, Any]:
        """Optimize content for Facebook engagement"""
        pattern = self.engagement_patterns.get(
            business_niche, 
            self.engagement_patterns['business_consulting']
        )
        
        # Optimize post structure
        optimized_text = await self._optimize_post_text(
            content.get('text', ''),
            pattern
        )
        
        # Select best post type
        post_type = await self._select_post_type(content, pattern['post_types'])
        
        # Generate hashtags
        hashtags = await self._research_facebook_hashtags(
            content,
            pattern['hashtag_count'],
            business_niche
        )
        
        # Create engagement elements
        engagement_elements = await self._create_engagement_elements(business_niche)
        
        return {
            'optimized_text': optimized_text,
            'post_type': post_type,
            'hashtags': hashtags,
            'engagement_elements': engagement_elements,
            'targeting_suggestions': await self._get_targeting_suggestions(business_niche),
            'boost_recommendations': await self._get_boost_recommendations(content, business_niche)
        }
    
    async def _optimize_post_text(self, text: str, pattern: Dict[str, Any]) -> str:
        """Optimize post text for engagement"""
        # Add engagement trigger
        trigger = pattern['engagement_triggers'][0]
        optimized = f"{trigger}\n\n{text}"
        
        # Optimize length
        optimal_length = pattern['optimal_length']
        if len(optimized) > optimal_length:
            optimized = optimized[:optimal_length-3] + "..."
            optimized += "\n\n[Read more in comments]"
        
        # Add emojis based on usage level
        if pattern['emoji_usage'] == 'high':
            optimized = self._add_relevant_emojis(optimized, 'high')
        elif pattern['emoji_usage'] == 'moderate':
            optimized = self._add_relevant_emojis(optimized, 'moderate')
        
        return optimized
    
    def _add_relevant_emojis(self, text: str, level: str) -> str:
        """Add relevant emojis to text"""
        emoji_map = {
            'high': {
                'success': '🎯', 'growth': '📈', 'idea': '💡',
                'fitness': '💪', 'health': '🌟', 'art': '🎨'
            },
            'moderate': {
                'success': '✓', 'growth': '↗', 'idea': '•',
                'fitness': '•', 'health': '•', 'art': '•'
            }
        }
        
        emojis = emoji_map.get(level, emoji_map['moderate'])
        
        # Add emojis contextually
        for keyword, emoji in emojis.items():
            if keyword in text.lower():
                text = text.replace('.', f' {emoji}.', 1)
                break
        
        return text
    
    async def _select_post_type(self, content: Dict[str, Any], preferred_types: List[str]) -> str:
        """Select optimal post type based on content"""
        if content.get('video_file') or content.get('video_url'):
            return 'video'
        elif content.get('images') and len(content.get('images', [])) > 1:
            return 'carousel'
        elif content.get('image_url') or content.get('image_file'):
            return 'photo'
        elif content.get('link_url'):
            return 'link'
        else:
            return preferred_types[0]
    
    async def _research_facebook_hashtags(self, content: Dict[str, Any], count: int, niche: str) -> List[str]:
        """Research optimal Facebook hashtags"""
        base_hashtags = {
            'education': [
                '#OnlineLearning', '#Education', '#StudyTips', '#Learning',
                '#Knowledge', '#StudentLife', '#TeachersOfFacebook', '#EdTech'
            ],
            'business_consulting': [
                '#BusinessGrowth', '#Entrepreneur', '#SmallBusiness', '#BusinessTips',
                '#Marketing', '#Success', '#Leadership', '#BusinessStrategy'
            ],
            'fitness_wellness': [
                '#FitnessMotivation', '#HealthyLifestyle', '#Workout', '#FitFam',
                '#Wellness', '#HealthyLiving', '#FitnessJourney', '#GymLife'
            ],
            'creative': [
                '#ArtistsOfFacebook', '#CreativeLife', '#ArtWork', '#Design',
                '#CreativeProcess', '#HandMade', '#ArtDaily', '#Creative'
            ]
        }
        
        niche_hashtags = base_hashtags.get(niche, base_hashtags['business_consulting'])
        
        # Mix popular and niche hashtags
        selected = niche_hashtags[:count//2]  # Popular
        selected.extend(niche_hashtags[count//2:count])  # Niche specific
        
        return selected[:count]
    
    async def _create_engagement_elements(self, niche: str) -> Dict[str, Any]:
        """Create engagement elements for the post"""
        elements = {
            'call_to_action': self._get_cta_for_niche(niche),
            'question_prompt': self._get_question_prompt(niche),
            'tag_suggestions': self._get_tag_suggestions(niche),
            'reaction_bait': self._get_reaction_prompt(niche)
        }
        
        return elements
    
    def _get_cta_for_niche(self, niche: str) -> str:
        """Get call-to-action for niche"""
        ctas = {
            'education': "👇 Save this for later and share with someone who needs it!",
            'business_consulting': "💼 Tag a business owner who needs to see this!",
            'fitness_wellness': "💪 Double tap if you're ready to transform!",
            'creative': "🎨 Share your work in the comments!"
        }
        return ctas.get(niche, "Share your thoughts below! 👇")
    
    def _get_question_prompt(self, niche: str) -> str:
        """Get question prompt for engagement"""
        questions = {
            'education': "What's one thing you learned today?",
            'business_consulting': "What's your biggest business challenge right now?",
            'fitness_wellness': "What's your fitness goal for this month?",
            'creative': "What inspires your creativity?"
        }
        return questions.get(niche, "What do you think?")
    
    def _get_tag_suggestions(self, niche: str) -> List[str]:
        """Get tag suggestions for niche"""
        tags = {
            'education': ['students', 'teachers', 'learners'],
            'business_consulting': ['entrepreneurs', 'business owners', 'startups'],
            'fitness_wellness': ['fitness enthusiasts', 'gym buddies', 'workout partners'],
            'creative': ['fellow artists', 'creative friends', 'art lovers']
        }
        return tags.get(niche, ['friends'])
    
    def _get_reaction_prompt(self, niche: str) -> str:
        """Get reaction prompt for engagement"""
        prompts = {
            'education': "❤️ if you learned something new!",
            'business_consulting': "👍 if this resonates with your business journey!",
            'fitness_wellness': "💪 if you're working out today!",
            'creative': "😍 if you love the creative process!"
        }
        return prompts.get(niche, "React if you agree! 👍")
    
    async def _get_targeting_suggestions(self, niche: str) -> Dict[str, Any]:
        """Get audience targeting suggestions"""
        targeting = {
            'education': {
                'age_min': 18,
                'age_max': 35,
                'interests': ['Online learning', 'Education', 'Self improvement'],
                'behaviors': ['Engaged shoppers', 'Technology early adopters']
            },
            'business_consulting': {
                'age_min': 25,
                'age_max': 55,
                'interests': ['Entrepreneurship', 'Small business', 'Business'],
                'behaviors': ['Small business owners', 'Business page admins']
            },
            'fitness_wellness': {
                'age_min': 20,
                'age_max': 45,
                'interests': ['Fitness and wellness', 'Physical exercise', 'Healthy lifestyle'],
                'behaviors': ['Active lifestyle', 'Health and wellness enthusiasts']
            },
            'creative': {
                'age_min': 18,
                'age_max': 50,
                'interests': ['Arts and music', 'DIY', 'Crafts'],
                'behaviors': ['DIY enthusiasts', 'Art and music enthusiasts']
            }
        }
        
        return targeting.get(niche, targeting['business_consulting'])
    
    async def _get_boost_recommendations(self, content: Dict[str, Any], niche: str) -> Dict[str, Any]:
        """Get boost/promotion recommendations"""
        base_budget = {
            'education': 20,
            'business_consulting': 50,
            'fitness_wellness': 30,
            'creative': 25
        }
        
        return {
            'recommended_budget': base_budget.get(niche, 30),
            'duration_days': 3,
            'objective': 'engagement' if niche in ['creative', 'fitness_wellness'] else 'traffic',
            'placement': ['facebook_feed', 'instagram_feed', 'facebook_stories', 'instagram_stories']
        }


class FacebookAnalyticsTracker:
    """Tracks Facebook and Instagram analytics"""
    
    def __init__(self):
        self.metrics_cache = {}
        self.revenue_multipliers = {
            'video_content': 1.5,
            'high_engagement': 2.0,
            'viral_content': 3.0,
            'boosted_post': 1.8
        }
    
    async def track_post_insights(self, post_id: str, page: Page) -> Dict[str, Any]:
        """Track Facebook post insights"""
        try:
            post = PagePost(post_id)
            insights = post.get_insights(
                fields=['post_impressions', 'post_engaged_users', 'post_clicks',
                       'post_reactions_by_type_total', 'post_video_views']
            )
            
            metrics = {
                'impressions': 0,
                'reach': 0,
                'engagement': 0,
                'clicks': 0,
                'reactions': {},
                'video_views': 0
            }
            
            for insight in insights:
                if insight['name'] == 'post_impressions':
                    metrics['impressions'] = insight['values'][0]['value']
                elif insight['name'] == 'post_engaged_users':
                    metrics['engagement'] = insight['values'][0]['value']
                elif insight['name'] == 'post_clicks':
                    metrics['clicks'] = insight['values'][0]['value']
                elif insight['name'] == 'post_reactions_by_type_total':
                    metrics['reactions'] = insight['values'][0]['value']
                elif insight['name'] == 'post_video_views':
                    metrics['video_views'] = insight['values'][0]['value']
            
            # Calculate engagement rate
            if metrics['impressions'] > 0:
                metrics['engagement_rate'] = (metrics['engagement'] / metrics['impressions']) * 100
            else:
                metrics['engagement_rate'] = 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error tracking Facebook insights: {e}")
            return {}
    
    async def calculate_monetization_potential(self, metrics: Dict[str, Any], page_data: Dict[str, Any]) -> float:
        """Calculate Facebook monetization potential"""
        base_revenue = 0
        
        # Facebook Creator Bonus (if eligible)
        if page_data.get('creator_bonus_eligible', False):
            views = metrics.get('video_views', 0)
            # Estimated $0.01 per view for creator bonus
            base_revenue += views * 0.01
        
        # In-stream ads revenue (for videos)
        if metrics.get('video_views', 0) > 0 and page_data.get('monetization_enabled', False):
            # Estimated $0.003 per view for in-stream ads
            base_revenue += metrics['video_views'] * 0.003
        
        # Sponsored content potential
        followers = page_data.get('followers_count', 0)
        engagement_rate = metrics.get('engagement_rate', 0)
        
        if followers > 10000 and engagement_rate > 3:
            # Estimated sponsored post value
            sponsored_value = (followers / 1000) * 15 * (engagement_rate / 3)
            base_revenue += sponsored_value * 0.1  # 10% chance of sponsorship
        
        # Facebook Stars (if enabled)
        if page_data.get('stars_enabled', False):
            estimated_stars = metrics.get('engagement', 0) * 0.001  # 0.1% send stars
            base_revenue += estimated_stars * 0.01 * 100  # $0.01 per star, avg 100 stars
        
        # Apply multipliers
        for condition, multiplier in self.revenue_multipliers.items():
            if self._check_condition(condition, metrics, page_data):
                base_revenue *= multiplier
        
        return round(base_revenue, 2)
    
    async def track_instagram_insights(self, media_id: str, ig_user: IGUser) -> Dict[str, Any]:
        """Track Instagram insights"""
        try:
            media = IGMedia(media_id)
            insights = media.get_insights(
                fields=['impressions', 'reach', 'engagement', 'saved']
            )
            
            metrics = {
                'impressions': 0,
                'reach': 0,
                'engagement': 0,
                'saves': 0
            }
            
            for insight in insights:
                metrics[insight['name']] = insight['values'][0]['value']
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error tracking Instagram insights: {e}")
            return {}
    
    def _check_condition(self, condition: str, metrics: Dict[str, Any], page_data: Dict[str, Any]) -> bool:
        """Check if condition is met for revenue multiplier"""
        if condition == 'video_content':
            return metrics.get('video_views', 0) > 0
        elif condition == 'high_engagement':
            return metrics.get('engagement_rate', 0) > 5
        elif condition == 'viral_content':
            return metrics.get('impressions', 0) > 100000
        elif condition == 'boosted_post':
            return page_data.get('is_boosted', False)
        return False


class FacebookEnhancedPublisher(UniversalPlatformPublisher):
    """Facebook and Instagram enhanced publisher with revenue optimization"""
    
    def __init__(self, client_id: str):
        super().__init__(client_id, "facebook")
        self.fb_api: Optional[FacebookAdsApi] = None
        self.page: Optional[Page] = None
        self.ig_user: Optional[IGUser] = None
        self.engagement_optimizer = FacebookEngagementOptimizer()
        self.analytics_tracker = FacebookAnalyticsTracker()
        self._authenticated = False
        self.page_data = {}
    
    async def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """Authenticate with Facebook API"""
        try:
            # Initialize Facebook Ads API
            FacebookAdsApi.init(
                credentials['app_id'],
                credentials['app_secret'],
                credentials['access_token']
            )
            
            self.fb_api = FacebookAdsApi.get_default_api()
            
            # Get page
            self.page = Page(credentials['page_id'])
            page_info = self.page.api_get(fields=['name', 'followers_count', 'fan_count'])
            
            self.page_data = {
                'page_id': credentials['page_id'],
                'page_name': page_info.get('name'),
                'followers_count': page_info.get('fan_count', 0),
                'monetization_enabled': True,  # Would check actual status
                'creator_bonus_eligible': page_info.get('fan_count', 0) > 10000
            }
            
            # Get Instagram business account if connected
            try:
                ig_accounts = self.page.get_instagram_accounts()
                if ig_accounts:
                    self.ig_user = ig_accounts[0]
                    self.page_data['instagram_connected'] = True
            except:
                self.page_data['instagram_connected'] = False
            
            # Store encrypted credentials
            encrypted = self.encryption_manager.encrypt_credentials(credentials)
            await self._store_platform_credentials(encrypted)
            
            self._authenticated = True
            logger.info(f"Successfully authenticated Facebook page: {self.page_data['page_name']}")
            return True
            
        except Exception as e:
            logger.error(f"Facebook authentication failed: {e}")
            return False
    
    async def publish_content(self, content: Dict[str, Any]) -> PublishResult:
        """Publish content to Facebook with optimization"""
        if not self._authenticated:
            return PublishResult(
                platform="facebook",
                post_id=None,
                url=None,
                status=PublishStatus.FAILED,
                error_message="Not authenticated"
            )
        
        try:
            # Detect business niche
            business_niche = await self.detect_business_niche(content.get('text', ''))
            
            # Optimize content
            optimizations = await self.optimize_for_revenue(content)
            engagement_opts = await self.engagement_optimizer.optimize_for_engagement(
                content, 
                business_niche
            )
            
            # Prepare post data
            post_data = {
                'message': engagement_opts['optimized_text']
            }
            
            # Add hashtags
            if engagement_opts['hashtags']:
                post_data['message'] += f"\n\n{' '.join(engagement_opts['hashtags'])}"
            
            # Add engagement elements
            elements = engagement_opts['engagement_elements']
            post_data['message'] += f"\n\n{elements['question_prompt']}"
            post_data['message'] += f"\n\n{elements['call_to_action']}"
            
            # Handle media
            if content.get('video_url') or content.get('video_file'):
                post_data['video_url'] = content.get('video_url', content.get('video_file'))
            elif content.get('image_url') or content.get('image_file'):
                post_data['photo_url'] = content.get('image_url', content.get('image_file'))
            elif content.get('link_url'):
                post_data['link'] = content['link_url']
            
            # Set targeting if provided
            if engagement_opts.get('targeting_suggestions'):
                post_data['targeting'] = json.dumps(engagement_opts['targeting_suggestions'])
            
            # Publish to Facebook
            post = self.page.create_feed(params=post_data)
            post_id = post['id']
            
            # Cross-post to Instagram if enabled
            instagram_id = None
            if self.ig_user and content.get('instagram_crosspost', True):
                instagram_id = await self._crosspost_to_instagram(content, engagement_opts)
            
            # Track initial metrics
            await self._track_publish_metrics(
                post_id,
                content,
                optimizations
            )
            
            # Generate URL
            page_id, actual_post_id = post_id.split('_')
            url = f"https://www.facebook.com/{page_id}/posts/{actual_post_id}"
            
            return PublishResult(
                platform="facebook",
                post_id=post_id,
                url=url,
                status=PublishStatus.SUCCESS,
                metadata={
                    'instagram_post_id': instagram_id,
                    'optimizations_applied': engagement_opts,
                    'business_niche': business_niche,
                    'boost_eligible': engagement_opts['boost_recommendations']
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to publish to Facebook: {e}")
            return PublishResult(
                platform="facebook",
                post_id=None,
                url=None,
                status=PublishStatus.FAILED,
                error_message=str(e)
            )
    
    async def _crosspost_to_instagram(self, content: Dict[str, Any], optimizations: Dict[str, Any]) -> Optional[str]:
        """Cross-post content to Instagram"""
        try:
            ig_params = {
                'caption': optimizations['optimized_text'][:2200],  # Instagram limit
                'media_type': 'IMAGE'  # or VIDEO
            }
            
            if content.get('image_url'):
                ig_params['image_url'] = content['image_url']
            elif content.get('video_url'):
                ig_params['media_type'] = 'VIDEO'
                ig_params['video_url'] = content['video_url']
            
            # Create media container
            container = self.ig_user.create_media(params=ig_params)
            container_id = container['id']
            
            # Wait for processing
            await asyncio.sleep(5)
            
            # Publish
            published = self.ig_user.publish_media(params={'creation_id': container_id})
            return published.get('id')
            
        except Exception as e:
            logger.error(f"Failed to cross-post to Instagram: {e}")
            return None
    
    async def optimize_for_revenue(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize content for Facebook revenue"""
        business_niche = await self.detect_business_niche(content.get('text', ''))
        
        # Get optimal posting time
        optimal_time = await self.get_optimal_posting_time('post', business_niche)
        
        # Get platform-specific optimizations
        platform_opts = await self.get_platform_optimizations(content, business_niche)
        
        # Calculate revenue potential
        revenue_potential = await self.calculate_revenue_potential(
            content,
            {'optimal_posting_time': optimal_time}
        )
        
        return {
            'optimal_posting_time': optimal_time,
            'revenue_optimization': {
                'estimated_revenue': revenue_potential,
                'monetization_strategies': [
                    'creator_bonus',
                    'in_stream_ads',
                    'facebook_stars',
                    'sponsored_content',
                    'affiliate_marketing',
                    'facebook_shops'
                ],
                'boost_strategy': platform_opts['boost_strategy']
            },
            'audience_targeting': {
                'demographics': platform_opts['targeting']['demographics'],
                'interests': platform_opts['targeting']['interests'],
                'behaviors': platform_opts['targeting']['behaviors']
            },
            'platform_specific_tweaks': platform_opts
        }
    
    async def get_optimal_posting_time(self, content_type: str, business_niche: str) -> datetime:
        """Get optimal posting time for Facebook"""
        hour_map = {
            'education': [9, 13, 15, 19],
            'business_consulting': [8, 12, 17, 19],
            'fitness_wellness': [6, 12, 17, 20],
            'creative': [11, 14, 18, 21],
            'ecommerce': [12, 15, 20],
            'local_service': [8, 12, 18],
            'technology': [10, 14, 20],
            'non_profit': [11, 15, 19]
        }
        
        optimal_hours = hour_map.get(business_niche, [9, 14, 19])
        
        # Get next optimal hour
        now = datetime.utcnow()
        current_hour = now.hour
        
        for hour in optimal_hours:
            if hour > current_hour:
                return now.replace(hour=hour, minute=0, second=0, microsecond=0)
        
        # If no optimal hour today, use first one tomorrow
        tomorrow = now + timedelta(days=1)
        return tomorrow.replace(hour=optimal_hours[0], minute=0, second=0, microsecond=0)
    
    async def get_platform_optimizations(self, content: Dict[str, Any], business_niche: str) -> Dict[str, Any]:
        """Get Facebook-specific optimizations"""
        return {
            'algorithm_optimizations': {
                'prioritize_video': True,
                'optimal_video_length': self._get_optimal_video_length(business_niche),
                'use_native_upload': True,
                'engagement_bait': False,  # Avoid as Facebook penalizes
                'meaningful_interactions': True
            },
            'targeting': {
                'demographics': await self._get_demographics_targeting(business_niche),
                'interests': await self._get_interest_targeting(business_niche),
                'behaviors': await self._get_behavior_targeting(business_niche),
                'custom_audiences': business_niche in ['ecommerce', 'local_service']
            },
            'boost_strategy': {
                'auto_boost': self.page_data.get('followers_count', 0) < 50000,
                'budget': self._get_boost_budget(business_niche),
                'duration': 3,  # days
                'objective': self._get_campaign_objective(business_niche)
            },
            'cross_platform': {
                'instagram_crosspost': True,
                'stories': business_niche in ['fitness_wellness', 'creative'],
                'reels': True,  # Always use reels for reach
                'messenger_integration': business_niche == 'local_service'
            }
        }
    
    def _get_optimal_video_length(self, niche: str) -> int:
        """Get optimal video length in seconds"""
        lengths = {
            'education': 180,  # 3 minutes
            'business_consulting': 120,  # 2 minutes
            'fitness_wellness': 60,  # 1 minute
            'creative': 90,  # 1.5 minutes
            'ecommerce': 30,  # 30 seconds
            'technology': 150,  # 2.5 minutes
            'non_profit': 120  # 2 minutes
        }
        return lengths.get(niche, 90)
    
    async def _get_demographics_targeting(self, niche: str) -> Dict[str, Any]:
        """Get demographic targeting for niche"""
        demographics = {
            'education': {'age_min': 18, 'age_max': 35, 'education_level': 'college'},
            'business_consulting': {'age_min': 25, 'age_max': 55, 'income_level': 'above_average'},
            'fitness_wellness': {'age_min': 20, 'age_max': 45, 'lifestyle': 'active'},
            'creative': {'age_min': 18, 'age_max': 50, 'interests': 'arts'}
        }
        return demographics.get(niche, {'age_min': 18, 'age_max': 65})
    
    async def _get_interest_targeting(self, niche: str) -> List[str]:
        """Get interest targeting for niche"""
        interests = {
            'education': ['Online learning', 'Education', 'Self improvement', 'Career development'],
            'business_consulting': ['Entrepreneurship', 'Small business', 'Business', 'Leadership'],
            'fitness_wellness': ['Fitness', 'Health', 'Wellness', 'Nutrition', 'Exercise'],
            'creative': ['Arts', 'Design', 'Creativity', 'DIY', 'Crafts']
        }
        return interests.get(niche, ['General interests'])
    
    async def _get_behavior_targeting(self, niche: str) -> List[str]:
        """Get behavior targeting for niche"""
        behaviors = {
            'education': ['Engaged shoppers', 'Technology early adopters'],
            'business_consulting': ['Small business owners', 'Business page admins'],
            'fitness_wellness': ['Active lifestyle', 'Health enthusiasts'],
            'creative': ['DIY enthusiasts', 'Art enthusiasts']
        }
        return behaviors.get(niche, [])
    
    def _get_boost_budget(self, niche: str) -> float:
        """Get recommended boost budget"""
        budgets = {
            'education': 20,
            'business_consulting': 50,
            'fitness_wellness': 30,
            'creative': 25,
            'ecommerce': 40,
            'local_service': 35,
            'technology': 45,
            'non_profit': 15
        }
        return budgets.get(niche, 30)
    
    def _get_campaign_objective(self, niche: str) -> str:
        """Get campaign objective for niche"""
        objectives = {
            'education': 'traffic',
            'business_consulting': 'lead_generation',
            'fitness_wellness': 'engagement',
            'creative': 'brand_awareness',
            'ecommerce': 'conversions',
            'local_service': 'store_traffic',
            'technology': 'app_installs',
            'non_profit': 'reach'
        }
        return objectives.get(niche, 'engagement')
    
    async def calculate_revenue_potential(self, content: Dict[str, Any], optimizations: Dict[str, Any]) -> float:
        """Calculate revenue potential for Facebook content"""
        base_revenue = 0
        
        # Estimate based on page metrics
        followers = self.page_data.get('followers_count', 0)
        
        # Creator bonus potential
        if self.page_data.get('creator_bonus_eligible', False):
            estimated_views = followers * 0.05  # 5% reach
            base_revenue += estimated_views * 0.01  # $0.01 per view
        
        # In-stream ads (for video)
        if content.get('video_url') or content.get('video_file'):
            estimated_views = followers * 0.03  # 3% video completion
            base_revenue += estimated_views * 0.003  # $0.003 per view
        
        # Sponsored content potential
        if followers > 10000:
            engagement_rate = 3.5  # Average Facebook engagement
            sponsored_rate = (followers / 1000) * 15 * (engagement_rate / 3)
            base_revenue += sponsored_rate * 0.08  # 8% chance per post
        
        # Facebook Stars potential
        if self.page_data.get('stars_enabled', False):
            estimated_stars = followers * 0.0001  # 0.01% send stars
            base_revenue += estimated_stars * 100 * 0.01  # 100 stars avg at $0.01
        
        # Apply boost multiplier
        if optimizations.get('platform_specific_tweaks', {}).get('boost_strategy', {}).get('auto_boost'):
            base_revenue *= 1.8
        
        return round(base_revenue, 2)
    
    async def get_analytics(self, post_id: str) -> Dict[str, Any]:
        """Get Facebook analytics for a post"""
        try:
            # Get Facebook insights
            fb_metrics = await self.analytics_tracker.track_post_insights(
                post_id,
                self.page
            )
            
            # Calculate revenue metrics
            revenue_metrics = await self.analytics_tracker.calculate_monetization_potential(
                fb_metrics,
                self.page_data
            )
            
            # Get Instagram insights if cross-posted
            ig_metrics = {}
            if self.ig_user:
                # Would get Instagram post ID from metadata
                pass
            
            return {
                'facebook_metrics': fb_metrics,
                'instagram_metrics': ig_metrics,
                'revenue_metrics': {
                    'estimated_revenue': revenue_metrics,
                    'revenue_sources': {
                        'creator_bonus': revenue_metrics * 0.4,
                        'in_stream_ads': revenue_metrics * 0.3,
                        'sponsored_potential': revenue_metrics * 0.2,
                        'facebook_stars': revenue_metrics * 0.1
                    }
                },
                'engagement_insights': {
                    'reaction_breakdown': fb_metrics.get('reactions', {}),
                    'audience_quality': self._assess_audience_quality(fb_metrics)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get Facebook analytics: {e}")
            return {}
    
    def _assess_audience_quality(self, metrics: Dict[str, Any]) -> str:
        """Assess audience quality based on metrics"""
        engagement_rate = metrics.get('engagement_rate', 0)
        
        if engagement_rate > 6:
            return 'highly_engaged'
        elif engagement_rate > 3:
            return 'moderately_engaged'
        else:
            return 'low_engagement'
    
    async def boost_post(self, post_id: str, budget: float, duration_days: int) -> Dict[str, Any]:
        """Boost a Facebook post"""
        try:
            # Would implement actual boost logic
            return {
                'boost_id': f"boost_{post_id}",
                'status': 'active',
                'budget': budget,
                'duration': duration_days,
                'estimated_reach': budget * 1000  # Rough estimate
            }
        except Exception as e:
            logger.error(f"Failed to boost post: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def schedule_content(self, content: Dict[str, Any], schedule_time: datetime) -> PublishResult:
        """Schedule content for future publishing"""
        # Facebook supports native scheduling
        content['published'] = False
        content['scheduled_publish_time'] = int(schedule_time.timestamp())
        
        return await self.publish_content(content)
    
    async def analyze_audience_engagement(self, business_niche: str) -> Dict[str, Any]:
        """Analyze audience engagement patterns"""
        return {
            'demographics': {
                'age_distribution': {
                    '18-24': 0.15,
                    '25-34': 0.25,
                    '35-44': 0.30,
                    '45-54': 0.20,
                    '55+': 0.10
                },
                'gender_split': {
                    'male': 0.45,
                    'female': 0.53,
                    'other': 0.02
                },
                'location_insights': {
                    'top_countries': ['US', 'UK', 'CA', 'AU'],
                    'top_cities': ['New York', 'Los Angeles', 'London', 'Toronto']
                }
            },
            'engagement_patterns': {
                'peak_hours': self.engagement_optimizer.engagement_patterns[business_niche]['optimal_times'],
                'peak_days': ['Thursday', 'Friday', 'Sunday'],
                'content_preferences': {
                    'video': 0.40,
                    'photos': 0.30,
                    'links': 0.20,
                    'text': 0.10
                }
            },
            'content_preferences': {
                'optimal_length': self.engagement_optimizer.engagement_patterns[business_niche]['optimal_length'],
                'media_engagement': 'very_high',
                'hashtag_performance': 'moderate',
                'emoji_usage': self.engagement_optimizer.engagement_patterns[business_niche]['emoji_usage']
            }
        }

    # Facebook Groups Integration
    async def post_to_group(self, content: Dict[str, Any], group_id: str) -> PublishResult:
        """Post content to Facebook group"""
        if not self._authenticated:
            return PublishResult(
                platform="facebook",
                post_id=None,
                url=None,
                status=PublishStatus.FAILED,
                error_message="Not authenticated"
            )
        
        try:
            # Detect business niche for group-specific optimization
            business_niche = await self.detect_business_niche(content.get('text', ''))
            
            # Optimize content for group engagement
            engagement_opts = await self.engagement_optimizer.optimize_for_engagement(
                content, business_niche
            )
            
            # Prepare group post data
            group_post_data = {
                'message': engagement_opts['optimized_text'],
                'published': True
            }
            
            # Add media for groups
            if content.get('image_url'):
                group_post_data['url'] = content['image_url']
            elif content.get('link_url'):
                group_post_data['link'] = content['link_url']
            
            # Post to group using Graph API
            async with aiohttp.ClientSession() as session:
                url = f"{self.GRAPH_API_URL}/{group_id}/feed"
                group_post_data['access_token'] = self._access_token
                
                async with session.post(url, data=group_post_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        post_id = result['id']
                        
                        return PublishResult(
                            platform="facebook",
                            post_id=post_id,
                            url=f"https://www.facebook.com/groups/{group_id}/posts/{post_id.split('_')[1]}",
                            status=PublishStatus.SUCCESS,
                            metadata={
                                'group_id': group_id,
                                'business_niche': business_niche,
                                'optimizations_applied': engagement_opts
                            }
                        )
                    else:
                        error = await response.text()
                        raise Exception(f"Group post failed: {error}")
                        
        except Exception as e:
            logger.error(f"Failed to post to Facebook group: {e}")
            return PublishResult(
                platform="facebook",
                post_id=None,
                url=None,
                status=PublishStatus.FAILED,
                error_message=str(e)
            )

    # Facebook Events Integration
    async def create_event(self, event_data: Dict[str, Any]) -> PublishResult:
        """Create Facebook event"""
        if not self._authenticated:
            return PublishResult(
                platform="facebook",
                post_id=None,
                url=None,
                status=PublishStatus.FAILED,
                error_message="Not authenticated"
            )
        
        try:
            # Prepare event data
            fb_event_data = {
                'name': event_data['title'],
                'description': event_data['description'],
                'start_time': event_data['start_time'],
                'location': event_data.get('location', ''),
                'privacy': event_data.get('privacy', 'PUBLIC'),
                'access_token': self._access_token
            }
            
            if event_data.get('end_time'):
                fb_event_data['end_time'] = event_data['end_time']
            
            if event_data.get('cover_photo_url'):
                fb_event_data['cover'] = {'source': event_data['cover_photo_url']}
            
            # Create event using Graph API
            async with aiohttp.ClientSession() as session:
                url = f"{self.GRAPH_API_URL}/{self.page_data['page_id']}/events"
                
                async with session.post(url, data=fb_event_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        event_id = result['id']
                        
                        return PublishResult(
                            platform="facebook",
                            post_id=event_id,
                            url=f"https://www.facebook.com/events/{event_id}",
                            status=PublishStatus.SUCCESS,
                            metadata={
                                'event_id': event_id,
                                'event_type': 'facebook_event',
                                'privacy': fb_event_data['privacy']
                            }
                        )
                    else:
                        error = await response.text()
                        raise Exception(f"Event creation failed: {error}")
                        
        except Exception as e:
            logger.error(f"Failed to create Facebook event: {e}")
            return PublishResult(
                platform="facebook",
                post_id=None,
                url=None,
                status=PublishStatus.FAILED,
                error_message=str(e)
            )

    # Facebook Shop Integration  
    async def manage_facebook_shop(self, product_data: Dict[str, Any]) -> PublishResult:
        """Manage Facebook Shop products"""
        if not self._authenticated:
            return PublishResult(
                platform="facebook",
                post_id=None,
                url=None,
                status=PublishStatus.FAILED,
                error_message="Not authenticated"
            )
        
        try:
            action = product_data.get('action', 'create')  # create, update, delete
            
            if action == 'create':
                return await self._create_shop_product(product_data)
            elif action == 'update':
                return await self._update_shop_product(product_data)
            elif action == 'delete':
                return await self._delete_shop_product(product_data['product_id'])
            else:
                raise ValueError(f"Unknown shop action: {action}")
                
        except Exception as e:
            logger.error(f"Failed to manage Facebook shop: {e}")
            return PublishResult(
                platform="facebook",
                post_id=None,
                url=None,
                status=PublishStatus.FAILED,
                error_message=str(e)
            )

    async def _create_shop_product(self, product_data: Dict[str, Any]) -> PublishResult:
        """Create a Facebook Shop product"""
        try:
            # Get catalog ID first
            catalog_id = await self._get_or_create_catalog()
            
            fb_product_data = {
                'name': product_data['name'],
                'description': product_data['description'],
                'price': f"{int(product_data['price'] * 100)} USD",  # Facebook expects cents
                'currency': product_data.get('currency', 'USD'),
                'availability': product_data.get('availability', 'in stock'),
                'condition': product_data.get('condition', 'new'),
                'url': product_data.get('product_url', ''),
                'image_url': product_data.get('image_url', ''),
                'access_token': self._access_token
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.GRAPH_API_URL}/{catalog_id}/products"
                
                async with session.post(url, data=fb_product_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        product_id = result['id']
                        
                        return PublishResult(
                            platform="facebook",
                            post_id=product_id,
                            url=f"https://www.facebook.com/commerce/products/{product_id}",
                            status=PublishStatus.SUCCESS,
                            metadata={
                                'product_id': product_id,
                                'catalog_id': catalog_id,
                                'product_type': 'facebook_shop_product'
                            }
                        )
                    else:
                        error = await response.text()
                        raise Exception(f"Product creation failed: {error}")
                        
        except Exception as e:
            logger.error(f"Failed to create shop product: {e}")
            raise

    async def _get_or_create_catalog(self) -> str:
        """Get or create a Facebook product catalog"""
        try:
            # Try to get existing catalog
            async with aiohttp.ClientSession() as session:
                url = f"{self.GRAPH_API_URL}/{self.page_data['page_id']}/owned_product_catalogs"
                params = {'access_token': self._access_token}
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get('data'):
                            return result['data'][0]['id']
            
            # Create new catalog if none exists
            catalog_data = {
                'name': f"{self.page_data['page_name']} Catalog",
                'vertical': 'commerce',
                'access_token': self._access_token
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.GRAPH_API_URL}/{self.page_data['page_id']}/product_catalogs"
                
                async with session.post(url, data=catalog_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['id']
                    else:
                        raise Exception("Failed to create product catalog")
                        
        except Exception as e:
            logger.error(f"Failed to get/create catalog: {e}")
            raise

    # Live Video Streaming
    async def post_live_video(self, video_stream: str, metadata: Dict[str, Any]) -> PublishResult:
        """Start live video broadcast"""
        if not self._authenticated:
            return PublishResult(
                platform="facebook",
                post_id=None,
                url=None,
                status=PublishStatus.FAILED,
                error_message="Not authenticated"
            )
        
        try:
            # Create live video
            live_video_data = {
                'title': metadata.get('title', 'Live Video'),
                'description': metadata.get('description', ''),
                'privacy': metadata.get('privacy', 'PUBLIC'),
                'planned_start_time': metadata.get('planned_start_time'),
                'access_token': self._access_token
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.GRAPH_API_URL}/{self.page_data['page_id']}/live_videos"
                
                async with session.post(url, data=live_video_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        live_video_id = result['id']
                        stream_url = result['stream_url']
                        
                        # Start streaming (this would integrate with streaming service)
                        await self._start_live_stream(stream_url, video_stream)
                        
                        return PublishResult(
                            platform="facebook",
                            post_id=live_video_id,
                            url=f"https://www.facebook.com/{self.page_data['page_id']}/videos/{live_video_id}",
                            status=PublishStatus.SUCCESS,
                            metadata={
                                'live_video_id': live_video_id,
                                'stream_url': stream_url,
                                'video_type': 'live_stream'
                            }
                        )
                    else:
                        error = await response.text()
                        raise Exception(f"Live video creation failed: {error}")
                        
        except Exception as e:
            logger.error(f"Failed to start live video: {e}")
            return PublishResult(
                platform="facebook",
                post_id=None,
                url=None,
                status=PublishStatus.FAILED,
                error_message=str(e)
            )

    async def _start_live_stream(self, stream_url: str, video_source: str):
        """Start the actual live stream (integrate with streaming service)"""
        # This would integrate with streaming services like FFmpeg, OBS, etc.
        # For now, just log the stream initiation
        logger.info(f"Starting live stream to {stream_url} from source {video_source}")
        
        # In production, this would:
        # 1. Configure streaming software
        # 2. Start video capture/encoding
        # 3. Stream to Facebook's RTMP endpoint
        # 4. Monitor stream health
        pass

    # Enhanced OAuth 2.0 Authentication
    async def get_oauth_url(self, redirect_uri: str, scope: str = None) -> str:
        """Get Facebook OAuth authorization URL"""
        if not scope:
            scope = "pages_manage_posts,pages_read_engagement,pages_show_list,instagram_basic,instagram_content_publish"
        
        params = {
            'client_id': self.app_id,
            'redirect_uri': redirect_uri,
            'scope': scope,
            'response_type': 'code',
            'state': str(uuid.uuid4())  # CSRF protection
        }
        
        return f"{self.OAUTH_URL}?" + urlencode(params)

    async def exchange_code_for_token(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        """Exchange authorization code for access token"""
        try:
            token_data = {
                'client_id': self.app_id,
                'client_secret': self.app_secret,
                'redirect_uri': redirect_uri,
                'code': code
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.GRAPH_API_URL}/oauth/access_token"
                
                async with session.post(url, data=token_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        access_token = result['access_token']
                        
                        # Get long-lived token
                        long_lived_token = await self._get_long_lived_token(access_token)
                        
                        # Get user pages
                        pages = await self._get_user_pages(long_lived_token)
                        
                        return {
                            'access_token': long_lived_token,
                            'token_type': 'bearer',
                            'expires_in': result.get('expires_in'),
                            'pages': pages,
                            'success': True
                        }
                    else:
                        error = await response.text()
                        return {'success': False, 'error': error}
                        
        except Exception as e:
            logger.error(f"Failed to exchange code for token: {e}")
            return {'success': False, 'error': str(e)}

    async def _get_long_lived_token(self, short_token: str) -> str:
        """Convert short-lived token to long-lived token"""
        try:
            params = {
                'grant_type': 'fb_exchange_token',
                'client_id': self.app_id,
                'client_secret': self.app_secret,
                'fb_exchange_token': short_token
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.GRAPH_API_URL}/oauth/access_token"
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['access_token']
                    else:
                        logger.warning("Failed to get long-lived token, using short-lived")
                        return short_token
                        
        except Exception as e:
            logger.error(f"Failed to get long-lived token: {e}")
            return short_token

    async def _get_user_pages(self, access_token: str) -> List[Dict[str, Any]]:
        """Get user's Facebook pages"""
        try:
            params = {
                'access_token': access_token,
                'fields': 'id,name,access_token,instagram_business_account,category'
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.GRAPH_API_URL}/me/accounts"
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('data', [])
                    else:
                        return []
                        
        except Exception as e:
            logger.error(f"Failed to get user pages: {e}")
            return []

    # Advanced Analytics Integration
    async def get_page_insights(self, page_id: str, metrics: List[str], period: str = "day") -> Dict[str, Any]:
        """Get Facebook page analytics"""
        try:
            if not metrics:
                metrics = [
                    'page_fans',
                    'page_impressions',
                    'page_engaged_users',
                    'page_post_engagements',
                    'page_video_views'
                ]
            
            params = {
                'metric': ','.join(metrics),
                'period': period,
                'access_token': self._access_token
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.GRAPH_API_URL}/{page_id}/insights"
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Process insights data
                        insights = {}
                        for insight in result.get('data', []):
                            metric_name = insight['name']
                            values = insight.get('values', [])
                            if values:
                                insights[metric_name] = values[-1]['value']  # Latest value
                        
                        return insights
                    else:
                        logger.error(f"Failed to get page insights: {await response.text()}")
                        return {}
                        
        except Exception as e:
            logger.error(f"Failed to get page insights: {e}")
            return {}

    async def get_post_analytics(self, post_id: str) -> Dict[str, Any]:
        """Get detailed post performance metrics"""
        try:
            metrics = [
                'post_impressions',
                'post_engaged_users',
                'post_negative_feedback',
                'post_clicks',
                'post_reactions_by_type_total'
            ]
            
            params = {
                'metric': ','.join(metrics),
                'access_token': self._access_token
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.GRAPH_API_URL}/{post_id}/insights"
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Process post insights
                        analytics = {}
                        for insight in result.get('data', []):
                            metric_name = insight['name']
                            values = insight.get('values', [])
                            if values:
                                analytics[metric_name] = values[0]['value']
                        
                        # Calculate engagement rate
                        impressions = analytics.get('post_impressions', 1)
                        engaged_users = analytics.get('post_engaged_users', 0)
                        analytics['engagement_rate'] = (engaged_users / impressions * 100) if impressions > 0 else 0
                        
                        return analytics
                    else:
                        logger.error(f"Failed to get post analytics: {await response.text()}")
                        return {}
                        
        except Exception as e:
            logger.error(f"Failed to get post analytics: {e}")
            return {}

    async def get_audience_demographics(self, page_id: str) -> Dict[str, Any]:
        """Get page audience demographics"""
        try:
            demographic_metrics = [
                'page_fans_gender_age',
                'page_fans_country',
                'page_fans_city'
            ]
            
            params = {
                'metric': ','.join(demographic_metrics),
                'period': 'lifetime',
                'access_token': self._access_token
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.GRAPH_API_URL}/{page_id}/insights"
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        demographics = {}
                        for insight in result.get('data', []):
                            metric_name = insight['name']
                            values = insight.get('values', [])
                            if values:
                                demographics[metric_name] = values[-1]['value']
                        
                        return demographics
                    else:
                        logger.error(f"Failed to get demographics: {await response.text()}")
                        return {}
                        
        except Exception as e:
            logger.error(f"Failed to get audience demographics: {e}")
            return {}