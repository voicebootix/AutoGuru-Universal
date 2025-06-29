"""
TikTok Enhanced Platform Publisher for AutoGuru Universal.

This module provides TikTok publishing capabilities with viral optimization,
creator monetization features, and trend analysis. It works universally
across all business niches without hardcoded business logic.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
import re
import random

from backend.platforms.enhanced_base_publisher import (
    UniversalPlatformPublisher,
    PublishResult,
    PublishStatus,
    RevenueMetrics,
    PerformanceMetrics
)
from backend.models.content_models import BusinessNicheType

logger = logging.getLogger(__name__)


@dataclass
class TikTokVideoMetadata:
    """TikTok video metadata"""
    description: str
    video_file: str
    cover_image: Optional[str] = None
    music_id: Optional[str] = None
    hashtags: Optional[List[str]] = None
    mentions: Optional[List[str]] = None
    privacy_level: str = "public"  # public, friends, private
    allow_comments: bool = True
    allow_duet: bool = True
    allow_stitch: bool = True


@dataclass
class ViralMetrics:
    """Viral performance metrics"""
    viral_score: float  # 0-100
    viral_velocity: float  # Views per hour growth rate
    share_rate: float
    completion_rate: float
    loop_count: int
    trend_alignment_score: float
    estimated_reach: int
    viral_peak_time: Optional[datetime] = None


class TikTokViralOptimizer:
    """Optimize content for TikTok virality"""
    
    VIRAL_CONTENT_PATTERNS = {
        "hook_types": [
            "question_hook",
            "controversial_statement",
            "visual_surprise",
            "relatable_moment",
            "transformation_reveal"
        ],
        "engagement_triggers": [
            "comment_bait",
            "share_worthy",
            "save_for_later",
            "watch_again",
            "duet_this"
        ],
        "trending_formats": [
            "tutorial",
            "before_after",
            "day_in_life",
            "challenge",
            "storytelling"
        ]
    }
    
    NICHE_VIRAL_STRATEGIES = {
        "education": {
            "formats": ["quick_tips", "mini_lessons", "study_with_me"],
            "hooks": ["Did you know...", "The secret to...", "Nobody talks about..."],
            "duration": (15, 30)
        },
        "business_consulting": {
            "formats": ["business_tips", "entrepreneur_journey", "success_stories"],
            "hooks": ["How I made $X...", "The mistake that cost...", "My biggest lesson..."],
            "duration": (30, 60)
        },
        "fitness_wellness": {
            "formats": ["workout_routine", "transformation", "form_check"],
            "hooks": ["30 day results...", "Common mistakes...", "Try this instead..."],
            "duration": (15, 45)
        },
        "creative": {
            "formats": ["process_video", "speed_art", "behind_scenes"],
            "hooks": ["Watch me create...", "From sketch to...", "The making of..."],
            "duration": (15, 60)
        }
    }
    
    async def optimize_for_algorithm(
        self,
        content: Dict[str, Any],
        business_niche: str
    ) -> Dict[str, Any]:
        """Optimize content for TikTok's algorithm"""
        strategy = self.NICHE_VIRAL_STRATEGIES.get(
            business_niche,
            {
                "formats": ["general_content"],
                "hooks": ["Check this out..."],
                "duration": (15, 60)
            }
        )
        
        optimizations = {
            'video_optimizations': {
                'ideal_duration': strategy['duration'],
                'hook_timing': 'First 3 seconds are crucial',
                'loop_potential': 'End with a reason to rewatch',
                'visual_pacing': 'Change scenes every 3-5 seconds'
            },
            'engagement_optimizations': {
                'caption_strategy': 'Ask a question or create curiosity',
                'hashtag_mix': '3-5 niche + 2-3 trending + 1-2 branded',
                'cta_placement': 'Middle or end of video',
                'comment_strategy': 'Reply to early comments to boost engagement'
            },
            'trend_alignment': {
                'audio_selection': 'Use trending sounds in your niche',
                'effect_usage': 'Apply popular effects subtly',
                'challenge_participation': 'Join relevant challenges',
                'timing_relevance': 'Post when trends are peaking'
            }
        }
        
        return optimizations
    
    def calculate_viral_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate viral potential score"""
        # Weighted factors for virality
        weights = {
            'completion_rate': 0.3,
            'share_rate': 0.25,
            'comment_rate': 0.2,
            'like_rate': 0.15,
            'save_rate': 0.1
        }
        
        score = 0
        for factor, weight in weights.items():
            score += metrics.get(factor, 0) * weight * 100
            
        return min(score, 100)  # Cap at 100


class TikTokAnalyticsTracker:
    """Track TikTok-specific analytics and monetization"""
    
    def __init__(self, tiktok_api):
        self.tiktok_api = tiktok_api
    
    async def get_video_analytics(self, video_id: str) -> Dict[str, Any]:
        """Get comprehensive video analytics"""
        try:
            # In production, fetch from TikTok API
            # Simulated analytics for demonstration
            views = random.randint(10000, 100000)
            likes = int(views * random.uniform(0.08, 0.15))
            comments = int(views * random.uniform(0.01, 0.03))
            shares = int(views * random.uniform(0.02, 0.05))
            
            return {
                'views': views,
                'likes': likes,
                'comments': comments,
                'shares': shares,
                'completion_rate': random.uniform(0.4, 0.8),
                'average_watch_time': random.uniform(10, 45),
                'traffic_sources': {
                    'for_you': 0.7,
                    'following': 0.15,
                    'search': 0.1,
                    'profile': 0.05
                }
            }
        except Exception as e:
            logger.error(f"Failed to get TikTok analytics: {str(e)}")
            return {}
    
    async def track_viral_metrics(self, video_id: str) -> ViralMetrics:
        """Track viral performance metrics"""
        analytics = await self.get_video_analytics(video_id)
        
        # Calculate viral metrics
        views = analytics.get('views', 0)
        viral_velocity = views / 24 if views > 10000 else 0  # Views per hour
        
        return ViralMetrics(
            viral_score=self._calculate_viral_score(analytics),
            viral_velocity=viral_velocity,
            share_rate=analytics.get('shares', 0) / max(views, 1),
            completion_rate=analytics.get('completion_rate', 0),
            loop_count=int(analytics.get('average_watch_time', 0) / 30),  # Assuming 30s videos
            trend_alignment_score=random.uniform(0.6, 0.9),  # Would analyze trend data
            estimated_reach=int(views * 1.5),  # Potential future reach
            viral_peak_time=datetime.utcnow() + timedelta(hours=random.randint(12, 48))
        )
    
    def _calculate_viral_score(self, analytics: Dict[str, Any]) -> float:
        """Calculate viral score based on metrics"""
        views = analytics.get('views', 0)
        engagement = (
            analytics.get('likes', 0) +
            analytics.get('comments', 0) * 2 +
            analytics.get('shares', 0) * 3
        )
        
        engagement_rate = engagement / max(views, 1)
        completion_rate = analytics.get('completion_rate', 0)
        
        # Viral score formula
        viral_score = (
            engagement_rate * 40 +
            completion_rate * 30 +
            (views / 100000) * 30  # Normalize views to 0-30 range
        )
        
        return min(viral_score, 100)


class TikTokEnhancedPublisher(UniversalPlatformPublisher):
    """Enhanced TikTok publisher with viral optimization and creator monetization"""
    
    def __init__(self, client_id: str):
        super().__init__(client_id, "tiktok")
        self.tiktok_api = None
        self.viral_optimizer = TikTokViralOptimizer()
        self.analytics_tracker = None
        self._account_id = None
        
    async def authenticate(self, credentials: Dict[str, str]) -> bool:
        """Authenticate with TikTok API"""
        try:
            # TikTok uses OAuth 2.0
            access_token = credentials.get('access_token')
            if not access_token:
                return False
            
            # Initialize API client
            self.tiktok_api = TikTokAPI(access_token)
            self.analytics_tracker = TikTokAnalyticsTracker(self.tiktok_api)
            
            # Verify token and get account info
            account_info = await self._get_account_info()
            if not account_info:
                return False
            
            self._account_id = account_info.get('account_id')
            self._authenticated = True
            self._credentials = credentials
            
            self.log_activity('authenticate', {'status': 'success'})
            return True
            
        except Exception as e:
            logger.error(f"TikTok authentication failed: {str(e)}")
            self.log_activity('authenticate', {'error': str(e)}, success=False)
            return False
    
    async def publish_content(self, content: Dict[str, Any]) -> PublishResult:
        """Publish to TikTok with viral optimization and monetization features"""
        try:
            if not self._authenticated:
                return self.handle_publish_error("tiktok", "Not authenticated")
            
            # 1. Viral content optimization
            viral_optimizations = await self.optimize_for_viral_potential(content)
            
            # 2. TikTok-specific content adaptation
            tiktok_content = await self.adapt_for_tiktok(content, viral_optimizations)
            
            # 3. Add viral elements (trending sounds, effects, hashtags)
            viral_content = await self.add_viral_elements(tiktok_content)
            
            # 4. Create video metadata
            metadata = await self._create_video_metadata(viral_content)
            
            # 5. Post at optimal viral times
            optimal_time = await self.get_viral_posting_time(
                await self.detect_business_niche(content.get('text', ''))
            )
            
            # 6. Publish video
            video_response = await self._publish_video(metadata)
            
            if not video_response or not video_response.get('video_id'):
                return self.handle_publish_error("tiktok", "Failed to publish video")
            
            video_id = video_response['video_id']
            
            # 7. Track viral metrics and revenue potential
            viral_metrics = await self.analytics_tracker.track_viral_metrics(video_id)
            
            # 8. Get initial analytics
            analytics = await self.analytics_tracker.get_video_analytics(video_id)
            
            # 9. Calculate creator revenue potential
            revenue_potential = await self.calculate_creator_revenue_potential(
                viral_content,
                analytics,
                viral_metrics
            )
            
            # 10. Create performance metrics
            performance_metrics = PerformanceMetrics(
                engagement_rate=self._calculate_engagement_rate(analytics),
                reach=analytics.get('views', 0),
                impressions=analytics.get('views', 0),
                clicks=0,  # TikTok doesn't provide click data
                shares=analytics.get('shares', 0),
                saves=0,  # Not directly available
                comments=analytics.get('comments', 0),
                likes=analytics.get('likes', 0),
                video_views=analytics.get('views', 0),
                video_retention_rate=analytics.get('completion_rate', 0)
            )
            
            # 11. Create revenue metrics
            revenue_metrics = RevenueMetrics(
                estimated_revenue_potential=revenue_potential,
                actual_revenue=0.0,
                conversion_rate=0.0,  # Will update as data comes in
                revenue_per_engagement=revenue_potential / max(
                    analytics.get('likes', 0) + analytics.get('comments', 0), 1
                ),
                revenue_per_impression=revenue_potential / max(analytics.get('views', 1), 1)
            )
            
            return PublishResult(
                platform="tiktok",
                status=PublishStatus.PUBLISHED,
                post_id=video_id,
                post_url=f"https://www.tiktok.com/@{self._account_id}/video/{video_id}",
                metrics=analytics,
                revenue_metrics=revenue_metrics,
                performance_metrics=performance_metrics,
                optimization_suggestions=viral_optimizations.get('suggestions', []),
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"TikTok publish failed: {str(e)}")
            return self.handle_publish_error("tiktok", str(e))
    
    async def optimize_for_viral_potential(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize content for maximum viral potential on TikTok"""
        business_niche = await self.detect_business_niche(content.get('text', ''))
        
        # Get algorithm optimizations
        algorithm_opts = await self.viral_optimizer.optimize_for_algorithm(content, business_niche)
        
        # Get trending data
        trending_data = await self._get_tiktok_trending_data()
        
        # Generate viral hashtags
        viral_hashtags = await self._get_viral_hashtags(content, trending_data)
        
        # Get optimal video specs
        video_specs = self._get_optimal_video_specs(business_niche)
        
        return {
            'algorithm_optimizations': algorithm_opts,
            'trending_elements': trending_data,
            'hashtags': viral_hashtags,
            'video_specifications': video_specs,
            'suggestions': [
                f"Use hook within first 3 seconds: {algorithm_opts['video_optimizations']['hook_timing']}",
                f"Optimal duration: {video_specs['duration']} seconds",
                "Include a trending sound for 2x more reach",
                "End with a loop or cliffhanger to increase watch time",
                "Reply to first 10 comments to boost engagement"
            ]
        }
    
    async def adapt_for_tiktok(
        self,
        content: Dict[str, Any],
        viral_optimizations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt content specifically for TikTok format"""
        adapted_content = {
            'video_file': content.get('video_file'),
            'description': self._create_tiktok_caption(
                content.get('text', ''),
                viral_optimizations
            ),
            'type': 'video'
        }
        
        # Add cover image if provided
        if content.get('thumbnail'):
            adapted_content['cover_image'] = content['thumbnail']
        
        return adapted_content
    
    async def add_viral_elements(self, tiktok_content: Dict[str, Any]) -> Dict[str, Any]:
        """Add viral elements to maximize reach"""
        enhanced_content = tiktok_content.copy()
        
        # Add trending sound
        trending_sound = await self._get_trending_sound_for_niche(
            await self.detect_business_niche(enhanced_content.get('description', ''))
        )
        if trending_sound:
            enhanced_content['music_id'] = trending_sound['id']
            enhanced_content['sound_attribution'] = trending_sound['name']
        
        # Add viral hashtags if not already present
        if 'hashtags' not in enhanced_content:
            enhanced_content['hashtags'] = await self._get_viral_hashtags(
                enhanced_content,
                await self._get_tiktok_trending_data()
            )
        
        # Add engagement prompts
        enhanced_content['description'] = self._add_engagement_prompts(
            enhanced_content['description']
        )
        
        return enhanced_content
    
    def _create_tiktok_caption(
        self,
        text: str,
        optimizations: Dict[str, Any]
    ) -> str:
        """Create engaging TikTok caption"""
        # Shorten for TikTok's caption limit (150 chars optimal)
        if len(text) > 150:
            # Keep the hook and add ellipsis
            text = text[:147] + "..."
        
        # Add call-to-action
        cta_options = [
            "Follow for more!",
            "What do you think?",
            "Try this yourself!",
            "Share if helpful!",
            "Which is your favorite?"
        ]
        
        caption = f"{text}\n\n{random.choice(cta_options)}"
        
        # Add hashtags
        if optimizations.get('hashtags'):
            hashtags_text = ' '.join(optimizations['hashtags'][:8])  # TikTok recommends 3-8
            caption += f"\n\n{hashtags_text}"
        
        return caption
    
    def _add_engagement_prompts(self, description: str) -> str:
        """Add engagement prompts to description"""
        prompts = [
            "Comment your thoughts below 👇",
            "Which one are you? Let me know!",
            "Tag someone who needs to see this",
            "Save this for later!",
            "Follow for part 2!"
        ]
        
        # Only add if not already has a prompt
        if not any(prompt.lower() in description.lower() for prompt in ['comment', 'follow', 'share', 'save']):
            description += f" {random.choice(prompts)}"
        
        return description
    
    async def _create_video_metadata(self, content: Dict[str, Any]) -> TikTokVideoMetadata:
        """Create TikTok video metadata"""
        return TikTokVideoMetadata(
            description=content['description'],
            video_file=content['video_file'],
            cover_image=content.get('cover_image'),
            music_id=content.get('music_id'),
            hashtags=self._extract_hashtags(content['description']),
            mentions=self._extract_mentions(content['description']),
            privacy_level="public",
            allow_comments=True,
            allow_duet=True,
            allow_stitch=True
        )
    
    async def _publish_video(self, metadata: TikTokVideoMetadata) -> Dict[str, Any]:
        """Publish video to TikTok"""
        # In production, this would use TikTok API
        return {
            'video_id': f'tiktok_{datetime.utcnow().timestamp()}',
            'status': 'published'
        }
    
    async def _get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get TikTok account information"""
        # In production, fetch from TikTok API
        return {
            'account_id': 'sample_tiktok_account',
            'username': 'sample_user',
            'followers': 10000
        }
    
    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text"""
        hashtag_pattern = r'#\w+'
        return re.findall(hashtag_pattern, text)
    
    def _extract_mentions(self, text: str) -> List[str]:
        """Extract mentions from text"""
        mention_pattern = r'@\w+'
        return re.findall(mention_pattern, text)
    
    async def _get_tiktok_trending_data(self) -> Dict[str, Any]:
        """Get current trending data from TikTok"""
        # In production, fetch from TikTok API or scraping service
        return {
            'trending_sounds': [
                {'id': 'sound1', 'name': 'Trending Sound 1', 'usage_count': 1000000},
                {'id': 'sound2', 'name': 'Trending Sound 2', 'usage_count': 800000}
            ],
            'trending_hashtags': ['#fyp', '#viral', '#trending2024', '#LearnOnTikTok'],
            'trending_effects': ['Green Screen', 'Time Warp', 'Face Zoom'],
            'trending_topics': ['education', 'smallbusiness', 'fitness']
        }
    
    async def _get_viral_hashtags(
        self,
        content: Dict[str, Any],
        trending_data: Dict[str, Any]
    ) -> List[str]:
        """Generate viral hashtags for content"""
        hashtags = []
        
        # Always include these for reach
        hashtags.extend(['#fyp', '#foryoupage', '#viral'])
        
        # Add trending hashtags
        trending = trending_data.get('trending_hashtags', [])
        hashtags.extend(trending[:2])
        
        # Add niche-specific hashtags
        business_niche = await self.detect_business_niche(content.get('description', ''))
        niche_hashtags = {
            "education": ['#LearnOnTikTok', '#EduTok', '#StudyTips'],
            "business_consulting": ['#BusinessTips', '#Entrepreneur', '#SmallBusiness'],
            "fitness_wellness": ['#FitTok', '#WorkoutMotivation', '#HealthyLifestyle'],
            "creative": ['#ArtTok', '#CreativeProcess', '#DIY'],
            "ecommerce": ['#TikTokMadeMeBuyIt', '#Shopping', '#ProductReview'],
            "technology": ['#TechTok', '#CodingLife', '#TechTips'],
            "non_profit": ['#ForGood', '#MakeADifference', '#Charity']
        }
        
        hashtags.extend(niche_hashtags.get(business_niche, ['#Content'])[:3])
        
        # Remove duplicates and return
        return list(dict.fromkeys(hashtags))[:8]  # TikTok optimal is 3-8 hashtags
    
    async def _get_trending_sound_for_niche(self, business_niche: str) -> Optional[Dict[str, Any]]:
        """Get trending sound appropriate for the niche"""
        # In production, analyze trending sounds by niche
        trending_sounds = {
            "education": {'id': 'edu_sound_1', 'name': 'Study Vibes'},
            "business_consulting": {'id': 'biz_sound_1', 'name': 'Success Mindset'},
            "fitness_wellness": {'id': 'fit_sound_1', 'name': 'Workout Beat'},
            "creative": {'id': 'art_sound_1', 'name': 'Creative Flow'}
        }
        
        return trending_sounds.get(business_niche)
    
    def _get_optimal_video_specs(self, business_niche: str) -> Dict[str, Any]:
        """Get optimal video specifications by niche"""
        specs = {
            "education": {
                'duration': 30,
                'aspect_ratio': '9:16',
                'resolution': '1080x1920',
                'fps': 30
            },
            "fitness_wellness": {
                'duration': 15,
                'aspect_ratio': '9:16',
                'resolution': '1080x1920',
                'fps': 30
            },
            "creative": {
                'duration': 60,
                'aspect_ratio': '9:16',
                'resolution': '1080x1920',
                'fps': 60  # Higher for smooth art videos
            }
        }
        
        return specs.get(business_niche, {
            'duration': 30,
            'aspect_ratio': '9:16',
            'resolution': '1080x1920',
            'fps': 30
        })
    
    async def calculate_creator_revenue_potential(
        self,
        content: Dict[str, Any],
        analytics: Dict[str, Any],
        viral_metrics: ViralMetrics
    ) -> float:
        """Calculate revenue potential for TikTok creators"""
        revenue_sources = {
            'creator_fund': 0.0,
            'live_gifts': 0.0,
            'brand_partnerships': 0.0,
            'affiliate_marketing': 0.0,
            'product_sales': 0.0
        }
        
        views = analytics.get('views', 0)
        
        # Creator Fund (roughly $0.02-0.04 per 1000 views)
        if views > 10000:  # Minimum for creator fund
            revenue_sources['creator_fund'] = (views / 1000) * 0.03
        
        # Brand partnership potential (based on engagement)
        engagement_rate = self._calculate_engagement_rate(analytics)
        if engagement_rate > 3.0 and views > 50000:
            # Rough estimate: $100 per 10k engaged followers
            revenue_sources['brand_partnerships'] = (views * engagement_rate / 100) * 0.1
        
        # Affiliate marketing potential
        if viral_metrics.viral_score > 70:
            revenue_sources['affiliate_marketing'] = views * 0.0001  # $0.10 per 1000 views
        
        # Product sales potential (for ecommerce niches)
        business_niche = await self.detect_business_niche(content.get('description', ''))
        if business_niche in ['ecommerce', 'creative', 'fitness_wellness']:
            conversion_rate = 0.001  # 0.1% conversion
            average_order_value = 50.0
            revenue_sources['product_sales'] = views * conversion_rate * average_order_value
        
        total_revenue = sum(revenue_sources.values())
        return round(total_revenue, 2)
    
    def _calculate_engagement_rate(self, analytics: Dict[str, Any]) -> float:
        """Calculate engagement rate"""
        views = analytics.get('views', 1)
        engagement = (
            analytics.get('likes', 0) +
            analytics.get('comments', 0) +
            analytics.get('shares', 0)
        )
        
        return (engagement / views) * 100
    
    async def get_viral_posting_time(self, business_niche: str) -> datetime:
        """Get optimal posting time for viral potential"""
        # TikTok peak times by niche
        optimal_times = {
            "education": [7, 12, 16, 20],      # Before school, lunch, after school, evening
            "business_consulting": [8, 12, 17, 21],  # Morning, lunch, after work, night
            "fitness_wellness": [5, 7, 12, 19],      # Early morning, breakfast, lunch, evening
            "creative": [10, 14, 19, 22],            # Mid-morning, afternoon, evening, late night
            "ecommerce": [12, 18, 20, 22],          # Lunch, after work, prime time
            "technology": [9, 13, 18, 21],           # Morning, lunch, evening, night
            "non_profit": [12, 17, 20]               # Lunch, after work, evening
        }
        
        hours = optimal_times.get(business_niche, [12, 18, 21])
        
        # TikTok engagement is highest on weekends
        now = datetime.utcnow()
        target_days = [4, 5, 6]  # Friday, Saturday, Sunday
        
        # Find next optimal day and time
        for days_ahead in range(7):
            check_date = now + timedelta(days=days_ahead)
            if check_date.weekday() in target_days:
                for hour in hours:
                    potential_time = check_date.replace(
                        hour=hour,
                        minute=random.randint(0, 59),  # Random minute for variety
                        second=0,
                        microsecond=0
                    )
                    if potential_time > now:
                        return potential_time
        
        # Fallback
        return now + timedelta(hours=1)
    
    async def get_optimal_posting_time(
        self,
        content_type: str,
        business_niche: str
    ) -> datetime:
        """Override to use viral posting time"""
        return await self.get_viral_posting_time(business_niche)
    
    async def analyze_audience_engagement(self, business_niche: str) -> Dict[str, Any]:
        """Analyze TikTok audience for this business niche"""
        try:
            audience_data = {
                'demographics': {
                    'age_groups': self._get_age_distribution(business_niche),
                    'gender_split': {'male': 0.45, 'female': 0.55},
                    'top_countries': ['US', 'UK', 'CA', 'AU', 'PH']
                },
                'engagement_patterns': {
                    'peak_hours': [6, 10, 19, 22],  # TikTok peak times
                    'peak_days': ['Friday', 'Saturday', 'Sunday'],
                    'average_session_time': 52,  # minutes per day
                    'content_consumption_rate': 8.5  # videos per minute
                },
                'content_preferences': {
                    'preferred_length': self._get_preferred_video_length(business_niche),
                    'top_content_types': ['entertainment', 'educational', 'inspirational'],
                    'engagement_triggers': ['humor', 'relatability', 'surprise', 'value']
                },
                'viral_indicators': {
                    'share_threshold': 0.03,  # 3% share rate indicates viral potential
                    'completion_threshold': 0.5,  # 50% completion rate
                    'loop_threshold': 1.5,  # 1.5 loops average
                    'early_engagement_importance': 0.8  # 80% of viral videos get high engagement in first hour
                }
            }
            
            return audience_data
            
        except Exception as e:
            logger.error(f"TikTok audience analysis failed: {str(e)}")
            return self._get_default_audience_data(business_niche)
    
    def _get_age_distribution(self, business_niche: str) -> Dict[str, float]:
        """Get age distribution by niche"""
        distributions = {
            "education": {"13-17": 0.25, "18-24": 0.40, "25-34": 0.25, "35+": 0.10},
            "business_consulting": {"18-24": 0.20, "25-34": 0.45, "35-44": 0.25, "45+": 0.10},
            "fitness_wellness": {"16-24": 0.35, "25-34": 0.40, "35-44": 0.20, "45+": 0.05},
            "creative": {"13-17": 0.30, "18-24": 0.35, "25-34": 0.25, "35+": 0.10}
        }
        
        return distributions.get(business_niche, {
            "13-17": 0.20,
            "18-24": 0.35,
            "25-34": 0.30,
            "35+": 0.15
        })
    
    def _get_preferred_video_length(self, business_niche: str) -> Dict[str, Any]:
        """Get preferred video length by niche"""
        preferences = {
            "education": {'optimal': 30, 'range': (15, 60)},
            "business_consulting": {'optimal': 45, 'range': (30, 60)},
            "fitness_wellness": {'optimal': 15, 'range': (15, 30)},
            "creative": {'optimal': 60, 'range': (30, 180)}
        }
        
        return preferences.get(business_niche, {'optimal': 30, 'range': (15, 60)})
    
    def _get_default_audience_data(self, business_niche: str) -> Dict[str, Any]:
        """Get default audience data when API unavailable"""
        return {
            'demographics': {
                'age_groups': self._get_age_distribution(business_niche),
                'gender_split': {'male': 0.45, 'female': 0.55},
                'top_countries': ['US', 'UK', 'CA']
            },
            'engagement_patterns': {
                'peak_hours': [10, 19, 22],
                'peak_days': ['Friday', 'Saturday', 'Sunday'],
                'average_session_time': 45,
                'content_consumption_rate': 7.0
            }
        }
    
    async def get_platform_optimizations(
        self,
        content: Dict[str, Any],
        business_niche: str
    ) -> Dict[str, Any]:
        """Get TikTok-specific optimizations"""
        return {
            'content_creation': {
                'hook_strategies': [
                    "Start with a question or bold statement",
                    "Show the end result first",
                    "Use text overlay for context",
                    "Create a pattern interrupt"
                ],
                'retention_tactics': [
                    "Change scenes every 3-5 seconds",
                    "Use trending transitions",
                    "Add captions for accessibility",
                    "End with a reason to rewatch"
                ]
            },
            'algorithm_hacks': {
                'engagement_boost': "Reply to comments within first hour",
                'completion_rate': "Keep videos under 30 seconds for higher completion",
                'share_triggers': "Create content that evokes emotion",
                'fyp_optimization': "Post when your audience is most active"
            },
            'monetization_strategies': {
                'creator_fund': "Need 10k followers and 100k views in 30 days",
                'live_gifts': "Go live regularly to receive virtual gifts",
                'brand_deals': "Focus on niche content for better partnerships",
                'product_placement': "Seamlessly integrate products into content"
            },
            'growth_tactics': {
                'consistency': "Post 1-3 times daily at peak times",
                'trends': "Jump on trends within 24-48 hours",
                'collaboration': "Duet and stitch with larger creators",
                'series': "Create content series to boost follows"
            }
        }


class TikTokAPI:
    """Simplified TikTok API client"""
    
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.base_url = "https://open-api.tiktok.com"
        
    async def upload_video(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Upload video to TikTok"""
        # In production, implement actual API call
        return {
            'video_id': f'tiktok_{datetime.utcnow().timestamp()}',
            'status': 'published'
        }