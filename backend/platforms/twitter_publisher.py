"""
Twitter Enhanced Publisher with built-in business intelligence.

Handles Twitter/X publishing with revenue optimization, engagement tracking,
and universal business niche support.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import tempfile
import os

import tweepy
from tweepy import API, OAuth1UserHandler, StreamingClient, Client
import aiohttp

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
class TwitterThreadMetadata:
    """Metadata for Twitter threads"""
    thread_tweets: List[str]
    media_urls: List[Optional[str]]
    poll_options: Optional[List[str]] = None
    scheduled_time: Optional[datetime] = None
    reply_settings: str = "everyone"  # everyone, following, mentioned
    

@dataclass 
class TwitterSpaceMetrics:
    """Metrics for Twitter Spaces"""
    space_id: str
    attendees: int
    speakers: int
    duration_minutes: int
    peak_listeners: int
    total_replays: int
    estimated_revenue: float


class TwitterEngagementOptimizer:
    """Optimizes content for Twitter's algorithm and engagement"""
    
    def __init__(self):
        self.viral_patterns = {
            'education': {
                'thread_length': (5, 15),
                'media_type': 'infographics',
                'optimal_times': [9, 12, 15, 20],
                'hashtag_count': 2,
                'engagement_hooks': [
                    "🧵 Thread:",
                    "Here's what nobody tells you about",
                    "I spent 100 hours researching",
                    "The complete guide to"
                ]
            },
            'business_consulting': {
                'thread_length': (7, 20),
                'media_type': 'charts',
                'optimal_times': [8, 13, 17],
                'hashtag_count': 3,
                'engagement_hooks': [
                    "How we went from $0 to $1M:",
                    "The framework that changed everything:",
                    "Stop doing X, start doing Y:",
                    "My biggest business mistake:"
                ]
            },
            'fitness_wellness': {
                'thread_length': (3, 10),
                'media_type': 'before_after',
                'optimal_times': [6, 12, 18],
                'hashtag_count': 4,
                'engagement_hooks': [
                    "30-day transformation:",
                    "The workout that changed my life:",
                    "Science-backed fitness tips:",
                    "Your workout is wrong if"
                ]
            },
            'creative': {
                'thread_length': (1, 5),
                'media_type': 'portfolio',
                'optimal_times': [11, 14, 19, 22],
                'hashtag_count': 5,
                'engagement_hooks': [
                    "My creative process:",
                    "From concept to creation:",
                    "Art tip of the day:",
                    "Behind the scenes:"
                ]
            }
        }
    
    async def optimize_for_virality(self, content: Dict[str, Any], business_niche: str) -> Dict[str, Any]:
        """Optimize content for Twitter virality"""
        pattern = self.viral_patterns.get(
            business_niche, 
            self.viral_patterns['business_consulting']
        )
        
        # Generate thread structure
        thread_structure = await self._generate_thread_structure(
            content, 
            pattern['thread_length']
        )
        
        # Add engagement hooks
        hook = await self._select_engagement_hook(content, pattern['engagement_hooks'])
        
        # Optimize hashtags
        hashtags = await self._research_trending_hashtags(
            content,
            pattern['hashtag_count'],
            business_niche
        )
        
        # Add call-to-action
        cta = await self._generate_viral_cta(business_niche)
        
        return {
            'thread_structure': thread_structure,
            'engagement_hook': hook,
            'hashtags': hashtags,
            'optimal_media': pattern['media_type'],
            'viral_cta': cta,
            'reply_strategy': await self._get_reply_strategy(business_niche),
            'retweet_optimization': await self._optimize_for_retweets(content)
        }
    
    async def _generate_thread_structure(self, content: Dict[str, Any], length_range: Tuple[int, int]) -> List[str]:
        """Generate optimized thread structure"""
        text = content.get('text', '')
        min_tweets, max_tweets = length_range
        
        # Split content into tweet-sized chunks
        tweets = []
        
        # First tweet - hook
        hook_tweet = text[:250] + "... 🧵"
        tweets.append(hook_tweet)
        
        # Body tweets
        remaining_text = text[250:]
        words = remaining_text.split()
        current_tweet = ""
        
        for word in words:
            if len(current_tweet) + len(word) + 1 <= 270:
                current_tweet += f" {word}"
            else:
                tweets.append(current_tweet.strip())
                current_tweet = word
        
        if current_tweet:
            tweets.append(current_tweet.strip())
        
        # Ensure we're within range
        if len(tweets) < min_tweets:
            # Add value-add tweets
            tweets.extend([
                "Key takeaway:",
                "Action step:",
                "Remember:"
            ][:min_tweets - len(tweets)])
        elif len(tweets) > max_tweets:
            tweets = tweets[:max_tweets]
        
        # Add thread numbering
        for i, tweet in enumerate(tweets[1:], 2):
            tweets[i-1] = f"{i}/{len(tweets)}\n\n{tweet}"
        
        return tweets
    
    async def _select_engagement_hook(self, content: Dict[str, Any], hooks: List[str]) -> str:
        """Select best engagement hook based on content"""
        # Analyze content sentiment and topic
        content_text = content.get('text', '').lower()
        
        # Score each hook
        best_hook = hooks[0]
        best_score = 0
        
        for hook in hooks:
            score = 0
            hook_lower = hook.lower()
            
            # Check relevance
            if "guide" in hook_lower and "how" in content_text:
                score += 2
            if "mistake" in hook_lower and "avoid" in content_text:
                score += 2
            if "framework" in hook_lower and "system" in content_text:
                score += 2
            
            if score > best_score:
                best_score = score
                best_hook = hook
        
        return best_hook
    
    async def _research_trending_hashtags(self, content: Dict[str, Any], count: int, niche: str) -> List[str]:
        """Research trending hashtags for the niche"""
        base_hashtags = {
            'education': ['#EdTech', '#OnlineLearning', '#StudyTips', '#Education', '#LearnOnTwitter'],
            'business_consulting': ['#BusinessTips', '#Entrepreneur', '#StartupLife', '#BusinessGrowth', '#Marketing'],
            'fitness_wellness': ['#FitnessMotivation', '#HealthyLiving', '#WorkoutWednesday', '#FitFam', '#Wellness'],
            'creative': ['#CreativeProcess', '#ArtistsOnTwitter', '#DesignInspiration', '#CreativeLife', '#Portfolio']
        }
        
        niche_hashtags = base_hashtags.get(niche, base_hashtags['business_consulting'])
        
        # Add trending variations
        trending_hashtags = []
        for tag in niche_hashtags[:count]:
            # Add day-specific variations
            day = datetime.utcnow().strftime('%A')
            if day == 'Monday' and '#Motivation' not in tag:
                trending_hashtags.append('#MondayMotivation')
            elif day == 'Friday' and niche == 'creative':
                trending_hashtags.append('#FollowFriday')
            else:
                trending_hashtags.append(tag)
        
        return trending_hashtags[:count]
    
    async def _generate_viral_cta(self, niche: str) -> str:
        """Generate viral call-to-action"""
        ctas = {
            'education': "RT to help someone learn something new today! 🎓",
            'business_consulting': "RT if this helped your business journey! 💼",
            'fitness_wellness': "RT to motivate someone's fitness journey! 💪",
            'creative': "RT to support creative work! 🎨"
        }
        
        return ctas.get(niche, "RT if you found this valuable! 🙏")
    
    async def _get_reply_strategy(self, niche: str) -> Dict[str, Any]:
        """Get reply strategy for engagement"""
        return {
            'auto_reply_templates': [
                "Thanks for engaging! What's your experience with this?",
                "Great question! Here's more detail:",
                "Appreciate the feedback! Have you tried..."
            ],
            'engagement_window': 60,  # minutes to actively engage
            'priority_replies': ['questions', 'high_followers', 'verified']
        }
    
    async def _optimize_for_retweets(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize content for maximum retweets"""
        return {
            'quote_tweet_prompts': [
                "What's your take on this?",
                "Tag someone who needs to see this",
                "Your thoughts? 👇"
            ],
            'visual_optimization': {
                'use_native_video': True,
                'image_alt_text': True,
                'gif_usage': 'strategic'
            },
            'timing_optimization': {
                'avoid_link_first_tweet': True,
                'link_in_comments': True,
                'space_out_media': True
            }
        }


class TwitterAnalyticsTracker:
    """Tracks Twitter analytics and monetization"""
    
    def __init__(self):
        self.metrics_cache = {}
        self.revenue_multipliers = {
            'verified_account': 1.5,
            'high_engagement': 2.0,
            'viral_content': 3.0,
            'twitter_blue': 1.2
        }
    
    async def track_tweet_performance(self, tweet_id: str, api: API) -> Dict[str, Any]:
        """Track individual tweet performance"""
        try:
            tweet = api.get_status(tweet_id, tweet_mode='extended')
            
            metrics = {
                'impressions': tweet.retweet_count * 150,  # Estimated
                'engagements': tweet.favorite_count + tweet.retweet_count,
                'retweets': tweet.retweet_count,
                'likes': tweet.favorite_count,
                'replies': 0,  # Would need to search for replies
                'profile_clicks': int(tweet.favorite_count * 0.05),
                'link_clicks': int(tweet.favorite_count * 0.02),
                'engagement_rate': 0
            }
            
            # Calculate engagement rate
            if metrics['impressions'] > 0:
                metrics['engagement_rate'] = (
                    metrics['engagements'] / metrics['impressions']
                ) * 100
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error tracking tweet performance: {e}")
            return {}
    
    async def calculate_monetization_potential(self, metrics: Dict[str, Any], account_data: Dict[str, Any]) -> float:
        """Calculate Twitter monetization potential"""
        base_revenue = 0
        
        # Twitter Blue revenue sharing (if eligible)
        if account_data.get('twitter_blue', False):
            impressions = metrics.get('impressions', 0)
            # Estimated $0.50 per 1000 impressions for verified accounts
            base_revenue += (impressions / 1000) * 0.50
        
        # Sponsored content potential
        followers = account_data.get('followers_count', 0)
        engagement_rate = metrics.get('engagement_rate', 0)
        
        if followers > 10000 and engagement_rate > 2:
            # Estimated sponsored post value
            sponsored_value = (followers / 1000) * 10 * (engagement_rate / 2)
            base_revenue += sponsored_value * 0.1  # 10% chance of sponsorship
        
        # Super Follows revenue (if enabled)
        if account_data.get('super_follows_enabled', False):
            super_followers = account_data.get('super_followers', 0)
            base_revenue += super_followers * 4.99 * 0.97 * 0.7  # Monthly revenue after fees
        
        # Apply multipliers
        for condition, multiplier in self.revenue_multipliers.items():
            if self._check_condition(condition, metrics, account_data):
                base_revenue *= multiplier
        
        return round(base_revenue, 2)
    
    async def track_spaces_performance(self, space_id: str) -> TwitterSpaceMetrics:
        """Track Twitter Spaces performance"""
        # Simulated metrics for Spaces
        return TwitterSpaceMetrics(
            space_id=space_id,
            attendees=250,
            speakers=5,
            duration_minutes=45,
            peak_listeners=180,
            total_replays=500,
            estimated_revenue=125.0  # Ticketed spaces or sponsorships
        )
    
    def _check_condition(self, condition: str, metrics: Dict[str, Any], account_data: Dict[str, Any]) -> bool:
        """Check if condition is met for revenue multiplier"""
        if condition == 'verified_account':
            return account_data.get('verified', False)
        elif condition == 'high_engagement':
            return metrics.get('engagement_rate', 0) > 5
        elif condition == 'viral_content':
            return metrics.get('retweets', 0) > 1000
        elif condition == 'twitter_blue':
            return account_data.get('twitter_blue', False)
        return False


class TwitterAPIv2:
    """Complete Twitter API v2 integration with advanced features"""
    
    def __init__(self, bearer_token: str, api_key: Optional[str] = None, api_secret: Optional[str] = None, 
                 access_token: Optional[str] = None, access_token_secret: Optional[str] = None):
        self.bearer_token = bearer_token
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret
        
        # Initialize Twitter API v2 client
        self.client = Client(
            bearer_token=bearer_token,
            consumer_key=api_key,
            consumer_secret=api_secret,
            access_token=access_token,
            access_token_secret=access_token_secret,
            wait_on_rate_limit=True
        )
        
        self.session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close_session(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def post_tweet(self, text: str, media_ids: Optional[List[str]] = None, reply_to: Optional[str] = None) -> Dict[str, Any]:
        """Post a tweet using Twitter API v2"""
        try:
            tweet_data = {
                'text': text
            }
            
            if media_ids:
                tweet_data['media'] = {'media_ids': media_ids}
            
            if reply_to:
                tweet_data['reply'] = {'in_reply_to_tweet_id': reply_to}
            
            response = self.client.create_tweet(**tweet_data)
            
            if response.data:
                return {
                    'success': True,
                    'tweet_id': response.data['id'],
                    'text': response.data['text']
                }
            else:
                return {'success': False, 'error': 'Failed to create tweet'}
                
        except Exception as e:
            logger.error(f"Tweet posting failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def post_thread(self, tweets: List[str], media_ids: Optional[List[List[str]]] = None) -> Dict[str, Any]:
        """Post a Twitter thread using API v2"""
        try:
            thread_results = []
            previous_tweet_id = None
            
            for i, tweet_text in enumerate(tweets):
                # Get media IDs for this tweet if available
                tweet_media = media_ids[i] if media_ids and i < len(media_ids) else None
                
                # Post the tweet
                result = await self.post_tweet(
                    text=tweet_text,
                    media_ids=tweet_media,
                    reply_to=previous_tweet_id
                )
                
                if result.get('success'):
                    tweet_id = result['tweet_id']
                    thread_results.append({
                        'tweet_id': tweet_id,
                        'text': tweet_text,
                        'position': i + 1
                    })
                    previous_tweet_id = tweet_id
                else:
                    logger.error(f"Failed to post tweet {i + 1} in thread: {result.get('error')}")
                    break
                
                # Small delay between tweets to avoid rate limiting
                await asyncio.sleep(1)
            
            return {
                'success': len(thread_results) > 0,
                'thread_results': thread_results,
                'thread_url': f"https://twitter.com/user/status/{thread_results[0]['tweet_id']}" if thread_results else None
            }
            
        except Exception as e:
            logger.error(f"Thread posting failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def upload_media(self, media_path: str, media_type: Optional[str] = None) -> Optional[str]:
        """Upload media using Twitter API v1.1 (required for media upload)"""
        try:
            # Media upload requires API v1.1
            auth = tweepy.OAuth1UserHandler(
                self.api_key, self.api_secret,
                self.access_token, self.access_token_secret
            )
            api_v1 = tweepy.API(auth)
            
            # Determine media category
            if not media_type:
                _, ext = os.path.splitext(media_path)
                if ext.lower() in ['.mp4', '.mov', '.avi']:
                    media_type = 'video'
                elif ext.lower() in ['.gif']:
                    media_type = 'gif'
                else:
                    media_type = 'image'
            
            # Upload media
            if media_type == 'video':
                media = api_v1.media_upload(
                    filename=media_path,
                    media_category='tweet_video'
                )
            elif media_type == 'gif':
                media = api_v1.media_upload(
                    filename=media_path,
                    media_category='tweet_gif'
                )
            else:  # image
                media = api_v1.media_upload(
                    filename=media_path
                )
            
            logger.info(f"Successfully uploaded media: {media.media_id}")
            return str(media.media_id)
            
        except Exception as e:
            logger.error(f"Media upload failed: {str(e)}")
            return None
    
    async def get_tweet_analytics(self, tweet_id: str) -> Dict[str, Any]:
        """Get tweet analytics using API v2"""
        try:
            tweet = self.client.get_tweet(
                id=tweet_id,
                tweet_fields=['public_metrics', 'created_at', 'context_annotations'],
                expansions=['author_id']
            )
            
            if tweet.data:
                metrics = tweet.data.public_metrics
                return {
                    'tweet_id': tweet_id,
                    'retweet_count': metrics.get('retweet_count', 0),
                    'like_count': metrics.get('like_count', 0),
                    'reply_count': metrics.get('reply_count', 0),
                    'quote_count': metrics.get('quote_count', 0),
                    'impression_count': metrics.get('impression_count', 0),
                    'created_at': str(tweet.data.created_at),
                    'total_engagements': (
                        metrics.get('retweet_count', 0) +
                        metrics.get('like_count', 0) +
                        metrics.get('reply_count', 0) +
                        metrics.get('quote_count', 0)
                    )
                }
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get tweet analytics: {str(e)}")
            return {}
    
    async def get_trending_hashtags(self, location_id: int = 1) -> List[str]:
        """Get trending hashtags using API v2"""
        try:
            # Note: Trending topics are available in API v1.1
            auth = tweepy.OAuth1UserHandler(
                self.api_key, self.api_secret,
                self.access_token, self.access_token_secret
            )
            api_v1 = tweepy.API(auth)
            
            trends = api_v1.get_place_trends(id=location_id)
            
            hashtags = []
            for trend in trends[0]['trends']:
                if trend['name'].startswith('#'):
                    hashtags.append(trend['name'])
            
            return hashtags[:10]  # Return top 10 trending hashtags
            
        except Exception as e:
            logger.error(f"Failed to get trending hashtags: {str(e)}")
            return []
    
    async def search_tweets(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search tweets using API v2"""
        try:
            tweets = self.client.search_recent_tweets(
                query=query,
                max_results=max_results,
                tweet_fields=['public_metrics', 'created_at', 'author_id'],
                expansions=['author_id'],
                user_fields=['username', 'verified']
            )
            
            results = []
            if tweets.data:
                users = {u.id: u for u in tweets.includes.get('users', [])}
                
                for tweet in tweets.data:
                    author = users.get(tweet.author_id)
                    results.append({
                        'tweet_id': tweet.id,
                        'text': tweet.text,
                        'author_username': author.username if author else None,
                        'author_verified': author.verified if author else False,
                        'created_at': str(tweet.created_at),
                        'metrics': tweet.public_metrics
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Tweet search failed: {str(e)}")
            return []
    
    async def create_twitter_space(self, space_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a Twitter Space"""
        try:
            # Note: Spaces creation is limited in API v2
            # This would require special permissions
            
            logger.info("Twitter Spaces creation requires special API access")
            return {
                'success': False,
                'error': 'Spaces creation requires special API permissions'
            }
            
        except Exception as e:
            logger.error(f"Twitter Space creation failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def get_user_info(self) -> Dict[str, Any]:
        """Get authenticated user information"""
        try:
            user = self.client.get_me(
                user_fields=['public_metrics', 'verified', 'description', 'location']
            )
            
            if user.data:
                return {
                    'user_id': user.data.id,
                    'username': user.data.username,
                    'display_name': user.data.name,
                    'verified': user.data.verified,
                    'description': user.data.description,
                    'location': user.data.location,
                    'followers_count': user.data.public_metrics.get('followers_count', 0),
                    'following_count': user.data.public_metrics.get('following_count', 0),
                    'tweet_count': user.data.public_metrics.get('tweet_count', 0),
                    'listed_count': user.data.public_metrics.get('listed_count', 0)
                }
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get user info: {str(e)}")
            return {}
    
    async def engage_with_tweet(self, tweet_id: str, action: str) -> bool:
        """Engage with a tweet (like, retweet, etc.)"""
        try:
            if action == 'like':
                response = self.client.like(tweet_id)
            elif action == 'retweet':
                response = self.client.retweet(tweet_id)
            elif action == 'unlike':
                response = self.client.unlike(tweet_id)
            elif action == 'unretweet':
                response = self.client.unretweet(tweet_id)
            else:
                logger.error(f"Unknown engagement action: {action}")
                return False
            
            return response.data.get('liked', False) or response.data.get('retweeted', False)
            
        except Exception as e:
            logger.error(f"Tweet engagement failed: {str(e)}")
            return False
    
    async def schedule_tweet(self, text: str, scheduled_time: datetime, media_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Schedule a tweet (requires external scheduling service)"""
        try:
            # Note: Native tweet scheduling is not available in API v2
            # This would require integration with external scheduling services
            
            logger.info("Tweet scheduling requires external service integration")
            return {
                'success': False,
                'error': 'Tweet scheduling requires external service integration'
            }
            
        except Exception as e:
            logger.error(f"Tweet scheduling failed: {str(e)}")
            return {'success': False, 'error': str(e)}


class TwitterEnhancedPublisher(UniversalPlatformPublisher):
    """Twitter/X enhanced publisher with revenue optimization"""
    
    def __init__(self, client_id: str):
        super().__init__(client_id, "twitter")
        self.twitter_api: Optional[TwitterAPIv2] = None
        self.engagement_optimizer = TwitterEngagementOptimizer()
        self.analytics_tracker = TwitterAnalyticsTracker()
        self._authenticated = False
        self.account_data = {}
    
    async def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """Authenticate with Twitter API v2"""
        try:
            # Twitter API v2 requires bearer token and OAuth credentials
            bearer_token = credentials.get('bearer_token')
            api_key = credentials.get('api_key')
            api_secret = credentials.get('api_secret')
            access_token = credentials.get('access_token')
            access_token_secret = credentials.get('access_token_secret')
            
            if not all([bearer_token, api_key, api_secret, access_token, access_token_secret]):
                logger.error("Missing required Twitter API credentials")
                return False
            
            # Initialize Twitter API v2
            self.twitter_api = TwitterAPIv2(
                bearer_token=bearer_token,
                api_key=api_key,
                api_secret=api_secret,
                access_token=access_token,
                access_token_secret=access_token_secret
            )
            
            # Verify credentials by getting user info
            user_info = await self.twitter_api.get_user_info()
            
            if not user_info:
                logger.error("Failed to verify Twitter credentials")
                return False
            
            self.account_data = user_info
            
            # Store encrypted credentials
            encrypted = self.encryption_manager.encrypt_credentials(credentials)
            await self._store_platform_credentials(encrypted)
            
            self._authenticated = True
            logger.info(f"Successfully authenticated Twitter account: @{user_info.get('username')}")
            return True
            
        except Exception as e:
            logger.error(f"Twitter authentication failed: {e}")
            return False
    
    async def publish_content(self, content: Dict[str, Any]) -> PublishResult:
        """Publish content to Twitter with optimization"""
        if not self._authenticated:
            return PublishResult(
                platform="twitter",
                post_id=None,
                url=None,
                status=PublishStatus.FAILED,
                error_message="Not authenticated"
            )
        
        try:
            # Detect business niche
            business_niche = await self.detect_business_niche(content.get('text', ''))
            
            # Optimize content for revenue and engagement
            optimizations = await self.optimize_for_revenue(content)
            viral_opts = await self.engagement_optimizer.optimize_for_virality(
                content, 
                business_niche
            )
            
            # Prepare content for posting
            if viral_opts.get('thread_structure') and len(viral_opts['thread_structure']) > 1:
                # Post as thread
                result = await self._post_thread(viral_opts['thread_structure'], content)
            else:
                # Post as single tweet
                result = await self._post_single_tweet(content, viral_opts)
            
            if not result.get('success'):
                return self.handle_publish_error("twitter", result.get('error', 'Unknown error'))
            
            # Track initial metrics
            tweet_id = result.get('tweet_id') or result.get('thread_results', [{}])[0].get('tweet_id')
            
            if tweet_id:
                await self._track_publish_metrics(
                    tweet_id,
                    content,
                    optimizations
                )
            
            # Generate URL
            username = self.account_data.get('username', 'user')
            url = result.get('thread_url') or f"https://twitter.com/{username}/status/{tweet_id}"
            
            return PublishResult(
                platform="twitter",
                status=PublishStatus.PUBLISHED,
                post_id=tweet_id,
                post_url=url,
                metrics=result.get('analytics', {}),
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Twitter publish failed: {str(e)}")
            return self.handle_publish_error("twitter", str(e))
    
    async def _post_single_tweet(self, content: Dict[str, Any], optimizations: Dict[str, Any]) -> Dict[str, Any]:
        """Post a single tweet with media"""
        try:
            text = content.get('text', '')
            
            # Add optimized hashtags
            if optimizations.get('hashtags'):
                hashtags_text = ' '.join(optimizations['hashtags'][:5])  # Twitter limit
                text += f"\n\n{hashtags_text}"
            
            # Add call-to-action
            if optimizations.get('viral_cta'):
                text += f"\n\n{optimizations['viral_cta']}"
            
            # Ensure tweet is within character limit
            if len(text) > 280:
                text = text[:277] + "..."
            
            # Upload media if present
            media_ids = []
            if content.get('media_url'):
                media_id = await self.twitter_api.upload_media(content['media_url'])
                if media_id:
                    media_ids.append(media_id)
            
            # Post tweet
            result = await self.twitter_api.post_tweet(
                text=text,
                media_ids=media_ids if media_ids else None
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Single tweet posting failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _post_thread(self, thread_tweets: List[str], content: Dict[str, Any]) -> Dict[str, Any]:
        """Post a Twitter thread"""
        try:
            # Upload media if present (attach to first tweet)
            media_ids = []
            if content.get('media_url'):
                media_id = await self.twitter_api.upload_media(content['media_url'])
                if media_id:
                    media_ids = [[media_id]]  # For first tweet only
            
            # Post thread
            result = await self.twitter_api.post_thread(
                tweets=thread_tweets,
                media_ids=media_ids if media_ids else None
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Thread posting failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def optimize_for_revenue(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize content for Twitter revenue"""
        business_niche = await self.detect_business_niche(content.get('text', ''))
        
        # Get optimal posting time
        optimal_time = await self.get_optimal_posting_time('tweet', business_niche)
        
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
                    'twitter_blue_revenue_sharing',
                    'sponsored_content',
                    'super_follows',
                    'ticketed_spaces',
                    'newsletter_subscriptions'
                ],
                'engagement_tactics': platform_opts['engagement_tactics']
            },
            'audience_targeting': {
                'best_times': platform_opts['algorithm_optimizations']['best_times'],
                'target_demographics': await self._get_target_demographics(business_niche)
            },
            'platform_specific_tweaks': platform_opts
        }
    
    async def get_optimal_posting_time(self, content_type: str, business_niche: str) -> datetime:
        """Get optimal posting time for Twitter"""
        hour_map = {
            'education': [9, 15, 20],
            'business_consulting': [8, 13, 17],
            'fitness_wellness': [6, 12, 18],
            'creative': [11, 14, 19, 22],
            'ecommerce': [10, 14, 20],
            'local_service': [8, 12, 17],
            'technology': [9, 14, 21],
            'non_profit': [10, 15, 19]
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
        """Get Twitter-specific optimizations"""
        return {
            'algorithm_optimizations': {
                'use_threads': True,
                'optimal_thread_length': self._get_optimal_thread_length(business_niche),
                'engagement_window': 60,  # minutes
                'reply_boost': True,
                'best_times': self._get_best_posting_times(business_niche)
            },
            'engagement_tactics': {
                'use_polls': business_niche in ['education', 'business_consulting'],
                'quote_tweet_strategy': True,
                'community_building': True,
                'twitter_spaces': business_niche in ['business_consulting', 'education']
            },
            'monetization_features': {
                'super_follows': self.account_data.get('followers_count', 0) > 10000,
                'ticketed_spaces': True,
                'tips_enabled': True,
                'newsletter': business_niche in ['education', 'business_consulting']
            },
            'growth_tactics': {
                'follow_trains': business_niche == 'creative',
                'engagement_pods': False,  # Not recommended
                'strategic_replies': True,
                'trending_hijack': True
            }
        }
    
    def _get_optimal_thread_length(self, niche: str) -> int:
        """Get optimal thread length by niche"""
        lengths = {
            'education': 10,
            'business_consulting': 15,
            'fitness_wellness': 7,
            'creative': 5,
            'ecommerce': 8,
            'technology': 12,
            'non_profit': 6
        }
        return lengths.get(niche, 8)
    
    def _get_best_posting_times(self, niche: str) -> List[int]:
        """Get best posting times by niche"""
        times = {
            'education': [9, 15, 20],
            'business_consulting': [8, 13, 17],
            'fitness_wellness': [6, 12, 18],
            'creative': [11, 14, 19, 22]
        }
        return times.get(niche, [9, 14, 19])
    
    async def _get_target_demographics(self, niche: str) -> Dict[str, Any]:
        """Get target demographics for niche"""
        demographics = {
            'education': {
                'age_range': '18-34',
                'interests': ['learning', 'self-improvement', 'career'],
                'behaviors': ['early_adopters', 'online_learners']
            },
            'business_consulting': {
                'age_range': '25-54',
                'interests': ['entrepreneurship', 'business', 'leadership'],
                'behaviors': ['business_decision_makers', 'small_business_owners']
            },
            'fitness_wellness': {
                'age_range': '22-45',
                'interests': ['health', 'fitness', 'nutrition'],
                'behaviors': ['gym_members', 'health_conscious']
            },
            'creative': {
                'age_range': '18-44',
                'interests': ['art', 'design', 'creativity'],
                'behaviors': ['content_creators', 'early_adopters']
            }
        }
        
        return demographics.get(niche, demographics['business_consulting'])
    
    async def calculate_revenue_potential(self, content: Dict[str, Any], optimizations: Dict[str, Any]) -> float:
        """Calculate revenue potential for Twitter content"""
        base_revenue = 0
        
        # Estimate based on account metrics
        followers = self.account_data.get('followers_count', 0)
        
        # Twitter Blue revenue sharing
        if self.account_data.get('twitter_blue', False):
            estimated_impressions = followers * 0.1  # 10% reach
            base_revenue += (estimated_impressions / 1000) * 0.50
        
        # Sponsored content potential
        if followers > 10000:
            engagement_rate = 2.5  # Average engagement rate
            sponsored_rate = (followers / 1000) * 10 * (engagement_rate / 2)
            base_revenue += sponsored_rate * 0.05  # 5% chance per post
        
        # Super Follows
        if followers > 10000:
            potential_super_followers = followers * 0.001  # 0.1% conversion
            base_revenue += potential_super_followers * 4.99 * 0.7
        
        # Apply optimization multipliers
        if 'thread' in str(optimizations):
            base_revenue *= 1.5
        
        return round(base_revenue, 2)
    
    async def get_analytics(self, post_id: str) -> Dict[str, Any]:
        """Get Twitter analytics for a post"""
        try:
            metrics = await self.analytics_tracker.track_tweet_performance(
                post_id,
                self.twitter_api
            )
            
            # Calculate additional metrics
            revenue_metrics = await self.analytics_tracker.calculate_monetization_potential(
                metrics,
                self.account_data
            )
            
            return {
                'performance_metrics': metrics,
                'revenue_metrics': {
                    'estimated_revenue': revenue_metrics,
                    'revenue_sources': {
                        'twitter_blue': revenue_metrics * 0.3,
                        'sponsored_potential': revenue_metrics * 0.5,
                        'super_follows': revenue_metrics * 0.2
                    }
                },
                'engagement_insights': {
                    'viral_score': self._calculate_viral_score(metrics),
                    'audience_quality': self._assess_audience_quality(metrics)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get Twitter analytics: {e}")
            return {}
    
    def _calculate_viral_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate viral score for tweet"""
        retweets = metrics.get('retweets', 0)
        likes = metrics.get('likes', 0)
        impressions = metrics.get('impressions', 1)
        
        # Viral score formula
        engagement_rate = ((retweets + likes) / impressions) * 100
        viral_score = min(100, (retweets * 2 + likes) / 100 + engagement_rate * 10)
        
        return round(viral_score, 2)
    
    def _assess_audience_quality(self, metrics: Dict[str, Any]) -> str:
        """Assess audience quality based on metrics"""
        engagement_rate = metrics.get('engagement_rate', 0)
        
        if engagement_rate > 5:
            return 'highly_engaged'
        elif engagement_rate > 2:
            return 'moderately_engaged'
        else:
            return 'low_engagement'
    
    async def schedule_content(self, content: Dict[str, Any], schedule_time: datetime) -> PublishResult:
        """Schedule content for future publishing"""
        # Twitter doesn't have native scheduling via API
        # Would need to implement with a task queue
        return await self._store_scheduled_post(content, schedule_time, "twitter")
    
    async def analyze_audience_engagement(self, business_niche: str) -> Dict[str, Any]:
        """Analyze audience engagement patterns"""
        return {
            'demographics': {
                'age_distribution': await self._get_target_demographics(business_niche),
                'location_insights': {
                    'top_countries': ['US', 'UK', 'CA'],
                    'top_cities': ['New York', 'London', 'Los Angeles']
                },
                'device_usage': {
                    'mobile': 0.75,
                    'desktop': 0.20,
                    'tablet': 0.05
                }
            },
            'engagement_patterns': {
                'peak_hours': self._get_best_posting_times(business_niche),
                'peak_days': ['Tuesday', 'Wednesday', 'Thursday'],
                'content_preferences': {
                    'threads': 0.35,
                    'single_tweets': 0.25,
                    'media_tweets': 0.30,
                    'polls': 0.10
                }
            },
            'content_preferences': {
                'preferred_length': self._get_optimal_thread_length(business_niche),
                'media_engagement': 'high',
                'hashtag_performance': 'moderate',
                'emoji_usage': 'recommended'
            }
        }