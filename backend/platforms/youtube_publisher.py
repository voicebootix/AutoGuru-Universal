"""
YouTube Enhanced Platform Publisher for AutoGuru Universal.

This module provides YouTube publishing capabilities with revenue optimization,
analytics, and comprehensive video management. It works universally across all
business niches without hardcoded business logic.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import aiohttp
from dataclasses import dataclass
import os
import tempfile

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

from backend.platforms.enhanced_base_publisher import (
    UniversalPlatformPublisher,
    PublishResult,
    PublishStatus,
    RevenueMetrics,
    PerformanceMetrics
)
from backend.models.content_models import BusinessNicheType
from backend.core.viral_engine import UniversalViralEngine
from backend.utils.encryption import EncryptionManager

logger = logging.getLogger(__name__)


@dataclass
class YouTubeVideoMetadata:
    """YouTube video metadata"""
    title: str
    description: str
    tags: List[str]
    category_id: str
    privacy_status: str = "public"
    made_for_kids: bool = False
    default_language: str = "en"
    recording_date: Optional[datetime] = None
    location: Optional[Dict[str, float]] = None
    custom_thumbnail: Optional[str] = None


class YouTubeVideoOptimizer:
    """Optimize videos for YouTube algorithm and revenue"""
    
    OPTIMAL_VIDEO_LENGTHS = {
        "education": (600, 1200),      # 10-20 minutes
        "business_consulting": (480, 900),  # 8-15 minutes
        "fitness_wellness": (300, 600),     # 5-10 minutes
        "creative": (180, 600),             # 3-10 minutes
        "ecommerce": (120, 300),            # 2-5 minutes
        "local_service": (60, 180),         # 1-3 minutes
        "technology": (600, 1800),          # 10-30 minutes
        "non_profit": (180, 480)            # 3-8 minutes
    }
    
    CATEGORY_MAPPING = {
        "education": "27",          # Education
        "business_consulting": "24", # Entertainment (for business content)
        "fitness_wellness": "17",    # Sports
        "creative": "1",            # Film & Animation
        "ecommerce": "22",          # People & Blogs
        "local_service": "22",      # People & Blogs
        "technology": "28",         # Science & Technology
        "non_profit": "29"          # Nonprofits & Activism
    }
    
    async def optimize_for_algorithm(
        self,
        video_file: str,
        content: Dict[str, Any],
        business_niche: str
    ) -> str:
        """Optimize video for YouTube algorithm"""
        # In production, this would process the video
        # For now, return the original file
        return video_file
    
    async def generate_optimal_thumbnail(
        self,
        video_file: str,
        content: Dict[str, Any]
    ) -> Optional[str]:
        """Generate revenue-optimized thumbnail"""
        # In production, this would analyze the video and create a thumbnail
        # that maximizes click-through rate
        return None
    
    def get_optimal_category(self, business_niche: str) -> str:
        """Get optimal YouTube category for the business niche"""
        return self.CATEGORY_MAPPING.get(business_niche, "22")  # Default to People & Blogs
    
    def get_optimal_length_range(self, business_niche: str) -> Tuple[int, int]:
        """Get optimal video length range for the niche"""
        return self.OPTIMAL_VIDEO_LENGTHS.get(business_niche, (180, 600))


class YouTubeAnalyticsTracker:
    """Track YouTube-specific analytics and revenue"""
    
    def __init__(self, youtube_service, analytics_service):
        self.youtube_service = youtube_service
        self.analytics_service = analytics_service
    
    async def get_video_analytics(self, video_id: str) -> Dict[str, Any]:
        """Get comprehensive video analytics"""
        try:
            # Get video statistics
            video_response = self.youtube_service.videos().list(
                part="statistics,contentDetails",
                id=video_id
            ).execute()
            
            if not video_response.get('items'):
                return {}
            
            stats = video_response['items'][0]['statistics']
            details = video_response['items'][0]['contentDetails']
            
            return {
                'views': int(stats.get('viewCount', 0)),
                'likes': int(stats.get('likeCount', 0)),
                'dislikes': int(stats.get('dislikeCount', 0)),
                'comments': int(stats.get('commentCount', 0)),
                'shares': 0,  # Not directly available
                'duration': details.get('duration'),
                'favorites': int(stats.get('favoriteCount', 0))
            }
            
        except Exception as e:
            logger.error(f"Failed to get YouTube analytics: {str(e)}")
            return {}
    
    async def calculate_estimated_revenue(
        self,
        video_id: str,
        views: int,
        business_niche: str
    ) -> float:
        """Calculate estimated ad revenue from video"""
        # CPM rates by niche (simplified)
        niche_cpm = {
            "education": 4.0,
            "business_consulting": 8.0,
            "fitness_wellness": 3.5,
            "creative": 2.5,
            "ecommerce": 5.0,
            "local_service": 3.0,
            "technology": 6.0,
            "non_profit": 2.0
        }
        
        cpm = niche_cpm.get(business_niche, 3.0)
        # YouTube takes 45%, creator gets 55%
        creator_share = 0.55
        
        estimated_revenue = (views / 1000) * cpm * creator_share
        return round(estimated_revenue, 2)


class YouTubeEnhancedPublisher(UniversalPlatformPublisher):
    """Enhanced YouTube publisher with revenue optimization and analytics"""
    
    SCOPES = [
        'https://www.googleapis.com/auth/youtube.upload',
        'https://www.googleapis.com/auth/youtube',
        'https://www.googleapis.com/auth/youtubepartner',
        'https://www.googleapis.com/auth/yt-analytics.readonly'
    ]
    
    def __init__(self, client_id: str):
        super().__init__(client_id, "youtube")
        self.youtube_service = None
        self.analytics_service = None
        self.video_optimizer = YouTubeVideoOptimizer()
        self.analytics_tracker = None
        self.viral_engine = UniversalViralEngine()
        
    async def authenticate(self, credentials: Dict[str, str]) -> bool:
        """Authenticate with YouTube API using OAuth2"""
        try:
            # Check if we have encrypted credentials
            if 'encrypted_credentials' in credentials:
                decrypted = self.encryption_manager.decrypt_credentials(
                    credentials['encrypted_credentials']
                )
                credentials = json.loads(decrypted)
            
            # Create credentials object
            creds = None
            if 'access_token' in credentials:
                creds = Credentials(
                    token=credentials['access_token'],
                    refresh_token=credentials.get('refresh_token'),
                    token_uri=credentials.get('token_uri', 'https://oauth2.googleapis.com/token'),
                    client_id=credentials.get('client_id'),
                    client_secret=credentials.get('client_secret'),
                    scopes=self.SCOPES
                )
            
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    return False
            
            # Build YouTube services
            self.youtube_service = build('youtube', 'v3', credentials=creds)
            self.analytics_service = build('youtubeAnalytics', 'v2', credentials=creds)
            
            # Initialize analytics tracker
            self.analytics_tracker = YouTubeAnalyticsTracker(
                self.youtube_service,
                self.analytics_service
            )
            
            # Test connection
            response = self.youtube_service.channels().list(
                part='snippet',
                mine=True
            ).execute()
            
            if not response.get('items'):
                return False
            
            self._authenticated = True
            self._credentials = credentials
            
            # Store encrypted credentials
            encrypted = self.encryption_manager.encrypt_credentials(
                json.dumps({
                    'access_token': creds.token,
                    'refresh_token': creds.refresh_token,
                    'client_id': credentials.get('client_id'),
                    'client_secret': credentials.get('client_secret')
                })
            )
            self._credentials['encrypted_credentials'] = encrypted
            
            self.log_activity('authenticate', {'status': 'success'})
            return True
            
        except Exception as e:
            logger.error(f"YouTube authentication failed: {str(e)}")
            self.log_activity('authenticate', {'error': str(e)}, success=False)
            return False
    
    async def publish_content(self, content: Dict[str, Any]) -> PublishResult:
        """Publish video content to YouTube with revenue optimization"""
        try:
            if not self._authenticated:
                return self.handle_publish_error("youtube", "Not authenticated")
            
            # 1. Pre-publish optimization
            optimizations = await self.optimize_for_revenue(content)
            
            # 2. Generate revenue-optimized metadata
            metadata = await self.generate_youtube_metadata(content, optimizations)
            
            # 3. Optimize video content if provided
            video_file = content.get('video_file')
            if not video_file:
                return self.handle_publish_error("youtube", "No video file provided")
            
            optimized_video = await self.video_optimizer.optimize_for_algorithm(
                video_file,
                content,
                await self.detect_business_niche(content.get('text', ''))
            )
            
            # 4. Create thumbnail that maximizes clicks
            thumbnail = await self.video_optimizer.generate_optimal_thumbnail(
                optimized_video,
                content
            )
            
            # 5. Upload video
            video_id = await self._upload_video(optimized_video, metadata)
            
            if not video_id:
                return self.handle_publish_error("youtube", "Failed to upload video")
            
            # 6. Upload custom thumbnail if generated
            if thumbnail:
                await self._upload_thumbnail(video_id, thumbnail)
            
            # 7. Add revenue optimization elements
            await self._add_revenue_elements(video_id, content)
            
            # 8. Track initial metrics
            metrics = await self.analytics_tracker.get_video_analytics(video_id)
            
            # 9. Calculate revenue potential
            business_niche = await self.detect_business_niche(content.get('text', ''))
            revenue_potential = await self.calculate_revenue_potential(content, metadata)
            
            # 10. Create performance metrics
            performance_metrics = PerformanceMetrics(
                engagement_rate=0.0,  # Will be updated after video gets views
                reach=0,
                impressions=0,
                clicks=0,
                shares=0,
                saves=0,
                comments=metrics.get('comments', 0),
                likes=metrics.get('likes', 0),
                video_views=metrics.get('views', 0)
            )
            
            # 11. Create revenue metrics
            revenue_metrics = RevenueMetrics(
                estimated_revenue_potential=revenue_potential,
                actual_revenue=0.0,
                conversion_rate=0.0,
                revenue_per_engagement=0.0,
                revenue_per_impression=0.0
            )
            
            return PublishResult(
                platform="youtube",
                status=PublishStatus.PUBLISHED,
                post_id=video_id,
                post_url=f"https://www.youtube.com/watch?v={video_id}",
                metrics=metrics,
                revenue_metrics=revenue_metrics,
                performance_metrics=performance_metrics,
                optimization_suggestions=optimizations.get('suggestions', []),
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"YouTube publish failed: {str(e)}")
            return self.handle_publish_error("youtube", str(e))
    
    async def _upload_video(self, video_file: str, metadata: YouTubeVideoMetadata) -> Optional[str]:
        """Upload video to YouTube"""
        try:
            body = {
                'snippet': {
                    'title': metadata.title,
                    'description': metadata.description,
                    'tags': metadata.tags,
                    'categoryId': metadata.category_id,
                    'defaultLanguage': metadata.default_language
                },
                'status': {
                    'privacyStatus': metadata.privacy_status,
                    'selfDeclaredMadeForKids': metadata.made_for_kids,
                    'embeddable': True,
                    'publicStatsViewable': True
                },
                'recordingDetails': {}
            }
            
            # Add location if provided
            if metadata.location:
                body['recordingDetails']['location'] = {
                    'latitude': metadata.location['latitude'],
                    'longitude': metadata.location['longitude']
                }
            
            # Add recording date if provided
            if metadata.recording_date:
                body['recordingDetails']['recordingDate'] = metadata.recording_date.isoformat()
            
            # Create media upload
            media = MediaFileUpload(
                video_file,
                chunksize=-1,
                resumable=True,
                mimetype='video/*'
            )
            
            # Insert video
            insert_request = self.youtube_service.videos().insert(
                part=','.join(body.keys()),
                body=body,
                media_body=media
            )
            
            response = insert_request.execute()
            video_id = response['id']
            
            logger.info(f"Successfully uploaded video: {video_id}")
            return video_id
            
        except HttpError as e:
            logger.error(f"YouTube API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Video upload failed: {str(e)}")
            return None
    
    async def _upload_thumbnail(self, video_id: str, thumbnail_path: str) -> bool:
        """Upload custom thumbnail for video"""
        try:
            self.youtube_service.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(thumbnail_path, mimetype='image/jpeg')
            ).execute()
            
            logger.info(f"Successfully uploaded thumbnail for video: {video_id}")
            return True
            
        except Exception as e:
            logger.error(f"Thumbnail upload failed: {str(e)}")
            return False
    
    async def _add_revenue_elements(self, video_id: str, content: Dict[str, Any]) -> None:
        """Add revenue optimization elements to video"""
        try:
            # Add end screens (if video is long enough)
            # Add cards
            # Add chapters
            # These would be implemented based on YouTube API capabilities
            pass
        except Exception as e:
            logger.error(f"Failed to add revenue elements: {str(e)}")
    
    async def generate_youtube_metadata(
        self,
        content: Dict[str, Any],
        optimizations: Dict[str, Any]
    ) -> YouTubeVideoMetadata:
        """Generate revenue-optimized YouTube metadata"""
        business_niche = await self.detect_business_niche(content.get('text', ''))
        
        # Generate optimized title
        title = await self._generate_optimized_title(content, business_niche)
        
        # Generate optimized description
        description = await self._generate_optimized_description(content, business_niche)
        
        # Generate revenue-optimized tags
        tags = await self._generate_optimized_tags(content, business_niche)
        
        # Get optimal category
        category_id = self.video_optimizer.get_optimal_category(business_niche)
        
        return YouTubeVideoMetadata(
            title=title,
            description=description,
            tags=tags,
            category_id=category_id,
            privacy_status=content.get('privacy_status', 'public'),
            made_for_kids=content.get('made_for_kids', False)
        )
    
    async def _generate_optimized_title(
        self,
        content: Dict[str, Any],
        business_niche: str
    ) -> str:
        """Generate SEO and CTR optimized title"""
        base_title = content.get('title', content.get('text', '')[:50])
        
        # Title optimization patterns by niche
        title_patterns = {
            "education": "{} | Step-by-Step Tutorial",
            "business_consulting": "How to {} (Proven Strategy)",
            "fitness_wellness": "{} - See Results in Days!",
            "creative": "{} | Creative Process Revealed",
            "ecommerce": "{} - Honest Review & Demo",
            "local_service": "{} | Local Expert Guide",
            "technology": "{} Explained (Beginner Friendly)",
            "non_profit": "{} | Make a Real Difference"
        }
        
        pattern = title_patterns.get(business_niche, "{}")
        optimized_title = pattern.format(base_title)
        
        # Ensure title is within YouTube's 100 character limit
        if len(optimized_title) > 100:
            optimized_title = optimized_title[:97] + "..."
            
        return optimized_title
    
    async def _generate_optimized_description(
        self,
        content: Dict[str, Any],
        business_niche: str
    ) -> str:
        """Generate revenue-optimized description with SEO"""
        base_description = content.get('description', content.get('text', ''))
        
        # Add timestamps section
        timestamps = "⏱️ TIMESTAMPS:\n0:00 Introduction\n"
        
        # Add links section
        links = "\n📱 CONNECT WITH US:\n"
        if content.get('website'):
            links += f"Website: {content['website']}\n"
        
        # Add call-to-action
        cta_map = {
            "education": "\n📚 Want to learn more? Check out our full course!",
            "business_consulting": "\n💼 Ready to scale your business? Book a free consultation!",
            "fitness_wellness": "\n💪 Start your transformation today!",
            "creative": "\n🎨 Get inspired and create something amazing!",
            "ecommerce": "\n🛍️ Get yours today with our special discount!",
            "local_service": "\n📞 Contact us for a free quote!",
            "technology": "\n💻 Try it yourself with our free resources!",
            "non_profit": "\n❤️ Join our mission and make a difference!"
        }
        
        cta = cta_map.get(business_niche, "\n✨ Thanks for watching!")
        
        # Combine all elements
        optimized_description = f"{base_description}\n\n{timestamps}\n{links}\n{cta}"
        
        # Add relevant hashtags
        hashtags = await self._get_trending_hashtags(business_niche)
        if hashtags:
            optimized_description += f"\n\n{' '.join(hashtags[:10])}"
        
        # Ensure description is within YouTube's 5000 character limit
        if len(optimized_description) > 5000:
            optimized_description = optimized_description[:4997] + "..."
            
        return optimized_description
    
    async def _generate_optimized_tags(
        self,
        content: Dict[str, Any],
        business_niche: str
    ) -> List[str]:
        """Generate revenue-optimized tags for discoverability"""
        tags = []
        
        # Extract keywords from content
        text = content.get('text', '')
        keywords = await self._extract_keywords(text)
        tags.extend(keywords[:10])
        
        # Add niche-specific tags
        niche_tags = {
            "education": ["tutorial", "how to", "learn", "course", "education"],
            "business_consulting": ["business", "entrepreneur", "consulting", "strategy", "growth"],
            "fitness_wellness": ["fitness", "workout", "health", "wellness", "exercise"],
            "creative": ["creative", "art", "design", "tutorial", "process"],
            "ecommerce": ["review", "unboxing", "product", "shopping", "demo"],
            "local_service": ["local", "service", "near me", "best", "professional"],
            "technology": ["tech", "technology", "tutorial", "software", "app"],
            "non_profit": ["charity", "nonprofit", "volunteer", "donation", "cause"]
        }
        
        tags.extend(niche_tags.get(business_niche, []))
        
        # Add trending tags
        trending = await self._get_trending_tags(business_niche)
        tags.extend(trending[:5])
        
        # Remove duplicates and limit to 500 characters total
        unique_tags = list(dict.fromkeys(tags))
        final_tags = []
        total_length = 0
        
        for tag in unique_tags:
            if total_length + len(tag) + 1 <= 500:  # +1 for comma separator
                final_tags.append(tag)
                total_length += len(tag) + 1
            else:
                break
                
        return final_tags
    
    async def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text using simple algorithm"""
        # In production, use NLP for better keyword extraction
        words = text.lower().split()
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        return keywords[:20]
    
    async def _get_trending_hashtags(self, business_niche: str) -> List[str]:
        """Get trending hashtags for the niche"""
        # In production, fetch from YouTube Trends API
        base_hashtags = ['#shorts', '#youtube', '#viral', '#trending']
        return base_hashtags
    
    async def _get_trending_tags(self, business_niche: str) -> List[str]:
        """Get trending tags for the niche"""
        # In production, analyze trending videos in the niche
        return ['2024', 'latest', 'new', 'best', 'top']
    
    async def get_optimal_posting_time(
        self,
        content_type: str,
        business_niche: str
    ) -> datetime:
        """AI-powered optimal YouTube posting time"""
        # Get audience timezone data
        audience_data = await self.analyze_audience_engagement(business_niche)
        
        # Optimal posting times by niche (simplified)
        optimal_times = {
            "education": [9, 15, 20],        # Morning, afternoon, evening
            "business_consulting": [8, 12, 17],  # Business hours
            "fitness_wellness": [6, 12, 18],     # Early morning, lunch, evening
            "creative": [10, 15, 21],            # Late morning, afternoon, night
            "ecommerce": [10, 14, 19],           # Shopping hours
            "local_service": [8, 13, 17],        # Local business hours
            "technology": [10, 16, 20],          # Tech audience hours
            "non_profit": [11, 14, 19]           # Midday and evening
        }
        
        hours = optimal_times.get(business_niche, [12, 18, 20])
        
        # Get next available optimal time
        now = datetime.utcnow()
        for hour in hours:
            potential_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if potential_time > now:
                return potential_time
        
        # If no time today, use first time tomorrow
        tomorrow = now + timedelta(days=1)
        return tomorrow.replace(hour=hours[0], minute=0, second=0, microsecond=0)
    
    async def analyze_audience_engagement(self, business_niche: str) -> Dict[str, Any]:
        """Analyze YouTube audience for this business niche"""
        try:
            if not self.analytics_service:
                return self._get_default_audience_data(business_niche)
            
            # Get channel analytics
            # In production, this would fetch real analytics data
            audience_data = {
                'demographics': {
                    'age_groups': self._get_niche_demographics(business_niche),
                    'gender_split': {'male': 0.55, 'female': 0.45},
                    'top_countries': ['US', 'UK', 'CA', 'AU', 'IN']
                },
                'engagement_patterns': {
                    'peak_hours': [9, 14, 20],
                    'peak_days': ['Tuesday', 'Thursday', 'Saturday'],
                    'average_view_duration': 6.5,  # minutes
                    'engagement_rate': 4.2  # percentage
                },
                'content_preferences': {
                    'preferred_length': self.video_optimizer.get_optimal_length_range(business_niche),
                    'top_performing_formats': ['tutorial', 'list', 'review'],
                    'engagement_triggers': ['questions', 'challenges', 'giveaways']
                },
                'revenue_metrics': {
                    'average_rpm': 3.50,  # Revenue per mille
                    'top_revenue_sources': ['ads', 'channel_memberships', 'super_thanks'],
                    'conversion_rate': 0.02
                }
            }
            
            return audience_data
            
        except Exception as e:
            logger.error(f"YouTube audience analysis failed: {str(e)}")
            return self._get_default_audience_data(business_niche)
    
    def _get_niche_demographics(self, business_niche: str) -> Dict[str, float]:
        """Get age demographics by niche"""
        demographics_map = {
            "education": {"13-17": 0.15, "18-24": 0.35, "25-34": 0.30, "35-44": 0.15, "45+": 0.05},
            "business_consulting": {"18-24": 0.10, "25-34": 0.40, "35-44": 0.30, "45-54": 0.15, "55+": 0.05},
            "fitness_wellness": {"18-24": 0.25, "25-34": 0.35, "35-44": 0.25, "45-54": 0.10, "55+": 0.05},
            "creative": {"13-17": 0.20, "18-24": 0.30, "25-34": 0.25, "35-44": 0.15, "45+": 0.10},
            "ecommerce": {"18-24": 0.20, "25-34": 0.30, "35-44": 0.25, "45-54": 0.15, "55+": 0.10},
            "local_service": {"25-34": 0.20, "35-44": 0.30, "45-54": 0.25, "55-64": 0.15, "65+": 0.10},
            "technology": {"13-17": 0.10, "18-24": 0.30, "25-34": 0.35, "35-44": 0.20, "45+": 0.05},
            "non_profit": {"18-24": 0.15, "25-34": 0.25, "35-44": 0.25, "45-54": 0.20, "55+": 0.15}
        }
        
        return demographics_map.get(business_niche, {"18-24": 0.25, "25-34": 0.35, "35-44": 0.25, "45+": 0.15})
    
    def _get_default_audience_data(self, business_niche: str) -> Dict[str, Any]:
        """Get default audience data when analytics unavailable"""
        return {
            'demographics': {
                'age_groups': self._get_niche_demographics(business_niche),
                'gender_split': {'male': 0.50, 'female': 0.50},
                'top_countries': ['US', 'UK', 'CA']
            },
            'engagement_patterns': {
                'peak_hours': [12, 18, 20],
                'peak_days': ['Wednesday', 'Friday', 'Sunday'],
                'average_view_duration': 5.0,
                'engagement_rate': 3.0
            },
            'content_preferences': {
                'preferred_length': (300, 600),
                'top_performing_formats': ['tutorial', 'review'],
                'engagement_triggers': ['questions', 'tips']
            },
            'revenue_metrics': {
                'average_rpm': 2.50,
                'top_revenue_sources': ['ads'],
                'conversion_rate': 0.015
            }
        }
    
    async def get_platform_optimizations(
        self,
        content: Dict[str, Any],
        business_niche: str
    ) -> Dict[str, Any]:
        """Get YouTube-specific optimizations"""
        optimizations = {
            'video_optimizations': {
                'optimal_length': self.video_optimizer.get_optimal_length_range(business_niche),
                'thumbnail_tips': [
                    "Use bright, contrasting colors",
                    "Include clear, readable text",
                    "Show faces with expressions",
                    "Create curiosity without clickbait"
                ],
                'title_formula': await self._get_title_formula(business_niche),
                'description_template': await self._get_description_template(business_niche)
            },
            'algorithm_optimizations': {
                'retention_hooks': [
                    "Start with a hook in first 5 seconds",
                    "Use pattern interrupts every 30 seconds",
                    "Add chapters for easy navigation",
                    "End with a strong CTA"
                ],
                'engagement_tactics': [
                    "Ask questions to encourage comments",
                    "Create polls and community posts",
                    "Respond to early comments",
                    "Use end screens effectively"
                ]
            },
            'monetization_optimizations': {
                'ad_placement': "Place ads at natural breaks",
                'affiliate_integration': "Add affiliate links in description",
                'membership_perks': "Offer exclusive content for members",
                'merchandise_shelf': "Enable merch shelf if eligible"
            }
        }
        
        return optimizations
    
    async def _get_title_formula(self, business_niche: str) -> str:
        """Get proven title formula for the niche"""
        formulas = {
            "education": "[Topic] - Complete Beginner's Guide (2024)",
            "business_consulting": "How I [Achievement] in [Timeframe]",
            "fitness_wellness": "[Number] [Exercise/Diet] Tips That Actually Work",
            "creative": "[Project] From Start to Finish - Full Process",
            "ecommerce": "[Product] Review - Is It Worth Your Money?",
            "local_service": "Best [Service] in [Location] - Honest Review",
            "technology": "[Tech Topic] Explained in [Number] Minutes",
            "non_profit": "How You Can Help [Cause] Today"
        }
        
        return formulas.get(business_niche, "How to [Topic] - Step by Step Guide")
    
    async def _get_description_template(self, business_niche: str) -> str:
        """Get description template for the niche"""
        return """
[Brief intro paragraph about the video]

⏱️ TIMESTAMPS:
0:00 Introduction
[Add more timestamps]

📝 KEY POINTS:
• [Point 1]
• [Point 2]
• [Point 3]

🔗 RESOURCES MENTIONED:
• [Resource 1]: [Link]
• [Resource 2]: [Link]

📱 CONNECT WITH ME:
• Website: [Your website]
• Instagram: [Your Instagram]
• Email: [Your email]

[Call to action paragraph]

#[Niche] #[Topic] #YouTube
"""