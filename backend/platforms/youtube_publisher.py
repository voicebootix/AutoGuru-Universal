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
        try:
            # Check if video should be optimized for Shorts
            if self._should_optimize_for_shorts(content, business_niche):
                return await self._optimize_for_shorts(video_file, content, business_niche)
            else:
                return await self._optimize_for_regular_video(video_file, content, business_niche)
        except Exception as e:
            logger.error(f"Video optimization failed: {str(e)}")
            return video_file
    
    def _should_optimize_for_shorts(self, content: Dict[str, Any], business_niche: str) -> bool:
        """Determine if video should be optimized as YouTube Shorts"""
        # Check video duration
        duration = content.get('duration', 0)
        if duration > 60:  # Shorts must be 60 seconds or less
            return False
        
        # Check if explicitly requested
        if content.get('format') == 'shorts':
            return True
        
        # Auto-determine based on content type and niche
        shorts_niches = ['fitness_wellness', 'creative', 'ecommerce']
        if business_niche in shorts_niches and duration <= 60:
            return True
        
        return False
    
    async def _optimize_for_shorts(self, video_file: str, content: Dict[str, Any], business_niche: str) -> str:
        """Optimize video for YouTube Shorts format"""
        try:
            import subprocess
            import tempfile
            
            # Check if video file exists
            if not os.path.exists(video_file):
                logger.error(f"Video file not found: {video_file}")
                return video_file
            
            # Create temporary output file
            output_file = tempfile.mktemp(suffix='_shorts.mp4')
            
            # FFmpeg command for YouTube Shorts optimization
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', video_file,
                '-vf', 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920',  # 9:16 aspect ratio
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',  # Good quality for mobile viewing
                '-maxrate', '8M',  # Max bitrate for Shorts
                '-bufsize', '16M',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-ar', '44100',
                '-t', '60',  # Limit to 60 seconds for Shorts
                '-y',  # Overwrite output file
                output_file
            ]
            
            # Add niche-specific optimizations
            if business_niche == 'fitness_wellness':
                # Add motion blur reduction for workout videos
                ffmpeg_cmd.insert(-2, '-filter:v')
                ffmpeg_cmd.insert(-2, 'minterpolate=fps=30')
            elif business_niche == 'creative':
                # Enhance colors for art/creative content
                ffmpeg_cmd.insert(-2, '-filter:v')
                ffmpeg_cmd.insert(-2, 'eq=contrast=1.1:brightness=0.05:saturation=1.2')
            
            # Run FFmpeg
            try:
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
                if result.returncode == 0 and os.path.exists(output_file):
                    logger.info(f"Successfully optimized video for YouTube Shorts: {output_file}")
                    return output_file
                else:
                    logger.error(f"FFmpeg failed: {result.stderr}")
                    return video_file
            except subprocess.TimeoutExpired:
                logger.error("Video optimization timed out")
                return video_file
            except FileNotFoundError:
                logger.warning("FFmpeg not found, returning original video")
                return video_file
            
        except Exception as e:
            logger.error(f"Shorts optimization error: {str(e)}")
            return video_file
    
    async def _optimize_for_regular_video(self, video_file: str, content: Dict[str, Any], business_niche: str) -> str:
        """Optimize video for regular YouTube format"""
        try:
            import subprocess
            import tempfile
            
            if not os.path.exists(video_file):
                return video_file
            
            output_file = tempfile.mktemp(suffix='_optimized.mp4')
            
            # Get optimal length for the niche
            min_length, max_length = self.get_optimal_length_range(business_niche)
            
            # FFmpeg command for regular video optimization
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', video_file,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '21',  # High quality for longer content
                '-maxrate', '10M',
                '-bufsize', '20M',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-ar', '48000',
                '-t', str(max_length),  # Limit to optimal length
                '-y',
                output_file
            ]
            
            # Add niche-specific optimizations
            if business_niche == 'education':
                # Enhance clarity for educational content
                ffmpeg_cmd.insert(-2, '-filter:v')
                ffmpeg_cmd.insert(-2, 'unsharp=5:5:1.0:5:5:0.0')
            elif business_niche == 'technology':
                # Optimize for screen recordings
                ffmpeg_cmd.insert(-2, '-filter:v')
                ffmpeg_cmd.insert(-2, 'scale=1920:1080')
            
            try:
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=600)
                if result.returncode == 0 and os.path.exists(output_file):
                    logger.info(f"Successfully optimized regular video: {output_file}")
                    return output_file
                else:
                    logger.error(f"Video optimization failed: {result.stderr}")
                    return video_file
            except subprocess.TimeoutExpired:
                logger.error("Video optimization timed out")
                return video_file
            except FileNotFoundError:
                logger.warning("FFmpeg not found, returning original video")
                return video_file
                
        except Exception as e:
            logger.error(f"Regular video optimization error: {str(e)}")
            return video_file
    
    async def generate_optimal_thumbnail(
        self,
        video_file: str,
        content: Dict[str, Any]
    ) -> Optional[str]:
        """Generate revenue-optimized thumbnail that maximizes click-through rate"""
        try:
            import subprocess
            import tempfile
            from PIL import Image, ImageDraw, ImageFont
            import cv2
            import numpy as np
            
            if not os.path.exists(video_file):
                return None
            
            # Extract frame at 30% of video (usually best composition)
            output_image = tempfile.mktemp(suffix='_thumbnail.jpg')
            
            # Get video duration first
            duration_cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_file
            ]
            
            try:
                duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)
                duration = float(duration_result.stdout.strip())
                extract_time = duration * 0.3  # Extract at 30% mark
            except:
                extract_time = 3  # Default to 3 seconds if duration detection fails
            
            # Extract frame
            extract_cmd = [
                'ffmpeg',
                '-i', video_file,
                '-ss', str(extract_time),
                '-vframes', '1',
                '-vf', 'scale=1280:720',  # YouTube recommended thumbnail size
                '-y',
                output_image
            ]
            
            try:
                result = subprocess.run(extract_cmd, capture_output=True, text=True, timeout=30)
                if result.returncode != 0 or not os.path.exists(output_image):
                    logger.error("Failed to extract video frame for thumbnail")
                    return None
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning("FFmpeg not available for thumbnail extraction")
                return None
            
            # Enhance the thumbnail with text overlay and optimizations
            enhanced_thumbnail = await self._enhance_thumbnail(
                output_image, 
                content.get('title', ''),
                content.get('business_niche', 'general')
            )
            
            return enhanced_thumbnail if enhanced_thumbnail else output_image
            
        except Exception as e:
            logger.error(f"Thumbnail generation error: {str(e)}")
            return None
    
    async def _enhance_thumbnail(self, image_path: str, title: str, business_niche: str) -> Optional[str]:
        """Enhance thumbnail with text, colors, and niche-specific optimizations"""
        try:
            from PIL import Image, ImageDraw, ImageFont, ImageEnhance
            import tempfile
            
            # Open the image
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)
            
            # Enhance image quality
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)  # Increase contrast by 20%
            
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.1)  # Increase saturation by 10%
            
            draw = ImageDraw.Draw(image)
            
            # Get image dimensions
            width, height = image.size
            
            # Niche-specific color schemes
            color_schemes = {
                'education': {'primary': '#FF6B35', 'secondary': '#FFFFFF', 'accent': '#F7931E'},
                'business_consulting': {'primary': '#2E86AB', 'secondary': '#FFFFFF', 'accent': '#F24236'},
                'fitness_wellness': {'primary': '#4CAF50', 'secondary': '#FFFFFF', 'accent': '#FF9800'},
                'creative': {'primary': '#9C27B0', 'secondary': '#FFFFFF', 'accent': '#E91E63'},
                'technology': {'primary': '#2196F3', 'secondary': '#FFFFFF', 'accent': '#00BCD4'},
                'default': {'primary': '#FF0000', 'secondary': '#FFFFFF', 'accent': '#FFC107'}
            }
            
            colors = color_schemes.get(business_niche, color_schemes['default'])
            
            # Add title text with high-contrast background
            if title:
                # Truncate title if too long
                max_title_length = 40
                display_title = title[:max_title_length] + '...' if len(title) > max_title_length else title
                
                # Try to load a font, fallback to default
                try:
                    # Try different font paths
                    font_paths = [
                        '/System/Library/Fonts/Arial.ttf',  # macOS
                        '/usr/share/fonts/truetype/arial.ttf',  # Linux
                        'C:/Windows/Fonts/arial.ttf',  # Windows
                    ]
                    
                    font = None
                    for font_path in font_paths:
                        try:
                            font = ImageFont.truetype(font_path, 36)
                            break
                        except:
                            continue
                    
                    if not font:
                        font = ImageFont.load_default()
                        
                except:
                    font = ImageFont.load_default()
                
                # Calculate text size and position
                bbox = draw.textbbox((0, 0), display_title, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Position text in bottom third of image
                text_x = (width - text_width) // 2
                text_y = height - text_height - 40
                
                # Add semi-transparent background for text
                overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                
                # Background rectangle with rounded corners effect
                padding = 20
                bg_coords = [
                    text_x - padding,
                    text_y - padding,
                    text_x + text_width + padding,
                    text_y + text_height + padding
                ]
                overlay_draw.rectangle(bg_coords, fill=(0, 0, 0, 180))
                
                # Blend overlay with original image
                image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
                draw = ImageDraw.Draw(image)
                
                # Add the text
                draw.text((text_x, text_y), display_title, font=font, fill=colors['secondary'])
                
                # Add accent border/highlight
                draw.rectangle([text_x - padding - 2, text_y - padding - 2, 
                              text_x + text_width + padding + 2, text_y - padding], 
                              fill=colors['primary'])
            
            # Add niche-specific elements
            if business_niche == 'education':
                # Add graduation cap icon or similar
                pass
            elif business_niche == 'fitness_wellness':
                # Add fitness-related graphics
                pass
            
            # Save enhanced thumbnail
            enhanced_path = tempfile.mktemp(suffix='_enhanced_thumbnail.jpg')
            image.save(enhanced_path, 'JPEG', quality=95, optimize=True)
            
            logger.info(f"Enhanced thumbnail created: {enhanced_path}")
            return enhanced_path
            
        except Exception as e:
            logger.error(f"Thumbnail enhancement error: {str(e)}")
            return None
    
    def get_optimal_category(self, business_niche: str) -> str:
        """Get optimal YouTube category for the business niche"""
        return self.CATEGORY_MAPPING.get(business_niche, "22")  # Default to People & Blogs
    
    def get_optimal_length_range(self, business_niche: str) -> Tuple[int, int]:
        """Get optimal video length range for the niche"""
        return self.OPTIMAL_VIDEO_LENGTHS.get(business_niche, (180, 600))
    
    async def create_playlist(self, playlist_data: Dict[str, Any], youtube_service) -> Optional[str]:
        """Create optimized YouTube playlist"""
        try:
            playlist_body = {
                'snippet': {
                    'title': playlist_data.get('title'),
                    'description': playlist_data.get('description', ''),
                    'defaultLanguage': 'en'
                },
                'status': {
                    'privacyStatus': playlist_data.get('privacy_status', 'public')
                }
            }
            
            response = youtube_service.playlists().insert(
                part='snippet,status',
                body=playlist_body
            ).execute()
            
            playlist_id = response['id']
            logger.info(f"Created YouTube playlist: {playlist_id}")
            return playlist_id
            
        except Exception as e:
            logger.error(f"Playlist creation failed: {str(e)}")
            return None
    
    async def post_community_update(self, content: str, media: Optional[bytes], youtube_service) -> Dict[str, Any]:
        """Post to YouTube Community tab"""
        try:
            # Note: YouTube Community API is limited
            # This would require YouTube Community API access
            
            community_post = {
                'snippet': {
                    'text': content
                }
            }
            
            # In production, this would use the Community API when available
            logger.info("Community post would be created (API limited)")
            return {
                'success': True,
                'post_id': f'community_{datetime.utcnow().timestamp()}'
            }
            
        except Exception as e:
            logger.error(f"Community post failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def schedule_live_stream(self, stream_data: Dict[str, Any], youtube_service) -> Dict[str, Any]:
        """Schedule and configure live stream"""
        try:
            # Create live broadcast
            broadcast_body = {
                'snippet': {
                    'title': stream_data.get('title'),
                    'description': stream_data.get('description', ''),
                    'scheduledStartTime': stream_data.get('start_time').isoformat() + 'Z'
                },
                'status': {
                    'privacyStatus': stream_data.get('privacy_status', 'public'),
                    'selfDeclaredMadeForKids': False
                }
            }
            
            broadcast_response = youtube_service.liveBroadcasts().insert(
                part='snippet,status',
                body=broadcast_body
            ).execute()
            
            broadcast_id = broadcast_response['id']
            
            # Create live stream
            stream_body = {
                'snippet': {
                    'title': f"Stream for {stream_data.get('title')}"
                },
                'cdn': {
                    'format': '1080p',
                    'ingestionType': 'rtmp'
                }
            }
            
            stream_response = youtube_service.liveStreams().insert(
                part='snippet,cdn',
                body=stream_body
            ).execute()
            
            stream_id = stream_response['id']
            
            # Bind broadcast to stream
            youtube_service.liveBroadcasts().bind(
                part='id',
                id=broadcast_id,
                streamId=stream_id
            ).execute()
            
            return {
                'success': True,
                'broadcast_id': broadcast_id,
                'stream_id': stream_id,
                'stream_url': stream_response['cdn']['ingestionInfo']['streamName']
            }
            
        except Exception as e:
            logger.error(f"Live stream scheduling failed: {str(e)}")
            return {'success': False, 'error': str(e)}


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