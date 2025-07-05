"""
Base Platform Publisher for AutoGuru Universal.

This module provides the abstract base class for all social media platform publishers.
It defines the universal interface that works across all business niches without
hardcoded business logic. All platform-specific publishers must inherit from this base.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import asyncio
import logging
from dataclasses import dataclass, field

from backend.models.content_models import (
    Platform, 
    PlatformContent, 
    BusinessNicheType,
    ContentFormat
)

logger = logging.getLogger(__name__)


class PublishStatus(str, Enum):
    """Status of a publish operation"""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    SCHEDULED = "scheduled"
    RATE_LIMITED = "rate_limited"
    CONTENT_REJECTED = "content_rejected"
    AUTH_FAILED = "auth_failed"


class MediaType(str, Enum):
    """Media types for content"""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    CAROUSEL = "carousel"
    REEL = "reel"
    STORY = "story"


@dataclass
class MediaAsset:
    """Represents a media asset for publishing"""
    type: MediaType
    url: Optional[str] = None
    data: Optional[bytes] = None
    filename: Optional[str] = None
    mime_type: Optional[str] = None
    dimensions: Optional[Dict[str, int]] = None  # width, height
    duration: Optional[int] = None  # For video/audio in seconds
    thumbnail_url: Optional[str] = None
    alt_text: Optional[str] = None  # Accessibility
    
    def __post_init__(self):
        if not self.url and not self.data:
            raise ValueError("MediaAsset must have either url or data")


@dataclass 
class VideoContent:
    """Video content for publishing"""
    video_asset: MediaAsset
    title: str
    description: str
    thumbnail: Optional[MediaAsset] = None
    captions: Optional[str] = None
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    location: Optional[Dict[str, Any]] = None
    is_reel: bool = False
    is_igtv: bool = False
    cover_frame_time: Optional[int] = None  # Time in ms for cover frame


@dataclass
class StoryContent:
    """Story content for publishing"""
    media_asset: MediaAsset
    text_overlay: Optional[str] = None
    stickers: List[Dict[str, Any]] = field(default_factory=list)
    mentions: List[Dict[str, Any]] = field(default_factory=list)  # With position
    hashtags: List[str] = field(default_factory=list)
    location: Optional[Dict[str, Any]] = None
    link: Optional[str] = None
    interactive_elements: List[Dict[str, Any]] = field(default_factory=list)  # Polls, questions, etc.
    duration_seconds: int = 15
    background_color: Optional[str] = None


@dataclass
class PublishResult:
    """Result of a publish operation"""
    status: PublishStatus
    platform: Platform
    post_id: Optional[str] = None
    post_url: Optional[str] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    published_at: Optional[datetime] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    rate_limit_info: Optional[Dict[str, Any]] = None
    
    @property
    def is_success(self) -> bool:
        return self.status == PublishStatus.SUCCESS


@dataclass
class ScheduleResult:
    """Result of a scheduling operation"""
    status: PublishStatus
    platform: Platform
    schedule_id: Optional[str] = None
    scheduled_time: Optional[datetime] = None
    error_message: Optional[str] = None
    confirmation_url: Optional[str] = None
    can_edit: bool = True
    can_cancel: bool = True


@dataclass
class InstagramAnalytics:
    """Instagram-specific analytics data"""
    post_id: str
    post_type: str  # feed, story, reel, igtv
    impressions: int = 0
    reach: int = 0
    engagement: int = 0  # likes + comments + saves + shares
    likes: int = 0
    comments: int = 0
    saves: int = 0
    shares: int = 0
    profile_visits: int = 0
    website_clicks: int = 0
    email_clicks: int = 0
    call_clicks: int = 0
    direction_clicks: int = 0
    follows: int = 0
    unfollows: int = 0
    demographic_data: Dict[str, Any] = field(default_factory=dict)
    discovery_data: Dict[str, Any] = field(default_factory=dict)  # How users found the post
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OptimizedContent:
    """Content optimized for platform algorithms"""
    original_content: PlatformContent
    optimized_text: str
    suggested_hashtags: List[str]
    suggested_mentions: List[str]
    best_posting_time: datetime
    algorithm_score: float  # 0.0 to 1.0
    optimization_reasons: List[str]
    trending_elements: List[Dict[str, Any]]
    engagement_predictions: Dict[str, float]
    a_b_test_variations: Optional[List[Dict[str, Any]]] = None


class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, calls_per_minute: int = 60, calls_per_hour: int = 1000):
        self.calls_per_minute = calls_per_minute
        self.calls_per_hour = calls_per_hour
        self.minute_calls: List[datetime] = []
        self.hour_calls: List[datetime] = []
        self._lock = asyncio.Lock()
    
    async def check_rate_limit(self) -> Tuple[bool, Optional[int]]:
        """
        Check if we can make a call without exceeding rate limits.
        
        Returns:
            Tuple of (can_proceed, wait_seconds)
        """
        async with self._lock:
            now = datetime.utcnow()
            
            # Clean old calls
            self.minute_calls = [
                call for call in self.minute_calls 
                if (now - call).total_seconds() < 60
            ]
            self.hour_calls = [
                call for call in self.hour_calls 
                if (now - call).total_seconds() < 3600
            ]
            
            # Check minute limit
            if len(self.minute_calls) >= self.calls_per_minute:
                oldest_minute_call = min(self.minute_calls)
                wait_seconds = 60 - (now - oldest_minute_call).total_seconds()
                return False, int(wait_seconds) + 1
            
            # Check hour limit
            if len(self.hour_calls) >= self.calls_per_hour:
                oldest_hour_call = min(self.hour_calls)
                wait_seconds = 3600 - (now - oldest_hour_call).total_seconds()
                return False, int(wait_seconds) + 1
            
            # Record the call
            self.minute_calls.append(now)
            self.hour_calls.append(now)
            
            return True, None
    
    async def wait_if_needed(self) -> None:
        """Wait if rate limited before proceeding"""
        can_proceed, wait_seconds = await self.check_rate_limit()
        if not can_proceed and wait_seconds:
            logger.warning(f"Rate limited, waiting {wait_seconds} seconds")
            await asyncio.sleep(wait_seconds)
            # Recursive call to check again
            await self.wait_if_needed()


class BasePlatformPublisher(ABC):
    """
    Abstract base class for all platform publishers.
    
    This class defines the universal interface that all platform-specific
    publishers must implement. It provides common functionality for rate
    limiting, error handling, and credential management.
    """
    
    def __init__(self, platform: Platform, business_id: str):
        """
        Initialize the base publisher.
        
        Args:
            platform: The social media platform
            business_id: Unique identifier for the business
        """
        self.platform = platform
        self.business_id = business_id
        self.rate_limiter = self._create_rate_limiter()
        self._credentials: Optional[Dict[str, Any]] = None
        self._authenticated = False
        
    @abstractmethod
    def _create_rate_limiter(self) -> RateLimiter:
        """Create platform-specific rate limiter"""
        pass
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """
        Authenticate with the platform.
        
        Args:
            credentials: Platform-specific credentials
            
        Returns:
            Success status
        """
        try:
            logger.info(f"Authenticating with {self.platform.value}")
            
            # Validate required credentials
            required_keys = self._get_required_credential_keys()
            missing_keys = [key for key in required_keys if key not in credentials]
            
            if missing_keys:
                logger.error(f"Missing required credentials: {missing_keys}")
                return False
            
            # Store encrypted credentials
            from backend.utils.encryption import EncryptionManager
            encryption_manager = EncryptionManager()
            
            encrypted_credentials = {}
            for key, value in credentials.items():
                encrypted_credentials[key] = encryption_manager.encrypt(str(value))
            
            self._credentials = encrypted_credentials
            
            # Test authentication with platform API
            auth_success = await self._test_platform_connection()
            
            if auth_success:
                self._authenticated = True
                logger.info(f"Successfully authenticated with {self.platform.value}")
                self._log_activity("authentication", {"status": "success"})
            else:
                self._authenticated = False
                logger.error(f"Authentication failed with {self.platform.value}")
                self._log_activity("authentication", {"status": "failed"}, success=False)
            
            return auth_success
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            self._log_activity("authentication", {"error": str(e)}, success=False)
            return False
    
    @abstractmethod
    async def validate_content(self, content: PlatformContent) -> Tuple[bool, Optional[str]]:
        """
        Validate content against platform requirements.
        
        Args:
            content: Content to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            logger.info(f"Validating content for {self.platform.value}")
            
            # Get platform-specific validation rules
            validation_rules = self._get_platform_validation_rules()
            
            # Validate content text length
            if content.content_text:
                text_length = len(content.content_text)
                max_length = validation_rules.get('max_text_length', 2000)
                if text_length > max_length:
                    return False, f"Content text exceeds maximum length of {max_length} characters"
            
            # Validate media assets
            if hasattr(content, 'media_assets') and content.media_assets:
                for media in content.media_assets:
                    is_valid, error = await self._validate_media_asset(media, validation_rules)
                    if not is_valid:
                        return False, error
            
            # Validate hashtags
            if hasattr(content, 'hashtags') and content.hashtags:
                max_hashtags = validation_rules.get('max_hashtags', 30)
                if len(content.hashtags) > max_hashtags:
                    return False, f"Too many hashtags. Maximum allowed: {max_hashtags}"
            
            # Check content policy compliance
            policy_check = await self.check_content_policy(content)
            if not policy_check[0]:
                return False, f"Content policy violation: {policy_check[1]}"
            
            # Platform-specific validation
            platform_valid, platform_error = await self._validate_platform_specific(content, validation_rules)
            if not platform_valid:
                return False, platform_error
            
            logger.info(f"Content validation passed for {self.platform.value}")
            return True, None
            
        except Exception as e:
            error_msg = f"Content validation failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    @abstractmethod
    async def publish_content(self, content: PlatformContent, **kwargs) -> PublishResult:
        """
        Publish content to the platform.
        
        Args:
            content: Content to publish
            **kwargs: Platform-specific parameters
            
        Returns:
            Publishing result
        """
        try:
            logger.info(f"Publishing content to {self.platform.value}")
            
            # Check authentication
            if not self._authenticated:
                return PublishResult(
                    status=PublishStatus.AUTH_FAILED,
                    platform=self.platform,
                    error_message="Not authenticated with platform"
                )
            
            # Wait for rate limiting
            await self.rate_limiter.wait_if_needed()
            
            # Validate content before publishing
            is_valid, validation_error = await self.validate_content(content)
            if not is_valid:
                return PublishResult(
                    status=PublishStatus.CONTENT_REJECTED,
                    platform=self.platform,
                    error_message=validation_error
                )
            
            # Prepare content for platform
            prepared_content = await self._prepare_content_for_platform(content, **kwargs)
            
            # Publish to platform API
            publish_response = await self._publish_to_platform_api(prepared_content, **kwargs)
            
            if publish_response.get('success'):
                result = PublishResult(
                    status=PublishStatus.SUCCESS,
                    platform=self.platform,
                    post_id=publish_response.get('post_id'),
                    post_url=publish_response.get('post_url'),
                    published_at=datetime.utcnow(),
                    metrics=publish_response.get('initial_metrics', {})
                )
                
                self._log_activity("publish", {
                    "post_id": result.post_id,
                    "content_type": content.content_format.value if hasattr(content, 'content_format') else 'unknown'
                })
                
                logger.info(f"Content published successfully to {self.platform.value}: {result.post_id}")
                return result
            else:
                error_message = publish_response.get('error', 'Unknown publishing error')
                logger.error(f"Publishing failed: {error_message}")
                
                return PublishResult(
                    status=PublishStatus.FAILED,
                    platform=self.platform,
                    error_message=error_message,
                    error_code=publish_response.get('error_code')
                )
                
        except Exception as e:
            error_msg = f"Publishing error: {str(e)}"
            logger.error(error_msg)
            self._log_activity("publish", {"error": error_msg}, success=False)
            
            return PublishResult(
                status=PublishStatus.FAILED,
                platform=self.platform,
                error_message=error_msg
            )
    
    @abstractmethod
    async def schedule_content(
        self, 
        content: PlatformContent, 
        publish_time: datetime,
        **kwargs
    ) -> ScheduleResult:
        """
        Schedule content for future publishing.
        
        Args:
            content: Content to schedule
            publish_time: When to publish
            **kwargs: Platform-specific parameters
            
        Returns:
            Scheduling result
        """
        try:
            logger.info(f"Scheduling content for {self.platform.value} at {publish_time}")
            
            # Check authentication
            if not self._authenticated:
                return ScheduleResult(
                    status=PublishStatus.AUTH_FAILED,
                    platform=self.platform,
                    error_message="Not authenticated with platform"
                )
            
            # Validate content before scheduling
            is_valid, validation_error = await self.validate_content(content)
            if not is_valid:
                return ScheduleResult(
                    status=PublishStatus.CONTENT_REJECTED,
                    platform=self.platform,
                    error_message=validation_error
                )
            
            # Validate publish time
            if publish_time <= datetime.utcnow():
                return ScheduleResult(
                    status=PublishStatus.FAILED,
                    platform=self.platform,
                    error_message="Publish time must be in the future"
                )
            
            # Prepare content for scheduling
            prepared_content = await self._prepare_content_for_platform(content, **kwargs)
            
            # Schedule with platform API
            schedule_response = await self._schedule_with_platform_api(prepared_content, publish_time, **kwargs)
            
            if schedule_response.get('success'):
                result = ScheduleResult(
                    status=PublishStatus.SCHEDULED,
                    platform=self.platform,
                    schedule_id=schedule_response.get('schedule_id'),
                    scheduled_time=publish_time,
                    confirmation_url=schedule_response.get('confirmation_url'),
                    can_edit=schedule_response.get('can_edit', True),
                    can_cancel=schedule_response.get('can_cancel', True)
                )
                
                self._log_activity("schedule", {
                    "schedule_id": result.schedule_id,
                    "scheduled_time": publish_time.isoformat(),
                    "content_type": content.content_format.value if hasattr(content, 'content_format') else 'unknown'
                })
                
                logger.info(f"Content scheduled successfully for {self.platform.value}: {result.schedule_id}")
                return result
            else:
                error_message = schedule_response.get('error', 'Unknown scheduling error')
                logger.error(f"Scheduling failed: {error_message}")
                
                return ScheduleResult(
                    status=PublishStatus.FAILED,
                    platform=self.platform,
                    error_message=error_message
                )
                
        except Exception as e:
            error_msg = f"Scheduling error: {str(e)}"
            logger.error(error_msg)
            self._log_activity("schedule", {"error": error_msg}, success=False)
            
            return ScheduleResult(
                status=PublishStatus.FAILED,
                platform=self.platform,
                error_message=error_msg
            )
    
    @abstractmethod
    async def get_analytics(self, post_id: str, **kwargs) -> Dict[str, Any]:
        """
        Get analytics for a published post.
        
        Args:
            post_id: Platform-specific post ID
            **kwargs: Additional parameters
            
        Returns:
            Analytics data
        """
        try:
            logger.info(f"Getting analytics for post {post_id} on {self.platform.value}")
            
            # Check authentication
            if not self._authenticated:
                return {"error": "Not authenticated with platform"}
            
            # Get analytics from platform API
            analytics_data = await self._get_analytics_from_platform_api(post_id, **kwargs)
            
            if analytics_data.get('success'):
                # Standardize analytics data format
                standardized_analytics = await self._standardize_analytics_data(analytics_data['data'])
                
                self._log_activity("get_analytics", {
                    "post_id": post_id,
                    "metrics_count": len(standardized_analytics)
                })
                
                logger.info(f"Analytics retrieved successfully for post {post_id}")
                return standardized_analytics
            else:
                error_message = analytics_data.get('error', 'Failed to retrieve analytics')
                logger.error(f"Analytics retrieval failed: {error_message}")
                return {"error": error_message}
                
        except Exception as e:
            error_msg = f"Analytics retrieval error: {str(e)}"
            logger.error(error_msg)
            self._log_activity("get_analytics", {"error": error_msg, "post_id": post_id}, success=False)
            return {"error": error_msg}
    
    @abstractmethod
    async def delete_content(self, post_id: str) -> bool:
        """
        Delete published content.
        
        Args:
            post_id: Platform-specific post ID
            
        Returns:
            Success status
        """
        try:
            logger.info(f"Deleting content {post_id} from {self.platform.value}")
            
            # Check authentication
            if not self._authenticated:
                logger.error("Not authenticated with platform")
                return False
            
            # Delete from platform API
            delete_response = await self._delete_from_platform_api(post_id)
            
            if delete_response.get('success'):
                self._log_activity("delete", {
                    "post_id": post_id,
                    "status": "success"
                })
                
                logger.info(f"Content deleted successfully: {post_id}")
                return True
            else:
                error_message = delete_response.get('error', 'Failed to delete content')
                logger.error(f"Content deletion failed: {error_message}")
                self._log_activity("delete", {
                    "post_id": post_id,
                    "error": error_message
                }, success=False)
                return False
                
        except Exception as e:
            error_msg = f"Content deletion error: {str(e)}"
            logger.error(error_msg)
            self._log_activity("delete", {"error": error_msg, "post_id": post_id}, success=False)
            return False
    
    async def check_content_policy(self, content: PlatformContent) -> Tuple[bool, Optional[str]]:
        """
        Check content against platform policies.
        
        Args:
            content: Content to check
            
        Returns:
            Tuple of (is_compliant, violation_reason)
        """
        # Default implementation - override for specific platforms
        prohibited_terms = []  # Platform-specific prohibited terms
        
        content_text = content.content_text.lower()
        for term in prohibited_terms:
            if term in content_text:
                return False, f"Content contains prohibited term: {term}"
        
        return True, None
    
    async def optimize_posting_time(
        self, 
        business_niche: BusinessNicheType,
        audience_timezone: str = "UTC"
    ) -> datetime:
        """
        Calculate optimal posting time based on business niche and audience.
        
        Args:
            business_niche: Type of business
            audience_timezone: Primary audience timezone
            
        Returns:
            Optimal posting time
        """
        # Default implementation - override with platform-specific logic
        # This is a simplified version - real implementation would use ML
        optimal_hours = {
            BusinessNicheType.FITNESS_WELLNESS: [6, 7, 17, 18],
            BusinessNicheType.BUSINESS_CONSULTING: [8, 12, 16],
            BusinessNicheType.CREATIVE: [11, 15, 19, 20],
            BusinessNicheType.EDUCATION: [9, 14, 18],
            BusinessNicheType.ECOMMERCE: [10, 13, 19, 20],
            BusinessNicheType.LOCAL_SERVICE: [8, 12, 17],
            BusinessNicheType.TECHNOLOGY: [9, 13, 16],
            BusinessNicheType.NON_PROFIT: [11, 14, 18],
        }
        
        hours = optimal_hours.get(business_niche, [9, 14, 18])
        # For now, return next available slot
        now = datetime.utcnow()
        for hour in sorted(hours):
            potential_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if potential_time > now:
                return potential_time
        
        # If no time today, use first slot tomorrow
        tomorrow = now + timedelta(days=1)
        return tomorrow.replace(hour=hours[0], minute=0, second=0, microsecond=0)
    
    def _log_activity(
        self, 
        action: str, 
        details: Dict[str, Any],
        success: bool = True
    ) -> None:
        """Log platform activity for monitoring and debugging"""
        log_data = {
            "platform": self.platform.value,
            "business_id": self.business_id,
            "action": action,
            "success": success,
            "timestamp": datetime.utcnow().isoformat(),
            **details
        }
        
        if success:
            logger.info(f"Platform activity: {log_data}")
        else:
            logger.error(f"Platform activity failed: {log_data}")
    
    @abstractmethod
    async def optimize_for_algorithm(self, content: PlatformContent, business_niche: str) -> OptimizedContent:
        """
        Optimize content for platform algorithm.
        
        Args:
            content: Content to optimize
            business_niche: Business niche for optimization
            
        Returns:
            Optimized content with algorithm recommendations
        """
        try:
            logger.info(f"Optimizing content for {self.platform.value} algorithm")
            
            # Get platform algorithm requirements
            algorithm_requirements = await self._get_algorithm_requirements(business_niche)
            
            # Optimize text content
            optimized_text = await self._optimize_text_for_algorithm(content.content_text, algorithm_requirements)
            
            # Generate hashtag suggestions
            suggested_hashtags = await self._generate_algorithm_hashtags(content, business_niche, algorithm_requirements)
            
            # Generate mention suggestions
            suggested_mentions = await self._generate_algorithm_mentions(content, business_niche, algorithm_requirements)
            
            # Calculate best posting time
            best_posting_time = await self.optimize_posting_time(
                BusinessNicheType(business_niche) if business_niche in [e.value for e in BusinessNicheType] else BusinessNicheType.GENERAL,
                algorithm_requirements.get('audience_timezone', 'UTC')
            )
            
            # Calculate algorithm score
            algorithm_score = await self._calculate_algorithm_score(content, algorithm_requirements)
            
            # Generate optimization reasons
            optimization_reasons = await self._generate_optimization_reasons(algorithm_requirements)
            
            # Identify trending elements
            trending_elements = await self._identify_trending_elements(business_niche, algorithm_requirements)
            
            # Predict engagement
            engagement_predictions = await self._predict_engagement(content, algorithm_requirements)
            
            optimized_content = OptimizedContent(
                original_content=content,
                optimized_text=optimized_text,
                suggested_hashtags=suggested_hashtags,
                suggested_mentions=suggested_mentions,
                best_posting_time=best_posting_time,
                algorithm_score=algorithm_score,
                optimization_reasons=optimization_reasons,
                trending_elements=trending_elements,
                engagement_predictions=engagement_predictions
            )
            
            logger.info(f"Content optimization completed with score: {algorithm_score}")
            return optimized_content
            
        except Exception as e:
            error_msg = f"Algorithm optimization failed: {str(e)}"
            logger.error(error_msg)
            self._log_activity("optimize_algorithm", {"error": error_msg}, success=False)
            
            # Return basic optimized content on failure
            return OptimizedContent(
                original_content=content,
                optimized_text=content.content_text,
                suggested_hashtags=[],
                suggested_mentions=[],
                best_posting_time=datetime.utcnow() + timedelta(hours=1),
                algorithm_score=0.5,
                optimization_reasons=["Optimization failed, using defaults"],
                trending_elements=[],
                engagement_predictions={}
            )
    
    # Helper methods for authentication and validation
    def _get_required_credential_keys(self) -> List[str]:
        """Get required credential keys for this platform"""
        # Platform-specific implementation should override this
        return ['access_token', 'api_key']
    
    async def _test_platform_connection(self) -> bool:
        """Test connection to platform API"""
        try:
            # This would make a test API call to verify credentials
            # For now, return True as placeholder
            return True
        except Exception as e:
            logger.error(f"Platform connection test failed: {str(e)}")
            return False
    
    def _get_platform_validation_rules(self) -> Dict[str, Any]:
        """Get platform-specific validation rules"""
        # Default rules - platform-specific implementations should override
        return {
            'max_text_length': 2000,
            'max_hashtags': 30,
            'max_media_files': 10,
            'supported_media_types': ['image/jpeg', 'image/png', 'video/mp4'],
            'max_file_size_mb': 100
        }
    
    async def _validate_media_asset(self, media: MediaAsset, validation_rules: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate individual media asset"""
        try:
            # Check media type
            supported_types = validation_rules.get('supported_media_types', [])
            if media.mime_type and media.mime_type not in supported_types:
                return False, f"Unsupported media type: {media.mime_type}"
            
            # Check file size
            max_size_mb = validation_rules.get('max_file_size_mb', 100)
            if media.data and len(media.data) > max_size_mb * 1024 * 1024:
                return False, f"File size exceeds maximum of {max_size_mb}MB"
            
            # Check dimensions for images
            if media.type == MediaType.IMAGE and media.dimensions:
                max_width = validation_rules.get('max_image_width', 4096)
                max_height = validation_rules.get('max_image_height', 4096)
                
                if (media.dimensions.get('width', 0) > max_width or 
                    media.dimensions.get('height', 0) > max_height):
                    return False, f"Image dimensions exceed maximum of {max_width}x{max_height}"
            
            # Check video duration
            if media.type == MediaType.VIDEO and media.duration:
                max_duration = validation_rules.get('max_video_duration_seconds', 600)
                if media.duration > max_duration:
                    return False, f"Video duration exceeds maximum of {max_duration} seconds"
            
            return True, None
            
        except Exception as e:
            return False, f"Media validation error: {str(e)}"
    
    async def _validate_platform_specific(self, content: PlatformContent, validation_rules: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Platform-specific content validation"""
        # Default implementation - platform-specific classes should override
        return True, None
    
    async def _prepare_content_for_platform(self, content: PlatformContent, **kwargs) -> Dict[str, Any]:
        """Prepare content for platform API"""
        prepared_content = {
            'text': content.content_text,
            'media': [],
            'hashtags': getattr(content, 'hashtags', []),
            'mentions': getattr(content, 'mentions', []),
            'location': getattr(content, 'location', None),
            'platform_specific': kwargs
        }
        
        # Prepare media assets
        if hasattr(content, 'media_assets') and content.media_assets:
            for media in content.media_assets:
                prepared_media = {
                    'type': media.type.value,
                    'data': media.data,
                    'filename': media.filename,
                    'mime_type': media.mime_type,
                    'alt_text': media.alt_text
                }
                prepared_content['media'].append(prepared_media)
        
        return prepared_content
    
    async def _publish_to_platform_api(self, prepared_content: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Publish content to platform API"""
        # Platform-specific implementation should override this
        # For now, return mock success response
        import uuid
        return {
            'success': True,
            'post_id': f"post_{uuid.uuid4().hex[:12]}",
            'post_url': f"https://{self.platform.value}.com/post/{uuid.uuid4().hex[:12]}",
            'initial_metrics': {
                'impressions': 0,
                'engagement': 0,
                'reach': 0
            }
        }
    
    async def _schedule_with_platform_api(self, prepared_content: Dict[str, Any], publish_time: datetime, **kwargs) -> Dict[str, Any]:
        """Schedule content with platform API"""
        # Platform-specific implementation should override this
        # For now, return mock success response
        import uuid
        return {
            'success': True,
            'schedule_id': f"schedule_{uuid.uuid4().hex[:12]}",
            'confirmation_url': f"https://{self.platform.value}.com/schedule/{uuid.uuid4().hex[:12]}",
            'can_edit': True,
            'can_cancel': True
        }
    
    async def _get_analytics_from_platform_api(self, post_id: str, **kwargs) -> Dict[str, Any]:
        """Get analytics from platform API"""
        # Platform-specific implementation should override this
        # For now, return mock analytics data
        return {
            'success': True,
            'data': {
                'post_id': post_id,
                'impressions': 1500,
                'reach': 1200,
                'engagement': 180,
                'likes': 120,
                'comments': 35,
                'shares': 25,
                'saves': 40,
                'clicks': 60,
                'date_range': {
                    'start': (datetime.utcnow() - timedelta(days=7)).isoformat(),
                    'end': datetime.utcnow().isoformat()
                }
            }
        }
    
    async def _standardize_analytics_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize analytics data format"""
        return {
            'post_id': raw_data.get('post_id'),
            'metrics': {
                'impressions': raw_data.get('impressions', 0),
                'reach': raw_data.get('reach', 0),
                'engagement': raw_data.get('engagement', 0),
                'engagement_rate': (raw_data.get('engagement', 0) / max(raw_data.get('impressions', 1), 1)) * 100,
                'likes': raw_data.get('likes', 0),
                'comments': raw_data.get('comments', 0),
                'shares': raw_data.get('shares', 0),
                'saves': raw_data.get('saves', 0),
                'clicks': raw_data.get('clicks', 0)
            },
            'date_range': raw_data.get('date_range', {}),
            'platform': self.platform.value,
            'retrieved_at': datetime.utcnow().isoformat()
        }
    
    async def _delete_from_platform_api(self, post_id: str) -> Dict[str, Any]:
        """Delete content from platform API"""
        # Platform-specific implementation should override this
        # For now, return mock success response
        return {
            'success': True,
            'message': f'Post {post_id} deleted successfully'
        }
    
    # Helper methods for algorithm optimization
    async def _get_algorithm_requirements(self, business_niche: str) -> Dict[str, Any]:
        """Get platform algorithm requirements for business niche"""
        # This would typically fetch from algorithm intelligence service
        return {
            'optimal_text_length': 150,
            'hashtag_count_range': (5, 15),
            'mention_count_range': (0, 3),
            'engagement_keywords': ['tips', 'how-to', 'guide', 'advice'],
            'trending_topics': ['productivity', 'wellness', 'growth'],
            'audience_timezone': 'UTC',
            'peak_activity_hours': [9, 12, 17, 20]
        }
    
    async def _optimize_text_for_algorithm(self, text: str, requirements: Dict[str, Any]) -> str:
        """Optimize text content for algorithm"""
        if not text:
            return text
        
        optimized_text = text
        
        # Add engagement keywords if missing
        engagement_keywords = requirements.get('engagement_keywords', [])
        for keyword in engagement_keywords[:1]:  # Add one keyword if none present
            if keyword.lower() not in text.lower():
                optimized_text = f"{keyword.title()}: {optimized_text}"
                break
        
        # Ensure optimal length
        optimal_length = requirements.get('optimal_text_length', 150)
        if len(optimized_text) > optimal_length * 2:
            # Truncate if too long
            optimized_text = optimized_text[:optimal_length] + "..."
        
        return optimized_text
    
    async def _generate_algorithm_hashtags(self, content: PlatformContent, business_niche: str, requirements: Dict[str, Any]) -> List[str]:
        """Generate hashtags optimized for algorithm"""
        hashtag_range = requirements.get('hashtag_count_range', (5, 15))
        trending_topics = requirements.get('trending_topics', [])
        
        # Start with existing hashtags
        existing_hashtags = getattr(content, 'hashtags', [])
        suggested_hashtags = existing_hashtags.copy()
        
        # Add niche-specific hashtags
        niche_hashtags = {
            'fitness': ['#fitness', '#health', '#wellness', '#workout', '#motivation'],
            'business': ['#business', '#entrepreneur', '#success', '#leadership', '#growth'],
            'education': ['#education', '#learning', '#knowledge', '#study', '#skills'],
            'creative': ['#creative', '#art', '#design', '#inspiration', '#artistic']
        }
        
        niche_tags = niche_hashtags.get(business_niche, ['#content', '#social', '#engagement'])
        
        # Add trending topic hashtags
        for topic in trending_topics:
            suggested_hashtags.append(f"#{topic}")
        
        # Add niche hashtags
        for tag in niche_tags:
            if tag not in suggested_hashtags:
                suggested_hashtags.append(tag)
        
        # Ensure within optimal range
        target_count = min(hashtag_range[1], max(hashtag_range[0], len(suggested_hashtags)))
        return suggested_hashtags[:target_count]
    
    async def _generate_algorithm_mentions(self, content: PlatformContent, business_niche: str, requirements: Dict[str, Any]) -> List[str]:
        """Generate mentions optimized for algorithm"""
        mention_range = requirements.get('mention_count_range', (0, 3))
        
        # Start with existing mentions
        existing_mentions = getattr(content, 'mentions', [])
        suggested_mentions = existing_mentions.copy()
        
        # Add strategic mentions based on business niche
        strategic_mentions = {
            'fitness': ['@fitnessmotivation', '@healthylifestyle'],
            'business': ['@entrepreneur', '@businesstips'],
            'education': ['@education', '@learning'],
            'creative': ['@creativity', '@design']
        }
        
        niche_mentions = strategic_mentions.get(business_niche, [])
        
        for mention in niche_mentions:
            if mention not in suggested_mentions and len(suggested_mentions) < mention_range[1]:
                suggested_mentions.append(mention)
        
        return suggested_mentions[:mention_range[1]]
    
    async def _calculate_algorithm_score(self, content: PlatformContent, requirements: Dict[str, Any]) -> float:
        """Calculate algorithm optimization score"""
        score = 0.0
        max_score = 10.0
        
        # Text length score
        text_length = len(content.content_text) if content.content_text else 0
        optimal_length = requirements.get('optimal_text_length', 150)
        if optimal_length * 0.8 <= text_length <= optimal_length * 1.5:
            score += 2.0
        elif text_length > 0:
            score += 1.0
        
        # Hashtag score
        hashtag_count = len(getattr(content, 'hashtags', []))
        hashtag_range = requirements.get('hashtag_count_range', (5, 15))
        if hashtag_range[0] <= hashtag_count <= hashtag_range[1]:
            score += 2.0
        elif hashtag_count > 0:
            score += 1.0
        
        # Engagement keywords score
        engagement_keywords = requirements.get('engagement_keywords', [])
        text_lower = content.content_text.lower() if content.content_text else ""
        keyword_count = sum(1 for keyword in engagement_keywords if keyword in text_lower)
        score += min(2.0, keyword_count * 0.5)
        
        # Media presence score
        if hasattr(content, 'media_assets') and content.media_assets:
            score += 2.0
        
        # Trending topics score
        trending_topics = requirements.get('trending_topics', [])
        trending_count = sum(1 for topic in trending_topics if topic in text_lower)
        score += min(2.0, trending_count * 0.5)
        
        return min(1.0, score / max_score)
    
    async def _generate_optimization_reasons(self, requirements: Dict[str, Any]) -> List[str]:
        """Generate optimization reason explanations"""
        return [
            f"Optimized text length for {requirements.get('optimal_text_length', 150)} character target",
            f"Added {len(requirements.get('engagement_keywords', []))} engagement keywords",
            f"Included trending topics: {', '.join(requirements.get('trending_topics', [])[:3])}",
            "Hashtag count optimized for maximum reach",
            "Posting time aligned with audience peak activity"
        ]
    
    async def _identify_trending_elements(self, business_niche: str, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify trending elements for content"""
        trending_elements = []
        
        # Trending topics
        for topic in requirements.get('trending_topics', []):
            trending_elements.append({
                'type': 'topic',
                'value': topic,
                'trend_score': 0.8,
                'description': f"Trending topic in {business_niche}"
            })
        
        # Trending hashtags
        trending_hashtags = ['#trending', '#viral', '#popular']
        for hashtag in trending_hashtags:
            trending_elements.append({
                'type': 'hashtag',
                'value': hashtag,
                'trend_score': 0.7,
                'description': f"High-performing hashtag"
            })
        
        return trending_elements[:5]  # Return top 5
    
    async def _predict_engagement(self, content: PlatformContent, requirements: Dict[str, Any]) -> Dict[str, float]:
        """Predict engagement metrics"""
        # Simple prediction based on content analysis
        text_length = len(content.content_text) if content.content_text else 0
        hashtag_count = len(getattr(content, 'hashtags', []))
        has_media = hasattr(content, 'media_assets') and bool(content.media_assets)
        
        # Base predictions
        base_impressions = 1000
        engagement_rate = 0.03  # 3% base rate
        
        # Adjust based on content factors
        if text_length > 100:
            engagement_rate += 0.005
        if hashtag_count > 5:
            engagement_rate += 0.01
        if has_media:
            engagement_rate += 0.015
        
        predicted_impressions = base_impressions
        predicted_engagement = predicted_impressions * engagement_rate
        
        return {
            'impressions': predicted_impressions,
            'engagement': predicted_engagement,
            'engagement_rate': engagement_rate * 100,
            'likes': predicted_engagement * 0.7,
            'comments': predicted_engagement * 0.15,
            'shares': predicted_engagement * 0.15
        }