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
        pass
    
    @abstractmethod
    async def validate_content(self, content: PlatformContent) -> Tuple[bool, Optional[str]]:
        """
        Validate content against platform requirements.
        
        Args:
            content: Content to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
    
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
        pass
    
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
        pass
    
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
        pass
    
    @abstractmethod
    async def delete_content(self, post_id: str) -> bool:
        """
        Delete published content.
        
        Args:
            post_id: Platform-specific post ID
            
        Returns:
            Success status
        """
        pass
    
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