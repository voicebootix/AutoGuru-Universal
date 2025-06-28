"""
Base Platform Publisher for AutoGuru Universal

This module provides the abstract base class for all social media platform publishers.
It implements universal interfaces and shared functionality that work across any
business niche without hardcoded logic. All platform-specific publishers inherit
from this base class.

Features:
- Standardized authentication, publishing, and analytics interfaces
- OAuth token management with automatic refresh
- Rate limiting with exponential backoff
- Content validation and adaptation
- Universal business type support
- Comprehensive error handling and retry logic
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
import hashlib
import json
from dataclasses import dataclass, field
import time

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from backend.utils.encryption import EncryptionManager
from backend.database.connection import PostgreSQLConnectionManager
from backend.models.content_models import (
    Platform,
    PlatformContent,
    BusinessNicheType,
    ContentFormat,
    ContentAnalysis
)
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class PublishStatus(str, Enum):
    """Status of content publishing"""
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RATE_LIMITED = "rate_limited"


class AuthStatus(str, Enum):
    """Authentication status"""
    AUTHENTICATED = "authenticated"
    EXPIRED = "expired"
    INVALID = "invalid"
    REFRESH_REQUIRED = "refresh_required"


class ValidationStatus(str, Enum):
    """Content validation status"""
    VALID = "valid"
    INVALID = "invalid"
    NEEDS_MODIFICATION = "needs_modification"
    WARNING = "warning"


@dataclass
class AuthResult:
    """Authentication result with token information"""
    status: AuthStatus
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: Optional[datetime] = None
    scope: List[str] = field(default_factory=list)
    user_info: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class PublishResult:
    """Result of content publishing"""
    status: PublishStatus
    content_id: Optional[str] = None
    platform_post_id: Optional[str] = None
    published_at: Optional[datetime] = None
    url: Optional[str] = None
    error_message: Optional[str] = None
    retry_after: Optional[datetime] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduleResult:
    """Result of content scheduling"""
    status: PublishStatus
    schedule_id: Optional[str] = None
    scheduled_for: Optional[datetime] = None
    platform_schedule_id: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class PlatformAnalytics:
    """Platform-specific analytics data"""
    content_id: str
    platform: Platform
    impressions: int = 0
    reach: int = 0
    engagement: int = 0
    clicks: int = 0
    shares: int = 0
    saves: int = 0
    comments: int = 0
    likes: int = 0
    views: int = 0
    watch_time_seconds: Optional[int] = None
    demographic_data: Dict[str, Any] = field(default_factory=dict)
    performance_score: float = 0.0
    retrieved_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ValidationResult:
    """Content validation result"""
    status: ValidationStatus
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    modified_content: Optional[PlatformContent] = None


@dataclass
class RateLimitInfo:
    """Rate limit information for a platform"""
    platform: Platform
    requests_remaining: int
    requests_limit: int
    reset_time: datetime
    current_usage_percent: float
    recommended_delay_seconds: int = 0


@dataclass
class OptimizedMedia:
    """Optimized media content for platform"""
    media_type: str
    file_path: str
    file_size_bytes: int
    dimensions: Optional[Tuple[int, int]] = None
    duration_seconds: Optional[int] = None
    format: str = ""
    optimizations_applied: List[str] = field(default_factory=list)


@dataclass
class MediaContent:
    """Input media content for optimization"""
    file_path: str
    media_type: str
    original_size_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class PlatformPublisherError(Exception):
    """Base exception for platform publisher errors"""
    pass


class AuthenticationError(PlatformPublisherError):
    """Authentication-related errors"""
    pass


class RateLimitError(PlatformPublisherError):
    """Rate limit exceeded error"""
    def __init__(self, message: str, retry_after: Optional[datetime] = None):
        super().__init__(message)
        self.retry_after = retry_after


class ContentValidationError(PlatformPublisherError):
    """Content validation error"""
    pass


class PublishingError(PlatformPublisherError):
    """Content publishing error"""
    pass


class BasePlatformPublisher(ABC):
    """
    Abstract base class for all social media platform publishers.
    
    This class provides universal interfaces and shared functionality
    that work across any business niche. All platform-specific
    implementations must inherit from this class.
    """
    
    def __init__(
        self,
        platform: Platform,
        encryption_manager: Optional[EncryptionManager] = None,
        db_manager: Optional[PostgreSQLConnectionManager] = None
    ):
        """
        Initialize the base platform publisher.
        
        Args:
            platform: The social media platform
            encryption_manager: Encryption manager for secure credential storage
            db_manager: Database connection manager for analytics and scheduling
        """
        self.platform = platform
        self.encryption_manager = encryption_manager or EncryptionManager()
        self.db_manager = db_manager
        self._rate_limiter = RateLimiter(platform)
        self._auth_cache: Dict[str, AuthResult] = {}
        self._http_client = httpx.AsyncClient(timeout=30.0)
        
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup"""
        await self._http_client.aclose()
    
    # Abstract methods that must be implemented by platform-specific classes
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> AuthResult:
        """
        Authenticate with the platform using provided credentials.
        
        Args:
            credentials: Platform-specific credentials
            
        Returns:
            AuthResult with authentication status and tokens
        """
        pass
    
    @abstractmethod
    async def publish_content(self, content: PlatformContent) -> PublishResult:
        """
        Publish content to the platform.
        
        Args:
            content: Platform-optimized content to publish
            
        Returns:
            PublishResult with publishing status and details
        """
        pass
    
    @abstractmethod
    async def schedule_content(
        self, 
        content: PlatformContent, 
        publish_time: datetime
    ) -> ScheduleResult:
        """
        Schedule content for future publishing.
        
        Args:
            content: Platform-optimized content to schedule
            publish_time: When to publish the content
            
        Returns:
            ScheduleResult with scheduling status
        """
        pass
    
    @abstractmethod
    async def get_analytics(self, content_id: str) -> PlatformAnalytics:
        """
        Retrieve analytics for published content.
        
        Args:
            content_id: Platform-specific content identifier
            
        Returns:
            PlatformAnalytics with performance metrics
        """
        pass
    
    @abstractmethod
    async def validate_content(self, content: PlatformContent) -> ValidationResult:
        """
        Validate content against platform requirements.
        
        Args:
            content: Content to validate
            
        Returns:
            ValidationResult with validation status and suggestions
        """
        pass
    
    @abstractmethod
    async def get_rate_limits(self) -> RateLimitInfo:
        """
        Get current rate limit information for the platform.
        
        Returns:
            RateLimitInfo with current usage and limits
        """
        pass
    
    # Shared functionality implemented in base class
    
    async def manage_oauth_tokens(
        self, 
        business_id: str,
        credentials: Dict[str, Any]
    ) -> AuthResult:
        """
        Manage OAuth tokens with automatic refresh and secure storage.
        
        Args:
            business_id: Unique business identifier
            credentials: Initial OAuth credentials
            
        Returns:
            AuthResult with current valid tokens
        """
        try:
            # Check cache first
            cache_key = f"{self.platform}_{business_id}"
            if cache_key in self._auth_cache:
                cached_auth = self._auth_cache[cache_key]
                if cached_auth.expires_at and cached_auth.expires_at > datetime.utcnow():
                    return cached_auth
            
            # Try to retrieve stored tokens
            stored_token_key = f"oauth_token_{self.platform}_{business_id}"
            try:
                encrypted_token = await self._retrieve_encrypted_token(stored_token_key)
                if encrypted_token:
                    token_data = self.encryption_manager.retrieve_oauth_token(encrypted_token)
                    
                    # Check if token needs refresh
                    expires_at = datetime.fromisoformat(token_data.get('expires_at', ''))
                    if expires_at > datetime.utcnow() + timedelta(minutes=5):
                        # Token still valid
                        auth_result = AuthResult(
                            status=AuthStatus.AUTHENTICATED,
                            access_token=token_data['access_token'],
                            refresh_token=token_data.get('refresh_token'),
                            expires_at=expires_at,
                            scope=token_data.get('scope', [])
                        )
                        self._auth_cache[cache_key] = auth_result
                        return auth_result
                    
                    # Token needs refresh
                    if 'refresh_token' in token_data:
                        refreshed_auth = await self._refresh_oauth_token(
                            token_data['refresh_token'],
                            business_id
                        )
                        if refreshed_auth.status == AuthStatus.AUTHENTICATED:
                            self._auth_cache[cache_key] = refreshed_auth
                            return refreshed_auth
            
            except Exception as e:
                logger.warning(f"Failed to retrieve stored token: {str(e)}")
            
            # Perform new authentication
            auth_result = await self.authenticate(credentials)
            
            if auth_result.status == AuthStatus.AUTHENTICATED:
                # Store tokens securely
                await self._store_oauth_token(auth_result, business_id)
                self._auth_cache[cache_key] = auth_result
            
            return auth_result
            
        except Exception as e:
            logger.error(f"OAuth token management failed: {str(e)}")
            return AuthResult(
                status=AuthStatus.INVALID,
                error_message=str(e)
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type(RateLimitError),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def publish_with_retry(
        self,
        content: PlatformContent,
        business_id: str
    ) -> PublishResult:
        """
        Publish content with automatic retry and rate limit handling.
        
        Args:
            content: Content to publish
            business_id: Business identifier for rate limiting
            
        Returns:
            PublishResult with status
        """
        # Check rate limits
        if not await self._rate_limiter.check_limit(business_id):
            rate_info = await self.get_rate_limits()
            raise RateLimitError(
                f"Rate limit exceeded for {self.platform}",
                retry_after=rate_info.reset_time
            )
        
        # Validate content
        validation_result = await self.validate_content(content)
        if validation_result.status == ValidationStatus.INVALID:
            raise ContentValidationError(
                f"Content validation failed: {', '.join(validation_result.errors)}"
            )
        
        # Use modified content if suggested
        if validation_result.modified_content:
            content = validation_result.modified_content
        
        # Publish content
        try:
            result = await self.publish_content(content)
            
            # Record usage for rate limiting
            await self._rate_limiter.record_usage(business_id)
            
            # Store analytics baseline if successful
            if result.status == PublishStatus.SUCCESS and self.db_manager:
                await self._store_publish_record(result, content, business_id)
            
            return result
            
        except Exception as e:
            logger.error(f"Publishing failed: {str(e)}")
            return PublishResult(
                status=PublishStatus.FAILED,
                error_message=str(e)
            )
    
    async def adapt_content_for_platform(
        self,
        content: str,
        platform: Platform,
        business_niche: BusinessNicheType
    ) -> str:
        """
        Adapt content for specific platform and business niche using AI.
        
        Args:
            content: Original content text
            platform: Target platform
            business_niche: Business niche type
            
        Returns:
            Platform-adapted content
        """
        # Platform-specific adaptations
        platform_limits = self._get_platform_limits()
        char_limit = platform_limits[platform]['text_limit']
        
        # Truncate if needed
        if len(content) > char_limit:
            # Smart truncation - try to end at sentence
            truncated = content[:char_limit-3]
            last_period = truncated.rfind('.')
            last_space = truncated.rfind(' ')
            
            if last_period > char_limit * 0.8:
                content = truncated[:last_period+1]
            elif last_space > char_limit * 0.8:
                content = truncated[:last_space] + '...'
            else:
                content = truncated + '...'
        
        # Platform-specific formatting
        if platform == Platform.TWITTER:
            # Twitter prefers concise, punchy content
            content = self._format_for_twitter(content)
        elif platform == Platform.LINKEDIN:
            # LinkedIn prefers professional, detailed content
            content = self._format_for_linkedin(content, business_niche)
        elif platform == Platform.INSTAGRAM:
            # Instagram focuses on visual storytelling
            content = self._format_for_instagram(content)
        elif platform == Platform.TIKTOK:
            # TikTok needs trendy, engaging hooks
            content = self._format_for_tiktok(content)
        
        return content
    
    async def optimize_media_for_platform(
        self,
        media: MediaContent,
        platform: Platform
    ) -> OptimizedMedia:
        """
        Optimize media files for platform requirements.
        
        Args:
            media: Original media content
            platform: Target platform
            
        Returns:
            Optimized media with platform specifications
        """
        platform_specs = self._get_platform_media_specs()
        specs = platform_specs.get(platform, {})
        
        optimizations = []
        
        # This is a placeholder for actual media optimization
        # In production, this would use image/video processing libraries
        optimized = OptimizedMedia(
            media_type=media.media_type,
            file_path=media.file_path,
            file_size_bytes=media.original_size_bytes,
            format=specs.get('format', 'jpg'),
            optimizations_applied=optimizations
        )
        
        # Apply platform-specific optimizations
        if media.media_type == 'image':
            optimized.dimensions = specs.get('image_dimensions', (1080, 1080))
            optimizations.append(f"Resized to {optimized.dimensions}")
        elif media.media_type == 'video':
            optimized.duration_seconds = min(
                media.metadata.get('duration', 60),
                specs.get('max_video_duration', 60)
            )
            optimizations.append(f"Duration limited to {optimized.duration_seconds}s")
        
        return optimized
    
    async def generate_platform_hashtags(
        self,
        base_hashtags: List[str],
        platform: Platform
    ) -> List[str]:
        """
        Generate platform-optimized hashtags.
        
        Args:
            base_hashtags: Original hashtags
            platform: Target platform
            
        Returns:
            Platform-optimized hashtag list
        """
        platform_limits = self._get_platform_limits()
        max_hashtags = platform_limits[platform]['hashtag_limit']
        
        # Filter and optimize hashtags
        optimized_hashtags = []
        
        for tag in base_hashtags[:max_hashtags]:
            # Remove spaces and special characters
            clean_tag = ''.join(c for c in tag if c.isalnum() or c == '_')
            
            # Platform-specific formatting
            if platform == Platform.INSTAGRAM:
                # Instagram allows longer hashtags
                if len(clean_tag) <= 100:
                    optimized_hashtags.append(f"#{clean_tag}")
            elif platform == Platform.TWITTER:
                # Twitter prefers shorter hashtags
                if len(clean_tag) <= 50:
                    optimized_hashtags.append(f"#{clean_tag}")
            elif platform == Platform.LINKEDIN:
                # LinkedIn uses fewer, more professional hashtags
                if len(clean_tag) <= 50 and len(optimized_hashtags) < 5:
                    optimized_hashtags.append(f"#{clean_tag}")
            else:
                optimized_hashtags.append(f"#{clean_tag}")
        
        return optimized_hashtags
    
    async def calculate_optimal_posting_time(
        self,
        business_niche: BusinessNicheType,
        platform: Platform
    ) -> datetime:
        """
        Calculate optimal posting time using AI and analytics.
        
        Args:
            business_niche: Type of business
            platform: Social media platform
            
        Returns:
            Optimal datetime for posting
        """
        # Default optimal times by platform (can be overridden by analytics)
        default_times = {
            Platform.INSTAGRAM: [9, 12, 17, 19],  # 9am, 12pm, 5pm, 7pm
            Platform.TWITTER: [8, 12, 17, 21],    # 8am, 12pm, 5pm, 9pm
            Platform.LINKEDIN: [7, 10, 12, 17],   # 7am, 10am, 12pm, 5pm
            Platform.FACEBOOK: [9, 13, 15, 19],   # 9am, 1pm, 3pm, 7pm
            Platform.TIKTOK: [6, 10, 15, 19],     # 6am, 10am, 3pm, 7pm
        }
        
        # Get platform-specific optimal hours
        optimal_hours = default_times.get(platform, [12])
        
        # Select next available optimal time
        now = datetime.utcnow()
        current_hour = now.hour
        
        # Find next optimal hour
        next_hour = None
        for hour in optimal_hours:
            if hour > current_hour:
                next_hour = hour
                break
        
        # If no optimal hour today, use first hour tomorrow
        if next_hour is None:
            next_hour = optimal_hours[0]
            target_date = now.date() + timedelta(days=1)
        else:
            target_date = now.date()
        
        # Create optimal posting time
        optimal_time = datetime.combine(
            target_date,
            datetime.min.time().replace(hour=next_hour)
        )
        
        return optimal_time
    
    # Helper methods
    
    def _get_platform_limits(self) -> Dict[Platform, Dict[str, Any]]:
        """Get platform-specific content limits"""
        return {
            Platform.TWITTER: {
                'text_limit': 280,
                'hashtag_limit': 5,
                'media_limit': 4,
                'video_duration': 140
            },
            Platform.INSTAGRAM: {
                'text_limit': 2200,
                'hashtag_limit': 30,
                'media_limit': 10,
                'video_duration': 60
            },
            Platform.LINKEDIN: {
                'text_limit': 3000,
                'hashtag_limit': 5,
                'media_limit': 9,
                'video_duration': 600
            },
            Platform.FACEBOOK: {
                'text_limit': 63206,
                'hashtag_limit': 10,
                'media_limit': 10,
                'video_duration': 240
            },
            Platform.TIKTOK: {
                'text_limit': 2200,
                'hashtag_limit': 10,
                'media_limit': 1,
                'video_duration': 180
            }
        }
    
    def _get_platform_media_specs(self) -> Dict[Platform, Dict[str, Any]]:
        """Get platform-specific media specifications"""
        return {
            Platform.INSTAGRAM: {
                'image_dimensions': (1080, 1080),
                'video_dimensions': (1080, 1920),
                'format': 'jpg',
                'max_file_size': 8388608,  # 8MB
                'max_video_duration': 60
            },
            Platform.TWITTER: {
                'image_dimensions': (1200, 675),
                'video_dimensions': (1280, 720),
                'format': 'jpg',
                'max_file_size': 5242880,  # 5MB
                'max_video_duration': 140
            },
            Platform.LINKEDIN: {
                'image_dimensions': (1200, 627),
                'video_dimensions': (1920, 1080),
                'format': 'jpg',
                'max_file_size': 10485760,  # 10MB
                'max_video_duration': 600
            }
        }
    
    def _format_for_twitter(self, content: str) -> str:
        """Format content specifically for Twitter"""
        # Twitter prefers concise, engaging content with clear CTAs
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            if line.strip():
                # Remove excessive punctuation
                line = line.strip()
                if not line.endswith(('.', '!', '?', ':')):
                    line += '.'
                formatted_lines.append(line)
        
        return '\n\n'.join(formatted_lines[:3])  # Keep it concise
    
    def _format_for_linkedin(self, content: str, business_niche: BusinessNicheType) -> str:
        """Format content specifically for LinkedIn"""
        # LinkedIn prefers professional, value-driven content
        if business_niche in [BusinessNicheType.BUSINESS_CONSULTING, BusinessNicheType.TECHNOLOGY]:
            # Add professional tone markers
            if not content.startswith(('Did you know', 'Here\'s', 'Today')):
                content = f"💡 {content}"
        
        return content
    
    def _format_for_instagram(self, content: str) -> str:
        """Format content specifically for Instagram"""
        # Instagram focuses on visual storytelling with emojis
        # Add line breaks for readability
        sentences = content.split('. ')
        formatted = []
        
        for i, sentence in enumerate(sentences):
            if i > 0 and i % 2 == 0:
                formatted.append('\n')
            formatted.append(sentence)
        
        return '. '.join(formatted)
    
    def _format_for_tiktok(self, content: str) -> str:
        """Format content specifically for TikTok"""
        # TikTok needs attention-grabbing hooks
        if not content.startswith(('POV:', 'Story time:', 'Wait for it')):
            # Add engaging hook
            content = f"Wait for it... {content}"
        
        return content
    
    async def _store_oauth_token(self, auth_result: AuthResult, business_id: str) -> None:
        """Store OAuth token securely"""
        try:
            token_data = {
                'access_token': auth_result.access_token,
                'refresh_token': auth_result.refresh_token,
                'expires_at': auth_result.expires_at.isoformat() if auth_result.expires_at else None,
                'scope': auth_result.scope
            }
            
            encrypted_token = self.encryption_manager.secure_oauth_token(
                token_data,
                self.platform.value,
                business_id
            )
            
            # Store in database if available
            if self.db_manager:
                await self._persist_encrypted_token(encrypted_token, business_id)
                
        except Exception as e:
            logger.error(f"Failed to store OAuth token: {str(e)}")
    
    async def _retrieve_encrypted_token(self, token_key: str) -> Optional[Dict[str, str]]:
        """Retrieve encrypted token from storage"""
        if not self.db_manager:
            return None
            
        try:
            # Placeholder for database retrieval
            # In production, this would query the database
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve token: {str(e)}")
            return None
    
    async def _persist_encrypted_token(
        self,
        encrypted_token: Dict[str, str],
        business_id: str
    ) -> None:
        """Persist encrypted token to database"""
        if not self.db_manager:
            return
            
        try:
            # Placeholder for database persistence
            # In production, this would insert/update in database
            pass
        except Exception as e:
            logger.error(f"Failed to persist token: {str(e)}")
    
    async def _refresh_oauth_token(
        self,
        refresh_token: str,
        business_id: str
    ) -> AuthResult:
        """Refresh OAuth token"""
        # This is platform-specific and should be overridden
        return AuthResult(
            status=AuthStatus.REFRESH_REQUIRED,
            error_message="Token refresh not implemented for this platform"
        )
    
    async def _store_publish_record(
        self,
        result: PublishResult,
        content: PlatformContent,
        business_id: str
    ) -> None:
        """Store publishing record for analytics"""
        if not self.db_manager:
            return
            
        try:
            # Placeholder for storing publish record
            # In production, this would insert into analytics table
            pass
        except Exception as e:
            logger.error(f"Failed to store publish record: {str(e)}")


class RateLimiter:
    """
    Rate limiter for platform API calls.
    
    Implements token bucket algorithm with per-business tracking.
    """
    
    def __init__(self, platform: Platform):
        self.platform = platform
        self._buckets: Dict[str, TokenBucket] = {}
        self._limits = self._get_platform_rate_limits()
    
    async def check_limit(self, business_id: str) -> bool:
        """Check if request is within rate limit"""
        bucket = self._get_bucket(business_id)
        return bucket.can_consume()
    
    async def record_usage(self, business_id: str) -> None:
        """Record API usage"""
        bucket = self._get_bucket(business_id)
        bucket.consume()
    
    def _get_bucket(self, business_id: str) -> 'TokenBucket':
        """Get or create token bucket for business"""
        key = f"{self.platform}_{business_id}"
        if key not in self._buckets:
            limit_config = self._limits.get(self.platform, {})
            self._buckets[key] = TokenBucket(
                capacity=limit_config.get('requests_per_hour', 100),
                refill_rate=limit_config.get('requests_per_hour', 100) / 3600,
                refill_amount=1
            )
        return self._buckets[key]
    
    def _get_platform_rate_limits(self) -> Dict[Platform, Dict[str, int]]:
        """Get platform-specific rate limits from settings"""
        return {
            Platform.TWITTER: {
                'requests_per_hour': settings.rate_limit.twitter_requests_per_15min * 4
            },
            Platform.LINKEDIN: {
                'requests_per_hour': settings.rate_limit.linkedin_requests_per_day // 24
            },
            Platform.INSTAGRAM: {
                'requests_per_hour': settings.rate_limit.instagram_requests_per_hour
            },
            Platform.FACEBOOK: {
                'requests_per_hour': settings.rate_limit.facebook_requests_per_hour
            },
            Platform.TIKTOK: {
                'requests_per_hour': settings.rate_limit.tiktok_requests_per_hour
            }
        }


class TokenBucket:
    """Token bucket implementation for rate limiting"""
    
    def __init__(self, capacity: int, refill_rate: float, refill_amount: int = 1):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.refill_amount = refill_amount
        self.tokens = capacity
        self.last_refill = time.time()
    
    def can_consume(self, tokens: int = 1) -> bool:
        """Check if tokens are available"""
        self._refill()
        return self.tokens >= tokens
    
    def consume(self, tokens: int = 1) -> bool:
        """Consume tokens if available"""
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = int(elapsed * self.refill_rate) * self.refill_amount
        
        if tokens_to_add > 0:
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now