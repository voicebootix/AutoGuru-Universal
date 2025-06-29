"""
Platform publishers module for AutoGuru Universal.

This module provides enhanced platform publishers with built-in business intelligence
and revenue optimization for YouTube, LinkedIn, TikTok, Twitter, and Facebook.
"""

from backend.platforms.enhanced_base_publisher import (
    UniversalPlatformPublisher,
    PublishResult,
    PublishStatus,
    RevenueMetrics,
    PerformanceMetrics,
    AudienceInsights,
    RevenueTracker,
    PerformanceMonitor
)

from backend.platforms.youtube_publisher import YouTubeEnhancedPublisher
from backend.platforms.linkedin_publisher import LinkedInEnhancedPublisher
from backend.platforms.tiktok_publisher import TikTokEnhancedPublisher
from backend.platforms.twitter_publisher import TwitterEnhancedPublisher
from backend.platforms.facebook_publisher import FacebookEnhancedPublisher

__all__ = [
    # Base classes and data structures
    'UniversalPlatformPublisher',
    'PublishResult',
    'PublishStatus',
    'RevenueMetrics',
    'PerformanceMetrics',
    'AudienceInsights',
    'RevenueTracker',
    'PerformanceMonitor',
    
    # Platform publishers
    'YouTubeEnhancedPublisher',
    'LinkedInEnhancedPublisher',
    'TikTokEnhancedPublisher',
    'TwitterEnhancedPublisher',
    'FacebookEnhancedPublisher'
]

# Platform registry for easy access
PLATFORM_PUBLISHERS = {
    'youtube': YouTubeEnhancedPublisher,
    'linkedin': LinkedInEnhancedPublisher,
    'tiktok': TikTokEnhancedPublisher,
    'twitter': TwitterEnhancedPublisher,
    'facebook': FacebookEnhancedPublisher
}

def get_publisher(platform: str, client_id: str) -> UniversalPlatformPublisher:
    """
    Factory function to get a platform publisher instance.
    
    Args:
        platform: Platform name (youtube, linkedin, tiktok, twitter, facebook)
        client_id: Client ID for the platform
        
    Returns:
        Platform publisher instance
        
    Raises:
        ValueError: If platform is not supported
    """
    platform_lower = platform.lower()
    if platform_lower not in PLATFORM_PUBLISHERS:
        raise ValueError(f"Unsupported platform: {platform}. Supported platforms: {list(PLATFORM_PUBLISHERS.keys())}")
    
    publisher_class = PLATFORM_PUBLISHERS[platform_lower]
    return publisher_class(client_id)
