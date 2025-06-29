"""
Enhanced Base Platform Publisher for AutoGuru Universal.

This module provides the enhanced abstract base class for all social media platform publishers
with built-in revenue optimization, business intelligence, and performance monitoring capabilities.
It works universally across all business niches without hardcoded business logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
import json

from backend.models.content_models import (
    Platform, 
    PlatformContent, 
    BusinessNicheType,
    ContentFormat
)
from backend.services.analytics_service import UniversalAnalyticsService
from backend.core.content_analyzer import UniversalContentAnalyzer, ContentAnalysisResult
from backend.utils.encryption import EncryptionManager
from backend.database.connection import PostgreSQLConnectionManager

logger = logging.getLogger(__name__)


class PublishStatus(Enum):
    """Enhanced status of a publish operation"""
    PENDING = "pending"
    PUBLISHED = "published"
    FAILED = "failed"
    SCHEDULED = "scheduled"
    ANALYZING = "analyzing"
    OPTIMIZING = "optimizing"
    RATE_LIMITED = "rate_limited"


@dataclass
class RevenueMetrics:
    """Revenue metrics for a published post"""
    estimated_revenue_potential: float
    actual_revenue: float = 0.0
    conversion_rate: float = 0.0
    revenue_per_engagement: float = 0.0
    revenue_per_impression: float = 0.0
    attribution_source: Dict[str, float] = field(default_factory=dict)
    optimization_score: float = 0.0
    revenue_trend: str = "stable"  # increasing, decreasing, stable


@dataclass
class PerformanceMetrics:
    """Performance metrics for content"""
    engagement_rate: float
    reach: int
    impressions: int
    clicks: int
    shares: int
    saves: int
    comments: int
    likes: int
    video_views: Optional[int] = None
    video_retention_rate: Optional[float] = None
    story_completion_rate: Optional[float] = None
    ctr: float = 0.0  # Click-through rate
    viral_coefficient: float = 0.0


@dataclass
class AudienceInsights:
    """Real-time audience insights"""
    demographics: Dict[str, Any]
    psychographics: Dict[str, Any]
    peak_activity_times: List[datetime]
    engagement_patterns: Dict[str, Any]
    content_preferences: List[str]
    purchase_indicators: Dict[str, float]
    sentiment_analysis: Dict[str, float]


@dataclass
class PublishResult:
    """Enhanced result of a publish operation with revenue tracking"""
    platform: str
    status: PublishStatus
    post_id: Optional[str]
    post_url: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    revenue_metrics: Optional[RevenueMetrics] = None
    performance_metrics: Optional[PerformanceMetrics] = None
    optimization_suggestions: List[str] = field(default_factory=list)
    ab_test_variant: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None
    next_optimization_time: Optional[datetime] = None


class RevenueTracker:
    """Track and optimize revenue across platforms"""
    
    def __init__(self, client_id: str, platform_name: str):
        self.client_id = client_id
        self.platform_name = platform_name
        self.analytics_service = UniversalAnalyticsService()
        self.db_manager = PostgreSQLConnectionManager()
        self.content_analyzer = UniversalContentAnalyzer()
        
    async def track_post_revenue(self, post_id: str, content: Dict[str, Any]) -> Dict[str, float]:
        """Track revenue attribution for a specific post"""
        try:
            # Get post performance data
            performance_data = await self._get_post_performance_data(post_id)
            
            # Calculate revenue attribution
            revenue_data = {
                'direct_sales': await self._calculate_direct_sales(post_id, performance_data),
                'lead_value': await self._calculate_lead_value(post_id, performance_data),
                'brand_value': await self._calculate_brand_value(performance_data),
                'affiliate_revenue': await self._calculate_affiliate_revenue(post_id),
                'ad_revenue': await self._calculate_ad_revenue(post_id, performance_data),
                'total_attributed': 0.0
            }
            
            revenue_data['total_attributed'] = sum(
                v for k, v in revenue_data.items() 
                if k != 'total_attributed'
            )
            
            # Store in database
            await self._store_revenue_data(post_id, revenue_data)
            
            return revenue_data
            
        except Exception as e:
            logger.error(f"Failed to track revenue for post {post_id}: {str(e)}")
            return {'total_attributed': 0.0}
    
    async def suggest_optimizations(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest revenue optimizations based on content and historical data"""
        business_niche = await self._detect_business_niche(content.get('text', ''))
        
        # Get historical performance data for this niche
        historical_data = await self._get_niche_performance_data(
            self.client_id,
            business_niche,
            self.platform_name
        )
        
        # Analyze top performing content
        top_performers = await self._analyze_top_performers(historical_data)
        
        optimizations = {
            'content_optimizations': await self._get_content_optimizations(
                content, 
                top_performers,
                business_niche
            ),
            'timing_optimizations': await self._get_timing_optimizations(
                business_niche,
                historical_data
            ),
            'format_optimizations': await self._get_format_optimizations(
                content,
                top_performers
            ),
            'cta_optimizations': await self._get_cta_optimizations(
                business_niche,
                top_performers
            ),
            'estimated_revenue_increase': await self._estimate_revenue_increase(
                content,
                historical_data
            )
        }
        
        return optimizations
    
    async def _get_post_performance_data(self, post_id: str) -> Dict[str, Any]:
        """Get performance data for a post"""
        # This would fetch from the platform APIs or database
        return {
            'impressions': 10000,
            'engagement': 500,
            'clicks': 200,
            'profile_visits': 50,
            'views': 8000
        }
    
    async def _calculate_direct_sales(self, post_id: str, performance_data: Dict) -> float:
        """Calculate direct sales attributed to the post"""
        # Implementation would track UTM parameters, promo codes, etc.
        # This is a simplified calculation
        clicks = performance_data.get('clicks', 0)
        conversion_rate = await self._get_platform_conversion_rate()
        average_order_value = await self._get_average_order_value()
        
        return clicks * conversion_rate * average_order_value
    
    async def _calculate_lead_value(self, post_id: str, performance_data: Dict) -> float:
        """Calculate the value of leads generated"""
        # Track email signups, contact form submissions, etc.
        profile_visits = performance_data.get('profile_visits', 0)
        lead_conversion_rate = 0.05  # 5% of profile visits convert to leads
        lead_value = await self._get_average_lead_value()
        
        return profile_visits * lead_conversion_rate * lead_value
    
    async def _calculate_brand_value(self, performance_data: Dict) -> float:
        """Calculate brand value increase from impressions and engagement"""
        impressions = performance_data.get('impressions', 0)
        engagement = performance_data.get('engagement', 0)
        
        # Simplified brand value calculation
        impression_value = 0.001  # $0.001 per impression
        engagement_value = 0.05   # $0.05 per engagement
        
        return (impressions * impression_value) + (engagement * engagement_value)
    
    async def _calculate_affiliate_revenue(self, post_id: str) -> float:
        """Calculate affiliate revenue if applicable"""
        # Track affiliate link clicks and conversions
        # This would integrate with affiliate platforms
        return 0.0  # Placeholder
    
    async def _calculate_ad_revenue(self, post_id: str, performance_data: Dict) -> float:
        """Calculate ad revenue for monetized content"""
        # Platform-specific ad revenue calculations
        views = performance_data.get('views', 0)
        platform_rpm = await self._get_platform_rpm()  # Revenue per 1000 views
        
        return (views / 1000) * platform_rpm
    
    async def _detect_business_niche(self, text: str) -> str:
        """Detect business niche from content"""
        if not text:
            return BusinessNicheType.OTHER.value
            
        result = await self.content_analyzer.analyze_content(text)
        return result.business_niche.value
    
    async def _get_platform_conversion_rate(self) -> float:
        """Get platform-specific conversion rate"""
        # This would be calculated from historical data
        platform_rates = {
            "youtube": 0.02,
            "instagram": 0.03,
            "linkedin": 0.05,
            "tiktok": 0.025,
            "twitter": 0.015,
            "facebook": 0.025
        }
        return platform_rates.get(self.platform_name.lower(), 0.02)
    
    async def _get_average_order_value(self) -> float:
        """Get average order value for the business"""
        # This would come from the business's e-commerce data
        return 75.0  # Placeholder
    
    async def _get_average_lead_value(self) -> float:
        """Get average value of a lead"""
        # This would be calculated from CRM data
        return 25.0  # Placeholder
    
    async def _get_platform_rpm(self) -> float:
        """Get platform-specific revenue per mille (thousand views)"""
        platform_rpm = {
            "youtube": 2.0,      # $2 per 1000 views
            "instagram": 0.0,    # No direct monetization
            "linkedin": 0.0,     # No direct monetization
            "tiktok": 0.5,       # Creator fund
            "twitter": 0.0,      # No direct monetization (yet)
            "facebook": 1.5      # Facebook video monetization
        }
        return platform_rpm.get(self.platform_name.lower(), 0.0)
    
    async def _store_revenue_data(self, post_id: str, revenue_data: Dict[str, float]):
        """Store revenue data in database"""
        # Implementation would store in database
        logger.info(f"Storing revenue data for post {post_id}: {revenue_data}")
    
    async def _get_niche_performance_data(
        self,
        client_id: str,
        business_niche: str,
        platform: str
    ) -> Dict[str, Any]:
        """Get historical performance data for a niche"""
        # This would query the database for historical data
        return {
            'avg_engagement_rate': 3.5,
            'avg_conversion_rate': 0.02,
            'top_performing_times': ['09:00', '14:00', '19:00'],
            'top_content_types': ['video', 'carousel']
        }
    
    async def _analyze_top_performers(self, historical_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze top performing content from historical data"""
        # This would analyze historical data to find patterns
        return [
            {
                'content_type': 'video',
                'engagement_rate': 5.2,
                'conversion_rate': 0.035,
                'characteristics': ['educational', 'under_60_seconds', 'call_to_action']
            }
        ]
    
    async def _get_content_optimizations(
        self,
        content: Dict[str, Any],
        top_performers: List[Dict[str, Any]],
        business_niche: str
    ) -> List[str]:
        """Get content optimization suggestions"""
        optimizations = []
        
        # Analyze content against top performers
        if top_performers:
            top_type = top_performers[0]['content_type']
            if content.get('type') != top_type:
                optimizations.append(f"Consider using {top_type} format for better performance")
        
        # Add niche-specific optimizations
        if 'call_to_action' not in content.get('text', '').lower():
            optimizations.append("Add a clear call-to-action to improve conversions")
            
        return optimizations
    
    async def _get_timing_optimizations(
        self,
        business_niche: str,
        historical_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get optimal posting time recommendations"""
        return {
            'best_times': historical_data.get('top_performing_times', ['09:00', '14:00', '19:00']),
            'best_days': ['Tuesday', 'Wednesday', 'Thursday'],
            'timezone_considerations': 'Post in your audience\'s primary timezone'
        }
    
    async def _get_format_optimizations(
        self,
        content: Dict[str, Any],
        top_performers: List[Dict[str, Any]]
    ) -> List[str]:
        """Get content format optimization suggestions"""
        suggestions = []
        
        if content.get('type') == 'text' and top_performers:
            if any(p['content_type'] == 'video' for p in top_performers):
                suggestions.append("Video content typically performs 2x better for this niche")
                
        return suggestions
    
    async def _get_cta_optimizations(
        self,
        business_niche: str,
        top_performers: List[Dict[str, Any]]
    ) -> List[str]:
        """Get call-to-action optimization suggestions"""
        cta_suggestions = {
            'education': ['Learn more', 'Start learning', 'Get the course'],
            'business_consulting': ['Book a consultation', 'Get your free strategy', 'Schedule a call'],
            'fitness_wellness': ['Start your journey', 'Join the challenge', 'Get your plan'],
            'creative': ['See the portfolio', 'Book a session', 'Get inspired'],
            'ecommerce': ['Shop now', 'Get 20% off', 'Limited time offer'],
            'local_service': ['Book now', 'Get a quote', 'Call today'],
            'technology': ['Try it free', 'Get started', 'See the demo'],
            'non_profit': ['Donate now', 'Join the cause', 'Make a difference']
        }
        
        return cta_suggestions.get(business_niche, ['Learn more', 'Get started'])
    
    async def _estimate_revenue_increase(
        self,
        content: Dict[str, Any],
        historical_data: Dict[str, Any]
    ) -> float:
        """Estimate potential revenue increase from optimizations"""
        # Simplified calculation based on historical improvements
        base_improvement = 0.15  # 15% base improvement
        
        # Add bonuses for specific optimizations
        if content.get('optimized_time'):
            base_improvement += 0.10
        if content.get('optimized_format'):
            base_improvement += 0.08
        if content.get('optimized_cta'):
            base_improvement += 0.12
            
        return base_improvement

    async def _estimate_performance(
        self, 
        content: Dict[str, Any], 
        optimizations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Estimate content performance based on optimizations"""
        # Use historical data and ML to predict performance
        historical_avg = await self._get_historical_average_performance()
        
        optimization_boost = 1.0
        if optimizations.get('revenue_optimization'):
            optimization_boost *= 1.2
        if optimizations.get('optimal_posting_time'):
            optimization_boost *= 1.15
        if optimizations.get('platform_specific_tweaks'):
            optimization_boost *= 1.1
            
        return {
            'estimated_reach': int(historical_avg['reach'] * optimization_boost),
            'estimated_engagement': int(historical_avg['engagement'] * optimization_boost),
            'estimated_conversions': int(historical_avg['conversions'] * optimization_boost),
            'confidence_level': 0.75  # 75% confidence in estimates
        }


class PerformanceMonitor:
    """Monitor and optimize content performance"""
    
    def __init__(self, client_id: str, platform_name: str):
        self.client_id = client_id
        self.platform_name = platform_name
        self.analytics_service = UniversalAnalyticsService()
        
    async def analyze_performance(self, post_id: str) -> PerformanceMetrics:
        """Analyze post performance metrics"""
        raw_metrics = await self.get_post_metrics(post_id)
        
        # Calculate engagement rate
        engagement = (
            raw_metrics.get('likes', 0) + 
            raw_metrics.get('comments', 0) + 
            raw_metrics.get('shares', 0) + 
            raw_metrics.get('saves', 0)
        )
        
        impressions = raw_metrics.get('impressions', 1)  # Avoid division by zero
        engagement_rate = (engagement / impressions) * 100
        
        # Calculate CTR
        clicks = raw_metrics.get('clicks', 0)
        ctr = (clicks / impressions) * 100 if impressions > 0 else 0
        
        # Calculate viral coefficient
        shares = raw_metrics.get('shares', 0)
        viral_coefficient = shares / (impressions / 1000) if impressions > 0 else 0
        
        return PerformanceMetrics(
            engagement_rate=engagement_rate,
            reach=raw_metrics.get('reach', 0),
            impressions=impressions,
            clicks=clicks,
            shares=shares,
            saves=raw_metrics.get('saves', 0),
            comments=raw_metrics.get('comments', 0),
            likes=raw_metrics.get('likes', 0),
            video_views=raw_metrics.get('video_views'),
            video_retention_rate=raw_metrics.get('video_retention_rate'),
            story_completion_rate=raw_metrics.get('story_completion_rate'),
            ctr=ctr,
            viral_coefficient=viral_coefficient
        )
    
    async def get_post_metrics(self, post_id: str) -> Dict[str, Any]:
        """Get raw metrics for a post"""
        # This would fetch from platform APIs
        return {
            'likes': 250,
            'comments': 35,
            'shares': 42,
            'saves': 78,
            'impressions': 10000,
            'reach': 8500,
            'clicks': 156,
            'video_views': 7500,
            'video_retention_rate': 0.65
        }
    
    async def get_optimization_suggestions(
        self, 
        performance: PerformanceMetrics,
        content: Dict[str, Any]
    ) -> List[str]:
        """Get performance optimization suggestions"""
        suggestions = []
        
        # Engagement rate optimization
        if performance.engagement_rate < 2.0:
            suggestions.append("Consider adding more engaging CTAs or questions to boost engagement")
        
        # CTR optimization
        if performance.ctr < 1.0:
            suggestions.append("Optimize your captions and thumbnails to improve click-through rate")
        
        # Viral potential
        if performance.viral_coefficient < 0.5:
            suggestions.append("Create more shareable content with broader appeal")
        
        # Video-specific optimizations
        if performance.video_retention_rate and performance.video_retention_rate < 50:
            suggestions.append("Improve video hooks in the first 3 seconds to increase retention")
        
        return suggestions


class UniversalPlatformPublisher(ABC):
    """Enhanced base class for all platform publishers with BI capabilities"""
    
    def __init__(self, client_id: str, platform_name: str):
        self.client_id = client_id
        self.platform_name = platform_name
        self.revenue_tracker = RevenueTracker(client_id, platform_name)
        self.performance_monitor = PerformanceMonitor(client_id, platform_name)
        self.encryption_manager = EncryptionManager()
        self.content_analyzer = UniversalContentAnalyzer()
        self.analytics_service = UniversalAnalyticsService()
        self._authenticated = False
        self._credentials = {}
        
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, str]) -> bool:
        """Platform-specific authentication"""
        pass
    
    @abstractmethod
    async def publish_content(self, content: Dict[str, Any]) -> PublishResult:
        """Publish content with revenue optimization"""
        pass
    
    @abstractmethod
    async def get_optimal_posting_time(self, content_type: str, business_niche: str) -> datetime:
        """AI-powered optimal posting time for this platform"""
        pass
    
    @abstractmethod
    async def analyze_audience_engagement(self, business_niche: str) -> Dict[str, Any]:
        """Analyze audience for this specific business type"""
        pass
    
    @abstractmethod
    async def get_platform_optimizations(self, content: Dict[str, Any], business_niche: str) -> Dict[str, Any]:
        """Get platform-specific optimizations"""
        pass
    
    async def track_revenue_impact(self, post_id: str, content: Dict[str, Any]) -> Dict[str, float]:
        """Track revenue attribution for this post"""
        return await self.revenue_tracker.track_post_revenue(post_id, content)
    
    async def optimize_for_revenue(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize content for maximum revenue potential"""
        business_niche = await self.detect_business_niche(content.get('text', ''))
        audience_data = await self.analyze_audience_engagement(business_niche)
        optimal_time = await self.get_optimal_posting_time(
            content.get('type', 'text'), 
            business_niche
        )
        
        optimizations = {
            'optimal_posting_time': optimal_time,
            'revenue_optimization': await self.revenue_tracker.suggest_optimizations(content),
            'audience_targeting': audience_data,
            'platform_specific_tweaks': await self.get_platform_optimizations(content, business_niche),
            'ab_test_variants': await self._generate_ab_test_variants(content, business_niche),
            'estimated_performance': await self._estimate_performance(content, optimizations)
        }
        
        return optimizations
    
    async def detect_business_niche(self, text: str) -> str:
        """Detect business niche using AI"""
        if not text:
            return BusinessNicheType.OTHER.value
            
        analysis = await self.content_analyzer.analyze_content(text)
        return analysis.business_niche.value
    
    async def calculate_revenue_potential(
        self, 
        content: Dict[str, Any], 
        metadata: Dict[str, Any]
    ) -> float:
        """Calculate estimated revenue potential for content"""
        # Base calculation on multiple factors
        factors = {
            'audience_size': await self._get_audience_size(),
            'engagement_rate': await self._get_average_engagement_rate(),
            'conversion_rate': await self._get_platform_conversion_rate(),
            'content_quality_score': await self._assess_content_quality(content),
            'timing_score': await self._assess_timing_score(metadata),
            'trending_score': await self._assess_trending_score(content, metadata)
        }
        
        # Weighted calculation
        base_potential = factors['audience_size'] * factors['engagement_rate'] * 0.01
        quality_multiplier = 1 + (factors['content_quality_score'] * 0.5)
        timing_multiplier = 1 + (factors['timing_score'] * 0.3)
        trending_multiplier = 1 + (factors['trending_score'] * 0.4)
        
        revenue_potential = (
            base_potential * 
            factors['conversion_rate'] * 
            quality_multiplier * 
            timing_multiplier * 
            trending_multiplier * 
            await self._get_average_transaction_value()
        )
        
        return round(revenue_potential, 2)
    
    async def _generate_ab_test_variants(
        self, 
        content: Dict[str, Any], 
        business_niche: str
    ) -> List[Dict[str, Any]]:
        """Generate A/B test variants for content"""
        variants = []
        
        # Original version
        variants.append({
            'variant_id': 'original',
            'changes': [],
            'hypothesis': 'Control version'
        })
        
        # CTA variation
        variants.append({
            'variant_id': 'cta_variation',
            'changes': ['Modified call-to-action'],
            'hypothesis': 'Stronger CTA will increase conversions'
        })
        
        # Hashtag variation
        variants.append({
            'variant_id': 'hashtag_variation',
            'changes': ['Optimized hashtag selection'],
            'hypothesis': 'Better hashtags will increase reach'
        })
        
        return variants
    
    async def _get_audience_size(self) -> int:
        """Get current audience size"""
        # This would fetch from platform API
        return 10000  # Placeholder
    
    async def _get_average_engagement_rate(self) -> float:
        """Get average engagement rate"""
        # Calculate from historical data
        return 3.5  # 3.5% engagement rate
    
    async def _get_platform_conversion_rate(self) -> float:
        """Get platform-specific conversion rate"""
        return await self.revenue_tracker._get_platform_conversion_rate()
    
    async def _assess_content_quality(self, content: Dict[str, Any]) -> float:
        """Assess content quality score (0-1)"""
        # AI-based quality assessment
        score = 0.0
        
        # Check for media
        if content.get('media_url') or content.get('video_file'):
            score += 0.3
            
        # Check content length
        text_length = len(content.get('text', ''))
        if 100 <= text_length <= 500:
            score += 0.2
            
        # Check for hashtags
        if '#' in content.get('text', ''):
            score += 0.1
            
        # Check for CTA
        cta_keywords = ['link', 'shop', 'buy', 'learn', 'discover', 'join']
        if any(keyword in content.get('text', '').lower() for keyword in cta_keywords):
            score += 0.2
            
        # Check for emotional appeal
        emotional_words = ['amazing', 'incredible', 'transform', 'discover', 'exclusive']
        if any(word in content.get('text', '').lower() for word in emotional_words):
            score += 0.2
            
        return min(score, 1.0)
    
    async def _assess_timing_score(self, metadata: Dict[str, Any]) -> float:
        """Assess timing optimization score (0-1)"""
        # Check if posting at optimal time
        if metadata.get('optimal_posting_time'):
            return 0.9
        return 0.5
    
    async def _assess_trending_score(
        self, 
        content: Dict[str, Any], 
        metadata: Dict[str, Any]
    ) -> float:
        """Assess trending/viral potential score (0-1)"""
        # Check for trending hashtags, topics, etc.
        score = 0.0
        
        # Check for trending hashtags
        if metadata.get('trending_hashtags'):
            score += 0.4
            
        # Check for trending topics
        if metadata.get('trending_topics'):
            score += 0.3
            
        # Check for viral elements
        viral_elements = ['challenge', 'trend', 'viral', 'breaking']
        if any(element in content.get('text', '').lower() for element in viral_elements):
            score += 0.3
            
        return min(score, 1.0)
    
    async def _get_average_transaction_value(self) -> float:
        """Get average transaction value"""
        return 75.0  # Placeholder
    
    async def _get_historical_average_performance(self) -> Dict[str, Any]:
        """Get historical average performance metrics"""
        # This would query the database for historical data
        return {
            'reach': 5000,
            'engagement': 150,
            'conversions': 5
        }
    
    def log_activity(self, action: str, details: Dict[str, Any], success: bool = True):
        """Log platform activity"""
        log_data = {
            'platform': self.platform_name,
            'client_id': self.client_id,
            'action': action,
            'success': success,
            'timestamp': datetime.utcnow().isoformat(),
            **details
        }
        
        if success:
            logger.info(f"Platform activity: {json.dumps(log_data)}")
        else:
            logger.error(f"Platform activity failed: {json.dumps(log_data)}")
    
    def handle_publish_error(self, platform: str, error_message: str) -> PublishResult:
        """Handle publishing errors consistently"""
        self.log_activity('publish', {'error': error_message}, success=False)
        
        return PublishResult(
            platform=platform,
            status=PublishStatus.FAILED,
            post_id=None,
            error_message=error_message,
            timestamp=datetime.utcnow()
        )