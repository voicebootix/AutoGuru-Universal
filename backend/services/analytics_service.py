"""
Universal Analytics Service for AutoGuru Universal

This module provides comprehensive analytics tracking for any business niche.
It tracks performance metrics across all social media platforms, calculates ROI,
measures business impact, and provides actionable insights without hardcoding
business-specific logic.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
import json
from collections import defaultdict, Counter
import statistics

from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.models.content_models import (
    Platform, BusinessNicheType, ContentFormat, BusinessNiche
)
from backend.config.settings import get_settings
from backend.utils.encryption import EncryptionService
from backend.database.connection import get_db_session, get_db_context

# Configure logging
logger = logging.getLogger(__name__)
settings = get_settings()


class MetricType(str, Enum):
    """Types of metrics tracked"""
    ENGAGEMENT = "engagement"
    REACH = "reach"
    CONVERSION = "conversion"
    GROWTH = "growth"
    REVENUE = "revenue"
    BRAND_AWARENESS = "brand_awareness"
    SENTIMENT = "sentiment"


class TimeFrame(str, Enum):
    """Time frame options for analytics"""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"
    CUSTOM = "custom"


class AttributionModel(str, Enum):
    """Attribution models for conversion tracking"""
    FIRST_TOUCH = "first_touch"
    LAST_TOUCH = "last_touch"
    LINEAR = "linear"
    TIME_DECAY = "time_decay"
    POSITION_BASED = "position_based"
    DATA_DRIVEN = "data_driven"


# Analytics Data Models
class EngagementMetrics(BaseModel):
    """Engagement metrics for content performance"""
    likes: int = Field(0, ge=0, description="Number of likes")
    comments: int = Field(0, ge=0, description="Number of comments")
    shares: int = Field(0, ge=0, description="Number of shares")
    saves: int = Field(0, ge=0, description="Number of saves/bookmarks")
    clicks: int = Field(0, ge=0, description="Number of clicks")
    reactions: Dict[str, int] = Field(default_factory=dict, description="Reaction breakdown")
    
    @property
    def total_engagement(self) -> int:
        """Calculate total engagement"""
        return self.likes + self.comments + self.shares + self.saves + self.clicks


class ReachMetrics(BaseModel):
    """Reach and impression metrics"""
    impressions: int = Field(0, ge=0, description="Total impressions")
    unique_reach: int = Field(0, ge=0, description="Unique users reached")
    organic_reach: int = Field(0, ge=0, description="Organic reach")
    paid_reach: int = Field(0, ge=0, description="Paid reach")
    viral_reach: int = Field(0, ge=0, description="Viral reach")
    
    @property
    def reach_rate(self) -> float:
        """Calculate reach rate"""
        if self.impressions == 0:
            return 0.0
        return self.unique_reach / self.impressions


class PerformanceMetrics(BaseModel):
    """Comprehensive content performance metrics"""
    content_id: str = Field(..., description="Unique content identifier")
    platform: Platform = Field(..., description="Social media platform")
    published_at: datetime = Field(..., description="Publication timestamp")
    content_type: ContentFormat = Field(..., description="Type of content")
    
    engagement: EngagementMetrics = Field(..., description="Engagement metrics")
    reach: ReachMetrics = Field(..., description="Reach metrics")
    
    engagement_rate: float = Field(0.0, ge=0.0, le=1.0, description="Engagement rate")
    click_through_rate: float = Field(0.0, ge=0.0, le=1.0, description="CTR")
    conversion_rate: float = Field(0.0, ge=0.0, le=1.0, description="Conversion rate")
    
    audience_sentiment: float = Field(0.0, ge=-1.0, le=1.0, description="Sentiment score")
    quality_score: float = Field(0.0, ge=0.0, le=1.0, description="Content quality score")
    
    peak_engagement_time: Optional[datetime] = Field(None, description="Peak engagement time")
    audience_demographics: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        schema_extra = {
            "example": {
                "content_id": "content_123",
                "platform": "instagram",
                "published_at": "2024-01-15T10:00:00Z",
                "content_type": "carousel",
                "engagement": {
                    "likes": 1250,
                    "comments": 87,
                    "shares": 43,
                    "saves": 156,
                    "clicks": 324
                },
                "reach": {
                    "impressions": 15000,
                    "unique_reach": 12000,
                    "organic_reach": 10000,
                    "paid_reach": 2000,
                    "viral_reach": 500
                },
                "engagement_rate": 0.128,
                "click_through_rate": 0.027,
                "conversion_rate": 0.015,
                "audience_sentiment": 0.85,
                "quality_score": 0.92
            }
        }


class ViralSuccessMetrics(BaseModel):
    """Metrics for measuring viral success"""
    viral_score: float = Field(..., ge=0.0, le=1.0, description="Overall viral score")
    viral_velocity: float = Field(..., ge=0.0, description="Speed of viral spread")
    amplification_rate: float = Field(..., ge=0.0, description="Content amplification rate")
    
    shares_per_hour: List[int] = Field(..., description="Shares per hour timeline")
    reach_growth_curve: List[int] = Field(..., description="Reach growth over time")
    
    viral_peak_time: datetime = Field(..., description="Time of peak virality")
    viral_duration_hours: float = Field(..., ge=0.0, description="Duration of viral activity")
    
    influencer_shares: List[Dict[str, Any]] = Field(default_factory=list)
    viral_communities: List[str] = Field(default_factory=list)
    
    prediction_accuracy: float = Field(..., ge=0.0, le=1.0, description="Accuracy vs prediction")


class ComparisonAnalytics(BaseModel):
    """Analytics for comparing multiple content pieces"""
    comparison_id: str = Field(..., description="Unique comparison identifier")
    content_ids: List[str] = Field(..., min_items=2, description="Content IDs being compared")
    comparison_period: Dict[str, datetime] = Field(..., description="Time period")
    
    best_performer: str = Field(..., description="Best performing content ID")
    worst_performer: str = Field(..., description="Worst performing content ID")
    
    metrics_comparison: Dict[str, Dict[str, float]] = Field(...)
    statistical_significance: Dict[str, float] = Field(...)
    
    key_differences: List[str] = Field(..., description="Key performance differences")
    recommendations: List[str] = Field(..., description="Improvement recommendations")


class GrowthAnalytics(BaseModel):
    """Audience growth analytics"""
    client_id: str = Field(..., description="Client identifier")
    timeframe: TimeFrame = Field(..., description="Analysis timeframe")
    period_start: datetime = Field(..., description="Period start")
    period_end: datetime = Field(..., description="Period end")
    
    starting_followers: Dict[Platform, int] = Field(..., description="Starting follower counts")
    ending_followers: Dict[Platform, int] = Field(..., description="Ending follower counts")
    
    total_growth: int = Field(..., description="Total follower growth")
    growth_rate: float = Field(..., description="Growth rate percentage")
    
    daily_growth: List[Dict[str, Any]] = Field(..., description="Daily growth breakdown")
    growth_sources: Dict[str, int] = Field(..., description="Growth by source")
    
    churn_rate: float = Field(..., ge=0.0, le=1.0, description="Follower churn rate")
    retention_rate: float = Field(..., ge=0.0, le=1.0, description="Follower retention rate")
    
    growth_velocity: float = Field(..., description="Growth acceleration")
    projected_growth: Dict[str, int] = Field(..., description="Projected future growth")


class DemographicAnalytics(BaseModel):
    """Audience demographic analytics"""
    client_id: str = Field(..., description="Client identifier")
    analysis_date: datetime = Field(default_factory=datetime.utcnow)
    
    age_distribution: Dict[str, float] = Field(..., description="Age range percentages")
    gender_distribution: Dict[str, float] = Field(..., description="Gender percentages")
    location_distribution: Dict[str, float] = Field(..., description="Geographic distribution")
    
    language_distribution: Dict[str, float] = Field(default_factory=dict)
    device_distribution: Dict[str, float] = Field(default_factory=dict)
    
    interests: List[Dict[str, Any]] = Field(..., description="Top audience interests")
    behaviors: List[Dict[str, Any]] = Field(..., description="Audience behaviors")
    
    income_brackets: Dict[str, float] = Field(default_factory=dict)
    education_levels: Dict[str, float] = Field(default_factory=dict)
    
    demographic_trends: List[str] = Field(..., description="Notable demographic trends")


class EngagementPatterns(BaseModel):
    """Audience engagement pattern analysis"""
    client_id: str = Field(..., description="Client identifier")
    analysis_period: Dict[str, datetime] = Field(..., description="Analysis period")
    
    peak_engagement_times: Dict[Platform, List[str]] = Field(...)
    engagement_by_content_type: Dict[ContentFormat, float] = Field(...)
    engagement_by_day_of_week: Dict[str, float] = Field(...)
    
    average_response_time: float = Field(..., description="Avg response time in minutes")
    engagement_depth: float = Field(..., description="Depth of engagement score")
    
    loyal_engagers: List[Dict[str, Any]] = Field(..., description="Most engaged users")
    engagement_clusters: List[Dict[str, Any]] = Field(..., description="User engagement clusters")
    
    content_preferences: Dict[str, float] = Field(...)
    engagement_triggers: List[str] = Field(...)


class AudienceSegment(BaseModel):
    """Individual audience segment"""
    segment_id: str = Field(..., description="Unique segment identifier")
    segment_name: str = Field(..., description="Segment name")
    size: int = Field(..., ge=0, description="Segment size")
    
    characteristics: Dict[str, Any] = Field(..., description="Segment characteristics")
    engagement_score: float = Field(..., ge=0.0, le=1.0)
    conversion_potential: float = Field(..., ge=0.0, le=1.0)
    lifetime_value: float = Field(..., ge=0.0)
    
    preferred_content: List[str] = Field(...)
    preferred_platforms: List[Platform] = Field(...)
    
    growth_rate: float = Field(..., description="Segment growth rate")


class AudienceSegments(BaseModel):
    """High-value audience segments analysis"""
    client_id: str = Field(..., description="Client identifier")
    total_audience: int = Field(..., ge=0, description="Total audience size")
    
    segments: List[AudienceSegment] = Field(..., min_items=1)
    segmentation_method: str = Field(..., description="Method used for segmentation")
    
    most_valuable_segment: AudienceSegment = Field(...)
    fastest_growing_segment: AudienceSegment = Field(...)
    
    segment_overlap: Dict[str, float] = Field(default_factory=dict)
    recommendations: List[str] = Field(...)


class ROIAnalysis(BaseModel):
    """Return on Investment analysis"""
    client_id: str = Field(..., description="Client identifier")
    timeframe: TimeFrame = Field(..., description="Analysis timeframe")
    analysis_period: Dict[str, datetime] = Field(...)
    
    total_investment: float = Field(..., ge=0.0, description="Total investment")
    investment_breakdown: Dict[str, float] = Field(..., description="Investment by category")
    
    total_revenue: float = Field(..., ge=0.0, description="Total revenue generated")
    revenue_breakdown: Dict[str, float] = Field(..., description="Revenue by source")
    
    roi_percentage: float = Field(..., description="ROI percentage")
    profit_margin: float = Field(..., description="Profit margin")
    
    cost_per_acquisition: float = Field(..., ge=0.0)
    customer_lifetime_value: float = Field(..., ge=0.0)
    
    platform_roi: Dict[Platform, float] = Field(..., description="ROI by platform")
    content_type_roi: Dict[ContentFormat, float] = Field(...)
    
    break_even_point: Optional[datetime] = Field(None)
    payback_period_days: Optional[int] = Field(None)
    
    attribution_analysis: Dict[str, float] = Field(...)
    recommendations: List[str] = Field(...)


class LeadMetrics(BaseModel):
    """Lead generation metrics"""
    client_id: str = Field(..., description="Client identifier")
    measurement_period: Dict[str, datetime] = Field(...)
    
    total_leads: int = Field(..., ge=0, description="Total leads generated")
    qualified_leads: int = Field(..., ge=0, description="Qualified leads")
    
    lead_sources: Dict[str, int] = Field(..., description="Leads by source")
    lead_quality_score: float = Field(..., ge=0.0, le=1.0)
    
    cost_per_lead: float = Field(..., ge=0.0)
    lead_conversion_rate: float = Field(..., ge=0.0, le=1.0)
    
    lead_velocity: float = Field(..., description="Lead generation velocity")
    lead_nurture_time: float = Field(..., description="Average nurture time in days")
    
    top_converting_content: List[Dict[str, Any]] = Field(...)
    lead_scoring_breakdown: Dict[str, Any] = Field(...)


class ConversionAnalytics(BaseModel):
    """Conversion funnel analytics"""
    client_id: str = Field(..., description="Client identifier")
    funnel_name: str = Field(..., description="Conversion funnel name")
    
    funnel_stages: List[Dict[str, Any]] = Field(..., description="Funnel stages with metrics")
    total_conversions: int = Field(..., ge=0)
    conversion_rate: float = Field(..., ge=0.0, le=1.0)
    
    drop_off_analysis: Dict[str, float] = Field(..., description="Drop-off rates by stage")
    time_to_conversion: Dict[str, float] = Field(..., description="Time metrics")
    
    conversion_paths: List[Dict[str, Any]] = Field(..., description="Common conversion paths")
    assisted_conversions: Dict[Platform, int] = Field(...)
    
    optimization_opportunities: List[str] = Field(...)


class AttributionAnalysis(BaseModel):
    """Social media attribution analysis"""
    client_id: str = Field(..., description="Client identifier")
    attribution_model: AttributionModel = Field(...)
    
    channel_attribution: Dict[Platform, float] = Field(..., description="Attribution by channel")
    content_attribution: Dict[str, float] = Field(..., description="Attribution by content")
    
    touchpoint_analysis: List[Dict[str, Any]] = Field(..., description="Touchpoint analysis")
    attribution_paths: List[Dict[str, Any]] = Field(..., description="Common paths")
    
    cross_channel_impact: Dict[str, float] = Field(...)
    attribution_confidence: float = Field(..., ge=0.0, le=1.0)


# Platform-specific analytics models
class InstagramAnalytics(BaseModel):
    """Instagram-specific analytics"""
    account_id: str = Field(..., description="Instagram account ID")
    analysis_period: Dict[str, datetime] = Field(...)
    
    profile_views: int = Field(..., ge=0)
    website_clicks: int = Field(..., ge=0)
    email_clicks: int = Field(..., ge=0)
    
    story_metrics: Dict[str, Any] = Field(..., description="Story performance")
    reel_metrics: Dict[str, Any] = Field(..., description="Reel performance")
    igtv_metrics: Dict[str, Any] = Field(..., description="IGTV performance")
    
    hashtag_performance: Dict[str, Dict[str, int]] = Field(...)
    mention_metrics: Dict[str, Any] = Field(...)
    
    shopping_metrics: Optional[Dict[str, Any]] = Field(None)
    discovery_metrics: Dict[str, Any] = Field(...)


class LinkedInAnalytics(BaseModel):
    """LinkedIn-specific analytics"""
    page_id: str = Field(..., description="LinkedIn page ID")
    analysis_period: Dict[str, datetime] = Field(...)
    
    page_views: int = Field(..., ge=0)
    unique_visitors: int = Field(..., ge=0)
    
    follower_demographics: Dict[str, Any] = Field(...)
    visitor_demographics: Dict[str, Any] = Field(...)
    
    post_metrics: Dict[str, Any] = Field(...)
    article_metrics: Dict[str, Any] = Field(...)
    
    employee_advocacy: Dict[str, Any] = Field(...)
    competitor_benchmarking: Dict[str, Any] = Field(...)


class TikTokAnalytics(BaseModel):
    """TikTok-specific analytics"""
    account_id: str = Field(..., description="TikTok account ID")
    analysis_period: Dict[str, datetime] = Field(...)
    
    video_views: int = Field(..., ge=0)
    profile_views: int = Field(..., ge=0)
    
    average_watch_time: float = Field(..., ge=0.0)
    completion_rate: float = Field(..., ge=0.0, le=1.0)
    
    trending_metrics: Dict[str, Any] = Field(...)
    sound_performance: Dict[str, Any] = Field(...)
    
    effect_usage: Dict[str, Any] = Field(...)
    duet_metrics: Dict[str, Any] = Field(...)


class YouTubeAnalytics(BaseModel):
    """YouTube-specific analytics"""
    channel_id: str = Field(..., description="YouTube channel ID")
    analysis_period: Dict[str, datetime] = Field(...)
    
    views: int = Field(..., ge=0)
    watch_time_hours: float = Field(..., ge=0.0)
    
    average_view_duration: float = Field(..., ge=0.0)
    click_through_rate: float = Field(..., ge=0.0, le=1.0)
    
    subscriber_change: int = Field(...)
    revenue_metrics: Optional[Dict[str, float]] = Field(None)
    
    traffic_sources: Dict[str, int] = Field(...)
    audience_retention: Dict[str, float] = Field(...)
    
    playlist_performance: Dict[str, Any] = Field(...)
    end_screen_metrics: Dict[str, Any] = Field(...)


# Universal analytics models
class BenchmarkData(BaseModel):
    """Industry benchmarking data"""
    client_id: str = Field(..., description="Client identifier")
    business_niche: BusinessNicheType = Field(...)
    
    industry_averages: Dict[str, float] = Field(...)
    client_performance: Dict[str, float] = Field(...)
    
    percentile_rankings: Dict[str, float] = Field(...)
    competitive_position: str = Field(...)
    
    strengths: List[str] = Field(...)
    weaknesses: List[str] = Field(...)
    
    improvement_opportunities: List[Dict[str, Any]] = Field(...)
    best_practices: List[str] = Field(...)


class ContentTypeAnalysis(BaseModel):
    """Analysis of content types performance"""
    client_id: str = Field(..., description="Client identifier")
    analysis_period: Dict[str, datetime] = Field(...)
    
    performance_by_type: Dict[ContentFormat, Dict[str, float]] = Field(...)
    optimal_mix: Dict[ContentFormat, float] = Field(..., description="Recommended content mix")
    
    trending_formats: List[ContentFormat] = Field(...)
    declining_formats: List[ContentFormat] = Field(...)
    
    audience_preferences: Dict[ContentFormat, float] = Field(...)
    platform_recommendations: Dict[Platform, List[ContentFormat]] = Field(...)


class PostingRecommendations(BaseModel):
    """Posting schedule optimization recommendations"""
    client_id: str = Field(..., description="Client identifier")
    
    optimal_times: Dict[Platform, List[str]] = Field(...)
    optimal_frequency: Dict[Platform, int] = Field(..., description="Posts per week")
    
    day_of_week_performance: Dict[str, float] = Field(...)
    time_zone_considerations: List[str] = Field(...)
    
    content_calendar: List[Dict[str, Any]] = Field(..., description="Recommended calendar")
    seasonal_adjustments: Dict[str, Any] = Field(...)


class InsightsList(BaseModel):
    """Actionable insights list"""
    client_id: str = Field(..., description="Client identifier")
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    insights: List[Dict[str, Any]] = Field(..., min_items=1)
    priority_actions: List[str] = Field(..., max_items=5)
    
    predicted_impact: Dict[str, float] = Field(...)
    implementation_difficulty: Dict[str, str] = Field(...)


# Reporting models
class WeeklyReport(BaseModel):
    """Weekly performance report"""
    client_id: str = Field(..., description="Client identifier")
    week_start: datetime = Field(...)
    week_end: datetime = Field(...)
    
    executive_summary: str = Field(...)
    key_metrics: Dict[str, Any] = Field(...)
    
    content_performance: List[PerformanceMetrics] = Field(...)
    audience_growth: GrowthAnalytics = Field(...)
    
    top_performing_content: List[Dict[str, Any]] = Field(...)
    engagement_trends: Dict[str, Any] = Field(...)
    
    action_items: List[str] = Field(...)
    next_week_focus: List[str] = Field(...)


class MonthlyDashboard(BaseModel):
    """Monthly dashboard data"""
    client_id: str = Field(..., description="Client identifier")
    month: str = Field(..., description="Month in YYYY-MM format")
    
    overview_metrics: Dict[str, Any] = Field(...)
    platform_breakdown: Dict[Platform, Dict[str, Any]] = Field(...)
    
    roi_analysis: ROIAnalysis = Field(...)
    audience_insights: Dict[str, Any] = Field(...)
    
    content_analysis: ContentTypeAnalysis = Field(...)
    competitive_analysis: BenchmarkData = Field(...)
    
    trends_identified: List[str] = Field(...)
    strategic_recommendations: List[str] = Field(...)


class ExportResult(BaseModel):
    """Data export result"""
    export_id: str = Field(..., description="Export identifier")
    format: str = Field(..., description="Export format")
    
    file_path: Optional[str] = Field(None)
    download_url: Optional[str] = Field(None)
    
    rows_exported: int = Field(..., ge=0)
    file_size_bytes: int = Field(..., ge=0)
    
    export_timestamp: datetime = Field(default_factory=datetime.utcnow)
    expiry_timestamp: Optional[datetime] = Field(None)


class ScheduleResult(BaseModel):
    """Report scheduling result"""
    schedule_id: str = Field(..., description="Schedule identifier")
    client_id: str = Field(...)
    
    report_type: str = Field(...)
    frequency: str = Field(...)
    
    next_run: datetime = Field(...)
    recipients: List[str] = Field(...)
    
    status: str = Field(..., description="Schedule status")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class UniversalAnalyticsService:
    """
    Universal Analytics Service for AutoGuru Universal.
    
    Provides comprehensive analytics tracking, performance measurement,
    and ROI calculation for any business niche. Uses AI-driven analysis
    without hardcoding business-specific logic.
    """
    
    def __init__(self):
        self.encryption_service = EncryptionService()
        self.logger = logger
        self._platform_clients = {}  # Cache for platform API clients
        
    async def initialize(self) -> None:
        """Initialize the analytics service and database connection"""
        async with get_db_session() as session:
            # use session for DB operations
            pass
            
    # CONTENT PERFORMANCE TRACKING
    async def track_content_performance(
        self,
        content_id: str,
        platform: Platform
    ) -> PerformanceMetrics:
        """
        Track performance metrics for a specific content piece.
        
        Args:
            content_id: Unique identifier for the content
            platform: Social media platform
            
        Returns:
            PerformanceMetrics: Comprehensive performance data
        """
        try:
            self.logger.info(f"Tracking performance for content {content_id} on {platform}")
            
            # Fetch raw metrics from platform API
            raw_metrics = await self._fetch_platform_metrics(content_id, platform)
            
            # Calculate derived metrics
            engagement_metrics = EngagementMetrics(
                likes=raw_metrics.get('likes', 0),
                comments=raw_metrics.get('comments', 0),
                shares=raw_metrics.get('shares', 0),
                saves=raw_metrics.get('saves', 0),
                clicks=raw_metrics.get('clicks', 0),
                reactions=raw_metrics.get('reactions', {})
            )
            
            reach_metrics = ReachMetrics(
                impressions=raw_metrics.get('impressions', 0),
                unique_reach=raw_metrics.get('reach', 0),
                organic_reach=raw_metrics.get('organic_reach', 0),
                paid_reach=raw_metrics.get('paid_reach', 0),
                viral_reach=raw_metrics.get('viral_reach', 0)
            )
            
            # Calculate rates
            engagement_rate = self._calculate_engagement_rate(
                engagement_metrics, reach_metrics.unique_reach
            )
            ctr = engagement_metrics.clicks / reach_metrics.impressions if reach_metrics.impressions > 0 else 0.0
            
            # Get content metadata
            content_metadata = await self._get_content_metadata(content_id)
            
            # Create performance metrics
            performance_metrics = PerformanceMetrics(
                content_id=content_id,
                platform=platform,
                published_at=content_metadata.get('published_at', datetime.utcnow()),
                content_type=content_metadata.get('content_type', ContentFormat.TEXT),
                engagement=engagement_metrics,
                reach=reach_metrics,
                engagement_rate=engagement_rate,
                click_through_rate=ctr,
                conversion_rate=raw_metrics.get('conversion_rate', 0.0),
                audience_sentiment=await self._analyze_sentiment(content_id),
                quality_score=await self._calculate_quality_score(content_id, raw_metrics),
                peak_engagement_time=raw_metrics.get('peak_engagement_time'),
                audience_demographics=raw_metrics.get('demographics', {})
            )
            
            # Store metrics in database
            await self._store_performance_metrics(performance_metrics)
            
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"Error tracking content performance: {str(e)}")
            raise
    
    async def calculate_engagement_rate(
        self,
        content_id: str,
        platform: Platform
    ) -> float:
        """
        Calculate the engagement rate for a specific content piece.
        
        Args:
            content_id: Content identifier
            platform: Social media platform
            
        Returns:
            float: Engagement rate (0.0 to 1.0)
        """
        try:
            # Fetch or retrieve cached performance metrics
            metrics = await self._get_or_fetch_metrics(content_id, platform)
            
            if not metrics:
                return 0.0
                
            return metrics.engagement_rate
            
        except Exception as e:
            self.logger.error(f"Error calculating engagement rate: {str(e)}")
            return 0.0
    
    async def measure_viral_success(
        self,
        content_id: str,
        viral_predictions: Dict[str, Any]
    ) -> ViralSuccessMetrics:
        """
        Measure how well content performed against viral predictions.
        
        Args:
            content_id: Content identifier
            viral_predictions: Original viral predictions
            
        Returns:
            ViralSuccessMetrics: Viral success analysis
        """
        try:
            self.logger.info(f"Measuring viral success for content {content_id}")
            
            # Get actual performance data
            performance = await self._get_content_performance(content_id)
            
            # Calculate viral metrics
            viral_score = await self._calculate_viral_score(performance)
            viral_velocity = await self._calculate_viral_velocity(content_id)
            amplification_rate = await self._calculate_amplification_rate(performance)
            
            # Get time-based metrics
            timeline_data = await self._get_viral_timeline(content_id)
            
            # Identify viral contributors
            influencer_shares = await self._identify_influencer_shares(content_id)
            viral_communities = await self._identify_viral_communities(content_id)
            
            # Calculate prediction accuracy
            prediction_accuracy = self._compare_with_predictions(
                performance, viral_predictions
            )
            
            viral_metrics = ViralSuccessMetrics(
                viral_score=viral_score,
                viral_velocity=viral_velocity,
                amplification_rate=amplification_rate,
                shares_per_hour=timeline_data['shares_per_hour'],
                reach_growth_curve=timeline_data['reach_growth'],
                viral_peak_time=timeline_data['peak_time'],
                viral_duration_hours=timeline_data['duration'],
                influencer_shares=influencer_shares,
                viral_communities=viral_communities,
                prediction_accuracy=prediction_accuracy
            )
            
            # Store viral metrics
            await self._store_viral_metrics(content_id, viral_metrics)
            
            return viral_metrics
            
        except Exception as e:
            self.logger.error(f"Error measuring viral success: {str(e)}")
            raise
    
    async def compare_content_performance(
        self,
        content_ids: List[str]
    ) -> ComparisonAnalytics:
        """
        Compare performance across multiple content pieces.
        
        Args:
            content_ids: List of content identifiers to compare
            
        Returns:
            ComparisonAnalytics: Detailed comparison analysis
        """
        try:
            self.logger.info(f"Comparing performance for {len(content_ids)} content pieces")
            
            # Fetch all performance metrics
            all_metrics = []
            for content_id in content_ids:
                metrics = await self._get_all_platform_metrics(content_id)
                all_metrics.append(metrics)
            
            # Aggregate and compare metrics
            comparison_data = await self._aggregate_comparison_metrics(all_metrics)
            
            # Identify best and worst performers
            best_performer = self._identify_best_performer(comparison_data)
            worst_performer = self._identify_worst_performer(comparison_data)
            
            # Calculate statistical significance
            statistical_significance = await self._calculate_statistical_significance(
                comparison_data
            )
            
            # Generate insights
            key_differences = await self._identify_key_differences(comparison_data)
            recommendations = await self._generate_comparison_recommendations(
                comparison_data, key_differences
            )
            
            comparison_analytics = ComparisonAnalytics(
                comparison_id=f"comp_{datetime.utcnow().timestamp()}",
                content_ids=content_ids,
                comparison_period={
                    'start': min(m['published_at'] for m in all_metrics),
                    'end': datetime.utcnow()
                },
                best_performer=best_performer,
                worst_performer=worst_performer,
                metrics_comparison=comparison_data,
                statistical_significance=statistical_significance,
                key_differences=key_differences,
                recommendations=recommendations
            )
            
            return comparison_analytics
            
        except Exception as e:
            self.logger.error(f"Error comparing content performance: {str(e)}")
            raise
    
    # Helper methods for content performance
    async def _fetch_platform_metrics(
        self,
        content_id: str,
        platform: Platform
    ) -> Dict[str, Any]:
        """Fetch metrics from platform API"""
        # This would integrate with platform-specific APIs
        # For now, returning simulated data structure
        query = """
            SELECT metrics_data 
            FROM content_metrics 
            WHERE content_id = $1 AND platform = $2
            ORDER BY tracked_at DESC
            LIMIT 1
        """
        
        async with get_db_session() as session:
            result = await session.fetch_one(query, content_id, platform.value)
        
        if result:
            return json.loads(result['metrics_data'])
        
        # If no stored metrics, fetch from platform API
        # This is where you'd integrate with actual platform APIs
        return {
            'likes': 0,
            'comments': 0,
            'shares': 0,
            'saves': 0,
            'clicks': 0,
            'impressions': 0,
            'reach': 0,
            'organic_reach': 0,
            'paid_reach': 0,
            'viral_reach': 0,
            'reactions': {},
            'conversion_rate': 0.0,
            'demographics': {}
        }
    
    def _calculate_engagement_rate(
        self,
        engagement: EngagementMetrics,
        reach: int
    ) -> float:
        """Calculate engagement rate"""
        if reach == 0:
            return 0.0
        
        total_engagement = engagement.total_engagement
        return min(total_engagement / reach, 1.0)
    
    async def _analyze_sentiment(self, content_id: str) -> float:
        """Analyze audience sentiment from comments and reactions"""
        # This would use NLP to analyze comments
        # For now, returning a placeholder
        return 0.85
    
    async def _calculate_quality_score(
        self,
        content_id: str,
        metrics: Dict[str, Any]
    ) -> float:
        """Calculate content quality score based on multiple factors"""
        # Quality score based on engagement, completion rate, etc.
        factors = {
            'engagement_rate': 0.3,
            'completion_rate': 0.2,
            'share_rate': 0.2,
            'save_rate': 0.1,
            'comment_quality': 0.2
        }
        
        score = 0.0
        # Calculate weighted score
        # Placeholder implementation
        return 0.85
    
    async def _get_content_metadata(self, content_id: str) -> Dict[str, Any]:
        """Get content metadata from database"""
        query = """
            SELECT content_type, published_at, platform, client_id
            FROM content
            WHERE content_id = $1
        """
        
        async with get_db_session() as session:
            result = await session.fetch_one(query, content_id)
        
        if result:
            return dict(result)
        
        return {
            'content_type': ContentFormat.TEXT,
            'published_at': datetime.utcnow()
        }
    
    async def _store_performance_metrics(
        self,
        metrics: PerformanceMetrics
    ) -> None:
        """Store performance metrics in database"""
        query = """
            INSERT INTO performance_metrics 
            (content_id, platform, metrics_data, tracked_at)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (content_id, platform) 
            DO UPDATE SET 
                metrics_data = $3,
                tracked_at = $4
        """
        
        metrics_json = json.dumps(metrics.dict())
        async with get_db_session() as session:
            await session.execute(
                query,
                metrics.content_id,
                metrics.platform.value,
                metrics_json,
                datetime.utcnow()
            )

    # AUDIENCE ANALYTICS
    async def analyze_audience_growth(
        self,
        client_id: str,
        timeframe: str
    ) -> GrowthAnalytics:
        """
        Analyze audience growth patterns over a specified timeframe.
        
        Args:
            client_id: Client identifier
            timeframe: Analysis timeframe (day, week, month, etc.)
            
        Returns:
            GrowthAnalytics: Detailed growth analysis
        """
        try:
            self.logger.info(f"Analyzing audience growth for client {client_id}")
            
            # Parse timeframe
            tf = TimeFrame(timeframe.lower())
            period_start, period_end = self._get_timeframe_bounds(tf)
            
            # Get follower counts across platforms
            starting_followers = await self._get_follower_counts(client_id, period_start)
            ending_followers = await self._get_follower_counts(client_id, period_end)
            
            # Calculate growth metrics
            total_growth = sum(ending_followers.values()) - sum(starting_followers.values())
            growth_rate = self._calculate_growth_rate(starting_followers, ending_followers)
            
            # Get daily breakdown
            daily_growth = await self._get_daily_growth_breakdown(
                client_id, period_start, period_end
            )
            
            # Analyze growth sources
            growth_sources = await self._analyze_growth_sources(client_id, period_start, period_end)
            
            # Calculate churn and retention
            churn_rate = await self._calculate_churn_rate(client_id, period_start, period_end)
            retention_rate = 1.0 - churn_rate
            
            # Calculate growth velocity
            growth_velocity = await self._calculate_growth_velocity(daily_growth)
            
            # Project future growth
            projected_growth = await self._project_growth(
                ending_followers, growth_rate, growth_velocity
            )
            
            growth_analytics = GrowthAnalytics(
                client_id=client_id,
                timeframe=tf,
                period_start=period_start,
                period_end=period_end,
                starting_followers=starting_followers,
                ending_followers=ending_followers,
                total_growth=total_growth,
                growth_rate=growth_rate,
                daily_growth=daily_growth,
                growth_sources=growth_sources,
                churn_rate=churn_rate,
                retention_rate=retention_rate,
                growth_velocity=growth_velocity,
                projected_growth=projected_growth
            )
            
            return growth_analytics
            
        except Exception as e:
            self.logger.error(f"Error analyzing audience growth: {str(e)}")
            raise
    
    async def track_audience_demographics(
        self,
        client_id: str
    ) -> DemographicAnalytics:
        """
        Track and analyze audience demographic data.
        
        Args:
            client_id: Client identifier
            
        Returns:
            DemographicAnalytics: Comprehensive demographic analysis
        """
        try:
            self.logger.info(f"Tracking audience demographics for client {client_id}")
            
            # Aggregate demographics across platforms
            demographics_data = await self._aggregate_platform_demographics(client_id)
            
            # Process age distribution
            age_distribution = self._normalize_distribution(
                demographics_data.get('age', {})
            )
            
            # Process gender distribution
            gender_distribution = self._normalize_distribution(
                demographics_data.get('gender', {})
            )
            
            # Process location distribution
            location_distribution = self._process_location_data(
                demographics_data.get('location', {})
            )
            
            # Get additional demographic data
            language_distribution = self._normalize_distribution(
                demographics_data.get('language', {})
            )
            device_distribution = self._normalize_distribution(
                demographics_data.get('device', {})
            )
            
            # Analyze interests and behaviors
            interests = await self._analyze_audience_interests(client_id)
            behaviors = await self._analyze_audience_behaviors(client_id)
            
            # Get socioeconomic data if available
            income_brackets = demographics_data.get('income', {})
            education_levels = demographics_data.get('education', {})
            
            # Identify demographic trends
            demographic_trends = await self._identify_demographic_trends(
                client_id, demographics_data
            )
            
            demographic_analytics = DemographicAnalytics(
                client_id=client_id,
                age_distribution=age_distribution,
                gender_distribution=gender_distribution,
                location_distribution=location_distribution,
                language_distribution=language_distribution,
                device_distribution=device_distribution,
                interests=interests,
                behaviors=behaviors,
                income_brackets=income_brackets,
                education_levels=education_levels,
                demographic_trends=demographic_trends
            )
            
            return demographic_analytics
            
        except Exception as e:
            self.logger.error(f"Error tracking audience demographics: {str(e)}")
            raise
    
    async def measure_audience_engagement_patterns(
        self,
        client_id: str
    ) -> EngagementPatterns:
        """
        Measure and analyze audience engagement patterns.
        
        Args:
            client_id: Client identifier
            
        Returns:
            EngagementPatterns: Detailed engagement pattern analysis
        """
        try:
            self.logger.info(f"Measuring engagement patterns for client {client_id}")
            
            # Define analysis period (last 90 days)
            analysis_period = {
                'start': datetime.utcnow() - timedelta(days=90),
                'end': datetime.utcnow()
            }
            
            # Analyze peak engagement times per platform
            peak_engagement_times = await self._analyze_peak_engagement_times(
                client_id, analysis_period
            )
            
            # Analyze engagement by content type
            engagement_by_content_type = await self._analyze_engagement_by_content_type(
                client_id, analysis_period
            )
            
            # Analyze engagement by day of week
            engagement_by_day_of_week = await self._analyze_engagement_by_day(
                client_id, analysis_period
            )
            
            # Calculate average response time
            average_response_time = await self._calculate_average_response_time(
                client_id, analysis_period
            )
            
            # Calculate engagement depth
            engagement_depth = await self._calculate_engagement_depth(
                client_id, analysis_period
            )
            
            # Identify loyal engagers
            loyal_engagers = await self._identify_loyal_engagers(
                client_id, analysis_period
            )
            
            # Identify engagement clusters
            engagement_clusters = await self._identify_engagement_clusters(
                client_id, analysis_period
            )
            
            # Analyze content preferences
            content_preferences = await self._analyze_content_preferences(
                client_id, analysis_period
            )
            
            # Identify engagement triggers
            engagement_triggers = await self._identify_engagement_triggers(
                client_id, analysis_period
            )
            
            engagement_patterns = EngagementPatterns(
                client_id=client_id,
                analysis_period=analysis_period,
                peak_engagement_times=peak_engagement_times,
                engagement_by_content_type=engagement_by_content_type,
                engagement_by_day_of_week=engagement_by_day_of_week,
                average_response_time=average_response_time,
                engagement_depth=engagement_depth,
                loyal_engagers=loyal_engagers,
                engagement_clusters=engagement_clusters,
                content_preferences=content_preferences,
                engagement_triggers=engagement_triggers
            )
            
            return engagement_patterns
            
        except Exception as e:
            self.logger.error(f"Error measuring engagement patterns: {str(e)}")
            raise
    
    async def identify_high_value_audience_segments(
        self,
        client_id: str
    ) -> AudienceSegments:
        """
        Identify and analyze high-value audience segments.
        
        Args:
            client_id: Client identifier
            
        Returns:
            AudienceSegments: High-value segment analysis
        """
        try:
            self.logger.info(f"Identifying high-value segments for client {client_id}")
            
            # Get total audience size
            total_audience = await self._get_total_audience_size(client_id)
            
            # Perform segmentation using multiple methods
            segments = await self._perform_audience_segmentation(client_id)
            
            # Calculate value metrics for each segment
            for segment in segments:
                segment.engagement_score = await self._calculate_segment_engagement_score(
                    client_id, segment.segment_id
                )
                segment.conversion_potential = await self._calculate_conversion_potential(
                    client_id, segment.segment_id
                )
                segment.lifetime_value = await self._calculate_segment_ltv(
                    client_id, segment.segment_id
                )
                segment.growth_rate = await self._calculate_segment_growth_rate(
                    client_id, segment.segment_id
                )
            
            # Identify most valuable and fastest growing segments
            most_valuable_segment = max(segments, key=lambda s: s.lifetime_value)
            fastest_growing_segment = max(segments, key=lambda s: s.growth_rate)
            
            # Calculate segment overlap
            segment_overlap = await self._calculate_segment_overlap(client_id, segments)
            
            # Generate recommendations
            recommendations = await self._generate_segment_recommendations(
                client_id, segments
            )
            
            audience_segments = AudienceSegments(
                client_id=client_id,
                total_audience=total_audience,
                segments=segments,
                segmentation_method="Multi-dimensional clustering with behavioral analysis",
                most_valuable_segment=most_valuable_segment,
                fastest_growing_segment=fastest_growing_segment,
                segment_overlap=segment_overlap,
                recommendations=recommendations
            )
            
            return audience_segments
            
        except Exception as e:
            self.logger.error(f"Error identifying audience segments: {str(e)}")
            raise
    
    # Helper methods that were referenced but not implemented
    async def _get_or_fetch_metrics(
        self,
        content_id: str,
        platform: Platform
    ) -> Optional[PerformanceMetrics]:
        """Get metrics from cache or fetch if needed"""
        # Check cache first
        cache_key = f"metrics:{content_id}:{platform.value}"
        
        # For now, fetch from database
        query = """
            SELECT metrics_data 
            FROM performance_metrics 
            WHERE content_id = $1 AND platform = $2
            ORDER BY tracked_at DESC
            LIMIT 1
        """
        
        async with get_db_session() as session:
            result = await session.fetch_one(query, content_id, platform.value)
        
        if result:
            return PerformanceMetrics(**json.loads(result['metrics_data']))
        
        # If not found, track new metrics
        return await self.track_content_performance(content_id, platform)
    
    async def _get_content_performance(self, content_id: str) -> Dict[str, Any]:
        """Get aggregated content performance across all platforms"""
        query = """
            SELECT platform, metrics_data 
            FROM performance_metrics 
            WHERE content_id = $1
        """
        
        async with get_db_session() as session:
            results = await session.fetch_all(query, content_id)
        
        aggregated = {
            'total_engagement': 0,
            'total_reach': 0,
            'platforms': {}
        }
        
        for result in results:
            metrics = json.loads(result['metrics_data'])
            platform = result['platform']
            aggregated['platforms'][platform] = metrics
            aggregated['total_engagement'] += metrics.get('engagement', {}).get('total_engagement', 0)
            aggregated['total_reach'] += metrics.get('reach', {}).get('unique_reach', 0)
        
        return aggregated
    
    async def _calculate_viral_score(self, performance: Dict[str, Any]) -> float:
        """Calculate viral score based on performance metrics"""
        # Viral score calculation based on multiple factors
        factors = {
            'share_rate': 0.3,
            'reach_growth': 0.25,
            'engagement_velocity': 0.25,
            'influencer_impact': 0.2
        }
        
        # Placeholder calculation
        return min(0.85, performance.get('total_engagement', 0) / max(1, performance.get('total_reach', 1)) * 10)
    
    async def _calculate_viral_velocity(self, content_id: str) -> float:
        """Calculate the speed of viral spread"""
        # This would analyze the rate of growth over time
        return 2.5  # Placeholder
    
    async def _calculate_amplification_rate(self, performance: Dict[str, Any]) -> float:
        """Calculate content amplification rate"""
        # Ratio of shares to initial reach
        total_shares = sum(
            p.get('engagement', {}).get('shares', 0) 
            for p in performance.get('platforms', {}).values()
        )
        initial_reach = performance.get('total_reach', 1)
        
        return total_shares / max(1, initial_reach)
    
    # BUSINESS IMPACT ANALYTICS
    async def calculate_roi(
        self,
        client_id: str,
        timeframe: str
    ) -> ROIAnalysis:
        """
        Calculate return on investment for social media activities.
        
        Args:
            client_id: Client identifier
            timeframe: Analysis timeframe
            
        Returns:
            ROIAnalysis: Comprehensive ROI analysis
        """
        try:
            self.logger.info(f"Calculating ROI for client {client_id}")
            
            # Parse timeframe
            tf = TimeFrame(timeframe.lower())
            period_start, period_end = self._get_timeframe_bounds(tf)
            analysis_period = {'start': period_start, 'end': period_end}
            
            # Calculate investment
            investment_data = await self._calculate_total_investment(
                client_id, analysis_period
            )
            total_investment = investment_data['total']
            investment_breakdown = investment_data['breakdown']
            
            # Calculate revenue
            revenue_data = await self._calculate_total_revenue(
                client_id, analysis_period
            )
            total_revenue = revenue_data['total']
            revenue_breakdown = revenue_data['breakdown']
            
            # Calculate ROI percentage
            roi_percentage = ((total_revenue - total_investment) / max(1, total_investment)) * 100
            
            # Calculate profit margin
            profit_margin = (total_revenue - total_investment) / max(1, total_revenue)
            
            # Calculate acquisition metrics
            cost_per_acquisition = await self._calculate_cost_per_acquisition(
                client_id, analysis_period, total_investment
            )
            customer_lifetime_value = await self._calculate_customer_lifetime_value(
                client_id
            )
            
            # Calculate platform-specific ROI
            platform_roi = await self._calculate_platform_roi(
                client_id, analysis_period
            )
            
            # Calculate content type ROI
            content_type_roi = await self._calculate_content_type_roi(
                client_id, analysis_period
            )
            
            # Find break-even point
            break_even_point = await self._find_break_even_point(
                client_id, analysis_period
            )
            
            # Calculate payback period
            payback_period_days = await self._calculate_payback_period(
                total_investment, revenue_data['daily_revenue']
            )
            
            # Perform attribution analysis
            attribution_analysis = await self._perform_attribution_analysis(
                client_id, analysis_period
            )
            
            # Generate recommendations
            recommendations = await self._generate_roi_recommendations(
                roi_percentage, platform_roi, content_type_roi
            )
            
            roi_analysis = ROIAnalysis(
                client_id=client_id,
                timeframe=tf,
                analysis_period=analysis_period,
                total_investment=total_investment,
                investment_breakdown=investment_breakdown,
                total_revenue=total_revenue,
                revenue_breakdown=revenue_breakdown,
                roi_percentage=roi_percentage,
                profit_margin=profit_margin,
                cost_per_acquisition=cost_per_acquisition,
                customer_lifetime_value=customer_lifetime_value,
                platform_roi=platform_roi,
                content_type_roi=content_type_roi,
                break_even_point=break_even_point,
                payback_period_days=payback_period_days,
                attribution_analysis=attribution_analysis,
                recommendations=recommendations
            )
            
            return roi_analysis
            
        except Exception as e:
            self.logger.error(f"Error calculating ROI: {str(e)}")
            raise
    
    async def measure_lead_generation(
        self,
        client_id: str
    ) -> LeadMetrics:
        """
        Measure lead generation performance from social media.
        
        Args:
            client_id: Client identifier
            
        Returns:
            LeadMetrics: Lead generation analytics
        """
        try:
            self.logger.info(f"Measuring lead generation for client {client_id}")
            
            # Define measurement period (last 30 days)
            measurement_period = {
                'start': datetime.utcnow() - timedelta(days=30),
                'end': datetime.utcnow()
            }
            
            # Get lead counts
            lead_data = await self._get_lead_data(client_id, measurement_period)
            total_leads = lead_data['total']
            qualified_leads = lead_data['qualified']
            
            # Analyze lead sources
            lead_sources = await self._analyze_lead_sources(
                client_id, measurement_period
            )
            
            # Calculate lead quality score
            lead_quality_score = await self._calculate_lead_quality_score(
                client_id, measurement_period
            )
            
            # Calculate cost per lead
            total_cost = await self._get_total_marketing_cost(
                client_id, measurement_period
            )
            cost_per_lead = total_cost / max(1, total_leads)
            
            # Calculate conversion rate
            lead_conversion_rate = qualified_leads / max(1, total_leads)
            
            # Calculate lead velocity
            lead_velocity = await self._calculate_lead_velocity(
                client_id, measurement_period
            )
            
            # Calculate average nurture time
            lead_nurture_time = await self._calculate_lead_nurture_time(
                client_id, measurement_period
            )
            
            # Identify top converting content
            top_converting_content = await self._identify_top_converting_content(
                client_id, measurement_period
            )
            
            # Get lead scoring breakdown
            lead_scoring_breakdown = await self._get_lead_scoring_breakdown(
                client_id
            )
            
            lead_metrics = LeadMetrics(
                client_id=client_id,
                measurement_period=measurement_period,
                total_leads=total_leads,
                qualified_leads=qualified_leads,
                lead_sources=lead_sources,
                lead_quality_score=lead_quality_score,
                cost_per_lead=cost_per_lead,
                lead_conversion_rate=lead_conversion_rate,
                lead_velocity=lead_velocity,
                lead_nurture_time=lead_nurture_time,
                top_converting_content=top_converting_content,
                lead_scoring_breakdown=lead_scoring_breakdown
            )
            
            return lead_metrics
            
        except Exception as e:
            self.logger.error(f"Error measuring lead generation: {str(e)}")
            raise
    
    async def track_conversion_funnel(
        self,
        client_id: str
    ) -> ConversionAnalytics:
        """
        Track and analyze the conversion funnel from social media.
        
        Args:
            client_id: Client identifier
            
        Returns:
            ConversionAnalytics: Funnel analysis
        """
        try:
            self.logger.info(f"Tracking conversion funnel for client {client_id}")
            
            # Get funnel configuration
            funnel_config = await self._get_funnel_configuration(client_id)
            funnel_name = funnel_config.get('name', 'Social Media Conversion Funnel')
            
            # Track funnel stages
            funnel_stages = await self._track_funnel_stages(client_id)
            
            # Calculate total conversions
            total_conversions = funnel_stages[-1]['conversions'] if funnel_stages else 0
            
            # Calculate overall conversion rate
            initial_visitors = funnel_stages[0]['visitors'] if funnel_stages else 1
            conversion_rate = total_conversions / max(1, initial_visitors)
            
            # Analyze drop-off rates
            drop_off_analysis = await self._analyze_funnel_dropoff(funnel_stages)
            
            # Calculate time metrics
            time_to_conversion = await self._calculate_time_to_conversion(
                client_id, funnel_stages
            )
            
            # Identify conversion paths
            conversion_paths = await self._identify_conversion_paths(client_id)
            
            # Track assisted conversions by platform
            assisted_conversions = await self._track_assisted_conversions(client_id)
            
            # Identify optimization opportunities
            optimization_opportunities = await self._identify_funnel_optimizations(
                funnel_stages, drop_off_analysis
            )
            
            conversion_analytics = ConversionAnalytics(
                client_id=client_id,
                funnel_name=funnel_name,
                funnel_stages=funnel_stages,
                total_conversions=total_conversions,
                conversion_rate=conversion_rate,
                drop_off_analysis=drop_off_analysis,
                time_to_conversion=time_to_conversion,
                conversion_paths=conversion_paths,
                assisted_conversions=assisted_conversions,
                optimization_opportunities=optimization_opportunities
            )
            
            return conversion_analytics
            
        except Exception as e:
            self.logger.error(f"Error tracking conversion funnel: {str(e)}")
            raise
    
    async def calculate_social_media_attribution(
        self,
        client_id: str
    ) -> AttributionAnalysis:
        """
        Calculate attribution across social media channels.
        
        Args:
            client_id: Client identifier
            
        Returns:
            AttributionAnalysis: Multi-touch attribution analysis
        """
        try:
            self.logger.info(f"Calculating social media attribution for client {client_id}")
            
            # Determine attribution model
            attribution_model = await self._get_attribution_model(client_id)
            
            # Get conversion data with touchpoints
            conversion_data = await self._get_conversion_touchpoints(client_id)
            
            # Calculate channel attribution
            channel_attribution = await self._calculate_channel_attribution(
                conversion_data, attribution_model
            )
            
            # Calculate content attribution
            content_attribution = await self._calculate_content_attribution(
                conversion_data, attribution_model
            )
            
            # Analyze touchpoints
            touchpoint_analysis = await self._analyze_touchpoints(conversion_data)
            
            # Identify common attribution paths
            attribution_paths = await self._identify_attribution_paths(
                conversion_data
            )
            
            # Calculate cross-channel impact
            cross_channel_impact = await self._calculate_cross_channel_impact(
                conversion_data
            )
            
            # Calculate attribution confidence
            attribution_confidence = await self._calculate_attribution_confidence(
                conversion_data, attribution_model
            )
            
            attribution_analysis = AttributionAnalysis(
                client_id=client_id,
                attribution_model=attribution_model,
                channel_attribution=channel_attribution,
                content_attribution=content_attribution,
                touchpoint_analysis=touchpoint_analysis,
                attribution_paths=attribution_paths,
                cross_channel_impact=cross_channel_impact,
                attribution_confidence=attribution_confidence
            )
            
            return attribution_analysis
            
        except Exception as e:
            self.logger.error(f"Error calculating attribution: {str(e)}")
            raise

    # PLATFORM-SPECIFIC ANALYTICS
    async def get_instagram_insights(
        self,
        account_id: str
    ) -> InstagramAnalytics:
        """
        Get Instagram-specific analytics and insights.
        
        Args:
            account_id: Instagram account identifier
            
        Returns:
            InstagramAnalytics: Platform-specific analytics
        """
        try:
            self.logger.info(f"Getting Instagram insights for account {account_id}")
            
            # Define analysis period (last 30 days)
            analysis_period = {
                'start': datetime.utcnow() - timedelta(days=30),
                'end': datetime.utcnow()
            }
            
            # Get profile metrics
            profile_metrics = await self._get_instagram_profile_metrics(account_id)
            
            # Get content-specific metrics
            story_metrics = await self._get_instagram_story_metrics(
                account_id, analysis_period
            )
            reel_metrics = await self._get_instagram_reel_metrics(
                account_id, analysis_period
            )
            igtv_metrics = await self._get_instagram_igtv_metrics(
                account_id, analysis_period
            )
            
            # Analyze hashtag performance
            hashtag_performance = await self._analyze_instagram_hashtags(
                account_id, analysis_period
            )
            
            # Get mention metrics
            mention_metrics = await self._get_instagram_mention_metrics(
                account_id, analysis_period
            )
            
            # Get shopping metrics if applicable
            shopping_metrics = await self._get_instagram_shopping_metrics(
                account_id, analysis_period
            )
            
            # Get discovery metrics
            discovery_metrics = await self._get_instagram_discovery_metrics(
                account_id, analysis_period
            )
            
            instagram_analytics = InstagramAnalytics(
                account_id=account_id,
                analysis_period=analysis_period,
                profile_views=profile_metrics.get('profile_views', 0),
                website_clicks=profile_metrics.get('website_clicks', 0),
                email_clicks=profile_metrics.get('email_clicks', 0),
                story_metrics=story_metrics,
                reel_metrics=reel_metrics,
                igtv_metrics=igtv_metrics,
                hashtag_performance=hashtag_performance,
                mention_metrics=mention_metrics,
                shopping_metrics=shopping_metrics,
                discovery_metrics=discovery_metrics
            )
            
            return instagram_analytics
            
        except Exception as e:
            self.logger.error(f"Error getting Instagram insights: {str(e)}")
            raise
    
    async def get_linkedin_metrics(
        self,
        page_id: str
    ) -> LinkedInAnalytics:
        """
        Get LinkedIn-specific metrics and analytics.
        
        Args:
            page_id: LinkedIn page identifier
            
        Returns:
            LinkedInAnalytics: Platform-specific analytics
        """
        try:
            self.logger.info(f"Getting LinkedIn metrics for page {page_id}")
            
            # Define analysis period
            analysis_period = {
                'start': datetime.utcnow() - timedelta(days=30),
                'end': datetime.utcnow()
            }
            
            # Get page view metrics
            page_metrics = await self._get_linkedin_page_metrics(page_id)
            
            # Get demographic data
            follower_demographics = await self._get_linkedin_follower_demographics(
                page_id
            )
            visitor_demographics = await self._get_linkedin_visitor_demographics(
                page_id
            )
            
            # Get content metrics
            post_metrics = await self._get_linkedin_post_metrics(
                page_id, analysis_period
            )
            article_metrics = await self._get_linkedin_article_metrics(
                page_id, analysis_period
            )
            
            # Get employee advocacy metrics
            employee_advocacy = await self._get_linkedin_employee_advocacy(
                page_id, analysis_period
            )
            
            # Get competitor benchmarking
            competitor_benchmarking = await self._get_linkedin_competitor_benchmarks(
                page_id
            )
            
            linkedin_analytics = LinkedInAnalytics(
                page_id=page_id,
                analysis_period=analysis_period,
                page_views=page_metrics.get('page_views', 0),
                unique_visitors=page_metrics.get('unique_visitors', 0),
                follower_demographics=follower_demographics,
                visitor_demographics=visitor_demographics,
                post_metrics=post_metrics,
                article_metrics=article_metrics,
                employee_advocacy=employee_advocacy,
                competitor_benchmarking=competitor_benchmarking
            )
            
            return linkedin_analytics
            
        except Exception as e:
            self.logger.error(f"Error getting LinkedIn metrics: {str(e)}")
            raise
    
    async def get_tiktok_analytics(
        self,
        account_id: str
    ) -> TikTokAnalytics:
        """
        Get TikTok-specific analytics.
        
        Args:
            account_id: TikTok account identifier
            
        Returns:
            TikTokAnalytics: Platform-specific analytics
        """
        try:
            self.logger.info(f"Getting TikTok analytics for account {account_id}")
            
            # Define analysis period
            analysis_period = {
                'start': datetime.utcnow() - timedelta(days=30),
                'end': datetime.utcnow()
            }
            
            # Get view metrics
            view_metrics = await self._get_tiktok_view_metrics(account_id)
            
            # Get engagement metrics
            engagement_metrics = await self._get_tiktok_engagement_metrics(
                account_id, analysis_period
            )
            
            # Get trending metrics
            trending_metrics = await self._get_tiktok_trending_metrics(
                account_id, analysis_period
            )
            
            # Get sound performance
            sound_performance = await self._get_tiktok_sound_performance(
                account_id, analysis_period
            )
            
            # Get effect usage
            effect_usage = await self._get_tiktok_effect_usage(
                account_id, analysis_period
            )
            
            # Get duet metrics
            duet_metrics = await self._get_tiktok_duet_metrics(
                account_id, analysis_period
            )
            
            tiktok_analytics = TikTokAnalytics(
                account_id=account_id,
                analysis_period=analysis_period,
                video_views=view_metrics.get('video_views', 0),
                profile_views=view_metrics.get('profile_views', 0),
                average_watch_time=engagement_metrics.get('average_watch_time', 0.0),
                completion_rate=engagement_metrics.get('completion_rate', 0.0),
                trending_metrics=trending_metrics,
                sound_performance=sound_performance,
                effect_usage=effect_usage,
                duet_metrics=duet_metrics
            )
            
            return tiktok_analytics
            
        except Exception as e:
            self.logger.error(f"Error getting TikTok analytics: {str(e)}")
            raise
    
    async def get_youtube_analytics(
        self,
        channel_id: str
    ) -> YouTubeAnalytics:
        """
        Get YouTube-specific analytics.
        
        Args:
            channel_id: YouTube channel identifier
            
        Returns:
            YouTubeAnalytics: Platform-specific analytics
        """
        try:
            self.logger.info(f"Getting YouTube analytics for channel {channel_id}")
            
            # Define analysis period
            analysis_period = {
                'start': datetime.utcnow() - timedelta(days=30),
                'end': datetime.utcnow()
            }
            
            # Get view and watch time metrics
            view_metrics = await self._get_youtube_view_metrics(
                channel_id, analysis_period
            )
            
            # Get engagement metrics
            engagement_metrics = await self._get_youtube_engagement_metrics(
                channel_id, analysis_period
            )
            
            # Get subscriber metrics
            subscriber_metrics = await self._get_youtube_subscriber_metrics(
                channel_id, analysis_period
            )
            
            # Get revenue metrics if monetized
            revenue_metrics = await self._get_youtube_revenue_metrics(
                channel_id, analysis_period
            )
            
            # Get traffic source data
            traffic_sources = await self._get_youtube_traffic_sources(
                channel_id, analysis_period
            )
            
            # Get audience retention data
            audience_retention = await self._get_youtube_audience_retention(
                channel_id, analysis_period
            )
            
            # Get playlist performance
            playlist_performance = await self._get_youtube_playlist_performance(
                channel_id, analysis_period
            )
            
            # Get end screen metrics
            end_screen_metrics = await self._get_youtube_end_screen_metrics(
                channel_id, analysis_period
            )
            
            youtube_analytics = YouTubeAnalytics(
                channel_id=channel_id,
                analysis_period=analysis_period,
                views=view_metrics.get('views', 0),
                watch_time_hours=view_metrics.get('watch_time_hours', 0.0),
                average_view_duration=engagement_metrics.get('average_view_duration', 0.0),
                click_through_rate=engagement_metrics.get('click_through_rate', 0.0),
                subscriber_change=subscriber_metrics.get('subscriber_change', 0),
                revenue_metrics=revenue_metrics,
                traffic_sources=traffic_sources,
                audience_retention=audience_retention,
                playlist_performance=playlist_performance,
                end_screen_metrics=end_screen_metrics
            )
            
            return youtube_analytics
            
        except Exception as e:
            self.logger.error(f"Error getting YouTube analytics: {str(e)}")
            raise
    
    # UNIVERSAL BUSINESS ANALYTICS
    async def benchmark_against_industry(
        self,
        client_id: str,
        business_niche: BusinessNicheType
    ) -> BenchmarkData:
        """
        Benchmark client performance against industry standards.
        
        Args:
            client_id: Client identifier
            business_niche: Business niche type
            
        Returns:
            BenchmarkData: Industry benchmarking analysis
        """
        try:
            self.logger.info(f"Benchmarking client {client_id} against {business_niche} industry")
            
            # Get industry benchmark data
            industry_averages = await self._get_industry_benchmarks(business_niche)
            
            # Get client performance metrics
            client_performance = await self._get_client_performance_summary(client_id)
            
            # Calculate percentile rankings
            percentile_rankings = await self._calculate_percentile_rankings(
                client_performance, industry_averages
            )
            
            # Determine competitive position
            competitive_position = self._determine_competitive_position(
                percentile_rankings
            )
            
            # Identify strengths and weaknesses
            strengths = await self._identify_performance_strengths(
                client_performance, industry_averages
            )
            weaknesses = await self._identify_performance_weaknesses(
                client_performance, industry_averages
            )
            
            # Find improvement opportunities
            improvement_opportunities = await self._find_improvement_opportunities(
                client_performance, industry_averages, business_niche
            )
            
            # Get industry best practices
            best_practices = await self._get_industry_best_practices(
                business_niche, weaknesses
            )
            
            benchmark_data = BenchmarkData(
                client_id=client_id,
                business_niche=business_niche,
                industry_averages=industry_averages,
                client_performance=client_performance,
                percentile_rankings=percentile_rankings,
                competitive_position=competitive_position,
                strengths=strengths,
                weaknesses=weaknesses,
                improvement_opportunities=improvement_opportunities,
                best_practices=best_practices
            )
            
            return benchmark_data
            
        except Exception as e:
            self.logger.error(f"Error benchmarking against industry: {str(e)}")
            raise
    
    async def identify_optimal_content_types(
        self,
        client_id: str
    ) -> ContentTypeAnalysis:
        """
        Identify optimal content types for the client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            ContentTypeAnalysis: Content type optimization analysis
        """
        try:
            self.logger.info(f"Identifying optimal content types for client {client_id}")
            
            # Define analysis period
            analysis_period = {
                'start': datetime.utcnow() - timedelta(days=90),
                'end': datetime.utcnow()
            }
            
            # Analyze performance by content type
            performance_by_type = await self._analyze_content_type_performance(
                client_id, analysis_period
            )
            
            # Calculate optimal content mix
            optimal_mix = await self._calculate_optimal_content_mix(
                client_id, performance_by_type
            )
            
            # Identify trending formats
            trending_formats = await self._identify_trending_formats(
                client_id, analysis_period
            )
            
            # Identify declining formats
            declining_formats = await self._identify_declining_formats(
                client_id, analysis_period
            )
            
            # Analyze audience preferences
            audience_preferences = await self._analyze_audience_content_preferences(
                client_id
            )
            
            # Get platform recommendations
            platform_recommendations = await self._get_platform_content_recommendations(
                client_id
            )
            
            content_type_analysis = ContentTypeAnalysis(
                client_id=client_id,
                analysis_period=analysis_period,
                performance_by_type=performance_by_type,
                optimal_mix=optimal_mix,
                trending_formats=trending_formats,
                declining_formats=declining_formats,
                audience_preferences=audience_preferences,
                platform_recommendations=platform_recommendations
            )
            
            return content_type_analysis
            
        except Exception as e:
            self.logger.error(f"Error identifying optimal content types: {str(e)}")
            raise

    async def recommend_posting_optimization(
        self,
        client_id: str
    ) -> PostingRecommendations:
        """
        Generate posting schedule optimization recommendations.
        
        Args:
            client_id: Client identifier
            
        Returns:
            PostingRecommendations: Optimized posting schedule
        """
        try:
            self.logger.info(f"Generating posting recommendations for client {client_id}")
            
            # Analyze historical performance data
            performance_data = await self._get_historical_posting_performance(client_id)
            
            # Identify optimal posting times per platform
            optimal_times = await self._identify_optimal_posting_times(
                client_id, performance_data
            )
            
            # Determine optimal posting frequency
            optimal_frequency = await self._calculate_optimal_posting_frequency(
                client_id, performance_data
            )
            
            # Analyze day of week performance
            day_of_week_performance = await self._analyze_day_of_week_performance(
                client_id, performance_data
            )
            
            # Consider time zone factors
            time_zone_considerations = await self._analyze_time_zone_impact(client_id)
            
            # Generate content calendar recommendations
            content_calendar = await self._generate_content_calendar(
                client_id, optimal_times, optimal_frequency
            )
            
            # Account for seasonal adjustments
            seasonal_adjustments = await self._calculate_seasonal_adjustments(
                client_id, performance_data
            )
            
            posting_recommendations = PostingRecommendations(
                client_id=client_id,
                optimal_times=optimal_times,
                optimal_frequency=optimal_frequency,
                day_of_week_performance=day_of_week_performance,
                time_zone_considerations=time_zone_considerations,
                content_calendar=content_calendar,
                seasonal_adjustments=seasonal_adjustments
            )
            
            return posting_recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating posting recommendations: {str(e)}")
            raise
    
    async def generate_performance_insights(
        self,
        client_id: str
    ) -> InsightsList:
        """
        Generate actionable performance insights using AI analysis.
        
        Args:
            client_id: Client identifier
            
        Returns:
            InsightsList: Prioritized actionable insights
        """
        try:
            self.logger.info(f"Generating performance insights for client {client_id}")
            
            # Collect comprehensive performance data
            performance_data = await self._collect_comprehensive_performance_data(client_id)
            
            # Use AI to analyze patterns and generate insights
            ai_insights = await self._generate_ai_insights(performance_data)
            
            # Prioritize insights by potential impact
            prioritized_insights = await self._prioritize_insights(ai_insights)
            
            # Extract top priority actions
            priority_actions = [
                insight['action'] for insight in prioritized_insights[:5]
            ]
            
            # Calculate predicted impact
            predicted_impact = await self._calculate_predicted_impact(
                client_id, prioritized_insights
            )
            
            # Assess implementation difficulty
            implementation_difficulty = await self._assess_implementation_difficulty(
                prioritized_insights
            )
            
            insights_list = InsightsList(
                client_id=client_id,
                insights=prioritized_insights,
                priority_actions=priority_actions,
                predicted_impact=predicted_impact,
                implementation_difficulty=implementation_difficulty
            )
            
            return insights_list
            
        except Exception as e:
            self.logger.error(f"Error generating performance insights: {str(e)}")
            raise
    
    # REPORTING AND INSIGHTS
    async def generate_weekly_report(
        self,
        client_id: str
    ) -> WeeklyReport:
        """
        Generate comprehensive weekly performance report.
        
        Args:
            client_id: Client identifier
            
        Returns:
            WeeklyReport: Weekly performance summary
        """
        try:
            self.logger.info(f"Generating weekly report for client {client_id}")
            
            # Define report period
            week_end = datetime.utcnow()
            week_start = week_end - timedelta(days=7)
            
            # Generate executive summary
            executive_summary = await self._generate_executive_summary(
                client_id, week_start, week_end
            )
            
            # Collect key metrics
            key_metrics = await self._collect_weekly_key_metrics(
                client_id, week_start, week_end
            )
            
            # Get content performance data
            content_performance = await self._get_weekly_content_performance(
                client_id, week_start, week_end
            )
            
            # Get audience growth analytics
            audience_growth = await self.analyze_audience_growth(client_id, "week")
            
            # Identify top performing content
            top_performing_content = await self._identify_top_weekly_content(
                client_id, week_start, week_end
            )
            
            # Analyze engagement trends
            engagement_trends = await self._analyze_weekly_engagement_trends(
                client_id, week_start, week_end
            )
            
            # Generate action items
            action_items = await self._generate_weekly_action_items(
                key_metrics, engagement_trends
            )
            
            # Set next week focus areas
            next_week_focus = await self._determine_next_week_focus(
                client_id, content_performance, engagement_trends
            )
            
            weekly_report = WeeklyReport(
                client_id=client_id,
                week_start=week_start,
                week_end=week_end,
                executive_summary=executive_summary,
                key_metrics=key_metrics,
                content_performance=content_performance,
                audience_growth=audience_growth,
                top_performing_content=top_performing_content,
                engagement_trends=engagement_trends,
                action_items=action_items,
                next_week_focus=next_week_focus
            )
            
            return weekly_report
            
        except Exception as e:
            self.logger.error(f"Error generating weekly report: {str(e)}")
            raise
    
    async def create_monthly_dashboard(
        self,
        client_id: str
    ) -> MonthlyDashboard:
        """
        Create comprehensive monthly analytics dashboard.
        
        Args:
            client_id: Client identifier
            
        Returns:
            MonthlyDashboard: Monthly performance dashboard
        """
        try:
            self.logger.info(f"Creating monthly dashboard for client {client_id}")
            
            # Get current month
            now = datetime.utcnow()
            month = now.strftime("%Y-%m")
            
            # Collect overview metrics
            overview_metrics = await self._collect_monthly_overview_metrics(
                client_id, month
            )
            
            # Get platform breakdown
            platform_breakdown = await self._get_monthly_platform_breakdown(
                client_id, month
            )
            
            # Generate ROI analysis
            roi_analysis = await self.calculate_roi(client_id, "month")
            
            # Collect audience insights
            audience_insights = await self._collect_monthly_audience_insights(
                client_id, month
            )
            
            # Analyze content performance
            content_analysis = await self.identify_optimal_content_types(client_id)
            
            # Get competitive analysis
            business_niche = await self._get_client_business_niche(client_id)
            competitive_analysis = await self.benchmark_against_industry(
                client_id, business_niche
            )
            
            # Identify trends
            trends_identified = await self._identify_monthly_trends(
                client_id, month
            )
            
            # Generate strategic recommendations
            strategic_recommendations = await self._generate_strategic_recommendations(
                client_id, overview_metrics, roi_analysis, trends_identified
            )
            
            monthly_dashboard = MonthlyDashboard(
                client_id=client_id,
                month=month,
                overview_metrics=overview_metrics,
                platform_breakdown=platform_breakdown,
                roi_analysis=roi_analysis,
                audience_insights=audience_insights,
                content_analysis=content_analysis,
                competitive_analysis=competitive_analysis,
                trends_identified=trends_identified,
                strategic_recommendations=strategic_recommendations
            )
            
            return monthly_dashboard
            
        except Exception as e:
            self.logger.error(f"Error creating monthly dashboard: {str(e)}")
            raise
    
    async def export_analytics_data(
        self,
        client_id: str,
        format: str
    ) -> ExportResult:
        """
        Export analytics data in specified format.
        
        Args:
            client_id: Client identifier
            format: Export format (csv, json, xlsx, pdf)
            
        Returns:
            ExportResult: Export details and download information
        """
        try:
            self.logger.info(f"Exporting analytics data for client {client_id} in {format}")
            
            # Validate format
            supported_formats = ['csv', 'json', 'xlsx', 'pdf']
            if format.lower() not in supported_formats:
                raise ValueError(f"Unsupported format: {format}")
            
            # Generate export ID
            export_id = f"export_{client_id}_{datetime.utcnow().timestamp()}"
            
            # Collect all analytics data
            analytics_data = await self._collect_all_analytics_data(client_id)
            
            # Export based on format
            if format.lower() == 'csv':
                export_result = await self._export_to_csv(export_id, analytics_data)
            elif format.lower() == 'json':
                export_result = await self._export_to_json(export_id, analytics_data)
            elif format.lower() == 'xlsx':
                export_result = await self._export_to_excel(export_id, analytics_data)
            else:  # pdf
                export_result = await self._export_to_pdf(export_id, analytics_data)
            
            # Generate download URL
            download_url = await self._generate_download_url(export_id, format)
            
            # Set expiry (7 days)
            expiry_timestamp = datetime.utcnow() + timedelta(days=7)
            
            export_result = ExportResult(
                export_id=export_id,
                format=format,
                file_path=export_result['file_path'],
                download_url=download_url,
                rows_exported=export_result['rows_exported'],
                file_size_bytes=export_result['file_size'],
                expiry_timestamp=expiry_timestamp
            )
            
            return export_result
            
        except Exception as e:
            self.logger.error(f"Error exporting analytics data: {str(e)}")
            raise
    
    async def schedule_automated_reports(
        self,
        client_id: str,
        frequency: str
    ) -> ScheduleResult:
        """
        Schedule automated report generation and delivery.
        
        Args:
            client_id: Client identifier
            frequency: Report frequency (daily, weekly, monthly)
            
        Returns:
            ScheduleResult: Schedule confirmation and details
        """
        try:
            self.logger.info(f"Scheduling {frequency} reports for client {client_id}")
            
            # Validate frequency
            valid_frequencies = ['daily', 'weekly', 'monthly']
            if frequency.lower() not in valid_frequencies:
                raise ValueError(f"Invalid frequency: {frequency}")
            
            # Generate schedule ID
            schedule_id = f"schedule_{client_id}_{frequency}_{datetime.utcnow().timestamp()}"
            
            # Determine report type based on frequency
            report_type = self._determine_report_type(frequency)
            
            # Calculate next run time
            next_run = self._calculate_next_run_time(frequency)
            
            # Get recipient list
            recipients = await self._get_report_recipients(client_id)
            
            # Create schedule in database
            await self._create_report_schedule(
                schedule_id, client_id, report_type, frequency, next_run, recipients
            )
            
            schedule_result = ScheduleResult(
                schedule_id=schedule_id,
                client_id=client_id,
                report_type=report_type,
                frequency=frequency,
                next_run=next_run,
                recipients=recipients,
                status="active"
            )
            
            return schedule_result
            
        except Exception as e:
            self.logger.error(f"Error scheduling automated reports: {str(e)}")
            raise
    
    # Timeframe helper method
    def _get_timeframe_bounds(self, timeframe: TimeFrame) -> Tuple[datetime, datetime]:
        """Get start and end dates for a timeframe"""
        end = datetime.utcnow()
        
        if timeframe == TimeFrame.HOUR:
            start = end - timedelta(hours=1)
        elif timeframe == TimeFrame.DAY:
            start = end - timedelta(days=1)
        elif timeframe == TimeFrame.WEEK:
            start = end - timedelta(weeks=1)
        elif timeframe == TimeFrame.MONTH:
            start = end - timedelta(days=30)
        elif timeframe == TimeFrame.QUARTER:
            start = end - timedelta(days=90)
        elif timeframe == TimeFrame.YEAR:
            start = end - timedelta(days=365)
        else:
            start = end - timedelta(days=30)  # Default to month
            
        return start, end