"""Usage Analytics Engine - Comprehensive usage tracking with business intelligence"""

from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import asyncio
import json
import logging
from dataclasses import dataclass
from sqlalchemy import select, func, and_, or_

from .base_intelligence import (
    UniversalIntelligenceEngine,
    AnalyticsTimeframe,
    BusinessMetricType,
    IntelligenceInsight,
    IntelligenceEngineError
)
from ..models.content_models import Content, ContentEngagement
from ..database.connection import get_db_connection

logger = logging.getLogger(__name__)

@dataclass
class PlatformUsageStats:
    """Platform-specific usage statistics"""
    platform: str
    total_posts: int
    successful_posts: int
    failed_posts: int
    success_rate: float
    average_engagement: float
    total_reach: int
    revenue_generated: float
    optimization_time_minutes: float
    automation_savings_hours: float
    peak_usage_hours: List[int]
    content_type_breakdown: Dict[str, int]
    business_niche_performance: Dict[str, float]

@dataclass
class FeatureUsageStats:
    """Feature-specific usage statistics"""
    feature: str
    usage_count: int
    unique_users: int
    average_session_duration_minutes: float
    success_rate: float
    user_satisfaction_score: float
    feature_adoption_rate: float
    revenue_correlation_score: float
    most_common_use_cases: List[str]
    business_niche_usage: Dict[str, int]

class UsageTracker:
    """Track usage metrics across the platform"""
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.db = get_db_connection()
        
    async def get_platform_stats(self, platform: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get comprehensive platform usage statistics"""
        try:
            # Query content data for the platform
            content_query = select(Content).where(
                and_(
                    Content.client_id == self.client_id,
                    Content.platform == platform,
                    Content.created_at >= start_date,
                    Content.created_at <= end_date
                )
            )
            
            contents = await self.db.fetch_all(content_query)
            
            # Calculate platform statistics
            total_posts = len(contents)
            successful_posts = sum(1 for c in contents if c.get('status') == 'published')
            failed_posts = sum(1 for c in contents if c.get('status') == 'failed')
            
            # Get engagement data
            engagement_query = select(ContentEngagement).where(
                and_(
                    ContentEngagement.client_id == self.client_id,
                    ContentEngagement.platform == platform,
                    ContentEngagement.recorded_at >= start_date,
                    ContentEngagement.recorded_at <= end_date
                )
            )
            
            engagements = await self.db.fetch_all(engagement_query)
            
            # Calculate engagement metrics
            total_engagement = sum(e.get('total_engagement', 0) for e in engagements)
            total_reach = sum(e.get('reach', 0) for e in engagements)
            revenue_generated = sum(e.get('attributed_revenue', 0) for e in engagements)
            
            # Calculate time spent optimizing
            optimization_time = sum(c.get('optimization_time_seconds', 0) for c in contents) / 60
            
            # Calculate automation savings
            manual_time_estimate = total_posts * 30  # 30 minutes per post manual
            automation_savings = (manual_time_estimate - optimization_time) / 60
            
            # Analyze peak usage hours
            posting_hours = [c.get('created_at').hour for c in contents if c.get('created_at')]
            hour_counter = Counter(posting_hours)
            peak_hours = [hour for hour, count in hour_counter.most_common(3)]
            
            # Content type breakdown
            content_types = [c.get('content_type', 'unknown') for c in contents]
            content_type_breakdown = dict(Counter(content_types))
            
            # Business niche performance
            niche_revenue = defaultdict(float)
            for engagement in engagements:
                niche = engagement.get('business_niche', 'unknown')
                niche_revenue[niche] += engagement.get('attributed_revenue', 0)
            
            return {
                'total_posts': total_posts,
                'successful_posts': successful_posts,
                'failed_posts': failed_posts,
                'success_rate': (successful_posts / total_posts * 100) if total_posts > 0 else 0,
                'average_engagement': (total_engagement / total_posts) if total_posts > 0 else 0,
                'total_reach': total_reach,
                'revenue_generated': revenue_generated,
                'optimization_time': optimization_time,
                'automation_savings': automation_savings,
                'peak_usage_hours': peak_hours,
                'content_type_breakdown': content_type_breakdown,
                'business_niche_performance': dict(niche_revenue)
            }
            
        except Exception as e:
            logger.error(f"Error getting platform stats: {str(e)}")
            return {}
    
    async def get_feature_stats(self, feature: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get comprehensive feature usage statistics"""
        try:
            # Query feature usage data
            feature_usage_query = f"""
                SELECT 
                    COUNT(*) as usage_count,
                    COUNT(DISTINCT user_id) as unique_users,
                    AVG(session_duration_seconds) / 60 as avg_session_duration,
                    AVG(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as success_rate,
                    AVG(satisfaction_score) as satisfaction_score
                FROM feature_usage_logs
                WHERE client_id = :client_id
                    AND feature_name = :feature
                    AND timestamp >= :start_date
                    AND timestamp <= :end_date
            """
            
            result = await self.db.fetch_one(
                feature_usage_query,
                {
                    'client_id': self.client_id,
                    'feature': feature,
                    'start_date': start_date,
                    'end_date': end_date
                }
            )
            
            # Calculate feature adoption rate
            total_users_query = """
                SELECT COUNT(DISTINCT user_id) as total_users
                FROM user_activities
                WHERE client_id = :client_id
                    AND timestamp >= :start_date
                    AND timestamp <= :end_date
            """
            
            total_users_result = await self.db.fetch_one(
                total_users_query,
                {
                    'client_id': self.client_id,
                    'start_date': start_date,
                    'end_date': end_date
                }
            )
            
            total_users = total_users_result.get('total_users', 1) if total_users_result else 1
            unique_feature_users = result.get('unique_users', 0) if result else 0
            adoption_rate = (unique_feature_users / total_users * 100) if total_users > 0 else 0
            
            # Get common use cases
            use_cases_query = """
                SELECT use_case, COUNT(*) as count
                FROM feature_usage_logs
                WHERE client_id = :client_id
                    AND feature_name = :feature
                    AND timestamp >= :start_date
                    AND timestamp <= :end_date
                GROUP BY use_case
                ORDER BY count DESC
                LIMIT 5
            """
            
            use_cases = await self.db.fetch_all(
                use_cases_query,
                {
                    'client_id': self.client_id,
                    'feature': feature,
                    'start_date': start_date,
                    'end_date': end_date
                }
            )
            
            common_use_cases = [uc.get('use_case') for uc in use_cases if uc.get('use_case')]
            
            # Get business niche usage
            niche_usage_query = """
                SELECT business_niche, COUNT(*) as usage_count
                FROM feature_usage_logs
                WHERE client_id = :client_id
                    AND feature_name = :feature
                    AND timestamp >= :start_date
                    AND timestamp <= :end_date
                GROUP BY business_niche
            """
            
            niche_usage = await self.db.fetch_all(
                niche_usage_query,
                {
                    'client_id': self.client_id,
                    'feature': feature,
                    'start_date': start_date,
                    'end_date': end_date
                }
            )
            
            business_niche_usage = {nu.get('business_niche'): nu.get('usage_count', 0) for nu in niche_usage}
            
            return {
                'usage_count': result.get('usage_count', 0) if result else 0,
                'unique_users': unique_feature_users,
                'avg_session_duration': result.get('avg_session_duration', 0) if result else 0,
                'success_rate': result.get('success_rate', 0) if result else 0,
                'satisfaction_score': result.get('satisfaction_score', 0) if result else 0,
                'adoption_rate': adoption_rate,
                'common_use_cases': common_use_cases,
                'business_niche_usage': business_niche_usage
            }
            
        except Exception as e:
            logger.error(f"Error getting feature stats: {str(e)}")
            return {}

class UsagePatternAnalyzer:
    """Analyze usage patterns for insights"""
    
    async def analyze_cross_platform_patterns(self, usage_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze patterns across platforms"""
        patterns = []
        
        # Find platforms with similar usage patterns
        platform_data = usage_data.get('platform_usage', {})
        
        # Group platforms by engagement levels
        high_engagement_platforms = []
        medium_engagement_platforms = []
        low_engagement_platforms = []
        
        for platform, data in platform_data.items():
            avg_engagement = data.get('average_engagement', 0)
            if avg_engagement > 1000:
                high_engagement_platforms.append(platform)
            elif avg_engagement > 100:
                medium_engagement_platforms.append(platform)
            else:
                low_engagement_platforms.append(platform)
        
        # Create pattern insights
        if high_engagement_platforms:
            patterns.append({
                'pattern_type': 'high_engagement_cluster',
                'platforms': high_engagement_platforms,
                'insight': f"Platforms {', '.join(high_engagement_platforms)} show consistently high engagement",
                'recommendation': "Focus content strategy on these high-performing platforms"
            })
        
        # Analyze time-based patterns
        peak_hours_by_platform = {}
        for platform, data in platform_data.items():
            peak_hours = data.get('peak_usage_hours', [])
            if peak_hours:
                peak_hours_by_platform[platform] = peak_hours
        
        # Find common peak hours across platforms
        all_peak_hours = []
        for hours in peak_hours_by_platform.values():
            all_peak_hours.extend(hours)
        
        hour_counter = Counter(all_peak_hours)
        most_common_hours = [hour for hour, count in hour_counter.most_common(3) if count >= 2]
        
        if most_common_hours:
            patterns.append({
                'pattern_type': 'temporal_alignment',
                'peak_hours': most_common_hours,
                'insight': f"Multiple platforms show peak engagement at {most_common_hours}",
                'recommendation': "Schedule important posts during these universal peak hours"
            })
        
        return patterns
    
    async def identify_feature_correlation_patterns(self, feature_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify correlations between feature usage"""
        correlations = []
        
        # Analyze which features are commonly used together
        feature_usage = feature_data
        
        # Find features with high adoption rates
        high_adoption_features = []
        for feature, data in feature_usage.items():
            if data.get('adoption_rate', 0) > 70:
                high_adoption_features.append(feature)
        
        if high_adoption_features:
            correlations.append({
                'correlation_type': 'high_adoption_cluster',
                'features': high_adoption_features,
                'insight': f"Features {', '.join(high_adoption_features)} have high adoption rates",
                'implication': "These are core features providing the most value to users"
            })
        
        return correlations

class UsageAnalyticsEngine(UniversalIntelligenceEngine):
    """Comprehensive usage analytics with business intelligence"""
    
    def __init__(self, client_id: str):
        super().__init__(client_id, "usage_analytics")
        self.usage_tracker = UsageTracker(client_id)
        self.pattern_analyzer = UsagePatternAnalyzer()
        
    async def collect_data(self, timeframe: AnalyticsTimeframe) -> Dict[str, Any]:
        """Collect comprehensive usage data across all platforms and features"""
        end_date = datetime.now()
        start_date = await self.get_timeframe_start_date(end_date, timeframe)
        
        # Platform usage data
        platform_usage = await self.collect_platform_usage_data(start_date, end_date)
        
        # Feature usage data
        feature_usage = await self.collect_feature_usage_data(start_date, end_date)
        
        # Content generation data
        content_generation = await self.collect_content_generation_data(start_date, end_date)
        
        # User behavior data
        user_behavior = await self.collect_user_behavior_data(start_date, end_date)
        
        # Performance correlation data
        performance_data = await self.collect_performance_correlation_data(start_date, end_date)
        
        return {
            'timeframe': timeframe.value,
            'date_range': {'start': start_date, 'end': end_date},
            'platform_usage': platform_usage,
            'feature_usage': feature_usage,
            'content_generation': content_generation,
            'user_behavior': user_behavior,
            'performance_data': performance_data,
            'metadata': {
                'total_sessions': await self.count_total_sessions(start_date, end_date),
                'unique_features_used': await self.count_unique_features_used(start_date, end_date),
                'automation_efficiency': await self.calculate_automation_efficiency(start_date, end_date)
            }
        }
    
    async def collect_platform_usage_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Collect detailed platform usage analytics"""
        platforms = ['youtube', 'linkedin', 'tiktok', 'twitter', 'facebook', 'instagram', 'pinterest']
        
        platform_data = {}
        
        for platform in platforms:
            platform_stats = await self.usage_tracker.get_platform_stats(
                platform=platform,
                start_date=start_date,
                end_date=end_date
            )
            
            platform_data[platform] = {
                'total_posts': platform_stats.get('total_posts', 0),
                'successful_posts': platform_stats.get('successful_posts', 0),
                'failed_posts': platform_stats.get('failed_posts', 0),
                'success_rate': platform_stats.get('success_rate', 0.0),
                'average_engagement': platform_stats.get('average_engagement', 0),
                'total_reach': platform_stats.get('total_reach', 0),
                'revenue_generated': platform_stats.get('revenue_generated', 0.0),
                'time_spent_optimizing': platform_stats.get('optimization_time', 0),
                'automation_savings_hours': platform_stats.get('automation_savings', 0.0),
                'peak_usage_hours': await self.get_peak_usage_hours(platform, start_date, end_date),
                'content_type_breakdown': await self.get_content_type_breakdown(platform, start_date, end_date),
                'business_niche_performance': await self.get_niche_performance(platform, start_date, end_date)
            }
        
        return platform_data
    
    async def collect_feature_usage_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Collect detailed feature usage analytics"""
        features = [
            'content_analysis', 'persona_generation', 'viral_optimization',
            'hashtag_generation', 'image_creation', 'video_editing',
            'scheduling', 'analytics_dashboard', 'revenue_tracking',
            'ai_suggestions', 'cross_platform_posting', 'audience_analysis'
        ]
        
        feature_data = {}
        
        for feature in features:
            feature_stats = await self.usage_tracker.get_feature_stats(
                feature=feature,
                start_date=start_date,
                end_date=end_date
            )
            
            feature_data[feature] = {
                'usage_count': feature_stats.get('usage_count', 0),
                'unique_users': feature_stats.get('unique_users', 0),
                'average_session_duration': feature_stats.get('avg_session_duration', 0.0),
                'success_rate': feature_stats.get('success_rate', 0.0),
                'user_satisfaction_score': feature_stats.get('satisfaction_score', 0.0),
                'feature_adoption_rate': feature_stats.get('adoption_rate', 0.0),
                'revenue_correlation': await self.calculate_feature_revenue_correlation(feature, start_date, end_date),
                'most_common_use_cases': await self.get_common_use_cases(feature, start_date, end_date),
                'business_niche_usage': await self.get_feature_niche_usage(feature, start_date, end_date)
            }
        
        return feature_data
    
    async def collect_content_generation_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Collect content generation statistics"""
        try:
            db = get_db_connection()
            
            # Query content generation metrics
            content_query = f"""
                SELECT 
                    COUNT(*) as total_content_generated,
                    AVG(generation_time_seconds) as avg_generation_time,
                    COUNT(DISTINCT content_type) as content_type_variety,
                    SUM(CASE WHEN ai_generated = true THEN 1 ELSE 0 END) as ai_generated_count,
                    AVG(quality_score) as avg_quality_score
                FROM content
                WHERE client_id = :client_id
                    AND created_at >= :start_date
                    AND created_at <= :end_date
            """
            
            result = await db.fetch_one(
                content_query,
                {
                    'client_id': self.client_id,
                    'start_date': start_date,
                    'end_date': end_date
                }
            )
            
            return {
                'total_content_generated': result.get('total_content_generated', 0) if result else 0,
                'average_generation_time_seconds': result.get('avg_generation_time', 0) if result else 0,
                'content_type_variety': result.get('content_type_variety', 0) if result else 0,
                'ai_generated_percentage': (result.get('ai_generated_count', 0) / max(result.get('total_content_generated', 1), 1) * 100) if result else 0,
                'average_quality_score': result.get('avg_quality_score', 0) if result else 0
            }
            
        except Exception as e:
            logger.error(f"Error collecting content generation data: {str(e)}")
            return {}
    
    async def collect_user_behavior_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Collect user behavior analytics"""
        try:
            db = get_db_connection()
            
            # Query user behavior metrics
            behavior_query = f"""
                SELECT 
                    COUNT(DISTINCT session_id) as total_sessions,
                    AVG(session_duration_seconds) / 60 as avg_session_duration_minutes,
                    COUNT(DISTINCT user_action) as action_variety,
                    AVG(actions_per_session) as avg_actions_per_session
                FROM user_sessions
                WHERE client_id = :client_id
                    AND session_start >= :start_date
                    AND session_start <= :end_date
            """
            
            result = await db.fetch_one(
                behavior_query,
                {
                    'client_id': self.client_id,
                    'start_date': start_date,
                    'end_date': end_date
                }
            )
            
            # Get user journey patterns
            journey_patterns = await self.analyze_user_journey_patterns(start_date, end_date)
            
            return {
                'total_sessions': result.get('total_sessions', 0) if result else 0,
                'average_session_duration_minutes': result.get('avg_session_duration_minutes', 0) if result else 0,
                'action_variety': result.get('action_variety', 0) if result else 0,
                'average_actions_per_session': result.get('avg_actions_per_session', 0) if result else 0,
                'user_journey_patterns': journey_patterns
            }
            
        except Exception as e:
            logger.error(f"Error collecting user behavior data: {str(e)}")
            return {}
    
    async def collect_performance_correlation_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Collect data correlating usage with performance"""
        try:
            # Correlate feature usage with revenue
            feature_revenue_correlation = await self.calculate_feature_revenue_correlations(start_date, end_date)
            
            # Correlate platform usage with engagement
            platform_engagement_correlation = await self.calculate_platform_engagement_correlations(start_date, end_date)
            
            # Correlate content types with viral success
            content_viral_correlation = await self.calculate_content_viral_correlations(start_date, end_date)
            
            return {
                'feature_revenue_correlation': feature_revenue_correlation,
                'platform_engagement_correlation': platform_engagement_correlation,
                'content_viral_correlation': content_viral_correlation
            }
            
        except Exception as e:
            logger.error(f"Error collecting performance correlation data: {str(e)}")
            return {}
    
    async def analyze_data(self, data: Dict[str, Any]) -> List[IntelligenceInsight]:
        """Analyze usage data for actionable insights"""
        insights = []
        
        # Platform performance insights
        platform_insights = await self.analyze_platform_performance(data['platform_usage'])
        insights.extend(platform_insights)
        
        # Feature adoption insights
        feature_insights = await self.analyze_feature_adoption(data['feature_usage'])
        insights.extend(feature_insights)
        
        # Usage pattern insights
        pattern_insights = await self.analyze_usage_patterns(data['user_behavior'])
        insights.extend(pattern_insights)
        
        # Efficiency insights
        efficiency_insights = await self.analyze_efficiency_metrics(data)
        insights.extend(efficiency_insights)
        
        # Revenue correlation insights
        revenue_insights = await self.analyze_revenue_correlations(data)
        insights.extend(revenue_insights)
        
        return insights
    
    async def analyze_platform_performance(self, platform_data: Dict[str, Any]) -> List[IntelligenceInsight]:
        """Analyze platform performance for insights"""
        insights = []
        
        # Find best performing platform
        platform_revenues = {platform: data['revenue_generated'] for platform, data in platform_data.items()}
        best_platform = max(platform_revenues.items(), key=lambda x: x[1])[0]
        worst_platform = min(platform_revenues.items(), key=lambda x: x[1])[0]
        
        if platform_revenues[best_platform] > 0:
            insights.append(IntelligenceInsight(
                metric_type=BusinessMetricType.REVENUE,
                insight_text=f"{best_platform.title()} is your highest revenue-generating platform with ${platform_revenues[best_platform]:.2f} generated",
                confidence_score=0.95,
                impact_level="high",
                actionable_recommendations=[
                    f"Increase posting frequency on {best_platform.title()} by 50%",
                    f"Allocate more content creation budget to {best_platform.title()}-optimized content",
                    f"Analyze successful {best_platform.title()} content patterns for replication"
                ],
                supporting_data={
                    'revenue_comparison': platform_revenues,
                    'engagement_rates': {platform: data['average_engagement'] for platform, data in platform_data.items()},
                    'success_rates': {platform: data['success_rate'] for platform, data in platform_data.items()}
                }
            ))
        
        # Identify underperforming platforms
        if platform_revenues[worst_platform] < platform_revenues[best_platform] * 0.1:
            insights.append(IntelligenceInsight(
                metric_type=BusinessMetricType.EFFICIENCY,
                insight_text=f"{worst_platform.title()} is underperforming with only ${platform_revenues[worst_platform]:.2f} revenue",
                confidence_score=0.88,
                impact_level="medium",
                actionable_recommendations=[
                    f"Review and optimize {worst_platform.title()} content strategy",
                    f"Consider pausing {worst_platform.title()} to focus on better-performing platforms",
                    f"Analyze {worst_platform.title()} audience demographics for better targeting"
                ],
                supporting_data={
                    'performance_gap': platform_revenues[best_platform] - platform_revenues[worst_platform],
                    'platform_details': platform_data[worst_platform]
                }
            ))
        
        # Analyze automation efficiency
        total_automation_savings = sum(data['automation_savings_hours'] for data in platform_data.values())
        if total_automation_savings > 100:
            insights.append(IntelligenceInsight(
                metric_type=BusinessMetricType.EFFICIENCY,
                insight_text=f"Automation saved {total_automation_savings:.1f} hours across all platforms",
                confidence_score=0.92,
                impact_level="high",
                actionable_recommendations=[
                    "Expand automation to cover more content types",
                    "Invest saved time in strategic planning and optimization",
                    "Calculate ROI of automation features for business case"
                ],
                supporting_data={
                    'savings_by_platform': {platform: data['automation_savings_hours'] for platform, data in platform_data.items()},
                    'monetary_value': total_automation_savings * 50  # $50/hour estimate
                }
            ))
        
        return insights
    
    async def analyze_feature_adoption(self, feature_data: Dict[str, Any]) -> List[IntelligenceInsight]:
        """Analyze feature adoption patterns"""
        insights = []
        
        # Identify high-value features
        revenue_correlations = {feature: data['revenue_correlation'] for feature, data in feature_data.items()}
        high_value_features = [f for f, corr in revenue_correlations.items() if corr > 0.7]
        
        if high_value_features:
            insights.append(IntelligenceInsight(
                metric_type=BusinessMetricType.REVENUE,
                insight_text=f"Features {', '.join(high_value_features[:3])} show strong correlation with revenue",
                confidence_score=0.85,
                impact_level="high",
                actionable_recommendations=[
                    f"Promote usage of {high_value_features[0]} to all users",
                    "Create tutorials highlighting revenue-driving features",
                    "Bundle high-value features in premium tiers"
                ],
                supporting_data={
                    'revenue_correlations': revenue_correlations,
                    'feature_details': {f: feature_data[f] for f in high_value_features[:3]}
                }
            ))
        
        # Identify underutilized features
        adoption_rates = {feature: data['feature_adoption_rate'] for feature, data in feature_data.items()}
        underutilized = [f for f, rate in adoption_rates.items() if rate < 20]
        
        if underutilized:
            insights.append(IntelligenceInsight(
                metric_type=BusinessMetricType.ENGAGEMENT,
                insight_text=f"Features {', '.join(underutilized[:3])} have low adoption rates",
                confidence_score=0.82,
                impact_level="medium",
                actionable_recommendations=[
                    "Improve feature discovery through better UI/UX",
                    "Create onboarding flows highlighting underutilized features",
                    "A/B test different feature promotion strategies"
                ],
                supporting_data={
                    'adoption_rates': adoption_rates,
                    'potential_users': sum(feature_data[f]['unique_users'] for f in underutilized)
                }
            ))
        
        return insights
    
    async def analyze_usage_patterns(self, behavior_data: Dict[str, Any]) -> List[IntelligenceInsight]:
        """Analyze user behavior patterns"""
        insights = []
        
        # Session duration insights
        avg_session_duration = behavior_data.get('average_session_duration_minutes', 0)
        if avg_session_duration < 5:
            insights.append(IntelligenceInsight(
                metric_type=BusinessMetricType.ENGAGEMENT,
                insight_text=f"Average session duration is only {avg_session_duration:.1f} minutes",
                confidence_score=0.78,
                impact_level="medium",
                actionable_recommendations=[
                    "Simplify workflows to reduce task completion time",
                    "Add engagement features to increase session duration",
                    "Analyze drop-off points in user journeys"
                ],
                supporting_data=behavior_data
            ))
        
        # User journey patterns
        journey_patterns = behavior_data.get('user_journey_patterns', [])
        if journey_patterns:
            most_common_journey = journey_patterns[0] if journey_patterns else None
            if most_common_journey:
                insights.append(IntelligenceInsight(
                    metric_type=BusinessMetricType.EFFICIENCY,
                    insight_text=f"Most common user journey: {most_common_journey.get('pattern', 'Unknown')}",
                    confidence_score=0.83,
                    impact_level="low",
                    actionable_recommendations=[
                        "Optimize the most common user flow for efficiency",
                        "Create shortcuts for frequent action sequences",
                        "Design features around natural usage patterns"
                    ],
                    supporting_data={'journey_patterns': journey_patterns[:5]}
                ))
        
        return insights
    
    async def analyze_efficiency_metrics(self, data: Dict[str, Any]) -> List[IntelligenceInsight]:
        """Analyze efficiency metrics"""
        insights = []
        
        automation_efficiency = data.get('metadata', {}).get('automation_efficiency', 0)
        if automation_efficiency > 80:
            insights.append(IntelligenceInsight(
                metric_type=BusinessMetricType.EFFICIENCY,
                insight_text=f"Automation efficiency at {automation_efficiency:.1f}% - excellent performance",
                confidence_score=0.91,
                impact_level="high",
                actionable_recommendations=[
                    "Document automation best practices for team training",
                    "Explore additional automation opportunities",
                    "Calculate and showcase automation ROI to stakeholders"
                ],
                supporting_data={'automation_metrics': data.get('metadata', {})}
            ))
        
        return insights
    
    async def analyze_revenue_correlations(self, data: Dict[str, Any]) -> List[IntelligenceInsight]:
        """Analyze correlations between usage and revenue"""
        insights = []
        
        performance_data = data.get('performance_data', {})
        feature_revenue_corr = performance_data.get('feature_revenue_correlation', {})
        
        # Find strongest revenue drivers
        if feature_revenue_corr:
            strongest_driver = max(feature_revenue_corr.items(), key=lambda x: x[1])[0]
            correlation_value = feature_revenue_corr[strongest_driver]
            
            if correlation_value > 0.8:
                insights.append(IntelligenceInsight(
                    metric_type=BusinessMetricType.REVENUE,
                    insight_text=f"{strongest_driver} usage strongly correlates with revenue (r={correlation_value:.2f})",
                    confidence_score=0.89,
                    impact_level="high",
                    actionable_recommendations=[
                        f"Make {strongest_driver} a core part of user onboarding",
                        f"Highlight {strongest_driver} in marketing materials",
                        f"Develop advanced features around {strongest_driver}"
                    ],
                    supporting_data={'correlations': feature_revenue_corr}
                ))
        
        return insights
    
    async def generate_recommendations(self, insights: List[IntelligenceInsight]) -> List[str]:
        """Generate comprehensive usage optimization recommendations"""
        recommendations = []
        
        # Extract all recommendations from insights
        for insight in insights:
            recommendations.extend(insight.actionable_recommendations)
        
        # Add strategic usage recommendations
        strategic_recommendations = await self.generate_strategic_usage_recommendations(insights)
        recommendations.extend(strategic_recommendations)
        
        # Add efficiency improvement recommendations
        efficiency_recommendations = await self.generate_efficiency_recommendations(insights)
        recommendations.extend(efficiency_recommendations)
        
        # Remove duplicates and prioritize
        unique_recommendations = list(set(recommendations))
        prioritized_recommendations = await self.prioritize_recommendations(unique_recommendations, insights)
        
        return prioritized_recommendations
    
    async def generate_strategic_usage_recommendations(self, insights: List[IntelligenceInsight]) -> List[str]:
        """Generate strategic recommendations based on usage patterns"""
        recommendations = []
        
        # Analyze insight patterns
        high_impact_insights = [i for i in insights if i.impact_level == "high"]
        
        if len(high_impact_insights) > 3:
            recommendations.append("Create a strategic roadmap focusing on high-impact optimizations")
            recommendations.append("Establish KPIs based on identified success patterns")
        
        # Revenue-focused recommendations
        revenue_insights = [i for i in insights if i.metric_type == BusinessMetricType.REVENUE]
        if revenue_insights:
            recommendations.append("Implement revenue attribution tracking for all features")
            recommendations.append("Create revenue dashboards for real-time monitoring")
        
        return recommendations
    
    async def generate_efficiency_recommendations(self, insights: List[IntelligenceInsight]) -> List[str]:
        """Generate efficiency improvement recommendations"""
        recommendations = []
        
        efficiency_insights = [i for i in insights if i.metric_type == BusinessMetricType.EFFICIENCY]
        
        if efficiency_insights:
            recommendations.append("Automate repetitive tasks identified in usage patterns")
            recommendations.append("Implement batch processing for common operations")
            recommendations.append("Create workflow templates based on successful patterns")
        
        return recommendations
    
    # Helper methods for data collection
    async def get_peak_usage_hours(self, platform: str, start_date: datetime, end_date: datetime) -> List[int]:
        """Get peak usage hours for a platform"""
        try:
            db = get_db_connection()
            query = """
                SELECT EXTRACT(HOUR FROM created_at) as hour, COUNT(*) as count
                FROM content
                WHERE client_id = :client_id
                    AND platform = :platform
                    AND created_at >= :start_date
                    AND created_at <= :end_date
                GROUP BY hour
                ORDER BY count DESC
                LIMIT 3
            """
            
            results = await db.fetch_all(
                query,
                {
                    'client_id': self.client_id,
                    'platform': platform,
                    'start_date': start_date,
                    'end_date': end_date
                }
            )
            
            return [int(r['hour']) for r in results]
            
        except Exception as e:
            logger.error(f"Error getting peak usage hours: {str(e)}")
            return []
    
    async def get_content_type_breakdown(self, platform: str, start_date: datetime, end_date: datetime) -> Dict[str, int]:
        """Get content type breakdown for a platform"""
        try:
            db = get_db_connection()
            query = """
                SELECT content_type, COUNT(*) as count
                FROM content
                WHERE client_id = :client_id
                    AND platform = :platform
                    AND created_at >= :start_date
                    AND created_at <= :end_date
                GROUP BY content_type
            """
            
            results = await db.fetch_all(
                query,
                {
                    'client_id': self.client_id,
                    'platform': platform,
                    'start_date': start_date,
                    'end_date': end_date
                }
            )
            
            return {r['content_type']: r['count'] for r in results}
            
        except Exception as e:
            logger.error(f"Error getting content type breakdown: {str(e)}")
            return {}
    
    async def get_niche_performance(self, platform: str, start_date: datetime, end_date: datetime) -> Dict[str, float]:
        """Get business niche performance for a platform"""
        try:
            db = get_db_connection()
            query = """
                SELECT business_niche, SUM(attributed_revenue) as total_revenue
                FROM content_engagement
                WHERE client_id = :client_id
                    AND platform = :platform
                    AND recorded_at >= :start_date
                    AND recorded_at <= :end_date
                GROUP BY business_niche
            """
            
            results = await db.fetch_all(
                query,
                {
                    'client_id': self.client_id,
                    'platform': platform,
                    'start_date': start_date,
                    'end_date': end_date
                }
            )
            
            return {r['business_niche']: float(r['total_revenue']) for r in results}
            
        except Exception as e:
            logger.error(f"Error getting niche performance: {str(e)}")
            return {}
    
    async def calculate_feature_revenue_correlation(self, feature: str, start_date: datetime, end_date: datetime) -> float:
        """Calculate correlation between feature usage and revenue"""
        try:
            # This would use actual statistical correlation calculation
            # For now, return a placeholder
            return 0.75
            
        except Exception as e:
            logger.error(f"Error calculating feature revenue correlation: {str(e)}")
            return 0.0
    
    async def get_common_use_cases(self, feature: str, start_date: datetime, end_date: datetime) -> List[str]:
        """Get common use cases for a feature"""
        # Placeholder implementation
        return ["Content creation", "Analytics review", "Scheduling"]
    
    async def get_feature_niche_usage(self, feature: str, start_date: datetime, end_date: datetime) -> Dict[str, int]:
        """Get feature usage by business niche"""
        # Placeholder implementation
        return {"business_consulting": 150, "fitness": 120, "ecommerce": 90}
    
    async def count_total_sessions(self, start_date: datetime, end_date: datetime) -> int:
        """Count total user sessions"""
        try:
            db = get_db_connection()
            query = """
                SELECT COUNT(DISTINCT session_id) as total_sessions
                FROM user_sessions
                WHERE client_id = :client_id
                    AND session_start >= :start_date
                    AND session_start <= :end_date
            """
            
            result = await db.fetch_one(
                query,
                {
                    'client_id': self.client_id,
                    'start_date': start_date,
                    'end_date': end_date
                }
            )
            
            return result.get('total_sessions', 0) if result else 0
            
        except Exception as e:
            logger.error(f"Error counting total sessions: {str(e)}")
            return 0
    
    async def count_unique_features_used(self, start_date: datetime, end_date: datetime) -> int:
        """Count unique features used"""
        try:
            db = get_db_connection()
            query = """
                SELECT COUNT(DISTINCT feature_name) as unique_features
                FROM feature_usage_logs
                WHERE client_id = :client_id
                    AND timestamp >= :start_date
                    AND timestamp <= :end_date
            """
            
            result = await db.fetch_one(
                query,
                {
                    'client_id': self.client_id,
                    'start_date': start_date,
                    'end_date': end_date
                }
            )
            
            return result.get('unique_features', 0) if result else 0
            
        except Exception as e:
            logger.error(f"Error counting unique features: {str(e)}")
            return 0
    
    async def calculate_automation_efficiency(self, start_date: datetime, end_date: datetime) -> float:
        """Calculate overall automation efficiency"""
        try:
            db = get_db_connection()
            query = """
                SELECT 
                    SUM(automated_time_saved_seconds) as time_saved,
                    SUM(total_task_time_seconds) as total_time
                FROM automation_metrics
                WHERE client_id = :client_id
                    AND timestamp >= :start_date
                    AND timestamp <= :end_date
            """
            
            result = await db.fetch_one(
                query,
                {
                    'client_id': self.client_id,
                    'start_date': start_date,
                    'end_date': end_date
                }
            )
            
            if result:
                time_saved = result.get('time_saved', 0)
                total_time = result.get('total_time', 1)
                return (time_saved / total_time * 100) if total_time > 0 else 0
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating automation efficiency: {str(e)}")
            return 0.0
    
    async def analyze_user_journey_patterns(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Analyze common user journey patterns"""
        # Placeholder implementation
        return [
            {
                'pattern': 'Dashboard → Content Creation → Publishing',
                'frequency': 450,
                'average_completion_time_minutes': 15
            },
            {
                'pattern': 'Analytics → Optimization → Re-publishing',
                'frequency': 320,
                'average_completion_time_minutes': 20
            }
        ]
    
    async def calculate_feature_revenue_correlations(self, start_date: datetime, end_date: datetime) -> Dict[str, float]:
        """Calculate correlations between features and revenue"""
        # Placeholder implementation - would use actual correlation analysis
        return {
            'viral_optimization': 0.85,
            'ai_suggestions': 0.78,
            'cross_platform_posting': 0.72,
            'analytics_dashboard': 0.65
        }
    
    async def calculate_platform_engagement_correlations(self, start_date: datetime, end_date: datetime) -> Dict[str, float]:
        """Calculate correlations between platforms and engagement"""
        # Placeholder implementation
        return {
            'instagram': 0.82,
            'tiktok': 0.79,
            'linkedin': 0.71,
            'twitter': 0.68
        }
    
    async def calculate_content_viral_correlations(self, start_date: datetime, end_date: datetime) -> Dict[str, float]:
        """Calculate correlations between content types and viral success"""
        # Placeholder implementation
        return {
            'video': 0.88,
            'carousel': 0.75,
            'image': 0.62,
            'text': 0.45
        }