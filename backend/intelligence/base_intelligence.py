from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import json
import logging

# Configure logging
logger = logging.getLogger(__name__)

class AnalyticsTimeframe(Enum):
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"

class BusinessMetricType(Enum):
    REVENUE = "revenue"
    ENGAGEMENT = "engagement"
    CONVERSION = "conversion"
    REACH = "reach"
    EFFICIENCY = "efficiency"
    ROI = "roi"

@dataclass
class IntelligenceInsight:
    metric_type: BusinessMetricType
    insight_text: str
    confidence_score: float
    impact_level: str  # "high", "medium", "low"
    actionable_recommendations: List[str]
    supporting_data: Dict[str, Any]
    generated_at: datetime = field(default_factory=datetime.now)

@dataclass
class BusinessMetrics:
    client_id: str
    business_niche: str
    timeframe: AnalyticsTimeframe
    
    # Revenue metrics
    total_revenue: float = 0.0
    revenue_growth_rate: float = 0.0
    revenue_per_post: float = 0.0
    conversion_revenue: float = 0.0
    
    # Engagement metrics
    total_engagement: int = 0
    engagement_rate: float = 0.0
    engagement_growth_rate: float = 0.0
    viral_coefficient: float = 0.0
    
    # Efficiency metrics
    cost_per_engagement: float = 0.0
    roi_percentage: float = 0.0
    time_efficiency_score: float = 0.0
    automation_savings: float = 0.0
    
    # Predictive metrics
    predicted_revenue_next_period: float = 0.0
    churn_risk_score: float = 0.0
    growth_potential_score: float = 0.0
    
    timestamp: datetime = field(default_factory=datetime.now)

class IntelligenceEngineError(Exception):
    """Custom exception for intelligence engine errors"""
    pass

class DataAggregator:
    """Aggregates data from multiple sources for analysis"""
    
    async def aggregate_platform_data(self, client_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Aggregate data from all platforms"""
        # This will be implemented to pull from actual platform APIs
        return {}
    
    async def aggregate_revenue_data(self, client_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Aggregate revenue data from all sources"""
        return {}

class AIAnalyzer:
    """AI-powered data analysis engine"""
    
    async def analyze_patterns(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze patterns in the data using ML"""
        return []
    
    async def predict_trends(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Predict future trends using ML models"""
        return {}

class UniversalIntelligenceEngine(ABC):
    """Base class for all business intelligence modules"""
    
    def __init__(self, client_id: str, intelligence_type: str):
        self.client_id = client_id
        self.intelligence_type = intelligence_type
        self.data_aggregator = DataAggregator()
        self.ai_analyzer = AIAnalyzer()
        
    @abstractmethod
    async def collect_data(self, timeframe: AnalyticsTimeframe) -> Dict[str, Any]:
        """Collect relevant data for analysis"""
        try:
            logger.info(f"Collecting data for {timeframe.value} analysis")
            
            # Calculate date range
            end_date = datetime.now()
            start_date = await self.get_timeframe_start_date(end_date, timeframe)
            
            # Collect data from multiple sources in parallel
            platform_data_task = self.data_aggregator.aggregate_platform_data(
                self.client_id, start_date, end_date
            )
            revenue_data_task = self.data_aggregator.aggregate_revenue_data(
                self.client_id, start_date, end_date
            )
            
            # Gather all data
            platform_data, revenue_data = await asyncio.gather(
                platform_data_task, revenue_data_task
            )
            
            # Combine and structure data
            collected_data = {
                'client_id': self.client_id,
                'timeframe': timeframe.value,
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'platform_data': platform_data,
                'revenue_data': revenue_data,
                
                # Revenue metrics
                'revenue_sources': revenue_data.get('revenue_by_source', {}),
                'total_revenue_current': revenue_data.get('total_revenue', 0),
                'previous_period_revenue': revenue_data.get('previous_period_revenue', 0),
                'conversion_revenue': revenue_data.get('conversion_revenue', 0),
                
                # Engagement metrics
                'engagement_by_platform': platform_data.get('engagement_by_platform', {}),
                'total_impressions': sum(platform_data.get('impressions_by_platform', {}).values()),
                'total_reach': sum(platform_data.get('reach_by_platform', {}).values()),
                'total_shares': sum(platform_data.get('shares_by_platform', {}).values()),
                'new_followers_from_shares': platform_data.get('follower_growth', {}).get('from_shares', 0),
                'previous_period_engagement': platform_data.get('previous_period_engagement', 0),
                
                # Content metrics
                'total_posts': platform_data.get('total_posts', 0),
                'content_performance': platform_data.get('content_performance', []),
                'viral_posts': platform_data.get('viral_posts', []),
                
                # Business metrics
                'total_marketing_cost': revenue_data.get('marketing_spend', 0),
                'automation_time_saved_hours': revenue_data.get('automation_savings', {}).get('time_hours', 0),
                'total_time_invested_hours': revenue_data.get('time_investment', {}).get('total_hours', 40),
                'average_hourly_rate': revenue_data.get('business_metrics', {}).get('hourly_rate', 50),
                
                # Customer metrics
                'engagement_decline_percentage': platform_data.get('engagement_trends', {}).get('decline_percentage', 0),
                'days_since_last_purchase': revenue_data.get('customer_metrics', {}).get('days_since_last_purchase', 0),
                
                # Market data
                'addressable_market_size': revenue_data.get('market_data', {}).get('addressable_market', 1000000),
                'current_total_reach': sum(platform_data.get('reach_by_platform', {}).values()),
                
                # Content analysis data
                'content_themes': platform_data.get('content_analysis', {}).get('themes', []),
                'engagement_patterns': platform_data.get('engagement_analysis', {}),
                'audience_demographics': platform_data.get('audience_data', {}),
                
                # Collected timestamp
                'collected_at': datetime.now().isoformat()
            }
            
            logger.info(f"Data collection completed for {self.client_id}")
            return collected_data
            
        except Exception as e:
            error_msg = f"Data collection failed: {str(e)}"
            await self.log_intelligence_error(error_msg)
            raise IntelligenceEngineError(error_msg)
    
    @abstractmethod
    async def analyze_data(self, data: Dict[str, Any]) -> List[IntelligenceInsight]:
        """Analyze data and generate insights"""
        try:
            logger.info(f"Analyzing data for intelligence insights")
            insights = []
            
            # Detect business niche for context
            business_niche = await self.detect_business_niche_from_data(data)
            
            # 1. Revenue Analysis Insights
            revenue_insights = await self._analyze_revenue_patterns(data, business_niche)
            insights.extend(revenue_insights)
            
            # 2. Engagement Analysis Insights
            engagement_insights = await self._analyze_engagement_patterns(data, business_niche)
            insights.extend(engagement_insights)
            
            # 3. Growth Opportunity Insights
            growth_insights = await self._analyze_growth_opportunities(data, business_niche)
            insights.extend(growth_insights)
            
            # 4. Risk Assessment Insights
            risk_insights = await self._analyze_risk_factors(data, business_niche)
            insights.extend(risk_insights)
            
            # 5. Content Performance Insights
            content_insights = await self._analyze_content_performance(data, business_niche)
            insights.extend(content_insights)
            
            # 6. Efficiency Insights
            efficiency_insights = await self._analyze_efficiency_metrics(data, business_niche)
            insights.extend(efficiency_insights)
            
            # 7. Competitive Insights
            competitive_insights = await self._analyze_competitive_position(data, business_niche)
            insights.extend(competitive_insights)
            
            # Sort insights by impact and confidence
            insights.sort(key=lambda x: (
                {'high': 3, 'medium': 2, 'low': 1}.get(x.impact_level, 0),
                x.confidence_score
            ), reverse=True)
            
            logger.info(f"Generated {len(insights)} intelligence insights")
            return insights
            
        except Exception as e:
            error_msg = f"Data analysis failed: {str(e)}"
            await self.log_intelligence_error(error_msg)
            raise IntelligenceEngineError(error_msg)
    
    @abstractmethod
    async def generate_recommendations(self, insights: List[IntelligenceInsight]) -> List[str]:
        """Generate actionable business recommendations"""
        try:
            logger.info(f"Generating recommendations from {len(insights)} insights")
            
            # Collect all recommendations from insights
            all_recommendations = []
            
            # Extract recommendations from high-impact insights first
            high_impact_insights = [i for i in insights if i.impact_level == "high"]
            medium_impact_insights = [i for i in insights if i.impact_level == "medium"]
            low_impact_insights = [i for i in insights if i.impact_level == "low"]
            
            # Process high impact first
            for insight in sorted(high_impact_insights, key=lambda x: x.confidence_score, reverse=True):
                for rec in insight.actionable_recommendations:
                    if rec not in all_recommendations:
                        all_recommendations.append(rec)
            
            # Add medium impact recommendations
            for insight in sorted(medium_impact_insights, key=lambda x: x.confidence_score, reverse=True):
                for rec in insight.actionable_recommendations:
                    if rec not in all_recommendations and len(all_recommendations) < 15:
                        all_recommendations.append(rec)
            
            # Add low impact if needed
            for insight in sorted(low_impact_insights, key=lambda x: x.confidence_score, reverse=True):
                for rec in insight.actionable_recommendations:
                    if rec not in all_recommendations and len(all_recommendations) < 20:
                        all_recommendations.append(rec)
            
            # Generate universal strategic recommendations
            strategic_recommendations = await self._generate_strategic_recommendations(insights)
            
            # Combine and prioritize
            final_recommendations = await self._prioritize_universal_recommendations(
                all_recommendations, strategic_recommendations, insights
            )
            
            logger.info(f"Generated {len(final_recommendations)} prioritized recommendations")
            return final_recommendations[:10]  # Return top 10
            
        except Exception as e:
            error_msg = f"Recommendation generation failed: {str(e)}"
            await self.log_intelligence_error(error_msg)
            return [
                "Review current marketing strategy for optimization opportunities",
                "Analyze competitor activities for market insights",
                "Focus on high-performing content formats",
                "Optimize posting schedule based on audience activity",
                "Implement automated reporting for better decision making"
            ]
    
    async def get_business_intelligence(self, timeframe: AnalyticsTimeframe = AnalyticsTimeframe.MONTH) -> Dict[str, Any]:
        """Main intelligence gathering function"""
        try:
            # 1. Collect comprehensive data
            raw_data = await self.collect_data(timeframe)
            
            # 2. Analyze for insights
            insights = await self.analyze_data(raw_data)
            
            # 3. Generate recommendations
            recommendations = await self.generate_recommendations(insights)
            
            # 4. Calculate business metrics
            metrics = await self.calculate_business_metrics(raw_data, timeframe)
            
            # 5. Predictive analysis
            predictions = await self.generate_predictions(raw_data, metrics)
            
            return {
                'intelligence_type': self.intelligence_type,
                'client_id': self.client_id,
                'timeframe': timeframe.value,
                'metrics': metrics,
                'insights': insights,
                'recommendations': recommendations,
                'predictions': predictions,
                'generated_at': datetime.now(),
                'confidence_score': await self.calculate_overall_confidence(insights)
            }
            
        except Exception as e:
            await self.log_intelligence_error(f"Intelligence gathering failed: {str(e)}")
            raise IntelligenceEngineError(f"Failed to generate intelligence: {str(e)}")
    
    async def get_timeframe_start_date(self, end_date: datetime, timeframe: AnalyticsTimeframe) -> datetime:
        """Calculate start date based on timeframe"""
        if timeframe == AnalyticsTimeframe.HOUR:
            return end_date - timedelta(hours=1)
        elif timeframe == AnalyticsTimeframe.DAY:
            return end_date - timedelta(days=1)
        elif timeframe == AnalyticsTimeframe.WEEK:
            return end_date - timedelta(weeks=1)
        elif timeframe == AnalyticsTimeframe.MONTH:
            return end_date - timedelta(days=30)
        elif timeframe == AnalyticsTimeframe.QUARTER:
            return end_date - timedelta(days=90)
        elif timeframe == AnalyticsTimeframe.YEAR:
            return end_date - timedelta(days=365)
        else:
            return end_date - timedelta(days=30)
    
    async def calculate_business_metrics(self, data: Dict[str, Any], timeframe: AnalyticsTimeframe) -> BusinessMetrics:
        """Calculate comprehensive business metrics"""
        business_niche = await self.detect_business_niche_from_data(data)
        
        return BusinessMetrics(
            client_id=self.client_id,
            business_niche=business_niche,
            timeframe=timeframe,
            total_revenue=await self.calculate_total_revenue(data),
            revenue_growth_rate=await self.calculate_revenue_growth(data, timeframe),
            revenue_per_post=await self.calculate_revenue_per_post(data),
            conversion_revenue=await self.calculate_conversion_revenue(data),
            total_engagement=await self.calculate_total_engagement(data),
            engagement_rate=await self.calculate_engagement_rate(data),
            engagement_growth_rate=await self.calculate_engagement_growth(data, timeframe),
            viral_coefficient=await self.calculate_viral_coefficient(data),
            cost_per_engagement=await self.calculate_cost_per_engagement(data),
            roi_percentage=await self.calculate_roi_percentage(data),
            time_efficiency_score=await self.calculate_time_efficiency(data),
            automation_savings=await self.calculate_automation_savings(data),
            predicted_revenue_next_period=await self.predict_next_period_revenue(data),
            churn_risk_score=await self.calculate_churn_risk(data),
            growth_potential_score=await self.calculate_growth_potential(data)
        )
    
    async def detect_business_niche_from_data(self, data: Dict[str, Any]) -> str:
        """Detect business niche from content and engagement patterns"""
        # AI-driven niche detection based on content analysis
        return "business_consulting"  # Placeholder
    
    async def calculate_total_revenue(self, data: Dict[str, Any]) -> float:
        """Calculate total revenue from all sources"""
        return sum(data.get('revenue_sources', {}).values())
    
    async def calculate_revenue_growth(self, data: Dict[str, Any], timeframe: AnalyticsTimeframe) -> float:
        """Calculate revenue growth rate"""
        current_revenue = await self.calculate_total_revenue(data)
        previous_revenue = data.get('previous_period_revenue', 0)
        
        if previous_revenue > 0:
            return ((current_revenue - previous_revenue) / previous_revenue) * 100
        return 0.0
    
    async def calculate_revenue_per_post(self, data: Dict[str, Any]) -> float:
        """Calculate average revenue per post"""
        total_revenue = await self.calculate_total_revenue(data)
        total_posts = data.get('total_posts', 1)
        
        return total_revenue / total_posts if total_posts > 0 else 0.0
    
    async def calculate_conversion_revenue(self, data: Dict[str, Any]) -> float:
        """Calculate revenue from conversions"""
        return data.get('conversion_revenue', 0.0)
    
    async def calculate_total_engagement(self, data: Dict[str, Any]) -> int:
        """Calculate total engagement across all platforms"""
        return sum(data.get('engagement_by_platform', {}).values())
    
    async def calculate_engagement_rate(self, data: Dict[str, Any]) -> float:
        """Calculate overall engagement rate"""
        total_engagement = await self.calculate_total_engagement(data)
        total_impressions = data.get('total_impressions', 1)
        
        return (total_engagement / total_impressions * 100) if total_impressions > 0 else 0.0
    
    async def calculate_engagement_growth(self, data: Dict[str, Any], timeframe: AnalyticsTimeframe) -> float:
        """Calculate engagement growth rate"""
        current_engagement = await self.calculate_total_engagement(data)
        previous_engagement = data.get('previous_period_engagement', 0)
        
        if previous_engagement > 0:
            return ((current_engagement - previous_engagement) / previous_engagement) * 100
        return 0.0
    
    async def calculate_viral_coefficient(self, data: Dict[str, Any]) -> float:
        """Calculate viral coefficient (K-factor)"""
        shares = data.get('total_shares', 0)
        new_followers = data.get('new_followers_from_shares', 0)
        
        return new_followers / shares if shares > 0 else 0.0
    
    async def calculate_cost_per_engagement(self, data: Dict[str, Any]) -> float:
        """Calculate cost per engagement"""
        total_cost = data.get('total_marketing_cost', 0)
        total_engagement = await self.calculate_total_engagement(data)
        
        return total_cost / total_engagement if total_engagement > 0 else 0.0
    
    async def calculate_roi_percentage(self, data: Dict[str, Any]) -> float:
        """Calculate return on investment percentage"""
        total_revenue = await self.calculate_total_revenue(data)
        total_cost = data.get('total_marketing_cost', 1)
        
        return ((total_revenue - total_cost) / total_cost * 100) if total_cost > 0 else 0.0
    
    async def calculate_time_efficiency(self, data: Dict[str, Any]) -> float:
        """Calculate time efficiency score (0-100)"""
        time_saved = data.get('automation_time_saved_hours', 0)
        total_time = data.get('total_time_invested_hours', 1)
        
        efficiency = (time_saved / total_time * 100) if total_time > 0 else 0.0
        return min(100, efficiency)
    
    async def calculate_automation_savings(self, data: Dict[str, Any]) -> float:
        """Calculate monetary savings from automation"""
        time_saved_hours = data.get('automation_time_saved_hours', 0)
        hourly_rate = data.get('average_hourly_rate', 50)  # Default $50/hour
        
        return time_saved_hours * hourly_rate
    
    async def predict_next_period_revenue(self, data: Dict[str, Any]) -> float:
        """Predict revenue for next period using ML"""
        # Simple linear prediction for now
        current_revenue = await self.calculate_total_revenue(data)
        growth_rate = await self.calculate_revenue_growth(data, AnalyticsTimeframe.MONTH)
        
        return current_revenue * (1 + growth_rate / 100)
    
    async def calculate_churn_risk(self, data: Dict[str, Any]) -> float:
        """Calculate customer churn risk score (0-1)"""
        engagement_decline = data.get('engagement_decline_percentage', 0)
        days_since_last_purchase = data.get('days_since_last_purchase', 0)
        
        risk_score = 0.0
        if engagement_decline > 50:
            risk_score += 0.5
        if days_since_last_purchase > 60:
            risk_score += 0.5
            
        return min(1.0, risk_score)
    
    async def calculate_growth_potential(self, data: Dict[str, Any]) -> float:
        """Calculate growth potential score (0-100)"""
        market_size = data.get('addressable_market_size', 1000000)
        current_reach = data.get('current_total_reach', 1000)
        engagement_rate = await self.calculate_engagement_rate(data)
        
        market_penetration = (current_reach / market_size * 100) if market_size > 0 else 0
        growth_potential = (100 - market_penetration) * (engagement_rate / 10)
        
        return min(100, growth_potential)
    
    async def generate_predictions(self, raw_data: Dict[str, Any], metrics: BusinessMetrics) -> Dict[str, Any]:
        """Generate predictive insights using ML"""
        return {
            'revenue_forecast': {
                'next_month': metrics.predicted_revenue_next_period,
                'confidence': 0.85
            },
            'growth_trajectory': {
                'direction': 'positive' if metrics.revenue_growth_rate > 0 else 'negative',
                'rate': metrics.revenue_growth_rate
            },
            'risk_indicators': {
                'churn_risk': metrics.churn_risk_score,
                'market_saturation': 1 - (metrics.growth_potential_score / 100)
            }
        }
    
    async def calculate_overall_confidence(self, insights: List[IntelligenceInsight]) -> float:
        """Calculate overall confidence score for the intelligence report"""
        if not insights:
            return 0.0
            
        confidence_scores = [insight.confidence_score for insight in insights]
        return sum(confidence_scores) / len(confidence_scores)
    
    async def log_intelligence_error(self, error_message: str):
        """Log intelligence engine errors"""
        logger.error(f"Intelligence Engine Error ({self.intelligence_type}): {error_message}")
    
    async def prioritize_recommendations(self, recommendations: List[str], insights: List[IntelligenceInsight]) -> List[str]:
        """Prioritize recommendations based on impact and confidence"""
        # Sort by impact level and confidence
        high_impact_insights = [i for i in insights if i.impact_level == "high"]
        
        # Extract recommendations from high impact insights first
        prioritized = []
        for insight in sorted(high_impact_insights, key=lambda x: x.confidence_score, reverse=True):
            for rec in insight.actionable_recommendations:
                if rec not in prioritized:
                    prioritized.append(rec)
        
        # Add remaining unique recommendations
        for rec in recommendations:
            if rec not in prioritized:
                prioritized.append(rec)
                
        return prioritized[:10]  # Return top 10 recommendations