"""Revenue Tracking Engine - Comprehensive revenue tracking and attribution system"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import logging
from collections import defaultdict
import statistics

from .base_intelligence import (
    UniversalIntelligenceEngine,
    AnalyticsTimeframe,
    BusinessMetricType,
    IntelligenceInsight,
    IntelligenceEngineError
)

logger = logging.getLogger(__name__)

class AttributionModel(Enum):
    FIRST_TOUCH = "first_touch"
    LAST_TOUCH = "last_touch"
    LINEAR = "linear"
    TIME_DECAY = "time_decay"
    POSITION_BASED = "position_based"
    DATA_DRIVEN = "data_driven"

class RevenueSource(Enum):
    DIRECT_SALE = "direct_sale"
    AFFILIATE = "affiliate"
    SUBSCRIPTION = "subscription"
    LEAD_GENERATION = "lead_generation"
    ADVERTISING = "advertising"
    SPONSORSHIP = "sponsorship"

@dataclass
class RevenueEvent:
    """Individual revenue event"""
    event_id: str
    client_id: str
    amount: float
    source: RevenueSource
    platform: str
    content_id: Optional[str]
    customer_id: str
    attribution_path: List[Dict[str, Any]]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class AttributionResult:
    """Attribution analysis result"""
    model: AttributionModel
    total_revenue: float
    touchpoint_credits: Dict[str, float]
    conversion_paths: List[Dict[str, Any]]
    confidence_score: float

class AttributionEngine:
    """Multi-touch attribution modeling engine"""
    
    def __init__(self):
        self.attribution_window_days = 30
        
    async def calculate_attributed_revenue(
        self, 
        model: str, 
        start_date: datetime, 
        end_date: datetime,
        client_id: str
    ) -> Dict[str, Any]:
        """Calculate revenue using specified attribution model"""
        
        # Get conversion data
        conversions = await self.get_conversions(client_id, start_date, end_date)
        
        # Apply attribution model
        if model == 'first_touch':
            return await self.apply_first_touch_attribution(conversions)
        elif model == 'last_touch':
            return await self.apply_last_touch_attribution(conversions)
        elif model == 'linear':
            return await self.apply_linear_attribution(conversions)
        elif model == 'time_decay':
            return await self.apply_time_decay_attribution(conversions)
        elif model == 'position_based':
            return await self.apply_position_based_attribution(conversions)
        else:
            return await self.apply_linear_attribution(conversions)  # Default
    
    async def get_conversions(self, client_id: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get conversion data with touchpoint history"""
        # In production, this would query actual conversion data
        # For now, return sample data
        return [
            {
                'conversion_id': 'conv_001',
                'revenue': 150.0,
                'touchpoints': [
                    {'platform': 'instagram', 'content_id': 'post_123', 'timestamp': start_date + timedelta(days=1)},
                    {'platform': 'linkedin', 'content_id': 'post_124', 'timestamp': start_date + timedelta(days=3)},
                    {'platform': 'instagram', 'content_id': 'post_125', 'timestamp': start_date + timedelta(days=5)}
                ]
            },
            {
                'conversion_id': 'conv_002',
                'revenue': 250.0,
                'touchpoints': [
                    {'platform': 'tiktok', 'content_id': 'post_126', 'timestamp': start_date + timedelta(days=2)},
                    {'platform': 'tiktok', 'content_id': 'post_127', 'timestamp': start_date + timedelta(days=4)}
                ]
            }
        ]
    
    async def apply_first_touch_attribution(self, conversions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply first-touch attribution model"""
        platform_credits = defaultdict(float)
        content_credits = defaultdict(float)
        
        for conversion in conversions:
            if conversion['touchpoints']:
                first_touch = conversion['touchpoints'][0]
                platform_credits[first_touch['platform']] += conversion['revenue']
                content_credits[first_touch['content_id']] += conversion['revenue']
        
        return {
            'total_revenue': sum(platform_credits.values()),
            'platform_attribution': dict(platform_credits),
            'content_attribution': dict(content_credits),
            'model': 'first_touch'
        }
    
    async def apply_last_touch_attribution(self, conversions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply last-touch attribution model"""
        platform_credits = defaultdict(float)
        content_credits = defaultdict(float)
        
        for conversion in conversions:
            if conversion['touchpoints']:
                last_touch = conversion['touchpoints'][-1]
                platform_credits[last_touch['platform']] += conversion['revenue']
                content_credits[last_touch['content_id']] += conversion['revenue']
        
        return {
            'total_revenue': sum(platform_credits.values()),
            'platform_attribution': dict(platform_credits),
            'content_attribution': dict(content_credits),
            'model': 'last_touch'
        }
    
    async def apply_linear_attribution(self, conversions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply linear attribution model - equal credit to all touchpoints"""
        platform_credits = defaultdict(float)
        content_credits = defaultdict(float)
        
        for conversion in conversions:
            if conversion['touchpoints']:
                credit_per_touch = conversion['revenue'] / len(conversion['touchpoints'])
                for touchpoint in conversion['touchpoints']:
                    platform_credits[touchpoint['platform']] += credit_per_touch
                    content_credits[touchpoint['content_id']] += credit_per_touch
        
        return {
            'total_revenue': sum(platform_credits.values()),
            'platform_attribution': dict(platform_credits),
            'content_attribution': dict(content_credits),
            'model': 'linear'
        }
    
    async def apply_time_decay_attribution(self, conversions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply time-decay attribution model - more recent touchpoints get more credit"""
        platform_credits = defaultdict(float)
        content_credits = defaultdict(float)
        decay_rate = 0.5  # 50% decay per week
        
        for conversion in conversions:
            if conversion['touchpoints']:
                conversion_time = conversion['touchpoints'][-1]['timestamp']
                total_weight = 0
                weights = []
                
                # Calculate weights based on time decay
                for touchpoint in conversion['touchpoints']:
                    days_before = (conversion_time - touchpoint['timestamp']).days
                    weeks_before = days_before / 7
                    weight = (1 - decay_rate) ** weeks_before
                    weights.append(weight)
                    total_weight += weight
                
                # Distribute revenue based on weights
                for i, touchpoint in enumerate(conversion['touchpoints']):
                    credit = conversion['revenue'] * (weights[i] / total_weight)
                    platform_credits[touchpoint['platform']] += credit
                    content_credits[touchpoint['content_id']] += credit
        
        return {
            'total_revenue': sum(platform_credits.values()),
            'platform_attribution': dict(platform_credits),
            'content_attribution': dict(content_credits),
            'model': 'time_decay'
        }
    
    async def apply_position_based_attribution(self, conversions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply position-based attribution - 40% first, 40% last, 20% middle"""
        platform_credits = defaultdict(float)
        content_credits = defaultdict(float)
        
        for conversion in conversions:
            touchpoints = conversion['touchpoints']
            if not touchpoints:
                continue
                
            if len(touchpoints) == 1:
                # Only one touchpoint gets all credit
                platform_credits[touchpoints[0]['platform']] += conversion['revenue']
                content_credits[touchpoints[0]['content_id']] += conversion['revenue']
            elif len(touchpoints) == 2:
                # Split 50/50 between first and last
                credit = conversion['revenue'] / 2
                platform_credits[touchpoints[0]['platform']] += credit
                platform_credits[touchpoints[1]['platform']] += credit
                content_credits[touchpoints[0]['content_id']] += credit
                content_credits[touchpoints[1]['content_id']] += credit
            else:
                # 40% first, 40% last, 20% split among middle
                first_credit = conversion['revenue'] * 0.4
                last_credit = conversion['revenue'] * 0.4
                middle_total = conversion['revenue'] * 0.2
                middle_credit = middle_total / (len(touchpoints) - 2)
                
                # First touchpoint
                platform_credits[touchpoints[0]['platform']] += first_credit
                content_credits[touchpoints[0]['content_id']] += first_credit
                
                # Middle touchpoints
                for touchpoint in touchpoints[1:-1]:
                    platform_credits[touchpoint['platform']] += middle_credit
                    content_credits[touchpoint['content_id']] += middle_credit
                
                # Last touchpoint
                platform_credits[touchpoints[-1]['platform']] += last_credit
                content_credits[touchpoints[-1]['content_id']] += last_credit
        
        return {
            'total_revenue': sum(platform_credits.values()),
            'platform_attribution': dict(platform_credits),
            'content_attribution': dict(content_credits),
            'model': 'position_based'
        }
    
    async def analyze_multi_touch_attribution(self, start_date: datetime, end_date: datetime, client_id: str) -> Dict[str, Any]:
        """Analyze attribution across multiple models"""
        models = ['first_touch', 'last_touch', 'linear', 'time_decay', 'position_based']
        results = {}
        
        for model in models:
            results[model] = await self.calculate_attributed_revenue(model, start_date, end_date, client_id)
        
        # Compare models and find consensus
        platform_consensus = await self.calculate_attribution_consensus(results, 'platform_attribution')
        content_consensus = await self.calculate_attribution_consensus(results, 'content_attribution')
        
        return {
            'model_results': results,
            'platform_consensus': platform_consensus,
            'content_consensus': content_consensus,
            'recommended_model': await self.recommend_attribution_model(results)
        }
    
    async def calculate_attribution_consensus(self, results: Dict[str, Any], attribution_type: str) -> Dict[str, Any]:
        """Calculate consensus across attribution models"""
        all_platforms = set()
        for model_result in results.values():
            all_platforms.update(model_result.get(attribution_type, {}).keys())
        
        consensus = {}
        for platform in all_platforms:
            values = []
            for model_result in results.values():
                if platform in model_result.get(attribution_type, {}):
                    values.append(model_result[attribution_type][platform])
            
            if values:
                consensus[platform] = {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values),
                    'max': max(values)
                }
        
        return consensus
    
    async def recommend_attribution_model(self, results: Dict[str, Any]) -> str:
        """Recommend best attribution model based on data patterns"""
        # Simple heuristic - in production would use ML
        # For now, recommend linear as it's most balanced
        return 'linear'
    
    async def compare_platform_attribution(self, start_date: datetime, end_date: datetime, client_id: str) -> Dict[str, Any]:
        """Compare platform performance across attribution models"""
        attribution_results = await self.analyze_multi_touch_attribution(start_date, end_date, client_id)
        
        platform_comparison = {}
        for platform in attribution_results['platform_consensus']:
            platform_comparison[platform] = {
                'consensus_revenue': attribution_results['platform_consensus'][platform]['mean'],
                'attribution_variance': attribution_results['platform_consensus'][platform]['std_dev'],
                'model_agreement': 1 - (attribution_results['platform_consensus'][platform]['std_dev'] / 
                                      attribution_results['platform_consensus'][platform]['mean'])
                                      if attribution_results['platform_consensus'][platform]['mean'] > 0 else 0
            }
        
        return platform_comparison
    
    async def calculate_post_attribution(self, post_id: str, platform: str, tracking_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate revenue attribution for a specific post"""
        total_revenue = tracking_results.get('total_attributed_revenue', 0)
        conversion_count = tracking_results.get('conversion_count', 0)
        
        return {
            'direct_revenue': tracking_results.get('direct_revenue', 0),
            'assisted_revenue': tracking_results.get('assisted_revenue', 0),
            'total_attributed_revenue': total_revenue,
            'conversion_count': conversion_count,
            'average_order_value': total_revenue / conversion_count if conversion_count > 0 else 0,
            'attribution_confidence': tracking_results.get('confidence_score', 0.75)
        }

class ConversionTracker:
    """Track conversions and attribute them to content"""
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.active_tracking = {}
        
    async def setup_post_tracking(self, post_id: str, platform: str, content: Dict[str, Any], tracking_start: datetime):
        """Set up conversion tracking for a post"""
        self.active_tracking[post_id] = {
            'platform': platform,
            'content': content,
            'tracking_start': tracking_start,
            'conversions': [],
            'engagement_metrics': []
        }
        
        # In production, this would set up actual tracking pixels/webhooks
        logger.info(f"Set up tracking for post {post_id} on {platform}")
    
    async def track_conversion(self, post_id: str, conversion_data: Dict[str, Any]):
        """Track a conversion event"""
        if post_id in self.active_tracking:
            self.active_tracking[post_id]['conversions'].append({
                'timestamp': datetime.now(),
                'revenue': conversion_data.get('revenue', 0),
                'customer_id': conversion_data.get('customer_id'),
                'conversion_type': conversion_data.get('type', 'direct_sale')
            })
    
    async def get_post_conversions(self, post_id: str) -> List[Dict[str, Any]]:
        """Get all conversions for a post"""
        if post_id in self.active_tracking:
            return self.active_tracking[post_id]['conversions']
        return []

class RevenueCalculator:
    """Calculate various revenue metrics"""
    
    async def calculate_direct_revenue(self, start_date: datetime, end_date: datetime) -> float:
        """Calculate direct revenue from social media"""
        # In production, this would query actual revenue data
        # For now, return sample calculation
        days = (end_date - start_date).days
        daily_revenue = 500.0  # Average $500/day
        return days * daily_revenue
    
    async def get_revenue_by_source(self, start_date: datetime, end_date: datetime) -> Dict[str, float]:
        """Get revenue breakdown by source"""
        return {
            'direct_sale': 5000.0,
            'affiliate': 2000.0,
            'subscription': 3500.0,
            'lead_generation': 1500.0,
            'advertising': 800.0,
            'sponsorship': 1200.0
        }
    
    async def get_revenue_by_campaign(self, start_date: datetime, end_date: datetime) -> Dict[str, float]:
        """Get revenue by marketing campaign"""
        return {
            'summer_sale_2024': 3500.0,
            'product_launch_q2': 4200.0,
            'brand_awareness': 1800.0,
            'holiday_special': 5500.0
        }
    
    async def get_revenue_by_content_type(self, start_date: datetime, end_date: datetime) -> Dict[str, float]:
        """Get revenue by content type"""
        return {
            'video': 6500.0,
            'carousel': 4200.0,
            'single_image': 2800.0,
            'story': 1500.0,
            'reel': 3000.0
        }
    
    async def get_revenue_by_niche(self, start_date: datetime, end_date: datetime) -> Dict[str, float]:
        """Get revenue by business niche"""
        return {
            'fitness': 4500.0,
            'business_consulting': 6200.0,
            'ecommerce': 5800.0,
            'education': 3500.0,
            'creative': 2000.0
        }
    
    async def get_subscription_revenue(self, start_date: datetime, end_date: datetime) -> float:
        """Get subscription-based revenue"""
        return 3500.0
    
    async def get_one_time_sales(self, start_date: datetime, end_date: datetime) -> float:
        """Get one-time sales revenue"""
        return 5000.0
    
    async def get_lead_gen_revenue(self, start_date: datetime, end_date: datetime) -> float:
        """Get lead generation revenue"""
        return 1500.0

class RevenueTrackingEngine(UniversalIntelligenceEngine):
    """Comprehensive revenue tracking and attribution system"""
    
    def __init__(self, client_id: str):
        super().__init__(client_id, "revenue_tracking")
        self.attribution_engine = AttributionEngine()
        self.conversion_tracker = ConversionTracker(client_id)
        self.revenue_calculator = RevenueCalculator()
        
    async def collect_data(self, timeframe: AnalyticsTimeframe) -> Dict[str, Any]:
        """Collect comprehensive revenue data across all sources"""
        end_date = datetime.now()
        start_date = await self.get_timeframe_start_date(end_date, timeframe)
        
        # Direct revenue data
        direct_revenue = await self.collect_direct_revenue_data(start_date, end_date)
        
        # Attribution revenue data
        attribution_revenue = await self.collect_attribution_revenue_data(start_date, end_date)
        
        # Conversion data
        conversion_data = await self.collect_conversion_data(start_date, end_date)
        
        # Platform-specific revenue
        platform_revenue = await self.collect_platform_revenue_data(start_date, end_date)
        
        # Content-type revenue
        content_revenue = await self.collect_content_type_revenue_data(start_date, end_date)
        
        # Customer lifetime value data
        clv_data = await self.collect_customer_lifetime_value_data(start_date, end_date)
        
        return {
            'timeframe': timeframe.value,
            'date_range': {'start': start_date, 'end': end_date},
            'direct_revenue': direct_revenue,
            'attribution_revenue': attribution_revenue,
            'conversion_data': conversion_data,
            'platform_revenue': platform_revenue,
            'content_revenue': content_revenue,
            'customer_lifetime_value': clv_data,
            'revenue_trends': await self.calculate_revenue_trends(start_date, end_date)
        }
    
    async def collect_direct_revenue_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Collect direct revenue attributed to social media"""
        return {
            'total_direct_revenue': await self.revenue_calculator.calculate_direct_revenue(start_date, end_date),
            'revenue_by_source': await self.revenue_calculator.get_revenue_by_source(start_date, end_date),
            'revenue_by_campaign': await self.revenue_calculator.get_revenue_by_campaign(start_date, end_date),
            'revenue_by_content_type': await self.revenue_calculator.get_revenue_by_content_type(start_date, end_date),
            'revenue_by_business_niche': await self.revenue_calculator.get_revenue_by_niche(start_date, end_date),
            'subscription_revenue': await self.revenue_calculator.get_subscription_revenue(start_date, end_date),
            'one_time_sales_revenue': await self.revenue_calculator.get_one_time_sales(start_date, end_date),
            'lead_generation_revenue': await self.revenue_calculator.get_lead_gen_revenue(start_date, end_date)
        }
    
    async def collect_attribution_revenue_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Collect revenue using attribution modeling"""
        attribution_models = ['first_touch', 'last_touch', 'linear', 'time_decay', 'position_based']
        
        attribution_data = {}
        
        for model in attribution_models:
            attribution_data[model] = await self.attribution_engine.calculate_attributed_revenue(
                model=model,
                start_date=start_date,
                end_date=end_date,
                client_id=self.client_id
            )
        
        # Multi-touch attribution analysis
        attribution_data['multi_touch_analysis'] = await self.attribution_engine.analyze_multi_touch_attribution(
            start_date=start_date,
            end_date=end_date,
            client_id=self.client_id
        )
        
        # Platform attribution comparison
        attribution_data['platform_attribution'] = await self.attribution_engine.compare_platform_attribution(
            start_date=start_date,
            end_date=end_date,
            client_id=self.client_id
        )
        
        return attribution_data
    
    async def collect_conversion_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Collect conversion tracking data"""
        return {
            'total_conversions': 245,
            'conversion_rate': 3.5,
            'average_order_value': 125.50,
            'conversion_by_platform': {
                'instagram': 85,
                'tiktok': 65,
                'linkedin': 45,
                'youtube': 30,
                'twitter': 20
            },
            'conversion_by_content_type': {
                'video': 95,
                'carousel': 65,
                'single_image': 45,
                'story': 25,
                'reel': 15
            },
            'conversion_funnel': {
                'impressions': 100000,
                'clicks': 5000,
                'landing_page_views': 2500,
                'add_to_cart': 800,
                'checkout_started': 400,
                'purchases': 245
            }
        }
    
    async def collect_platform_revenue_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Collect platform-specific revenue data"""
        platforms = ['instagram', 'tiktok', 'linkedin', 'youtube', 'twitter', 'facebook', 'pinterest']
        platform_data = {}
        
        # Sample revenue distribution
        revenue_distribution = {
            'instagram': 0.35,
            'tiktok': 0.25,
            'linkedin': 0.15,
            'youtube': 0.10,
            'twitter': 0.05,
            'facebook': 0.07,
            'pinterest': 0.03
        }
        
        total_revenue = await self.revenue_calculator.calculate_direct_revenue(start_date, end_date)
        
        for platform in platforms:
            platform_revenue = total_revenue * revenue_distribution.get(platform, 0)
            platform_data[platform] = {
                'total_revenue': platform_revenue,
                'conversion_count': int(245 * revenue_distribution.get(platform, 0)),
                'average_order_value': 125.50,
                'revenue_per_post': platform_revenue / 50,  # Assume 50 posts per platform
                'top_performing_content': await self.get_top_revenue_content(platform, start_date, end_date),
                'revenue_growth_rate': 15.5 * (1 + revenue_distribution.get(platform, 0))
            }
        
        return platform_data
    
    async def collect_content_type_revenue_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Collect revenue data by content type"""
        content_types = ['video', 'carousel', 'single_image', 'story', 'reel']
        content_data = {}
        
        revenue_by_type = await self.revenue_calculator.get_revenue_by_content_type(start_date, end_date)
        
        for content_type in content_types:
            content_data[content_type] = {
                'total_revenue': revenue_by_type.get(content_type, 0),
                'average_revenue_per_post': revenue_by_type.get(content_type, 0) / 20,  # Assume 20 posts per type
                'conversion_rate': 3.5 * (1.2 if content_type == 'video' else 1.0),
                'engagement_to_revenue_ratio': 0.025,  # $0.025 per engagement
                'optimal_posting_frequency': await self.calculate_optimal_frequency(content_type)
            }
        
        return content_data
    
    async def collect_customer_lifetime_value_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Collect customer lifetime value data"""
        return {
            'average_clv': 850.0,
            'clv_by_acquisition_channel': {
                'instagram': 950.0,
                'tiktok': 750.0,
                'linkedin': 1100.0,
                'youtube': 900.0,
                'twitter': 650.0
            },
            'clv_by_customer_segment': {
                'high_value': 2500.0,
                'medium_value': 850.0,
                'low_value': 250.0
            },
            'retention_metrics': {
                'month_1_retention': 0.85,
                'month_3_retention': 0.65,
                'month_6_retention': 0.45,
                'month_12_retention': 0.30
            },
            'repeat_purchase_rate': 0.35,
            'average_purchases_per_customer': 2.8
        }
    
    async def calculate_revenue_trends(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Calculate revenue trend data"""
        days = (end_date - start_date).days
        daily_revenues = []
        
        # Generate sample trend data
        base_revenue = 500
        for i in range(days):
            # Add some variability and growth trend
            daily_revenue = base_revenue + (i * 5) + (50 * (i % 7))  # Weekly pattern
            daily_revenues.append(daily_revenue)
        
        return {
            'daily_revenues': daily_revenues,
            'weekly_revenues': [sum(daily_revenues[i:i+7]) for i in range(0, len(daily_revenues), 7)],
            'growth_rate': 15.5,
            'trend_direction': 'increasing',
            'seasonality_factor': 1.2,  # 20% seasonal boost
            'projected_next_period': sum(daily_revenues) * 1.155  # 15.5% growth
        }
    
    async def track_post_revenue_impact(self, post_id: str, platform: str, content: Dict[str, Any]) -> Dict[str, Any]:
        """Track revenue impact of a specific post"""
        try:
            # Start tracking from post publication
            tracking_start = datetime.now()
            
            # Set up conversion tracking
            await self.conversion_tracker.setup_post_tracking(
                post_id=post_id,
                platform=platform,
                content=content,
                tracking_start=tracking_start
            )
            
            # Track for 30 days (typical attribution window)
            tracking_results = await self.track_post_performance_over_time(
                post_id=post_id,
                platform=platform,
                tracking_duration_days=30
            )
            
            # Calculate revenue attribution
            revenue_attribution = await self.attribution_engine.calculate_post_attribution(
                post_id=post_id,
                platform=platform,
                tracking_results=tracking_results
            )
            
            # Analyze revenue factors
            revenue_factors = await self.analyze_revenue_factors(
                post_id=post_id,
                content=content,
                tracking_results=tracking_results
            )
            
            return {
                'post_id': post_id,
                'platform': platform,
                'tracking_period': {'start': tracking_start, 'duration_days': 30},
                'revenue_attribution': revenue_attribution,
                'revenue_factors': revenue_factors,
                'performance_metrics': tracking_results,
                'optimization_suggestions': await self.generate_post_optimization_suggestions(tracking_results)
            }
            
        except Exception as e:
            await self.log_revenue_tracking_error(f"Post revenue tracking failed for {post_id}: {str(e)}")
            return {'error': str(e), 'post_id': post_id}
    
    async def track_post_performance_over_time(self, post_id: str, platform: str, tracking_duration_days: int) -> Dict[str, Any]:
        """Track post performance over specified duration"""
        # In production, this would track actual performance
        # For now, return sample tracking data
        return {
            'total_impressions': 15000,
            'total_engagements': 750,
            'click_throughs': 225,
            'conversions': 12,
            'total_attributed_revenue': 1500.0,
            'direct_revenue': 900.0,
            'assisted_revenue': 600.0,
            'engagement_timeline': {
                'day_1': 400,
                'day_7': 200,
                'day_14': 100,
                'day_30': 50
            },
            'conversion_timeline': {
                'day_1': 5,
                'day_7': 4,
                'day_14': 2,
                'day_30': 1
            },
            'confidence_score': 0.85
        }
    
    async def analyze_revenue_factors(self, post_id: str, content: Dict[str, Any], tracking_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze factors contributing to post revenue"""
        return {
            'content_factors': {
                'content_type': content.get('type', 'unknown'),
                'hashtag_effectiveness': 0.75,
                'caption_sentiment': 'positive',
                'visual_appeal_score': 8.5,
                'call_to_action_strength': 'strong'
            },
            'timing_factors': {
                'day_of_week': 'Thursday',
                'time_of_day': '14:00',
                'seasonality_impact': 1.2
            },
            'audience_factors': {
                'audience_match_score': 0.85,
                'engagement_quality': 'high',
                'follower_value_segment': 'premium'
            },
            'competitive_factors': {
                'market_saturation': 'low',
                'competitor_activity': 'moderate',
                'uniqueness_score': 0.9
            }
        }
    
    async def generate_post_optimization_suggestions(self, tracking_results: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions for post revenue"""
        suggestions = []
        
        conversion_rate = tracking_results['conversions'] / tracking_results['click_throughs'] if tracking_results['click_throughs'] > 0 else 0
        
        if conversion_rate < 0.05:
            suggestions.append("Improve call-to-action clarity and placement")
            suggestions.append("A/B test different landing pages")
        
        if tracking_results['engagement_timeline']['day_1'] < 500:
            suggestions.append("Post during peak engagement hours")
            suggestions.append("Boost post in first 24 hours for momentum")
        
        if tracking_results['assisted_revenue'] > tracking_results['direct_revenue']:
            suggestions.append("This content works well as part of multi-touch journey")
            suggestions.append("Create follow-up content to capitalize on interest")
        
        return suggestions
    
    async def analyze_data(self, data: Dict[str, Any]) -> List[IntelligenceInsight]:
        """Analyze revenue data for actionable insights"""
        insights = []
        
        # Revenue trend analysis
        trend_insights = await self.analyze_revenue_trends(data['revenue_trends'])
        insights.extend(trend_insights)
        
        # Platform revenue performance
        platform_insights = await self.analyze_platform_revenue_performance(data['platform_revenue'])
        insights.extend(platform_insights)
        
        # Content type revenue analysis
        content_insights = await self.analyze_content_revenue_performance(data['content_revenue'])
        insights.extend(content_insights)
        
        # Attribution model insights
        attribution_insights = await self.analyze_attribution_models(data['attribution_revenue'])
        insights.extend(attribution_insights)
        
        # Conversion optimization insights
        conversion_insights = await self.analyze_conversion_opportunities(data['conversion_data'])
        insights.extend(conversion_insights)
        
        # Customer lifetime value insights
        clv_insights = await self.analyze_customer_lifetime_value(data['customer_lifetime_value'])
        insights.extend(clv_insights)
        
        return insights
    
    async def analyze_revenue_trends(self, trend_data: Dict[str, Any]) -> List[IntelligenceInsight]:
        """Analyze revenue trends for insights"""
        insights = []
        
        growth_rate = trend_data.get('growth_rate', 0)
        trend_direction = trend_data.get('trend_direction', 'stable')
        
        if growth_rate > 20:
            insights.append(IntelligenceInsight(
                metric_type=BusinessMetricType.REVENUE,
                insight_text=f"Revenue growing rapidly at {growth_rate}% - capitalize on momentum",
                confidence_score=0.92,
                impact_level="high",
                actionable_recommendations=[
                    "Scale successful campaigns while maintaining quality",
                    "Increase content production for high-revenue platforms",
                    "Invest in customer retention to maximize growth"
                ],
                supporting_data={'growth_rate': growth_rate, 'trend': trend_direction}
            ))
        elif growth_rate < 5:
            insights.append(IntelligenceInsight(
                metric_type=BusinessMetricType.REVENUE,
                insight_text=f"Revenue growth slowing to {growth_rate}% - intervention needed",
                confidence_score=0.88,
                impact_level="high",
                actionable_recommendations=[
                    "Analyze and replicate past high-revenue content",
                    "Test new content formats and platforms",
                    "Review pricing and offer strategies"
                ],
                supporting_data={'growth_rate': growth_rate, 'trend': trend_direction}
            ))
        
        # Seasonality insights
        seasonality_factor = trend_data.get('seasonality_factor', 1.0)
        if seasonality_factor > 1.15:
            insights.append(IntelligenceInsight(
                metric_type=BusinessMetricType.REVENUE,
                insight_text=f"Strong seasonal opportunity detected - {(seasonality_factor-1)*100:.0f}% boost potential",
                confidence_score=0.85,
                impact_level="medium",
                actionable_recommendations=[
                    "Prepare seasonal campaigns in advance",
                    "Increase inventory for seasonal demand",
                    "Create season-specific content calendars"
                ],
                supporting_data={'seasonality_factor': seasonality_factor}
            ))
        
        return insights
    
    async def analyze_platform_revenue_performance(self, platform_data: Dict[str, Any]) -> List[IntelligenceInsight]:
        """Analyze platform-specific revenue performance"""
        insights = []
        
        # Find best and worst performing platforms
        platform_revenues = {platform: data['total_revenue'] for platform, data in platform_data.items()}
        best_platform = max(platform_revenues.items(), key=lambda x: x[1])[0]
        worst_platform = min(platform_revenues.items(), key=lambda x: x[1])[0]
        
        # Best platform insights
        best_revenue = platform_revenues[best_platform]
        best_data = platform_data[best_platform]
        
        insights.append(IntelligenceInsight(
            metric_type=BusinessMetricType.REVENUE,
            insight_text=f"{best_platform.title()} generating ${best_revenue:,.2f} - your revenue champion",
            confidence_score=0.94,
            impact_level="high",
            actionable_recommendations=[
                f"Increase {best_platform.title()} content frequency by 30%",
                f"Study top-performing {best_platform.title()} content for patterns",
                f"Allocate more ad budget to {best_platform.title()}"
            ],
            supporting_data={
                'platform': best_platform,
                'revenue': best_revenue,
                'metrics': best_data
            }
        ))
        
        # Underperforming platform insights
        worst_revenue = platform_revenues[worst_platform]
        if worst_revenue < best_revenue * 0.2:
            insights.append(IntelligenceInsight(
                metric_type=BusinessMetricType.REVENUE,
                insight_text=f"{worst_platform.title()} underperforming at ${worst_revenue:,.2f}",
                confidence_score=0.87,
                impact_level="medium",
                actionable_recommendations=[
                    f"Audit {worst_platform.title()} strategy and content",
                    f"Test {best_platform.title()} successful content on {worst_platform.title()}",
                    f"Consider reducing {worst_platform.title()} investment"
                ],
                supporting_data={
                    'platform': worst_platform,
                    'revenue': worst_revenue,
                    'gap': best_revenue - worst_revenue
                }
            ))
        
        return insights
    
    async def analyze_content_revenue_performance(self, content_data: Dict[str, Any]) -> List[IntelligenceInsight]:
        """Analyze content type revenue performance"""
        insights = []
        
        # Find most profitable content type
        content_revenues = {ctype: data['total_revenue'] for ctype, data in content_data.items()}
        best_content_type = max(content_revenues.items(), key=lambda x: x[1])[0]
        
        insights.append(IntelligenceInsight(
            metric_type=BusinessMetricType.REVENUE,
            insight_text=f"{best_content_type.title()} content driving ${content_revenues[best_content_type]:,.2f} in revenue",
            confidence_score=0.89,
            impact_level="high",
            actionable_recommendations=[
                f"Prioritize {best_content_type} content creation",
                f"Train team on {best_content_type} best practices",
                f"Invest in {best_content_type} production tools"
            ],
            supporting_data={'content_revenues': content_revenues}
        ))
        
        # Content mix optimization
        total_content_revenue = sum(content_revenues.values())
        content_distribution = {ctype: revenue/total_content_revenue for ctype, revenue in content_revenues.items()}
        
        # Check if content mix is too concentrated
        max_concentration = max(content_distribution.values())
        if max_concentration > 0.5:
            insights.append(IntelligenceInsight(
                metric_type=BusinessMetricType.REVENUE,
                insight_text=f"Revenue overly concentrated in one content type ({max_concentration*100:.0f}%)",
                confidence_score=0.82,
                impact_level="medium",
                actionable_recommendations=[
                    "Diversify content portfolio for risk mitigation",
                    "Test new content formats with small budgets",
                    "Create hybrid content combining successful elements"
                ],
                supporting_data={'content_distribution': content_distribution}
            ))
        
        return insights
    
    async def analyze_attribution_models(self, attribution_data: Dict[str, Any]) -> List[IntelligenceInsight]:
        """Analyze attribution model results"""
        insights = []
        
        multi_touch = attribution_data.get('multi_touch_analysis', {})
        recommended_model = multi_touch.get('recommended_model', 'linear')
        
        insights.append(IntelligenceInsight(
            metric_type=BusinessMetricType.REVENUE,
            insight_text=f"Attribution analysis recommends '{recommended_model}' model for most accurate revenue tracking",
            confidence_score=0.86,
            impact_level="medium",
            actionable_recommendations=[
                f"Implement {recommended_model} attribution in reporting",
                "Track customer journey touchpoints more comprehensively",
                "Adjust platform budgets based on attribution insights"
            ],
            supporting_data={'attribution_analysis': multi_touch}
        ))
        
        # Platform attribution insights
        platform_attribution = attribution_data.get('platform_attribution', {})
        if platform_attribution:
            high_variance_platforms = [
                platform for platform, data in platform_attribution.items()
                if data.get('attribution_variance', 0) > 500
            ]
            
            if high_variance_platforms:
                insights.append(IntelligenceInsight(
                    metric_type=BusinessMetricType.REVENUE,
                    insight_text=f"High attribution variance on {', '.join(high_variance_platforms)} - complex customer journeys",
                    confidence_score=0.83,
                    impact_level="low",
                    actionable_recommendations=[
                        "Implement more sophisticated tracking for these platforms",
                        "Consider multi-touch attribution for better accuracy",
                        "Analyze full customer journey paths"
                    ],
                    supporting_data={'high_variance_platforms': high_variance_platforms}
                ))
        
        return insights
    
    async def analyze_conversion_opportunities(self, conversion_data: Dict[str, Any]) -> List[IntelligenceInsight]:
        """Analyze conversion funnel for optimization opportunities"""
        insights = []
        
        conversion_rate = conversion_data.get('conversion_rate', 0)
        avg_order_value = conversion_data.get('average_order_value', 0)
        
        # Conversion rate insights
        if conversion_rate < 2:
            insights.append(IntelligenceInsight(
                metric_type=BusinessMetricType.CONVERSION,
                insight_text=f"Conversion rate at {conversion_rate}% - significant improvement potential",
                confidence_score=0.87,
                impact_level="high",
                actionable_recommendations=[
                    "Optimize landing pages for conversions",
                    "Implement retargeting for engaged users",
                    "A/B test different offers and CTAs"
                ],
                supporting_data={'conversion_rate': conversion_rate}
            ))
        
        # Funnel analysis
        funnel = conversion_data.get('conversion_funnel', {})
        if funnel:
            # Calculate drop-off rates
            impressions_to_clicks = (funnel.get('clicks', 0) / funnel.get('impressions', 1)) * 100
            clicks_to_landing = (funnel.get('landing_page_views', 0) / funnel.get('clicks', 1)) * 100
            
            if impressions_to_clicks < 5:
                insights.append(IntelligenceInsight(
                    metric_type=BusinessMetricType.CONVERSION,
                    insight_text=f"Low click-through rate ({impressions_to_clicks:.1f}%) limiting revenue potential",
                    confidence_score=0.91,
                    impact_level="high",
                    actionable_recommendations=[
                        "Improve ad creative and messaging",
                        "Test different audience targeting",
                        "Optimize posting times for engagement"
                    ],
                    supporting_data={'ctr': impressions_to_clicks, 'funnel': funnel}
                ))
        
        return insights
    
    async def analyze_customer_lifetime_value(self, clv_data: Dict[str, Any]) -> List[IntelligenceInsight]:
        """Analyze customer lifetime value data"""
        insights = []
        
        avg_clv = clv_data.get('average_clv', 0)
        clv_by_channel = clv_data.get('clv_by_acquisition_channel', {})
        
        # Find highest CLV channel
        if clv_by_channel:
            best_clv_channel = max(clv_by_channel.items(), key=lambda x: x[1])[0]
            best_clv = clv_by_channel[best_clv_channel]
            
            if best_clv > avg_clv * 1.2:
                insights.append(IntelligenceInsight(
                    metric_type=BusinessMetricType.REVENUE,
                    insight_text=f"{best_clv_channel.title()} customers have {((best_clv/avg_clv)-1)*100:.0f}% higher lifetime value",
                    confidence_score=0.88,
                    impact_level="high",
                    actionable_recommendations=[
                        f"Focus acquisition efforts on {best_clv_channel}",
                        f"Study {best_clv_channel} customer behavior for insights",
                        "Create premium offerings for high-CLV segments"
                    ],
                    supporting_data={'clv_by_channel': clv_by_channel, 'average_clv': avg_clv}
                ))
        
        # Retention insights
        retention = clv_data.get('retention_metrics', {})
        month_3_retention = retention.get('month_3_retention', 0)
        
        if month_3_retention < 0.5:
            insights.append(IntelligenceInsight(
                metric_type=BusinessMetricType.REVENUE,
                insight_text=f"Only {month_3_retention*100:.0f}% customer retention after 3 months - revenue leakage",
                confidence_score=0.85,
                impact_level="high",
                actionable_recommendations=[
                    "Implement customer retention program",
                    "Create engaging post-purchase content",
                    "Develop loyalty rewards system"
                ],
                supporting_data={'retention_metrics': retention}
            ))
        
        return insights
    
    async def generate_recommendations(self, insights: List[IntelligenceInsight]) -> List[str]:
        """Generate revenue optimization recommendations"""
        recommendations = []
        
        # Extract all recommendations from insights
        for insight in insights:
            recommendations.extend(insight.actionable_recommendations)
        
        # Add revenue-specific strategic recommendations
        revenue_recommendations = await self.generate_revenue_optimization_suggestions(insights)
        recommendations.extend(revenue_recommendations)
        
        # Remove duplicates and prioritize
        unique_recommendations = list(set(recommendations))
        prioritized_recommendations = await self.prioritize_recommendations(unique_recommendations, insights)
        
        return prioritized_recommendations
    
    async def generate_revenue_optimization_suggestions(self, insights: List[IntelligenceInsight]) -> List[str]:
        """Generate specific revenue optimization suggestions"""
        suggestions = []
        
        # Count revenue-related insights
        revenue_insights = [i for i in insights if i.metric_type == BusinessMetricType.REVENUE]
        
        if len(revenue_insights) > 3:
            suggestions.append("Create comprehensive revenue optimization roadmap")
            suggestions.append("Implement weekly revenue performance reviews")
            suggestions.append("Establish revenue targets by platform and content type")
        
        # High-impact revenue opportunities
        high_impact_revenue = [i for i in revenue_insights if i.impact_level == "high"]
        if high_impact_revenue:
            suggestions.append("Prioritize high-impact revenue initiatives immediately")
            suggestions.append("Allocate resources to proven revenue drivers")
        
        return suggestions
    
    async def get_top_revenue_content(self, platform: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get top revenue-generating content for a platform"""
        # Sample data - in production would query actual content
        return [
            {
                'content_id': 'post_123',
                'revenue': 450.0,
                'content_type': 'video',
                'engagement_rate': 5.2
            },
            {
                'content_id': 'post_124',
                'revenue': 380.0,
                'content_type': 'carousel',
                'engagement_rate': 4.8
            }
        ]
    
    async def calculate_optimal_frequency(self, content_type: str) -> Dict[str, Any]:
        """Calculate optimal posting frequency for content type"""
        frequency_map = {
            'video': {'posts_per_week': 3, 'optimal_days': ['Monday', 'Wednesday', 'Friday']},
            'carousel': {'posts_per_week': 4, 'optimal_days': ['Tuesday', 'Thursday', 'Saturday', 'Sunday']},
            'single_image': {'posts_per_week': 7, 'optimal_days': ['Daily']},
            'story': {'posts_per_week': 14, 'optimal_days': ['Daily', 'Multiple']},
            'reel': {'posts_per_week': 5, 'optimal_days': ['Weekdays']}
        }
        
        return frequency_map.get(content_type, {'posts_per_week': 5, 'optimal_days': ['Weekdays']})
    
    async def log_revenue_tracking_error(self, error_message: str):
        """Log revenue tracking errors"""
        logger.error(f"Revenue Tracking Error: {error_message}")