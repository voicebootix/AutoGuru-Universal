"""Performance Monitoring System - Real-time performance tracking with predictive alerts"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import logging
import statistics
from collections import deque, defaultdict

from .base_intelligence import (
    UniversalIntelligenceEngine,
    AnalyticsTimeframe,
    BusinessMetricType,
    IntelligenceInsight,
    IntelligenceEngineError
)

logger = logging.getLogger(__name__)

class PerformanceStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DEGRADED = "degraded"

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    metric_name: str
    value: float
    unit: str
    status: PerformanceStatus
    threshold_warning: float
    threshold_critical: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PerformanceAlert:
    """Performance alert structure"""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    message: str
    metric_data: Dict[str, Any]
    recommended_actions: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

class MetricCollector:
    """Collect various performance metrics"""
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        
    async def get_api_response_times(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get API response time metrics"""
        # In production, this would query actual monitoring data
        return {
            'average_response_time_ms': 125.5,
            'p95_response_time_ms': 250.0,
            'p99_response_time_ms': 500.0,
            'max_response_time_ms': 1200.0,
            'endpoints': {
                '/api/content/create': {'avg_ms': 180.0, 'calls': 1500},
                '/api/analytics/get': {'avg_ms': 95.0, 'calls': 3200},
                '/api/platforms/publish': {'avg_ms': 320.0, 'calls': 890}
            }
        }
    
    async def get_system_uptime(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get system uptime metrics"""
        total_minutes = (end_date - start_date).total_seconds() / 60
        downtime_minutes = 12  # Example downtime
        
        return {
            'uptime_percentage': ((total_minutes - downtime_minutes) / total_minutes * 100),
            'total_downtime_minutes': downtime_minutes,
            'incidents': [
                {
                    'timestamp': start_date + timedelta(days=5),
                    'duration_minutes': 8,
                    'cause': 'Database maintenance',
                    'impact': 'Reduced functionality'
                },
                {
                    'timestamp': start_date + timedelta(days=12),
                    'duration_minutes': 4,
                    'cause': 'API rate limit',
                    'impact': 'Delayed posting'
                }
            ]
        }
    
    async def get_error_rates(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get error rate metrics"""
        return {
            'overall_error_rate': 0.23,  # 0.23% error rate
            'error_by_type': {
                '4xx_errors': 0.15,
                '5xx_errors': 0.08,
                'timeout_errors': 0.05,
                'api_errors': 0.10
            },
            'error_trends': {
                'increasing': False,
                'rate_change_percentage': -12.5  # Decreasing by 12.5%
            }
        }
    
    async def get_throughput_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get system throughput metrics"""
        return {
            'requests_per_second': 45.8,
            'posts_per_hour': 320,
            'content_generated_per_hour': 180,
            'peak_throughput': {
                'timestamp': start_date + timedelta(days=7, hours=14),
                'requests_per_second': 125.5
            }
        }
    
    async def get_resource_utilization(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get resource utilization metrics"""
        return {
            'cpu_usage': {
                'average_percentage': 45.2,
                'peak_percentage': 78.5,
                'trend': 'stable'
            },
            'memory_usage': {
                'average_percentage': 62.1,
                'peak_percentage': 85.0,
                'available_gb': 8.5
            },
            'storage_usage': {
                'used_percentage': 55.3,
                'available_gb': 445.7,
                'growth_rate_gb_per_month': 12.5
            },
            'database_connections': {
                'average_active': 25,
                'peak_active': 85,
                'connection_pool_size': 100
            }
        }
    
    async def get_concurrent_capacity(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get concurrent user capacity metrics"""
        return {
            'average_concurrent_users': 125,
            'peak_concurrent_users': 450,
            'capacity_limit': 1000,
            'capacity_utilization_percentage': 45.0
        }
    
    async def get_processing_speeds(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get data processing speed metrics"""
        return {
            'content_analysis_speed': {
                'average_seconds': 2.5,
                'items_per_minute': 24
            },
            'image_generation_speed': {
                'average_seconds': 8.5,
                'items_per_minute': 7
            },
            'viral_optimization_speed': {
                'average_seconds': 1.8,
                'items_per_minute': 33
            }
        }
    
    async def get_ai_model_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get AI model performance metrics"""
        return {
            'model_accuracy': {
                'content_classification': 94.5,
                'engagement_prediction': 87.2,
                'viral_potential_scoring': 82.8
            },
            'model_latency': {
                'average_ms': 150,
                'p95_ms': 320,
                'p99_ms': 550
            },
            'model_throughput': {
                'predictions_per_second': 85
            }
        }
    
    async def get_content_generation_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get content generation performance metrics"""
        return {
            'average_generation_time_seconds': 4.5,
            'success_rate_percentage': 98.5,
            'quality_score_average': 8.7,
            'generation_by_type': {
                'text': {'avg_seconds': 2.1, 'success_rate': 99.2},
                'image': {'avg_seconds': 8.5, 'success_rate': 97.5},
                'video': {'avg_seconds': 45.0, 'success_rate': 95.8}
            }
        }
    
    async def get_viral_optimization_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get viral optimization performance metrics"""
        return {
            'optimization_success_rate': 78.5,
            'average_score_improvement': 35.2,
            'viral_hit_rate': 12.5,  # % of content that goes viral
            'engagement_lift_percentage': 145.0
        }
    
    async def get_hashtag_performance_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get hashtag generation performance metrics"""
        return {
            'generation_accuracy': 89.5,
            'trending_hashtag_hit_rate': 65.2,
            'average_hashtags_per_post': 8.5,
            'engagement_correlation': 0.72
        }
    
    async def get_image_generation_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get image generation performance metrics"""
        return {
            'generation_success_rate': 96.8,
            'average_generation_time_seconds': 8.5,
            'quality_score_average': 8.9,
            'style_accuracy': 91.2
        }
    
    async def get_content_quality_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get content quality metrics"""
        return {
            'overall_quality_score': 8.6,
            'quality_by_type': {
                'text': 8.8,
                'image': 8.5,
                'video': 8.3
            },
            'quality_trend': 'improving',
            'quality_improvement_rate': 5.2  # % improvement
        }
    
    async def get_personalization_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get personalization effectiveness metrics"""
        return {
            'personalization_accuracy': 85.5,
            'engagement_lift_from_personalization': 48.2,
            'persona_match_rate': 91.0,
            'conversion_improvement': 25.5
        }
    
    async def get_adaptation_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get cross-platform adaptation metrics"""
        return {
            'adaptation_success_rate': 94.2,
            'content_consistency_score': 87.5,
            'platform_optimization_effectiveness': {
                'instagram': 92.5,
                'tiktok': 89.2,
                'linkedin': 91.8,
                'twitter': 88.5
            }
        }

class AnomalyDetector:
    """Detect anomalies in performance metrics"""
    
    def __init__(self):
        self.baseline_metrics = {}
        self.anomaly_thresholds = {
            'response_time': 2.5,  # Standard deviations
            'error_rate': 3.0,
            'throughput': 2.0,
            'revenue': 2.5
        }
    
    async def detect_response_time_anomalies(self, response_times: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in API response times"""
        anomalies = []
        
        avg_response_time = response_times.get('average_response_time_ms', 0)
        baseline_avg = 100  # Expected baseline
        
        if avg_response_time > baseline_avg * 2:
            anomalies.append({
                'type': 'response_time_spike',
                'severity': 'high',
                'current_value': avg_response_time,
                'expected_value': baseline_avg,
                'deviation_percentage': ((avg_response_time - baseline_avg) / baseline_avg * 100),
                'impact': 'User experience degradation',
                'recommended_actions': [
                    'Check server load and scale if needed',
                    'Review recent deployments for performance issues',
                    'Analyze slow query logs'
                ]
            })
        
        # Check for endpoint-specific anomalies
        endpoints = response_times.get('endpoints', {})
        for endpoint, metrics in endpoints.items():
            if metrics['avg_ms'] > 300:  # Threshold for slow endpoints
                anomalies.append({
                    'type': 'slow_endpoint',
                    'severity': 'medium',
                    'endpoint': endpoint,
                    'response_time_ms': metrics['avg_ms'],
                    'call_volume': metrics['calls'],
                    'recommended_actions': [
                        f'Optimize {endpoint} endpoint logic',
                        'Consider caching for this endpoint',
                        'Review database queries used by this endpoint'
                    ]
                })
        
        return anomalies
    
    async def detect_revenue_anomalies(self, revenue_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in revenue performance"""
        anomalies = []
        
        # This would use statistical methods in production
        # For now, simple threshold-based detection
        daily_revenue = revenue_data.get('daily_revenue', [])
        if daily_revenue:
            avg_revenue = statistics.mean(daily_revenue) if daily_revenue else 0
            std_dev = statistics.stdev(daily_revenue) if len(daily_revenue) > 1 else 0
            
            # Check for sudden drops
            recent_revenue = daily_revenue[-1] if daily_revenue else 0
            if recent_revenue < avg_revenue - (2 * std_dev):
                anomalies.append({
                    'type': 'revenue_drop',
                    'severity': 'critical',
                    'current_revenue': recent_revenue,
                    'expected_range': (avg_revenue - std_dev, avg_revenue + std_dev),
                    'drop_percentage': ((avg_revenue - recent_revenue) / avg_revenue * 100),
                    'recommended_actions': [
                        'Check platform API connectivity',
                        'Review content publishing success rates',
                        'Analyze conversion funnel for issues'
                    ]
                })
        
        return anomalies
    
    async def detect_engagement_anomalies(self, engagement_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in engagement metrics"""
        anomalies = []
        
        engagement_rate = engagement_data.get('engagement_rate', 0)
        baseline_engagement = 5.0  # Expected 5% engagement rate
        
        if engagement_rate < baseline_engagement * 0.5:
            anomalies.append({
                'type': 'low_engagement',
                'severity': 'medium',
                'current_rate': engagement_rate,
                'expected_rate': baseline_engagement,
                'decline_percentage': ((baseline_engagement - engagement_rate) / baseline_engagement * 100),
                'recommended_actions': [
                    'Review content quality and relevance',
                    'Check posting times and frequency',
                    'Analyze audience demographics shifts'
                ]
            })
        
        return anomalies
    
    async def detect_quality_anomalies(self, quality_scores: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in content quality"""
        anomalies = []
        
        overall_quality = quality_scores.get('overall_quality_score', 0)
        quality_threshold = 7.5
        
        if overall_quality < quality_threshold:
            anomalies.append({
                'type': 'quality_degradation',
                'severity': 'medium',
                'current_score': overall_quality,
                'threshold': quality_threshold,
                'quality_by_type': quality_scores.get('quality_by_type', {}),
                'recommended_actions': [
                    'Review AI model performance',
                    'Check content generation prompts',
                    'Analyze user feedback on recent content'
                ]
            })
        
        return anomalies

class AlertManager:
    """Manage performance alerts and notifications"""
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.active_alerts = []
        self.alert_history = deque(maxlen=1000)
        
    async def send_alert(self, alert_type: str, severity: str, message: str, data: Dict[str, Any]):
        """Send performance alert"""
        alert = PerformanceAlert(
            alert_id=f"alert_{datetime.now().timestamp()}",
            alert_type=alert_type,
            severity=AlertSeverity(severity),
            message=message,
            metric_data=data,
            recommended_actions=data.get('recommended_actions', [])
        )
        
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        # In production, this would send actual notifications
        logger.warning(f"Performance Alert: {alert_type} - {message}")
        
        return alert

class PerformanceMonitoringSystem(UniversalIntelligenceEngine):
    """Real-time performance monitoring with predictive alerts"""
    
    def __init__(self, client_id: str):
        super().__init__(client_id, "performance_monitoring")
        self.metric_collector = MetricCollector(client_id)
        self.anomaly_detector = AnomalyDetector()
        self.alert_manager = AlertManager(client_id)
        self.monitoring_active = False
        
    async def collect_data(self, timeframe: AnalyticsTimeframe) -> Dict[str, Any]:
        """Collect comprehensive performance data across all systems"""
        end_date = datetime.now()
        start_date = await self.get_timeframe_start_date(end_date, timeframe)
        
        # System performance metrics
        system_metrics = await self.collect_system_performance_metrics(start_date, end_date)
        
        # Content performance metrics
        content_metrics = await self.collect_content_performance_metrics(start_date, end_date)
        
        # Revenue performance metrics
        revenue_metrics = await self.collect_revenue_performance_metrics(start_date, end_date)
        
        # User engagement metrics
        engagement_metrics = await self.collect_engagement_performance_metrics(start_date, end_date)
        
        # Platform API performance
        api_performance = await self.collect_api_performance_metrics(start_date, end_date)
        
        return {
            'timeframe': timeframe.value,
            'date_range': {'start': start_date, 'end': end_date},
            'system_performance': system_metrics,
            'content_performance': content_metrics,
            'revenue_performance': revenue_metrics,
            'engagement_performance': engagement_metrics,
            'api_performance': api_performance,
            'real_time_status': await self.get_real_time_system_status()
        }
    
    async def collect_system_performance_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Collect system-level performance metrics"""
        return {
            'api_response_times': await self.metric_collector.get_api_response_times(start_date, end_date),
            'system_uptime': await self.metric_collector.get_system_uptime(start_date, end_date),
            'error_rates': await self.metric_collector.get_error_rates(start_date, end_date),
            'throughput_metrics': await self.metric_collector.get_throughput_metrics(start_date, end_date),
            'resource_utilization': await self.metric_collector.get_resource_utilization(start_date, end_date),
            'concurrent_user_capacity': await self.metric_collector.get_concurrent_capacity(start_date, end_date),
            'data_processing_speed': await self.metric_collector.get_processing_speeds(start_date, end_date),
            'ai_model_performance': await self.metric_collector.get_ai_model_metrics(start_date, end_date)
        }
    
    async def collect_content_performance_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Collect content creation and optimization performance"""
        return {
            'content_generation_speed': await self.metric_collector.get_content_generation_metrics(start_date, end_date),
            'viral_optimization_success_rate': await self.metric_collector.get_viral_optimization_metrics(start_date, end_date),
            'hashtag_performance': await self.metric_collector.get_hashtag_performance_metrics(start_date, end_date),
            'image_generation_metrics': await self.metric_collector.get_image_generation_metrics(start_date, end_date),
            'content_quality_scores': await self.metric_collector.get_content_quality_metrics(start_date, end_date),
            'personalization_effectiveness': await self.metric_collector.get_personalization_metrics(start_date, end_date),
            'cross_platform_adaptation': await self.metric_collector.get_adaptation_metrics(start_date, end_date)
        }
    
    async def collect_revenue_performance_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Collect revenue performance metrics"""
        # Generate sample daily revenue data
        days = (end_date - start_date).days
        daily_revenue = [1000 + (i * 50) + (100 * (i % 7)) for i in range(days)]
        
        return {
            'total_revenue': sum(daily_revenue),
            'daily_revenue': daily_revenue,
            'revenue_growth_rate': 15.5,
            'revenue_per_platform': {
                'instagram': 4500.0,
                'tiktok': 3200.0,
                'linkedin': 2800.0,
                'youtube': 2100.0
            },
            'conversion_metrics': {
                'conversion_rate': 3.5,
                'average_order_value': 125.0,
                'customer_lifetime_value': 850.0
            }
        }
    
    async def collect_engagement_performance_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Collect engagement performance metrics"""
        return {
            'engagement_rate': 4.8,
            'average_likes_per_post': 250,
            'average_comments_per_post': 35,
            'average_shares_per_post': 45,
            'follower_growth_rate': 8.5,
            'audience_retention_rate': 85.2
        }
    
    async def collect_api_performance_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Collect platform API performance metrics"""
        return {
            'instagram_api': {
                'availability': 99.5,
                'average_latency_ms': 120,
                'rate_limit_usage': 65.0
            },
            'tiktok_api': {
                'availability': 99.2,
                'average_latency_ms': 150,
                'rate_limit_usage': 45.0
            },
            'linkedin_api': {
                'availability': 99.8,
                'average_latency_ms': 95,
                'rate_limit_usage': 30.0
            }
        }
    
    async def get_real_time_system_status(self) -> Dict[str, Any]:
        """Get real-time system status"""
        return {
            'overall_status': 'healthy',
            'active_users': 125,
            'active_processes': 45,
            'queue_depth': 12,
            'last_check': datetime.now(),
            'critical_alerts': len([a for a in self.alert_manager.active_alerts if a.severity == AlertSeverity.CRITICAL])
        }
    
    async def analyze_data(self, data: Dict[str, Any]) -> List[IntelligenceInsight]:
        """Analyze performance data for insights and alerts"""
        insights = []
        
        # System performance insights
        system_insights = await self.analyze_system_performance(data['system_performance'])
        insights.extend(system_insights)
        
        # Content performance insights
        content_insights = await self.analyze_content_performance(data['content_performance'])
        insights.extend(content_insights)
        
        # Revenue performance insights
        revenue_insights = await self.analyze_revenue_performance(data['revenue_performance'])
        insights.extend(revenue_insights)
        
        # Anomaly detection
        anomalies = await self.detect_performance_anomalies(data)
        anomaly_insights = await self.convert_anomalies_to_insights(anomalies)
        insights.extend(anomaly_insights)
        
        # Predictive performance insights
        predictive_insights = await self.generate_predictive_performance_insights(data)
        insights.extend(predictive_insights)
        
        return insights
    
    async def analyze_system_performance(self, system_data: Dict[str, Any]) -> List[IntelligenceInsight]:
        """Analyze system performance metrics"""
        insights = []
        
        # API response time analysis
        response_times = system_data['api_response_times']
        avg_response = response_times['average_response_time_ms']
        
        if avg_response > 200:
            insights.append(IntelligenceInsight(
                metric_type=BusinessMetricType.EFFICIENCY,
                insight_text=f"API response times averaging {avg_response}ms - above optimal threshold",
                confidence_score=0.92,
                impact_level="medium",
                actionable_recommendations=[
                    "Implement caching for frequently accessed endpoints",
                    "Optimize database queries and indexes",
                    "Consider upgrading server infrastructure"
                ],
                supporting_data=response_times
            ))
        
        # Resource utilization analysis
        resources = system_data['resource_utilization']
        cpu_usage = resources['cpu_usage']['average_percentage']
        memory_usage = resources['memory_usage']['average_percentage']
        
        if cpu_usage > 70 or memory_usage > 80:
            insights.append(IntelligenceInsight(
                metric_type=BusinessMetricType.EFFICIENCY,
                insight_text=f"High resource utilization detected: CPU {cpu_usage}%, Memory {memory_usage}%",
                confidence_score=0.95,
                impact_level="high",
                actionable_recommendations=[
                    "Scale infrastructure horizontally",
                    "Optimize resource-intensive processes",
                    "Implement auto-scaling policies"
                ],
                supporting_data=resources
            ))
        
        # Uptime analysis
        uptime = system_data['system_uptime']
        uptime_percentage = uptime['uptime_percentage']
        
        if uptime_percentage < 99.9:
            insights.append(IntelligenceInsight(
                metric_type=BusinessMetricType.EFFICIENCY,
                insight_text=f"System uptime at {uptime_percentage:.2f}% - below target SLA",
                confidence_score=0.98,
                impact_level="high",
                actionable_recommendations=[
                    "Implement redundancy and failover systems",
                    "Review and address root causes of downtime",
                    "Establish proactive monitoring and alerting"
                ],
                supporting_data=uptime
            ))
        
        return insights
    
    async def analyze_content_performance(self, content_data: Dict[str, Any]) -> List[IntelligenceInsight]:
        """Analyze content performance metrics"""
        insights = []
        
        # Content generation speed
        gen_speed = content_data['content_generation_speed']
        avg_gen_time = gen_speed['average_generation_time_seconds']
        
        if avg_gen_time > 5:
            insights.append(IntelligenceInsight(
                metric_type=BusinessMetricType.EFFICIENCY,
                insight_text=f"Content generation taking {avg_gen_time}s on average - optimization needed",
                confidence_score=0.88,
                impact_level="medium",
                actionable_recommendations=[
                    "Optimize AI model inference",
                    "Implement content generation queuing",
                    "Use GPU acceleration for AI models"
                ],
                supporting_data=gen_speed
            ))
        
        # Content quality analysis
        quality_scores = content_data['content_quality_scores']
        overall_quality = quality_scores['overall_quality_score']
        
        if overall_quality < 8:
            insights.append(IntelligenceInsight(
                metric_type=BusinessMetricType.ENGAGEMENT,
                insight_text=f"Content quality score at {overall_quality}/10 - room for improvement",
                confidence_score=0.85,
                impact_level="medium",
                actionable_recommendations=[
                    "Fine-tune AI models with high-performing content",
                    "Implement stricter quality filters",
                    "Analyze top-performing content patterns"
                ],
                supporting_data=quality_scores
            ))
        
        # Viral optimization success
        viral_metrics = content_data['viral_optimization_success_rate']
        viral_success = viral_metrics['optimization_success_rate']
        
        if viral_success > 75:
            insights.append(IntelligenceInsight(
                metric_type=BusinessMetricType.ENGAGEMENT,
                insight_text=f"Viral optimization achieving {viral_success}% success rate - excellent performance",
                confidence_score=0.91,
                impact_level="high",
                actionable_recommendations=[
                    "Document viral optimization best practices",
                    "Expand viral features to more content types",
                    "Create case studies of viral successes"
                ],
                supporting_data=viral_metrics
            ))
        
        return insights
    
    async def analyze_revenue_performance(self, revenue_data: Dict[str, Any]) -> List[IntelligenceInsight]:
        """Analyze revenue performance metrics"""
        insights = []
        
        # Revenue growth analysis
        growth_rate = revenue_data['revenue_growth_rate']
        
        if growth_rate > 10:
            insights.append(IntelligenceInsight(
                metric_type=BusinessMetricType.REVENUE,
                insight_text=f"Revenue growing at {growth_rate}% - strong performance",
                confidence_score=0.93,
                impact_level="high",
                actionable_recommendations=[
                    "Maintain current growth strategies",
                    "Identify and replicate success factors",
                    "Prepare infrastructure for increased scale"
                ],
                supporting_data={'growth_rate': growth_rate}
            ))
        elif growth_rate < 5:
            insights.append(IntelligenceInsight(
                metric_type=BusinessMetricType.REVENUE,
                insight_text=f"Revenue growth slowing to {growth_rate}% - intervention needed",
                confidence_score=0.87,
                impact_level="high",
                actionable_recommendations=[
                    "Analyze conversion funnel for bottlenecks",
                    "Test new pricing strategies",
                    "Expand to new platforms or markets"
                ],
                supporting_data={'growth_rate': growth_rate}
            ))
        
        # Platform revenue distribution
        platform_revenue = revenue_data['revenue_per_platform']
        total_revenue = sum(platform_revenue.values())
        
        for platform, revenue in platform_revenue.items():
            percentage = (revenue / total_revenue * 100) if total_revenue > 0 else 0
            if percentage > 40:
                insights.append(IntelligenceInsight(
                    metric_type=BusinessMetricType.REVENUE,
                    insight_text=f"{platform.title()} contributing {percentage:.1f}% of revenue - high concentration risk",
                    confidence_score=0.86,
                    impact_level="medium",
                    actionable_recommendations=[
                        "Diversify revenue sources across platforms",
                        f"Develop contingency plans for {platform} changes",
                        "Invest in growing other platform revenues"
                    ],
                    supporting_data={'platform_revenue': platform_revenue}
                ))
        
        return insights
    
    async def detect_performance_anomalies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect performance anomalies using ML"""
        anomalies = []
        
        # API response time anomalies
        api_response_times = data['system_performance']['api_response_times']
        response_time_anomalies = await self.anomaly_detector.detect_response_time_anomalies(api_response_times)
        anomalies.extend(response_time_anomalies)
        
        # Revenue performance anomalies
        revenue_data = data['revenue_performance']
        revenue_anomalies = await self.anomaly_detector.detect_revenue_anomalies(revenue_data)
        anomalies.extend(revenue_anomalies)
        
        # Engagement rate anomalies
        engagement_data = data['engagement_performance']
        engagement_anomalies = await self.anomaly_detector.detect_engagement_anomalies(engagement_data)
        anomalies.extend(engagement_anomalies)
        
        # Content quality anomalies
        content_quality = data['content_performance']['content_quality_scores']
        quality_anomalies = await self.anomaly_detector.detect_quality_anomalies(content_quality)
        anomalies.extend(quality_anomalies)
        
        return anomalies
    
    async def convert_anomalies_to_insights(self, anomalies: List[Dict[str, Any]]) -> List[IntelligenceInsight]:
        """Convert detected anomalies to actionable insights"""
        insights = []
        
        for anomaly in anomalies:
            severity_to_impact = {
                'low': 'low',
                'medium': 'medium',
                'high': 'high',
                'critical': 'high'
            }
            
            insights.append(IntelligenceInsight(
                metric_type=BusinessMetricType.EFFICIENCY,
                insight_text=f"Anomaly detected: {anomaly['type']} - {anomaly.get('impact', 'Performance impact')}",
                confidence_score=0.85,
                impact_level=severity_to_impact.get(anomaly['severity'], 'medium'),
                actionable_recommendations=anomaly.get('recommended_actions', []),
                supporting_data=anomaly
            ))
        
        return insights
    
    async def generate_predictive_performance_insights(self, data: Dict[str, Any]) -> List[IntelligenceInsight]:
        """Generate predictive insights about future performance"""
        insights = []
        
        # Predict system capacity issues
        capacity_prediction = await self.predict_capacity_issues(data['system_performance'])
        if capacity_prediction['risk_level'] > 0.7:
            insights.append(IntelligenceInsight(
                metric_type=BusinessMetricType.EFFICIENCY,
                insight_text=f"System capacity risk detected: {capacity_prediction['predicted_issue']} in {capacity_prediction['days_until_issue']} days",
                confidence_score=capacity_prediction['confidence'],
                impact_level="high",
                actionable_recommendations=[
                    "Scale server infrastructure before capacity issues occur",
                    "Optimize resource-intensive processes",
                    "Implement load balancing improvements"
                ],
                supporting_data=capacity_prediction
            ))
        
        # Predict revenue performance trends
        revenue_trend = await self.predict_revenue_trends(data['revenue_performance'])
        if revenue_trend['trend_direction'] == 'declining':
            insights.append(IntelligenceInsight(
                metric_type=BusinessMetricType.REVENUE,
                insight_text=f"Revenue declining trend detected: {revenue_trend['decline_percentage']:.1f}% decrease predicted",
                confidence_score=revenue_trend['confidence'],
                impact_level="high",
                actionable_recommendations=[
                    "Analyze top-performing content patterns for replication",
                    "Increase posting frequency on highest-revenue platforms",
                    "Review and optimize content strategy"
                ],
                supporting_data=revenue_trend
            ))
        
        return insights
    
    async def predict_capacity_issues(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict future capacity issues"""
        # Analyze resource utilization trends
        resources = system_data['resource_utilization']
        storage_usage = resources['storage_usage']['used_percentage']
        growth_rate = resources['storage_usage']['growth_rate_gb_per_month']
        
        # Simple linear projection
        months_until_full = (100 - storage_usage) / (growth_rate / 100 * storage_usage) if growth_rate > 0 else 999
        
        risk_level = 0.9 if months_until_full < 3 else 0.5 if months_until_full < 6 else 0.2
        
        return {
            'risk_level': risk_level,
            'predicted_issue': 'Storage capacity exhaustion',
            'days_until_issue': int(months_until_full * 30),
            'confidence': 0.85,
            'current_usage': storage_usage,
            'growth_rate': growth_rate
        }
    
    async def predict_revenue_trends(self, revenue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict revenue trends"""
        daily_revenue = revenue_data.get('daily_revenue', [])
        
        if len(daily_revenue) < 7:
            return {
                'trend_direction': 'insufficient_data',
                'confidence': 0.0
            }
        
        # Simple trend analysis
        recent_avg = statistics.mean(daily_revenue[-7:])
        previous_avg = statistics.mean(daily_revenue[-14:-7])
        
        change_percentage = ((recent_avg - previous_avg) / previous_avg * 100) if previous_avg > 0 else 0
        
        trend_direction = 'growing' if change_percentage > 5 else 'declining' if change_percentage < -5 else 'stable'
        
        return {
            'trend_direction': trend_direction,
            'change_percentage': abs(change_percentage),
            'decline_percentage': -change_percentage if change_percentage < 0 else 0,
            'confidence': 0.75,
            'recent_average': recent_avg,
            'previous_average': previous_avg
        }
    
    async def generate_recommendations(self, insights: List[IntelligenceInsight]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Extract all recommendations from insights
        for insight in insights:
            recommendations.extend(insight.actionable_recommendations)
        
        # Add monitoring-specific recommendations
        monitoring_recommendations = await self.generate_monitoring_recommendations(insights)
        recommendations.extend(monitoring_recommendations)
        
        # Remove duplicates and prioritize
        unique_recommendations = list(set(recommendations))
        prioritized_recommendations = await self.prioritize_recommendations(unique_recommendations, insights)
        
        return prioritized_recommendations
    
    async def generate_monitoring_recommendations(self, insights: List[IntelligenceInsight]) -> List[str]:
        """Generate monitoring-specific recommendations"""
        recommendations = []
        
        # Count high-impact insights
        high_impact_count = sum(1 for i in insights if i.impact_level == "high")
        
        if high_impact_count > 3:
            recommendations.append("Implement automated remediation for common performance issues")
            recommendations.append("Create performance dashboards for real-time monitoring")
            recommendations.append("Establish SLAs and automated alerting thresholds")
        
        # Check for efficiency issues
        efficiency_issues = [i for i in insights if i.metric_type == BusinessMetricType.EFFICIENCY]
        if len(efficiency_issues) > 2:
            recommendations.append("Conduct comprehensive performance audit")
            recommendations.append("Implement performance testing in CI/CD pipeline")
        
        return recommendations
    
    async def start_real_time_monitoring(self):
        """Start real-time performance monitoring with alerts"""
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                # Check system health every minute
                system_health = await self.check_system_health()
                
                if system_health['status'] != 'healthy':
                    await self.alert_manager.send_alert(
                        alert_type='system_health',
                        severity=system_health['severity'],
                        message=system_health['message'],
                        data=system_health
                    )
                
                # Check for performance degradation
                performance_status = await self.check_performance_degradation()
                if performance_status['degraded']:
                    await self.alert_manager.send_alert(
                        alert_type='performance_degradation',
                        severity='medium',
                        message=performance_status['message'],
                        data=performance_status
                    )
                
                # Check revenue anomalies
                revenue_status = await self.check_revenue_anomalies()
                if revenue_status['anomaly_detected']:
                    await self.alert_manager.send_alert(
                        alert_type='revenue_anomaly',
                        severity='high',
                        message=revenue_status['message'],
                        data=revenue_status
                    )
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                await self.log_monitoring_error(f"Real-time monitoring error: {str(e)}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def stop_real_time_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
    
    async def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        # This would check actual system metrics in production
        return {
            'status': 'healthy',
            'severity': 'low',
            'message': 'All systems operating normally',
            'timestamp': datetime.now(),
            'metrics': {
                'api_status': 'operational',
                'database_status': 'operational',
                'queue_status': 'operational'
            }
        }
    
    async def check_performance_degradation(self) -> Dict[str, Any]:
        """Check for performance degradation"""
        # This would analyze real-time metrics in production
        return {
            'degraded': False,
            'message': 'Performance within normal parameters',
            'metrics': {
                'response_time_increase': 0,
                'error_rate_increase': 0,
                'throughput_decrease': 0
            }
        }
    
    async def check_revenue_anomalies(self) -> Dict[str, Any]:
        """Check for revenue anomalies in real-time"""
        # This would analyze real-time revenue data in production
        return {
            'anomaly_detected': False,
            'message': 'Revenue metrics normal',
            'current_revenue_rate': 125.50,
            'expected_revenue_rate': 120.00,
            'variance_percentage': 4.5
        }
    
    async def log_monitoring_error(self, error_message: str):
        """Log monitoring system errors"""
        logger.error(f"Performance Monitoring Error: {error_message}")