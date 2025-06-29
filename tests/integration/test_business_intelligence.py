"""Integration tests for Business Intelligence Engine"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any

from backend.intelligence import (
    UsageAnalyticsEngine,
    PerformanceMonitoringSystem,
    RevenueTrackingEngine,
    AIPricingOptimization,
    AnalyticsTimeframe,
    BusinessMetricType,
    IntelligenceInsight
)

class TestBusinessIntelligenceIntegration:
    """Comprehensive integration tests for BI modules"""
    
    @pytest.fixture
    def test_client_id(self):
        """Test client ID for all tests"""
        return "test_client_bi_integration"
    
    @pytest.mark.asyncio
    async def test_complete_bi_workflow(self, test_client_id):
        """Test complete BI workflow from data collection to recommendations"""
        
        # 1. Test usage analytics
        usage_engine = UsageAnalyticsEngine(test_client_id)
        usage_data = await usage_engine.collect_data(AnalyticsTimeframe.MONTH)
        usage_insights = await usage_engine.analyze_data(usage_data)
        usage_recommendations = await usage_engine.generate_recommendations(usage_insights)
        
        # Verify usage analytics results
        assert 'platform_usage' in usage_data
        assert 'feature_usage' in usage_data
        assert len(usage_insights) > 0
        assert all(isinstance(insight, IntelligenceInsight) for insight in usage_insights)
        assert len(usage_recommendations) > 0
        
        # 2. Test performance monitoring
        performance_monitor = PerformanceMonitoringSystem(test_client_id)
        performance_data = await performance_monitor.collect_data(AnalyticsTimeframe.WEEK)
        performance_insights = await performance_monitor.analyze_data(performance_data)
        
        # Verify performance monitoring results
        assert 'system_performance' in performance_data
        assert 'content_performance' in performance_data
        assert 'real_time_status' in performance_data
        assert performance_data['real_time_status']['overall_status'] in ['healthy', 'warning', 'critical']
        
        # 3. Test revenue tracking
        revenue_tracker = RevenueTrackingEngine(test_client_id)
        revenue_data = await revenue_tracker.collect_data(AnalyticsTimeframe.MONTH)
        revenue_insights = await revenue_tracker.analyze_data(revenue_data)
        
        # Verify revenue tracking results
        assert 'direct_revenue' in revenue_data
        assert 'attribution_revenue' in revenue_data
        assert 'platform_revenue' in revenue_data
        assert revenue_data['direct_revenue']['total_direct_revenue'] > 0
        
        # 4. Test AI pricing
        pricing_optimizer = AIPricingOptimization(test_client_id)
        pricing_data = await pricing_optimizer.collect_data(AnalyticsTimeframe.QUARTER)
        pricing_insights = await pricing_optimizer.analyze_data(pricing_data)
        pricing_suggestions = await pricing_optimizer.generate_pricing_suggestions(pricing_data, pricing_insights)
        
        # Verify pricing optimization results
        assert 'demand_data' in pricing_data
        assert 'competitor_data' in pricing_data
        assert len(pricing_suggestions) >= 0
        if pricing_suggestions:
            assert all('confidence_score' in suggestion for suggestion in pricing_suggestions)
            assert all('requires_admin_approval' in suggestion for suggestion in pricing_suggestions)
    
    @pytest.mark.asyncio
    async def test_cross_module_data_flow(self, test_client_id):
        """Test data flow between BI modules"""
        
        # Test that revenue data flows to pricing optimization
        revenue_tracker = RevenueTrackingEngine(test_client_id)
        pricing_optimizer = AIPricingOptimization(test_client_id)
        
        # Generate revenue data
        revenue_data = await revenue_tracker.collect_data(AnalyticsTimeframe.MONTH)
        
        # Use revenue data in pricing optimization
        pricing_data = await pricing_optimizer.collect_data(AnalyticsTimeframe.MONTH)
        pricing_data['revenue_integration'] = revenue_data
        
        pricing_suggestions = await pricing_optimizer.generate_pricing_suggestions(pricing_data, [])
        
        # Verify integration
        assert any('revenue' in str(suggestion).lower() for suggestion in pricing_suggestions)
    
    @pytest.mark.asyncio
    async def test_insight_quality_and_consistency(self, test_client_id):
        """Test quality and consistency of insights across modules"""
        
        # Collect insights from all modules
        all_insights = []
        
        # Usage insights
        usage_engine = UsageAnalyticsEngine(test_client_id)
        usage_data = await usage_engine.collect_data(AnalyticsTimeframe.MONTH)
        usage_insights = await usage_engine.analyze_data(usage_data)
        all_insights.extend(usage_insights)
        
        # Performance insights
        performance_monitor = PerformanceMonitoringSystem(test_client_id)
        performance_data = await performance_monitor.collect_data(AnalyticsTimeframe.MONTH)
        performance_insights = await performance_monitor.analyze_data(performance_data)
        all_insights.extend(performance_insights)
        
        # Revenue insights
        revenue_tracker = RevenueTrackingEngine(test_client_id)
        revenue_data = await revenue_tracker.collect_data(AnalyticsTimeframe.MONTH)
        revenue_insights = await revenue_tracker.analyze_data(revenue_data)
        all_insights.extend(revenue_insights)
        
        # Verify insight quality
        for insight in all_insights:
            assert isinstance(insight, IntelligenceInsight)
            assert 0 <= insight.confidence_score <= 1
            assert insight.impact_level in ['high', 'medium', 'low']
            assert len(insight.actionable_recommendations) > 0
            assert insight.metric_type in BusinessMetricType
            assert isinstance(insight.supporting_data, dict)
    
    @pytest.mark.asyncio
    async def test_real_time_monitoring_capabilities(self, test_client_id):
        """Test real-time monitoring capabilities"""
        
        performance_monitor = PerformanceMonitoringSystem(test_client_id)
        
        # Start monitoring (run for a short time)
        monitoring_task = asyncio.create_task(performance_monitor.start_real_time_monitoring())
        
        # Let it run for 2 seconds
        await asyncio.sleep(2)
        
        # Stop monitoring
        await performance_monitor.stop_real_time_monitoring()
        
        # Cancel the task
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass
        
        # Verify monitoring ran
        assert not performance_monitor.monitoring_active
    
    @pytest.mark.asyncio
    async def test_revenue_attribution_models(self, test_client_id):
        """Test multi-touch attribution models"""
        
        revenue_tracker = RevenueTrackingEngine(test_client_id)
        
        # Test different attribution models
        attribution_data = await revenue_tracker.collect_attribution_revenue_data(
            datetime.now() - timedelta(days=30),
            datetime.now()
        )
        
        # Verify all attribution models are present
        expected_models = ['first_touch', 'last_touch', 'linear', 'time_decay', 'position_based']
        for model in expected_models:
            assert model in attribution_data
            assert 'total_revenue' in attribution_data[model]
            assert 'platform_attribution' in attribution_data[model]
        
        # Verify multi-touch analysis
        assert 'multi_touch_analysis' in attribution_data
        assert 'recommended_model' in attribution_data['multi_touch_analysis']
    
    @pytest.mark.asyncio
    async def test_pricing_approval_workflow(self, test_client_id):
        """Test pricing suggestion approval workflow"""
        
        pricing_optimizer = AIPricingOptimization(test_client_id)
        
        # Generate a pricing suggestion
        test_suggestion = {
            'type': 'price_increase',
            'tier': 'professional',
            'current_price': 149.0,
            'suggested_price': 179.0,
            'price_change_percentage': 20.0,
            'reasoning': 'Test pricing suggestion',
            'confidence_score': 0.85,
            'predicted_impact': {'revenue_change_percentage': 15.0},
            'risk_assessment': {'overall_risk_level': 'medium'}
        }
        
        # Submit for approval
        approval_id = await pricing_optimizer.submit_pricing_suggestion_for_approval(test_suggestion)
        
        # Verify approval was created
        assert approval_id is not None
        assert approval_id.startswith('price_approval_')
        
        # Verify suggestion is tracked
        assert approval_id in pricing_optimizer.approval_workflow.pending_approvals
    
    @pytest.mark.asyncio
    async def test_post_revenue_tracking(self, test_client_id):
        """Test individual post revenue impact tracking"""
        
        revenue_tracker = RevenueTrackingEngine(test_client_id)
        
        # Track a test post
        test_post = {
            'post_id': 'test_post_123',
            'platform': 'instagram',
            'content': {
                'type': 'video',
                'hashtags': ['#test', '#revenue'],
                'caption': 'Test revenue tracking post'
            }
        }
        
        tracking_result = await revenue_tracker.track_post_revenue_impact(
            test_post['post_id'],
            test_post['platform'],
            test_post['content']
        )
        
        # Verify tracking results
        assert 'revenue_attribution' in tracking_result
        assert 'revenue_factors' in tracking_result
        assert 'optimization_suggestions' in tracking_result
        assert tracking_result['revenue_attribution']['total_attributed_revenue'] > 0
    
    @pytest.mark.asyncio
    async def test_performance_anomaly_detection(self, test_client_id):
        """Test anomaly detection in performance monitoring"""
        
        performance_monitor = PerformanceMonitoringSystem(test_client_id)
        
        # Create test data with anomalies
        test_data = {
            'system_performance': {
                'api_response_times': {
                    'average_response_time_ms': 500,  # High - should trigger anomaly
                    'endpoints': {
                        '/api/slow': {'avg_ms': 1000, 'calls': 100}
                    }
                }
            },
            'revenue_performance': {
                'daily_revenue': [1000, 1200, 1100, 500]  # Drop should trigger anomaly
            },
            'engagement_performance': {
                'engagement_rate': 1.5  # Low - should trigger anomaly
            },
            'content_performance': {
                'content_quality_scores': {
                    'overall_quality_score': 6.5  # Below threshold
                }
            }
        }
        
        # Detect anomalies
        anomalies = await performance_monitor.detect_performance_anomalies(test_data)
        
        # Verify anomalies were detected
        assert len(anomalies) > 0
        assert any(a['type'] == 'response_time_spike' for a in anomalies)
    
    @pytest.mark.asyncio
    async def test_elasticity_based_pricing(self, test_client_id):
        """Test price elasticity calculations for pricing optimization"""
        
        pricing_optimizer = AIPricingOptimization(test_client_id)
        
        # Test elasticity analysis
        test_demand_data = {
            'elasticity_by_tier': {
                'starter': {
                    'coefficient': -1.2,
                    'interpretation': 'elastic',
                    'confidence_interval': (-1.4, -1.0)
                },
                'enterprise': {
                    'coefficient': -0.3,
                    'interpretation': 'inelastic',
                    'confidence_interval': (-0.4, -0.2)
                }
            }
        }
        
        current_pricing = {'starter': 49.0, 'enterprise': 599.0}
        
        # Generate demand-based suggestions
        suggestions = await pricing_optimizer.generate_demand_based_suggestions(
            test_demand_data,
            current_pricing
        )
        
        # Verify elasticity-based suggestions
        assert len(suggestions) > 0
        
        # Enterprise (inelastic) should suggest price increase
        enterprise_suggestions = [s for s in suggestions if s.get('tier') == 'enterprise' and s['type'] == 'price_increase']
        assert len(enterprise_suggestions) > 0
        
        # Starter (elastic) should suggest value optimization
        starter_suggestions = [s for s in suggestions if s.get('tier') == 'starter' and s['type'] == 'value_optimization']
        assert len(starter_suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_comprehensive_business_metrics(self, test_client_id):
        """Test calculation of comprehensive business metrics"""
        
        # Test with each engine
        engines = [
            UsageAnalyticsEngine(test_client_id),
            PerformanceMonitoringSystem(test_client_id),
            RevenueTrackingEngine(test_client_id)
        ]
        
        for engine in engines:
            # Get business intelligence
            bi_data = await engine.get_business_intelligence(AnalyticsTimeframe.MONTH)
            
            # Verify comprehensive metrics
            assert 'metrics' in bi_data
            metrics = bi_data['metrics']
            
            # Verify all key metrics are present
            assert hasattr(metrics, 'total_revenue')
            assert hasattr(metrics, 'revenue_growth_rate')
            assert hasattr(metrics, 'engagement_rate')
            assert hasattr(metrics, 'roi_percentage')
            assert hasattr(metrics, 'predicted_revenue_next_period')
            assert hasattr(metrics, 'churn_risk_score')
            assert hasattr(metrics, 'growth_potential_score')
            
            # Verify predictions
            assert 'predictions' in bi_data
            assert 'revenue_forecast' in bi_data['predictions']
            
            # Verify confidence score
            assert 'confidence_score' in bi_data
            assert 0 <= bi_data['confidence_score'] <= 1

class TestIndividualModules:
    """Test individual module functionality"""
    
    @pytest.mark.asyncio
    async def test_usage_analytics_peak_hours(self):
        """Test peak usage hour detection"""
        usage_engine = UsageAnalyticsEngine("test_peak_hours")
        
        # Get peak hours for a platform
        peak_hours = await usage_engine.get_peak_usage_hours(
            'instagram',
            datetime.now() - timedelta(days=7),
            datetime.now()
        )
        
        # Verify peak hours format
        assert isinstance(peak_hours, list)
        assert len(peak_hours) <= 3
        assert all(0 <= hour <= 23 for hour in peak_hours)
    
    @pytest.mark.asyncio
    async def test_revenue_optimal_frequency(self):
        """Test optimal posting frequency calculation"""
        revenue_tracker = RevenueTrackingEngine("test_frequency")
        
        # Test for different content types
        content_types = ['video', 'carousel', 'single_image', 'story', 'reel']
        
        for content_type in content_types:
            frequency = await revenue_tracker.calculate_optimal_frequency(content_type)
            
            assert 'posts_per_week' in frequency
            assert 'optimal_days' in frequency
            assert frequency['posts_per_week'] > 0
            assert len(frequency['optimal_days']) > 0
    
    @pytest.mark.asyncio
    async def test_pricing_competitive_analysis(self):
        """Test competitive pricing analysis"""
        pricing_optimizer = AIPricingOptimization("test_competitive")
        
        # Get competitor data
        competitor_data = await pricing_optimizer.collect_competitor_pricing_data()
        
        # Verify competitor analysis
        assert 'total_competitors_analyzed' in competitor_data
        assert 'competitor_details' in competitor_data
        assert 'market_positioning' in competitor_data
        
        # Verify positioning analysis
        positioning = competitor_data['market_positioning']
        assert 'recommended_position' in positioning
        assert positioning['recommended_position'] in ['value_leader', 'competitive_parity', 'premium_provider']