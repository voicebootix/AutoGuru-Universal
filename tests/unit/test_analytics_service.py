"""
Unit tests for UniversalAnalyticsService

Tests the analytics service functionality to ensure it works universally
across all business niches without hardcoding business logic.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json

from backend.services.analytics_service import (
    UniversalAnalyticsService,
    PerformanceMetrics,
    EngagementMetrics,
    ReachMetrics,
    GrowthAnalytics,
    ROIAnalysis,
    Platform,
    ContentFormat,
    BusinessNicheType,
    TimeFrame
)
from backend.database.connection import PostgreSQLConnectionManager


@pytest.fixture
async def analytics_service():
    """Create analytics service instance with mocked database"""
    mock_db_manager = Mock(spec=PostgreSQLConnectionManager)
    mock_db_manager._is_initialized = True
    mock_db_manager.fetch_one = AsyncMock()
    mock_db_manager.fetch_all = AsyncMock()
    mock_db_manager.execute = AsyncMock()
    
    service = UniversalAnalyticsService(db_manager=mock_db_manager)
    return service


@pytest.fixture
def sample_performance_data():
    """Sample performance data for testing"""
    return {
        'likes': 1500,
        'comments': 120,
        'shares': 85,
        'saves': 200,
        'clicks': 450,
        'impressions': 25000,
        'reach': 20000,
        'organic_reach': 18000,
        'paid_reach': 2000,
        'viral_reach': 1000,
        'reactions': {'love': 300, 'wow': 50, 'haha': 25},
        'conversion_rate': 0.035,
        'demographics': {
            'age': {'18-24': 0.25, '25-34': 0.45, '35-44': 0.20, '45+': 0.10},
            'gender': {'female': 0.60, 'male': 0.38, 'other': 0.02}
        }
    }


class TestContentPerformanceTracking:
    """Test content performance tracking methods"""
    
    @pytest.mark.asyncio
    async def test_track_content_performance_success(self, analytics_service, sample_performance_data):
        """Test successful content performance tracking"""
        # Mock database responses
        analytics_service.db_manager.fetch_one.side_effect = [
            {'metrics_data': json.dumps(sample_performance_data)},  # Platform metrics
            {  # Content metadata
                'content_type': ContentFormat.CAROUSEL.value,
                'published_at': datetime.utcnow(),
                'platform': Platform.INSTAGRAM.value,
                'client_id': 'test_client'
            }
        ]
        
        # Track performance
        result = await analytics_service.track_content_performance(
            content_id='test_content_123',
            platform=Platform.INSTAGRAM
        )
        
        # Verify result
        assert isinstance(result, PerformanceMetrics)
        assert result.content_id == 'test_content_123'
        assert result.platform == Platform.INSTAGRAM
        assert result.engagement.likes == 1500
        assert result.reach.impressions == 25000
        assert 0 <= result.engagement_rate <= 1.0
        
        # Verify database interaction
        assert analytics_service.db_manager.execute.called
    
    @pytest.mark.asyncio
    async def test_calculate_engagement_rate(self, analytics_service):
        """Test engagement rate calculation"""
        # Create mock metrics
        engagement = EngagementMetrics(
            likes=1000,
            comments=50,
            shares=30,
            saves=20,
            clicks=100
        )
        reach = 10000
        
        # Calculate rate
        rate = analytics_service._calculate_engagement_rate(engagement, reach)
        
        # Verify calculation
        expected_rate = (1000 + 50 + 30 + 20 + 100) / 10000
        assert rate == expected_rate
        assert 0 <= rate <= 1.0
    
    @pytest.mark.asyncio
    async def test_compare_content_performance(self, analytics_service):
        """Test content performance comparison"""
        # Mock data for multiple content pieces
        analytics_service._get_all_platform_metrics = AsyncMock(side_effect=[
            {
                'content_id': 'content_1',
                'published_at': datetime.utcnow() - timedelta(days=7),
                'total_engagement': 5000,
                'total_reach': 50000
            },
            {
                'content_id': 'content_2',
                'published_at': datetime.utcnow() - timedelta(days=3),
                'total_engagement': 8000,
                'total_reach': 60000
            },
            {
                'content_id': 'content_3',
                'published_at': datetime.utcnow() - timedelta(days=1),
                'total_engagement': 3000,
                'total_reach': 40000
            }
        ])
        
        analytics_service._aggregate_comparison_metrics = AsyncMock(return_value={
            'content_1': {'engagement_rate': 0.10},
            'content_2': {'engagement_rate': 0.13},
            'content_3': {'engagement_rate': 0.075}
        })
        
        analytics_service._identify_best_performer = Mock(return_value='content_2')
        analytics_service._identify_worst_performer = Mock(return_value='content_3')
        analytics_service._calculate_statistical_significance = AsyncMock(return_value={'p_value': 0.03})
        analytics_service._identify_key_differences = AsyncMock(return_value=['engagement timing', 'content format'])
        analytics_service._generate_comparison_recommendations = AsyncMock(return_value=['Post during peak hours', 'Use more carousel posts'])
        
        # Compare performance
        result = await analytics_service.compare_content_performance(
            content_ids=['content_1', 'content_2', 'content_3']
        )
        
        # Verify results
        assert result.best_performer == 'content_2'
        assert result.worst_performer == 'content_3'
        assert len(result.content_ids) == 3
        assert len(result.key_differences) == 2
        assert len(result.recommendations) == 2


class TestAudienceAnalytics:
    """Test audience analytics methods"""
    
    @pytest.mark.asyncio
    async def test_analyze_audience_growth(self, analytics_service):
        """Test audience growth analysis"""
        # Mock helper methods
        analytics_service._get_timeframe_bounds = Mock(return_value=(
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow()
        ))
        
        analytics_service._get_follower_counts = AsyncMock(side_effect=[
            {Platform.INSTAGRAM: 10000, Platform.TWITTER: 5000},  # Starting
            {Platform.INSTAGRAM: 12000, Platform.TWITTER: 5500}   # Ending
        ])
        
        analytics_service._calculate_growth_rate = Mock(return_value=16.67)
        analytics_service._get_daily_growth_breakdown = AsyncMock(return_value=[
            {'date': '2024-01-01', 'growth': 50},
            {'date': '2024-01-02', 'growth': 75}
        ])
        analytics_service._analyze_growth_sources = AsyncMock(return_value={
            'organic': 1500, 'paid': 500, 'viral': 500
        })
        analytics_service._calculate_churn_rate = AsyncMock(return_value=0.05)
        analytics_service._calculate_growth_velocity = AsyncMock(return_value=1.2)
        analytics_service._project_growth = AsyncMock(return_value={
            '30_days': 14000, '60_days': 16000, '90_days': 18500
        })
        
        # Analyze growth
        result = await analytics_service.analyze_audience_growth(
            client_id='test_client',
            timeframe='month'
        )
        
        # Verify results
        assert isinstance(result, GrowthAnalytics)
        assert result.total_growth == 2500
        assert result.growth_rate == 16.67
        assert result.retention_rate == 0.95
        assert len(result.daily_growth) == 2
    
    @pytest.mark.asyncio
    async def test_identify_high_value_segments(self, analytics_service):
        """Test high-value audience segment identification"""
        # Mock helper methods
        analytics_service._get_total_audience_size = AsyncMock(return_value=50000)
        analytics_service._perform_audience_segmentation = AsyncMock(return_value=[
            Mock(
                segment_id='seg_1',
                segment_name='Engaged Professionals',
                size=15000,
                characteristics={'age': '25-34', 'interests': ['business', 'tech']},
                lifetime_value=0,
                growth_rate=0
            ),
            Mock(
                segment_id='seg_2',
                segment_name='Creative Enthusiasts',
                size=10000,
                characteristics={'age': '18-24', 'interests': ['art', 'design']},
                lifetime_value=0,
                growth_rate=0
            )
        ])
        
        analytics_service._calculate_segment_engagement_score = AsyncMock(return_value=0.85)
        analytics_service._calculate_conversion_potential = AsyncMock(return_value=0.65)
        analytics_service._calculate_segment_ltv = AsyncMock(side_effect=[1500.0, 1200.0])
        analytics_service._calculate_segment_growth_rate = AsyncMock(side_effect=[2.5, 3.8])
        analytics_service._calculate_segment_overlap = AsyncMock(return_value={'seg_1_seg_2': 0.15})
        analytics_service._generate_segment_recommendations = AsyncMock(return_value=[
            'Focus on creative content for fastest growing segment',
            'Develop professional content for highest value segment'
        ])
        
        # Identify segments
        result = await analytics_service.identify_high_value_audience_segments('test_client')
        
        # Verify results
        assert result.total_audience == 50000
        assert len(result.segments) == 2
        assert result.most_valuable_segment.lifetime_value == 1500.0
        assert result.fastest_growing_segment.growth_rate == 3.8


class TestBusinessImpactAnalytics:
    """Test business impact analytics methods"""
    
    @pytest.mark.asyncio
    async def test_calculate_roi(self, analytics_service):
        """Test ROI calculation"""
        # Mock helper methods
        analytics_service._get_timeframe_bounds = Mock(return_value=(
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow()
        ))
        
        analytics_service._calculate_total_investment = AsyncMock(return_value={
            'total': 10000,
            'breakdown': {'advertising': 6000, 'content': 3000, 'tools': 1000}
        })
        
        analytics_service._calculate_total_revenue = AsyncMock(return_value={
            'total': 25000,
            'breakdown': {'direct_sales': 20000, 'affiliate': 5000},
            'daily_revenue': [800, 850, 900]
        })
        
        analytics_service._calculate_cost_per_acquisition = AsyncMock(return_value=50.0)
        analytics_service._calculate_customer_lifetime_value = AsyncMock(return_value=500.0)
        analytics_service._calculate_platform_roi = AsyncMock(return_value={
            Platform.INSTAGRAM: 180.0,
            Platform.FACEBOOK: 120.0
        })
        analytics_service._calculate_content_type_roi = AsyncMock(return_value={
            ContentFormat.VIDEO: 200.0,
            ContentFormat.CAROUSEL: 150.0
        })
        analytics_service._find_break_even_point = AsyncMock(return_value=datetime.utcnow() - timedelta(days=10))
        analytics_service._calculate_payback_period = AsyncMock(return_value=12)
        analytics_service._perform_attribution_analysis = AsyncMock(return_value={
            'first_touch': 0.3, 'last_touch': 0.5, 'linear': 0.2
        })
        analytics_service._generate_roi_recommendations = AsyncMock(return_value=[
            'Increase investment in video content',
            'Focus on Instagram for higher ROI'
        ])
        
        # Calculate ROI
        result = await analytics_service.calculate_roi(
            client_id='test_client',
            timeframe='month'
        )
        
        # Verify results
        assert isinstance(result, ROIAnalysis)
        assert result.roi_percentage == 150.0  # (25000 - 10000) / 10000 * 100
        assert result.profit_margin == 0.6  # (25000 - 10000) / 25000
        assert result.cost_per_acquisition == 50.0
        assert result.customer_lifetime_value == 500.0


class TestUniversalNicheSupport:
    """Test that analytics work universally across business niches"""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("business_niche", list(BusinessNicheType))
    async def test_analytics_work_for_all_niches(self, analytics_service, business_niche):
        """Test that analytics methods work for all business niches"""
        # Mock data that works for any niche
        analytics_service._get_industry_benchmarks = AsyncMock(return_value={
            'engagement_rate': 0.05,
            'growth_rate': 2.5,
            'conversion_rate': 0.02
        })
        
        analytics_service._get_client_performance_summary = AsyncMock(return_value={
            'engagement_rate': 0.07,
            'growth_rate': 3.2,
            'conversion_rate': 0.025
        })
        
        analytics_service._calculate_percentile_rankings = AsyncMock(return_value={
            'engagement_rate': 75,
            'growth_rate': 80,
            'conversion_rate': 70
        })
        
        analytics_service._determine_competitive_position = Mock(return_value='Above Average')
        analytics_service._identify_performance_strengths = AsyncMock(return_value=[
            'High engagement rate',
            'Strong growth trajectory'
        ])
        analytics_service._identify_performance_weaknesses = AsyncMock(return_value=[
            'Below average post frequency'
        ])
        analytics_service._find_improvement_opportunities = AsyncMock(return_value=[
            {'area': 'Content frequency', 'impact': 'high'},
            {'area': 'Platform diversification', 'impact': 'medium'}
        ])
        analytics_service._get_industry_best_practices = AsyncMock(return_value=[
            'Post consistently',
            'Engage with audience'
        ])
        
        # Run benchmark analysis for each niche
        result = await analytics_service.benchmark_against_industry(
            client_id='test_client',
            business_niche=business_niche
        )
        
        # Verify it works for all niches
        assert result.business_niche == business_niche
        assert result.competitive_position == 'Above Average'
        assert len(result.strengths) > 0
        assert len(result.improvement_opportunities) > 0
        
        # Ensure no hardcoded business logic
        assert 'fitness' not in str(result).lower() or business_niche == BusinessNicheType.FITNESS_WELLNESS
        assert 'consulting' not in str(result).lower() or business_niche == BusinessNicheType.BUSINESS_CONSULTING


class TestReportingAndExports:
    """Test reporting and export functionality"""
    
    @pytest.mark.asyncio
    async def test_export_analytics_data(self, analytics_service):
        """Test analytics data export in multiple formats"""
        # Mock data collection
        analytics_service._collect_all_analytics_data = AsyncMock(return_value={
            'performance': {'total_engagement': 50000},
            'audience': {'total_followers': 25000},
            'roi': {'roi_percentage': 150.0}
        })
        
        # Mock export methods
        export_result = {
            'file_path': '/tmp/export.csv',
            'rows_exported': 1000,
            'file_size': 50000
        }
        analytics_service._export_to_csv = AsyncMock(return_value=export_result)
        analytics_service._export_to_json = AsyncMock(return_value=export_result)
        analytics_service._export_to_excel = AsyncMock(return_value=export_result)
        analytics_service._export_to_pdf = AsyncMock(return_value=export_result)
        analytics_service._generate_download_url = AsyncMock(
            return_value='https://download.example.com/export.csv'
        )
        
        # Test CSV export
        result = await analytics_service.export_analytics_data(
            client_id='test_client',
            format='csv'
        )
        
        # Verify export
        assert result.format == 'csv'
        assert result.rows_exported == 1000
        assert result.download_url.startswith('https://')
        assert result.expiry_timestamp > datetime.utcnow()
        
        # Test invalid format
        with pytest.raises(ValueError):
            await analytics_service.export_analytics_data(
                client_id='test_client',
                format='invalid'
            )
    
    @pytest.mark.asyncio
    async def test_schedule_automated_reports(self, analytics_service):
        """Test automated report scheduling"""
        # Mock helper methods
        analytics_service._determine_report_type = Mock(return_value='weekly_performance')
        analytics_service._calculate_next_run_time = Mock(
            return_value=datetime.utcnow() + timedelta(days=7)
        )
        analytics_service._get_report_recipients = AsyncMock(
            return_value=['admin@example.com', 'client@example.com']
        )
        analytics_service._create_report_schedule = AsyncMock()
        
        # Schedule report
        result = await analytics_service.schedule_automated_reports(
            client_id='test_client',
            frequency='weekly'
        )
        
        # Verify schedule
        assert result.frequency == 'weekly'
        assert result.status == 'active'
        assert len(result.recipients) == 2
        assert result.next_run > datetime.utcnow()
        
        # Test invalid frequency
        with pytest.raises(ValueError):
            await analytics_service.schedule_automated_reports(
                client_id='test_client',
                frequency='hourly'
            )