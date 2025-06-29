"""
AutoGuru Universal - Advanced Analytics Module
Comprehensive analytics engine for business intelligence and insights
"""

from .base_analytics import (
    UniversalAnalyticsEngine,
    AnalyticsScope,
    InsightPriority,
    ReportFormat,
    AnalyticsInsight,
    BusinessKPI,
    AnalyticsRequest,
    AnalyticsError
)

from .cross_platform_analytics import CrossPlatformAnalyticsEngine
from .bi_reports import BusinessIntelligenceReports
from .customer_success_analytics import CustomerSuccessAnalytics
from .predictive_modeling import PredictiveBusinessModeling
from .competitive_intelligence import CompetitiveIntelligenceSystem
from .executive_dashboard import ExecutiveDashboardGenerator

__all__ = [
    'UniversalAnalyticsEngine',
    'AnalyticsScope',
    'InsightPriority',
    'ReportFormat',
    'AnalyticsInsight',
    'BusinessKPI',
    'AnalyticsRequest',
    'AnalyticsError',
    'CrossPlatformAnalyticsEngine',
    'BusinessIntelligenceReports',
    'CustomerSuccessAnalytics',
    'PredictiveBusinessModeling',
    'CompetitiveIntelligenceSystem',
    'ExecutiveDashboardGenerator'
]