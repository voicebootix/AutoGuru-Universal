"""Business Intelligence Engine for AutoGuru Universal"""

from .base_intelligence import (
    UniversalIntelligenceEngine,
    AnalyticsTimeframe,
    BusinessMetricType,
    IntelligenceInsight,
    BusinessMetrics,
    IntelligenceEngineError
)

from .usage_analytics import UsageAnalyticsEngine
from .performance_monitor import PerformanceMonitoringSystem
from .revenue_tracker import RevenueTrackingEngine
from .ai_pricing import AIPricingOptimization

# Enhanced ML Models
from .enhanced_ml_models import (
    EnhancedAnomalyDetector,
    PredictiveRevenueModel,
    CustomerSegmentationEngine,
    AnomalyScore
)

# A/B Testing Framework
from .ab_testing import (
    ABTestingEngine,
    ExperimentStatus,
    ExperimentType,
    Experiment,
    ExperimentResult
)

# Real-time Streaming
from .realtime_streaming import (
    RealTimeMetricsStreamer,
    WebSocketMetricsHandler,
    MetricType,
    MetricUpdate
)

# Caching Strategy
from .caching_strategy import (
    IntelligenceCacheManager,
    cache_intelligence_data,
    SmartCacheInvalidator,
    CacheWarmer
)

# Advanced Alerting
from .advanced_alerting import (
    IntelligentAlertManager,
    AlertSeverity,
    AlertChannel,
    AlertPattern,
    Alert,
    AlertRule
)

__all__ = [
    # Base classes
    'UniversalIntelligenceEngine',
    'AnalyticsTimeframe',
    'BusinessMetricType',
    'IntelligenceInsight',
    'BusinessMetrics',
    'IntelligenceEngineError',
    
    # Core Intelligence engines
    'UsageAnalyticsEngine',
    'PerformanceMonitoringSystem',
    'RevenueTrackingEngine',
    'AIPricingOptimization',
    
    # Enhanced ML Models
    'EnhancedAnomalyDetector',
    'PredictiveRevenueModel',
    'CustomerSegmentationEngine',
    'AnomalyScore',
    
    # A/B Testing
    'ABTestingEngine',
    'ExperimentStatus',
    'ExperimentType',
    'Experiment',
    'ExperimentResult',
    
    # Real-time Streaming
    'RealTimeMetricsStreamer',
    'WebSocketMetricsHandler',
    'MetricType',
    'MetricUpdate',
    
    # Caching
    'IntelligenceCacheManager',
    'cache_intelligence_data',
    'SmartCacheInvalidator',
    'CacheWarmer',
    
    # Alerting
    'IntelligentAlertManager',
    'AlertSeverity',
    'AlertChannel',
    'AlertPattern',
    'Alert',
    'AlertRule'
]