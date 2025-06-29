"""
Database models for platform publishers.

This module defines the database models for tracking posts, analytics,
and revenue metrics across all social media platforms.
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, Float, JSON, Boolean, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class PlatformPost(Base):
    """Track posts across all platforms with revenue metrics"""
    __tablename__ = 'platform_posts'
    
    id = Column(Integer, primary_key=True)
    client_id = Column(String(255), nullable=False, index=True)
    platform = Column(String(50), nullable=False, index=True)
    post_id = Column(String(255), nullable=False, unique=True)
    content = Column(JSON, nullable=False)
    metadata = Column(JSON)
    business_niche = Column(String(100), nullable=False, index=True)
    publish_status = Column(String(50), default='pending')
    scheduled_time = Column(DateTime)
    published_time = Column(DateTime)
    
    # Revenue tracking
    revenue_potential = Column(Float, default=0.0)
    actual_revenue = Column(Float, default=0.0)
    conversion_rate = Column(Float, default=0.0)
    lead_value = Column(Float, default=0.0)
    
    # Performance metrics
    engagement_metrics = Column(JSON)  # likes, comments, shares, etc.
    reach_metrics = Column(JSON)       # impressions, reach, views
    conversion_metrics = Column(JSON)  # clicks, conversions, sales
    viral_metrics = Column(JSON)       # viral score, share rate, etc.
    
    # Optimization data
    ab_test_variant = Column(String(50))
    optimization_score = Column(Float, default=0.0)
    optimization_suggestions = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_client_platform', 'client_id', 'platform'),
        Index('idx_business_niche', 'business_niche'),
        Index('idx_publish_status', 'publish_status'),
        Index('idx_published_time', 'published_time'),
    )


class PlatformAnalytics(Base):
    """Daily analytics aggregation by platform"""
    __tablename__ = 'platform_analytics'
    
    id = Column(Integer, primary_key=True)
    client_id = Column(String(255), nullable=False, index=True)
    platform = Column(String(50), nullable=False)
    date = Column(DateTime, nullable=False)
    business_niche = Column(String(100), nullable=False)
    
    # Engagement metrics
    total_posts = Column(Integer, default=0)
    total_engagement = Column(Integer, default=0)
    total_reach = Column(Integer, default=0)
    total_impressions = Column(Integer, default=0)
    avg_engagement_rate = Column(Float, default=0.0)
    
    # Revenue metrics
    total_revenue = Column(Float, default=0.0)
    revenue_per_post = Column(Float, default=0.0)
    conversion_revenue = Column(Float, default=0.0)
    lead_generation_value = Column(Float, default=0.0)
    
    # Performance scores
    engagement_score = Column(Float, default=0.0)
    viral_score = Column(Float, default=0.0)
    revenue_score = Column(Float, default=0.0)
    overall_performance_score = Column(Float, default=0.0)
    
    # Platform-specific metrics
    platform_metrics = Column(JSON)  # Store platform-specific data
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_analytics_client_date', 'client_id', 'date'),
        Index('idx_analytics_platform_date', 'platform', 'date'),
        Index('idx_analytics_niche', 'business_niche'),
    )


class RevenueTracking(Base):
    """Detailed revenue tracking per post"""
    __tablename__ = 'revenue_tracking'
    
    id = Column(Integer, primary_key=True)
    post_id = Column(String(255), ForeignKey('platform_posts.post_id'), nullable=False)
    client_id = Column(String(255), nullable=False, index=True)
    platform = Column(String(50), nullable=False)
    
    # Revenue breakdown
    direct_sales = Column(Float, default=0.0)
    lead_value = Column(Float, default=0.0)
    brand_value = Column(Float, default=0.0)
    affiliate_revenue = Column(Float, default=0.0)
    ad_revenue = Column(Float, default=0.0)
    creator_fund_revenue = Column(Float, default=0.0)
    
    # Attribution data
    attribution_window_days = Column(Integer, default=30)
    attribution_model = Column(String(50), default='last_touch')
    confidence_score = Column(Float, default=0.0)
    
    # Conversion tracking
    total_conversions = Column(Integer, default=0)
    conversion_value = Column(Float, default=0.0)
    conversion_sources = Column(JSON)  # Source breakdown
    
    tracked_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    post = relationship("PlatformPost", backref="revenue_tracking")


class AudienceInsights(Base):
    """Store audience insights by platform and niche"""
    __tablename__ = 'audience_insights'
    
    id = Column(Integer, primary_key=True)
    client_id = Column(String(255), nullable=False, index=True)
    platform = Column(String(50), nullable=False)
    business_niche = Column(String(100), nullable=False)
    
    # Demographics
    age_distribution = Column(JSON)
    gender_distribution = Column(JSON)
    location_distribution = Column(JSON)
    interest_categories = Column(JSON)
    
    # Behavior patterns
    peak_activity_times = Column(JSON)
    content_preferences = Column(JSON)
    engagement_patterns = Column(JSON)
    purchase_behavior = Column(JSON)
    
    # Platform-specific data
    platform_specific_insights = Column(JSON)
    
    # Validity
    insights_date = Column(DateTime, default=datetime.utcnow)
    validity_days = Column(Integer, default=30)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_audience_client_platform', 'client_id', 'platform'),
        Index('idx_audience_niche', 'business_niche'),
    )


class ContentOptimizationHistory(Base):
    """Track content optimization history and results"""
    __tablename__ = 'content_optimization_history'
    
    id = Column(Integer, primary_key=True)
    client_id = Column(String(255), nullable=False, index=True)
    platform = Column(String(50), nullable=False)
    original_content = Column(JSON, nullable=False)
    optimized_content = Column(JSON, nullable=False)
    optimization_type = Column(String(100))  # revenue, viral, engagement, etc.
    
    # Results
    original_performance = Column(JSON)
    optimized_performance = Column(JSON)
    improvement_percentage = Column(Float)
    
    # A/B test data
    is_ab_test = Column(Boolean, default=False)
    test_variant = Column(String(50))
    control_variant_id = Column(Integer, ForeignKey('content_optimization_history.id'))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Self-referential relationship for A/B tests
    control_variant = relationship("ContentOptimizationHistory", remote_side=[id])


class PlatformCredentials(Base):
    """Securely store platform credentials"""
    __tablename__ = 'platform_credentials'
    
    id = Column(Integer, primary_key=True)
    client_id = Column(String(255), nullable=False, index=True)
    platform = Column(String(50), nullable=False)
    
    # Encrypted credentials
    encrypted_credentials = Column(Text, nullable=False)
    credential_type = Column(String(50))  # oauth, api_key, etc.
    
    # Token management
    access_token_expires = Column(DateTime)
    refresh_token_expires = Column(DateTime)
    requires_refresh = Column(Boolean, default=False)
    
    # Status
    is_active = Column(Boolean, default=True)
    last_verified = Column(DateTime)
    verification_status = Column(String(50))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Unique constraint
    __table_args__ = (
        Index('idx_credentials_client_platform', 'client_id', 'platform', unique=True),
    )


class PublishingSchedule(Base):
    """Track scheduled posts and optimal posting times"""
    __tablename__ = 'publishing_schedule'
    
    id = Column(Integer, primary_key=True)
    client_id = Column(String(255), nullable=False, index=True)
    platform = Column(String(50), nullable=False)
    
    # Schedule data
    scheduled_time = Column(DateTime, nullable=False)
    content_id = Column(String(255))
    content_data = Column(JSON)
    
    # Optimization data
    is_optimal_time = Column(Boolean, default=True)
    optimization_reason = Column(String(255))
    expected_performance = Column(JSON)
    
    # Status
    status = Column(String(50), default='scheduled')  # scheduled, published, cancelled, failed
    published_at = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_schedule_client_time', 'client_id', 'scheduled_time'),
        Index('idx_schedule_status', 'status'),
    )