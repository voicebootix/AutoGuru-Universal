"""
Example usage of the AutoGuru Universal Business Intelligence Engine

This script demonstrates how to use all BI modules to get comprehensive
business insights for ANY business niche.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any

from backend.intelligence import (
    UsageAnalyticsEngine,
    PerformanceMonitoringSystem,
    RevenueTrackingEngine,
    AIPricingOptimization,
    AnalyticsTimeframe
)


async def demonstrate_usage_analytics(client_id: str):
    """Demonstrate usage analytics capabilities"""
    print("\n=== USAGE ANALYTICS DEMO ===")
    
    engine = UsageAnalyticsEngine(client_id)
    
    # Get monthly usage intelligence
    intelligence = await engine.get_business_intelligence(AnalyticsTimeframe.MONTH)
    
    print(f"\nTotal Insights Generated: {len(intelligence['insights'])}")
    print(f"Confidence Score: {intelligence['confidence_score']:.2f}")
    
    # Display top insights
    print("\nTop Usage Insights:")
    for i, insight in enumerate(intelligence['insights'][:3], 1):
        print(f"\n{i}. {insight.insight_text}")
        print(f"   Impact: {insight.impact_level} | Confidence: {insight.confidence_score:.2f}")
        print("   Recommendations:")
        for rec in insight.actionable_recommendations[:2]:
            print(f"   - {rec}")
    
    # Display key metrics
    metrics = intelligence['metrics']
    print(f"\nKey Metrics:")
    print(f"- Total Engagement: {metrics.total_engagement:,}")
    print(f"- Engagement Rate: {metrics.engagement_rate:.2f}%")
    print(f"- ROI: {metrics.roi_percentage:.2f}%")
    print(f"- Automation Savings: ${metrics.automation_savings:,.2f}")


async def demonstrate_performance_monitoring(client_id: str):
    """Demonstrate performance monitoring capabilities"""
    print("\n=== PERFORMANCE MONITORING DEMO ===")
    
    monitor = PerformanceMonitoringSystem(client_id)
    
    # Get system performance data
    intelligence = await monitor.get_business_intelligence(AnalyticsTimeframe.WEEK)
    
    # Check system health
    real_time_status = await monitor.check_system_health()
    print(f"\nSystem Health: {real_time_status['overall_status']}")
    print(f"Performance Score: {real_time_status['health_score']}/100")
    
    # Display any anomalies or alerts
    anomalies = [i for i in intelligence['insights'] if 'anomaly' in i.insight_text.lower()]
    if anomalies:
        print(f"\n⚠️  {len(anomalies)} Anomalies Detected:")
        for anomaly in anomalies[:3]:
            print(f"- {anomaly.insight_text}")
    
    # Display performance metrics
    print("\nPerformance Metrics:")
    print("- API Response Time: 125ms average")
    print("- System Uptime: 99.9%")
    print("- Content Generation Success Rate: 98.5%")
    print("- AI Model Accuracy: 94.5%")


async def demonstrate_revenue_tracking(client_id: str):
    """Demonstrate revenue tracking and attribution"""
    print("\n=== REVENUE TRACKING DEMO ===")
    
    tracker = RevenueTrackingEngine(client_id)
    
    # Get revenue intelligence
    intelligence = await tracker.get_business_intelligence(AnalyticsTimeframe.MONTH)
    
    metrics = intelligence['metrics']
    print(f"\nRevenue Summary:")
    print(f"- Total Revenue: ${metrics.total_revenue:,.2f}")
    print(f"- Growth Rate: {metrics.revenue_growth_rate:.1f}%")
    print(f"- Revenue per Post: ${metrics.revenue_per_post:.2f}")
    print(f"- Predicted Next Month: ${metrics.predicted_revenue_next_period:,.2f}")
    
    # Track a specific post
    print("\nTracking Individual Post Revenue Impact:")
    post_impact = await tracker.track_post_revenue_impact(
        post_id="demo_post_123",
        platform="instagram",
        content={
            "type": "video",
            "hashtags": ["#business", "#growth", "#entrepreneur"],
            "caption": "5 strategies to scale your business in 2024"
        }
    )
    
    print(f"Post ID: demo_post_123")
    print(f"- Direct Revenue: ${post_impact['revenue_attribution']['direct_revenue']:.2f}")
    print(f"- Assisted Revenue: ${post_impact['revenue_attribution']['assisted_revenue']:.2f}")
    print(f"- Total Attribution: ${post_impact['revenue_attribution']['total_attributed_revenue']:.2f}")
    
    # Show optimization suggestions
    print("\nPost Optimization Suggestions:")
    for suggestion in post_impact['optimization_suggestions'][:3]:
        print(f"- {suggestion}")


async def demonstrate_pricing_optimization(client_id: str):
    """Demonstrate AI pricing optimization"""
    print("\n=== AI PRICING OPTIMIZATION DEMO ===")
    
    optimizer = AIPricingOptimization(client_id)
    
    # Get pricing intelligence
    intelligence = await optimizer.get_business_intelligence(AnalyticsTimeframe.QUARTER)
    
    # Generate pricing suggestions
    suggestions = await optimizer.generate_pricing_suggestions(
        intelligence['metrics'],
        intelligence['insights']
    )
    
    print(f"\n{len(suggestions)} Pricing Suggestions Generated:")
    
    for i, suggestion in enumerate(suggestions[:3], 1):
        print(f"\n{i}. {suggestion['tier'].upper()} Tier")
        print(f"   Current Price: ${suggestion['current_price']}")
        print(f"   Suggested Price: ${suggestion['suggested_price']}")
        print(f"   Change: {suggestion['price_change_percentage']:.1f}%")
        print(f"   Confidence: {suggestion['confidence_score']:.2f}")
        print(f"   Expected Revenue Impact: {suggestion['predicted_impact']['revenue_change_percentage']:.1f}%")
        print(f"   Risk Level: {suggestion['risk_assessment']['overall_risk_level']}")
        
        # Show admin approval requirement
        if suggestion.get('requires_admin_approval'):
            print("   ⚠️  REQUIRES ADMIN APPROVAL")
    
    # Simulate approval workflow
    if suggestions:
        print("\n📋 Creating Approval Request for Top Suggestion...")
        approval_id = await optimizer.submit_pricing_suggestion_for_approval(suggestions[0])
        print(f"Approval Request ID: {approval_id}")
        print("Status: Pending Admin Review")


async def demonstrate_comprehensive_dashboard(client_id: str):
    """Demonstrate comprehensive BI dashboard"""
    print("\n=== COMPREHENSIVE BI DASHBOARD ===")
    
    # Simulate gathering all BI data
    print("\nGathering intelligence from all modules...")
    
    # In real implementation, this would be done in parallel
    usage_engine = UsageAnalyticsEngine(client_id)
    performance_monitor = PerformanceMonitoringSystem(client_id)
    revenue_tracker = RevenueTrackingEngine(client_id)
    pricing_optimizer = AIPricingOptimization(client_id)
    
    print("\n📊 Executive Dashboard Summary:")
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n🎯 Key Performance Indicators:")
    print("- Total Revenue: $45,350 (+15.2% MoM)")
    print("- User Engagement: 125,450 interactions (+25.5% MoM)")
    print("- System Health Score: 92/100")
    print("- ROI: 385%")
    
    print("\n📈 Platform Performance:")
    print("1. Instagram: $18,500 revenue | 45K engagements")
    print("2. LinkedIn: $12,200 revenue | 28K engagements")
    print("3. TikTok: $8,650 revenue | 52K engagements")
    
    print("\n🚨 Active Alerts:")
    print("- [MEDIUM] Instagram API response time increased by 35%")
    print("- [LOW] TikTok engagement rate dropped 12% this week")
    
    print("\n💡 AI Recommendations:")
    print("1. Increase Instagram posting frequency to 2x daily (High Impact)")
    print("2. Review pricing for Professional tier (+20% suggested)")
    print("3. Optimize video content for TikTok algorithm changes")
    
    print("\n🎬 Action Items:")
    print("✅ Review 3 pricing suggestions pending approval")
    print("✅ Investigate Instagram API performance issue")
    print("✅ Schedule strategy session for TikTok optimization")


async def main():
    """Run all demonstrations"""
    client_id = "demo_business_client"
    
    print("=" * 60)
    print("AutoGuru Universal - Business Intelligence Engine Demo")
    print("Works for ANY business niche automatically!")
    print("=" * 60)
    
    # Run all demonstrations
    await demonstrate_usage_analytics(client_id)
    await demonstrate_performance_monitoring(client_id)
    await demonstrate_revenue_tracking(client_id)
    await demonstrate_pricing_optimization(client_id)
    await demonstrate_comprehensive_dashboard(client_id)
    
    print("\n" + "=" * 60)
    print("Demo Complete! The BI Engine provides:")
    print("- Real-time insights for ANY business")
    print("- Data-driven recommendations")
    print("- Revenue optimization strategies")
    print("- Predictive analytics")
    print("- Admin-controlled pricing changes")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())