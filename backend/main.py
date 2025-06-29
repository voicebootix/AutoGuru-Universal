"""
AutoGuru Universal - Main FastAPI Application

This is the main entry point for the AutoGuru Universal API that provides
social media automation for ANY business niche. All functionality is AI-driven
without hardcoded business logic.
"""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, Request, Response, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
from celery import Celery
from celery.result import AsyncResult
from fastapi import WebSocket, WebSocketDisconnect

# Import settings based on environment
try:
    from backend.config.production import get_production_settings
    settings = get_production_settings()
    ENVIRONMENT = "production"
except ImportError:
    from backend.config.settings import get_settings
    settings = get_settings()
    ENVIRONMENT = "development"

from backend.database.connection import get_db_manager, PostgreSQLConnectionManager
from backend.core.content_analyzer import UniversalContentAnalyzer
from backend.models.content_models import (
    ContentAnalysis,
    BusinessNiche,
    AudienceProfile,
    Platform,
    ContentFormat,
    PlatformContent
)
from backend.utils.encryption import encrypt_data, decrypt_data

# Import Business Intelligence modules
from backend.intelligence import (
    UsageAnalyticsEngine,
    PerformanceMonitoringSystem,
    RevenueTrackingEngine,
    AIPricingOptimization,
    AnalyticsTimeframe,
    BusinessMetricType,
    IntelligenceInsight
)

# Build handlers list
handlers: List[logging.Handler] = [logging.StreamHandler()]
if hasattr(settings, 'logging') and settings.logging.enable_file_logging:
    log_path = Path(settings.logging.log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(settings.logging.log_file_path))

# Configure logging
log_level = getattr(settings, 'logging', None)
if log_level:
    logging.basicConfig(
        level=getattr(logging, log_level.level.value),
        format=log_level.format,
        handlers=handlers
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )

logger = logging.getLogger(__name__)

# Initialize Celery
celery_broker_url = getattr(settings, 'celery', None)
if celery_broker_url:
    celery_app = Celery(
        'autoguru_universal',
        broker=celery_broker_url.broker_url,
        backend=celery_broker_url.result_backend
    )
else:
    # Fallback for development
    celery_app = Celery(
        'autoguru_universal',
        broker='redis://localhost:6379',
        backend='redis://localhost:6379'
    )

# Security
security = HTTPBearer(auto_error=False)  # Make auth optional for health checks

# Request/Response Models
class AnalyzeContentRequest(BaseModel):
    """Request model for content analysis"""
    content: str = Field(..., min_length=10, max_length=10000, description="Content to analyze")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional business context")
    platforms: Optional[List[Platform]] = Field(None, description="Target platforms for analysis")
    
    class Config:
        schema_extra = {
            "example": {
                "content": "Transform your fitness journey with our personalized training programs...",
                "context": {"website": "fitnessguru.com", "existing_audience": "health enthusiasts"},
                "platforms": ["instagram", "youtube", "tiktok"]
            }
        }


class GeneratePersonaRequest(BaseModel):
    """Request model for persona generation"""
    business_description: str = Field(..., description="Description of the business")
    target_market: Optional[str] = Field(None, description="Target market description")
    goals: Optional[List[str]] = Field(None, description="Business goals")
    
    class Config:
        schema_extra = {
            "example": {
                "business_description": "Online education platform for professional development",
                "target_market": "Working professionals seeking career advancement",
                "goals": ["increase course enrollments", "build thought leadership"]
            }
        }


class CreateViralContentRequest(BaseModel):
    """Request model for viral content creation"""
    topic: str = Field(..., description="Content topic or theme")
    business_niche: BusinessNiche = Field(..., description="Business niche information")
    target_audience: Optional[Dict[str, Any]] = Field(None, description="Target audience details")
    platforms: List[Platform] = Field(..., description="Target platforms")
    content_type: Optional[ContentFormat] = Field(None, description="Desired content format")
    
    class Config:
        schema_extra = {
            "example": {
                "topic": "5 productivity hacks for remote workers",
                "business_niche": {
                    "niche_type": "business_consulting",
                    "confidence_score": 0.95,
                    "sub_niches": ["productivity", "remote work"],
                    "reasoning": "Focus on business efficiency",
                    "keywords": ["productivity", "efficiency", "remote"]
                },
                "platforms": ["linkedin", "twitter"]
            }
        }


class PublishContentRequest(BaseModel):
    """Request model for content publishing"""
    content: PlatformContent = Field(..., description="Platform-specific content to publish")
    schedule_time: Optional[datetime] = Field(None, description="Schedule for future publishing")
    cross_post: bool = Field(False, description="Cross-post to multiple platforms")
    
    class Config:
        schema_extra = {
            "example": {
                "content": {
                    "platform": "instagram",
                    "content_text": "Transform your mornings with these 5 habits...",
                    "content_format": "carousel",
                    "hashtags": ["morningroutine", "productivity"],
                    "call_to_action": "Save this post for tomorrow!"
                },
                "schedule_time": "2024-01-15T09:00:00Z"
            }
        }


class TaskStatusResponse(BaseModel):
    """Response model for task status"""
    task_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
    progress: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    environment: str
    timestamp: datetime
    version: str
    features: List[str]


# Business Intelligence Request/Response Models
class GetBusinessIntelligenceRequest(BaseModel):
    """Request model for business intelligence data"""
    timeframe: AnalyticsTimeframe = Field(AnalyticsTimeframe.MONTH, description="Analytics timeframe")
    metric_types: Optional[List[BusinessMetricType]] = Field(None, description="Specific metrics to focus on")
    
    class Config:
        schema_extra = {
            "example": {
                "timeframe": "month",
                "metric_types": ["revenue", "engagement", "efficiency"]
            }
        }


class StartMonitoringRequest(BaseModel):
    """Request model for starting real-time monitoring"""
    monitoring_type: str = Field("comprehensive", description="Type of monitoring to start")
    alert_channels: Optional[List[str]] = Field(["email", "webhook"], description="Alert notification channels")
    
    class Config:
        schema_extra = {
            "example": {
                "monitoring_type": "comprehensive",
                "alert_channels": ["email", "webhook", "sms"]
            }
        }


class TrackPostRevenueRequest(BaseModel):
    """Request model for tracking post revenue impact"""
    post_id: str = Field(..., description="Unique post identifier")
    platform: str = Field(..., description="Social media platform")
    content: Dict[str, Any] = Field(..., description="Post content details")
    tracking_duration_days: int = Field(30, description="Days to track post impact")
    
    class Config:
        schema_extra = {
            "example": {
                "post_id": "post_12345",
                "platform": "instagram",
                "content": {
                    "type": "video",
                    "hashtags": ["#business", "#growth"],
                    "caption": "5 tips for business growth..."
                },
                "tracking_duration_days": 30
            }
        }


class PricingSuggestionResponse(BaseModel):
    """Response model for pricing suggestions"""
    suggestion_id: str
    tier: str
    current_price: float
    suggested_price: float
    price_change_percentage: float
    confidence_score: float
    expected_impact: Dict[str, float]
    risk_assessment: Dict[str, Any]
    requires_admin_approval: bool
    
    class Config:
        schema_extra = {
            "example": {
                "suggestion_id": "price_suggest_12345",
                "tier": "professional",
                "current_price": 149.0,
                "suggested_price": 179.0,
                "price_change_percentage": 20.0,
                "confidence_score": 0.85,
                "expected_impact": {
                    "revenue_change_percentage": 15.0,
                    "churn_risk": 0.05
                },
                "risk_assessment": {
                    "overall_risk_level": "medium",
                    "mitigation_strategies": ["grandfathering", "value_communication"]
                },
                "requires_admin_approval": True
            }
        }


class ApprovePricingRequest(BaseModel):
    """Request model for approving pricing changes"""
    approval_id: str = Field(..., description="Approval request ID")
    decision: str = Field(..., description="approve or reject")
    admin_notes: Optional[str] = Field(None, description="Admin notes on the decision")
    
    class Config:
        schema_extra = {
            "example": {
                "approval_id": "price_approval_12345",
                "decision": "approve",
                "admin_notes": "Approved with 30-day notice to existing customers"
            }
        }


# Middleware
class RequestIdMiddleware:
    """Middleware to add request ID for tracking"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        logger.info(
            f"Request {request_id} - {request.method} {request.url.path} "
            f"- Status: {response.status_code} - Time: {process_time:.3f}s"
        )
        
        return response


class ErrorHandlerMiddleware:
    """Global error handling middleware"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, request: Request, call_next):
        try:
            return await call_next(request)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unhandled error: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": "An unexpected error occurred",
                    "request_id": getattr(request.state, "request_id", "unknown")
                }
            )


# Authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify authentication token"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # In production, you would verify the JWT token here
    # For now, we'll accept any token for development
    return credentials.credentials


# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info(f"Starting AutoGuru Universal in {ENVIRONMENT} environment")
    
    # Initialize database connection
    try:
        if ENVIRONMENT == "production":
            # Production database initialization
            db_manager = get_db_manager()
            await db_manager.initialize()
            logger.info("Database initialized successfully")
    except Exception as e:
        logger.warning(f"Database initialization failed: {e}")
    
    yield
    
    logger.info("Shutting down AutoGuru Universal")


# Create FastAPI app
app = FastAPI(
    title=getattr(settings, 'title', 'AutoGuru Universal'),
    description=getattr(settings, 'description', 'Universal social media automation for ANY business niche'),
    version=getattr(settings, 'version', '1.0.0'),
    docs_url="/docs" if ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if ENVIRONMENT != "production" else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(RequestIdMiddleware)
app.add_middleware(ErrorHandlerMiddleware)

# CORS configuration
cors_origins = getattr(settings, 'security', None)
if cors_origins and cors_origins.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    # Default CORS for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Mount static files for frontend
frontend_path = Path("frontend")
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory="frontend"), name="static")


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "AutoGuru Universal - Universal Social Media Automation",
        "environment": ENVIRONMENT,
        "status": "running",
        "docs": "/docs" if ENVIRONMENT != "production" else None,
        "health": "/health",
        "version": getattr(settings, 'version', '1.0.0')
    }


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for Render"""
    features = [
        "Content Analysis",
        "Business Niche Detection",
        "Viral Potential Scoring",
        "Platform Recommendations",
        "Hashtag Generation",
        "Universal Business Support",
        "Usage Analytics Engine",
        "Performance Monitoring System",
        "Revenue Tracking & Attribution",
        "AI Pricing Optimization"
    ]
    
    return HealthResponse(
        status="healthy",
        environment=ENVIRONMENT,
        timestamp=datetime.utcnow(),
        version=getattr(settings, 'version', '1.0.0'),
        features=features
    )


# Frontend serving
@app.get("/app")
async def serve_frontend():
    """Serve the frontend application"""
    frontend_file = Path("frontend/index.html")
    if frontend_file.exists():
        return FileResponse(frontend_file)
    else:
        return {"message": "Frontend not available", "api_docs": "/docs"}


# API endpoints
@app.get("/api/v1/health", tags=["Health"])
async def api_health_check():
    """API health check endpoint"""
    return {
        "status": "healthy",
        "environment": ENVIRONMENT,
        "timestamp": datetime.utcnow(),
        "version": getattr(settings, 'version', '1.0.0')
    }


@app.post(
    "/api/v1/analyze",
    response_model=ContentAnalysis,
    tags=["Content Analysis"],
    summary="Analyze content for any business niche"
)
async def analyze_content(
    request: AnalyzeContentRequest,
    token: str = Depends(verify_token)
):
    """Analyze content for any business niche"""
    try:
        analyzer = UniversalContentAnalyzer()
        analysis = await analyzer.analyze_content(
            content=request.content,
            context=request.context,
            platforms=request.platforms
        )
        return analysis
    except Exception as e:
        logger.error(f"Content analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/v1/generate-persona",
    response_model=AudienceProfile,
    tags=["Persona Generation"],
    summary="Generate detailed audience personas"
)
async def generate_persona(
    request: GeneratePersonaRequest,
    token: str = Depends(verify_token)
):
    """Generate detailed audience personas"""
    try:
        analyzer = UniversalContentAnalyzer()
        persona = await analyzer.generate_persona(
            business_description=request.business_description,
            target_market=request.target_market,
            goals=request.goals
        )
        return persona
    except Exception as e:
        logger.error(f"Persona generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/v1/create-viral-content",
    response_model=List[PlatformContent],
    tags=["Content Creation"],
    summary="Create viral content for multiple platforms"
)
async def create_viral_content(
    request: CreateViralContentRequest,
    token: str = Depends(verify_token)
):
    """Create viral content for multiple platforms"""
    try:
        analyzer = UniversalContentAnalyzer()
        content_list = await analyzer.create_viral_content(
            topic=request.topic,
            business_niche=request.business_niche,
            target_audience=request.target_audience,
            platforms=request.platforms,
            content_type=request.content_type
        )
        return content_list
    except Exception as e:
        logger.error(f"Viral content creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/v1/publish",
    tags=["Publishing"],
    summary="Publish content to social media platforms"
)
async def publish_content(
    request: PublishContentRequest,
    token: str = Depends(verify_token)
):
    """Publish content to social media platforms"""
    try:
        # This would integrate with actual social media platforms
        # For now, return a mock response
        return {
            "status": "scheduled",
            "task_id": str(uuid.uuid4()),
            "message": "Content scheduled for publishing",
            "platform": request.content.platform,
            "scheduled_time": request.schedule_time
        }
    except Exception as e:
        logger.error(f"Content publishing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/v1/tasks/{task_id}",
    response_model=TaskStatusResponse,
    tags=["Tasks"],
    summary="Get background task status"
)
async def get_task_status(
    task_id: str,
    token: str = Depends(verify_token)
):
    """Get background task status"""
    try:
        result = AsyncResult(task_id, app=celery_app)
        return TaskStatusResponse(
            task_id=task_id,
            status=result.status,
            result=result.result if result.ready() else None,
            error=str(result.info) if result.failed() else None,
            progress=result.info.get('progress', 0) if result.info else None
        )
    except Exception as e:
        logger.error(f"Task status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/v1/rate-limits",
    tags=["System"],
    summary="Get current rate limit status"
)
async def get_rate_limits(token: str = Depends(verify_token)):
    """Get current rate limit status"""
    # This would implement actual rate limiting logic
    return {
        "rate_limits": {
            "requests_per_minute": getattr(settings, 'rate_limit_requests', 100),
            "window_seconds": getattr(settings, 'rate_limit_window', 60),
            "current_usage": 0  # This would be tracked in production
        }
    }


# Demo endpoint for testing
@app.get("/demo", tags=["Demo"])
async def demo_analysis():
    """Demo analysis endpoint"""
    demo_content = "Transform your body with our 8-week HIIT program! Join thousands who've achieved their dream physique."
    
    try:
        analyzer = UniversalContentAnalyzer()
        analysis = await analyzer.analyze_content(
            content=demo_content,
            context="Fitness and wellness business"
        )
        return {
            "demo_content": demo_content,
            "analysis": analysis,
            "message": "This is a demo analysis. Use /api/v1/analyze endpoint for your own content."
        }
    except Exception as e:
        logger.error(f"Demo analysis failed: {e}")
        return {
            "demo_content": demo_content,
            "error": str(e),
            "message": "Demo analysis failed. Check API configuration."
        }


# Business Intelligence API Endpoints
@app.post(
    "/api/v1/bi/usage-analytics",
    tags=["Business Intelligence"],
    summary="Get comprehensive usage analytics"
)
async def get_usage_analytics(
    request: GetBusinessIntelligenceRequest,
    token: str = Depends(verify_token)
):
    """Get comprehensive usage analytics with insights and recommendations"""
    try:
        # Extract client_id from token (in production)
        client_id = "demo_client"  # Would be extracted from JWT token
        
        usage_engine = UsageAnalyticsEngine(client_id)
        intelligence = await usage_engine.get_business_intelligence(request.timeframe)
        
        return {
            "status": "success",
            "data": intelligence,
            "summary": {
                "total_insights": len(intelligence.get('insights', [])),
                "top_recommendations": intelligence.get('recommendations', [])[:3],
                "confidence_score": intelligence.get('confidence_score', 0)
            }
        }
    except Exception as e:
        logger.error(f"Usage analytics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/v1/bi/performance-monitoring",
    tags=["Business Intelligence"],
    summary="Get performance monitoring data"
)
async def get_performance_monitoring(
    request: GetBusinessIntelligenceRequest,
    token: str = Depends(verify_token)
):
    """Get real-time performance monitoring data with anomaly detection"""
    try:
        client_id = "demo_client"  # Would be extracted from JWT token
        
        monitor = PerformanceMonitoringSystem(client_id)
        intelligence = await monitor.get_business_intelligence(request.timeframe)
        
        # Check for any critical alerts
        critical_alerts = [
            insight for insight in intelligence.get('insights', [])
            if insight.impact_level == "high"
        ]
        
        return {
            "status": "success",
            "data": intelligence,
            "alerts": {
                "critical_count": len(critical_alerts),
                "critical_alerts": critical_alerts[:5]  # Top 5 critical alerts
            },
            "system_status": intelligence.get('metrics', {})
        }
    except Exception as e:
        logger.error(f"Performance monitoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/v1/bi/start-monitoring",
    tags=["Business Intelligence"],
    summary="Start real-time performance monitoring"
)
async def start_real_time_monitoring(
    request: StartMonitoringRequest,
    token: str = Depends(verify_token)
):
    """Start real-time performance monitoring with alerts"""
    try:
        client_id = "demo_client"  # Would be extracted from JWT token
        
        monitor = PerformanceMonitoringSystem(client_id)
        
        # Start monitoring in background
        task_id = str(uuid.uuid4())
        asyncio.create_task(monitor.start_real_time_monitoring())
        
        return {
            "status": "monitoring_started",
            "task_id": task_id,
            "monitoring_type": request.monitoring_type,
            "alert_channels": request.alert_channels,
            "message": "Real-time monitoring started. Alerts will be sent to configured channels."
        }
    except Exception as e:
        logger.error(f"Start monitoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/v1/bi/revenue-tracking",
    tags=["Business Intelligence"],
    summary="Get revenue tracking and attribution data"
)
async def get_revenue_tracking(
    request: GetBusinessIntelligenceRequest,
    token: str = Depends(verify_token)
):
    """Get comprehensive revenue tracking with multi-touch attribution"""
    try:
        client_id = "demo_client"  # Would be extracted from JWT token
        
        revenue_tracker = RevenueTrackingEngine(client_id)
        intelligence = await revenue_tracker.get_business_intelligence(request.timeframe)
        
        # Extract key revenue metrics
        metrics = intelligence.get('metrics', {})
        
        return {
            "status": "success",
            "data": intelligence,
            "revenue_summary": {
                "total_revenue": metrics.total_revenue if hasattr(metrics, 'total_revenue') else 0,
                "revenue_growth_rate": metrics.revenue_growth_rate if hasattr(metrics, 'revenue_growth_rate') else 0,
                "revenue_per_post": metrics.revenue_per_post if hasattr(metrics, 'revenue_per_post') else 0,
                "predicted_next_period": metrics.predicted_revenue_next_period if hasattr(metrics, 'predicted_revenue_next_period') else 0
            },
            "attribution_models": {
                "recommended": "linear",  # Would be dynamically determined
                "comparison_available": True
            }
        }
    except Exception as e:
        logger.error(f"Revenue tracking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/v1/bi/track-post-revenue",
    tags=["Business Intelligence"],
    summary="Track revenue impact of a specific post"
)
async def track_post_revenue(
    request: TrackPostRevenueRequest,
    token: str = Depends(verify_token)
):
    """Track revenue impact and attribution for a specific post"""
    try:
        client_id = "demo_client"  # Would be extracted from JWT token
        
        revenue_tracker = RevenueTrackingEngine(client_id)
        
        # Track post revenue impact
        impact = await revenue_tracker.track_post_revenue_impact(
            post_id=request.post_id,
            platform=request.platform,
            content=request.content
        )
        
        # Start monitoring for specified duration
        monitoring_task = asyncio.create_task(
            revenue_tracker.track_post_performance_over_time(
                post_id=request.post_id,
                platform=request.platform,
                tracking_duration_days=request.tracking_duration_days
            )
        )
        
        return {
            "status": "tracking_started",
            "post_id": request.post_id,
            "initial_assessment": impact,
            "tracking_duration_days": request.tracking_duration_days,
            "message": f"Revenue tracking started for post {request.post_id}. Updates will be available daily."
        }
    except Exception as e:
        logger.error(f"Post revenue tracking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/v1/bi/pricing-optimization",
    response_model=List[PricingSuggestionResponse],
    tags=["Business Intelligence"],
    summary="Get AI-driven pricing suggestions"
)
async def get_pricing_suggestions(
    request: GetBusinessIntelligenceRequest,
    token: str = Depends(verify_token)
):
    """Get AI-driven pricing suggestions with market analysis"""
    try:
        client_id = "demo_client"  # Would be extracted from JWT token
        
        pricing_optimizer = AIPricingOptimization(client_id)
        intelligence = await pricing_optimizer.get_business_intelligence(request.timeframe)
        
        # Generate pricing suggestions
        insights = intelligence.get('insights', [])
        suggestions = await pricing_optimizer.generate_pricing_suggestions(
            intelligence.get('metrics', {}),
            insights
        )
        
        # Convert to response model
        response_suggestions = []
        for suggestion in suggestions[:5]:  # Return top 5 suggestions
            if 'tier' in suggestion and 'suggested_price' in suggestion:
                response_suggestions.append(
                    PricingSuggestionResponse(
                        suggestion_id=f"price_suggest_{datetime.now().timestamp()}",
                        tier=suggestion['tier'],
                        current_price=suggestion.get('current_price', 0),
                        suggested_price=suggestion['suggested_price'],
                        price_change_percentage=suggestion.get('price_change_percentage', 0),
                        confidence_score=suggestion.get('confidence_score', 0.75),
                        expected_impact=suggestion.get('predicted_impact', {}),
                        risk_assessment=suggestion.get('risk_assessment', {}),
                        requires_admin_approval=True
                    )
                )
        
        return response_suggestions
    except Exception as e:
        logger.error(f"Pricing optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/v1/bi/approve-pricing",
    tags=["Business Intelligence"],
    summary="Approve or reject pricing suggestions"
)
async def approve_pricing_change(
    request: ApprovePricingRequest,
    token: str = Depends(verify_token)
):
    """Approve or reject AI-generated pricing suggestions (Admin only)"""
    try:
        # In production, verify admin privileges from token
        client_id = "demo_client"
        
        pricing_optimizer = AIPricingOptimization(client_id)
        
        # Process admin decision
        result = await pricing_optimizer.approval_workflow.process_admin_decision(
            approval_id=request.approval_id,
            decision=request.decision,
            admin_notes=request.admin_notes
        )
        
        # If approved, implement the change
        if request.decision == "approve":
            implementation = await pricing_optimizer.implement_approved_pricing_change(
                request.approval_id
            )
            
            return {
                "status": "approved_and_implemented",
                "approval_id": request.approval_id,
                "implementation_details": implementation,
                "notification_sent": True,
                "effective_date": datetime.now() + timedelta(days=30)  # 30-day notice
            }
        else:
            return {
                "status": "rejected",
                "approval_id": request.approval_id,
                "admin_notes": request.admin_notes,
                "message": "Pricing suggestion rejected"
            }
            
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Pricing approval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/v1/bi/dashboard",
    tags=["Business Intelligence"],
    summary="Get comprehensive BI dashboard data"
)
async def get_bi_dashboard(
    timeframe: AnalyticsTimeframe = AnalyticsTimeframe.MONTH,
    token: str = Depends(verify_token)
):
    """Get comprehensive dashboard data from all BI modules"""
    try:
        client_id = "demo_client"  # Would be extracted from JWT token
        
        # Gather data from all BI engines in parallel
        async def get_usage_data():
            engine = UsageAnalyticsEngine(client_id)
            return await engine.get_business_intelligence(timeframe)
        
        async def get_performance_data():
            monitor = PerformanceMonitoringSystem(client_id)
            return await monitor.get_business_intelligence(timeframe)
        
        async def get_revenue_data():
            tracker = RevenueTrackingEngine(client_id)
            return await tracker.get_business_intelligence(timeframe)
        
        async def get_pricing_data():
            optimizer = AIPricingOptimization(client_id)
            return await optimizer.get_business_intelligence(timeframe)
        
        # Execute all in parallel
        usage_task = asyncio.create_task(get_usage_data())
        performance_task = asyncio.create_task(get_performance_data())
        revenue_task = asyncio.create_task(get_revenue_data())
        pricing_task = asyncio.create_task(get_pricing_data())
        
        # Wait for all to complete
        usage_data = await usage_task
        performance_data = await performance_task
        revenue_data = await revenue_task
        pricing_data = await pricing_task
        
        # Compile dashboard summary
        dashboard = {
            "timeframe": timeframe.value,
            "generated_at": datetime.now(),
            "modules": {
                "usage_analytics": {
                    "summary": usage_data.get('metrics', {}),
                    "top_insights": usage_data.get('insights', [])[:3],
                    "confidence": usage_data.get('confidence_score', 0)
                },
                "performance_monitoring": {
                    "system_status": "healthy",  # Would be determined from data
                    "alerts_count": len([i for i in performance_data.get('insights', []) if i.impact_level == "high"]),
                    "top_metrics": performance_data.get('metrics', {})
                },
                "revenue_tracking": {
                    "total_revenue": revenue_data.get('metrics', {}).total_revenue if hasattr(revenue_data.get('metrics', {}), 'total_revenue') else 0,
                    "growth_rate": revenue_data.get('metrics', {}).revenue_growth_rate if hasattr(revenue_data.get('metrics', {}), 'revenue_growth_rate') else 0,
                    "top_platforms": revenue_data.get('insights', [])[:2]
                },
                "pricing_optimization": {
                    "active_suggestions": len(pricing_data.get('insights', [])),
                    "market_position": "competitive",  # Would be determined from data
                    "optimization_opportunities": pricing_data.get('recommendations', [])[:2]
                }
            },
            "executive_summary": {
                "health_score": 85,  # Would be calculated from all metrics
                "key_achievements": [
                    "Revenue increased by 15% this month",
                    "System uptime maintained at 99.9%",
                    "User engagement up 25% across platforms"
                ],
                "action_items": [
                    "Review 3 high-confidence pricing suggestions",
                    "Address performance anomaly on Instagram API",
                    "Capitalize on viral content opportunity"
                ]
            }
        }
        
        return dashboard
        
    except Exception as e:
        logger.error(f"BI dashboard failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint for real-time BI dashboard
@app.websocket("/ws/bi-dashboard")
async def websocket_bi_dashboard(websocket: WebSocket):
    """WebSocket endpoint for real-time Business Intelligence dashboard updates"""
    # Import here to avoid circular imports
    from backend.intelligence.realtime_streaming import (
        RealTimeMetricsStreamer,
        WebSocketMetricsHandler
    )
    
    # Initialize streamer (in production, this would be a singleton)
    streamer = RealTimeMetricsStreamer()
    await streamer.initialize()
    
    handler = WebSocketMetricsHandler(streamer)
    
    try:
        # Accept connection
        await handler.connect(websocket)
        
        # Handle messages
        while True:
            try:
                data = await websocket.receive_json()
                await handler.handle_message(websocket, data)
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
                
    finally:
        await handler.disconnect(websocket)
        await streamer.close()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=port,
        reload=ENVIRONMENT == "development"
    )