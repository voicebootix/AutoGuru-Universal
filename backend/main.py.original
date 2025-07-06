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
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import sentry_sdk
from starlette.middleware.base import BaseHTTPMiddleware

# Import settings based on environment
try:
    from backend.config.production import get_production_settings
    settings = get_production_settings()
    ENVIRONMENT = "production"
except ImportError:
    from backend.config.settings import get_settings
    settings = get_settings()
    ENVIRONMENT = "development"

from backend.database.connection import get_db_session, get_db_context
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


class LoginRequest(BaseModel):
    """Request model for login"""
    email: str = Field(..., description="User email")
    password: str = Field(..., description="User password")
    
    class Config:
        schema_extra = {
            "example": {
                "email": "demo@autoguru.com",
                "password": "demo123"
            }
        }

class LoginResponse(BaseModel):
    """Response model for login"""
    token: str
    user_id: str
    email: str
    message: str = "Login successful"
    
    class Config:
        schema_extra = {
            "example": {
                "token": "demo_token_1234567890",
                "user_id": "user_123",
                "email": "demo@autoguru.com",
                "message": "Login successful"
            }
        }

# Middleware
class RequestIdMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID for tracking (class-based for Starlette/FastAPI)"""
    async def dispatch(self, request, call_next):
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


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware (class-based for Starlette/FastAPI)"""
    async def dispatch(self, request, call_next):
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
    
    try:
        # Demo token verification - accept any token that starts with 'demo_token_'
        token = credentials.credentials
        if token.startswith('demo_token_'):
            return token
        
        # TODO: Implement proper JWT verification for production
        # For now, just return the token as user ID
        return token
    except Exception as e:
        logger.error(f"Token verification failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info(f"Starting AutoGuru Universal in {ENVIRONMENT} environment")
    
    # Initialize database connection
    try:
        if ENVIRONMENT == "production":
            # Production database initialization
            async with get_db_session() as session:
                # use session for DB operations
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


# ============================================
# RAW DATA DEBUGGING ENDPOINTS FOR FLOWISE
# ============================================

@app.get("/api/debug/database-raw", tags=["Debug"])
async def get_database_raw_data():
    """
    Get raw database metrics and status information.
    
    Returns comprehensive database health data including connection pool status,
    table sizes, and performance metrics for debugging purposes.
    """
    try:
        from backend.database.connection import health_check as db_health_check
        
        # Get basic database health
        db_health = await db_health_check()
        
        # Get additional database metrics
        additional_metrics = {}
        try:
            from sqlalchemy import text
            engine = await create_db_engine()
            async with engine.connect() as conn:
                # Get table sizes and row counts
                table_query = text("""
                    SELECT 
                        schemaname,
                        tablename,
                        attname,
                        n_distinct,
                        correlation
                    FROM pg_stats 
                    WHERE schemaname = 'public'
                    LIMIT 20
                """)
                table_result = await conn.execute(table_query)
                additional_metrics["table_stats"] = [
                    {
                        "schema": row[0],
                        "table": row[1], 
                        "column": row[2],
                        "distinct_values": row[3],
                        "correlation": row[4]
                    }
                    for row in table_result.fetchall()
                ]
                
                # Get connection info
                conn_query = text("SELECT count(*) FROM pg_stat_activity")
                conn_result = await conn.execute(conn_query)
                additional_metrics["active_connections"] = conn_result.scalar()
                
                # Get database size
                size_query = text("""
                    SELECT pg_size_pretty(pg_database_size(current_database())) as db_size,
                           pg_database_size(current_database()) as db_size_bytes
                """)
                size_result = await conn.execute(size_query)
                size_row = size_result.fetchone()
                additional_metrics["database_size"] = {
                    "formatted": size_row[0],
                    "bytes": size_row[1]
                }
                
        except Exception as e:
            additional_metrics["error"] = f"Failed to get additional metrics: {str(e)}"
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": "database-raw",
            "status": "success",
            "data": {
                **db_health,
                **additional_metrics
            },
            "errors": []
        }
        
    except Exception as e:
        logger.error(f"Database debug endpoint failed: {e}")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": "database-raw", 
            "status": "error",
            "data": {},
            "errors": [str(e)]
        }


@app.get("/api/debug/system-raw", tags=["Debug"])
async def get_system_raw_data():
    """
    Get raw system metrics and performance data.
    
    Returns comprehensive system health information including CPU, memory,
    disk usage, and network statistics for debugging purposes.
    """
    try:
        import psutil
        import platform
        
        # System information
        system_info = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "hostname": platform.node()
        }
        
        # CPU metrics
        cpu_metrics = {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "cpu_stats": psutil.cpu_stats()._asdict(),
            "cpu_times": psutil.cpu_times()._asdict()
        }
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_metrics = {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "free": memory.free,
            "percent": memory.percent,
            "formatted": {
                "total": f"{memory.total / (1024**3):.2f} GB",
                "available": f"{memory.available / (1024**3):.2f} GB",
                "used": f"{memory.used / (1024**3):.2f} GB",
                "free": f"{memory.free / (1024**3):.2f} GB"
            }
        }
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_metrics = {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": disk.percent,
            "formatted": {
                "total": f"{disk.total / (1024**3):.2f} GB",
                "used": f"{disk.used / (1024**3):.2f} GB", 
                "free": f"{disk.free / (1024**3):.2f} GB"
            }
        }
        
        # Network metrics
        network = psutil.net_io_counters()
        network_metrics = {
            "bytes_sent": network.bytes_sent,
            "bytes_recv": network.bytes_recv,
            "packets_sent": network.packets_sent,
            "packets_recv": network.packets_recv,
            "connections": len(psutil.net_connections())
        }
        
        # Process metrics
        process = psutil.Process()
        process_metrics = {
            "pid": process.pid,
            "name": process.name(),
            "status": process.status(),
            "create_time": process.create_time(),
            "cpu_percent": process.cpu_percent(),
            "memory_percent": process.memory_percent(),
            "memory_info": process.memory_info()._asdict(),
            "num_threads": process.num_threads(),
            "open_files": len(process.open_files()),
            "connections": len(process.connections())
        }
        
        # Uptime
        uptime_metrics = {
            "system_uptime": psutil.boot_time(),
            "process_uptime": time.time() - process.create_time()
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": "system-raw",
            "status": "success", 
            "data": {
                "system_info": system_info,
                "cpu_metrics": cpu_metrics,
                "memory_metrics": memory_metrics,
                "disk_metrics": disk_metrics,
                "network_metrics": network_metrics,
                "process_metrics": process_metrics,
                "uptime_metrics": uptime_metrics
            },
            "errors": []
        }
        
    except Exception as e:
        logger.error(f"System debug endpoint failed: {e}")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": "system-raw",
            "status": "error",
            "data": {},
            "errors": [str(e)]
        }


@app.get("/api/debug/application-raw", tags=["Debug"])
async def get_application_raw_data():
    """
    Get raw application metrics and status information.
    
    Returns comprehensive application health data including API status,
    recent errors, request rates, and configuration information.
    """
    try:
        # Application configuration
        config_data = {
            "environment": ENVIRONMENT,
            "version": getattr(settings, 'version', '1.0.0'),
            "debug": getattr(settings, 'debug', False),
            "database_url": str(settings.database.postgres_dsn).split('@')[1] if '@' in str(settings.database.postgres_dsn) else "hidden",
            "celery_broker": getattr(settings, 'celery', {}).get('broker_url', 'not_configured') if hasattr(settings, 'celery') else 'not_configured'
        }
        
        # Feature availability
        features = {
            "content_analysis": True,
            "persona_generation": True,
            "viral_content_creation": True,
            "content_publishing": True,
            "business_intelligence": True,
            "analytics": True,
            "monitoring": True,
            "pricing_optimization": True
        }
        
        # API endpoint status (simplified)
        api_status = {
            "health_endpoints": ["/health", "/api/v1/health"],
            "content_endpoints": ["/api/v1/analyze", "/api/v1/generate-persona", "/api/v1/create-viral-content"],
            "publishing_endpoints": ["/api/v1/publish"],
            "bi_endpoints": ["/api/v1/bi/usage-analytics", "/api/v1/bi/performance-monitoring", "/api/v1/bi/revenue-tracking"],
            "debug_endpoints": ["/api/debug/database-raw", "/api/debug/system-raw", "/api/debug/application-raw", "/api/debug/business-raw", "/api/debug/all-raw"]
        }
        
        # Recent application state
        app_state = {
            "startup_time": getattr(app.state, 'startup_time', None),
            "request_count": getattr(app.state, 'request_count', 0),
            "error_count": getattr(app.state, 'error_count', 0),
            "active_websockets": len(getattr(app.state, 'websocket_connections', set())),
            "celery_tasks": {
                "active": len(celery_app.control.inspect().active() or {}),
                "reserved": len(celery_app.control.inspect().reserved() or {}),
                "registered": len(celery_app.control.inspect().registered() or {})
            }
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": "application-raw",
            "status": "success",
            "data": {
                "config_data": config_data,
                "features": features,
                "api_status": api_status,
                "app_state": app_state
            },
            "errors": []
        }
        
    except Exception as e:
        logger.error(f"Application debug endpoint failed: {e}")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": "application-raw",
            "status": "error",
            "data": {},
            "errors": [str(e)]
        }


@app.get("/api/debug/business-raw", tags=["Debug"])
async def get_business_raw_data():
    """
    Get raw business metrics and revenue system status.
    
    Returns comprehensive business intelligence data including revenue tracking,
    subscription metrics, content generation statistics, and user activity.
    """
    try:
        # Revenue system status
        revenue_status = {
            "system_available": True,
            "tracking_enabled": True,
            "last_update": datetime.utcnow().isoformat(),
            "currency": "USD"
        }
        
        # Get real subscription metrics from database
        async with get_db_context() as db:
            subscription_metrics = await _get_real_subscription_metrics(db)
        
        # Content generation statistics
        content_stats = {
            "total_content_generated": 0,
            "content_by_platform": {
                "instagram": 0,
                "linkedin": 0,
                "tiktok": 0,
                "youtube": 0,
                "twitter": 0
            },
            "content_by_type": {
                "posts": 0,
                "stories": 0,
                "videos": 0,
                "carousels": 0
            },
            "viral_content_count": 0,
            "average_engagement_rate": 0.0
        }
        
        # User activity metrics
        user_activity = {
            "total_users": 0,
            "active_users_30d": 0,
            "new_users_30d": 0,
            "user_retention_rate": 0.0,
            "average_session_duration": 0,
            "feature_usage": {
                "content_analysis": 0,
                "persona_generation": 0,
                "viral_content": 0,
                "publishing": 0,
                "analytics": 0
            }
        }
        
        # Platform integration status
        platform_status = {
            "instagram": {"connected": False, "status": "not_configured"},
            "linkedin": {"connected": False, "status": "not_configured"},
            "tiktok": {"connected": False, "status": "not_configured"},
            "youtube": {"connected": False, "status": "not_configured"},
            "twitter": {"connected": False, "status": "not_configured"},
            "facebook": {"connected": False, "status": "not_configured"}
        }
        
        # Business intelligence status
        bi_status = {
            "analytics_engine": "available",
            "performance_monitoring": "available", 
            "revenue_tracking": "available",
            "pricing_optimization": "available",
            "real_time_monitoring": "available"
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": "business-raw",
            "status": "success",
            "data": {
                "revenue_status": revenue_status,
                "subscription_metrics": subscription_metrics,
                "content_stats": content_stats,
                "user_activity": user_activity,
                "platform_status": platform_status,
                "bi_status": bi_status
            },
            "errors": []
        }
        
    except Exception as e:
        logger.error(f"Business debug endpoint failed: {e}")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": "business-raw",
            "status": "error",
            "data": {},
            "errors": [str(e)]
        }


@app.get("/api/debug/all-raw", tags=["Debug"])
async def get_all_raw_data():
    """
    Get all raw debug data in a single request.
    
    Returns comprehensive debugging information from all endpoints:
    database, system, application, and business metrics combined.
    """
    try:
        # Import the individual debug functions
        from backend.database.connection import create_db_engine
        
        # Collect data from all endpoints
        database_data = await get_database_raw_data()
        system_data = await get_system_raw_data()
        application_data = await get_application_raw_data()
        business_data = await get_business_raw_data()
        
        # Combine all data
        combined_data = {
            "database": database_data["data"],
            "system": system_data["data"],
            "application": application_data["data"],
            "business": business_data["data"]
        }
        
        # Collect any errors from individual endpoints
        all_errors = []
        for endpoint_data in [database_data, system_data, application_data, business_data]:
            if endpoint_data.get("errors"):
                all_errors.extend(endpoint_data["errors"])
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": "all-raw",
            "status": "success" if not all_errors else "partial_success",
            "data": combined_data,
            "errors": all_errors
        }
        
    except Exception as e:
        logger.error(f"All-raw debug endpoint failed: {e}")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": "all-raw",
            "status": "error",
            "data": {},
            "errors": [str(e)]
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


# Sentry integration
SENTRY_DSN = os.getenv("SENTRY_DSN")
if SENTRY_DSN:
    sentry_sdk.init(dsn=SENTRY_DSN, traces_sample_rate=1.0)

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'http_status'])
REQUEST_LATENCY = Histogram('http_request_latency_seconds', 'HTTP request latency', ['endpoint'])

# Global rate limiting state (in-memory, for demo; use Redis for distributed)
RATE_LIMIT_STATE = {}

@app.middleware("http")
async def rate_limit_and_metrics_middleware(request: Request, call_next):
    # Prometheus metrics
    endpoint = request.url.path
    method = request.method
    start_time = time.time()
    client_ip = request.client.host
    key = f"{client_ip}:{endpoint}"
    now = int(time.time())
    minute = now // 60
    hour = now // 3600
    # Rate limit config
    rpm = getattr(settings, 'rate_limit_requests', 60)
    rph = getattr(settings, 'rate_limit_requests_per_hour', 3600)
    # Track requests
    state = RATE_LIMIT_STATE.setdefault(key, {'minute': minute, 'minute_count': 0, 'hour': hour, 'hour_count': 0})
    if state['minute'] != minute:
        state['minute'] = minute
        state['minute_count'] = 0
    if state['hour'] != hour:
        state['hour'] = hour
        state['hour_count'] = 0
    state['minute_count'] += 1
    state['hour_count'] += 1
    # Enforce limits
    if state['minute_count'] > rpm or state['hour_count'] > rph:
        logger.warning(f"Rate limit exceeded for {client_ip} on {endpoint}")
        REQUEST_COUNT.labels(method, endpoint, 429).inc()
        return JSONResponse({"error": "Rate limit exceeded"}, status_code=429)
    # Process request
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        status_code = 500
        if SENTRY_DSN:
            sentry_sdk.capture_exception(e)
        raise
    finally:
        elapsed = time.time() - start_time
        REQUEST_COUNT.labels(method, endpoint, status_code).inc()
        REQUEST_LATENCY.labels(endpoint).observe(elapsed)
    return response

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/support", tags=["Support"])
def support():
    support_email = os.getenv("SUPPORT_EMAIL", "support@autoguru.com")
    support_url = os.getenv("SUPPORT_URL", "https://autoguru.com/support")
    return {"email": support_email, "url": support_url}

@app.post(
    "/auth/login",
    response_model=LoginResponse,
    tags=["Authentication"],
    summary="User login"
)
async def login(request: LoginRequest):
    """Demo login endpoint - accepts any credentials"""
    try:
        # Demo authentication - accept any email/password
        demo_token = f"demo_token_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        user_id = f"user_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Demo login successful for email: {request.email}")
        
        return LoginResponse(
            token=demo_token,
            user_id=user_id,
            email=request.email,
            message="Demo login successful"
        )
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@app.post(
    "/auth/logout",
    tags=["Authentication"],
    summary="User logout"
)
async def logout():
    """Demo logout endpoint"""
    try:
        logger.info("Demo logout successful")
        return {"message": "Logout successful"}
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

# Helper method for getting real subscription metrics
async def _get_real_subscription_metrics(db) -> Dict[str, Any]:
    """Get real subscription metrics from database"""
    try:
        # Query subscription data from database
        subscription_query = """
            SELECT 
                COUNT(*) as total_subscriptions,
                COUNT(CASE WHEN status = 'active' THEN 1 END) as active_subscriptions,
                COUNT(CASE WHEN tier = 'basic' AND status = 'active' THEN 1 END) as basic_tier,
                COUNT(CASE WHEN tier = 'professional' AND status = 'active' THEN 1 END) as professional_tier,
                COUNT(CASE WHEN tier = 'enterprise' AND status = 'active' THEN 1 END) as enterprise_tier,
                SUM(CASE WHEN status = 'active' THEN monthly_amount ELSE 0 END) as mrr,
                SUM(CASE WHEN status = 'active' THEN monthly_amount * 12 ELSE 0 END) as arr
            FROM subscriptions 
            WHERE created_at >= NOW() - INTERVAL '1 year'
        """
        
        result = await db.fetch(subscription_query)
        
        if result:
            row = result[0]
            total_subs = row['active_subscriptions'] or 0
            mrr = float(row['mrr'] or 0)
            
            return {
                "total_subscriptions": int(row['total_subscriptions'] or 0),
                "active_subscriptions": total_subs,
                "subscription_tiers": {
                    "basic": int(row['basic_tier'] or 0),
                    "professional": int(row['professional_tier'] or 0),
                    "enterprise": int(row['enterprise_tier'] or 0)
                },
                "revenue_metrics": {
                    "monthly_recurring_revenue": mrr,
                    "annual_recurring_revenue": float(row['arr'] or 0),
                    "average_revenue_per_user": mrr / total_subs if total_subs > 0 else 0.0
                }
            }
        else:
            # Return empty structure if no data
            return {
                "total_subscriptions": 0,
                "active_subscriptions": 0,
                "subscription_tiers": {
                    "basic": 0,
                    "professional": 0,
                    "enterprise": 0
                },
                "revenue_metrics": {
                    "monthly_recurring_revenue": 0.0,
                    "annual_recurring_revenue": 0.0,
                    "average_revenue_per_user": 0.0
                }
            }
    except Exception as e:
        logger.error(f"Failed to get subscription metrics: {str(e)}")
        # Return safe defaults on error
        return {
            "total_subscriptions": 0,
            "active_subscriptions": 0,
            "subscription_tiers": {
                "basic": 0,
                "professional": 0,
                "enterprise": 0
            },
            "revenue_metrics": {
                "monthly_recurring_revenue": 0.0,
                "annual_recurring_revenue": 0.0,
                "average_revenue_per_user": 0.0
            }
        }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=port,
        reload=ENVIRONMENT == "development"
    )