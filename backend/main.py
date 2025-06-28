"""
AutoGuru Universal - Main FastAPI Application

This is the main entry point for the AutoGuru Universal API that provides
social media automation for ANY business niche. All functionality is AI-driven
without hardcoded business logic.
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, Request, Response, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn
from celery import Celery
from celery.result import AsyncResult

from backend.config.settings import get_settings
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

# Initialize settings and logging
settings = get_settings()

# Build handlers list
handlers: List[logging.Handler] = [logging.StreamHandler()]
if settings.logging.enable_file_logging:
    handlers.append(logging.FileHandler(settings.logging.log_file_path))

logging.basicConfig(
    level=getattr(logging, settings.logging.log_level.value),
    format=settings.logging.log_format,
    handlers=handlers
)
logger = logging.getLogger(__name__)

# Initialize Celery
celery_app = Celery(
    'autoguru_universal',
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend
)

# Security
security = HTTPBearer()


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


# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify JWT token for API authentication"""
    token = credentials.credentials
    
    try:
        # TODO: Implement actual JWT verification
        # For now, this is a placeholder
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        return token
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )


# Lifespan manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting AutoGuru Universal API...")
    
    # Initialize database connection
    db_manager = await get_db_manager()
    await db_manager.initialize()
    logger.info("Database connection pool initialized")
    
    # Initialize content analyzer
    app.state.content_analyzer = UniversalContentAnalyzer(
        openai_api_key=settings.ai_service.openai_api_key.get_secret_value() if settings.ai_service.openai_api_key else None,
        anthropic_api_key=settings.ai_service.anthropic_api_key.get_secret_value() if settings.ai_service.anthropic_api_key else None
    )
    logger.info("Content analyzer initialized")
    
    # Verify Celery connection
    try:
        celery_app.control.inspect().stats()
        logger.info("Celery connection verified")
    except Exception as e:
        logger.warning(f"Celery connection failed: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AutoGuru Universal API...")
    await db_manager.close()
    logger.info("Database connections closed")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    docs_url="/docs" if settings.api_docs_enabled else None,
    redoc_url="/redoc" if settings.api_docs_enabled else None,
    openapi_url="/openapi.json" if settings.api_docs_enabled else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(RequestIdMiddleware)
app.add_middleware(ErrorHandlerMiddleware)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.allowed_origins,
    allow_credentials=True,
    allow_methods=settings.security.allowed_methods,
    allow_headers=settings.security.allowed_headers,
)

# Trusted host middleware for security
if settings.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["autoguru.com", "*.autoguru.com"]
    )


# Health check endpoint
@app.get("/api/v1/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for monitoring.
    
    Returns system health status including database connectivity,
    AI service availability, and worker status.
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.app_version,
        "environment": settings.environment.value
    }
    
    # Check database
    try:
        db_manager = await get_db_manager()
        db_health = await db_manager.health_check()
        health_status["database"] = db_health
    except Exception as e:
        health_status["database"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"
    
    # Check Celery workers
    try:
        stats = celery_app.control.inspect().stats()
        health_status["workers"] = {"status": "healthy", "active_workers": len(stats) if stats else 0}
    except Exception as e:
        health_status["workers"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"
    
    # Check AI services
    health_status["ai_services"] = {
        "openai": "configured" if settings.ai_service.openai_api_key else "not configured",
        "anthropic": "configured" if settings.ai_service.anthropic_api_key else "not configured"
    }
    
    return health_status


# Content analysis endpoint
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
    """
    Analyze content using AI to determine business niche, target audience,
    brand voice, and viral potential across platforms.
    
    This endpoint works universally for ANY business type without
    hardcoded logic. The AI automatically adapts to different industries.
    """
    try:
        analyzer = app.state.content_analyzer
        
        # Start analysis task
        task = celery_app.send_task(
            'tasks.analyze_content',
            args=[request.content, request.context, request.platforms]
        )
        
        # For now, we'll wait for the result (in production, return task ID for async)
        result = await analyzer.analyze_content(
            content=request.content,
            context=request.context,
            platforms=request.platforms
        )
        
        # Log analysis completion
        logger.info(
            f"Content analysis completed - Niche: {result.business_niche.value}, "
            f"Confidence: {result.confidence_score}"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Content analysis failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


# Persona generation endpoint
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
    """
    Generate detailed audience personas based on business description.
    
    Uses AI to create comprehensive personas that include demographics,
    psychographics, pain points, and content preferences for any business type.
    """
    try:
        # Start persona generation task
        task = celery_app.send_task(
            'tasks.generate_persona',
            args=[request.business_description, request.target_market, request.goals]
        )
        
        # Return task ID for async tracking
        return {
            "task_id": task.id,
            "status": "processing",
            "message": "Persona generation started"
        }
        
    except Exception as e:
        logger.error(f"Persona generation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Persona generation failed: {str(e)}"
        )


# Viral content creation endpoint
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
    """
    Create platform-optimized viral content based on topic and audience.
    
    Generates content that's optimized for each platform while maintaining
    brand consistency. Works for any business niche automatically.
    """
    try:
        # Start content creation task
        task = celery_app.send_task(
            'tasks.create_viral_content',
            args=[
                request.topic,
                request.business_niche.dict(),
                request.target_audience,
                [p.value for p in request.platforms],
                request.content_type.value if request.content_type else None
            ]
        )
        
        # Return task ID for async tracking
        return {
            "task_id": task.id,
            "status": "processing",
            "message": "Content creation started"
        }
        
    except Exception as e:
        logger.error(f"Content creation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Content creation failed: {str(e)}"
        )


# Content publishing endpoint
@app.post(
    "/api/v1/publish",
    tags=["Publishing"],
    summary="Publish content to social media platforms"
)
async def publish_content(
    request: PublishContentRequest,
    token: str = Depends(verify_token)
):
    """
    Publish content to social media platforms with optional scheduling.
    
    Handles authentication, rate limiting, and platform-specific requirements
    automatically. Supports immediate publishing or scheduling for later.
    """
    try:
        # Validate platform credentials
        platform_config = settings.get_platform_config(request.content.platform.value)
        if not platform_config or not any(platform_config.values()):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Platform {request.content.platform} not configured"
            )
        
        # Start publishing task
        task = celery_app.send_task(
            'tasks.publish_content',
            args=[
                request.content.dict(),
                request.schedule_time.isoformat() if request.schedule_time else None,
                request.cross_post
            ]
        )
        
        return {
            "task_id": task.id,
            "status": "scheduled" if request.schedule_time else "publishing",
            "message": f"Content {'scheduled for publishing' if request.schedule_time else 'publishing'}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Publishing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Publishing failed: {str(e)}"
        )


# Task status endpoint
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
    """
    Get the status of a background task.
    
    Returns task status, progress, and results when available.
    """
    try:
        result = AsyncResult(task_id, app=celery_app)
        
        response = TaskStatusResponse(
            task_id=task_id,
            status=result.status
        )
        
        if result.ready():
            if result.successful():
                response.result = result.result
            else:
                response.error = str(result.info)
        elif result.status == "PENDING":
            response.progress = 0.0
        else:
            response.progress = result.info.get("progress", 0.0) if isinstance(result.info, dict) else None
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get task status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task status: {str(e)}"
        )


# Rate limiting endpoint info
@app.get(
    "/api/v1/rate-limits",
    tags=["System"],
    summary="Get current rate limit status"
)
async def get_rate_limits(token: str = Depends(verify_token)):
    """
    Get current rate limit status for all platforms and services.
    
    Returns remaining API calls and reset times for each integrated service.
    """
    rate_limits = {
        "global": {
            "requests_per_minute": settings.rate_limit.requests_per_minute,
            "requests_per_hour": settings.rate_limit.requests_per_hour
        },
        "platforms": {
            "twitter": {
                "limit": settings.rate_limit.twitter_requests_per_15min,
                "window": "15 minutes"
            },
            "linkedin": {
                "limit": settings.rate_limit.linkedin_requests_per_day,
                "window": "24 hours"
            },
            "facebook": {
                "limit": settings.rate_limit.facebook_requests_per_hour,
                "window": "1 hour"
            },
            "instagram": {
                "limit": settings.rate_limit.instagram_requests_per_hour,
                "window": "1 hour"
            },
            "youtube": {
                "limit": settings.rate_limit.youtube_requests_per_day,
                "window": "24 hours"
            },
            "tiktok": {
                "limit": settings.rate_limit.tiktok_requests_per_hour,
                "window": "1 hour"
            }
        },
        "ai_services": {
            "openai": {
                "limit": settings.rate_limit.openai_requests_per_minute,
                "window": "1 minute"
            },
            "anthropic": {
                "limit": settings.rate_limit.anthropic_requests_per_minute,
                "window": "1 minute"
            }
        }
    }
    
    return rate_limits


# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.logging.log_level.value.lower()
    )