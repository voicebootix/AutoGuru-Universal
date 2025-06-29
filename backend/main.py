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
from datetime import datetime
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

# Import settings based on environment
try:
    from backend.config.production import production_settings as settings
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
        "Universal Business Support"
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


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=port,
        reload=ENVIRONMENT == "development"
    )