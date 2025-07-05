"""
AutoGuru Universal - Main FastAPI Application

This is the main entry point for the AutoGuru Universal API that provides
social media automation for ANY business niche. All functionality is AI-driven
without hardcoded business logic.
"""

import asyncio
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

# Configure basic logging FIRST to ensure logger is always available
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add current directory to Python path to handle import issues
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from fastapi import FastAPI, Request, Response, HTTPException, Depends, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.responses import JSONResponse, FileResponse
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
    import uvicorn
    from starlette.middleware.base import BaseHTTPMiddleware
    logger.info("FastAPI and core dependencies imported successfully")
except ImportError as e:
    logger.error(f"Failed to import FastAPI core dependencies: {e}")
    sys.exit(1)

# Initialize basic settings to prevent errors
class BasicSettings:
    def __init__(self):
        self.title = "AutoGuru Universal"
        self.description = "Universal social media automation for ANY business niche"
        self.version = "1.0.0"
        self.debug = False
        self.environment = "development"

# Set default environment variables if not present
if not os.getenv("ENVIRONMENT"):
    os.environ["ENVIRONMENT"] = "development"
    
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
logger.info(f"Environment: {ENVIRONMENT}")

# Try to import settings, fall back to basic settings
settings = BasicSettings()
try:
    if ENVIRONMENT == "production":
        from backend.config.production import get_production_settings
        settings = get_production_settings()
        logger.info("Production settings loaded successfully")
    else:
        from backend.config.settings import get_settings
        settings = get_settings()
        logger.info("Development settings loaded successfully")
except ImportError as e:
    logger.warning(f"Could not load settings module: {e}")
    logger.info("Using basic settings")
except Exception as e:
    logger.warning(f"Error loading settings: {e}")
    logger.info("Using basic settings")

# Try to import optional dependencies
try:
    from celery import Celery
    from celery.result import AsyncResult
    CELERY_AVAILABLE = True
    logger.info("Celery imported successfully")
except ImportError as e:
    logger.warning(f"Celery not available: {e}")
    CELERY_AVAILABLE = False
    # Create dummy Celery class
    class Celery:
        def __init__(self, *args, **kwargs):
            pass
        def control(self):
            return self
        def inspect(self):
            return self
        def active(self):
            return {}
        def reserved(self):
            return {}
        def registered(self):
            return {}

try:
    from fastapi import WebSocket, WebSocketDisconnect
    WEBSOCKET_AVAILABLE = True
    logger.info("WebSocket support imported successfully")
except ImportError as e:
    logger.warning(f"WebSocket not available: {e}")
    WEBSOCKET_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
    logger.info("Prometheus metrics imported successfully")
except ImportError as e:
    logger.warning(f"Prometheus not available: {e}")
    PROMETHEUS_AVAILABLE = False

try:
    import sentry_sdk
    SENTRY_AVAILABLE = True
    logger.info("Sentry SDK imported successfully")
except ImportError as e:
    logger.warning(f"Sentry not available: {e}")
    SENTRY_AVAILABLE = False

# Try to import database modules
try:
    from backend.database.connection import get_db_session, get_db_context
    DATABASE_AVAILABLE = True
    logger.info("Database modules imported successfully")
except ImportError as e:
    logger.warning(f"Database modules not available: {e}")
    DATABASE_AVAILABLE = False
    # Create dummy functions
    async def get_db_session():
        return None
    async def get_db_context():
        return None

# Try to import content analyzer
try:
    from backend.core.content_analyzer import UniversalContentAnalyzer
    CONTENT_ANALYZER_AVAILABLE = True
    logger.info("Content analyzer imported successfully")
except ImportError as e:
    logger.warning(f"Content analyzer not available: {e}")
    CONTENT_ANALYZER_AVAILABLE = False
    UniversalContentAnalyzer = None

# Try to import models
try:
    from backend.models.content_models import (
        ContentAnalysis,
        BusinessNiche,
        AudienceProfile,
        Platform,
        ContentFormat,
        PlatformContent
    )
    MODELS_AVAILABLE = True
    logger.info("Content models imported successfully")
except ImportError as e:
    logger.warning(f"Content models not available: {e}")
    MODELS_AVAILABLE = False
    # Create basic model classes
    class ContentAnalysis(BaseModel):
        content: str
        confidence: float = 0.8
        recommendations: List[str] = []
    
    class BusinessNiche(BaseModel):
        niche_type: str
        confidence_score: float = 0.8
        sub_niches: List[str] = []
        reasoning: str = ""
        keywords: List[str] = []
    
    class AudienceProfile(BaseModel):
        demographics: Dict[str, Any] = {}
        interests: List[str] = []
        behavior_patterns: Dict[str, Any] = {}
        platform_preferences: List[str] = []
    
    class Platform(BaseModel):
        name: str
        enabled: bool = True
    
    class ContentFormat(BaseModel):
        format_type: str
        specifications: Dict[str, Any] = {}
    
    class PlatformContent(BaseModel):
        platform: str
        content_text: str
        content_format: str
        hashtags: List[str] = []
        call_to_action: str = ""

# Try to import encryption utilities
try:
    from backend.utils.encryption import encrypt_data, decrypt_data
    ENCRYPTION_AVAILABLE = True
    logger.info("Encryption utilities imported successfully")
except ImportError as e:
    logger.warning(f"Encryption utilities not available: {e}")
    ENCRYPTION_AVAILABLE = False
    # Create dummy functions
    def encrypt_data(data):
        return data
    def decrypt_data(data):
        return data

# Try to import Business Intelligence modules
try:
    from backend.intelligence import (
        UsageAnalyticsEngine,
        PerformanceMonitoringSystem,
        RevenueTrackingEngine,
        AIPricingOptimization,
        AnalyticsTimeframe,
        BusinessMetricType,
        IntelligenceInsight
    )
    INTELLIGENCE_AVAILABLE = True
    logger.info("Business Intelligence modules imported successfully")
except ImportError as e:
    logger.warning(f"Business Intelligence modules not available: {e}")
    INTELLIGENCE_AVAILABLE = False
    # Create dummy classes
    class UsageAnalyticsEngine:
        def __init__(self, client_id): 
            self.client_id = client_id
        async def get_business_intelligence(self, timeframe): 
            return {"metrics": {}, "insights": []}
    
    class PerformanceMonitoringSystem:
        def __init__(self, client_id): 
            self.client_id = client_id
        async def get_business_intelligence(self, timeframe): 
            return {"metrics": {}, "insights": []}
    
    class RevenueTrackingEngine:
        def __init__(self, client_id): 
            self.client_id = client_id
        async def get_business_intelligence(self, timeframe): 
            return {"metrics": {}, "insights": []}
        async def track_post_revenue_impact(self, post_id, platform, content):
            return {"impact": "tracked"}
        async def track_post_performance_over_time(self, post_id, platform, tracking_duration_days):
            return {"tracking": "started"}
    
    class AIPricingOptimization:
        def __init__(self, client_id): 
            self.client_id = client_id
        async def get_business_intelligence(self, timeframe): 
            return {"metrics": {}, "insights": []}
        async def generate_pricing_suggestions(self, metrics, insights):
            return []
        @property
        def approval_workflow(self):
            return self
        async def process_admin_decision(self, approval_id, decision, admin_notes):
            return {"processed": True}
        async def implement_approved_pricing_change(self, approval_id):
            return {"implemented": True}
    
    class AnalyticsTimeframe:
        MONTH = "month"
        WEEK = "week"
        DAY = "day"
        def __init__(self, value):
            self.value = value
    
    class BusinessMetricType:
        pass
    
    class IntelligenceInsight:
        pass

# Initialize Celery
try:
    if CELERY_AVAILABLE:
        celery_broker_url = getattr(settings, 'celery_broker_url', None)
        if celery_broker_url:
            celery_app = Celery(
                'autoguru_universal',
                broker=celery_broker_url,
                backend=getattr(settings, 'celery_result_backend', celery_broker_url)
            )
            logger.info("Celery initialized with configured broker")
        else:
            celery_app = Celery(
                'autoguru_universal',
                broker='redis://localhost:6379',
                backend='redis://localhost:6379'
            )
            logger.info("Celery initialized with default Redis broker")
    else:
        celery_app = Celery('autoguru_universal')
        logger.info("Celery dummy instance created")
except Exception as e:
    logger.warning(f"Could not initialize Celery: {e}")
    celery_app = Celery('autoguru_universal')

# Security
security = HTTPBearer(auto_error=False)

# Basic Request/Response Models
class HealthResponse(BaseModel):
    status: str
    environment: str
    timestamp: datetime
    version: str
    features: List[str]

class LoginRequest(BaseModel):
    email: str = Field(..., description="User email")
    password: str = Field(..., description="User password")

class LoginResponse(BaseModel):
    token: str
    user_id: str
    email: str
    message: str = "Login successful"

# Basic middleware classes
class RequestIdMiddleware(BaseHTTPMiddleware):
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
        # For demo purposes, allow access without token
        return "demo_token"
    
    token = credentials.credentials
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return token

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info(f"Starting AutoGuru Universal in {ENVIRONMENT} environment")
    
    # Initialize database connection if available
    if DATABASE_AVAILABLE:
        try:
            async with get_db_context() as session:
                logger.info("Database connection test successful")
        except Exception as e:
            logger.warning(f"Database connection test failed: {e}")
    
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
try:
    cors_origins = getattr(settings, 'allowed_origins', ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS configured successfully")
except Exception as e:
    logger.warning(f"Could not configure CORS: {e}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Mount static files if available
frontend_path = Path("frontend")
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory="frontend"), name="static")
    logger.info("Frontend static files mounted")

# Routes
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

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
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

@app.get("/api/v1/health", tags=["Health"])
async def api_health_check():
    """API health check endpoint"""
    return {
        "status": "healthy",
        "environment": ENVIRONMENT,
        "timestamp": datetime.utcnow(),
        "version": getattr(settings, 'version', '1.0.0')
    }

@app.post("/auth/login", response_model=LoginResponse, tags=["Authentication"])
async def login(request: LoginRequest):
    """Demo login endpoint"""
    try:
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

@app.get("/demo", tags=["Demo"])
async def demo_analysis():
    """Demo endpoint for testing"""
    return {
        "message": "AutoGuru Universal Demo",
        "status": "available",
        "features": {
            "content_analysis": CONTENT_ANALYZER_AVAILABLE,
            "database": DATABASE_AVAILABLE,
            "models": MODELS_AVAILABLE,
            "encryption": ENCRYPTION_AVAILABLE,
            "intelligence": INTELLIGENCE_AVAILABLE,
            "celery": CELERY_AVAILABLE,
            "websocket": WEBSOCKET_AVAILABLE,
            "prometheus": PROMETHEUS_AVAILABLE,
            "sentry": SENTRY_AVAILABLE
        },
        "environment": ENVIRONMENT
    }

# Initialize Prometheus metrics if available
if PROMETHEUS_AVAILABLE:
    REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'http_status'])
    REQUEST_LATENCY = Histogram('http_request_latency_seconds', 'HTTP request latency', ['endpoint'])
    
    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Initialize Sentry if available
if SENTRY_AVAILABLE:
    sentry_dsn = os.getenv("SENTRY_DSN")
    if sentry_dsn:
        sentry_sdk.init(dsn=sentry_dsn, traces_sample_rate=1.0)
        logger.info("Sentry initialized")

# Global rate limiting state
RATE_LIMIT_STATE = {}

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Basic rate limiting middleware"""
    client_ip = request.client.host
    now = int(time.time())
    minute = now // 60
    
    # Simple rate limiting
    key = f"{client_ip}:{minute}"
    RATE_LIMIT_STATE[key] = RATE_LIMIT_STATE.get(key, 0) + 1
    
    if RATE_LIMIT_STATE[key] > 60:  # 60 requests per minute
        return JSONResponse(
            {"error": "Rate limit exceeded"}, 
            status_code=429
        )
    
    # Clean up old entries
    if len(RATE_LIMIT_STATE) > 1000:
        current_minute = now // 60
        keys_to_remove = [k for k in RATE_LIMIT_STATE.keys() if int(k.split(':')[1]) < current_minute - 5]
        for key in keys_to_remove:
            del RATE_LIMIT_STATE[key]
    
    response = await call_next(request)
    return response

# Add basic endpoints for testing
@app.get("/test", tags=["Test"])
async def test_endpoint():
    """Test endpoint to verify the application is working"""
    return {
        "message": "AutoGuru Universal is working correctly",
        "timestamp": datetime.utcnow(),
        "environment": ENVIRONMENT,
        "status": "success"
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting AutoGuru Universal on port {port}")
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=port,
        reload=ENVIRONMENT == "development"
    )