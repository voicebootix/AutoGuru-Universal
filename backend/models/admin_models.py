"""
Admin Models for AutoGuru Universal

This module defines database models for admin functionality including
platform credentials, user management, and system configuration.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from sqlalchemy import Column, String, DateTime, JSON, Boolean, Integer, Text, Float
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, Field

Base = declarative_base()


class PlatformType(str, Enum):
    """Supported social media platforms"""
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    TIKTOK = "tiktok"
    YOUTUBE = "youtube"
    PINTEREST = "pinterest"
    SNAPCHAT = "snapchat"
    REDDIT = "reddit"


class AIServiceType(str, Enum):
    """Supported AI services"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE_PALM = "google_palm"
    STABLE_DIFFUSION = "stable_diffusion"
    ELEVENLABS = "elevenlabs"
    RUNWAY_ML = "runway_ml"
    REPLICATE = "replicate"


class CredentialStatus(str, Enum):
    """Status of platform credentials"""
    ACTIVE = "active"
    EXPIRED = "expired"
    INVALID = "invalid"
    PENDING = "pending"
    REVOKED = "revoked"


class AdminRole(str, Enum):
    """Admin user roles"""
    SUPER_ADMIN = "super_admin"
    PLATFORM_ADMIN = "platform_admin"
    USER_ADMIN = "user_admin"
    REVENUE_ADMIN = "revenue_admin"
    SUPPORT_ADMIN = "support_admin"


class ConnectionStatus(str, Enum):
    """API connection status"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    TESTING = "testing"
    RATE_LIMITED = "rate_limited"


# Database Models
class PlatformCredentials(Base):
    """Store encrypted platform API credentials"""
    __tablename__ = "platform_credentials"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    platform_type = Column(String(50), nullable=False)
    credential_name = Column(String(100), nullable=False)
    encrypted_value = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    status = Column(String(20), default=CredentialStatus.ACTIVE.value)
    extra_data = Column(JSON, default=dict)
    created_by = Column(String(100), nullable=False)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "platform_type": self.platform_type,
            "credential_name": self.credential_name,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "extra_data": self.extra_data
        }


class AdminUser(Base):
    """Admin user accounts"""
    __tablename__ = "admin_users"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    permissions = Column(JSON, default=list)
    extra_data = Column(JSON, default=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "permissions": self.permissions
        }


class SystemConfiguration(Base):
    """Dynamic system configuration"""
    __tablename__ = "system_configuration"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    config_key = Column(String(100), unique=True, nullable=False)
    config_value = Column(JSON, nullable=False)
    config_type = Column(String(50), nullable=False)
    description = Column(Text, nullable=True)
    is_sensitive = Column(Boolean, default=False)
    requires_restart = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String(100), nullable=False)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "config_key": self.config_key,
            "config_value": self.config_value if not self.is_sensitive else "***HIDDEN***",
            "config_type": self.config_type,
            "description": self.description,
            "requires_restart": self.requires_restart,
            "updated_at": self.updated_at.isoformat(),
            "updated_by": self.updated_by
        }


class APIConnectionLog(Base):
    """Log API connection attempts and status"""
    __tablename__ = "api_connection_logs"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    platform_type = Column(String(50), nullable=False)
    connection_status = Column(String(20), nullable=False)
    response_time_ms = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    tested_at = Column(DateTime, default=datetime.utcnow)
    tested_by = Column(String(100), nullable=False)
    extra_data = Column(JSON, default=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "platform_type": self.platform_type,
            "connection_status": self.connection_status,
            "response_time_ms": self.response_time_ms,
            "error_message": self.error_message,
            "tested_at": self.tested_at.isoformat(),
            "tested_by": self.tested_by,
            "extra_data": self.extra_data
        }


class UserPlatformConnection(Base):
    """Track user's platform connections"""
    __tablename__ = "user_platform_connections"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(String(100), nullable=False)
    platform_type = Column(String(50), nullable=False)
    platform_user_id = Column(String(255), nullable=False)
    platform_username = Column(String(255), nullable=True)
    access_token = Column(Text, nullable=False)  # Encrypted
    refresh_token = Column(Text, nullable=True)  # Encrypted
    token_expires_at = Column(DateTime, nullable=True)
    connection_status = Column(String(20), default=ConnectionStatus.CONNECTED.value)
    connected_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)
    permissions = Column(JSON, default=list)
    extra_data = Column(JSON, default=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "platform_type": self.platform_type,
            "platform_username": self.platform_username,
            "connection_status": self.connection_status,
            "connected_at": self.connected_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "token_expires_at": self.token_expires_at.isoformat() if self.token_expires_at else None,
            "permissions": self.permissions
        }


# Pydantic Models for API
class PlatformCredentialCreate(BaseModel):
    """Create platform credentials"""
    platform_type: PlatformType
    credential_name: str = Field(..., min_length=1, max_length=100)
    credential_value: str = Field(..., min_length=1)
    expires_at: Optional[datetime] = None
    extra_data: Dict[str, Any] = Field(default_factory=dict)


class PlatformCredentialUpdate(BaseModel):
    """Update platform credentials"""
    credential_value: Optional[str] = None
    expires_at: Optional[datetime] = None
    status: Optional[CredentialStatus] = None
    extra_data: Optional[Dict[str, Any]] = None


class AdminUserCreate(BaseModel):
    """Create admin user"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    password: str = Field(..., min_length=8)
    role: AdminRole
    permissions: List[str] = Field(default_factory=list)


class AdminUserUpdate(BaseModel):
    """Update admin user"""
    email: Optional[str] = Field(None, regex=r'^[^@]+@[^@]+\.[^@]+$')
    role: Optional[AdminRole] = None
    is_active: Optional[bool] = None
    permissions: Optional[List[str]] = None


class SystemConfigUpdate(BaseModel):
    """Update system configuration"""
    config_value: Any
    description: Optional[str] = None


class ConnectionTestRequest(BaseModel):
    """Test platform connection"""
    platform_type: PlatformType
    test_endpoints: List[str] = Field(default_factory=list)
    timeout_seconds: int = Field(default=30, ge=5, le=120)


class ConnectionTestResult(BaseModel):
    """Connection test result"""
    platform_type: PlatformType
    success: bool
    response_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    endpoint_results: List[Dict[str, Any]] = Field(default_factory=list)
    tested_at: datetime = Field(default_factory=datetime.utcnow)


@dataclass
class PlatformConfiguration:
    """Platform-specific configuration"""
    platform_type: PlatformType
    app_id: Optional[str] = None
    app_secret: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    webhook_url: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    rate_limits: Dict[str, int] = field(default_factory=dict)
    endpoints: Dict[str, str] = field(default_factory=dict)
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform_type": self.platform_type.value,
            "app_id": self.app_id,
            "app_secret": "***HIDDEN***" if self.app_secret else None,
            "access_token": "***HIDDEN***" if self.access_token else None,
            "webhook_url": self.webhook_url,
            "permissions": self.permissions,
            "rate_limits": self.rate_limits,
            "endpoints": self.endpoints,
            "extra_data": self.extra_data
        }


@dataclass
class AIServiceConfiguration:
    """AI service configuration"""
    service_type: AIServiceType
    api_key: Optional[str] = None
    organization_id: Optional[str] = None
    model_preferences: Dict[str, str] = field(default_factory=dict)
    usage_limits: Dict[str, int] = field(default_factory=dict)
    cost_tracking: Dict[str, float] = field(default_factory=dict)
    endpoints: Dict[str, str] = field(default_factory=dict)
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_type": self.service_type.value,
            "api_key": "***HIDDEN***" if self.api_key else None,
            "organization_id": self.organization_id,
            "model_preferences": self.model_preferences,
            "usage_limits": self.usage_limits,
            "cost_tracking": self.cost_tracking,
            "endpoints": self.endpoints,
            "extra_data": self.extra_data
        }


class AdminDashboardData(BaseModel):
    """Admin dashboard data"""
    system_health: Dict[str, Any]
    platform_status: Dict[str, Any]
    user_metrics: Dict[str, Any]
    revenue_metrics: Dict[str, Any]
    recent_activities: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]
    quick_stats: Dict[str, Any]


class UserConnectionRequest(BaseModel):
    """User platform connection request"""
    platform_type: PlatformType
    oauth_code: Optional[str] = None
    redirect_uri: Optional[str] = None
    permissions: List[str] = Field(default_factory=list)


class BulkOperationRequest(BaseModel):
    """Bulk operation request"""
    operation_type: str
    target_ids: List[str]
    parameters: Dict[str, Any] = Field(default_factory=dict)


class BulkOperationResult(BaseModel):
    """Bulk operation result"""
    operation_type: str
    total_targets: int
    successful: int
    failed: int
    results: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]