"""
Universal Admin Base Module

Provides core functionality for all admin control modules including
permission management, audit logging, approval workflows, and more.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import uuid
import json
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, insert
from backend.database.connection import get_db_session
from backend.utils.encryption import encrypt_data, decrypt_data
from backend.config.settings import settings

# Configure logging
logger = logging.getLogger(__name__)


class AdminPermissionLevel(Enum):
    """Admin permission levels"""
    READ_ONLY = "read_only"
    REVIEWER = "reviewer"
    MODERATOR = "moderator"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class ApprovalStatus(Enum):
    """Approval workflow statuses"""
    PENDING = "pending"
    REVIEWING = "reviewing"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"
    FAILED = "failed"


class AdminActionType(Enum):
    """Types of admin actions"""
    PRICING_CHANGE = "pricing_change"
    FEATURE_TOGGLE = "feature_toggle"
    CLIENT_SUSPENSION = "client_suspension"
    SYSTEM_CONFIGURATION = "system_configuration"
    AI_MODEL_UPDATE = "ai_model_update"
    REVENUE_ADJUSTMENT = "revenue_adjustment"


@dataclass
class AdminAction:
    """Represents an admin action requiring approval"""
    action_id: str
    admin_id: str
    action_type: AdminActionType
    target_client_id: Optional[str]
    description: str
    data: Dict[str, Any]
    status: ApprovalStatus
    requested_at: datetime
    reviewed_at: Optional[datetime] = None
    reviewed_by: Optional[str] = None
    implemented_at: Optional[datetime] = None
    impact_prediction: Dict[str, Any] = field(default_factory=dict)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdminUser:
    """Represents an admin user"""
    admin_id: str
    email: str
    name: str
    permission_level: AdminPermissionLevel
    active: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    permissions: List[str] = field(default_factory=list)
    managed_clients: List[str] = field(default_factory=list)


class AdminPermissionError(Exception):
    """Raised when admin lacks required permissions"""
    pass


class AdminActionError(Exception):
    """Raised when admin action fails"""
    pass


class AdminAuditLogger:
    """Handles audit logging for admin actions"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.audit")
    
    async def log_action(self, admin_id: str, action: AdminAction, result: Dict[str, Any], timestamp: datetime):
        """Log an admin action for audit trail"""
        try:
            async with get_db_session() as session:
                audit_entry = {
                    'audit_id': str(uuid.uuid4()),
                    'admin_id': admin_id,
                    'action_id': action.action_id,
                    'action_type': action.action_type.value,
                    'target_client_id': action.target_client_id,
                    'action_data': encrypt_data(json.dumps(action.data)),
                    'result_data': encrypt_data(json.dumps(result)),
                    'timestamp': timestamp,
                    'status': action.status.value
                }
                
                # Store in audit log table
                await session.execute(
                    insert(settings.ADMIN_AUDIT_TABLE).values(**audit_entry)
                )
                await session.commit()
                
                # Also log to file
                self.logger.info(f"Admin action logged: {admin_id} - {action.action_type.value} - {action.status.value}")
                
        except Exception as e:
            self.logger.error(f"Failed to log admin action: {str(e)}")
            raise


class AdminNotificationManager:
    """Manages notifications between admins"""
    
    def __init__(self):
        self.notification_queue = asyncio.Queue()
        self.logger = logging.getLogger(f"{__name__}.notifications")
    
    async def send_notification(self, notification_type: str, data: Dict[str, Any], sender_admin_id: str):
        """Send notification to relevant admins"""
        try:
            notification = {
                'notification_id': str(uuid.uuid4()),
                'type': notification_type,
                'data': data,
                'sender_admin_id': sender_admin_id,
                'created_at': datetime.now(),
                'priority': self._determine_priority(notification_type, data)
            }
            
            # Add to queue for processing
            await self.notification_queue.put(notification)
            
            # For critical notifications, also send email/SMS
            if notification['priority'] == 'critical':
                await self._send_critical_notification(notification)
            
            self.logger.info(f"Notification sent: {notification_type} from {sender_admin_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send notification: {str(e)}")
            raise
    
    def _determine_priority(self, notification_type: str, data: Dict[str, Any]) -> str:
        """Determine notification priority"""
        critical_types = ['system_failure', 'security_breach', 'revenue_loss']
        high_types = ['approval_required', 'urgent_review', 'client_issue']
        
        if notification_type in critical_types:
            return 'critical'
        elif notification_type in high_types:
            return 'high'
        else:
            return 'normal'
    
    async def _send_critical_notification(self, notification: Dict[str, Any]):
        """Send critical notifications via email/SMS"""
        # Implementation would integrate with email/SMS service
        pass


class PermissionValidator:
    """Validates admin permissions"""
    
    def __init__(self):
        self.permission_hierarchy = {
            AdminPermissionLevel.SUPER_ADMIN: 5,
            AdminPermissionLevel.ADMIN: 4,
            AdminPermissionLevel.MODERATOR: 3,
            AdminPermissionLevel.REVIEWER: 2,
            AdminPermissionLevel.READ_ONLY: 1
        }
        
        self.permission_requirements = {
            'review_pricing': AdminPermissionLevel.REVIEWER,
            'approve_actions': AdminPermissionLevel.MODERATOR,
            'export_reports': AdminPermissionLevel.REVIEWER,
            'modify_optimization_settings': AdminPermissionLevel.ADMIN,
            'create_ab_tests': AdminPermissionLevel.MODERATOR,
            'bulk_review_suggestions': AdminPermissionLevel.MODERATOR,
            'rollback_optimization_changes': AdminPermissionLevel.ADMIN,
            'system_configuration': AdminPermissionLevel.ADMIN,
            'client_management': AdminPermissionLevel.MODERATOR,
            'revenue_management': AdminPermissionLevel.ADMIN
        }
    
    async def validate_permission(self, admin_id: str, required_permission: str, permission_level: AdminPermissionLevel) -> bool:
        """Validate if admin has required permission"""
        try:
            # Check permission level hierarchy
            required_level = self.permission_requirements.get(required_permission, AdminPermissionLevel.ADMIN)
            admin_level_value = self.permission_hierarchy[permission_level]
            required_level_value = self.permission_hierarchy[required_level]
            
            if admin_level_value >= required_level_value:
                return True
            
            # Check specific permissions list
            async with get_db_session() as session:
                result = await session.execute(
                    select(settings.ADMIN_USERS_TABLE.c.permissions).where(
                        settings.ADMIN_USERS_TABLE.c.admin_id == admin_id
                    )
                )
                admin_permissions = result.scalar()
                
                if admin_permissions and required_permission in admin_permissions:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to validate permission: {str(e)}")
            return False


class UniversalAdminController(ABC):
    """Base class for all admin control modules"""
    
    def __init__(self, admin_id: str, permission_level: AdminPermissionLevel):
        self.admin_id = admin_id
        self.permission_level = permission_level
        self.audit_logger = AdminAuditLogger()
        self.notification_manager = AdminNotificationManager()
        self.permission_validator = PermissionValidator()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def get_dashboard_data(self, timeframe: str = "month") -> Dict[str, Any]:
        """Get dashboard data for this admin module"""
        pass
    
    @abstractmethod
    async def process_admin_action(self, action: AdminAction) -> Dict[str, Any]:
        """Process an admin action with proper validation"""
        pass
    
    async def validate_admin_permission(self, required_permission: str) -> bool:
        """Validate admin has required permission"""
        return await self.permission_validator.validate_permission(
            admin_id=self.admin_id,
            required_permission=required_permission,
            permission_level=self.permission_level
        )
    
    async def log_admin_action(self, action: AdminAction, result: Dict[str, Any]):
        """Log admin action for audit trail"""
        await self.audit_logger.log_action(
            admin_id=self.admin_id,
            action=action,
            result=result,
            timestamp=datetime.now()
        )
    
    async def send_admin_notification(self, notification_type: str, data: Dict[str, Any]):
        """Send notification to relevant admins"""
        await self.notification_manager.send_notification(
            notification_type=notification_type,
            data=data,
            sender_admin_id=self.admin_id
        )
    
    async def create_approval_request(self, action_type: AdminActionType, data: Dict[str, Any], target_client_id: Optional[str] = None) -> str:
        """Create an approval request for admin action"""
        action = AdminAction(
            action_id=str(uuid.uuid4()),
            admin_id=self.admin_id,
            action_type=action_type,
            target_client_id=target_client_id,
            description=data.get('description', ''),
            data=data,
            status=ApprovalStatus.PENDING,
            requested_at=datetime.now(),
            impact_prediction=await self.predict_action_impact(action_type, data),
            risk_assessment=await self.assess_action_risk(action_type, data)
        )
        
        # Store approval request
        await self.store_approval_request(action)
        
        # Notify relevant admins
        await self.notify_approval_required(action)
        
        return action.action_id
    
    async def approve_action(self, action_id: str, reviewer_admin_id: str, review_notes: str = "") -> Dict[str, Any]:
        """Approve an admin action"""
        if not await self.validate_admin_permission("approve_actions"):
            raise AdminPermissionError("Insufficient permissions to approve actions")
        
        action = await self.get_approval_request(action_id)
        if not action:
            raise AdminActionError(f"Action {action_id} not found")
        
        if action.status != ApprovalStatus.PENDING:
            raise AdminActionError(f"Action {action_id} is not pending approval")
        
        # Update action status
        action.status = ApprovalStatus.APPROVED
        action.reviewed_at = datetime.now()
        action.reviewed_by = reviewer_admin_id
        
        # Store approval
        await self.store_approval_decision(action, review_notes)
        
        # Implement the action
        implementation_result = await self.implement_approved_action(action)
        
        # Log the approval and implementation
        await self.log_admin_action(action, implementation_result)
        
        return implementation_result
    
    async def reject_action(self, action_id: str, reviewer_admin_id: str, rejection_reason: str) -> Dict[str, Any]:
        """Reject an admin action"""
        if not await self.validate_admin_permission("approve_actions"):
            raise AdminPermissionError("Insufficient permissions to reject actions")
        
        action = await self.get_approval_request(action_id)
        if not action:
            raise AdminActionError(f"Action {action_id} not found")
        
        if action.status != ApprovalStatus.PENDING:
            raise AdminActionError(f"Action {action_id} is not pending approval")
        
        # Update action status
        action.status = ApprovalStatus.REJECTED
        action.reviewed_at = datetime.now()
        action.reviewed_by = reviewer_admin_id
        
        # Store rejection
        await self.store_approval_decision(action, rejection_reason)
        
        # Notify the requester
        await self.notify_action_rejected(action, rejection_reason)
        
        # Log the rejection
        await self.log_admin_action(action, {'status': 'rejected', 'reason': rejection_reason})
        
        return {
            'action_id': action_id,
            'status': 'rejected',
            'rejection_reason': rejection_reason,
            'rejected_by': reviewer_admin_id,
            'rejected_at': datetime.now()
        }
    
    async def predict_action_impact(self, action_type: AdminActionType, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict the impact of an admin action"""
        # Base implementation - override in subclasses for specific predictions
        return {
            'predicted_impact': 'medium',
            'confidence': 0.75,
            'affected_users': 'unknown',
            'revenue_impact': 'unknown',
            'risk_level': 'medium'
        }
    
    async def assess_action_risk(self, action_type: AdminActionType, data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the risk of an admin action"""
        # Base implementation - override in subclasses for specific assessments
        return {
            'risk_level': 'medium',
            'risk_factors': [],
            'mitigation_strategies': [],
            'rollback_plan_available': True
        }
    
    async def store_approval_request(self, action: AdminAction):
        """Store an approval request in the database"""
        try:
            async with get_db_session() as session:
                await session.execute(
                    insert(settings.ADMIN_ACTIONS_TABLE).values(
                        action_id=action.action_id,
                        admin_id=action.admin_id,
                        action_type=action.action_type.value,
                        target_client_id=action.target_client_id,
                        description=action.description,
                        data=encrypt_data(json.dumps(action.data)),
                        status=action.status.value,
                        requested_at=action.requested_at,
                        impact_prediction=json.dumps(action.impact_prediction),
                        risk_assessment=json.dumps(action.risk_assessment)
                    )
                )
                await session.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to store approval request: {str(e)}")
            raise
    
    async def get_approval_request(self, action_id: str) -> Optional[AdminAction]:
        """Retrieve an approval request from the database"""
        try:
            async with get_db_session() as session:
                result = await session.execute(
                    select(settings.ADMIN_ACTIONS_TABLE).where(
                        settings.ADMIN_ACTIONS_TABLE.c.action_id == action_id
                    )
                )
                row = result.first()
                
                if row:
                    return AdminAction(
                        action_id=row.action_id,
                        admin_id=row.admin_id,
                        action_type=AdminActionType(row.action_type),
                        target_client_id=row.target_client_id,
                        description=row.description,
                        data=json.loads(decrypt_data(row.data)),
                        status=ApprovalStatus(row.status),
                        requested_at=row.requested_at,
                        reviewed_at=row.reviewed_at,
                        reviewed_by=row.reviewed_by,
                        implemented_at=row.implemented_at,
                        impact_prediction=json.loads(row.impact_prediction) if row.impact_prediction else {},
                        risk_assessment=json.loads(row.risk_assessment) if row.risk_assessment else {}
                    )
                
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get approval request: {str(e)}")
            raise
    
    async def store_approval_decision(self, action: AdminAction, notes: str):
        """Store the approval/rejection decision"""
        try:
            async with get_db_session() as session:
                await session.execute(
                    update(settings.ADMIN_ACTIONS_TABLE).where(
                        settings.ADMIN_ACTIONS_TABLE.c.action_id == action.action_id
                    ).values(
                        status=action.status.value,
                        reviewed_at=action.reviewed_at,
                        reviewed_by=action.reviewed_by,
                        review_notes=encrypt_data(notes)
                    )
                )
                await session.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to store approval decision: {str(e)}")
            raise
    
    async def implement_approved_action(self, action: AdminAction) -> Dict[str, Any]:
        """Implement an approved action - override in subclasses"""
        # Base implementation
        return {
            'status': 'not_implemented',
            'message': 'Implementation must be defined in subclass'
        }
    
    async def notify_approval_required(self, action: AdminAction):
        """Notify admins that approval is required"""
        await self.send_admin_notification(
            notification_type='approval_required',
            data={
                'action_id': action.action_id,
                'action_type': action.action_type.value,
                'description': action.description,
                'requester': self.admin_id,
                'priority': 'high' if action.risk_assessment.get('risk_level') == 'high' else 'normal'
            }
        )
    
    async def notify_action_rejected(self, action: AdminAction, rejection_reason: str):
        """Notify requester that action was rejected"""
        await self.send_admin_notification(
            notification_type='action_rejected',
            data={
                'action_id': action.action_id,
                'action_type': action.action_type.value,
                'rejection_reason': rejection_reason,
                'rejected_by': action.reviewed_by
            }
        )
    
    async def get_admin_activity_log(self, timeframe: str) -> List[Dict[str, Any]]:
        """Get activity log for this admin"""
        try:
            start_date = self._calculate_start_date(timeframe)
            
            async with get_db_session() as session:
                result = await session.execute(
                    select(settings.ADMIN_AUDIT_TABLE).where(
                        settings.ADMIN_AUDIT_TABLE.c.admin_id == self.admin_id,
                        settings.ADMIN_AUDIT_TABLE.c.timestamp >= start_date
                    ).order_by(settings.ADMIN_AUDIT_TABLE.c.timestamp.desc())
                )
                
                activities = []
                for row in result:
                    activities.append({
                        'audit_id': row.audit_id,
                        'action_type': row.action_type,
                        'timestamp': row.timestamp,
                        'status': row.status,
                        'target_client_id': row.target_client_id
                    })
                
                return activities
                
        except Exception as e:
            self.logger.error(f"Failed to get admin activity log: {str(e)}")
            return []
    
    def _calculate_start_date(self, timeframe: str) -> datetime:
        """Calculate start date based on timeframe"""
        now = datetime.now()
        
        if timeframe == "day":
            return now - timedelta(days=1)
        elif timeframe == "week":
            return now - timedelta(weeks=1)
        elif timeframe == "month":
            return now - timedelta(days=30)
        elif timeframe == "quarter":
            return now - timedelta(days=90)
        elif timeframe == "year":
            return now - timedelta(days=365)
        else:
            return now - timedelta(days=30)  # Default to month