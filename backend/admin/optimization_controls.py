"""
Optimization Controls Module for AutoGuru Universal.

This module provides comprehensive platform optimization controls that adapt
to any business niche, allowing admins to tune performance, resource allocation,
and system behavior.
"""
import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

from sqlalchemy import and_, func, or_, select, update

from backend.admin.base_admin import (
    AdminAction,
    AdminActionType,
    AdminPermissionLevel,
    AdminPermissionError,
    AdminUser,
    ApprovalStatus,
    PermissionValidator,
    UniversalAdminController,
)
from backend.config.settings import settings
from backend.database.connection import get_db_context
from backend.services.analytics_service import AnalyticsService
from backend.utils.encryption import encrypt_data, decrypt_data


class OptimizationCategory(str, Enum):
    """Categories of optimization controls."""
    
    PERFORMANCE = "performance"
    RESOURCE_ALLOCATION = "resource_allocation"
    ALGORITHM_TUNING = "algorithm_tuning"
    CACHING = "caching"
    RATE_LIMITING = "rate_limiting"
    QUEUE_MANAGEMENT = "queue_management"
    AI_MODEL_CONFIG = "ai_model_config"
    PLATFORM_LIMITS = "platform_limits"
    COST_OPTIMIZATION = "cost_optimization"
    SCALING = "scaling"


class ResourceType(str, Enum):
    """Types of system resources."""
    
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    API_CALLS = "api_calls"
    AI_TOKENS = "ai_tokens"
    BANDWIDTH = "bandwidth"
    QUEUE_SIZE = "queue_size"
    CACHE_SIZE = "cache_size"


@dataclass
class OptimizationRule:
    """Represents an optimization rule or setting."""
    
    rule_id: UUID = field(default_factory=uuid4)
    category: OptimizationCategory = OptimizationCategory.PERFORMANCE
    name: str = ""
    description: str = ""
    current_value: Any = None
    target_value: Any = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    value_type: str = "float"  # float, int, bool, string, json
    unit: Optional[str] = None
    impact_level: str = "medium"  # low, medium, high, critical
    auto_adjust: bool = False
    adjustment_algorithm: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def validate_value(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate a proposed value against constraints."""
        # Type validation
        try:
            if self.value_type == "float":
                value = float(value)
            elif self.value_type == "int":
                value = int(value)
            elif self.value_type == "bool":
                value = bool(value)
            elif self.value_type == "json":
                if isinstance(value, str):
                    value = json.loads(value)
        except (ValueError, json.JSONDecodeError) as e:
            return False, f"Invalid value type: {str(e)}"
        
        # Range validation
        if self.min_value is not None and value < self.min_value:
            return False, f"Value below minimum: {self.min_value}"
        
        if self.max_value is not None and value > self.max_value:
            return False, f"Value above maximum: {self.max_value}"
        
        # Custom constraints
        for constraint_name, constraint_value in self.constraints.items():
            if constraint_name == "divisible_by" and value % constraint_value != 0:
                return False, f"Value must be divisible by {constraint_value}"
            elif constraint_name == "allowed_values" and value not in constraint_value:
                return False, f"Value must be one of: {constraint_value}"
            elif constraint_name == "regex" and not re.match(constraint_value, str(value)):
                return False, f"Value doesn't match pattern: {constraint_value}"
        
        return True, None


@dataclass
class PerformanceMetric:
    """Represents a performance metric."""
    
    metric_id: UUID = field(default_factory=uuid4)
    name: str = ""
    category: str = ""
    current_value: float = 0.0
    target_value: Optional[float] = None
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    unit: str = ""
    trend: str = "stable"  # improving, stable, degrading
    trend_percentage: float = 0.0
    measurement_interval: timedelta = timedelta(minutes=5)
    last_measurement: datetime = field(default_factory=datetime.utcnow)
    historical_data: List[Dict[str, Any]] = field(default_factory=list)


class SystemOptimizer:
    """Handles automatic system optimization based on metrics."""
    
    def __init__(self, analytics_service: AnalyticsService):
        self.analytics_service = analytics_service
        self.optimization_history: List[Dict[str, Any]] = []
    
    async def analyze_performance(
        self,
        metrics: List[PerformanceMetric],
        time_range: timedelta = timedelta(hours=24)
    ) -> Dict[str, Any]:
        """Analyze system performance and identify optimization opportunities."""
        analysis = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_health": "good",
            "score": 85.0,
            "bottlenecks": [],
            "recommendations": [],
            "resource_utilization": {},
            "trending_issues": [],
            "optimization_opportunities": []
        }
        
        # Analyze each metric
        critical_count = 0
        warning_count = 0
        
        for metric in metrics:
            # Check thresholds
            if metric.threshold_critical and metric.current_value >= metric.threshold_critical:
                critical_count += 1
                analysis["bottlenecks"].append({
                    "metric": metric.name,
                    "severity": "critical",
                    "current": metric.current_value,
                    "threshold": metric.threshold_critical,
                    "recommendation": await self._generate_optimization_recommendation(metric)
                })
            elif metric.threshold_warning and metric.current_value >= metric.threshold_warning:
                warning_count += 1
                analysis["trending_issues"].append({
                    "metric": metric.name,
                    "severity": "warning",
                    "current": metric.current_value,
                    "threshold": metric.threshold_warning,
                    "trend": metric.trend
                })
            
            # Analyze trends
            if metric.trend == "degrading" and abs(metric.trend_percentage) > 10:
                analysis["optimization_opportunities"].append({
                    "metric": metric.name,
                    "trend": f"Degrading by {abs(metric.trend_percentage):.1f}%",
                    "suggested_action": await self._suggest_optimization_action(metric)
                })
        
        # Calculate overall health
        if critical_count > 0:
            analysis["overall_health"] = "critical"
            analysis["score"] = max(0, 50 - (critical_count * 10))
        elif warning_count > 0:
            analysis["overall_health"] = "warning"
            analysis["score"] = max(50, 85 - (warning_count * 5))
        
        # Add resource utilization summary
        analysis["resource_utilization"] = await self._calculate_resource_utilization(metrics)
        
        # Generate AI-powered recommendations
        analysis["recommendations"] = await self._generate_ai_recommendations(analysis)
        
        return analysis
    
    async def _generate_optimization_recommendation(
        self,
        metric: PerformanceMetric
    ) -> str:
        """Generate specific optimization recommendation for a metric."""
        recommendations = {
            "cpu": "Consider scaling horizontally or optimizing CPU-intensive operations",
            "memory": "Implement memory caching strategies or increase available RAM",
            "api_calls": "Enable request batching or implement rate limiting",
            "ai_tokens": "Optimize prompt engineering or implement token caching",
            "response_time": "Enable caching, optimize database queries, or add CDN",
            "queue_size": "Increase worker processes or optimize processing logic"
        }
        
        # Get base recommendation
        base_rec = recommendations.get(
            metric.name.lower(),
            "Monitor closely and consider optimization"
        )
        
        # Enhance with specific context
        if metric.trend == "degrading":
            base_rec += f" (degrading by {abs(metric.trend_percentage):.1f}% recently)"
        
        return base_rec
    
    async def _suggest_optimization_action(
        self,
        metric: PerformanceMetric
    ) -> Dict[str, Any]:
        """Suggest specific optimization action for a metric."""
        return {
            "action": "auto_optimize",
            "parameters": {
                "metric": metric.name,
                "target_improvement": 20,  # percentage
                "strategy": "gradual",
                "monitor_period": "1h"
            },
            "expected_impact": "Medium to High",
            "risk_level": "Low"
        }
    
    async def _calculate_resource_utilization(
        self,
        metrics: List[PerformanceMetric]
    ) -> Dict[str, float]:
        """Calculate resource utilization percentages."""
        utilization = {}
        
        for metric in metrics:
            if metric.target_value:
                utilization[metric.name] = (metric.current_value / metric.target_value) * 100
            else:
                # Estimate based on thresholds
                if metric.threshold_critical:
                    utilization[metric.name] = (metric.current_value / metric.threshold_critical) * 100
                else:
                    utilization[metric.name] = 0.0
        
        return utilization
    
    async def _generate_ai_recommendations(
        self,
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate AI-powered optimization recommendations."""
        recommendations = []
        
        # High-priority recommendations based on bottlenecks
        for bottleneck in analysis["bottlenecks"]:
            recommendations.append({
                "priority": "high",
                "category": "performance",
                "title": f"Resolve {bottleneck['metric']} bottleneck",
                "description": bottleneck["recommendation"],
                "estimated_impact": "High",
                "implementation_time": "1-2 hours",
                "auto_implementable": True
            })
        
        # Medium-priority recommendations for trending issues
        for issue in analysis["trending_issues"]:
            recommendations.append({
                "priority": "medium",
                "category": "preventive",
                "title": f"Address {issue['metric']} trend",
                "description": f"Metric trending {issue['trend']}, consider preventive action",
                "estimated_impact": "Medium",
                "implementation_time": "2-4 hours",
                "auto_implementable": True
            })
        
        # Low-priority optimization opportunities
        for opportunity in analysis["optimization_opportunities"]:
            recommendations.append({
                "priority": "low",
                "category": "optimization",
                "title": f"Optimize {opportunity['metric']}",
                "description": opportunity["suggested_action"]["action"],
                "estimated_impact": "Low to Medium",
                "implementation_time": "4-8 hours",
                "auto_implementable": False
            })
        
        return recommendations[:10]  # Top 10 recommendations


class AutoScaler:
    """Handles automatic scaling decisions."""
    
    def __init__(self):
        self.scaling_policies: Dict[str, Dict[str, Any]] = {}
        self.scaling_history: List[Dict[str, Any]] = []
    
    async def evaluate_scaling_need(
        self,
        metrics: Dict[str, float],
        current_resources: Dict[str, int]
    ) -> Optional[Dict[str, Any]]:
        """Evaluate if scaling is needed based on metrics."""
        scaling_decision = None
        
        # Check CPU-based scaling
        if metrics.get("cpu_utilization", 0) > 80:
            scaling_decision = {
                "action": "scale_up",
                "resource": "compute_instances",
                "reason": "High CPU utilization",
                "recommended_increase": 2,
                "urgency": "high"
            }
        
        # Check memory-based scaling
        elif metrics.get("memory_utilization", 0) > 85:
            scaling_decision = {
                "action": "scale_up",
                "resource": "memory",
                "reason": "High memory utilization",
                "recommended_increase": "4GB",
                "urgency": "medium"
            }
        
        # Check request-based scaling
        elif metrics.get("request_rate", 0) > current_resources.get("max_rps", 1000) * 0.9:
            scaling_decision = {
                "action": "scale_out",
                "resource": "load_balancer",
                "reason": "Approaching request rate limit",
                "recommended_action": "Add load balancer node",
                "urgency": "high"
            }
        
        # Check for scale-down opportunities
        elif all(
            metrics.get(m, 100) < 30
            for m in ["cpu_utilization", "memory_utilization", "request_rate"]
        ):
            scaling_decision = {
                "action": "scale_down",
                "resource": "compute_instances",
                "reason": "Low resource utilization",
                "recommended_decrease": 1,
                "urgency": "low",
                "wait_period": "30m"
            }
        
        if scaling_decision:
            scaling_decision["timestamp"] = datetime.utcnow().isoformat()
            scaling_decision["current_metrics"] = metrics
            scaling_decision["current_resources"] = current_resources
            self.scaling_history.append(scaling_decision)
        
        return scaling_decision


class OptimizationControls(UniversalAdminController):
    """
    Admin control system for platform optimization.
    
    Provides comprehensive controls for performance tuning, resource allocation,
    and system optimization that adapt to any business niche.
    """
    
    def __init__(self, admin_id: str, permission_level: AdminPermissionLevel):
        super().__init__(admin_id, permission_level)
        self.optimizer = SystemOptimizer(AnalyticsService())
        self.auto_scaler = AutoScaler()
        self.optimization_rules: Dict[UUID, OptimizationRule] = {}
        self.performance_metrics: Dict[str, PerformanceMetric] = {}
        self.active_optimizations: Dict[UUID, Dict[str, Any]] = {}
    
    async def get_optimization_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive optimization dashboard data."""
        try:
            # Get current performance metrics
            metrics = await self._get_current_metrics()
            
            # Analyze performance
            performance_analysis = await self.optimizer.analyze_performance(
                list(self.performance_metrics.values())
            )
            
            # Get resource utilization
            resource_data = await self._get_resource_utilization()
            
            # Get active optimizations
            active_opts = await self._get_active_optimizations()
            
            # Get optimization history
            history = await self._get_optimization_history(days=7)
            
            # Get AI recommendations
            recommendations = performance_analysis.get("recommendations", [])
            
            dashboard = {
                "summary": {
                    "overall_health": performance_analysis["overall_health"],
                    "performance_score": performance_analysis["score"],
                    "active_optimizations": len(active_opts),
                    "pending_recommendations": len(recommendations),
                    "last_optimization": history[0]["timestamp"] if history else None
                },
                "metrics": metrics,
                "resource_utilization": resource_data,
                "bottlenecks": performance_analysis.get("bottlenecks", []),
                "trending_issues": performance_analysis.get("trending_issues", []),
                "active_optimizations": active_opts,
                "recommendations": recommendations,
                "optimization_categories": await self._get_category_status(),
                "auto_scaling_status": await self._get_auto_scaling_status(),
                "cost_analysis": await self._get_cost_optimization_analysis(),
                "recent_changes": history[:10]
            }
            
            return dashboard
            
        except Exception as e:
            self.logger.error(
                f"Error getting optimization dashboard: {str(e)}",
                extra={"admin_id": self.admin_id}
            )
            raise
    
    async def update_optimization_rule(
        self,
        rule_id: UUID,
        updates: Dict[str, Any],
        reason: str
    ) -> Dict[str, Any]:
        """Update an optimization rule."""
        # Validate permissions
        if not await self.validate_admin_permission("modify_optimization_settings"):
            raise AdminPermissionError("Insufficient permissions to modify optimization settings")
        
        try:
            rule = self.optimization_rules.get(rule_id)
            if not rule:
                raise ValueError(f"Optimization rule {rule_id} not found")
            
            # Validate new value if provided
            if "target_value" in updates:
                is_valid, error_msg = rule.validate_value(updates["target_value"])
                if not is_valid:
                    raise ValueError(f"Invalid value: {error_msg}")
            
            # Check if change requires approval
            if rule.impact_level in ["high", "critical"]:
                # Create approval request
                action = AdminAction(
                    action_type="optimization_rule_change",
                    target_id=str(rule_id),
                    changes=updates,
                    reason=reason,
                    impact_level=rule.impact_level,
                    created_by=self.admin_user.user_id
                )
                
                approval_id = await self._create_approval_request(
                    action,
                    required_approvers=2 if rule.impact_level == "critical" else 1
                )
                
                return {
                    "status": "pending_approval",
                    "approval_id": str(approval_id),
                    "rule_id": str(rule_id),
                    "message": f"High-impact change requires approval"
                }
            
            # Apply the change
            old_value = rule.target_value
            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            
            rule.updated_at = datetime.utcnow()
            
            # Log the action
            await self.logger.log_action(
                AdminAction(
                    action_type="optimization_rule_updated",
                    target_id=str(rule_id),
                    changes={
                        "rule": rule.name,
                        "old_value": old_value,
                        "new_value": rule.target_value,
                        "updates": updates
                    },
                    reason=reason,
                    created_by=self.admin_user.user_id,
                    status=ApprovalStatus.IMPLEMENTED
                )
            )
            
            # Apply optimization if auto-adjust is enabled
            if rule.auto_adjust:
                await self._apply_optimization(rule)
            
            return {
                "status": "success",
                "rule_id": str(rule_id),
                "applied_updates": updates,
                "message": "Optimization rule updated successfully"
            }
            
        except Exception as e:
            await self.logger.log_error(
                f"Error updating optimization rule: {str(e)}",
                {
                    "admin_id": str(self.admin_user.user_id),
                    "rule_id": str(rule_id),
                    "updates": updates
                }
            )
            raise
    
    async def create_optimization_experiment(
        self,
        name: str,
        category: OptimizationCategory,
        hypothesis: str,
        changes: List[Dict[str, Any]],
        success_metrics: List[str],
        duration_hours: int = 24
    ) -> Dict[str, Any]:
        """Create an A/B test for optimization changes."""
        self._validate_permission(AdminPermission.MODERATOR)
        
        try:
            experiment_id = uuid4()
            
            # Validate all changes
            for change in changes:
                rule_id = UUID(change["rule_id"])
                rule = self.optimization_rules.get(rule_id)
                if not rule:
                    raise ValueError(f"Rule {rule_id} not found")
                
                is_valid, error_msg = rule.validate_value(change["new_value"])
                if not is_valid:
                    raise ValueError(f"Invalid value for {rule.name}: {error_msg}")
            
            # Create experiment
            experiment = {
                "experiment_id": str(experiment_id),
                "name": name,
                "category": category.value,
                "hypothesis": hypothesis,
                "changes": changes,
                "success_metrics": success_metrics,
                "duration_hours": duration_hours,
                "status": "pending",
                "created_at": datetime.utcnow().isoformat(),
                "created_by": str(self.admin_user.user_id),
                "start_time": None,
                "end_time": None,
                "control_group": {
                    "size_percentage": 50,
                    "current_values": {}
                },
                "test_group": {
                    "size_percentage": 50,
                    "new_values": {}
                },
                "results": None
            }
            
            # Store current values
            for change in changes:
                rule_id = UUID(change["rule_id"])
                rule = self.optimization_rules.get(rule_id)
                experiment["control_group"]["current_values"][str(rule_id)] = rule.current_value
                experiment["test_group"]["new_values"][str(rule_id)] = change["new_value"]
            
            # Store experiment
            async with get_async_db_session() as session:
                await session.execute(
                    """
                    INSERT INTO optimization_experiments 
                    (id, data, created_by, created_at, status)
                    VALUES (:id, :data, :created_by, :created_at, :status)
                    """,
                    {
                        "id": experiment_id,
                        "data": encrypt_data(json.dumps(experiment)),
                        "created_by": self.admin_user.user_id,
                        "created_at": datetime.utcnow(),
                        "status": "pending"
                    }
                )
                await session.commit()
            
            # Log action
            await self.logger.log_action(
                AdminAction(
                    action_type="optimization_experiment_created",
                    target_id=str(experiment_id),
                    changes=experiment,
                    reason=f"Testing: {hypothesis}",
                    created_by=self.admin_user.user_id
                )
            )
            
            return experiment
            
        except Exception as e:
            await self.logger.log_error(
                f"Error creating optimization experiment: {str(e)}",
                {
                    "admin_id": str(self.admin_user.user_id),
                    "name": name,
                    "category": category.value
                }
            )
            raise
    
    async def apply_auto_scaling_decision(
        self,
        decision_id: UUID,
        override_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Apply an auto-scaling decision."""
        self._validate_permission(AdminPermission.ADMIN)
        
        try:
            # Get scaling decision
            metrics = await self._get_current_metrics()
            resources = await self._get_current_resources()
            
            scaling_decision = await self.auto_scaler.evaluate_scaling_need(
                metrics,
                resources
            )
            
            if not scaling_decision:
                return {
                    "status": "no_action_needed",
                    "message": "No scaling required at this time"
                }
            
            # Apply overrides if provided
            if override_params:
                scaling_decision.update(override_params)
            
            # Check if approval needed for high-urgency scaling
            if scaling_decision.get("urgency") == "high":
                action = AdminAction(
                    action_type="auto_scaling",
                    target_id=str(decision_id),
                    changes=scaling_decision,
                    reason="Automatic scaling based on metrics",
                    impact_level="high",
                    created_by=self.admin_user.user_id
                )
                
                approval_id = await self._create_approval_request(action)
                
                return {
                    "status": "pending_approval",
                    "approval_id": str(approval_id),
                    "decision": scaling_decision,
                    "message": "High-urgency scaling requires approval"
                }
            
            # Apply scaling
            result = await self._execute_scaling(scaling_decision)
            
            # Log action
            await self.logger.log_action(
                AdminAction(
                    action_type="auto_scaling_applied",
                    target_id=str(decision_id),
                    changes={
                        "decision": scaling_decision,
                        "result": result
                    },
                    reason="Automatic scaling based on metrics",
                    created_by=self.admin_user.user_id,
                    status=ApprovalStatus.IMPLEMENTED
                )
            )
            
            return {
                "status": "success",
                "decision_id": str(decision_id),
                "scaling_result": result,
                "message": "Scaling applied successfully"
            }
            
        except Exception as e:
            await self.logger.log_error(
                f"Error applying auto-scaling: {str(e)}",
                {
                    "admin_id": str(self.admin_user.user_id),
                    "decision_id": str(decision_id)
                }
            )
            raise
    
    async def configure_cost_optimization(
        self,
        settings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Configure cost optimization settings."""
        self._validate_permission(AdminPermission.ADMIN)
        
        try:
            # Validate settings
            valid_settings = {
                "enable_spot_instances": bool,
                "reserved_instance_percentage": float,
                "idle_resource_timeout_minutes": int,
                "auto_shutdown_dev_resources": bool,
                "compress_cold_storage": bool,
                "optimize_ai_token_usage": bool,
                "batch_api_calls": bool,
                "cache_ttl_seconds": int
            }
            
            validated_settings = {}
            for key, value in settings.items():
                if key in valid_settings:
                    try:
                        validated_settings[key] = valid_settings[key](value)
                    except ValueError:
                        raise ValueError(f"Invalid value for {key}")
            
            # Apply cost optimization settings
            cost_rules = []
            
            for key, value in validated_settings.items():
                rule = OptimizationRule(
                    category=OptimizationCategory.COST_OPTIMIZATION,
                    name=key,
                    description=f"Cost optimization: {key}",
                    current_value=value,
                    target_value=value,
                    value_type=valid_settings[key].__name__,
                    impact_level="medium",
                    auto_adjust=True
                )
                
                self.optimization_rules[rule.rule_id] = rule
                cost_rules.append(rule)
            
            # Calculate estimated savings
            estimated_savings = await self._calculate_cost_savings(validated_settings)
            
            # Log action
            await self.logger.log_action(
                AdminAction(
                    action_type="cost_optimization_configured",
                    target_id="cost_settings",
                    changes=validated_settings,
                    reason="Configure cost optimization",
                    created_by=self.admin_user.user_id,
                    metadata={"estimated_savings": estimated_savings}
                )
            )
            
            return {
                "status": "success",
                "applied_settings": validated_settings,
                "estimated_monthly_savings": estimated_savings,
                "rules_created": len(cost_rules),
                "message": "Cost optimization configured successfully"
            }
            
        except Exception as e:
            await self.logger.log_error(
                f"Error configuring cost optimization: {str(e)}",
                {
                    "admin_id": str(self.admin_user.user_id),
                    "settings": settings
                }
            )
            raise
    
    async def _get_current_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        # This would connect to monitoring systems
        return {
            "cpu_utilization": 65.5,
            "memory_utilization": 72.3,
            "request_rate": 850.0,
            "response_time_ms": 125.5,
            "error_rate": 0.02,
            "queue_depth": 1250,
            "cache_hit_rate": 0.85,
            "ai_token_usage": 15000
        }
    
    async def _get_resource_utilization(self) -> Dict[str, Any]:
        """Get detailed resource utilization."""
        return {
            "compute": {
                "cpu": {"current": 65.5, "allocated": 80, "unit": "%"},
                "memory": {"current": 14.5, "allocated": 20, "unit": "GB"},
                "instances": {"current": 8, "allocated": 10, "unit": "count"}
            },
            "storage": {
                "database": {"current": 450, "allocated": 1000, "unit": "GB"},
                "cache": {"current": 12, "allocated": 16, "unit": "GB"},
                "files": {"current": 2.1, "allocated": 5, "unit": "TB"}
            },
            "network": {
                "bandwidth": {"current": 850, "allocated": 2000, "unit": "Mbps"},
                "connections": {"current": 15000, "allocated": 50000, "unit": "count"}
            },
            "api": {
                "calls": {"current": 850000, "allocated": 1000000, "unit": "per day"},
                "tokens": {"current": 15000000, "allocated": 20000000, "unit": "per day"}
            }
        }
    
    async def _get_active_optimizations(self) -> List[Dict[str, Any]]:
        """Get list of active optimizations."""
        active = []
        
        for opt_id, optimization in self.active_optimizations.items():
            if optimization.get("status") == "active":
                active.append({
                    "optimization_id": str(opt_id),
                    "name": optimization["name"],
                    "category": optimization["category"],
                    "started_at": optimization["started_at"],
                    "impact": optimization.get("measured_impact", "Measuring..."),
                    "status": optimization["status"]
                })
        
        return active
    
    async def _get_optimization_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get optimization history."""
        # This would query the database
        return [
            {
                "timestamp": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                "type": "auto_scaling",
                "action": "Scaled up compute instances",
                "result": "Success",
                "impact": "+15% capacity"
            },
            {
                "timestamp": (datetime.utcnow() - timedelta(days=1)).isoformat(),
                "type": "cache_optimization",
                "action": "Increased cache TTL",
                "result": "Success",
                "impact": "+10% hit rate"
            }
        ]
    
    async def _get_category_status(self) -> Dict[str, Any]:
        """Get optimization status by category."""
        categories = {}
        
        for category in OptimizationCategory:
            rules_in_category = [
                r for r in self.optimization_rules.values()
                if r.category == category
            ]
            
            categories[category.value] = {
                "total_rules": len(rules_in_category),
                "active_rules": sum(1 for r in rules_in_category if r.auto_adjust),
                "last_optimized": datetime.utcnow().isoformat(),
                "health_score": 85.0  # Would be calculated
            }
        
        return categories
    
    async def _get_auto_scaling_status(self) -> Dict[str, Any]:
        """Get auto-scaling status."""
        return {
            "enabled": True,
            "mode": "reactive",  # reactive, predictive, scheduled
            "last_scaling_event": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
            "current_policy": {
                "scale_up_threshold": 80,
                "scale_down_threshold": 30,
                "cooldown_minutes": 10
            },
            "pending_decisions": 0,
            "scaling_history": self.auto_scaler.scaling_history[-5:]
        }
    
    async def _get_cost_optimization_analysis(self) -> Dict[str, Any]:
        """Get cost optimization analysis."""
        return {
            "current_monthly_cost": 25000.00,
            "projected_monthly_cost": 23500.00,
            "potential_savings": 1500.00,
            "savings_percentage": 6.0,
            "top_cost_drivers": [
                {"resource": "AI API Calls", "cost": 8000, "percentage": 32},
                {"resource": "Compute Instances", "cost": 6000, "percentage": 24},
                {"resource": "Storage", "cost": 4000, "percentage": 16}
            ],
            "optimization_opportunities": [
                {
                    "opportunity": "Use spot instances for batch jobs",
                    "potential_savings": 500,
                    "implementation": "easy"
                },
                {
                    "opportunity": "Compress cold storage data",
                    "potential_savings": 300,
                    "implementation": "medium"
                }
            ]
        }
    
    async def _apply_optimization(self, rule: OptimizationRule) -> None:
        """Apply an optimization rule."""
        # This would implement the actual optimization
        optimization_id = uuid4()
        
        self.active_optimizations[optimization_id] = {
            "name": rule.name,
            "category": rule.category.value,
            "started_at": datetime.utcnow().isoformat(),
            "status": "active",
            "rule_id": str(rule.rule_id)
        }
    
    async def _execute_scaling(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a scaling decision."""
        # This would integrate with cloud provider APIs
        return {
            "execution_time": datetime.utcnow().isoformat(),
            "action": decision["action"],
            "resource": decision["resource"],
            "result": "success",
            "new_capacity": "Scaled as requested"
        }
    
    async def _calculate_cost_savings(self, settings: Dict[str, Any]) -> float:
        """Calculate estimated cost savings from settings."""
        savings = 0.0
        
        # Estimate savings based on settings
        if settings.get("enable_spot_instances"):
            savings += 500  # $500/month estimated
        
        if settings.get("compress_cold_storage"):
            savings += 300  # $300/month estimated
        
        if settings.get("optimize_ai_token_usage"):
            savings += 800  # $800/month estimated
        
        return savings
    
    async def _get_current_resources(self) -> Dict[str, int]:
        """Get current resource allocation."""
        return {
            "compute_instances": 8,
            "memory_gb": 64,
            "storage_tb": 2,
            "max_rps": 1000,
            "cache_nodes": 3
        }