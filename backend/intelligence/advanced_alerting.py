"""Advanced Alerting System for Business Intelligence"""

from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import logging
from collections import defaultdict, deque
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import httpx

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertChannel(Enum):
    EMAIL = "email"
    WEBHOOK = "webhook"
    SMS = "sms"
    SLACK = "slack"
    IN_APP = "in_app"

class AlertPattern(Enum):
    SPIKE = "spike"
    DROP = "drop"
    THRESHOLD = "threshold"
    ANOMALY = "anomaly"
    TREND = "trend"
    RECURRING = "recurring"

@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    description: str
    metric_type: str
    condition: Dict[str, Any]
    severity: AlertSeverity
    channels: List[AlertChannel]
    cooldown_minutes: int = 60
    enabled: bool = True

@dataclass
class Alert:
    """Individual alert instance"""
    alert_id: str
    rule_id: str
    severity: AlertSeverity
    title: str
    message: str
    metric_data: Dict[str, Any]
    pattern: AlertPattern
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False

@dataclass
class AlertRecipient:
    """Alert recipient configuration"""
    recipient_id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    webhook_url: Optional[str] = None
    slack_webhook: Optional[str] = None
    channels: List[AlertChannel] = field(default_factory=list)
    severity_filter: Optional[List[AlertSeverity]] = None

class IntelligentAlertManager:
    """Smart alert routing and management"""
    
    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.recipients: Dict[str, AlertRecipient] = {}
        self.alert_cooldowns: Dict[str, datetime] = {}
        self.pattern_detector = AlertPatternDetector()
        self.escalation_manager = EscalationManager()
        
    async def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule"""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    async def process_metric(self, metric_type: str, value: float, metadata: Dict[str, Any]):
        """Process a metric and check for alerts"""
        # Check all rules for this metric type
        for rule in self.alert_rules.values():
            if not rule.enabled or rule.metric_type != metric_type:
                continue
            
            # Check if in cooldown
            if self._is_in_cooldown(rule.rule_id):
                continue
            
            # Evaluate condition
            if await self._evaluate_condition(rule, value, metadata):
                # Detect pattern
                pattern = await self.pattern_detector.detect_pattern(
                    metric_type, value, metadata
                )
                
                # Create alert
                alert = await self._create_alert(rule, value, metadata, pattern)
                
                # Route alert
                await self.route_alert(alert)
    
    def _is_in_cooldown(self, rule_id: str) -> bool:
        """Check if rule is in cooldown period"""
        if rule_id not in self.alert_cooldowns:
            return False
        
        cooldown_end = self.alert_cooldowns[rule_id]
        return datetime.now() < cooldown_end
    
    async def _evaluate_condition(self, rule: AlertRule, value: float, metadata: Dict[str, Any]) -> bool:
        """Evaluate if alert condition is met"""
        condition = rule.condition
        condition_type = condition.get("type")
        
        if condition_type == "threshold":
            operator = condition.get("operator", ">")
            threshold = condition.get("value", 0)
            
            if operator == ">":
                return value > threshold
            elif operator == "<":
                return value < threshold
            elif operator == ">=":
                return value >= threshold
            elif operator == "<=":
                return value <= threshold
            elif operator == "==":
                return value == threshold
        
        elif condition_type == "change":
            # Check percentage change
            previous_value = metadata.get("previous_value", value)
            if previous_value == 0:
                return False
            
            change_percent = abs((value - previous_value) / previous_value * 100)
            return change_percent > condition.get("threshold", 10)
        
        elif condition_type == "pattern":
            # Check for specific patterns
            pattern_type = condition.get("pattern")
            if pattern_type:
                return await self.pattern_detector.has_pattern(
                    rule.metric_type, pattern_type, metadata
                )
            return False
        
        return False
    
    async def _create_alert(self, rule: AlertRule, value: float, 
                          metadata: Dict[str, Any], pattern: AlertPattern) -> Alert:
        """Create an alert from a triggered rule"""
        alert_id = f"alert_{datetime.now().timestamp()}"
        
        # Generate alert message
        message = self._generate_alert_message(rule, value, metadata, pattern)
        
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            severity=rule.severity,
            title=f"{rule.name} Alert",
            message=message,
            metric_data={
                "metric_type": rule.metric_type,
                "value": value,
                "metadata": metadata
            },
            pattern=pattern
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Set cooldown
        cooldown_rule = self.alert_rules[rule.rule_id]
        self.alert_cooldowns[rule.rule_id] = datetime.now() + timedelta(
            minutes=cooldown_rule.cooldown_minutes
        )
        
        return alert
    
    def _generate_alert_message(self, rule: AlertRule, value: float, 
                              metadata: Dict[str, Any], pattern: AlertPattern) -> str:
        """Generate human-readable alert message"""
        messages = {
            AlertPattern.SPIKE: f"{rule.metric_type} spiked to {value:.2f}",
            AlertPattern.DROP: f"{rule.metric_type} dropped to {value:.2f}",
            AlertPattern.THRESHOLD: f"{rule.metric_type} crossed threshold at {value:.2f}",
            AlertPattern.ANOMALY: f"Anomaly detected in {rule.metric_type}: {value:.2f}",
            AlertPattern.TREND: f"Concerning trend in {rule.metric_type}: {value:.2f}",
            AlertPattern.RECURRING: f"Recurring issue with {rule.metric_type}: {value:.2f}"
        }
        
        base_message = messages.get(pattern, f"{rule.metric_type} alert: {value:.2f}")
        
        # Add context
        if "platform" in metadata:
            base_message += f" on {metadata['platform']}"
        
        if "change_percent" in metadata:
            base_message += f" ({metadata['change_percent']:.1f}% change)"
        
        return base_message
    
    async def route_alert(self, alert: Alert):
        """Route alert to appropriate channels"""
        rule = self.alert_rules.get(alert.rule_id)
        if not rule:
            return
        
        # Determine if escalation is needed
        if alert.pattern == AlertPattern.RECURRING:
            await self.escalation_manager.escalate_to_engineering(alert)
        
        # Send to configured channels
        for channel in rule.channels:
            await self._send_to_channel(alert, channel)
    
    async def _send_to_channel(self, alert: Alert, channel: AlertChannel):
        """Send alert to specific channel"""
        try:
            if channel == AlertChannel.EMAIL:
                await self._send_email_alert(alert)
            elif channel == AlertChannel.WEBHOOK:
                await self._send_webhook_alert(alert)
            elif channel == AlertChannel.SLACK:
                await self._send_slack_alert(alert)
            elif channel == AlertChannel.IN_APP:
                await self._send_in_app_alert(alert)
            elif channel == AlertChannel.SMS:
                await self._send_sms_alert(alert)
        except Exception as e:
            logger.error(f"Failed to send alert to {channel.value}: {e}")
    
    async def _send_email_alert(self, alert: Alert):
        """Send email alert"""
        # Get email recipients
        recipients = [
            r for r in self.recipients.values()
            if AlertChannel.EMAIL in r.channels and r.email
        ]
        
        if not recipients:
            return
        
        # In production, use proper email service
        logger.info(f"Sending email alert {alert.alert_id} to {len(recipients)} recipients")
    
    async def _send_webhook_alert(self, alert: Alert):
        """Send webhook alert"""
        webhook_recipients = [
            r for r in self.recipients.values()
            if AlertChannel.WEBHOOK in r.channels and r.webhook_url
        ]
        
        async with httpx.AsyncClient() as client:
            for recipient in webhook_recipients:
                try:
                    await client.post(
                        recipient.webhook_url,
                        json={
                            "alert_id": alert.alert_id,
                            "severity": alert.severity.value,
                            "title": alert.title,
                            "message": alert.message,
                            "timestamp": alert.timestamp.isoformat(),
                            "metric_data": alert.metric_data
                        },
                        timeout=10.0
                    )
                except Exception as e:
                    logger.error(f"Webhook failed for {recipient.webhook_url}: {e}")
    
    async def _send_slack_alert(self, alert: Alert):
        """Send Slack alert"""
        slack_recipients = [
            r for r in self.recipients.values()
            if AlertChannel.SLACK in r.channels and r.slack_webhook
        ]
        
        # Format for Slack
        slack_message = {
            "text": alert.title,
            "attachments": [{
                "color": self._get_severity_color(alert.severity),
                "fields": [
                    {"title": "Severity", "value": alert.severity.value, "short": True},
                    {"title": "Pattern", "value": alert.pattern.value, "short": True},
                    {"title": "Message", "value": alert.message, "short": False}
                ],
                "timestamp": int(alert.timestamp.timestamp())
            }]
        }
        
        async with httpx.AsyncClient() as client:
            for recipient in slack_recipients:
                try:
                    await client.post(recipient.slack_webhook, json=slack_message)
                except Exception as e:
                    logger.error(f"Slack webhook failed: {e}")
    
    async def _send_in_app_alert(self, alert: Alert):
        """Send in-app notification"""
        # This would integrate with your notification system
        logger.info(f"In-app alert created: {alert.alert_id}")
    
    async def _send_sms_alert(self, alert: Alert):
        """Send SMS alert (for critical alerts only)"""
        if alert.severity != AlertSeverity.CRITICAL:
            return
        
        sms_recipients = [
            r for r in self.recipients.values()
            if AlertChannel.SMS in r.channels and r.phone
        ]
        
        # In production, use SMS service like Twilio
        logger.info(f"SMS alert {alert.alert_id} to {len(sms_recipients)} recipients")
    
    def _get_severity_color(self, severity: AlertSeverity) -> str:
        """Get color for severity level"""
        colors = {
            AlertSeverity.LOW: "#36a64f",      # Green
            AlertSeverity.MEDIUM: "#ff9900",   # Orange
            AlertSeverity.HIGH: "#ff0000",     # Red
            AlertSeverity.CRITICAL: "#8b0000"  # Dark Red
        }
        return colors.get(severity, "#808080")
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
    
    async def resolve_alert(self, alert_id: str, resolved_by: str, notes: Optional[str] = None):
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            
            # Move to history
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert {alert_id} resolved by {resolved_by}")

class AlertPatternDetector:
    """Detect patterns in metric behavior"""
    
    def __init__(self):
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
    async def detect_pattern(self, metric_type: str, value: float, 
                           metadata: Dict[str, Any]) -> AlertPattern:
        """Detect the pattern type for this metric behavior"""
        history = self.metric_history[metric_type]
        history.append((value, datetime.now()))
        
        if len(history) < 3:
            return AlertPattern.THRESHOLD
        
        # Check for spike/drop
        recent_values = [v for v, _ in list(history)[-10:]]
        if recent_values:
            avg = sum(recent_values[:-1]) / len(recent_values[:-1])
            current = recent_values[-1]
            
            change_percent = abs((current - avg) / avg * 100) if avg != 0 else 0
            
            if change_percent > 50:
                return AlertPattern.SPIKE if current > avg else AlertPattern.DROP
        
        # Check for recurring pattern
        if self._is_recurring_pattern(history):
            return AlertPattern.RECURRING
        
        # Check for trend
        if self._has_concerning_trend(recent_values):
            return AlertPattern.TREND
        
        # Check for anomaly
        if metadata.get("is_anomaly", False):
            return AlertPattern.ANOMALY
        
        return AlertPattern.THRESHOLD
    
    def _is_recurring_pattern(self, history: deque) -> bool:
        """Check if the pattern is recurring"""
        if len(history) < 20:
            return False
        
        # Simple check: if similar spikes happen regularly
        values = [v for v, _ in history]
        threshold = sum(values) / len(values) * 1.5
        
        spike_times = [t for v, t in history if v > threshold]
        
        if len(spike_times) < 3:
            return False
        
        # Check if spikes are regular
        intervals = []
        for i in range(1, len(spike_times)):
            interval = (spike_times[i] - spike_times[i-1]).total_seconds()
            intervals.append(interval)
        
        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            # If intervals are consistent (within 20% of average)
            consistent = all(abs(i - avg_interval) / avg_interval < 0.2 for i in intervals)
            return consistent
        
        return False
    
    def _has_concerning_trend(self, values: List[float]) -> bool:
        """Check if there's a concerning trend"""
        if len(values) < 5:
            return False
        
        # Simple linear trend check
        x = list(range(len(values)))
        y = values
        
        # Calculate slope
        n = len(x)
        if n == 0:
            return False
            
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return False
        
        slope = numerator / denominator
        
        # Concerning if strong positive or negative trend
        return abs(slope) > y_mean * 0.1  # 10% of mean per time unit
    
    async def has_pattern(self, metric_type: str, pattern_type: str, 
                         metadata: Dict[str, Any]) -> bool:
        """Check if a specific pattern exists"""
        history = self.metric_history.get(metric_type, deque())
        
        if pattern_type == "recurring":
            return self._is_recurring_pattern(history)
        elif pattern_type == "trend":
            values = [v for v, _ in history]
            return self._has_concerning_trend(values)
        
        return False

class EscalationManager:
    """Manage alert escalation"""
    
    def __init__(self):
        self.escalation_queue = asyncio.Queue()
        self.escalation_history = deque(maxlen=100)
        
    async def escalate_to_engineering(self, alert: Alert):
        """Escalate alert to engineering team"""
        escalation = {
            "alert": alert,
            "escalated_at": datetime.now(),
            "reason": "Recurring pattern detected",
            "priority": "high" if alert.severity == AlertSeverity.CRITICAL else "medium"
        }
        
        await self.escalation_queue.put(escalation)
        self.escalation_history.append(escalation)
        
        logger.warning(f"Alert {alert.alert_id} escalated to engineering")
        
        # In production, this would create a ticket, page on-call, etc.
        await self._create_engineering_ticket(alert)
    
    async def _create_engineering_ticket(self, alert: Alert):
        """Create engineering ticket for escalated alert"""
        # This would integrate with your ticketing system (Jira, PagerDuty, etc.)
        ticket_data = {
            "title": f"[ESCALATED] {alert.title}",
            "description": f"""
                Alert ID: {alert.alert_id}
                Severity: {alert.severity.value}
                Pattern: {alert.pattern.value}
                Message: {alert.message}
                
                Metric Data: {json.dumps(alert.metric_data, indent=2)}
                
                This alert has been automatically escalated due to recurring pattern.
            """,
            "priority": "high" if alert.severity == AlertSeverity.CRITICAL else "medium",
            "labels": ["autoguru", "alert", "escalated", alert.pattern.value]
        }
        
        logger.info(f"Engineering ticket created for alert {alert.alert_id}")