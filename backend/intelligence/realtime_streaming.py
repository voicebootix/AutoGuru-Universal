"""Real-time Streaming Support for Business Intelligence Dashboard"""

from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as redis
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class MetricType(Enum):
    REVENUE = "revenue"
    ENGAGEMENT = "engagement"
    PERFORMANCE = "performance"
    CONVERSION = "conversion"
    SYSTEM_HEALTH = "system_health"

@dataclass
class MetricUpdate:
    """Real-time metric update"""
    metric_type: MetricType
    metric_name: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class DashboardSubscription:
    """Dashboard subscription configuration"""
    client_id: str
    subscription_id: str
    metric_types: List[MetricType]
    update_frequency_seconds: int = 30
    active: bool = True

class RealTimeMetricsStreamer:
    """Stream real-time metrics to dashboards"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.pubsub = None
        self.subscriptions: Dict[str, DashboardSubscription] = {}
        self.metric_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.streaming_tasks: Dict[str, asyncio.Task] = {}
        
    async def initialize(self):
        """Initialize Redis connection"""
        self.redis_client = await redis.from_url(self.redis_url)
        self.pubsub = self.redis_client.pubsub()
        logger.info("Real-time metrics streamer initialized")
    
    async def close(self):
        """Close connections"""
        if self.pubsub:
            await self.pubsub.close()
        if self.redis_client:
            await self.redis_client.close()
    
    async def subscribe_dashboard(
        self,
        client_id: str,
        metric_types: List[MetricType],
        update_frequency: int = 30
    ) -> str:
        """Subscribe a dashboard to real-time updates"""
        subscription_id = f"sub_{client_id}_{datetime.now().timestamp()}"
        
        subscription = DashboardSubscription(
            client_id=client_id,
            subscription_id=subscription_id,
            metric_types=metric_types,
            update_frequency_seconds=update_frequency
        )
        
        self.subscriptions[subscription_id] = subscription
        
        # Start streaming task
        task = asyncio.create_task(
            self._stream_metrics_to_dashboard(subscription)
        )
        self.streaming_tasks[subscription_id] = task
        
        logger.info(f"Dashboard subscribed: {subscription_id}")
        
        return subscription_id
    
    async def unsubscribe_dashboard(self, subscription_id: str):
        """Unsubscribe a dashboard from updates"""
        if subscription_id in self.subscriptions:
            self.subscriptions[subscription_id].active = False
            
            # Cancel streaming task
            if subscription_id in self.streaming_tasks:
                self.streaming_tasks[subscription_id].cancel()
                del self.streaming_tasks[subscription_id]
            
            del self.subscriptions[subscription_id]
            
            logger.info(f"Dashboard unsubscribed: {subscription_id}")
    
    async def publish_metric_update(self, update: MetricUpdate):
        """Publish a metric update"""
        # Buffer the update
        key = f"{update.metric_type.value}:{update.metric_name}"
        self.metric_buffers[key].append(update)
        
        # Publish to Redis channel
        channel = f"metrics:{update.metric_type.value}"
        message = json.dumps({
            "metric_name": update.metric_name,
            "value": update.value,
            "timestamp": update.timestamp.isoformat(),
            "metadata": update.metadata
        })
        
        await self.redis_client.publish(channel, message)
    
    async def _stream_metrics_to_dashboard(self, subscription: DashboardSubscription):
        """Stream metrics to a specific dashboard subscription"""
        while subscription.active:
            try:
                # Collect latest metrics for subscribed types
                metrics_data = await self._collect_latest_metrics(
                    subscription.client_id,
                    subscription.metric_types
                )
                
                # Send update via Redis pub/sub
                channel = f"dashboard:{subscription.subscription_id}"
                await self.redis_client.publish(
                    channel,
                    json.dumps({
                        "subscription_id": subscription.subscription_id,
                        "timestamp": datetime.now().isoformat(),
                        "metrics": metrics_data
                    })
                )
                
                # Wait for next update
                await asyncio.sleep(subscription.update_frequency_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error streaming metrics: {e}")
                await asyncio.sleep(5)  # Brief pause on error
    
    async def _collect_latest_metrics(
        self,
        client_id: str,
        metric_types: List[MetricType]
    ) -> Dict[str, Any]:
        """Collect latest metrics for specified types"""
        metrics_data = {}
        
        for metric_type in metric_types:
            if metric_type == MetricType.REVENUE:
                metrics_data["revenue"] = await self._get_revenue_metrics(client_id)
            elif metric_type == MetricType.ENGAGEMENT:
                metrics_data["engagement"] = await self._get_engagement_metrics(client_id)
            elif metric_type == MetricType.PERFORMANCE:
                metrics_data["performance"] = await self._get_performance_metrics(client_id)
            elif metric_type == MetricType.CONVERSION:
                metrics_data["conversion"] = await self._get_conversion_metrics(client_id)
            elif metric_type == MetricType.SYSTEM_HEALTH:
                metrics_data["system_health"] = await self._get_system_health_metrics()
        
        return metrics_data
    
    async def _get_revenue_metrics(self, client_id: str) -> Dict[str, Any]:
        """Get latest revenue metrics"""
        # Get from buffer or calculate
        revenue_updates = [
            update for key, updates in self.metric_buffers.items()
            if key.startswith("revenue:") 
            for update in updates
            if update.timestamp > datetime.now() - timedelta(minutes=5)
        ]
        
        if revenue_updates:
            latest_revenue = revenue_updates[-1].value
            revenue_trend = self._calculate_trend([u.value for u in revenue_updates])
        else:
            # Fallback to calculated values
            latest_revenue = 5000.0  # Would query from database
            revenue_trend = 0.05
        
        return {
            "current_revenue": latest_revenue,
            "revenue_trend": revenue_trend,
            "revenue_per_minute": latest_revenue / 1440,  # Daily revenue / minutes
            "top_revenue_source": "instagram",  # Would be calculated
            "revenue_forecast_today": latest_revenue * 1.05
        }
    
    async def _get_engagement_metrics(self, client_id: str) -> Dict[str, Any]:
        """Get latest engagement metrics"""
        engagement_updates = [
            update for key, updates in self.metric_buffers.items()
            if key.startswith("engagement:")
            for update in updates
            if update.timestamp > datetime.now() - timedelta(minutes=5)
        ]
        
        total_engagement = sum(u.value for u in engagement_updates) if engagement_updates else 1000
        
        return {
            "total_engagements": total_engagement,
            "engagement_rate": 5.2,  # Would be calculated
            "trending_content": "video",
            "peak_engagement_hour": 14,
            "engagement_velocity": len(engagement_updates) / 5  # Per minute
        }
    
    async def _get_performance_metrics(self, client_id: str) -> Dict[str, Any]:
        """Get latest performance metrics"""
        return {
            "api_response_time_ms": 125,
            "system_cpu_usage": 45.2,
            "memory_usage_percent": 62.1,
            "active_requests": 42,
            "error_rate": 0.23,
            "uptime_percentage": 99.9
        }
    
    async def _get_conversion_metrics(self, client_id: str) -> Dict[str, Any]:
        """Get latest conversion metrics"""
        return {
            "conversion_rate": 3.5,
            "conversions_today": 145,
            "average_order_value": 85.50,
            "cart_abandonment_rate": 25.5,
            "conversion_trend": "increasing"
        }
    
    async def _get_system_health_metrics(self) -> Dict[str, Any]:
        """Get system health metrics"""
        return {
            "overall_health": "healthy",
            "health_score": 92,
            "active_alerts": 2,
            "critical_issues": 0,
            "last_incident": "2024-01-10T14:30:00Z"
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend from a series of values"""
        if len(values) < 2:
            return 0.0
        
        # Simple trend calculation
        first_half = sum(values[:len(values)//2]) / (len(values)//2)
        second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
        
        if first_half == 0:
            return 0.0
        
        return (second_half - first_half) / first_half

class WebSocketMetricsHandler:
    """Handle WebSocket connections for real-time metrics"""
    
    def __init__(self, streamer: RealTimeMetricsStreamer):
        self.streamer = streamer
        self.active_connections: Set[Any] = set()  # WebSocket connections
        
    async def connect(self, websocket):
        """Handle new WebSocket connection"""
        self.active_connections.add(websocket)
        await websocket.accept()
        
        # Send initial data
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "timestamp": datetime.now().isoformat()
        })
    
    async def disconnect(self, websocket):
        """Handle WebSocket disconnection"""
        self.active_connections.remove(websocket)
    
    async def handle_message(self, websocket, message: Dict[str, Any]):
        """Handle incoming WebSocket message"""
        message_type = message.get("type")
        
        if message_type == "subscribe":
            # Subscribe to metrics
            client_id = message.get("client_id")
            if not client_id:
                await websocket.send_json({
                    "type": "error",
                    "message": "client_id is required"
                })
                return
                
            metric_types = [MetricType(t) for t in message.get("metric_types", [])]
            update_frequency = message.get("update_frequency", 30)
            
            subscription_id = await self.streamer.subscribe_dashboard(
                client_id=client_id,
                metric_types=metric_types,
                update_frequency=update_frequency
            )
            
            # Start forwarding updates to this WebSocket
            asyncio.create_task(
                self._forward_updates_to_websocket(websocket, subscription_id)
            )
            
            await websocket.send_json({
                "type": "subscribed",
                "subscription_id": subscription_id,
                "status": "success"
            })
            
        elif message_type == "unsubscribe":
            subscription_id = message.get("subscription_id")
            if not subscription_id:
                await websocket.send_json({
                    "type": "error",
                    "message": "subscription_id is required"
                })
                return
                
            await self.streamer.unsubscribe_dashboard(subscription_id)
            
            await websocket.send_json({
                "type": "unsubscribed",
                "subscription_id": subscription_id,
                "status": "success"
            })
    
    async def _forward_updates_to_websocket(self, websocket, subscription_id: str):
        """Forward metric updates to WebSocket client"""
        channel = f"dashboard:{subscription_id}"
        
        # Subscribe to Redis channel
        await self.streamer.pubsub.subscribe(channel)
        
        try:
            async for message in self.streamer.pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    
                    # Send to WebSocket
                    await websocket.send_json({
                        "type": "metric_update",
                        "data": data
                    })
                    
        except Exception as e:
            logger.error(f"Error forwarding updates: {e}")
        finally:
            await self.streamer.pubsub.unsubscribe(channel)

class MetricsAggregator:
    """Aggregate metrics for efficient streaming"""
    
    def __init__(self):
        self.aggregation_windows = {
            "1min": deque(maxlen=60),
            "5min": deque(maxlen=300),
            "15min": deque(maxlen=900)
        }
        
    async def add_metric(self, metric: MetricUpdate):
        """Add metric to aggregation windows"""
        for window in self.aggregation_windows.values():
            window.append(metric)
    
    async def get_aggregated_metrics(self, window: str = "5min") -> Dict[str, Any]:
        """Get aggregated metrics for a time window"""
        if window not in self.aggregation_windows:
            raise ValueError(f"Invalid window: {window}")
        
        metrics = list(self.aggregation_windows[window])
        
        if not metrics:
            return {}
        
        # Group by metric type and name
        grouped = defaultdict(list)
        for metric in metrics:
            key = f"{metric.metric_type.value}:{metric.metric_name}"
            grouped[key].append(metric.value)
        
        # Calculate aggregations
        aggregated = {}
        for key, values in grouped.items():
            aggregated[key] = {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "count": len(values),
                "latest": values[-1]
            }
        
        return aggregated