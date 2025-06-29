"""Caching Strategy for Business Intelligence Data"""

from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import json
import logging
import hashlib
from functools import wraps
import asyncio
import redis.asyncio as redis
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Cache configuration"""
    expire_minutes: int = 30
    key_prefix: str = "bi"
    enable_compression: bool = True
    max_size_mb: float = 10.0

class IntelligenceCacheManager:
    """Manage caching for Business Intelligence data"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", config: Optional[CacheConfig] = None):
        self.redis_url = redis_url
        self.redis_client = None
        self.config = config or CacheConfig()
        self._local_cache = {}  # In-memory cache for frequently accessed data
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "errors": 0
        }
        
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = await redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Cache manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize cache: {e}")
            # Continue without cache
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
    
    def _generate_cache_key(self, namespace: str, params: Dict[str, Any]) -> str:
        """Generate consistent cache key from parameters"""
        # Sort parameters for consistent key generation
        sorted_params = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(sorted_params.encode()).hexdigest()[:8]
        
        return f"{self.config.key_prefix}:{namespace}:{param_hash}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        # Check local cache first
        if key in self._local_cache:
            self._cache_stats["hits"] += 1
            return self._local_cache[key]
        
        if not self.redis_client:
            self._cache_stats["misses"] += 1
            return None
        
        try:
            # Get from Redis
            value = await self.redis_client.get(key)
            
            if value:
                self._cache_stats["hits"] += 1
                # Decompress if needed
                if self.config.enable_compression:
                    import zlib
                    value = zlib.decompress(value)
                
                # Deserialize
                result = json.loads(value)
                
                # Store in local cache
                self._local_cache[key] = result
                
                return result
            else:
                self._cache_stats["misses"] += 1
                return None
                
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self._cache_stats["errors"] += 1
            return None
    
    async def set(self, key: str, value: Any, expire_minutes: Optional[int] = None):
        """Set value in cache"""
        if not self.redis_client:
            return
        
        try:
            # Serialize
            serialized = json.dumps(value)
            
            # Check size
            size_mb = len(serialized) / (1024 * 1024)
            if size_mb > self.config.max_size_mb:
                logger.warning(f"Cache value too large ({size_mb:.2f} MB), skipping cache")
                return
            
            # Compress if enabled
            if self.config.enable_compression:
                import zlib
                serialized = zlib.compress(serialized.encode())
            else:
                serialized = serialized.encode()
            
            # Set in Redis with expiration
            expire_seconds = (expire_minutes or self.config.expire_minutes) * 60
            await self.redis_client.setex(key, expire_seconds, serialized)
            
            # Also store in local cache
            self._local_cache[key] = value
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            self._cache_stats["errors"] += 1
    
    async def delete(self, key: str):
        """Delete value from cache"""
        # Remove from local cache
        if key in self._local_cache:
            del self._local_cache[key]
        
        if self.redis_client:
            try:
                await self.redis_client.delete(key)
            except Exception as e:
                logger.error(f"Cache delete error: {e}")
    
    async def clear_namespace(self, namespace: str):
        """Clear all keys in a namespace"""
        if not self.redis_client:
            return
        
        try:
            pattern = f"{self.config.key_prefix}:{namespace}:*"
            cursor = 0
            
            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor, match=pattern, count=100
                )
                
                if keys:
                    await self.redis_client.delete(*keys)
                
                if cursor == 0:
                    break
                    
            # Clear from local cache
            keys_to_remove = [k for k in self._local_cache.keys() if k.startswith(f"{self.config.key_prefix}:{namespace}:")]
            for key in keys_to_remove:
                del self._local_cache[key]
                
        except Exception as e:
            logger.error(f"Cache clear namespace error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = (self._cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "hits": self._cache_stats["hits"],
            "misses": self._cache_stats["misses"],
            "errors": self._cache_stats["errors"],
            "hit_rate": f"{hit_rate:.2f}%",
            "local_cache_size": len(self._local_cache)
        }

def cache_intelligence_data(
    namespace: str,
    expire_minutes: int = 30,
    key_params: Optional[List[str]] = None
):
    """Decorator for caching Business Intelligence data"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get cache manager (should be injected or global)
            cache_manager = getattr(args[0], '_cache_manager', None) if args else None
            
            if not cache_manager:
                # No cache available, execute function
                return await func(*args, **kwargs)
            
            # Generate cache key
            cache_params = {}
            
            # Add specified parameters to cache key
            if key_params:
                for param in key_params:
                    if param in kwargs:
                        cache_params[param] = kwargs[param]
            
            # Add method name
            cache_params["method"] = func.__name__
            
            # Add client_id if available
            if hasattr(args[0], 'client_id'):
                cache_params["client_id"] = args[0].client_id
            
            cache_key = cache_manager._generate_cache_key(namespace, cache_params)
            
            # Try to get from cache
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache the result
            await cache_manager.set(cache_key, result, expire_minutes)
            
            return result
            
        return wrapper
    return decorator

class SmartCacheInvalidator:
    """Intelligent cache invalidation based on data changes"""
    
    def __init__(self, cache_manager: IntelligenceCacheManager):
        self.cache_manager = cache_manager
        self.invalidation_rules = self._setup_invalidation_rules()
        
    def _setup_invalidation_rules(self) -> Dict[str, List[str]]:
        """Define which events invalidate which cache namespaces"""
        return {
            "revenue_update": ["revenue", "dashboard", "pricing"],
            "engagement_update": ["engagement", "usage", "dashboard"],
            "content_published": ["content", "performance", "dashboard"],
            "pricing_changed": ["pricing", "revenue", "dashboard"],
            "system_error": ["performance", "dashboard"],
            "user_segment_changed": ["segmentation", "pricing", "dashboard"]
        }
    
    async def handle_event(self, event_type: str, event_data: Dict[str, Any]):
        """Handle cache invalidation based on event"""
        if event_type not in self.invalidation_rules:
            return
        
        namespaces_to_invalidate = self.invalidation_rules[event_type]
        
        for namespace in namespaces_to_invalidate:
            logger.info(f"Invalidating cache namespace '{namespace}' due to event '{event_type}'")
            await self.cache_manager.clear_namespace(namespace)
    
    async def schedule_periodic_invalidation(self):
        """Schedule periodic cache invalidation for certain namespaces"""
        while True:
            try:
                # Invalidate real-time data every 5 minutes
                await asyncio.sleep(300)  # 5 minutes
                await self.cache_manager.clear_namespace("realtime")
                
                # Invalidate dashboard data every 15 minutes
                await asyncio.sleep(600)  # 10 more minutes
                await self.cache_manager.clear_namespace("dashboard")
                
            except Exception as e:
                logger.error(f"Periodic invalidation error: {e}")
                await asyncio.sleep(60)  # Wait a minute on error

class CacheWarmer:
    """Pre-warm cache with frequently accessed data"""
    
    def __init__(self, cache_manager: IntelligenceCacheManager):
        self.cache_manager = cache_manager
        self.warming_tasks = []
        
    async def warm_dashboard_cache(self, client_id: str):
        """Pre-warm dashboard cache for a client"""
        from backend.intelligence import (
            UsageAnalyticsEngine,
            PerformanceMonitoringSystem,
            RevenueTrackingEngine,
            AnalyticsTimeframe
        )
        
        logger.info(f"Warming dashboard cache for client {client_id}")
        
        # Common timeframes to pre-warm
        timeframes = [AnalyticsTimeframe.DAY, AnalyticsTimeframe.WEEK, AnalyticsTimeframe.MONTH]
        
        for timeframe in timeframes:
            # Usage analytics
            usage_engine = UsageAnalyticsEngine(client_id)
            usage_engine._cache_manager = self.cache_manager
            
            try:
                await usage_engine.get_business_intelligence(timeframe)
            except Exception as e:
                logger.error(f"Error warming usage cache: {e}")
            
            # Performance monitoring
            perf_monitor = PerformanceMonitoringSystem(client_id)
            perf_monitor._cache_manager = self.cache_manager
            
            try:
                await perf_monitor.get_business_intelligence(timeframe)
            except Exception as e:
                logger.error(f"Error warming performance cache: {e}")
            
            # Revenue tracking
            revenue_tracker = RevenueTrackingEngine(client_id)
            revenue_tracker._cache_manager = self.cache_manager
            
            try:
                await revenue_tracker.get_business_intelligence(timeframe)
            except Exception as e:
                logger.error(f"Error warming revenue cache: {e}")
        
        logger.info(f"Dashboard cache warming completed for client {client_id}")
    
    async def schedule_cache_warming(self, client_ids: List[str]):
        """Schedule periodic cache warming for active clients"""
        while True:
            try:
                for client_id in client_ids:
                    await self.warm_dashboard_cache(client_id)
                    await asyncio.sleep(60)  # Space out warming tasks
                
                # Wait before next warming cycle
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"Cache warming error: {e}")
                await asyncio.sleep(300)  # 5 minutes on error

class CacheMetrics:
    """Track cache performance metrics"""
    
    def __init__(self):
        self.metrics = {
            "cache_size_bytes": 0,
            "eviction_count": 0,
            "avg_response_time_ms": 0,
            "cache_operations": []
        }
        
    async def record_operation(self, operation: str, duration_ms: float, size_bytes: int = 0):
        """Record a cache operation"""
        self.metrics["cache_operations"].append({
            "operation": operation,
            "duration_ms": duration_ms,
            "size_bytes": size_bytes,
            "timestamp": datetime.now()
        })
        
        # Keep only last 1000 operations
        if len(self.metrics["cache_operations"]) > 1000:
            self.metrics["cache_operations"] = self.metrics["cache_operations"][-1000:]
        
        # Update average response time
        recent_ops = self.metrics["cache_operations"][-100:]
        if recent_ops:
            self.metrics["avg_response_time_ms"] = sum(op["duration_ms"] for op in recent_ops) / len(recent_ops)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics"""
        return {
            "cache_size_mb": self.metrics["cache_size_bytes"] / (1024 * 1024),
            "eviction_count": self.metrics["eviction_count"],
            "avg_response_time_ms": round(self.metrics["avg_response_time_ms"], 2),
            "recent_operations": len(self.metrics["cache_operations"])
        }