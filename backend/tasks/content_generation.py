"""
Celery Background Tasks for Content Generation - AutoGuru Universal

This module provides asynchronous background tasks for content workflow automation
that works universally across ANY business niche. It includes tasks for content
analysis, persona generation, viral content creation, publishing, monitoring,
and continuous optimization using AI-driven strategies.

Features:
- Universal task handling for any business type (no hardcoded logic)
- Async task execution with proper error handling and retries
- Task status tracking and result caching
- High-volume content generation support
- Platform-specific publishing with credential encryption
- Real-time analytics updates and strategy optimization
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import asdict
import traceback
from functools import wraps

from celery import Celery, Task, states
from celery.exceptions import Retry, MaxRetriesExceededError
from celery.result import AsyncResult
from kombu import Queue
import asyncpg
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config.settings import get_settings
from backend.core.content_analyzer import UniversalContentAnalyzer, BusinessNiche, Platform
from backend.core.persona_factory import PersonaFactory  # To be implemented
from backend.core.viral_engine import ViralContentEngine  # To be implemented
from backend.database.connection import get_db_session, get_db_context
from backend.utils.encryption import EncryptionManager
from backend.models.content_models import (
    ContentAnalysis, PlatformContent, AudienceProfile,
    BrandVoice, ContentTheme, ViralScore
)
from backend.platforms.publishers import PlatformPublisherFactory  # To be implemented

# Configure logging
logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize Celery app
app = Celery('autoguru')
app.config_from_object({
    'broker_url': settings.celery_broker_url or settings.redis_dsn,
    'result_backend': settings.celery_result_backend or settings.redis_dsn,
    'task_serializer': 'json',
    'result_serializer': 'json',
    'accept_content': ['json'],
    'timezone': 'UTC',
    'enable_utc': True,
    'task_track_started': True,
    'task_send_sent_event': True,
    'task_acks_late': True,
    'worker_prefetch_multiplier': 1,
    'task_default_retry_delay': 30,
    'task_max_retries': 3,
    'task_soft_time_limit': 300,
    'task_time_limit': 600,
    'task_always_eager': settings.environment == 'test',
    'task_eager_propagates': True,
    'task_default_queue': 'default',
    'task_queues': (
        Queue('default', routing_key='task.#'),
        Queue('content_analysis', routing_key='content.analysis.#'),
        Queue('content_generation', routing_key='content.generation.#'),
        Queue('publishing', routing_key='publishing.#'),
        Queue('analytics', routing_key='analytics.#'),
        Queue('optimization', routing_key='optimization.#'),
    ),
    'task_routes': {
        'content_generation.analyze_content_task': {'queue': 'content_analysis'},
        'content_generation.generate_persona_task': {'queue': 'content_analysis'},
        'content_generation.create_viral_content_task': {'queue': 'content_generation'},
        'content_generation.publish_content_task': {'queue': 'publishing'},
        'content_generation.bulk_publish_task': {'queue': 'publishing'},
        'content_generation.update_analytics_task': {'queue': 'analytics'},
        'content_generation.optimize_strategy_task': {'queue': 'optimization'},
    }
})


class DatabaseTask(Task):
    """Base task class with database connection management"""
    
    _db_manager: Optional[PostgreSQLConnectionManager] = None
    _encryption_manager: Optional[EncryptionManager] = None
    _content_analyzer: Optional[UniversalContentAnalyzer] = None
    
    async def get_db_manager(self) -> PostgreSQLConnectionManager:
        """Get or create database manager instance"""
        if self._db_manager is None:
            # Create a new instance instead of using the singleton
            self._db_manager = PostgreSQLConnectionManager()
            await self._db_manager.initialize()
        return self._db_manager
    
    @property
    def encryption_manager(self) -> EncryptionManager:
        if self._encryption_manager is None:
            self._encryption_manager = EncryptionManager()
        return self._encryption_manager
    
    @property
    def content_analyzer(self) -> UniversalContentAnalyzer:
        if self._content_analyzer is None:
            self._content_analyzer = UniversalContentAnalyzer(
                openai_api_key=settings.openai_api_key,
                anthropic_api_key=settings.anthropic_api_key,
                default_llm=settings.default_llm_provider
            )
        return self._content_analyzer


def async_task(func):
    """Decorator to run async functions in Celery tasks"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(func(*args, **kwargs))
        finally:
            loop.close()
    return wrapper


# ============== CONTENT PROCESSING TASKS ==============

@app.task(
    bind=True,
    base=DatabaseTask,
    name='content_generation.analyze_content_task',
    retry_kwargs={'max_retries': 3, 'countdown': 30}
)
@async_task
async def analyze_content_task(
    self: DatabaseTask,
    content: str,
    client_id: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Background content analysis for any business type.
    
    Args:
        content: Content to analyze
        client_id: Unique client identifier
        metadata: Additional metadata for analysis
        
    Returns:
        Dictionary with analysis_id and summary results
    """
    task_id = self.request.id
    logger.info(f"Starting content analysis task {task_id} for client {client_id}")
    
    try:
        # Update task state
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Analyzing content...', 'progress': 10}
        )
        
        # Perform AI-powered content analysis
        analysis_result = await self.content_analyzer.analyze_content(
            content=content,
            context=metadata,
            platforms=None  # Analyze for all platforms
        )
        
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Storing analysis results...', 'progress': 70}
        )
        
        # Store analysis in database
        analysis_data = {
            'client_id': client_id,
            'business_niche': analysis_result.business_niche.value,
            'confidence_score': analysis_result.confidence_score,
            'target_audience': analysis_result.target_audience,  # Already a dict
            'brand_voice': analysis_result.brand_voice,  # Already a dict
            'viral_potential': {
                platform.value: score 
                for platform, score in analysis_result.viral_potential.items()
            },
            'key_themes': analysis_result.key_themes,
            'recommendations': analysis_result.recommendations,
            'metadata': {
                **analysis_result.metadata,
                'task_id': task_id,
                'original_metadata': metadata or {}
            },
            'created_at': datetime.utcnow()
        }
        
        # Insert into database
        query = """
            INSERT INTO content_analyses 
            (client_id, business_niche, confidence_score, target_audience, 
             brand_voice, viral_potential, key_themes, recommendations, 
             metadata, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING id
        """
        
        analysis_id = await self.db_manager.fetch_value(
            query,
            client_id,
            analysis_data['business_niche'],
            analysis_data['confidence_score'],
            json.dumps(analysis_data['target_audience']),
            json.dumps(analysis_data['brand_voice']),
            json.dumps(analysis_data['viral_potential']),
            analysis_data['key_themes'],
            analysis_data['recommendations'],
            json.dumps(analysis_data['metadata']),
            analysis_data['created_at']
        )
        
        logger.info(f"Content analysis completed. Analysis ID: {analysis_id}")
        
        # Return summary results
        return {
            'status': 'success',
            'analysis_id': str(analysis_id),
            'business_niche': analysis_result.business_niche.value,
            'confidence_score': analysis_result.confidence_score,
            'top_platforms': sorted(
                analysis_result.viral_potential.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3],
            'primary_theme': analysis_result.key_themes[0] if analysis_result.key_themes else None,
            'task_id': task_id
        }
        
    except Exception as e:
        logger.error(f"Content analysis task failed: {str(e)}\n{traceback.format_exc()}")
        
        # Store error state
        error_data = {
            'task_id': task_id,
            'client_id': client_id,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Log error to database for debugging
        try:
            await self.db_manager.execute(
                """
                INSERT INTO task_errors (task_id, task_type, client_id, error_data, created_at)
                VALUES ($1, $2, $3, $4, $5)
                """,
                task_id,
                'content_analysis',
                client_id,
                json.dumps(error_data),
                datetime.utcnow()
            )
        except:
            pass
        
        # Retry if possible
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=60 * (self.request.retries + 1))
        
        return {
            'status': 'error',
            'error': str(e),
            'task_id': task_id
        }


@app.task(
    bind=True,
    base=DatabaseTask,
    name='content_generation.generate_persona_task',
    retry_kwargs={'max_retries': 3, 'countdown': 45}
)
@async_task
async def generate_persona_task(
    self: DatabaseTask,
    content_analysis_id: str,
    client_preferences: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate business persona based on content analysis.
    
    Args:
        content_analysis_id: ID of the content analysis
        client_preferences: Optional client preferences for persona
        
    Returns:
        Dictionary with persona_id and configuration
    """
    task_id = self.request.id
    logger.info(f"Starting persona generation task {task_id} for analysis {content_analysis_id}")
    
    try:
        # Update task state
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Loading analysis data...', 'progress': 10}
        )
        
        # Fetch content analysis from database
        analysis_data = await self.db_manager.fetch_one(
            """
            SELECT * FROM content_analyses WHERE id = $1
            """,
            int(content_analysis_id)
        )
        
        if not analysis_data:
            raise ValueError(f"Content analysis {content_analysis_id} not found")
        
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Generating persona...', 'progress': 30}
        )
        
        # Initialize persona factory
        persona_factory = PersonaFactory(
            encryption_manager=self.encryption_manager,
            content_analyzer=self.content_analyzer
        )
        
        # Generate persona based on analysis
        persona = await persona_factory.create_persona(
            business_niche=BusinessNiche(analysis_data['business_niche']),
            target_audience=json.loads(analysis_data['target_audience']),
            brand_voice=json.loads(analysis_data['brand_voice']),
            preferences=client_preferences
        )
        
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Storing persona configuration...', 'progress': 80}
        )
        
        # Store persona in database
        persona_data = {
            'client_id': analysis_data['client_id'],
            'analysis_id': content_analysis_id,
            'persona_type': persona.persona_type,
            'configuration': persona.to_dict(),
            'niche_adaptations': persona.get_niche_adaptations(),
            'platform_strategies': persona.get_platform_strategies(),
            'metadata': {
                'task_id': task_id,
                'preferences': client_preferences or {},
                'generated_at': datetime.utcnow().isoformat()
            },
            'is_active': True,
            'created_at': datetime.utcnow()
        }
        
        query = """
            INSERT INTO personas
            (client_id, analysis_id, persona_type, configuration, 
             niche_adaptations, platform_strategies, metadata, is_active, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id
        """
        
        persona_id = await self.db_manager.fetch_value(
            query,
            persona_data['client_id'],
            int(content_analysis_id),
            persona_data['persona_type'],
            json.dumps(persona_data['configuration']),
            json.dumps(persona_data['niche_adaptations']),
            json.dumps(persona_data['platform_strategies']),
            json.dumps(persona_data['metadata']),
            persona_data['is_active'],
            persona_data['created_at']
        )
        
        logger.info(f"Persona generation completed. Persona ID: {persona_id}")
        
        return {
            'status': 'success',
            'persona_id': str(persona_id),
            'persona_type': persona.persona_type,
            'business_niche': analysis_data['business_niche'],
            'key_traits': persona.get_key_traits()[:5],
            'task_id': task_id
        }
        
    except Exception as e:
        logger.error(f"Persona generation task failed: {str(e)}\n{traceback.format_exc()}")
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=90 * (self.request.retries + 1))
        
        return {
            'status': 'error',
            'error': str(e),
            'task_id': task_id
        }


@app.task(
    bind=True,
    base=DatabaseTask,
    name='content_generation.create_viral_content_task',
    retry_kwargs={'max_retries': 3, 'countdown': 60}
)
@async_task
async def create_viral_content_task(
    self: DatabaseTask,
    analysis_id: str,
    persona_id: str,
    target_platforms: List[str],
    content_preferences: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate viral content for specified platforms.
    
    Args:
        analysis_id: Content analysis ID
        persona_id: Persona ID to use
        target_platforms: List of platform names
        content_preferences: Optional content generation preferences
        
    Returns:
        Dictionary with content_id and generated variations
    """
    task_id = self.request.id
    logger.info(f"Starting viral content creation task {task_id}")
    
    try:
        # Update task state
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Loading analysis and persona data...', 'progress': 10}
        )
        
        # Fetch analysis and persona data
        analysis_data = await self.db_manager.fetch_one(
            "SELECT * FROM content_analyses WHERE id = $1",
            int(analysis_id)
        )
        
        persona_data = await self.db_manager.fetch_one(
            "SELECT * FROM personas WHERE id = $1",
            int(persona_id)
        )
        
        if not analysis_data or not persona_data:
            raise ValueError("Analysis or persona not found")
        
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Initializing viral content engine...', 'progress': 20}
        )
        
        # Initialize viral content engine
        viral_engine = ViralContentEngine(
            content_analyzer=self.content_analyzer,
            encryption_manager=self.encryption_manager
        )
        
        # Convert platform strings to Platform enums
        platforms = [Platform(p) for p in target_platforms]
        
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Generating viral content variations...', 'progress': 40}
        )
        
        # Generate viral content
        viral_content = await viral_engine.generate_viral_content(
            business_niche=BusinessNiche(analysis_data['business_niche']),
            target_audience=json.loads(analysis_data['target_audience']),
            brand_voice=json.loads(analysis_data['brand_voice']),
            persona_config=json.loads(persona_data['configuration']),
            platforms=platforms,
            preferences=content_preferences
        )
        
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Optimizing for each platform...', 'progress': 70}
        )
        
        # Store content variations
        content_data = {
            'client_id': analysis_data['client_id'],
            'analysis_id': analysis_id,
            'persona_id': persona_id,
            'content_type': 'viral_generated',
            'base_content': viral_content.base_content,
            'platform_variations': {},
            'hashtags': {},
            'posting_schedules': {},
            'metadata': {
                'task_id': task_id,
                'preferences': content_preferences or {},
                'generated_at': datetime.utcnow().isoformat()
            },
            'created_at': datetime.utcnow()
        }
        
        # Process each platform variation
        for platform in platforms:
            variation = viral_content.get_platform_variation(platform)
            if variation:
                content_data['platform_variations'][platform.value] = {
                    'content': variation.content_text,
                    'format': variation.content_format.value,
                    'media_requirements': variation.media_requirements,
                    'call_to_action': variation.call_to_action,
                    'character_count': variation.character_count
                }
                content_data['hashtags'][platform.value] = variation.hashtags
                
                # Generate optimal posting schedule
                schedule = await viral_engine.generate_posting_schedule(
                    platform=platform,
                    audience_data=json.loads(analysis_data['target_audience']),
                    content_type=variation.content_format
                )
                content_data['posting_schedules'][platform.value] = schedule
        
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Storing generated content...', 'progress': 90}
        )
        
        # Store in database
        query = """
            INSERT INTO generated_content
            (client_id, analysis_id, persona_id, content_type, base_content,
             platform_variations, hashtags, posting_schedules, metadata, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING id
        """
        
        content_id = await self.db_manager.fetch_value(
            query,
            content_data['client_id'],
            int(analysis_id),
            int(persona_id),
            content_data['content_type'],
            content_data['base_content'],
            json.dumps(content_data['platform_variations']),
            json.dumps(content_data['hashtags']),
            json.dumps(content_data['posting_schedules']),
            json.dumps(content_data['metadata']),
            content_data['created_at']
        )
        
        logger.info(f"Viral content creation completed. Content ID: {content_id}")
        
        return {
            'status': 'success',
            'content_id': str(content_id),
            'platforms_generated': list(content_data['platform_variations'].keys()),
            'total_variations': len(content_data['platform_variations']),
            'preview': {
                platform: variation['content'][:100] + '...'
                for platform, variation in list(content_data['platform_variations'].items())[:2]
            },
            'task_id': task_id
        }
        
    except Exception as e:
        logger.error(f"Viral content creation failed: {str(e)}\n{traceback.format_exc()}")
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=120 * (self.request.retries + 1))
        
        return {
            'status': 'error',
            'error': str(e),
            'task_id': task_id
        }


# ============== PUBLISHING TASKS ==============

@app.task(
    bind=True,
    base=DatabaseTask,
    name='content_generation.publish_content_task',
    retry_kwargs={'max_retries': 3, 'countdown': 30}
)
@async_task
async def publish_content_task(
    self: DatabaseTask,
    content_id: str,
    platform: str,
    credentials: Dict[str, Any],
    schedule_time: Optional[str] = None
) -> Dict[str, Any]:
    """
    Publish content to social media platform.
    
    Args:
        content_id: Generated content ID
        platform: Platform name
        credentials: Encrypted platform credentials
        schedule_time: Optional ISO format datetime for scheduled posting
        
    Returns:
        Dictionary with publishing results
    """
    task_id = self.request.id
    logger.info(f"Starting content publishing task {task_id} for platform {platform}")
    
    try:
        # Update task state
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Loading content data...', 'progress': 10}
        )
        
        # Fetch content data
        content_data = await self.db_manager.fetch_one(
            "SELECT * FROM generated_content WHERE id = $1",
            int(content_id)
        )
        
        if not content_data:
            raise ValueError(f"Content {content_id} not found")
        
        # Validate platform
        if platform not in content_data['platform_variations']:
            raise ValueError(f"No content variation found for platform {platform}")
        
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Decrypting credentials...', 'progress': 20}
        )
        
        # Decrypt platform credentials
        decrypted_creds = self.encryption_manager.decrypt_credentials(credentials)
        
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Initializing platform publisher...', 'progress': 30}
        )
        
        # Get platform publisher
        publisher_factory = PlatformPublisherFactory()
        publisher = await publisher_factory.get_publisher(
            platform=Platform(platform),
            credentials=decrypted_creds
        )
        
        # Get platform-specific content
        platform_content = content_data['platform_variations'][platform]
        hashtags = content_data['hashtags'].get(platform, [])
        
        # Handle scheduled posting
        if schedule_time:
            schedule_dt = datetime.fromisoformat(schedule_time)
            if schedule_dt > datetime.utcnow():
                self.update_state(
                    state='PROCESSING',
                    meta={'status': 'Scheduling content...', 'progress': 50}
                )
                
                # Schedule the post
                scheduled_result = await publisher.schedule_post(
                    content=platform_content['content'],
                    media_urls=platform_content.get('media_urls', []),
                    hashtags=hashtags,
                    schedule_time=schedule_dt
                )
                
                # Store scheduling info
                await self._store_publishing_result(
                    content_id=content_id,
                    platform=platform,
                    status='scheduled',
                    result_data=scheduled_result,
                    scheduled_for=schedule_dt
                )
                
                return {
                    'status': 'scheduled',
                    'platform': platform,
                    'scheduled_for': schedule_time,
                    'scheduled_post_id': scheduled_result.get('scheduled_id'),
                    'task_id': task_id
                }
        
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Publishing content...', 'progress': 60}
        )
        
        # Publish immediately
        publish_result = await publisher.publish_content(
            content=platform_content['content'],
            media_urls=platform_content.get('media_urls', []),
            hashtags=hashtags,
            mentions=platform_content.get('mentions', [])
        )
        
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Storing publishing results...', 'progress': 90}
        )
        
        # Store publishing result
        await self._store_publishing_result(
            content_id=content_id,
            platform=platform,
            status='published',
            result_data=publish_result,
            published_at=datetime.utcnow()
        )
        
        logger.info(f"Content published successfully to {platform}")
        
        return {
            'status': 'published',
            'platform': platform,
            'post_id': publish_result.get('post_id'),
            'post_url': publish_result.get('post_url'),
            'published_at': datetime.utcnow().isoformat(),
            'initial_metrics': publish_result.get('initial_metrics', {}),
            'task_id': task_id
        }
        
    except Exception as e:
        logger.error(f"Content publishing failed: {str(e)}\n{traceback.format_exc()}")
        
        # Store failure
        try:
            await self._store_publishing_result(
                content_id=content_id,
                platform=platform,
                status='failed',
                error_data={'error': str(e), 'traceback': traceback.format_exc()}
            )
        except:
            pass
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=60 * (self.request.retries + 1))
        
        return {
            'status': 'error',
            'platform': platform,
            'error': str(e),
            'task_id': task_id
        }
    
    async def _store_publishing_result(
        self,
        content_id: str,
        platform: str,
        status: str,
        result_data: Optional[Dict] = None,
        error_data: Optional[Dict] = None,
        published_at: Optional[datetime] = None,
        scheduled_for: Optional[datetime] = None
    ):
        """Store publishing result in database"""
        query = """
            INSERT INTO publishing_results
            (content_id, platform, status, result_data, error_data, 
             published_at, scheduled_for, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """
        
        await self.db_manager.execute(
            query,
            int(content_id),
            platform,
            status,
            json.dumps(result_data) if result_data else None,
            json.dumps(error_data) if error_data else None,
            published_at,
            scheduled_for,
            datetime.utcnow()
        )


@app.task(
    bind=True,
    base=DatabaseTask,
    name='content_generation.bulk_publish_task',
    retry_kwargs={'max_retries': 2, 'countdown': 60}
)
@async_task
async def bulk_publish_task(
    self: DatabaseTask,
    content_id: str,
    platforms: List[str],
    credentials: Dict[str, Dict[str, Any]],
    stagger_minutes: int = 5
) -> Dict[str, Any]:
    """
    Publish same content to multiple platforms.
    
    Args:
        content_id: Generated content ID
        platforms: List of platform names
        credentials: Dictionary mapping platform to encrypted credentials
        stagger_minutes: Minutes to wait between platform posts
        
    Returns:
        Dictionary with bulk publishing results
    """
    task_id = self.request.id
    logger.info(f"Starting bulk publishing task {task_id} for {len(platforms)} platforms")
    
    results = {
        'total_platforms': len(platforms),
        'successful': [],
        'failed': [],
        'task_id': task_id
    }
    
    try:
        # Update task state
        self.update_state(
            state='PROCESSING',
            meta={
                'status': 'Starting bulk publishing...',
                'progress': 0,
                'platforms_completed': 0,
                'total_platforms': len(platforms)
            }
        )
        
        # Publish to each platform with staggering
        for i, platform in enumerate(platforms):
            try:
                # Update progress
                self.update_state(
                    state='PROCESSING',
                    meta={
                        'status': f'Publishing to {platform}...',
                        'progress': int((i / len(platforms)) * 100),
                        'platforms_completed': i,
                        'total_platforms': len(platforms)
                    }
                )
                
                # Schedule with stagger
                if i > 0:
                    schedule_time = (
                        datetime.utcnow() + timedelta(minutes=stagger_minutes * i)
                    ).isoformat()
                else:
                    schedule_time = None
                
                # Get platform credentials
                platform_creds = credentials.get(platform)
                if not platform_creds:
                    raise ValueError(f"No credentials provided for {platform}")
                
                # Publish to platform
                result = await publish_content_task(
                    content_id=content_id,
                    platform=platform,
                    credentials=platform_creds,
                    schedule_time=schedule_time
                )
                
                if result['status'] in ['published', 'scheduled']:
                    results['successful'].append({
                        'platform': platform,
                        'status': result['status'],
                        'post_id': result.get('post_id'),
                        'scheduled_for': result.get('scheduled_for')
                    })
                else:
                    results['failed'].append({
                        'platform': platform,
                        'error': result.get('error', 'Unknown error')
                    })
                
            except Exception as e:
                logger.error(f"Failed to publish to {platform}: {str(e)}")
                results['failed'].append({
                    'platform': platform,
                    'error': str(e)
                })
        
        # Final summary
        results['status'] = 'completed'
        results['success_rate'] = len(results['successful']) / len(platforms)
        
        logger.info(
            f"Bulk publishing completed. Success: {len(results['successful'])}, "
            f"Failed: {len(results['failed'])}"
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Bulk publishing task failed: {str(e)}")
        results['status'] = 'error'
        results['error'] = str(e)
        return results


# ============== MONITORING TASKS ==============

@app.task(
    bind=True,
    base=DatabaseTask,
    name='content_generation.update_analytics_task',
    retry_kwargs={'max_retries': 3, 'countdown': 300}
)
@async_task
async def update_analytics_task(
    self: DatabaseTask,
    client_id: str,
    platforms: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Update analytics data for client content.
    
    Args:
        client_id: Client identifier
        platforms: Optional list of platforms to update (all if None)
        
    Returns:
        Dictionary with analytics update results
    """
    task_id = self.request.id
    logger.info(f"Starting analytics update task {task_id} for client {client_id}")
    
    try:
        # Update task state
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Fetching published content...', 'progress': 10}
        )
        
        # Get all published content for client
        published_content = await self.db_manager.fetch_all(
            """
            SELECT pr.*, gc.platform_variations
            FROM publishing_results pr
            JOIN generated_content gc ON pr.content_id = gc.id
            WHERE gc.client_id = $1 
            AND pr.status = 'published'
            AND pr.published_at > NOW() - INTERVAL '30 days'
            """ + (f" AND pr.platform = ANY($2)" if platforms else ""),
            client_id,
            *(platforms if platforms else [])
        )
        
        if not published_content:
            logger.info(f"No published content found for client {client_id}")
            return {
                'status': 'success',
                'message': 'No published content to analyze',
                'task_id': task_id
            }
        
        self.update_state(
            state='PROCESSING',
            meta={
                'status': 'Fetching analytics from platforms...',
                'progress': 20,
                'total_posts': len(published_content)
            }
        )
        
        # Group by platform
        platform_groups = {}
        for content in published_content:
            platform = content['platform']
            if platform not in platform_groups:
                platform_groups[platform] = []
            platform_groups[platform].append(content)
        
        # Analytics results
        analytics_data = {
            'client_id': client_id,
            'platforms': {},
            'total_reach': 0,
            'total_engagement': 0,
            'top_performing_content': [],
            'insights': [],
            'updated_at': datetime.utcnow()
        }
        
        # Fetch analytics for each platform
        for platform, posts in platform_groups.items():
            try:
                # Get platform credentials
                creds = await self._get_client_platform_credentials(client_id, platform)
                if not creds:
                    logger.warning(f"No credentials found for {platform}")
                    continue
                
                # Decrypt credentials
                decrypted_creds = self.encryption_manager.decrypt_credentials(creds)
                
                # Get publisher
                publisher_factory = PlatformPublisherFactory()
                publisher = await publisher_factory.get_publisher(
                    platform=Platform(platform),
                    credentials=decrypted_creds
                )
                
                # Fetch analytics for posts
                platform_analytics = await publisher.get_bulk_analytics(
                    post_ids=[p['result_data'].get('post_id') for p in posts if p['result_data']]
                )
                
                # Process analytics
                platform_summary = self._process_platform_analytics(
                    platform_analytics,
                    posts
                )
                
                analytics_data['platforms'][platform] = platform_summary
                analytics_data['total_reach'] += platform_summary['total_reach']
                analytics_data['total_engagement'] += platform_summary['total_engagement']
                
                # Update individual post analytics
                for post_id, metrics in platform_analytics.items():
                    await self.db_manager.execute(
                        """
                        UPDATE publishing_results
                        SET analytics_data = $1, analytics_updated_at = $2
                        WHERE result_data->>'post_id' = $3
                        """,
                        json.dumps(metrics),
                        datetime.utcnow(),
                        post_id
                    )
                
            except Exception as e:
                logger.error(f"Failed to fetch analytics for {platform}: {str(e)}")
                analytics_data['platforms'][platform] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Generating insights...', 'progress': 80}
        )
        
        # Generate AI-powered insights
        insights = await self._generate_analytics_insights(analytics_data)
        analytics_data['insights'] = insights
        
        # Identify top performing content
        analytics_data['top_performing_content'] = await self._identify_top_content(
            client_id,
            limit=10
        )
        
        # Calculate performance trends
        analytics_data['trends'] = await self._calculate_performance_trends(
            client_id,
            days=30
        )
        
        # Store analytics summary
        await self.db_manager.execute(
            """
            INSERT INTO analytics_summaries
            (client_id, analytics_data, period_start, period_end, created_at)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (client_id, period_start) 
            DO UPDATE SET analytics_data = $2, updated_at = $5
            """,
            client_id,
            json.dumps(analytics_data),
            datetime.utcnow().date() - timedelta(days=30),
            datetime.utcnow().date(),
            datetime.utcnow()
        )
        
        logger.info(f"Analytics update completed for client {client_id}")
        
        return {
            'status': 'success',
            'client_id': client_id,
            'platforms_updated': list(analytics_data['platforms'].keys()),
            'total_reach': analytics_data['total_reach'],
            'total_engagement': analytics_data['total_engagement'],
            'insights_generated': len(analytics_data['insights']),
            'task_id': task_id
        }
        
    except Exception as e:
        logger.error(f"Analytics update failed: {str(e)}\n{traceback.format_exc()}")
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=600 * (self.request.retries + 1))
        
        return {
            'status': 'error',
            'error': str(e),
            'task_id': task_id
        }
    
    def _process_platform_analytics(
        self,
        platform_analytics: Dict[str, Dict],
        posts: List[Record]
    ) -> Dict[str, Any]:
        """Process raw platform analytics into summary"""
        summary = {
            'total_posts': len(posts),
            'total_reach': 0,
            'total_engagement': 0,
            'average_engagement_rate': 0,
            'top_metrics': {},
            'content_performance': []
        }
        
        for post_id, metrics in platform_analytics.items():
            reach = metrics.get('reach', 0) or metrics.get('impressions', 0)
            engagement = (
                metrics.get('likes', 0) +
                metrics.get('comments', 0) +
                metrics.get('shares', 0) +
                metrics.get('saves', 0)
            )
            
            summary['total_reach'] += reach
            summary['total_engagement'] += engagement
            
            # Track content performance
            post_data = next((p for p in posts if p['result_data'].get('post_id') == post_id), None)
            if post_data:
                summary['content_performance'].append({
                    'post_id': post_id,
                    'reach': reach,
                    'engagement': engagement,
                    'engagement_rate': (engagement / reach * 100) if reach > 0 else 0,
                    'published_at': post_data['published_at'].isoformat()
                })
        
        # Calculate average engagement rate
        if summary['total_reach'] > 0:
            summary['average_engagement_rate'] = (
                summary['total_engagement'] / summary['total_reach'] * 100
            )
        
        # Sort by engagement rate
        summary['content_performance'].sort(
            key=lambda x: x['engagement_rate'],
            reverse=True
        )
        
        return summary
    
    async def _get_client_platform_credentials(
        self,
        client_id: str,
        platform: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch encrypted platform credentials for client"""
        result = await self.db_manager.fetch_one(
            """
            SELECT encrypted_credentials
            FROM platform_credentials
            WHERE client_id = $1 AND platform = $2 AND is_active = true
            """,
            client_id,
            platform
        )
        
        return json.loads(result['encrypted_credentials']) if result else None
    
    async def _generate_analytics_insights(
        self,
        analytics_data: Dict[str, Any]
    ) -> List[str]:
        """Generate AI-powered insights from analytics data"""
        try:
            # Prepare data for AI analysis
            prompt = f"""
            Analyze the following social media performance data and provide 5-7 actionable insights:
            
            Total Reach: {analytics_data['total_reach']:,}
            Total Engagement: {analytics_data['total_engagement']:,}
            Platforms: {list(analytics_data['platforms'].keys())}
            
            Platform Performance:
            {json.dumps(analytics_data['platforms'], indent=2)}
            
            Provide specific, actionable insights about:
            1. Best performing content types
            2. Optimal posting times
            3. Engagement patterns
            4. Platform-specific recommendations
            5. Content improvement suggestions
            
            Return as JSON: {{"insights": ["insight1", "insight2", ...]}}
            """
            
            response = await self.content_analyzer._call_llm(prompt, temperature=0.7)
            result = json.loads(response)
            return result.get('insights', [])
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {str(e)}")
            return [
                "Consider increasing posting frequency for better engagement",
                "Focus on platforms with highest engagement rates",
                "Analyze top performing content for common patterns"
            ]
    
    async def _identify_top_content(
        self,
        client_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Identify top performing content"""
        results = await self.db_manager.fetch_all(
            """
            SELECT 
                gc.id,
                gc.base_content,
                pr.platform,
                pr.analytics_data,
                pr.published_at,
                (pr.analytics_data->>'engagement')::int as engagement,
                (pr.analytics_data->>'reach')::int as reach,
                CASE 
                    WHEN (pr.analytics_data->>'reach')::int > 0 
                    THEN ((pr.analytics_data->>'engagement')::float / (pr.analytics_data->>'reach')::int * 100)
                    ELSE 0 
                END as engagement_rate
            FROM publishing_results pr
            JOIN generated_content gc ON pr.content_id = gc.id
            WHERE gc.client_id = $1
            AND pr.status = 'published'
            AND pr.analytics_data IS NOT NULL
            ORDER BY engagement_rate DESC, engagement DESC
            LIMIT $2
            """,
            client_id,
            limit
        )
        
        return [
            {
                'content_id': str(r['id']),
                'preview': r['base_content'][:100] + '...',
                'platform': r['platform'],
                'engagement': r['engagement'],
                'reach': r['reach'],
                'engagement_rate': round(r['engagement_rate'], 2),
                'published_at': r['published_at'].isoformat()
            }
            for r in results
        ]
    
    async def _calculate_performance_trends(
        self,
        client_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Calculate performance trends over time"""
        results = await self.db_manager.fetch_all(
            """
            SELECT 
                DATE(pr.published_at) as date,
                COUNT(*) as posts,
                SUM((pr.analytics_data->>'reach')::int) as reach,
                SUM((pr.analytics_data->>'engagement')::int) as engagement
            FROM publishing_results pr
            JOIN generated_content gc ON pr.content_id = gc.id
            WHERE gc.client_id = $1
            AND pr.status = 'published'
            AND pr.published_at > NOW() - INTERVAL '%s days'
            AND pr.analytics_data IS NOT NULL
            GROUP BY DATE(pr.published_at)
            ORDER BY date
            """,
            client_id,
            days
        )
        
        if not results:
            return {}
        
        # Calculate trends
        dates = [r['date'].isoformat() for r in results]
        reach_values = [r['reach'] or 0 for r in results]
        engagement_values = [r['engagement'] or 0 for r in results]
        
        return {
            'dates': dates,
            'reach_trend': reach_values,
            'engagement_trend': engagement_values,
            'average_daily_reach': sum(reach_values) / len(reach_values) if reach_values else 0,
            'average_daily_engagement': sum(engagement_values) / len(engagement_values) if engagement_values else 0,
            'growth_rate': self._calculate_growth_rate(reach_values) if len(reach_values) > 1 else 0
        }
    
    def _calculate_growth_rate(self, values: List[int]) -> float:
        """Calculate growth rate from a list of values"""
        if len(values) < 2 or values[0] == 0:
            return 0.0
        
        # Calculate compound growth rate
        start_value = values[0]
        end_value = values[-1]
        periods = len(values) - 1
        
        if start_value <= 0 or end_value <= 0:
            return 0.0
        
        growth_rate = ((end_value / start_value) ** (1 / periods) - 1) * 100
        return round(growth_rate, 2)


# ============== OPTIMIZATION TASKS ==============

@app.task(
    bind=True,
    base=DatabaseTask,
    name='content_generation.optimize_strategy_task',
    retry_kwargs={'max_retries': 2, 'countdown': 600}
)
@async_task
async def optimize_strategy_task(
    self: DatabaseTask,
    client_id: str,
    optimization_window_days: int = 30
) -> Dict[str, Any]:
    """
    Continuously optimize client's content strategy.
    
    Args:
        client_id: Client identifier
        optimization_window_days: Days of data to analyze for optimization
        
    Returns:
        Dictionary with optimization results and recommendations
    """
    task_id = self.request.id
    logger.info(f"Starting strategy optimization task {task_id} for client {client_id}")
    
    try:
        # Update task state
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Loading analytics data...', 'progress': 10}
        )
        
        # Get recent analytics summary
        analytics_summary = await self.db_manager.fetch_one(
            """
            SELECT * FROM analytics_summaries
            WHERE client_id = $1
            AND created_at > NOW() - INTERVAL '1 day'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            client_id
        )
        
        if not analytics_summary:
            # Trigger analytics update first
            await update_analytics_task.delay(client_id)
            raise ValueError("No recent analytics data available. Analytics update triggered.")
        
        analytics_data = json.loads(analytics_summary['analytics_data'])
        
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Analyzing performance patterns...', 'progress': 20}
        )
        
        # Get current persona and settings
        current_persona = await self.db_manager.fetch_one(
            """
            SELECT * FROM personas
            WHERE client_id = $1 AND is_active = true
            ORDER BY created_at DESC
            LIMIT 1
            """,
            client_id
        )
        
        if not current_persona:
            raise ValueError("No active persona found for client")
        
        # Analyze performance patterns
        performance_analysis = await self._analyze_performance_patterns(
            client_id,
            analytics_data,
            window_days=optimization_window_days
        )
        
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Generating optimization recommendations...', 'progress': 40}
        )
        
        # Generate optimization recommendations
        optimization_engine = OptimizationEngine(
            content_analyzer=self.content_analyzer,
            db_manager=self.db_manager
        )
        
        recommendations = await optimization_engine.generate_recommendations(
            client_id=client_id,
            current_persona=json.loads(current_persona['configuration']),
            performance_data=performance_analysis,
            analytics_data=analytics_data
        )
        
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Adjusting content strategy...', 'progress': 60}
        )
        
        # Apply optimizations
        optimizations_applied = {}
        
        # 1. Update persona if needed
        if recommendations.get('persona_adjustments'):
            persona_update = await self._update_persona_strategy(
                persona_id=current_persona['id'],
                adjustments=recommendations['persona_adjustments']
            )
            optimizations_applied['persona_updated'] = persona_update
        
        # 2. Update posting schedule
        if recommendations.get('schedule_adjustments'):
            schedule_update = await self._update_posting_schedule(
                client_id=client_id,
                schedule_adjustments=recommendations['schedule_adjustments']
            )
            optimizations_applied['schedule_updated'] = schedule_update
        
        # 3. Update content themes
        if recommendations.get('theme_adjustments'):
            theme_update = await self._update_content_themes(
                client_id=client_id,
                theme_adjustments=recommendations['theme_adjustments']
            )
            optimizations_applied['themes_updated'] = theme_update
        
        # 4. Update platform strategies
        if recommendations.get('platform_adjustments'):
            platform_update = await self._update_platform_strategies(
                client_id=client_id,
                platform_adjustments=recommendations['platform_adjustments']
            )
            optimizations_applied['platforms_updated'] = platform_update
        
        self.update_state(
            state='PROCESSING',
            meta={'status': 'Storing optimization results...', 'progress': 90}
        )
        
        # Store optimization results
        optimization_data = {
            'client_id': client_id,
            'performance_analysis': performance_analysis,
            'recommendations': recommendations,
            'optimizations_applied': optimizations_applied,
            'expected_improvements': recommendations.get('expected_improvements', {}),
            'metadata': {
                'task_id': task_id,
                'optimization_window_days': optimization_window_days,
                'analytics_period': {
                    'start': analytics_summary['period_start'].isoformat(),
                    'end': analytics_summary['period_end'].isoformat()
                }
            },
            'created_at': datetime.utcnow()
        }
        
        await self.db_manager.execute(
            """
            INSERT INTO strategy_optimizations
            (client_id, optimization_data, created_at)
            VALUES ($1, $2, $3)
            """,
            client_id,
            json.dumps(optimization_data),
            optimization_data['created_at']
        )
        
        logger.info(f"Strategy optimization completed for client {client_id}")
        
        return {
            'status': 'success',
            'client_id': client_id,
            'optimizations_applied': list(optimizations_applied.keys()),
            'total_recommendations': len(recommendations),
            'expected_improvements': recommendations.get('expected_improvements', {}),
            'next_optimization': (datetime.utcnow() + timedelta(days=7)).isoformat(),
            'task_id': task_id
        }
        
    except Exception as e:
        logger.error(f"Strategy optimization failed: {str(e)}\n{traceback.format_exc()}")
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=1800 * (self.request.retries + 1))
        
        return {
            'status': 'error',
            'error': str(e),
            'task_id': task_id
        }
    
    async def _analyze_performance_patterns(
        self,
        client_id: str,
        analytics_data: Dict[str, Any],
        window_days: int
    ) -> Dict[str, Any]:
        """Analyze performance patterns from historical data"""
        # Get detailed performance data
        performance_data = await self.db_manager.fetch_all(
            """
            SELECT 
                gc.id,
                gc.base_content,
                gc.platform_variations,
                gc.hashtags,
                pr.platform,
                pr.published_at,
                pr.analytics_data,
                EXTRACT(HOUR FROM pr.published_at) as hour,
                EXTRACT(DOW FROM pr.published_at) as day_of_week
            FROM publishing_results pr
            JOIN generated_content gc ON pr.content_id = gc.id
            WHERE gc.client_id = $1
            AND pr.status = 'published'
            AND pr.published_at > NOW() - INTERVAL '%s days'
            AND pr.analytics_data IS NOT NULL
            """,
            client_id,
            window_days
        )
        
        # Analyze patterns
        patterns = {
            'content_type_performance': {},
            'posting_time_performance': {},
            'day_of_week_performance': {},
            'hashtag_performance': {},
            'platform_performance': {},
            'content_length_performance': {},
            'engagement_drivers': []
        }
        
        # Process each post
        for post in performance_data:
            platform = post['platform']
            analytics = json.loads(post['analytics_data'])
            engagement_rate = self._calculate_engagement_rate(analytics)
            
            # Platform performance
            if platform not in patterns['platform_performance']:
                patterns['platform_performance'][platform] = {
                    'posts': 0,
                    'total_engagement_rate': 0,
                    'total_reach': 0
                }
            
            patterns['platform_performance'][platform]['posts'] += 1
            patterns['platform_performance'][platform]['total_engagement_rate'] += engagement_rate
            patterns['platform_performance'][platform]['total_reach'] += analytics.get('reach', 0)
            
            # Time performance
            hour = post['hour']
            if hour not in patterns['posting_time_performance']:
                patterns['posting_time_performance'][hour] = {
                    'posts': 0,
                    'total_engagement_rate': 0
                }
            
            patterns['posting_time_performance'][hour]['posts'] += 1
            patterns['posting_time_performance'][hour]['total_engagement_rate'] += engagement_rate
            
            # Day of week performance
            dow = post['day_of_week']
            if dow not in patterns['day_of_week_performance']:
                patterns['day_of_week_performance'][dow] = {
                    'posts': 0,
                    'total_engagement_rate': 0
                }
            
            patterns['day_of_week_performance'][dow]['posts'] += 1
            patterns['day_of_week_performance'][dow]['total_engagement_rate'] += engagement_rate
            
            # Hashtag performance
            hashtags = json.loads(post['hashtags']).get(platform, [])
            for hashtag in hashtags:
                if hashtag not in patterns['hashtag_performance']:
                    patterns['hashtag_performance'][hashtag] = {
                        'uses': 0,
                        'total_engagement_rate': 0
                    }
                
                patterns['hashtag_performance'][hashtag]['uses'] += 1
                patterns['hashtag_performance'][hashtag]['total_engagement_rate'] += engagement_rate
        
        # Calculate averages
        for platform, data in patterns['platform_performance'].items():
            if data['posts'] > 0:
                data['avg_engagement_rate'] = data['total_engagement_rate'] / data['posts']
                data['avg_reach'] = data['total_reach'] / data['posts']
        
        for hour, data in patterns['posting_time_performance'].items():
            if data['posts'] > 0:
                data['avg_engagement_rate'] = data['total_engagement_rate'] / data['posts']
        
        for dow, data in patterns['day_of_week_performance'].items():
            if data['posts'] > 0:
                data['avg_engagement_rate'] = data['total_engagement_rate'] / data['posts']
        
        for hashtag, data in patterns['hashtag_performance'].items():
            if data['uses'] > 0:
                data['avg_engagement_rate'] = data['total_engagement_rate'] / data['uses']
        
        # Identify top patterns
        patterns['top_performing_times'] = sorted(
            patterns['posting_time_performance'].items(),
            key=lambda x: x[1].get('avg_engagement_rate', 0),
            reverse=True
        )[:5]
        
        patterns['top_performing_days'] = sorted(
            patterns['day_of_week_performance'].items(),
            key=lambda x: x[1].get('avg_engagement_rate', 0),
            reverse=True
        )[:3]
        
        patterns['top_hashtags'] = sorted(
            patterns['hashtag_performance'].items(),
            key=lambda x: x[1].get('avg_engagement_rate', 0),
            reverse=True
        )[:20]
        
        return patterns
    
    def _calculate_engagement_rate(self, analytics: Dict[str, Any]) -> float:
        """Calculate engagement rate from analytics data"""
        reach = analytics.get('reach', 0) or analytics.get('impressions', 0)
        if reach == 0:
            return 0.0
        
        engagement = (
            analytics.get('likes', 0) +
            analytics.get('comments', 0) +
            analytics.get('shares', 0) +
            analytics.get('saves', 0)
        )
        
        return (engagement / reach) * 100
    
    async def _update_persona_strategy(
        self,
        persona_id: int,
        adjustments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update persona configuration with optimizations"""
        # Fetch current persona
        current_persona = await self.db_manager.fetch_one(
            "SELECT * FROM personas WHERE id = $1",
            persona_id
        )
        
        if not current_persona:
            return {'status': 'error', 'message': 'Persona not found'}
        
        # Apply adjustments
        persona_config = json.loads(current_persona['configuration'])
        
        # Update configuration with adjustments
        for key, value in adjustments.items():
            if key in persona_config:
                if isinstance(persona_config[key], dict) and isinstance(value, dict):
                    persona_config[key].update(value)
                else:
                    persona_config[key] = value
        
        # Update in database
        await self.db_manager.execute(
            """
            UPDATE personas
            SET configuration = $1, updated_at = $2
            WHERE id = $3
            """,
            json.dumps(persona_config),
            datetime.utcnow(),
            persona_id
        )
        
        return {
            'status': 'success',
            'adjustments_applied': list(adjustments.keys())
        }
    
    async def _update_posting_schedule(
        self,
        client_id: str,
        schedule_adjustments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update posting schedule based on performance data"""
        # Store new schedule
        await self.db_manager.execute(
            """
            INSERT INTO posting_schedules
            (client_id, schedule_data, is_active, created_at)
            VALUES ($1, $2, true, $3)
            ON CONFLICT (client_id) WHERE is_active = true
            DO UPDATE SET schedule_data = $2, updated_at = $3
            """,
            client_id,
            json.dumps(schedule_adjustments),
            datetime.utcnow()
        )
        
        return {
            'status': 'success',
            'platforms_updated': list(schedule_adjustments.keys())
        }
    
    async def _update_content_themes(
        self,
        client_id: str,
        theme_adjustments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update content themes based on performance"""
        # Store updated themes
        await self.db_manager.execute(
            """
            INSERT INTO content_themes
            (client_id, theme_data, is_active, created_at)
            VALUES ($1, $2, true, $3)
            ON CONFLICT (client_id) WHERE is_active = true
            DO UPDATE SET theme_data = $2, updated_at = $3
            """,
            client_id,
            json.dumps(theme_adjustments),
            datetime.utcnow()
        )
        
        return {
            'status': 'success',
            'themes_updated': len(theme_adjustments.get('themes', []))
        }
    
    async def _update_platform_strategies(
        self,
        client_id: str,
        platform_adjustments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update platform-specific strategies"""
        # Store platform strategies
        for platform, strategy in platform_adjustments.items():
            await self.db_manager.execute(
                """
                INSERT INTO platform_strategies
                (client_id, platform, strategy_data, is_active, created_at)
                VALUES ($1, $2, $3, true, $4)
                ON CONFLICT (client_id, platform) WHERE is_active = true
                DO UPDATE SET strategy_data = $3, updated_at = $4
                """,
                client_id,
                platform,
                json.dumps(strategy),
                datetime.utcnow()
            )
        
        return {
            'status': 'success',
            'platforms_updated': list(platform_adjustments.keys())
        }


# Placeholder classes for modules to be implemented
class PersonaFactory:
    """Factory for creating AI-powered personas"""
    def __init__(self, encryption_manager, content_analyzer):
        self.encryption_manager = encryption_manager
        self.content_analyzer = content_analyzer
    
    async def create_persona(self, business_niche, target_audience, brand_voice, preferences):
        # Placeholder implementation
        return type('Persona', (), {
            'persona_type': 'universal_optimized',
            'to_dict': lambda: {
                'type': 'universal_optimized',
                'niche': business_niche.value,
                'audience': target_audience,
                'voice': brand_voice
            },
            'get_niche_adaptations': lambda: {'adaptations': []},
            'get_platform_strategies': lambda: {'strategies': {}},
            'get_key_traits': lambda: ['adaptive', 'data-driven', 'engaging', 'authentic', 'strategic']
        })()


class ViralContentEngine:
    """Engine for generating viral content"""
    def __init__(self, content_analyzer, encryption_manager):
        self.content_analyzer = content_analyzer
        self.encryption_manager = encryption_manager
    
    async def generate_viral_content(self, business_niche, target_audience, brand_voice, persona_config, platforms, preferences):
        # Placeholder implementation
        return type('ViralContent', (), {
            'base_content': 'AI-generated viral content optimized for engagement',
            'get_platform_variation': lambda p: type('Variation', (), {
                'content_text': f'Platform-optimized content for {p.value}',
                'content_format': type('Format', (), {'value': 'text'})(),
                'media_requirements': {},
                'call_to_action': 'Engage with us!',
                'character_count': 100,
                'hashtags': ['trending', 'viral', business_niche.value]
            })()
        })()
    
    async def generate_posting_schedule(self, platform, audience_data, content_type):
        # Placeholder implementation
        return {
            'optimal_times': ['09:00', '13:00', '18:00'],
            'optimal_days': ['Monday', 'Wednesday', 'Friday'],
            'frequency': 'daily'
        }


class PlatformPublisherFactory:
    """Factory for platform-specific publishers"""
    async def get_publisher(self, platform, credentials):
        # Placeholder implementation
        return type('Publisher', (), {
            'publish_content': lambda **kwargs: {
                'post_id': f'{platform.value}_123456',
                'post_url': f'https://{platform.value}.com/post/123456',
                'initial_metrics': {'views': 0, 'likes': 0}
            },
            'schedule_post': lambda **kwargs: {
                'scheduled_id': f'{platform.value}_scheduled_123456'
            },
            'get_bulk_analytics': lambda post_ids: {
                post_id: {
                    'reach': 1000,
                    'impressions': 1500,
                    'likes': 100,
                    'comments': 20,
                    'shares': 10,
                    'saves': 5,
                    'engagement': 135
                } for post_id in post_ids
            }
        })()


class OptimizationEngine:
    """Engine for strategy optimization"""
    def __init__(self, content_analyzer, db_manager):
        self.content_analyzer = content_analyzer
        self.db_manager = db_manager
    
    async def generate_recommendations(self, client_id, current_persona, performance_data, analytics_data):
        # Placeholder implementation
        return {
            'persona_adjustments': {
                'tone': 'more_conversational',
                'engagement_style': 'question_based'
            },
            'schedule_adjustments': {
                'instagram': {'times': ['09:00', '18:00'], 'days': ['Mon', 'Wed', 'Fri']},
                'linkedin': {'times': ['08:00', '12:00'], 'days': ['Tue', 'Thu']}
            },
            'theme_adjustments': {
                'themes': ['educational', 'behind_the_scenes', 'user_generated']
            },
            'platform_adjustments': {
                'instagram': {'focus': 'reels', 'hashtag_strategy': 'niche_specific'},
                'linkedin': {'focus': 'articles', 'engagement_strategy': 'thought_leadership'}
            },
            'expected_improvements': {
                'engagement_rate': '+25%',
                'reach': '+40%',
                'conversions': '+15%'
            }
        }