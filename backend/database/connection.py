"""
Database connection management for AutoGuru Universal.

This module provides async database connection handling using SQLAlchemy
with PostgreSQL. It supports connection pooling and session management
for optimal performance across all business niches.
"""

import logging
from typing import AsyncGenerator, Optional, Dict, Any
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool, AsyncAdaptedQueuePool

from ..config.settings import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Create declarative base for models
Base = declarative_base()

# Global engine instance
_engine: Optional[AsyncEngine] = None
_async_session_maker: Optional[async_sessionmaker] = None


def get_database_url() -> str:
    """
    Get the database URL from settings.
    
    Returns:
        str: PostgreSQL connection URL
    """
    return str(settings.database.postgres_dsn)


async def create_db_engine() -> AsyncEngine:
    """
    Create an async database engine with connection pooling.
    
    Returns:
        AsyncEngine: Configured async engine instance
    """
    global _engine
    
    if _engine is not None:
        return _engine
    
    # Configure engine with appropriate pool settings
    engine_config = {
        "echo": settings.debug,  # SQL logging in debug mode
        "echo_pool": settings.debug,
        "max_overflow": settings.database.max_overflow,
        "pool_size": settings.database.pool_size,
        "pool_timeout": settings.database.pool_timeout,
        "pool_recycle": settings.database.pool_recycle,
    }
    
    # Use NullPool for testing to avoid connection issues
    if settings.environment.value == "testing":
        engine_config["poolclass"] = NullPool
    else:
        engine_config["poolclass"] = AsyncAdaptedQueuePool
    
    _engine = create_async_engine(
        get_database_url(),
        **engine_config
    )
    
    logger.info("Database engine created successfully")
    return _engine


async def get_session_maker() -> async_sessionmaker:
    """
    Get or create an async session maker.
    
    Returns:
        async_sessionmaker: Session factory
    """
    global _async_session_maker
    
    if _async_session_maker is not None:
        return _async_session_maker
    
    engine = await create_db_engine()
    _async_session_maker = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )
    
    return _async_session_maker


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session.
    
    This is the main function to use in FastAPI dependencies.
    
    Yields:
        AsyncSession: Database session
    """
    session_maker = await get_session_maker()
    async with session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {str(e)}")
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions.
    
    Use this for non-FastAPI contexts where you need a database session.
    
    Yields:
        AsyncSession: Database session
    """
    session_maker = await get_session_maker()
    async with session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_database() -> None:
    """
    Initialize the database by creating all tables.
    
    This should be called during application startup.
    """
    engine = await create_db_engine()
    
    try:
        async with engine.begin() as conn:
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise


async def close_database() -> None:
    """
    Close database connections.
    
    This should be called during application shutdown.
    """
    global _engine, _async_session_maker
    
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _async_session_maker = None
        logger.info("Database connections closed")


async def check_database_connection() -> bool:
    """
    Check if database connection is working.
    
    Returns:
        bool: True if connection is successful
    """
    try:
        engine = await create_db_engine()
        async with engine.connect() as conn:
            await conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {str(e)}")
        return False


# Health check query for monitoring
async def health_check() -> Dict[str, Any]:
    """
    Perform a database health check.
    
    Returns:
        Dict with health status information
    """
    try:
        engine = await create_db_engine()
        async with engine.connect() as conn:
            result = await conn.execute("SELECT version();")
            version = result.scalar()
            
        pool = engine.pool
        return {
            "status": "healthy",
            "database": "postgresql",
            "version": version,
            "pool_size": pool.size(),
            "pool_checked_in": pool.checkedin(),
            "pool_checked_out": pool.checkedout(),
            "pool_overflow": pool.overflow(),
            "pool_total": pool.total()
        }
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }