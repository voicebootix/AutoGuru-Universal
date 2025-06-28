"""
PostgreSQL Connection Management for AutoGuru Universal

This module provides async PostgreSQL connection management that works universally
across all business niches. It includes connection pooling, retry logic, transaction
management, and comprehensive error handling using asyncpg.
"""

import asyncio
import logging
import json
from typing import Optional, Dict, Any, List, Union, TypeVar, Callable, AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
import asyncpg
from asyncpg import Pool, Connection, Record
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from backend.config.database import get_database_settings, DatabaseSettings

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar('T')


class DatabaseConnectionError(Exception):
    """Base exception for database connection errors"""
    pass


class DatabasePoolError(DatabaseConnectionError):
    """Exception for connection pool related errors"""
    pass


class DatabaseQueryError(DatabaseConnectionError):
    """Exception for query execution errors"""
    pass


class DatabaseTransactionError(DatabaseConnectionError):
    """Exception for transaction related errors"""
    pass


class PostgreSQLConnectionManager:
    """
    Async PostgreSQL connection manager with pooling and retry logic.
    Works universally across all business niches without hardcoded logic.
    """
    
    def __init__(self, settings: Optional[DatabaseSettings] = None):
        """
        Initialize the connection manager.
        
        Args:
            settings: Database settings instance (uses default if not provided)
        """
        self.settings = settings or get_database_settings()
        self.pool: Optional[Pool] = None
        self._is_initialized = False
        self._connection_semaphore = asyncio.Semaphore(self.settings.pool_max_size)
        
    async def initialize(self) -> None:
        """
        Initialize the connection pool.
        
        Raises:
            DatabasePoolError: If pool initialization fails
        """
        if self._is_initialized:
            logger.warning("Connection pool already initialized")
            return
            
        try:
            logger.info(f"Initializing PostgreSQL connection pool to {self.settings.postgres_host}:{self.settings.postgres_port}")
            
            # Create connection pool with configuration
            self.pool = await asyncpg.create_pool(
                dsn=self.settings.asyncpg_url,
                **self.settings.get_pool_config()
            )
            
            # Test the connection
            await self.health_check()
            
            self._is_initialized = True
            logger.info("PostgreSQL connection pool initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {str(e)}")
            raise DatabasePoolError(f"Pool initialization failed: {str(e)}") from e
    
    async def close(self) -> None:
        """
        Close the connection pool and cleanup resources.
        """
        if not self.pool:
            logger.warning("No connection pool to close")
            return
            
        try:
            logger.info("Closing PostgreSQL connection pool")
            await self.pool.close()
            self.pool = None
            self._is_initialized = False
            logger.info("PostgreSQL connection pool closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing connection pool: {str(e)}")
            raise DatabasePoolError(f"Failed to close pool: {str(e)}") from e
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[Connection, None]:
        """
        Get a connection from the pool with automatic cleanup.
        
        Yields:
            asyncpg.Connection: Database connection
            
        Raises:
            DatabaseConnectionError: If unable to acquire connection
        """
        if not self.pool:
            raise DatabaseConnectionError("Connection pool not initialized")
            
        connection = None
        try:
            # Acquire connection with semaphore to prevent overload
            async with self._connection_semaphore:
                connection = await self.pool.acquire()
                yield connection
                
        except Exception as e:
            logger.error(f"Failed to acquire connection: {str(e)}")
            raise DatabaseConnectionError(f"Connection acquisition failed: {str(e)}") from e
            
        finally:
            if connection:
                await self.pool.release(connection)
    
    @asynccontextmanager
    async def transaction(self, isolation_level: str = 'read_committed') -> AsyncGenerator[Connection, None]:
        """
        Create a database transaction with automatic rollback on error.
        
        Args:
            isolation_level: Transaction isolation level
            
        Yields:
            asyncpg.Connection: Connection with active transaction
            
        Raises:
            DatabaseTransactionError: If transaction fails
        """
        async with self.get_connection() as connection:
            transaction = connection.transaction(isolation=isolation_level)
            
            try:
                await transaction.start()
                yield connection
                await transaction.commit()
                
            except Exception as e:
                logger.error(f"Transaction failed, rolling back: {str(e)}")
                await transaction.rollback()
                raise DatabaseTransactionError(f"Transaction failed: {str(e)}") from e
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((asyncpg.PostgresConnectionError, asyncpg.InterfaceError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def execute(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> str:
        """
        Execute a query without returning results (INSERT, UPDATE, DELETE).
        
        Args:
            query: SQL query to execute
            *args: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            Query status string
            
        Raises:
            DatabaseQueryError: If query execution fails
        """
        timeout = timeout or self.settings.query_timeout
        
        try:
            async with self.get_connection() as connection:
                status = await connection.execute(query, *args, timeout=timeout)
                logger.debug(f"Query executed successfully: {status}")
                return status
                
        except asyncpg.PostgresError as e:
            logger.error(f"PostgreSQL error: {str(e)}")
            raise DatabaseQueryError(f"Query execution failed: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error during query execution: {str(e)}")
            raise DatabaseQueryError(f"Unexpected error: {str(e)}") from e
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((asyncpg.PostgresConnectionError, asyncpg.InterfaceError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def fetch_all(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> List[Record]:
        """
        Execute a query and fetch all results.
        
        Args:
            query: SQL query to execute
            *args: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            List of query results
            
        Raises:
            DatabaseQueryError: If query execution fails
        """
        timeout = timeout or self.settings.query_timeout
        
        try:
            async with self.get_connection() as connection:
                results = await connection.fetch(query, *args, timeout=timeout)
                logger.debug(f"Fetched {len(results)} rows")
                return results
                
        except asyncpg.PostgresError as e:
            logger.error(f"PostgreSQL error: {str(e)}")
            raise DatabaseQueryError(f"Query fetch failed: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error during fetch: {str(e)}")
            raise DatabaseQueryError(f"Unexpected error: {str(e)}") from e
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((asyncpg.PostgresConnectionError, asyncpg.InterfaceError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def fetch_one(
        self,
        query: str,
        *args,
        timeout: Optional[float] = None
    ) -> Optional[Record]:
        """
        Execute a query and fetch one result.
        
        Args:
            query: SQL query to execute
            *args: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            Single query result or None
            
        Raises:
            DatabaseQueryError: If query execution fails
        """
        timeout = timeout or self.settings.query_timeout
        
        try:
            async with self.get_connection() as connection:
                result = await connection.fetchrow(query, *args, timeout=timeout)
                logger.debug(f"Fetched {'1 row' if result else 'no rows'}")
                return result
                
        except asyncpg.PostgresError as e:
            logger.error(f"PostgreSQL error: {str(e)}")
            raise DatabaseQueryError(f"Query fetch failed: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error during fetch: {str(e)}")
            raise DatabaseQueryError(f"Unexpected error: {str(e)}") from e
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((asyncpg.PostgresConnectionError, asyncpg.InterfaceError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def fetch_value(
        self,
        query: str,
        *args,
        column: int = 0,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Execute a query and fetch a single value.
        
        Args:
            query: SQL query to execute
            *args: Query parameters
            column: Column index to fetch
            timeout: Query timeout in seconds
            
        Returns:
            Single value from query result
            
        Raises:
            DatabaseQueryError: If query execution fails
        """
        timeout = timeout or self.settings.query_timeout
        
        try:
            async with self.get_connection() as connection:
                value = await connection.fetchval(query, *args, column=column, timeout=timeout)
                logger.debug(f"Fetched value: {value}")
                return value
                
        except asyncpg.PostgresError as e:
            logger.error(f"PostgreSQL error: {str(e)}")
            raise DatabaseQueryError(f"Query fetch failed: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error during fetch: {str(e)}")
            raise DatabaseQueryError(f"Unexpected error: {str(e)}") from e
    
    async def execute_many(
        self,
        query: str,
        args_list: List[tuple],
        timeout: Optional[float] = None
    ) -> None:
        """
        Execute the same query multiple times with different parameters.
        
        Args:
            query: SQL query to execute
            args_list: List of parameter tuples
            timeout: Query timeout in seconds
            
        Raises:
            DatabaseQueryError: If execution fails
        """
        timeout = timeout or self.settings.query_timeout
        
        try:
            async with self.get_connection() as connection:
                await connection.executemany(query, args_list, timeout=timeout)
                logger.debug(f"Executed query {len(args_list)} times")
                
        except asyncpg.PostgresError as e:
            logger.error(f"PostgreSQL error in batch execution: {str(e)}")
            raise DatabaseQueryError(f"Batch execution failed: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error during batch execution: {str(e)}")
            raise DatabaseQueryError(f"Unexpected error: {str(e)}") from e
    
    async def copy_to_table(
        self,
        table_name: str,
        source: Union[str, bytes, asyncio.StreamReader],
        columns: Optional[List[str]] = None,
        format: str = 'csv',
        delimiter: str = ',',
        null: str = ''
    ) -> int:
        """
        Bulk copy data to a table using COPY command.
        
        Args:
            table_name: Target table name
            source: Data source (file path, bytes, or stream)
            columns: List of column names (all if None)
            format: Data format (csv, text, binary)
            delimiter: Field delimiter
            null: Null value representation
            
        Returns:
            Number of rows copied
            
        Raises:
            DatabaseQueryError: If copy fails
        """
        try:
            async with self.get_connection() as connection:
                if columns:
                    columns_str = f"({', '.join(columns)})"
                else:
                    columns_str = ""
                    
                copy_query = f"COPY {table_name}{columns_str} FROM STDIN WITH (FORMAT {format}, DELIMITER '{delimiter}', NULL '{null}')"
                
                if isinstance(source, str):
                    with open(source, 'rb') as f:
                        result = await connection.copy_to_table(table_name, source=f.read())
                elif isinstance(source, bytes):
                    result = await connection.copy_to_table(table_name, source=source)
                else:
                    result = await connection.copy_to_table(table_name, source=source)
                    
                logger.info(f"Copied {result} rows to {table_name}")
                return result
                
        except Exception as e:
            logger.error(f"Failed to copy data to {table_name}: {str(e)}")
            raise DatabaseQueryError(f"Copy operation failed: {str(e)}") from e
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the database connection.
        
        Returns:
            Dictionary with health check results
            
        Raises:
            DatabaseConnectionError: If health check fails
        """
        try:
            start_time = datetime.utcnow()
            
            # Test basic connectivity
            version = await self.fetch_value("SELECT version()")
            
            # Get connection pool stats
            pool_stats = {}
            if self.pool:
                pool_stats = {
                    "size": self.pool.get_size(),
                    "free_size": self.pool.get_free_size(),
                    "min_size": self.pool.get_min_size(),
                    "max_size": self.pool.get_max_size(),
                }
            
            # Calculate response time
            response_time = (datetime.utcnow() - start_time).total_seconds()
            
            health_status = {
                "status": "healthy",
                "database_version": version,
                "connection_pool": pool_stats,
                "response_time_seconds": response_time,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            logger.info(f"Health check passed: {json.dumps(health_status)}")
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            raise DatabaseConnectionError(f"Health check failed: {str(e)}") from e
    
    async def create_tables_if_not_exist(self, schema_sql: str) -> None:
        """
        Create database tables from schema SQL if they don't exist.
        
        Args:
            schema_sql: SQL statements to create tables
            
        Raises:
            DatabaseQueryError: If table creation fails
        """
        try:
            async with self.transaction():
                await self.execute(schema_sql)
                logger.info("Database tables created/verified successfully")
                
        except Exception as e:
            logger.error(f"Failed to create tables: {str(e)}")
            raise DatabaseQueryError(f"Table creation failed: {str(e)}") from e
    
    def __repr__(self) -> str:
        """String representation of the connection manager"""
        return (
            f"<PostgreSQLConnectionManager "
            f"host={self.settings.postgres_host}:{self.settings.postgres_port} "
            f"database={self.settings.postgres_db} "
            f"initialized={self._is_initialized}>"
        )


# Singleton instance for easy access
_connection_manager: Optional[PostgreSQLConnectionManager] = None


async def get_db_manager() -> PostgreSQLConnectionManager:
    """
    Get the singleton database connection manager instance.
    
    Returns:
        PostgreSQLConnectionManager instance
        
    Raises:
        DatabaseConnectionError: If manager not initialized
    """
    global _connection_manager
    
    if not _connection_manager:
        _connection_manager = PostgreSQLConnectionManager()
        await _connection_manager.initialize()
        
    return _connection_manager


async def close_db_manager() -> None:
    """Close the singleton database connection manager."""
    global _connection_manager
    
    if _connection_manager:
        await _connection_manager.close()
        _connection_manager = None