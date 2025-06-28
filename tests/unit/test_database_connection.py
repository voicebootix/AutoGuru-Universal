"""
Unit tests for PostgreSQL Connection Manager

Tests the database connection management including connection pooling,
retry logic, transaction management, and error handling.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from datetime import datetime
import asyncpg

from backend.database.connection import (
    PostgreSQLConnectionManager,
    get_db_manager,
    close_db_manager,
    DatabaseConnectionError,
    DatabasePoolError,
    DatabaseQueryError,
    DatabaseTransactionError,
)
from backend.config.database import DatabaseSettings


class TestDatabaseSettings:
    """Test database configuration settings"""
    
    def test_default_settings(self):
        """Test default database settings initialization"""
        with patch.dict('os.environ', {'POSTGRES_PASSWORD': 'test_password'}):
            settings = DatabaseSettings()
            assert settings.postgres_host == "localhost"
            assert settings.postgres_port == 5432
            assert settings.postgres_user == "autoguru"
            assert settings.postgres_password == "test_password"
            assert settings.postgres_db == "autoguru_universal"
            assert settings.pool_min_size == 10
            assert settings.pool_max_size == 20
    
    def test_settings_from_environment(self):
        """Test loading settings from environment variables"""
        env_vars = {
            'POSTGRES_HOST': 'db.example.com',
            'POSTGRES_PORT': '5433',
            'POSTGRES_USER': 'custom_user',
            'POSTGRES_PASSWORD': 'secure_password',
            'POSTGRES_DB': 'custom_db',
            'DB_POOL_MIN_SIZE': '5',
            'DB_POOL_MAX_SIZE': '15',
        }
        
        with patch.dict('os.environ', env_vars):
            settings = DatabaseSettings()
            assert settings.postgres_host == "db.example.com"
            assert settings.postgres_port == 5433
            assert settings.postgres_user == "custom_user"
            assert settings.postgres_db == "custom_db"
            assert settings.pool_min_size == 5
            assert settings.pool_max_size == 15
    
    def test_database_url_generation(self):
        """Test database URL generation"""
        with patch.dict('os.environ', {'POSTGRES_PASSWORD': 'test_pass'}):
            settings = DatabaseSettings()
            expected_url = "postgresql://autoguru:test_pass@localhost:5432/autoguru_universal"
            assert settings.database_url == expected_url
            assert settings.asyncpg_url == expected_url.replace("postgresql://", "postgres://")
    
    def test_ssl_context_generation(self):
        """Test SSL context configuration"""
        env_vars = {
            'POSTGRES_PASSWORD': 'test_pass',
            'DB_SSL_MODE': 'require',
            'DB_SSL_CERT_PATH': '/path/to/cert',
            'DB_SSL_KEY_PATH': '/path/to/key',
            'DB_SSL_CA_PATH': '/path/to/ca',
        }
        
        with patch.dict('os.environ', env_vars):
            settings = DatabaseSettings()
            ssl_context = settings.ssl_context
            assert ssl_context is not None
            assert ssl_context['ssl'] == 'require'
            assert ssl_context['ssl_cert'] == '/path/to/cert'
            assert ssl_context['ssl_key'] == '/path/to/key'
            assert ssl_context['ssl_ca'] == '/path/to/ca'


class TestPostgreSQLConnectionManager:
    """Test suite for PostgreSQLConnectionManager"""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock database settings"""
        settings = MagicMock(spec=DatabaseSettings)
        settings.postgres_host = "localhost"
        settings.postgres_port = 5432
        settings.postgres_user = "test_user"
        settings.postgres_password = "test_pass"
        settings.postgres_db = "test_db"
        settings.pool_min_size = 5
        settings.pool_max_size = 10
        settings.pool_max_queries = 50000
        settings.pool_max_inactive_connection_lifetime = 300.0
        settings.query_timeout = 30
        settings.statement_cache_size = 20
        settings.asyncpg_url = "postgres://test_user:test_pass@localhost:5432/test_db"
        settings.get_pool_config.return_value = {
            "min_size": 5,
            "max_size": 10,
            "max_queries": 50000,
            "max_inactive_connection_lifetime": 300.0,
            "command_timeout": 30,
            "statement_cache_size": 20,
        }
        return settings
    
    @pytest.fixture
    def manager(self, mock_settings):
        """Create connection manager with mock settings"""
        return PostgreSQLConnectionManager(settings=mock_settings)
    
    @pytest.mark.asyncio
    async def test_initialization(self, manager, mock_settings):
        """Test connection manager initialization"""
        assert manager.settings == mock_settings
        assert manager.pool is None
        assert not manager._is_initialized
        assert manager._connection_semaphore._value == 10  # pool_max_size
    
    @pytest.mark.asyncio
    async def test_initialize_pool_success(self, manager):
        """Test successful pool initialization"""
        mock_pool = AsyncMock()
        mock_pool.get_size.return_value = 10
        mock_pool.get_free_size.return_value = 10
        mock_pool.get_min_size.return_value = 5
        mock_pool.get_max_size.return_value = 10
        
        with patch('asyncpg.create_pool', return_value=mock_pool) as mock_create_pool:
            # Mock health check
            manager.health_check = AsyncMock(return_value={"status": "healthy"})
            
            await manager.initialize()
            
            assert manager.pool == mock_pool
            assert manager._is_initialized
            mock_create_pool.assert_called_once_with(
                dsn=manager.settings.asyncpg_url,
                **manager.settings.get_pool_config()
            )
            manager.health_check.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_pool_already_initialized(self, manager):
        """Test initialization when pool is already initialized"""
        manager._is_initialized = True
        manager.pool = AsyncMock()
        
        with patch('asyncpg.create_pool') as mock_create_pool:
            await manager.initialize()
            mock_create_pool.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_initialize_pool_failure(self, manager):
        """Test pool initialization failure"""
        with patch('asyncpg.create_pool', side_effect=Exception("Connection failed")):
            with pytest.raises(DatabasePoolError, match="Pool initialization failed"):
                await manager.initialize()
            
            assert manager.pool is None
            assert not manager._is_initialized
    
    @pytest.mark.asyncio
    async def test_close_pool(self, manager):
        """Test closing the connection pool"""
        mock_pool = AsyncMock()
        manager.pool = mock_pool
        manager._is_initialized = True
        
        await manager.close()
        
        mock_pool.close.assert_called_once()
        assert manager.pool is None
        assert not manager._is_initialized
    
    @pytest.mark.asyncio
    async def test_close_pool_not_initialized(self, manager):
        """Test closing when pool is not initialized"""
        await manager.close()  # Should not raise
    
    @pytest.mark.asyncio
    async def test_get_connection_success(self, manager):
        """Test getting a connection from the pool"""
        mock_connection = AsyncMock()
        mock_pool = AsyncMock()
        mock_pool.acquire.return_value = mock_connection
        manager.pool = mock_pool
        
        async with manager.get_connection() as conn:
            assert conn == mock_connection
        
        mock_pool.acquire.assert_called_once()
        mock_pool.release.assert_called_once_with(mock_connection)
    
    @pytest.mark.asyncio
    async def test_get_connection_pool_not_initialized(self, manager):
        """Test getting connection when pool is not initialized"""
        with pytest.raises(DatabaseConnectionError, match="Connection pool not initialized"):
            async with manager.get_connection() as conn:
                pass
    
    @pytest.mark.asyncio
    async def test_transaction_success(self, manager):
        """Test successful transaction execution"""
        mock_connection = AsyncMock()
        mock_transaction = AsyncMock()
        mock_connection.transaction.return_value = mock_transaction
        
        manager.get_connection = AsyncMock()
        manager.get_connection.return_value.__aenter__.return_value = mock_connection
        manager.get_connection.return_value.__aexit__.return_value = None
        
        async with manager.transaction() as conn:
            assert conn == mock_connection
        
        mock_connection.transaction.assert_called_once_with(isolation='read_committed')
        mock_transaction.start.assert_called_once()
        mock_transaction.commit.assert_called_once()
        mock_transaction.rollback.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(self, manager):
        """Test transaction rollback on error"""
        mock_connection = AsyncMock()
        mock_transaction = AsyncMock()
        mock_connection.transaction.return_value = mock_transaction
        
        manager.get_connection = AsyncMock()
        manager.get_connection.return_value.__aenter__.return_value = mock_connection
        manager.get_connection.return_value.__aexit__.return_value = None
        
        with pytest.raises(DatabaseTransactionError):
            async with manager.transaction() as conn:
                raise Exception("Test error")
        
        mock_transaction.start.assert_called_once()
        mock_transaction.commit.assert_not_called()
        mock_transaction.rollback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_query_success(self, manager):
        """Test successful query execution"""
        mock_connection = AsyncMock()
        mock_connection.execute.return_value = "INSERT 1"
        
        manager.get_connection = AsyncMock()
        manager.get_connection.return_value.__aenter__.return_value = mock_connection
        manager.get_connection.return_value.__aexit__.return_value = None
        
        result = await manager.execute("INSERT INTO test VALUES ($1)", 1)
        
        assert result == "INSERT 1"
        mock_connection.execute.assert_called_once_with(
            "INSERT INTO test VALUES ($1)", 1, timeout=30
        )
    
    @pytest.mark.asyncio
    async def test_execute_query_with_retry(self, manager):
        """Test query execution with retry on connection error"""
        mock_connection = AsyncMock()
        mock_connection.execute.side_effect = [
            asyncpg.PostgresConnectionError("Connection lost"),
            asyncpg.PostgresConnectionError("Connection lost"),
            "INSERT 1"  # Success on third try
        ]
        
        manager.get_connection = AsyncMock()
        manager.get_connection.return_value.__aenter__.return_value = mock_connection
        manager.get_connection.return_value.__aexit__.return_value = None
        
        result = await manager.execute("INSERT INTO test VALUES ($1)", 1)
        
        assert result == "INSERT 1"
        assert mock_connection.execute.call_count == 3
    
    @pytest.mark.asyncio
    async def test_fetch_all_success(self, manager):
        """Test fetching all results"""
        mock_records = [{"id": 1, "name": "test1"}, {"id": 2, "name": "test2"}]
        mock_connection = AsyncMock()
        mock_connection.fetch.return_value = mock_records
        
        manager.get_connection = AsyncMock()
        manager.get_connection.return_value.__aenter__.return_value = mock_connection
        manager.get_connection.return_value.__aexit__.return_value = None
        
        results = await manager.fetch_all("SELECT * FROM test")
        
        assert results == mock_records
        mock_connection.fetch.assert_called_once_with("SELECT * FROM test", timeout=30)
    
    @pytest.mark.asyncio
    async def test_fetch_one_success(self, manager):
        """Test fetching one result"""
        mock_record = {"id": 1, "name": "test"}
        mock_connection = AsyncMock()
        mock_connection.fetchrow.return_value = mock_record
        
        manager.get_connection = AsyncMock()
        manager.get_connection.return_value.__aenter__.return_value = mock_connection
        manager.get_connection.return_value.__aexit__.return_value = None
        
        result = await manager.fetch_one("SELECT * FROM test WHERE id = $1", 1)
        
        assert result == mock_record
        mock_connection.fetchrow.assert_called_once_with(
            "SELECT * FROM test WHERE id = $1", 1, timeout=30
        )
    
    @pytest.mark.asyncio
    async def test_fetch_value_success(self, manager):
        """Test fetching a single value"""
        mock_connection = AsyncMock()
        mock_connection.fetchval.return_value = 42
        
        manager.get_connection = AsyncMock()
        manager.get_connection.return_value.__aenter__.return_value = mock_connection
        manager.get_connection.return_value.__aexit__.return_value = None
        
        result = await manager.fetch_value("SELECT COUNT(*) FROM test")
        
        assert result == 42
        mock_connection.fetchval.assert_called_once_with(
            "SELECT COUNT(*) FROM test", column=0, timeout=30
        )
    
    @pytest.mark.asyncio
    async def test_execute_many_success(self, manager):
        """Test executing multiple queries"""
        mock_connection = AsyncMock()
        
        manager.get_connection = AsyncMock()
        manager.get_connection.return_value.__aenter__.return_value = mock_connection
        manager.get_connection.return_value.__aexit__.return_value = None
        
        args_list = [(1, "test1"), (2, "test2"), (3, "test3")]
        await manager.execute_many("INSERT INTO test VALUES ($1, $2)", args_list)
        
        mock_connection.executemany.assert_called_once_with(
            "INSERT INTO test VALUES ($1, $2)", args_list, timeout=30
        )
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, manager):
        """Test successful health check"""
        mock_connection = AsyncMock()
        mock_pool = AsyncMock()
        mock_pool.get_size.return_value = 10
        mock_pool.get_free_size.return_value = 8
        mock_pool.get_min_size.return_value = 5
        mock_pool.get_max_size.return_value = 10
        
        manager.pool = mock_pool
        manager.fetch_value = AsyncMock(return_value="PostgreSQL 14.5")
        
        health_status = await manager.health_check()
        
        assert health_status["status"] == "healthy"
        assert health_status["database_version"] == "PostgreSQL 14.5"
        assert health_status["connection_pool"]["size"] == 10
        assert health_status["connection_pool"]["free_size"] == 8
        assert "response_time_seconds" in health_status
        assert "timestamp" in health_status
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, manager):
        """Test health check failure"""
        manager.fetch_value = AsyncMock(side_effect=Exception("Database unreachable"))
        
        with pytest.raises(DatabaseConnectionError, match="Health check failed"):
            await manager.health_check()
    
    @pytest.mark.asyncio
    async def test_create_tables_success(self, manager):
        """Test creating tables from schema"""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100)
        );
        """
        
        manager.transaction = AsyncMock()
        manager.execute = AsyncMock()
        
        # Setup context manager mocks
        manager.transaction.return_value.__aenter__.return_value = AsyncMock()
        manager.transaction.return_value.__aexit__.return_value = None
        
        await manager.create_tables_if_not_exist(schema_sql)
        
        manager.execute.assert_called_once_with(schema_sql)
    
    def test_repr(self, manager):
        """Test string representation"""
        repr_str = repr(manager)
        assert "PostgreSQLConnectionManager" in repr_str
        assert "localhost:5432" in repr_str
        assert "test_db" in repr_str
        assert "initialized=False" in repr_str


class TestConnectionManagerSingleton:
    """Test singleton connection manager functions"""
    
    @pytest.mark.asyncio
    async def test_get_db_manager_creates_singleton(self):
        """Test that get_db_manager creates and returns singleton"""
        with patch('backend.database.connection._connection_manager', None):
            with patch('backend.database.connection.PostgreSQLConnectionManager') as MockManager:
                mock_instance = AsyncMock()
                MockManager.return_value = mock_instance
                
                manager = await get_db_manager()
                
                assert manager == mock_instance
                mock_instance.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_db_manager_returns_existing(self):
        """Test that get_db_manager returns existing singleton"""
        mock_manager = AsyncMock()
        
        with patch('backend.database.connection._connection_manager', mock_manager):
            manager = await get_db_manager()
            
            assert manager == mock_manager
            mock_manager.initialize.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_close_db_manager(self):
        """Test closing the singleton manager"""
        mock_manager = AsyncMock()
        
        with patch('backend.database.connection._connection_manager', mock_manager):
            await close_db_manager()
            
            mock_manager.close.assert_called_once()
            
        # Verify manager is set to None
        from backend.database.connection import _connection_manager
        assert _connection_manager is None


class TestErrorHandling:
    """Test error handling scenarios"""
    
    @pytest.mark.asyncio
    async def test_query_error_handling(self):
        """Test proper error handling for query failures"""
        manager = PostgreSQLConnectionManager()
        mock_connection = AsyncMock()
        mock_connection.execute.side_effect = asyncpg.PostgresSyntaxError("Syntax error")
        
        manager.get_connection = AsyncMock()
        manager.get_connection.return_value.__aenter__.return_value = mock_connection
        manager.get_connection.return_value.__aexit__.return_value = None
        
        with pytest.raises(DatabaseQueryError, match="Query execution failed"):
            await manager.execute("INVALID SQL")
    
    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test connection error handling"""
        manager = PostgreSQLConnectionManager()
        manager.pool = None
        
        with pytest.raises(DatabaseConnectionError, match="Connection pool not initialized"):
            async with manager.get_connection():
                pass
    
    @pytest.mark.asyncio
    async def test_transaction_error_handling(self):
        """Test transaction error handling"""
        manager = PostgreSQLConnectionManager()
        mock_connection = AsyncMock()
        mock_transaction = AsyncMock()
        mock_transaction.start.side_effect = Exception("Transaction start failed")
        mock_connection.transaction.return_value = mock_transaction
        
        manager.get_connection = AsyncMock()
        manager.get_connection.return_value.__aenter__.return_value = mock_connection
        manager.get_connection.return_value.__aexit__.return_value = None
        
        with pytest.raises(DatabaseTransactionError, match="Transaction failed"):
            async with manager.transaction():
                pass