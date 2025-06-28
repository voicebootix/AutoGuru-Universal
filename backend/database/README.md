# Database Module - AutoGuru Universal

This module provides PostgreSQL database connectivity and management for AutoGuru Universal, working universally across all business niches.

## Features

- **Async Operations**: Built on asyncpg for high-performance async database operations
- **Connection Pooling**: Automatic connection pool management for scalability
- **Retry Logic**: Automatic retry on connection failures with exponential backoff
- **Transaction Management**: Easy-to-use transaction context managers with automatic rollback
- **Environment-based Configuration**: Flexible configuration through environment variables
- **Health Checks**: Built-in health check functionality for monitoring
- **Type Safety**: Full type hints for better IDE support

## Setup

1. **Install PostgreSQL**: Ensure PostgreSQL 12+ is installed and running

2. **Create Database**:
   ```sql
   CREATE DATABASE autoguru_universal;
   CREATE USER autoguru WITH ENCRYPTED PASSWORD 'your_secure_password';
   GRANT ALL PRIVILEGES ON DATABASE autoguru_universal TO autoguru;
   ```

3. **Configure Environment**: Copy `.env.example` to `.env` and update with your database credentials:
   ```bash
   cp .env.example .env
   # Edit .env with your database credentials
   ```

## Usage

### Basic Usage

```python
from backend.database import get_db_manager

# Get the database manager (singleton)
db = await get_db_manager()

# Execute a query
await db.execute("INSERT INTO users (name, email) VALUES ($1, $2)", "John Doe", "john@example.com")

# Fetch results
users = await db.fetch_all("SELECT * FROM users WHERE active = $1", True)

# Fetch single row
user = await db.fetch_one("SELECT * FROM users WHERE id = $1", user_id)

# Fetch single value
count = await db.fetch_value("SELECT COUNT(*) FROM users")
```

### Using Transactions

```python
from backend.database import get_db_manager

db = await get_db_manager()

# Transaction with automatic rollback on error
async with db.transaction():
    await db.execute("INSERT INTO accounts (name) VALUES ($1)", "Account 1")
    await db.execute("INSERT INTO users (account_id, name) VALUES ($1, $2)", 1, "User 1")
    # If any query fails, all changes are rolled back
```

### Direct Connection Access

```python
from backend.database import get_db_manager

db = await get_db_manager()

# Get a connection from the pool
async with db.get_connection() as conn:
    # Use asyncpg connection directly
    await conn.execute("CREATE TEMP TABLE temp_data (id INT)")
    await conn.copy_records_to_table('temp_data', records=[(1,), (2,), (3,)])
```

### Health Checks

```python
from backend.database import get_db_manager

db = await get_db_manager()

# Perform health check
health_status = await db.health_check()
print(health_status)
# Output: {
#     "status": "healthy",
#     "database_version": "PostgreSQL 14.5",
#     "connection_pool": {
#         "size": 10,
#         "free_size": 8,
#         "min_size": 5,
#         "max_size": 20
#     },
#     "response_time_seconds": 0.023,
#     "timestamp": "2024-01-15T12:00:00"
# }
```

### Bulk Operations

```python
# Execute many queries with different parameters
data = [(1, "User 1"), (2, "User 2"), (3, "User 3")]
await db.execute_many("INSERT INTO users (id, name) VALUES ($1, $2)", data)

# Bulk copy from CSV
await db.copy_to_table(
    table_name="users",
    source="/path/to/users.csv",
    columns=["id", "name", "email"],
    format="csv",
    delimiter=","
)
```

## Configuration Options

All configuration is done through environment variables:

- `POSTGRES_HOST`: Database host (default: localhost)
- `POSTGRES_PORT`: Database port (default: 5432)
- `POSTGRES_USER`: Database username (default: autoguru)
- `POSTGRES_PASSWORD`: Database password (required)
- `POSTGRES_DB`: Database name (default: autoguru_universal)
- `DB_POOL_MIN_SIZE`: Minimum pool size (default: 10)
- `DB_POOL_MAX_SIZE`: Maximum pool size (default: 20)
- `DB_RETRY_ATTEMPTS`: Number of retry attempts (default: 3)
- `DB_QUERY_TIMEOUT`: Query timeout in seconds (default: 30)

For SSL connections:
- `DB_SSL_MODE`: SSL mode (disable, require, verify-ca, verify-full)
- `DB_SSL_CERT_PATH`: Path to client certificate
- `DB_SSL_KEY_PATH`: Path to client key
- `DB_SSL_CA_PATH`: Path to CA certificate

## Error Handling

The module provides specific exception types for different error scenarios:

- `DatabaseConnectionError`: Connection-related errors
- `DatabasePoolError`: Connection pool errors
- `DatabaseQueryError`: Query execution errors
- `DatabaseTransactionError`: Transaction-related errors

```python
from backend.database import get_db_manager, DatabaseQueryError

try:
    db = await get_db_manager()
    result = await db.fetch_one("SELECT * FROM users WHERE id = $1", user_id)
except DatabaseQueryError as e:
    logger.error(f"Query failed: {e}")
    # Handle query error
```

## Testing

Run the unit tests:

```bash
pytest tests/unit/test_database_connection.py -v
```

## Best Practices

1. **Always use parameterized queries** to prevent SQL injection
2. **Use transactions** for multi-step operations that should be atomic
3. **Handle exceptions** appropriately in your application code
4. **Monitor pool usage** through health checks in production
5. **Set appropriate timeouts** for long-running queries
6. **Use bulk operations** for inserting/updating many records

## Universal Design

This module is designed to work universally across all business niches supported by AutoGuru Universal:
- No hardcoded business logic
- Flexible schema support
- Works with any PostgreSQL database structure
- Suitable for educational, consulting, fitness, creative, and all other business types