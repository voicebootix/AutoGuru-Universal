# AutoGuru Universal - Docker Setup Guide

This guide will help you set up AutoGuru Universal using Docker for local development.

## Quick Start

### Prerequisites

1. **Docker Desktop** installed and running
   - [Download for Windows](https://docs.docker.com/desktop/install/windows/)
   - [Download for macOS](https://docs.docker.com/desktop/install/mac/)
   - [Download for Linux](https://docs.docker.com/desktop/install/linux/)

2. **Docker Compose** (usually included with Docker Desktop)

### One-Command Setup

**Linux/macOS:**
```bash
chmod +x docker-manage.sh
./docker-manage.sh start
```

**Windows PowerShell:**
```powershell
.\docker-manage.ps1 start
```

## Services Overview

The Docker Compose setup includes:

### Core Services
- **PostgreSQL 15** - Main database
  - Port: 5432
  - Database: `autoguru_universal`
  - User: `autoguru`
  - Password: `password`

- **Redis 7** - Caching and message broker
  - Port: 6379
  - Persistence enabled
  - Memory limit: 256MB

### Optional Management Tools
- **pgAdmin 4** - PostgreSQL web interface
  - Port: 5050
  - Email: `admin@autoguru.com`
  - Password: `admin123`

- **Redis Commander** - Redis web interface
  - Port: 8081
  - No authentication required

## Management Commands

### Using the Management Scripts

**Linux/macOS:**
```bash
# Start core services
./docker-manage.sh start

# Start with management tools
./docker-manage.sh start-tools

# Stop all services
./docker-manage.sh stop

# Show logs
./docker-manage.sh logs

# Show service status
./docker-manage.sh status

# Reset database (removes all data)
./docker-manage.sh reset-db

# Clean up everything
./docker-manage.sh cleanup
```

**Windows PowerShell:**
```powershell
# Start core services
.\docker-manage.ps1 start

# Start with management tools
.\docker-manage.ps1 start-tools

# Stop all services
.\docker-manage.ps1 stop

# Show logs
.\docker-manage.ps1 logs

# Show service status
.\docker-manage.ps1 status

# Reset database (removes all data)
.\docker-manage.ps1 reset-db

# Clean up everything
.\docker-manage.ps1 cleanup
```

### Using Docker Compose Directly

```bash
# Start core services
docker-compose up -d postgres redis

# Start with management tools
docker-compose --profile tools up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f postgres

# Restart services
docker-compose restart

# Remove everything including volumes
docker-compose down -v
```

## Configuration Details

### PostgreSQL Configuration

- **Image**: `postgres:15-alpine`
- **Database**: `autoguru_universal`
- **User**: `autoguru`
- **Password**: `password`
- **Extensions**: UUID, pg_trgm, btree_gin
- **Schemas**: public, analytics, content, social
- **Health Check**: Enabled with 30s intervals

### Redis Configuration

- **Image**: `redis:7-alpine`
- **Persistence**: AOF (Append Only File) enabled
- **Memory Limit**: 256MB
- **Eviction Policy**: LRU (Least Recently Used)
- **Health Check**: Enabled with 30s intervals

### Network Configuration

- **Network**: `autoguru_network`
- **Subnet**: `172.20.0.0/16`
- **Driver**: Bridge

## Environment Variables

The services use the following environment variables (configured in `docker-compose.yml`):

### PostgreSQL
```env
POSTGRES_DB=autoguru_universal
POSTGRES_USER=autoguru
POSTGRES_PASSWORD=password
POSTGRES_INITDB_ARGS=--encoding=UTF-8 --lc-collate=C --lc-ctype=C
```

### Redis
```env
# Configured via redis.conf file
# Memory limit: 256MB
# Persistence: AOF enabled
```

## Data Persistence

### Volumes
- `postgres_data` - PostgreSQL data directory
- `redis_data` - Redis data directory
- `pgadmin_data` - pgAdmin configuration and data

### Backup and Restore

**PostgreSQL Backup:**
```bash
docker exec autoguru_postgres pg_dump -U autoguru autoguru_universal > backup.sql
```

**PostgreSQL Restore:**
```bash
docker exec -i autoguru_postgres psql -U autoguru autoguru_universal < backup.sql
```

**Redis Backup:**
```bash
docker exec autoguru_redis redis-cli BGSAVE
docker cp autoguru_redis:/data/dump.rdb ./redis_backup.rdb
```

## Troubleshooting

### Common Issues

**1. Port Already in Use**
```bash
# Check what's using the port
lsof -i :5432  # macOS/Linux
netstat -ano | findstr :5432  # Windows

# Stop conflicting services or change ports in docker-compose.yml
```

**2. Permission Denied**
```bash
# Make script executable (Linux/macOS)
chmod +x docker-manage.sh

# Run PowerShell as Administrator (Windows)
```

**3. Docker Not Running**
```bash
# Start Docker Desktop
# Wait for it to fully start
# Try the command again
```

**4. Database Connection Issues**
```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# Check logs
docker-compose logs postgres

# Test connection
docker exec autoguru_postgres pg_isready -U autoguru -d autoguru_universal
```

**5. Redis Connection Issues**
```bash
# Check if Redis is running
docker-compose ps redis

# Check logs
docker-compose logs redis

# Test connection
docker exec autoguru_redis redis-cli ping
```

### Logs and Debugging

**View all logs:**
```bash
docker-compose logs -f
```

**View specific service logs:**
```bash
docker-compose logs -f postgres
docker-compose logs -f redis
```

**Access service shells:**
```bash
# PostgreSQL shell
docker exec -it autoguru_postgres psql -U autoguru -d autoguru_universal

# Redis shell
docker exec -it autoguru_redis redis-cli
```

## Performance Optimization

### PostgreSQL
- Connection pooling recommended for production
- Consider increasing shared_buffers and work_mem
- Monitor slow queries with pg_stat_statements

### Redis
- Current memory limit: 256MB (adjust based on usage)
- AOF persistence for data durability
- Consider Redis Cluster for high availability

## Security Considerations

### Development Environment
- Default passwords are used for simplicity
- Services are exposed on localhost only
- No SSL/TLS encryption (add for production)

### Production Recommendations
- Use strong, unique passwords
- Enable SSL/TLS encryption
- Restrict network access
- Regular security updates
- Backup and disaster recovery

## Integration with AutoGuru Universal

### Environment Variables
Update your `.env` file to use the Docker services:

```env
# Database
DATABASE_URL=postgresql://autoguru:password@localhost:5432/autoguru_universal

# Redis/Celery
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

### Application Startup
After starting the Docker services:

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run database migrations:**
   ```bash
   python backend/main.py --migrate
   ```

3. **Start the application:**
   ```bash
   python backend/main.py
   ```

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the service logs
3. Ensure Docker Desktop is running
4. Verify ports are not in use
5. Check the project documentation
6. Create an issue in the project repository

## Next Steps

After setting up Docker:

1. **Set up your environment variables** (see `ENVIRONMENT_SETUP.md`)
2. **Install Python dependencies**
3. **Run the application**
4. **Access management tools** (if started with tools)
5. **Begin development!** 