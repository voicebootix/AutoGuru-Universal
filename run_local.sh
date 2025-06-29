#!/bin/bash

# AutoGuru Universal - Local Development Startup Script
# This script provides a complete startup solution for local development

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  AutoGuru Universal${NC}"
    echo -e "${BLUE}  Local Development${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if port is in use
port_in_use() {
    lsof -i :$1 >/dev/null 2>&1
}

# Function to wait for service health
wait_for_service() {
    local service_name=$1
    local port=$2
    local max_attempts=30
    local attempt=1
    
    print_step "Waiting for $service_name to be ready on port $port..."
    
    while [ $attempt -le $max_attempts ]; do
        if port_in_use $port; then
            print_success "$service_name is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "$service_name failed to start within $((max_attempts * 2)) seconds"
    return 1
}

# Function to check Docker services health
wait_for_docker_health() {
    local max_attempts=30
    local attempt=1
    
    print_step "Waiting for Docker services to be healthy..."
    
    while [ $attempt -le $max_attempts ]; do
        if docker-compose ps | grep -q "healthy"; then
            print_success "Docker services are healthy!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "Docker services failed to become healthy within $((max_attempts * 2)) seconds"
    return 1
}

# Function to check database connection
check_database_connection() {
    print_step "Checking database connection..."
    
    python3 -c "
import asyncio
import sys
from backend.database.connection import check_database_connection

async def test_db():
    try:
        is_connected = await check_database_connection()
        if is_connected:
            print('✅ Database connection successful')
            sys.exit(0)
        else:
            print('❌ Database connection failed')
            sys.exit(1)
    except Exception as e:
        print(f'❌ Database connection error: {e}')
        sys.exit(1)

asyncio.run(test_db())
"
}

# Function to initialize database
initialize_database() {
    print_step "Initializing database..."
    
    python3 -c "
import asyncio
import sys
from backend.database.connection import init_database

async def setup_db():
    try:
        await init_database()
        print('✅ Database initialized successfully')
        sys.exit(0)
    except Exception as e:
        print(f'❌ Database initialization failed: {e}')
        sys.exit(1)

asyncio.run(setup_db())
"
}

# Function to start Celery worker
start_celery_worker() {
    print_step "Starting Celery worker..."
    
    # Check if Celery is already running
    if pgrep -f "celery.*worker" > /dev/null; then
        print_warning "Celery worker is already running"
        return 0
    fi
    
    # Start Celery worker in background
    celery -A backend.tasks.content_generation worker \
        --loglevel=info \
        --concurrency=2 \
        --queues=default,content_analysis,content_generation,publishing,analytics,optimization \
        --hostname=worker@%h \
        --detach
    
    sleep 3
    
    # Check if worker started successfully
    if pgrep -f "celery.*worker" > /dev/null; then
        print_success "Celery worker started successfully"
    else
        print_error "Failed to start Celery worker"
        return 1
    fi
}

# Function to start FastAPI server
start_fastapi_server() {
    print_step "Starting FastAPI server..."
    
    # Check if server is already running
    if port_in_use 8000; then
        print_warning "FastAPI server is already running on port 8000"
        return 0
    fi
    
    # Start FastAPI server in background
    uvicorn backend.main:app \
        --reload \
        --host 0.0.0.0 \
        --port 8000 \
        --log-level info \
        --access-log \
        --workers 1 \
        --detach
    
    # Wait for server to start
    wait_for_service "FastAPI server" 8000
}

# Function to show final status
show_final_status() {
    echo ""
    print_header
    print_success "AutoGuru Universal is running!"
    echo ""
    echo -e "${CYAN}Services:${NC}"
    echo -e "  🌐 API Server:     ${GREEN}http://localhost:8000${NC}"
    echo -e "  📖 API Docs:       ${GREEN}http://localhost:8000/docs${NC}"
    echo -e "  🎨 Frontend:       ${GREEN}file://$(pwd)/frontend/index.html${NC}"
    echo -e "  🗄️  PostgreSQL:     ${GREEN}localhost:5432${NC}"
    echo -e "  🔴 Redis:          ${GREEN}localhost:6379${NC}"
    echo -e "  ⚙️  Celery Worker:  ${GREEN}Running${NC}"
    echo ""
    echo -e "${CYAN}Management:${NC}"
    echo -e "  📊 pgAdmin:        ${GREEN}http://localhost:5050${NC} (admin@autoguru.com / admin123)"
    echo -e "  🔧 Redis Commander: ${GREEN}http://localhost:8081${NC}"
    echo ""
    echo -e "${CYAN}Commands:${NC}"
    echo -e "  🧪 Test API:       ${YELLOW}python test_integration.py${NC}"
    echo -e "  📋 View Logs:      ${YELLOW}docker-compose logs -f${NC}"
    echo -e "  🛑 Stop Services:  ${YELLOW}docker-compose down${NC}"
    echo -e "  🔄 Restart:        ${YELLOW}./run_local.sh${NC}"
    echo ""
    echo -e "${PURPLE}Happy coding! 🚀${NC}"
}

# Function to cleanup on exit
cleanup() {
    print_warning "Shutting down AutoGuru Universal..."
    
    # Stop FastAPI server
    pkill -f "uvicorn.*backend.main:app" 2>/dev/null || true
    
    # Stop Celery worker
    pkill -f "celery.*worker" 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# Set up trap for cleanup on script exit
trap cleanup EXIT

# Main startup function
main() {
    print_header
    echo ""
    
    # Check prerequisites
    print_step "Checking prerequisites..."
    
    # Check if .env exists
    if [ ! -f .env ]; then
        print_error ".env file not found. Please run: python setup_env.py"
        exit 1
    fi
    
    # Check if Docker is running
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker Desktop."
        exit 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker Desktop."
        exit 1
    fi
    
    # Check if docker-compose exists
    if ! command_exists docker-compose; then
        print_error "docker-compose is not installed. Please install it."
        exit 1
    fi
    
    # Check if Python 3.8+ is available
    if ! command_exists python3; then
        print_error "Python 3 is not installed. Please install Python 3.8+."
        exit 1
    fi
    
    # Check Python version
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    required_version="3.8"
    
    if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
        print_error "Python 3.8+ is required. Current version: $python_version"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
    
    # Load environment variables
    print_step "Loading environment variables..."
    if [ -f .env ]; then
        export $(grep -v '^#' .env | xargs)
        print_success "Environment variables loaded"
    else
        print_error "Failed to load environment variables"
        exit 1
    fi
    
    # Start Docker services
    print_step "Starting Docker services..."
    if [ -f docker-compose.yml ]; then
        docker-compose up -d postgres redis
        
        # Wait for Docker services to be healthy
        if ! wait_for_docker_health; then
            print_error "Docker services failed to start properly"
            docker-compose logs
            exit 1
        fi
    else
        print_error "docker-compose.yml not found"
        exit 1
    fi
    
    # Install/upgrade Python dependencies
    print_step "Installing Python dependencies..."
    if [ -f requirements.txt ]; then
        pip3 install -r requirements.txt --quiet
        print_success "Dependencies installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
    
    # Check database connection
    if ! check_database_connection; then
        print_error "Database connection failed"
        exit 1
    fi
    
    # Initialize database
    if ! initialize_database; then
        print_error "Database initialization failed"
        exit 1
    fi
    
    # Start Celery worker
    if ! start_celery_worker; then
        print_error "Failed to start Celery worker"
        exit 1
    fi
    
    # Start FastAPI server
    if ! start_fastapi_server; then
        print_error "Failed to start FastAPI server"
        exit 1
    fi
    
    # Show final status
    show_final_status
    
    # Keep script running to maintain services
    print_step "Press Ctrl+C to stop all services..."
    while true; do
        sleep 1
    done
}

# Run main function
main "$@" 