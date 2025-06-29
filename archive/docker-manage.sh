#!/bin/bash

# AutoGuru Universal - Docker Management Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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
    echo -e "${BLUE}  AutoGuru Universal Docker${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to check if docker-compose is available
check_docker_compose() {
    if ! command -v docker-compose > /dev/null 2>&1; then
        print_error "docker-compose is not installed. Please install it and try again."
        exit 1
    fi
}

# Function to start services
start_services() {
    print_status "Starting AutoGuru Universal services..."
    
    # Start core services
    docker-compose up -d postgres redis
    
    print_status "Waiting for services to be ready..."
    sleep 10
    
    # Check service health
    if docker-compose ps | grep -q "Up"; then
        print_status "Core services started successfully!"
        print_status "PostgreSQL: localhost:5432"
        print_status "Redis: localhost:6379"
    else
        print_error "Failed to start services. Check logs with: docker-compose logs"
        exit 1
    fi
}

# Function to start with tools
start_with_tools() {
    print_status "Starting AutoGuru Universal services with management tools..."
    
    # Start all services including tools
    docker-compose --profile tools up -d
    
    print_status "Waiting for services to be ready..."
    sleep 15
    
    print_status "All services started successfully!"
    print_status "PostgreSQL: localhost:5432"
    print_status "Redis: localhost:6379"
    print_status "pgAdmin: http://localhost:5050 (admin@autoguru.com / admin123)"
    print_status "Redis Commander: http://localhost:8081"
}

# Function to stop services
stop_services() {
    print_status "Stopping AutoGuru Universal services..."
    docker-compose down
    print_status "Services stopped successfully!"
}

# Function to restart services
restart_services() {
    print_status "Restarting AutoGuru Universal services..."
    docker-compose restart
    print_status "Services restarted successfully!"
}

# Function to show logs
show_logs() {
    if [ -z "$1" ]; then
        print_status "Showing logs for all services..."
        docker-compose logs -f
    else
        print_status "Showing logs for service: $1"
        docker-compose logs -f "$1"
    fi
}

# Function to show status
show_status() {
    print_status "Service status:"
    docker-compose ps
}

# Function to clean up
cleanup() {
    print_warning "This will remove all containers, networks, and volumes!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Cleaning up Docker resources..."
        docker-compose down -v --remove-orphans
        docker system prune -f
        print_status "Cleanup completed!"
    else
        print_status "Cleanup cancelled."
    fi
}

# Function to reset database
reset_database() {
    print_warning "This will remove all data from PostgreSQL!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Resetting database..."
        docker-compose down postgres
        docker volume rm autoguru-universal_postgres_data
        docker-compose up -d postgres
        print_status "Database reset completed!"
    else
        print_status "Database reset cancelled."
    fi
}

# Function to show help
show_help() {
    print_header
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start           Start core services (PostgreSQL, Redis)"
    echo "  start-tools     Start services with management tools (pgAdmin, Redis Commander)"
    echo "  stop            Stop all services"
    echo "  restart         Restart all services"
    echo "  logs [SERVICE]  Show logs (all services or specific service)"
    echo "  status          Show service status"
    echo "  cleanup         Remove all containers, networks, and volumes"
    echo "  reset-db        Reset PostgreSQL database (removes all data)"
    echo "  help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start"
    echo "  $0 logs postgres"
    echo "  $0 start-tools"
}

# Main script logic
main() {
    check_docker
    check_docker_compose
    
    case "${1:-help}" in
        start)
            start_services
            ;;
        start-tools)
            start_with_tools
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        logs)
            show_logs "$2"
            ;;
        status)
            show_status
            ;;
        cleanup)
            cleanup
            ;;
        reset-db)
            reset_database
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@" 