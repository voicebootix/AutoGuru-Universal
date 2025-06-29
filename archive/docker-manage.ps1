# AutoGuru Universal - Docker Management Script (PowerShell)

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Write-Header {
    Write-Host "================================" -ForegroundColor Blue
    Write-Host "  AutoGuru Universal Docker" -ForegroundColor Blue
    Write-Host "================================" -ForegroundColor Blue
}

# Function to check if Docker is running
function Test-Docker {
    try {
        docker info | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Function to check if docker-compose is available
function Test-DockerCompose {
    try {
        docker-compose --version | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Function to start services
function Start-Services {
    Write-Status "Starting AutoGuru Universal services..."
    
    # Start core services
    docker-compose up -d postgres redis
    
    Write-Status "Waiting for services to be ready..."
    Start-Sleep -Seconds 10
    
    # Check service health
    $status = docker-compose ps
    if ($status -match "Up") {
        Write-Status "Core services started successfully!"
        Write-Status "PostgreSQL: localhost:5432"
        Write-Status "Redis: localhost:6379"
    }
    else {
        Write-Error "Failed to start services. Check logs with: docker-compose logs"
        exit 1
    }
}

# Function to start with tools
function Start-ServicesWithTools {
    Write-Status "Starting AutoGuru Universal services with management tools..."
    
    # Start all services including tools
    docker-compose --profile tools up -d
    
    Write-Status "Waiting for services to be ready..."
    Start-Sleep -Seconds 15
    
    Write-Status "All services started successfully!"
    Write-Status "PostgreSQL: localhost:5432"
    Write-Status "Redis: localhost:6379"
    Write-Status "pgAdmin: http://localhost:5050 (admin@autoguru.com / admin123)"
    Write-Status "Redis Commander: http://localhost:8081"
}

# Function to stop services
function Stop-Services {
    Write-Status "Stopping AutoGuru Universal services..."
    docker-compose down
    Write-Status "Services stopped successfully!"
}

# Function to restart services
function Restart-Services {
    Write-Status "Restarting AutoGuru Universal services..."
    docker-compose restart
    Write-Status "Services restarted successfully!"
}

# Function to show logs
function Show-Logs {
    param([string]$Service = "")
    
    if ([string]::IsNullOrEmpty($Service)) {
        Write-Status "Showing logs for all services..."
        docker-compose logs -f
    }
    else {
        Write-Status "Showing logs for service: $Service"
        docker-compose logs -f $Service
    }
}

# Function to show status
function Show-Status {
    Write-Status "Service status:"
    docker-compose ps
}

# Function to clean up
function Clear-DockerResources {
    Write-Warning "This will remove all containers, networks, and volumes!"
    $response = Read-Host "Are you sure? (y/N)"
    
    if ($response -eq "y" -or $response -eq "Y") {
        Write-Status "Cleaning up Docker resources..."
        docker-compose down -v --remove-orphans
        docker system prune -f
        Write-Status "Cleanup completed!"
    }
    else {
        Write-Status "Cleanup cancelled."
    }
}

# Function to reset database
function Reset-Database {
    Write-Warning "This will remove all data from PostgreSQL!"
    $response = Read-Host "Are you sure? (y/N)"
    
    if ($response -eq "y" -or $response -eq "Y") {
        Write-Status "Resetting database..."
        docker-compose down postgres
        docker volume rm autoguru-universal_postgres_data
        docker-compose up -d postgres
        Write-Status "Database reset completed!"
    }
    else {
        Write-Status "Database reset cancelled."
    }
}

# Function to show help
function Show-Help {
    Write-Header
    Write-Host "Usage: .\docker-manage.ps1 [COMMAND]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  start           Start core services (PostgreSQL, Redis)"
    Write-Host "  start-tools     Start services with management tools (pgAdmin, Redis Commander)"
    Write-Host "  stop            Stop all services"
    Write-Host "  restart         Restart all services"
    Write-Host "  logs [SERVICE]  Show logs (all services or specific service)"
    Write-Host "  status          Show service status"
    Write-Host "  cleanup         Remove all containers, networks, and volumes"
    Write-Host "  reset-db        Reset PostgreSQL database (removes all data)"
    Write-Host "  help            Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\docker-manage.ps1 start"
    Write-Host "  .\docker-manage.ps1 logs postgres"
    Write-Host "  .\docker-manage.ps1 start-tools"
}

# Main script logic
function Main {
    # Check Docker
    if (-not (Test-Docker)) {
        Write-Error "Docker is not running. Please start Docker and try again."
        exit 1
    }
    
    # Check docker-compose
    if (-not (Test-DockerCompose)) {
        Write-Error "docker-compose is not installed. Please install it and try again."
        exit 1
    }
    
    switch ($Command) {
        "start" { Start-Services }
        "start-tools" { Start-ServicesWithTools }
        "stop" { Stop-Services }
        "restart" { Restart-Services }
        "logs" { Show-Logs $args[0] }
        "status" { Show-Status }
        "cleanup" { Clear-DockerResources }
        "reset-db" { Reset-Database }
        "help" { Show-Help }
        default {
            Write-Error "Unknown command: $Command"
            Write-Host ""
            Show-Help
            exit 1
        }
    }
}

# Run main function
Main 