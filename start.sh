#!/bin/bash
set -euo pipefail

# PenguLLM System Startup Script
# Orchestrates the complete privacy-preserving AI advisor system

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.yml"
PROJECT_NAME="pengu-llm"
LOG_DIR="./logs"
MODELS_DIR="./models"
DATA_DIR="./data"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_header() {
    echo -e "${PURPLE}================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}================================${NC}"
}

# Check if Docker is installed and running
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    log_success "Docker and Docker Compose are available"
}

# Check NVIDIA GPU support
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            log_success "NVIDIA GPU detected:"
            nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

            # Check NVIDIA Container Runtime
            if docker info 2>/dev/null | grep -q nvidia; then
                log_success "NVIDIA Container Runtime is configured"
            else
                log_warn "NVIDIA Container Runtime not detected. GPU acceleration may not work."
                log_warn "Please install nvidia-container-runtime and restart Docker."
            fi
        else
            log_warn "NVIDIA drivers not properly installed"
        fi
    else
        log_warn "NVIDIA GPU not detected. System will run in CPU-only mode."
    fi
}

# Create required directories
setup_directories() {
    log_info "Setting up directory structure..."

    mkdir -p "$LOG_DIR"
    mkdir -p "$MODELS_DIR"
    mkdir -p "$DATA_DIR"
    mkdir -p "./database/backup"
    mkdir -p "./config"

    log_success "Directory structure created"
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."

    # Check available memory
    local memory_gb=$(free -g | awk 'NR==2{printf "%.1f", $2}')
    log_info "Available memory: ${memory_gb}GB"

    if (( $(echo "$memory_gb < 16" | bc -l) )); then
        log_warn "Recommended memory: 16GB+. Current: ${memory_gb}GB"
        log_warn "System may experience performance issues."
    fi

    # Check available disk space
    local disk_avail=$(df -h . | awk 'NR==2 {print $4}')
    log_info "Available disk space: $disk_avail"

    # Check for required files
    if [ ! -f "$COMPOSE_FILE" ]; then
        log_error "docker-compose.yml not found in current directory"
        exit 1
    fi

    log_success "System requirements check completed"
}

# Download model files (placeholder)
setup_models() {
    log_info "Checking model files..."

    if [ "$(ls -A $MODELS_DIR 2>/dev/null)" ]; then
        log_success "Model files found in $MODELS_DIR"
    else
        log_warn "No model files found in $MODELS_DIR"
        log_warn "Please download Llama model files to $MODELS_DIR"
        log_warn "Expected structure:"
        log_warn "  $MODELS_DIR/"
        log_warn "  ├── llama-3.1-70b/"
        log_warn "  │   ├── tokenizer.model"
        log_warn "  │   ├── consolidated.*.pth"
        log_warn "  │   └── params.json"
        echo
        read -p "Continue without models? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Please add model files and run the script again."
            exit 1
        fi
    fi
}

# Create environment configuration
create_env_config() {
    log_info "Creating environment configuration..."

    if [ ! -f ".env" ]; then
        cat > .env << EOF
# PenguLLM Environment Configuration

# Database Configuration
POSTGRES_DB=zkp_llm
POSTGRES_USER=zkp_user
POSTGRES_PASSWORD=zkp_secure_password_$(openssl rand -hex 8)
DATABASE_URL=postgres://zkp_user:zkp_secure_password_$(openssl rand -hex 8)@postgres:5432/zkp_llm

# Redis Configuration
REDIS_URL=redis://redis:6379

# Entity 1 (ZK Proof Service) Configuration
ZKPROOF_SERVICE_URL=http://zkproof-service:8001
RUST_LOG=info
RISC0_DEV_MODE=0
ENABLE_GPU=true
RATE_LIMIT=10
CACHE_TTL=3600

# Entity 2 (LLM Service) Configuration
LLM_SERVICE_URL=http://llm-advisor:8002
MODEL_PATH=/app/models
MODEL_NAME=llama-3.1-70b
GPU_ENABLED=true
CUDA_VISIBLE_DEVICES=0
TEMPERATURE=0.7
MAX_CONTEXT_LENGTH=4096
MAX_RESPONSE_TOKENS=512
RATE_LIMIT_PER_MINUTE=60

# Web Interface Configuration
NEXT_PUBLIC_ZK_SERVICE_URL=http://localhost:8001
NEXT_PUBLIC_LLM_SERVICE_URL=http://localhost:8002

# Security
JWT_SECRET=$(openssl rand -hex 32)
API_KEY=$(openssl rand -hex 16)

# Monitoring
ENABLE_METRICS=true
LOG_LEVEL=info
EOF
        log_success "Environment configuration created (.env)"
    else
        log_info "Environment configuration already exists (.env)"
    fi
}

# Start the system
start_system() {
    log_header "Starting PenguLLM System"

    log_info "Building Docker images..."
    docker-compose -p "$PROJECT_NAME" build

    log_info "Starting services..."
    docker-compose -p "$PROJECT_NAME" up -d

    log_info "Waiting for services to be ready..."

    # Wait for database
    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if docker-compose -p "$PROJECT_NAME" exec -T postgres pg_isready -U zkp_user -d zkp_llm 2>/dev/null; then
            log_success "Database is ready"
            break
        fi

        if [ $attempt -eq $max_attempts ]; then
            log_error "Database failed to start after $max_attempts attempts"
            return 1
        fi

        log_info "Waiting for database... (attempt $attempt/$max_attempts)"
        sleep 5
        ((attempt++))
    done

    # Wait for Redis
    attempt=1
    while [ $attempt -le $max_attempts ]; do
        if docker-compose -p "$PROJECT_NAME" exec -T redis redis-cli ping 2>/dev/null | grep -q PONG; then
            log_success "Redis is ready"
            break
        fi

        if [ $attempt -eq $max_attempts ]; then
            log_error "Redis failed to start after $max_attempts attempts"
            return 1
        fi

        log_info "Waiting for Redis... (attempt $attempt/$max_attempts)"
        sleep 3
        ((attempt++))
    done

    # Wait for ZK Proof Service
    attempt=1
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s http://localhost:8001/health >/dev/null 2>&1; then
            log_success "ZK Proof Service is ready"
            break
        fi

        if [ $attempt -eq $max_attempts ]; then
            log_warn "ZK Proof Service may still be starting (this can take several minutes)"
            break
        fi

        log_info "Waiting for ZK Proof Service... (attempt $attempt/$max_attempts)"
        sleep 10
        ((attempt++))
    done

    # Wait for LLM Service
    attempt=1
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s http://localhost:8002/health >/dev/null 2>&1; then
            log_success "LLM Advisor Service is ready"
            break
        fi

        if [ $attempt -eq $max_attempts ]; then
            log_warn "LLM Advisor Service may still be loading models (this can take several minutes)"
            break
        fi

        log_info "Waiting for LLM Advisor Service... (attempt $attempt/$max_attempts)"
        sleep 10
        ((attempt++))
    done

    # Wait for Web Interface
    attempt=1
    while [ $attempt -le 15 ]; do
        if curl -f -s http://localhost:3000 >/dev/null 2>&1; then
            log_success "Web Interface is ready"
            break
        fi

        if [ $attempt -eq 15 ]; then
            log_warn "Web Interface may still be starting"
            break
        fi

        log_info "Waiting for Web Interface... (attempt $attempt/15)"
        sleep 5
        ((attempt++))
    done
}

# Show system status
show_status() {
    log_header "System Status"

    docker-compose -p "$PROJECT_NAME" ps

    echo
    log_header "Service Endpoints"
    echo -e "${CYAN}Web Interface:${NC}        http://localhost:3000"
    echo -e "${CYAN}ZK Proof Service:${NC}     http://localhost:8001"
    echo -e "${CYAN}LLM Advisor Service:${NC}  http://localhost:8002"
    echo -e "${CYAN}Metrics:${NC}              http://localhost:8002/metrics"
    echo

    # Check service health
    log_info "Checking service health..."

    services=("postgres:5432" "redis:6379" "localhost:8001" "localhost:8002" "localhost:3000")
    service_names=("Database" "Redis" "ZK Proof Service" "LLM Service" "Web Interface")

    for i in "${!services[@]}"; do
        service="${services[$i]}"
        name="${service_names[$i]}"

        if [[ "$service" == *"localhost"* ]]; then
            if curl -f -s "http://$service/health" >/dev/null 2>&1 || curl -f -s "http://$service" >/dev/null 2>&1; then
                echo -e "${GREEN}✓${NC} $name"
            else
                echo -e "${RED}✗${NC} $name"
            fi
        else
            if nc -z ${service/:/ } 2>/dev/null; then
                echo -e "${GREEN}✓${NC} $name"
            else
                echo -e "${RED}✗${NC} $name"
            fi
        fi
    done
}

# Show logs
show_logs() {
    log_header "System Logs"
    echo "Press Ctrl+C to exit log viewing"
    echo
    docker-compose -p "$PROJECT_NAME" logs -f
}

# Stop the system
stop_system() {
    log_header "Stopping PenguLLM System"

    docker-compose -p "$PROJECT_NAME" down
    log_success "System stopped"
}

# Clean up the system
cleanup_system() {
    log_header "Cleaning Up PenguLLM System"

    read -p "This will remove all containers, volumes, and images. Continue? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose -p "$PROJECT_NAME" down -v --rmi all
        docker system prune -f
        log_success "Cleanup completed"
    else
        log_info "Cleanup cancelled"
    fi
}

# Show usage
show_usage() {
    echo "PenguLLM - Privacy-Preserving Personalized Advice System"
    echo
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  start     Start the complete system"
    echo "  stop      Stop all services"
    echo "  restart   Restart all services"
    echo "  status    Show system status"
    echo "  logs      Show system logs"
    echo "  cleanup   Remove all containers and images"
    echo "  check     Check system requirements"
    echo "  help      Show this help message"
    echo
    echo "Examples:"
    echo "  $0 start     # Start the system"
    echo "  $0 status    # Check if services are running"
    echo "  $0 logs      # View real-time logs"
    echo
}

# Main execution
main() {
    case "${1:-}" in
        start)
            log_header "PenguLLM System Startup"
            check_docker
            check_gpu
            check_requirements
            setup_directories
            setup_models
            create_env_config
            start_system
            show_status
            echo
            log_success "PenguLLM system is starting up!"
            log_info "Access the web interface at: http://localhost:3000"
            log_info "View logs with: $0 logs"
            ;;
        stop)
            stop_system
            ;;
        restart)
            stop_system
            start_system
            show_status
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        cleanup)
            cleanup_system
            ;;
        check)
            check_docker
            check_gpu
            check_requirements
            ;;
        help|--help|-h)
            show_usage
            ;;
        "")
            show_usage
            ;;
        *)
            log_error "Unknown command: $1"
            echo
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
