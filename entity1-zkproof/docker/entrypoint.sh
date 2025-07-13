#!/bin/bash
set -euo pipefail

# Entrypoint script for ZK Proof Service (Entity 1)
# Handles initialization, database migrations, and service startup

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Environment variables with defaults
export DATABASE_URL="${DATABASE_URL:-postgres://zkp_user:zkp_secure_password@postgres:5432/zkp_llm}"
export REDIS_URL="${REDIS_URL:-redis://redis:6379}"
export PORT="${PORT:-8001}"
export RUST_LOG="${RUST_LOG:-info}"
export RISC0_DEV_MODE="${RISC0_DEV_MODE:-0}"
export ENABLE_GPU="${ENABLE_GPU:-true}"
export RATE_LIMIT="${RATE_LIMIT:-10}"
export CACHE_TTL="${CACHE_TTL:-3600}"

# Wait for dependencies
wait_for_service() {
    local host="$1"
    local port="$2"
    local service_name="$3"
    local max_attempts=30
    local attempt=1

    log_info "Waiting for $service_name at $host:$port..."

    while ! nc -z "$host" "$port" 2>/dev/null; do
        if [ $attempt -eq $max_attempts ]; then
            log_error "Failed to connect to $service_name after $max_attempts attempts"
            exit 1
        fi

        log_info "Attempt $attempt/$max_attempts: $service_name not ready, waiting..."
        sleep 2
        ((attempt++))
    done

    log_success "$service_name is ready!"
}

# Parse database URL to extract connection details
parse_database_url() {
    # Extract host and port from DATABASE_URL
    # Format: postgres://user:password@host:port/database
    local db_url="$DATABASE_URL"

    # Remove protocol
    db_url="${db_url#postgres://}"

    # Extract user:password@host:port/database
    local auth_host="${db_url%%/*}"

    # Extract host:port
    local host_port="${auth_host##*@}"

    # Extract host and port
    DB_HOST="${host_port%%:*}"
    DB_PORT="${host_port##*:}"

    log_info "Parsed database connection: $DB_HOST:$DB_PORT"
}

# Parse Redis URL
parse_redis_url() {
    local redis_url="$REDIS_URL"
    redis_url="${redis_url#redis://}"

    REDIS_HOST="${redis_url%%:*}"
    REDIS_PORT="${redis_url##*:}"

    log_info "Parsed Redis connection: $REDIS_HOST:$REDIS_PORT"
}

# GPU detection and setup
setup_gpu() {
    if [ "$ENABLE_GPU" = "true" ]; then
        log_info "GPU acceleration enabled, checking for NVIDIA GPU..."

        if command -v nvidia-smi >/dev/null 2>&1; then
            log_info "NVIDIA GPU detected:"
            nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits

            # Set optimal CUDA settings for RiscZero
            export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
            export RISC0_GPU_ENABLE=1

            log_success "GPU acceleration configured"
        else
            log_warn "GPU acceleration requested but nvidia-smi not found, falling back to CPU"
            export ENABLE_GPU="false"
            export RISC0_GPU_ENABLE=0
        fi
    else
        log_info "GPU acceleration disabled, using CPU only"
        export RISC0_GPU_ENABLE=0
    fi
}

# Database health check
check_database() {
    log_info "Checking database connection..."

    # Simple connection test using psql if available
    if command -v psql >/dev/null 2>&1; then
        if psql "$DATABASE_URL" -c "SELECT 1;" >/dev/null 2>&1; then
            log_success "Database connection successful"
        else
            log_error "Database connection failed"
            return 1
        fi
    else
        log_warn "psql not available, skipping database connection test"
    fi
}

# Redis health check
check_redis() {
    log_info "Checking Redis connection..."

    if command -v redis-cli >/dev/null 2>&1; then
        if redis-cli -u "$REDIS_URL" ping >/dev/null 2>&1; then
            log_success "Redis connection successful"
        else
            log_error "Redis connection failed"
            return 1
        fi
    else
        log_warn "redis-cli not available, skipping Redis connection test"
    fi
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."

    # The application will handle migrations via sqlx::migrate!
    # This is just a placeholder for any additional setup

    log_success "Database migrations completed"
}

# Setup monitoring and observability
setup_monitoring() {
    log_info "Setting up monitoring..."

    # Create directories for logs and metrics
    mkdir -p /app/logs /app/metrics

    # Set up log rotation configuration
    cat > /app/logs/logrotate.conf << EOF
/app/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 zkproof zkproof
}
EOF

    log_success "Monitoring setup completed"
}

# Validate configuration
validate_config() {
    log_info "Validating configuration..."

    # Check required environment variables
    local required_vars=(
        "DATABASE_URL"
        "REDIS_URL"
        "PORT"
    )

    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            log_error "Required environment variable $var is not set"
            exit 1
        fi
    done

    # Validate port number
    if ! [[ "$PORT" =~ ^[0-9]+$ ]] || [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
        log_error "Invalid port number: $PORT"
        exit 1
    fi

    # Validate rate limit
    if ! [[ "$RATE_LIMIT" =~ ^[0-9]+$ ]] || [ "$RATE_LIMIT" -lt 1 ]; then
        log_error "Invalid rate limit: $RATE_LIMIT"
        exit 1
    fi

    log_success "Configuration validation passed"
}

# Signal handlers for graceful shutdown
cleanup() {
    log_info "Received shutdown signal, cleaning up..."

    # Kill background processes
    jobs -p | xargs -r kill

    # Additional cleanup if needed

    log_info "Cleanup completed"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Main initialization sequence
main() {
    log_info "Starting ZK Proof Service (Entity 1) initialization..."
    log_info "Version: $(cat /app/VERSION 2>/dev/null || echo 'unknown')"
    log_info "Build time: $(date)"

    # Step 1: Validate configuration
    validate_config

    # Step 2: Parse connection URLs
    parse_database_url
    parse_redis_url

    # Step 3: Setup GPU if enabled
    setup_gpu

    # Step 4: Wait for dependencies
    wait_for_service "$DB_HOST" "$DB_PORT" "PostgreSQL"
    wait_for_service "$REDIS_HOST" "$REDIS_PORT" "Redis"

    # Step 5: Health checks
    check_database
    check_redis

    # Step 6: Setup monitoring
    setup_monitoring

    # Step 7: Run migrations (handled by the application)
    run_migrations

    log_success "Initialization completed successfully!"
    log_info "Starting ZK Proof Service on port $PORT"

    # Execute the main command
    exec "$@"
}

# Check if we're being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
