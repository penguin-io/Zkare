#!/bin/bash
set -euo pipefail

# Health check script for ZK Proof Service (Entity 1)
# Performs comprehensive health checks including service availability,
# database connectivity, Redis connectivity, and GPU status

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
HEALTH_ENDPOINT="http://localhost:${PORT:-8001}/health"
TIMEOUT=10
MAX_RETRIES=3

# Logging functions
log_info() {
    echo -e "[INFO] $1" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

# Check if service is responding
check_service_endpoint() {
    local attempt=1

    while [ $attempt -le $MAX_RETRIES ]; do
        if curl -f -s --max-time $TIMEOUT "$HEALTH_ENDPOINT" >/dev/null 2>&1; then
            return 0
        fi

        log_warn "Health endpoint check failed (attempt $attempt/$MAX_RETRIES)"
        sleep 1
        ((attempt++))
    done

    return 1
}

# Check detailed health status
check_detailed_health() {
    local response
    response=$(curl -f -s --max-time $TIMEOUT "$HEALTH_ENDPOINT" 2>/dev/null || echo '{}')

    # Parse JSON response
    local status
    local db_connected
    local redis_connected
    local uptime

    status=$(echo "$response" | grep -o '"status":"[^"]*"' | cut -d'"' -f4 2>/dev/null || echo "unknown")
    db_connected=$(echo "$response" | grep -o '"database_connected":[^,}]*' | cut -d':' -f2 2>/dev/null || echo "false")
    redis_connected=$(echo "$response" | grep -o '"redis_connected":[^,}]*' | cut -d':' -f2 2>/dev/null || echo "false")
    uptime=$(echo "$response" | grep -o '"uptime_seconds":[^,}]*' | cut -d':' -f2 2>/dev/null || echo "0")

    log_info "Service status: $status"
    log_info "Database connected: $db_connected"
    log_info "Redis connected: $redis_connected"
    log_info "Uptime: ${uptime}s"

    # Check if all components are healthy
    if [ "$status" = "healthy" ] && [ "$db_connected" = "true" ] && [ "$redis_connected" = "true" ]; then
        return 0
    else
        return 1
    fi
}

# Check process is running
check_process() {
    if pgrep -f "zkproof-service" >/dev/null 2>&1; then
        return 0
    else
        log_error "ZK Proof Service process not found"
        return 1
    fi
}

# Check GPU status if enabled
check_gpu_status() {
    if [ "${ENABLE_GPU:-false}" = "true" ]; then
        if command -v nvidia-smi >/dev/null 2>&1; then
            if nvidia-smi >/dev/null 2>&1; then
                log_info "GPU status: Available"
                return 0
            else
                log_warn "GPU status: Error"
                return 1
            fi
        else
            log_warn "GPU status: nvidia-smi not available"
            return 1
        fi
    else
        log_info "GPU status: Disabled"
        return 0
    fi
}

# Check memory usage
check_memory_usage() {
    local memory_usage
    memory_usage=$(ps -o pid,ppid,rss,comm -p $(pgrep -f "zkproof-service") 2>/dev/null | tail -n +2 | awk '{sum+=$3} END {print sum}' || echo "0")

    if [ "$memory_usage" -gt 0 ]; then
        local memory_mb=$((memory_usage / 1024))
        log_info "Memory usage: ${memory_mb}MB"

        # Warning if memory usage is too high (> 4GB)
        if [ "$memory_mb" -gt 4096 ]; then
            log_warn "High memory usage detected: ${memory_mb}MB"
            return 1
        fi
        return 0
    else
        log_error "Could not determine memory usage"
        return 1
    fi
}

# Check disk space
check_disk_space() {
    local disk_usage
    disk_usage=$(df /app | tail -1 | awk '{print $5}' | sed 's/%//')

    if [ "$disk_usage" -gt 90 ]; then
        log_error "Disk usage critical: ${disk_usage}%"
        return 1
    elif [ "$disk_usage" -gt 80 ]; then
        log_warn "Disk usage high: ${disk_usage}%"
        return 0
    else
        log_info "Disk usage: ${disk_usage}%"
        return 0
    fi
}

# Check log file sizes
check_log_files() {
    local log_dir="/app/logs"

    if [ -d "$log_dir" ]; then
        local total_size
        total_size=$(du -s "$log_dir" 2>/dev/null | cut -f1 || echo "0")
        local size_mb=$((total_size / 1024))

        log_info "Log files size: ${size_mb}MB"

        # Warning if logs are too large (> 1GB)
        if [ "$size_mb" -gt 1024 ]; then
            log_warn "Large log files detected: ${size_mb}MB"
            return 1
        fi
        return 0
    else
        log_info "Log directory not found"
        return 0
    fi
}

# Main health check function
main() {
    local exit_code=0
    local checks_passed=0
    local total_checks=7

    log_info "Starting health check for ZK Proof Service..."

    # Core service checks
    if check_process; then
        log_success "✓ Process check passed"
        ((checks_passed++))
    else
        log_error "✗ Process check failed"
        exit_code=1
    fi

    if check_service_endpoint; then
        log_success "✓ Service endpoint check passed"
        ((checks_passed++))
    else
        log_error "✗ Service endpoint check failed"
        exit_code=1
    fi

    if check_detailed_health; then
        log_success "✓ Detailed health check passed"
        ((checks_passed++))
    else
        log_error "✗ Detailed health check failed"
        exit_code=1
    fi

    # Resource checks
    if check_memory_usage; then
        log_success "✓ Memory usage check passed"
        ((checks_passed++))
    else
        log_error "✗ Memory usage check failed"
        exit_code=1
    fi

    if check_disk_space; then
        log_success "✓ Disk space check passed"
        ((checks_passed++))
    else
        log_error "✗ Disk space check failed"
        exit_code=1
    fi

    if check_log_files; then
        log_success "✓ Log files check passed"
        ((checks_passed++))
    else
        log_error "✗ Log files check failed"
        exit_code=1
    fi

    # GPU check (optional)
    if check_gpu_status; then
        log_success "✓ GPU status check passed"
        ((checks_passed++))
    else
        log_warn "⚠ GPU status check failed (non-critical)"
        # GPU failure is not critical for container health
    fi

    # Summary
    log_info "Health check completed: $checks_passed/$total_checks checks passed"

    if [ $exit_code -eq 0 ]; then
        log_success "ZK Proof Service is healthy"
    else
        log_error "ZK Proof Service has health issues"
    fi

    exit $exit_code
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
