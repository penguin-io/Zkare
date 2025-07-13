#!/bin/bash
set -euo pipefail

# Health check script for Entity 2 LLM Advisor Service
# Performs comprehensive health checks including service availability,
# LLM readiness, cache connectivity, and external service connectivity

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
HEALTH_ENDPOINT="http://localhost:${PORT:-8002}/health"
TIMEOUT=15
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
        sleep 2
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
    local llm_ready
    local cache_connected
    local proof_verifier_connected
    local uptime

    status=$(echo "$response" | grep -o '"status":"[^"]*"' | cut -d'"' -f4 2>/dev/null || echo "unknown")
    llm_ready=$(echo "$response" | grep -o '"llm_ready":[^,}]*' | cut -d':' -f2 2>/dev/null || echo "false")
    cache_connected=$(echo "$response" | grep -o '"cache_connected":[^,}]*' | cut -d':' -f2 2>/dev/null || echo "false")
    proof_verifier_connected=$(echo "$response" | grep -o '"proof_verifier_connected":[^,}]*' | cut -d':' -f2 2>/dev/null || echo "false")
    uptime=$(echo "$response" | grep -o '"uptime_seconds":[^,}]*' | cut -d':' -f2 2>/dev/null || echo "0")

    log_info "Service status: $status"
    log_info "LLM ready: $llm_ready"
    log_info "Cache connected: $cache_connected"
    log_info "Proof verifier connected: $proof_verifier_connected"
    log_info "Uptime: ${uptime}s"

    # Check if all components are healthy
    if [ "$status" = "healthy" ] && [ "$llm_ready" = "true" ] && [ "$cache_connected" = "true" ] && [ "$proof_verifier_connected" = "true" ]; then
        return 0
    else
        return 1
    fi
}

# Check process is running
check_process() {
    if pgrep -f "app.main" >/dev/null 2>&1; then
        return 0
    else
        log_error "LLM Advisor Service process not found"
        return 1
    fi
}

# Check GPU status if enabled
check_gpu_status() {
    if [ "${GPU_ENABLED:-false}" = "true" ]; then
        if command -v nvidia-smi >/dev/null 2>&1; then
            if nvidia-smi >/dev/null 2>&1; then
                # Check GPU utilization
                local gpu_util
                gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
                log_info "GPU status: Available (${gpu_util:-0}% utilization)"
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
    memory_usage=$(ps -o pid,ppid,rss,comm -p $(pgrep -f "app.main") 2>/dev/null | tail -n +2 | awk '{sum+=$3} END {print sum}' || echo "0")

    if [ "$memory_usage" -gt 0 ]; then
        local memory_mb=$((memory_usage / 1024))
        log_info "Memory usage: ${memory_mb}MB"

        # Warning if memory usage is too high (> 8GB)
        if [ "$memory_mb" -gt 8192 ]; then
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

# Check model files availability
check_model_files() {
    local model_path="${MODEL_PATH:-/app/models}"

    if [ -d "$model_path" ]; then
        local model_count
        model_count=$(find "$model_path" -name "*.bin" -o -name "*.pth" -o -name "*.safetensors" 2>/dev/null | wc -l)

        if [ "$model_count" -gt 0 ]; then
            log_info "Model files: $model_count files found"
            return 0
        else
            log_warn "Model files: No model files found in $model_path"
            return 1
        fi
    else
        log_error "Model path not found: $model_path"
        return 1
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

# Check external service connectivity
check_external_services() {
    local zkproof_url="${ZKPROOF_SERVICE_URL:-http://zkproof-service:8001}"
    local redis_url="${REDIS_URL:-redis://redis:6379}"

    # Extract host and port from URLs
    local zkproof_host
    local zkproof_port
    zkproof_host=$(echo "$zkproof_url" | sed 's|.*://||' | cut -d':' -f1)
    zkproof_port=$(echo "$zkproof_url" | sed 's|.*://||' | cut -d':' -f2 | cut -d'/' -f1)

    local redis_host
    local redis_port
    redis_host=$(echo "$redis_url" | sed 's|.*://||' | cut -d':' -f1)
    redis_port=$(echo "$redis_url" | sed 's|.*://||' | cut -d':' -f2)

    local services_ok=0

    # Check ZK Proof Service
    if nc -z "$zkproof_host" "$zkproof_port" 2>/dev/null; then
        log_info "ZK Proof Service: Connected ($zkproof_host:$zkproof_port)"
        ((services_ok++))
    else
        log_warn "ZK Proof Service: Connection failed ($zkproof_host:$zkproof_port)"
    fi

    # Check Redis
    if nc -z "$redis_host" "$redis_port" 2>/dev/null; then
        log_info "Redis: Connected ($redis_host:$redis_port)"
        ((services_ok++))
    else
        log_warn "Redis: Connection failed ($redis_host:$redis_port)"
    fi

    # At least one external service should be reachable
    if [ $services_ok -gt 0 ]; then
        return 0
    else
        return 1
    fi
}

# Main health check function
main() {
    local exit_code=0
    local checks_passed=0
    local total_checks=9

    log_info "Starting health check for LLM Advisor Service..."

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

    # Model and service checks
    if check_model_files; then
        log_success "✓ Model files check passed"
        ((checks_passed++))
    else
        log_warn "⚠ Model files check failed (non-critical)"
        # Model files failure is not critical for container health during startup
    fi

    if check_external_services; then
        log_success "✓ External services check passed"
        ((checks_passed++))
    else
        log_warn "⚠ External services check failed (non-critical)"
        # External service failure is not critical if the main service is running
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
        log_success "LLM Advisor Service is healthy"
    else
        log_error "LLM Advisor Service has health issues"
    fi

    exit $exit_code
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
