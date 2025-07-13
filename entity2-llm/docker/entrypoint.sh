#!/bin/bash
set -euo pipefail

# Entrypoint script for Entity 2 LLM Advisor Service
# Handles initialization, model loading, and service startup

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
export MODEL_PATH="${MODEL_PATH:-/app/models}"
export CONFIG_PATH="${CONFIG_PATH:-/app/config}"
export ZKPROOF_SERVICE_URL="${ZKPROOF_SERVICE_URL:-http://zkproof-service:8001}"
export REDIS_URL="${REDIS_URL:-redis://redis:6379}"
export PORT="${PORT:-8002}"
export GPU_ENABLED="${GPU_ENABLED:-true}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TEMPERATURE="${TEMPERATURE:-0.7}"
export MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-4096}"
export MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-512}"
export RATE_LIMIT_PER_MINUTE="${RATE_LIMIT_PER_MINUTE:-60}"

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

# Parse service URLs
parse_service_urls() {
    # Extract host and port from ZKPROOF_SERVICE_URL
    local zkproof_url="${ZKPROOF_SERVICE_URL#http://}"
    zkproof_url="${zkproof_url#https://}"
    ZKPROOF_HOST="${zkproof_url%%:*}"
    ZKPROOF_PORT="${zkproof_url##*:}"
    ZKPROOF_PORT="${ZKPROOF_PORT%%/*}"

    # Extract host and port from REDIS_URL
    local redis_url="${REDIS_URL#redis://}"
    REDIS_HOST="${redis_url%%:*}"
    REDIS_PORT="${redis_url##*:}"

    log_info "Parsed service connections:"
    log_info "  ZK Proof Service: $ZKPROOF_HOST:$ZKPROOF_PORT"
    log_info "  Redis: $REDIS_HOST:$REDIS_PORT"
}

# GPU detection and setup
setup_gpu() {
    if [ "$GPU_ENABLED" = "true" ]; then
        log_info "GPU acceleration enabled, checking for NVIDIA GPU..."

        if command -v nvidia-smi >/dev/null 2>&1; then
            log_info "NVIDIA GPU detected:"
            nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits

            # Set optimal GPU settings
            export CUDA_LAUNCH_BLOCKING=0
            export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
            export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

            log_success "GPU acceleration configured"
        else
            log_warn "GPU acceleration requested but nvidia-smi not found, falling back to CPU"
            export GPU_ENABLED="false"
        fi
    else
        log_info "GPU acceleration disabled, using CPU only"
        export OMP_NUM_THREADS=4
        export MKL_NUM_THREADS=4
    fi
}

# Check model availability
check_model() {
    log_info "Checking model availability..."

    if [ ! -d "$MODEL_PATH" ]; then
        log_warn "Model directory does not exist, creating: $MODEL_PATH"
        mkdir -p "$MODEL_PATH"
    fi

    # List available models
    if [ "$(ls -A $MODEL_PATH 2>/dev/null)" ]; then
        log_info "Found models in $MODEL_PATH:"
        ls -la "$MODEL_PATH" | head -10
    else
        log_warn "No models found in $MODEL_PATH"
        log_warn "Please ensure Llama model files are mounted to $MODEL_PATH"
        log_warn "Expected structure:"
        log_warn "  $MODEL_PATH/"
        log_warn "  ├── llama-3.1-70b/"
        log_warn "  │   ├── tokenizer.model"
        log_warn "  │   ├── consolidated.*.pth"
        log_warn "  │   └── params.json"
        log_warn "Service will continue but may fail during model loading"
    fi
}

# Setup configuration
setup_config() {
    log_info "Setting up configuration..."

    # Create config directory if it doesn't exist
    mkdir -p "$CONFIG_PATH"

    # Create default prompts configuration if it doesn't exist
    if [ ! -f "$CONFIG_PATH/prompts.yaml" ]; then
        log_info "Creating default prompts configuration..."
        cat > "$CONFIG_PATH/prompts.yaml" << 'EOF'
contexts:
  c0:
    name: baseline
    description: No specific trait emphasis
    d0_weight: 0.5
    d1_weight: 0.5
  c1:
    name: emphasize_unverifiable
    description: Emphasize unverifiable traits
    d0_weight: 0.8
    d1_weight: 0.2
  c2:
    name: emphasize_verifiable
    description: Emphasize verifiable traits
    d0_weight: 0.2
    d1_weight: 0.8
  c3:
    name: moderate_verifiable
    description: Moderate emphasis on verifiable traits
    d0_weight: 0.3
    d1_weight: 0.7

domains:
  financial:
    system_prompt: "You are a helpful financial advisor providing personalized investment and financial planning advice."
    emphasis_keywords: ["investment", "risk", "portfolio", "financial goals", "market conditions"]
  healthcare:
    system_prompt: "You are a knowledgeable healthcare advisor providing general health and wellness guidance."
    emphasis_keywords: ["health", "wellness", "symptoms", "lifestyle", "medical history"]
  general:
    system_prompt: "You are a helpful advisor providing personalized guidance and recommendations."
    emphasis_keywords: ["preferences", "situation", "goals", "context", "circumstances"]

response_format:
  max_length: 500
  include_reasoning: true
  formal_tone: false
  structured_response: true
EOF
    fi

    # Create templates directory
    mkdir -p "$CONFIG_PATH/templates"

    log_success "Configuration setup completed"
}

# Health check for dependencies
check_dependencies() {
    log_info "Checking dependencies..."

    # Check Python packages
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

    # Check CUDA availability if GPU enabled
    if [ "$GPU_ENABLED" = "true" ]; then
        python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
        if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
            python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
        else
            log_warn "CUDA not available in PyTorch, falling back to CPU"
            export GPU_ENABLED="false"
        fi
    fi

    log_success "Dependencies check completed"
}

# Setup monitoring
setup_monitoring() {
    log_info "Setting up monitoring..."

    # Create log directory
    mkdir -p /app/logs

    # Setup log rotation
    cat > /app/logs/logrotate.conf << EOF
/app/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 llmuser llmuser
}
EOF

    log_success "Monitoring setup completed"
}

# Validate configuration
validate_config() {
    log_info "Validating configuration..."

    # Check required environment variables
    local required_vars=(
        "MODEL_PATH"
        "ZKPROOF_SERVICE_URL"
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

    # Validate temperature
    if ! python -c "
temp = float('$TEMPERATURE')
if not 0.0 <= temp <= 2.0:
    exit(1)
"; then
        log_error "Invalid temperature: $TEMPERATURE (must be between 0.0 and 2.0)"
        exit 1
    fi

    log_success "Configuration validation passed"
}

# Test model loading
test_model_loading() {
    log_info "Testing model loading..."

    python -c "
import sys
sys.path.append('/app')
from app.config import get_settings, validate_configuration

try:
    settings = get_settings()
    issues = validate_configuration()

    if issues:
        print('Configuration issues found:')
        for issue in issues:
            print(f'  - {issue}')
        sys.exit(1)
    else:
        print('Configuration validation passed')

except Exception as e:
    print(f'Configuration test failed: {e}')
    sys.exit(1)
"

    if [ $? -eq 0 ]; then
        log_success "Model configuration test passed"
    else
        log_error "Model configuration test failed"
        exit 1
    fi
}

# Pre-flight checks
preflight_checks() {
    log_info "Running pre-flight checks..."

    # Check disk space
    local disk_usage=$(df /app | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$disk_usage" -gt 90 ]; then
        log_error "Disk usage critical: ${disk_usage}%"
        exit 1
    elif [ "$disk_usage" -gt 80 ]; then
        log_warn "Disk usage high: ${disk_usage}%"
    fi

    # Check memory
    local memory_gb=$(free -g | awk 'NR==2{printf "%.1f", $2}')
    log_info "Available memory: ${memory_gb}GB"

    if (( $(echo "$memory_gb < 8" | bc -l) )); then
        log_warn "Low memory detected: ${memory_gb}GB (recommended: 16GB+)"
    fi

    log_success "Pre-flight checks completed"
}

# Signal handlers for graceful shutdown
cleanup() {
    log_info "Received shutdown signal, cleaning up..."

    # Kill background processes
    jobs -p | xargs -r kill

    log_info "Cleanup completed"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Main initialization sequence
main() {
    log_info "Starting LLM Advisor Service (Entity 2) initialization..."
    log_info "Version: $(cat /app/VERSION 2>/dev/null || echo 'unknown')"
    log_info "Build time: $(date)"

    # Step 1: Validate configuration
    validate_config

    # Step 2: Parse service URLs
    parse_service_urls

    # Step 3: Setup GPU if enabled
    setup_gpu

    # Step 4: Check dependencies
    check_dependencies

    # Step 5: Wait for external dependencies
    wait_for_service "$ZKPROOF_HOST" "$ZKPROOF_PORT" "ZK Proof Service"
    wait_for_service "$REDIS_HOST" "$REDIS_PORT" "Redis"

    # Step 6: Setup configuration
    setup_config

    # Step 7: Check model availability
    check_model

    # Step 8: Test model configuration
    test_model_loading

    # Step 9: Setup monitoring
    setup_monitoring

    # Step 10: Pre-flight checks
    preflight_checks

    log_success "Initialization completed successfully!"
    log_info "Starting LLM Advisor Service on port $PORT"

    # Execute the main command
    exec "$@"
}

# Check if we're being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
