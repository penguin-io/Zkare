# Environment Setup Guide

This guide explains how to configure the environment variables for the Zero Knowledge Llama LLM deployment system.

## Quick Start

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit the `.env` file with your secure values:**
   ```bash
   nano .env
   ```

3. **Update the critical security values** (see sections below)

## Critical Security Settings

### Database Passwords
**MUST CHANGE** these default passwords before deployment:

```bash
# PostgreSQL - Change this password!
POSTGRES_PASSWORD=your_secure_postgres_password_here

# Redis - Change this password!
REDIS_PASSWORD=your_redis_password_here
```

### JWT and API Secrets
**MUST CHANGE** these secrets (minimum 32 characters):

```bash
# JWT Secret for token signing
JWT_SECRET=your_jwt_secret_key_here_min_32_chars

# Session secret for web sessions
SESSION_SECRET=your_session_secret_here_min_32_chars

# API key secret
API_KEY_SECRET=your_api_key_secret_here
```

## Environment Variables Reference

### Database Configuration
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `POSTGRES_DB` | PostgreSQL database name | `zkp_llm` | Yes |
| `POSTGRES_USER` | PostgreSQL username | `zkp_user` | Yes |
| `POSTGRES_PASSWORD` | PostgreSQL password | - | **Yes** |
| `POSTGRES_HOST` | PostgreSQL host | `postgres-db` | Yes |
| `POSTGRES_PORT` | PostgreSQL port | `5432` | Yes |
| `DATABASE_URL` | Full database connection string | Auto-generated | Yes |

### Redis Configuration
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `REDIS_HOST` | Redis host | `redis-cache` | Yes |
| `REDIS_PORT` | Redis port | `6379` | Yes |
| `REDIS_PASSWORD` | Redis password | - | **Yes** |
| `REDIS_URL` | Full Redis connection string | Auto-generated | Yes |

### Service Ports
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ZK_SERVICE_PORT` | ZK Proof service port | `8001` | Yes |
| `LLM_SERVICE_PORT` | LLM Advisor service port | `8002` | Yes |
| `WEB_PORT` | Web interface port | `3000` | Yes |

### GPU Configuration
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `CUDA_VISIBLE_DEVICES` | GPU device ID | `0` | Yes |
| `NVIDIA_VISIBLE_DEVICES` | NVIDIA GPU visibility | `all` | Yes |

### Model Configuration
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MODEL_NAME` | LLM model name | `llama-2-7b-chat` | Yes |
| `MODEL_PATH` | Path to model files | `/app/models` | Yes |
| `MODEL_MAX_LENGTH` | Maximum sequence length | `4096` | No |
| `MODEL_TEMPERATURE` | Sampling temperature | `0.7` | No |
| `MODEL_TOP_P` | Top-p sampling | `0.9` | No |
| `MODEL_BATCH_SIZE` | Batch size | `1` | No |

### Security Settings
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `JWT_SECRET` | JWT signing secret | - | **Yes** |
| `SESSION_SECRET` | Session encryption secret | - | **Yes** |
| `API_KEY_SECRET` | API key encryption secret | - | **Yes** |

### Performance Tuning
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MAX_CONCURRENT_PROOFS` | Max concurrent ZK proofs | `5` | No |
| `MAX_CONCURRENT_LLM_REQUESTS` | Max concurrent LLM requests | `10` | No |
| `WORKER_THREADS` | Number of worker threads | `4` | No |
| `ZK_PROOF_CACHE_TTL` | Proof cache TTL (seconds) | `3600` | No |

### Rate Limiting
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `RATE_LIMIT_WINDOW_MS` | Rate limit window (ms) | `900000` | No |
| `RATE_LIMIT_MAX_REQUESTS` | Max requests per window | `100` | No |

### Logging and Monitoring
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `LOG_LEVEL` | Logging level | `info` | No |
| `RUST_LOG` | Rust logging level | `info` | No |
| `ENABLE_METRICS` | Enable Prometheus metrics | `true` | No |
| `METRICS_PORT` | Metrics endpoint port | `9090` | No |

## Environment-Specific Configurations

### Development Environment
```bash
# Development settings
DEBUG_MODE=true
LOG_LEVEL=debug
RUST_LOG=debug
RISC0_DEV_MODE=1
ENABLE_CORS=true
```

### Production Environment
```bash
# Production settings
DEBUG_MODE=false
LOG_LEVEL=info
RUST_LOG=info
RISC0_DEV_MODE=0
ENABLE_CORS=false
```

### GPU Configuration
For different GPU setups:

```bash
# Single GPU (default)
CUDA_VISIBLE_DEVICES=0

# Multiple GPUs
CUDA_VISIBLE_DEVICES=0,1

# Specific GPU
CUDA_VISIBLE_DEVICES=1

# CPU-only (not recommended)
CUDA_VISIBLE_DEVICES=""
```

## Security Best Practices

### 1. Password Generation
Generate strong passwords using:
```bash
# Generate a 32-character password
openssl rand -base64 32

# Generate a hex password
openssl rand -hex 32
```

### 2. JWT Secret Generation
```bash
# Generate a secure JWT secret
openssl rand -base64 64
```

### 3. File Permissions
Secure your environment files:
```bash
chmod 600 .env
chown root:root .env
```

### 4. Environment Isolation
Use different `.env` files for different environments:
- `.env` - Local development
- `.env.staging` - Staging environment
- `.env.production` - Production environment

## Common Issues and Solutions

### Issue: Database Connection Failed
**Solution:** Check PostgreSQL credentials and ensure the database is running:
```bash
docker-compose logs postgres
```

### Issue: GPU Not Detected
**Solution:** Verify NVIDIA Docker runtime:
```bash
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Issue: Model Loading Failed
**Solution:** Ensure model files are in the correct path:
```bash
ls -la ./models/
```

### Issue: Redis Connection Failed
**Solution:** Check Redis password and connection:
```bash
docker-compose logs redis
```

## Validation Script

Create a simple script to validate your environment:

```bash
#!/bin/bash
# validate_env.sh

echo "Validating environment configuration..."

# Check required files
if [ ! -f .env ]; then
    echo "❌ .env file not found"
    exit 1
fi

# Check critical variables
source .env

if [ -z "$POSTGRES_PASSWORD" ]; then
    echo "❌ POSTGRES_PASSWORD not set"
    exit 1
fi

if [ -z "$JWT_SECRET" ]; then
    echo "❌ JWT_SECRET not set"
    exit 1
fi

if [ ${#JWT_SECRET} -lt 32 ]; then
    echo "❌ JWT_SECRET too short (minimum 32 characters)"
    exit 1
fi

echo "✅ Environment validation passed"
```

Run the validation:
```bash
chmod +x validate_env.sh
./validate_env.sh
```

## Support

If you encounter issues with environment configuration:

1. Check the logs: `docker-compose logs`
2. Validate your `.env` file syntax
3. Ensure all required variables are set
4. Verify file permissions
5. Check Docker and GPU runtime setup

For more help, see the main [README.md](./README.md) and [DEPLOYMENT.md](./DEPLOYMENT.md) files.