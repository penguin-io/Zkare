# PenguLLM Deployment Guide

A comprehensive guide for deploying the privacy-preserving personalized advice system using Zero-Knowledge Proofs and Large Language Models.

## 🎯 Overview

PenguLLM implements the research paper "Generating Privacy-Preserving Personalized Advice with Zero-Knowledge Proofs and LLMs" as a production-ready system with the following components:

- **Entity 1**: ZK Proof Service (Rust + RiscZero zkVM)
- **Entity 2**: LLM Advisor Service (Python + Llama)
- **Web Interface**: Next.js demonstration interface
- **Supporting Services**: PostgreSQL, Redis

## 🔧 Prerequisites

### Hardware Requirements

#### Minimum Configuration
- **CPU**: 8 cores (x86_64)
- **RAM**: 16GB
- **Storage**: 100GB SSD
- **GPU**: None (CPU-only mode)

#### Recommended Configuration
- **CPU**: 16+ cores (x86_64)
- **RAM**: 64GB+
- **Storage**: 500GB NVMe SSD
- **GPU**: NVIDIA A100 (24GB) or RTX 4090

#### Optimal Configuration
- **CPU**: 32+ cores (x86_64)
- **RAM**: 128GB+
- **Storage**: 1TB NVMe SSD
- **GPU**: NVIDIA A100 (80GB) or H100

### Software Requirements

1. **Docker Engine** (≥ 20.10)
2. **Docker Compose** (≥ 2.0)
3. **NVIDIA Container Runtime** (for GPU support)
4. **Git**
5. **OpenSSL**
6. **Basic Unix tools** (curl, nc, bc)

### Model Requirements

Download Llama model files and place them in the `models/` directory:

```
models/
├── llama-3.1-70b/
│   ├── config.json
│   ├── tokenizer.model
│   ├── tokenizer.json
│   ├── pytorch_model-*.bin
│   └── generation_config.json
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd PenguLLM
```

### 2. Install NVIDIA Container Runtime (for GPU support)

#### Ubuntu/Debian
```bash
# Add NVIDIA package repositories
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install NVIDIA Container Runtime
sudo apt-get update
sudo apt-get install -y nvidia-container-runtime

# Restart Docker
sudo systemctl restart docker
```

#### CentOS/RHEL
```bash
# Add NVIDIA package repositories
curl -s -L https://nvidia.github.io/libnvidia-container/centos8/libnvidia-container.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# Install NVIDIA Container Runtime
sudo yum install -y nvidia-container-runtime

# Restart Docker
sudo systemctl restart docker
```

### 3. Prepare Model Files

```bash
# Create models directory
mkdir -p models

# Download Llama models (example - adjust for your access method)
# Option 1: Using Hugging Face Hub
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='meta-llama/Llama-2-70b-chat-hf',
    local_dir='./models/llama-3.1-70b',
    token='your_hf_token'
)
"

# Option 2: Manual placement
# Place your Llama model files in ./models/llama-3.1-70b/
```

### 4. Start the System

```bash
# Make startup script executable
chmod +x start.sh

# Start the complete system
./start.sh start
```

### 5. Access the Services

- **Web Interface**: http://localhost:3000
- **ZK Proof Service**: http://localhost:8001
- **LLM Advisor Service**: http://localhost:8002
- **Health Checks**: Use `./start.sh status`

## 📊 Performance Benchmarks

Based on the paper's evaluation and our testing:

### ZK Proof Generation Performance

| Configuration | Proof Generation | Verification | Memory Usage |
|---------------|------------------|--------------|--------------|
| CPU Only (32 cores) | 51.8s | 0.02s | ~4GB |
| A100 GPU | 1.45s | 0.02s | ~12GB GPU |
| H100 GPU | ~0.8s | 0.02s | ~15GB GPU |

### LLM Inference Performance

| Model Size | Hardware | Tokens/sec | Memory Usage | Response Time |
|------------|----------|------------|--------------|---------------|
| 70B | CPU (32 cores) | 2-5 | 140GB | 10-30s |
| 70B | A100 (80GB) | 15-25 | 45GB | 2-8s |
| 70B | H100 (80GB) | 25-40 | 45GB | 1-5s |

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following configuration:

```env
# Database Configuration
POSTGRES_DB=zkp_llm
POSTGRES_USER=zkp_user
POSTGRES_PASSWORD=your_secure_password
DATABASE_URL=postgres://zkp_user:your_secure_password@postgres:5432/zkp_llm

# Redis Configuration  
REDIS_URL=redis://redis:6379

# Entity 1 (ZK Proof Service)
RUST_LOG=info
RISC0_DEV_MODE=0
ENABLE_GPU=true
RATE_LIMIT=10
CACHE_TTL=3600

# Entity 2 (LLM Service)
MODEL_PATH=/app/models
MODEL_NAME=llama-3.1-70b
GPU_ENABLED=true
CUDA_VISIBLE_DEVICES=0
TEMPERATURE=0.7
MAX_CONTEXT_LENGTH=4096
MAX_RESPONSE_TOKENS=512
RATE_LIMIT_PER_MINUTE=60

# Security
JWT_SECRET=your_jwt_secret_here
API_KEY=your_api_key_here

# Monitoring
ENABLE_METRICS=true
LOG_LEVEL=info
```

### Custom Configuration Files

#### Prompt Configuration (`config/prompts.yaml`)

```yaml
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
```

## 🔒 Security Configuration

### Production Security Checklist

- [ ] Change all default passwords
- [ ] Generate strong JWT secrets
- [ ] Configure TLS/SSL certificates
- [ ] Set up firewall rules
- [ ] Enable audit logging
- [ ] Configure rate limiting
- [ ] Set up backup encryption
- [ ] Review network policies

### SSL/TLS Setup

For production deployment, configure reverse proxy with SSL:

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    
    # ZK Proof Service
    location /api/zkproof/ {
        proxy_pass http://localhost:8001/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # LLM Service
    location /api/llm/ {
        proxy_pass http://localhost:8002/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # Web Interface
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📈 Monitoring and Observability

### Metrics Collection

Access Prometheus metrics at:
- ZK Proof Service: http://localhost:8001/metrics
- LLM Service: http://localhost:8002/metrics

### Key Metrics to Monitor

1. **ZK Proof Generation**
   - Proof generation time
   - Verification time
   - Cache hit rate
   - Error rate

2. **LLM Performance**
   - Inference latency
   - Tokens per second
   - GPU utilization
   - Memory usage

3. **System Health**
   - CPU usage
   - Memory usage
   - Disk I/O
   - Network throughput

### Log Management

Logs are stored in:
- Container logs: `docker-compose logs`
- Application logs: `./logs/`
- Database logs: PostgreSQL container

## 🛠 Troubleshooting

### Common Issues

#### 1. GPU Not Detected

```bash
# Check NVIDIA drivers
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Verify NVIDIA Container Runtime
docker info | grep nvidia
```

#### 2. Out of Memory Errors

```bash
# Check available memory
free -h

# Monitor GPU memory
nvidia-smi

# Adjust model parameters in docker-compose.yml
environment:
  - GPU_MEMORY_FRACTION=0.6
  - MAX_CONTEXT_LENGTH=2048
```

#### 3. Slow Proof Generation

```bash
# Enable GPU acceleration
export RISC0_GPU_ENABLE=1

# Check CUDA version compatibility
nvcc --version

# Monitor GPU utilization during proof generation
nvidia-smi -l 1
```

#### 4. Service Won't Start

```bash
# Check service logs
docker-compose logs zkproof-service
docker-compose logs llm-advisor

# Verify ports are available
netstat -tulpn | grep -E ":(8001|8002|3000|5432|6379)"

# Check disk space
df -h
```

### Performance Optimization

#### For CPU-Only Deployment

```bash
# Optimize for CPU
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
```

#### For GPU Deployment

```bash
# Optimize GPU memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
```

## 📦 Deployment Options

### Option 1: Docker Compose (Recommended)

```bash
# Standard deployment
./start.sh start

# View status
./start.sh status

# View logs
./start.sh logs

# Stop system
./start.sh stop
```

### Option 2: Kubernetes

```yaml
# k8s-deployment.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: pengu-llm

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zkproof-service
  namespace: pengu-llm
spec:
  replicas: 1
  selector:
    matchLabels:
      app: zkproof-service
  template:
    metadata:
      labels:
        app: zkproof-service
    spec:
      containers:
      - name: zkproof-service
        image: pengu-llm/zkproof-service:latest
        ports:
        - containerPort: 8001
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 8Gi
          requests:
            memory: 4Gi
```

### Option 3: Cloud Deployment

#### AWS ECS with GPU instances

```json
{
  "family": "pengu-llm",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "requiresCompatibilities": ["EC2"],
  "cpu": "4096",
  "memory": "16384",
  "containerDefinitions": [
    {
      "name": "zkproof-service",
      "image": "your-ecr-repo/zkproof-service:latest",
      "portMappings": [
        {
          "containerPort": 8001,
          "protocol": "tcp"
        }
      ],
      "resourceRequirements": [
        {
          "type": "GPU",
          "value": "1"
        }
      ]
    }
  ]
}
```

## 🔄 Backup and Recovery

### Database Backup

```bash
# Create backup
docker-compose exec postgres pg_dump -U zkp_user zkp_llm > backup_$(date +%Y%m%d).sql

# Restore backup
docker-compose exec -T postgres psql -U zkp_user zkp_llm < backup_20240101.sql
```

### Model Backup

```bash
# Backup model files
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# Restore models
tar -xzf models_backup_20240101.tar.gz
```

## 📋 Production Checklist

### Pre-deployment

- [ ] Hardware requirements met
- [ ] NVIDIA drivers installed
- [ ] Docker and Docker Compose installed
- [ ] Model files downloaded
- [ ] Configuration files created
- [ ] Security settings configured
- [ ] SSL certificates ready
- [ ] Firewall rules configured

### Post-deployment

- [ ] All services healthy
- [ ] API endpoints responding
- [ ] GPU acceleration working
- [ ] Monitoring configured
- [ ] Backup procedures tested
- [ ] Load testing completed
- [ ] Security audit passed
- [ ] Documentation updated

## 🆘 Support

### Getting Help

1. **Check the logs**: `./start.sh logs`
2. **Verify status**: `./start.sh status`
3. **Review configuration**: Check `.env` and config files
4. **GitHub Issues**: Report bugs and feature requests
5. **Community**: Join our Discord/Slack for support

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### License

This project is licensed under the MIT License. See the LICENSE file for details.

---

**PenguLLM** - Building the future of privacy-preserving AI, one proof at a time. 🐧🔐