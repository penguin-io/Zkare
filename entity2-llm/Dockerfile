# Multi-stage build for Entity 2 LLM Advisor Service
# Based on NVIDIA PyTorch image with CUDA support for GPU acceleration

FROM nvidia/cuda:12.1-devel-ubuntu22.04 as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
FROM base as pytorch-install
RUN pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121

# Main application stage
FROM pytorch-install as app

# Set working directory
WORKDIR /app

# Create app user for security
RUN useradd -m -u 1000 llmuser && \
    mkdir -p /app/models /app/config /app/logs /app/cache && \
    chown -R llmuser:llmuser /app

# Copy requirements first for better layer caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ /app/app/
COPY config/ /app/config/

# Copy entrypoint and health check scripts
COPY docker/entrypoint.sh /app/entrypoint.sh
COPY docker/healthcheck.sh /app/healthcheck.sh

# Make scripts executable
RUN chmod +x /app/entrypoint.sh /app/healthcheck.sh

# Create version file
RUN echo "0.1.0" > /app/VERSION

# Set up environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models
ENV CONFIG_PATH=/app/config
ENV CUDA_VISIBLE_DEVICES=0
ENV TOKENIZERS_PARALLELISM=false

# GPU optimization environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Switch to non-root user
USER llmuser

# Expose service port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD /app/healthcheck.sh

# Entry point
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "-m", "app.main"]
