# Multi-stage build for ZK Proof Service (Entity 1)
# Based on Ubuntu with CUDA support for GPU acceleration

FROM nvidia/cuda:12.1-devel-ubuntu22.04 as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    libpq-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Rust with the latest stable toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set up the workspace
WORKDIR /app

# Copy dependency files first for better layer caching
COPY Cargo.toml Cargo.lock ./
COPY build.rs ./

# Create a dummy main.rs to build dependencies
RUN mkdir -p src && echo "fn main() {}" > src/main.rs && echo "fn main() {}" > src/guest.rs

# Build dependencies (this layer will be cached unless Cargo.toml changes)
RUN cargo build --release && rm -rf src

# Copy the actual source code
COPY src/ ./src/
COPY migrations/ ./migrations/

# Build the application
RUN cargo build --release

# Runtime stage with CUDA runtime
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    libssl3 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN useradd -m -u 1000 zkproof && \
    mkdir -p /app/data /app/logs && \
    chown -R zkproof:zkproof /app

# Copy the built binary from builder stage
COPY --from=builder /app/target/release/server /app/zkproof-service
COPY --from=builder /app/migrations /app/migrations

# Copy configuration and scripts
COPY docker/entrypoint.sh /app/entrypoint.sh
COPY docker/healthcheck.sh /app/healthcheck.sh

# Make scripts executable
RUN chmod +x /app/entrypoint.sh /app/healthcheck.sh

# Set up environment
ENV RUST_LOG=info
ENV RISC0_DEV_MODE=0
ENV CUDA_VISIBLE_DEVICES=0

# Switch to non-root user
USER zkproof

# Set working directory
WORKDIR /app

# Expose the service port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /app/healthcheck.sh

# Entry point
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["./zkproof-service"]
