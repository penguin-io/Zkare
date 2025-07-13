#!/usr/bin/env python3
"""
Configuration module for Entity 2 LLM Advisor Service

Handles all configuration settings, environment variables, and application setup.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from functools import lru_cache

from pydantic import BaseSettings, Field, validator
from pydantic.env_settings import SettingsSourceCallable


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Service Configuration
    service_name: str = Field(default="llm-advisor", env="SERVICE_NAME")
    environment: str = Field(default="production", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    port: int = Field(default=8002, env="PORT")
    host: str = Field(default="0.0.0.0", env="HOST")

    # Model Configuration
    model_path: str = Field(default="/app/models", env="MODEL_PATH")
    model_name: str = Field(default="llama-3.1-70b", env="MODEL_NAME")
    max_context_length: int = Field(default=4096, env="MAX_CONTEXT_LENGTH")
    max_response_tokens: int = Field(default=512, env="MAX_RESPONSE_TOKENS")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    top_p: float = Field(default=0.9, env="TOP_P")
    top_k: int = Field(default=40, env="TOP_K")

    # GPU Configuration
    gpu_enabled: bool = Field(default=True, env="GPU_ENABLED")
    cuda_visible_devices: str = Field(default="0", env="CUDA_VISIBLE_DEVICES")
    gpu_memory_fraction: float = Field(default=0.8, env="GPU_MEMORY_FRACTION")

    # External Services
    zkproof_service_url: str = Field(
        default="http://zkproof-service:8001",
        env="ZKPROOF_SERVICE_URL"
    )
    redis_url: str = Field(default="redis://redis:6379", env="REDIS_URL")
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")

    # Security and Rate Limiting
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    max_request_size: int = Field(default=1024 * 1024, env="MAX_REQUEST_SIZE")  # 1MB
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    jwt_secret: Optional[str] = Field(default=None, env="JWT_SECRET")

    # Caching Configuration
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    conversation_ttl: int = Field(default=86400, env="CONVERSATION_TTL")  # 24 hours
    max_chat_history: int = Field(default=10, env="MAX_CHAT_HISTORY")

    # Prompt Strategy Configuration
    prompt_config_path: str = Field(
        default="/app/config/prompts.yaml",
        env="PROMPT_CONFIG_PATH"
    )
    enable_prompt_caching: bool = Field(default=True, env="ENABLE_PROMPT_CACHING")

    # Monitoring and Observability
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")

    # Tracing Configuration
    jaeger_endpoint: Optional[str] = Field(default=None, env="JAEGER_ENDPOINT")
    enable_tracing: bool = Field(default=False, env="ENABLE_TRACING")
    trace_sample_rate: float = Field(default=0.1, env="TRACE_SAMPLE_RATE")

    # Advanced LLM Configuration
    batch_size: int = Field(default=1, env="BATCH_SIZE")
    max_concurrent_requests: int = Field(default=10, env="MAX_CONCURRENT_REQUESTS")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    model_warmup: bool = Field(default=True, env="MODEL_WARMUP")

    # Privacy and Compliance
    data_retention_days: int = Field(default=30, env="DATA_RETENTION_DAYS")
    enable_audit_logging: bool = Field(default=True, env="ENABLE_AUDIT_LOGGING")
    anonymize_logs: bool = Field(default=True, env="ANONYMIZE_LOGS")

    # Feature Flags
    enable_chat: bool = Field(default=True, env="ENABLE_CHAT")
    enable_advice: bool = Field(default=True, env="ENABLE_ADVICE")
    enable_proof_verification: bool = Field(default=True, env="ENABLE_PROOF_VERIFICATION")
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")

    # Health Check Configuration
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    service_startup_timeout: int = Field(default=300, env="SERVICE_STARTUP_TIMEOUT")

    @validator("model_path")
    def validate_model_path(cls, v):
        """Validate that model path exists"""
        path = Path(v)
        if not path.exists():
            # In production, create the directory if it doesn't exist
            path.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())

    @validator("temperature")
    def validate_temperature(cls, v):
        """Validate temperature is in valid range"""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    @validator("top_p")
    def validate_top_p(cls, v):
        """Validate top_p is in valid range"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")
        return v

    @validator("gpu_memory_fraction")
    def validate_gpu_memory_fraction(cls, v):
        """Validate GPU memory fraction"""
        if not 0.1 <= v <= 1.0:
            raise ValueError("GPU memory fraction must be between 0.1 and 1.0")
        return v

    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    @validator("trace_sample_rate")
    def validate_trace_sample_rate(cls, v):
        """Validate trace sample rate"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Trace sample rate must be between 0.0 and 1.0")
        return v

    def get_full_model_path(self) -> str:
        """Get the full path to the model files"""
        return str(Path(self.model_path) / self.model_name)

    def get_prompt_config_path(self) -> str:
        """Get the full path to prompt configuration"""
        return str(Path(self.prompt_config_path).absolute())

    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment.lower() in ["development", "dev", "local"]

    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment.lower() in ["production", "prod"]

    def get_cors_origins(self) -> List[str]:
        """Get CORS origins based on environment"""
        if self.is_development():
            return ["*"]
        else:
            # In production, specify exact origins
            origins = os.getenv("CORS_ORIGINS", "").split(",")
            return [origin.strip() for origin in origins if origin.strip()]

    def get_trusted_hosts(self) -> List[str]:
        """Get trusted hosts based on environment"""
        if self.is_development():
            return ["*"]
        else:
            hosts = os.getenv("TRUSTED_HOSTS", "").split(",")
            return [host.strip() for host in hosts if host.strip()]

    def get_database_config(self) -> Optional[Dict[str, Any]]:
        """Get database configuration if available"""
        if not self.database_url:
            return None

        return {
            "url": self.database_url,
            "pool_size": int(os.getenv("DB_POOL_SIZE", "10")),
            "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "20")),
            "pool_timeout": int(os.getenv("DB_POOL_TIMEOUT", "30")),
            "pool_recycle": int(os.getenv("DB_POOL_RECYCLE", "3600"))
        }

    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration"""
        return {
            "url": self.redis_url,
            "encoding": "utf-8",
            "decode_responses": True,
            "max_connections": int(os.getenv("REDIS_MAX_CONNECTIONS", "20")),
            "retry_on_timeout": True,
            "socket_timeout": int(os.getenv("REDIS_SOCKET_TIMEOUT", "5")),
            "socket_connect_timeout": int(os.getenv("REDIS_CONNECT_TIMEOUT", "5"))
        }

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        return {
            "model_path": self.get_full_model_path(),
            "max_context_length": self.max_context_length,
            "max_response_tokens": self.max_response_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "gpu_enabled": self.gpu_enabled,
            "gpu_memory_fraction": self.gpu_memory_fraction,
            "batch_size": self.batch_size,
            "model_warmup": self.model_warmup
        }

    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return {
            "api_key": self.api_key,
            "jwt_secret": self.jwt_secret,
            "max_request_size": self.max_request_size,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "request_timeout": self.request_timeout
        }

    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return {
            "enable_metrics": self.enable_metrics,
            "metrics_port": self.metrics_port,
            "enable_tracing": self.enable_tracing,
            "jaeger_endpoint": self.jaeger_endpoint,
            "trace_sample_rate": self.trace_sample_rate,
            "log_level": self.log_level,
            "log_format": self.log_format
        }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

        # Custom environment variable sources
        @classmethod
        def customise_sources(
            cls,
            init_settings: SettingsSourceCallable,
            env_settings: SettingsSourceCallable,
            file_secret_settings: SettingsSourceCallable,
        ) -> tuple[SettingsSourceCallable, ...]:
            return (
                init_settings,
                env_settings,
                file_secret_settings,
            )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


def get_environment_info() -> Dict[str, Any]:
    """Get environment information for debugging"""
    settings = get_settings()

    return {
        "service_name": settings.service_name,
        "environment": settings.environment,
        "debug": settings.debug,
        "model_name": settings.model_name,
        "gpu_enabled": settings.gpu_enabled,
        "cuda_visible_devices": settings.cuda_visible_devices,
        "python_version": os.sys.version,
        "platform": os.name,
        "working_directory": os.getcwd(),
        "environment_variables": {
            key: value for key, value in os.environ.items()
            if not any(sensitive in key.lower() for sensitive in ["password", "secret", "key", "token"])
        }
    }


def validate_configuration() -> List[str]:
    """Validate configuration and return list of issues"""
    issues = []
    settings = get_settings()

    # Check model path
    model_path = Path(settings.get_full_model_path())
    if not model_path.exists():
        issues.append(f"Model path does not exist: {model_path}")

    # Check prompt config path
    prompt_path = Path(settings.get_prompt_config_path())
    if not prompt_path.parent.exists():
        issues.append(f"Prompt config directory does not exist: {prompt_path.parent}")

    # Check GPU configuration
    if settings.gpu_enabled:
        cuda_devices = settings.cuda_visible_devices.split(",")
        try:
            import torch
            if not torch.cuda.is_available():
                issues.append("GPU enabled but CUDA not available")
            elif len(cuda_devices) > torch.cuda.device_count():
                issues.append(f"More CUDA devices specified ({len(cuda_devices)}) than available ({torch.cuda.device_count()})")
        except ImportError:
            issues.append("GPU enabled but PyTorch not available")

    # Check service URLs
    try:
        from urllib.parse import urlparse
        zkproof_url = urlparse(settings.zkproof_service_url)
        if not zkproof_url.scheme or not zkproof_url.netloc:
            issues.append(f"Invalid ZK proof service URL: {settings.zkproof_service_url}")

        redis_url = urlparse(settings.redis_url)
        if not redis_url.scheme or not redis_url.netloc:
            issues.append(f"Invalid Redis URL: {settings.redis_url}")
    except Exception as e:
        issues.append(f"URL validation error: {e}")

    return issues


if __name__ == "__main__":
    # CLI for configuration validation and info
    import json
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        issues = validate_configuration()
        if issues:
            print("Configuration issues found:")
            for issue in issues:
                print(f"  - {issue}")
            sys.exit(1)
        else:
            print("Configuration is valid")
    elif len(sys.argv) > 1 and sys.argv[1] == "info":
        info = get_environment_info()
        print(json.dumps(info, indent=2))
    else:
        settings = get_settings()
        print(json.dumps(settings.dict(), indent=2, default=str))
