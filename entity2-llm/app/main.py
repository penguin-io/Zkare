#!/usr/bin/env python3
"""
Entity 2: LLM Advisor Service

This service implements the privacy-preserving personalized advice system
using Llama LLM and zero-knowledge proof verification as described in the paper.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Union

import httpx
import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app, Counter, Histogram, Gauge

from .config import Settings, get_settings
from .models import (
    AdviceRequest,
    AdviceResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ProofVerificationRequest,
    VerifiedTraits,
)
from .llm_engine import LlamaEngine
from .prompt_strategy import PromptStrategy
from .proof_verifier import ProofVerifier
from .cache import CacheManager
from .rate_limiter import RateLimiter

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter(
    'llm_advisor_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'llm_advisor_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'llm_advisor_active_connections',
    'Number of active connections'
)

ADVICE_GENERATION_TIME = Histogram(
    'llm_advisor_advice_generation_seconds',
    'Time taken to generate advice',
    ['domain', 'traits_type']
)

PROOF_VERIFICATION_TIME = Histogram(
    'llm_advisor_proof_verification_seconds',
    'Time taken to verify proofs'
)

# Global state
llm_engine: Optional[LlamaEngine] = None
prompt_strategy: Optional[PromptStrategy] = None
proof_verifier: Optional[ProofVerifier] = None
cache_manager: Optional[CacheManager] = None
rate_limiter: Optional[RateLimiter] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global llm_engine, prompt_strategy, proof_verifier, cache_manager, rate_limiter

    settings = get_settings()

    logger.info("Starting LLM Advisor Service initialization")

    try:
        # Initialize cache manager
        cache_manager = CacheManager(settings.redis_url)
        await cache_manager.initialize()
        logger.info("Cache manager initialized")

        # Initialize rate limiter
        rate_limiter = RateLimiter(cache_manager, settings.rate_limit_per_minute)
        logger.info("Rate limiter initialized")

        # Initialize proof verifier
        proof_verifier = ProofVerifier(settings.zkproof_service_url)
        logger.info("Proof verifier initialized")

        # Initialize LLM engine
        llm_engine = LlamaEngine(
            model_path=settings.model_path,
            max_context_length=settings.max_context_length,
            temperature=settings.temperature,
            gpu_enabled=settings.gpu_enabled
        )
        await llm_engine.initialize()
        logger.info("LLM engine initialized")

        # Initialize prompt strategy
        prompt_strategy = PromptStrategy(settings.prompt_config_path)
        logger.info("Prompt strategy initialized")

        logger.info("LLM Advisor Service initialization completed successfully")

        yield

    except Exception as e:
        logger.error("Failed to initialize LLM Advisor Service", error=str(e))
        raise
    finally:
        # Cleanup
        if llm_engine:
            await llm_engine.cleanup()
        if cache_manager:
            await cache_manager.cleanup()
        logger.info("LLM Advisor Service shutdown completed")


# Create FastAPI application
app = FastAPI(
    title="LLM Advisor Service (Entity 2)",
    description="Privacy-preserving personalized advice system using Zero-Knowledge Proofs and LLMs",
    version="0.1.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Add processing time and metrics"""
    start_time = time.time()
    ACTIVE_CONNECTIONS.inc()

    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)

        # Record metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()

        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(process_time)

        return response

    finally:
        ACTIVE_CONNECTIONS.dec()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check LLM engine status
        llm_status = llm_engine.is_ready() if llm_engine else False

        # Check cache connectivity
        cache_status = await cache_manager.health_check() if cache_manager else False

        # Check proof verifier connectivity
        proof_verifier_status = await proof_verifier.health_check() if proof_verifier else False

        overall_status = llm_status and cache_status and proof_verifier_status

        return HealthResponse(
            status="healthy" if overall_status else "degraded",
            version="0.1.0",
            llm_ready=llm_status,
            cache_connected=cache_status,
            proof_verifier_connected=proof_verifier_status,
            uptime_seconds=time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
        )

    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/advice", response_model=AdviceResponse)
async def generate_advice(
    request: AdviceRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings)
):
    """Generate personalized advice using verified traits and ZK proofs"""

    start_time = time.time()

    try:
        # Rate limiting
        if rate_limiter:
            await rate_limiter.check_rate_limit(request.user_id or "anonymous")

        # Verify zero-knowledge proof
        verification_start = time.time()
        if request.proof and request.verification_key:
            is_valid = await proof_verifier.verify_proof(
                proof_data=request.proof,
                verification_key=request.verification_key,
                expected_traits=request.verified_traits
            )

            if not is_valid:
                raise HTTPException(status_code=400, detail="Invalid zero-knowledge proof")

        verification_time = time.time() - verification_start
        PROOF_VERIFICATION_TIME.observe(verification_time)

        # Check cache for similar request
        cache_key = f"advice:{hash(str(request.dict()))}"
        cached_response = await cache_manager.get(cache_key) if cache_manager else None

        if cached_response:
            logger.info("Returning cached advice", cache_key=cache_key)
            return AdviceResponse.parse_raw(cached_response)

        # Generate advice using prompt strategy
        advice_start = time.time()

        # Create context based on verified and unverifiable traits
        context = prompt_strategy.create_context(
            verified_traits=request.verified_traits,
            unverifiable_traits=request.unverifiable_traits,
            domain=request.domain
        )

        # Generate proposed answer
        proposed_answer = await llm_engine.generate(
            prompt=prompt_strategy.create_proposal_prompt(
                query=request.query,
                context=context["proposal"]
            ),
            max_tokens=settings.max_response_tokens,
            temperature=settings.temperature
        )

        # Generate explanation
        explanation = await llm_engine.generate(
            prompt=prompt_strategy.create_explanation_prompt(
                query=request.query,
                proposed_answer=proposed_answer,
                context=context["explanation"]
            ),
            max_tokens=settings.max_response_tokens,
            temperature=settings.temperature
        )

        advice_time = time.time() - advice_start
        ADVICE_GENERATION_TIME.labels(
            domain=request.domain or "general",
            traits_type="verified" if request.verified_traits else "unverified"
        ).observe(advice_time)

        # Create response
        response = AdviceResponse(
            advice=proposed_answer,
            explanation=explanation,
            confidence_score=_calculate_confidence_score(request, verification_time, advice_time),
            domain=request.domain,
            response_time_ms=int((time.time() - start_time) * 1000),
            used_verified_traits=bool(request.verified_traits),
            proof_verified=bool(request.proof and request.verification_key)
        )

        # Cache the response
        if cache_manager:
            background_tasks.add_task(
                cache_manager.set,
                cache_key,
                response.json(),
                expire=settings.cache_ttl
            )

        # Log advice generation
        background_tasks.add_task(
            _log_advice_generation,
            request,
            response,
            verification_time,
            advice_time
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to generate advice", error=str(e), request_id=request.user_id)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/chat", response_model=ChatResponse)
async def chat_with_advisor(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings)
):
    """Interactive chat with privacy-preserving context"""

    try:
        # Rate limiting
        if rate_limiter:
            await rate_limiter.check_rate_limit(request.user_id or "anonymous")

        # Get conversation history from cache
        conversation_key = f"conversation:{request.session_id}"
        conversation_history = await cache_manager.get(conversation_key) if cache_manager else None

        if conversation_history:
            history = eval(conversation_history)  # In production, use proper JSON parsing
        else:
            history = []

        # Add current message to history
        history.append({"role": "user", "content": request.message})

        # Create context with verified traits if available
        context = ""
        if request.verified_traits:
            context = prompt_strategy.create_chat_context(request.verified_traits)

        # Generate response
        chat_prompt = prompt_strategy.create_chat_prompt(
            message=request.message,
            history=history[-settings.max_chat_history:],  # Limit history
            context=context
        )

        response_text = await llm_engine.generate(
            prompt=chat_prompt,
            max_tokens=settings.max_response_tokens,
            temperature=settings.temperature
        )

        # Add assistant response to history
        history.append({"role": "assistant", "content": response_text})

        # Update conversation cache
        if cache_manager:
            background_tasks.add_task(
                cache_manager.set,
                conversation_key,
                str(history),
                expire=settings.conversation_ttl
            )

        return ChatResponse(
            response=response_text,
            session_id=request.session_id,
            message_count=len(history),
            used_verified_traits=bool(request.verified_traits)
        )

    except Exception as e:
        logger.error("Failed to process chat", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/verify-proof")
async def verify_proof_endpoint(request: ProofVerificationRequest):
    """Standalone proof verification endpoint"""

    try:
        is_valid = await proof_verifier.verify_proof(
            proof_data=request.proof_data,
            verification_key=request.verification_key
        )

        return {"is_valid": is_valid}

    except Exception as e:
        logger.error("Failed to verify proof", error=str(e))
        raise HTTPException(status_code=400, detail="Proof verification failed")


@app.get("/stats")
async def get_statistics():
    """Get service statistics"""

    try:
        stats = {
            "total_requests": REQUEST_COUNT._value.sum(),
            "active_connections": ACTIVE_CONNECTIONS._value.get(),
            "average_response_time": REQUEST_DURATION._sum.sum() / max(REQUEST_DURATION._count.sum(), 1),
            "llm_status": llm_engine.get_stats() if llm_engine else {},
            "cache_stats": await cache_manager.get_stats() if cache_manager else {}
        }

        return stats

    except Exception as e:
        logger.error("Failed to get statistics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get statistics")


def _calculate_confidence_score(
    request: AdviceRequest,
    verification_time: float,
    advice_time: float
) -> float:
    """Calculate confidence score based on various factors"""

    base_confidence = 0.7

    # Boost confidence if verified traits are used
    if request.verified_traits:
        base_confidence += 0.2

    # Boost confidence if proof is verified
    if request.proof and request.verification_key:
        base_confidence += 0.1

    # Reduce confidence for very fast responses (might indicate cached/simple responses)
    if advice_time < 0.1:
        base_confidence -= 0.1

    # Reduce confidence for very slow verification
    if verification_time > 1.0:
        base_confidence -= 0.05

    return min(max(base_confidence, 0.0), 1.0)


async def _log_advice_generation(
    request: AdviceRequest,
    response: AdviceResponse,
    verification_time: float,
    advice_time: float
):
    """Log advice generation for analytics"""

    logger.info(
        "Advice generated",
        user_id=request.user_id,
        domain=request.domain,
        used_verified_traits=response.used_verified_traits,
        proof_verified=response.proof_verified,
        confidence_score=response.confidence_score,
        verification_time_ms=int(verification_time * 1000),
        advice_time_ms=int(advice_time * 1000),
        response_time_ms=response.response_time_ms
    )


if __name__ == "__main__":
    # Set start time for uptime calculation
    app.state.start_time = time.time()

    # Run the application
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8002")),
        reload=os.getenv("ENVIRONMENT") == "development",
        log_level="info"
    )
