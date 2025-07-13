#!/usr/bin/env python3
"""
Pydantic models for Entity 2 LLM Advisor Service

Defines all data models, request/response schemas, and validation logic
for the privacy-preserving personalized advice system.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator, root_validator


class RiskCategory(str, Enum):
    """Risk tolerance categories from ZK proof verification"""
    CONSERVATIVE = "conservative"
    STEADY_GROWTH = "steady_growth"
    BALANCED = "balanced"
    AGGRESSIVE_INVESTMENT = "aggressive_investment"


class AgeBracket(str, Enum):
    """Age brackets for privacy-preserving demographics"""
    YOUNG = "young"          # 18-35
    MIDDLE_AGE = "middle_age"  # 36-50
    MATURE = "mature"        # 51-65
    SENIOR = "senior"        # 66+


class IncomeLevel(str, Enum):
    """Income levels for financial advice context"""
    LOW = "low"              # < 50k
    MEDIUM = "medium"        # 50k-100k
    HIGH = "high"            # 100k-200k
    VERY_HIGH = "very_high"  # 200k+


class ExperienceLevel(str, Enum):
    """Investment experience levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class Domain(str, Enum):
    """Advice domains supported by the system"""
    FINANCIAL = "financial"
    HEALTHCARE = "healthcare"
    CAREER = "career"
    EDUCATION = "education"
    LIFESTYLE = "lifestyle"
    GENERAL = "general"


class TraitType(str, Enum):
    """Types of user traits for prompting strategy"""
    VERIFIED = "verified"      # d1: Verifiable traits with ZK proof
    UNVERIFIABLE = "unverifiable"  # d0: Exploratory traits


# Base Models

class BaseResponse(BaseModel):
    """Base response model with common fields"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


class VerifiedTraits(BaseModel):
    """Verified user traits from zero-knowledge proofs"""
    risk_category: Optional[RiskCategory] = None
    confidence_score: Optional[int] = Field(None, ge=0, le=100)
    age_bracket: Optional[AgeBracket] = None
    income_level: Optional[IncomeLevel] = None
    experience_level: Optional[ExperienceLevel] = None

    # Additional verified attributes
    has_dependents: Optional[bool] = None
    has_debt: Optional[bool] = None
    has_insurance: Optional[bool] = None
    time_horizon_years: Optional[int] = Field(None, ge=1, le=50)

    @validator("confidence_score")
    def validate_confidence_score(cls, v):
        if v is not None and not 0 <= v <= 100:
            raise ValueError("Confidence score must be between 0 and 100")
        return v


class UnverifiableTraits(BaseModel):
    """Unverifiable user traits for context enhancement"""
    personality_type: Optional[str] = None
    communication_style: Optional[str] = None
    decision_making_style: Optional[str] = None
    stress_tolerance: Optional[str] = None
    learning_preference: Optional[str] = None
    social_influence: Optional[str] = None

    # Free-form additional context
    additional_context: Optional[Dict[str, Any]] = None
    preferences: Optional[List[str]] = None
    goals: Optional[List[str]] = None
    concerns: Optional[List[str]] = None


class ProofData(BaseModel):
    """Zero-knowledge proof data"""
    proof_id: str
    proof_data: str = Field(..., description="Base64 encoded proof")
    verification_key: str = Field(..., description="Base64 encoded verification key")
    generated_at: datetime
    expires_at: datetime

    @validator("proof_data", "verification_key")
    def validate_base64_fields(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Proof data and verification key cannot be empty")
        return v.strip()


# Request Models

class AdviceRequest(BaseModel):
    """Request for personalized advice generation"""
    query: str = Field(..., min_length=1, max_length=2000, description="User's question or request")
    domain: Optional[Domain] = Field(None, description="Domain for context-specific advice")

    # User identification and session
    user_id: Optional[str] = Field(None, description="Optional user identifier for rate limiting")
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid4()), description="Session identifier")

    # Verified traits from ZK proof
    verified_traits: Optional[VerifiedTraits] = None
    proof: Optional[str] = Field(None, description="Base64 encoded ZK proof")
    verification_key: Optional[str] = Field(None, description="Base64 encoded verification key")

    # Unverifiable traits for enhanced context
    unverifiable_traits: Optional[UnverifiableTraits] = None

    # Request configuration
    max_response_length: Optional[int] = Field(512, ge=50, le=2000)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    include_explanation: bool = Field(True, description="Whether to include explanation")

    @validator("query")
    def validate_query(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Query cannot be empty")
        return v.strip()

    @root_validator
    def validate_proof_consistency(cls, values):
        proof = values.get("proof")
        verification_key = values.get("verification_key")
        verified_traits = values.get("verified_traits")

        # If proof is provided, verification key must also be provided
        if proof and not verification_key:
            raise ValueError("Verification key required when proof is provided")

        # If verification key is provided, proof must also be provided
        if verification_key and not proof:
            raise ValueError("Proof required when verification key is provided")

        # If proof/verification_key provided, verified_traits should be present
        if (proof or verification_key) and not verified_traits:
            raise ValueError("Verified traits should be provided with proof data")

        return values


class ChatRequest(BaseModel):
    """Request for interactive chat conversation"""
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: Optional[str] = None

    # Context from verified traits
    verified_traits: Optional[VerifiedTraits] = None
    domain: Optional[Domain] = None

    # Conversation settings
    maintain_context: bool = Field(True, description="Whether to maintain conversation history")
    max_history_length: Optional[int] = Field(10, ge=1, le=50)

    @validator("message")
    def validate_message(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Message cannot be empty")
        return v.strip()


class ProofVerificationRequest(BaseModel):
    """Request for standalone proof verification"""
    proof_data: str = Field(..., description="Base64 encoded proof")
    verification_key: str = Field(..., description="Base64 encoded verification key")
    expected_traits: Optional[VerifiedTraits] = None

    @validator("proof_data", "verification_key")
    def validate_base64_fields(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Proof data and verification key cannot be empty")
        return v.strip()


# Response Models

class AdviceResponse(BaseResponse):
    """Response containing personalized advice"""
    advice: str = Field(..., description="Generated personalized advice")
    explanation: Optional[str] = Field(None, description="Explanation of the advice reasoning")

    # Metadata
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in the advice")
    domain: Optional[Domain] = None
    response_time_ms: int = Field(..., ge=0, description="Time taken to generate response")

    # Privacy and verification status
    used_verified_traits: bool = Field(False, description="Whether verified traits were used")
    proof_verified: bool = Field(False, description="Whether ZK proof was verified")
    trait_categories_used: List[str] = Field(default_factory=list, description="Categories of traits used")

    # Token usage statistics
    input_tokens: Optional[int] = Field(None, ge=0)
    output_tokens: Optional[int] = Field(None, ge=0)
    total_tokens: Optional[int] = Field(None, ge=0)


class ChatResponse(BaseResponse):
    """Response for chat conversation"""
    response: str = Field(..., description="Chat response")
    session_id: str = Field(..., description="Session identifier")
    message_count: int = Field(..., ge=1, description="Number of messages in conversation")

    # Context status
    used_verified_traits: bool = Field(False)
    context_maintained: bool = Field(True)

    # Conversation metadata
    conversation_summary: Optional[str] = Field(None, description="Brief summary of conversation topic")
    suggested_follow_ups: List[str] = Field(default_factory=list, description="Suggested follow-up questions")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Overall service status")
    version: str = Field(..., description="Service version")
    uptime_seconds: float = Field(..., ge=0)

    # Component status
    llm_ready: bool = Field(..., description="LLM engine readiness")
    cache_connected: bool = Field(..., description="Cache connectivity")
    proof_verifier_connected: bool = Field(..., description="Proof verifier connectivity")

    # Performance metrics
    active_requests: int = Field(0, ge=0)
    total_requests: int = Field(0, ge=0)
    average_response_time_ms: Optional[float] = Field(None, ge=0)

    # Resource usage
    memory_usage_mb: Optional[float] = Field(None, ge=0)
    gpu_memory_usage_mb: Optional[float] = Field(None, ge=0)
    cpu_usage_percent: Optional[float] = Field(None, ge=0, le=100)


class ProofVerificationResponse(BaseModel):
    """Response for proof verification"""
    is_valid: bool = Field(..., description="Whether the proof is valid")
    verification_time_ms: int = Field(..., ge=0, description="Time taken for verification")
    verified_traits: Optional[VerifiedTraits] = Field(None, description="Extracted verified traits")
    error_message: Optional[str] = Field(None, description="Error message if verification failed")


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StatsResponse(BaseModel):
    """Service statistics response"""
    # Request statistics
    total_requests: int = Field(0, ge=0)
    successful_requests: int = Field(0, ge=0)
    failed_requests: int = Field(0, ge=0)

    # Performance statistics
    average_response_time_ms: float = Field(0.0, ge=0)
    median_response_time_ms: float = Field(0.0, ge=0)
    p95_response_time_ms: float = Field(0.0, ge=0)

    # Feature usage statistics
    advice_requests: int = Field(0, ge=0)
    chat_requests: int = Field(0, ge=0)
    proof_verifications: int = Field(0, ge=0)

    # Privacy statistics
    requests_with_verified_traits: int = Field(0, ge=0)
    requests_with_proofs: int = Field(0, ge=0)
    cache_hit_rate: float = Field(0.0, ge=0, le=1)

    # Model statistics
    total_tokens_processed: int = Field(0, ge=0)
    average_tokens_per_request: float = Field(0.0, ge=0)

    # Time period for statistics
    period_start: datetime
    period_end: datetime


# Prompt Strategy Models

class PromptContext(BaseModel):
    """Context for prompt generation"""
    traits_type: TraitType
    domain: Optional[Domain] = None
    emphasis_level: float = Field(0.5, ge=0.0, le=1.0, description="Emphasis on traits (0=none, 1=strong)")

    # Context components
    verified_traits: Optional[VerifiedTraits] = None
    unverifiable_traits: Optional[UnverifiableTraits] = None

    # Prompt configuration
    include_explanations: bool = True
    formal_tone: bool = False
    detailed_response: bool = True


class PromptTemplate(BaseModel):
    """Template for prompt generation"""
    template_id: str
    name: str
    domain: Optional[Domain] = None

    # Template components
    system_prompt: str
    user_prompt_template: str
    context_template: Optional[str] = None

    # Template metadata
    version: str = "1.0"
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

    # Usage constraints
    max_context_length: int = Field(2048, ge=100)
    supports_verified_traits: bool = True
    supports_unverifiable_traits: bool = True


# Cache Models

class CacheEntry(BaseModel):
    """Cache entry for storing responses"""
    key: str
    value: str
    ttl_seconds: int = Field(3600, ge=60)  # Default 1 hour
    created_at: datetime = Field(default_factory=datetime.utcnow)
    accessed_count: int = Field(0, ge=0)
    last_accessed: Optional[datetime] = None

    # Cache metadata
    content_type: str = "application/json"
    tags: List[str] = Field(default_factory=list)

    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if not self.created_at:
            return True

        expiry_time = self.created_at.timestamp() + self.ttl_seconds
        return datetime.utcnow().timestamp() > expiry_time


# Configuration Models

class LLMConfig(BaseModel):
    """LLM engine configuration"""
    model_name: str
    model_path: str
    max_context_length: int = Field(4096, ge=512)
    max_response_tokens: int = Field(512, ge=50)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(40, ge=1)

    # GPU configuration
    gpu_enabled: bool = True
    gpu_memory_fraction: float = Field(0.8, ge=0.1, le=1.0)

    # Performance settings
    batch_size: int = Field(1, ge=1)
    num_threads: Optional[int] = Field(None, ge=1)


class RateLimitConfig(BaseModel):
    """Rate limiting configuration"""
    requests_per_minute: int = Field(60, ge=1)
    requests_per_hour: int = Field(1000, ge=1)
    requests_per_day: int = Field(10000, ge=1)

    # Burst settings
    burst_size: int = Field(10, ge=1)
    burst_window_seconds: int = Field(60, ge=1)

    # Whitelist/blacklist
    whitelisted_users: List[str] = Field(default_factory=list)
    blacklisted_users: List[str] = Field(default_factory=list)
