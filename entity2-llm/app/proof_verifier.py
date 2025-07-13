#!/usr/bin/env python3
"""
Proof Verifier for Entity 2 LLM Advisor Service

This module handles verification of zero-knowledge proofs received from Entity 1,
ensuring that verified traits are authentic without accessing the underlying data.
"""

import asyncio
import base64
import hashlib
import json
import time
from typing import Dict, Optional, Any, Tuple
from datetime import datetime, timedelta

import httpx
import structlog
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import (
    load_pem_public_key,
    Encoding,
    PublicFormat
)

from .models import VerifiedTraits, RiskCategory, AgeBracket, IncomeLevel, ExperienceLevel

logger = structlog.get_logger()


class ProofVerificationError(Exception):
    """Exception raised when proof verification fails"""
    pass


class ProofVerifier:
    """
    Zero-knowledge proof verifier for Entity 2

    Communicates with Entity 1 to verify proofs and extract verified traits
    without accessing the underlying sensitive data.
    """

    def __init__(
        self,
        zkproof_service_url: str,
        timeout: int = 30,
        max_retries: int = 3,
        cache_ttl: int = 300  # 5 minutes
    ):
        self.zkproof_service_url = zkproof_service_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_ttl = cache_ttl

        # HTTP client for communication with Entity 1
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )

        # Verification cache to avoid redundant verifications
        self.verification_cache: Dict[str, Tuple[bool, datetime, Optional[VerifiedTraits]]] = {}

        # Statistics tracking
        self.stats = {
            "total_verifications": 0,
            "successful_verifications": 0,
            "failed_verifications": 0,
            "cache_hits": 0,
            "average_verification_time": 0.0,
            "last_verification_time": None
        }

    async def verify_proof(
        self,
        proof_data: str,
        verification_key: str,
        expected_traits: Optional[VerifiedTraits] = None,
        proof_id: Optional[str] = None
    ) -> bool:
        """
        Verify a zero-knowledge proof and optionally validate expected traits

        Args:
            proof_data: Base64 encoded proof
            verification_key: Base64 encoded verification key
            expected_traits: Expected verified traits to validate against
            proof_id: Optional proof identifier for tracking

        Returns:
            True if proof is valid, False otherwise
        """
        start_time = time.time()

        try:
            # Check cache first
            cache_key = self._create_cache_key(proof_data, verification_key)
            cached_result = self._get_cached_verification(cache_key)

            if cached_result is not None:
                is_valid, cached_traits = cached_result
                self.stats["cache_hits"] += 1

                # Validate expected traits if provided
                if expected_traits and is_valid:
                    traits_match = self._validate_traits(cached_traits, expected_traits)
                    return is_valid and traits_match

                return is_valid

            # Perform verification
            is_valid, verified_traits = await self._verify_with_entity1(
                proof_data, verification_key, proof_id
            )

            # Cache the result
            self._cache_verification(cache_key, is_valid, verified_traits)

            # Validate expected traits if provided
            if expected_traits and is_valid and verified_traits:
                traits_match = self._validate_traits(verified_traits, expected_traits)
                is_valid = is_valid and traits_match

            # Update statistics
            verification_time = time.time() - start_time
            self._update_stats(is_valid, verification_time)

            logger.info(
                "Proof verification completed",
                proof_id=proof_id,
                is_valid=is_valid,
                verification_time=f"{verification_time:.3f}s",
                cached=False
            )

            return is_valid

        except Exception as e:
            logger.error("Proof verification failed", error=str(e), proof_id=proof_id)
            self.stats["failed_verifications"] += 1
            return False

    async def _verify_with_entity1(
        self,
        proof_data: str,
        verification_key: str,
        proof_id: Optional[str] = None
    ) -> Tuple[bool, Optional[VerifiedTraits]]:
        """Verify proof with Entity 1 service"""

        verification_request = {
            "proof_data": proof_data,
            "verification_key": verification_key
        }

        if proof_id:
            verification_request["proof_id"] = proof_id

        for attempt in range(self.max_retries):
            try:
                response = await self.client.post(
                    f"{self.zkproof_service_url}/verify-proof",
                    json=verification_request
                )

                if response.status_code == 200:
                    result = response.json()
                    is_valid = result.get("is_valid", False)

                    # Extract verified traits if available
                    verified_traits = None
                    if is_valid and "verified_traits" in result:
                        verified_traits = self._parse_verified_traits(result["verified_traits"])

                    return is_valid, verified_traits

                elif response.status_code == 400:
                    logger.warning("Invalid proof format", status_code=response.status_code)
                    return False, None

                else:
                    logger.warning(
                        "Verification request failed",
                        status_code=response.status_code,
                        attempt=attempt + 1
                    )

            except httpx.TimeoutException:
                logger.warning(f"Verification timeout (attempt {attempt + 1})")

            except httpx.RequestError as e:
                logger.warning(f"Verification request error: {e} (attempt {attempt + 1})")

            # Wait before retry
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        return False, None

    def _parse_verified_traits(self, traits_data: Dict[str, Any]) -> Optional[VerifiedTraits]:
        """Parse verified traits from Entity 1 response"""
        try:
            # Convert string enums back to enum objects
            parsed_traits = {}

            if "risk_category" in traits_data:
                parsed_traits["risk_category"] = RiskCategory(traits_data["risk_category"])

            if "age_bracket" in traits_data:
                parsed_traits["age_bracket"] = AgeBracket(traits_data["age_bracket"])

            if "income_level" in traits_data:
                parsed_traits["income_level"] = IncomeLevel(traits_data["income_level"])

            if "experience_level" in traits_data:
                parsed_traits["experience_level"] = ExperienceLevel(traits_data["experience_level"])

            # Copy other fields directly
            for field in ["confidence_score", "has_dependents", "has_debt", "has_insurance", "time_horizon_years"]:
                if field in traits_data:
                    parsed_traits[field] = traits_data[field]

            return VerifiedTraits(**parsed_traits)

        except Exception as e:
            logger.error("Failed to parse verified traits", error=str(e))
            return None

    def _validate_traits(
        self,
        actual_traits: Optional[VerifiedTraits],
        expected_traits: VerifiedTraits
    ) -> bool:
        """Validate that actual traits match expected traits"""
        if not actual_traits:
            return False

        # Compare key fields
        comparisons = [
            actual_traits.risk_category == expected_traits.risk_category,
            actual_traits.age_bracket == expected_traits.age_bracket,
            actual_traits.income_level == expected_traits.income_level,
            actual_traits.experience_level == expected_traits.experience_level
        ]

        # Allow some tolerance for confidence score
        if (actual_traits.confidence_score is not None and
            expected_traits.confidence_score is not None):
            confidence_diff = abs(actual_traits.confidence_score - expected_traits.confidence_score)
            comparisons.append(confidence_diff <= 5)  # 5% tolerance

        # At least 80% of comparisons must match
        valid_comparisons = [c for c in comparisons if c is not False]
        match_rate = len(valid_comparisons) / len(comparisons)

        return match_rate >= 0.8

    def _create_cache_key(self, proof_data: str, verification_key: str) -> str:
        """Create cache key for verification result"""
        combined = f"{proof_data}:{verification_key}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _get_cached_verification(
        self,
        cache_key: str
    ) -> Optional[Tuple[bool, Optional[VerifiedTraits]]]:
        """Get cached verification result if still valid"""
        if cache_key not in self.verification_cache:
            return None

        is_valid, timestamp, traits = self.verification_cache[cache_key]

        # Check if cache entry is expired
        if datetime.utcnow() - timestamp > timedelta(seconds=self.cache_ttl):
            del self.verification_cache[cache_key]
            return None

        return is_valid, traits

    def _cache_verification(
        self,
        cache_key: str,
        is_valid: bool,
        verified_traits: Optional[VerifiedTraits]
    ) -> None:
        """Cache verification result"""
        # Limit cache size
        if len(self.verification_cache) >= 1000:
            # Remove oldest entries
            oldest_keys = sorted(
                self.verification_cache.keys(),
                key=lambda k: self.verification_cache[k][1]
            )[:100]

            for key in oldest_keys:
                del self.verification_cache[key]

        self.verification_cache[cache_key] = (is_valid, datetime.utcnow(), verified_traits)

    def _update_stats(self, is_valid: bool, verification_time: float) -> None:
        """Update verification statistics"""
        self.stats["total_verifications"] += 1

        if is_valid:
            self.stats["successful_verifications"] += 1
        else:
            self.stats["failed_verifications"] += 1

        # Update average verification time
        total_time = (self.stats["average_verification_time"] *
                     (self.stats["total_verifications"] - 1) + verification_time)
        self.stats["average_verification_time"] = total_time / self.stats["total_verifications"]

        self.stats["last_verification_time"] = datetime.utcnow()

    async def verify_proof_integrity(self, proof_data: str) -> bool:
        """
        Verify basic proof integrity without full verification

        This is a lightweight check to ensure the proof data is well-formed
        before attempting full verification.
        """
        try:
            # Decode base64 proof data
            decoded_proof = base64.b64decode(proof_data)

            # Basic size checks
            if len(decoded_proof) < 32:  # Minimum expected proof size
                return False

            if len(decoded_proof) > 1024 * 1024:  # Maximum 1MB
                return False

            # Check for valid structure (this would be RiscZero specific)
            # For now, just verify it's valid base64 and reasonable size
            return True

        except Exception:
            return False

    async def batch_verify_proofs(
        self,
        proof_requests: list[Dict[str, str]]
    ) -> list[bool]:
        """Verify multiple proofs in batch"""
        tasks = []

        for request in proof_requests:
            task = self.verify_proof(
                proof_data=request.get("proof_data", ""),
                verification_key=request.get("verification_key", ""),
                proof_id=request.get("proof_id")
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to False
        return [result if isinstance(result, bool) else False for result in results]

    async def health_check(self) -> bool:
        """Check if the proof verifier can communicate with Entity 1"""
        try:
            response = await self.client.get(
                f"{self.zkproof_service_url}/health",
                timeout=5.0
            )

            if response.status_code == 200:
                health_data = response.json()
                return health_data.get("status") in ["healthy", "degraded"]

            return False

        except Exception as e:
            logger.warning("Health check failed", error=str(e))
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get verification statistics"""
        success_rate = 0.0
        if self.stats["total_verifications"] > 0:
            success_rate = (self.stats["successful_verifications"] /
                          self.stats["total_verifications"])

        cache_hit_rate = 0.0
        if self.stats["total_verifications"] > 0:
            cache_hit_rate = self.stats["cache_hits"] / self.stats["total_verifications"]

        return {
            **self.stats,
            "success_rate": success_rate,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.verification_cache),
            "entity1_url": self.zkproof_service_url
        }

    def clear_cache(self) -> None:
        """Clear verification cache"""
        self.verification_cache.clear()
        logger.info("Verification cache cleared")

    async def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            await self.client.aclose()
            self.clear_cache()
            logger.info("Proof verifier cleanup completed")
        except Exception as e:
            logger.error("Error during proof verifier cleanup", error=str(e))

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            if hasattr(self, 'client') and self.client:
                # Note: This won't work in async context, but prevents warnings
                pass
        except:
            pass


class MockProofVerifier(ProofVerifier):
    """Mock proof verifier for testing and development"""

    def __init__(self, always_valid: bool = True):
        # Don't call parent __init__ to avoid setting up HTTP client
        self.always_valid = always_valid
        self.verification_cache = {}
        self.stats = {
            "total_verifications": 0,
            "successful_verifications": 0,
            "failed_verifications": 0,
            "cache_hits": 0,
            "average_verification_time": 0.001,
            "last_verification_time": None
        }

    async def verify_proof(
        self,
        proof_data: str,
        verification_key: str,
        expected_traits: Optional[VerifiedTraits] = None,
        proof_id: Optional[str] = None
    ) -> bool:
        """Mock verification that returns predetermined result"""

        # Simulate some processing time
        await asyncio.sleep(0.001)

        # Update stats
        self.stats["total_verifications"] += 1

        if self.always_valid:
            self.stats["successful_verifications"] += 1
            return True
        else:
            self.stats["failed_verifications"] += 1
            return False

    async def health_check(self) -> bool:
        """Mock health check"""
        return True

    async def cleanup(self) -> None:
        """Mock cleanup"""
        pass
