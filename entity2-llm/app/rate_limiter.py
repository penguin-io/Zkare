#!/usr/bin/env python3
"""
Rate Limiter for Entity 2 LLM Advisor Service

This module provides rate limiting functionality to prevent abuse and ensure
fair resource usage across users and sessions.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from enum import Enum

import structlog

from .cache import CacheManager

logger = structlog.get_logger()


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded"""
    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(message)
        self.retry_after = retry_after


class RateLimitType(str, Enum):
    """Types of rate limits"""
    PER_MINUTE = "per_minute"
    PER_HOUR = "per_hour"
    PER_DAY = "per_day"
    BURST = "burst"


class RateLimiter:
    """
    Redis-based distributed rate limiter with multiple time windows

    Supports different rate limiting strategies:
    - Fixed window
    - Sliding window
    - Token bucket
    - Leaky bucket
    """

    def __init__(
        self,
        cache_manager: CacheManager,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        requests_per_day: int = 10000,
        burst_size: int = 10,
        burst_window: int = 60,
        algorithm: str = "sliding_window"
    ):
        self.cache_manager = cache_manager
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests_per_day = requests_per_day
        self.burst_size = burst_size
        self.burst_window = burst_window
        self.algorithm = algorithm

        # Rate limit configurations
        self.limits = {
            RateLimitType.PER_MINUTE: {
                "limit": requests_per_minute,
                "window": 60,
                "key_suffix": "min"
            },
            RateLimitType.PER_HOUR: {
                "limit": requests_per_hour,
                "window": 3600,
                "key_suffix": "hour"
            },
            RateLimitType.PER_DAY: {
                "limit": requests_per_day,
                "window": 86400,
                "key_suffix": "day"
            },
            RateLimitType.BURST: {
                "limit": burst_size,
                "window": burst_window,
                "key_suffix": "burst"
            }
        }

        # Statistics
        self.stats = {
            "total_requests": 0,
            "allowed_requests": 0,
            "blocked_requests": 0,
            "rate_limit_hits": 0
        }

        # Whitelist and blacklist
        self.whitelisted_users: set = set()
        self.blacklisted_users: set = set()

    async def check_rate_limit(
        self,
        identifier: str,
        resource: str = "default",
        cost: int = 1
    ) -> bool:
        """
        Check if request is within rate limits

        Args:
            identifier: User/client identifier
            resource: Resource being accessed
            cost: Cost of the request (for weighted rate limiting)

        Returns:
            True if request is allowed, False otherwise

        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        self.stats["total_requests"] += 1

        # Check blacklist first
        if identifier in self.blacklisted_users:
            self.stats["blocked_requests"] += 1
            raise RateLimitExceeded(
                f"User {identifier} is blacklisted",
                retry_after=3600
            )

        # Check whitelist
        if identifier in self.whitelisted_users:
            self.stats["allowed_requests"] += 1
            return True

        # Check all rate limit types
        for limit_type, config in self.limits.items():
            try:
                if not await self._check_limit(
                    identifier, resource, limit_type, config, cost
                ):
                    self.stats["blocked_requests"] += 1
                    self.stats["rate_limit_hits"] += 1

                    raise RateLimitExceeded(
                        f"Rate limit exceeded for {limit_type.value}: "
                        f"{config['limit']} requests per {config['window']} seconds",
                        retry_after=config['window']
                    )
            except RateLimitExceeded:
                raise
            except Exception as e:
                logger.warning(
                    "Rate limit check failed",
                    identifier=identifier,
                    limit_type=limit_type,
                    error=str(e)
                )
                # Allow request on rate limiter failure
                continue

        self.stats["allowed_requests"] += 1
        return True

    async def _check_limit(
        self,
        identifier: str,
        resource: str,
        limit_type: RateLimitType,
        config: Dict,
        cost: int
    ) -> bool:
        """Check specific rate limit"""
        if self.algorithm == "sliding_window":
            return await self._sliding_window_check(
                identifier, resource, limit_type, config, cost
            )
        elif self.algorithm == "fixed_window":
            return await self._fixed_window_check(
                identifier, resource, limit_type, config, cost
            )
        elif self.algorithm == "token_bucket":
            return await self._token_bucket_check(
                identifier, resource, limit_type, config, cost
            )
        else:
            # Default to sliding window
            return await self._sliding_window_check(
                identifier, resource, limit_type, config, cost
            )

    async def _sliding_window_check(
        self,
        identifier: str,
        resource: str,
        limit_type: RateLimitType,
        config: Dict,
        cost: int
    ) -> bool:
        """Sliding window rate limiting using sorted sets"""
        key = f"rate_limit:sliding:{identifier}:{resource}:{config['key_suffix']}"
        window = config["window"]
        limit = config["limit"]
        now = time.time()
        window_start = now - window

        try:
            # Remove expired entries
            await self.cache_manager.redis_client.zremrangebyscore(
                key, 0, window_start
            )

            # Count current requests
            current_count = await self.cache_manager.redis_client.zcard(key)

            if current_count + cost > limit:
                return False

            # Add current request
            await self.cache_manager.redis_client.zadd(
                key, {str(now): now}
            )

            # Set expiration
            await self.cache_manager.redis_client.expire(key, window)

            return True

        except Exception as e:
            logger.error(
                "Sliding window rate limit check failed",
                identifier=identifier,
                error=str(e)
            )
            return True  # Allow on error

    async def _fixed_window_check(
        self,
        identifier: str,
        resource: str,
        limit_type: RateLimitType,
        config: Dict,
        cost: int
    ) -> bool:
        """Fixed window rate limiting"""
        window = config["window"]
        limit = config["limit"]
        now = int(time.time())
        window_start = (now // window) * window

        key = f"rate_limit:fixed:{identifier}:{resource}:{config['key_suffix']}:{window_start}"

        try:
            current_count = await self.cache_manager.redis_client.get(key)
            current_count = int(current_count) if current_count else 0

            if current_count + cost > limit:
                return False

            # Increment counter
            await self.cache_manager.redis_client.incr(key, cost)
            await self.cache_manager.redis_client.expire(key, window)

            return True

        except Exception as e:
            logger.error(
                "Fixed window rate limit check failed",
                identifier=identifier,
                error=str(e)
            )
            return True  # Allow on error

    async def _token_bucket_check(
        self,
        identifier: str,
        resource: str,
        limit_type: RateLimitType,
        config: Dict,
        cost: int
    ) -> bool:
        """Token bucket rate limiting"""
        key = f"rate_limit:bucket:{identifier}:{resource}:{config['key_suffix']}"
        capacity = config["limit"]
        refill_rate = capacity / config["window"]  # tokens per second
        now = time.time()

        try:
            bucket_data = await self.cache_manager.redis_client.hmget(
                key, "tokens", "last_refill"
            )

            tokens = float(bucket_data[0]) if bucket_data[0] else capacity
            last_refill = float(bucket_data[1]) if bucket_data[1] else now

            # Calculate tokens to add
            time_passed = now - last_refill
            tokens_to_add = time_passed * refill_rate
            tokens = min(capacity, tokens + tokens_to_add)

            if tokens < cost:
                return False

            # Consume tokens
            tokens -= cost

            # Update bucket
            await self.cache_manager.redis_client.hmset(
                key, {
                    "tokens": tokens,
                    "last_refill": now
                }
            )
            await self.cache_manager.redis_client.expire(key, config["window"] * 2)

            return True

        except Exception as e:
            logger.error(
                "Token bucket rate limit check failed",
                identifier=identifier,
                error=str(e)
            )
            return True  # Allow on error

    async def get_rate_limit_status(
        self,
        identifier: str,
        resource: str = "default"
    ) -> Dict[str, Dict[str, int]]:
        """Get current rate limit status for a user"""
        status = {}

        for limit_type, config in self.limits.items():
            try:
                if self.algorithm == "sliding_window":
                    remaining = await self._get_sliding_window_remaining(
                        identifier, resource, limit_type, config
                    )
                elif self.algorithm == "fixed_window":
                    remaining = await self._get_fixed_window_remaining(
                        identifier, resource, limit_type, config
                    )
                elif self.algorithm == "token_bucket":
                    remaining = await self._get_token_bucket_remaining(
                        identifier, resource, limit_type, config
                    )
                else:
                    remaining = config["limit"]

                status[limit_type.value] = {
                    "limit": config["limit"],
                    "remaining": max(0, remaining),
                    "reset_time": int(time.time() + config["window"])
                }

            except Exception as e:
                logger.warning(
                    "Failed to get rate limit status",
                    identifier=identifier,
                    limit_type=limit_type,
                    error=str(e)
                )
                status[limit_type.value] = {
                    "limit": config["limit"],
                    "remaining": config["limit"],
                    "reset_time": int(time.time() + config["window"])
                }

        return status

    async def _get_sliding_window_remaining(
        self,
        identifier: str,
        resource: str,
        limit_type: RateLimitType,
        config: Dict
    ) -> int:
        """Get remaining requests for sliding window"""
        key = f"rate_limit:sliding:{identifier}:{resource}:{config['key_suffix']}"
        window = config["window"]
        limit = config["limit"]
        now = time.time()
        window_start = now - window

        try:
            # Remove expired entries
            await self.cache_manager.redis_client.zremrangebyscore(
                key, 0, window_start
            )

            # Count current requests
            current_count = await self.cache_manager.redis_client.zcard(key)
            return limit - current_count

        except Exception:
            return limit

    async def _get_fixed_window_remaining(
        self,
        identifier: str,
        resource: str,
        limit_type: RateLimitType,
        config: Dict
    ) -> int:
        """Get remaining requests for fixed window"""
        window = config["window"]
        limit = config["limit"]
        now = int(time.time())
        window_start = (now // window) * window

        key = f"rate_limit:fixed:{identifier}:{resource}:{config['key_suffix']}:{window_start}"

        try:
            current_count = await self.cache_manager.redis_client.get(key)
            current_count = int(current_count) if current_count else 0
            return limit - current_count

        except Exception:
            return limit

    async def _get_token_bucket_remaining(
        self,
        identifier: str,
        resource: str,
        limit_type: RateLimitType,
        config: Dict
    ) -> int:
        """Get remaining tokens in bucket"""
        key = f"rate_limit:bucket:{identifier}:{resource}:{config['key_suffix']}"
        capacity = config["limit"]
        refill_rate = capacity / config["window"]
        now = time.time()

        try:
            bucket_data = await self.cache_manager.redis_client.hmget(
                key, "tokens", "last_refill"
            )

            tokens = float(bucket_data[0]) if bucket_data[0] else capacity
            last_refill = float(bucket_data[1]) if bucket_data[1] else now

            # Calculate current tokens
            time_passed = now - last_refill
            tokens_to_add = time_passed * refill_rate
            tokens = min(capacity, tokens + tokens_to_add)

            return int(tokens)

        except Exception:
            return capacity

    async def reset_rate_limit(
        self,
        identifier: str,
        resource: str = "default",
        limit_type: Optional[RateLimitType] = None
    ) -> bool:
        """Reset rate limit for a user"""
        try:
            if limit_type:
                # Reset specific limit type
                config = self.limits[limit_type]
                patterns = [
                    f"rate_limit:*:{identifier}:{resource}:{config['key_suffix']}*"
                ]
            else:
                # Reset all limit types
                patterns = [
                    f"rate_limit:*:{identifier}:{resource}:*"
                ]

            for pattern in patterns:
                keys = await self.cache_manager.get_keys_pattern(pattern)
                if keys:
                    await self.cache_manager.redis_client.delete(*keys)

            logger.info(
                "Rate limit reset",
                identifier=identifier,
                resource=resource,
                limit_type=limit_type
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to reset rate limit",
                identifier=identifier,
                error=str(e)
            )
            return False

    def add_to_whitelist(self, identifier: str) -> None:
        """Add user to whitelist"""
        self.whitelisted_users.add(identifier)
        logger.info("User added to whitelist", identifier=identifier)

    def remove_from_whitelist(self, identifier: str) -> None:
        """Remove user from whitelist"""
        self.whitelisted_users.discard(identifier)
        logger.info("User removed from whitelist", identifier=identifier)

    def add_to_blacklist(self, identifier: str) -> None:
        """Add user to blacklist"""
        self.blacklisted_users.add(identifier)
        logger.info("User added to blacklist", identifier=identifier)

    def remove_from_blacklist(self, identifier: str) -> None:
        """Remove user from blacklist"""
        self.blacklisted_users.discard(identifier)
        logger.info("User removed from blacklist", identifier=identifier)

    async def get_top_users(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get top users by request count"""
        try:
            # This would require additional tracking
            # For now, return empty list
            return []
        except Exception as e:
            logger.error("Failed to get top users", error=str(e))
            return []

    def get_stats(self) -> Dict[str, any]:
        """Get rate limiter statistics"""
        success_rate = 0.0
        if self.stats["total_requests"] > 0:
            success_rate = self.stats["allowed_requests"] / self.stats["total_requests"]

        return {
            **self.stats,
            "success_rate": success_rate,
            "algorithm": self.algorithm,
            "limits": {
                limit_type.value: {
                    "limit": config["limit"],
                    "window": config["window"]
                }
                for limit_type, config in self.limits.items()
            },
            "whitelisted_users": len(self.whitelisted_users),
            "blacklisted_users": len(self.blacklisted_users)
        }

    async def cleanup_expired_keys(self) -> int:
        """Clean up expired rate limit keys"""
        cleaned = 0
        try:
            patterns = [
                "rate_limit:sliding:*",
                "rate_limit:fixed:*",
                "rate_limit:bucket:*"
            ]

            for pattern in patterns:
                keys = await self.cache_manager.get_keys_pattern(pattern)
                for key in keys:
                    ttl = await self.cache_manager.redis_client.ttl(key)
                    if ttl == -2:  # Key doesn't exist (expired)
                        cleaned += 1

        except Exception as e:
            logger.error("Failed to cleanup expired keys", error=str(e))

        return cleaned

    async def health_check(self) -> bool:
        """Check rate limiter health"""
        try:
            # Test basic functionality
            test_identifier = "health_check_test"
            await self.check_rate_limit(test_identifier, "health", 1)
            await self.reset_rate_limit(test_identifier, "health")
            return True
        except Exception as e:
            logger.error("Rate limiter health check failed", error=str(e))
            return False


class MockRateLimiter(RateLimiter):
    """Mock rate limiter for testing"""

    def __init__(self, always_allow: bool = True):
        self.always_allow = always_allow
        self.stats = {
            "total_requests": 0,
            "allowed_requests": 0,
            "blocked_requests": 0,
            "rate_limit_hits": 0
        }

    async def check_rate_limit(
        self,
        identifier: str,
        resource: str = "default",
        cost: int = 1
    ) -> bool:
        self.stats["total_requests"] += 1

        if self.always_allow:
            self.stats["allowed_requests"] += 1
            return True
        else:
            self.stats["blocked_requests"] += 1
            raise RateLimitExceeded("Mock rate limit exceeded")

    async def get_rate_limit_status(
        self,
        identifier: str,
        resource: str = "default"
    ) -> Dict[str, Dict[str, int]]:
        return {
            "per_minute": {
                "limit": 60,
                "remaining": 60,
                "reset_time": int(time.time() + 60)
            }
        }

    async def health_check(self) -> bool:
        return True
