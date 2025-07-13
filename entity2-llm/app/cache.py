#!/usr/bin/env python3
"""
Cache Manager for Entity 2 LLM Advisor Service

This module provides caching functionality for LLM responses, conversation history,
and proof verification results using Redis as the backend.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

import redis.asyncio as redis
import structlog

logger = structlog.get_logger()


@dataclass
class CacheStats:
    """Cache statistics data class"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_keys: int = 0
    memory_usage_bytes: int = 0
    hit_rate: float = 0.0


class CacheManager:
    """
    Redis-based cache manager for the LLM Advisor Service

    Provides caching for:
    - LLM responses and advice
    - Conversation history
    - Proof verification results
    - User session data
    """

    def __init__(
        self,
        redis_url: str = "redis://redis:6379",
        default_ttl: int = 3600,  # 1 hour
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Redis connection
        self.redis_client: Optional[redis.Redis] = None

        # Cache statistics
        self.stats = CacheStats()

        # Cache key prefixes for different data types
        self.KEY_PREFIXES = {
            "advice": "advice:",
            "chat": "chat:",
            "conversation": "conv:",
            "proof": "proof:",
            "session": "session:",
            "user": "user:",
            "stats": "stats:",
            "config": "config:"
        }

    async def initialize(self) -> None:
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20,
                retry_on_timeout=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )

            # Test connection
            await self.redis_client.ping()
            logger.info("Cache manager initialized successfully", redis_url=self.redis_url)

        except Exception as e:
            logger.error("Failed to initialize cache manager", error=str(e))
            raise

    async def get(self, key: str, default: Any = None) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        if not self.redis_client:
            return default

        try:
            self.stats.total_requests += 1

            value = await self.redis_client.get(key)

            if value is not None:
                self.stats.cache_hits += 1
                self._update_hit_rate()

                # Try to deserialize JSON
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
            else:
                self.stats.cache_misses += 1
                self._update_hit_rate()
                return default

        except Exception as e:
            logger.warning("Cache get failed", key=key, error=str(e))
            self.stats.cache_misses += 1
            self._update_hit_rate()
            return default

    async def set(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = None,
        nx: bool = False,
        xx: bool = False
    ) -> bool:
        """
        Set value in cache

        Args:
            key: Cache key
            value: Value to cache
            expire: TTL in seconds (uses default_ttl if None)
            nx: Only set if key doesn't exist
            xx: Only set if key exists

        Returns:
            True if value was set, False otherwise
        """
        if not self.redis_client:
            return False

        try:
            # Serialize value if it's not a string
            if not isinstance(value, str):
                value = json.dumps(value, default=str)

            expire = expire or self.default_ttl

            result = await self.redis_client.set(
                key, value, ex=expire, nx=nx, xx=xx
            )

            return bool(result)

        except Exception as e:
            logger.warning("Cache set failed", key=key, error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.redis_client:
            return False

        try:
            result = await self.redis_client.delete(key)
            return bool(result)
        except Exception as e:
            logger.warning("Cache delete failed", key=key, error=str(e))
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.redis_client:
            return False

        try:
            result = await self.redis_client.exists(key)
            return bool(result)
        except Exception as e:
            logger.warning("Cache exists check failed", key=key, error=str(e))
            return False

    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration time for key"""
        if not self.redis_client:
            return False

        try:
            result = await self.redis_client.expire(key, seconds)
            return bool(result)
        except Exception as e:
            logger.warning("Cache expire failed", key=key, error=str(e))
            return False

    async def ttl(self, key: str) -> int:
        """Get time to live for key"""
        if not self.redis_client:
            return -1

        try:
            return await self.redis_client.ttl(key)
        except Exception as e:
            logger.warning("Cache TTL check failed", key=key, error=str(e))
            return -1

    # Specialized cache methods for different data types

    async def cache_advice_response(
        self,
        query: str,
        domain: str,
        verified_traits_hash: str,
        response: Dict[str, Any],
        ttl: int = 3600
    ) -> str:
        """Cache advice response with structured key"""
        import hashlib

        # Create deterministic cache key
        key_data = f"{query}:{domain}:{verified_traits_hash}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        cache_key = f"{self.KEY_PREFIXES['advice']}{key_hash}"

        await self.set(cache_key, response, expire=ttl)
        return cache_key

    async def get_cached_advice(
        self,
        query: str,
        domain: str,
        verified_traits_hash: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached advice response"""
        import hashlib

        key_data = f"{query}:{domain}:{verified_traits_hash}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        cache_key = f"{self.KEY_PREFIXES['advice']}{key_hash}"

        return await self.get(cache_key)

    async def store_conversation(
        self,
        session_id: str,
        messages: List[Dict[str, str]],
        ttl: int = 86400  # 24 hours
    ) -> bool:
        """Store conversation history"""
        key = f"{self.KEY_PREFIXES['conversation']}{session_id}"
        return await self.set(key, messages, expire=ttl)

    async def get_conversation(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history"""
        key = f"{self.KEY_PREFIXES['conversation']}{session_id}"
        messages = await self.get(key, [])
        return messages if isinstance(messages, list) else []

    async def append_to_conversation(
        self,
        session_id: str,
        message: Dict[str, str],
        max_messages: int = 50
    ) -> bool:
        """Append message to conversation history"""
        messages = await self.get_conversation(session_id)
        messages.append(message)

        # Trim to max_messages
        if len(messages) > max_messages:
            messages = messages[-max_messages:]

        return await self.store_conversation(session_id, messages)

    async def cache_proof_verification(
        self,
        proof_hash: str,
        is_valid: bool,
        verified_traits: Optional[Dict[str, Any]] = None,
        ttl: int = 300  # 5 minutes
    ) -> bool:
        """Cache proof verification result"""
        key = f"{self.KEY_PREFIXES['proof']}{proof_hash}"

        data = {
            "is_valid": is_valid,
            "verified_traits": verified_traits,
            "timestamp": datetime.utcnow().isoformat()
        }

        return await self.set(key, data, expire=ttl)

    async def get_cached_proof_verification(
        self, proof_hash: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached proof verification result"""
        key = f"{self.KEY_PREFIXES['proof']}{proof_hash}"
        return await self.get(key)

    async def store_user_session(
        self,
        user_id: str,
        session_data: Dict[str, Any],
        ttl: int = 7200  # 2 hours
    ) -> bool:
        """Store user session data"""
        key = f"{self.KEY_PREFIXES['session']}{user_id}"

        session_data["last_updated"] = datetime.utcnow().isoformat()
        return await self.set(key, session_data, expire=ttl)

    async def get_user_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user session data"""
        key = f"{self.KEY_PREFIXES['session']}{user_id}"
        return await self.get(key)

    async def increment_counter(self, key: str, amount: int = 1, expire: Optional[int] = None) -> int:
        """Increment a counter in cache"""
        if not self.redis_client:
            return 0

        try:
            result = await self.redis_client.incr(key, amount)

            if expire and result == amount:  # First time setting
                await self.redis_client.expire(key, expire)

            return result
        except Exception as e:
            logger.warning("Counter increment failed", key=key, error=str(e))
            return 0

    async def get_keys_pattern(self, pattern: str) -> List[str]:
        """Get keys matching pattern"""
        if not self.redis_client:
            return []

        try:
            return await self.redis_client.keys(pattern)
        except Exception as e:
            logger.warning("Keys pattern search failed", pattern=pattern, error=str(e))
            return []

    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern"""
        if not self.redis_client:
            return 0

        try:
            keys = await self.get_keys_pattern(pattern)
            if keys:
                return await self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.warning("Pattern delete failed", pattern=pattern, error=str(e))
            return 0

    async def clear_expired_keys(self) -> int:
        """Clear expired keys (Redis handles this automatically, but useful for stats)"""
        # This is mainly for statistics - Redis handles expiration automatically
        expired_count = 0

        try:
            # Get some sample keys and check their TTL
            for prefix in self.KEY_PREFIXES.values():
                keys = await self.get_keys_pattern(f"{prefix}*")
                for key in keys[:10]:  # Sample first 10 keys
                    ttl = await self.ttl(key)
                    if ttl == -2:  # Key doesn't exist (expired)
                        expired_count += 1

        except Exception as e:
            logger.warning("Expired keys check failed", error=str(e))

        return expired_count

    async def health_check(self) -> bool:
        """Check cache health"""
        if not self.redis_client:
            return False

        try:
            # Simple ping test
            result = await self.redis_client.ping()
            return result is True
        except Exception as e:
            logger.warning("Cache health check failed", error=str(e))
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats_dict = {
            "total_requests": self.stats.total_requests,
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses,
            "hit_rate": self.stats.hit_rate,
            "connected": bool(self.redis_client)
        }

        if self.redis_client:
            try:
                # Get Redis info
                info = await self.redis_client.info()
                stats_dict.update({
                    "redis_version": info.get("redis_version", "unknown"),
                    "used_memory": info.get("used_memory", 0),
                    "used_memory_human": info.get("used_memory_human", "0B"),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0)
                })

                # Calculate Redis hit rate
                redis_hits = info.get("keyspace_hits", 0)
                redis_misses = info.get("keyspace_misses", 0)
                if redis_hits + redis_misses > 0:
                    stats_dict["redis_hit_rate"] = redis_hits / (redis_hits + redis_misses)
                else:
                    stats_dict["redis_hit_rate"] = 0.0

            except Exception as e:
                logger.warning("Failed to get Redis info", error=str(e))

        return stats_dict

    def _update_hit_rate(self) -> None:
        """Update cache hit rate"""
        if self.stats.total_requests > 0:
            self.stats.hit_rate = self.stats.cache_hits / self.stats.total_requests
        else:
            self.stats.hit_rate = 0.0

    async def cleanup(self) -> None:
        """Cleanup cache resources"""
        try:
            if self.redis_client:
                await self.redis_client.close()
                logger.info("Cache manager cleanup completed")
        except Exception as e:
            logger.error("Error during cache cleanup", error=str(e))

    # Context manager support
    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            if hasattr(self, 'redis_client') and self.redis_client:
                # Note: This won't work in async context, but prevents warnings
                pass
        except:
            pass


class MemoryCache:
    """In-memory cache fallback when Redis is not available"""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.stats = CacheStats()

    async def initialize(self) -> None:
        """Initialize memory cache (no-op)"""
        logger.info("Using in-memory cache fallback")

    async def get(self, key: str, default: Any = None) -> Optional[Any]:
        """Get value from memory cache"""
        self.stats.total_requests += 1

        if key in self.cache:
            entry = self.cache[key]

            # Check expiration
            if datetime.utcnow() < entry["expires_at"]:
                self.stats.cache_hits += 1
                self._update_hit_rate()
                return entry["value"]
            else:
                # Remove expired entry
                del self.cache[key]

        self.stats.cache_misses += 1
        self._update_hit_rate()
        return default

    async def set(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = None,
        nx: bool = False,
        xx: bool = False
    ) -> bool:
        """Set value in memory cache"""
        # Check conditions
        if nx and key in self.cache:
            return False
        if xx and key not in self.cache:
            return False

        # Evict if at max size
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Remove oldest entry
            oldest_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k]["created_at"]
            )
            del self.cache[oldest_key]

        expire = expire or self.default_ttl
        expires_at = datetime.utcnow() + timedelta(seconds=expire)

        self.cache[key] = {
            "value": value,
            "created_at": datetime.utcnow(),
            "expires_at": expires_at
        }

        return True

    async def delete(self, key: str) -> bool:
        """Delete key from memory cache"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False

    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired"""
        return await self.get(key) is not None

    async def health_check(self) -> bool:
        """Memory cache is always healthy"""
        return True

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory cache statistics"""
        return {
            "total_requests": self.stats.total_requests,
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses,
            "hit_rate": self.stats.hit_rate,
            "total_keys": len(self.cache),
            "max_size": self.max_size,
            "cache_type": "memory"
        }

    def _update_hit_rate(self) -> None:
        """Update cache hit rate"""
        if self.stats.total_requests > 0:
            self.stats.hit_rate = self.stats.cache_hits / self.stats.total_requests
        else:
            self.stats.hit_rate = 0.0

    async def cleanup(self) -> None:
        """Cleanup memory cache"""
        self.cache.clear()
        logger.info("Memory cache cleanup completed")
