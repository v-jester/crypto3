import json
import pickle
from typing import Any, Optional, List

import redis.asyncio as aioredis

from src.monitoring.logger import logger
from src.config.settings import settings


class RedisClient:
    """Асинхронный Redis клиент с безопасными операциями (SCAN/UNLINK)."""

    def __init__(self):
        self._redis: Optional[aioredis.Redis] = None
        self._connected = False
        self._url: Optional[str] = None

    async def connect(self, url: Optional[str] = None):
        """Подключение к Redis. Можно передать url, иначе берём из settings.database.redis_url."""
        try:
            if url:
                self._url = url
            if not self._url:
                self._url = settings.database.redis_url
            self._redis = await aioredis.from_url(
                self._url,
                encoding="utf-8",
                decode_responses=False,
                max_connections=10,
            )
            await self._redis.ping()
            self._connected = True
            logger.logger.info("Redis connected", url=self._url)
        except Exception as e:
            logger.logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False

    async def disconnect(self):
        if self._redis:
            await self._redis.close()
        self._connected = False
        logger.logger.info("Redis disconnected")

    async def get(self, key: str) -> Optional[Any]:
        if not self._connected:
            await self.connect()
        if not self._connected:
            return None
        try:
            data = await self._redis.get(key)
            if data is None:
                return None
            try:
                return json.loads(data)
            except Exception:
                try:
                    return pickle.loads(data)
                except Exception:
                    return data.decode("utf-8") if isinstance(data, bytes) else data
        except Exception as e:
            logger.logger.error(f"Failed to get key {key}: {e}")
            return None

    async def set(self, key: str, value: Any, expire: int | None = None) -> bool:
        if not self._connected:
            await self.connect()
        if not self._connected:
            return False
        try:
            if isinstance(value, (dict, list)):
                data = json.dumps(value)
            elif isinstance(value, str):
                data = value
            else:
                data = pickle.dumps(value)
            if expire:
                await self._redis.setex(key, expire, data)
            else:
                await self._redis.set(key, data)
            return True
        except Exception as e:
            logger.logger.error(f"Failed to set key {key}: {e}")
            return False

    async def delete(self, *keys: str) -> int:
        if not self._connected:
            await self.connect()
        if not self._connected:
            return 0
        try:
            return await self._redis.delete(*keys)
        except Exception:
            return 0

    async def keys(self, pattern: str, count: int = 500) -> List[str]:
        """Безопасный аналог KEYS на SCAN."""
        if not self._connected:
            await self.connect()
        if not self._connected:
            return []
        out: List[str] = []
        cursor = 0
        try:
            while True:
                cursor, batch = await self._redis.scan(cursor=cursor, match=pattern, count=count)
                if batch:
                    out.extend(k.decode("utf-8") if isinstance(k, bytes) else k for k in batch)
                if cursor == 0:
                    break
        except Exception as e:
            logger.logger.debug(f"Redis keys error for pattern {pattern}: {e}")
        return out

    async def delete_pattern(self, pattern: str, count: int = 500) -> int:
        """Удаление по паттерну через SCAN + UNLINK (fallback DEL)."""
        if not self._connected:
            await self.connect()
        if not self._connected:
            return 0
        deleted = 0
        cursor = 0
        try:
            while True:
                cursor, batch = await self._redis.scan(cursor=cursor, match=pattern, count=count)
                if batch:
                    try:
                        deleted += await self._redis.unlink(*batch)
                    except Exception:
                        deleted += await self._redis.delete(*batch)
                if cursor == 0:
                    break
        except Exception as e:
            logger.logger.debug(f"Redis delete_pattern error for {pattern}: {e}")
        return deleted

    async def flushdb(self, asynchronous: bool = True) -> bool:
        if not self._connected:
            await self.connect()
        if not self._connected:
            return False
        try:
            await self._redis.flushdb(asynchronous=asynchronous)
            logger.logger.info("Redis database flushed")
            return True
        except Exception as e:
            logger.logger.debug(f"Redis flushdb error: {e}")
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected


class CacheManager:
    def __init__(self, redis_client: RedisClient):
        self.redis = redis_client

    async def invalidate_pattern(self, pattern: str) -> int:
        return await self.redis.delete_pattern(pattern)

    async def cache_indicators(self, symbol: str, timeframe: str, data: dict, ttl: int = 60):
        if not self.redis._connected:
            await self.redis.connect()
        if not self.redis._connected:
            return False
        key = f"indicators:{symbol}:{timeframe}"
        return await self.redis.set(key, data, expire=ttl)

    async def get_indicators(self, symbol: str, timeframe: str):
        key = f"indicators:{symbol}:{timeframe}"
        return await self.redis.get(key)

    async def invalidate_symbol(self, symbol: str) -> int:
        total = 0
        for p in (f"historical:{symbol}:*", f"indicators:{symbol}:*", f"kline:{symbol}:*", f"ticker:{symbol}:*", f"market:{symbol}:*"):
            total += await self.redis.delete_pattern(p)
        return total


# Глобальные экземпляры
redis_client = RedisClient()
cache_manager = CacheManager(redis_client)


async def close_redis():
    await redis_client.disconnect()
