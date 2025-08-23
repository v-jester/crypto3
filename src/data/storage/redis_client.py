# src/data/storage/redis_client.py
"""
Redis клиент с поддержкой кеширования, pub/sub и управлением TTL
"""
import json
import pickle
import asyncio
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
import redis.asyncio as aioredis
from redis.asyncio import ConnectionPool
from functools import wraps
import hashlib
from src.monitoring.logger import logger, log_performance


class RedisClient:
    """Асинхронный Redis клиент с расширенным функционалом"""

    def __init__(self, url: str = "redis://localhost:6379/0", max_connections: int = 50):
        self.url = url
        self.max_connections = max_connections
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[aioredis.Redis] = None
        self._pubsub: Optional[aioredis.client.PubSub] = None
        self._connected = False

    async def connect(self):
        """Установка соединения с Redis"""
        if self._connected:
            return

        try:
            self._pool = ConnectionPool.from_url(
                self.url,
                max_connections=self.max_connections,
                decode_responses=False  # Для поддержки бинарных данных
            )
            self._client = aioredis.Redis(connection_pool=self._pool)

            # Проверка соединения
            await self._client.ping()
            self._connected = True
            logger.logger.info("Redis connected", url=self.url)

        except Exception as e:
            logger.log_error(e, {"context": "Redis connection failed"})
            raise

    async def disconnect(self):
        """Закрытие соединения"""
        if self._pubsub:
            await self._pubsub.close()
        if self._client:
            await self._client.close()
        if self._pool:
            await self._pool.disconnect()
        self._connected = False
        logger.logger.info("Redis disconnected")

    # ============ Базовые операции ============

    async def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """Установка значения с опциональным TTL"""
        if not self._connected:
            await self.connect()

        try:
            # Сериализация значения
            if isinstance(value, (str, bytes)):
                data = value
            else:
                data = json.dumps(value)

            if isinstance(data, str):
                data = data.encode('utf-8')

            result = await self._client.set(key, data, ex=expire)
            return bool(result)

        except Exception as e:
            logger.log_error(e, {"context": "Redis set", "key": key})
            return False

    async def get(self, key: str, default: Any = None) -> Any:
        """Получение значения"""
        if not self._connected:
            await self.connect()

        try:
            value = await self._client.get(key)
            if value is None:
                return default

            # Десериализация
            if isinstance(value, bytes):
                value = value.decode('utf-8')

            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value

        except Exception as e:
            logger.log_error(e, {"context": "Redis get", "key": key})
            return default

    async def delete(self, *keys: str) -> int:
        """Удаление ключей"""
        if not self._connected:
            await self.connect()
        return await self._client.delete(*keys)

    async def exists(self, *keys: str) -> int:
        """Проверка существования ключей"""
        if not self._connected:
            await self.connect()
        return await self._client.exists(*keys)

    # ============ Работа с хешами ============

    async def hset(self, name: str, key: str, value: Any) -> int:
        """Установка значения в хеш"""
        if not self._connected:
            await self.connect()

        if not isinstance(value, (str, bytes)):
            value = json.dumps(value)
        if isinstance(value, str):
            value = value.encode('utf-8')

        return await self._client.hset(name, key, value)

    async def hget(self, name: str, key: str) -> Any:
        """Получение значения из хеша"""
        if not self._connected:
            await self.connect()

        value = await self._client.hget(name, key)
        if value is None:
            return None

        if isinstance(value, bytes):
            value = value.decode('utf-8')

        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    async def hgetall(self, name: str) -> Dict[str, Any]:
        """Получение всех значений хеша"""
        if not self._connected:
            await self.connect()

        data = await self._client.hgetall(name)
        result = {}

        for key, value in data.items():
            if isinstance(key, bytes):
                key = key.decode('utf-8')
            if isinstance(value, bytes):
                value = value.decode('utf-8')

            try:
                result[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                result[key] = value

        return result

    # ============ Работа со списками ============

    async def lpush(self, key: str, *values: Any) -> int:
        """Добавление в начало списка"""
        if not self._connected:
            await self.connect()

        encoded_values = []
        for value in values:
            if not isinstance(value, (str, bytes)):
                value = json.dumps(value)
            if isinstance(value, str):
                value = value.encode('utf-8')
            encoded_values.append(value)

        return await self._client.lpush(key, *encoded_values)

    async def lrange(self, key: str, start: int = 0, stop: int = -1) -> List[Any]:
        """Получение диапазона из списка"""
        if not self._connected:
            await self.connect()

        values = await self._client.lrange(key, start, stop)
        result = []

        for value in values:
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            try:
                result.append(json.loads(value))
            except (json.JSONDecodeError, TypeError):
                result.append(value)

        return result

    async def ltrim(self, key: str, start: int, stop: int) -> bool:
        """Обрезка списка"""
        if not self._connected:
            await self.connect()
        return await self._client.ltrim(key, start, stop)

    # ============ Pub/Sub ============

    async def publish(self, channel: str, message: Any) -> int:
        """Публикация сообщения в канал"""
        if not self._connected:
            await self.connect()

        if not isinstance(message, (str, bytes)):
            message = json.dumps(message)
        if isinstance(message, str):
            message = message.encode('utf-8')

        return await self._client.publish(channel, message)

    async def subscribe(self, *channels: str) -> aioredis.client.PubSub:
        """Подписка на каналы"""
        if not self._connected:
            await self.connect()

        if not self._pubsub:
            self._pubsub = self._client.pubsub()

        await self._pubsub.subscribe(*channels)
        return self._pubsub

    # ============ Специализированные методы для трейдинга ============

    async def cache_market_data(
            self,
            symbol: str,
            data: Dict[str, Any],
            ttl: int = 300
    ) -> bool:
        """Кеширование рыночных данных"""
        key = f"market:{symbol}"
        return await self.set(key, data, expire=ttl)

    async def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Получение рыночных данных из кеша"""
        key = f"market:{symbol}"
        return await self.get(key)

    async def cache_indicators(
            self,
            symbol: str,
            timeframe: str,
            indicators: Dict[str, float],
            ttl: int = 300
    ) -> bool:
        """Кеширование индикаторов"""
        key = f"indicators:{symbol}:{timeframe}"
        for indicator_name, value in indicators.items():
            await self.hset(key, indicator_name, value)
        await self._client.expire(key, ttl)
        return True

    async def get_indicators(
            self,
            symbol: str,
            timeframe: str
    ) -> Dict[str, float]:
        """Получение индикаторов из кеша"""
        key = f"indicators:{symbol}:{timeframe}"
        return await self.hgetall(key)

    async def add_price_history(
            self,
            symbol: str,
            price: float,
            timestamp: Optional[datetime] = None,
            max_length: int = 1000
    ) -> None:
        """Добавление цены в историю"""
        key = f"prices:{symbol}"
        ts = timestamp or datetime.utcnow()

        data = {
            "price": price,
            "timestamp": ts.isoformat()
        }

        await self.lpush(key, data)
        await self.ltrim(key, 0, max_length - 1)

    async def get_price_history(
            self,
            symbol: str,
            limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Получение истории цен"""
        key = f"prices:{symbol}"
        return await self.lrange(key, 0, limit - 1)

    async def set_position(
            self,
            position_id: str,
            position_data: Dict[str, Any]
    ) -> bool:
        """Сохранение информации о позиции"""
        key = f"position:{position_id}"
        return await self.set(key, position_data)

    async def get_position(self, position_id: str) -> Optional[Dict[str, Any]]:
        """Получение информации о позиции"""
        key = f"position:{position_id}"
        return await self.get(key)

    async def get_all_positions(self) -> List[Dict[str, Any]]:
        """Получение всех открытых позиций"""
        pattern = "position:*"
        positions = []

        cursor = 0
        while True:
            cursor, keys = await self._client.scan(
                cursor,
                match=pattern,
                count=100
            )

            for key in keys:
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                position = await self.get(key)
                if position:
                    positions.append(position)

            if cursor == 0:
                break

        return positions

    # ============ Метрики и статистика ============

    async def increment_counter(self, key: str, amount: int = 1) -> int:
        """Инкремент счётчика"""
        if not self._connected:
            await self.connect()
        return await self._client.incrby(key, amount)

    async def get_counter(self, key: str) -> int:
        """Получение значения счётчика"""
        value = await self.get(key)
        return int(value) if value else 0


class CacheManager:
    """Менеджер кеширования с декораторами"""

    def __init__(self, redis_client: RedisClient):
        self.redis = redis_client

    def cache_result(
            self,
            ttl: int = 300,
            prefix: str = "cache",
            key_builder: Optional[Callable] = None
    ):
        """
        Декоратор для кеширования результатов функций

        Usage:
            @cache_manager.cache_result(ttl=600)
            async def expensive_calculation(param1, param2):
                ...
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Построение ключа кеша
                if key_builder:
                    cache_key = key_builder(*args, **kwargs)
                else:
                    # Генерация ключа на основе аргументов
                    key_data = f"{func.__name__}:{args}:{kwargs}"
                    key_hash = hashlib.md5(key_data.encode()).hexdigest()
                    cache_key = f"{prefix}:{key_hash}"

                # Проверка кеша
                cached = await self.redis.get(cache_key)
                if cached is not None:
                    logger.logger.debug(
                        "Cache hit",
                        function=func.__name__,
                        key=cache_key
                    )
                    return cached

                # Выполнение функции
                result = await func(*args, **kwargs)

                # Сохранение в кеш
                await self.redis.set(cache_key, result, expire=ttl)
                logger.logger.debug(
                    "Cache set",
                    function=func.__name__,
                    key=cache_key,
                    ttl=ttl
                )

                return result

            return wrapper

        return decorator

    async def invalidate_pattern(self, pattern: str) -> int:
        """Инвалидация кеша по паттерну"""
        if not self.redis._connected:
            await self.redis.connect()

        deleted = 0
        cursor = 0

        while True:
            cursor, keys = await self.redis._client.scan(
                cursor,
                match=pattern,
                count=100
            )

            if keys:
                deleted += await self.redis.delete(*keys)

            if cursor == 0:
                break

        logger.logger.info(
            "Cache invalidated",
            pattern=pattern,
            deleted_keys=deleted
        )

        return deleted


# Создание глобальных экземпляров
redis_client = RedisClient()
cache_manager = CacheManager(redis_client)


# Вспомогательные функции
async def init_redis(url: str = None):
    """Инициализация Redis соединения"""
    if url:
        global redis_client, cache_manager
        redis_client = RedisClient(url)
        cache_manager = CacheManager(redis_client)

    await redis_client.connect()
    return redis_client


async def close_redis():
    """Закрытие Redis соединения"""
    await redis_client.disconnect()