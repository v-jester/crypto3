# src/data/collectors/websocket_client.py
"""
WebSocket клиент для получения данных в реальном времени с Binance
Поддерживает множественные потоки, автоматическое переподключение и обработку ошибок
"""
import json
import asyncio
import websockets
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime, timedelta
import time
from collections import deque
from src.monitoring.logger import logger, log_performance
from src.data.storage.redis_client import redis_client
from src.config.settings import settings
from src.monitoring.metrics import metrics_collector


class BinanceWebSocketClient:
    """Асинхронный WebSocket клиент для Binance"""

    def __init__(self):
        self.base_url = "wss://stream.binance.com:9443/ws" if not settings.api.TESTNET else "wss://testnet.binance.vision/ws"
        self.streams = {}
        self.connections = {}
        self.handlers = {}
        self.running = False
        self.reconnect_delay = 5  # секунд
        self.max_reconnect_attempts = 10
        self.ping_interval = 180  # 3 минуты
        self.message_buffer = deque(maxlen=10000)
        self.stats = {
            "messages_received": 0,
            "errors": 0,
            "reconnects": 0,
            "last_message_time": None
        }

    async def subscribe_klines(
            self,
            symbols: List[str],
            intervals: List[str],
            handler: Optional[Callable] = None
    ):
        """
        Подписка на свечные данные

        Args:
            symbols: Список торговых пар (например, ['BTCUSDT', 'ETHUSDT'])
            intervals: Интервалы свечей (например, ['5m', '15m'])
            handler: Функция-обработчик для данных
        """
        streams = []
        for symbol in symbols:
            for interval in intervals:
                stream = f"{symbol.lower()}@kline_{interval}"
                streams.append(stream)

                if handler:
                    self.handlers[stream] = handler

        stream_name = "klines"
        await self._connect_streams(stream_name, streams)

    async def subscribe_ticker(
            self,
            symbols: List[str],
            handler: Optional[Callable] = None
    ):
        """
        Подписка на тикеры (24hr изменения)

        Args:
            symbols: Список торговых пар
            handler: Функция-обработчик
        """
        streams = [f"{symbol.lower()}@ticker" for symbol in symbols]

        if handler:
            for stream in streams:
                self.handlers[stream] = handler

        stream_name = "ticker"
        await self._connect_streams(stream_name, streams)

    async def subscribe_depth(
            self,
            symbols: List[str],
            levels: int = 20,
            update_speed: int = 100,  # ms
            handler: Optional[Callable] = None
    ):
        """
        Подписка на стакан заявок

        Args:
            symbols: Список торговых пар
            levels: Глубина стакана (5, 10, 20)
            update_speed: Скорость обновления в мс (100 или 1000)
            handler: Функция-обработчик
        """
        speed = "100ms" if update_speed == 100 else "1000ms"
        streams = [f"{symbol.lower()}@depth{levels}@{speed}" for symbol in symbols]

        if handler:
            for stream in streams:
                self.handlers[stream] = handler

        stream_name = "depth"
        await self._connect_streams(stream_name, streams)

    async def subscribe_trades(
            self,
            symbols: List[str],
            handler: Optional[Callable] = None
    ):
        """
        Подписка на поток сделок

        Args:
            symbols: Список торговых пар
            handler: Функция-обработчик
        """
        streams = [f"{symbol.lower()}@aggTrade" for symbol in symbols]

        if handler:
            for stream in streams:
                self.handlers[stream] = handler

        stream_name = "trades"
        await self._connect_streams(stream_name, streams)

    async def _connect_streams(self, name: str, streams: List[str]):
        """Подключение к множественным потокам"""
        if not streams:
            return

        # Формируем URL с множественными потоками
        combined_url = f"{self.base_url.replace('/ws', '/stream')}?streams={'/'.join(streams)}"

        self.streams[name] = {
            "url": combined_url,
            "streams": streams,
            "reconnect_attempts": 0
        }

        # Запускаем подключение
        asyncio.create_task(self._maintain_connection(name))

    async def _maintain_connection(self, name: str):
        """Поддержание соединения с автоматическим переподключением"""
        stream_info = self.streams.get(name)
        if not stream_info:
            return

        while self.running:
            try:
                logger.logger.info(f"Connecting to {name} WebSocket streams")

                async with websockets.connect(
                        stream_info["url"],
                        ping_interval=self.ping_interval,
                        ping_timeout=20,
                        close_timeout=10
                ) as websocket:

                    self.connections[name] = websocket
                    stream_info["reconnect_attempts"] = 0

                    logger.logger.info(f"Connected to {name} WebSocket")
                    metrics_collector.record_api_latency("websocket", "connect", 0)

                    # Обработка сообщений
                    await self._handle_messages(name, websocket)

            except websockets.exceptions.ConnectionClosed as e:
                logger.logger.warning(f"WebSocket {name} connection closed: {e}")
                self.stats["errors"] += 1

            except Exception as e:
                logger.log_error(e, {"context": f"WebSocket {name} error"})
                self.stats["errors"] += 1
                metrics_collector.record_error("websocket_error", name)

            finally:
                if name in self.connections:
                    del self.connections[name]

            # Переподключение с экспоненциальной задержкой
            if self.running:
                stream_info["reconnect_attempts"] += 1
                self.stats["reconnects"] += 1

                if stream_info["reconnect_attempts"] >= self.max_reconnect_attempts:
                    logger.logger.error(f"Max reconnection attempts reached for {name}")
                    break

                delay = min(
                    self.reconnect_delay * (2 ** stream_info["reconnect_attempts"]),
                    300  # Максимум 5 минут
                )

                logger.logger.info(f"Reconnecting {name} in {delay} seconds...")
                await asyncio.sleep(delay)

    async def _handle_messages(self, name: str, websocket):
        """Обработка входящих сообщений"""
        async for message in websocket:
            try:
                data = json.loads(message)

                # Обновление статистики
                self.stats["messages_received"] += 1
                self.stats["last_message_time"] = time.time()
                metrics_collector.websocket_messages.labels(stream_type=name).inc()

                # Определяем тип потока из данных
                stream_name = data.get("stream", "")

                # Обработка через пользовательский handler
                if stream_name in self.handlers:
                    await self.handlers[stream_name](data.get("data", data))

                # Обработка по умолчанию
                await self._process_message(name, data)

                # Буферизация для анализа
                self.message_buffer.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "stream": name,
                    "data": data
                })

            except json.JSONDecodeError as e:
                logger.logger.error(f"Failed to decode WebSocket message: {e}")

            except Exception as e:
                logger.log_error(e, {"context": f"Error processing {name} message"})

    async def _process_message(self, stream_type: str, data: Dict[str, Any]):
        """Обработка сообщения по типу потока"""
        try:
            if "data" in data:
                # Множественный поток
                stream_data = data["data"]
                stream_name = data.get("stream", "")
            else:
                stream_data = data
                stream_name = stream_type

            # Определяем тип данных по stream name
            if "kline" in stream_name:
                await self._process_kline(stream_data)
            elif "ticker" in stream_name:
                await self._process_ticker(stream_data)
            elif "depth" in stream_name:
                await self._process_depth(stream_data)
            elif "aggTrade" in stream_name:
                await self._process_trade(stream_data)

        except Exception as e:
            logger.log_error(e, {"context": f"Failed to process {stream_type} message"})

    async def _process_kline(self, data: Dict[str, Any]):
        """Обработка данных свечей"""
        kline = data.get("k", {})

        if not kline:
            return

        symbol = kline.get("s")
        interval = kline.get("i")

        candle_data = {
            "symbol": symbol,
            "interval": interval,
            "open_time": kline.get("t"),
            "close_time": kline.get("T"),
            "open": float(kline.get("o", 0)),
            "high": float(kline.get("h", 0)),
            "low": float(kline.get("l", 0)),
            "close": float(kline.get("c", 0)),
            "volume": float(kline.get("v", 0)),
            "quote_volume": float(kline.get("q", 0)),
            "trades": int(kline.get("n", 0)),
            "is_closed": kline.get("x", False)
        }

        # Кешируем в Redis
        cache_key = f"kline:{symbol}:{interval}:latest"
        await redis_client.set(cache_key, candle_data, expire=300)

        # Добавляем в историю цен
        if candle_data["is_closed"]:
            await redis_client.add_price_history(
                symbol,
                candle_data["close"],
                datetime.fromtimestamp(candle_data["close_time"] / 1000)
            )

        # Обновляем метрики
        metrics_collector.update_market_data(symbol, candle_data["close"], candle_data["volume"])

    async def _process_ticker(self, data: Dict[str, Any]):
        """Обработка данных тикера"""
        ticker_data = {
            "symbol": data.get("s"),
            "price_change": float(data.get("p", 0)),
            "price_change_percent": float(data.get("P", 0)),
            "weighted_avg_price": float(data.get("w", 0)),
            "last_price": float(data.get("c", 0)),
            "last_qty": float(data.get("Q", 0)),
            "bid_price": float(data.get("b", 0)),
            "bid_qty": float(data.get("B", 0)),
            "ask_price": float(data.get("a", 0)),
            "ask_qty": float(data.get("A", 0)),
            "open": float(data.get("o", 0)),
            "high": float(data.get("h", 0)),
            "low": float(data.get("l", 0)),
            "volume": float(data.get("v", 0)),
            "quote_volume": float(data.get("q", 0)),
            "trades": int(data.get("n", 0))
        }

        # Кешируем
        cache_key = f"ticker:{ticker_data['symbol']}"
        await redis_client.set(cache_key, ticker_data, expire=60)

    async def _process_depth(self, data: Dict[str, Any]):
        """Обработка данных стакана"""
        depth_data = {
            "symbol": data.get("s", ""),
            "last_update_id": data.get("u"),
            "bids": [[float(p), float(q)] for p, q in data.get("b", [])],
            "asks": [[float(p), float(q)] for p, q in data.get("a", [])]
        }

        # Вычисляем спред
        if depth_data["bids"] and depth_data["asks"]:
            best_bid = depth_data["bids"][0][0]
            best_ask = depth_data["asks"][0][0]
            spread = (best_ask - best_bid) / best_ask * 10000  # в базисных пунктах

            depth_data["spread_bps"] = spread
            depth_data["mid_price"] = (best_bid + best_ask) / 2

        # Кешируем
        cache_key = f"depth:{depth_data['symbol']}"
        await redis_client.set(cache_key, depth_data, expire=10)

    async def _process_trade(self, data: Dict[str, Any]):
        """Обработка данных сделок"""
        trade_data = {
            "symbol": data.get("s"),
            "trade_id": data.get("a"),
            "price": float(data.get("p", 0)),
            "quantity": float(data.get("q", 0)),
            "timestamp": data.get("T"),
            "is_buyer_maker": data.get("m", False)
        }

        # Добавляем в список последних сделок
        cache_key = f"trades:{trade_data['symbol']}"
        await redis_client.lpush(cache_key, trade_data)
        await redis_client.ltrim(cache_key, 0, 99)  # Сохраняем последние 100 сделок

    async def start(self):
        """Запуск WebSocket клиента"""
        self.running = True
        logger.logger.info("Starting WebSocket client")

        # Запускаем мониторинг состояния
        asyncio.create_task(self._monitor_health())

    async def stop(self):
        """Остановка WebSocket клиента"""
        self.running = False
        logger.logger.info("Stopping WebSocket client")

        # Закрываем все соединения
        for name, ws in list(self.connections.items()):
            try:
                await ws.close()
            except:
                pass

        self.connections.clear()
        self.streams.clear()

    async def _monitor_health(self):
        """Мониторинг здоровья соединений"""
        while self.running:
            await asyncio.sleep(30)

            current_time = time.time()

            # Проверяем активность соединений
            if self.stats["last_message_time"]:
                time_since_last_message = current_time - self.stats["last_message_time"]

                if time_since_last_message > 60:
                    logger.logger.warning(
                        f"No messages received for {time_since_last_message:.0f} seconds"
                    )

            # Логируем статистику
            logger.logger.debug(
                "WebSocket statistics",
                messages=self.stats["messages_received"],
                errors=self.stats["errors"],
                reconnects=self.stats["reconnects"],
                active_connections=len(self.connections)
            )

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики работы"""
        return {
            **self.stats,
            "active_connections": len(self.connections),
            "registered_handlers": len(self.handlers),
            "buffer_size": len(self.message_buffer)
        }


# Глобальный экземпляр
ws_client = BinanceWebSocketClient()


# Вспомогательные функции
async def init_websocket_client():
    """Инициализация WebSocket клиента"""
    await ws_client.start()

    # Подписываемся на основные потоки
    symbols = settings.trading.SYMBOLS

    # Подписка на свечи
    await ws_client.subscribe_klines(
        symbols=symbols,
        intervals=[settings.trading.PRIMARY_TIMEFRAME]
    )

    # Подписка на тикеры
    await ws_client.subscribe_ticker(symbols=symbols)

    # Подписка на стакан для топ символов
    top_symbols = symbols[:3]  # Ограничиваем для экономии ресурсов
    await ws_client.subscribe_depth(
        symbols=top_symbols,
        levels=20,
        update_speed=100
    )

    logger.logger.info("WebSocket client initialized and subscribed to streams")

    return ws_client


async def close_websocket_client():
    """Закрытие WebSocket клиента"""
    await ws_client.stop()