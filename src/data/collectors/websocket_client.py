# src/data/collectors/websocket_client.py
"""
WebSocket РєР»РёРµРЅС‚ РґР»СЏ РїРѕР»СѓС‡РµРЅРёСЏ РґР°РЅРЅС‹С… РІ СЂРµР°Р»СЊРЅРѕРј РІСЂРµРјРµРЅРё СЃ Binance
РџРѕРґРґРµСЂР¶РёРІР°РµС‚ РјРЅРѕР¶РµСЃС‚РІРµРЅРЅС‹Рµ РїРѕС‚РѕРєРё, Р°РІС‚РѕРјР°С‚РёС‡РµСЃРєРѕРµ РїРµСЂРµРїРѕРґРєР»СЋС‡РµРЅРёРµ Рё РѕР±СЂР°Р±РѕС‚РєСѓ РѕС€РёР±РѕРє
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
    """РђСЃРёРЅС…СЂРѕРЅРЅС‹Р№ WebSocket РєР»РёРµРЅС‚ РґР»СЏ Binance"""

    def __init__(self):
        self.base_url = "wss://stream.binance.com:9443/ws" if not settings.api.TESTNET else "wss://testnet.binance.vision/ws"
        self.streams = {}
        self.connections = {}
        self.handlers = {}
        self.running = False
        self.reconnect_delay = 5  # СЃРµРєСѓРЅРґ
        self.max_reconnect_attempts = 10
        self.ping_interval = 180  # 3 РјРёРЅСѓС‚С‹
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
        РџРѕРґРїРёСЃРєР° РЅР° СЃРІРµС‡РЅС‹Рµ РґР°РЅРЅС‹Рµ

        Args:
            symbols: РЎРїРёСЃРѕРє С‚РѕСЂРіРѕРІС‹С… РїР°СЂ (РЅР°РїСЂРёРјРµСЂ, ['BTCUSDT', 'ETHUSDT'])
            intervals: РРЅС‚РµСЂРІР°Р»С‹ СЃРІРµС‡РµР№ (РЅР°РїСЂРёРјРµСЂ, ['5m', '15m'])
            handler: Р¤СѓРЅРєС†РёСЏ-РѕР±СЂР°Р±РѕС‚С‡РёРє РґР»СЏ РґР°РЅРЅС‹С…
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
        РџРѕРґРїРёСЃРєР° РЅР° С‚РёРєРµСЂС‹ (24hr РёР·РјРµРЅРµРЅРёСЏ)

        Args:
            symbols: РЎРїРёСЃРѕРє С‚РѕСЂРіРѕРІС‹С… РїР°СЂ
            handler: Р¤СѓРЅРєС†РёСЏ-РѕР±СЂР°Р±РѕС‚С‡РёРє
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
        РџРѕРґРїРёСЃРєР° РЅР° СЃС‚Р°РєР°РЅ Р·Р°СЏРІРѕРє

        Args:
            symbols: РЎРїРёСЃРѕРє С‚РѕСЂРіРѕРІС‹С… РїР°СЂ
            levels: Р“Р»СѓР±РёРЅР° СЃС‚Р°РєР°РЅР° (5, 10, 20)
            update_speed: РЎРєРѕСЂРѕСЃС‚СЊ РѕР±РЅРѕРІР»РµРЅРёСЏ РІ РјСЃ (100 РёР»Рё 1000)
            handler: Р¤СѓРЅРєС†РёСЏ-РѕР±СЂР°Р±РѕС‚С‡РёРє
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
        РџРѕРґРїРёСЃРєР° РЅР° РїРѕС‚РѕРє СЃРґРµР»РѕРє

        Args:
            symbols: РЎРїРёСЃРѕРє С‚РѕСЂРіРѕРІС‹С… РїР°СЂ
            handler: Р¤СѓРЅРєС†РёСЏ-РѕР±СЂР°Р±РѕС‚С‡РёРє
        """
        streams = [f"{symbol.lower()}@aggTrade" for symbol in symbols]

        if handler:
            for stream in streams:
                self.handlers[stream] = handler

        stream_name = "trades"
        await self._connect_streams(stream_name, streams)

    async def _connect_streams(self, name: str, streams: List[str]):
        """РџРѕРґРєР»СЋС‡РµРЅРёРµ Рє РјРЅРѕР¶РµСЃС‚РІРµРЅРЅС‹Рј РїРѕС‚РѕРєР°Рј"""
        if not streams:
            return

        # Р¤РѕСЂРјРёСЂСѓРµРј URL СЃ РјРЅРѕР¶РµСЃС‚РІРµРЅРЅС‹РјРё РїРѕС‚РѕРєР°РјРё
        combined_url = f"{self.base_url.replace('/ws', '/stream')}?streams={'/'.join(streams)}"

        self.streams[name] = {
            "url": combined_url,
            "streams": streams,
            "reconnect_attempts": 0
        }

        # Р—Р°РїСѓСЃРєР°РµРј РїРѕРґРєР»СЋС‡РµРЅРёРµ
        asyncio.create_task(self._maintain_connection(name))

    async def _maintain_connection(self, name: str):
        """РџРѕРґРґРµСЂР¶Р°РЅРёРµ СЃРѕРµРґРёРЅРµРЅРёСЏ СЃ Р°РІС‚РѕРјР°С‚РёС‡РµСЃРєРёРј РїРµСЂРµРїРѕРґРєР»СЋС‡РµРЅРёРµРј"""
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

                    # РћР±СЂР°Р±РѕС‚РєР° СЃРѕРѕР±С‰РµРЅРёР№
                    await self._handle_messages(name, websocket)

            except websockets.exceptions.ConnectionClosed as e:
                logger.logger.warning(f"WebSocket {name} connection closed: {e}")
                self.stats["errors"] += 1

            except Exception as e:
                logger.logger.error(f"Error: {e}, context: {"context": f"WebSocket {name} error"}")
                self.stats["errors"] += 1
                metrics_collector.record_error("websocket_error", name)

            finally:
                if name in self.connections:
                    del self.connections[name]

            # РџРµСЂРµРїРѕРґРєР»СЋС‡РµРЅРёРµ СЃ СЌРєСЃРїРѕРЅРµРЅС†РёР°Р»СЊРЅРѕР№ Р·Р°РґРµСЂР¶РєРѕР№
            if self.running:
                stream_info["reconnect_attempts"] += 1
                self.stats["reconnects"] += 1

                if stream_info["reconnect_attempts"] >= self.max_reconnect_attempts:
                    logger.logger.error(f"Max reconnection attempts reached for {name}")
                    break

                delay = min(
                    self.reconnect_delay * (2 ** stream_info["reconnect_attempts"]),
                    300  # РњР°РєСЃРёРјСѓРј 5 РјРёРЅСѓС‚
                )

                logger.logger.info(f"Reconnecting {name} in {delay} seconds...")
                await asyncio.sleep(delay)

    async def _handle_messages(self, name: str, websocket):
        """РћР±СЂР°Р±РѕС‚РєР° РІС…РѕРґСЏС‰РёС… СЃРѕРѕР±С‰РµРЅРёР№"""
        async for message in websocket:
            try:
                data = json.loads(message)

                # РћР±РЅРѕРІР»РµРЅРёРµ СЃС‚Р°С‚РёСЃС‚РёРєРё
                self.stats["messages_received"] += 1
                self.stats["last_message_time"] = time.time()
                metrics_collector.websocket_messages.labels(stream_type=name).inc()

                # РћРїСЂРµРґРµР»СЏРµРј С‚РёРї РїРѕС‚РѕРєР° РёР· РґР°РЅРЅС‹С…
                stream_name = data.get("stream", "")

                # РћР±СЂР°Р±РѕС‚РєР° С‡РµСЂРµР· РїРѕР»СЊР·РѕРІР°С‚РµР»СЊСЃРєРёР№ handler
                if stream_name in self.handlers:
                    await self.handlers[stream_name](data.get("data", data))

                # РћР±СЂР°Р±РѕС‚РєР° РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ
                await self._process_message(name, data)

                # Р‘СѓС„РµСЂРёР·Р°С†РёСЏ РґР»СЏ Р°РЅР°Р»РёР·Р°
                self.message_buffer.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "stream": name,
                    "data": data
                })

            except json.JSONDecodeError as e:
                logger.logger.error(f"Failed to decode WebSocket message: {e}")

            except Exception as e:
                logger.logger.error(f"Error: {e}, context: {"context": f"Error processing {name} message"}")

    async def _process_message(self, stream_type: str, data: Dict[str, Any]):
        """РћР±СЂР°Р±РѕС‚РєР° СЃРѕРѕР±С‰РµРЅРёСЏ РїРѕ С‚РёРїСѓ РїРѕС‚РѕРєР°"""
        try:
            if "data" in data:
                # РњРЅРѕР¶РµСЃС‚РІРµРЅРЅС‹Р№ РїРѕС‚РѕРє
                stream_data = data["data"]
                stream_name = data.get("stream", "")
            else:
                stream_data = data
                stream_name = stream_type

            # РћРїСЂРµРґРµР»СЏРµРј С‚РёРї РґР°РЅРЅС‹С… РїРѕ stream name
            if "kline" in stream_name:
                await self._process_kline(stream_data)
            elif "ticker" in stream_name:
                await self._process_ticker(stream_data)
            elif "depth" in stream_name:
                await self._process_depth(stream_data)
            elif "aggTrade" in stream_name:
                await self._process_trade(stream_data)

        except Exception as e:
            logger.logger.error(f"Error: {e}, context: {"context": f"Failed to process {stream_type} message"}")

    async def _process_kline(self, data: Dict[str, Any]):
        """РћР±СЂР°Р±РѕС‚РєР° РґР°РЅРЅС‹С… СЃРІРµС‡РµР№"""
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

        # РљРµС€РёСЂСѓРµРј РІ Redis
        cache_key = f"kline:{symbol}:{interval}:latest"
        await redis_client.set(cache_key, candle_data, expire=300)

        # Р”РѕР±Р°РІР»СЏРµРј РІ РёСЃС‚РѕСЂРёСЋ С†РµРЅ
        if candle_data["is_closed"]:
            await redis_client.add_price_history(
                symbol,
                candle_data["close"],
                datetime.fromtimestamp(candle_data["close_time"] / 1000)
            )

        # РћР±РЅРѕРІР»СЏРµРј РјРµС‚СЂРёРєРё
        metrics_collector.update_market_data(symbol, candle_data["close"], candle_data["volume"])

    async def _process_ticker(self, data: Dict[str, Any]):
        """РћР±СЂР°Р±РѕС‚РєР° РґР°РЅРЅС‹С… С‚РёРєРµСЂР°"""
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

        # РљРµС€РёСЂСѓРµРј
        cache_key = f"ticker:{ticker_data['symbol']}"
        await redis_client.set(cache_key, ticker_data, expire=60)

    async def _process_depth(self, data: Dict[str, Any]):
        """РћР±СЂР°Р±РѕС‚РєР° РґР°РЅРЅС‹С… СЃС‚Р°РєР°РЅР°"""
        depth_data = {
            "symbol": data.get("s", ""),
            "last_update_id": data.get("u"),
            "bids": [[float(p), float(q)] for p, q in data.get("b", [])],
            "asks": [[float(p), float(q)] for p, q in data.get("a", [])]
        }

        # Р’С‹С‡РёСЃР»СЏРµРј СЃРїСЂРµРґ
        if depth_data["bids"] and depth_data["asks"]:
            best_bid = depth_data["bids"][0][0]
            best_ask = depth_data["asks"][0][0]
            spread = (best_ask - best_bid) / best_ask * 10000  # РІ Р±Р°Р·РёСЃРЅС‹С… РїСѓРЅРєС‚Р°С…

            depth_data["spread_bps"] = spread
            depth_data["mid_price"] = (best_bid + best_ask) / 2

        # РљРµС€РёСЂСѓРµРј
        cache_key = f"depth:{depth_data['symbol']}"
        await redis_client.set(cache_key, depth_data, expire=10)

    async def _process_trade(self, data: Dict[str, Any]):
        """РћР±СЂР°Р±РѕС‚РєР° РґР°РЅРЅС‹С… СЃРґРµР»РѕРє"""
        trade_data = {
            "symbol": data.get("s"),
            "trade_id": data.get("a"),
            "price": float(data.get("p", 0)),
            "quantity": float(data.get("q", 0)),
            "timestamp": data.get("T"),
            "is_buyer_maker": data.get("m", False)
        }

        # Р”РѕР±Р°РІР»СЏРµРј РІ СЃРїРёСЃРѕРє РїРѕСЃР»РµРґРЅРёС… СЃРґРµР»РѕРє
        cache_key = f"trades:{trade_data['symbol']}"
        await redis_client.lpush(cache_key, trade_data)
        await redis_client.ltrim(cache_key, 0, 99)  # РЎРѕС…СЂР°РЅСЏРµРј РїРѕСЃР»РµРґРЅРёРµ 100 СЃРґРµР»РѕРє

    async def start(self):
        """Р—Р°РїСѓСЃРє WebSocket РєР»РёРµРЅС‚Р°"""
        self.running = True
        logger.logger.info("Starting WebSocket client")

        # Р—Р°РїСѓСЃРєР°РµРј РјРѕРЅРёС‚РѕСЂРёРЅРі СЃРѕСЃС‚РѕСЏРЅРёСЏ
        asyncio.create_task(self._monitor_health())

    async def stop(self):
        """РћСЃС‚Р°РЅРѕРІРєР° WebSocket РєР»РёРµРЅС‚Р°"""
        self.running = False
        logger.logger.info("Stopping WebSocket client")

        # Р—Р°РєСЂС‹РІР°РµРј РІСЃРµ СЃРѕРµРґРёРЅРµРЅРёСЏ
        for name, ws in list(self.connections.items()):
            try:
                await ws.close()
            except:
                pass

        self.connections.clear()
        self.streams.clear()

    async def _monitor_health(self):
        """РњРѕРЅРёС‚РѕСЂРёРЅРі Р·РґРѕСЂРѕРІСЊСЏ СЃРѕРµРґРёРЅРµРЅРёР№"""
        while self.running:
            await asyncio.sleep(30)

            current_time = time.time()

            # РџСЂРѕРІРµСЂСЏРµРј Р°РєС‚РёРІРЅРѕСЃС‚СЊ СЃРѕРµРґРёРЅРµРЅРёР№
            if self.stats["last_message_time"]:
                time_since_last_message = current_time - self.stats["last_message_time"]

                if time_since_last_message > 60:
                    logger.logger.warning(
                        f"No messages received for {time_since_last_message:.0f} seconds"
                    )

            # Р›РѕРіРёСЂСѓРµРј СЃС‚Р°С‚РёСЃС‚РёРєСѓ
            logger.logger.debug(
                "WebSocket statistics",
                messages=self.stats["messages_received"],
                errors=self.stats["errors"],
                reconnects=self.stats["reconnects"],
                active_connections=len(self.connections)
            )

    def get_stats(self) -> Dict[str, Any]:
        """РџРѕР»СѓС‡РµРЅРёРµ СЃС‚Р°С‚РёСЃС‚РёРєРё СЂР°Р±РѕС‚С‹"""
        return {
            **self.stats,
            "active_connections": len(self.connections),
            "registered_handlers": len(self.handlers),
            "buffer_size": len(self.message_buffer)
        }


# Р“Р»РѕР±Р°Р»СЊРЅС‹Р№ СЌРєР·РµРјРїР»СЏСЂ
ws_client = BinanceWebSocketClient()


# Р’СЃРїРѕРјРѕРіР°С‚РµР»СЊРЅС‹Рµ С„СѓРЅРєС†РёРё
async def init_websocket_client():
    """РРЅРёС†РёР°Р»РёР·Р°С†РёСЏ WebSocket РєР»РёРµРЅС‚Р°"""
    await ws_client.start()

    # РџРѕРґРїРёСЃС‹РІР°РµРјСЃСЏ РЅР° РѕСЃРЅРѕРІРЅС‹Рµ РїРѕС‚РѕРєРё
    symbols = settings.trading.SYMBOLS

    # РџРѕРґРїРёСЃРєР° РЅР° СЃРІРµС‡Рё
    await ws_client.subscribe_klines(
        symbols=symbols,
        intervals=[settings.trading.PRIMARY_TIMEFRAME]
    )

    # РџРѕРґРїРёСЃРєР° РЅР° С‚РёРєРµСЂС‹
    await ws_client.subscribe_ticker(symbols=symbols)

    # РџРѕРґРїРёСЃРєР° РЅР° СЃС‚Р°РєР°РЅ РґР»СЏ С‚РѕРї СЃРёРјРІРѕР»РѕРІ
    top_symbols = symbols[:3]  # РћРіСЂР°РЅРёС‡РёРІР°РµРј РґР»СЏ СЌРєРѕРЅРѕРјРёРё СЂРµСЃСѓСЂСЃРѕРІ
    await ws_client.subscribe_depth(
        symbols=top_symbols,
        levels=20,
        update_speed=100
    )

    logger.logger.info("WebSocket client initialized and subscribed to streams")

    return ws_client


async def close_websocket_client():
    """Р—Р°РєСЂС‹С‚РёРµ WebSocket РєР»РёРµРЅС‚Р°"""
    await ws_client.stop()
