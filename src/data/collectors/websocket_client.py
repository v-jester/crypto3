"""
WebSocketClient wrapper для Binance streams.
- Подписка на kline и ticker
- Колбэки вызываются с исходными объектами Binance
Примечание: в тестнете у тебя включён REST polling; WS используется на лайве.
"""

import asyncio
from typing import Callable, List, Optional

from binance import AsyncClient, BinanceSocketManager
from src.monitoring.logger import logger


class WebSocketClient:
    def __init__(self):
        self.client: Optional[AsyncClient] = None
        self.bsm: Optional[BinanceSocketManager] = None

        self._kline_handlers: List[tuple[str, str, Callable]] = []
        self._ticker_handlers: List[tuple[str, Callable]] = []

        self._running = False
        self._tasks: List[asyncio.Task] = []

    def set_client(self, client: AsyncClient):
        self.client = client
        self.bsm = BinanceSocketManager(self.client)

    async def subscribe_klines(self, symbols: List[str], intervals: List[str], handler: Callable):
        for s in symbols:
            for itv in intervals:
                self._kline_handlers.append((s, itv, handler))

    async def subscribe_ticker(self, symbols: List[str], handler: Callable):
        for s in symbols:
            self._ticker_handlers.append((s, handler))

    async def start(self):
        if self.client is None or self.bsm is None:
            raise RuntimeError("WebSocket client is not initialized. Call set_client(client).")

        if self._running:
            return

        self._running = True
        logger.logger.info("Starting WebSocket client")

        for sym, itv, handler in self._kline_handlers:
            task = asyncio.create_task(self._run_kline(sym, itv, handler))
            self._tasks.append(task)

        for sym, handler in self._ticker_handlers:
            task = asyncio.create_task(self._run_ticker(sym, handler))
            self._tasks.append(task)

    async def stop(self):
        if not self._running:
            return
        self._running = False
        logger.logger.info("Stopping WebSocket client")

        for t in self._tasks:
            t.cancel()
        self._tasks.clear()

        if self.bsm:
            try:
                await self.bsm.close()
            except Exception:
                pass

    async def _run_kline(self, symbol: str, interval: str, handler: Callable):
        assert self.bsm is not None
        stream_symbol = symbol.lower()
        while self._running:
            try:
                async with self.bsm.kline_socket(symbol=stream_symbol, interval=interval) as stream:
                    async for msg in stream:
                        await handler(msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.logger.debug(f"kline socket error for {symbol}: {e}")
                await asyncio.sleep(2.0)

    async def _run_ticker(self, symbol: str, handler: Callable):
        assert self.bsm is not None
        stream_symbol = symbol.lower()
        while self._running:
            try:
                async with self.bsm.symbol_ticker_socket(symbol=stream_symbol) as stream:
                    async for msg in stream:
                        await handler(msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.logger.debug(f"ticker socket error for {symbol}: {e}")
                await asyncio.sleep(2.0)


ws_client = WebSocketClient()
