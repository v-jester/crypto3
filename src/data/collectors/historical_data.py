"""
HistoricalDataCollector
- Инициализируется AsyncClient-ом Binance
- Загружает klines и формирует DataFrame с техническими индикаторами
- Возвращает df со столбцами: open, high, low, close, volume, close_time (UTC Timestamp), rsi, macd, macd_signal, bb_percent, volume_ratio, atr
"""

from __future__ import annotations

from typing import Optional, Dict, Any
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pandas_ta as ta
from binance import AsyncClient

from src.monitoring.logger import logger
from src.config.settings import settings


class HistoricalDataCollector:
    def __init__(self):
        self.client: Optional[AsyncClient] = None
        self.symbol_info: Dict[str, Any] = {}
        self.force_refresh: bool = True
        self.use_cache: bool = False  # сейчас отключено

    async def initialize(self, client: AsyncClient):
        self.client = client
        await self._load_symbol_info()
        logger.logger.info("Historical data collector initialized")

    async def _load_symbol_info(self):
        try:
            if self.client is None:
                return
            ex = await self.client.get_exchange_info()
            symbols = ex.get("symbols", [])
            self.symbol_info = {s["symbol"]: s for s in symbols if s.get("status") == "TRADING"}
            logger.logger.info(f"Loaded info for {len(self.symbol_info)} symbols")
        except Exception as e:
            logger.logger.warning(f"Failed to load symbol info: {e}")

    async def fetch_historical_data(
        self,
        symbol: str,
        interval: str,
        days_back: int = 1,
        limit: int = 200,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Загружает исторические свечи (spot klines) и рассчитывает индикаторы.
        Возвращает DataFrame (UTC).
        """
        if self.client is None:
            raise RuntimeError("HistoricalDataCollector is not initialized")

        interval = interval or settings.trading.PRIMARY_TIMEFRAME
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days_back)

        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        try:
            raw = await self.client.get_klines(
                symbol=symbol,
                interval=interval,
                startTime=start_ms,
                endTime=end_ms,
                limit=limit
            )
        except Exception as e:
            logger.logger.warning(f"Failed to fetch klines for {symbol}: {e}")
            return pd.DataFrame()

        if not raw:
            return pd.DataFrame()

        cols = [
            "open_time","open","high","low","close","volume","close_time",
            "qav","trades","taker_base","taker_quote","ignore"
        ]
        df = pd.DataFrame(raw, columns=cols)

        # типы
        for col in ("open","high","low","close","volume"):
            df[col] = df[col].astype(float)

        # Время → UTC Timestamp
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        # Индикаторы
        try:
            # RSI
            df["rsi"] = ta.rsi(df["close"], length=14)

            # MACD
            macd = ta.macd(df["close"])
            if macd is not None and not macd.empty:
                df["macd"] = macd["MACD_12_26_9"]
                df["macd_signal"] = macd["MACDs_12_26_9"]
            else:
                df["macd"] = 0.0
                df["macd_signal"] = 0.0

            # Bollinger Bands → позиция [0..1]
            bb = ta.bbands(df["close"], length=20, std=2.0)
            if bb is not None and not bb.empty:
                lower = bb["BBL_20_2.0"]
                upper = bb["BBU_20_2.0"]
                df["bb_percent"] = np.clip((df["close"] - lower) / (upper - lower), 0.0, 1.0)
            else:
                df["bb_percent"] = 0.5

            # Volume ratio
            vol_mean = df["volume"].rolling(20, min_periods=1).mean()
            df["volume_ratio"] = (df["volume"] / vol_mean).replace([np.inf, -np.inf], np.nan).fillna(1.0)

            # ATR
            df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

        except Exception as e:
            logger.logger.debug(f"Indicator calc failed for {symbol}: {e}")

        # Индекс по close_time (колонка остаётся)
        df.set_index("close_time", inplace=True)

        return df
