# src/data/collectors/historical_data.py
"""
Сборщик исторических данных с Binance с поддержкой
мультитаймфреймов и расчётом индикаторов
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from binance.client import AsyncClient
import talib as ta
from src.monitoring.logger import logger, log_performance
from src.data.storage.redis_client import cache_manager
from src.config.settings import settings


class HistoricalDataCollector:
    """Сборщик исторических данных"""

    def __init__(self):
        self.client: Optional[AsyncClient] = None
        self.symbol_info_cache: Dict[str, Dict] = {}

    async def initialize(self, client: AsyncClient):
        """Инициализация с клиентом Binance"""
        self.client = client
        await self._load_symbol_info()
        logger.logger.info("Historical data collector initialized")

    async def _load_symbol_info(self):
        """Загрузка информации о символах"""
        try:
            if settings.TRADING_MODE.value == "futures":
                info = await self.client.futures_exchange_info()
            else:
                info = await self.client.get_exchange_info()

            for symbol_info in info["symbols"]:
                if symbol_info["status"] != "TRADING":
                    continue

                symbol = symbol_info["symbol"]

                # Извлекаем важную информацию
                filters = {f["filterType"]: f for f in symbol_info["filters"]}

                self.symbol_info_cache[symbol] = {
                    "pricePrecision": symbol_info.get("pricePrecision", 8),
                    "quantityPrecision": symbol_info.get("quantityPrecision", 8),
                    "minNotional": float(filters.get("MIN_NOTIONAL", {}).get("minNotional", 10.0)),
                    "minQty": float(filters.get("LOT_SIZE", {}).get("minQty", 0.001)),
                    "maxQty": float(filters.get("LOT_SIZE", {}).get("maxQty", 9999999)),
                    "stepSize": float(filters.get("LOT_SIZE", {}).get("stepSize", 0.001)),
                    "tickSize": float(filters.get("PRICE_FILTER", {}).get("tickSize", 0.01)),
                }

            logger.logger.info(f"Loaded info for {len(self.symbol_info_cache)} symbols")

        except Exception as e:
            logger.log_error(e, {"context": "Failed to load symbol info"})

    @log_performance("fetch_historical_data")
    @cache_manager.cache_result(ttl=300, prefix="historical")
    async def fetch_historical_data(
            self,
            symbol: str,
            interval: str,
            days_back: int = 2,
            limit: int = 1000
    ) -> pd.DataFrame:
        """
        Получение исторических данных

        Args:
            symbol: Торговая пара (например, BTCUSDT)
            interval: Интервал свечей (1m, 5m, 15m, 1h, 4h, 1d)
            days_back: Количество дней назад
            limit: Максимальное количество свечей

        Returns:
            DataFrame с OHLCV данными и индикаторами
        """
        try:
            # Расчёт временных меток
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days_back)

            # Получение данных с Binance
            if settings.TRADING_MODE.value == "futures":
                klines = await self.client.futures_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=int(start_time.timestamp() * 1000),
                    endTime=int(end_time.timestamp() * 1000),
                    limit=limit
                )
            else:
                klines = await self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=int(start_time.timestamp() * 1000),
                    endTime=int(end_time.timestamp() * 1000),
                    limit=limit
                )

            # Преобразование в DataFrame
            df = pd.DataFrame(klines, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "trades",
                "taker_buy_base", "taker_buy_quote", "ignore"
            ])

            # Конвертация типов
            numeric_columns = ["open", "high", "low", "close", "volume",
                               "quote_asset_volume", "taker_buy_base", "taker_buy_quote"]
            df[numeric_columns] = df[numeric_columns].astype(float)
            df["trades"] = df["trades"].astype(int)

            # Временные метки
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
            df.set_index("open_time", inplace=True)

            # Добавление индикаторов
            self._add_indicators(df)

            # Добавление рыночной микроструктуры
            self._add_market_microstructure(df)

            logger.logger.debug(
                f"Fetched {len(df)} candles for {symbol} {interval}",
                symbol=symbol,
                interval=interval,
                records=len(df)
            )

            return df

        except Exception as e:
            logger.log_error(e, {
                "context": "Failed to fetch historical data",
                "symbol": symbol,
                "interval": interval
            })
            raise

    def _add_indicators(self, df: pd.DataFrame):
        """Добавление технических индикаторов"""
        try:
            # Проверка минимального количества данных
            if len(df) < 50:
                logger.logger.warning("Not enough data for indicators calculation")
                return

            # EMA
            df["ema_9"] = ta.EMA(df["close"], timeperiod=9)
            df["ema_20"] = ta.EMA(df["close"], timeperiod=20)
            df["ema_50"] = ta.EMA(df["close"], timeperiod=50)
            df["ema_200"] = ta.EMA(df["close"], timeperiod=200)

            # SMA
            df["sma_20"] = ta.SMA(df["close"], timeperiod=20)
            df["sma_50"] = ta.SMA(df["close"], timeperiod=50)

            # RSI
            df["rsi"] = ta.RSI(df["close"], timeperiod=settings.signals.RSI_PERIOD)
            df["rsi_14"] = ta.RSI(df["close"], timeperiod=14)

            # MACD
            macd, signal, hist = ta.MACD(
                df["close"],
                fastperiod=settings.signals.MACD_FAST,
                slowperiod=settings.signals.MACD_SLOW,
                signalperiod=settings.signals.MACD_SIGNAL
            )
            df["macd"] = macd
            df["macd_signal"] = signal
            df["macd_hist"] = hist

            # Bollinger Bands
            upper, middle, lower = ta.BBANDS(
                df["close"],
                timeperiod=settings.signals.BB_PERIOD,
                nbdevup=settings.signals.BB_STD,
                nbdevdn=settings.signals.BB_STD
            )
            df["bb_upper"] = upper
            df["bb_middle"] = middle
            df["bb_lower"] = lower
            df["bb_width"] = upper - lower
            df["bb_percent"] = (df["close"] - lower) / (upper - lower)

            # ATR
            df["atr"] = ta.ATR(df["high"], df["low"], df["close"], timeperiod=14)
            df["atr_percent"] = df["atr"] / df["close"] * 100

            # Volume indicators
            df["obv"] = ta.OBV(df["close"], df["volume"])
            df["ad"] = ta.AD(df["high"], df["low"], df["close"], df["volume"])
            df["adosc"] = ta.ADOSC(df["high"], df["low"], df["close"], df["volume"])

            # Volatility
            df["volatility"] = df["close"].pct_change().rolling(50).std()
            df["volatility_rank"] = df["volatility"].rolling(252).rank(pct=True)

            # Stochastic
            slowk, slowd = ta.STOCH(
                df["high"], df["low"], df["close"],
                fastk_period=14, slowk_period=3, slowd_period=3
            )
            df["stoch_k"] = slowk
            df["stoch_d"] = slowd

            # Williams %R
            df["williams_r"] = ta.WILLR(df["high"], df["low"], df["close"], timeperiod=14)

            # CCI
            df["cci"] = ta.CCI(df["high"], df["low"], df["close"], timeperiod=20)

            # MFI
            df["mfi"] = ta.MFI(df["high"], df["low"], df["close"], df["volume"], timeperiod=14)

            # ROC
            df["roc"] = ta.ROC(df["close"], timeperiod=10)

        except Exception as e:
            logger.log_error(e, {"context": "Failed to calculate indicators"})

    def _add_market_microstructure(self, df: pd.DataFrame):
        """Добавление метрик рыночной микроструктуры"""
        try:
            # Спред
            df["spread"] = df["high"] - df["low"]
            df["spread_percent"] = df["spread"] / df["close"] * 100

            # Объёмные метрики
            df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
            df["volume_delta"] = df["taker_buy_base"] - (df["volume"] - df["taker_buy_base"])
            df["buy_pressure"] = df["taker_buy_base"] / df["volume"]

            # Ценовые уровни
            df["distance_from_high"] = (df["high"] - df["close"]) / df["close"] * 100
            df["distance_from_low"] = (df["close"] - df["low"]) / df["close"] * 100

            # Моментум
            df["momentum_1"] = df["close"].pct_change(1)
            df["momentum_5"] = df["close"].pct_change(5)
            df["momentum_10"] = df["close"].pct_change(10)

            # Эффективность движения
            df["efficiency_ratio"] = abs(df["close"].diff(10)) / df["spread"].rolling(10).sum()

            # VWAP
            df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
            df["vwap_distance"] = (df["close"] - df["vwap"]) / df["vwap"] * 100

            # Накопление/распределение
            df["accumulation"] = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / df["spread"] * df["volume"]

        except Exception as e:
            logger.log_error(e, {"context": "Failed to calculate market microstructure"})

    async def fetch_multi_timeframe_data(
            self,
            symbol: str,
            timeframes: List[str] = None,
            days_back: int = 2
    ) -> Dict[str, pd.DataFrame]:
        """
        Получение данных для нескольких таймфреймов

        Args:
            symbol: Торговая пара
            timeframes: Список таймфреймов (по умолчанию из настроек)
            days_back: Количество дней назад

        Returns:
            Словарь {timeframe: DataFrame}
        """
        if timeframes is None:
            timeframes = settings.trading.TIMEFRAMES

        data = {}

        for tf in timeframes:
            try:
                df = await self.fetch_historical_data(symbol, tf, days_back)
                data[tf] = df
            except Exception as e:
                logger.logger.error(f"Failed to fetch {tf} data for {symbol}: {e}")

        return data

    async def get_latest_candle(self, symbol: str, interval: str) -> Optional[Dict]:
        """Получение последней свечи"""
        try:
            df = await self.fetch_historical_data(symbol, interval, days_back=1, limit=2)
            if not df.empty:
                latest = df.iloc[-1]
                return {
                    "symbol": symbol,
                    "interval": interval,
                    "open": float(latest["open"]),
                    "high": float(latest["high"]),
                    "low": float(latest["low"]),
                    "close": float(latest["close"]),
                    "volume": float(latest["volume"]),
                    "rsi": float(latest.get("rsi", 50)),
                    "timestamp": latest.name.isoformat()
                }
        except Exception as e:
            logger.log_error(e, {"context": "Failed to get latest candle"})

        return None

    def round_price(self, price: float, symbol: str) -> float:
        """Округление цены согласно правилам биржи"""
        info = self.symbol_info_cache.get(symbol, {})
        tick_size = info.get("tickSize", 0.01)

        if tick_size:
            return round(price / tick_size) * tick_size

        precision = info.get("pricePrecision", 2)
        return round(price, precision)

    def round_quantity(self, quantity: float, symbol: str) -> float:
        """Округление количества согласно правилам биржи"""
        info = self.symbol_info_cache.get(symbol, {})
        step_size = info.get("stepSize", 0.001)

        if step_size:
            return round(quantity / step_size) * step_size

        precision = info.get("quantityPrecision", 3)
        return round(quantity, precision)

    def validate_order_params(
            self,
            symbol: str,
            quantity: float,
            price: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Валидация параметров ордера

        Returns:
            (valid, error_message)
        """
        info = self.symbol_info_cache.get(symbol)
        if not info:
            return False, f"No symbol info for {symbol}"

        # Проверка минимального количества
        min_qty = info.get("minQty", 0)
        if quantity < min_qty:
            return False, f"Quantity {quantity} below minimum {min_qty}"

        # Проверка максимального количества
        max_qty = info.get("maxQty", float("inf"))
        if quantity > max_qty:
            return False, f"Quantity {quantity} above maximum {max_qty}"

        # Проверка минимального нотионала
        min_notional = info.get("minNotional", 0)
        notional = quantity * price
        if notional < min_notional:
            return False, f"Notional {notional} below minimum {min_notional}"

        return True, None