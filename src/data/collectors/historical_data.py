# src/data/collectors/historical_data.py
"""
Сборщик исторических данных с Binance с поддержкой
мультитаймфреймов и расчётом индикаторов
ИСПРАВЛЕНО: Проблема с кешированием и застрявшими индикаторами
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
from binance import AsyncClient
import pandas_ta as ta
from src.monitoring.logger import logger
from src.data.storage.redis_client import redis_client
from src.config.settings import settings
from functools import wraps
import time


def log_performance(name: str = None):
    """Декоратор для логирования производительности"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.perf_counter() - start_time
                logger.logger.debug(
                    f"{name or func.__name__} executed in {execution_time:.3f}s"
                )
                return result
            except Exception as e:
                execution_time = time.perf_counter() - start_time
                logger.logger.error(
                    f"{name or func.__name__} failed after {execution_time:.3f}s: {e}"
                )
                raise

        return wrapper

    return decorator


class HistoricalDataCollector:
    """Сборщик исторических данных"""

    def __init__(self):
        self.client: Optional[AsyncClient] = None
        self.symbol_info_cache: Dict[str, Dict] = {}
        self.force_refresh = True  # По умолчанию всегда обновляем данные
        self.cache_ttl = 30  # Время жизни кеша в секундах
        self.use_cache = False  # ВАЖНО: Полностью отключаем кеш для решения проблемы

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
            logger.logger.error(f"Failed to load symbol info: {e}")

    @log_performance("fetch_historical_data")
    async def fetch_historical_data(
            self,
            symbol: str,
            interval: str,
            days_back: int = 2,
            limit: int = 1000,
            force_refresh: bool = None
    ) -> pd.DataFrame:
        """
        Получение исторических данных

        Args:
            symbol: Торговая пара (например, BTCUSDT)
            interval: Интервал свечей (1m, 5m, 15m, 1h, 4h, 1d)
            days_back: Количество дней назад
            limit: Максимальное количество свечей
            force_refresh: Принудительное обновление, игнорируя кеш

        Returns:
            DataFrame с OHLCV данными и индикаторами
        """
        # Определяем нужно ли обновить данные
        should_refresh = force_refresh if force_refresh is not None else self.force_refresh

        # Ключ кеша
        cache_key = f"historical:{symbol}:{interval}:{days_back}"

        # Проверяем кеш ТОЛЬКО если явно разрешено и не требуется обновление
        if self.use_cache and not should_refresh:
            try:
                cached_data = await redis_client.get(cache_key)
                if cached_data and isinstance(cached_data, dict):
                    # Проверяем актуальность кешированных данных
                    if 'cache_time' in cached_data:
                        cache_time = datetime.fromisoformat(cached_data['cache_time'])
                        age_seconds = (datetime.now(timezone.utc).replace(tzinfo=None) - cache_time).total_seconds()

                        # Если данные старше cache_ttl секунд, обновляем
                        if age_seconds > self.cache_ttl:
                            logger.logger.debug(f"Cache expired for {symbol} (age: {age_seconds:.0f}s)")
                        else:
                            logger.logger.debug(f"Using cached data for {symbol} (age: {age_seconds:.0f}s)")
                            try:
                                df = pd.DataFrame(cached_data['data'])
                                if 'open_time' in df.columns:
                                    df['open_time'] = pd.to_datetime(df['open_time'])
                                    df.set_index('open_time', inplace=True)
                                return df
                            except Exception as e:
                                logger.logger.warning(f"Failed to restore from cache: {e}")
            except Exception as e:
                logger.logger.debug(f"Cache check failed: {e}")

        # Загружаем свежие данные с Binance
        try:
            logger.logger.debug(f"Fetching fresh data for {symbol} {interval} (force_refresh={should_refresh})")

            # Расчёт временных меток
            end_time = datetime.now(timezone.utc).replace(tzinfo=None)
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

            # Сохраняем индекс для восстановления из кеша
            df_for_cache = df.copy()

            df.set_index("open_time", inplace=True)

            # Добавление индикаторов
            self._add_indicators(df)

            # Добавление рыночной микроструктуры
            self._add_market_microstructure(df)

            logger.logger.debug(
                f"Fetched {len(df)} candles for {symbol} {interval} | "
                f"Latest RSI: {df['rsi'].iloc[-1]:.2f} | "
                f"Latest Price: {df['close'].iloc[-1]:.2f}"
            )

            # Кешируем результат ТОЛЬКО если разрешено
            if self.use_cache and redis_client._connected:
                try:
                    # Подготавливаем данные для кеширования
                    cache_data = {
                        'data': df.reset_index().to_dict('records'),
                        'cache_time': datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
                    }

                    # Сохраняем с TTL
                    await redis_client.set(cache_key, cache_data, expire=self.cache_ttl)
                    logger.logger.debug(f"Cached data for {symbol} with {self.cache_ttl}s TTL")
                except Exception as e:
                    logger.logger.debug(f"Failed to cache data: {e}")

            return df

        except Exception as e:
            logger.logger.error(
                f"Failed to fetch historical data for {symbol}: {e}"
            )
            # Возвращаем пустой DataFrame вместо исключения
            return pd.DataFrame()

    def _add_indicators(self, df: pd.DataFrame):
        """Добавление технических индикаторов используя pandas_ta"""
        try:
            # Проверка минимального количества данных
            if len(df) < 50:
                logger.logger.warning("Not enough data for indicators calculation")
                return

            # EMA
            df["ema_9"] = ta.ema(df["close"], length=9)
            df["ema_20"] = ta.ema(df["close"], length=20)
            df["ema_50"] = ta.ema(df["close"], length=50)
            df["ema_200"] = ta.ema(df["close"], length=200)

            # SMA
            df["sma_20"] = ta.sma(df["close"], length=20)
            df["sma_50"] = ta.sma(df["close"], length=50)

            # RSI - КРИТИЧЕСКИ ВАЖНО для проблемы
            df["rsi"] = ta.rsi(df["close"], length=settings.signals.RSI_PERIOD)
            df["rsi_14"] = ta.rsi(df["close"], length=14)

            # MACD
            macd_result = ta.macd(
                df["close"],
                fast=settings.signals.MACD_FAST,
                slow=settings.signals.MACD_SLOW,
                signal=settings.signals.MACD_SIGNAL
            )
            if macd_result is not None and not macd_result.empty:
                macd_cols = macd_result.columns
                if len(macd_cols) >= 3:
                    df["macd"] = macd_result.iloc[:, 0]  # MACD line
                    df["macd_signal"] = macd_result.iloc[:, 2]  # Signal line
                    df["macd_hist"] = macd_result.iloc[:, 1]  # Histogram

            # Bollinger Bands
            bb_result = ta.bbands(
                df["close"],
                length=settings.signals.BB_PERIOD,
                std=settings.signals.BB_STD
            )
            if bb_result is not None and not bb_result.empty:
                bb_cols = bb_result.columns
                if len(bb_cols) >= 3:
                    df["bb_lower"] = bb_result.iloc[:, 0]  # Lower band
                    df["bb_middle"] = bb_result.iloc[:, 1]  # Middle band
                    df["bb_upper"] = bb_result.iloc[:, 2]  # Upper band
                    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
                    # Безопасное вычисление bb_percent
                    df["bb_percent"] = df.apply(
                        lambda row: (row["close"] - row["bb_lower"]) / row["bb_width"]
                        if row["bb_width"] > 0 else 0.5,
                        axis=1
                    )

            # ATR
            df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
            df["atr_percent"] = (df["atr"] / df["close"]) * 100

            # Volume indicators
            df["obv"] = ta.obv(df["close"], df["volume"])
            df["ad"] = ta.ad(df["high"], df["low"], df["close"], df["volume"])
            df["adosc"] = ta.adosc(df["high"], df["low"], df["close"], df["volume"])

            # Volatility
            df["volatility"] = df["close"].pct_change().rolling(50).std()
            df["volatility_rank"] = df["volatility"].rolling(252).rank(pct=True)

            # Stochastic
            stoch_result = ta.stoch(df["high"], df["low"], df["close"])
            if stoch_result is not None and not stoch_result.empty:
                stoch_cols = stoch_result.columns
                if len(stoch_cols) >= 2:
                    df["stoch_k"] = stoch_result.iloc[:, 0]
                    df["stoch_d"] = stoch_result.iloc[:, 1]

            # Williams %R
            df["williams_r"] = ta.willr(df["high"], df["low"], df["close"], length=14)

            # CCI
            df["cci"] = ta.cci(df["high"], df["low"], df["close"], length=20)

            # MFI
            df["mfi"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)

            # ROC
            df["roc"] = ta.roc(df["close"], length=10)

            # Заполняем пропущенные значения
            df.bfill(inplace=True)
            df.ffill(inplace=True)

        except Exception as e:
            logger.logger.error(f"Failed to calculate indicators: {e}")

    def _add_market_microstructure(self, df: pd.DataFrame):
        """Добавление метрик рыночной микроструктуры"""
        try:
            # Спред
            df["spread"] = df["high"] - df["low"]
            df["spread_percent"] = (df["spread"] / df["close"]) * 100

            # Объёмные метрики
            volume_ma = df["volume"].rolling(20).mean()
            df["volume_ratio"] = df["volume"] / volume_ma.where(volume_ma > 0, 1)

            # Безопасное вычисление volume_delta
            df["volume_delta"] = df["taker_buy_base"] - (df["volume"] - df["taker_buy_base"])

            # Безопасное вычисление buy_pressure
            df["buy_pressure"] = df.apply(
                lambda row: row["taker_buy_base"] / row["volume"]
                if row["volume"] > 0 else 0.5,
                axis=1
            )

            # Ценовые уровни
            df["distance_from_high"] = ((df["high"] - df["close"]) / df["close"]) * 100
            df["distance_from_low"] = ((df["close"] - df["low"]) / df["close"]) * 100

            # Моментум
            df["momentum_1"] = df["close"].pct_change(1)
            df["momentum_5"] = df["close"].pct_change(5)
            df["momentum_10"] = df["close"].pct_change(10)

            # Эффективность движения
            price_change = df["close"].diff(10).abs()
            spread_sum = df["spread"].rolling(10).sum()
            df["efficiency_ratio"] = price_change / spread_sum.where(spread_sum > 0, 1)

            # VWAP
            df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
            df["vwap_distance"] = ((df["close"] - df["vwap"]) / df["vwap"]) * 100

            # Накопление/распределение
            df["accumulation"] = df.apply(
                lambda row: ((row["close"] - row["low"]) - (row["high"] - row["close"])) / row["spread"] * row["volume"]
                if row["spread"] > 0 else 0,
                axis=1
            )

        except Exception as e:
            logger.logger.error(f"Failed to calculate market microstructure: {e}")

    async def fetch_multi_timeframe_data(
            self,
            symbol: str,
            timeframes: List[str] = None,
            days_back: int = 2,
            force_refresh: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Получение данных для нескольких таймфреймов

        Args:
            symbol: Торговая пара
            timeframes: Список таймфреймов (по умолчанию из настроек)
            days_back: Количество дней назад
            force_refresh: Принудительное обновление данных

        Returns:
            Словарь {timeframe: DataFrame}
        """
        if timeframes is None:
            timeframes = settings.trading.TIMEFRAMES

        data = {}

        for tf in timeframes:
            try:
                df = await self.fetch_historical_data(
                    symbol,
                    tf,
                    days_back,
                    force_refresh=force_refresh
                )
                if not df.empty:
                    data[tf] = df
            except Exception as e:
                logger.logger.error(f"Failed to fetch {tf} data for {symbol}: {e}")

        return data

    async def get_latest_candle(self, symbol: str, interval: str) -> Optional[Dict]:
        """Получение последней свечи (всегда свежие данные)"""
        try:
            # Всегда получаем свежие данные для последней свечи
            df = await self.fetch_historical_data(
                symbol,
                interval,
                days_back=1,
                limit=2,
                force_refresh=True
            )
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
                    "timestamp": latest.name.isoformat() if hasattr(latest.name, 'isoformat') else str(latest.name)
                }
        except Exception as e:
            logger.logger.error(f"Failed to get latest candle: {e}")

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