# src/data/collectors/historical_data.py
"""
Сборщик исторических данных с Binance с поддержкой
мультитаймфреймов и расчётом индикаторов
ИСПРАВЛЕНО: Устранены все FutureWarning для pandas 3.0
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
        self.last_prices = {}  # Храним последние известные цены

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

            # Адаптируем limit в зависимости от интервала
            interval_limits = {
                '1m': min(limit, 500),
                '5m': min(limit, 288),  # ~1 день
                '15m': min(limit, 96),  # ~1 день
                '30m': min(limit, 48),  # ~1 день
                '1h': min(limit, 24),  # 1 день
                '4h': min(limit, 42),  # 1 неделя
                '1d': min(limit, 30)  # 1 месяц
            }

            adjusted_limit = interval_limits.get(interval, limit)

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
                    limit=adjusted_limit
                )
            else:
                klines = await self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=int(start_time.timestamp() * 1000),
                    endTime=int(end_time.timestamp() * 1000),
                    limit=adjusted_limit
                )

            if not klines:
                logger.logger.warning(f"No klines data received for {symbol} {interval}")
                return pd.DataFrame()

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

            # Сохраняем последнюю цену
            if not df.empty:
                self.last_prices[symbol] = df['close'].iloc[-1]

            df.set_index("open_time", inplace=True)

            # Добавление индикаторов с обработкой недостаточных данных
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
        """Добавление технических индикаторов с обработкой недостаточных данных"""
        try:
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Проверяем минимальное количество данных
            if len(df) < 14:  # Минимум для базовых индикаторов
                logger.logger.warning(f"Not enough data for indicators: {len(df)} rows")
                # Заполняем дефолтными значениями
                df["rsi"] = 50.0
                df["rsi_14"] = 50.0
                df["macd"] = 0.0
                df["macd_signal"] = 0.0
                df["macd_hist"] = 0.0
                df["bb_lower"] = df["close"] * 0.98 if "close" in df.columns else 0
                df["bb_middle"] = df["close"] if "close" in df.columns else 0
                df["bb_upper"] = df["close"] * 1.02 if "close" in df.columns else 0
                df["bb_width"] = df["close"] * 0.04 if "close" in df.columns else 1
                df["bb_percent"] = 0.5
                df["atr"] = df["close"] * 0.02 if "close" in df.columns else 0
                df["atr_percent"] = 2.0
                df["volume_ratio"] = 1.0
                return

            # EMA - проверяем достаточность данных для каждой
            if len(df) >= 9:
                df["ema_9"] = ta.ema(df["close"], length=9)
            else:
                df["ema_9"] = df["close"]

            if len(df) >= 20:
                df["ema_20"] = ta.ema(df["close"], length=20)
                df["sma_20"] = ta.sma(df["close"], length=20)
            else:
                df["ema_20"] = df["close"]
                df["sma_20"] = df["close"]

            if len(df) >= 50:
                df["ema_50"] = ta.ema(df["close"], length=50)
                df["sma_50"] = ta.sma(df["close"], length=50)
            else:
                df["ema_50"] = df["close"]
                df["sma_50"] = df["close"]

            if len(df) >= 200:
                df["ema_200"] = ta.ema(df["close"], length=200)
            else:
                df["ema_200"] = df["close"]

            # RSI - безопасный расчет с проверкой (ИСПРАВЛЕНО для pandas 3.0)
            min_rsi_period = min(settings.signals.RSI_PERIOD, len(df) - 1)
            if min_rsi_period > 2:
                df["rsi"] = ta.rsi(df["close"], length=min_rsi_period)
                if df["rsi"].isna().all():
                    df["rsi"] = 50.0
                else:
                    df["rsi"] = df["rsi"].bfill()
                    df["rsi"] = df["rsi"].fillna(50.0)
            else:
                df["rsi"] = 50.0

            # RSI 14
            if len(df) >= 14:
                df["rsi_14"] = ta.rsi(df["close"], length=14)
                if df["rsi_14"].isna().all():
                    df["rsi_14"] = 50.0
                else:
                    df["rsi_14"] = df["rsi_14"].fillna(50.0)
            else:
                df["rsi_14"] = 50.0

            # MACD - требует минимум 26 баров
            if len(df) >= max(settings.signals.MACD_SLOW, 26):
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
                        # Заполняем NaN (ИСПРАВЛЕНО для pandas 3.0)
                        df["macd"] = df["macd"].fillna(0)
                        df["macd_signal"] = df["macd_signal"].fillna(0)
                        df["macd_hist"] = df["macd_hist"].fillna(0)
                    else:
                        df["macd"] = 0
                        df["macd_signal"] = 0
                        df["macd_hist"] = 0
                else:
                    df["macd"] = 0
                    df["macd_signal"] = 0
                    df["macd_hist"] = 0
            else:
                df["macd"] = 0
                df["macd_signal"] = 0
                df["macd_hist"] = 0

            # Bollinger Bands
            if len(df) >= settings.signals.BB_PERIOD:
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
                    else:
                        self._set_default_bb(df)
                else:
                    self._set_default_bb(df)
            else:
                self._set_default_bb(df)

            # ATR
            if len(df) >= 14:
                df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
                if df["atr"].isna().all():
                    df["atr"] = df["close"] * 0.02
                else:
                    df["atr"] = df["atr"].fillna(df["close"] * 0.02)
                df["atr_percent"] = (df["atr"] / df["close"]) * 100
            else:
                df["atr"] = df["close"] * 0.02
                df["atr_percent"] = 2.0

            # Volume indicators - только если достаточно данных
            if len(df) >= 20:
                df["obv"] = ta.obv(df["close"], df["volume"])
                df["ad"] = ta.ad(df["high"], df["low"], df["close"], df["volume"])
                if len(df) >= 20:
                    df["adosc"] = ta.adosc(df["high"], df["low"], df["close"], df["volume"])
                else:
                    df["adosc"] = 0
            else:
                df["obv"] = df["volume"].cumsum() if "volume" in df.columns else 0
                df["ad"] = 0
                df["adosc"] = 0

            # Volatility
            if len(df) >= 20:
                df["volatility"] = df["close"].pct_change().rolling(20).std()
                df["volatility"] = df["volatility"].fillna(0.01)
            else:
                df["volatility"] = 0.01

            if len(df) >= 50:
                df["volatility_rank"] = df["volatility"].rolling(50).rank(pct=True)
                df["volatility_rank"] = df["volatility_rank"].fillna(0.5)
            else:
                df["volatility_rank"] = 0.5

            # Другие индикаторы только если достаточно данных
            if len(df) >= 14:
                # Stochastic
                stoch_result = ta.stoch(df["high"], df["low"], df["close"])
                if stoch_result is not None and not stoch_result.empty:
                    stoch_cols = stoch_result.columns
                    if len(stoch_cols) >= 2:
                        df["stoch_k"] = stoch_result.iloc[:, 0]
                        df["stoch_d"] = stoch_result.iloc[:, 1]
                    else:
                        df["stoch_k"] = 50
                        df["stoch_d"] = 50
                else:
                    df["stoch_k"] = 50
                    df["stoch_d"] = 50

                # Williams %R
                df["williams_r"] = ta.willr(df["high"], df["low"], df["close"], length=14)
                if df["williams_r"].isna().all():
                    df["williams_r"] = -50
                else:
                    df["williams_r"] = df["williams_r"].fillna(-50)

                # MFI
                df["mfi"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)
                if df["mfi"].isna().all():
                    df["mfi"] = 50
                else:
                    df["mfi"] = df["mfi"].fillna(50)

                # ROC
                df["roc"] = ta.roc(df["close"], length=10)
                if df["roc"].isna().all():
                    df["roc"] = 0
                else:
                    df["roc"] = df["roc"].fillna(0)
            else:
                df["stoch_k"] = 50
                df["stoch_d"] = 50
                df["williams_r"] = -50
                df["mfi"] = 50
                df["roc"] = 0

            # CCI - требует минимум 20 баров
            if len(df) >= 20:
                df["cci"] = ta.cci(df["high"], df["low"], df["close"], length=20)
                if df["cci"].isna().all():
                    df["cci"] = 0
                else:
                    df["cci"] = df["cci"].fillna(0)
            else:
                df["cci"] = 0

            # Финальное заполнение пропущенных значений (ИСПРАВЛЕНО для pandas 3.0)
            df = df.bfill()
            df = df.ffill()

            # Проверка что основные индикаторы не NaN
            critical_indicators = ['rsi', 'macd', 'bb_percent', 'atr', 'volume_ratio']
            for indicator in critical_indicators:
                if indicator in df.columns:
                    if df[indicator].isna().any():
                        default_values = {
                            'rsi': 50.0,
                            'macd': 0.0,
                            'bb_percent': 0.5,
                            'atr': df["close"].mean() * 0.02 if "close" in df.columns else 1,
                            'volume_ratio': 1.0
                        }
                        df[indicator] = df[indicator].fillna(default_values.get(indicator, 0))

        except Exception as e:
            logger.logger.error(f"Failed to calculate indicators: {e}")
            # В случае ошибки заполняем базовыми значениями
            self._set_default_indicators(df)

    def _set_default_bb(self, df: pd.DataFrame):
        """Установка дефолтных значений для Bollinger Bands"""
        if "close" in df.columns:
            df["bb_middle"] = df["close"]
            df["bb_lower"] = df["close"] * 0.98
            df["bb_upper"] = df["close"] * 1.02
            df["bb_width"] = df["close"] * 0.04
            df["bb_percent"] = 0.5
        else:
            df["bb_middle"] = 0
            df["bb_lower"] = 0
            df["bb_upper"] = 0
            df["bb_width"] = 1
            df["bb_percent"] = 0.5

    def _set_default_indicators(self, df: pd.DataFrame):
        """Установка дефолтных значений для всех индикаторов"""
        default_price = df["close"].mean() if "close" in df.columns and not df["close"].empty else 100

        df["rsi"] = 50.0
        df["rsi_14"] = 50.0
        df["macd"] = 0.0
        df["macd_signal"] = 0.0
        df["macd_hist"] = 0.0
        df["atr"] = default_price * 0.02
        df["atr_percent"] = 2.0
        df["bb_percent"] = 0.5
        df["volume_ratio"] = 1.0
        df["volatility"] = 0.01
        df["volatility_rank"] = 0.5

    def _add_market_microstructure(self, df: pd.DataFrame):
        """Добавление метрик рыночной микроструктуры"""
        try:
            # Спред
            df["spread"] = df["high"] - df["low"]
            df["spread_percent"] = (df["spread"] / df["close"]) * 100

            # Объёмные метрики (ИСПРАВЛЕНО для pandas 3.0)
            if len(df) >= 20:
                volume_ma = df["volume"].rolling(20).mean()
                df["volume_ratio"] = df["volume"] / volume_ma.where(volume_ma > 0, 1)
                df["volume_ratio"] = df["volume_ratio"].fillna(1.0)
            else:
                df["volume_ratio"] = 1.0

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

            # Моментум (ИСПРАВЛЕНО для pandas 3.0)
            for period in [1, 5, 10]:
                if len(df) > period:
                    df[f"momentum_{period}"] = df["close"].pct_change(period)
                    df[f"momentum_{period}"] = df[f"momentum_{period}"].fillna(0)
                else:
                    df[f"momentum_{period}"] = 0

            # Эффективность движения (ИСПРАВЛЕНО для pandas 3.0)
            if len(df) >= 10:
                price_change = df["close"].diff(10).abs()
                spread_sum = df["spread"].rolling(10).sum()
                df["efficiency_ratio"] = price_change / spread_sum.where(spread_sum > 0, 1)
                df["efficiency_ratio"] = df["efficiency_ratio"].fillna(0.5)
            else:
                df["efficiency_ratio"] = 0.5

            # VWAP (ИСПРАВЛЕНО для pandas 3.0)
            df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
            df["vwap_distance"] = ((df["close"] - df["vwap"]) / df["vwap"]) * 100
            df["vwap"] = df["vwap"].fillna(df["close"])
            df["vwap_distance"] = df["vwap_distance"].fillna(0)

            # Накопление/распределение
            df["accumulation"] = df.apply(
                lambda row: ((row["close"] - row["low"]) - (row["high"] - row["close"])) / row["spread"] * row["volume"]
                if row["spread"] > 0 else 0,
                axis=1
            )

        except Exception as e:
            logger.logger.error(f"Failed to calculate market microstructure: {e}")
            # Заполняем дефолтными значениями
            df["volume_ratio"] = 1.0
            df["buy_pressure"] = 0.5
            df["efficiency_ratio"] = 0.5

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