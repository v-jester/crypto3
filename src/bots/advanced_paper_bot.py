"""
Продвинутый Paper Trading Bot с реальными данными Binance и ML стратегиями.

ИСПРАВЛЕНИЯ v3.0:
- КРИТИЧЕСКИЙ ФИКС: Фильтры применяются только к реальным сигналам (не HOLD)
- Адаптивные пороги волатильности по символам
- Снижены требования: confidence=0.65, signals=2.0
- Минимальная волатильность: 0.2% (адаптивно по символам)
- Время между сделками: 30 минут (было 2 часа)
- Улучшенная корреляция: блокирует только противоположные направления
- Правильный подсчет метрик
- Добавлены новые метрики: hold_signals, total_analyses, skipped_time_of_day, skipped_low_volume
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta
from binance import AsyncClient

from src.monitoring.logger import logger
from src.config.settings import settings
from src.data.storage.redis_client import cache_manager
from src.risk.risk_manager import RiskManager, Position
from src.data.collectors.historical_data import HistoricalDataCollector
from src.data.collectors.websocket_client import ws_client
from src.ml.models.ml_engine import ml_engine


class AdvancedPaperTradingBot:
    """Продвинутый бот с ML, риск-менеджментом и реальными данными"""

    def __init__(
        self,
        initial_balance: float = 10000.0,
        maker_fee: float = 0.001,
        taker_fee: float = 0.001,
        slippage_bps: float = 5.0
    ):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_bps = slippage_bps

        # Компоненты системы
        self.binance_client: Optional[AsyncClient] = None
        self.data_collector = HistoricalDataCollector()
        self.risk_manager = RiskManager(
            initial_capital=initial_balance,
            max_drawdown=settings.trading.MAX_DRAWDOWN_PERCENT,
            max_daily_loss=settings.trading.MAX_DAILY_LOSS_PERCENT,
            max_positions=settings.trading.MAX_POSITIONS
        )

        # Торговые данные/индикаторы
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.indicators: Dict[str, Dict[str, Any]] = {}

        # Данные старшего таймфрейма для тренда
        self.market_data_h1: Dict[str, pd.DataFrame] = {}
        self.trend_direction: Dict[str, str] = {}

        # Для логики "RSI stuck"
        self._last_bar_close_ts: Dict[str, int] = {}
        self._rsi_stuck: Dict[str, int] = {}
        self.last_rsi_values: Dict[str, float] = {}

        # Для оценки движения цены между опросами
        self._last_price: Dict[str, float] = {}
        self._tf_seconds: int = self._timeframe_to_seconds(settings.trading.PRIMARY_TIMEFRAME)

        # Серии лоссов и кулдауны
        self._loss_streak: Dict[str, int] = {}
        self._cooldown_until: Dict[str, datetime] = {}

        # Время последней сделки по символу
        self._last_trade_time: Dict[str, datetime] = {}

        # Частично закрытые позиции
        self._partially_closed: Dict[str, bool] = {}

        # Технические поля
        self.last_data_update: Dict[str, datetime] = {}
        self.data_update_interval = 60
        self.running = False
        self.connected = False

        # Настройки индикаторов/пуллинга
        self.rsi_length = 14
        self.poll_interval = 30  # оптимально для 15м таймфрейма

        # ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ v3.0
        self.min_confidence = 0.50  # снижено с 0.75
        self.min_signals_required = 1.5  # снижено с 2.5
        self.min_volatility_percent = 0.2  # базовое значение, снижено с 0.5
        self.min_time_between_trades = timedelta(minutes=30)  # снижено с 2 часов
        self.position_size_percent = 0.05  # вернули безопасные 5% вместо 10%

        # Адаптивные пороги волатильности по символам
        self.volatility_thresholds = {
            'BTCUSDT': 0.15,   # BTC менее волатилен
            'ETHUSDT': 0.20,   # ETH средняя волатильность
            'BNBUSDT': 0.20,   # BNB средняя волатильность
            'SOLUSDT': 0.25,   # SOL более волатилен
            'ADAUSDT': 0.25,   # ADA волатильна
            'DOTUSDT': 0.30,   # DOT очень волатильна
            'default': 0.20
        }

        # Коррелированные пары
        self.correlated_pairs = [
            {'BTCUSDT', 'ETHUSDT'},
            {'BNBUSDT', 'SOLUSDT'}
        ]

        # ИСПРАВЛЕННЫЕ метрики производительности
        self.performance_metrics = {
            'total_analyses': 0,           # Общее количество анализов
            'total_signals': 0,            # Сигналы != HOLD
            'hold_signals': 0,             # HOLD сигналы (нет торговли)
            'executed_signals': 0,         # Исполненные сигналы
            'skipped_low_confidence': 0,   # Отклонено по уверенности
            'skipped_low_volatility': 0,   # Отклонено по волатильности
            'skipped_time_of_day': 0,      # Отклонено по времени суток
            'skipped_cooldown': 0,         # Отклонено из-за кулдауна
            'skipped_correlation': 0,      # Отклонено по корреляции
            'skipped_time_limit': 0,       # Отклонено по времени между сделками
            'skipped_low_volume': 0,       # Отклонено по объему
            'skipped_position_exists': 0   # Отклонено - позиция уже открыта
        }

    async def initialize(self):
        """Полная инициализация бота"""
        try:
            logger.logger.info("Initializing Advanced Paper Trading Bot v3.0")
            await self._connect_binance()
            await self._load_historical_data()
            await self._initialize_ml()
            await self._start_data_streams()

            logger.logger.info(
                f"Advanced Paper Trading Bot v3.0 initialized | "
                f"Initial balance: ${self.initial_balance:,.2f} | "
                f"Symbols: {settings.trading.SYMBOLS[:3]} | "
                f"Timeframe: {settings.trading.PRIMARY_TIMEFRAME} | "
                f"Position size: {self.position_size_percent*100:.0f}%"
            )
        except Exception as e:
            logger.logger.error(f"Failed to initialize advanced bot: {e}")
            raise

    def _timeframe_to_seconds(self, tf: str) -> int:
        mapping = {
            "1m": 60, "3m": 180, "5m": 300, "15m": 900,
            "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400,
            "6h": 21600, "8h": 28800, "12h": 43200, "1d": 86400
        }
        return mapping.get(tf, 300)

    def get_adaptive_volatility_threshold(self, symbol: str) -> float:
        """Получить адаптивный порог волатильности для символа"""
        return self.volatility_thresholds.get(symbol, self.volatility_thresholds['default'])

    async def _connect_binance(self):
        """Подключение к Binance API"""
        try:
            self.binance_client = await AsyncClient.create(
                api_key=settings.api.BINANCE_API_KEY.get_secret_value(),
                api_secret=settings.api.BINANCE_API_SECRET.get_secret_value(),
                testnet=settings.api.TESTNET
            )

            await self.binance_client.ping()
            _ = await self.binance_client.get_account()

            await self.data_collector.initialize(self.binance_client)
            self.data_collector.force_refresh = True
            self.data_collector.use_cache = False

            self.connected = True
            logger.logger.info(f"Connected to Binance {'Testnet' if settings.api.TESTNET else 'Live'}")
        except Exception as e:
            logger.logger.error(f"Binance connection failed: {e}")
            self.connected = False

    async def _load_historical_data(self):
        """Загрузка исторических данных для анализа"""
        logger.logger.info("Loading historical data...")

        for symbol in settings.trading.SYMBOLS[:3]:
            try:
                # Основной таймфрейм
                self.data_collector.force_refresh = True
                self.data_collector.use_cache = False

                df = await self.data_collector.fetch_historical_data(
                    symbol=symbol,
                    interval=settings.trading.PRIMARY_TIMEFRAME,
                    days_back=1,
                    limit=100,
                    force_refresh=True
                )

                # Старший таймфрейм для тренда (H1)
                df_h1 = await self.data_collector.fetch_historical_data(
                    symbol=symbol,
                    interval='1h',
                    days_back=3,
                    limit=72,
                    force_refresh=True
                )

                if df is not None and not df.empty:
                    self.market_data[symbol] = df
                    self.last_data_update[symbol] = datetime.utcnow()

                    # Сохраняем данные H1
                    if df_h1 is not None and not df_h1.empty:
                        self.market_data_h1[symbol] = df_h1
                        # Определяем тренд
                        ema50 = ta.ema(df_h1['close'], length=50)
                        if ema50 is not None and not ema50.empty:
                            self.trend_direction[symbol] = 'UP' if df_h1['close'].iloc[-1] > ema50.iloc[-1] else 'DOWN'
                        else:
                            self.trend_direction[symbol] = 'NEUTRAL'

                    self.indicators[symbol] = {
                        'rsi': float(df['rsi'].iloc[-1]) if 'rsi' in df.columns else 50.0,
                        'macd': float(df['macd'].iloc[-1]) if 'macd' in df.columns else 0.0,
                        'bb_position': float(df['bb_percent'].iloc[-1]) if 'bb_percent' in df.columns else 0.5,
                        'volume_ratio': float(df['volume_ratio'].iloc[-1]) if 'volume_ratio' in df.columns else 1.0,
                        'atr': float(df['atr'].iloc[-1]) if 'atr' in df.columns else 0.0,
                        'price': float(df['close'].iloc[-1]) if 'close' in df.columns else 0.0
                    }
                    self.last_rsi_values[symbol] = self.indicators[symbol]['rsi']

                    # Расчет волатильности для логирования
                    atr = self.indicators[symbol]['atr']
                    price = self.indicators[symbol]['price']
                    atr_percent = (atr / price * 100) if price > 0 else 0

                    logger.logger.info(
                        f"Loaded {len(df)} candles for {symbol} | "
                        f"Price: ${price:,.2f} | "
                        f"RSI: {self.indicators[symbol]['rsi']:.1f} | "
                        f"ATR%: {atr_percent:.3f}% | "
                        f"Trend: {self.trend_direction.get(symbol, 'UNKNOWN')}"
                    )
                else:
                    logger.logger.warning(f"No data loaded for {symbol}")

            except Exception as e:
                logger.logger.warning(f"Failed to load data for {symbol}: {e}")

    async def _initialize_ml(self):
        """Инициализация ML моделей"""
        try:
            await ml_engine.load_models()
            logger.logger.info("ML models loaded from disk")
        except Exception:
            logger.logger.info("No saved models found, will train on first suitable data")

    async def _start_data_streams(self):
        """Запуск потоков данных"""
        if settings.api.TESTNET:
            logger.logger.warning("Using REST API polling for testnet (WebSocket issues)")
            self.running = True
            asyncio.create_task(self._poll_market_data())
        else:
            await self._start_websocket_streams()

    async def _start_websocket_streams(self):
        """Запуск WebSocket стримов для реальных данных"""
        if not self.connected:
            logger.logger.warning("Skipping websocket streams - not connected to Binance")
            return

        try:
            if hasattr(ws_client, "set_client"):
                ws_client.set_client(self.binance_client)

            await ws_client.subscribe_klines(
                symbols=settings.trading.SYMBOLS[:3],
                intervals=[settings.trading.PRIMARY_TIMEFRAME],
                handler=self._handle_kline_update
            )

            await ws_client.subscribe_ticker(
                symbols=settings.trading.SYMBOLS[:3],
                handler=self._handle_ticker_update
            )

            await ws_client.start()
            logger.logger.info("WebSocket streams started")
        except Exception as e:
            logger.logger.error(f"Failed to start websocket streams: {e}")
            self.running = True
            asyncio.create_task(self._poll_market_data())

    async def _get_current_price(self, symbol: str, df_fallback: Optional[pd.DataFrame] = None) -> float:
        """Получить текущую цену symbol через REST-тикер"""
        try:
            if self.binance_client is not None:
                ticker = await self.binance_client.get_ticker(symbol=symbol)
                price = float(ticker['lastPrice'])
                if price > 0:
                    return price
        except Exception:
            pass

        if df_fallback is not None and not df_fallback.empty and 'close' in df_fallback.columns:
            return float(df_fallback['close'].iloc[-1])

        raise RuntimeError(f"No price source for {symbol}")

    def _extract_close_ts(self, df: pd.DataFrame) -> int:
        """Достаём timestamp последней свечи (UTC, сек) из 'close_time' или индекса"""
        if df is None or df.empty:
            return int(time.time())

        if 'close_time' in df.columns:
            ct = df['close_time'].iloc[-1]
        else:
            try:
                ct = df.index[-1]
            except Exception:
                return int(time.time())

        if isinstance(ct, pd.Timestamp):
            return int(ct.tz_convert('UTC').timestamp()) if ct.tzinfo else int(ct.timestamp())
        if isinstance(ct, (int, float)):
            return int(ct / 1000) if ct > 1_000_000_000_000 else int(ct)
        if isinstance(ct, str):
            return int(pd.to_datetime(ct, utc=True).timestamp())

        return int(time.time())

    async def _poll_market_data(self):
        """Обновление данных через REST API с оптимизированным интервалом"""
        logger.logger.info(f"Starting REST API polling for market data (interval: {self.poll_interval}s)")

        while self.running:
            try:
                for symbol in settings.trading.SYMBOLS[:3]:
                    try:
                        now = datetime.utcnow()

                        # Инвалидация кэша
                        for pattern in (
                            f"historical:{symbol}:*",
                            f"kline:{symbol}:*",
                            f"market:{symbol}:*",
                            f"ticker:{symbol}:*",
                            f"indicators:{symbol}:*"
                        ):
                            try:
                                deleted = await cache_manager.invalidate_pattern(pattern)
                                if deleted > 0:
                                    logger.logger.debug(f"Cleared {deleted} cache keys for pattern {pattern}")
                            except Exception as e:
                                logger.logger.debug(f"Cache clear failed for {pattern}: {e}")

                        # Обновляем исторические данные
                        self.data_collector.force_refresh = True
                        self.data_collector.use_cache = False

                        df = await self.data_collector.fetch_historical_data(
                            symbol=symbol,
                            interval=settings.trading.PRIMARY_TIMEFRAME,
                            days_back=1,
                            limit=100,
                            force_refresh=True
                        )

                        # Обновляем H1 каждые 5 минут
                        if symbol not in self.market_data_h1 or \
                           (now - self.last_data_update.get(f"{symbol}_h1", datetime.min)) > timedelta(minutes=5):
                            df_h1 = await self.data_collector.fetch_historical_data(
                                symbol=symbol,
                                interval='1h',
                                days_back=3,
                                limit=72,
                                force_refresh=True
                            )
                            if df_h1 is not None and not df_h1.empty:
                                self.market_data_h1[symbol] = df_h1
                                self.last_data_update[f"{symbol}_h1"] = now
                                # Обновляем тренд
                                ema50 = ta.ema(df_h1['close'], length=50)
                                if ema50 is not None and not ema50.empty:
                                    old_trend = self.trend_direction.get(symbol, 'UNKNOWN')
                                    new_trend = 'UP' if df_h1['close'].iloc[-1] > ema50.iloc[-1] else 'DOWN'
                                    self.trend_direction[symbol] = new_trend
                                    if old_trend != new_trend:
                                        logger.logger.info(f"Trend changed for {symbol}: {old_trend} → {new_trend}")

                        if df is None or df.empty:
                            logger.logger.warning(f"Empty dataframe received for {symbol}")
                            continue

                        # Текущая цена
                        current_price = await self._get_current_price(symbol, df_fallback=df)

                        # ЖИВОЙ RSI
                        df_live = df.copy()
                        if 'close' in df_live.columns:
                            df_live.iloc[-1, df_live.columns.get_loc('close')] = float(current_price)

                        rsi_series = ta.rsi(df_live['close'], length=self.rsi_length)
                        rsi_live = float(rsi_series.iloc[-1]) if rsi_series is not None and not rsi_series.empty else float('nan')

                        # Сохраняем свежие данные и индикаторы
                        old_rsi = self.indicators.get(symbol, {}).get('rsi', np.nan)
                        old_price = self.indicators.get(symbol, {}).get('price', np.nan)

                        self.market_data[symbol] = df
                        self.last_data_update[symbol] = now
                        self.indicators.setdefault(symbol, {})
                        self.indicators[symbol].update({
                            'rsi': rsi_live,
                            'price': float(current_price),
                            'macd': float(df['macd'].iloc[-1]) if 'macd' in df.columns else 0.0,
                            'bb_position': float(df['bb_percent'].iloc[-1]) if 'bb_percent' in df.columns else 0.5,
                            'volume_ratio': float(df['volume_ratio'].iloc[-1]) if 'volume_ratio' in df.columns else 1.0,
                            'atr': float(df['atr'].iloc[-1]) if 'atr' in df.columns else 0.0,
                        })

                        # Логика "RSI stuck"
                        last_close_ts = self._extract_close_ts(df)
                        prev_ts = self._last_bar_close_ts.get(symbol)
                        prev_rsi = self.indicators[symbol].get('rsi_prev')

                        prev_price = self._last_price.get(symbol)
                        price_move_bp = 0.0
                        if prev_price and prev_price > 0:
                            price_move_bp = abs(float(current_price) - float(prev_price)) / float(prev_price) * 10000.0
                        self._last_price[symbol] = float(current_price)

                        MIN_BP_TO_CHECK = 5.0
                        RSI_EPS = 0.20

                        if prev_ts is None or last_close_ts != prev_ts:
                            self._last_bar_close_ts[symbol] = last_close_ts
                            self._rsi_stuck[symbol] = 0
                        else:
                            if price_move_bp < MIN_BP_TO_CHECK or pd.isna(rsi_live) or prev_rsi is None:
                                pass
                            else:
                                if abs(rsi_live - prev_rsi) < RSI_EPS:
                                    cnt = self._rsi_stuck.get(symbol, 0) + 1
                                    self._rsi_stuck[symbol] = cnt
                                    msg = (f"⚠️ RSI stuck for {symbol}: {rsi_live:.2f} "
                                           f"(no change for {cnt} polls; ΔP≈{price_move_bp:.1f}bp)")
                                    if cnt in (3, 4, 5):
                                        if getattr(settings.api, "TESTNET", False):
                                            logger.logger.debug(msg)
                                        else:
                                            logger.logger.warning(msg)
                                else:
                                    self._rsi_stuck[symbol] = 0

                        self.indicators[symbol]['rsi_prev'] = rsi_live

                        # Лог при существенном изменении
                        if pd.notna(old_rsi) and pd.notna(rsi_live) and abs(rsi_live - old_rsi) >= 0.1:
                            atr = self.indicators[symbol]['atr']
                            atr_percent = (atr / current_price * 100) if current_price > 0 else 0
                            logger.logger.info(
                                f"✅ Data updated for {symbol} | "
                                f"Price: {old_price if not np.isnan(old_price) else '—'} → ${current_price:.2f} | "
                                f"RSI: {old_rsi if not np.isnan(old_rsi) else '—'} → {rsi_live:.1f} | "
                                f"ATR%: {atr_percent:.3f}% | "
                                f"Trend: {self.trend_direction.get(symbol, 'UNKNOWN')}"
                            )
                        self.last_rsi_values[symbol] = rsi_live

                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.logger.error(f"Failed to poll data for {symbol}: {e}", exc_info=True)

                await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                logger.logger.info("Market data polling cancelled")
                break
            except Exception as e:
                logger.logger.error(f"Error in polling loop: {e}", exc_info=True)
                await asyncio.sleep(10)

    async def _handle_kline_update(self, data: Dict):
        """Обработка обновлений свечей (WS)"""
        try:
            kline = data.get('k', {})
            if kline.get('x'):  # Свеча закрылась
                symbol = kline.get('s')
                logger.logger.debug(f"Closed candle for {symbol}: {kline.get('c')}")
                await self._update_market_data(symbol)
        except Exception as e:
            logger.logger.error(f"Error handling kline update: {e}")

    async def _handle_ticker_update(self, data: Dict):
        """Обработка обновлений тикера (WS)"""
        try:
            symbol = data.get('s')
            price = float(data.get('c', 0))
            if symbol and price > 0 and symbol in self.positions:
                await self._update_position_pnl(symbol, price)
        except Exception as e:
            logger.logger.error(f"Error handling ticker update: {e}")

    async def _update_market_data(self, symbol: str):
        """Принудительное обновление рыночных данных и индикаторов"""
        try:
            self.data_collector.force_refresh = True
            self.data_collector.use_cache = False

            deleted = await cache_manager.invalidate_symbol(symbol)
            if deleted > 0:
                logger.logger.debug(f"Cleared {deleted} cache keys for {symbol}")

            df = await self.data_collector.fetch_historical_data(
                symbol=symbol,
                interval=settings.trading.PRIMARY_TIMEFRAME,
                days_back=1,
                limit=100,
                force_refresh=True
            )

            if df is not None and not df.empty:
                self.market_data[symbol] = df
                self.last_data_update[symbol] = datetime.utcnow()

                self.indicators.setdefault(symbol, {})
                self.indicators[symbol].update({
                    'rsi': float(df['rsi'].iloc[-1]) if 'rsi' in df.columns else 50.0,
                    'macd': float(df['macd'].iloc[-1]) if 'macd' in df.columns else 0.0,
                    'bb_position': float(df['bb_percent'].iloc[-1]) if 'bb_percent' in df.columns else 0.5,
                    'volume_ratio': float(df['volume_ratio'].iloc[-1]) if 'volume_ratio' in df.columns else 1.0,
                    'atr': float(df['atr'].iloc[-1]) if 'atr' in df.columns else 0.0,
                    'price': float(df['close'].iloc[-1]) if 'close' in df.columns else 0.0
                })

        except Exception as e:
            logger.logger.error(f"Failed to update market data for {symbol}: {e}")

    async def run(self):
        """Основной торговый цикл"""
        self.running = True
        logger.logger.info("Starting Advanced Paper Trading Bot v3.0 main loop")

        analysis_counter = 0

        try:
            while self.running:
                # Анализ каждые 60 секунд
                if analysis_counter % 2 == 0:  # 60 секунд (30 сек * 2)
                    await self._analyze_and_trade()

                # Обновляем позиции
                await self._update_positions()

                # Проверяем риски
                await self._check_risk_limits()

                # Логируем статус каждые 2 минуты
                if analysis_counter % 4 == 0:  # 120 секунд
                    await self._log_status()
                    analysis_counter = 0

                analysis_counter += 1
                await asyncio.sleep(30)

        except asyncio.CancelledError:
            logger.logger.info("Trading loop cancelled")
        except Exception as e:
            logger.logger.error(f"Error in trading loop: {e}")
            raise
        finally:
            logger.logger.info("Advanced Paper Trading Bot main loop stopped")

    async def _analyze_and_trade(self):
        """ИСПРАВЛЕННЫЙ метод анализа рынка - фильтры применяются только к реальным сигналам"""
        for symbol in settings.trading.SYMBOLS[:3]:
            try:
                # Увеличиваем счетчик общих анализов
                self.performance_metrics['total_analyses'] += 1

                # === БАЗОВЫЕ ПРОВЕРКИ ДАННЫХ ===
                if symbol not in self.market_data:
                    logger.logger.debug(f"No market data for {symbol}")
                    continue

                df = self.market_data[symbol]
                if df.empty or len(df) < 50:
                    logger.logger.debug(f"Insufficient data for {symbol}: {len(df)} candles")
                    continue

                indicators = self.indicators.get(symbol, {})
                current_price = indicators.get('price', float(df['close'].iloc[-1]) if not df.empty else 0.0)
                if current_price == 0:
                    logger.logger.warning(f"Zero price for {symbol}, skipping")
                    continue

                # === ГЕНЕРАЦИЯ СИГНАЛА (БЕЗ ФИЛЬТРОВ) ===
                signal = await self._generate_trading_signal(symbol, df, indicators)

                # Проверяем тип сигнала
                if signal['action'] == 'HOLD':
                    self.performance_metrics['hold_signals'] += 1
                    # Нет сигнала - нет проверок фильтров, просто переходим к следующему символу
                    continue

                # === У НАС ЕСТЬ РЕАЛЬНЫЙ СИГНАЛ (BUY/SELL) ===
                self.performance_metrics['total_signals'] += 1

                logger.logger.info(
                    f"📊 Signal generated for {symbol} | "
                    f"Action: {signal['action']} | "
                    f"Confidence: {signal['confidence']:.2f} | "
                    f"Trend: {self.trend_direction.get(symbol, 'UNKNOWN')} | "
                    f"Reasons: {', '.join(signal['reasons'])}"
                )

                # === ТЕПЕРЬ ПРИМЕНЯЕМ ФИЛЬТРЫ К РЕАЛЬНОМУ СИГНАЛУ ===

                # 1. Проверка существующей позиции
                if symbol in self.positions:
                    logger.logger.debug(f"Position already exists for {symbol}")
                    self.performance_metrics['skipped_position_exists'] += 1
                    continue

                # 2. Фильтр уверенности
                if signal['confidence'] < self.min_confidence:
                    logger.logger.debug(
                        f"Signal confidence too low for {symbol}: "
                        f"{signal['confidence']:.2f} < {self.min_confidence}"
                    )
                    self.performance_metrics['skipped_low_confidence'] += 1
                    continue

                # 3. Фильтр времени суток (низкая ликвидность)
                current_hour = datetime.utcnow().hour
                if current_hour in [22, 23, 0, 1, 2, 3]:
                    logger.logger.debug(f"Low liquidity hours for {symbol} (hour: {current_hour} UTC)")
                    self.performance_metrics['skipped_time_of_day'] += 1
                    continue

                # 4. Фильтр кулдауна после убытков
                if symbol in self._cooldown_until and datetime.utcnow() < self._cooldown_until[symbol]:
                    remaining = (self._cooldown_until[symbol] - datetime.utcnow()).total_seconds() / 60
                    logger.logger.debug(f"Cooldown active for {symbol} ({remaining:.1f} min remaining)")
                    self.performance_metrics['skipped_cooldown'] += 1
                    continue

                # 5. Фильтр времени между сделками
                if symbol in self._last_trade_time:
                    time_since_last = datetime.utcnow() - self._last_trade_time[symbol]
                    if time_since_last < self.min_time_between_trades:
                        minutes_since = time_since_last.total_seconds() / 60
                        logger.logger.debug(
                            f"Too soon for new trade on {symbol}: "
                            f"{minutes_since:.1f} min < {self.min_time_between_trades.total_seconds()/60:.0f} min"
                        )
                        self.performance_metrics['skipped_time_limit'] += 1
                        continue

                # 6. АДАПТИВНЫЙ фильтр волатильности
                atr = indicators.get('atr', 0.0)
                atr_percent = (atr / current_price) * 100 if current_price > 0 else 0
                min_vol = self.get_adaptive_volatility_threshold(symbol)

                if atr_percent < min_vol:
                    logger.logger.debug(
                        f"Low volatility for {symbol}: ATR={atr_percent:.3f}% < {min_vol:.2f}% threshold"
                    )
                    self.performance_metrics['skipped_low_volatility'] += 1
                    continue

                # 7. Фильтр объема
                volume_ratio = indicators.get('volume_ratio', 1.0)
                if volume_ratio < 1.0:
                    logger.logger.debug(f"Insufficient volume for {symbol}: ratio={volume_ratio:.2f} < 1.2")
                    self.performance_metrics['skipped_low_volume'] += 1
                    continue

                # 8. УЛУЧШЕННЫЙ фильтр корреляции (только противоположные направления)
                if await self._check_correlation_conflict(symbol, signal['action']):
                    logger.logger.debug(f"Correlation conflict for {symbol} ({signal['action']})")
                    self.performance_metrics['skipped_correlation'] += 1
                    continue

                # === ВСЕ ФИЛЬТРЫ ПРОЙДЕНЫ - ИСПОЛНЯЕМ СДЕЛКУ ===
                logger.logger.info(
                    f"✅ All filters passed for {symbol} | "
                    f"Executing {signal['action']} trade | "
                    f"ATR: {atr_percent:.3f}% | Volume: {volume_ratio:.1f}x"
                )

                await self._execute_trade(symbol, signal, current_price)
                self.performance_metrics['executed_signals'] += 1

            except Exception as e:
                logger.logger.error(f"Analysis failed for {symbol}: {e}", exc_info=True)

    async def _check_correlation_conflict(self, symbol: str, direction: str) -> bool:
        """УЛУЧШЕННАЯ проверка корреляции - блокирует только противоположные позиции"""
        for group in self.correlated_pairs:
            if symbol in group:
                for other_symbol in group:
                    if other_symbol != symbol and other_symbol in self.positions:
                        other_position = self.positions[other_symbol]
                        # Блокируем только если направления ПРОТИВОПОЛОЖНЫ
                        if (direction == 'BUY' and other_position.side == 'SELL') or \
                           (direction == 'SELL' and other_position.side == 'BUY'):
                            logger.logger.debug(
                                f"Correlation conflict: {symbol} {direction} vs "
                                f"{other_symbol} {other_position.side}"
                            )
                            return True
        return False

    async def _generate_trading_signal(self, symbol: str, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Генерация торгового сигнала с оптимизированными параметрами"""
        signal = {
            'action': 'HOLD',
            'confidence': 0.0,
            'reasons': [],
            'stop_loss': None,
            'take_profit': None
        }

        buy_signals = 0.0
        sell_signals = 0.0

        # 1) RSI с умеренными порогами (30/70 вместо 35/65)
        rsi = indicators.get('rsi', 50.0)
        if rsi < 30:
            buy_signals += 1.5
            signal['reasons'].append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 70:
            sell_signals += 1.5
            signal['reasons'].append(f"RSI overbought ({rsi:.1f})")
        elif rsi < 40:
            buy_signals += 0.5
            signal['reasons'].append(f"RSI low ({rsi:.1f})")
        elif rsi > 60:
            sell_signals += 0.5
            signal['reasons'].append(f"RSI high ({rsi:.1f})")

        # 2) MACD
        macd = indicators.get('macd', 0.0)
        if 'macd_signal' in df.columns and len(df) > 0:
            macd_signal_val = float(df['macd_signal'].iloc[-1])
            macd_diff = macd - macd_signal_val
            if macd_diff > 0:
                buy_signals += 0.8
                signal['reasons'].append(f"MACD bullish ({macd_diff:.4f})")
            elif macd_diff < 0:
                sell_signals += 0.8
                signal['reasons'].append(f"MACD bearish ({macd_diff:.4f})")

        # 3) Bollinger Bands (умеренные пороги)
        bb_position = indicators.get('bb_position', 0.5)
        if bb_position < 0.20:
            buy_signals += 1.0
            signal['reasons'].append(f"Price near lower BB ({bb_position:.2f})")
        elif bb_position > 0.80:
            sell_signals += 1.0
            signal['reasons'].append(f"Price near upper BB ({bb_position:.2f})")

        # 4) Volume (умеренный порог)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:  # снижено с 1.8
            if buy_signals > sell_signals:
                buy_signals += 0.5
                signal['reasons'].append(f"Volume confirmation ({volume_ratio:.1f}x)")
            elif sell_signals > buy_signals:
                sell_signals += 0.5
                signal['reasons'].append(f"Volume confirmation ({volume_ratio:.1f}x)")

        # 5) Тренд по старшему ТФ (H1)
        trend = self.trend_direction.get(symbol, 'NEUTRAL')
        close_ = float(df['close'].iloc[-1])

        if trend == 'UP':
            if buy_signals > sell_signals:
                buy_signals += 0.8
                signal['reasons'].append("Trend alignment (UP)")
            elif sell_signals > buy_signals:
                sell_signals *= 0.5  # умеренный штраф за контртренд
                signal['reasons'].append("Counter-trend SHORT")
        elif trend == 'DOWN':
            if sell_signals > buy_signals:
                sell_signals += 0.8
                signal['reasons'].append("Trend alignment (DOWN)")
            elif buy_signals > sell_signals:
                buy_signals *= 0.5
                signal['reasons'].append("Counter-trend LONG")

        # 6) EMA200 на текущем ТФ
        ema200 = float(df['ema200'].iloc[-1]) if 'ema200' in df.columns else None
        if ema200 is None:
            try:
                ema200_series = ta.ema(df['close'], length=200)
                ema200 = float(ema200_series.iloc[-1]) if ema200_series is not None and not ema200_series.empty else None
            except Exception:
                ema200 = None

        if ema200:
            local_trend_up = close_ > ema200
            if local_trend_up and buy_signals > sell_signals:
                buy_signals += 0.3
                signal['reasons'].append("Above EMA200")
            elif not local_trend_up and sell_signals > buy_signals:
                sell_signals += 0.3
                signal['reasons'].append("Below EMA200")

        # 7) ML предсказания (если доступны)
        try:
            if hasattr(ml_engine, 'is_trained') and ml_engine.is_trained:
                ml_prediction = await ml_engine.predict(df, return_proba=True)
                if ml_prediction and 'prediction_label' in ml_prediction:
                    if ml_prediction['prediction_label'] == 'UP' and ml_prediction.get('confidence', 0) > 0.55:
                        buy_signals += 1.0
                        signal['reasons'].append(f"ML: UP ({ml_prediction['confidence']:.1%})")
                    elif ml_prediction['prediction_label'] == 'DOWN' and ml_prediction.get('confidence', 0) > 0.55:
                        sell_signals += 1.0
                        signal['reasons'].append(f"ML: DOWN ({ml_prediction['confidence']:.1%})")
        except Exception as e:
            logger.logger.debug(f"ML prediction skipped: {e}")

        # Финальное решение с умеренными порогами
        atr = float(indicators.get('atr', 0))
        if atr == 0 or pd.isna(atr):
            atr = close_ * 0.003  # 0.3% fallback

        # Убедимся что ATR реалистичный
        min_atr = close_ * 0.002  # 0.2% минимум
        max_atr = close_ * 0.05   # 5% максимум
        atr = max(min_atr, min(atr, max_atr))

        atr_percent = (atr / close_) * 100

        # Адаптивные множители для SL/TP
        volatility_multiplier_sl = 2.0 if atr_percent < 0.5 else 2.5
        volatility_multiplier_tp = 3.0 if atr_percent < 0.5 else 3.5

        if buy_signals >= self.min_signals_required and buy_signals > sell_signals:
            signal['action'] = 'BUY'
            signal['confidence'] = min(buy_signals / 4.0, 0.85)
            sl = close_ - (volatility_multiplier_sl * atr)
            tp = close_ + (volatility_multiplier_tp * atr)
            if sl >= close_:
                sl = close_ - max(atr, close_ * 0.005)
            if tp <= close_:
                tp = close_ + max(atr, close_ * 0.005)
            signal['stop_loss'] = float(sl)
            signal['take_profit'] = float(tp)

        elif sell_signals >= self.min_signals_required and sell_signals > buy_signals:
            signal['action'] = 'SELL'
            signal['confidence'] = min(sell_signals / 4.0, 0.85)
            sl = close_ + (volatility_multiplier_sl * atr)
            tp = close_ - (volatility_multiplier_tp * atr)
            if sl <= close_:
                sl = close_ + max(atr, close_ * 0.005)
            if tp >= close_:
                tp = close_ - max(atr, close_ * 0.005)
            signal['stop_loss'] = float(sl)
            signal['take_profit'] = float(tp)

        return signal

    async def _execute_trade(self, symbol: str, signal: Dict, current_price: float):
        """Выполнение торговой операции"""
        try:
            # Размер позиции: 5% от баланса (безопасный размер)
            position_value = self.current_balance * self.position_size_percent

            # Минимальный размер позиции
            min_position_values = {
                'BTCUSDT': 10.0,
                'ETHUSDT': 10.0,
                'BNBUSDT': 10.0,
                'SOLUSDT': 10.0,
                'default': 10.0
            }
            min_value = min_position_values.get(symbol, min_position_values['default'])
            if position_value < min_value:
                logger.logger.warning(
                    f"Position size too small for {symbol}: ${position_value:.2f} < ${min_value}"
                )
                return

            position_size = position_value / current_price

            # Округление
            step_sizes = {
                'BTCUSDT': 0.00001,
                'ETHUSDT': 0.0001,
                'BNBUSDT': 0.001,
                'SOLUSDT': 0.001,
                'default': 0.001
            }
            step_size = step_sizes.get(symbol, step_sizes['default'])
            position_size = round(position_size / step_size) * step_size

            final_value = position_size * current_price
            if final_value < min_value:
                position_size = (min_value * 1.1) / current_price
                position_size = round(position_size / step_size) * step_size

            required_balance = position_size * current_price * 1.01
            if required_balance > self.current_balance:
                logger.logger.warning(
                    f"Insufficient balance for {symbol}: "
                    f"Required ${required_balance:.2f} > Available ${self.current_balance:.2f}"
                )
                return

            can_open, reason = self.risk_manager.can_open_position(
                symbol=symbol,
                proposed_size=position_size * current_price
            )
            if not can_open:
                logger.logger.warning(f"Risk manager rejected position for {symbol}: {reason}")
                return

            # Слиппаж
            slippage = self.slippage_bps / 10000
            entry_price = current_price * (1 + slippage) if signal['action'] == 'BUY' else current_price * (1 - slippage)

            # Финальная валидация SL/TP
            atr = float(self.indicators.get(symbol, {}).get('atr', current_price * 0.003))
            atr = abs(atr) if atr == atr else current_price * 0.003
            sl = signal.get('stop_loss')
            tp = signal.get('take_profit')

            if signal['action'] == 'BUY':
                if sl is None or sl >= entry_price:
                    sl = entry_price - max(atr * 2, entry_price * 0.01)
                if tp is None or tp <= entry_price:
                    tp = entry_price + max(atr * 3, entry_price * 0.015)
            else:  # SELL
                if sl is None or sl <= entry_price:
                    sl = entry_price + max(atr * 2, entry_price * 0.01)
                if tp is None or tp >= entry_price:
                    tp = entry_price - max(atr * 3, entry_price * 0.015)

            signal['stop_loss'], signal['take_profit'] = float(sl), float(tp)

            position = Position(
                id=f"{symbol}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                symbol=symbol,
                side=signal['action'],
                entry_price=entry_price,
                quantity=position_size,
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profit'],
                entry_time=datetime.utcnow(),
                current_price=entry_price,
                risk_amount=abs(entry_price - signal['stop_loss']) * position_size if signal['stop_loss'] is not None else 0.0
            )

            # Регистрируем позицию
            self.positions[symbol] = position
            self.risk_manager.add_position(position)
            self._last_trade_time[symbol] = datetime.utcnow()
            self._partially_closed[symbol] = False

            # Списываем стоимость позиции и комиссию
            position_cost = position_size * entry_price
            open_fee = position_cost * self.taker_fee
            setattr(position, "open_fee", float(open_fee))
            self.current_balance -= (position_cost + open_fee)

            logger.logger.info(
                f"✅ POSITION OPENED | Symbol: {symbol} | "
                f"Side: {signal['action']} | Entry: ${entry_price:.2f} | "
                f"Size: {position_size:.6f} | Value: ${position_cost:.2f} | "
                f"Balance: ${self.current_balance:.2f}"
            )
            logger.logger.info(
                f"Position details | SL: ${signal['stop_loss']:.2f} | "
                f"TP: ${signal['take_profit']:.2f} | "
                f"Risk: ${position.risk_amount:.2f} | "
                f"Trend: {self.trend_direction.get(symbol, 'UNKNOWN')} | "
                f"Reasons: {', '.join(signal['reasons'])}"
            )

        except Exception as e:
            logger.logger.error(f"Failed to execute trade for {symbol}: {e}", exc_info=True)

    async def _update_positions(self):
        """Обновление открытых позиций с частичным закрытием и трейлингом"""
        for symbol, position in list(self.positions.items()):
            try:
                current_price = self.indicators.get(symbol, {}).get('price')
                if (not current_price) or abs(current_price - position.current_price) < 0.01:
                    if self.binance_client is not None:
                        ticker = await self.binance_client.get_ticker(symbol=symbol)
                        current_price = float(ticker['lastPrice'])
                        logger.logger.debug(f"Fetched fresh price for {symbol}: ${current_price:.2f}")

                # Обновляем PnL
                await self._update_position_pnl(symbol, current_price)

                # Частичное закрытие при достижении 1.5*ATR прибыли
                atr = self.indicators.get(symbol, {}).get('atr')
                if atr and atr > 0 and not self._partially_closed.get(symbol, False):
                    if position.side == 'BUY':
                        profit_distance = current_price - position.entry_price
                    else:
                        profit_distance = position.entry_price - current_price

                    if profit_distance > 1.5 * atr:
                        # Закрываем 50% позиции
                        await self._partial_close_position(symbol, current_price, 0.5, "partial_tp_1.5atr")
                        self._partially_closed[symbol] = True

                # Трейлинг-стоп по ATR
                if atr and atr > 0:
                    if position.side == 'BUY':
                        if current_price - position.entry_price > 1.5 * atr:
                            new_sl = max(position.stop_loss or 0.0, position.entry_price + 0.5 * atr)
                            if position.stop_loss is None or new_sl > position.stop_loss:
                                position.stop_loss = new_sl
                                logger.logger.debug(f"[{symbol}] Trailing SL moved to ${new_sl:.2f} (BUY)")
                    else:
                        if position.entry_price - current_price > 1.5 * atr:
                            new_sl = min(position.stop_loss or float('inf'), position.entry_price - 0.5 * atr)
                            if position.stop_loss is None or new_sl < position.stop_loss:
                                position.stop_loss = new_sl
                                logger.logger.debug(f"[{symbol}] Trailing SL moved to ${new_sl:.2f} (SELL)")

                # Обновление риск-менеджера
                actions = self.risk_manager.update_position(
                    position_id=position.id,
                    current_price=current_price,
                    atr=atr
                )
                if actions and actions.get('action') == 'close':
                    await self._close_position(symbol, current_price, actions.get('reason', 'risk_management'))
                    continue

                # Проверка SL/TP
                if position.stop_loss is not None:
                    if position.side == 'BUY' and current_price <= position.stop_loss:
                        await self._close_position(symbol, current_price, "stop_loss_hit")
                        continue
                    if position.side == 'SELL' and current_price >= position.stop_loss:
                        await self._close_position(symbol, current_price, "stop_loss_hit")
                        continue

                if position.take_profit is not None:
                    if position.side == 'BUY' and current_price >= position.take_profit:
                        await self._close_position(symbol, current_price, "take_profit_hit")
                        continue
                    if position.side == 'SELL' and current_price <= position.take_profit:
                        await self._close_position(symbol, current_price, "take_profit_hit")
                        continue

            except Exception as e:
                logger.logger.error(f"Failed to update position for {symbol}: {e}")

    async def _partial_close_position(self, symbol: str, close_price: float, percentage: float, reason: str):
        """Частичное закрытие позиции"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        close_quantity = position.quantity * percentage

        # Расчёт PnL для частичного закрытия
        gross = (close_price - position.entry_price) * close_quantity if position.side == 'BUY' \
            else (position.entry_price - close_price) * close_quantity

        close_fee = close_quantity * close_price * self.taker_fee
        partial_pnl = gross - close_fee

        # Обновляем баланс
        if position.side == 'BUY':
            self.current_balance += (close_quantity * close_price - close_fee)
        else:
            entry_value = close_quantity * position.entry_price
            self.current_balance += entry_value + partial_pnl

        # Уменьшаем размер позиции
        position.quantity -= close_quantity

        logger.logger.info(
            f"💰 PARTIAL CLOSE | Symbol: {symbol} | "
            f"Closed: {percentage*100:.0f}% | Price: ${close_price:.2f} | "
            f"Partial PnL: ${partial_pnl:+.2f} | Reason: {reason}"
        )

        # Сохраняем в историю
        self.trade_history.append({
            'symbol': symbol,
            'side': position.side,
            'type': 'partial_close',
            'entry_price': position.entry_price,
            'close_price': close_price,
            'quantity': close_quantity,
            'pnl': partial_pnl,
            'fees': {'close': close_fee},
            'reason': reason,
            'timestamp': datetime.utcnow()
        })

    async def _close_position(self, symbol: str, close_price: float, reason: str):
        """Полное закрытие позиции"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        # Валовая прибыль/убыток
        gross = (close_price - position.entry_price) * position.quantity if position.side == 'BUY' \
            else (position.entry_price - close_price) * position.quantity

        # Комиссии
        close_fee = position.quantity * close_price * self.taker_fee
        open_fee = float(getattr(position, "open_fee", 0.0))

        # Итоговый PnL
        pnl = gross - (open_fee + close_fee)

        # Обновляем баланс
        if position.side == 'BUY':
            self.current_balance += (position.quantity * close_price - close_fee)
        else:
            entry_value = position.quantity * position.entry_price
            self.current_balance += entry_value + pnl

        # Закрываем в риск-менеджере
        self.risk_manager.close_position(position.id, close_price, reason)

        # Удаляем позицию
        del self.positions[symbol]
        if symbol in self._partially_closed:
            del self._partially_closed[symbol]

        # Сохраняем в историю
        self.trade_history.append({
            'symbol': symbol,
            'side': position.side,
            'type': 'full_close',
            'entry_price': position.entry_price,
            'close_price': close_price,
            'quantity': position.quantity,
            'pnl': pnl,
            'fees': {'open': open_fee, 'close': close_fee, 'total': open_fee + close_fee},
            'reason': reason,
            'timestamp': datetime.utcnow()
        })

        # Обновляем streak/cooldown для убыточных сделок
        if pnl < 0:
            self._loss_streak[symbol] = self._loss_streak.get(symbol, 0) + 1
            if self._loss_streak[symbol] >= 2:
                self._cooldown_until[symbol] = datetime.utcnow() + timedelta(minutes=45)
                self._loss_streak[symbol] = 0
                logger.logger.info(
                    f"[{symbol}] Loss streak reached (2 losses). "
                    f"Cooldown 45 min until {self._cooldown_until[symbol].strftime('%H:%M:%S')} UTC"
                )
        else:
            self._loss_streak[symbol] = 0

        # Логируем закрытие
        pnl_emoji = "💰" if pnl > 0 else "💸"
        logger.logger.info(
            f"{pnl_emoji} POSITION CLOSED | Symbol: {symbol} | "
            f"Side: {position.side} | Close: ${close_price:.2f} | "
            f"PnL: ${pnl:+.2f} | Reason: {reason} | "
            f"Fees: open ${open_fee:.2f}, close ${close_fee:.2f}"
        )

    async def _update_position_pnl(self, symbol: str, current_price: float):
        """Обновление P&L позиции"""
        if symbol in self.positions:
            position = self.positions[symbol]
            position.current_price = current_price
            if position.side == 'BUY':
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            else:
                position.unrealized_pnl = (position.entry_price - current_price) * position.quantity

    async def _check_risk_limits(self):
        """Проверка лимитов риска"""
        try:
            risk_metrics = self.risk_manager.get_risk_metrics()
            if risk_metrics.risk_level.value == "critical":
                logger.logger.warning("⚠️ CRITICAL RISK LEVEL - closing all positions")
                for symbol in list(self.positions.keys()):
                    current_price = self.indicators.get(symbol, {}).get('price', self.positions[symbol].entry_price)
                    await self._close_position(symbol, current_price, "risk_limit_exceeded")
        except Exception as e:
            logger.logger.error(f"Error checking risk limits: {e}")

    async def _log_status(self):
        """Расширенное логирование статуса бота с исправленными метриками"""
        try:
            positions_value = sum(
                pos.quantity * (pos.current_price if pos.current_price > 0 else pos.entry_price)
                for pos in self.positions.values()
            )
            total_unrealized_pnl = sum(
                getattr(pos, 'unrealized_pnl', 0.0)
                for pos in self.positions.values()
            )
            equity = self.current_balance + positions_value
            total_pnl = equity - self.initial_balance

            # Расчёт дополнительных метрик
            all_trades = [t for t in self.trade_history if t.get('type') != 'partial_close']
            wins = [t for t in all_trades if t['pnl'] > 0]
            losses = [t for t in all_trades if t['pnl'] < 0]

            win_rate = (len(wins) / len(all_trades) * 100) if all_trades else 0
            avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
            avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
            profit_factor = abs(sum(t['pnl'] for t in wins) / sum(t['pnl'] for t in losses)) if losses and sum(t['pnl'] for t in losses) != 0 else 0

            status = {
                "balance": round(self.current_balance, 2),
                "positions_value": round(positions_value, 2),
                "equity": round(equity, 2),
                "positions": len(self.positions),
                "trades": len(all_trades),
                "realized_pnl": round(sum(t.get('pnl', 0.0) for t in all_trades), 2),
                "unrealized_pnl": round(total_unrealized_pnl, 2),
                "total_pnl": round(total_pnl, 2),
                "return_percent": round((total_pnl / self.initial_balance) * 100, 2) if self.initial_balance > 0 else 0.0,
                "win_rate": round(win_rate, 1),
                "profit_factor": round(profit_factor, 2),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2)
            }

            logger.logger.info(
                f"📈 Bot Status | Equity: ${status['equity']:.2f} | "
                f"Free Balance: ${status['balance']:.2f} | "
                f"In Positions: ${status['positions_value']:.2f} | "
                f"Positions: {status['positions']} | Trades: {status['trades']} | "
                f"Win Rate: {status['win_rate']:.1f}% | "
                f"Profit Factor: {status['profit_factor']:.2f}"
            )

            logger.logger.info(
                f"💰 P&L | Realized: ${status['realized_pnl']:.2f} | "
                f"Unrealized: ${status['unrealized_pnl']:.2f} | "
                f"Total: ${status['total_pnl']:.2f} | "
                f"Return: {status['return_percent']:.2f}% | "
                f"Avg Win: ${status['avg_win']:.2f} | Avg Loss: ${status['avg_loss']:.2f}"
            )

            # ИСПРАВЛЕННЫЕ метрики фильтров v3.0
            logger.logger.info(
                f"📊 Analysis Metrics | "
                f"Total Analyses: {self.performance_metrics['total_analyses']} | "
                f"Signals (non-HOLD): {self.performance_metrics['total_signals']} | "
                f"HOLD signals: {self.performance_metrics['hold_signals']} | "
                f"Executed: {self.performance_metrics['executed_signals']}"
            )

            if self.performance_metrics['total_signals'] > 0:
                execution_rate = (self.performance_metrics['executed_signals'] /
                                self.performance_metrics['total_signals'] * 100)

                logger.logger.info(
                    f"🎯 Filter Statistics | "
                    f"Execution Rate: {execution_rate:.1f}% | "
                    f"Rejected by: Confidence={self.performance_metrics['skipped_low_confidence']}, "
                    f"Volatility={self.performance_metrics['skipped_low_volatility']}, "
                    f"Time of Day={self.performance_metrics['skipped_time_of_day']}, "
                    f"Cooldown={self.performance_metrics['skipped_cooldown']}, "
                    f"Correlation={self.performance_metrics['skipped_correlation']}, "
                    f"Time Limit={self.performance_metrics['skipped_time_limit']}, "
                    f"Volume={self.performance_metrics['skipped_low_volume']}, "
                    f"Position Exists={self.performance_metrics['skipped_position_exists']}"
                )

            # Позиции
            for symbol, position in self.positions.items():
                if hasattr(position, 'unrealized_pnl') and hasattr(position, 'current_price'):
                    base = position.entry_price * position.quantity
                    pnl_percent = (position.unrealized_pnl / base) * 100 if base > 0 else 0.0
                    pnl_emoji = "📈" if position.unrealized_pnl > 0 else "📉"
                    partially_closed = " (50% closed)" if self._partially_closed.get(symbol, False) else ""
                    logger.logger.info(
                        f"{pnl_emoji} Position: {symbol} | Side: {position.side} | "
                        f"Entry: ${position.entry_price:.2f} | Current: ${position.current_price:.2f} | "
                        f"PnL: ${position.unrealized_pnl:.2f} ({pnl_percent:.2f}%){partially_closed}"
                    )
        except Exception as e:
            logger.logger.error(f"Error logging status: {e}")

    async def stop(self):
        """Остановка бота с финальным отчётом v3.0"""
        logger.logger.info("Stopping Advanced Paper Trading Bot v3.0")
        self.running = False

        try:
            # Закрываем все позиции
            for symbol in list(self.positions.keys()):
                current_price = self.indicators.get(symbol, {}).get('price', self.positions[symbol].entry_price)
                await self._close_position(symbol, current_price, "bot_stopped")

            # Отключаемся от Binance
            if self.binance_client:
                await self.binance_client.close_connection()

            # Останавливаем WebSocket
            await ws_client.stop()

            # Финальный отчёт
            all_trades = [t for t in self.trade_history if t.get('type') != 'partial_close']
            final_pnl = sum(t.get('pnl', 0.0) for t in all_trades)
            final_balance = self.current_balance
            final_return = ((final_balance - self.initial_balance) / self.initial_balance) * 100 if self.initial_balance else 0.0

            wins = [t for t in all_trades if t['pnl'] > 0]
            losses = [t for t in all_trades if t['pnl'] < 0]

            # Расчёт дополнительных метрик
            if all_trades:
                win_rate = len(wins) / len(all_trades) * 100
                avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
                avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
                profit_factor = abs(sum(t['pnl'] for t in wins) / sum(t['pnl'] for t in losses)) if losses and sum(t['pnl'] for t in losses) != 0 else 0

                logger.logger.info("=" * 80)
                logger.logger.info("📊 FINAL TRADING REPORT v3.0")
                logger.logger.info("=" * 80)
                logger.logger.info(
                    f"💰 Financial Results:\n"
                    f"   Initial Balance: ${self.initial_balance:.2f}\n"
                    f"   Final Balance: ${final_balance:.2f}\n"
                    f"   Total PnL: ${final_pnl:.2f}\n"
                    f"   Return: {final_return:.2f}%"
                )
                logger.logger.info(
                    f"📈 Trading Statistics:\n"
                    f"   Total Trades: {len(all_trades)}\n"
                    f"   Wins: {len(wins)} | Losses: {len(losses)}\n"
                    f"   Win Rate: {win_rate:.1f}%\n"
                    f"   Profit Factor: {profit_factor:.2f}\n"
                    f"   Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}"
                )

                # Топ прибыльные и убыточные сделки
                if wins:
                    best_trade = max(wins, key=lambda x: x['pnl'])
                    logger.logger.info(
                        f"   Best Trade: {best_trade['symbol']} | "
                        f"PnL: ${best_trade['pnl']:.2f}"
                    )
                if losses:
                    worst_trade = min(losses, key=lambda x: x['pnl'])
                    logger.logger.info(
                        f"   Worst Trade: {worst_trade['symbol']} | "
                        f"PnL: ${worst_trade['pnl']:.2f}"
                    )

                # ИСПРАВЛЕННЫЙ анализ сигналов v3.0
                logger.logger.info(
                    f"🎯 Signal Analysis v3.0:\n"
                    f"   Total Market Analyses: {self.performance_metrics['total_analyses']}\n"
                    f"   Trading Signals Generated (non-HOLD): {self.performance_metrics['total_signals']}\n"
                    f"   HOLD Signals (no action): {self.performance_metrics['hold_signals']}\n"
                    f"   Signals Executed: {self.performance_metrics['executed_signals']}"
                )

                if self.performance_metrics['total_signals'] > 0:
                    exec_rate = self.performance_metrics['executed_signals'] / self.performance_metrics['total_signals'] * 100
                    logger.logger.info(
                        f"   Execution Rate: {exec_rate:.1f}%\n"
                        f"   Filter Rejections (from real signals only):\n"
                        f"      - Low Confidence: {self.performance_metrics['skipped_low_confidence']}\n"
                        f"      - Low Volatility: {self.performance_metrics['skipped_low_volatility']}\n"
                        f"      - Time of Day: {self.performance_metrics['skipped_time_of_day']}\n"
                        f"      - Cooldown Period: {self.performance_metrics['skipped_cooldown']}\n"
                        f"      - Correlation Conflict: {self.performance_metrics['skipped_correlation']}\n"
                        f"      - Time Between Trades: {self.performance_metrics['skipped_time_limit']}\n"
                        f"      - Low Volume: {self.performance_metrics['skipped_low_volume']}\n"
                        f"      - Position Already Exists: {self.performance_metrics['skipped_position_exists']}"
                    )

                    # Анализ причин отклонения
                    total_rejected = (
                        self.performance_metrics['skipped_low_confidence'] +
                        self.performance_metrics['skipped_low_volatility'] +
                        self.performance_metrics['skipped_time_of_day'] +
                        self.performance_metrics['skipped_cooldown'] +
                        self.performance_metrics['skipped_correlation'] +
                        self.performance_metrics['skipped_time_limit'] +
                        self.performance_metrics['skipped_low_volume'] +
                        self.performance_metrics['skipped_position_exists']
                    )

                    if total_rejected > 0:
                        logger.logger.info(
                            f"\n   Top Rejection Reasons:"
                        )
                        rejection_reasons = [
                            ('Low Volatility', self.performance_metrics['skipped_low_volatility']),
                            ('Time Between Trades', self.performance_metrics['skipped_time_limit']),
                            ('Correlation', self.performance_metrics['skipped_correlation']),
                            ('Low Confidence', self.performance_metrics['skipped_low_confidence']),
                            ('Low Volume', self.performance_metrics['skipped_low_volume']),
                            ('Time of Day', self.performance_metrics['skipped_time_of_day']),
                            ('Cooldown', self.performance_metrics['skipped_cooldown']),
                            ('Position Exists', self.performance_metrics['skipped_position_exists']),
                        ]
                        rejection_reasons.sort(key=lambda x: x[1], reverse=True)

                        for reason, count in rejection_reasons[:3]:
                            if count > 0:
                                pct = count / total_rejected * 100
                                logger.logger.info(f"      {reason}: {count} ({pct:.1f}%)")

                logger.logger.info("=" * 80)
            else:
                logger.logger.info("No trades executed during this session")
                logger.logger.info(
                    f"Analysis summary: {self.performance_metrics['total_analyses']} analyses, "
                    f"{self.performance_metrics['total_signals']} signals generated, "
                    f"{self.performance_metrics['hold_signals']} HOLD signals"
                )

        except Exception as e:
            logger.logger.error(f"Error stopping bot: {e}")