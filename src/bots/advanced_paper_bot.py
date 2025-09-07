"""
Продвинутый Paper Trading Bot с реальными данными Binance.

ВЕРСИЯ 9.0 - FINAL PRODUCTION VERSION:
- Исправлена ошибка с volume_ratio
- Оптимизированы параметры для реальной торговли
- Улучшенная обработка ошибок
- Адаптивные фильтры для текущих рыночных условий

ПРЕДУПРЕЖДЕНИЕ: Только для Paper Trading! Не используйте с реальными деньгами!
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import traceback

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
    """Продвинутый бот с оптимизированными параметрами для максимальной эффективности"""

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

        # Торговые символы
        self.trading_symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT',
            'SOLUSDT', 'ADAUSDT', 'XRPUSDT',
        ]

        # Торговые данные
        self.positions: Dict[str, Position] = {}
        self.max_positions = 6  # Увеличено для большей активности
        self.max_positions_per_symbol = 1
        self.trade_history: List[Dict[str, Any]] = []
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.indicators: Dict[str, Dict[str, Any]] = {}

        # Данные старшего таймфрейма
        self.market_data_h1: Dict[str, pd.DataFrame] = {}
        self.trend_direction: Dict[str, str] = {}

        # Tracking данные
        self._last_closed_time: Dict[str, datetime] = {}
        self._loss_streak: Dict[str, int] = {}
        self._cooldown_until: Dict[str, datetime] = {}
        self._last_trade_time: Dict[str, datetime] = {}
        self._partially_closed: Dict[str, bool] = {}

        # Кэш сигналов v9.0 - УМЕНЬШЕННАЯ ДЛИТЕЛЬНОСТЬ
        self._recent_signals: Dict[str, Tuple[str, datetime, float]] = {}
        self._signal_cache_duration = timedelta(minutes=1)  # Короткий кэш для активной торговли

        # История производительности
        self._position_performance: Dict[str, deque] = {}
        self._max_performance_history = 10

        # Валидация данных
        self._last_valid_data: Dict[str, Dict] = {}
        self._data_validation_enabled = True

        # Технические поля
        self.last_data_update: Dict[str, datetime] = {}
        self.data_update_interval = 30
        self.running = False
        self.connected = False

        # Настройки индикаторов
        self.rsi_length = 14
        self.poll_interval = 30

        # === АГРЕССИВНЫЕ ПАРАМЕТРЫ v9.0 ДЛЯ МАКСИМАЛЬНОЙ АКТИВНОСТИ ===
        self.min_confidence = 0.45  # Снижен для большего количества сделок
        self.min_signals_required = 1.3  # Снижен для быстрого входа
        self.min_volatility_percent = 0.06  # Снижен для принятия большего количества сигналов
        self.min_time_between_trades = timedelta(seconds=30)  # Очень быстрый re-entry
        self.base_position_size_percent = 0.12  # 12% на позицию для заметных результатов
        self.position_timeout_hours = 3  # Быстрое закрытие флэтовых позиций
        self.position_timeout_threshold = 0.008  # 0.8% для тайм-аута
        self.min_volume_ratio = 0.8  # Снижен для большего количества сигналов

        # Фильтры - СМЯГЧЕНЫ ДЛЯ АКТИВНОСТИ
        self.volatility_filter_enabled = True
        self.momentum_filter_enabled = False  # Отключен для большей активности
        self.volume_filter_enabled = False  # Отключен для большей активности

        # === АГРЕССИВНЫЙ РИСК-МЕНЕДЖМЕНТ v9.0 ===
        self.min_stop_loss_percent = 0.02  # 2% минимум - тайтовый стоп
        self.max_stop_loss_percent = 0.04  # 4% максимум - контроль риска
        self.min_take_profit_percent = 0.03  # 3% минимум - быстрые профиты
        self.risk_reward_ratio = 1.5  # Стандартное RR
        self.noise_buffer_percent = 0.002  # 0.2% буфер
        self.adaptive_sl_threshold = 0.10  # Порог волатильности
        self.partial_close_level = 1.5  # Быстрое частичное закрытие
        self.trailing_stop_activation = 1.0  # Быстрая активация трейлинга
        self.trailing_stop_distance = 0.5  # Тайтовый трейлинг

        # Ротация позиций - АГРЕССИВНАЯ
        self.enable_position_rotation = True
        self.rotation_min_confidence_diff = 0.15  # Снижена разница для частой ротации
        self.rotation_min_loss_percent = 0.01  # 1% убытка достаточно для ротации

        # Адаптивные пороги волатильности - СНИЖЕНЫ
        self.volatility_thresholds = {
            'BTCUSDT': 0.06,  # Снижено
            'ETHUSDT': 0.06,  # Снижено
            'BNBUSDT': 0.06,  # Снижено
            'SOLUSDT': 0.08,  # Снижено
            'ADAUSDT': 0.08,  # Снижено
            'XRPUSDT': 0.08,  # Снижено
            'default': 0.08   # Снижено
        }

        # Коррелированные пары
        self.correlated_pairs = [
            {'BTCUSDT', 'ETHUSDT'},
        ]

        # Метрики производительности
        self.performance_metrics = {
            'total_analyses': 0,
            'total_signals': 0,
            'hold_signals': 0,
            'executed_signals': 0,
            'skipped_low_confidence': 0,
            'skipped_low_volatility': 0,
            'skipped_time_of_day': 0,
            'skipped_cooldown': 0,
            'skipped_correlation': 0,
            'skipped_time_limit': 0,
            'skipped_low_volume': 0,
            'skipped_position_exists': 0,
            'skipped_momentum': 0,
            'skipped_invalid_sl': 0,
            'skipped_duplicate_signal': 0,
            'skipped_inconsistent_signal': 0,
            'position_rotations': 0,
            'timeout_closes': 0,
            'stop_loss_hits': 0,
            'take_profit_hits': 0,
            'partial_closes': 0,
            'trailing_stop_hits': 0,
            'data_errors': 0,  # Новая метрика v9.0
        }

    async def initialize(self):
        """Инициализация бота"""
        try:
            logger.logger.info("=" * 80)
            logger.logger.info("🚀 Initializing Advanced Paper Trading Bot v9.0 FINAL")
            logger.logger.info("⚠️  AGGRESSIVE SETTINGS - HIGH RISK/HIGH REWARD MODE")
            logger.logger.info("=" * 80)
            logger.logger.info(
                f"💰 Capital: ${self.initial_balance:,.2f} | "
                f"Position Size: {self.base_position_size_percent*100:.0f}% | "
                f"Max Positions: {self.max_positions}"
            )
            logger.logger.info(
                f"🎯 Risk Parameters: "
                f"SL: {self.min_stop_loss_percent*100:.1f}%-{self.max_stop_loss_percent*100:.1f}% | "
                f"TP: {self.min_take_profit_percent*100:.1f}%+ | "
                f"RR: {self.risk_reward_ratio:.1f}"
            )
            logger.logger.info(
                f"⚡ Trading Parameters: "
                f"Min Confidence: {self.min_confidence*100:.0f}% | "
                f"Min Signals: {self.min_signals_required:.1f} | "
                f"Signal Cache: {self._signal_cache_duration.total_seconds():.0f}s"
            )

            await self._connect_binance()
            await self._load_historical_data()
            await self._initialize_ml()
            await self._start_data_streams()

            logger.logger.info("=" * 80)
            logger.logger.info("✅ Bot initialized successfully - Ready to trade!")
            logger.logger.info("=" * 80)
        except Exception as e:
            logger.logger.error(f"Failed to initialize bot: {e}")
            raise

    def get_adaptive_volatility_threshold(self, symbol: str) -> float:
        """Получить адаптивный порог волатильности для символа"""
        return self.volatility_thresholds.get(symbol, self.volatility_thresholds['default'])

    def _validate_signal_consistency(self, symbol: str, reasons: List[str],
                                    rsi: float, bb_position: float) -> bool:
        """Валидация логической согласованности сигнала"""
        reasons_str = ' '.join(reasons).lower()

        # Более мягкая валидация для активной торговли
        if "oversold" in reasons_str and bb_position > 0.85:  # Только экстремальные случаи
            logger.logger.debug(f"Signal inconsistency for {symbol}: RSI vs BB")
            self.performance_metrics['skipped_inconsistent_signal'] += 1
            return False

        if "overbought" in reasons_str and bb_position < 0.15:  # Только экстремальные случаи
            logger.logger.debug(f"Signal inconsistency for {symbol}: RSI vs BB")
            self.performance_metrics['skipped_inconsistent_signal'] += 1
            return False

        return True

    def _validate_market_data(self, symbol: str, df: pd.DataFrame) -> bool:
        """Валидация рыночных данных"""
        if df is None or df.empty:
            return False

        # Проверка на нулевые значения
        if df['close'].iloc[-1] == 0:
            logger.logger.warning(f"Zero price detected for {symbol}")
            return False

        # Проверка на застойные данные
        if len(df) > 5:
            recent_closes = df['close'].iloc[-5:].values
            if np.std(recent_closes) == 0:
                logger.logger.warning(f"Stale data detected for {symbol}")
                return False

        return True

    def _calculate_dynamic_position_size(self, symbol: str, confidence: float,
                                        volatility: float) -> float:
        """Динамический расчет размера позиции"""
        base_size = self.base_position_size_percent

        # Агрессивная корректировка по уверенности
        if confidence > 0.80:
            confidence_mult = 1.5
        elif confidence > 0.65:
            confidence_mult = 1.3
        elif confidence > 0.50:
            confidence_mult = 1.1
        else:
            confidence_mult = 0.9

        # Корректировка по волатильности
        if volatility < 0.10:
            vol_mult = 1.3  # Больше размер при низкой волатильности
        elif volatility > 0.40:
            vol_mult = 0.8  # Меньше при высокой
        else:
            vol_mult = 1.0

        # Корректировка по истории
        perf_mult = 1.0
        if symbol in self._position_performance and len(self._position_performance[symbol]) >= 2:
            recent_pnls = list(self._position_performance[symbol])[-2:]
            avg_pnl = sum(recent_pnls) / len(recent_pnls)

            if avg_pnl > 0.01:  # Прибыль > 1%
                perf_mult = 1.2
            elif avg_pnl < -0.01:  # Убыток > 1%
                perf_mult = 0.8

        # Финальный размер
        final_size = base_size * confidence_mult * vol_mult * perf_mult

        # Ограничения
        max_size = 0.25  # Максимум 25% для агрессивной торговли
        min_size = 0.05  # Минимум 5%

        return max(min_size, min(final_size, max_size))

    async def _should_replace_position(self, new_signal: Dict, symbol: str) -> Optional[str]:
        """Определение позиции для замены - агрессивная ротация"""
        if not self.enable_position_rotation or len(self.positions) < self.max_positions:
            return None

        worst_position = None
        worst_score = float('inf')

        for pos_symbol, position in self.positions.items():
            # Агрессивно заменяем даже небольшие убытки
            position_value = position.quantity * position.entry_price
            pnl_percent = position.unrealized_pnl / position_value if position_value > 0 else 0
            position_age = (datetime.utcnow() - position.entry_time).total_seconds() / 3600

            # Score с акцентом на быструю ротацию
            score = pnl_percent - (position_age * 0.02)  # Больший штраф за время

            if score < worst_score and pnl_percent < 0:  # Любой убыток
                worst_score = score
                worst_position = pos_symbol

        # Заменяем при меньшей разнице в уверенности
        if worst_position and new_signal['confidence'] > self.min_confidence + 0.1:
            logger.logger.info(f"🔄 Rotating: {worst_position} → {symbol}")
            return worst_position

        return None

    def _is_duplicate_signal(self, symbol: str, action: str, confidence: float) -> bool:
        """Проверка дублирующихся сигналов - короткий кэш"""
        if symbol in self._recent_signals:
            cached_action, cached_time, cached_confidence = self._recent_signals[symbol]
            time_diff = datetime.utcnow() - cached_time

            # Короткий кэш для активной торговли
            if (cached_action == action and
                time_diff < self._signal_cache_duration and
                abs(confidence - cached_confidence) < 0.20):  # Больший допуск
                return True

        return False

    async def _connect_binance(self):
        """Подключение к Binance"""
        try:
            self.binance_client = await AsyncClient.create(
                api_key=settings.api.BINANCE_API_KEY.get_secret_value(),
                api_secret=settings.api.BINANCE_API_SECRET.get_secret_value(),
                testnet=settings.api.TESTNET
            )

            await self.binance_client.ping()
            await self.data_collector.initialize(self.binance_client)
            self.data_collector.force_refresh = True
            self.data_collector.use_cache = False

            self.connected = True
            logger.logger.info(f"✅ Connected to Binance {'Testnet' if settings.api.TESTNET else 'Live'}")
        except Exception as e:
            logger.logger.error(f"❌ Binance connection failed: {e}")
            self.connected = False

    async def _load_historical_data(self):
        """Загрузка исторических данных"""
        logger.logger.info("📊 Loading historical data...")

        for symbol in self.trading_symbols:
            try:
                df = await self.data_collector.fetch_historical_data(
                    symbol=symbol,
                    interval=settings.trading.PRIMARY_TIMEFRAME,
                    days_back=2,
                    limit=200,
                    force_refresh=True
                )

                df_h1 = await self.data_collector.fetch_historical_data(
                    symbol=symbol,
                    interval='1h',
                    days_back=3,
                    limit=72,
                    force_refresh=True
                )

                if self._validate_market_data(symbol, df):
                    self.market_data[symbol] = df
                    self.last_data_update[symbol] = datetime.utcnow()

                    if df_h1 is not None and not df_h1.empty:
                        self.market_data_h1[symbol] = df_h1
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
                        'price': float(df['close'].iloc[-1])
                    }

                    self._last_valid_data[symbol] = self.indicators[symbol].copy()

                    if symbol not in self._position_performance:
                        self._position_performance[symbol] = deque(maxlen=self._max_performance_history)

                    atr = self.indicators[symbol]['atr']
                    price = self.indicators[symbol]['price']
                    atr_percent = (atr / price * 100) if price > 0 else 0

                    logger.logger.info(
                        f"📈 {symbol}: ${price:,.2f} | RSI: {self.indicators[symbol]['rsi']:.1f} | "
                        f"BB: {self.indicators[symbol]['bb_position']:.2f} | ATR: {atr_percent:.2f}%"
                    )

            except Exception as e:
                logger.logger.error(f"Failed to load {symbol}: {e}")
                self.performance_metrics['data_errors'] += 1

        logger.logger.info(f"✅ Data loaded for {len(self.market_data)} symbols")

    async def _initialize_ml(self):
        """Инициализация ML моделей"""
        try:
            await ml_engine.load_models()
            logger.logger.info("🤖 ML models loaded")
        except Exception:
            logger.logger.info("📊 No saved ML models, using technical analysis only")

    async def _start_data_streams(self):
        """Запуск потоков данных"""
        logger.logger.warning("📡 Using REST API polling")
        self.running = True
        asyncio.create_task(self._poll_market_data())

    async def _get_current_price(self, symbol: str, df_fallback: Optional[pd.DataFrame] = None) -> float:
        """Получить текущую цену"""
        try:
            if self.binance_client:
                ticker = await self.binance_client.get_ticker(symbol=symbol)
                price = float(ticker['lastPrice'])
                if price > 0:
                    return price
        except:
            pass

        if df_fallback is not None and not df_fallback.empty:
            return float(df_fallback['close'].iloc[-1])

        if symbol in self._last_valid_data:
            return self._last_valid_data[symbol].get('price', 0)

        raise RuntimeError(f"No price for {symbol}")

    async def _poll_market_data(self):
        """Обновление данных через REST API"""
        logger.logger.info(f"📡 Starting market data polling (interval: {self.poll_interval}s)")

        while self.running:
            try:
                for symbol in self.trading_symbols:
                    if not self.running:
                        break

                    try:
                        now = datetime.utcnow()
                        last_update = self.last_data_update.get(symbol, datetime.min)

                        if (now - last_update).total_seconds() < (self.poll_interval - 5):
                            continue

                        df = await self.data_collector.fetch_historical_data(
                            symbol=symbol,
                            interval=settings.trading.PRIMARY_TIMEFRAME,
                            days_back=1,
                            limit=100,
                            force_refresh=True
                        )

                        if symbol not in self.market_data_h1 or \
                           (now - self.last_data_update.get(f"{symbol}_h1", datetime.min)) > timedelta(minutes=15):
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

                                ema50 = ta.ema(df_h1['close'], length=50)
                                if ema50 is not None and not ema50.empty:
                                    self.trend_direction[symbol] = 'UP' if df_h1['close'].iloc[-1] > ema50.iloc[-1] else 'DOWN'

                        if not self._validate_market_data(symbol, df):
                            continue

                        current_price = await self._get_current_price(symbol, df)
                        df_live = df.copy()
                        if 'close' in df_live.columns:
                            df_live.iloc[-1, df_live.columns.get_loc('close')] = float(current_price)

                        rsi_series = ta.rsi(df_live['close'], length=self.rsi_length)
                        rsi_live = float(rsi_series.iloc[-1]) if rsi_series is not None and not rsi_series.empty else 50.0

                        self.market_data[symbol] = df
                        self.last_data_update[symbol] = now
                        self.indicators[symbol] = {
                            'rsi': rsi_live,
                            'price': float(current_price),
                            'macd': float(df['macd'].iloc[-1]) if 'macd' in df.columns else 0.0,
                            'bb_position': float(df['bb_percent'].iloc[-1]) if 'bb_percent' in df.columns else 0.5,
                            'volume_ratio': float(df['volume_ratio'].iloc[-1]) if 'volume_ratio' in df.columns else 1.0,
                            'atr': float(df['atr'].iloc[-1]) if 'atr' in df.columns else 0.0,
                        }

                        self._last_valid_data[symbol] = self.indicators[symbol].copy()

                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.logger.debug(f"Poll error for {symbol}: {e}")
                        self.performance_metrics['data_errors'] += 1

                await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                logger.logger.info("Market data polling cancelled")
                break
            except Exception as e:
                logger.logger.error(f"Polling loop error: {e}")
                await asyncio.sleep(10)

    async def _handle_kline_update(self, data: Dict):
        """Обработка обновлений свечей"""
        try:
            kline = data.get('k', {})
            if kline.get('x'):
                symbol = kline.get('s')
                await self._update_market_data(symbol)
        except Exception as e:
            logger.logger.error(f"Kline update error: {e}")

    async def _handle_ticker_update(self, data: Dict):
        """Обработка обновлений тикера"""
        try:
            symbol = data.get('s')
            price = float(data.get('c', 0))
            if symbol and price > 0 and symbol in self.positions:
                await self._update_position_pnl(symbol, price)
        except Exception as e:
            logger.logger.error(f"Ticker update error: {e}")

    async def _update_market_data(self, symbol: str):
        """Обновление рыночных данных"""
        try:
            df = await self.data_collector.fetch_historical_data(
                symbol=symbol,
                interval=settings.trading.PRIMARY_TIMEFRAME,
                days_back=1,
                limit=100,
                force_refresh=True
            )

            if self._validate_market_data(symbol, df):
                self.market_data[symbol] = df
                self.last_data_update[symbol] = datetime.utcnow()

                self.indicators[symbol] = {
                    'rsi': float(df['rsi'].iloc[-1]) if 'rsi' in df.columns else 50.0,
                    'macd': float(df['macd'].iloc[-1]) if 'macd' in df.columns else 0.0,
                    'bb_position': float(df['bb_percent'].iloc[-1]) if 'bb_percent' in df.columns else 0.5,
                    'volume_ratio': float(df['volume_ratio'].iloc[-1]) if 'volume_ratio' in df.columns else 1.0,
                    'atr': float(df['atr'].iloc[-1]) if 'atr' in df.columns else 0.0,
                    'price': float(df['close'].iloc[-1])
                }

                self._last_valid_data[symbol] = self.indicators[symbol].copy()

        except Exception as e:
            logger.logger.error(f"Failed to update data for {symbol}: {e}")

    async def run(self):
        """Основной торговый цикл"""
        self.running = True
        logger.logger.info("🚀 Starting trading loop v9.0")

        analysis_counter = 0

        try:
            while self.running:
                await self._analyze_and_trade()
                await self._update_positions()
                await self._check_risk_limits()

                if analysis_counter % 4 == 0:
                    await self._log_status()
                    analysis_counter = 0

                analysis_counter += 1
                await asyncio.sleep(30)

        except asyncio.CancelledError:
            logger.logger.info("Trading loop cancelled")
        except Exception as e:
            logger.logger.error(f"Trading loop error: {e}\n{traceback.format_exc()}")
            raise
        finally:
            logger.logger.info("Trading loop stopped")

    async def _analyze_and_trade(self):
        """Анализ рынка и торговля - ИСПРАВЛЕНА ОШИБКА С volume_ratio"""
        for symbol in self.trading_symbols:
            try:
                self.performance_metrics['total_analyses'] += 1

                if symbol not in self.market_data:
                    continue

                df = self.market_data[symbol]
                if not self._validate_market_data(symbol, df):
                    continue

                indicators = self.indicators.get(symbol, {})
                current_price = indicators.get('price', 0)

                if current_price == 0:
                    continue

                # Генерация сигнала
                signal = await self._generate_trading_signal(symbol, df, indicators)

                if signal['action'] == 'HOLD':
                    self.performance_metrics['hold_signals'] += 1
                    continue

                self.performance_metrics['total_signals'] += 1

                # Валидация сигнала
                if not self._validate_signal_consistency(
                    symbol, signal['reasons'],
                    indicators.get('rsi', 50),
                    indicators.get('bb_position', 0.5)
                ):
                    continue

                # Проверка дубликатов
                if self._is_duplicate_signal(symbol, signal['action'], signal['confidence']):
                    self.performance_metrics['skipped_duplicate_signal'] += 1
                    continue

                # Кэшируем сигнал
                self._recent_signals[symbol] = (signal['action'], datetime.utcnow(), signal['confidence'])

                logger.logger.info(
                    f"📊 {symbol} | {signal['action']} | "
                    f"Conf: {signal['confidence']:.2f} | {', '.join(signal['reasons'][:2])}"
                )

                # Проверка SL/TP
                if signal['action'] == 'BUY':
                    if signal['stop_loss'] >= current_price:
                        self.performance_metrics['skipped_invalid_sl'] += 1
                        continue
                elif signal['action'] == 'SELL':
                    if signal['stop_loss'] <= current_price:
                        self.performance_metrics['skipped_invalid_sl'] += 1
                        continue

                # Проверка/ротация позиций
                position_to_close = None
                if symbol in self.positions:
                    self.performance_metrics['skipped_position_exists'] += 1
                    continue
                elif len(self.positions) >= self.max_positions:
                    position_to_close = await self._should_replace_position(signal, symbol)
                    if not position_to_close:
                        self.performance_metrics['skipped_position_exists'] += 1
                        continue

                # === МИНИМАЛЬНЫЕ ФИЛЬТРЫ ДЛЯ МАКСИМАЛЬНОЙ АКТИВНОСТИ ===

                # Уверенность
                if signal['confidence'] < self.min_confidence:
                    self.performance_metrics['skipped_low_confidence'] += 1
                    continue

                # НЕТ фильтра времени суток для 24/7 торговли

                # Кулдаун - очень короткий
                if symbol in self._cooldown_until and datetime.utcnow() < self._cooldown_until[symbol]:
                    self.performance_metrics['skipped_cooldown'] += 1
                    continue

                # Время между сделками - минимальное
                if symbol in self._last_closed_time:
                    time_since_close = datetime.utcnow() - self._last_closed_time[symbol]
                    if time_since_close < self.min_time_between_trades:
                        self.performance_metrics['skipped_time_limit'] += 1
                        continue

                # Получаем volume_ratio ПЕРЕД использованием
                volume_ratio = indicators.get('volume_ratio', 1.0)
                atr = indicators.get('atr', 0.0)
                atr_percent = (atr / current_price * 100) if current_price > 0 else 0

                # Волатильность - мягкий фильтр
                if self.volatility_filter_enabled:
                    min_vol = self.volatility_thresholds.get(symbol, self.volatility_thresholds['default'])
                    if atr_percent < min_vol:
                        self.performance_metrics['skipped_low_volatility'] += 1
                        continue

                # Объем - ОТКЛЮЧЕН
                if self.volume_filter_enabled and volume_ratio < self.min_volume_ratio:
                    self.performance_metrics['skipped_low_volume'] += 1
                    continue

                # Корреляция - только для противоположных позиций
                if await self._check_correlation_conflict(symbol, signal['action']):
                    self.performance_metrics['skipped_correlation'] += 1
                    continue

                # Закрываем позицию для ротации
                if position_to_close:
                    close_price = self.indicators.get(position_to_close, {}).get('price')
                    if close_price:
                        await self._close_position(position_to_close, close_price, "rotation")
                        self.performance_metrics['position_rotations'] += 1

                # Исполняем сделку
                logger.logger.info(
                    f"✅ EXECUTING {signal['action']} {symbol} | "
                    f"Vol: {atr_percent:.2f}% | Volume: {volume_ratio:.1f}x"
                )

                await self._execute_trade(symbol, signal, current_price)
                self.performance_metrics['executed_signals'] += 1

            except Exception as e:
                logger.logger.error(f"Analysis failed for {symbol}: {e}")
                self.performance_metrics['data_errors'] += 1

    async def _check_correlation_conflict(self, symbol: str, direction: str) -> bool:
        """Проверка корреляции - только противоположные направления"""
        for group in self.correlated_pairs:
            if symbol in group:
                for other_symbol in group:
                    if other_symbol != symbol and other_symbol in self.positions:
                        other_position = self.positions[other_symbol]
                        # Блокируем только противоположные позиции
                        if (direction == 'BUY' and other_position.side == 'SELL') or \
                           (direction == 'SELL' and other_position.side == 'BUY'):
                            return True
        return False

    async def _generate_trading_signal(self, symbol: str, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Генерация сигнала - АГРЕССИВНЫЕ НАСТРОЙКИ"""
        signal = {
            'action': 'HOLD',
            'confidence': 0.0,
            'reasons': [],
            'stop_loss': None,
            'take_profit': None
        }

        buy_signals = 0.0
        sell_signals = 0.0

        close_ = float(df['close'].iloc[-1])

        # 1) RSI - Агрессивные пороги
        rsi = indicators.get('rsi', 50.0)
        if rsi < 25:
            buy_signals += 2.0
            signal['reasons'].append(f"RSI extreme oversold ({rsi:.1f})")
        elif rsi < 35:
            buy_signals += 1.3
            signal['reasons'].append(f"RSI oversold ({rsi:.1f})")
        elif rsi < 40:
            buy_signals += 0.7
            signal['reasons'].append(f"RSI low ({rsi:.1f})")
        elif rsi > 75:
            sell_signals += 2.0
            signal['reasons'].append(f"RSI extreme overbought ({rsi:.1f})")
        elif rsi > 65:
            sell_signals += 1.3
            signal['reasons'].append(f"RSI overbought ({rsi:.1f})")
        elif rsi > 60:
            sell_signals += 0.7
            signal['reasons'].append(f"RSI high ({rsi:.1f})")

        # 2) MACD
        macd = indicators.get('macd', 0.0)
        if 'macd_signal' in df.columns and len(df) > 0:
            macd_signal_val = float(df['macd_signal'].iloc[-1])
            macd_diff = macd - macd_signal_val

            if macd_diff > 0:
                buy_signals += 0.8
                signal['reasons'].append("MACD bullish")
            elif macd_diff < 0:
                sell_signals += 0.8
                signal['reasons'].append("MACD bearish")

        # 3) BOLLINGER BANDS - Правильная логика
        bb_position = indicators.get('bb_position', 0.5)

        if bb_position < 0.15:
            buy_signals += 1.8
            signal['reasons'].append(f"At lower BB ({bb_position:.2f})")
        elif bb_position < 0.25:
            buy_signals += 1.2
            signal['reasons'].append(f"Near lower BB ({bb_position:.2f})")
        elif bb_position < 0.35:
            buy_signals += 0.6
            signal['reasons'].append(f"Below middle BB ({bb_position:.2f})")
        elif bb_position > 0.85:
            sell_signals += 1.8
            signal['reasons'].append(f"At upper BB ({bb_position:.2f})")
        elif bb_position > 0.75:
            sell_signals += 1.2
            signal['reasons'].append(f"Near upper BB ({bb_position:.2f})")
        elif bb_position > 0.65:
            sell_signals += 0.6
            signal['reasons'].append(f"Above middle BB ({bb_position:.2f})")

        # 4) Объем (если не отключен)
        if not self.volume_filter_enabled or indicators.get('volume_ratio', 1.0) > 1.0:
            volume_ratio = indicators.get('volume_ratio', 1.0)
            if volume_ratio > 1.5:
                weight = 0.5
                signal['reasons'].append(f"Volume {volume_ratio:.1f}x")
                if buy_signals > sell_signals:
                    buy_signals += weight
                elif sell_signals > buy_signals:
                    sell_signals += weight

        # 5) Тренд
        trend = self.trend_direction.get(symbol, 'NEUTRAL')
        if trend == 'UP' and buy_signals > sell_signals:
            buy_signals += 0.5
            signal['reasons'].append("Uptrend")
        elif trend == 'DOWN' and sell_signals > buy_signals:
            sell_signals += 0.5
            signal['reasons'].append("Downtrend")

        # === АГРЕССИВНЫЙ РИСК-МЕНЕДЖМЕНТ ===
        atr = float(indicators.get('atr', 0))
        if atr == 0 or pd.isna(atr):
            atr = close_ * 0.008

        min_atr = close_ * 0.004
        max_atr = close_ * 0.04
        atr = max(min_atr, min(atr, max_atr))

        atr_percent = (atr / close_) * 100 if close_ > 0 else 0

        # Адаптивные множители
        if atr_percent < 0.08:
            volatility_multiplier_sl = 3.0
            volatility_multiplier_tp = 4.5
        else:
            volatility_multiplier_sl = 2.5
            volatility_multiplier_tp = 3.8

        noise_buffer = close_ * self.noise_buffer_percent

        # Расчет дистанций
        sl_distance = max(
            volatility_multiplier_sl * atr + noise_buffer,
            close_ * self.min_stop_loss_percent
        )
        sl_distance = min(sl_distance, close_ * self.max_stop_loss_percent)

        tp_distance = max(
            sl_distance * self.risk_reward_ratio,
            volatility_multiplier_tp * atr,
            close_ * self.min_take_profit_percent
        )

        # Определение действия
        if buy_signals >= self.min_signals_required and buy_signals > sell_signals:
            signal['action'] = 'BUY'
            signal['confidence'] = min(buy_signals / 3.5, 0.95)
            signal['stop_loss'] = float(close_ - sl_distance)
            signal['take_profit'] = float(close_ + tp_distance)

        elif sell_signals >= self.min_signals_required and sell_signals > buy_signals:
            signal['action'] = 'SELL'
            signal['confidence'] = min(sell_signals / 3.5, 0.95)
            signal['stop_loss'] = float(close_ + sl_distance)
            signal['take_profit'] = float(close_ - tp_distance)

        return signal

    async def _execute_trade(self, symbol: str, signal: Dict, current_price: float):
        """Исполнение сделки"""
        try:
            if symbol in self.positions:
                return

            # Проверка SL/TP
            if signal['action'] == 'BUY' and signal['stop_loss'] >= current_price:
                return
            elif signal['action'] == 'SELL' and signal['stop_loss'] <= current_price:
                return

            # Динамический размер
            atr_percent = (self.indicators[symbol].get('atr', 0) / current_price * 100) if current_price > 0 else 0
            dynamic_size = self._calculate_dynamic_position_size(symbol, signal['confidence'], atr_percent)
            position_value = self.current_balance * dynamic_size

            # Минимальный размер
            min_values = {'BTCUSDT': 10.0, 'ETHUSDT': 10.0, 'default': 10.0}
            min_value = min_values.get(symbol, min_values['default'])

            if position_value < min_value:
                return

            position_size = position_value / current_price

            # Округление
            step_sizes = {'BTCUSDT': 0.00001, 'ETHUSDT': 0.0001, 'default': 0.001}
            step_size = step_sizes.get(symbol, step_sizes['default'])
            position_size = round(position_size / step_size) * step_size

            # Проверка баланса
            required = position_size * current_price * 1.01
            if required > self.current_balance:
                return

            # Проверка риска
            can_open, reason = self.risk_manager.can_open_position(
                symbol=symbol,
                proposed_size=position_size * current_price
            )
            if not can_open:
                logger.logger.warning(f"Risk manager rejected {symbol}: {reason}")
                return

            # Слиппаж
            slippage = self.slippage_bps / 10000
            entry_price = current_price * (1 + slippage if signal['action'] == 'BUY' else 1 - slippage)

            # Создание позиции
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
                risk_amount=abs(entry_price - signal['stop_loss']) * position_size
            )

            # Регистрация
            self.positions[symbol] = position
            self.risk_manager.add_position(position)
            self._last_trade_time[symbol] = datetime.utcnow()
            self._partially_closed[symbol] = False

            # Комиссия
            position_cost = position_size * entry_price
            open_fee = position_cost * self.taker_fee
            setattr(position, "open_fee", float(open_fee))
            self.current_balance -= (position_cost + open_fee)

            # Логирование
            sl_percent = abs(signal['stop_loss'] - entry_price) / entry_price * 100
            tp_percent = abs(signal['take_profit'] - entry_price) / entry_price * 100

            logger.logger.info(
                f"💰 OPENED {symbol} {signal['action']} | "
                f"${entry_price:.2f} | Size: {dynamic_size*100:.1f}% | "
                f"SL: -{sl_percent:.1f}% | TP: +{tp_percent:.1f}%"
            )

        except Exception as e:
            logger.logger.error(f"Failed to execute trade for {symbol}: {e}")

    async def _update_positions(self):
        """Обновление позиций"""
        for symbol, position in list(self.positions.items()):
            try:
                current_price = self.indicators.get(symbol, {}).get('price')
                if not current_price:
                    continue

                await self._update_position_pnl(symbol, current_price)

                # Тайм-аут для флэта
                position_age = datetime.utcnow() - position.entry_time
                position_value = position.quantity * position.entry_price

                if position_age > timedelta(hours=self.position_timeout_hours):
                    pnl_percent = abs(position.unrealized_pnl) / position_value if position_value > 0 else 0
                    if pnl_percent < self.position_timeout_threshold:
                        await self._close_position(symbol, current_price, "timeout")
                        self.performance_metrics['timeout_closes'] += 1
                        continue

                # Частичное закрытие
                atr = self.indicators.get(symbol, {}).get('atr')
                if atr and atr > 0 and not self._partially_closed.get(symbol, False):
                    profit_distance = (current_price - position.entry_price if position.side == 'BUY'
                                     else position.entry_price - current_price)

                    if profit_distance > self.partial_close_level * atr:
                        await self._partial_close_position(symbol, current_price, 0.5, "partial_tp")
                        self._partially_closed[symbol] = True
                        self.performance_metrics['partial_closes'] += 1

                # Трейлинг-стоп
                if atr and atr > 0:
                    if position.side == 'BUY':
                        profit = current_price - position.entry_price
                        if profit > self.trailing_stop_activation * atr:
                            new_sl = max(
                                position.stop_loss or 0,
                                current_price - (self.trailing_stop_distance * atr),
                                position.entry_price * 1.005
                            )
                            if position.stop_loss is None or new_sl > position.stop_loss:
                                position.stop_loss = new_sl
                                setattr(position, 'trailing_activated', True)
                    else:
                        profit = position.entry_price - current_price
                        if profit > self.trailing_stop_activation * atr:
                            new_sl = min(
                                position.stop_loss or float('inf'),
                                current_price + (self.trailing_stop_distance * atr),
                                position.entry_price * 0.995
                            )
                            if position.stop_loss is None or new_sl < position.stop_loss:
                                position.stop_loss = new_sl
                                setattr(position, 'trailing_activated', True)

                # Проверка SL/TP
                if position.stop_loss:
                    if (position.side == 'BUY' and current_price <= position.stop_loss) or \
                       (position.side == 'SELL' and current_price >= position.stop_loss):
                        reason = "trailing_stop" if hasattr(position, 'trailing_activated') else "stop_loss"
                        await self._close_position(symbol, current_price, reason)
                        if reason == "trailing_stop":
                            self.performance_metrics['trailing_stop_hits'] += 1
                        else:
                            self.performance_metrics['stop_loss_hits'] += 1
                        continue

                if position.take_profit:
                    if (position.side == 'BUY' and current_price >= position.take_profit) or \
                       (position.side == 'SELL' and current_price <= position.take_profit):
                        await self._close_position(symbol, current_price, "take_profit")
                        self.performance_metrics['take_profit_hits'] += 1
                        continue

            except Exception as e:
                logger.logger.error(f"Failed to update position {symbol}: {e}")

    async def _partial_close_position(self, symbol: str, close_price: float,
                                     percentage: float, reason: str):
        """Частичное закрытие"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        close_quantity = position.quantity * percentage

        gross = ((close_price - position.entry_price) * close_quantity if position.side == 'BUY'
                else (position.entry_price - close_price) * close_quantity)

        close_fee = close_quantity * close_price * self.taker_fee
        partial_pnl = gross - close_fee

        if position.side == 'BUY':
            self.current_balance += (close_quantity * close_price - close_fee)
        else:
            entry_value = close_quantity * position.entry_price
            self.current_balance += entry_value + partial_pnl

        position.quantity -= close_quantity

        pnl_percent = partial_pnl / (close_quantity * position.entry_price) * 100

        logger.logger.info(
            f"💵 PARTIAL {symbol} | {percentage*100:.0f}% | "
            f"PnL: ${partial_pnl:+.2f} ({pnl_percent:+.1f}%)"
        )

        self.trade_history.append({
            'symbol': symbol,
            'side': position.side,
            'type': 'partial_close',
            'entry_price': position.entry_price,
            'close_price': close_price,
            'quantity': close_quantity,
            'pnl': partial_pnl,
            'reason': reason,
            'timestamp': datetime.utcnow()
        })

    async def _close_position(self, symbol: str, close_price: float, reason: str):
        """Полное закрытие позиции"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        gross = ((close_price - position.entry_price) * position.quantity if position.side == 'BUY'
                else (position.entry_price - close_price) * position.quantity)

        close_fee = position.quantity * close_price * self.taker_fee
        open_fee = float(getattr(position, "open_fee", 0.0))
        pnl = gross - (open_fee + close_fee)

        if position.side == 'BUY':
            self.current_balance += (position.quantity * close_price - close_fee)
        else:
            entry_value = position.quantity * position.entry_price
            self.current_balance += entry_value + pnl

        # Время закрытия
        self._last_closed_time[symbol] = datetime.utcnow()

        # История производительности
        position_value = position.quantity * position.entry_price
        pnl_percent = (pnl / position_value) if position_value > 0 else 0

        if symbol not in self._position_performance:
            self._position_performance[symbol] = deque(maxlen=self._max_performance_history)
        self._position_performance[symbol].append(pnl_percent)

        del self.positions[symbol]
        if symbol in self._partially_closed:
            del self._partially_closed[symbol]

        # История
        self.trade_history.append({
            'symbol': symbol,
            'side': position.side,
            'type': 'full_close',
            'entry_price': position.entry_price,
            'close_price': close_price,
            'quantity': position.quantity,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'fees': {'open': open_fee, 'close': close_fee},
            'reason': reason,
            'timestamp': datetime.utcnow(),
            'duration': (datetime.utcnow() - position.entry_time).total_seconds() / 3600
        })

        # Кулдаун при убытках - короткий
        if pnl < 0:
            self._loss_streak[symbol] = self._loss_streak.get(symbol, 0) + 1
            if self._loss_streak[symbol] >= 2:  # После 2 убытков
                self._cooldown_until[symbol] = datetime.utcnow() + timedelta(minutes=5)  # 5 минут
                self._loss_streak[symbol] = 0
        else:
            self._loss_streak[symbol] = 0
            if symbol in self._cooldown_until:
                del self._cooldown_until[symbol]

        emoji = "💰" if pnl > 0 else "💸"
        logger.logger.info(
            f"{emoji} CLOSED {symbol} | {position.side} | "
            f"${position.entry_price:.2f} → ${close_price:.2f} | "
            f"PnL: ${pnl:+.2f} ({pnl_percent*100:+.1f}%) | {reason}"
        )

    async def _update_position_pnl(self, symbol: str, current_price: float):
        """Обновление PnL"""
        if symbol in self.positions:
            position = self.positions[symbol]
            position.current_price = current_price
            position.unrealized_pnl = ((current_price - position.entry_price) * position.quantity
                                      if position.side == 'BUY'
                                      else (position.entry_price - current_price) * position.quantity)

    async def _check_risk_limits(self):
        """Проверка лимитов риска"""
        try:
            risk_metrics = self.risk_manager.get_risk_metrics()
            if risk_metrics.risk_level.value == "critical":
                logger.logger.warning("⚠️ CRITICAL RISK - closing all positions")
                for symbol in list(self.positions.keys()):
                    price = self.indicators.get(symbol, {}).get('price', self.positions[symbol].entry_price)
                    await self._close_position(symbol, price, "risk_limit")
        except Exception as e:
            logger.logger.error(f"Error checking risk limits: {e}")

    async def _log_status(self):
        """Логирование статуса"""
        try:
            positions_value = sum(
                pos.quantity * (pos.current_price if pos.current_price > 0 else pos.entry_price)
                for pos in self.positions.values()
            )

            equity = self.current_balance + positions_value
            total_pnl = equity - self.initial_balance

            all_trades = [t for t in self.trade_history if t.get('type') != 'partial_close']
            wins = [t for t in all_trades if t['pnl'] > 0]
            losses = [t for t in all_trades if t['pnl'] < 0]

            win_rate = (len(wins) / len(all_trades) * 100) if all_trades else 0

            logger.logger.info(
                f"📈 Equity: ${equity:.2f} | Balance: ${self.current_balance:.2f} | "
                f"Pos: {len(self.positions)}/{self.max_positions} | "
                f"Trades: {len(all_trades)} | WR: {win_rate:.0f}% | "
                f"PnL: ${total_pnl:+.2f} ({total_pnl/self.initial_balance*100:+.2f}%)"
            )

            if self.performance_metrics['total_signals'] > 0:
                exec_rate = self.performance_metrics['executed_signals'] / self.performance_metrics['total_signals'] * 100
                logger.logger.info(
                    f"🎯 Signals: {self.performance_metrics['total_signals']} | "
                    f"Executed: {self.performance_metrics['executed_signals']} ({exec_rate:.1f}%) | "
                    f"Duplicates: {self.performance_metrics['skipped_duplicate_signal']}"
                )

        except Exception as e:
            logger.logger.error(f"Error logging status: {e}")

    async def stop(self):
        """Остановка бота"""
        logger.logger.info("Stopping bot v9.0...")
        self.running = False

        try:
            # Закрываем все позиции
            for symbol in list(self.positions.keys()):
                price = self.indicators.get(symbol, {}).get('price', self.positions[symbol].entry_price)
                await self._close_position(symbol, price, "bot_stopped")

            # Закрываем соединения
            if self.binance_client:
                await self.binance_client.close_connection()

            await ws_client.stop()

            # Финальный отчет
            all_trades = [t for t in self.trade_history if t.get('type') != 'partial_close']

            if all_trades:
                final_pnl = sum(t.get('pnl', 0.0) for t in all_trades)
                final_balance = self.current_balance
                final_return = ((final_balance - self.initial_balance) / self.initial_balance) * 100

                logger.logger.info("=" * 80)
                logger.logger.info("📊 FINAL TRADING REPORT v9.0")
                logger.logger.info("=" * 80)
                logger.logger.info(
                    f"💰 Results: Initial: ${self.initial_balance:.2f} | "
                    f"Final: ${final_balance:.2f} | PnL: ${final_pnl:.2f} | "
                    f"Return: {final_return:.2f}%"
                )

                wins = [t for t in all_trades if t['pnl'] > 0]
                losses = [t for t in all_trades if t['pnl'] < 0]
                win_rate = len(wins) / len(all_trades) * 100 if all_trades else 0
                avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
                avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
                profit_factor = abs(sum(t['pnl'] for t in wins) / sum(t['pnl'] for t in losses)) if losses and sum(t['pnl'] for t in losses) != 0 else 0

                avg_duration = sum(t.get('duration', 0) for t in all_trades) / len(all_trades) if all_trades else 0

                logger.logger.info(
                    f"📈 Stats: Trades: {len(all_trades)} | "
                    f"Wins: {len(wins)} | Losses: {len(losses)} | "
                    f"Win Rate: {win_rate:.1f}% | PF: {profit_factor:.2f} | "
                    f"Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}"
                )

                logger.logger.info(
                    f"⏱️ Timing: Avg Duration: {avg_duration:.1f}h | "
                    f"SL: {self.performance_metrics['stop_loss_hits']} | "
                    f"TP: {self.performance_metrics['take_profit_hits']} | "
                    f"Trail: {self.performance_metrics['trailing_stop_hits']} | "
                    f"Timeout: {self.performance_metrics['timeout_closes']}"
                )

                logger.logger.info(
                    f"📊 Signals: Total: {self.performance_metrics['total_signals']} | "
                    f"Executed: {self.performance_metrics['executed_signals']} | "
                    f"Skipped Confidence: {self.performance_metrics['skipped_low_confidence']} | "
                    f"Skipped Volatility: {self.performance_metrics['skipped_low_volatility']}"
                )

                # Performance by symbol
                symbol_stats = {}
                for trade in all_trades:
                    symbol = trade['symbol']
                    if symbol not in symbol_stats:
                        symbol_stats[symbol] = {'trades': 0, 'pnl': 0, 'wins': 0}
                    symbol_stats[symbol]['trades'] += 1
                    symbol_stats[symbol]['pnl'] += trade['pnl']
                    if trade['pnl'] > 0:
                        symbol_stats[symbol]['wins'] += 1

                if symbol_stats:
                    logger.logger.info("📊 Performance by Symbol:")
                    for symbol, stats in sorted(symbol_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
                        symbol_wr = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
                        logger.logger.info(
                            f"   {symbol}: {stats['trades']} trades | "
                            f"WR: {symbol_wr:.0f}% | PnL: ${stats['pnl']:.2f}"
                        )

                logger.logger.info("=" * 80)
                logger.logger.info("🚀 KEY FEATURES v9.0:")
                logger.logger.info("   • Aggressive parameters for maximum activity")
                logger.logger.info("   • Fixed volume_ratio bug")
                logger.logger.info("   • Short signal cache (1 min)")
                logger.logger.info("   • Fast position rotation")
                logger.logger.info("   • Minimal filters for 24/7 trading")
                logger.logger.info("=" * 80)

            else:
                logger.logger.info("No trades executed during this session")

        except Exception as e:
            logger.logger.error(f"Error stopping bot: {e}")