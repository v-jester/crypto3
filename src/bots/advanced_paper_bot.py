# src/bots/advanced_paper_bot.py
"""
Продвинутый Paper Trading Bot с реальными данными Binance и ML стратегиями
ВЕРСИЯ: Агрессивная торговля с оптимизированными параметрами
"""
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from binance import AsyncClient
from binance.streams import BinanceSocketManager
from binance.enums import *
import pandas_ta as ta

from src.monitoring.logger import logger
from src.config.settings import settings
from src.data.storage.redis_client import redis_client, cache_manager
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

        # АГРЕССИВНЫЙ РЕЖИМ ДЛЯ АКТИВНОЙ ТОРГОВЛИ
        self.aggressive_mode = True
        self.position_percent = 0.10  # 10% от баланса на позицию (увеличено с 5%)
        self.min_confidence = 0.40  # Снижен порог уверенности (было 0.55)
        self.min_signals_required = 1.2  # Снижено требование к сигналам (было 2.0)

        # Компоненты системы
        self.binance_client = None
        self.socket_manager = None
        self.data_collector = HistoricalDataCollector()
        self.risk_manager = RiskManager(
            initial_capital=initial_balance,
            max_drawdown=settings.trading.MAX_DRAWDOWN_PERCENT,
            max_daily_loss=settings.trading.MAX_DAILY_LOSS_PERCENT,
            max_positions=settings.trading.MAX_POSITIONS
        )

        # Торговые данные
        self.positions = {}
        self.trade_history = []
        self.market_data = {}
        self.indicators = {}

        # Добавляем timestamp последнего обновления и последние значения RSI
        self.last_data_update = {}
        self.data_update_interval = 30  # секунд
        self.last_rsi_values = {}  # Для отслеживания изменений RSI
        self.data_fetch_failures = {}  # Счетчик неудачных попыток получения данных
        self.last_trade_time = {}  # Время последней сделки по символу

        # НЕ используем альтернативные таймфреймы - всегда основной
        self.primary_timeframe = settings.trading.PRIMARY_TIMEFRAME
        self.use_fallback_timeframes = False  # Отключаем переключение таймфреймов

        # Флаги состояния
        self.running = False
        self.connected = False

        # Метрики производительности
        self.performance_metrics = {
            'total_analyses': 0,
            'total_signals': 0,
            'executed_signals': 0,
            'hold_signals': 0,
            'skipped_low_volatility': 0,
            'skipped_time_limit': 0,
            'skipped_low_confidence': 0,
            'skipped_insufficient_data': 0,
            'skipped_position_exists': 0
        }

    def get_adaptive_volatility_threshold(self, symbol: str) -> float:
        """
        Возвращает адаптивный порог волатильности для символа

        В агрессивном режиме пороги снижены для большей активности
        """
        base_thresholds = {
            'BTCUSDT': 0.015,  # 1.5% для BTC
            'ETHUSDT': 0.020,  # 2.0% для ETH
            'SOLUSDT': 0.025,  # 2.5% для SOL
            'DOGEUSDT': 0.030,  # 3.0% для DOGE
            'AVAXUSDT': 0.025,  # 2.5% для AVAX
            'BNBUSDT': 0.018,  # 1.8% для BNB
            'ADAUSDT': 0.022,  # 2.2% для ADA
            'XRPUSDT': 0.023,  # 2.3% для XRP
            'MATICUSDT': 0.028,  # 2.8% для MATIC
            'DOTUSDT': 0.024,  # 2.4% для DOT
        }

        threshold = base_thresholds.get(symbol, 0.020)

        # В агрессивном режиме снижаем пороги на 30%
        if self.aggressive_mode:
            threshold *= 0.7

        return threshold

    async def initialize(self):
        """Полная инициализация бота"""
        try:
            logger.logger.info("Initializing Advanced Paper Trading Bot (AGGRESSIVE MODE)")

            # 1. Подключение к Binance API
            await self._connect_binance()

            # 2. Загрузка исторических данных
            await self._load_historical_data()

            # 3. Инициализация ML моделей
            await self._initialize_ml()

            # 4. Запуск WebSocket стримов или polling
            await self._start_data_streams()

            logger.logger.info(
                f"Advanced Paper Trading Bot initialized | "
                f"Initial balance: ${self.initial_balance:,.2f} | "
                f"Symbols: {settings.trading.SYMBOLS[:3]} | "
                f"Timeframe: {settings.trading.PRIMARY_TIMEFRAME} | "
                f"Mode: {'AGGRESSIVE' if self.aggressive_mode else 'NORMAL'}"
            )

        except Exception as e:
            logger.logger.error(f"Failed to initialize advanced bot: {e}")
            raise

    async def _connect_binance(self):
        """Подключение к Binance API"""
        try:
            # Используем testnet для paper trading
            self.binance_client = await AsyncClient.create(
                api_key=settings.api.BINANCE_API_KEY.get_secret_value(),
                api_secret=settings.api.BINANCE_API_SECRET.get_secret_value(),
                testnet=settings.api.TESTNET
            )

            # Проверяем подключение
            await self.binance_client.ping()

            # Получаем информацию об аккаунте (для testnet)
            account_info = await self.binance_client.get_account()
            logger.logger.info(f"Connected to Binance {'Testnet' if settings.api.TESTNET else 'Live'}")

            # Инициализируем data collector с клиентом
            await self.data_collector.initialize(self.binance_client)

            # ВАЖНО: Устанавливаем флаги для отключения кеша
            self.data_collector.force_refresh = True
            self.data_collector.use_cache = False

            self.connected = True

        except Exception as e:
            logger.logger.error(f"Binance connection failed: {e}")
            # Продолжаем работу в оффлайн режиме
            self.connected = False

    async def _load_historical_data(self):
        """Загрузка исторических данных для анализа"""
        logger.logger.info("Loading historical data...")

        for symbol in settings.trading.SYMBOLS[:3]:
            try:
                # ИСПРАВЛЕНО: Принудительно обновляем данные
                self.data_collector.force_refresh = True
                self.data_collector.use_cache = False

                # Загружаем данные для основного таймфрейма
                df = await self.data_collector.fetch_historical_data(
                    symbol=symbol,
                    interval=self.primary_timeframe,
                    days_back=1,
                    limit=100,
                    force_refresh=True
                )

                if not df.empty:
                    self.market_data[symbol] = df
                    self.last_data_update[symbol] = datetime.utcnow()
                    self.data_fetch_failures[symbol] = 0
                    self.last_trade_time[symbol] = datetime.utcnow() - timedelta(hours=1)  # Разрешаем сразу торговать

                    # Сохраняем последние индикаторы
                    self.indicators[symbol] = {
                        'rsi': df['rsi'].iloc[-1] if 'rsi' in df.columns else 50,
                        'macd': df['macd'].iloc[-1] if 'macd' in df.columns else 0,
                        'bb_position': df['bb_percent'].iloc[-1] if 'bb_percent' in df.columns else 0.5,
                        'volume_ratio': df['volume_ratio'].iloc[-1] if 'volume_ratio' in df.columns else 1,
                        'atr': df['atr'].iloc[-1] if 'atr' in df.columns else 0,
                        'price': df['close'].iloc[-1] if 'close' in df.columns else 0
                    }

                    # Сохраняем начальное значение RSI
                    self.last_rsi_values[symbol] = self.indicators[symbol]['rsi']

                    logger.logger.info(
                        f"Loaded {len(df)} candles for {symbol} | "
                        f"Price: ${self.indicators[symbol]['price']:,.2f} | "
                        f"RSI: {self.indicators[symbol]['rsi']:.1f}"
                    )
                else:
                    logger.logger.warning(f"No data loaded for {symbol}")
                    self.data_fetch_failures[symbol] = 1

            except Exception as e:
                logger.logger.warning(f"Failed to load data for {symbol}: {e}")
                self.data_fetch_failures[symbol] = 1

    async def _initialize_ml(self):
        """Инициализация ML моделей"""
        try:
            # Пытаемся загрузить существующие модели
            await ml_engine.load_models()
            logger.logger.info("ML models loaded from disk")
        except:
            logger.logger.info("No saved models found, will train on first suitable data")

    async def _start_data_streams(self):
        """Запуск потоков данных (WebSocket или REST API polling)"""
        # Для testnet используем REST API polling вместо WebSocket
        if settings.api.TESTNET:
            logger.logger.warning("Using REST API polling for testnet (WebSocket issues)")
            # ВАЖНО: Устанавливаем флаг running ДО запуска polling
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
            # Подписываемся на стримы свечей
            await ws_client.subscribe_klines(
                symbols=settings.trading.SYMBOLS[:3],
                intervals=[self.primary_timeframe],
                handler=self._handle_kline_update
            )

            # Подписываемся на тикеры для отслеживания цен
            await ws_client.subscribe_ticker(
                symbols=settings.trading.SYMBOLS[:3],
                handler=self._handle_ticker_update
            )

            # Запускаем WebSocket клиент
            await ws_client.start()

            logger.logger.info("WebSocket streams started")

        except Exception as e:
            logger.logger.error(f"Failed to start websocket streams: {e}")
            # Fallback на REST API
            self.running = True
            asyncio.create_task(self._poll_market_data())

    async def _poll_market_data(self):
        """Альтернативный метод обновления данных через REST API - ФИКСИРОВАННЫЙ"""
        logger.logger.info("Starting REST API polling for market data")

        poll_interval = 10  # секунд между обновлениями данных
        consecutive_failures = {}

        while self.running:
            try:
                for symbol in settings.trading.SYMBOLS[:3]:
                    try:
                        now = datetime.utcnow()

                        # Инициализируем счетчик если его нет
                        if symbol not in consecutive_failures:
                            consecutive_failures[symbol] = 0

                        # Получаем текущую цену
                        ticker = await self.binance_client.get_ticker(symbol=symbol)
                        current_price = float(ticker['lastPrice'])

                        # Очистка кешей
                        try:
                            deleted = await cache_manager.invalidate_symbol(symbol)
                            if deleted > 0:
                                logger.logger.debug(f"Cleared {deleted} cache keys for {symbol}")
                        except:
                            pass

                        # Устанавливаем флаги для принудительного обновления
                        self.data_collector.force_refresh = True
                        self.data_collector.use_cache = False

                        # ВАЖНО: Используем ТОЛЬКО основной таймфрейм
                        df = await self.data_collector.fetch_historical_data(
                            symbol=symbol,
                            interval=self.primary_timeframe,
                            days_back=1,
                            limit=100,
                            force_refresh=True
                        )

                        if not df.empty and 'rsi' in df.columns:
                            # Сохраняем старые значения для сравнения
                            old_rsi = self.indicators.get(symbol, {}).get('rsi', 50)
                            old_price = self.indicators.get(symbol, {}).get('price', 0)

                            # Обновляем данные
                            self.market_data[symbol] = df
                            self.last_data_update[symbol] = now
                            consecutive_failures[symbol] = 0

                            # Обновляем индикаторы
                            new_rsi = df['rsi'].iloc[-1] if not pd.isna(df['rsi'].iloc[-1]) else 50

                            self.indicators[symbol] = {
                                'rsi': new_rsi,
                                'macd': df['macd'].iloc[-1] if 'macd' in df.columns else 0,
                                'bb_position': df['bb_percent'].iloc[-1] if 'bb_percent' in df.columns else 0.5,
                                'volume_ratio': df['volume_ratio'].iloc[-1] if 'volume_ratio' in df.columns else 1,
                                'atr': df['atr'].iloc[-1] if 'atr' in df.columns else current_price * 0.02,
                                'price': current_price,
                                'momentum_5': df['momentum_5'].iloc[-1] if 'momentum_5' in df.columns else 0,
                                'ema_9': df['ema_9'].iloc[-1] if 'ema_9' in df.columns else current_price,
                                'ema_20': df['ema_20'].iloc[-1] if 'ema_20' in df.columns else current_price,
                                'volatility': df['volatility'].iloc[-1] if 'volatility' in df.columns else 0.01
                            }

                            # Логируем изменения
                            if abs(new_rsi - old_rsi) > 0.1 or abs(current_price - old_price) > 0.01:
                                logger.logger.info(
                                    f"✅ Data updated for {symbol} | "
                                    f"Price: ${old_price:.2f} → ${current_price:.2f} | "
                                    f"RSI: {old_rsi:.1f} → {new_rsi:.1f}"
                                )

                            self.last_rsi_values[symbol] = new_rsi

                        else:
                            consecutive_failures[symbol] += 1
                            logger.logger.warning(
                                f"Failed to get valid data for {symbol} (attempt {consecutive_failures[symbol]})")

                            # Если много неудач подряд, используем последние известные данные
                            if consecutive_failures[symbol] > 3 and symbol in self.market_data:
                                logger.logger.info(f"Using last known data for {symbol}")
                                # Обновляем только цену
                                if symbol in self.indicators:
                                    self.indicators[symbol]['price'] = current_price

                    except Exception as e:
                        logger.logger.error(f"Failed to poll data for {symbol}: {e}")
                        consecutive_failures[symbol] = consecutive_failures.get(symbol, 0) + 1

                # Ждём перед следующим обновлением
                await asyncio.sleep(poll_interval)

            except asyncio.CancelledError:
                logger.logger.info("Market data polling cancelled")
                break
            except Exception as e:
                logger.logger.error(f"Error in polling loop: {e}")
                await asyncio.sleep(10)

    async def _handle_kline_update(self, data: Dict):
        """Обработка обновлений свечей"""
        try:
            kline = data.get('k', {})
            if kline.get('x'):  # Свеча закрылась
                symbol = kline.get('s')
                logger.logger.debug(f"Closed candle for {symbol}: {kline.get('c')}")

                # Обновляем рыночные данные
                await self._update_market_data(symbol)
        except Exception as e:
            logger.logger.error(f"Error handling kline update: {e}")

    async def _handle_ticker_update(self, data: Dict):
        """Обработка обновлений тикера"""
        try:
            symbol = data.get('s')
            price = float(data.get('c', 0))

            if symbol and price > 0:
                # Обновляем текущие цены для позиций
                if symbol in self.positions:
                    await self._update_position_pnl(symbol, price)
        except Exception as e:
            logger.logger.error(f"Error handling ticker update: {e}")

    async def _update_market_data(self, symbol: str):
        """Обновление рыночных данных и индикаторов"""
        try:
            # Принудительное обновление
            self.data_collector.force_refresh = True
            self.data_collector.use_cache = False

            # Очищаем кеш
            deleted = await cache_manager.invalidate_symbol(symbol)
            if deleted > 0:
                logger.logger.debug(f"Cleared {deleted} cache keys for {symbol}")

            # Получаем свежие данные ТОЛЬКО для основного таймфрейма
            df = await self.data_collector.fetch_historical_data(
                symbol=symbol,
                interval=self.primary_timeframe,
                days_back=1,
                limit=100,
                force_refresh=True
            )

            if not df.empty:
                self.market_data[symbol] = df
                self.last_data_update[symbol] = datetime.utcnow()

                # Обновляем индикаторы
                self.indicators[symbol] = {
                    'rsi': df['rsi'].iloc[-1] if 'rsi' in df.columns else 50,
                    'macd': df['macd'].iloc[-1] if 'macd' in df.columns else 0,
                    'bb_position': df['bb_percent'].iloc[-1] if 'bb_percent' in df.columns else 0.5,
                    'volume_ratio': df['volume_ratio'].iloc[-1] if 'volume_ratio' in df.columns else 1,
                    'atr': df['atr'].iloc[-1] if 'atr' in df.columns else 0,
                    'price': df['close'].iloc[-1],
                    'momentum_5': df['momentum_5'].iloc[-1] if 'momentum_5' in df.columns else 0,
                    'ema_9': df['ema_9'].iloc[-1] if 'ema_9' in df.columns else df['close'].iloc[-1],
                    'ema_20': df['ema_20'].iloc[-1] if 'ema_20' in df.columns else df['close'].iloc[-1],
                    'volatility': df['volatility'].iloc[-1] if 'volatility' in df.columns else 0.01
                }

        except Exception as e:
            logger.logger.error(f"Failed to update market data for {symbol}: {e}")

    async def run(self):
        """Основной торговый цикл с улучшенной обработкой ошибок"""
        self.running = True
        logger.logger.info("Starting Advanced Paper Trading Bot main loop (AGGRESSIVE MODE)")

        analysis_counter = 0

        try:
            while self.running:
                try:
                    # АГРЕССИВНЫЙ РЕЖИМ: Анализируем рынок каждые 15 секунд
                    if analysis_counter % 3 == 0:  # 15 секунд (5 сек * 3)
                        await self._analyze_and_trade()

                    # Обновляем позиции
                    await self._update_positions()

                    # Проверяем риски
                    await self._check_risk_limits()

                    # Логируем статус каждую минуту
                    if analysis_counter % 12 == 0:  # 60 секунд
                        await self._log_status()
                        analysis_counter = 0

                    analysis_counter += 1
                    await asyncio.sleep(5)

                except asyncio.CancelledError:
                    logger.logger.info("Trading loop cancelled by user")
                    break
                except Exception as e:
                    logger.logger.error(f"Error in trading iteration: {e}")
                    # Продолжаем работу после ошибки
                    await asyncio.sleep(5)

        except asyncio.CancelledError:
            logger.logger.info("Trading loop cancelled")
        except Exception as e:
            logger.logger.error(f"Critical error in trading loop: {e}")
            raise
        finally:
            logger.logger.info("Advanced Paper Trading Bot main loop stopped")

    async def _analyze_and_trade(self):
        """Анализ рынка и принятие торговых решений с валидацией данных"""
        self.performance_metrics['total_analyses'] += 1

        for symbol in settings.trading.SYMBOLS[:3]:
            try:
                # Проверяем наличие и валидность данных
                if symbol not in self.market_data:
                    logger.logger.debug(f"No market data for {symbol}")
                    self.performance_metrics['skipped_insufficient_data'] += 1
                    continue

                df = self.market_data[symbol]

                # ИСПРАВЛЕНИЕ: Более строгая проверка данных
                if df.empty or len(df) < 20:  # Минимум 20 свечей для анализа
                    logger.logger.debug(f"Insufficient data for {symbol}: {len(df)} candles")
                    self.performance_metrics['skipped_insufficient_data'] += 1
                    continue

                # Проверяем что индикаторы присутствуют и валидны
                required_columns = ['close', 'volume', 'rsi']
                if not all(col in df.columns for col in required_columns):
                    logger.logger.warning(f"Missing required columns for {symbol}")
                    self.performance_metrics['skipped_insufficient_data'] += 1
                    continue

                # Проверяем что RSI не все NaN
                if df['rsi'].isna().all():
                    logger.logger.warning(f"All RSI values are NaN for {symbol}")
                    self.performance_metrics['skipped_insufficient_data'] += 1
                    continue

                # Получаем текущие индикаторы
                indicators = self.indicators.get(symbol, {})
                current_price = indicators.get('price', df['close'].iloc[-1] if not df.empty else 0)

                if current_price == 0:
                    logger.logger.warning(f"Zero price for {symbol}, skipping")
                    continue

                # Генерируем торговые сигналы
                signal = await self._generate_trading_signal(symbol, df, indicators)

                if signal['action'] == 'HOLD':
                    self.performance_metrics['hold_signals'] += 1
                else:
                    self.performance_metrics['total_signals'] += 1

                    logger.logger.info(
                        f"📊 Signal generated for {symbol} | "
                        f"Action: {signal['action']} | "
                        f"Confidence: {signal['confidence']:.2f} | "
                        f"Reasons: {', '.join(signal['reasons'])}"
                    )

                    # АГРЕССИВНЫЙ РЕЖИМ: Сниженный порог уверенности
                    if signal['confidence'] >= self.min_confidence:
                        # Проверяем время последней сделки (минимум 5 минут между сделками)
                        time_since_last_trade = (
                                    datetime.utcnow() - self.last_trade_time.get(symbol, datetime.min)).total_seconds()
                        if time_since_last_trade > 300:  # 5 минут
                            await self._execute_trade(symbol, signal, current_price)
                            self.performance_metrics['executed_signals'] += 1
                        else:
                            logger.logger.debug(f"Too soon after last trade for {symbol}: {time_since_last_trade:.0f}s")
                            self.performance_metrics['skipped_time_limit'] += 1
                    else:
                        logger.logger.debug(
                            f"Signal confidence too low for {symbol}: {signal['confidence']:.2f}"
                        )
                        self.performance_metrics['skipped_low_confidence'] += 1

            except Exception as e:
                logger.logger.error(f"Analysis failed for {symbol}: {e}", exc_info=True)

    async def _generate_trading_signal(self, symbol: str, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Генерация торгового сигнала на основе множественных стратегий (АГРЕССИВНАЯ ВЕРСИЯ)"""
        signal = {
            'action': 'HOLD',
            'confidence': 0.0,
            'reasons': [],
            'stop_loss': None,
            'take_profit': None
        }

        buy_signals = 0
        sell_signals = 0

        # 1. RSI стратегия (АГРЕССИВНЫЕ УРОВНИ)
        rsi = indicators.get('rsi', 50)
        if not pd.isna(rsi):
            if rsi < 35:  # Oversold (было 30)
                buy_signals += 1.5
                signal['reasons'].append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 65:  # Overbought (было 70)
                sell_signals += 1.5
                signal['reasons'].append(f"RSI overbought ({rsi:.1f})")
            elif rsi < 45:  # Mild oversold (было 40)
                buy_signals += 0.8
                signal['reasons'].append(f"RSI low ({rsi:.1f})")
            elif rsi > 55:  # Mild overbought (было 60)
                sell_signals += 0.8
                signal['reasons'].append(f"RSI high ({rsi:.1f})")

        # 2. MACD стратегия
        macd = indicators.get('macd', 0)
        if 'macd_signal' in df.columns and len(df) > 0:
            macd_signal_val = df['macd_signal'].iloc[-1]
            if not pd.isna(macd) and not pd.isna(macd_signal_val):
                macd_diff = macd - macd_signal_val

                if macd_diff > 0:
                    buy_signals += 0.8
                    signal['reasons'].append(f"MACD bullish ({macd_diff:.4f})")
                elif macd_diff < 0:
                    sell_signals += 0.8
                    signal['reasons'].append(f"MACD bearish ({macd_diff:.4f})")

        # 3. Bollinger Bands стратегия
        bb_position = indicators.get('bb_position', 0.5)
        if not pd.isna(bb_position):
            if bb_position < 0.25:  # Near lower band (было 0.2)
                buy_signals += 1.0
                signal['reasons'].append(f"Price near lower BB ({bb_position:.2f})")
            elif bb_position > 0.75:  # Near upper band (было 0.8)
                sell_signals += 1.0
                signal['reasons'].append(f"Price near upper BB ({bb_position:.2f})")

        # 4. Volume подтверждение
        volume_ratio = indicators.get('volume_ratio', 1)
        if volume_ratio > 1.3:  # High volume (снижено с 1.5)
            if buy_signals > sell_signals:
                buy_signals += 0.5
                signal['reasons'].append(f"High volume confirmation ({volume_ratio:.1f}x)")
            elif sell_signals > buy_signals:
                sell_signals += 0.5
                signal['reasons'].append(f"High volume confirmation ({volume_ratio:.1f}x)")

        # 5. ML предсказание (если доступно)
        try:
            if hasattr(ml_engine, 'is_trained') and ml_engine.is_trained:
                ml_prediction = await ml_engine.predict(df, return_proba=True)
                if ml_prediction and 'prediction_label' in ml_prediction:
                    if ml_prediction['prediction_label'] == 'UP' and ml_prediction.get('confidence', 0) > 0.50:
                        buy_signals += 1.2
                        signal['reasons'].append(f"ML: UP ({ml_prediction['confidence']:.1%})")
                    elif ml_prediction['prediction_label'] == 'DOWN' and ml_prediction.get('confidence', 0) > 0.50:
                        sell_signals += 1.2
                        signal['reasons'].append(f"ML: DOWN ({ml_prediction['confidence']:.1%})")
        except Exception as e:
            logger.logger.debug(f"ML prediction skipped: {e}")

        # 6. Momentum стратегия (НОВАЯ)
        momentum = indicators.get('momentum_5', 0)
        if momentum > 0.01:  # Рост больше 1%
            buy_signals += 0.5
            signal['reasons'].append(f"Positive momentum ({momentum:.2%})")
        elif momentum < -0.01:  # Падение больше 1%
            sell_signals += 0.5
            signal['reasons'].append(f"Negative momentum ({momentum:.2%})")

        # 7. EMA crossover (НОВАЯ)
        ema_short = indicators.get('ema_9', 0)
        ema_long = indicators.get('ema_20', 0)
        if ema_short > 0 and ema_long > 0:
            if ema_short > ema_long:
                buy_signals += 0.3
                signal['reasons'].append("EMA bullish crossover")
            else:
                sell_signals += 0.3
                signal['reasons'].append("EMA bearish crossover")

        # 8. Volatility bonus в агрессивном режиме
        if self.aggressive_mode:
            volatility = indicators.get('volatility', 0)
            if volatility > 0.015:  # Высокая волатильность
                if buy_signals > sell_signals:
                    buy_signals += 0.4
                    signal['reasons'].append(f"High volatility bonus ({volatility:.3f})")
                elif sell_signals > buy_signals:
                    sell_signals += 0.4
                    signal['reasons'].append(f"High volatility bonus ({volatility:.3f})")

        # Определяем финальное действие (СНИЖЕННЫЕ ТРЕБОВАНИЯ)
        min_signals_required = self.min_signals_required  # 1.2 в агрессивном режиме

        if buy_signals >= min_signals_required and buy_signals > sell_signals:
            signal['action'] = 'BUY'
            signal['confidence'] = min(buy_signals / 4.0, 0.90)

            # Рассчитываем стоп-лосс и тейк-профит (БОЛЕЕ БЛИЗКИЕ УРОВНИ)
            atr = indicators.get('atr', df['close'].iloc[-1] * 0.02)
            signal['stop_loss'] = df['close'].iloc[-1] - (atr * 1.0)  # Было 2.0
            signal['take_profit'] = df['close'].iloc[-1] + (atr * 1.5)  # Было 3.0

        elif sell_signals >= min_signals_required and sell_signals > buy_signals:
            signal['action'] = 'SELL'
            signal['confidence'] = min(sell_signals / 4.0, 0.90)

            atr = indicators.get('atr', df['close'].iloc[-1] * 0.02)
            signal['stop_loss'] = df['close'].iloc[-1] + (atr * 1.0)  # Было 2.0
            signal['take_profit'] = df['close'].iloc[-1] - (atr * 1.5)  # Было 3.0

        return signal

    async def _execute_trade(self, symbol: str, signal: Dict, current_price: float):
        """Выполнение торговой операции с исправленным учетом баланса"""
        try:
            # Проверяем, есть ли уже позиция по этому символу
            if symbol in self.positions:
                logger.logger.info(f"Position already exists for {symbol}")
                self.performance_metrics['skipped_position_exists'] += 1
                return

            # Расчет размера позиции (УВЕЛИЧЕННЫЙ)
            position_value = self.current_balance * self.position_percent

            # Минимальный размер позиции для Binance (СНИЖЕННЫЙ)
            min_position_values = {
                'BTCUSDT': 5.0,  # Снижено с 10
                'ETHUSDT': 5.0,
                'BNBUSDT': 5.0,
                'SOLUSDT': 5.0,
                'default': 5.0
            }

            min_value = min_position_values.get(symbol, min_position_values['default'])

            # Проверяем минимальный размер
            if position_value < min_value:
                logger.logger.warning(
                    f"Position size too small for {symbol}: ${position_value:.2f} < ${min_value}"
                )
                return

            # Рассчитываем количество
            position_size = position_value / current_price

            # Округляем количество согласно правилам биржи
            step_sizes = {
                'BTCUSDT': 0.00001,
                'ETHUSDT': 0.0001,
                'BNBUSDT': 0.001,
                'SOLUSDT': 0.001,
                'default': 0.001
            }

            step_size = step_sizes.get(symbol, step_sizes['default'])
            position_size = round(position_size / step_size) * step_size

            # Финальная проверка
            final_value = position_size * current_price
            if final_value < min_value:
                position_size = (min_value * 1.1) / current_price
                position_size = round(position_size / step_size) * step_size

            # ИСПРАВЛЕНИЕ: Правильная проверка баланса
            position_cost = position_size * current_price
            fee = position_cost * self.taker_fee
            total_required = position_cost + fee

            # Проверяем достаточность баланса
            if total_required > self.current_balance * 0.95:  # Оставляем 5% резерв
                logger.logger.warning(
                    f"Insufficient balance for {symbol}: "
                    f"Required ${total_required:.2f} > Available ${self.current_balance * 0.95:.2f}"
                )
                return

            # Проверяем с риск-менеджером
            can_open, reason = self.risk_manager.can_open_position(
                symbol=symbol,
                proposed_size=position_cost
            )

            if not can_open:
                logger.logger.warning(f"Risk manager rejected position for {symbol}: {reason}")
                return

            # Применяем слиппаж
            slippage = self.slippage_bps / 10000
            if signal['action'] == 'BUY':
                entry_price = current_price * (1 + slippage)
            else:
                entry_price = current_price * (1 - slippage)

            # Пересчитываем с учетом слиппажа
            position_cost = position_size * entry_price
            fee = position_cost * self.taker_fee

            # Создаём позицию
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

            # Добавляем позицию
            self.positions[symbol] = position
            self.risk_manager.add_position(position)

            # ИСПРАВЛЕНИЕ: Правильный учет баланса для всех типов позиций
            # Для paper trading вычитаем стоимость позиции из баланса
            self.current_balance -= (position_cost + fee)

            # Обновляем время последней сделки
            self.last_trade_time[symbol] = datetime.utcnow()

            # Логируем сделку
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
                f"Reasons: {', '.join(signal['reasons'])}"
            )

            # Логируем торговое событие
            logger.log_trade(
                action="open",
                symbol=symbol,
                side=signal['action'],
                price=entry_price,
                quantity=position_size,
                pnl=0
            )

        except Exception as e:
            logger.logger.error(f"Failed to execute trade for {symbol}: {e}", exc_info=True)

    async def _update_positions(self):
        """Обновление открытых позиций"""
        for symbol, position in list(self.positions.items()):
            try:
                # Получаем текущую цену
                current_price = self.indicators.get(symbol, {}).get('price')

                # Если нет цены в индикаторах, запрашиваем напрямую
                if not current_price or abs(current_price - position.current_price) < 0.01:
                    try:
                        ticker = await self.binance_client.get_ticker(symbol=symbol)
                        current_price = float(ticker['lastPrice'])
                        logger.logger.debug(f"Fetched fresh price for {symbol}: ${current_price:.2f}")
                    except Exception as e:
                        logger.logger.debug(f"Failed to fetch price for {symbol}: {e}")
                        continue

                # Обновляем P&L позиции
                await self._update_position_pnl(symbol, current_price)

                # Обновляем позицию в риск-менеджере
                actions = self.risk_manager.update_position(
                    position_id=position.id,
                    current_price=current_price,
                    atr=self.indicators.get(symbol, {}).get('atr')
                )

                # Закрываем позицию если нужно
                if actions and actions.get('action') == 'close':
                    await self._close_position(symbol, current_price, actions.get('reason', 'risk_management'))

            except Exception as e:
                logger.logger.error(f"Failed to update position for {symbol}: {e}")

    async def _close_position(self, symbol: str, close_price: float, reason: str):
        """Закрытие позиции с корректным расчетом баланса"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        # Рассчитываем P&L
        if position.side == 'BUY':
            # Для длинной позиции: прибыль = (цена закрытия - цена входа) * количество
            gross_pnl = (close_price - position.entry_price) * position.quantity
            # Возвращаем выручку от продажи
            proceeds = position.quantity * close_price
        else:  # SELL
            # Для короткой позиции: прибыль = (цена входа - цена закрытия) * количество
            gross_pnl = (position.entry_price - close_price) * position.quantity
            # Возвращаем изначальную стоимость позиции + прибыль
            proceeds = position.quantity * position.entry_price + gross_pnl

        # Вычитаем комиссию за закрытие
        fee = position.quantity * close_price * self.taker_fee
        net_pnl = gross_pnl - fee

        # ИСПРАВЛЕНИЕ: Правильное обновление баланса
        # Возвращаем средства на баланс с учетом P&L и комиссии
        if position.side == 'BUY':
            # Для лонга: возвращаем выручку минус комиссия
            self.current_balance += (proceeds - fee)
        else:  # SELL
            # Для шорта: возвращаем изначальные средства + чистую прибыль/убыток
            initial_cost = position.quantity * position.entry_price
            self.current_balance += initial_cost + net_pnl

        # Закрываем в риск-менеджере
        self.risk_manager.close_position(position.id, close_price, reason)

        # Удаляем позицию
        del self.positions[symbol]

        # Сохраняем в историю
        self.trade_history.append({
            'symbol': symbol,
            'side': position.side,
            'entry_price': position.entry_price,
            'close_price': close_price,
            'quantity': position.quantity,
            'pnl': net_pnl,
            'reason': reason,
            'timestamp': datetime.utcnow()
        })

        # Логируем закрытие
        pnl_emoji = "💰" if net_pnl > 0 else "💸"
        logger.logger.info(
            f"{pnl_emoji} POSITION CLOSED | Symbol: {symbol} | "
            f"Side: {position.side} | Close: ${close_price:.2f} | "
            f"PnL: ${net_pnl:+.2f} | Reason: {reason}"
        )

        # Логируем торговое событие
        logger.log_trade(
            action="close",
            symbol=symbol,
            side=position.side,
            price=close_price,
            quantity=position.quantity,
            pnl=net_pnl
        )

    async def _update_position_pnl(self, symbol: str, current_price: float):
        """Обновление P&L позиции"""
        if symbol in self.positions:
            position = self.positions[symbol]
            position.current_price = current_price

            if position.side == 'BUY':
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            else:  # SELL
                position.unrealized_pnl = (position.entry_price - current_price) * position.quantity

    async def _check_risk_limits(self):
        """Проверка лимитов риска"""
        try:
            risk_metrics = self.risk_manager.get_risk_metrics()

            # Экстренное закрытие при критическом риске
            if risk_metrics.risk_level.value == "critical":
                logger.logger.warning("⚠️ CRITICAL RISK LEVEL - closing all positions")
                for symbol in list(self.positions.keys()):
                    current_price = self.indicators.get(symbol, {}).get('price', self.positions[symbol].entry_price)
                    await self._close_position(symbol, current_price, "risk_limit_exceeded")
        except Exception as e:
            logger.logger.error(f"Error checking risk limits: {e}")

    async def _log_status(self):
        """Логирование статуса бота"""
        try:
            # Расчет полного капитала (equity)
            positions_value = sum(
                pos.quantity * (pos.current_price if pos.current_price > 0 else pos.entry_price)
                for pos in self.positions.values()
            )

            total_unrealized_pnl = sum(
                getattr(pos, 'unrealized_pnl', 0)
                for pos in self.positions.values()
            )

            # Equity = свободный баланс + стоимость позиций
            equity = self.current_balance + positions_value

            # Реальный P&L = разница от начального капитала
            total_pnl = equity - self.initial_balance

            status = {
                "balance": round(self.current_balance, 2),
                "positions_value": round(positions_value, 2),
                "equity": round(equity, 2),
                "positions": len(self.positions),
                "trades": len(self.trade_history),
                "realized_pnl": round(sum(t.get('pnl', 0) for t in self.trade_history), 2),
                "unrealized_pnl": round(total_unrealized_pnl, 2),
                "total_pnl": round(total_pnl, 2),
                "return_percent": round((total_pnl / self.initial_balance) * 100, 2) if self.initial_balance > 0 else 0
            }

            logger.logger.info(
                f"📈 Bot Status | Mode: {'AGGRESSIVE' if self.aggressive_mode else 'NORMAL'} | "
                f"Equity: ${status['equity']:.2f} | "
                f"Free Balance: ${status['balance']:.2f} | "
                f"In Positions: ${status['positions_value']:.2f} | "
                f"Positions: {status['positions']} | Trades: {status['trades']} | "
                f"Realized PnL: ${status['realized_pnl']:.2f} | "
                f"Unrealized PnL: ${status['unrealized_pnl']:.2f} | "
                f"Total PnL: ${status['total_pnl']:.2f} | "
                f"Return: {status['return_percent']:.2f}%"
            )

            # Логируем открытые позиции
            for symbol, position in self.positions.items():
                if hasattr(position, 'unrealized_pnl') and hasattr(position, 'current_price'):
                    pnl_percent = (position.unrealized_pnl / (position.entry_price * position.quantity)) * 100
                    pnl_emoji = "📈" if position.unrealized_pnl > 0 else "📉"
                    logger.logger.info(
                        f"{pnl_emoji} Position: {symbol} | Side: {position.side} | "
                        f"Entry: ${position.entry_price:.2f} | Current: ${position.current_price:.2f} | "
                        f"PnL: ${position.unrealized_pnl:.2f} ({pnl_percent:.2f}%)"
                    )

            # Логируем метрики производительности
            if self.performance_metrics['total_analyses'] > 0:
                logger.logger.debug(
                    f"Performance Metrics | "
                    f"Analyses: {self.performance_metrics['total_analyses']} | "
                    f"Signals: {self.performance_metrics['total_signals']} | "
                    f"Executed: {self.performance_metrics['executed_signals']} | "
                    f"Hold: {self.performance_metrics['hold_signals']} | "
                    f"Skipped (confidence): {self.performance_metrics['skipped_low_confidence']} | "
                    f"Skipped (data): {self.performance_metrics['skipped_insufficient_data']} | "
                    f"Skipped (position): {self.performance_metrics['skipped_position_exists']} | "
                    f"Skipped (time): {self.performance_metrics['skipped_time_limit']}"
                )

        except Exception as e:
            logger.logger.error(f"Error logging status: {e}")

    async def stop(self):
        """Остановка бота"""
        logger.logger.info("Stopping Advanced Paper Trading Bot")
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
            final_pnl = sum(t.get('pnl', 0) for t in self.trade_history)
            final_balance = self.current_balance
            final_return = ((final_balance - self.initial_balance) / self.initial_balance) * 100

            win_trades = [t for t in self.trade_history if t['pnl'] > 0]
            lose_trades = [t for t in self.trade_history if t['pnl'] < 0]

            if self.trade_history:
                logger.logger.info(
                    f"📊 Final Report | "
                    f"Final Balance: ${final_balance:.2f} | "
                    f"Total PnL: ${final_pnl:.2f} | "
                    f"Return: {final_return:.2f}% | "
                    f"Total Trades: {len(self.trade_history)} | "
                    f"Wins: {len(win_trades)} | "
                    f"Losses: {len(lose_trades)} | "
                    f"Win Rate: {len(win_trades) / len(self.trade_history) * 100:.2f}%"
                )
            else:
                logger.logger.info("No trades executed")

            # Финальные метрики производительности
            logger.logger.info(
                f"Session Performance | "
                f"Total Analyses: {self.performance_metrics['total_analyses']} | "
                f"Total Signals: {self.performance_metrics['total_signals']} | "
                f"Execution Rate: {self.performance_metrics['executed_signals'] / max(1, self.performance_metrics['total_signals']) * 100:.1f}%"
            )

        except Exception as e:
            logger.logger.error(f"Error stopping bot: {e}")