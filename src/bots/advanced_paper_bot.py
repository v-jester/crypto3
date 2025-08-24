"""
Продвинутый Paper Trading Bot с реальными данными Binance и ML стратегиями
"""
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from binance import AsyncClient
from binance.streams import BinanceSocketManager
from binance.enums import *

from src.monitoring.logger import logger
from src.config.settings import settings
from src.data.storage.redis_client import redis_client
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

        # Флаги состояния
        self.running = False
        self.connected = False

    async def initialize(self):
        """Полная инициализация бота"""
        try:
            logger.logger.info("Initializing Advanced Paper Trading Bot")

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
                f"Initial balance: {self.initial_balance} | "
                f"Symbols: {settings.trading.SYMBOLS[:3]} | "
                f"Timeframe: {settings.trading.PRIMARY_TIMEFRAME}"
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

            self.connected = True

        except Exception as e:
            logger.logger.error(f"Binance connection failed: {e}")
            # Продолжаем работу в оффлайн режиме
            self.connected = False

    async def _load_historical_data(self):
        """Загрузка исторических данных для анализа"""
        logger.logger.info("Loading historical data...")

        for symbol in settings.trading.SYMBOLS[:3]:  # Начнём с первых 3 символов
            try:
                # Загружаем данные для основного таймфрейма
                df = await self.data_collector.fetch_historical_data(
                    symbol=symbol,
                    interval=settings.trading.PRIMARY_TIMEFRAME,
                    days_back=7,
                    limit=500
                )

                if not df.empty:
                    self.market_data[symbol] = df

                    # Сохраняем последние индикаторы
                    self.indicators[symbol] = {
                        'rsi': df['rsi'].iloc[-1] if 'rsi' in df.columns else 50,
                        'macd': df['macd'].iloc[-1] if 'macd' in df.columns else 0,
                        'bb_position': df['bb_percent'].iloc[-1] if 'bb_percent' in df.columns else 0.5,
                        'volume_ratio': df['volume_ratio'].iloc[-1] if 'volume_ratio' in df.columns else 1,
                        'atr': df['atr'].iloc[-1] if 'atr' in df.columns else 0,
                        'price': df['close'].iloc[-1] if 'close' in df.columns else 0
                    }

                    # Исправленное логирование
                    logger.logger.info(
                        f"Loaded {len(df)} candles for {symbol} | "
                        f"Price: {self.indicators[symbol]['price']:.2f} | "
                        f"RSI: {self.indicators[symbol]['rsi']:.1f}"
                    )
                else:
                    logger.logger.warning(f"No data loaded for {symbol}")

            except Exception as e:
                logger.logger.warning(f"Failed to load data for {symbol}: {e}")

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
                intervals=[settings.trading.PRIMARY_TIMEFRAME],
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
            asyncio.create_task(self._poll_market_data())

    async def _poll_market_data(self):
        """Альтернативный метод обновления данных через REST API"""
        logger.logger.info("Starting REST API polling for market data")

        poll_interval = 30  # секунд
        last_update = {}

        while self.running:
            try:
                for symbol in settings.trading.SYMBOLS[:3]:
                    try:
                        # Проверяем, нужно ли обновление
                        now = datetime.utcnow()
                        if symbol in last_update:
                            time_since_update = (now - last_update[symbol]).total_seconds()
                            if time_since_update < poll_interval:
                                continue

                        # Получаем текущую цену
                        ticker = await self.binance_client.get_ticker(symbol=symbol)
                        price = float(ticker['lastPrice'])

                        # Проверяем изменение цены
                        old_price = self.indicators.get(symbol, {}).get('price', 0)
                        if old_price > 0:
                            price_change = abs((price - old_price) / old_price)
                        else:
                            price_change = 1

                        # Обновляем данные если цена изменилась более чем на 0.1% или прошло достаточно времени
                        if price_change > 0.001 or symbol not in last_update:
                            # Загружаем свежие данные
                            df = await self.data_collector.fetch_historical_data(
                                symbol=symbol,
                                interval=settings.trading.PRIMARY_TIMEFRAME,
                                days_back=1,
                                limit=100
                            )

                            if not df.empty:
                                self.market_data[symbol] = df

                                # Обновляем индикаторы
                                self.indicators[symbol] = {
                                    'rsi': df['rsi'].iloc[-1] if 'rsi' in df.columns else 50,
                                    'macd': df['macd'].iloc[-1] if 'macd' in df.columns else 0,
                                    'bb_position': df['bb_percent'].iloc[-1] if 'bb_percent' in df.columns else 0.5,
                                    'volume_ratio': df['volume_ratio'].iloc[-1] if 'volume_ratio' in df.columns else 1,
                                    'atr': df['atr'].iloc[-1] if 'atr' in df.columns else 0,
                                    'price': price
                                }

                                last_update[symbol] = now

                                logger.logger.debug(
                                    f"Updated {symbol} via REST API | "
                                    f"Price: {price:.2f} | "
                                    f"RSI: {self.indicators[symbol]['rsi']:.1f} | "
                                    f"BB: {self.indicators[symbol]['bb_position']:.2f}"
                                )

                    except Exception as e:
                        logger.logger.error(f"Failed to poll data for {symbol}: {e}")

                # Ждём перед следующей итерацией
                await asyncio.sleep(5)  # Проверяем каждые 5 секунд, но обновляем реже

            except asyncio.CancelledError:
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
            # Получаем свежие данные
            df = await self.data_collector.fetch_historical_data(
                symbol=symbol,
                interval=settings.trading.PRIMARY_TIMEFRAME,
                days_back=1,
                limit=100
            )

            if not df.empty:
                self.market_data[symbol] = df

                # Обновляем индикаторы
                self.indicators[symbol] = {
                    'rsi': df['rsi'].iloc[-1] if 'rsi' in df.columns else 50,
                    'macd': df['macd'].iloc[-1] if 'macd' in df.columns else 0,
                    'bb_position': df['bb_percent'].iloc[-1] if 'bb_percent' in df.columns else 0.5,
                    'volume_ratio': df['volume_ratio'].iloc[-1] if 'volume_ratio' in df.columns else 1,
                    'atr': df['atr'].iloc[-1] if 'atr' in df.columns else 0,
                    'price': df['close'].iloc[-1]
                }

        except Exception as e:
            logger.logger.error(f"Failed to update market data for {symbol}: {e}")

    async def run(self):
        """Основной торговый цикл"""
        self.running = True
        logger.logger.info("Starting Advanced Paper Trading Bot main loop")

        analysis_counter = 0

        try:
            while self.running:
                # Анализируем рынок каждые 30 секунд
                if analysis_counter % 6 == 0:  # 30 секунд (5 сек * 6)
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
            logger.logger.info("Trading loop cancelled")
        except Exception as e:
            logger.logger.error(f"Error in trading loop: {e}")
            raise
        finally:
            logger.logger.info("Advanced Paper Trading Bot main loop stopped")

    async def _analyze_and_trade(self):
        """Анализ рынка и принятие торговых решений"""
        for symbol in settings.trading.SYMBOLS[:3]:
            try:
                # Пропускаем если нет данных
                if symbol not in self.market_data:
                    logger.logger.debug(f"No market data for {symbol}")
                    continue

                df = self.market_data[symbol]
                if df.empty or len(df) < 50:
                    logger.logger.debug(f"Insufficient data for {symbol}: {len(df)} candles")
                    continue

                # Получаем текущие индикаторы
                indicators = self.indicators.get(symbol, {})
                current_price = indicators.get('price', df['close'].iloc[-1] if not df.empty else 0)

                if current_price == 0:
                    logger.logger.warning(f"Zero price for {symbol}, skipping")
                    continue

                # Генерируем торговые сигналы
                signal = await self._generate_trading_signal(symbol, df, indicators)

                if signal['action'] != 'HOLD':
                    logger.logger.info(
                        f"Signal generated for {symbol} | "
                        f"Action: {signal['action']} | "
                        f"Confidence: {signal['confidence']:.2f} | "
                        f"Reasons: {', '.join(signal['reasons'])}"
                    )

                    # Проверяем уверенность сигнала
                    if signal['confidence'] >= 0.55:  # Снижен порог для большей активности
                        await self._execute_trade(symbol, signal, current_price)
                    else:
                        logger.logger.debug(
                            f"Signal confidence too low for {symbol}: {signal['confidence']:.2f}"
                        )

            except Exception as e:
                logger.logger.error(f"Analysis failed for {symbol}: {e}", exc_info=True)

    async def _generate_trading_signal(self, symbol: str, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Генерация торгового сигнала на основе множественных стратегий"""
        signal = {
            'action': 'HOLD',
            'confidence': 0.0,
            'reasons': [],
            'stop_loss': None,
            'take_profit': None
        }

        buy_signals = 0
        sell_signals = 0

        # 1. RSI стратегия
        rsi = indicators.get('rsi', 50)
        if rsi < 35:  # Oversold
            buy_signals += 1.2
            signal['reasons'].append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 65:  # Overbought
            sell_signals += 1.2
            signal['reasons'].append(f"RSI overbought ({rsi:.1f})")
        elif rsi < 45:  # Mild oversold
            buy_signals += 0.5
            signal['reasons'].append(f"RSI low ({rsi:.1f})")
        elif rsi > 55:  # Mild overbought
            sell_signals += 0.5
            signal['reasons'].append(f"RSI high ({rsi:.1f})")

        # 2. MACD стратегия
        macd = indicators.get('macd', 0)
        if 'macd_signal' in df.columns and len(df) > 0:
            macd_signal_val = df['macd_signal'].iloc[-1]
            macd_diff = macd - macd_signal_val

            if macd_diff > 0:
                buy_signals += 0.8
                signal['reasons'].append(f"MACD bullish ({macd_diff:.4f})")
            elif macd_diff < 0:
                sell_signals += 0.8
                signal['reasons'].append(f"MACD bearish ({macd_diff:.4f})")

        # 3. Bollinger Bands стратегия
        bb_position = indicators.get('bb_position', 0.5)
        if bb_position < 0.2:  # Near lower band
            buy_signals += 1.0
            signal['reasons'].append(f"Price near lower BB ({bb_position:.2f})")
        elif bb_position > 0.8:  # Near upper band
            sell_signals += 1.0
            signal['reasons'].append(f"Price near upper BB ({bb_position:.2f})")
        elif bb_position < 0.35:  # Below middle
            buy_signals += 0.3
            signal['reasons'].append(f"Price below BB middle ({bb_position:.2f})")
        elif bb_position > 0.65:  # Above middle
            sell_signals += 0.3
            signal['reasons'].append(f"Price above BB middle ({bb_position:.2f})")

        # 4. Volume подтверждение
        volume_ratio = indicators.get('volume_ratio', 1)
        if volume_ratio > 1.5:  # High volume
            if buy_signals > sell_signals:
                buy_signals += 0.5
                signal['reasons'].append(f"High volume confirmation ({volume_ratio:.1f}x)")
            elif sell_signals > buy_signals:
                sell_signals += 0.5
                signal['reasons'].append(f"High volume confirmation ({volume_ratio:.1f}x)")

        # 5. ML предсказание (если доступно)
        try:
            if hasattr(ml_engine, 'models_trained') and ml_engine.models_trained:
                ml_prediction = await ml_engine.predict(df, return_proba=True)
                if ml_prediction and 'prediction_label' in ml_prediction:
                    if ml_prediction['prediction_label'] == 'UP' and ml_prediction.get('confidence', 0) > 0.55:
                        buy_signals += 1.5
                        signal['reasons'].append(f"ML: UP ({ml_prediction['confidence']:.1%})")
                    elif ml_prediction['prediction_label'] == 'DOWN' and ml_prediction.get('confidence', 0) > 0.55:
                        sell_signals += 1.5
                        signal['reasons'].append(f"ML: DOWN ({ml_prediction['confidence']:.1%})")
        except Exception as e:
            logger.logger.debug(f"ML prediction skipped: {e}")

        # Определяем финальное действие
        min_signals_required = 1.2  # Минимум сигналов для действия

        if buy_signals >= min_signals_required and buy_signals > sell_signals:
            signal['action'] = 'BUY'
            signal['confidence'] = min(buy_signals / 4.0, 0.85)

            # Рассчитываем стоп-лосс и тейк-профит
            atr = indicators.get('atr', df['close'].iloc[-1] * 0.02)
            signal['stop_loss'] = df['close'].iloc[-1] - (atr * 1.5)
            signal['take_profit'] = df['close'].iloc[-1] + (atr * 2.5)

        elif sell_signals >= min_signals_required and sell_signals > buy_signals:
            signal['action'] = 'SELL'
            signal['confidence'] = min(sell_signals / 4.0, 0.85)

            atr = indicators.get('atr', df['close'].iloc[-1] * 0.02)
            signal['stop_loss'] = df['close'].iloc[-1] + (atr * 1.5)
            signal['take_profit'] = df['close'].iloc[-1] - (atr * 2.5)

        # Логируем детальную информацию о сигналах
        if buy_signals > 0 or sell_signals > 0:
            logger.logger.debug(
                f"Signal analysis for {symbol} | "
                f"Buy signals: {buy_signals:.2f} | "
                f"Sell signals: {sell_signals:.2f} | "
                f"Action: {signal['action']} | "
                f"Confidence: {signal['confidence']:.2f}"
            )

        return signal

    async def _execute_trade(self, symbol: str, signal: Dict, current_price: float):
        """Выполнение торговой операции"""
        try:
            # Проверяем, есть ли уже позиция по этому символу
            if symbol in self.positions:
                logger.logger.info(f"Position already exists for {symbol}")
                return

            # Рассчитываем размер позиции
            position_size = self.risk_manager.position_sizer.calculate_position_size(
                entry_price=current_price,
                stop_loss_price=signal['stop_loss'],
                account_balance=self.current_balance
            )

            if position_size <= 0:
                logger.logger.warning(f"Invalid position size for {symbol}: {position_size}")
                return

            # Проверяем возможность открытия позиции
            can_open, reason = self.risk_manager.can_open_position(
                symbol=symbol,
                proposed_size=position_size * current_price
            )

            if not can_open:
                logger.logger.warning(f"Cannot open position for {symbol}: {reason}")
                return

            # Применяем слиппаж
            slippage = self.slippage_bps / 10000
            if signal['action'] == 'BUY':
                entry_price = current_price * (1 + slippage)
            else:
                entry_price = current_price * (1 - slippage)

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

            # Вычитаем комиссию
            fee = position_size * entry_price * self.taker_fee
            self.current_balance -= fee

            # Логируем сделку - исправленный вызов
            # Не используем logger.log_trade с именованными параметрами
            logger.logger.info(
                f"Trade executed | Action: open | Symbol: {symbol} | "
                f"Side: {signal['action']} | Price: {entry_price:.2f} | "
                f"Quantity: {position_size:.4f} | Order ID: {position.id}"
            )

            logger.logger.info(
                f"Position opened | Symbol: {symbol} | Side: {signal['action']} | "
                f"Entry: {entry_price:.2f} | Size: {position_size:.4f} | "
                f"SL: {signal['stop_loss']:.2f} | TP: {signal['take_profit']:.2f} | "
                f"Reasons: {', '.join(signal['reasons'])}"
            )

        except Exception as e:
            logger.logger.error(f"Failed to execute trade for {symbol}: {e}", exc_info=True)

    async def _update_positions(self):
        """Обновление открытых позиций"""
        for symbol, position in list(self.positions.items()):
            try:
                # Получаем текущую цену
                current_price = self.indicators.get(symbol, {}).get('price')
                if not current_price:
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
        """Закрытие позиции"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        # Рассчитываем P&L
        if position.side == 'BUY':
            pnl = (close_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - close_price) * position.quantity

        # Вычитаем комиссию
        fee = position.quantity * close_price * self.taker_fee
        pnl -= fee

        # Обновляем баланс
        self.current_balance += (position.quantity * close_price - fee)

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
            'pnl': pnl,
            'reason': reason,
            'timestamp': datetime.utcnow()
        })

        # Логируем закрытие - исправленный вызов
        logger.logger.info(
            f"Trade executed | Action: close | Symbol: {symbol} | "
            f"Side: {position.side} | Price: {close_price:.2f} | "
            f"Quantity: {position.quantity:.4f} | PnL: {pnl:.2f} | "
            f"Order ID: {position.id}"
        )

        logger.logger.info(
            f"Position closed | Symbol: {symbol} | "
            f"PnL: {pnl:.2f} | Reason: {reason} | "
            f"Close price: {close_price:.2f}"
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

            # Экстренное закрытие при критическом риске
            if risk_metrics.risk_level.value == "critical":
                logger.logger.warning("CRITICAL RISK LEVEL - closing all positions")
                for symbol in list(self.positions.keys()):
                    current_price = self.indicators.get(symbol, {}).get('price', self.positions[symbol].entry_price)
                    await self._close_position(symbol, current_price, "risk_limit_exceeded")
        except Exception as e:
            logger.logger.error(f"Error checking risk limits: {e}")

    async def _log_status(self):
        """Логирование статуса бота"""
        try:
            total_pnl = sum(
                getattr(pos, 'unrealized_pnl', 0)
                for pos in self.positions.values()
            )

            status = {
                "balance": round(self.current_balance, 2),
                "positions": len(self.positions),
                "trades": len(self.trade_history),
                "realized_pnl": round(self.current_balance - self.initial_balance, 2),
                "unrealized_pnl": round(total_pnl, 2),
                "total_pnl": round((self.current_balance - self.initial_balance) + total_pnl, 2),
                "return_percent": round(
                    ((self.current_balance + total_pnl - self.initial_balance) / self.initial_balance) * 100,
                    2
                )
            }

            logger.logger.info(
                f"Advanced Bot Status | Balance: ${status['balance']:.2f} | "
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
                    logger.logger.info(
                        f"Position: {symbol} | Side: {position.side} | "
                        f"Entry: {position.entry_price:.2f} | Current: {position.current_price:.2f} | "
                        f"PnL: ${position.unrealized_pnl:.2f} ({pnl_percent:.2f}%)"
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
            final_pnl = self.current_balance - self.initial_balance
            final_return = (final_pnl / self.initial_balance) * 100

            win_trades = [t for t in self.trade_history if t['pnl'] > 0]
            lose_trades = [t for t in self.trade_history if t['pnl'] < 0]

            logger.logger.info(
                f"Final Report | "
                f"Final Balance: ${self.current_balance:.2f} | "
                f"Total PnL: ${final_pnl:.2f} | "
                f"Return: {final_return:.2f}% | "
                f"Total Trades: {len(self.trade_history)} | "
                f"Wins: {len(win_trades)} | "
                f"Losses: {len(lose_trades)} | "
                f"Win Rate: {len(win_trades) / len(self.trade_history) * 100:.2f}%" if self.trade_history else "No trades executed"
            )
        except Exception as e:
            logger.logger.error(f"Error stopping bot: {e}")