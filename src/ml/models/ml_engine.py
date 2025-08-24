# src/bots/advanced_paper_bot.py
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
from src.ml.models.ml_engine import ml_engine  # Исправлено: ml_engine вместо ml


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

            # 4. Запуск WebSocket стримов
            await self._start_websocket_streams()

            logger.logger.info(
                "Advanced Paper Trading Bot initialized",
                initial_balance=self.initial_balance,
                symbols=settings.trading.SYMBOLS,
                timeframe=settings.trading.PRIMARY_TIMEFRAME
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

                self.market_data[symbol] = df

                # Сохраняем последние индикаторы
                if not df.empty:
                    self.indicators[symbol] = {
                        'rsi': df['rsi'].iloc[-1] if 'rsi' in df.columns else 50,
                        'macd': df['macd'].iloc[-1] if 'macd' in df.columns else 0,
                        'bb_position': df['bb_percent'].iloc[-1] if 'bb_percent' in df.columns else 0.5,
                        'volume_ratio': df['volume_ratio'].iloc[-1] if 'volume_ratio' in df.columns else 1,
                        'atr': df['atr'].iloc[-1] if 'atr' in df.columns else 0
                    }

                logger.logger.info(f"Loaded {len(df)} candles for {symbol}")

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

    async def _handle_kline_update(self, data: Dict):
        """Обработка обновлений свечей"""
        kline = data.get('k', {})
        if kline.get('x'):  # Свеча закрылась
            symbol = kline.get('s')
            logger.logger.debug(f"Closed candle for {symbol}: {kline.get('c')}")

            # Обновляем рыночные данные
            await self._update_market_data(symbol)

    async def _handle_ticker_update(self, data: Dict):
        """Обработка обновлений тикера"""
        symbol = data.get('s')
        price = float(data.get('c', 0))

        if symbol and price > 0:
            # Обновляем текущие цены для позиций
            if symbol in self.positions:
                await self._update_position_pnl(symbol, price)

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
                    continue

                df = self.market_data[symbol]
                if df.empty or len(df) < 50:
                    continue

                # Получаем текущие индикаторы
                indicators = self.indicators.get(symbol, {})
                current_price = indicators.get('price', df['close'].iloc[-1])

                # Генерируем торговые сигналы
                signal = await self._generate_trading_signal(symbol, df, indicators)

                if signal['action'] != 'HOLD':
                    logger.logger.info(
                        f"Signal generated for {symbol}",
                        action=signal['action'],
                        confidence=signal['confidence'],
                        reasons=signal['reasons']
                    )

                    # Выполняем сделку если сигнал достаточно сильный
                    if signal['confidence'] >= 0.65:
                        await self._execute_trade(symbol, signal, current_price)

            except Exception as e:
                logger.logger.error(f"Analysis failed for {symbol}: {e}")

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
        total_weight = 0

        # 1. RSI стратегия
        rsi = indicators.get('rsi', 50)
        if rsi < settings.signals.RSI_OVERSOLD:
            buy_signals += 1
            signal['reasons'].append(f"RSI oversold ({rsi:.1f})")
        elif rsi > settings.signals.RSI_OVERBOUGHT:
            sell_signals += 1
            signal['reasons'].append(f"RSI overbought ({rsi:.1f})")

        # 2. MACD стратегия
        macd = indicators.get('macd', 0)
        macd_signal = df['macd_signal'].iloc[-1] if 'macd_signal' in df.columns else 0
        if macd > macd_signal and macd < 0:
            buy_signals += 1
            signal['reasons'].append("MACD bullish crossover")
        elif macd < macd_signal and macd > 0:
            sell_signals += 1
            signal['reasons'].append("MACD bearish crossover")

        # 3. Bollinger Bands стратегия
        bb_position = indicators.get('bb_position', 0.5)
        if bb_position < 0.2:
            buy_signals += 1
            signal['reasons'].append(f"Price near lower BB ({bb_position:.2f})")
        elif bb_position > 0.8:
            sell_signals += 1
            signal['reasons'].append(f"Price near upper BB ({bb_position:.2f})")

        # 4. Volume подтверждение
        volume_ratio = indicators.get('volume_ratio', 1)
        if volume_ratio > settings.signals.MIN_VOLUME_RATIO:
            total_weight += 0.2
            signal['reasons'].append(f"High volume ({volume_ratio:.1f}x)")

        # 5. ML предсказание (если модель обучена)
        if hasattr(ml_engine, 'xgb_model') and ml_engine.xgb_model is not None:
            try:
                ml_prediction = await ml_engine.predict(df, return_proba=True)
                if ml_prediction['prediction_label'] == 'UP' and ml_prediction['confidence'] > 0.65:
                    buy_signals += 2  # ML сигнал имеет больший вес
                    signal['reasons'].append(f"ML: UP ({ml_prediction['confidence']:.1%})")
                elif ml_prediction['prediction_label'] == 'DOWN' and ml_prediction['confidence'] > 0.65:
                    sell_signals += 2
                    signal['reasons'].append(f"ML: DOWN ({ml_prediction['confidence']:.1%})")
            except:
                pass  # Игнорируем ошибки ML

        # Определяем финальное действие
        total_signals = buy_signals + sell_signals
        if total_signals > 0:
            if buy_signals > sell_signals and buy_signals >= 2:
                signal['action'] = 'BUY'
                signal['confidence'] = buy_signals / 5.0  # Нормализуем к [0, 1]

                # Рассчитываем стоп-лосс и тейк-профит
                atr = indicators.get('atr', df['close'].iloc[-1] * 0.02)
                signal['stop_loss'] = df['close'].iloc[-1] - (atr * settings.trading.STOP_LOSS_ATR_MULTIPLIER)
                signal['take_profit'] = df['close'].iloc[-1] + (
                        atr * settings.trading.STOP_LOSS_ATR_MULTIPLIER * settings.trading.TAKE_PROFIT_RR_RATIO)

            elif sell_signals > buy_signals and sell_signals >= 2:
                signal['action'] = 'SELL'
                signal['confidence'] = sell_signals / 5.0

                atr = indicators.get('atr', df['close'].iloc[-1] * 0.02)
                signal['stop_loss'] = df['close'].iloc[-1] + (atr * settings.trading.STOP_LOSS_ATR_MULTIPLIER)
                signal['take_profit'] = df['close'].iloc[-1] - (
                        atr * settings.trading.STOP_LOSS_ATR_MULTIPLIER * settings.trading.TAKE_PROFIT_RR_RATIO)

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

            # Логируем сделку
            logger.log_trade(
                action="open",
                symbol=symbol,
                side=signal['action'],
                price=entry_price,
                quantity=position_size,
                order_id=position.id
            )

            logger.logger.info(
                f"Position opened",
                symbol=symbol,
                side=signal['action'],
                entry_price=entry_price,
                size=position_size,
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profit'],
                reasons=signal['reasons']
            )

        except Exception as e:
            logger.logger.error(f"Failed to execute trade for {symbol}: {e}")

    async def _update_positions(self):
        """Обновление открытых позиций"""
        for symbol, position in list(self.positions.items()):
            try:
                # Получаем текущую цену
                current_price = self.indicators.get(symbol, {}).get('price')
                if not current_price:
                    continue

                # Обновляем позицию в риск-менеджере
                actions = self.risk_manager.update_position(
                    position_id=position.id,
                    current_price=current_price,
                    atr=self.indicators.get(symbol, {}).get('atr')
                )

                # Закрываем позицию если нужно
                if actions.get('action') == 'close':
                    await self._close_position(symbol, current_price, actions.get('reason', 'manual'))

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
        self.current_balance += pnl

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

        # Логируем
        logger.log_trade(
            action="close",
            symbol=symbol,
            side=position.side,
            price=close_price,
            quantity=position.quantity,
            pnl=pnl,
            order_id=position.id
        )

        logger.logger.info(
            f"Position closed",
            symbol=symbol,
            pnl=pnl,
            reason=reason
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
        risk_metrics = self.risk_manager.get_risk_metrics()

        # Экстренное закрытие при критическом риске
        if risk_metrics.risk_level.value == "critical":
            logger.logger.warning("CRITICAL RISK LEVEL - closing all positions")
            for symbol in list(self.positions.keys()):
                current_price = self.indicators.get(symbol, {}).get('price', self.positions[symbol].entry_price)
                await self._close_position(symbol, current_price, "risk_limit")

    async def _log_status(self):
        """Логирование статуса бота"""
        total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())

        status = {
            "balance": self.current_balance,
            "positions": len(self.positions),
            "trades": len(self.trade_history),
            "realized_pnl": self.current_balance - self.initial_balance,
            "unrealized_pnl": total_pnl,
            "total_pnl": (self.current_balance - self.initial_balance) + total_pnl,
            "return_percent": ((self.current_balance + total_pnl - self.initial_balance) / self.initial_balance) * 100
        }

        logger.logger.info("Advanced Bot Status", **status)

        # Логируем открытые позиции
        for symbol, position in self.positions.items():
            logger.logger.info(
                f"Position: {symbol}",
                side=position.side,
                entry=position.entry_price,
                current=position.current_price,
                pnl=position.unrealized_pnl,
                pnl_percent=(position.unrealized_pnl / (position.entry_price * position.quantity)) * 100
            )

    async def stop(self):
        """Остановка бота"""
        logger.logger.info("Stopping Advanced Paper Trading Bot")
        self.running = False

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
            "Final Report",
            final_balance=self.current_balance,
            total_pnl=final_pnl,
            total_return_percent=final_return,
            total_trades=len(self.trade_history),
            winning_trades=len(win_trades),
            losing_trades=len(lose_trades),
            win_rate=len(win_trades) / len(self.trade_history) * 100 if self.trade_history else 0
        )
