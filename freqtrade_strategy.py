# freqtrade_strategy.py
"""
Стратегия Freqtrade на основе Advanced Paper Trading Bot
Для быстрого бэктестинга и оптимизации
"""
from freqtrade.strategy import IStrategy, informative
from pandas import DataFrame
import talib.abstract as ta
import pandas_ta as pta
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from freqtrade.persistence import Trade


class AdvancedCryptoStrategy(IStrategy):
    """
    Стратегия основанная на вашем Advanced Paper Trading Bot
    Адаптированная для Freqtrade
    """
    
    # Основные параметры стратегии
    INTERFACE_VERSION = 3
    
    # Минимальный ROI
    minimal_roi = {
        "0": 0.10,    # 10% через любое время
        "30": 0.05,   # 5% через 30 минут
        "60": 0.02,   # 2% через час
        "120": 0.01   # 1% через 2 часа
    }
    
    # Стоп-лосс
    stoploss = -0.05  # -5%
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True
    
    # Оптимальный таймфрейм
    timeframe = '5m'
    
    # Параметры для оптимизации (те же что в вашем боте)
    buy_rsi_low = 35
    buy_rsi_high = 45
    sell_rsi_low = 55
    sell_rsi_high = 65
    
    buy_bb_percent = 0.2
    sell_bb_percent = 0.8
    
    min_volume_ratio = 1.5
    min_signals_required = 1.2
    
    # Защита от покупки на пампах
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    
    # Лимит открытых сделок
    max_open_trades = 5
    
    # Параметры позиций
    stake_amount = "unlimited"
    stake_currency = "USDT"
    
    def informative_pairs(self):
        """Дополнительные пары для анализа корреляций"""
        return []
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Добавление всех индикаторов из вашего бота
        """
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=7)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        
        # MACD (Linda Raschke settings из вашего бота)
        macd = ta.MACD(dataframe, fastperiod=3, slowperiod=10, signalperiod=16)
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['macdsignal']
        dataframe['macd_hist'] = macd['macdhist']
        
        # Bollinger Bands
        bollinger = pta.bbands(dataframe['close'], length=20, std=2.0)
        dataframe['bb_lower'] = bollinger['BBL_20_2.0']
        dataframe['bb_middle'] = bollinger['BBM_20_2.0']
        dataframe['bb_upper'] = bollinger['BBU_20_2.0']
        dataframe['bb_width'] = dataframe['bb_upper'] - dataframe['bb_lower']
        dataframe['bb_percent'] = (
            (dataframe['close'] - dataframe['bb_lower']) / dataframe['bb_width']
        )
        
        # ATR для динамических стоп-лоссов
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_percent'] = (dataframe['atr'] / dataframe['close']) * 100
        
        # Volume indicators
        dataframe['volume_ma'] = dataframe['volume'].rolling(20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_ma']
        
        # EMA
        dataframe['ema_9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        
        # Momentum
        dataframe['momentum_1'] = dataframe['close'].pct_change(1)
        dataframe['momentum_5'] = dataframe['close'].pct_change(5)
        dataframe['momentum_10'] = dataframe['close'].pct_change(10)
        
        # Volatility для адаптивных порогов
        dataframe['volatility'] = dataframe['close'].pct_change().rolling(50).std()
        
        # MFI
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        
        # Stochastic
        stoch = ta.STOCH(dataframe)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Логика входа в позицию (из вашего _generate_trading_signal)
        """
        conditions = []
        
        # RSI oversold
        conditions.append(
            (dataframe['rsi'] < self.buy_rsi_low) |
            ((dataframe['rsi'] < self.buy_rsi_high) & (dataframe['rsi'].shift(1) > dataframe['rsi']))
        )
        
        # Bollinger Bands - цена у нижней границы
        conditions.append(dataframe['bb_percent'] < self.buy_bb_percent)
        
        # MACD bullish crossover
        conditions.append(
            (dataframe['macd'] > dataframe['macd_signal']) &
            (dataframe['macd'].shift(1) <= dataframe['macd_signal'].shift(1))
        )
        
        # Volume confirmation
        conditions.append(dataframe['volume_ratio'] > self.min_volume_ratio)
        
        # Momentum положительный
        conditions.append(dataframe['momentum_5'] > 0)
        
        # Дополнительные фильтры для качества сигнала
        conditions.append(dataframe['volume'] > 0)
        
        # Комбинируем условия - нужно минимум 2 сигнала
        buy_signal = np.zeros(len(dataframe), dtype=bool)
        
        for i, cond in enumerate(conditions):
            buy_signal = buy_signal | cond
        
        # Подсчитываем количество сигналов
        signal_count = sum([cond.astype(int) for cond in conditions])
        
        dataframe.loc[
            (signal_count >= self.min_signals_required) &
            (dataframe['volume'] > 0),
            'enter_long'
        ] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Логика выхода из позиции
        """
        conditions = []
        
        # RSI overbought
        conditions.append(
            (dataframe['rsi'] > self.sell_rsi_high) |
            ((dataframe['rsi'] > self.sell_rsi_low) & (dataframe['rsi'].shift(1) < dataframe['rsi']))
        )
        
        # Bollinger Bands - цена у верхней границы
        conditions.append(dataframe['bb_percent'] > self.sell_bb_percent)
        
        # MACD bearish crossover
        conditions.append(
            (dataframe['macd'] < dataframe['macd_signal']) &
            (dataframe['macd'].shift(1) >= dataframe['macd_signal'].shift(1))
        )
        
        # Momentum отрицательный
        conditions.append(dataframe['momentum_5'] < 0)
        
        # Комбинируем условия
        sell_signal = np.zeros(len(dataframe), dtype=bool)
        
        for cond in conditions:
            sell_signal = sell_signal | cond
        
        signal_count = sum([cond.astype(int) for cond in conditions])
        
        dataframe.loc[
            (signal_count >= self.min_signals_required) &
            (dataframe['volume'] > 0),
            'exit_long'
        ] = 1
        
        return dataframe
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        """
        Кастомная логика выхода - трейлинг стоп и частичная фиксация прибыли
        """
        # Частичная фиксация прибыли
        if current_profit > 0.02:  # 2% прибыли
            if trade.amount > trade.min_stake:
                return 'partial_take_profit'
        
        # Выход при резком движении против позиции
        if current_profit < -0.03:  # -3% убыток
            return 'stop_loss_emergency'
        
        return None
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: float, max_stake: float,
                           entry_tag: Optional[str], side: str, **kwargs) -> float:
        """
        Кастомный размер позиции (как в вашем position_sizer)
        """
        # 5% от капитала на позицию
        return proposed_stake * 0.05


# Конфигурация для Freqtrade
def get_freqtrade_config():
    """
    Генерация конфига для Freqtrade на основе вашего .env
    """
    return {
        "max_open_trades": 5,
        "stake_currency": "USDT",
        "stake_amount": "unlimited",
        "tradable_balance_ratio": 0.99,
        "fiat_display_currency": "USD",
        "dry_run": True,
        "dry_run_wallet": 50000,
        "cancel_open_orders_on_exit": False,
        "trading_mode": "futures",
        "margin_mode": "isolated",
        
        "unfilledtimeout": {
            "entry": 10,
            "exit": 10,
            "exit_timeout_count": 0,
            "unit": "minutes"
        },
        
        "entry_pricing": {
            "price_side": "same",
            "use_order_book": True,
            "order_book_top": 1,
            "price_last_balance": 0.0,
            "check_depth_of_market": {
                "enabled": False,
                "bids_to_ask_delta": 1
            }
        },
        
        "exit_pricing": {
            "price_side": "same",
            "use_order_book": True,
            "order_book_top": 1
        },
        
        "exchange": {
            "name": "binance",
            "key": "",  # Из вашего .env
            "secret": "",  # Из вашего .env
            "ccxt_config": {},
            "ccxt_async_config": {},
            "pair_whitelist": [
                "BTC/USDT:USDT",
                "ETH/USDT:USDT",
                "BNB/USDT:USDT",
                "SOL/USDT:USDT",
                "XRP/USDT:USDT",
                "DOGE/USDT:USDT",
                "ADA/USDT:USDT",
                "AVAX/USDT:USDT",
                "DOT/USDT:USDT"
            ],
            "pair_blacklist": []
        },
        
        "pairlists": [
            {
                "method": "StaticPairList",
                "number_assets": 20,
                "sort_key": "quoteVolume",
                "min_value": 0,
                "refresh_period": 1800
            }
        ],
        
        "telegram": {
            "enabled": False,
            "token": "",
            "chat_id": ""
        },
        
        "api_server": {
            "enabled": True,
            "listen_ip_address": "127.0.0.1",
            "listen_port": 8080,
            "verbosity": "error",
            "enable_openapi": False,
            "jwt_secret_key": "somethingrandom",
            "CORS_origins": [],
            "username": "freqtrader",
            "password": "freqtrader"
        },
        
        "bot_name": "AdvancedCryptoBot",
        "initial_state": "running",
        "force_entry_enable": False,
        "internals": {
            "process_throttle_secs": 5
        }
    }