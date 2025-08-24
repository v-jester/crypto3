# src/risk/risk_manager.py
"""
РљРѕРјРїР»РµРєСЃРЅР°СЏ СЃРёСЃС‚РµРјР° СѓРїСЂР°РІР»РµРЅРёСЏ СЂРёСЃРєР°РјРё РґР»СЏ РєСЂРёРїС‚РѕРІР°Р»СЋС‚РЅРѕРіРѕ С‚СЂРµР№РґРёРЅРіР°
Р’РєР»СЋС‡Р°РµС‚ Van Tharp position sizing, РґРёРЅР°РјРёС‡РµСЃРєРёРµ СЃС‚РѕРї-Р»РѕСЃСЃС‹, РєРѕРЅС‚СЂРѕР»СЊ РєРѕСЂСЂРµР»СЏС†РёР№
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from src.monitoring.logger import logger
from src.config.settings import settings
from src.monitoring.metrics import metrics_collector


class RiskLevel(Enum):
    """РЈСЂРѕРІРЅРё СЂРёСЃРєР°"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Position:
    """РљР»Р°СЃСЃ РґР»СЏ С…СЂР°РЅРµРЅРёСЏ РёРЅС„РѕСЂРјР°С†РёРё Рѕ РїРѕР·РёС†РёРё"""
    id: str
    symbol: str
    side: str  # BUY/SELL
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: Optional[float]
    entry_time: datetime
    current_price: float = 0
    unrealized_pnl: float = 0
    risk_amount: float = 0


@dataclass
class RiskMetrics:
    """РњРµС‚СЂРёРєРё СЂРёСЃРєР° РїРѕСЂС‚С„РµР»СЏ"""
    total_exposure: float
    current_drawdown: float
    max_drawdown: float
    daily_loss: float
    var_95: float
    sharpe_ratio: float
    risk_level: RiskLevel
    open_positions: int
    correlation_risk: float


class PositionSizer:
    """РљР°Р»СЊРєСѓР»СЏС‚РѕСЂ СЂР°Р·РјРµСЂР° РїРѕР·РёС†РёРё РїРѕ РјРµС‚РѕРґСѓ Van Tharp"""

    def __init__(
            self,
            account_balance: float,
            risk_per_trade: float = 0.01,
            max_position_percent: float = 0.25
    ):
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade
        self.max_position_percent = max_position_percent

    def calculate_position_size(
            self,
            entry_price: float,
            stop_loss_price: float,
            account_balance: Optional[float] = None
    ) -> float:
        """
        Р Р°СЃС‡С‘С‚ СЂР°Р·РјРµСЂР° РїРѕР·РёС†РёРё РїРѕ Van Tharp

        Args:
            entry_price: Р¦РµРЅР° РІС…РѕРґР°
            stop_loss_price: Р¦РµРЅР° СЃС‚РѕРї-Р»РѕСЃСЃР°
            account_balance: РўРµРєСѓС‰РёР№ Р±Р°Р»Р°РЅСЃ (РµСЃР»Рё РёР·РјРµРЅРёР»СЃСЏ)

        Returns:
            Р Р°Р·РјРµСЂ РїРѕР·РёС†РёРё РІ Р±Р°Р·РѕРІРѕР№ РІР°Р»СЋС‚Рµ
        """
        if account_balance:
            self.account_balance = account_balance

        # Р РёСЃРє РІ РґРѕР»Р»Р°СЂР°С…
        risk_amount = self.account_balance * self.risk_per_trade

        # Р РёСЃРє РЅР° РµРґРёРЅРёС†Сѓ Р°РєС‚РёРІР°
        price_risk = abs(entry_price - stop_loss_price)

        if price_risk == 0:
            logger.logger.warning("Price risk is zero, returning zero position size")
            return 0

        # Р Р°Р·РјРµСЂ РїРѕР·РёС†РёРё
        position_size = risk_amount / price_risk

        # РћРіСЂР°РЅРёС‡РµРЅРёРµ РјР°РєСЃРёРјР°Р»СЊРЅС‹Рј СЂР°Р·РјРµСЂРѕРј
        max_position = self.account_balance * self.max_position_percent / entry_price
        position_size = min(position_size, max_position)

        logger.logger.debug(
            f"Position size calculated",
            risk_amount=risk_amount,
            price_risk=price_risk,
            position_size=position_size
        )

        return position_size

    def calculate_with_kelly_criterion(
            self,
            win_rate: float,
            avg_win: float,
            avg_loss: float,
            kelly_fraction: float = 0.25
    ) -> float:
        """
        Р Р°СЃС‡С‘С‚ СЂР°Р·РјРµСЂР° РїРѕР·РёС†РёРё РїРѕ РєСЂРёС‚РµСЂРёСЋ РљРµР»Р»Рё

        Args:
            win_rate: РџСЂРѕС†РµРЅС‚ РІС‹РёРіСЂС‹С€РЅС‹С… СЃРґРµР»РѕРє
            avg_win: РЎСЂРµРґРЅРёР№ РІС‹РёРіСЂС‹С€
            avg_loss: РЎСЂРµРґРЅРёР№ РїСЂРѕРёРіСЂС‹С€
            kelly_fraction: Р”РѕР»СЏ РѕС‚ РїРѕР»РЅРѕРіРѕ РљРµР»Р»Рё (РґР»СЏ РєРѕРЅСЃРµСЂРІР°С‚РёРІРЅРѕСЃС‚Рё)

        Returns:
            РћРїС‚РёРјР°Р»СЊРЅС‹Р№ РїСЂРѕС†РµРЅС‚ РєР°РїРёС‚Р°Р»Р° РґР»СЏ СЂРёСЃРєР°
        """
        if avg_loss == 0:
            return self.risk_per_trade

        # Р¤РѕСЂРјСѓР»Р° РљРµР»Р»Рё: f = (p * b - q) / b
        # РіРґРµ p = РІРµСЂРѕСЏС‚РЅРѕСЃС‚СЊ РІС‹РёРіСЂС‹С€Р°, q = РІРµСЂРѕСЏС‚РЅРѕСЃС‚СЊ РїСЂРѕРёРіСЂС‹С€Р°, b = РѕС‚РЅРѕС€РµРЅРёРµ РІС‹РёРіСЂС‹С€Р° Рє РїСЂРѕРёРіСЂС‹С€Сѓ
        b = avg_win / abs(avg_loss)
        p = win_rate
        q = 1 - win_rate

        kelly_percentage = (p * b - q) / b

        # РџСЂРёРјРµРЅСЏРµРј РґСЂРѕР±РЅРѕРµ РљРµР»Р»Рё РґР»СЏ РєРѕРЅСЃРµСЂРІР°С‚РёРІРЅРѕСЃС‚Рё
        optimal_risk = kelly_percentage * kelly_fraction

        # РћРіСЂР°РЅРёС‡РёРІР°РµРј РјР°РєСЃРёРјР°Р»СЊРЅС‹Рј СЂРёСЃРєРѕРј
        return min(max(optimal_risk, 0.001), 0.05)  # РћС‚ 0.1% РґРѕ 5%


class StopLossManager:
    """РњРµРЅРµРґР¶РµСЂ РґРёРЅР°РјРёС‡РµСЃРєРёС… СЃС‚РѕРї-Р»РѕСЃСЃРѕРІ"""

    def __init__(self, atr_multiplier: float = 2.5):
        self.atr_multiplier = atr_multiplier

    def calculate_atr_stop(
            self,
            current_price: float,
            atr: float,
            is_long: bool = True
    ) -> float:
        """
        Р Р°СЃС‡С‘С‚ СЃС‚РѕРї-Р»РѕСЃСЃР° РЅР° РѕСЃРЅРѕРІРµ ATR

        Args:
            current_price: РўРµРєСѓС‰Р°СЏ С†РµРЅР°
            atr: Average True Range
            is_long: Р”Р»РёРЅРЅР°СЏ РїРѕР·РёС†РёСЏ РёР»Рё РєРѕСЂРѕС‚РєР°СЏ

        Returns:
            Р¦РµРЅР° СЃС‚РѕРї-Р»РѕСЃСЃР°
        """
        stop_distance = atr * self.atr_multiplier

        if is_long:
            stop_loss = current_price - stop_distance
        else:
            stop_loss = current_price + stop_distance

        return stop_loss

    def calculate_trailing_stop(
            self,
            entry_price: float,
            current_price: float,
            highest_price: float,
            trail_percent: float = 0.02,
            is_long: bool = True
    ) -> float:
        """
        Р Р°СЃС‡С‘С‚ С‚СЂРµР№Р»РёРЅРі СЃС‚РѕРї-Р»РѕСЃСЃР°

        Args:
            entry_price: Р¦РµРЅР° РІС…РѕРґР°
            current_price: РўРµРєСѓС‰Р°СЏ С†РµРЅР°
            highest_price: РњР°РєСЃРёРјР°Р»СЊРЅР°СЏ С†РµРЅР° СЃ РјРѕРјРµРЅС‚Р° РІС…РѕРґР°
            trail_percent: РџСЂРѕС†РµРЅС‚ С‚СЂРµР№Р»РёРЅРіР°
            is_long: Р”Р»РёРЅРЅР°СЏ РїРѕР·РёС†РёСЏ РёР»Рё РєРѕСЂРѕС‚РєР°СЏ

        Returns:
            Р¦РµРЅР° С‚СЂРµР№Р»РёРЅРі СЃС‚РѕРїР°
        """
        if is_long:
            # Р”Р»СЏ Р»РѕРЅРіР° СЃР»РµРґРёРј Р·Р° РјР°РєСЃРёРјСѓРјРѕРј
            trail_distance = highest_price * trail_percent
            trailing_stop = highest_price - trail_distance

            # РЎС‚РѕРї РЅРµ РјРѕР¶РµС‚ Р±С‹С‚СЊ РЅРёР¶Рµ С†РµРЅС‹ РІС…РѕРґР° РјРёРЅСѓСЃ РЅР°С‡Р°Р»СЊРЅС‹Р№ СЂРёСЃРє
            min_stop = entry_price * (1 - trail_percent)
            return max(trailing_stop, min_stop)
        else:
            # Р”Р»СЏ С€РѕСЂС‚Р° СЃР»РµРґРёРј Р·Р° РјРёРЅРёРјСѓРјРѕРј
            trail_distance = current_price * trail_percent
            trailing_stop = current_price + trail_distance

            # РЎС‚РѕРї РЅРµ РјРѕР¶РµС‚ Р±С‹С‚СЊ РІС‹С€Рµ С†РµРЅС‹ РІС…РѕРґР° РїР»СЋСЃ РЅР°С‡Р°Р»СЊРЅС‹Р№ СЂРёСЃРє
            max_stop = entry_price * (1 + trail_percent)
            return min(trailing_stop, max_stop)

    def calculate_breakeven_stop(
            self,
            entry_price: float,
            current_price: float,
            breakeven_trigger: float = 1.01,
            is_long: bool = True
    ) -> Optional[float]:
        """
        РџРµСЂРµРЅРѕСЃ СЃС‚РѕРїР° РІ Р±РµР·СѓР±С‹С‚РѕРє

        Args:
            entry_price: Р¦РµРЅР° РІС…РѕРґР°
            current_price: РўРµРєСѓС‰Р°СЏ С†РµРЅР°
            breakeven_trigger: РљРѕСЌС„С„РёС†РёРµРЅС‚ РґР»СЏ Р°РєС‚РёРІР°С†РёРё (1.01 = +1%)
            is_long: Р”Р»РёРЅРЅР°СЏ РїРѕР·РёС†РёСЏ РёР»Рё РєРѕСЂРѕС‚РєР°СЏ

        Returns:
            РќРѕРІР°СЏ С†РµРЅР° СЃС‚РѕРїР° РёР»Рё None
        """
        if is_long:
            if current_price >= entry_price * breakeven_trigger:
                # РџРµСЂРµРЅРѕСЃРёРј СЃС‚РѕРї С‡СѓС‚СЊ РІС‹С€Рµ С‚РѕС‡РєРё РІС…РѕРґР°
                return entry_price * 1.001
        else:
            if current_price <= entry_price / breakeven_trigger:
                return entry_price * 0.999

        return None


class RiskManager:
    """Р“Р»Р°РІРЅС‹Р№ РјРµРЅРµРґР¶РµСЂ СЂРёСЃРєРѕРІ"""

    def __init__(
            self,
            initial_capital: float,
            max_drawdown: float = 0.20,
            max_daily_loss: float = 0.05,
            max_positions: int = 5,
            max_correlation: float = 0.7
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_drawdown = max_drawdown
        self.max_daily_loss = max_daily_loss
        self.max_positions = max_positions
        self.max_correlation = max_correlation

        self.position_sizer = PositionSizer(
            initial_capital,
            settings.trading.RISK_PER_TRADE,
            settings.trading.MAX_POSITION_SIZE_PERCENT
        )
        self.stop_loss_manager = StopLossManager(
            settings.trading.STOP_LOSS_ATR_MULTIPLIER
        )

        self.positions: Dict[str, Position] = {}
        self.trade_history = []
        self.daily_pnl = []
        self.peak_balance = initial_capital
        self.daily_start_balance = initial_capital
        self.last_reset_date = datetime.utcnow().date()

    def can_open_position(
            self,
            symbol: str,
            proposed_size: float,
            correlation_matrix: Optional[pd.DataFrame] = None
    ) -> Tuple[bool, str]:
        """
        РџСЂРѕРІРµСЂРєР° РІРѕР·РјРѕР¶РЅРѕСЃС‚Рё РѕС‚РєСЂС‹С‚РёСЏ РїРѕР·РёС†РёРё

        Args:
            symbol: РЎРёРјРІРѕР» РґР»СЏ РЅРѕРІРѕР№ РїРѕР·РёС†РёРё
            proposed_size: РџСЂРµРґР»Р°РіР°РµРјС‹Р№ СЂР°Р·РјРµСЂ РїРѕР·РёС†РёРё
            correlation_matrix: РњР°С‚СЂРёС†Р° РєРѕСЂСЂРµР»СЏС†РёР№ РјРµР¶РґСѓ Р°РєС‚РёРІР°РјРё

        Returns:
            (РјРѕР¶РЅРѕ_РѕС‚РєСЂС‹С‚СЊ, РїСЂРёС‡РёРЅР°_РѕС‚РєР°Р·Р°)
        """
        # РџСЂРѕРІРµСЂРєР° РєРѕР»РёС‡РµСЃС‚РІР° РїРѕР·РёС†РёР№
        if len(self.positions) >= self.max_positions:
            return False, f"Maximum positions ({self.max_positions}) reached"

        # РџСЂРѕРІРµСЂРєР° РґРЅРµРІРЅРѕРіРѕ Р»РёРјРёС‚Р° РїРѕС‚РµСЂСЊ
        daily_loss_percent = self._calculate_daily_loss_percent()
        if daily_loss_percent >= self.max_daily_loss:
            return False, f"Daily loss limit reached ({daily_loss_percent:.2%})"

        # РџСЂРѕРІРµСЂРєР° РјР°РєСЃРёРјР°Р»СЊРЅРѕР№ РїСЂРѕСЃР°РґРєРё
        current_drawdown = self._calculate_drawdown()
        if current_drawdown >= self.max_drawdown:
            return False, f"Maximum drawdown reached ({current_drawdown:.2%})"

        # РџСЂРѕРІРµСЂРєР° РєРѕСЂСЂРµР»СЏС†РёРё СЃ СЃСѓС‰РµСЃС‚РІСѓСЋС‰РёРјРё РїРѕР·РёС†РёСЏРјРё
        if correlation_matrix is not None and symbol in correlation_matrix.columns:
            for pos_symbol in self.positions.keys():
                if pos_symbol in correlation_matrix.columns:
                    correlation = correlation_matrix.loc[symbol, pos_symbol]
                    if abs(correlation) > self.max_correlation:
                        return False, f"High correlation with {pos_symbol} ({correlation:.2f})"

        # РџСЂРѕРІРµСЂРєР° РґРѕСЃС‚Р°С‚РѕС‡РЅРѕСЃС‚Рё РєР°РїРёС‚Р°Р»Р°
        total_exposure = self._calculate_total_exposure() + proposed_size
        if total_exposure > self.current_capital * 0.95:
            return False, "Insufficient capital for position"

        return True, "OK"

    def add_position(self, position: Position) -> bool:
        """
        Р”РѕР±Р°РІР»РµРЅРёРµ РЅРѕРІРѕР№ РїРѕР·РёС†РёРё

        Args:
            position: РћР±СЉРµРєС‚ РїРѕР·РёС†РёРё

        Returns:
            РЈСЃРїРµС€РЅРѕСЃС‚СЊ РґРѕР±Р°РІР»РµРЅРёСЏ
        """
        can_open, reason = self.can_open_position(
            position.symbol,
            position.quantity * position.entry_price
        )

        if not can_open:
            logger.logger.warning(f"Cannot open position: {reason}")
            return False

        self.positions[position.id] = position

        # РћР±РЅРѕРІР»СЏРµРј РјРµС‚СЂРёРєРё
        metrics_collector.open_positions.set(len(self.positions))
        metrics_collector.update_position_metrics(
            position.symbol,
            position.quantity * position.entry_price,
            0
        )

        logger.logger.info(
            f"Position added",
            position_id=position.id,
            symbol=position.symbol,
            side=position.side,
            size=position.quantity
        )

        return True

    def update_position(
            self,
            position_id: str,
            current_price: float,
            atr: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        РћР±РЅРѕРІР»РµРЅРёРµ РїРѕР·РёС†РёРё Рё РїСЂРѕРІРµСЂРєР° СЃС‚РѕРї-Р»РѕСЃСЃРѕРІ

        Args:
            position_id: ID РїРѕР·РёС†РёРё
            current_price: РўРµРєСѓС‰Р°СЏ С†РµРЅР°
            atr: ATR РґР»СЏ РґРёРЅР°РјРёС‡РµСЃРєРѕРіРѕ СЃС‚РѕРїР°

        Returns:
            РЎР»РѕРІР°СЂСЊ СЃ РґРµР№СЃС‚РІРёСЏРјРё
        """
        if position_id not in self.positions:
            return {"action": "none", "reason": "position not found"}

        position = self.positions[position_id]
        position.current_price = current_price

        # Р Р°СЃС‡С‘С‚ P&L
        if position.side == "BUY":
            position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            price_change = (current_price - position.entry_price) / position.entry_price
        else:
            position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
            price_change = (position.entry_price - current_price) / position.entry_price

        # РћР±РЅРѕРІР»СЏРµРј РјРµС‚СЂРёРєРё
        metrics_collector.update_position_metrics(
            position.symbol,
            position.quantity * current_price,
            position.unrealized_pnl
        )

        actions = {"action": "none"}

        # РџСЂРѕРІРµСЂРєР° СЃС‚РѕРї-Р»РѕСЃСЃР°
        if position.side == "BUY" and current_price <= position.stop_loss:
            actions = {"action": "close", "reason": "stop_loss_hit", "price": current_price}
        elif position.side == "SELL" and current_price >= position.stop_loss:
            actions = {"action": "close", "reason": "stop_loss_hit", "price": current_price}

        # РџСЂРѕРІРµСЂРєР° С‚РµР№Рє-РїСЂРѕС„РёС‚Р°
        if position.take_profit:
            if position.side == "BUY" and current_price >= position.take_profit:
                actions = {"action": "close", "reason": "take_profit_hit", "price": current_price}
            elif position.side == "SELL" and current_price <= position.take_profit:
                actions = {"action": "close", "reason": "take_profit_hit", "price": current_price}

        # РћР±РЅРѕРІР»РµРЅРёРµ С‚СЂРµР№Р»РёРЅРі СЃС‚РѕРїР° РµСЃР»Рё РІ РїСЂРёР±С‹Р»Рё
        if price_change > 0.02 and atr:  # Р’ РїСЂРёР±С‹Р»Рё Р±РѕР»РµРµ 2%
            new_stop = self.stop_loss_manager.calculate_trailing_stop(
                position.entry_price,
                current_price,
                current_price,  # Р’ СЂРµР°Р»СЊРЅРѕСЃС‚Рё РЅСѓР¶РЅРѕ РѕС‚СЃР»РµР¶РёРІР°С‚СЊ РјР°РєСЃРёРјСѓРј
                trail_percent=0.015,
                is_long=(position.side == "BUY")
            )

            if position.side == "BUY" and new_stop > position.stop_loss:
                position.stop_loss = new_stop
                actions["trailing_stop_updated"] = new_stop
            elif position.side == "SELL" and new_stop < position.stop_loss:
                position.stop_loss = new_stop
                actions["trailing_stop_updated"] = new_stop

        return actions

    def close_position(
            self,
            position_id: str,
            close_price: float,
            reason: str = "manual"
    ) -> Dict[str, Any]:
        """
        Р—Р°РєСЂС‹С‚РёРµ РїРѕР·РёС†РёРё

        Args:
            position_id: ID РїРѕР·РёС†РёРё
            close_price: Р¦РµРЅР° Р·Р°РєСЂС‹С‚РёСЏ
            reason: РџСЂРёС‡РёРЅР° Р·Р°РєСЂС‹С‚РёСЏ

        Returns:
            РРЅС„РѕСЂРјР°С†РёСЏ Рѕ Р·Р°РєСЂС‹С‚РѕР№ РїРѕР·РёС†РёРё
        """
        if position_id not in self.positions:
            return {"error": "position not found"}

        position = self.positions[position_id]

        # Р Р°СЃС‡С‘С‚ С„РёРЅР°Р»СЊРЅРѕРіРѕ P&L
        if position.side == "BUY":
            realized_pnl = (close_price - position.entry_price) * position.quantity
        else:
            realized_pnl = (position.entry_price - close_price) * position.quantity

        # РћР±РЅРѕРІР»РµРЅРёРµ РєР°РїРёС‚Р°Р»Р°
        self.current_capital += realized_pnl

        # РЎРѕС…СЂР°РЅРµРЅРёРµ РІ РёСЃС‚РѕСЂРёСЋ
        trade_result = {
            "position_id": position_id,
            "symbol": position.symbol,
            "side": position.side,
            "entry_price": position.entry_price,
            "close_price": close_price,
            "quantity": position.quantity,
            "realized_pnl": realized_pnl,
            "return_pct": realized_pnl / (position.quantity * position.entry_price) * 100,
            "duration": (datetime.utcnow() - position.entry_time).total_seconds() / 60,
            "reason": reason,
            "timestamp": datetime.utcnow()
        }

        self.trade_history.append(trade_result)

        # РЈРґР°Р»РµРЅРёРµ РїРѕР·РёС†РёРё
        del self.positions[position_id]

        # РћР±РЅРѕРІР»РµРЅРёРµ РјРµС‚СЂРёРє
        metrics_collector.open_positions.set(len(self.positions))
        metrics_collector.pnl_realized.set(
            sum(t["realized_pnl"] for t in self.trade_history)
        )

        # Р›РѕРіРёСЂРѕРІР°РЅРёРµ
        logger.log_trade(
            action="close",
            symbol=position.symbol,
            side=position.side,
            price=close_price,
            quantity=position.quantity,
            pnl=realized_pnl
        )

        return trade_result

    def get_risk_metrics(self) -> RiskMetrics:
        """РџРѕР»СѓС‡РµРЅРёРµ С‚РµРєСѓС‰РёС… РјРµС‚СЂРёРє СЂРёСЃРєР°"""
        total_exposure = self._calculate_total_exposure()
        current_drawdown = self._calculate_drawdown()
        daily_loss = self._calculate_daily_loss_percent()
        var_95 = self._calculate_var()
        sharpe = self._calculate_sharpe_ratio()
        correlation_risk = self._calculate_correlation_risk()

        # РћРїСЂРµРґРµР»РµРЅРёРµ СѓСЂРѕРІРЅСЏ СЂРёСЃРєР°
        if current_drawdown > 0.15 or daily_loss > 0.04:
            risk_level = RiskLevel.CRITICAL
        elif current_drawdown > 0.10 or daily_loss > 0.03:
            risk_level = RiskLevel.HIGH
        elif current_drawdown > 0.05 or daily_loss > 0.02:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        metrics = RiskMetrics(
            total_exposure=total_exposure,
            current_drawdown=current_drawdown,
            max_drawdown=self._calculate_max_drawdown(),
            daily_loss=daily_loss,
            var_95=var_95,
            sharpe_ratio=sharpe,
            risk_level=risk_level,
            open_positions=len(self.positions),
            correlation_risk=correlation_risk
        )

        # РћР±РЅРѕРІР»РµРЅРёРµ РјРµС‚СЂРёРє Prometheus
        metrics_collector.update_portfolio_metrics({
            'current_drawdown': current_drawdown * 100,
            'sharpe_ratio': sharpe
        })
        metrics_collector.exposure.set(total_exposure)
        metrics_collector.var_95.set(var_95)
        metrics_collector.daily_loss.set(daily_loss * 100)

        # Р›РѕРіРёСЂРѕРІР°РЅРёРµ СЂРёСЃРє-СЃРѕР±С‹С‚РёР№
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            logger.log_risk_event(
                event_type="high_risk_detected",
                severity=risk_level.value.upper(),
                message=f"Risk level: {risk_level.value}",
                current_drawdown=current_drawdown,
                current_exposure=total_exposure
            )

        return metrics

    def _calculate_total_exposure(self) -> float:
        """Р Р°СЃС‡С‘С‚ РѕР±С‰РµР№ СЌРєСЃРїРѕР·РёС†РёРё"""
        return sum(
            pos.quantity * pos.current_price if pos.current_price > 0
            else pos.quantity * pos.entry_price
            for pos in self.positions.values()
        )

    def _calculate_drawdown(self) -> float:
        """Р Р°СЃС‡С‘С‚ С‚РµРєСѓС‰РµР№ РїСЂРѕСЃР°РґРєРё"""
        if self.current_capital >= self.peak_balance:
            self.peak_balance = self.current_capital
            return 0

        return (self.peak_balance - self.current_capital) / self.peak_balance

    def _calculate_max_drawdown(self) -> float:
        """Р Р°СЃС‡С‘С‚ РјР°РєСЃРёРјР°Р»СЊРЅРѕР№ РїСЂРѕСЃР°РґРєРё РёР· РёСЃС‚РѕСЂРёРё"""
        if not self.trade_history:
            return 0

        cumulative_pnl = []
        running_total = 0

        for trade in self.trade_history:
            running_total += trade["realized_pnl"]
            cumulative_pnl.append(running_total)

        if not cumulative_pnl:
            return 0

        peak = cumulative_pnl[0]
        max_dd = 0

        for value in cumulative_pnl:
            if value > peak:
                peak = value
            drawdown = (peak - value) / (self.initial_capital + peak) if peak > 0 else 0
            max_dd = max(max_dd, drawdown)

        return max_dd

    def _calculate_daily_loss_percent(self) -> float:
        """Р Р°СЃС‡С‘С‚ РґРЅРµРІРЅС‹С… РїРѕС‚РµСЂСЊ РІ РїСЂРѕС†РµРЅС‚Р°С…"""
        # РЎР±СЂРѕСЃ СЃС‡С‘С‚С‡РёРєР° РµСЃР»Рё РЅРѕРІС‹Р№ РґРµРЅСЊ
        current_date = datetime.utcnow().date()
        if current_date != self.last_reset_date:
            self.daily_start_balance = self.current_capital
            self.last_reset_date = current_date
            self.daily_pnl = []

        daily_loss = (self.daily_start_balance - self.current_capital) / self.daily_start_balance
        return max(0, daily_loss)

    def _calculate_var(self, confidence: float = 0.95, window: int = 100) -> float:
        """Р Р°СЃС‡С‘С‚ Value at Risk"""
        if len(self.trade_history) < 20:
            return self.current_capital * 0.05  # РџРѕ СѓРјРѕР»С‡Р°РЅРёСЋ 5%

        # РСЃРїРѕР»СЊР·СѓРµРј РїРѕСЃР»РµРґРЅРёРµ N СЃРґРµР»РѕРє
        recent_trades = self.trade_history[-window:] if len(self.trade_history) > window else self.trade_history
        returns = [t["return_pct"] / 100 for t in recent_trades]

        if not returns:
            return self.current_capital * 0.05

        # РЎРѕСЂС‚РёСЂСѓРµРј Рё РЅР°С…РѕРґРёРј РєРІР°РЅС‚РёР»СЊ
        sorted_returns = sorted(returns)
        var_index = int((1 - confidence) * len(sorted_returns))
        var_return = abs(sorted_returns[var_index]) if var_index < len(sorted_returns) else abs(sorted_returns[0])

        return self.current_capital * var_return

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Р Р°СЃС‡С‘С‚ РєРѕСЌС„С„РёС†РёРµРЅС‚Р° РЁР°СЂРїР°"""
        if len(self.trade_history) < 10:
            return 0

        returns = [t["return_pct"] / 100 for t in self.trade_history]

        if not returns or np.std(returns) == 0:
            return 0

        # Р“РѕРґРѕРІР°СЏ РґРѕС…РѕРґРЅРѕСЃС‚СЊ (РїСЂРµРґРїРѕР»Р°РіР°РµРј ~250 С‚РѕСЂРіРѕРІС‹С… РґРЅРµР№)
        avg_return = np.mean(returns)
        std_return = np.std(returns)

        # РљРѕР»РёС‡РµСЃС‚РІРѕ СЃРґРµР»РѕРє РІ РґРµРЅСЊ (РїСЂРёР±Р»РёР·РёС‚РµР»СЊРЅРѕ)
        first_trade_time = self.trade_history[0].get("timestamp", datetime.utcnow())
        if isinstance(first_trade_time, str):
            first_trade_time = datetime.fromisoformat(first_trade_time)

        days_trading = max(1, (datetime.utcnow() - first_trade_time).days)
        trades_per_day = len(self.trade_history) / days_trading

        annualized_return = avg_return * trades_per_day * 250
        annualized_std = std_return * np.sqrt(trades_per_day * 250)

        if annualized_std == 0:
            return 0

        return (annualized_return - risk_free_rate) / annualized_std

    def _calculate_correlation_risk(self) -> float:
        """Р Р°СЃС‡С‘С‚ СЂРёСЃРєР° РєРѕСЂСЂРµР»СЏС†РёРё РїРѕСЂС‚С„РµР»СЏ"""
        if len(self.positions) < 2:
            return 0

        # РЈРїСЂРѕС‰С‘РЅРЅС‹Р№ СЂР°СЃС‡С‘С‚ - РІ СЂРµР°Р»СЊРЅРѕСЃС‚Рё РЅСѓР¶РЅР° РјР°С‚СЂРёС†Р° РєРѕСЂСЂРµР»СЏС†РёР№
        # Р’РѕР·РІСЂР°С‰Р°РµРј РїСЂРѕС†РµРЅС‚ РїРѕР·РёС†РёР№ РІ РѕРґРЅРѕРј РЅР°РїСЂР°РІР»РµРЅРёРё
        long_positions = sum(1 for p in self.positions.values() if p.side == "BUY")
        short_positions = len(self.positions) - long_positions

        concentration = max(long_positions, short_positions) / len(self.positions)
        return concentration

    def emergency_close_all(self, reason: str = "emergency") -> List[Dict[str, Any]]:
        """Р­РєСЃС‚СЂРµРЅРЅРѕРµ Р·Р°РєСЂС‹С‚РёРµ РІСЃРµС… РїРѕР·РёС†РёР№"""
        logger.logger.warning(f"Emergency close all positions: {reason}")

        closed_positions = []
        for position_id, position in list(self.positions.items()):
            result = self.close_position(
                position_id,
                position.current_price if position.current_price > 0 else position.entry_price,
                reason=reason
            )
            closed_positions.append(result)

        # Р—Р°РїРёСЃС‹РІР°РµРј СЂРёСЃРє-СЃРѕР±С‹С‚РёРµ
        logger.log_risk_event(
            event_type="emergency_close",
            severity="CRITICAL",
            message=f"All positions closed: {reason}",
            current_drawdown=self._calculate_drawdown(),
            current_exposure=0
        )

        metrics_collector.record_risk_event("emergency_close", "critical")

        return closed_positions

    def calculate_correlation_matrix(
            self,
            symbols: List[str],
            price_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Р Р°СЃС‡С‘С‚ РјР°С‚СЂРёС†С‹ РєРѕСЂСЂРµР»СЏС†РёР№ РјРµР¶РґСѓ СЃРёРјРІРѕР»Р°РјРё

        Args:
            symbols: РЎРїРёСЃРѕРє СЃРёРјРІРѕР»РѕРІ
            price_data: DataFrame СЃ С†РµРЅР°РјРё (РєРѕР»РѕРЅРєРё = СЃРёРјРІРѕР»С‹)

        Returns:
            РњР°С‚СЂРёС†Р° РєРѕСЂСЂРµР»СЏС†РёР№
        """
        returns = price_data.pct_change().dropna()
        correlation_matrix = returns.corr()

        return correlation_matrix

    def optimize_portfolio_weights(
            self,
            expected_returns: Dict[str, float],
            covariance_matrix: pd.DataFrame,
            risk_tolerance: float = 0.5
    ) -> Dict[str, float]:
        """
        РћРїС‚РёРјРёР·Р°С†РёСЏ РІРµСЃРѕРІ РїРѕСЂС‚С„РµР»СЏ (СѓРїСЂРѕС‰С‘РЅРЅР°СЏ РІРµСЂСЃРёСЏ РњР°СЂРєРѕРІРёС†Р°)

        Args:
            expected_returns: РћР¶РёРґР°РµРјС‹Рµ РґРѕС…РѕРґРЅРѕСЃС‚Рё РїРѕ СЃРёРјРІРѕР»Р°Рј
            covariance_matrix: РњР°С‚СЂРёС†Р° РєРѕРІР°СЂРёР°С†РёР№
            risk_tolerance: РўРѕР»РµСЂР°РЅС‚РЅРѕСЃС‚СЊ Рє СЂРёСЃРєСѓ (0=РјРёРЅРёРјР°Р»СЊРЅС‹Р№ СЂРёСЃРє, 1=РјР°РєСЃРёРјР°Р»СЊРЅР°СЏ РґРѕС…РѕРґРЅРѕСЃС‚СЊ)

        Returns:
            РћРїС‚РёРјР°Р»СЊРЅС‹Рµ РІРµСЃР° РїРѕСЂС‚С„РµР»СЏ
        """
        symbols = list(expected_returns.keys())
        n = len(symbols)

        # Р Р°РІРЅС‹Рµ РІРµСЃР° РєР°Рє Р±Р°Р·РѕРІС‹Р№ СЃР»СѓС‡Р°Р№
        equal_weights = {symbol: 1 / n for symbol in symbols}

        if n < 2:
            return equal_weights

        try:
            # РЈРїСЂРѕС‰С‘РЅРЅР°СЏ РѕРїС‚РёРјРёР·Р°С†РёСЏ (РІ СЂРµР°Р»СЊРЅРѕСЃС‚Рё РЅСѓР¶РµРЅ scipy.optimize)
            # РСЃРїРѕР»СЊР·СѓРµРј РїСЂР°РІРёР»Рѕ: Р±РѕР»СЊС€РёР№ РІРµСЃ РґР»СЏ Р°РєС‚РёРІРѕРІ СЃ Р»СѓС‡С€РёРј СЃРѕРѕС‚РЅРѕС€РµРЅРёРµРј РґРѕС…РѕРґРЅРѕСЃС‚СЊ/СЂРёСЃРє
            sharpe_ratios = {}

            for symbol in symbols:
                expected_return = expected_returns[symbol]
                volatility = np.sqrt(covariance_matrix.loc[symbol, symbol])

                if volatility > 0:
                    sharpe_ratios[symbol] = expected_return / volatility
                else:
                    sharpe_ratios[symbol] = 0

            # РќРѕСЂРјР°Р»РёР·СѓРµРј РІРµСЃР° РЅР° РѕСЃРЅРѕРІРµ Sharpe ratios
            total_sharpe = sum(max(0, sr) for sr in sharpe_ratios.values())

            if total_sharpe > 0:
                weights = {
                    symbol: max(0, sharpe_ratios[symbol]) / total_sharpe
                    for symbol in symbols
                }
            else:
                weights = equal_weights

            # РџСЂРёРјРµРЅСЏРµРј РѕРіСЂР°РЅРёС‡РµРЅРёСЏ (РјР°РєСЃ 40% РЅР° Р°РєС‚РёРІ)
            max_weight = 0.4
            for symbol in weights:
                weights[symbol] = min(weights[symbol], max_weight)

            # РџРµСЂРµРЅРѕСЂРјР°Р»РёР·Р°С†РёСЏ
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}

            return weights

        except Exception as e:
            logger.logger.error(f"Portfolio optimization failed: {e}")
            return equal_weights

    def get_portfolio_stats(self) -> Dict[str, Any]:
        """РџРѕР»СѓС‡РµРЅРёРµ СЃС‚Р°С‚РёСЃС‚РёРєРё РїРѕСЂС‚С„РµР»СЏ"""
        if not self.trade_history:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "current_capital": self.current_capital,
                "total_return": 0
            }

        winning_trades = [t for t in self.trade_history if t["realized_pnl"] > 0]
        losing_trades = [t for t in self.trade_history if t["realized_pnl"] < 0]

        total_wins = sum(t["realized_pnl"] for t in winning_trades)
        total_losses = abs(sum(t["realized_pnl"] for t in losing_trades))

        stats = {
            "total_trades": len(self.trade_history),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(self.trade_history) if self.trade_history else 0,
            "total_pnl": sum(t["realized_pnl"] for t in self.trade_history),
            "avg_win": total_wins / len(winning_trades) if winning_trades else 0,
            "avg_loss": total_losses / len(losing_trades) if losing_trades else 0,
            "profit_factor": total_wins / total_losses if total_losses > 0 else float('inf'),
            "current_capital": self.current_capital,
            "total_return": (self.current_capital - self.initial_capital) / self.initial_capital * 100,
            "max_drawdown": self._calculate_max_drawdown() * 100,
            "sharpe_ratio": self._calculate_sharpe_ratio(),
            "var_95": self._calculate_var()
        }

        return stats


# Р“Р»РѕР±Р°Р»СЊРЅС‹Р№ СЌРєР·РµРјРїР»СЏСЂ
risk_manager = None


def init_risk_manager(initial_capital: float) -> RiskManager:
    """РРЅРёС†РёР°Р»РёР·Р°С†РёСЏ РјРµРЅРµРґР¶РµСЂР° СЂРёСЃРєРѕРІ"""
    global risk_manager
    risk_manager = RiskManager(
        initial_capital=initial_capital,
        max_drawdown=settings.trading.MAX_DRAWDOWN_PERCENT,
        max_daily_loss=settings.trading.MAX_DAILY_LOSS_PERCENT,
        max_positions=settings.trading.MAX_POSITIONS,
        max_correlation=settings.trading.MAX_CORRELATION
    )
    logger.logger.info(f"Risk manager initialized with capital: ${initial_capital:,.2f}")
    return risk_manager
