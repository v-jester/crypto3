# src/risk/risk_manager.py
"""
Комплексная система управления рисками для криптовалютного трейдинга
Включает Van Tharp position sizing, динамические стоп-лоссы, контроль корреляций
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
    """Уровни риска"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Position:
    """Класс для хранения информации о позиции"""
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
    """Метрики риска портфеля"""
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
    """Калькулятор размера позиции по методу Van Tharp"""

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
        Расчёт размера позиции по Van Tharp

        Args:
            entry_price: Цена входа
            stop_loss_price: Цена стоп-лосса
            account_balance: Текущий баланс (если изменился)

        Returns:
            Размер позиции в базовой валюте
        """
        if account_balance:
            self.account_balance = account_balance

        # Риск в долларах
        risk_amount = self.account_balance * self.risk_per_trade

        # Риск на единицу актива
        price_risk = abs(entry_price - stop_loss_price)

        if price_risk == 0:
            logger.logger.warning("Price risk is zero, returning zero position size")
            return 0

        # Размер позиции
        position_size = risk_amount / price_risk

        # Ограничение максимальным размером
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
        Расчёт размера позиции по критерию Келли

        Args:
            win_rate: Процент выигрышных сделок
            avg_win: Средний выигрыш
            avg_loss: Средний проигрыш
            kelly_fraction: Доля от полного Келли (для консервативности)

        Returns:
            Оптимальный процент капитала для риска
        """
        if avg_loss == 0:
            return self.risk_per_trade

        # Формула Келли: f = (p * b - q) / b
        # где p = вероятность выигрыша, q = вероятность проигрыша, b = отношение выигрыша к проигрышу
        b = avg_win / abs(avg_loss)
        p = win_rate
        q = 1 - win_rate

        kelly_percentage = (p * b - q) / b

        # Применяем дробное Келли для консервативности
        optimal_risk = kelly_percentage * kelly_fraction

        # Ограничиваем максимальным риском
        return min(max(optimal_risk, 0.001), 0.05)  # От 0.1% до 5%


class StopLossManager:
    """Менеджер динамических стоп-лоссов"""

    def __init__(self, atr_multiplier: float = 2.5):
        self.atr_multiplier = atr_multiplier

    def calculate_atr_stop(
            self,
            current_price: float,
            atr: float,
            is_long: bool = True
    ) -> float:
        """
        Расчёт стоп-лосса на основе ATR

        Args:
            current_price: Текущая цена
            atr: Average True Range
            is_long: Длинная позиция или короткая

        Returns:
            Цена стоп-лосса
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
        Расчёт трейлинг стоп-лосса

        Args:
            entry_price: Цена входа
            current_price: Текущая цена
            highest_price: Максимальная цена с момента входа
            trail_percent: Процент трейлинга
            is_long: Длинная позиция или короткая

        Returns:
            Цена трейлинг стопа
        """
        if is_long:
            # Для лонга следим за максимумом
            trail_distance = highest_price * trail_percent
            trailing_stop = highest_price - trail_distance

            # Стоп не может быть ниже цены входа минус начальный риск
            min_stop = entry_price * (1 - trail_percent)
            return max(trailing_stop, min_stop)
        else:
            # Для шорта следим за минимумом
            trail_distance = current_price * trail_percent
            trailing_stop = current_price + trail_distance

            # Стоп не может быть выше цены входа плюс начальный риск
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
        Перенос стопа в безубыток

        Args:
            entry_price: Цена входа
            current_price: Текущая цена
            breakeven_trigger: Коэффициент для активации (1.01 = +1%)
            is_long: Длинная позиция или короткая

        Returns:
            Новая цена стопа или None
        """
        if is_long:
            if current_price >= entry_price * breakeven_trigger:
                # Переносим стоп чуть выше точки входа
                return entry_price * 1.001
        else:
            if current_price <= entry_price / breakeven_trigger:
                return entry_price * 0.999

        return None


class RiskManager:
    """Главный менеджер рисков"""

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
        Проверка возможности открытия позиции

        Args:
            symbol: Символ для новой позиции
            proposed_size: Предлагаемый размер позиции
            correlation_matrix: Матрица корреляций между активами

        Returns:
            (можно_открыть, причина_отказа)
        """
        # Проверка количества позиций
        if len(self.positions) >= self.max_positions:
            return False, f"Maximum positions ({self.max_positions}) reached"

        # Проверка дневного лимита потерь
        daily_loss_percent = self._calculate_daily_loss_percent()
        if daily_loss_percent >= self.max_daily_loss:
            return False, f"Daily loss limit reached ({daily_loss_percent:.2%})"

        # Проверка максимальной просадки
        current_drawdown = self._calculate_drawdown()
        if current_drawdown >= self.max_drawdown:
            return False, f"Maximum drawdown reached ({current_drawdown:.2%})"

        # Проверка корреляции с существующими позициями
        if correlation_matrix is not None and symbol in correlation_matrix.columns:
            for pos_symbol in self.positions.keys():
                if pos_symbol in correlation_matrix.columns:
                    correlation = correlation_matrix.loc[symbol, pos_symbol]
                    if abs(correlation) > self.max_correlation:
                        return False, f"High correlation with {pos_symbol} ({correlation:.2f})"

        # Проверка достаточности капитала
        total_exposure = self._calculate_total_exposure() + proposed_size
        if total_exposure > self.current_capital * 0.95:
            return False, "Insufficient capital for position"

        return True, "OK"

    def add_position(self, position: Position) -> bool:
        """
        Добавление новой позиции

        Args:
            position: Объект позиции

        Returns:
            Успешность добавления
        """
        can_open, reason = self.can_open_position(
            position.symbol,
            position.quantity * position.entry_price
        )

        if not can_open:
            logger.logger.warning(f"Cannot open position: {reason}")
            return False

        self.positions[position.id] = position

        # Обновляем метрики
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
        Обновление позиции и проверка стоп-лоссов

        Args:
            position_id: ID позиции
            current_price: Текущая цена
            atr: ATR для динамического стопа

        Returns:
            Словарь с действиями
        """
        if position_id not in self.positions:
            return {"action": "none", "reason": "position not found"}

        position = self.positions[position_id]
        position.current_price = current_price

        # Расчёт P&L
        if position.side == "BUY":
            position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            price_change = (current_price - position.entry_price) / position.entry_price
        else:
            position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
            price_change = (position.entry_price - current_price) / position.entry_price

        # Обновляем метрики
        metrics_collector.update_position_metrics(
            position.symbol,
            position.quantity * current_price,
            position.unrealized_pnl
        )

        actions = {"action": "none"}

        # Проверка стоп-лосса
        if position.side == "BUY" and current_price <= position.stop_loss:
            actions = {"action": "close", "reason": "stop_loss_hit", "price": current_price}
        elif position.side == "SELL" and current_price >= position.stop_loss:
            actions = {"action": "close", "reason": "stop_loss_hit", "price": current_price}

        # Проверка тейк-профита
        if position.take_profit:
            if position.side == "BUY" and current_price >= position.take_profit:
                actions = {"action": "close", "reason": "take_profit_hit", "price": current_price}
            elif position.side == "SELL" and current_price <= position.take_profit:
                actions = {"action": "close", "reason": "take_profit_hit", "price": current_price}

        # Обновление трейлинг стопа если в прибыли
        if price_change > 0.02 and atr:  # В прибыли более 2%
            new_stop = self.stop_loss_manager.calculate_trailing_stop(
                position.entry_price,
                current_price,
                current_price,  # В реальности нужно отслеживать максимум
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
        Закрытие позиции

        Args:
            position_id: ID позиции
            close_price: Цена закрытия
            reason: Причина закрытия

        Returns:
            Информация о закрытой позиции
        """
        if position_id not in self.positions:
            return {"error": "position not found"}

        position = self.positions[position_id]

        # Расчёт финального P&L
        if position.side == "BUY":
            realized_pnl = (close_price - position.entry_price) * position.quantity
        else:
            realized_pnl = (position.entry_price - close_price) * position.quantity

        # Обновление капитала
        self.current_capital += realized_pnl

        # Сохранение в историю
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

        # Удаление позиции
        del self.positions[position_id]

        # Обновление метрик
        metrics_collector.open_positions.set(len(self.positions))
        metrics_collector.pnl_realized.set(
            sum(t["realized_pnl"] for t in self.trade_history)
        )

        # Логирование
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
        """Получение текущих метрик риска"""
        total_exposure = self._calculate_total_exposure()
        current_drawdown = self._calculate_drawdown()
        daily_loss = self._calculate_daily_loss_percent()
        var_95 = self._calculate_var()
        sharpe = self._calculate_sharpe_ratio()
        correlation_risk = self._calculate_correlation_risk()

        # Определение уровня риска
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

        # Обновление метрик Prometheus
        metrics_collector.update_portfolio_metrics({
            'current_drawdown': current_drawdown * 100,
            'sharpe_ratio': sharpe
        })
        metrics_collector.exposure.set(total_exposure)
        metrics_collector.var_95.set(var_95)
        metrics_collector.daily_loss.set(daily_loss * 100)

        # Логирование риск-событий
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
        """Расчёт общей экспозиции"""
        return sum(
            pos.quantity * pos.current_price if pos.current_price > 0
            else pos.quantity * pos.entry_price
            for pos in self.positions.values()
        )

    def _calculate_drawdown(self) -> float:
        """Расчёт текущей просадки"""
        if self.current_capital >= self.peak_balance:
            self.peak_balance = self.current_capital
            return 0

        return (self.peak_balance - self.current_capital) / self.peak_balance

    def _calculate_max_drawdown(self) -> float:
        """Расчёт максимальной просадки из истории"""
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
        """Расчёт дневных потерь в процентах"""
        # Сброс счётчика если новый день
        current_date = datetime.utcnow().date()
        if current_date != self.last_reset_date:
            self.daily_start_balance = self.current_capital
            self.last_reset_date = current_date
            self.daily_pnl = []

        daily_loss = (self.daily_start_balance - self.current_capital) / self.daily_start_balance
        return max(0, daily_loss)

    def _calculate_var(self, confidence: float = 0.95, window: int = 100) -> float:
        """Расчёт Value at Risk"""
        if len(self.trade_history) < 20:
            return self.current_capital * 0.05  # По умолчанию 5%

        # Используем последние N сделок
        recent_trades = self.trade_history[-window:] if len(self.trade_history) > window else self.trade_history
        returns = [t["return_pct"] / 100 for t in recent_trades]

        if not returns:
            return self.current_capital * 0.05

        # Сортируем и находим квантиль
        sorted_returns = sorted(returns)
        var_index = int((1 - confidence) * len(sorted_returns))
        var_return = abs(sorted_returns[var_index]) if var_index < len(sorted_returns) else abs(sorted_returns[0])

        return self.current_capital * var_return

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Расчёт коэффициента Шарпа"""
        if len(self.trade_history) < 10:
            return 0

        returns = [t["return_pct"] / 100 for t in self.trade_history]

        if not returns or np.std(returns) == 0:
            return 0

        # Годовая доходность (предполагаем ~250 торговых дней)
        avg_return = np.mean(returns)
        std_return = np.std(returns)

        # Количество сделок в день (приблизительно)
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
        """Расчёт риска корреляции портфеля"""
        if len(self.positions) < 2:
            return 0

        # Упрощённый расчёт - в реальности нужна матрица корреляций
        # Возвращаем процент позиций в одном направлении
        long_positions = sum(1 for p in self.positions.values() if p.side == "BUY")
        short_positions = len(self.positions) - long_positions

        concentration = max(long_positions, short_positions) / len(self.positions)
        return concentration

    def emergency_close_all(self, reason: str = "emergency") -> List[Dict[str, Any]]:
        """Экстренное закрытие всех позиций"""
        logger.logger.warning(f"Emergency close all positions: {reason}")

        closed_positions = []
        for position_id, position in list(self.positions.items()):
            result = self.close_position(
                position_id,
                position.current_price if position.current_price > 0 else position.entry_price,
                reason=reason
            )
            closed_positions.append(result)

        # Записываем риск-событие
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
        Расчёт матрицы корреляций между символами

        Args:
            symbols: Список символов
            price_data: DataFrame с ценами (колонки = символы)

        Returns:
            Матрица корреляций
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
        Оптимизация весов портфеля (упрощённая версия Марковица)

        Args:
            expected_returns: Ожидаемые доходности по символам
            covariance_matrix: Матрица ковариаций
            risk_tolerance: Толерантность к риску (0=минимальный риск, 1=максимальная доходность)

        Returns:
            Оптимальные веса портфеля
        """
        symbols = list(expected_returns.keys())
        n = len(symbols)

        # Равные веса как базовый случай
        equal_weights = {symbol: 1 / n for symbol in symbols}

        if n < 2:
            return equal_weights

        try:
            # Упрощённая оптимизация (в реальности нужен scipy.optimize)
            # Используем правило: больший вес для активов с лучшим соотношением доходность/риск
            sharpe_ratios = {}

            for symbol in symbols:
                expected_return = expected_returns[symbol]
                volatility = np.sqrt(covariance_matrix.loc[symbol, symbol])

                if volatility > 0:
                    sharpe_ratios[symbol] = expected_return / volatility
                else:
                    sharpe_ratios[symbol] = 0

            # Нормализуем веса на основе Sharpe ratios
            total_sharpe = sum(max(0, sr) for sr in sharpe_ratios.values())

            if total_sharpe > 0:
                weights = {
                    symbol: max(0, sharpe_ratios[symbol]) / total_sharpe
                    for symbol in symbols
                }
            else:
                weights = equal_weights

            # Применяем ограничения (макс 40% на актив)
            max_weight = 0.4
            for symbol in weights:
                weights[symbol] = min(weights[symbol], max_weight)

            # Перенормализация
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}

            return weights

        except Exception as e:
            logger.logger.error(f"Portfolio optimization failed: {e}")
            return equal_weights

    def get_portfolio_stats(self) -> Dict[str, Any]:
        """Получение статистики портфеля"""
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


# Глобальный экземпляр
risk_manager = None


def init_risk_manager(initial_capital: float) -> RiskManager:
    """Инициализация менеджера рисков"""
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