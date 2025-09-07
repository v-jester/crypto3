"""
Unit tests for Risk Manager
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
import pandas as pd
import numpy as np

from src.risk.risk_manager import (
    RiskManager, Position, RiskLevel,
    PositionSizer, StopLossManager
)


class TestPositionSizer:
    """Tests for PositionSizer class"""

    @pytest.fixture
    def position_sizer(self):
        return PositionSizer(
            account_balance=10000,
            risk_per_trade=0.02,
            max_position_percent=0.25
        )

    def test_calculate_position_size_basic(self, position_sizer):
        """Test basic position size calculation"""
        entry_price = 100
        stop_loss_price = 95

        size = position_sizer.calculate_position_size(
            entry_price=entry_price,
            stop_loss_price=stop_loss_price
        )

        # Risk = 10000 * 0.02 = 200
        # Price risk = 100 - 95 = 5
        # Expected size = 200 / 5 = 40
        assert size == 40

    def test_position_size_with_max_limit(self, position_sizer):
        """Test position size respects maximum limit"""
        entry_price = 100
        stop_loss_price = 99.5  # Very tight stop

        size = position_sizer.calculate_position_size(
            entry_price=entry_price,
            stop_loss_price=stop_loss_price
        )

        # Max position = 10000 * 0.25 / 100 = 25
        assert size == 25  # Limited by max position

    def test_kelly_criterion_calculation(self, position_sizer):
        """Test Kelly criterion position sizing"""
        optimal_risk = position_sizer.calculate_with_kelly_criterion(
            win_rate=0.6,
            avg_win=150,
            avg_loss=100,
            kelly_fraction=0.25
        )

        # Kelly formula: f = (p*b - q) / b
        # b = 150/100 = 1.5
        # f = (0.6*1.5 - 0.4) / 1.5 = 0.333...
        # With kelly_fraction = 0.25: 0.333 * 0.25 = 0.083
        assert 0.05 <= optimal_risk <= 0.1


class TestStopLossManager:
    """Tests for StopLossManager class"""

    @pytest.fixture
    def stop_manager(self):
        return StopLossManager(atr_multiplier=2.5)

    def test_atr_stop_calculation(self, stop_manager):
        """Test ATR-based stop loss calculation"""
        current_price = 100
        atr = 2

        # Long position
        stop_long = stop_manager.calculate_atr_stop(
            current_price=current_price,
            atr=atr,
            is_long=True
        )
        assert stop_long == 95  # 100 - (2 * 2.5)

        # Short position
        stop_short = stop_manager.calculate_atr_stop(
            current_price=current_price,
            atr=atr,
            is_long=False
        )
        assert stop_short == 105  # 100 + (2 * 2.5)

    def test_trailing_stop_calculation(self, stop_manager):
        """Test trailing stop calculation"""
        entry_price = 100
        current_price = 110
        highest_price = 112

        trailing_stop = stop_manager.calculate_trailing_stop(
            entry_price=entry_price,
            current_price=current_price,
            highest_price=highest_price,
            trail_percent=0.02,
            is_long=True
        )

        # Trail distance = 112 * 0.02 = 2.24
        # Trailing stop = 112 - 2.24 = 109.76
        assert trailing_stop == pytest.approx(109.76, rel=0.01)

    def test_breakeven_stop(self, stop_manager):
        """Test breakeven stop calculation"""
        entry_price = 100

        # Price hasn't reached trigger
        stop = stop_manager.calculate_breakeven_stop(
            entry_price=entry_price,
            current_price=100.5,
            breakeven_trigger=1.01,
            is_long=True
        )
        assert stop is None

        # Price reached trigger
        stop = stop_manager.calculate_breakeven_stop(
            entry_price=entry_price,
            current_price=101.5,
            breakeven_trigger=1.01,
            is_long=True
        )
        assert stop == pytest.approx(100.1, rel=0.01)


class TestRiskManager:
    """Tests for main RiskManager class"""

    @pytest.fixture
    def risk_manager(self):
        return RiskManager(
            initial_capital=10000,
            max_drawdown=0.20,
            max_daily_loss=0.05,
            max_positions=5,
            max_correlation=0.7
        )

    @pytest.fixture
    def sample_position(self):
        return Position(
            id="TEST001",
            symbol="BTCUSDT",
            side="BUY",
            entry_price=50000,
            quantity=0.001,
            stop_loss=48000,
            take_profit=52000,
            entry_time=datetime.utcnow(),
            current_price=50000,
            risk_amount=100
        )

    @pytest.mark.asyncio
    async def test_can_open_position_basic(self, risk_manager):
        """Test basic position opening checks"""
        can_open, reason = risk_manager.can_open_position(
            symbol="BTCUSDT",
            proposed_size=1000
        )
        assert can_open is True
        assert reason == "OK"

    @pytest.mark.asyncio
    async def test_max_positions_limit(self, risk_manager, sample_position):
        """Test maximum positions limit"""
        # Add maximum number of positions
        for i in range(5):
            position = Position(
                id=f"TEST{i:03d}",
                symbol=f"TEST{i}USDT",
                side="BUY",
                entry_price=100,
                quantity=1,
                stop_loss=95,
                take_profit=105,
                entry_time=datetime.utcnow(),
                current_price=100,
                risk_amount=50
            )
            risk_manager.add_position(position)

        can_open, reason = risk_manager.can_open_position(
            symbol="NEWUSDT",
            proposed_size=100
        )
        assert can_open is False
        assert "Maximum positions" in reason

    def test_daily_loss_limit(self, risk_manager):
        """Test daily loss limit check"""
        # Simulate daily loss
        risk_manager.daily_start_balance = 10000
        risk_manager.current_capital = 9400  # 6% loss

        can_open, reason = risk_manager.can_open_position(
            symbol="BTCUSDT",
            proposed_size=100
        )
        assert can_open is False
        assert "Daily loss limit" in reason

    def test_drawdown_limit(self, risk_manager):
        """Test maximum drawdown limit"""
        # Simulate drawdown
        risk_manager.peak_balance = 12000
        risk_manager.current_capital = 9500  # >20% drawdown

        can_open, reason = risk_manager.can_open_position(
            symbol="BTCUSDT",
            proposed_size=100
        )
        assert can_open is False
        assert "Maximum drawdown" in reason

    def test_correlation_check(self, risk_manager, sample_position):
        """Test correlation check between positions"""
        risk_manager.add_position(sample_position)

        # Create correlation matrix
        correlation_matrix = pd.DataFrame({
            'BTCUSDT': [1.0, 0.8],
            'ETHUSDT': [0.8, 1.0]
        }, index=['BTCUSDT', 'ETHUSDT'])

        can_open, reason = risk_manager.can_open_position(
            symbol="ETHUSDT",
            proposed_size=100,
            correlation_matrix=correlation_matrix
        )
        assert can_open is False
        assert "High correlation" in reason

    def test_position_update_and_close(self, risk_manager, sample_position):
        """Test position update and closing logic"""
        risk_manager.add_position(sample_position)

        # Update position - price moved up
        actions = risk_manager.update_position(
            position_id=sample_position.id,
            current_price=52500,  # Above take profit
            atr=1000
        )

        assert actions['action'] == 'close'
        assert actions['reason'] == 'take_profit_hit'

        # Close position
        result = risk_manager.close_position(
            position_id=sample_position.id,
            close_price=52500,
            reason='take_profit_hit'
        )

        assert result['realized_pnl'] == pytest.approx(2.5, rel=0.1)
        assert sample_position.id not in risk_manager.positions

    def test_risk_metrics_calculation(self, risk_manager, sample_position):
        """Test risk metrics calculation"""
        risk_manager.add_position(sample_position)

        metrics = risk_manager.get_risk_metrics()

        assert metrics.open_positions == 1
        assert metrics.total_exposure == 50  # 0.001 * 50000
        assert metrics.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]
        assert metrics.current_drawdown >= 0
        assert metrics.var_95 > 0

    def test_emergency_close_all(self, risk_manager, sample_position):
        """Test emergency close all positions"""
        risk_manager.add_position(sample_position)

        closed = risk_manager.emergency_close_all(reason="test_emergency")

        assert len(closed) == 1
        assert len(risk_manager.positions) == 0
        assert closed[0]['position_id'] == sample_position.id

    def test_portfolio_statistics(self, risk_manager):
        """Test portfolio statistics calculation"""
        # Add some trade history
        for i in range(10):
            risk_manager.trade_history.append({
                'position_id': f'TEST{i:03d}',
                'symbol': 'BTCUSDT',
                'side': 'BUY' if i % 2 == 0 else 'SELL',
                'entry_price': 50000,
                'close_price': 51000 if i < 6 else 49000,
                'quantity': 0.001,
                'realized_pnl': 50 if i < 6 else -50,
                'return_pct': 2 if i < 6 else -2,
                'duration': 60,
                'reason': 'test',
                'timestamp': datetime.utcnow()
            })

        stats = risk_manager.get_portfolio_stats()

        assert stats['total_trades'] == 10
        assert stats['winning_trades'] == 6
        assert stats['losing_trades'] == 4
        assert stats['win_rate'] == 0.6
        assert stats['profit_factor'] == 1.5
        assert stats['avg_win'] == 50
        assert stats['avg_loss'] == 50


@pytest.mark.integration
class TestRiskManagerIntegration:
    """Integration tests for risk management system"""

    @pytest.mark.asyncio
    async def test_full_trading_cycle(self):
        """Test complete trading cycle with risk management"""
        risk_manager = RiskManager(
            initial_capital=10000,
            max_drawdown=0.20,
            max_daily_loss=0.05,
            max_positions=3
        )

        # Open first position
        position1 = Position(
            id="POS001",
            symbol="BTCUSDT",
            side="BUY",
            entry_price=50000,
            quantity=0.002,
            stop_loss=48000,
            take_profit=52000,
            entry_time=datetime.utcnow(),
            current_price=50000,
            risk_amount=100
        )

        assert risk_manager.add_position(position1) is True

        # Try to open position exceeding capital
        can_open, reason = risk_manager.can_open_position(
            symbol="ETHUSDT",
            proposed_size=10000  # Too large
        )
        assert can_open is False
        assert "Insufficient capital" in reason

        # Update position with profit
        actions = risk_manager.update_position(
            position_id="POS001",
            current_price=51000,
            atr=500
        )

        # Check trailing stop update
        if 'trailing_stop_updated' in actions:
            assert actions['trailing_stop_updated'] > position1.stop_loss

        # Close position with profit
        result = risk_manager.close_position(
            position_id="POS001",
            close_price=52000,
            reason="take_profit"
        )

        assert result['realized_pnl'] > 0
        assert risk_manager.current_capital > risk_manager.initial_capital

        # Check final metrics
        metrics = risk_manager.get_risk_metrics()
        assert metrics.open_positions == 0
        assert metrics.risk_level == RiskLevel.LOW


if __name__ == "__main__":
    pytest.main([__file__, "-v"])