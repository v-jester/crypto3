"""
Unit tests for ML Engine
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from src.ml.models.ml_engine import MLEngine


class TestMLEngine:
    """Tests for ML Engine"""

    @pytest.fixture
    def ml_engine(self):
        return MLEngine()

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample dataframe with indicators"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='15min')

        # Generate synthetic price data
        np.random.seed(42)
        prices = 50000 + np.cumsum(np.random.randn(100) * 100)

        df = pd.DataFrame({
            'open': prices + np.random.randn(100) * 50,
            'high': prices + abs(np.random.randn(100) * 100),
            'low': prices - abs(np.random.randn(100) * 100),
            'close': prices,
            'volume': np.random.uniform(100, 1000, 100),
            'rsi': np.random.uniform(20, 80, 100),
            'macd': np.random.randn(100) * 0.001,
            'macd_signal': np.random.randn(100) * 0.001,
            'bb_percent': np.random.uniform(0, 1, 100),
            'volume_ratio': np.random.uniform(0.5, 2, 100),
            'atr': abs(np.random.randn(100) * 100),
            'momentum_5': np.random.uniform(-0.05, 0.05, 100)
        }, index=dates)

        return df

    @pytest.mark.asyncio
    async def test_load_models(self, ml_engine):
        """Test model loading"""
        await ml_engine.load_models()
        assert ml_engine.is_trained is True

    @pytest.mark.asyncio
    async def test_predict_with_empty_dataframe(self, ml_engine):
        """Test prediction with empty dataframe"""
        result = await ml_engine.predict(pd.DataFrame())

        assert result['prediction'] == 0
        assert result['prediction_label'] == 'HOLD'
        assert result['confidence'] == 0.5

    @pytest.mark.asyncio
    async def test_predict_oversold_condition(self, ml_engine, sample_dataframe):
        """Test prediction with oversold RSI"""
        # Set oversold RSI
        sample_dataframe.loc[sample_dataframe.index[-1], 'rsi'] = 25
        sample_dataframe.loc[sample_dataframe.index[-1], 'bb_percent'] = 0.1

        result = await ml_engine.predict(sample_dataframe)

        assert result['prediction_label'] == 'UP'
        assert result['confidence'] > 0.6
        assert result['prediction'] == 1

    @pytest.mark.asyncio
    async def test_predict_overbought_condition(self, ml_engine, sample_dataframe):
        """Test prediction with overbought RSI"""
        # Set overbought RSI
        sample_dataframe.loc[sample_dataframe.index[-1], 'rsi'] = 75
        sample_dataframe.loc[sample_dataframe.index[-1], 'bb_percent'] = 0.9

        result = await ml_engine.predict(sample_dataframe)

        assert result['prediction_label'] == 'DOWN'
        assert result['confidence'] > 0.6
        assert result['prediction'] == -1

    @pytest.mark.asyncio
    async def test_predict_neutral_condition(self, ml_engine, sample_dataframe):
        """Test prediction with neutral indicators"""
        # Set neutral values
        sample_dataframe.loc[sample_dataframe.index[-1], 'rsi'] = 50
        sample_dataframe.loc[sample_dataframe.index[-1], 'bb_percent'] = 0.5
        sample_dataframe.loc[sample_dataframe.index[-1], 'macd'] = 0

        result = await ml_engine.predict(sample_dataframe)

        assert result['prediction_label'] == 'HOLD'
        assert result['confidence'] == 0.5
        assert result['prediction'] == 0

    @pytest.mark.asyncio
    async def test_predict_with_volume_confirmation(self, ml_engine, sample_dataframe):
        """Test prediction with volume confirmation"""
        # Set bullish signals with high volume
        sample_dataframe.loc[sample_dataframe.index[-1], 'rsi'] = 30
        sample_dataframe.loc[sample_dataframe.index[-1], 'volume_ratio'] = 2.0
        sample_dataframe.loc[sample_dataframe.index[-1], 'macd'] = 0.001
        sample_dataframe.loc[sample_dataframe.index[-1], 'macd_signal'] = -0.001

        result = await ml_engine.predict(sample_dataframe)

        assert result['prediction_label'] == 'UP'
        # Volume should amplify confidence
        assert result['confidence'] > 0.65

    @pytest.mark.asyncio
    async def test_predict_with_momentum(self, ml_engine, sample_dataframe):
        """Test prediction with momentum indicator"""
        # Strong upward momentum
        sample_dataframe.loc[sample_dataframe.index[-1], 'momentum_5'] = 0.03
        sample_dataframe.loc[sample_dataframe.index[-1], 'rsi'] = 45

        result = await ml_engine.predict(sample_dataframe)

        # Momentum should contribute to bullish signal
        assert result['prediction_label'] in ['UP', 'HOLD']

    @pytest.mark.asyncio
    async def test_predict_exception_handling(self, ml_engine):
        """Test exception handling in prediction"""
        # Create invalid dataframe
        df = pd.DataFrame({'invalid': [1, 2, 3]})

        result = await ml_engine.predict(df)

        assert result['prediction'] == 0
        assert result['prediction_label'] == 'HOLD'
        assert result['confidence'] == 0.5

    def test_get_feature_importance(self, ml_engine):
        """Test feature importance retrieval"""
        importance = ml_engine.get_feature_importance()

        assert isinstance(importance, dict)
        assert 'rsi' in importance
        assert 'macd' in importance
        assert sum(importance.values()) == pytest.approx(1.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_train_placeholder(self, ml_engine, sample_dataframe):
        """Test training method (placeholder)"""
        result = await ml_engine.train(sample_dataframe)

        assert result is False  # Placeholder returns False
        assert ml_engine.is_trained is True  # But sets is_trained

    @pytest.mark.asyncio
    async def test_prediction_consistency(self, ml_engine, sample_dataframe):
        """Test that predictions are consistent for same input"""
        # Set specific values
        sample_dataframe.loc[sample_dataframe.index[-1], 'rsi'] = 28
        sample_dataframe.loc[sample_dataframe.index[-1], 'bb_percent'] = 0.15

        result1 = await ml_engine.predict(sample_dataframe)
        result2 = await ml_engine.predict(sample_dataframe)

        assert result1['prediction_label'] == result2['prediction_label']
        assert result1['confidence'] == result2['confidence']
        assert result1['prediction'] == result2['prediction']


@pytest.mark.integration
class TestMLEngineIntegration:
    """Integration tests for ML Engine"""

    @pytest.mark.asyncio
    async def test_full_prediction_pipeline(self):
        """Test complete prediction pipeline"""
        engine = MLEngine()
        await engine.load_models()

        # Create realistic market data
        dates = pd.date_range(start='2024-01-01', periods=200, freq='15min')

        # Simulate trending market
        trend = np.linspace(50000, 51000, 200)
        noise = np.random.randn(200) * 50
        prices = trend + noise

        df = pd.DataFrame({
            'close': prices,
            'volume': np.random.uniform(100, 1000, 200)
        }, index=dates)

        # Calculate real indicators
        # RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Simple MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        sma20 = df['close'].rolling(window=20).mean()
        std20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma20 + (std20 * 2)
        df['bb_lower'] = sma20 - (std20 * 2)
        df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()

        # ATR (simplified)
        df['atr'] = df['close'].rolling(window=14).std()

        # Momentum
        df['momentum_5'] = df['close'].pct_change(5)

        # Fill NaN values
        df.fillna(method='ffill', inplace=True)
        df.fillna(50, inplace=True)  # Default RSI

        # Get predictions for different market conditions
        predictions = []

        for i in range(50, 200, 10):
            result = await engine.predict(df.iloc[:i])
            predictions.append(result)

        # Verify predictions make sense
        up_predictions = [p for p in predictions if p['prediction_label'] == 'UP']
        down_predictions = [p for p in predictions if p['prediction_label'] == 'DOWN']
        hold_predictions = [p for p in predictions if p['prediction_label'] == 'HOLD']

        # Should have mix of predictions
        assert len(up_predictions) > 0
        assert len(hold_predictions) > 0

        # Confidence should vary
        confidences = [p['confidence'] for p in predictions]
        assert min(confidences) < max(confidences)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])