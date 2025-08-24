# src/ml/models/ml_engine.py
"""
ML Engine СЃ Р°РЅСЃР°РјР±Р»РµРј XGBoost Рё LightGBM РґР»СЏ РїСЂРµРґСЃРєР°Р·Р°РЅРёСЏ РґРІРёР¶РµРЅРёСЏ С†РµРЅС‹
Р’РєР»СЋС‡Р°РµС‚ feature engineering, online learning Рё РІР°Р»РёРґР°С†РёСЋ
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List, Optional, Tuple, Any
import joblib
from pathlib import Path
import asyncio
from datetime import datetime, timedelta
from src.monitoring.logger import logger, log_performance
from src.config.settings import settings
from src.monitoring.metrics import metrics_collector


class FeatureEngineering:
    """Р“РµРЅРµСЂР°С‚РѕСЂ РїСЂРёР·РЅР°РєРѕРІ РґР»СЏ ML РјРѕРґРµР»РµР№"""

    def __init__(self):
        self.feature_names = []
        self.scaler = RobustScaler()  # РЈСЃС‚РѕР№С‡РёРІ Рє РІС‹Р±СЂРѕСЃР°Рј

    def create_features(
            self,
            df: pd.DataFrame,
            timeframes: List[int] = [5, 15, 60, 240]
    ) -> pd.DataFrame:
        """
        РЎРѕР·РґР°РЅРёРµ 3000+ РїСЂРёР·РЅР°РєРѕРІ РёР· OHLCV РґР°РЅРЅС‹С…

        Args:
            df: DataFrame СЃ OHLCV РґР°РЅРЅС‹РјРё
            timeframes: РЎРїРёСЃРѕРє С‚Р°Р№РјС„СЂРµР№РјРѕРІ РґР»СЏ РјСѓР»СЊС‚РёС‚Р°Р№РјС„СЂРµР№РјРѕРІРѕРіРѕ Р°РЅР°Р»РёР·Р°

        Returns:
            DataFrame СЃ РїСЂРёР·РЅР°РєР°РјРё
        """
        features = pd.DataFrame(index=df.index)

        # Р‘Р°Р·РѕРІС‹Рµ С†РµРЅРѕРІС‹Рµ РїСЂРёР·РЅР°РєРё
        features = self._add_price_features(features, df)

        # РўРµС…РЅРёС‡РµСЃРєРёРµ РёРЅРґРёРєР°С‚РѕСЂС‹ РґР»СЏ СЂР°Р·РЅС‹С… С‚Р°Р№РјС„СЂРµР№РјРѕРІ
        for tf in timeframes:
            features = self._add_technical_features(features, df, tf)

        # РњРёРєСЂРѕСЃС‚СЂСѓРєС‚СѓСЂР° СЂС‹РЅРєР°
        features = self._add_microstructure_features(features, df)

        # РћР±СЉС‘РјРЅС‹Рµ РїСЂРёР·РЅР°РєРё
        features = self._add_volume_features(features, df)

        # РџР°С‚С‚РµСЂРЅС‹ Рё СЃРІРµС‡РЅС‹Рµ С„РѕСЂРјР°С†РёРё
        features = self._add_pattern_features(features, df)

        # РЎС‚Р°С‚РёСЃС‚РёС‡РµСЃРєРёРµ РїСЂРёР·РЅР°РєРё
        features = self._add_statistical_features(features, df)

        # Р’Р·Р°РёРјРѕРґРµР№СЃС‚РІРёСЏ РїСЂРёР·РЅР°РєРѕРІ
        features = self._add_interaction_features(features)

        # РЎРѕС…СЂР°РЅСЏРµРј РёРјРµРЅР° РїСЂРёР·РЅР°РєРѕРІ
        self.feature_names = features.columns.tolist()

        logger.logger.info(f"Created {len(self.feature_names)} features")

        return features

    def _add_price_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Р”РѕР±Р°РІР»РµРЅРёРµ С†РµРЅРѕРІС‹С… РїСЂРёР·РЅР°РєРѕРІ"""
        # РР·РјРµРЅРµРЅРёСЏ С†РµРЅС‹
        for period in [1, 3, 5, 10, 20, 50]:
            features[f'returns_{period}'] = df['close'].pct_change(period)
            features[f'log_returns_{period}'] = np.log(df['close'] / df['close'].shift(period))

        # Р¦РµРЅРѕРІС‹Рµ СѓСЂРѕРІРЅРё
        for period in [10, 20, 50, 100]:
            features[f'high_{period}'] = df['high'].rolling(period).max()
            features[f'low_{period}'] = df['low'].rolling(period).min()
            features[f'distance_from_high_{period}'] = (features[f'high_{period}'] - df['close']) / df['close']
            features[f'distance_from_low_{period}'] = (df['close'] - features[f'low_{period}']) / df['close']

        # Р¦РµРЅРѕРІС‹Рµ РєР°РЅР°Р»С‹
        features['hl_ratio'] = df['high'] / df['low']
        features['co_ratio'] = df['close'] / df['open']
        features['body_size'] = abs(df['close'] - df['open']) / df['open']
        features['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['open']
        features['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['open']

        return features

    def _add_technical_features(
            self,
            features: pd.DataFrame,
            df: pd.DataFrame,
            timeframe: int
    ) -> pd.DataFrame:
        """Р”РѕР±Р°РІР»РµРЅРёРµ С‚РµС…РЅРёС‡РµСЃРєРёС… РёРЅРґРёРєР°С‚РѕСЂРѕРІ"""
        prefix = f'tf{timeframe}_'

        # Moving Averages
        for ma_period in [7, 14, 21, 50]:
            ma = df['close'].rolling(ma_period).mean()
            features[f'{prefix}sma_{ma_period}'] = ma
            features[f'{prefix}sma_{ma_period}_slope'] = ma.diff()
            features[f'{prefix}price_to_sma_{ma_period}'] = df['close'] / ma

        # EMA
        for ema_period in [9, 12, 26]:
            ema = df['close'].ewm(span=ema_period, adjust=False).mean()
            features[f'{prefix}ema_{ema_period}'] = ema
            features[f'{prefix}price_to_ema_{ema_period}'] = df['close'] / ema

        # RSI РІР°СЂРёР°РЅС‚С‹
        for rsi_period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features[f'{prefix}rsi_{rsi_period}'] = rsi
            features[f'{prefix}rsi_{rsi_period}_oversold'] = (rsi < 30).astype(int)
            features[f'{prefix}rsi_{rsi_period}_overbought'] = (rsi > 70).astype(int)

        # MACD РІР°СЂРёР°РЅС‚С‹
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        features[f'{prefix}macd'] = macd
        features[f'{prefix}macd_signal'] = signal
        features[f'{prefix}macd_hist'] = macd - signal
        features[f'{prefix}macd_cross'] = np.where(macd > signal, 1, -1)

        # Bollinger Bands
        for bb_period in [10, 20, 50]:
            sma = df['close'].rolling(bb_period).mean()
            std = df['close'].rolling(bb_period).std()
            upper = sma + (std * 2)
            lower = sma - (std * 2)
            features[f'{prefix}bb_upper_{bb_period}'] = upper
            features[f'{prefix}bb_lower_{bb_period}'] = lower
            features[f'{prefix}bb_width_{bb_period}'] = upper - lower
            features[f'{prefix}bb_position_{bb_period}'] = (df['close'] - lower) / (upper - lower)

        # ATR Рё РІРѕР»Р°С‚РёР»СЊРЅРѕСЃС‚СЊ
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)

        for atr_period in [7, 14, 21]:
            atr = true_range.rolling(atr_period).mean()
            features[f'{prefix}atr_{atr_period}'] = atr
            features[f'{prefix}atr_ratio_{atr_period}'] = atr / df['close']

        # Stochastic Oscillator
        for stoch_period in [5, 14, 21]:
            low_min = df['low'].rolling(stoch_period).min()
            high_max = df['high'].rolling(stoch_period).max()
            stoch_k = 100 * ((df['close'] - low_min) / (high_max - low_min))
            stoch_d = stoch_k.rolling(3).mean()
            features[f'{prefix}stoch_k_{stoch_period}'] = stoch_k
            features[f'{prefix}stoch_d_{stoch_period}'] = stoch_d

        return features

    def _add_microstructure_features(
            self,
            features: pd.DataFrame,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """Р”РѕР±Р°РІР»РµРЅРёРµ РїСЂРёР·РЅР°РєРѕРІ РјРёРєСЂРѕСЃС‚СЂСѓРєС‚СѓСЂС‹ СЂС‹РЅРєР°"""
        # РЎРїСЂРµРґ Рё РґРёР°РїР°Р·РѕРЅ
        features['spread'] = df['high'] - df['low']
        features['spread_pct'] = features['spread'] / df['close']
        features['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        features['weighted_close'] = (df['high'] + df['low'] + 2 * df['close']) / 4

        # Р­С„С„РµРєС‚РёРІРЅРѕСЃС‚СЊ РґРІРёР¶РµРЅРёСЏ
        for period in [5, 10, 20]:
            price_change = df['close'].diff(period).abs()
            path_length = df['spread'].rolling(period).sum()
            features[f'efficiency_ratio_{period}'] = price_change / path_length

        # РђРјРїР»РёС‚СѓРґР° Рё СЂР°Р·РјР°С…
        for period in [5, 10, 20, 50]:
            features[f'amplitude_{period}'] = (
                                                      df['high'].rolling(period).max() -
                                                      df['low'].rolling(period).min()
                                              ) / df['close'].rolling(period).mean()

        # Р”РёСЃР±Р°Р»Р°РЅСЃ РѕР±СЉС‘РјРѕРІ
        if 'taker_buy_base' in df.columns:
            features['buy_sell_imbalance'] = (
                                                     2 * df['taker_buy_base'] - df['volume']
                                             ) / df['volume']
            features['buy_pressure'] = df['taker_buy_base'] / df['volume']

        return features

    def _add_volume_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Р”РѕР±Р°РІР»РµРЅРёРµ РѕР±СЉС‘РјРЅС‹С… РїСЂРёР·РЅР°РєРѕРІ"""
        # РћР±СЉС‘РјРЅС‹Рµ РѕС‚РЅРѕС€РµРЅРёСЏ
        for period in [5, 10, 20, 50]:
            features[f'volume_ma_{period}'] = df['volume'].rolling(period).mean()
            features[f'volume_ratio_{period}'] = df['volume'] / features[f'volume_ma_{period}']
            features[f'volume_std_{period}'] = df['volume'].rolling(period).std()

        # VWAP
        for period in [10, 20, 50]:
            cum_volume = df['volume'].rolling(period).sum()
            cum_pv = (df['close'] * df['volume']).rolling(period).sum()
            vwap = cum_pv / cum_volume
            features[f'vwap_{period}'] = vwap
            features[f'price_to_vwap_{period}'] = df['close'] / vwap

        # OBV (On-Balance Volume)
        obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        features['obv'] = obv
        features['obv_ma_10'] = obv.rolling(10).mean()
        features['obv_slope'] = obv.diff()

        # Accumulation/Distribution
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfv = mfm * df['volume']
        features['acc_dist'] = mfv.cumsum()

        # Chaikin Money Flow
        for period in [10, 20]:
            features[f'cmf_{period}'] = mfv.rolling(period).sum() / df['volume'].rolling(period).sum()

        # Volume Rate of Change
        for period in [5, 10, 20]:
            features[f'vroc_{period}'] = df['volume'].pct_change(period)

        return features

    def _add_pattern_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Р”РѕР±Р°РІР»РµРЅРёРµ СЃРІРµС‡РЅС‹С… РїР°С‚С‚РµСЂРЅРѕРІ"""
        # Р‘Р°Р·РѕРІС‹Рµ СЃРІРµС‡РЅС‹Рµ РїР°С‚С‚РµСЂРЅС‹
        body = df['close'] - df['open']
        body_abs = abs(body)

        # Doji
        features['is_doji'] = (body_abs / df['open'] < 0.001).astype(int)

        # Hammer/Hanging Man
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        features['is_hammer'] = (
                (lower_shadow > body_abs * 2) &
                (upper_shadow < body_abs * 0.1)
        ).astype(int)

        # Engulfing
        prev_body = body.shift(1)
        features['bullish_engulfing'] = (
                (body > 0) &
                (prev_body < 0) &
                (df['open'] < df['close'].shift(1)) &
                (df['close'] > df['open'].shift(1))
        ).astype(int)

        features['bearish_engulfing'] = (
                (body < 0) &
                (prev_body > 0) &
                (df['open'] > df['close'].shift(1)) &
                (df['close'] < df['open'].shift(1))
        ).astype(int)

        # РџР°С‚С‚РµСЂРЅС‹ С‚СЂРµРЅРґР°
        for period in [5, 10, 20]:
            # Higher Highs and Higher Lows (uptrend)
            hh = df['high'] > df['high'].shift(period)
            hl = df['low'] > df['low'].shift(period)
            features[f'uptrend_{period}'] = (hh & hl).astype(int)

            # Lower Highs and Lower Lows (downtrend)
            lh = df['high'] < df['high'].shift(period)
            ll = df['low'] < df['low'].shift(period)
            features[f'downtrend_{period}'] = (lh & ll).astype(int)

        return features

    def _add_statistical_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Р”РѕР±Р°РІР»РµРЅРёРµ СЃС‚Р°С‚РёСЃС‚РёС‡РµСЃРєРёС… РїСЂРёР·РЅР°РєРѕРІ"""
        # РЎРєРѕР»СЊР·СЏС‰РёРµ СЃС‚Р°С‚РёСЃС‚РёРєРё
        for period in [5, 10, 20, 50]:
            returns = df['close'].pct_change()

            # РњРѕРјРµРЅС‚С‹ СЂР°СЃРїСЂРµРґРµР»РµРЅРёСЏ
            features[f'returns_mean_{period}'] = returns.rolling(period).mean()
            features[f'returns_std_{period}'] = returns.rolling(period).std()
            features[f'returns_skew_{period}'] = returns.rolling(period).skew()
            features[f'returns_kurt_{period}'] = returns.rolling(period).kurt()

            # РљРІР°РЅС‚РёР»Рё
            features[f'returns_q25_{period}'] = returns.rolling(period).quantile(0.25)
            features[f'returns_q75_{period}'] = returns.rolling(period).quantile(0.75)
            features[f'returns_iqr_{period}'] = features[f'returns_q75_{period}'] - features[f'returns_q25_{period}']

            # РђРІС‚РѕРєРѕСЂСЂРµР»СЏС†РёСЏ
            for lag in [1, 3, 5]:
                features[f'autocorr_{period}_lag{lag}'] = returns.rolling(period).apply(
                    lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
                )

        # Z-score
        for period in [10, 20, 50]:
            mean = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            features[f'zscore_{period}'] = (df['close'] - mean) / std

        # Percentile rank
        for period in [20, 50, 100]:
            features[f'percentile_rank_{period}'] = df['close'].rolling(period).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            )

        return features

    def _add_interaction_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Р”РѕР±Р°РІР»РµРЅРёРµ РІР·Р°РёРјРѕРґРµР№СЃС‚РІРёР№ РјРµР¶РґСѓ РїСЂРёР·РЅР°РєР°РјРё"""
        # Р’Р°Р¶РЅС‹Рµ РІР·Р°РёРјРѕРґРµР№СЃС‚РІРёСЏ
        if 'tf15_rsi_14' in features.columns and 'tf15_macd' in features.columns:
            features['rsi_macd_interaction'] = features['tf15_rsi_14'] * features['tf15_macd']

        if 'volume_ratio_10' in features.columns and 'returns_1' in features.columns:
            features['volume_return_interaction'] = features['volume_ratio_10'] * features['returns_1']

        if 'tf15_bb_position_20' in features.columns and 'tf15_rsi_14' in features.columns:
            features['bb_rsi_interaction'] = features['tf15_bb_position_20'] * features['tf15_rsi_14']

        # РџРѕР»РёРЅРѕРјРёР°Р»СЊРЅС‹Рµ РїСЂРёР·РЅР°РєРё РґР»СЏ РєР»СЋС‡РµРІС‹С… РёРЅРґРёРєР°С‚РѕСЂРѕРІ
        key_features = ['returns_1', 'volume_ratio_10', 'tf15_rsi_14']
        for feat in key_features:
            if feat in features.columns:
                features[f'{feat}_squared'] = features[feat] ** 2
                features[f'{feat}_cubed'] = features[feat] ** 3

        return features

    def fit_transform(self, features: pd.DataFrame) -> np.ndarray:
        """РќРѕСЂРјР°Р»РёР·Р°С†РёСЏ РїСЂРёР·РЅР°РєРѕРІ"""
        # Р—Р°РїРѕР»РЅСЏРµРј РїСЂРѕРїСѓСЃРєРё
        features = features.fillna(method='ffill').fillna(0)

        # Р—Р°РјРµРЅСЏРµРј Р±РµСЃРєРѕРЅРµС‡РЅРѕСЃС‚Рё
        features = features.replace([np.inf, -np.inf], 0)

        # РќРѕСЂРјР°Р»РёР·Р°С†РёСЏ
        return self.scaler.fit_transform(features)

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        """РўСЂР°РЅСЃС„РѕСЂРјР°С†РёСЏ РїСЂРёР·РЅР°РєРѕРІ"""
        features = features.fillna(method='ffill').fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        return self.scaler.transform(features)


class CryptoMLEngine:
    """РћСЃРЅРѕРІРЅРѕР№ ML РґРІРёР¶РѕРє РґР»СЏ РїСЂРµРґСЃРєР°Р·Р°РЅРёР№"""

    def __init__(self):
        self.feature_engineering = FeatureEngineering()
        self.xgb_model = None
        self.lgb_model = None
        self.ensemble_weights = {'xgboost': 0.6, 'lightgbm': 0.4}  # XGBoost РїРѕРєР°Р·Р°Р» Р»СѓС‡С€РёРµ СЂРµР·СѓР»СЊС‚Р°С‚С‹
        self.models_path = Path("models")
        self.models_path.mkdir(exist_ok=True)

        # РџР°СЂР°РјРµС‚СЂС‹ РјРѕРґРµР»РµР№ (РѕРїС‚РёРјРёР·РёСЂРѕРІР°РЅРЅС‹Рµ РґР»СЏ РєСЂРёРїС‚Рѕ)
        self.xgb_params = {
            'objective': 'multi:softprob',
            'num_class': 3,  # DOWN, NEUTRAL, UP
            'max_depth': 6,
            'learning_rate': 0.01,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',  # Р‘С‹СЃС‚СЂРµРµ РґР»СЏ Р±РѕР»СЊС€РёС… РґР°С‚Р°СЃРµС‚РѕРІ
            'predictor': 'cpu_predictor'
        }

        self.lgb_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'num_leaves': 31,
            'learning_rate': 0.01,
            'n_estimators': 1000,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.1,
            'lambda_l2': 1,
            'min_child_samples': 20,
            'random_state': 42,
            'n_jobs': -1,
            'device': 'cpu'
        }

        # Р‘СѓС„РµСЂ РґР»СЏ online learning
        self.online_buffer = []
        self.online_buffer_size = 50
        self.last_training_time = None

    def prepare_target(
            self,
            df: pd.DataFrame,
            lookahead: int = 5,
            threshold: float = 0.002
    ) -> pd.Series:
        """
        РџРѕРґРіРѕС‚РѕРІРєР° С†РµР»РµРІРѕР№ РїРµСЂРµРјРµРЅРЅРѕР№

        Args:
            df: DataFrame СЃ С†РµРЅР°РјРё
            lookahead: РџРµСЂРёРѕРґ РїСЂРµРґСЃРєР°Р·Р°РЅРёСЏ (СЃРІРµС‡РµР№ РІРїРµСЂС‘Рґ)
            threshold: РџРѕСЂРѕРі РґР»СЏ РєР»Р°СЃСЃРёС„РёРєР°С†РёРё (0.2%)

        Returns:
            Series СЃ РјРµС‚РєР°РјРё РєР»Р°СЃСЃРѕРІ: 0=DOWN, 1=NEUTRAL, 2=UP
        """
        future_returns = df['close'].shift(-lookahead) / df['close'] - 1

        # РљР»Р°СЃСЃРёС„РёРєР°С†РёСЏ
        target = pd.Series(index=df.index, dtype=int)
        target[future_returns < -threshold] = 0  # DOWN
        target[future_returns > threshold] = 2  # UP
        target[(future_returns >= -threshold) & (future_returns <= threshold)] = 1  # NEUTRAL

        return target

    @log_performance("train_models")
    async def train(
            self,
            df: pd.DataFrame,
            validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        РћР±СѓС‡РµРЅРёРµ РјРѕРґРµР»РµР№

        Args:
            df: DataFrame СЃ OHLCV РґР°РЅРЅС‹РјРё
            validation_split: Р”РѕР»СЏ РІР°Р»РёРґР°С†РёРѕРЅРЅРѕР№ РІС‹Р±РѕСЂРєРё

        Returns:
            РЎР»РѕРІР°СЂСЊ СЃ РјРµС‚СЂРёРєР°РјРё РѕР±СѓС‡РµРЅРёСЏ
        """
        try:
            logger.logger.info("Starting model training")

            # РЎРѕР·РґР°РЅРёРµ РїСЂРёР·РЅР°РєРѕРІ
            features = self.feature_engineering.create_features(df)
            X = self.feature_engineering.fit_transform(features)

            # РџРѕРґРіРѕС‚РѕРІРєР° С†РµР»РµРІРѕР№ РїРµСЂРµРјРµРЅРЅРѕР№
            y = self.prepare_target(df)

            # РЈРґР°Р»СЏРµРј NaN
            mask = ~(np.isnan(X).any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask].values

            # Р Р°Р·РґРµР»РµРЅРёРµ РЅР° train/validation (РІСЂРµРјРµРЅРЅРѕР№ СЃРїР»РёС‚)
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            logger.logger.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")

            # РћР±СѓС‡РµРЅРёРµ XGBoost
            start_time = datetime.utcnow()
            self.xgb_model = xgb.XGBClassifier(**self.xgb_params)
            self.xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            xgb_time = (datetime.utcnow() - start_time).total_seconds()

            # РџСЂРµРґСЃРєР°Р·Р°РЅРёСЏ XGBoost
            xgb_pred = self.xgb_model.predict(X_val)
            xgb_proba = self.xgb_model.predict_proba(X_val)
            xgb_accuracy = accuracy_score(y_val, xgb_pred)

            logger.logger.info(f"XGBoost trained in {xgb_time:.2f}s, accuracy: {xgb_accuracy:.4f}")
            metrics_collector.ml_training_time.labels(model="xgboost").observe(xgb_time)
            metrics_collector.ml_accuracy.labels(model="xgboost").set(xgb_accuracy)

            # РћР±СѓС‡РµРЅРёРµ LightGBM
            start_time = datetime.utcnow()
            self.lgb_model = lgb.LGBMClassifier(**self.lgb_params)
            self.lgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            lgb_time = (datetime.utcnow() - start_time).total_seconds()

            # РџСЂРµРґСЃРєР°Р·Р°РЅРёСЏ LightGBM
            lgb_pred = self.lgb_model.predict(X_val)
            lgb_proba = self.lgb_model.predict_proba(X_val)
            lgb_accuracy = accuracy_score(y_val, lgb_pred)

            logger.logger.info(f"LightGBM trained in {lgb_time:.2f}s, accuracy: {lgb_accuracy:.4f}")
            metrics_collector.ml_training_time.labels(model="lightgbm").observe(lgb_time)
            metrics_collector.ml_accuracy.labels(model="lightgbm").set(lgb_accuracy)

            # РђРЅСЃР°РјР±Р»РµРІС‹Рµ РїСЂРµРґСЃРєР°Р·Р°РЅРёСЏ
            ensemble_proba = (
                    self.ensemble_weights['xgboost'] * xgb_proba +
                    self.ensemble_weights['lightgbm'] * lgb_proba
            )
            ensemble_pred = np.argmax(ensemble_proba, axis=1)
            ensemble_accuracy = accuracy_score(y_val, ensemble_pred)

            logger.logger.info(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
            metrics_collector.ml_accuracy.labels(model="ensemble").set(ensemble_accuracy)

            # Р”РѕРїРѕР»РЅРёС‚РµР»СЊРЅС‹Рµ РјРµС‚СЂРёРєРё
            metrics = {
                'xgboost': {
                    'accuracy': xgb_accuracy,
                    'precision': precision_score(y_val, xgb_pred, average='weighted'),
                    'recall': recall_score(y_val, xgb_pred, average='weighted'),
                    'f1': f1_score(y_val, xgb_pred, average='weighted'),
                    'training_time': xgb_time
                },
                'lightgbm': {
                    'accuracy': lgb_accuracy,
                    'precision': precision_score(y_val, lgb_pred, average='weighted'),
                    'recall': recall_score(y_val, lgb_pred, average='weighted'),
                    'f1': f1_score(y_val, lgb_pred, average='weighted'),
                    'training_time': lgb_time
                },
                'ensemble': {
                    'accuracy': ensemble_accuracy,
                    'precision': precision_score(y_val, ensemble_pred, average='weighted'),
                    'recall': recall_score(y_val, ensemble_pred, average='weighted'),
                    'f1': f1_score(y_val, ensemble_pred, average='weighted')
                }
            }

            # Р’Р°Р¶РЅРѕСЃС‚СЊ РїСЂРёР·РЅР°РєРѕРІ
            feature_importance = self._get_feature_importance()

            # РЎРѕС…СЂР°РЅРµРЅРёРµ РјРѕРґРµР»РµР№
            await self.save_models()

            self.last_training_time = datetime.utcnow()

            return {
                'metrics': metrics,
                'feature_importance': feature_importance,
                'training_samples': len(X_train),
                'validation_samples': len(X_val)
            }

        except Exception as e:
            logger.logger.error(f"Error: {e}, context: {"context": "Model training failed"}")
            raise

    async def predict(
            self,
            df: pd.DataFrame,
            return_proba: bool = True
    ) -> Dict[str, Any]:
        """
        РџСЂРµРґСЃРєР°Р·Р°РЅРёРµ РЅР° РЅРѕРІС‹С… РґР°РЅРЅС‹С…

        Args:
            df: DataFrame СЃ OHLCV РґР°РЅРЅС‹РјРё
            return_proba: Р’РѕР·РІСЂР°С‰Р°С‚СЊ РІРµСЂРѕСЏС‚РЅРѕСЃС‚Рё РєР»Р°СЃСЃРѕРІ

        Returns:
            РЎР»РѕРІР°СЂСЊ СЃ РїСЂРµРґСЃРєР°Р·Р°РЅРёСЏРјРё
        """
        if not self.xgb_model or not self.lgb_model:
            raise ValueError("Models not trained. Call train() first.")

        try:
            # РЎРѕР·РґР°РЅРёРµ РїСЂРёР·РЅР°РєРѕРІ
            features = self.feature_engineering.create_features(df)
            X = self.feature_engineering.transform(features)

            # РџСЂРµРґСЃРєР°Р·Р°РЅРёСЏ XGBoost
            xgb_proba = self.xgb_model.predict_proba(X[-1:])

            # РџСЂРµРґСЃРєР°Р·Р°РЅРёСЏ LightGBM
            lgb_proba = self.lgb_model.predict_proba(X[-1:])

            # РђРЅСЃР°РјР±Р»СЊ
            ensemble_proba = (
                    self.ensemble_weights['xgboost'] * xgb_proba +
                    self.ensemble_weights['lightgbm'] * lgb_proba
            )[0]

            # РљР»Р°СЃСЃ СЃ РјР°РєСЃРёРјР°Р»СЊРЅРѕР№ РІРµСЂРѕСЏС‚РЅРѕСЃС‚СЊСЋ
            prediction = np.argmax(ensemble_proba)
            confidence = ensemble_proba[prediction]

            # Р—Р°РїРёСЃС‹РІР°РµРј РјРµС‚СЂРёРєРё
            metrics_collector.ml_predictions.labels(
                model="ensemble",
                prediction_type=["DOWN", "NEUTRAL", "UP"][prediction]
            ).inc()
            metrics_collector.ml_confidence.labels(model="ensemble").observe(confidence)

            result = {
                'prediction': prediction,
                'prediction_label': ["DOWN", "NEUTRAL", "UP"][prediction],
                'confidence': float(confidence),
                'timestamp': datetime.utcnow().isoformat()
            }

            if return_proba:
                result['probabilities'] = {
                    'down': float(ensemble_proba[0]),
                    'neutral': float(ensemble_proba[1]),
                    'up': float(ensemble_proba[2])
                }
                result['model_probabilities'] = {
                    'xgboost': {
                        'down': float(xgb_proba[0][0]),
                        'neutral': float(xgb_proba[0][1]),
                        'up': float(xgb_proba[0][2])
                    },
                    'lightgbm': {
                        'down': float(lgb_proba[0][0]),
                        'neutral': float(lgb_proba[0][1]),
                        'up': float(lgb_proba[0][2])
                    }
                }

            return result

        except Exception as e:
            logger.logger.error(f"Error: {e}, context: {"context": "Prediction failed"}")
            raise

    async def online_update(self, X_new: np.ndarray, y_new: int):
        """
        РћРЅР»Р°Р№РЅ РѕР±РЅРѕРІР»РµРЅРёРµ РјРѕРґРµР»Рё

        Args:
            X_new: РќРѕРІС‹Рµ РїСЂРёР·РЅР°РєРё
            y_new: РќРѕРІР°СЏ РјРµС‚РєР°
        """
        if not settings.ml.ENABLE_ONLINE_LEARNING:
            return

        # Р”РѕР±Р°РІР»СЏРµРј РІ Р±СѓС„РµСЂ
        self.online_buffer.append((X_new, y_new))

        # РћР±РЅРѕРІР»СЏРµРј РµСЃР»Рё Р±СѓС„РµСЂ Р·Р°РїРѕР»РЅРµРЅ
        if len(self.online_buffer) >= self.online_buffer_size:
            X_batch = np.array([x for x, _ in self.online_buffer])
            y_batch = np.array([y for _, y in self.online_buffer])

            # Р§Р°СЃС‚РёС‡РЅРѕРµ РѕР±СѓС‡РµРЅРёРµ XGBoost
            if self.xgb_model:
                self.xgb_model.fit(
                    X_batch, y_batch,
                    xgb_model=self.xgb_model.get_booster(),
                    verbose=False
                )

            # LightGBM РЅРµ РїРѕРґРґРµСЂР¶РёРІР°РµС‚ РёРЅРєСЂРµРјРµРЅС‚Р°Р»СЊРЅРѕРµ РѕР±СѓС‡РµРЅРёРµ РЅР°РїСЂСЏРјСѓСЋ,
            # РЅРѕ РјРѕР¶РЅРѕ РёСЃРїРѕР»СЊР·РѕРІР°С‚СЊ init_model
            if self.lgb_model:
                self.lgb_model.fit(
                    X_batch, y_batch,
                    init_model=self.lgb_model,
                    keep_training_booster=True
                )

            # РћС‡РёС‰Р°РµРј Р±СѓС„РµСЂ, РѕСЃС‚Р°РІР»СЏСЏ РїРѕСЃР»РµРґРЅРёРµ СЌР»РµРјРµРЅС‚С‹
            self.online_buffer = self.online_buffer[-25:]

            logger.logger.info("Models updated with online learning")

    def _get_feature_importance(self, top_n: int = 50) -> Dict[str, float]:
        """РџРѕР»СѓС‡РµРЅРёРµ РІР°Р¶РЅРѕСЃС‚Рё РїСЂРёР·РЅР°РєРѕРІ"""
        importance = {}

        if self.xgb_model:
            xgb_importance = self.xgb_model.feature_importances_
            for i, imp in enumerate(xgb_importance):
                feat_name = self.feature_engineering.feature_names[i] if i < len(
                    self.feature_engineering.feature_names) else f"feature_{i}"
                importance[feat_name] = importance.get(feat_name, 0) + imp * self.ensemble_weights['xgboost']

        if self.lgb_model:
            lgb_importance = self.lgb_model.feature_importances_
            for i, imp in enumerate(lgb_importance):
                feat_name = self.feature_engineering.feature_names[i] if i < len(
                    self.feature_engineering.feature_names) else f"feature_{i}"
                importance[feat_name] = importance.get(feat_name, 0) + imp * self.ensemble_weights['lightgbm']

        # РЎРѕСЂС‚РёСЂРѕРІРєР° Рё РІРѕР·РІСЂР°С‚ С‚РѕРї РїСЂРёР·РЅР°РєРѕРІ
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n])

        return sorted_importance

    async def save_models(self):
        """РЎРѕС…СЂР°РЅРµРЅРёРµ РјРѕРґРµР»РµР№ РЅР° РґРёСЃРє"""
        try:
            # XGBoost
            if self.xgb_model:
                xgb_path = self.models_path / f"xgboost_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pkl"
                joblib.dump(self.xgb_model, xgb_path)

                # РЎРѕС…СЂР°РЅСЏРµРј С‚Р°РєР¶Рµ РїРѕСЃР»РµРґРЅСЋСЋ РІРµСЂСЃРёСЋ
                joblib.dump(self.xgb_model, self.models_path / "xgboost_latest.pkl")

            # LightGBM
            if self.lgb_model:
                lgb_path = self.models_path / f"lightgbm_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pkl"
                joblib.dump(self.lgb_model, lgb_path)
                joblib.dump(self.lgb_model, self.models_path / "lightgbm_latest.pkl")

            # Scaler
            scaler_path = self.models_path / "scaler_latest.pkl"
            joblib.dump(self.feature_engineering.scaler, scaler_path)

            logger.logger.info("Models saved successfully")

        except Exception as e:
            logger.logger.error(f"Error: {e}, context: {"context": "Failed to save models"}")

    async def load_models(self):
        """Р—Р°РіСЂСѓР·РєР° РјРѕРґРµР»РµР№ СЃ РґРёСЃРєР°"""
        try:
            xgb_path = self.models_path / "xgboost_latest.pkl"
            if xgb_path.exists():
                self.xgb_model = joblib.load(xgb_path)
                logger.logger.info("XGBoost model loaded")

            lgb_path = self.models_path / "lightgbm_latest.pkl"
            if lgb_path.exists():
                self.lgb_model = joblib.load(lgb_path)
                logger.logger.info("LightGBM model loaded")

            scaler_path = self.models_path / "scaler_latest.pkl"
            if scaler_path.exists():
                self.feature_engineering.scaler = joblib.load(scaler_path)
                logger.logger.info("Feature scaler loaded")

        except Exception as e:
            logger.logger.error(f"Error: {e}, context: {"context": "Failed to load models"}")

    def should_retrain(self) -> bool:
        """РџСЂРѕРІРµСЂРєР° РЅРµРѕР±С…РѕРґРёРјРѕСЃС‚Рё РїРµСЂРµРѕР±СѓС‡РµРЅРёСЏ"""
        if not self.last_training_time:
            return True

        hours_since_training = (datetime.utcnow() - self.last_training_time).total_seconds() / 3600
        return hours_since_training >= settings.ml.RETRAIN_INTERVAL_HOURS


# Р“Р»РѕР±Р°Р»СЊРЅС‹Р№ СЌРєР·РµРјРїР»СЏСЂ
ml_engine = CryptoMLEngine()


# Р’СЃРїРѕРјРѕРіР°С‚РµР»СЊРЅС‹Рµ С„СѓРЅРєС†РёРё
async def init_ml_engine():
    """РРЅРёС†РёР°Р»РёР·Р°С†РёСЏ ML РґРІРёР¶РєР°"""
    await ml_engine.load_models()
    logger.logger.info("ML Engine initialized")
    return ml_engine
