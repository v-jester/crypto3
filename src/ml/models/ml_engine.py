# src/ml/models/ml_engine.py
"""
ML Engine для торгового бота - упрощенная версия с эвристическими предсказаниями
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from src.monitoring.logger import logger


class MLEngine:
    """Упрощенный ML движок с эвристическими предсказаниями"""

    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        self.is_trained = True  # Всегда True для использования эвристик

    async def load_models(self):
        """Загрузка моделей (заглушка)"""
        logger.logger.info("ML models initialization - using heuristic predictions")
        # Устанавливаем is_trained в True для использования эвристических предсказаний
        self.is_trained = True
        return False

    async def predict(self, df: pd.DataFrame, return_proba: bool = False) -> Dict[str, Any]:
        """Простое предсказание на основе технических индикаторов"""

        if df.empty:
            return {
                'prediction': 0,
                'prediction_label': 'HOLD',
                'confidence': 0.5
            }

        try:
            # Простая логика на основе индикаторов
            last_row = df.iloc[-1]

            score = 0
            factors = 0

            # RSI - основной индикатор
            if 'rsi' in df.columns:
                rsi = last_row['rsi']
                if rsi < 30:
                    score += 1.5  # Сильный сигнал покупки
                elif rsi < 40:
                    score += 0.7  # Умеренный сигнал покупки
                elif rsi > 70:
                    score -= 1.5  # Сильный сигнал продажи
                elif rsi > 60:
                    score -= 0.7  # Умеренный сигнал продажи
                factors += 1

            # MACD
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                macd_diff = last_row['macd'] - last_row['macd_signal']
                if macd_diff > 0:
                    score += 0.5
                    # Усиливаем сигнал если MACD растёт
                    if len(df) > 1 and 'macd' in df.columns:
                        prev_macd = df['macd'].iloc[-2]
                        if last_row['macd'] > prev_macd:
                            score += 0.3
                else:
                    score -= 0.5
                    # Усиливаем сигнал если MACD падает
                    if len(df) > 1 and 'macd' in df.columns:
                        prev_macd = df['macd'].iloc[-2]
                        if last_row['macd'] < prev_macd:
                            score -= 0.3
                factors += 1

            # Bollinger Bands
            if 'bb_percent' in df.columns:
                bb = last_row['bb_percent']
                if bb < 0.2:
                    score += 0.8  # Сильная перепроданность
                elif bb < 0.3:
                    score += 0.4
                elif bb > 0.8:
                    score -= 0.8  # Сильная перекупленность
                elif bb > 0.7:
                    score -= 0.4
                factors += 1

            # Volume confirmation
            if 'volume_ratio' in df.columns:
                vol_ratio = last_row['volume_ratio']
                if vol_ratio > 1.5:
                    # Усиливаем текущий сигнал при высоком объёме
                    if score > 0:
                        score *= 1.2
                    elif score < 0:
                        score *= 1.2

            # Momentum
            if 'momentum_5' in df.columns:
                momentum = last_row['momentum_5']
                if momentum > 0.02:  # Рост больше 2%
                    score += 0.3
                elif momentum < -0.02:  # Падение больше 2%
                    score -= 0.3

            # Нормализация и определение сигнала
            if factors > 0:
                normalized_score = score / factors

                # Снижаем пороги для более частых сигналов
                if normalized_score > 0.25:  # Было 0.3
                    prediction_label = 'UP'
                    # Расчёт confidence на основе силы сигнала
                    confidence = min(0.60 + abs(normalized_score) * 0.25, 0.85)
                elif normalized_score < -0.25:  # Было -0.3
                    prediction_label = 'DOWN'
                    confidence = min(0.60 + abs(normalized_score) * 0.25, 0.85)
                else:
                    prediction_label = 'HOLD'
                    confidence = 0.5
            else:
                prediction_label = 'HOLD'
                confidence = 0.5

            # Логируем предсказание для отладки
            if prediction_label != 'HOLD':
                logger.logger.debug(
                    f"ML Prediction: {prediction_label} with confidence {confidence:.2f}",
                    score=score,
                    normalized_score=normalized_score if factors > 0 else 0,
                    rsi=last_row.get('rsi', 'N/A'),
                    bb_percent=last_row.get('bb_percent', 'N/A')
                )

            return {
                'prediction': 1 if prediction_label == 'UP' else (-1 if prediction_label == 'DOWN' else 0),
                'prediction_label': prediction_label,
                'confidence': confidence
            }

        except Exception as e:
            logger.logger.error(f"ML prediction failed: {e}")
            return {
                'prediction': 0,
                'prediction_label': 'HOLD',
                'confidence': 0.5
            }

    async def train(self, df: pd.DataFrame):
        """Обучение модели (заглушка)"""
        logger.logger.info("ML training not implemented - using heuristic predictions")
        self.is_trained = True
        return False

    def get_feature_importance(self) -> Dict[str, float]:
        """Возвращает важность признаков (заглушка)"""
        return {
            'rsi': 0.35,
            'macd': 0.25,
            'bb_percent': 0.20,
            'volume_ratio': 0.10,
            'momentum': 0.10
        }


# Глобальный экземпляр
ml_engine = MLEngine()