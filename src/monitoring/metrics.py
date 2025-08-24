# src/monitoring/metrics.py
"""
Система метрик с Prometheus для мониторинга торгового бота
"""
from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
from prometheus_client.core import CollectorRegistry
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime
from src.monitoring.logger import logger


class MetricsCollector:
    """Сборщик метрик для Prometheus"""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._init_metrics()

    def _init_metrics(self):
        """Инициализация всех метрик"""

        # ============ Информационные метрики ============
        self.bot_info = Info(
            'trading_bot_info',
            'Information about the trading bot',
            registry=self.registry
        )

        # ============ Торговые метрики ============
        self.trades_total = Counter(
            'trades_total',
            'Total number of trades executed',
            ['symbol', 'side', 'status'],
            registry=self.registry
        )

        self.orders_total = Counter(
            'orders_total',
            'Total number of orders placed',
            ['symbol', 'side', 'order_type'],
            registry=self.registry
        )

        self.signals_generated = Counter(
            'signals_generated_total',
            'Total number of trading signals generated',
            ['symbol', 'signal_type'],
            registry=self.registry
        )

        self.signals_executed = Counter(
            'signals_executed_total',
            'Total number of signals that led to trades',
            ['symbol', 'signal_type'],
            registry=self.registry
        )

        # ============ Финансовые метрики ============
        self.portfolio_value = Gauge(
            'portfolio_value_usdt',
            'Current portfolio value in USDT',
            registry=self.registry
        )

        self.pnl_total = Gauge(
            'pnl_total_usdt',
            'Total P&L in USDT',
            registry=self.registry
        )

        self.pnl_realized = Gauge(
            'pnl_realized_usdt',
            'Realized P&L in USDT',
            registry=self.registry
        )

        self.pnl_unrealized = Gauge(
            'pnl_unrealized_usdt',
            'Unrealized P&L in USDT',
            registry=self.registry
        )

        self.win_rate = Gauge(
            'win_rate_percent',
            'Current win rate percentage',
            registry=self.registry
        )

        self.sharpe_ratio = Gauge(
            'sharpe_ratio',
            'Current Sharpe ratio',
            registry=self.registry
        )

        self.max_drawdown = Gauge(
            'max_drawdown_percent',
            'Maximum drawdown percentage',
            registry=self.registry
        )

        self.current_drawdown = Gauge(
            'current_drawdown_percent',
            'Current drawdown percentage',
            registry=self.registry
        )

        # ============ Позиции ============
        self.open_positions = Gauge(
            'open_positions_count',
            'Number of open positions',
            registry=self.registry
        )

        self.position_value = Gauge(
            'position_value_usdt',
            'Value of position in USDT',
            ['symbol'],
            registry=self.registry
        )

        self.position_pnl = Gauge(
            'position_pnl_usdt',
            'P&L of position in USDT',
            ['symbol'],
            registry=self.registry
        )

        # ============ Риск метрики ============
        self.risk_events = Counter(
            'risk_events_total',
            'Total number of risk events',
            ['event_type', 'severity'],
            registry=self.registry
        )

        self.daily_loss = Gauge(
            'daily_loss_percent',
            'Current daily loss percentage',
            registry=self.registry
        )

        self.exposure = Gauge(
            'total_exposure_usdt',
            'Total market exposure in USDT',
            registry=self.registry
        )

        self.var_95 = Gauge(
            'value_at_risk_95',
            '95% Value at Risk',
            registry=self.registry
        )

        # ============ ML метрики ============
        self.ml_predictions = Counter(
            'ml_predictions_total',
            'Total ML predictions made',
            ['model', 'prediction_type'],
            registry=self.registry
        )

        self.ml_accuracy = Gauge(
            'ml_model_accuracy',
            'ML model accuracy',
            ['model'],
            registry=self.registry
        )

        self.ml_confidence = Histogram(
            'ml_prediction_confidence',
            'ML prediction confidence distribution',
            ['model'],
            registry=self.registry
        )

        self.ml_training_time = Histogram(
            'ml_training_duration_seconds',
            'ML model training duration',
            ['model'],
            registry=self.registry
        )

        # ============ Производительность системы ============
        self.api_latency = Histogram(
            'api_latency_seconds',
            'API call latency',
            ['endpoint', 'method'],
            registry=self.registry
        )

        self.websocket_messages = Counter(
            'websocket_messages_total',
            'Total WebSocket messages received',
            ['stream_type'],
            registry=self.registry
        )

        self.data_processing_time = Histogram(
            'data_processing_seconds',
            'Data processing time',
            ['operation'],
            registry=self.registry
        )

        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )

        self.cpu_usage = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )

        # ============ Ошибки и предупреждения ============
        self.errors_total = Counter(
            'errors_total',
            'Total number of errors',
            ['error_type', 'component'],
            registry=self.registry
        )

        self.warnings_total = Counter(
            'warnings_total',
            'Total number of warnings',
            ['warning_type', 'component'],
            registry=self.registry
        )

        # ============ Рыночные данные ============
        self.market_price = Gauge(
            'market_price_usdt',
            'Current market price',
            ['symbol'],
            registry=self.registry
        )

        self.market_volume = Gauge(
            'market_volume_24h',
            '24h market volume',
            ['symbol'],
            registry=self.registry
        )

        self.market_volatility = Gauge(
            'market_volatility',
            'Market volatility',
            ['symbol', 'timeframe'],
            registry=self.registry
        )

    # ============ Методы обновления метрик ============

    def update_bot_info(self, info: Dict[str, str]):
        """Обновление информации о боте"""
        self.bot_info.info(info)

    def record_trade(self, symbol: str, side: str, status: str = "success"):
        """Запись сделки"""
        self.trades_total.labels(symbol=symbol, side=side, status=status).inc()

    def record_order(self, symbol: str, side: str, order_type: str):
        """Запись ордера"""
        self.orders_total.labels(
            symbol=symbol,
            side=side,
            order_type=order_type
        ).inc()

    def record_signal(self, symbol: str, signal_type: str, executed: bool = False):
        """Запись сигнала"""
        self.signals_generated.labels(
            symbol=symbol,
            signal_type=signal_type
        ).inc()

        if executed:
            self.signals_executed.labels(
                symbol=symbol,
                signal_type=signal_type
            ).inc()

    def update_portfolio_metrics(self, metrics: Dict[str, float]):
        """Обновление метрик портфеля"""
        if 'portfolio_value' in metrics:
            self.portfolio_value.set(metrics['portfolio_value'])
        if 'pnl_total' in metrics:
            self.pnl_total.set(metrics['pnl_total'])
        if 'pnl_realized' in metrics:
            self.pnl_realized.set(metrics['pnl_realized'])
        if 'pnl_unrealized' in metrics:
            self.pnl_unrealized.set(metrics['pnl_unrealized'])
        if 'win_rate' in metrics:
            self.win_rate.set(metrics['win_rate'])
        if 'sharpe_ratio' in metrics:
            self.sharpe_ratio.set(metrics['sharpe_ratio'])
        if 'max_drawdown' in metrics:
            self.max_drawdown.set(metrics['max_drawdown'])
        if 'current_drawdown' in metrics:
            self.current_drawdown.set(metrics['current_drawdown'])

    def update_position_metrics(self, symbol: str, value: float, pnl: float):
        """Обновление метрик позиции"""
        self.position_value.labels(symbol=symbol).set(value)
        self.position_pnl.labels(symbol=symbol).set(pnl)

    def record_risk_event(self, event_type: str, severity: str):
        """Запись риск-события"""
        self.risk_events.labels(
            event_type=event_type,
            severity=severity
        ).inc()

    def record_ml_prediction(
            self,
            model: str,
            prediction_type: str,
            confidence: float
    ):
        """Запись ML предсказания"""
        self.ml_predictions.labels(
            model=model,
            prediction_type=prediction_type
        ).inc()
        self.ml_confidence.labels(model=model).observe(confidence)

    def record_api_latency(self, endpoint: str, method: str, latency: float):
        """Запись задержки API"""
        self.api_latency.labels(
            endpoint=endpoint,
            method=method
        ).observe(latency)

    def record_error(self, error_type: str, component: str):
        """Запись ошибки"""
        self.errors_total.labels(
            error_type=error_type,
            component=component
        ).inc()

    def update_market_data(self, symbol: str, price: float, volume: float = None):
        """Обновление рыночных данных"""
        self.market_price.labels(symbol=symbol).set(price)
        if volume:
            self.market_volume.labels(symbol=symbol).set(volume)

    def update_system_metrics(self, cpu: float, memory: float):
        """Обновление системных метрик"""
        self.cpu_usage.set(cpu)
        self.memory_usage.set(memory)


class MetricsServer:
    """HTTP сервер для Prometheus метрик"""

    def __init__(self, port: int = 8000):
        self.port = port
        self.collector = MetricsCollector()
        self.server_task = None

    async def start(self):
        """Запуск HTTP сервера метрик"""
        try:
            # Запуск в отдельном потоке
            await asyncio.get_event_loop().run_in_executor(
                None,
                start_http_server,
                self.port,
                registry=self.collector.registry
            )

            logger.logger.info(f"Metrics server started on port {self.port}")

            # Установка начальной информации
            self.collector.update_bot_info({
                'version': '1.0.0',
                'environment': 'development',
                'started_at': datetime.utcnow().isoformat()
            })

        except Exception as e:
            logger.logger.error(f"Error: {e}, context: {"context": "Failed to start metrics server"}")
            raise

    async def stop(self):
        """Остановка сервера"""
        if self.server_task:
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass

        logger.logger.info("Metrics server stopped")

    def get_collector(self) -> MetricsCollector:
        """Получение коллектора метрик"""
        return self.collector


# Глобальный экземпляр
metrics_collector = MetricsCollector()


# Вспомогательные функции для быстрого доступа
def record_trade(symbol: str, side: str, status: str = "success"):
    """Быстрая запись сделки"""
    metrics_collector.record_trade(symbol, side, status)


def record_signal(symbol: str, signal_type: str, executed: bool = False):
    """Быстрая запись сигнала"""
    metrics_collector.record_signal(symbol, signal_type, executed)


def update_portfolio_value(value: float):
    """Быстрое обновление стоимости портфеля"""
    metrics_collector.portfolio_value.set(value)


def record_error(error_type: str, component: str):
    """Быстрая запись ошибки"""
    metrics_collector.record_error(error_type, component)
