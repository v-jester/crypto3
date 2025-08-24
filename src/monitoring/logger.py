# src/monitoring/logger.py
"""
Продвинутая система логирования с структурированными логами,
производительностью и торговыми событиями
"""
import logging
import time
import json
import sys
from pathlib import Path
from datetime import datetime
from functools import wraps
from typing import Any, Dict, Optional, Callable
from contextlib import contextmanager
import structlog
from pythonjsonlogger import jsonlogger


class TradingLogger:
    """Специализированный логгер для торговой системы"""

    def __init__(self, name: str = "trading_bot"):
        self.name = name
        self._setup_directories()
        self._setup_structured_logging()
        self.logger = structlog.get_logger(name)

    def _setup_directories(self):
        """Создание директорий для логов"""
        log_dirs = ["logs", "logs/trading", "logs/system", "logs/errors", "logs/performance"]
        for dir_path in log_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def _setup_structured_logging(self):
        """Настройка структурированного логирования"""
        # Процессоры для structlog
        processors = [
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                ]
            ),
            structlog.processors.dict_tracebacks,
            structlog.dev.ConsoleRenderer() if sys.stdout.isatty() else structlog.processors.JSONRenderer(),
        ]

        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        # Настройка стандартного логгера
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=logging.INFO,
        )

    def _get_file_handler(self, filename: str, level=logging.INFO) -> logging.FileHandler:
        """Создание файлового обработчика с JSON форматированием"""
        handler = logging.FileHandler(filename)
        handler.setLevel(level)

        # JSON formatter
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        return handler

    def log_trade(
            self,
            action: str,
            symbol: str,
            side: str,
            price: float,
            quantity: float,
            order_id: Optional[str] = None,
            pnl: Optional[float] = None,
            **kwargs
    ):
        """Логирование торговых операций"""
        trade_data = {
            "event": "trade",
            "action": action,
            "symbol": symbol,
            "side": side,
            "price": price,
            "quantity": quantity,
            "order_id": order_id,
            "pnl": pnl,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }

        # Логируем в основной лог
        self.logger.info("Trade executed", **trade_data)

        # Дополнительно сохраняем в отдельный файл сделок
        try:
            with open("logs/trading/trades.jsonl", "a") as f:
                f.write(json.dumps(trade_data) + "\n")
        except Exception as e:
            self.logger.warning(f"Failed to write trade to file: {e}")

    def log_signal(
            self,
            symbol: str,
            signal_type: str,
            confidence: float,
            entry_price: float,
            stop_loss: Optional[float] = None,
            take_profit: Optional[float] = None,
            metadata: Optional[Dict] = None,
            **kwargs
    ):
        """Логирование торговых сигналов"""
        signal_data = {
            "event": "signal",
            "symbol": symbol,
            "signal_type": signal_type,
            "confidence": confidence,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }

        self.logger.info("Signal generated", **signal_data)

        # Сохраняем историю сигналов
        try:
            with open("logs/trading/signals.jsonl", "a") as f:
                f.write(json.dumps(signal_data) + "\n")
        except Exception as e:
            self.logger.warning(f"Failed to write signal to file: {e}")

    def log_risk_event(
            self,
            event_type: str,
            severity: str,
            message: str,
            current_drawdown: Optional[float] = None,
            current_exposure: Optional[float] = None,
            **kwargs
    ):
        """Логирование риск-событий"""
        risk_data = {
            "event": "risk",
            "event_type": event_type,
            "severity": severity,
            "message": message,
            "current_drawdown": current_drawdown,
            "current_exposure": current_exposure,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }

        log_method = getattr(self.logger, severity.lower(), self.logger.warning)
        log_method("Risk event", **risk_data)

        # Критические риск-события в отдельный файл
        if severity in ["ERROR", "CRITICAL"]:
            try:
                with open("logs/errors/risk_events.jsonl", "a") as f:
                    f.write(json.dumps(risk_data) + "\n")
            except Exception as e:
                self.logger.warning(f"Failed to write risk event to file: {e}")

    def log_performance(
            self,
            metric: str,
            value: float,
            unit: str = "",
            **kwargs
    ):
        """Логирование метрик производительности"""
        perf_data = {
            "event": "performance",
            "metric": metric,
            "value": value,
            "unit": unit,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }

        self.logger.debug("Performance metric", **perf_data)

        # Сохраняем метрики производительности
        try:
            with open("logs/performance/metrics.jsonl", "a") as f:
                f.write(json.dumps(perf_data) + "\n")
        except Exception as e:
            self.logger.warning(f"Failed to write performance metric to file: {e}")

    def log_error(self, error: Exception, context: Optional[Dict] = None):
        """Логирование ошибок с контекстом - ИСПРАВЛЕННЫЙ МЕТОД"""
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Добавляем контекст если есть
        if context:
            error_data.update(context)

        # Логируем ошибку
        self.logger.error("Error occurred", exc_info=True, **error_data)

        # Сохраняем в файл ошибок
        try:
            with open("logs/errors/errors.jsonl", "a") as f:
                f.write(json.dumps(error_data) + "\n")
        except Exception as write_error:
            self.logger.warning(f"Failed to write error to file: {write_error}")


# Глобальный экземпляр логгера
logger = TradingLogger()


# Декораторы для логирования
def log_performance(name: str = None):
    """
    Декоратор для логирования производительности функций

    Usage:
        @log_performance("fetch_market_data")
        async def fetch_data():
            ...
    """

    def decorator(func: Callable) -> Callable:
        func_name = name or func.__name__

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.perf_counter() - start_time
                logger.log_performance(
                    metric=f"{func_name}_execution_time",
                    value=execution_time,
                    unit="seconds",
                    status="success"
                )
                return result
            except Exception as e:
                execution_time = time.perf_counter() - start_time
                logger.log_performance(
                    metric=f"{func_name}_execution_time",
                    value=execution_time,
                    unit="seconds",
                    status="error",
                    error=str(e)
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                execution_time = time.perf_counter() - start_time
                logger.log_performance(
                    metric=f"{func_name}_execution_time",
                    value=execution_time,
                    unit="seconds",
                    status="success"
                )
                return result
            except Exception as e:
                execution_time = time.perf_counter() - start_time
                logger.log_performance(
                    metric=f"{func_name}_execution_time",
                    value=execution_time,
                    unit="seconds",
                    status="error",
                    error=str(e)
                )
                raise

        # Возвращаем правильную обёртку в зависимости от типа функции
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def log_exceptions(func: Callable) -> Callable:
    """
    Декоратор для логирования исключений

    Usage:
        @log_exceptions
        async def risky_operation():
            ...
    """

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.log_error(
                e,
                context={
                    "function": func.__name__,
                    "args": str(args)[:200],  # Ограничиваем размер
                    "kwargs": str(kwargs)[:200]
                }
            )
            raise

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.log_error(
                e,
                context={
                    "function": func.__name__,
                    "args": str(args)[:200],
                    "kwargs": str(kwargs)[:200]
                }
            )
            raise

    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


@contextmanager
def log_context(**kwargs):
    """
    Контекстный менеджер для добавления контекста к логам

    Usage:
        with log_context(user_id=123, operation="place_order"):
            logger.logger.info("Processing order")
    """
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(**kwargs)
    try:
        yield
    finally:
        structlog.contextvars.clear_contextvars()


# Экспортируемые функции для обратной совместимости
def log_trade(**kwargs):
    """Обёртка для логирования сделок"""
    logger.log_trade(**kwargs)


def log_signal(**kwargs):
    """Обёртка для логирования сигналов"""
    logger.log_signal(**kwargs)


def log_risk_event(**kwargs):
    """Обёртка для логирования риск-событий"""
    logger.log_risk_event(**kwargs)


def log_performance_metric(metric: str, value: float, **kwargs):
    """Обёртка для логирования метрик"""
    logger.log_performance(metric, value, **kwargs)


# Установка уровня логирования
def set_log_level(level: str):
    """Установить уровень логирования"""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.getLogger().setLevel(numeric_level)