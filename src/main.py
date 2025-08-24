# src/main.py
"""
Главная точка входа для криптовалютного торгового бота
Управляет режимами работы: monitor, paper, live
"""
import asyncio
import sys
import signal
import platform
from pathlib import Path
from typing import Optional

# Добавляем корневую директорию в path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Импортируем uvloop только на Unix системах
try:
    if platform.system() != "Windows":
        import uvloop

        USE_UVLOOP = True
    else:
        USE_UVLOOP = False
except ImportError:
    USE_UVLOOP = False

from src.config.settings import settings, BotMode
from src.monitoring.logger import logger, set_log_level
from src.data.storage.redis_client import init_redis
from src.utils.banner import print_banner

# Импортируем боты только при необходимости
try:
    from src.bots.paper_trading_bot_v5 import EnhancedPaperTradingBotV5
except ImportError:
    EnhancedPaperTradingBotV5 = None

try:
    from src.bots.trading_bot import TradingBot
except ImportError:
    TradingBot = None

try:
    from src.monitoring.metrics import MetricsServer
except ImportError:
    MetricsServer = None


class Application:
    """Главное приложение"""

    def __init__(self):
        self.bot = None
        self.metrics_server = None
        self.shutdown_event = asyncio.Event()

    async def check_environment(self) -> bool:
        """Проверка окружения и зависимостей"""
        logger.logger.info("Checking environment...")

        # Проверка Redis
        try:
            redis_url = settings.database.redis_url
            await init_redis(redis_url)

            # Простая проверка Redis
            from src.data.storage.redis_client import redis_client
            await redis_client.set("health_check", "OK", expire=10)
            result = await redis_client.get("health_check")

            if result != "OK":
                logger.logger.error("✗ Redis health check failed")
                return False

            logger.logger.info("✅ Redis connection OK")

        except Exception as e:
            logger.logger.warning(f"⚠️ Redis connection failed: {e}")
            logger.logger.info("Will continue without Redis caching")

        # Проверка базы данных (если не в monitor режиме)
        if settings.BOT_MODE != BotMode.MONITOR:
            try:
                # TODO: Добавить проверку PostgreSQL/TimescaleDB
                logger.logger.info("✅ Database connection OK (not implemented)")
            except Exception as e:
                logger.logger.warning(f"Database not configured: {e}")

        # Проверка API ключей
        if settings.BOT_MODE in [BotMode.PAPER, BotMode.LIVE]:
            api_key = settings.api.BINANCE_API_KEY.get_secret_value() if settings.api.BINANCE_API_KEY else None
            api_secret = settings.api.BINANCE_API_SECRET.get_secret_value() if settings.api.BINANCE_API_SECRET else None

            if not api_key or not api_secret:
                logger.logger.error("✗ Binance API credentials not configured")
                return False
            logger.logger.info("✅ API credentials configured")

        return True

    async def initialize_bot(self):
        """Инициализация бота в зависимости от режима"""
        mode = settings.BOT_MODE

        logger.logger.info(f"Initializing bot in {mode.value} mode...")

        if mode == BotMode.MONITOR:
            # Режим мониторинга - только сбор данных без торговли
            logger.logger.info("Monitor mode: collecting data without trading")
            # TODO: Инициализировать MonitorBot
            logger.logger.warning("Monitor mode not implemented yet")

        elif mode == BotMode.PAPER:
            # Paper trading
            if EnhancedPaperTradingBotV5 is None:
                raise ImportError("Paper trading bot not available")

            self.bot = EnhancedPaperTradingBotV5(
                initial_balance=settings.PAPER_STARTING_BALANCE,
                maker_fee=settings.PAPER_MAKER_FEE,
                taker_fee=settings.PAPER_TAKER_FEE,
                slippage_bps=settings.PAPER_SLIPPAGE_BPS
            )

            try:
                await self.bot.initialize()
                logger.logger.info("✅ Paper trading bot initialized")
            except Exception as e:
                logger.logger.error(f"Failed to initialize paper trading bot: {e}")
                raise

        elif mode == BotMode.LIVE:
            # Live trading
            if settings.ENVIRONMENT.value != "production":
                logger.logger.warning("⚠️ Live trading in non-production environment!")

            if TradingBot is None:
                raise ImportError("Live trading bot not available")

            self.bot = TradingBot()
            await self.bot.initialize()
            logger.logger.info("✅ Live trading bot initialized")

        else:
            raise ValueError(f"Unknown bot mode: {mode}")

    async def start_metrics_server(self):
        """Запуск сервера метрик"""
        if settings.monitoring.ENABLE_PROMETHEUS and MetricsServer:
            try:
                self.metrics_server = MetricsServer(port=settings.monitoring.PROMETHEUS_PORT)
                await self.metrics_server.start()
                logger.logger.info(f"✅ Metrics server started on port {settings.monitoring.PROMETHEUS_PORT}")
            except Exception as e:
                logger.logger.warning(f"Failed to start metrics server: {e}")
        else:
            logger.logger.info("Metrics server disabled or not available")

    async def run(self):
        """Основной цикл выполнения"""
        try:
            # Печать баннера
            print_banner()

            # Установка уровня логирования
            set_log_level(settings.monitoring.LOG_LEVEL)

            # Информация о конфигурации
            logger.logger.info(
                "Starting Crypto Trading Bot",
                environment=settings.ENVIRONMENT.value,
                mode=settings.BOT_MODE.value,
                trading_mode=settings.TRADING_MODE.value,
                version=settings.VERSION,
                platform=platform.system()
            )

            # Проверка окружения
            if not await self.check_environment():
                logger.logger.error("Environment check failed. Exiting...")
                sys.exit(1)

            # Запуск метрик
            await self.start_metrics_server()

            # Инициализация бота
            await self.initialize_bot()

            # Запуск бота
            if self.bot:
                logger.logger.info("🚀 Starting bot main loop...")

                # Создаем задачу для бота
                bot_task = asyncio.create_task(self.bot.run())

                # Ждём сигнала остановки или завершения бота
                done, pending = await asyncio.wait(
                    [bot_task, asyncio.create_task(self.shutdown_event.wait())],
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Останавливаем бота
                logger.logger.info("🛑 Stopping bot...")

                if hasattr(self.bot, 'stop'):
                    await self.bot.stop()

                # Отменяем незавершённые задачи
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            else:
                # Если бот не инициализирован, просто ждём сигнала
                logger.logger.info("No bot initialized, waiting for shutdown signal...")
                await self.shutdown_event.wait()

        except KeyboardInterrupt:
            logger.logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.logger.error(f"Application run failed: {e}")
            logger.log_error(e, {"context": "Application run failed"})
            raise
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Очистка ресурсов"""
        logger.logger.info("🧹 Cleaning up resources...")

        if self.metrics_server:
            try:
                await self.metrics_server.stop()
            except Exception as e:
                logger.logger.warning(f"Error stopping metrics server: {e}")

        # Закрываем Redis соединение
        try:
            from src.data.storage.redis_client import close_redis
            await close_redis()
        except Exception as e:
            logger.logger.warning(f"Error closing Redis connection: {e}")

        logger.logger.info("✅ Cleanup completed")

    def handle_signal(self, sig, frame):
        """Обработчик сигналов"""
        logger.logger.info(f"Received signal {sig}")

        # Устанавливаем событие остановки
        if not self.shutdown_event.is_set():
            asyncio.create_task(self._set_shutdown_event())

    async def _set_shutdown_event(self):
        """Установка события остановки"""
        self.shutdown_event.set()


def setup_signal_handlers(app: Application):
    """Настройка обработчиков сигналов"""
    try:
        # В Windows доступны только SIGINT и SIGTERM
        signal.signal(signal.SIGINT, app.handle_signal)

        if platform.system() != "Windows":
            signal.signal(signal.SIGTERM, app.handle_signal)

    except Exception as e:
        logger.logger.warning(f"Failed to setup signal handlers: {e}")


async def main():
    """Главная функция"""
    # Используем uvloop только если доступен (Unix системы)
    if USE_UVLOOP:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        logger.logger.info("Using uvloop for better performance")
    else:
        logger.logger.info(f"Using default event loop on {platform.system()}")

    app = Application()

    # Регистрация обработчиков сигналов
    setup_signal_handlers(app)

    try:
        await app.run()
    except Exception as e:
        logger.logger.error(f"Fatal error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        if exit_code == 0:
            print("\n✅ Shutdown complete")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n✅ Shutdown complete")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)