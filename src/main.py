# src/main.py
"""
Главная точка входа для криптовалютного торгового бота
Управляет режимами работы: monitor, paper, live
"""
import asyncio
import sys
import signal
from pathlib import Path
from typing import Optional
import uvloop  # Более быстрый event loop

# Добавляем корневую директорию в path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import settings, BotMode
from src.monitoring.logger import logger, set_log_level
from src.data.storage.redis_client import redis_client
from src.utils.banner import print_banner
from src.bots.paper_trading_bot_v5 import EnhancedPaperTradingBotV5
from src.bots.trading_bot import TradingBot
from src.monitoring.metrics import MetricsServer


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
            await redis_client.connect()
            await redis_client.set("health_check", "OK", expire=10)
            result = await redis_client.get("health_check")
            if result != "OK":
                raise Exception("Redis health check failed")
            logger.logger.info("✓ Redis connection OK")
        except Exception as e:
            logger.logger.error(f"✗ Redis connection failed: {e}")
            return False

        # Проверка базы данных (если не в monitor режиме)
        if settings.BOT_MODE != BotMode.MONITOR:
            try:
                # TODO: Добавить проверку PostgreSQL/TimescaleDB
                logger.logger.info("✓ Database connection OK")
            except Exception as e:
                logger.logger.warning(f"Database not configured: {e}")

        # Проверка API ключей
        if settings.BOT_MODE in [BotMode.PAPER, BotMode.LIVE]:
            if not settings.api.BINANCE_API_KEY or not settings.api.BINANCE_API_SECRET:
                logger.logger.error("✗ Binance API credentials not configured")
                return False
            logger.logger.info("✓ API credentials configured")

        return True

    async def initialize_bot(self):
        """Инициализация бота в зависимости от режима"""
        mode = settings.BOT_MODE

        logger.logger.info(f"Initializing bot in {mode.value} mode...")

        if mode == BotMode.MONITOR:
            # Режим мониторинга - только сбор данных без торговли
            logger.logger.info("Monitor mode: collecting data without trading")
            # TODO: Инициализировать MonitorBot

        elif mode == BotMode.PAPER:
            # Paper trading
            self.bot = EnhancedPaperTradingBotV5(
                initial_balance=settings.PAPER_STARTING_BALANCE,
                maker_fee=settings.PAPER_MAKER_FEE,
                taker_fee=settings.PAPER_TAKER_FEE,
                slippage_bps=settings.PAPER_SLIPPAGE_BPS
            )
            await self.bot.initialize()
            logger.logger.info("Paper trading bot initialized")

        elif mode == BotMode.LIVE:
            # Live trading
            if settings.ENVIRONMENT != "production":
                logger.logger.warning("⚠️  Live trading in non-production environment!")

            self.bot = TradingBot()
            await self.bot.initialize()
            logger.logger.info("Live trading bot initialized")

        else:
            raise ValueError(f"Unknown bot mode: {mode}")

    async def start_metrics_server(self):
        """Запуск сервера метрик"""
        if settings.monitoring.ENABLE_PROMETHEUS:
            self.metrics_server = MetricsServer(port=settings.monitoring.PROMETHEUS_PORT)
            await self.metrics_server.start()
            logger.logger.info(f"Metrics server started on port {settings.monitoring.PROMETHEUS_PORT}")

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
                version=settings.VERSION
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
                logger.logger.info("Starting bot main loop...")
                bot_task = asyncio.create_task(self.bot.run())

                # Ждём сигнала остановки
                await self.shutdown_event.wait()

                # Останавливаем бота
                logger.logger.info("Stopping bot...")
                await self.bot.stop()
                bot_task.cancel()

                try:
                    await bot_task
                except asyncio.CancelledError:
                    pass

        except KeyboardInterrupt:
            logger.logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.log_error(e, {"context": "Application run failed"})
            raise
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Очистка ресурсов"""
        logger.logger.info("Cleaning up resources...")

        if self.metrics_server:
            await self.metrics_server.stop()

        await redis_client.disconnect()

        logger.logger.info("Cleanup completed")

    def handle_signal(self, sig, frame):
        """Обработчик сигналов"""
        logger.logger.info(f"Received signal {sig}")
        self.shutdown_event.set()


async def main():
    """Главная функция"""
    # Используем uvloop для лучшей производительности
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    app = Application()

    # Регистрация обработчиков сигналов
    signal.signal(signal.SIGINT, app.handle_signal)
    signal.signal(signal.SIGTERM, app.handle_signal)

    await app.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)