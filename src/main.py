"""
Main entrypoint for the Crypto Trading Bot v3.0
With fixed metrics tracking and improved monitoring
"""

# --- PATH FIX: позволяет запускать `python src/main.py` из корня проекта ---
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ---------------------------------------------------------------------------

import asyncio
import signal
import platform
from datetime import datetime, timedelta, timezone

from src.monitoring.logger import logger
from src.config.settings import settings
from src.data.storage.redis_client import redis_client, close_redis
from src.bots.advanced_paper_bot import AdvancedPaperTradingBot

# Опциональные метрики
try:
    from src.monitoring.metrics import start as start_metrics_server, stop as stop_metrics_server
    METRICS_AVAILABLE = True
except Exception:
    METRICS_AVAILABLE = False


async def check_environment():
    """Расширенная проверка окружения с рекомендациями"""
    logger.logger.info("=" * 80)
    logger.logger.info("ENVIRONMENT CHECK v3.0")
    logger.logger.info("=" * 80)

    checks_passed = 0
    checks_total = 0

    # 1) Redis
    checks_total += 1
    try:
        await redis_client.connect()
        await redis_client.set("health_check", "OK", expire=10)
        result = await redis_client.get("health_check")
        if result == "OK":
            logger.logger.info("✅ Redis connection OK")
            checks_passed += 1
        else:
            logger.logger.warning("⚠️ Redis test value mismatch (will continue without caching)")
    except Exception as e:
        logger.logger.warning(f"⚠️ Redis connection failed: {e}")
        logger.logger.info("   → Bot will work without caching (may affect performance)")

    # 2) Database
    checks_total += 1
    try:
        if hasattr(settings.database, 'postgres_url'):
            logger.logger.info("✅ Database configuration found")
            checks_passed += 1
        else:
            logger.logger.info("ℹ️ Database not configured (using in-memory storage)")
    except Exception:
        logger.logger.info("ℹ️ Database not configured (using in-memory storage)")

    # 3) API credentials
    checks_total += 1
    if settings.api.BINANCE_API_KEY and settings.api.BINANCE_API_SECRET:
        logger.logger.info("✅ API credentials configured")
        if settings.api.TESTNET:
            logger.logger.info("   → Using TESTNET mode (paper trading)")
        else:
            logger.logger.info("   → Using LIVE mode")
        checks_passed += 1
    else:
        logger.logger.warning("⚠️ API credentials missing")
        logger.logger.info("   → Add BINANCE_API_KEY and BINANCE_API_SECRET to .env file")

    # 4) Trading configuration
    checks_total += 1
    logger.logger.info("✅ Trading configuration:")
    logger.logger.info(f"   → Symbols: {', '.join(settings.trading.SYMBOLS[:3])}")
    logger.logger.info(f"   → Timeframe: {settings.trading.PRIMARY_TIMEFRAME}")
    logger.logger.info(f"   → Initial capital: ${settings.trading.INITIAL_CAPITAL:,.2f}")
    logger.logger.info(f"   → Max positions: {settings.trading.MAX_POSITIONS}")
    logger.logger.info(f"   → Max drawdown: {settings.trading.MAX_DRAWDOWN_PERCENT}%")
    checks_passed += 1

    # 5) Python packages
    checks_total += 1
    packages_ok = True
    required_packages = {
        'binance': 'python-binance',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'pandas_ta': 'pandas-ta',
        'asyncpg': 'asyncpg (optional)',
        'redis': 'redis'
    }

    missing_packages = []
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing_packages.append(package)
            packages_ok = False

    if packages_ok:
        logger.logger.info("✅ All required Python packages installed")
        checks_passed += 1
    else:
        logger.logger.warning(f"⚠️ Missing packages: {', '.join(missing_packages)}")
        logger.logger.info("   → Install with: pip install " + " ".join(missing_packages))

    # 6) ML models
    checks_total += 1
    try:
        from src.ml.models.ml_engine import ml_engine
        logger.logger.info("✅ ML engine available")
        checks_passed += 1
    except Exception:
        logger.logger.info("ℹ️ ML engine not available (bot will work without ML signals)")

    # Summary
    logger.logger.info("=" * 80)
    logger.logger.info(f"Environment check complete: {checks_passed}/{checks_total} checks passed")

    if checks_passed < 3:
        logger.logger.warning("⚠️ Some critical components are missing. Bot may not work properly.")
        logger.logger.info("Continuing anyway... Press Ctrl+C to stop.")
    else:
        logger.logger.info("✅ Environment is ready for trading!")

    logger.logger.info("=" * 80)


async def start_metrics():
    """Запуск сервера метрик с улучшенным логированием"""
    if not METRICS_AVAILABLE:
        return
    try:
        port = 8000
        await start_metrics_server(port=port)
        logger.logger.info("✅ Metrics server started")
        logger.logger.info(f"   → Dashboard: http://localhost:{port}/")
        logger.logger.info(f"   → Metrics endpoint: http://localhost:{port}/metrics")
    except Exception as e:
        logger.logger.warning(f"⚠️ Metrics server failed to start: {e}")


async def initialize_bot() -> AdvancedPaperTradingBot:
    """Инициализация бота с расширенным логированием"""
    logger.logger.info("=" * 80)
    logger.logger.info("INITIALIZING ADVANCED PAPER TRADING BOT v3.0")
    logger.logger.info("=" * 80)

    logger.logger.info("Critical fixes in v3.0:")
    logger.logger.info("  ✅ FIXED: Filters now apply only to real signals (not HOLD)")
    logger.logger.info("  ✅ FIXED: Proper metric counting logic")
    logger.logger.info("  ✅ FIXED: Correlation checks only opposite directions")
    logger.logger.info("")
    logger.logger.info("Optimized parameters:")
    logger.logger.info("  • Confidence threshold: 0.65 (reduced from 0.75)")
    logger.logger.info("  • Signals required: 2.0 (reduced from 2.5)")
    logger.logger.info("  • RSI levels: 30/70 (standard)")
    logger.logger.info("  • Time between trades: 30 minutes (reduced from 2 hours)")
    logger.logger.info("  • Adaptive volatility thresholds by symbol")
    logger.logger.info("  • Position size: 5% (safe sizing)")
    logger.logger.info("  • Partial profit taking at 1.5*ATR")
    logger.logger.info("  • Trailing stop-loss implementation")

    bot = AdvancedPaperTradingBot(
        initial_balance=settings.trading.INITIAL_CAPITAL
    )
    await bot.initialize()

    # Показываем адаптивные пороги волатильности
    logger.logger.info("")
    logger.logger.info("Adaptive volatility thresholds:")
    for symbol in settings.trading.SYMBOLS[:3]:
        threshold = bot.get_adaptive_volatility_threshold(symbol)
        logger.logger.info(f"  • {symbol}: {threshold:.2f}% minimum ATR")

    logger.logger.info("=" * 80)
    logger.logger.info("✅ Bot initialized successfully!")
    logger.logger.info("=" * 80)

    return bot


async def periodic_summary(bot: AdvancedPaperTradingBot, interval_minutes: int = 30):
    """Периодический отчёт о производительности с исправленными метриками"""
    while bot.running:
        await asyncio.sleep(interval_minutes * 60)

        if not bot.running:
            break

        # Расчёт метрик
        all_trades = [t for t in bot.trade_history if t.get('type') != 'partial_close']
        if not all_trades and bot.performance_metrics['total_signals'] == 0:
            continue

        now = datetime.now(timezone.utc)
        recent_trades = [
            t for t in all_trades
            if (now - t['timestamp'].replace(tzinfo=timezone.utc)).total_seconds() < interval_minutes * 60
        ] if all_trades else []

        logger.logger.info("=" * 80)
        logger.logger.info(f"📊 {interval_minutes}-MINUTE SUMMARY")
        logger.logger.info("=" * 80)

        # Анализ сигналов
        if bot.performance_metrics['total_analyses'] > 0:
            signal_rate = bot.performance_metrics['total_signals'] / bot.performance_metrics['total_analyses'] * 100
            logger.logger.info(
                f"Market Analysis: {bot.performance_metrics['total_analyses']} scans | "
                f"Signal Rate: {signal_rate:.1f}% | "
                f"Signals: {bot.performance_metrics['total_signals']} | "
                f"HOLD: {bot.performance_metrics['hold_signals']}"
            )

        # Торговая активность
        if recent_trades:
            recent_pnl = sum(t['pnl'] for t in recent_trades)
            wins = [t for t in recent_trades if t['pnl'] > 0]
            win_rate = len(wins) / len(recent_trades) * 100 if recent_trades else 0

            logger.logger.info(
                f"Trades in period: {len(recent_trades)} | "
                f"PnL: ${recent_pnl:.2f} | "
                f"Win rate: {win_rate:.1f}%"
            )
        else:
            logger.logger.info("No trades executed in this period")

        # Фильтры
        if bot.performance_metrics['total_signals'] > 0:
            exec_rate = bot.performance_metrics['executed_signals'] / bot.performance_metrics['total_signals'] * 100
            logger.logger.info(
                f"Filter Performance: Execution Rate {exec_rate:.1f}% | "
                f"Top rejection: Volatility={bot.performance_metrics['skipped_low_volatility']}, "
                f"Time={bot.performance_metrics['skipped_time_limit']}"
            )

        logger.logger.info("=" * 80)


async def run():
    """Основная функция запуска с улучшенным управлением"""
    if platform.system() == "Windows":
        logger.logger.info("Detected Windows OS - using default event loop")

    # Время запуска
    start_time = datetime.now(timezone.utc)
    logger.logger.info(f"Bot starting at {start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    await check_environment()

    if METRICS_AVAILABLE:
        await start_metrics()

    bot = await initialize_bot()

    logger.logger.info("=" * 80)
    logger.logger.info("🚀 STARTING MAIN TRADING LOOP v3.0")
    logger.logger.info("=" * 80)
    logger.logger.info("")
    logger.logger.info("Bot is now actively monitoring markets...")
    logger.logger.info("Signals will be generated based on technical indicators.")
    logger.logger.info("Filters will be applied only to actionable signals (BUY/SELL).")
    logger.logger.info("")
    logger.logger.info("Expected behavior:")
    logger.logger.info("  • Market scan every 60 seconds")
    logger.logger.info("  • Signals require min 2.0 indicator confirmations")
    logger.logger.info("  • Filters check: confidence, volatility, volume, time, correlation")
    logger.logger.info("  • Positions: 5% of balance, max 5 concurrent")
    logger.logger.info("  • Risk management: SL at 2*ATR, TP at 3*ATR")
    logger.logger.info("")
    logger.logger.info("Commands:")
    logger.logger.info("  • Press Ctrl+C to stop the bot gracefully")
    logger.logger.info("  • Check logs for detailed trading activity")
    if METRICS_AVAILABLE:
        logger.logger.info("  • View metrics at http://localhost:8000/")
    logger.logger.info("")
    logger.logger.info("Happy trading! 🎯")
    logger.logger.info("=" * 80)

    # Грейсфул шутдаун
    stop_event = asyncio.Event()

    def _handle_sig(*_):
        logger.logger.info("")
        logger.logger.info("=" * 80)
        logger.logger.info("📛 SHUTDOWN SIGNAL RECEIVED")
        logger.logger.info("=" * 80)
        logger.logger.info("Closing positions and saving state...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, _handle_sig)
        except NotImplementedError:
            # Windows не поддерживает add_signal_handler
            signal.signal(sig, _handle_sig)

    # Запускаем основные задачи
    bot_task = asyncio.create_task(bot.run())
    summary_task = asyncio.create_task(periodic_summary(bot, interval_minutes=30))

    # Ждём сигнал остановки
    await stop_event.wait()

    # Останавливаем бота
    logger.logger.info("Stopping bot...")
    try:
        await bot.stop()
    except Exception as e:
        logger.logger.warning(f"Error while stopping bot: {e}")

    # Останавливаем периодические задачи
    summary_task.cancel()
    try:
        await summary_task
    except asyncio.CancelledError:
        pass

    # Останавливаем метрики
    if METRICS_AVAILABLE:
        try:
            await stop_metrics_server()
            logger.logger.info("✅ Metrics server stopped")
        except Exception:
            pass

    # Закрываем Redis
    try:
        await close_redis()
        logger.logger.info("✅ Redis connection closed")
    except Exception as e:
        logger.logger.warning(f"Error closing Redis connection: {e}")

    # Отменяем основную задачу бота
    if not bot_task.done():
        bot_task.cancel()
        try:
            await bot_task
        except asyncio.CancelledError:
            logger.logger.info("✅ Trading loop cancelled")

    # Финальная статистика
    end_time = datetime.now(timezone.utc)
    runtime = end_time - start_time
    hours = runtime.total_seconds() / 3600

    logger.logger.info("=" * 80)
    logger.logger.info("✅ BOT STOPPED SUCCESSFULLY")
    logger.logger.info("=" * 80)
    logger.logger.info(f"Total runtime: {hours:.1f} hours")
    logger.logger.info(f"Stopped at: {end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    # Финальная сводка по метрикам
    if bot.performance_metrics['total_analyses'] > 0:
        logger.logger.info("")
        logger.logger.info("Session Summary:")
        logger.logger.info(f"  • Market analyses: {bot.performance_metrics['total_analyses']}")
        logger.logger.info(f"  • Trading signals: {bot.performance_metrics['total_signals']}")
        logger.logger.info(f"  • Trades executed: {bot.performance_metrics['executed_signals']}")

        if bot.performance_metrics['total_signals'] > 0:
            exec_rate = bot.performance_metrics['executed_signals'] / bot.performance_metrics['total_signals'] * 100
            logger.logger.info(f"  • Execution rate: {exec_rate:.1f}%")

    logger.logger.info("=" * 80)
    logger.logger.info("Thank you for using Advanced Paper Trading Bot v3.0!")
    logger.logger.info("=" * 80)


def main():
    """Точка входа с обработкой исключений"""
    try:
        print("\n" + "=" * 80)
        print("ADVANCED PAPER TRADING BOT v3.0")
        print("With Critical Bug Fixes")
        print("=" * 80)
        print("Starting async event loop...")
        print("")

        asyncio.run(run())

    except KeyboardInterrupt:
        print("\n✅ Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        raise
    finally:
        print("\nGoodbye! 👋")


if __name__ == "__main__":
    main()