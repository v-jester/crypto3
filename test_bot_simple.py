import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.bots.advanced_paper_bot import AdvancedPaperTradingBot
from src.monitoring.logger import logger


async def test_bot():
    """Простой тест бота"""

    print("\n" + "=" * 60)
    print("ТЕСТ ТОРГОВОГО БОТА")
    print("=" * 60 + "\n")

    # Создаем бота с большим капиталом
    bot = AdvancedPaperTradingBot(
        initial_balance=50000.0,  # $50k для тестирования
        maker_fee=0.001,
        taker_fee=0.001,
        slippage_bps=5.0
    )

    try:
        # Инициализация
        print("1. Инициализация бота...")
        await bot.initialize()
        print("✅ Бот инициализирован\n")

        # Ждем обновления данных
        print("2. Ожидание данных (30 сек)...")
        await asyncio.sleep(30)

        # Проверяем данные
        print("\n3. Проверка загруженных данных:")
        for symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']:
            if symbol in bot.indicators:
                ind = bot.indicators[symbol]
                print(f"\n{symbol}:")
                print(f"  Цена: ${ind.get('price', 0):,.2f}")
                print(f"  RSI: {ind.get('rsi', 0):.2f}")
                print(f"  MACD: {ind.get('macd', 0):.4f}")
            else:
                print(f"\n{symbol}: Нет данных")

        # Форсируем анализ
        print("\n4. Форсированный анализ рынка...")
        await bot._analyze_and_trade()

        # Проверяем позиции
        print(f"\n5. Открытые позиции: {len(bot.positions)}")
        for symbol, pos in bot.positions.items():
            print(f"  {symbol}: {pos.side} @ ${pos.entry_price:.2f}")

        # Ждем немного
        print("\n6. Мониторинг (30 сек)...")
        await asyncio.sleep(30)

        # Финальный статус
        await bot._log_status()

    finally:
        print("\n7. Остановка бота...")
        await bot.stop()
        print("✅ Бот остановлен")


if __name__ == "__main__":
    try:
        asyncio.run(test_bot())
    except KeyboardInterrupt:
        print("\n\nТест прерван пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback

        traceback.print_exc()