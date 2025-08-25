# test_real_balance.py
"""
Проверка реального баланса с учётом позиций
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


async def check_real_balance():
    from src.bots.advanced_paper_bot import AdvancedPaperTradingBot

    print("=" * 60)
    print("ПРОВЕРКА РАСЧЁТА БАЛАНСА")
    print("=" * 60)

    # Создаём бота
    bot = AdvancedPaperTradingBot(
        initial_balance=50000.0,
        maker_fee=0.001,
        taker_fee=0.001,
        slippage_bps=5.0
    )

    # Инициализируем
    await bot.initialize()

    print(f"\n1. Начальное состояние:")
    print(f"   Начальный капитал: ${bot.initial_balance:,.2f}")
    print(f"   Текущий баланс: ${bot.current_balance:,.2f}")
    print(f"   Позиций: {len(bot.positions)}")

    # Ждём 30 секунд для загрузки данных
    print(f"\n2. Ожидание данных (30 сек)...")
    await asyncio.sleep(30)

    # Форсируем анализ для открытия позиций
    print(f"\n3. Анализ рынка...")
    await bot._analyze_and_trade()

    # Показываем результат
    print(f"\n4. После открытия позиций:")
    print(f"   Текущий баланс: ${bot.current_balance:,.2f}")
    print(f"   Позиций открыто: {len(bot.positions)}")

    # Рассчитываем полный капитал
    positions_value = 0
    for symbol, pos in bot.positions.items():
        value = pos.quantity * pos.entry_price
        positions_value += value
        print(f"   - {symbol}: ${value:,.2f}")

    total_equity = bot.current_balance + positions_value

    print(f"\n5. РЕАЛЬНЫЙ КАПИТАЛ:")
    print(f"   Свободный баланс: ${bot.current_balance:,.2f}")
    print(f"   В позициях: ${positions_value:,.2f}")
    print(f"   ИТОГО EQUITY: ${total_equity:,.2f}")

    # Расчёт реального P&L
    real_pnl = total_equity - bot.initial_balance
    real_return = (real_pnl / bot.initial_balance) * 100

    print(f"\n6. РЕАЛЬНЫЙ P&L:")
    print(f"   P&L: ${real_pnl:+,.2f}")
    print(f"   Return: {real_return:+.2f}%")

    # Останавливаем бота
    await bot.stop()


if __name__ == "__main__":
    try:
        asyncio.run(check_real_balance())
    except KeyboardInterrupt:
        print("\nПрервано пользователем")