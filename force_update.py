# force_update.py
"""
Принудительное обновление данных
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


async def test_updates():
    from src.bots.advanced_paper_bot import AdvancedPaperTradingBot

    print("Тестируем обновление данных...\n")

    bot = AdvancedPaperTradingBot(
        initial_balance=50000.0,
        maker_fee=0.001,
        taker_fee=0.001,
        slippage_bps=5.0
    )

    await bot.initialize()

    print("Ждём 10 секунд для первой загрузки...")
    await asyncio.sleep(10)

    # Сохраняем начальные значения
    initial_rsi = {}
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
        if symbol in bot.indicators:
            initial_rsi[symbol] = bot.indicators[symbol].get('rsi', 0)
            print(f"{symbol} начальный RSI: {initial_rsi[symbol]:.2f}")

    print("\nЖдём 30 секунд для обновления...")
    await asyncio.sleep(30)

    # Проверяем изменения
    print("\nПроверка изменений:")
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
        if symbol in bot.indicators:
            current_rsi = bot.indicators[symbol].get('rsi', 0)
            change = current_rsi - initial_rsi.get(symbol, 0)

            if abs(change) > 0.01:
                print(f"✅ {symbol}: RSI изменился {initial_rsi[symbol]:.2f} → {current_rsi:.2f} (Δ{change:+.2f})")
            else:
                print(f"❌ {symbol}: RSI НЕ изменился ({current_rsi:.2f})")

    await bot.stop()


if __name__ == "__main__":
    asyncio.run(test_updates())