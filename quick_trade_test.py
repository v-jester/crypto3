# quick_trade_test.py
"""
Быстрый тест с измененными параметрами
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


async def test_with_modified_params():
    from src.bots.advanced_paper_bot import AdvancedPaperTradingBot

    bot = AdvancedPaperTradingBot(
        initial_balance=50000.0,
        maker_fee=0.001,
        taker_fee=0.001,
        slippage_bps=5.0
    )

    await bot.initialize()
    print("✅ Бот инициализирован\n")

    # Ждём данные
    await asyncio.sleep(20)

    # МОДИФИЦИРУЕМ пороги прямо в runtime
    print("📊 Текущие RSI:")
    for symbol in bot.indicators:
        rsi = bot.indicators[symbol].get('rsi', 0)
        print(f"  {symbol}: {rsi:.2f}")

    print("\n🔧 Применяем модифицированную стратегию...")

    # Хакаем метод для теста
    original_method = bot._generate_trading_signal

    async def modified_signal(symbol, df, indicators):
        signal = await original_method(symbol, df, indicators)
        # Усиливаем сигналы для текущих условий
        rsi = indicators.get('rsi', 50)
        if 55 < rsi < 65:  # Текущий диапазон
            signal['action'] = 'SELL'
            signal['confidence'] = 0.65
            signal['reasons'].append(f"Modified: RSI high range ({rsi:.1f})")
        return signal

    bot._generate_trading_signal = modified_signal

    # Запускаем анализ
    await bot._analyze_and_trade()

    print(f"\n✅ Результат:")
    print(f"Позиций открыто: {len(bot.positions)}")
    for symbol, pos in bot.positions.items():
        print(f"  {symbol}: {pos.side} @ ${pos.entry_price:.2f}")

    await bot.stop()


if __name__ == "__main__":
    asyncio.run(test_with_modified_params())