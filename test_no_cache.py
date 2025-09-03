# test_no_cache.py
"""
Тест без использования кеша вообще
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from binance import AsyncClient

sys.path.insert(0, str(Path(__file__).parent))


async def test_direct_api():
    """Тест прямых запросов к API"""

    from src.config.settings import settings

    print("Тестируем прямые запросы к Binance API...\n")

    client = await AsyncClient.create(
        api_key=settings.api.BINANCE_API_KEY.get_secret_value(),
        api_secret=settings.api.BINANCE_API_SECRET.get_secret_value(),
        testnet=True
    )

    symbol = "BTCUSDT"

    print(f"Отслеживаем {symbol} в течение 30 секунд...\n")

    for i in range(7):  # 7 проверок каждые 5 секунд
        # Получаем свечи напрямую
        klines = await client.get_klines(
            symbol=symbol,
            interval="5m",
            limit=50
        )

        # Последняя свеча
        last_candle = klines[-1]
        close_price = float(last_candle[4])
        volume = float(last_candle[5])

        # Простой расчет RSI
        closes = [float(k[4]) for k in klines]

        # Упрощенный RSI
        gains = []
        losses = []
        for j in range(1, len(closes)):
            change = closes[j] - closes[j - 1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = sum(gains[-14:]) / 14 if len(gains) >= 14 else 0
        avg_loss = sum(losses[-14:]) / 14 if len(losses) >= 14 else 0

        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 100 if avg_gain > 0 else 50

        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] Цена: ${close_price:,.2f} | RSI: {rsi:.2f} | Объем: {volume:.4f}")

        if i < 6:
            await asyncio.sleep(5)

    await client.close_connection()

    print("\n✅ Если цены и RSI менялись - API работает корректно")
    print("❌ Если не менялись - проблема с Binance Testnet")


if __name__ == "__main__":
    asyncio.run(test_direct_api())