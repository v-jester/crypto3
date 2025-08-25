# test_live_data.py
"""
Проверка актуальности данных с Binance
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from binance import AsyncClient

sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import settings
from src.monitoring.logger import logger
import pandas as pd
import numpy as np


async def check_data_freshness():
    """Проверяем свежесть данных"""

    print("\n" + "=" * 60)
    print("ТЕСТ АКТУАЛЬНОСТИ ДАННЫХ BINANCE")
    print("=" * 60 + "\n")

    # Подключаемся к Binance
    client = await AsyncClient.create(
        api_key=settings.api.BINANCE_API_KEY.get_secret_value(),
        api_secret=settings.api.BINANCE_API_SECRET.get_secret_value(),
        testnet=settings.api.TESTNET
    )

    symbol = "BTCUSDT"

    try:
        # 1. Проверяем текущую цену
        print("1. Проверка текущей цены:")
        ticker = await client.get_ticker(symbol=symbol)
        current_price = float(ticker['lastPrice'])
        price_time = datetime.fromtimestamp(ticker['closeTime'] / 1000)

        print(f"   Цена: ${current_price:,.2f}")
        print(f"   Время: {price_time}")
        print(f"   Задержка: {(datetime.utcnow() - price_time).total_seconds():.1f} сек")

        # 2. Получаем последние свечи
        print("\n2. Проверка последних свечей (15m):")
        klines = await client.get_klines(
            symbol=symbol,
            interval="15m",
            limit=5
        )

        for i, kline in enumerate(klines[-3:]):
            open_time = datetime.fromtimestamp(int(kline[0]) / 1000)
            close_time = datetime.fromtimestamp(int(kline[6]) / 1000)
            close_price = float(kline[4])
            volume = float(kline[5])

            age = (datetime.utcnow() - close_time).total_seconds()

            print(f"\n   Свеча {i + 1}:")
            print(f"     Время: {open_time.strftime('%H:%M:%S')} - {close_time.strftime('%H:%M:%S')}")
            print(f"     Цена закрытия: ${close_price:,.2f}")
            print(f"     Объем: {volume:.4f} BTC")
            print(f"     Возраст данных: {age:.0f} сек")

            if age > 900:  # 15 минут
                print(f"     ⚠️ ДАННЫЕ УСТАРЕЛИ!")

        # 3. Проверяем RSI в реальном времени
        print("\n3. Расчет RSI из свежих данных:")

        # Получаем больше данных для RSI
        klines = await client.get_klines(
            symbol=symbol,
            interval="15m",
            limit=50
        )

        # Преобразуем в DataFrame
        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])

        df['close'] = df['close'].astype(float)

        # Вычисляем RSI
        def calculate_rsi(prices, period=14):
            deltas = np.diff(prices)
            seed = deltas[:period + 1]
            up = seed[seed >= 0].sum() / period
            down = -seed[seed < 0].sum() / period
            rs = up / down if down != 0 else 100
            rsi = np.zeros_like(prices)
            rsi[:period] = 100. - 100. / (1. + rs)

            for i in range(period, len(prices)):
                delta = deltas[i - 1]
                if delta > 0:
                    upval = delta
                    downval = 0.
                else:
                    upval = 0.
                    downval = -delta

                up = (up * (period - 1) + upval) / period
                down = (down * (period - 1) + downval) / period
                rs = up / down if down != 0 else 100
                rsi[i] = 100. - 100. / (1. + rs)

            return rsi

        rsi_values = calculate_rsi(df['close'].values, 14)
        current_rsi = rsi_values[-1]
        prev_rsi = rsi_values[-2]

        print(f"   Текущий RSI: {current_rsi:.2f}")
        print(f"   Предыдущий RSI: {prev_rsi:.2f}")
        print(f"   Изменение: {current_rsi - prev_rsi:+.2f}")

        if abs(current_rsi - prev_rsi) < 0.01:
            print("   ⚠️ RSI НЕ МЕНЯЕТСЯ - ПРОБЛЕМА С ДАННЫМИ!")
        else:
            print("   ✅ RSI обновляется корректно")

        # 4. Проверяем скорость обновления
        print("\n4. Тест обновления в реальном времени:")
        print("   Отслеживаем изменения цены в течение 30 секунд...")

        prices = []
        timestamps = []

        for i in range(6):  # 6 проверок по 5 секунд = 30 секунд
            ticker = await client.get_ticker(symbol=symbol)
            price = float(ticker['lastPrice'])
            timestamp = datetime.utcnow()

            prices.append(price)
            timestamps.append(timestamp)

            if i > 0:
                price_change = price - prices[i - 1]
                time_diff = (timestamp - timestamps[i - 1]).total_seconds()

                print(f"   [{i}] ${price:,.2f} | Изменение: ${price_change:+.2f} за {time_diff:.1f} сек")
            else:
                print(f"   [{i}] ${price:,.2f} | Начальная цена")

            if i < 5:
                await asyncio.sleep(5)

        # Анализ изменений
        unique_prices = len(set(prices))
        if unique_prices == 1:
            print("\n   ⚠️ ЦЕНА НЕ МЕНЯЛАСЬ - возможна проблема с данными!")
        else:
            print(f"\n   ✅ Цена обновлялась {unique_prices} раз из 6 проверок")

        # 5. Проверка WebSocket (если доступен)
        print("\n5. Проверка WebSocket потока:")
        if settings.api.TESTNET:
            print("   ⚠️ Testnet может иметь проблемы с WebSocket")
            print("   Рекомендуется использовать REST API polling")

    finally:
        await client.close_connection()

    print("\n" + "=" * 60)
    print("РЕКОМЕНДАЦИИ:")
    print("=" * 60)
    print("""
1. Если данные не обновляются:
   - Проверьте подключение к интернету
   - Попробуйте переключиться с testnet на mainnet
   - Увеличьте интервал обновления данных

2. Если RSI застрял:
   - Очистите кеш Redis: python clear_cache_and_restart.py
   - Проверьте расчет индикаторов в historical_data.py

3. Для исправления проблемы с капиталом:
   - Уменьшите минимальный размер позиции в настройках
   - Увеличьте начальный капитал
   """)


async def main():
    try:
        await check_data_freshness()
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())