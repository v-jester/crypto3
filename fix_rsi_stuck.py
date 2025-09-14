#!/usr/bin/env python3
"""
Скрипт для исправления проблемы с застрявшим RSI
Запускать перед стартом бота
"""
import asyncio
import sys
import os
from datetime import datetime

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.storage.redis_client import redis_client
from src.monitoring.logger import logger


async def clear_all_caches():
    """Полная очистка всех кешей"""
    print("\n" + "=" * 60)
    print("FIX RSI STUCK PROBLEM")
    print("=" * 60 + "\n")

    try:
        # Подключаемся к Redis
        print("🔄 Connecting to Redis...")
        await redis_client.connect()
        print("✅ Redis connected\n")

        # Паттерны для очистки
        patterns = [
            'historical:*',
            'market:*',
            'kline:*',
            'ticker:*',
            'indicators:*',
            'depth:*',
            'trades:*',
            'prices:*',
            'position:*',
            'cache:*'
        ]

        print("🗑️ Clearing all cached data...")
        total_deleted = 0

        for pattern in patterns:
            try:
                keys = await redis_client.keys(pattern)
                if keys:
                    count = len(keys)
                    for key in keys:
                        await redis_client.delete(key)
                    total_deleted += count
                    print(f"   ✅ Deleted {count} keys for pattern '{pattern}'")
                else:
                    print(f"   ℹ️ No keys found for pattern '{pattern}'")
            except Exception as e:
                print(f"   ⚠️ Error clearing pattern '{pattern}': {e}")

        print(f"\n✅ Total keys deleted: {total_deleted}")

        # Опционально: полная очистка базы данных
        print("\n🔥 Performing complete database flush...")
        await redis_client.flushdb()
        print("✅ Redis database completely cleared!")

        # Отключаемся
        await redis_client.disconnect()
        print("✅ Redis disconnected\n")

        print("=" * 60)
        print("NEXT STEPS:")
        print("=" * 60)
        print("\n1. ✅ Cache has been cleared")
        print("\n2. 📝 Make sure the following files are updated:")
        print("   - src/data/collectors/historical_data.py")
        print("   - src/bots/advanced_paper_bot.py")
        print("\n3. 🚀 Start the bot:")
        print("   python src/main.py")
        print("\n4. 📊 Monitor the logs for:")
        print("   - '✅ Data updated for BTCUSDT' messages")
        print("   - RSI values changing (not stuck at same value)")
        print("   - Price updates every 5 seconds")
        print("\n5. 🎯 Expected behavior:")
        print("   - RSI should change dynamically")
        print("   - New trading signals should appear")
        print("   - Positions should open/close properly")
        print("\n" + "=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.logger.error(f"Failed to clear caches: {e}")
        raise


async def test_data_fetching():
    """Тестирование получения данных"""
    print("\n" + "=" * 60)
    print("TESTING DATA FETCHING")
    print("=" * 60 + "\n")

    try:
        from binance import AsyncClient
        from src.data.collectors.historical_data import HistoricalDataCollector
        from src.config.settings import settings

        # Создаем клиент
        print("🔄 Connecting to Binance...")
        client = await AsyncClient.create(
            api_key=settings.api.BINANCE_API_KEY.get_secret_value(),
            api_secret=settings.api.BINANCE_API_SECRET.get_secret_value(),
            testnet=settings.api.TESTNET
        )

        # Инициализируем коллектор
        collector = HistoricalDataCollector()
        collector.force_refresh = True  # Принудительное обновление
        collector.use_cache = False  # Отключаем кеш
        await collector.initialize(client)

        # Тестируем получение данных
        symbol = "BTCUSDT"
        print(f"\n📊 Fetching data for {symbol}...")

        for i in range(3):
            df = await collector.fetch_historical_data(
                symbol=symbol,
                interval="5m",
                days_back=1,
                limit=100,
                force_refresh=True
            )

            if not df.empty:
                latest_rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 0
                latest_price = df['close'].iloc[-1] if 'close' in df.columns else 0
                print(f"   Test {i + 1}: Price=${latest_price:.2f}, RSI={latest_rsi:.2f}")
            else:
                print(f"   Test {i + 1}: Empty dataframe")

            await asyncio.sleep(2)

        await client.close_connection()
        print("\n✅ Data fetching test completed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        logger.logger.error(f"Data fetching test failed: {e}")


async def main():
    """Главная функция"""
    try:
        # Очищаем кеши
        await clear_all_caches()

        # Опционально: тестируем получение данных
        print("\nDo you want to test data fetching? (y/n): ", end="")
        answer = input().strip().lower()
        if answer == 'y':
            await test_data_fetching()

        print("\n✅ All fixes applied successfully!")
        print("You can now start the bot with: python src/main.py")

    except Exception as e:
        print(f"\n❌ Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Script interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)