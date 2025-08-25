# clear_cache_and_restart.py
"""
Скрипт для очистки кеша Redis и перезапуска бота со свежими данными
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.storage.redis_client import redis_client
from src.monitoring.logger import logger


async def clear_all_cache():
    """Полная очистка кеша Redis"""
    try:
        print("🔄 Connecting to Redis...")
        await redis_client.connect()

        if not redis_client._connected:
            print("❌ Failed to connect to Redis")
            print("   Make sure Redis is running on port 16379")
            return False

        print("🗑️ Clearing all cached data...")

        # Паттерны для удаления
        patterns = [
            "historical:*",
            "market:*",
            "kline:*",
            "ticker:*",
            "indicators:*",
            "depth:*",
            "trades:*",
            "prices:*",
            "position:*",
            "cache:*"
        ]

        total_deleted = 0

        for pattern in patterns:
            try:
                # Получаем все ключи по паттерну
                cursor = 0
                keys_to_delete = []

                while True:
                    cursor, keys = await redis_client._client.scan(
                        cursor,
                        match=pattern,
                        count=100
                    )

                    keys_to_delete.extend(keys)

                    if cursor == 0:
                        break

                # Удаляем найденные ключи
                if keys_to_delete:
                    # Декодируем bytes в строки если нужно
                    keys_str = []
                    for key in keys_to_delete:
                        if isinstance(key, bytes):
                            keys_str.append(key.decode('utf-8'))
                        else:
                            keys_str.append(key)

                    deleted = await redis_client.delete(*keys_str)
                    total_deleted += deleted
                    print(f"   ✅ Deleted {deleted} keys matching '{pattern}'")
                else:
                    print(f"   ℹ️ No keys found for pattern '{pattern}'")

            except Exception as e:
                print(f"   ⚠️ Error deleting keys for pattern '{pattern}': {e}")

        print(f"\n✅ Total keys deleted: {total_deleted}")

        # Проверка, что кеш очищен
        remaining_keys = 0
        cursor = 0
        while True:
            cursor, keys = await redis_client._client.scan(cursor, count=100)
            remaining_keys += len(keys)
            if cursor == 0:
                break

        if remaining_keys > 0:
            print(f"ℹ️ {remaining_keys} keys remaining in Redis (probably system keys)")
        else:
            print("✅ Redis cache completely cleared!")

        await redis_client.disconnect()
        return True

    except Exception as e:
        print(f"❌ Error clearing cache: {e}")
        import traceback
        traceback.print_exc()
        return False


async def restart_bot():
    """Инструкции для перезапуска бота"""
    print("\n" + "=" * 60)
    print("NEXT STEPS TO FIX THE BOT:")
    print("=" * 60)

    print("""
1. ✅ Cache has been cleared

2. 📝 Apply the fix to your bot:
   - Replace src/bots/advanced_paper_bot.py with the fixed version
   - The key changes are:
     * Added force_refresh flag to bypass cache
     * Clear cache before each data fetch
     * Update data every 30 seconds regardless of price change
     * Store last update timestamps

3. 🚀 Restart the bot:
   python src/main.py

4. 📊 Monitor the logs:
   You should see messages like:
   "Data updated for BTCUSDT | Price: 50000.00 → 50100.00 | RSI: 90.5 → 65.3"

5. 🎯 Expected behavior:
   - RSI should change from the stuck value (90.5)
   - Prices should update every 30 seconds
   - New trading signals should appear
   - Positions should close when hitting SL/TP

If data is still stuck after applying these fixes:
- Check Binance Testnet status (might be down)
- Try with fewer symbols (just BTCUSDT)
- Increase update interval to 60 seconds
""")


async def main():
    print("\n" + "=" * 60)
    print("CRYPTO BOT CACHE CLEANER")
    print("=" * 60 + "\n")

    success = await clear_all_cache()

    if success:
        await restart_bot()
    else:
        print("\n❌ Failed to clear cache. Please check Redis connection.")
        print("   Make sure Redis is running on port 16379")
        print("   You can start it with: docker-compose up -d redis")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()