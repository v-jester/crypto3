# clear_redis.py
import redis

try:
    # Подключаемся к Redis
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

    # Проверяем подключение
    r.ping()
    print("✅ Connected to Redis")

    # Очищаем все данные
    r.flushall()
    print("✅ Redis cache cleared successfully")

except redis.ConnectionError:
    print("❌ Could not connect to Redis. Make sure Redis is running.")
except Exception as e:
    print(f"❌ Error: {e}")