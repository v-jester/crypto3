import asyncio
import asyncpg
from pathlib import Path


async def init_database():
    """Инициализация базы данных"""

    conn = await asyncpg.connect(
        host='localhost',
        port=5432,
        user='trader',
        password='trading_password_123',
        database='trading_bot'
    )

    print("📊 Инициализация базы данных...")

    # Включаем TimescaleDB
    await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
    print("✅ TimescaleDB extension установлен")

    # Создаём основные таблицы
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS klines (
            time TIMESTAMPTZ NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            interval VARCHAR(10) NOT NULL,
            open DECIMAL(20,8),
            high DECIMAL(20,8),
            low DECIMAL(20,8),
            close DECIMAL(20,8),
            volume DECIMAL(20,8),
            PRIMARY KEY (time, symbol, interval)
        );
    """)

    # Делаем гипертаблицу
    try:
        await conn.execute("""
            SELECT create_hypertable('klines', 'time', if_not_exists => TRUE);
        """)
        print("✅ Hypertable 'klines' создана")
    except:
        print("ℹ️ Hypertable 'klines' уже существует")

    await conn.close()
    print("✨ База данных готова!")


if __name__ == "__main__":
    asyncio.run(init_database())