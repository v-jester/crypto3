# src/data/storage/database.py
"""
TimescaleDB менеджер для хранения исторических данных и торговой статистики
Оптимизирован для временных рядов с автоматическим партиционированием
"""
import asyncio
import asyncpg
from asyncpg import Pool
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from src.monitoring.logger import logger, log_performance
from src.config.settings import settings


class TimescaleDBManager:
    """Менеджер для работы с TimescaleDB"""

    def __init__(self):
        self.pool: Optional[Pool] = None
        self.db_url = settings.database.postgres_url

    async def initialize(self):
        """Инициализация пула соединений и создание таблиц"""
        try:
            # Создание пула соединений
            self.pool = await asyncpg.create_pool(
                self.db_url,
                min_size=5,
                max_size=20,
                max_queries=50000,
                max_inactive_connection_lifetime=300,
                command_timeout=60
            )

            logger.logger.info("TimescaleDB pool created")

            # Создание таблиц
            await self._create_tables()

            logger.logger.info("TimescaleDB initialized successfully")

        except Exception as e:
            logger.log_error(e, {"context": "Failed to initialize TimescaleDB"})
            raise

    async def close(self):
        """Закрытие пула соединений"""
        if self.pool:
            await self.pool.close()
            logger.logger.info("TimescaleDB pool closed")

    @asynccontextmanager
    async def acquire(self):
        """Контекстный менеджер для получения соединения"""
        async with self.pool.acquire() as connection:
            yield connection

    async def _create_tables(self):
        """Создание таблиц и гипертаблиц"""
        async with self.acquire() as conn:
            # Включаем расширение TimescaleDB
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")

            # Таблица свечей
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS klines (
                    time TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    interval VARCHAR(10) NOT NULL,
                    open DECIMAL(20,8) NOT NULL,
                    high DECIMAL(20,8) NOT NULL,
                    low DECIMAL(20,8) NOT NULL,
                    close DECIMAL(20,8) NOT NULL,
                    volume DECIMAL(20,8) NOT NULL,
                    quote_volume DECIMAL(20,8),
                    trades INTEGER,
                    taker_buy_base DECIMAL(20,8),
                    taker_buy_quote DECIMAL(20,8),
                    PRIMARY KEY (time, symbol, interval)
                );
            """)

            # Создаём гипертаблицу
            await conn.execute("""
                SELECT create_hypertable(
                    'klines', 'time',
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 day'
                );
            """)

            # Индексы для оптимизации
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_klines_symbol_time 
                ON klines (symbol, time DESC);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_klines_symbol_interval_time 
                ON klines (symbol, interval, time DESC);
            """)

            # Таблица сделок
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL,
                    time TIMESTAMPTZ NOT NULL,
                    position_id VARCHAR(50) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    order_type VARCHAR(20),
                    price DECIMAL(20,8) NOT NULL,
                    quantity DECIMAL(20,8) NOT NULL,
                    value DECIMAL(20,8) NOT NULL,
                    fee DECIMAL(20,8),
                    pnl DECIMAL(20,8),
                    cumulative_pnl DECIMAL(20,8),
                    status VARCHAR(20),
                    metadata JSONB,
                    PRIMARY KEY (time, id)
                );
            """)

            await conn.execute("""
                SELECT create_hypertable(
                    'trades', 'time',
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '7 days'
                );
            """)

            # Таблица позиций
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id VARCHAR(50) PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    entry_time TIMESTAMPTZ NOT NULL,
                    exit_time TIMESTAMPTZ,
                    entry_price DECIMAL(20,8) NOT NULL,
                    exit_price DECIMAL(20,8),
                    quantity DECIMAL(20,8) NOT NULL,
                    stop_loss DECIMAL(20,8),
                    take_profit DECIMAL(20,8),
                    realized_pnl DECIMAL(20,8),
                    unrealized_pnl DECIMAL(20,8),
                    max_profit DECIMAL(20,8),
                    max_loss DECIMAL(20,8),
                    duration_minutes INTEGER,
                    status VARCHAR(20),
                    close_reason VARCHAR(50),
                    metadata JSONB
                );
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_positions_symbol 
                ON positions (symbol);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_positions_status 
                ON positions (status);
            """)

            # Таблица сигналов
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    time TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    signal_type VARCHAR(50) NOT NULL,
                    direction VARCHAR(10) NOT NULL,
                    confidence DECIMAL(5,4) NOT NULL,
                    entry_price DECIMAL(20,8),
                    stop_loss DECIMAL(20,8),
                    take_profit DECIMAL(20,8),
                    executed BOOLEAN DEFAULT FALSE,
                    metadata JSONB,
                    PRIMARY KEY (time, symbol, signal_type)
                );
            """)

            await conn.execute("""
                SELECT create_hypertable(
                    'signals', 'time',
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 day'
                );
            """)

            # Таблица метрик производительности
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    time TIMESTAMPTZ NOT NULL,
                    metric_name VARCHAR(100) NOT NULL,
                    value DECIMAL(20,8) NOT NULL,
                    metadata JSONB,
                    PRIMARY KEY (time, metric_name)
                );
            """)

            await conn.execute("""
                SELECT create_hypertable(
                    'performance_metrics', 'time',
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 hour'
                );
            """)

            # Настройка политик сжатия для старых данных
            await self._setup_compression_policies(conn)

            logger.logger.info("All TimescaleDB tables created successfully")

    async def _setup_compression_policies(self, conn):
        """Настройка автоматического сжатия старых данных"""
        try:
            # Включаем сжатие для таблицы свечей
            await conn.execute("""
                ALTER TABLE klines SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby = 'symbol,interval'
                );
            """)

            # Политика сжатия - сжимать данные старше 7 дней
            await conn.execute("""
                SELECT add_compression_policy(
                    'klines',
                    INTERVAL '7 days',
                    if_not_exists => TRUE
                );
            """)

            # Сжатие для сделок
            await conn.execute("""
                ALTER TABLE trades SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby = 'symbol'
                );
            """)

            await conn.execute("""
                SELECT add_compression_policy(
                    'trades',
                    INTERVAL '30 days',
                    if_not_exists => TRUE
                );
            """)

            logger.logger.info("Compression policies configured")

        except Exception as e:
            logger.logger.warning(f"Could not setup compression policies: {e}")

    @log_performance("insert_klines")
    async def insert_klines(self, klines_data: List[Dict[str, Any]]) -> int:
        """
        Вставка свечных данных

        Args:
            klines_data: Список словарей с данными свечей

        Returns:
            Количество вставленных записей
        """
        if not klines_data:
            return 0

        async with self.acquire() as conn:
            # Подготовка данных для batch insert
            values = []
            for kline in klines_data:
                values.append((
                    kline['time'],
                    kline['symbol'],
                    kline['interval'],
                    kline['open'],
                    kline['high'],
                    kline['low'],
                    kline['close'],
                    kline['volume'],
                    kline.get('quote_volume'),
                    kline.get('trades'),
                    kline.get('taker_buy_base'),
                    kline.get('taker_buy_quote')
                ))

            # Используем ON CONFLICT для обновления существующих записей
            result = await conn.executemany("""
                INSERT INTO klines (
                    time, symbol, interval, open, high, low, close, volume,
                    quote_volume, trades, taker_buy_base, taker_buy_quote
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (time, symbol, interval) 
                DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    quote_volume = EXCLUDED.quote_volume,
                    trades = EXCLUDED.trades,
                    taker_buy_base = EXCLUDED.taker_buy_base,
                    taker_buy_quote = EXCLUDED.taker_buy_quote;
            """, values)

            count = len(values)
            logger.logger.debug(f"Inserted {count} klines")
            return count

    async def get_klines(
            self,
            symbol: str,
            interval: str,
            start_time: datetime,
            end_time: Optional[datetime] = None,
            limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Получение свечных данных

        Args:
            symbol: Торговая пара
            interval: Интервал свечей
            start_time: Начальное время
            end_time: Конечное время
            limit: Максимальное количество записей

        Returns:
            DataFrame с данными
        """
        if not end_time:
            end_time = datetime.utcnow()

        async with self.acquire() as conn:
            query = """
                SELECT time, open, high, low, close, volume,
                       quote_volume, trades, taker_buy_base, taker_buy_quote
                FROM klines
                WHERE symbol = $1 AND interval = $2 
                AND time >= $3 AND time <= $4
                ORDER BY time DESC
            """

            if limit:
                query += f" LIMIT {limit}"

            rows = await conn.fetch(query, symbol, interval, start_time, end_time)

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows)
            df.set_index('time', inplace=True)
            df = df.sort_index()

            return df

    async def insert_trade(self, trade_data: Dict[str, Any]) -> str:
        """Вставка информации о сделке"""
        async with self.acquire() as conn:
            trade_id = await conn.fetchval("""
                INSERT INTO trades (
                    time, position_id, symbol, side, order_type,
                    price, quantity, value, fee, pnl, cumulative_pnl,
                    status, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                RETURNING id;
            """,
                                           datetime.utcnow(),
                                           trade_data['position_id'],
                                           trade_data['symbol'],
                                           trade_data['side'],
                                           trade_data.get('order_type', 'MARKET'),
                                           trade_data['price'],
                                           trade_data['quantity'],
                                           trade_data['price'] * trade_data['quantity'],
                                           trade_data.get('fee', 0),
                                           trade_data.get('pnl', 0),
                                           trade_data.get('cumulative_pnl', 0),
                                           trade_data.get('status', 'FILLED'),
                                           trade_data.get('metadata', {})
                                           )

            logger.logger.debug(f"Trade inserted with ID: {trade_id}")
            return str(trade_id)

    async def insert_signal(self, signal_data: Dict[str, Any]):
        """Вставка торгового сигнала"""
        async with self.acquire() as conn:
            await conn.execute("""
                INSERT INTO signals (
                    time, symbol, signal_type, direction, confidence,
                    entry_price, stop_loss, take_profit, executed, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (time, symbol, signal_type) DO NOTHING;
            """,
                               datetime.utcnow(),
                               signal_data['symbol'],
                               signal_data['signal_type'],
                               signal_data['direction'],
                               signal_data['confidence'],
                               signal_data.get('entry_price'),
                               signal_data.get('stop_loss'),
                               signal_data.get('take_profit'),
                               signal_data.get('executed', False),
                               signal_data.get('metadata', {})
                               )

    async def get_trade_statistics(
            self,
            symbol: Optional[str] = None,
            days_back: int = 30
    ) -> Dict[str, Any]:
        """Получение статистики торговли"""
        start_time = datetime.utcnow() - timedelta(days=days_back)

        async with self.acquire() as conn:
            # Базовый запрос
            base_where = "WHERE time >= $1"
            params = [start_time]

            if symbol:
                base_where += " AND symbol = $2"
                params.append(symbol)

            # Общая статистика
            stats = await conn.fetchrow(f"""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    MAX(pnl) as max_win,
                    MIN(pnl) as min_loss,
                    SUM(value) as total_volume,
                    SUM(fee) as total_fees
                FROM trades
                {base_where};
            """, *params)

            # Преобразование в словарь
            result = dict(stats) if stats else {}

            # Расчёт дополнительных метрик
            if result.get('total_trades', 0) > 0:
                result['win_rate'] = result.get('winning_trades', 0) / result['total_trades']
                result['avg_win'] = await conn.fetchval(f"""
                    SELECT AVG(pnl) FROM trades 
                    {base_where} AND pnl > 0;
                """, *params) or 0

                result['avg_loss'] = await conn.fetchval(f"""
                    SELECT AVG(pnl) FROM trades 
                    {base_where} AND pnl < 0;
                """, *params) or 0

                # Profit factor
                total_wins = await conn.fetchval(f"""
                    SELECT SUM(pnl) FROM trades 
                    {base_where} AND pnl > 0;
                """, *params) or 0

                total_losses = abs(await conn.fetchval(f"""
                    SELECT SUM(pnl) FROM trades 
                    {base_where} AND pnl < 0;
                """, *params) or 1)

                result['profit_factor'] = total_wins / total_losses if total_losses > 0 else 0

            return result

    async def get_performance_metrics(
            self,
            metric_names: List[str],
            hours_back: int = 24
    ) -> pd.DataFrame:
        """Получение метрик производительности"""
        start_time = datetime.utcnow() - timedelta(hours=hours_back)

        async with self.acquire() as conn:
            rows = await conn.fetch("""
                SELECT time, metric_name, value, metadata
                FROM performance_metrics
                WHERE metric_name = ANY($1) AND time >= $2
                ORDER BY time DESC;
            """, metric_names, start_time)

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows)
            df.set_index('time', inplace=True)

            return df

    async def cleanup_old_data(self, days_to_keep: int = 90):
        """Очистка старых данных"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

        async with self.acquire() as conn:
            # Удаляем старые несжатые данные
            deleted_klines = await conn.fetchval("""
                DELETE FROM klines 
                WHERE time < $1 
                AND NOT EXISTS (
                    SELECT 1 FROM timescaledb_information.chunks c
                    WHERE c.hypertable_name = 'klines'
                    AND c.is_compressed = true
                    AND klines.time >= c.range_start 
                    AND klines.time < c.range_end
                )
                RETURNING COUNT(*);
            """, cutoff_date)

            deleted_signals = await conn.fetchval("""
                DELETE FROM signals 
                WHERE time < $1 AND NOT executed
                RETURNING COUNT(*);
            """, cutoff_date)

            logger.logger.info(
                f"Cleanup completed",
                deleted_klines=deleted_klines,
                deleted_signals=deleted_signals
            )

    async def get_continuous_aggregate(
            self,
            symbol: str,
            interval: str,
            aggregation: str = '1h'
    ) -> pd.DataFrame:
        """
        Получение агрегированных данных
        Использует continuous aggregates TimescaleDB для быстрых запросов
        """
        async with self.acquire() as conn:
            # Создаём continuous aggregate если не существует
            await conn.execute(f"""
                CREATE MATERIALIZED VIEW IF NOT EXISTS klines_{aggregation}_agg
                WITH (timescaledb.continuous) AS
                SELECT 
                    time_bucket('{aggregation}', time) AS bucket,
                    symbol,
                    first(open, time) as open,
                    max(high) as high,
                    min(low) as low,
                    last(close, time) as close,
                    sum(volume) as volume
                FROM klines
                GROUP BY bucket, symbol
                WITH NO DATA;
            """)

            # Обновляем материализованное представление
            await conn.execute(f"""
                CALL refresh_continuous_aggregate(
                    'klines_{aggregation}_agg',
                    NULL, 
                    NULL
                );
            """)

            # Получаем данные
            rows = await conn.fetch(f"""
                SELECT bucket as time, open, high, low, close, volume
                FROM klines_{aggregation}_agg
                WHERE symbol = $1
                ORDER BY bucket DESC
                LIMIT 1000;
            """, symbol)

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows)
            df.set_index('time', inplace=True)

            return df


# Глобальный экземпляр
db_manager = TimescaleDBManager()


# Вспомогательные функции
async def init_database():
    """Инициализация базы данных"""
    await db_manager.initialize()
    logger.logger.info("Database initialized")
    return db_manager


async def close_database():
    """Закрытие соединения с базой данных"""
    await db_manager.close()