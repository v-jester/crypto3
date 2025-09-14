# src/config/settings.py
"""
Централизованная конфигурация системы с валидацией через Pydantic
"""
import os
from typing import Optional, List, Dict, Any
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import SecretStr, Field, validator
from pydantic.types import PositiveFloat, PositiveInt
from enum import Enum

# Загружаем .env файл в начале
from dotenv import load_dotenv
load_dotenv()


class Environment(str, Enum):
    """Окружения запуска"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class BotMode(str, Enum):
    """Режимы работы бота"""
    MONITOR = "monitor"  # Только мониторинг без торговли
    PAPER = "paper"  # Paper trading
    LIVE = "live"  # Реальная торговля


class TradingMode(str, Enum):
    """Режимы торговли"""
    SPOT = "spot"
    FUTURES = "futures"


class APISettings(BaseSettings):
    """Настройки API биржи"""
    BINANCE_API_KEY: SecretStr = Field(default=SecretStr("test_api_key"))
    BINANCE_API_SECRET: SecretStr = Field(default=SecretStr("test_api_secret"))
    TESTNET: bool = True

    # Endpoints
    REST_URL: str = Field(
        default="https://testnet.binance.vision",
        description="REST API endpoint"
    )
    WS_URL: str = Field(
        default="wss://testnet.binance.vision",
        description="WebSocket endpoint"
    )

    # Rate limits
    REQUEST_WEIGHT_LIMIT: int = 6000
    ORDER_LIMIT: int = 50

    class Config:
        env_prefix = "API_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


class DatabaseSettings(BaseSettings):
    """Настройки базы данных"""
    # TimescaleDB
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "trading_bot"
    POSTGRES_USER: str = "trader"
    POSTGRES_PASSWORD: SecretStr = Field(default=SecretStr("trading_password_123"))

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[SecretStr] = None

    # QuestDB (optional for tick data)
    QUESTDB_HOST: str = "localhost"
    QUESTDB_PORT: int = 9009
    USE_QUESTDB: bool = False

    @property
    def postgres_url(self) -> str:
        """Получить PostgreSQL connection URL"""
        pwd = self.POSTGRES_PASSWORD.get_secret_value()
        return f"postgresql://{self.POSTGRES_USER}:{pwd}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def redis_url(self) -> str:
        """Получить Redis connection URL"""
        if self.REDIS_PASSWORD:
            pwd = self.REDIS_PASSWORD.get_secret_value()
            return f"redis://:{pwd}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    class Config:
        env_prefix = "DB_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


class TradingSettings(BaseSettings):
    """Настройки торговли"""
    # Основные параметры
    INITIAL_CAPITAL: PositiveFloat = 10000.0
    RISK_PER_TRADE: PositiveFloat = Field(
        default=0.02,
        ge=0.001,
        le=0.05,
        description="Риск на сделку (1-5%)"
    )
    MAX_POSITIONS: PositiveInt = Field(
        default=5,
        description="Максимум открытых позиций"
    )

    # Лимиты позиций
    MIN_POSITION_SIZE_USDT: PositiveFloat = 10.0
    MAX_POSITION_SIZE_PERCENT: PositiveFloat = Field(
        default=0.25,
        description="Макс размер позиции от капитала"
    )

    # Риск-менеджмент
    MAX_DAILY_LOSS_PERCENT: PositiveFloat = 0.05  # 5%
    MAX_DRAWDOWN_PERCENT: PositiveFloat = 0.20  # 20%
    STOP_LOSS_ATR_MULTIPLIER: PositiveFloat = 2.5
    TAKE_PROFIT_RR_RATIO: PositiveFloat = 2.0  # Risk:Reward 1:2

    # Корреляция
    MAX_CORRELATION: PositiveFloat = Field(
        default=0.7,
        description="Макс корреляция между позициями"
    )

    # Таймфреймы для анализа
    TIMEFRAMES: List[str] = Field(default=["5m", "15m", "1h", "4h"])
    PRIMARY_TIMEFRAME: str = "15m"

    # Торговые пары
    SYMBOLS: List[str] = Field(default=[
        "BTCUSDT", "ETHUSDT", "BNBUSDT",
        "ADAUSDT", "SOLUSDT", "DOTUSDT"
    ])

    # Время торговли (UTC)
    TRADING_HOURS: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Ограничения по времени торговли"
    )

    class Config:
        env_prefix = "TRADING_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


class MLSettings(BaseSettings):
    """Настройки машинного обучения"""
    # Модели
    USE_XGBOOST: bool = True
    USE_LIGHTGBM: bool = True
    ENSEMBLE_METHOD: str = "voting"  # voting, stacking

    # Параметры обучения
    RETRAIN_INTERVAL_HOURS: int = 2
    MIN_TRAINING_SAMPLES: int = 1000
    VALIDATION_SPLIT: float = 0.2

    # Online learning
    ENABLE_ONLINE_LEARNING: bool = True
    ONLINE_BATCH_SIZE: int = 50
    ONLINE_UPDATE_FREQUENCY: int = 300  # секунд

    # Feature engineering
    NUM_FEATURES: int = 3266  # Количество признаков
    FEATURE_SELECTION_METHOD: str = "importance"  # importance, pca, mutual_info

    # Пороги
    MIN_CONFIDENCE_THRESHOLD: float = 0.65
    MIN_WIN_PROBABILITY: float = 0.55

    class Config:
        env_prefix = "ML_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


class SignalSettings(BaseSettings):
    """Настройки генерации сигналов"""
    # RSI
    RSI_PERIOD: int = 14
    RSI_OVERSOLD: float = 40.0
    RSI_OVERBOUGHT: float = 60.0

    # MACD (Linda Raschke settings)
    MACD_FAST: int = 3
    MACD_SLOW: int = 10
    MACD_SIGNAL: int = 16

    # Bollinger Bands
    BB_PERIOD: int = 20
    BB_STD: float = 2.0

    # Volume
    MIN_VOLUME_RATIO: float = 1.5  # Мин объём относительно среднего

    # Фильтры сигналов
    MIN_RR_RATIO: float = 1.5  # Risk:Reward
    MAX_SPREAD_BPS: float = 10.0  # Макс спред в базисных пунктах

    class Config:
        env_prefix = "SIGNAL_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


class MonitoringSettings(BaseSettings):
    """Настройки мониторинга"""
    # Логирование
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json, text
    LOG_DIR: Path = Path("logs")

    # Метрики
    ENABLE_PROMETHEUS: bool = True
    PROMETHEUS_PORT: int = 8000

    # Алерты
    ENABLE_ALERTS: bool = True
    ALERT_WEBHOOK_URL: Optional[str] = None

    # Производительность
    TRACK_PERFORMANCE: bool = True
    PERFORMANCE_LOG_INTERVAL: int = 60  # секунд

    class Config:
        env_prefix = "MONITORING_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


class Settings(BaseSettings):
    """Основные настройки приложения"""
    # Окружение
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    BOT_MODE: BotMode = BotMode.PAPER
    TRADING_MODE: TradingMode = TradingMode.FUTURES

    # Проект
    PROJECT_NAME: str = "Crypto Trading Bot"
    VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Подмодули настроек - lazy loading через свойства
    _api: Optional[APISettings] = None
    _database: Optional[DatabaseSettings] = None
    _trading: Optional[TradingSettings] = None
    _ml: Optional[MLSettings] = None
    _signals: Optional[SignalSettings] = None
    _monitoring: Optional[MonitoringSettings] = None

    @property
    def api(self) -> APISettings:
        """Получить настройки API"""
        if self._api is None:
            self._api = APISettings()
        return self._api

    @property
    def database(self) -> DatabaseSettings:
        """Получить настройки базы данных"""
        if self._database is None:
            self._database = DatabaseSettings()
        return self._database

    @property
    def trading(self) -> TradingSettings:
        """Получить настройки торговли"""
        if self._trading is None:
            self._trading = TradingSettings()
        return self._trading

    @property
    def ml(self) -> MLSettings:
        """Получить настройки ML"""
        if self._ml is None:
            self._ml = MLSettings()
        return self._ml

    @property
    def signals(self) -> SignalSettings:
        """Получить настройки сигналов"""
        if self._signals is None:
            self._signals = SignalSettings()
        return self._signals

    @property
    def monitoring(self) -> MonitoringSettings:
        """Получить настройки мониторинга"""
        if self._monitoring is None:
            self._monitoring = MonitoringSettings()
        return self._monitoring

    # Интервалы обновления (секунды)
    MARKET_UPDATE_INTERVAL: int = 1
    POSITION_CHECK_INTERVAL: int = 5
    RISK_CHECK_INTERVAL: int = 10
    UNIVERSE_UPDATE_INTERVAL: int = 300

    # Paper Trading специфичные настройки
    PAPER_STARTING_BALANCE: float = 10000.0
    PAPER_MAKER_FEE: float = 0.001  # 0.1%
    PAPER_TAKER_FEE: float = 0.001  # 0.1%
    PAPER_SLIPPAGE_BPS: float = 5.0  # 0.05%

    @validator("ENVIRONMENT")
    def validate_environment(cls, v, values):
        """Валидация окружения"""
        if v == Environment.PRODUCTION and values.get("DEBUG"):
            raise ValueError("DEBUG не может быть True в PRODUCTION")
        return v

    @validator("BOT_MODE")
    def validate_bot_mode(cls, v, values):
        """Валидация режима бота"""
        if v == BotMode.LIVE and values.get("ENVIRONMENT") == Environment.DEVELOPMENT:
            raise ValueError("LIVE режим недоступен в DEVELOPMENT окружении")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # Игнорируем дополнительные поля из подмодулей


# Singleton instance
settings = Settings()


# Вспомогательные функции
def get_settings() -> Settings:
    """Получить текущие настройки"""
    return settings


def is_production() -> bool:
    """Проверка на продакшен окружение"""
    return settings.ENVIRONMENT == Environment.PRODUCTION


def is_paper_trading() -> bool:
    """Проверка на paper trading режим"""
    return settings.BOT_MODE == BotMode.PAPER


def is_live_trading() -> bool:
    """Проверка на live trading режим"""
    return settings.BOT_MODE == BotMode.LIVE
