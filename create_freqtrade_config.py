# create_freqtrade_config.py
import json
import os
from pathlib import Path


def create_full_config():
    """Создание полной конфигурации для Freqtrade"""

    # Путь к Freqtrade
    freqtrade_path = Path("C:/freqtrade-nfi")
    user_data = freqtrade_path / "ft_userdata"

    # Создаем директории
    (user_data / "strategies").mkdir(parents=True, exist_ok=True)
    (user_data / "data").mkdir(parents=True, exist_ok=True)

    config = {
        "max_open_trades": 5,
        "stake_currency": "USDT",
        "stake_amount": "unlimited",
        "tradable_balance_ratio": 0.99,
        "fiat_display_currency": "USD",
        "dry_run": True,
        "dry_run_wallet": 50000,  # Ваш начальный капитал
        "cancel_open_orders_on_exit": False,
        "trading_mode": "spot",
        "margin_mode": "",

        "timeframe": "5m",  # ВАЖНО: добавляем таймфрейм

        "unfilledtimeout": {
            "entry": 10,
            "exit": 10,
            "exit_timeout_count": 0,
            "unit": "minutes"
        },

        # ВАЖНО: Добавляем entry_pricing
        "entry_pricing": {
            "price_side": "same",
            "use_order_book": False,  # Изменено на False для dry-run
            "order_book_top": 1,
            "price_last_balance": 0.0,
            "check_depth_of_market": {
                "enabled": False,
                "bids_to_ask_delta": 1
            }
        },

        # ВАЖНО: Добавляем exit_pricing
        "exit_pricing": {
            "price_side": "same",
            "use_order_book": False,  # Изменено на False для dry-run
            "order_book_top": 1,
            "price_last_balance": 0.0
        },

        "exchange": {
            "name": "binance",
            "key": "",  # Оставляем пустым для dry-run
            "secret": "",  # Оставляем пустым для dry-run
            "ccxt_config": {},
            "ccxt_async_config": {},
            "pair_whitelist": [
                "BTC/USDT",
                "ETH/USDT",
                "SOL/USDT",
                "BNB/USDT",
                "DOGE/USDT",
                "AVAX/USDT"
            ],
            "pair_blacklist": []
        },

        "pairlists": [
            {
                "method": "StaticPairList"
            }
        ],

        "dataformat_ohlcv": "json",
        "dataformat_trades": "json",

        "user_data_dir": str(user_data),
        "datadir": str(user_data / "data"),
        "strategy_path": str(user_data / "strategies"),

        # ИСПРАВЛЕНИЕ: Полная конфигурация API сервера
        "api_server": {
            "enabled": False,
            "listen_ip_address": "127.0.0.1",  # ОБЯЗАТЕЛЬНО
            "listen_port": 8080,  # ОБЯЗАТЕЛЬНО
            "username": "freqtrader",  # ОБЯЗАТЕЛЬНО
            "password": "password",  # ОБЯЗАТЕЛЬНО
            "verbosity": "error",
            "enable_openapi": False,
            "jwt_secret_key": "somethingrandom",
            "ws_token": "",
            "CORS_origins": []
        },

        "bot_name": "AdvancedCryptoBot",
        "initial_state": "running",
        "force_entry_enable": False
    }

    # Сохраняем конфигурацию
    config_file = user_data / "config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"✅ Конфигурация создана: {config_file}")

    # Копируем стратегию
    strategy_source = Path("freqtrade_strategy.py")
    strategy_dest = user_data / "strategies" / "AdvancedCryptoStrategy.py"

    if strategy_source.exists():
        with open(strategy_source, "r", encoding="utf-8") as src:
            content = src.read()
        with open(strategy_dest, "w", encoding="utf-8") as dst:
            dst.write(content)
        print(f"✅ Стратегия скопирована: {strategy_dest}")

    return config_file


if __name__ == "__main__":
    config_path = create_full_config()
    print("\n📝 Конфигурация готова!")
    print("\nТеперь запустите:")
    print("python run_freqtrade_setup.py")