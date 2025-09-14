# run_backtest.py
"""
Запуск бэктестинга вашей стратегии через Freqtrade
"""
import os
import sys
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

# Добавляем путь к вашему проекту
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import settings
from freqtrade_strategy import AdvancedCryptoStrategy, get_freqtrade_config


def prepare_freqtrade_env():
    """
    Подготовка окружения Freqtrade
    """
    # Путь к Freqtrade
    freqtrade_path = Path("C:/freqtrade-nfi")

    # Создаем директории если их нет
    user_data = freqtrade_path / "ft_userdata"
    strategies_dir = user_data / "strategies"
    data_dir = user_data / "data"

    strategies_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Копируем стратегию с правильной кодировкой
    strategy_file = strategies_dir / "AdvancedCryptoStrategy.py"

    # ИСПРАВЛЕНО: добавлена кодировка UTF-8
    with open("freqtrade_strategy.py", "r", encoding="utf-8") as source:
        strategy_content = source.read()

    with open(strategy_file, "w", encoding="utf-8") as target:
        target.write(strategy_content)

    print(f"✅ Стратегия скопирована в {strategy_file}")

    # Создаем конфиг
    config = get_freqtrade_config()

    # Добавляем ваши API ключи из settings
    config["exchange"]["key"] = settings.api.BINANCE_API_KEY.get_secret_value()
    config["exchange"]["secret"] = settings.api.BINANCE_API_SECRET.get_secret_value()

    # Настройки для правильной работы
    config["user_data_dir"] = str(user_data)
    config["datadir"] = str(data_dir)
    config["strategy_path"] = str(strategies_dir)

    # Сохраняем конфиг
    config_file = user_data / "config_advanced.json"

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"✅ Конфиг создан: {config_file}")

    return freqtrade_path, config_file


def download_data(freqtrade_path: Path, pairs: list, timeframe: str, days: int):
    """
    Загрузка исторических данных
    """
    print(f"\n📥 Загрузка данных для {len(pairs)} пар...")

    # Переходим в директорию Freqtrade
    os.chdir(freqtrade_path)

    # Активируем виртуальное окружение и выполняем команду
    cmd_ps1 = f"""
cd {freqtrade_path}
.\\venv\\Scripts\\Activate.ps1
freqtrade download-data --userdir ft_userdata --exchange binance --pairs {' '.join(pairs)} --timeframe {timeframe} --days {days} --data-format json
"""

    # Создаем временный PowerShell скрипт
    temp_script = freqtrade_path / "temp_download.ps1"
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(cmd_ps1)

    # Запускаем скрипт
    result = subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(temp_script)],
        capture_output=True,
        text=True,
        encoding='utf-8'
    )

    # Удаляем временный скрипт
    temp_script.unlink(missing_ok=True)

    if result.returncode == 0:
        print("✅ Данные загружены")
    else:
        print(f"⚠️ Предупреждение: {result.stderr}")
        # Не прерываем выполнение, так как данные могут быть уже загружены


def run_backtest(freqtrade_path: Path, config_file: Path,
                 strategy: str = "AdvancedCryptoStrategy",
                 timerange: str = None):
    """
    Запуск бэктеста
    """
    print(f"\n🚀 Запуск бэктеста стратегии {strategy}...")

    # Переходим в директорию Freqtrade
    os.chdir(freqtrade_path)

    # Формируем команду
    cmd = f"freqtrade backtesting --userdir ft_userdata --strategy {strategy}"

    if timerange:
        cmd += f" --timerange {timerange}"

    # PowerShell скрипт
    cmd_ps1 = f"""
cd {freqtrade_path}
.\\venv\\Scripts\\Activate.ps1
{cmd}
"""

    # Создаем временный скрипт
    temp_script = freqtrade_path / "temp_backtest.ps1"
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(cmd_ps1)

    # Запускаем бэктест
    result = subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(temp_script)],
        capture_output=True,
        text=True,
        encoding='utf-8'
    )

    # Выводим результаты
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ БЭКТЕСТА:")
    print("=" * 60)
    print(result.stdout)

    # Удаляем временный скрипт
    temp_script.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"❌ Ошибка: {result.stderr}")

    return result.returncode == 0


def quick_test():
    """
    Быстрый тест стратегии
    """
    print("=" * 60)
    print("БЫСТРЫЙ ТЕСТ СТРАТЕГИИ ЧЕРЕЗ FREQTRADE")
    print("=" * 60)

    try:
        # Подготовка
        freqtrade_path, config_file = prepare_freqtrade_env()

        # Пары для теста
        pairs = [
            "BTC/USDT",
            "ETH/USDT",
            "SOL/USDT"
        ]

        # Загрузка данных (последние 30 дней)
        download_data(freqtrade_path, pairs, "5m", 30)

        # Запуск бэктеста за последнюю неделю
        timerange = "20250907-20250914"
        success = run_backtest(freqtrade_path, config_file, timerange=timerange)

        if success:
            print("\n✅ Бэктест завершен успешно!")
            print("\nДля детального анализа:")
            print(f"1. Откройте PowerShell")
            print(f"2. cd {freqtrade_path}")
            print(f"3. .\\venv\\Scripts\\Activate.ps1")
            print(
                f"4. freqtrade plot-dataframe --userdir ft_userdata --strategy AdvancedCryptoStrategy --pairs BTC/USDT")

        return success

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


def interactive_mode():
    """
    Интерактивный режим для выбора действий
    """
    print("\n" + "=" * 60)
    print("FREQTRADE ИНТЕГРАЦИЯ")
    print("=" * 60)

    print("\nВыберите действие:")
    print("1. Быстрый тест (последняя неделя)")
    print("2. Полный бэктест (2 месяца)")
    print("3. Загрузить свежие данные")
    print("4. Оптимизация параметров")
    print("5. Запустить веб-интерфейс")

    choice = input("\nВаш выбор (1-5): ").strip()

    freqtrade_path, config_file = prepare_freqtrade_env()

    if choice == "1":
        quick_test()
    elif choice == "2":
        # Полный бэктест
        timerange = "20250715-20250914"
        run_backtest(freqtrade_path, config_file, timerange=timerange)
    elif choice == "3":
        # Загрузка данных
        pairs = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "DOGE/USDT"]
        download_data(freqtrade_path, pairs, "5m", 60)
    elif choice == "4":
        # Оптимизация
        print("\n🔧 Запуск оптимизации...")
        os.chdir(freqtrade_path)
        cmd = f"""
cd {freqtrade_path}
.\\venv\\Scripts\\Activate.ps1
freqtrade hyperopt --userdir ft_userdata --strategy AdvancedCryptoStrategy --hyperopt-loss SharpeHyperOptLoss --epochs 100 --spaces buy sell
"""
        subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-Command", cmd])
    elif choice == "5":
        # Веб-интерфейс
        print("\n🌐 Запуск веб-интерфейса...")
        print("Откройте браузер: http://localhost:8080")
        os.chdir(freqtrade_path)
        cmd = f"""
cd {freqtrade_path}
.\\venv\\Scripts\\Activate.ps1
freqtrade webserver --userdir ft_userdata
"""
        subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-Command", cmd])
    else:
        print("❌ Неверный выбор")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Freqtrade интеграция")
    parser.add_argument(
        "--mode",
        choices=["quick", "interactive"],
        default="quick",
        help="Режим запуска"
    )

    args = parser.parse_args()

    if args.mode == "quick":
        quick_test()
    else:
        interactive_mode()