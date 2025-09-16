# run_freqtrade_setup.py
import os
import subprocess
from pathlib import Path
from datetime import datetime, timedelta


def setup_and_run_freqtrade():
    """Настройка и запуск Freqtrade"""

    freqtrade_path = Path("C:/freqtrade-nfi")
    os.chdir(freqtrade_path)

    print("=" * 60)
    print("FREQTRADE SETUP & ANALYSIS")
    print("=" * 60)
    print()

    # Проверяем наличие виртуального окружения
    venv_path = freqtrade_path / ".venv"
    if not venv_path.exists():
        venv_path = freqtrade_path / "venv"

    if not venv_path.exists():
        print("❌ Виртуальное окружение Freqtrade не найдено!")
        print("   Установите Freqtrade согласно документации")
        return

    # Путь к python в виртуальном окружении
    if os.name == 'nt':  # Windows
        python_exe = venv_path / "Scripts" / "python.exe"
        activate_cmd = f"{venv_path}\\Scripts\\activate.bat && "
    else:  # Unix/Linux
        python_exe = venv_path / "bin" / "python"
        activate_cmd = f"source {venv_path}/bin/activate && "

    print("Выберите действие:")
    print("1. Загрузить исторические данные")
    print("2. Запустить бэктест")
    print("3. Создать визуальный отчет")
    print("4. Полная настройка (все шаги)")

    choice = input("\nВаш выбор (1-4): ")

    # Таймрейндж для тестирования (последняя неделя)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    timerange = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"

    if choice in ["1", "4"]:
        print("\n📥 Загрузка данных...")
        cmd = f'"{python_exe}" -m freqtrade download-data --userdir ft_userdata --exchange binance --pairs BTC/USDT ETH/USDT SOL/USDT --timeframes 5m 15m 1h --days 30'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Данные загружены")
        else:
            print(f"⚠️ Предупреждение: {result.stderr}")

    if choice in ["2", "4"]:
        print("\n🚀 Запуск бэктеста...")
        cmd = f'"{python_exe}" -m freqtrade backtesting --userdir ft_userdata --strategy AdvancedCryptoStrategy --timerange {timerange}'

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)

        if result.returncode != 0:
            print(f"❌ Ошибка: {result.stderr}")
        else:
            print("\n✅ Бэктест завершен!")

    if choice == "3":
        print("\n📊 Создание отчета...")
        # Для отчета нужно указать правильный путь к результатам
        cmd1 = f'"{python_exe}" -m freqtrade plot-dataframe --userdir ft_userdata --strategy AdvancedCryptoStrategy --pairs BTC/USDT --timerange {timerange}'
        cmd2 = f'"{python_exe}" -m freqtrade plot-profit --userdir ft_userdata --timerange {timerange}'

        subprocess.run(cmd1, shell=True)
        subprocess.run(cmd2, shell=True)

        print("✅ Отчеты созданы в ft_userdata/plot/")

        # Открываем папку с отчетами
        plot_dir = freqtrade_path / "ft_userdata" / "plot"
        if plot_dir.exists():
            os.startfile(plot_dir)


if __name__ == "__main__":
    # Сначала создаем конфигурацию
    print("🔧 Создание конфигурации...")
    exec(open("create_freqtrade_config.py").read())

    print("\n" + "=" * 60)
    # Затем запускаем Freqtrade
    setup_and_run_freqtrade()