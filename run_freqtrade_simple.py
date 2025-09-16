# run_freqtrade_simple.py
import os
import subprocess
from pathlib import Path

# Переходим в директорию Freqtrade
os.chdir("C:/freqtrade-nfi")

print("Выберите действие:")
print("1. Загрузить данные")
print("2. Бэктест")
print("3. Показать результаты")

choice = input("Выбор: ")

python_exe = r"C:\freqtrade-nfi\.venv\Scripts\python.exe"

if choice == "1":
    cmd = f'"{python_exe}" -m freqtrade download-data --userdir ft_userdata --exchange binance --pairs BTC/USDT ETH/USDT --timeframes 5m --days 30'
    subprocess.run(cmd, shell=True)

elif choice == "2":
    cmd = f'"{python_exe}" -m freqtrade backtesting --userdir ft_userdata --strategy AdvancedCryptoStrategy --timerange 20250910-20250916'
    subprocess.run(cmd, shell=True)

elif choice == "3":
    cmd = f'"{python_exe}" -m freqtrade backtesting-show --userdir ft_userdata'
    subprocess.run(cmd, shell=True)