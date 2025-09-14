# run_backtest_improved.py
"""
Улучшенная версия с детальным выводом результатов
"""
import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from src.config.settings import settings


def run_freqtrade_backtest():
    """
    Запуск бэктеста с полным выводом результатов
    """
    freqtrade_path = Path("C:/freqtrade-nfi")

    # PowerShell команда с выводом результатов
    ps_script = """
$ErrorActionPreference = "Continue"
Set-Location "C:\freqtrade-nfi"
& .\venv\Scripts\Activate.ps1

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "ЗАПУСК БЭКТЕСТА" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Cyan

# Запуск бэктеста
freqtrade backtesting `
    --userdir ft_userdata `
    --strategy AdvancedCryptoStrategy `
    --timerange 20250907-20250914 `
    --enable-position-stacking `
    --max-open-trades 5

# Показ результатов
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Cyan

freqtrade backtesting-show --userdir ft_userdata

# Если есть результаты, показываем статистику
$results_file = "ft_userdata\backtest_results\.last_result.json"
if (Test-Path $results_file) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "АНАЛИЗ РЕЗУЛЬТАТОВ" -ForegroundColor Yellow
    Write-Host "========================================`n" -ForegroundColor Cyan

    $json = Get-Content $results_file | ConvertFrom-Json
    $strategy_name = $json.strategy_comparison[0].key
    $results = $json.strategy.$strategy_name

    Write-Host "Стратегия: $strategy_name" -ForegroundColor Green
    Write-Host "Период: $($json.backtest_start_time) - $($json.backtest_end_time)"
    Write-Host ""
    Write-Host "СТАТИСТИКА:" -ForegroundColor Yellow
    Write-Host "  Всего сделок: $($results.total_trades)"
    Write-Host "  Прибыльных: $($results.wins) ($($results.win_rate)%)"
    Write-Host "  Убыточных: $($results.losses)"
    Write-Host ""
    Write-Host "ФИНАНСЫ:" -ForegroundColor Yellow
    Write-Host "  Начальный капитал: $($results.starting_balance) USDT"
    Write-Host "  Финальный капитал: $($results.final_balance) USDT"
    Write-Host "  Общая прибыль: $($results.profit_total) USDT ($($results.profit_total_pct)%)"
    Write-Host "  Макс. просадка: $($results.max_drawdown_account)%"
    Write-Host ""
    Write-Host "МЕТРИКИ:" -ForegroundColor Yellow
    Write-Host "  Sharpe Ratio: $($results.sharpe)"
    Write-Host "  Sortino Ratio: $($results.sortino)"
    Write-Host "  Calmar Ratio: $($results.calmar)"
    Write-Host "  Win Rate: $($results.win_rate)%"
}
"""

    # Сохраняем и выполняем скрипт
    temp_script = freqtrade_path / "show_results.ps1"
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(ps_script)

    # Запускаем
    result = subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(temp_script)],
        encoding='utf-8',
        capture_output=False  # Показываем вывод напрямую
    )

    # Удаляем временный файл
    temp_script.unlink(missing_ok=True)

    # Читаем результаты из JSON файла
    results_file = freqtrade_path / "ft_userdata" / "backtest_results" / ".last_result.json"
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        print("\n" + "=" * 60)
        print("📊 СВОДКА РЕЗУЛЬТАТОВ")
        print("=" * 60)

        # Парсим результаты
        if "strategy" in results:
            for strategy_name, strategy_results in results["strategy"].items():
                print(f"\nСтратегия: {strategy_name}")
                print(
                    f"Период: {results.get('backtest_start_time', 'N/A')} - {results.get('backtest_end_time', 'N/A')}")
                print(f"\n📈 Производительность:")
                print(f"  • Всего сделок: {strategy_results.get('total_trades', 0)}")
                print(f"  • Прибыльных: {strategy_results.get('wins', 0)}")
                print(f"  • Убыточных: {strategy_results.get('losses', 0)}")

                profit_total = strategy_results.get('profit_total', 0)
                profit_pct = strategy_results.get('profit_total_pct', 0)

                if profit_total > 0:
                    print(f"\n💰 Прибыль: ${profit_total:.2f} ({profit_pct:.2f}%)")
                else:
                    print(f"\n💸 Убыток: ${profit_total:.2f} ({profit_pct:.2f}%)")

                print(f"\n📊 Метрики:")
                print(f"  • Sharpe Ratio: {strategy_results.get('sharpe', 0):.2f}")
                print(f"  • Max Drawdown: {strategy_results.get('max_drawdown_account', 0):.2f}%")
                print(f"  • Win Rate: {strategy_results.get('win_rate', 0):.1f}%")

        # Предложения по улучшению
        print("\n" + "=" * 60)
        print("💡 РЕКОМЕНДАЦИИ")
        print("=" * 60)

        if results.get("strategy"):
            strategy_results = list(results["strategy"].values())[0]
            total_trades = strategy_results.get('total_trades', 0)

            if total_trades == 0:
                print("\n⚠️ Нет сделок! Возможные причины:")
                print("  1. Слишком строгие условия входа")
                print("  2. Недостаточно данных")
                print("  3. Высокие пороги индикаторов")
                print("\n🔧 Попробуйте:")
                print("  • Снизить buy_rsi с 35 до 30")
                print("  • Увеличить sell_rsi с 65 до 70")
                print("  • Уменьшить volume_ratio с 1.5 до 1.2")
            elif total_trades < 10:
                print("\n⚠️ Мало сделок. Попробуйте:")
                print("  • Расширить условия входа")
                print("  • Добавить больше пар")
                print("  • Увеличить период тестирования")
            else:
                win_rate = strategy_results.get('win_rate', 0)
                if win_rate < 40:
                    print("\n⚠️ Низкий Win Rate. Улучшите:")
                    print("  • Фильтры входа (добавьте тренд)")
                    print("  • Стоп-лоссы (уменьшите с 5% до 3%)")
                    print("  • Тайминг входа (используйте старший таймфрейм)")


def analyze_and_optimize():
    """
    Анализ результатов и предложения по оптимизации
    """
    print("\n" + "=" * 60)
    print("🔧 ЗАПУСК ОПТИМИЗАЦИИ")
    print("=" * 60)

    freqtrade_path = Path("C:/freqtrade-nfi")

    ps_script = """
Set-Location "C:\freqtrade-nfi"
& .\venv\Scripts\Activate.ps1

Write-Host "`nЗапуск оптимизации параметров..." -ForegroundColor Yellow
Write-Host "Это может занять несколько минут...`n" -ForegroundColor Cyan

freqtrade hyperopt `
    --userdir ft_userdata `
    --strategy AdvancedCryptoStrategy `
    --hyperopt-loss SharpeHyperOptLoss `
    --epochs 50 `
    --spaces buy sell `
    --timerange 20250801-20250914

Write-Host "`n✅ Оптимизация завершена!" -ForegroundColor Green
Write-Host "Лучшие параметры сохранены в ft_userdata\hyperopt_results\" -ForegroundColor Yellow
"""

    temp_script = freqtrade_path / "optimize.ps1"
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(ps_script)

    subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(temp_script)],
        encoding='utf-8'
    )

    temp_script.unlink(missing_ok=True)


def create_report():
    """
    Создание HTML отчета с графиками
    """
    print("\n📊 Создание визуального отчета...")

    freqtrade_path = Path("C:/freqtrade-nfi")

    ps_script = """
Set-Location "C:\freqtrade-nfi"
& .\venv\Scripts\Activate.ps1

# Создание графиков
freqtrade plot-dataframe `
    --userdir ft_userdata `
    --strategy AdvancedCryptoStrategy `
    --pairs BTC/USDT ETH/USDT `
    --timerange 20250907-20250914

# Создание графика прибыли
freqtrade plot-profit `
    --userdir ft_userdata `
    --strategy AdvancedCryptoStrategy `
    --timerange 20250907-20250914

Write-Host "`n✅ Отчеты созданы в ft_userdata\plot\" -ForegroundColor Green
Write-Host "Откройте HTML файлы в браузере для просмотра" -ForegroundColor Yellow
"""

    temp_script = freqtrade_path / "create_report.ps1"
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(ps_script)

    subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(temp_script)],
        encoding='utf-8'
    )

    temp_script.unlink(missing_ok=True)

    # Открываем папку с отчетами
    plot_dir = freqtrade_path / "ft_userdata" / "plot"
    if plot_dir.exists():
        os.startfile(plot_dir)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FREQTRADE АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 60)

    print("\nВыберите действие:")
    print("1. Запустить бэктест с полным выводом")
    print("2. Оптимизировать параметры")
    print("3. Создать визуальный отчет")
    print("4. Все вместе")

    choice = input("\nВаш выбор (1-4): ").strip()

    if choice == "1":
        run_freqtrade_backtest()
    elif choice == "2":
        analyze_and_optimize()
    elif choice == "3":
        create_report()
    elif choice == "4":
        run_freqtrade_backtest()
        print("\nХотите запустить оптимизацию? (y/n): ", end="")
        if input().lower() == 'y':
            analyze_and_optimize()
        create_report()
    else:
        # По умолчанию запускаем бэктест
        run_freqtrade_backtest()