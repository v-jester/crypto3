# run_backtest_improved.py
"""
Improved version with detailed output of results - Fixed encoding issues
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
    Run backtest with full output of results
    """
    freqtrade_path = Path("C:/freqtrade-nfi")

    # PowerShell command with output results - using raw strings to avoid escape issues
    ps_script = r"""
$ErrorActionPreference = "Continue"
Set-Location "C:\freqtrade-nfi"
& .\venv\Scripts\Activate.ps1

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "STARTING BACKTEST" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Cyan

# Run backtest
freqtrade backtesting `
    --userdir ft_userdata `
    --strategy AdvancedCryptoStrategy `
    --timerange 20250907-20250914 `
    --enable-position-stacking `
    --max-open-trades 5

# Show results
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "DETAILED RESULTS" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Cyan

freqtrade backtesting-show --userdir ft_userdata

# If results exist, show statistics
$results_file = "ft_userdata\backtest_results\.last_result.json"
if (Test-Path $results_file) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "RESULTS ANALYSIS" -ForegroundColor Yellow
    Write-Host "========================================`n" -ForegroundColor Cyan

    $json = Get-Content $results_file | ConvertFrom-Json
    $strategy_name = $json.strategy_comparison[0].key
    $results = $json.strategy.$strategy_name

    Write-Host "Strategy: $strategy_name" -ForegroundColor Green
    Write-Host "Period: $($json.backtest_start_time) - $($json.backtest_end_time)"
    Write-Host ""
    Write-Host "STATISTICS:" -ForegroundColor Yellow
    Write-Host "  Total trades: $($results.total_trades)"
    Write-Host "  Wins: $($results.wins) ($($results.win_rate)%)"
    Write-Host "  Losses: $($results.losses)"
    Write-Host ""
    Write-Host "FINANCES:" -ForegroundColor Yellow
    Write-Host "  Starting balance: $($results.starting_balance) USDT"
    Write-Host "  Final balance: $($results.final_balance) USDT"
    Write-Host "  Total profit: $($results.profit_total) USDT ($($results.profit_total_pct)%)"
    Write-Host "  Max drawdown: $($results.max_drawdown_account)%"
    Write-Host ""
    Write-Host "METRICS:" -ForegroundColor Yellow
    Write-Host "  Sharpe Ratio: $($results.sharpe)"
    Write-Host "  Sortino Ratio: $($results.sortino)"
    Write-Host "  Calmar Ratio: $($results.calmar)"
    Write-Host "  Win Rate: $($results.win_rate)%"
}
"""

    # Save and execute script
    temp_script = freqtrade_path / "show_results.ps1"
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(ps_script)

    # Run
    result = subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(temp_script)],
        encoding='utf-8',
        capture_output=False  # Show output directly
    )

    # Delete temporary file
    temp_script.unlink(missing_ok=True)

    # Read results from JSON file
    results_file = freqtrade_path / "ft_userdata" / "backtest_results" / ".last_result.json"
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)

        # Parse results
        if "strategy" in results:
            for strategy_name, strategy_results in results["strategy"].items():
                print(f"\nStrategy: {strategy_name}")
                print(f"Period: {results.get('backtest_start_time', 'N/A')} - {results.get('backtest_end_time', 'N/A')}")
                print(f"\nPerformance:")
                print(f"  • Total trades: {strategy_results.get('total_trades', 0)}")
                print(f"  • Wins: {strategy_results.get('wins', 0)}")
                print(f"  • Losses: {strategy_results.get('losses', 0)}")

                profit_total = strategy_results.get('profit_total', 0)
                profit_pct = strategy_results.get('profit_total_pct', 0)

                if profit_total > 0:
                    print(f"\nProfit: ${profit_total:.2f} ({profit_pct:.2f}%)")
                else:
                    print(f"\nLoss: ${profit_total:.2f} ({profit_pct:.2f}%)")

                print(f"\nMetrics:")
                print(f"  • Sharpe Ratio: {strategy_results.get('sharpe', 0):.2f}")
                print(f"  • Max Drawdown: {strategy_results.get('max_drawdown_account', 0):.2f}%")
                print(f"  • Win Rate: {strategy_results.get('win_rate', 0):.1f}%")

        # Recommendations
        print("\n" + "=" * 60)
        print("RECOMMENDATIONS")
        print("=" * 60)

        if results.get("strategy"):
            strategy_results = list(results["strategy"].values())[0]
            total_trades = strategy_results.get('total_trades', 0)

            if total_trades == 0:
                print("\nWarning: No trades! Possible causes:")
                print("  1. Entry conditions too strict")
                print("  2. Not enough data")
                print("  3. High indicator thresholds")
                print("\nTry:")
                print("  • Reduce buy_rsi from 35 to 30")
                print("  • Increase sell_rsi from 65 to 70")
                print("  • Reduce volume_ratio from 1.5 to 1.2")
            elif total_trades < 10:
                print("\nFew trades. Try:")
                print("  • Relaxing entry conditions")
                print("  • Adding more pairs")
                print("  • Increasing test period")
            else:
                win_rate = strategy_results.get('win_rate', 0)
                if win_rate < 40:
                    print("\nLow Win Rate. Improve:")
                    print("  • Entry filters (add trend)")
                    print("  • Stop losses (reduce from 5% to 3%)")
                    print("  • Entry timing (use higher timeframe)")


def analyze_and_optimize():
    """
    Analyze results and suggest optimizations
    """
    print("\n" + "=" * 60)
    print("STARTING OPTIMIZATION")
    print("=" * 60)

    freqtrade_path = Path("C:/freqtrade-nfi")

    ps_script = r"""
Set-Location "C:\freqtrade-nfi"
& .\venv\Scripts\Activate.ps1

Write-Host "`nStarting parameter optimization..." -ForegroundColor Yellow
Write-Host "This may take several minutes...`n" -ForegroundColor Cyan

freqtrade hyperopt `
    --userdir ft_userdata `
    --strategy AdvancedCryptoStrategy `
    --hyperopt-loss SharpeHyperOptLoss `
    --epochs 50 `
    --spaces buy sell `
    --timerange 20250801-20250914

Write-Host "`nOptimization complete!" -ForegroundColor Green
Write-Host "Best parameters saved in ft_userdata\hyperopt_results\" -ForegroundColor Yellow
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
    Create HTML report with charts
    """
    print("\nCreating visual report...")

    freqtrade_path = Path("C:/freqtrade-nfi")

    ps_script = r"""
Set-Location "C:\freqtrade-nfi"
& .\venv\Scripts\Activate.ps1

# Create charts
freqtrade plot-dataframe `
    --userdir ft_userdata `
    --strategy AdvancedCryptoStrategy `
    --pairs BTC/USDT ETH/USDT `
    --timerange 20250907-20250914

# Create profit chart
freqtrade plot-profit `
    --userdir ft_userdata `
    --strategy AdvancedCryptoStrategy `
    --timerange 20250907-20250914

Write-Host "`nReports created in ft_userdata\plot\" -ForegroundColor Green
Write-Host "Open HTML files in browser to view" -ForegroundColor Yellow
"""

    temp_script = freqtrade_path / "create_report.ps1"
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(ps_script)

    subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(temp_script)],
        encoding='utf-8'
    )

    temp_script.unlink(missing_ok=True)

    # Open the plot directory
    plot_dir = freqtrade_path / "ft_userdata" / "plot"
    if plot_dir.exists():
        os.startfile(plot_dir)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FREQTRADE RESULTS ANALYSIS")
    print("=" * 60)

    print("\nSelect action:")
    print("1. Run backtest with full output")
    print("2. Optimize parameters")
    print("3. Create visual report")
    print("4. All together")

    choice = input("\nYour choice (1-4): ").strip()

    if choice == "1":
        run_freqtrade_backtest()
    elif choice == "2":
        analyze_and_optimize()
    elif choice == "3":
        create_report()
    elif choice == "4":
        run_freqtrade_backtest()
        print("\nWould you like to run optimization? (y/n): ", end="")
        if input().lower() == 'y':
            analyze_and_optimize()
        create_report()
    else:
        # Default - run backtest
        run_freqtrade_backtest()