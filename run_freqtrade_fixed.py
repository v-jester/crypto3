# run_freqtrade_fixed.py
"""
Fixed Freqtrade runner with proper configuration handling
"""
import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))


def check_freqtrade_setup():
    """Check if Freqtrade is properly set up"""
    freqtrade_path = Path("C:/freqtrade-nfi")

    # Check for different possible virtual environment locations
    venv_paths = [
        freqtrade_path / ".venv",
        freqtrade_path / "venv",
        freqtrade_path / ".env"
    ]

    venv_path = None
    for path in venv_paths:
        if path.exists():
            venv_path = path
            break

    if not venv_path:
        print("Warning: Virtual environment not found in expected locations")
        print("Freqtrade might be installed globally")
        return None

    return venv_path


def create_config_if_needed():
    """Create a minimal config file if it doesn't exist"""
    freqtrade_path = Path("C:/freqtrade-nfi")
    config_path = freqtrade_path / "ft_userdata" / "config.json"

    if not config_path.exists():
        print("Creating minimal config file...")

        minimal_config = {
            "max_open_trades": 5,
            "stake_currency": "USDT",
            "stake_amount": "unlimited",
            "tradable_balance_ratio": 0.99,
            "fiat_display_currency": "USD",
            "dry_run": True,
            "dry_run_wallet": 50000,
            "datadir": "ft_userdata/data",
            "user_data_dir": "ft_userdata",
            "exchange": {
                "name": "binance",
                "key": "",
                "secret": "",
                "ccxt_config": {},
                "ccxt_async_config": {},
                "pair_whitelist": [
                    "BTC/USDT",
                    "ETH/USDT"
                ]
            }
        }

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(minimal_config, f, indent=2)

        print(f"Config created at: {config_path}")
        return config_path

    return config_path


def run_freqtrade_backtest():
    """Run backtest with proper configuration"""
    freqtrade_path = Path("C:/freqtrade-nfi")
    venv_path = check_freqtrade_setup()
    config_path = create_config_if_needed()

    # Build the command based on available setup
    if venv_path:
        # Use virtual environment Python
        python_exe = venv_path / "Scripts" / "python.exe"
        freqtrade_cmd = f'"{python_exe}" -m freqtrade'
    else:
        # Try global freqtrade
        freqtrade_cmd = "freqtrade"

    # PowerShell script with proper paths
    ps_script = f"""
$ErrorActionPreference = "Continue"
Set-Location "{freqtrade_path}"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "STARTING BACKTEST" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Cyan

# Run backtest with explicit config
{freqtrade_cmd} backtesting `
    --config "{config_path}" `
    --datadir "ft_userdata/data" `
    --strategy AdvancedCryptoStrategy `
    --timerange 20250907-20250914

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "BACKTEST COMPLETE" -ForegroundColor Yellow
Write-Host "========================================`n" -ForegroundColor Cyan
"""

    # Save and execute script
    temp_script = freqtrade_path / "run_backtest.ps1"
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(ps_script)

    result = subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(temp_script)],
        capture_output=False
    )

    temp_script.unlink(missing_ok=True)

    # Try to read results
    results_file = freqtrade_path / "ft_userdata" / "backtest_results" / ".last_result.json"
    if results_file.exists():
        print("\n" + "=" * 60)
        print("READING RESULTS")
        print("=" * 60)

        with open(results_file, "r") as f:
            results = json.load(f)

        if "strategy" in results:
            for strategy_name, strategy_results in results["strategy"].items():
                print(f"\nStrategy: {strategy_name}")
                print(f"Total trades: {strategy_results.get('total_trades', 0)}")
                print(f"Win rate: {strategy_results.get('win_rate', 0):.1f}%")
                print(f"Total profit: {strategy_results.get('profit_total_pct', 0):.2f}%")


def create_report():
    """Create visual report with proper configuration"""
    print("\nCreating visual report...")

    freqtrade_path = Path("C:/freqtrade-nfi")
    venv_path = check_freqtrade_setup()
    config_path = create_config_if_needed()

    # Ensure plot directory exists
    plot_dir = freqtrade_path / "ft_userdata" / "plot"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Build the command
    if venv_path:
        python_exe = venv_path / "Scripts" / "python.exe"
        freqtrade_cmd = f'"{python_exe}" -m freqtrade'
    else:
        freqtrade_cmd = "freqtrade"

    # Create plots using direct commands
    commands = [
        # Plot dataframe
        f'{freqtrade_cmd} plot-dataframe '
        f'--config "{config_path}" '
        f'--datadir "ft_userdata/data" '
        f'--strategy AdvancedCryptoStrategy '
        f'--pairs BTC/USDT ETH/USDT '
        f'--timerange 20250907-20250914',

        # Plot profit
        f'{freqtrade_cmd} plot-profit '
        f'--config "{config_path}" '
        f'--datadir "ft_userdata/data" '
        f'--strategy AdvancedCryptoStrategy '
        f'--timerange 20250907-20250914'
    ]

    # Execute commands
    for cmd in commands:
        print(f"\nExecuting: {cmd[:50]}...")
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=str(freqtrade_path),
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Error: {result.stderr}")
        else:
            print("Success!")

    # Check for generated files
    html_files = list(plot_dir.glob("*.html"))
    if html_files:
        print(f"\n✅ Generated {len(html_files)} HTML report(s):")
        for file in html_files:
            print(f"   - {file.name}")

        # Open the plot directory
        os.startfile(plot_dir)
    else:
        print("\n⚠️ No HTML reports were generated")
        print("This could mean:")
        print("  1. No data available for the selected pairs")
        print("  2. Strategy file not found")
        print("  3. Invalid timerange")


def download_data():
    """Download historical data for backtesting"""
    print("\nDownloading historical data...")

    freqtrade_path = Path("C:/freqtrade-nfi")
    venv_path = check_freqtrade_setup()
    config_path = create_config_if_needed()

    if venv_path:
        python_exe = venv_path / "Scripts" / "python.exe"
        freqtrade_cmd = f'"{python_exe}" -m freqtrade'
    else:
        freqtrade_cmd = "freqtrade"

    # Download data command
    cmd = (
        f'{freqtrade_cmd} download-data '
        f'--config "{config_path}" '
        f'--datadir "ft_userdata/data" '
        f'--exchange binance '
        f'--pairs BTC/USDT ETH/USDT BNB/USDT SOL/USDT '
        f'--timeframe 5m 15m 1h '
        f'--days 30'
    )

    print(f"Executing: {cmd[:50]}...")
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=str(freqtrade_path),
        capture_output=False
    )

    if result.returncode == 0:
        print("\n✅ Data download complete!")
    else:
        print("\n⚠️ Data download had issues")


def check_strategy():
    """Check if the strategy file exists"""
    freqtrade_path = Path("C:/freqtrade-nfi")
    strategy_path = freqtrade_path / "ft_userdata" / "strategies" / "AdvancedCryptoStrategy.py"

    if not strategy_path.exists():
        print(f"\n⚠️ Strategy file not found at: {strategy_path}")
        print("Creating a simple test strategy...")

        # Create a minimal strategy for testing
        strategy_code = '''
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class AdvancedCryptoStrategy(IStrategy):
    INTERFACE_VERSION = 3

    minimal_roi = {
        "0": 0.10,
        "30": 0.05,
        "60": 0.02,
        "120": 0.01
    }

    stoploss = -0.05
    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['rsi'] < 30) &
            (dataframe['volume'] > 0),
            'enter_long'
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe['rsi'] > 70) &
            (dataframe['volume'] > 0),
            'exit_long'
        ] = 1
        return dataframe
'''

        strategy_path.parent.mkdir(parents=True, exist_ok=True)
        with open(strategy_path, 'w') as f:
            f.write(strategy_code)

        print(f"✅ Created test strategy at: {strategy_path}")
    else:
        print(f"✅ Strategy found at: {strategy_path}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FREQTRADE SETUP & ANALYSIS")
    print("=" * 60)

    # Check setup first
    print("\nChecking Freqtrade setup...")
    check_strategy()

    print("\nSelect action:")
    print("1. Download historical data")
    print("2. Run backtest")
    print("3. Create visual report")
    print("4. Full setup (download data, then backtest)")

    choice = input("\nYour choice (1-4): ").strip()

    if choice == "1":
        download_data()
    elif choice == "2":
        run_freqtrade_backtest()
    elif choice == "3":
        create_report()
    elif choice == "4":
        download_data()
        print("\n" + "=" * 60)
        run_freqtrade_backtest()
    else:
        print("Invalid choice")