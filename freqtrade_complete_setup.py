# freqtrade_complete_setup.py
"""
Complete Freqtrade setup with all necessary configurations
"""
import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta


def setup_complete_config():
    """Create a complete configuration file with all required fields"""
    freqtrade_path = Path("C:/freqtrade-nfi")
    config_path = freqtrade_path / "ft_userdata" / "config.json"

    complete_config = {
        "max_open_trades": 5,
        "stake_currency": "USDT",
        "stake_amount": "unlimited",
        "tradable_balance_ratio": 0.99,
        "fiat_display_currency": "USD",
        "timeframe": "5m",
        "dry_run": True,
        "dry_run_wallet": 50000,
        "cancel_open_orders_on_exit": False,
        "trading_mode": "spot",
        "margin_mode": "",
        "unfilledtimeout": {
            "entry": 10,
            "exit": 10,
            "exit_timeout_count": 0,
            "unit": "minutes"
        },
        "entry_pricing": {
            "price_side": "same",
            "use_order_book": True,
            "order_book_top": 1,
            "price_last_balance": 0.0,
            "check_depth_of_market": {
                "enabled": False,
                "bids_to_ask_delta": 1
            }
        },
        "exit_pricing": {
            "price_side": "same",
            "use_order_book": True,
            "order_book_top": 1
        },
        "exchange": {
            "name": "binance",
            "key": "",
            "secret": "",
            "ccxt_config": {},
            "ccxt_async_config": {},
            "pair_whitelist": [
                "BTC/USDT",
                "ETH/USDT",
                "BNB/USDT",
                "SOL/USDT"
            ],
            "pair_blacklist": []
        },
        "pairlists": [
            {"method": "StaticPairList"}
        ],
        "telegram": {
            "enabled": False,
            "token": "",
            "chat_id": ""
        },
        "api_server": {
            "enabled": False,
            "listen_ip_address": "127.0.0.1",
            "listen_port": 8080,
            "verbosity": "error",
            "enable_openapi": False,
            "jwt_secret_key": "somethingrandom",
            "CORS_origins": [],
            "username": "freqtrader",
            "password": "freqtrader"
        },
        "bot_name": "freqtrade",
        "initial_state": "running",
        "force_entry_enable": False,
        "internals": {
            "process_throttle_secs": 5
        },
        "datadir": "ft_userdata/data",
        "user_data_dir": "ft_userdata"
    }

    # Create directories if they don't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the config
    with open(config_path, 'w') as f:
        json.dump(complete_config, f, indent=2)

    print(f"✅ Complete config saved to: {config_path}")
    return config_path


def download_data():
    """Download historical data for backtesting"""
    print("\n📥 Downloading historical data...")

    freqtrade_path = Path("C:/freqtrade-nfi")
    os.chdir(freqtrade_path)

    # Build the command
    cmd = [
        sys.executable, "-m", "freqtrade",
        "download-data",
        "--config", "ft_userdata/config.json",
        "--exchange", "binance",
        "--pairs", "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
        "--timeframe", "5m", "15m", "1h",
        "--days", "30"
    ]

    print(f"Executing: {' '.join(cmd[:5])}...")
    result = subprocess.run(cmd, cwd=str(freqtrade_path))

    if result.returncode == 0:
        print("✅ Data download complete!")
        return True
    else:
        print("⚠️ Data download had issues")
        return False


def run_backtest():
    """Run a backtest with the strategy"""
    print("\n🚀 Running backtest...")

    freqtrade_path = Path("C:/freqtrade-nfi")
    os.chdir(freqtrade_path)

    # Calculate timerange (last 7 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    timerange = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"

    cmd = [
        sys.executable, "-m", "freqtrade",
        "backtesting",
        "--config", "ft_userdata/config.json",
        "--strategy", "AdvancedCryptoStrategy",
        "--timerange", timerange
    ]

    print(f"Executing backtest for timerange: {timerange}")
    result = subprocess.run(cmd, cwd=str(freqtrade_path))

    if result.returncode == 0:
        print("✅ Backtest complete!")

        # Try to read and display results
        results_file = freqtrade_path / "ft_userdata" / "backtest_results" / ".last_result.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)

            print("\n📊 RESULTS SUMMARY:")
            print("=" * 50)

            if "strategy" in results:
                for strategy_name, data in results["strategy"].items():
                    print(f"\nStrategy: {strategy_name}")
                    print(f"Total trades: {data.get('total_trades', 0)}")
                    print(f"Win rate: {data.get('win_rate', 0):.1f}%")
                    print(f"Total profit: {data.get('profit_total_pct', 0):.2f}%")
                    print(f"Max drawdown: {data.get('max_drawdown_account', 0):.2f}%")
                    print(f"Sharpe ratio: {data.get('sharpe', 0):.2f}")
    else:
        print("⚠️ Backtest failed")


def create_plots():
    """Create visual plots"""
    print("\n📊 Creating visual plots...")

    freqtrade_path = Path("C:/freqtrade-nfi")
    os.chdir(freqtrade_path)

    # Calculate timerange
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    timerange = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"

    # Create plot directory
    plot_dir = freqtrade_path / "ft_userdata" / "plot"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Plot dataframe
    cmd1 = [
        sys.executable, "-m", "freqtrade",
        "plot-dataframe",
        "--config", "ft_userdata/config.json",
        "--strategy", "AdvancedCryptoStrategy",
        "--pairs", "BTC/USDT", "ETH/USDT",
        "--timerange", timerange
    ]

    print("Creating dataframe plot...")
    result1 = subprocess.run(cmd1, cwd=str(freqtrade_path), capture_output=True, text=True)

    if result1.returncode != 0:
        print(f"Warning: {result1.stderr[:200]}")

    # Plot profit
    cmd2 = [
        sys.executable, "-m", "freqtrade",
        "plot-profit",
        "--config", "ft_userdata/config.json",
        "--strategy", "AdvancedCryptoStrategy",
        "--timerange", timerange
    ]

    print("Creating profit plot...")
    result2 = subprocess.run(cmd2, cwd=str(freqtrade_path), capture_output=True, text=True)

    if result2.returncode != 0:
        print(f"Warning: {result2.stderr[:200]}")

    # Check for generated files
    html_files = list(plot_dir.glob("*.html"))
    if html_files:
        print(f"\n✅ Generated {len(html_files)} HTML reports:")
        for file in html_files:
            print(f"   - {file.name}")

        # Open the directory
        os.startfile(plot_dir)
    else:
        print("\n⚠️ No plots were generated. This might be because:")
        print("  1. No backtest results available")
        print("  2. No trades were made during backtest")
        print("  3. Data issues")


def main():
    """Main function to run the complete setup"""
    print("\n" + "=" * 60)
    print("FREQTRADE COMPLETE SETUP")
    print("=" * 60)

    freqtrade_path = Path("C:/freqtrade-nfi")

    # Check if we're in the virtual environment
    if not sys.prefix.startswith(str(freqtrade_path)):
        print(f"\n⚠️ Not in Freqtrade virtual environment!")
        print(f"Current Python: {sys.executable}")
        print(f"Expected path: {freqtrade_path}\\.venv\\Scripts\\python.exe")
        print("\nTrying to use the virtual environment...")

        venv_python = freqtrade_path / ".venv" / "Scripts" / "python.exe"
        if venv_python.exists():
            # Re-run this script with the correct Python
            cmd = [str(venv_python), __file__]
            subprocess.run(cmd)
            return

    print("\nSelect action:")
    print("1. Setup complete config")
    print("2. Download historical data")
    print("3. Run backtest")
    print("4. Create visual plots")
    print("5. Complete setup (all of the above)")

    choice = input("\nYour choice (1-5): ").strip()

    if choice == "1":
        setup_complete_config()
    elif choice == "2":
        setup_complete_config()
        download_data()
    elif choice == "3":
        setup_complete_config()
        run_backtest()
    elif choice == "4":
        setup_complete_config()
        create_plots()
    elif choice == "5":
        # Complete setup
        setup_complete_config()

        if download_data():
            print("\n" + "=" * 60)
            run_backtest()
            print("\n" + "=" * 60)
            create_plots()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()