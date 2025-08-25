# monitor_bot.py
"""
Мониторинг работы торгового бота в реальном времени
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent))

from src.data.storage.redis_client import redis_client
from colorama import init, Fore, Style

init(autoreset=True)


async def monitor():
    """Мониторинг данных в Redis"""

    print(f"\n{Fore.CYAN}{'=' * 60}")
    print(f"{Fore.WHITE}REAL-TIME BOT MONITOR")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}\n")

    await redis_client.connect()

    if not redis_client._connected:
        print(f"{Fore.RED}❌ Redis not connected!{Style.RESET_ALL}")
        return

    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']

    try:
        iteration = 0
        while True:
            iteration += 1
            print(f"\n{Fore.YELLOW}[{datetime.now().strftime('%H:%M:%S')}] Iteration #{iteration}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'-' * 50}{Style.RESET_ALL}")

            for symbol in symbols:
                # Получаем последние данные
                market_key = f"market:{symbol}"
                market_data = await redis_client.get(market_key)

                indicators_key = f"indicators:{symbol}:15m"
                indicators = await redis_client.hgetall(indicators_key)

                kline_key = f"kline:{symbol}:15m:latest"
                kline_data = await redis_client.get(kline_key)

                # Позиции
                positions = await redis_client.get_all_positions()
                position = next((p for p in positions if p.get('symbol') == symbol), None)

                print(f"\n{Fore.GREEN}{symbol}:{Style.RESET_ALL}")

                if kline_data and isinstance(kline_data, dict):
                    price = kline_data.get('close', 0)
                    volume = kline_data.get('volume', 0)
                    print(f"  Price: ${price:,.2f} | Volume: {volume:.4f}")
                else:
                    print(f"  {Fore.YELLOW}No kline data{Style.RESET_ALL}")

                if indicators:
                    rsi = float(indicators.get('rsi', 0))
                    macd = float(indicators.get('macd', 0))
                    bb_percent = float(indicators.get('bb_percent', 0.5))

                    # Цветовая индикация RSI
                    if rsi < 30:
                        rsi_color = Fore.RED
                        rsi_status = "OVERSOLD"
                    elif rsi > 70:
                        rsi_color = Fore.GREEN
                        rsi_status = "OVERBOUGHT"
                    else:
                        rsi_color = Fore.WHITE
                        rsi_status = "NEUTRAL"

                    print(f"  RSI: {rsi_color}{rsi:.2f} ({rsi_status}){Style.RESET_ALL}")
                    print(f"  MACD: {macd:.4f}")
                    print(f"  BB%: {bb_percent:.2f}")

                if position:
                    side_color = Fore.GREEN if position['side'] == 'BUY' else Fore.RED
                    pnl = position.get('unrealized_pnl', 0)
                    pnl_color = Fore.GREEN if pnl > 0 else Fore.RED

                    print(f"  {Fore.YELLOW}POSITION:{Style.RESET_ALL}")
                    print(f"    Side: {side_color}{position['side']}{Style.RESET_ALL}")
                    print(f"    Entry: ${position.get('entry_price', 0):.2f}")
                    print(f"    P&L: {pnl_color}${pnl:.2f}{Style.RESET_ALL}")

            # Проверяем историю цен
            print(f"\n{Fore.CYAN}Price History Check:{Style.RESET_ALL}")
            for symbol in symbols:
                prices = await redis_client.get_price_history(symbol, limit=3)
                if prices:
                    latest = prices[0]
                    if isinstance(latest, dict):
                        timestamp = latest.get('timestamp', '')
                        if timestamp:
                            try:
                                price_time = datetime.fromisoformat(timestamp)
                                age = (datetime.utcnow() - price_time).total_seconds()

                                if age > 60:
                                    status = f"{Fore.RED}STALE ({age:.0f}s old){Style.RESET_ALL}"
                                else:
                                    status = f"{Fore.GREEN}FRESH ({age:.0f}s old){Style.RESET_ALL}"

                                print(f"  {symbol}: {status}")
                            except:
                                print(f"  {symbol}: {Fore.YELLOW}Parse error{Style.RESET_ALL}")
                else:
                    print(f"  {symbol}: {Fore.RED}No history{Style.RESET_ALL}")

            print(f"\n{Fore.CYAN}Press Ctrl+C to stop monitoring{Style.RESET_ALL}")
            await asyncio.sleep(10)  # Обновление каждые 10 секунд

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Monitoring stopped{Style.RESET_ALL}")
    finally:
        await redis_client.disconnect()


async def check_bot_health():
    """Проверка здоровья бота"""

    print(f"\n{Fore.CYAN}BOT HEALTH CHECK{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}\n")

    await redis_client.connect()

    checks = {
        "Redis Connection": False,
        "Market Data": False,
        "Indicators": False,
        "Price Updates": False,
        "Positions": False
    }

    # 1. Redis
    if redis_client._connected:
        checks["Redis Connection"] = True

    # 2. Market Data
    market_data = await redis_client.get("market:BTCUSDT")
    if market_data:
        checks["Market Data"] = True

    # 3. Indicators
    indicators = await redis_client.hgetall("indicators:BTCUSDT:15m")
    if indicators and 'rsi' in indicators:
        checks["Indicators"] = True

    # 4. Price Updates
    prices = await redis_client.get_price_history("BTCUSDT", limit=2)
    if len(prices) >= 2:
        # Проверяем, что цены разные
        if prices[0] != prices[1]:
            checks["Price Updates"] = True

    # 5. Positions
    positions = await redis_client.get_all_positions()
    checks["Positions"] = True  # Может быть 0 позиций

    # Вывод результатов
    for check, status in checks.items():
        if status:
            print(f"  ✅ {check}")
        else:
            print(f"  ❌ {check}")

    # Итоговая оценка
    passed = sum(checks.values())
    total = len(checks)

    print(f"\n{Fore.CYAN}Overall: {passed}/{total} checks passed{Style.RESET_ALL}")

    if passed < total:
        print(f"\n{Fore.YELLOW}Recommendations:{Style.RESET_ALL}")
        if not checks["Redis Connection"]:
            print("  - Start Redis: docker-compose up -d redis")
        if not checks["Market Data"]:
            print("  - Check Binance API connection")
        if not checks["Indicators"]:
            print("  - Check indicator calculations in historical_data.py")
        if not checks["Price Updates"]:
            print("  - Data may be stale, restart the bot")

    await redis_client.disconnect()


async def main():
    """Главная функция"""

    if len(sys.argv) > 1 and sys.argv[1] == "health":
        await check_bot_health()
    else:
        await monitor()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Stopped by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}")