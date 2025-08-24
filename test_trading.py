# test_trading.py
"""
Простой скрипт для тестирования торговли
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.monitoring.logger import logger
import random
from datetime import datetime


class SimplePaperBot:
    def __init__(self, balance=10000):
        self.balance = balance
        self.initial_balance = balance
        self.positions = []
        self.trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

    async def run(self):
        logger.logger.info(f"Starting Simple Paper Bot with ${self.balance}")
        logger.logger.info("=" * 60)

        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']

        for i in range(30):  # 30 итераций для демонстрации
            # Выбираем случайный символ
            symbol = random.choice(symbols)

            # Генерируем случайный сигнал с большей вероятностью торговли
            signal = random.choices(
                ['BUY', 'SELL', 'HOLD'],
                weights=[35, 35, 30],  # 70% вероятность торговли
                k=1
            )[0]

            if signal != 'HOLD':
                # Генерируем случайную цену
                if symbol == 'BTCUSDT':
                    price = 50000 + random.randint(-5000, 5000)
                elif symbol == 'ETHUSDT':
                    price = 3000 + random.randint(-300, 300)
                else:
                    price = 400 + random.randint(-50, 50)

                size = 0.001 if symbol == 'BTCUSDT' else 0.01
                value = price * size

                # Симулируем P&L
                pnl = random.uniform(-50, 100)
                self.balance += pnl

                if pnl > 0:
                    self.winning_trades += 1
                    status = "WIN"
                    color = "\033[92m"  # Green
                else:
                    self.losing_trades += 1
                    status = "LOSS"
                    color = "\033[91m"  # Red

                self.trades += 1

                logger.logger.info(
                    f"Trade #{self.trades}: {signal} {symbol} | "
                    f"Price: ${price:,.2f} | Size: {size} | "
                    f"Value: ${value:,.2f} | {color}P&L: ${pnl:+.2f}\033[0m | "
                    f"Balance: ${self.balance:,.2f}"
                )

                # Добавляем небольшую задержку для реалистичности
                await asyncio.sleep(1)
            else:
                logger.logger.debug(f"No signal for {symbol}, holding...")
                await asyncio.sleep(0.5)

        # Финальная статистика
        total_pnl = self.balance - self.initial_balance
        win_rate = (self.winning_trades / self.trades * 100) if self.trades > 0 else 0

        logger.logger.info("=" * 60)
        logger.logger.info("TRADING SESSION COMPLETE")
        logger.logger.info("=" * 60)
        logger.logger.info(f"Initial Balance: ${self.initial_balance:,.2f}")
        logger.logger.info(f"Final Balance: ${self.balance:,.2f}")
        logger.logger.info(f"Total P&L: ${total_pnl:+,.2f} ({total_pnl / self.initial_balance * 100:+.2f}%)")
        logger.logger.info(f"Total Trades: {self.trades}")
        logger.logger.info(f"Winning Trades: {self.winning_trades}")
        logger.logger.info(f"Losing Trades: {self.losing_trades}")
        logger.logger.info(f"Win Rate: {win_rate:.1f}%")
        logger.logger.info("=" * 60)


async def main():
    print("\n" + "=" * 60)
    print("SIMPLE PAPER TRADING BOT TEST")
    print("=" * 60 + "\n")

    bot = SimplePaperBot(balance=10000)
    await bot.run()

    print("\nTest completed successfully!")
    print("If you see trades above, the basic trading logic is working.")
    print("Now check the main bot for issues with data fetching.\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback

        traceback.print_exc()