# test_and_monitor.py
"""
Комплексный тест и мониторинг торгового бота
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from colorama import init, Fore, Style
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.bots.advanced_paper_bot import AdvancedPaperTradingBot
from src.monitoring.logger import logger
from src.data.storage.redis_client import redis_client
from src.config.settings import settings
from binance import AsyncClient

init(autoreset=True)


class BotTester:
    """Класс для тестирования и мониторинга бота"""

    def __init__(self):
        self.bot = None
        self.monitor_task = None
        self.running = False

    async def initialize_bot(self):
        """Инициализация бота с правильными настройками"""
        print(f"\n{Fore.CYAN}{'=' * 60}")
        print(f"{Fore.WHITE}ИНИЦИАЛИЗАЦИЯ ТОРГОВОГО БОТА")
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}\n")

        # Создаем бота с настройками из .env
        self.bot = AdvancedPaperTradingBot(
            initial_balance=settings.PAPER_STARTING_BALANCE,
            maker_fee=settings.PAPER_MAKER_FEE,
            taker_fee=settings.PAPER_TAKER_FEE,
            slippage_bps=settings.PAPER_SLIPPAGE_BPS
        )

        try:
            await self.bot.initialize()
            print(f"{Fore.GREEN}✅ Бот успешно инициализирован{Style.RESET_ALL}")
            print(f"   Начальный капитал: ${self.bot.initial_balance:,.2f}")
            print(f"   Символы: {', '.join(settings.trading.SYMBOLS[:3])}")
            print(f"   Таймфрейм: {settings.trading.PRIMARY_TIMEFRAME}\n")
            return True
        except Exception as e:
            print(f"{Fore.RED}❌ Ошибка инициализации: {e}{Style.RESET_ALL}")
            return False

    async def test_data_freshness(self):
        """Проверка свежести данных"""
        print(f"\n{Fore.CYAN}ПРОВЕРКА ДАННЫХ:{Style.RESET_ALL}")

        for symbol in settings.trading.SYMBOLS[:3]:
            if symbol in self.bot.indicators:
                ind = self.bot.indicators[symbol]
                last_update = self.bot.last_data_update.get(symbol)

                if last_update:
                    age = (datetime.now(timezone.utc).replace(tzinfo=None) - last_update).total_seconds()
                    status = f"{Fore.GREEN}FRESH{Style.RESET_ALL}" if age < 60 else f"{Fore.YELLOW}STALE{Style.RESET_ALL}"
                else:
                    status = f"{Fore.RED}NO DATA{Style.RESET_ALL}"
                    age = 0

                print(f"\n{Fore.GREEN}{symbol}:{Style.RESET_ALL}")
                print(f"  Цена: ${ind.get('price', 0):,.2f}")
                print(f"  RSI: {ind.get('rsi', 0):.2f}")
                print(f"  Статус данных: {status} ({age:.0f}s)")

    async def monitor_loop(self):
        """Цикл мониторинга"""
        iteration = 0
        while self.running:
            iteration += 1

            print(f"\n{Fore.YELLOW}[{datetime.now().strftime('%H:%M:%S')}] Итерация #{iteration}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'-' * 50}{Style.RESET_ALL}")

            # Статус бота
            total_pnl = sum(
                getattr(pos, 'unrealized_pnl', 0)
                for pos in self.bot.positions.values()
            )

            balance_color = Fore.GREEN if self.bot.current_balance >= self.bot.initial_balance else Fore.RED
            print(f"Баланс: {balance_color}${self.bot.current_balance:,.2f}{Style.RESET_ALL}")
            print(f"Позиций: {len(self.bot.positions)}")
            print(f"Сделок: {len(self.bot.trade_history)}")

            if total_pnl != 0:
                pnl_color = Fore.GREEN if total_pnl > 0 else Fore.RED
                print(f"Unrealized P&L: {pnl_color}${total_pnl:+.2f}{Style.RESET_ALL}")

            # Позиции
            if self.bot.positions:
                print(f"\n{Fore.CYAN}Открытые позиции:{Style.RESET_ALL}")
                for symbol, pos in self.bot.positions.items():
                    side_color = Fore.GREEN if pos.side == 'BUY' else Fore.RED
                    if hasattr(pos, 'unrealized_pnl'):
                        pnl_color = Fore.GREEN if pos.unrealized_pnl > 0 else Fore.RED
                        print(
                            f"  {symbol}: {side_color}{pos.side}{Style.RESET_ALL} @ ${pos.entry_price:.2f} | P&L: {pnl_color}${pos.unrealized_pnl:.2f}{Style.RESET_ALL}")
                    else:
                        print(f"  {symbol}: {side_color}{pos.side}{Style.RESET_ALL} @ ${pos.entry_price:.2f}")

            # RSI мониторинг
            print(f"\n{Fore.CYAN}RSI индикаторы:{Style.RESET_ALL}")
            for symbol in settings.trading.SYMBOLS[:3]:
                if symbol in self.bot.indicators:
                    rsi = self.bot.indicators[symbol].get('rsi', 0)
                    if rsi < 30:
                        rsi_color = Fore.RED
                        status = "OVERSOLD"
                    elif rsi > 70:
                        rsi_color = Fore.GREEN
                        status = "OVERBOUGHT"
                    else:
                        rsi_color = Fore.WHITE
                        status = "NEUTRAL"
                    print(f"  {symbol}: {rsi_color}{rsi:.2f} ({status}){Style.RESET_ALL}")

            await asyncio.sleep(30)  # Обновление каждые 30 секунд

    async def run_test(self, duration_minutes: int = 5):
        """Запуск теста на определенное время"""
        self.running = True

        # Инициализация
        if not await self.initialize_bot():
            return

        # Запуск бота
        bot_task = asyncio.create_task(self.bot.run())

        # Запуск мониторинга
        monitor_task = asyncio.create_task(self.monitor_loop())

        print(f"\n{Fore.GREEN}🚀 Бот запущен. Тестирование {duration_minutes} минут...{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Нажмите Ctrl+C для остановки{Style.RESET_ALL}\n")

        try:
            # Ждем указанное время
            await asyncio.sleep(duration_minutes * 60)

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Остановка по запросу пользователя...{Style.RESET_ALL}")

        finally:
            self.running = False

            # Останавливаем задачи
            bot_task.cancel()
            monitor_task.cancel()

            # Ждем завершения
            try:
                await bot_task
            except asyncio.CancelledError:
                pass

            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            # Останавливаем бота
            await self.bot.stop()

            # Финальный отчет
            await self.print_final_report()

    async def print_final_report(self):
        """Печать финального отчета"""
        print(f"\n{Fore.CYAN}{'=' * 60}")
        print(f"{Fore.WHITE}ФИНАЛЬНЫЙ ОТЧЕТ")
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}\n")

        final_balance = self.bot.current_balance
        total_pnl = final_balance - self.bot.initial_balance
        return_pct = (total_pnl / self.bot.initial_balance) * 100

        # Статистика сделок
        if self.bot.trade_history:
            wins = [t for t in self.bot.trade_history if t['pnl'] > 0]
            losses = [t for t in self.bot.trade_history if t['pnl'] < 0]
            win_rate = (len(wins) / len(self.bot.trade_history)) * 100

            print(f"Начальный баланс: ${self.bot.initial_balance:,.2f}")
            print(f"Финальный баланс: ${final_balance:,.2f}")

            pnl_color = Fore.GREEN if total_pnl > 0 else Fore.RED
            print(f"Общий P&L: {pnl_color}${total_pnl:+,.2f} ({return_pct:+.2f}%){Style.RESET_ALL}")

            print(f"\nВсего сделок: {len(self.bot.trade_history)}")
            print(f"Прибыльных: {Fore.GREEN}{len(wins)}{Style.RESET_ALL}")
            print(f"Убыточных: {Fore.RED}{len(losses)}{Style.RESET_ALL}")
            print(f"Win Rate: {win_rate:.1f}%")

            if wins:
                avg_win = sum(t['pnl'] for t in wins) / len(wins)
                print(f"Средняя прибыль: {Fore.GREEN}${avg_win:.2f}{Style.RESET_ALL}")

            if losses:
                avg_loss = sum(abs(t['pnl']) for t in losses) / len(losses)
                print(f"Средний убыток: {Fore.RED}${avg_loss:.2f}{Style.RESET_ALL}")
        else:
            print(f"Сделок не было выполнено")

        print(f"\n{Fore.GREEN}✅ Тест завершен{Style.RESET_ALL}")


async def quick_test():
    """Быстрый тест на 1 минуту"""
    tester = BotTester()
    await tester.run_test(duration_minutes=1)


async def full_test():
    """Полный тест на 10 минут"""
    tester = BotTester()
    await tester.run_test(duration_minutes=10)


async def check_rsi_updates():
    """Проверка обновления RSI"""
    print(f"\n{Fore.CYAN}ПРОВЕРКА ОБНОВЛЕНИЯ RSI{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}\n")

    client = await AsyncClient.create(
        api_key=settings.api.BINANCE_API_KEY.get_secret_value(),
        api_secret=settings.api.BINANCE_API_SECRET.get_secret_value(),
        testnet=True
    )

    symbol = "BTCUSDT"

    try:
        print(f"Отслеживаем RSI для {symbol} в течение 2 минут...\n")

        previous_rsi = None

        for i in range(4):  # 4 проверки по 30 секунд
            # Получаем свечи
            klines = await client.get_klines(
                symbol=symbol,
                interval="1m",
                limit=50
            )

            # Преобразуем в DataFrame
            df = pd.DataFrame(klines, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore"
            ])

            df['close'] = df['close'].astype(float)

            # Вычисляем RSI
            import pandas_ta as ta
            rsi = ta.rsi(df['close'], length=14)
            current_rsi = rsi.iloc[-1]

            timestamp = datetime.now().strftime('%H:%M:%S')

            if previous_rsi is not None:
                change = current_rsi - previous_rsi
                if abs(change) > 0.01:
                    print(f"[{timestamp}] RSI: {current_rsi:.2f} | Изменение: {change:+.2f} ✅")
                else:
                    print(f"[{timestamp}] RSI: {current_rsi:.2f} | Не изменился ⚠️")
            else:
                print(f"[{timestamp}] RSI: {current_rsi:.2f} | Начальное значение")

            previous_rsi = current_rsi

            if i < 3:
                await asyncio.sleep(30)

        print(f"\n{Fore.GREEN}✅ Проверка завершена{Style.RESET_ALL}")

    finally:
        await client.close_connection()


async def main():
    """Главная функция"""
    print(f"\n{Fore.CYAN}{'=' * 60}")
    print(f"{Fore.WHITE}ТЕСТИРОВАНИЕ ТОРГОВОГО БОТА")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}\n")

    print("Выберите тип теста:")
    print("1. Быстрый тест (1 минута)")
    print("2. Полный тест (10 минут)")
    print("3. Проверка обновления RSI")
    print("4. Запустить основной бот")

    choice = input("\nВаш выбор (1-4): ")

    if choice == "1":
        await quick_test()
    elif choice == "2":
        await full_test()
    elif choice == "3":
        await check_rsi_updates()
    elif choice == "4":
        print(f"\n{Fore.YELLOW}Запустите: python src/main.py{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Неверный выбор{Style.RESET_ALL}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Остановлено пользователем{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Ошибка: {e}{Style.RESET_ALL}")
        import traceback

        traceback.print_exc()