# src/utils/banner.py
"""
Баннер и визуальные утилиты для торгового бота
"""
from colorama import init, Fore, Style
from src.config.settings import settings
import platform
import psutil

# Инициализация colorama для поддержки цветов в Windows
init(autoreset=True)


def print_banner():
    """Печать красивого баннера при запуске"""

    banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                                  ║
║  {Fore.WHITE}  ██████╗██████╗ ██╗   ██╗██████╗ ████████╗ ██████╗     ██████╗  ██████╗ ████████╗{Fore.CYAN} ║
║  {Fore.WHITE} ██╔════╝██╔══██╗╚██╗ ██╔╝██╔══██╗╚══██╔══╝██╔═══██╗    ██╔══██╗██╔═══██╗╚══██╔══╝{Fore.CYAN} ║
║  {Fore.WHITE} ██║     ██████╔╝ ╚████╔╝ ██████╔╝   ██║   ██║   ██║    ██████╔╝██║   ██║   ██║   {Fore.CYAN} ║
║  {Fore.WHITE} ██║     ██╔══██╗  ╚██╔╝  ██╔═══╝    ██║   ██║   ██║    ██╔══██╗██║   ██║   ██║   {Fore.CYAN} ║
║  {Fore.WHITE} ╚██████╗██║  ██║   ██║   ██║        ██║   ╚██████╔╝    ██████╔╝╚██████╔╝   ██║   {Fore.CYAN} ║
║  {Fore.WHITE}  ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═╝        ╚═╝    ╚═════╝     ╚═════╝  ╚═════╝    ╚═╝   {Fore.CYAN} ║
║                                                                                  ║
║  {Fore.YELLOW}                     Advanced Cryptocurrency Trading System                      {Fore.CYAN} ║
║                                                                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║"""

    # Информация о системе
    mode_color = {
        "monitor": Fore.BLUE,
        "paper": Fore.YELLOW,
        "live": Fore.RED
    }.get(settings.BOT_MODE.value, Fore.WHITE)

    env_color = {
        "development": Fore.GREEN,
        "testing": Fore.YELLOW,
        "staging": Fore.MAGENTA,
        "production": Fore.RED
    }.get(settings.ENVIRONMENT.value, Fore.WHITE)

    # Системная информация
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()

    info_lines = [
        (f"  Version", f"{settings.VERSION}", Fore.GREEN),
        (f"  Environment", f"{settings.ENVIRONMENT.value.upper()}", env_color),
        (f"  Bot Mode", f"{settings.BOT_MODE.value.upper()}", mode_color),
        (f"  Trading Mode", f"{settings.TRADING_MODE.value.upper()}", Fore.CYAN),
        ("", "", ""),
        (f"  Initial Capital", f"${settings.trading.INITIAL_CAPITAL:,.2f}", Fore.GREEN),
        (f"  Risk per Trade", f"{settings.trading.RISK_PER_TRADE * 100:.1f}%", Fore.YELLOW),
        (f"  Max Positions", f"{settings.trading.MAX_POSITIONS}", Fore.CYAN),
        (f"  Primary Timeframe", f"{settings.trading.PRIMARY_TIMEFRAME}", Fore.MAGENTA),
        ("", "", ""),
        (f"  System", f"{platform.system()} {platform.release()}", Fore.WHITE),
        (f"  Python", f"{platform.python_version()}", Fore.WHITE),
        (f"  CPU Usage", f"{cpu_percent:.1f}%", Fore.GREEN if cpu_percent < 50 else Fore.YELLOW),
        (f"  Memory Usage",
         f"{memory.percent:.1f}% ({memory.used / (1024 ** 3):.1f}GB / {memory.total / (1024 ** 3):.1f}GB)",
         Fore.GREEN if memory.percent < 70 else Fore.YELLOW),
    ]

    for label, value, color in info_lines:
        if label == "" and value == "":
            banner += f"\n║  {' ' * 78} ║"
        else:
            formatted_line = f"  {label:<20} : {color}{value}{Fore.CYAN}"
            # Подсчёт реальной длины без ANSI кодов
            visible_length = len(f"  {label:<20} : {value}")
            padding = 78 - visible_length
            banner += f"\n║{formatted_line}{' ' * padding} ║"

    banner += f"""
║                                                                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  {Fore.WHITE}Trading Pairs:{Fore.CYAN}                                                                   ║"""

    # Торговые пары
    symbols = settings.trading.SYMBOLS
    for i in range(0, len(symbols), 6):
        batch = symbols[i:i + 6]
        symbols_str = ", ".join(batch)
        formatted_line = f"  {Fore.GREEN}{symbols_str}{Fore.CYAN}"
        visible_length = len(f"  {symbols_str}")
        padding = 78 - visible_length
        banner += f"\n║{formatted_line}{' ' * padding} ║"

    # Предупреждения
    warnings = []

    if settings.BOT_MODE.value == "live":
        if settings.ENVIRONMENT.value != "production":
            warnings.append((f"⚠️  LIVE TRADING IN {settings.ENVIRONMENT.value.upper()} ENVIRONMENT!", Fore.RED))
        if settings.api.TESTNET:
            warnings.append(("⚠️  TESTNET MODE ENABLED FOR LIVE TRADING!", Fore.YELLOW))

    if settings.DEBUG:
        warnings.append(("⚠️  DEBUG MODE ENABLED", Fore.YELLOW))

    if warnings:
        banner += f"""
║                                                                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  {Fore.RED}Warnings:{Fore.CYAN}                                                                        ║"""

        for warning, color in warnings:
            formatted_line = f"  {color}{warning}{Fore.CYAN}"
            visible_length = len(f"  {warning}")
            padding = 78 - visible_length
            banner += f"\n║{formatted_line}{' ' * padding} ║"

    banner += f"""
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
{Style.RESET_ALL}"""

    print(banner)


def print_section_header(title: str, color=Fore.CYAN):
    """Печать заголовка секции"""
    width = 80
    padding = (width - len(title) - 2) // 2
    header = f"""
{color}{'═' * width}
{'═' * padding} {Fore.WHITE}{title}{color} {'═' * (width - padding - len(title) - 2)}
{'═' * width}{Style.RESET_ALL}
"""
    print(header)


def print_status_table(data: dict, title: str = "Status"):
    """Печать таблицы статуса"""
    print(f"\n{Fore.CYAN}┌{'─' * 78}┐")
    print(f"│ {Fore.WHITE}{title:^76}{Fore.CYAN} │")
    print(f"├{'─' * 78}┤")

    for key, value in data.items():
        # Определение цвета на основе значения
        if isinstance(value, bool):
            color = Fore.GREEN if value else Fore.RED
            value_str = "✓" if value else "✗"
        elif isinstance(value, (int, float)):
            if value > 0:
                color = Fore.GREEN
            elif value < 0:
                color = Fore.RED
            else:
                color = Fore.YELLOW
            value_str = f"{value:,.2f}" if isinstance(value, float) else str(value)
        else:
            color = Fore.WHITE
            value_str = str(value)

        print(f"│ {Fore.YELLOW}{key:<35}{color}{value_str:>40}{Fore.CYAN} │")

    print(f"└{'─' * 78}┘{Style.RESET_ALL}")


def format_pnl(pnl: float, with_sign: bool = True) -> str:
    """Форматирование P&L с цветом"""
    if pnl > 0:
        color = Fore.GREEN
        sign = "+" if with_sign else ""
    elif pnl < 0:
        color = Fore.RED
        sign = ""
    else:
        color = Fore.YELLOW
        sign = ""

    return f"{color}{sign}{pnl:,.2f}{Style.RESET_ALL}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Форматирование процентов с цветом"""
    if value > 0:
        color = Fore.GREEN
    elif value < 0:
        color = Fore.RED
    else:
        color = Fore.YELLOW

    return f"{color}{value:.{decimals}f}%{Style.RESET_ALL}"


def progress_bar(current: int, total: int, width: int = 50, title: str = "") -> str:
    """Создание прогресс-бара"""
    if total == 0:
        percent = 0
    else:
        percent = current / total

    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)

    if percent < 0.33:
        color = Fore.RED
    elif percent < 0.66:
        color = Fore.YELLOW
    else:
        color = Fore.GREEN

    return f"{title} {color}[{bar}] {percent * 100:.1f}%{Style.RESET_ALL} ({current}/{total})"


def print_trade_summary(trade_data: dict):
    """Печать сводки по сделке"""
    side_color = Fore.GREEN if trade_data.get("side") == "BUY" else Fore.RED
    pnl = trade_data.get("pnl", 0)

    print(f"""
{Fore.CYAN}╔══════════════════════ TRADE EXECUTED ══════════════════════╗
║ {Fore.WHITE}Symbol:     {Fore.YELLOW}{trade_data.get('symbol', 'N/A'):<20}{Fore.WHITE} Side: {side_color}{trade_data.get('side', 'N/A'):<10}{Fore.CYAN} ║
║ {Fore.WHITE}Price:      {Fore.GREEN}${trade_data.get('price', 0):,.4f}{' ' * (39 - len(f"{trade_data.get('price', 0):,.4f}"))}{Fore.CYAN} ║
║ {Fore.WHITE}Quantity:   {Fore.MAGENTA}{trade_data.get('quantity', 0):.8f}{' ' * (39 - len(f"{trade_data.get('quantity', 0):.8f}"))}{Fore.CYAN} ║
║ {Fore.WHITE}Value:      {Fore.BLUE}${trade_data.get('value', 0):,.2f}{' ' * (39 - len(f"{trade_data.get('value', 0):,.2f}"))}{Fore.CYAN} ║
║ {Fore.WHITE}P&L:        {format_pnl(pnl)}{' ' * (39 - len(f"{pnl:,.2f}"))}{Fore.CYAN} ║
║ {Fore.WHITE}Order ID:   {Fore.CYAN}{trade_data.get('order_id', 'N/A'):<45}{Fore.CYAN} ║
╚═════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")


def clear_screen():
    """Очистка экрана (кроссплатформенная)"""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')