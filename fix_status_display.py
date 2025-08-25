# fix_status_display.py
"""
Улучшение отображения статуса бота
"""


def improve_status_display():
    file_path = "src/bots/advanced_paper_bot.py"

    print("Для улучшения отображения статуса, найдите метод _log_status")
    print("и добавьте расчёт equity:")
    print("\n" + "=" * 60)
    print("ДОБАВИТЬ В МЕТОД _log_status:")
    print("=" * 60)
    print("""
    # Расчёт полного капитала (equity)
    positions_value = sum(
        pos.quantity * pos.current_price if pos.current_price > 0 
        else pos.quantity * pos.entry_price
        for pos in self.positions.values()
    )

    equity = self.current_balance + positions_value + total_pnl
    equity_return = ((equity - self.initial_balance) / self.initial_balance) * 100

    status = {
        "free_balance": round(self.current_balance, 2),
        "positions_value": round(positions_value, 2),
        "equity": round(equity, 2),
        "positions": len(self.positions),
        "trades": len(self.trade_history),
        "unrealized_pnl": round(total_pnl, 2),
        "equity_return": round(equity_return, 2)
    }

    logger.logger.info(
        f"📈 Bot Status | Equity: ${status['equity']:.2f} | "
        f"Free: ${status['free_balance']:.2f} | "
        f"In Positions: ${status['positions_value']:.2f} | "
        f"Positions: {status['positions']} | "
        f"Unrealized PnL: ${status['unrealized_pnl']:.2f} | "
        f"Return: {status['equity_return']:.2f}%"
    )
    """)


if __name__ == "__main__":
    improve_status_display()