# fix_paper_trading_logic.py
"""
Исправление логики paper trading
"""

print("""
ВАРИАНТЫ ИСПРАВЛЕНИЯ:

1. ПРОСТОЙ (рекомендуется):
   При открытии позиции вычитать ВСЮ стоимость:
   self.current_balance -= (position_cost + fee)

   Тогда:
   - Баланс после открытия: $20,000
   - В позициях: $30,000
   - Equity = $20,000 + $30,000 = $50,000 ✅

2. СЛОЖНЫЙ (текущий):
   Вычитать только комиссию:
   self.current_balance -= fee

   НО тогда нужно изменить расчёт equity:
   equity = initial_capital - total_fees + unrealized_pnl

   А НЕ:
   equity = current_balance + positions_value

ВЫБЕРИТЕ ВАРИАНТ 1 - он логичнее для paper trading!
""")