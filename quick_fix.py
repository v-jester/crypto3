# quick_fix.py
"""
Возвращаем правильную логику paper trading
"""

def apply_fix():
    print("Откройте src/bots/advanced_paper_bot.py")
    print("Найдите строку 633:")
    print("")
    print("БЫЛО:")
    print("  self.current_balance -= fee")
    print("")
    print("ИЗМЕНИТЕ НА:")
    print("  self.current_balance -= (position_cost + fee)")
    print("")
    print("Это правильная логика для paper trading!")
    print("После этого:")
    print("  - Баланс будет ~$20,000 после открытия 3 позиций")
    print("  - Equity будет правильным: $20,000 + $30,000 = $50,000")

if __name__ == "__main__":
    apply_fix()