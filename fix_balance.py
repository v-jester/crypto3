# fix_balance.py
"""
Исправление расчета баланса в advanced_paper_bot.py
"""
import fileinput
import sys

file_path = "src/bots/advanced_paper_bot.py"

# Ищем и заменяем строку
with fileinput.FileInput(file_path, inplace=True, backup='.bak') as file:
    for line in file:
        if "self.current_balance -= (position_cost + fee)" in line:
            print("            # Paper trading: не блокируем средства, только комиссия")
            print("            self.current_balance -= fee  # Только комиссия")
        else:
            print(line, end='')

print("✅ Файл исправлен!")
print("Старая версия сохранена как advanced_paper_bot.py.bak")