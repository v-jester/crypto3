# check_balance_issue.py
"""
Поиск проблемной строки в коде
"""


def check_balance_calculation():
    file_path = "src/bots/advanced_paper_bot.py"

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        print("Поиск строк с расчётом баланса...\n")

        for i, line in enumerate(lines, 1):
            # Ищем все упоминания current_balance
            if "current_balance -=" in line and "position" in line.lower():
                print(f"Строка {i}: {line.strip()}")

                # Показываем контекст
                if i > 1:
                    print(f"  Предыдущая: {lines[i - 2].strip()}")
                if i < len(lines):
                    print(f"  Следующая: {lines[i].strip()}")
                print("-" * 50)

        # Также ищем execute_trade
        print("\nМетод execute_trade:")
        in_execute_trade = False
        for i, line in enumerate(lines, 1):
            if "async def _execute_trade" in line:
                in_execute_trade = True
                start_line = i

            if in_execute_trade:
                if "self.current_balance" in line:
                    print(f"Строка {i}: {line.strip()}")

                # Выходим из метода
                if in_execute_trade and i > start_line + 50 and line.strip() and not line.startswith(' '):
                    break

    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    check_balance_calculation()