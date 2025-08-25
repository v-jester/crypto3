# fix_balance.py
"""
Исправление расчета баланса в advanced_paper_bot.py
"""


def fix_balance():
    # Читаем файл с правильной кодировкой
    file_path = "src/bots/advanced_paper_bot.py"

    try:
        # Читаем файл
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Ищем и заменяем нужную строку
        modified = False
        for i, line in enumerate(lines):
            if "self.current_balance -= (position_cost + fee)" in line:
                # Заменяем строку
                indent = len(line) - len(line.lstrip())
                lines[i] = " " * indent + "# Paper trading: не блокируем средства, только комиссия\n"
                lines.insert(i + 1, " " * indent + "self.current_balance -= fee  # Только комиссия\n")
                modified = True
                break

        if modified:
            # Сохраняем оригинал
            with open(file_path + '.bak', 'w', encoding='utf-8') as f:
                with open(file_path, 'r', encoding='utf-8') as orig:
                    f.write(orig.read())

            # Записываем изменения
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            print("✅ Файл исправлен!")
            print("Старая версия сохранена как advanced_paper_bot.py.bak")
        else:
            print("⚠️ Строка для замены не найдена")
            print("Возможно, файл уже исправлен")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("\nПопробуйте ручное исправление:")
        print("1. Откройте src/bots/advanced_paper_bot.py")
        print("2. Найдите строку (около 636):")
        print("   self.current_balance -= (position_cost + fee)")
        print("3. Замените на:")
        print("   self.current_balance -= fee  # Только комиссия")


if __name__ == "__main__":
    fix_balance()