# fix_rsi_stuck_v2.py
"""
Исправление проблемы с застрявшим RSI (версия с правильной кодировкой)
"""


def fix_historical_data():
    """Исправляем кеширование в historical_data.py"""

    file_path = "src/data/collectors/historical_data.py"

    print("Читаем файл...")

    # Читаем файл с правильной кодировкой
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Прочитано {len(lines)} строк")

    # Изменяем нужные строки
    modified = False
    for i, line in enumerate(lines):
        # Уменьшаем время кеша
        if "await redis_client.set(cache_key, cache_data, expire=60)" in line:
            lines[i] = line.replace("expire=60", "expire=5")
            print(f"✅ Строка {i + 1}: Изменено время кеша с 60 на 5 секунд")
            modified = True

        # Отключаем кеш для force_refresh
        elif "if not should_refresh:" in line and "# FIXED" not in lines[i - 1] if i > 0 else True:
            # Добавляем комментарий перед строкой
            lines[i] = "        # FIXED: Временно усиливаем обновление данных\n" + \
                       "        if not should_refresh and False:  # Кеш временно отключен\n"
            print(f"✅ Строка {i + 1}: Отключена проверка кеша")
            modified = True

    if modified:
        # Сохраняем оригинал
        print("\nСохраняем backup...")
        with open(file_path + '.bak', 'w', encoding='utf-8') as f:
            with open(file_path, 'r', encoding='utf-8') as orig:
                f.write(orig.read())

        # Записываем изменения
        print("Записываем изменения...")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        print("\n✅ Файл успешно исправлен!")
        print(f"   Backup сохранен как {file_path}.bak")
    else:
        print("\n⚠️ Изменения не требуются или уже применены")


if __name__ == "__main__":
    try:
        fix_historical_data()
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("\nПопробуйте ручное исправление (см. инструкцию ниже)")