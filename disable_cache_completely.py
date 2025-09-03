# disable_cache_completely.py
"""
Полное отключение кеша для исправления RSI
"""
import os
from pathlib import Path


def disable_cache():
    """Полностью отключаем кеш в historical_data.py"""

    file_path = Path("src/data/collectors/historical_data.py")

    if not file_path.exists():
        print(f"❌ Файл не найден: {file_path}")
        return False

    print("Отключаем кеш полностью...")

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Сохраняем оригинал
    backup_path = file_path.with_suffix('.py.original')
    if not backup_path.exists():
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"✅ Оригинал сохранен: {backup_path}")

    modified = False
    for i, line in enumerate(lines):
        # Полностью отключаем проверку кеша
        if "if not should_refresh:" in line:
            lines[i] = "        if False:  # КЕШИРОВАНИЕ ПОЛНОСТЬЮ ОТКЛЮЧЕНО\n"
            print(f"✅ Строка {i + 1}: Кеш отключен")
            modified = True

        # Также отключаем сохранение в кеш
        elif "await redis_client.set(cache_key" in line:
            lines[i] = "                    # " + line  # Комментируем строку
            print(f"✅ Строка {i + 1}: Сохранение в кеш отключено")
            modified = True

    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print("\n✅ Кеш полностью отключен!")
        return True
    else:
        print("⚠️ Изменения уже применены или не найдены")
        return False


def update_bot_refresh():
    """Устанавливаем принудительное обновление в боте"""

    file_path = Path("src/bots/advanced_paper_bot.py")

    if not file_path.exists():
        print(f"❌ Файл бота не найден: {file_path}")
        return False

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Находим метод _poll_market_data
    search_text = "self.data_collector.force_refresh = True"

    if search_text not in content:
        print("⚠️ force_refresh не найден, добавляем...")

        # Ищем место для вставки
        insert_after = "# ПРИНУДИТЕЛЬНО обновляем данные"
        if insert_after in content:
            content = content.replace(
                insert_after,
                insert_after + "\n                        self.data_collector.force_refresh = True  # ВСЕГДА обновлять"
            )
        else:
            # Альтернативный поиск
            insert_after = "logger.logger.debug(f\"Updating data for {symbol}\")"
            if insert_after in content:
                content = content.replace(
                    insert_after,
                    insert_after + "\n                        self.data_collector.force_refresh = True  # ВСЕГДА обновлять"
                )

    # Уменьшаем интервал
    content = content.replace("poll_interval = 30", "poll_interval = 5")
    content = content.replace("poll_interval = 10", "poll_interval = 5")

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("✅ Бот настроен на принудительное обновление каждые 5 секунд")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("ПОЛНОЕ ОТКЛЮЧЕНИЕ КЕША")
    print("=" * 60)

    success1 = disable_cache()
    print()
    success2 = update_bot_refresh()

    if success1 or success2:
        print("\n✅ ГОТОВО!")
        print("\nТеперь выполните:")
        print("1. python clear_cache_and_restart.py")
        print("2. python force_update.py")