# fix_cache_final.py
"""
Финальное исправление - полное отключение кеша
"""
from pathlib import Path


def fix_historical_data():
    """Полностью отключаем использование кеша"""

    file_path = Path("src/data/collectors/historical_data.py")

    print("Исправляем historical_data.py...")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Backup
    backup_path = file_path.with_suffix('.py.backup_final')
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Backup: {backup_path}")

    # Заменяем проверку кеша на всегда False
    content = content.replace(
        "if not should_refresh:",
        "if False:  # CACHE DISABLED - always fetch fresh data"
    )

    # Также комментируем сохранение в кеш
    lines = content.split('\n')
    new_lines = []

    for line in lines:
        if "await redis_client.set(cache_key" in line and not line.strip().startswith('#'):
            new_lines.append("                    # " + line.strip() + "  # CACHE DISABLED")
        else:
            new_lines.append(line)

    content = '\n'.join(new_lines)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("✅ Кеш полностью отключен в historical_data.py")


def fix_bot_polling():
    """Исправляем интервал опроса в боте"""

    file_path = Path("src/bots/advanced_paper_bot.py")

    print("\nИсправляем advanced_paper_bot.py...")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Backup
    backup_path = file_path.with_suffix('.py.backup_final')
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)

    # Меняем интервал опроса на 5 секунд
    content = content.replace("poll_interval = 30", "poll_interval = 5")
    content = content.replace("poll_interval = 10", "poll_interval = 5")

    # Добавляем принудительное обновление
    if "self.data_collector.force_refresh = True" not in content:
        # Находим место для вставки
        lines = content.split('\n')
        new_lines = []

        for line in lines:
            new_lines.append(line)
            if "# ПРИНУДИТЕЛЬНО обновляем данные" in line:
                new_lines.append("                        self.data_collector.force_refresh = True  # FORCE UPDATE")

        content = '\n'.join(new_lines)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("✅ Интервал изменен на 5 секунд")
    print("✅ Принудительное обновление включено")


if __name__ == "__main__":
    print("=" * 60)
    print("ФИНАЛЬНОЕ ИСПРАВЛЕНИЕ ПРОБЛЕМЫ С RSI")
    print("=" * 60)
    print()

    fix_historical_data()
    fix_bot_polling()

    print("\n" + "=" * 60)
    print("✅ ВСЕ ИСПРАВЛЕНИЯ ПРИМЕНЕНЫ!")
    print("=" * 60)
    print("\nТеперь выполните:")
    print("1. python clear_cache_and_restart.py")
    print("2. python src/main.py")