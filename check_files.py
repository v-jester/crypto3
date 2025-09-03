# check_files.py
import os
from pathlib import Path


def check_structure():
    """Проверка структуры проекта"""

    files_to_check = [
        "src/data/collectors/historical_data.py",
        "src/bots/advanced_paper_bot.py",
        "src/config/settings.py"
    ]

    print("Проверка файлов:\n")

    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            print(f"✅ {file_path} - {size} байт")
        else:
            print(f"❌ {file_path} - НЕ НАЙДЕН")

    # Проверяем текущую директорию
    print(f"\nТекущая директория: {os.getcwd()}")

    # Проверяем содержимое src
    src_path = Path("src")
    if src_path.exists():
        print("\nСодержимое src/:")
        for item in src_path.iterdir():
            print(f"  - {item.name}/")


if __name__ == "__main__":
    check_structure()