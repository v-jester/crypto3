# install_dependencies.py
"""
Скрипт для установки зависимостей с учетом операционной системы
"""
import subprocess
import sys
import platform
import os


def run_command(command):
    """Выполнение команды и вывод результата"""
    print(f"Executing: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def install_basic_requirements():
    """Установка базовых зависимостей"""
    print("Installing basic requirements...")
    return run_command("pip install -r requirements.txt")


def install_platform_specific():
    """Установка зависимостей для конкретной платформы"""
    system = platform.system()
    print(f"Detected system: {system}")

    if system != "Windows":
        print("Installing Unix-specific packages...")
        # uvloop для лучшей производительности на Unix
        run_command("pip install uvloop==0.19.0")

    # TA-Lib - сложная установка
    print("Attempting to install TA-Lib...")
    if system == "Windows":
        print("For Windows, you may need to install TA-Lib manually:")
        print("1. Download wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib")
        print("2. pip install downloaded_wheel.whl")
        print("Skipping automatic TA-Lib installation...")
    else:
        # Пытаемся установить TA-Lib на Unix
        success = run_command("pip install TA-Lib==0.4.28")
        if not success:
            print("Failed to install TA-Lib automatically")
            print("Please install it manually or system will use pandas-ta instead")


def install_optional_packages():
    """Установка опциональных пакетов"""
    optional_packages = [
        "questdb==1.0.1",  # Для высокопроизводительного хранения тиков
    ]

    for package in optional_packages:
        print(f"Installing optional package: {package}")
        success = run_command(f"pip install {package}")
        if not success:
            print(f"Failed to install {package}, continuing...")


def main():
    """Основная функция установки"""
    print("=" * 60)
    print("Crypto Trading Bot - Dependencies Installation")
    print("=" * 60)

    # Проверяем версию Python
    python_version = sys.version_info
    if python_version < (3, 9):
        print(f"Warning: Python {python_version.major}.{python_version.minor} detected.")
        print("Recommended Python version: 3.9+")

    # Обновляем pip
    print("Updating pip...")
    run_command("python -m pip install --upgrade pip")

    # Устанавливаем базовые зависимости
    if not install_basic_requirements():
        print("Failed to install basic requirements!")
        return 1

    # Устанавливаем платформо-специфичные пакеты
    install_platform_specific()

    # Устанавливаем опциональные пакеты
    install_optional_packages()

    print("=" * 60)
    print("Installation completed!")
    print("=" * 60)

    # Проверяем ключевые пакеты
    print("Checking key packages...")
    key_packages = ["pandas", "numpy", "aiohttp", "redis", "structlog"]

    for package in key_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - FAILED")

    print("\nNow you can run: python src/main.py")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)