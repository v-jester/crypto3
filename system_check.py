# system_check.py
"""
Проверка готовности системы для запуска торгового бота
"""
import sys
import platform
import subprocess
import importlib
from pathlib import Path


def check_python_version():
    """Проверка версии Python"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version < (3, 9):
        print("❌ Python 3.9+ required")
        return False
    else:
        print("✅ Python version OK")
        return True


def check_package(package_name, import_name=None):
    """Проверка установки пакета"""
    if import_name is None:
        import_name = package_name

    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name} - not installed")
        return False


def check_files():
    """Проверка наличия необходимых файлов"""
    required_files = [
        "src/main.py",
        "src/config/settings.py",
        "requirements.txt",
        ".env.example"
    ]

    all_ok = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - missing")
            all_ok = False

    return all_ok


def check_env_file():
    """Проверка .env файла"""
    if Path(".env").exists():
        print("✅ .env file exists")

        # Проверяем ключевые настройки
        try:
            with open(".env", "r") as f:
                content = f.read()

            required_settings = [
                "BOT_MODE",
                "API_BINANCE_API_KEY",
                "API_BINANCE_API_SECRET"
            ]

            for setting in required_settings:
                if setting in content and not content.find(f"{setting}=your_") != -1:
                    print(f"✅ {setting} configured")
                else:
                    print(f"⚠️ {setting} needs configuration")

        except Exception as e:
            print(f"⚠️ Error reading .env file: {e}")

        return True
    else:
        print("⚠️ .env file not found")
        print("   Copy .env.example to .env and configure it")
        return False


def check_optional_services():
    """Проверка опциональных сервисов"""
    print("\nOptional services:")

    # Redis
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("✅ Redis - running")
    except:
        print("⚠️ Redis - not running (optional for basic functionality)")

    # PostgreSQL
    try:
        import psycopg2
        print("✅ PostgreSQL driver installed")
    except ImportError:
        print("⚠️ PostgreSQL driver not installed")


def main():
    """Основная функция проверки"""
    print("=" * 60)
    print("Crypto Trading Bot - System Check")
    print("=" * 60)

    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print()

    # Проверка Python
    print("Checking Python...")
    python_ok = check_python_version()
    print()

    # Проверка файлов
    print("Checking project files...")
    files_ok = check_files()
    print()

    # Проверка .env
    print("Checking configuration...")
    env_ok = check_env_file()
    print()

    # Проверка основных пакетов
    print("Checking core packages...")
    core_packages = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("aiohttp", "aiohttp"),
        ("redis", "redis"),
        ("structlog", "structlog"),
        ("colorama", "colorama"),
        ("python-dotenv", "dotenv"),
        ("pydantic", "pydantic"),
    ]

    packages_ok = 0
    for pkg_name, import_name in core_packages:
        if check_package(pkg_name, import_name):
            packages_ok += 1

    print()

    # Проверка ML пакетов
    print("Checking ML packages...")
    ml_packages = [
        ("scikit-learn", "sklearn"),
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
    ]

    ml_ok = 0
    for pkg_name, import_name in ml_packages:
        if check_package(pkg_name, import_name):
            ml_ok += 1

    print()

    # Проверка опциональных сервисов
    check_optional_services()
    print()

    # Итоговый отчет
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if python_ok and files_ok and packages_ok >= 6:
        print("🎉 System ready to run the trading bot!")
        print("   Execute: python src/main.py")
    else:
        print("❌ System not ready. Please fix the issues above.")

        if not python_ok:
            print("   - Upgrade Python to 3.9+")
        if not files_ok:
            print("   - Restore missing project files")
        if packages_ok < 6:
            print("   - Install missing packages: python install_dependencies.py")

    if not env_ok:
        print("⚠️  Don't forget to configure .env file")

    if ml_ok < 3:
        print("⚠️  Some ML packages missing - advanced features may not work")

    print()
    print("Core packages:", f"{packages_ok}/{len(core_packages)}")
    print("ML packages:", f"{ml_ok}/{len(ml_packages)}")

    return python_ok and files_ok and packages_ok >= 6


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)