@echo off
REM run_bot.bat - Запуск торгового бота на Windows

echo ========================================
echo    Crypto Trading Bot - Windows Runner
echo ========================================

REM Проверяем наличие виртуального окружения
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Активируем виртуальное окружение
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Проверяем установлены ли зависимости
echo Checking dependencies...
pip show pandas >nul 2>&1
if errorlevel 1 (
    echo Dependencies not found. Installing...
    python install_dependencies.py
    if errorlevel 1 (
        echo Failed to install dependencies!
        pause
        exit /b 1
    )
)

REM Проверяем конфигурацию
if not exist ".env" (
    echo WARNING: .env file not found!
    echo Copying from .env.example...
    copy .env.example .env
    echo Please edit .env file with your settings before running the bot.
    pause
)

REM Запускаем бота
echo Starting Crypto Trading Bot...
echo.
python src/main.py

REM Если произошла ошибка
if errorlevel 1 (
    echo.
    echo Bot stopped with error code %errorlevel%
    echo Check the logs for details.
)

echo.
echo Press any key to exit...
pause >nul