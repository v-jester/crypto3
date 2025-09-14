# run_freqtrade_test.ps1
# Автоматический запуск тестов через Freqtrade

# Активируем виртуальное окружение Freqtrade
Set-Location "C:\freqtrade-nfi"
.\venv\Scripts\Activate.ps1

# Копируем стратегию
Write-Host "Подготовка стратегии..." -ForegroundColor Yellow
python C:\path\to\your\project\run_backtest.py --mode quick

# Запускаем веб-интерфейс для анализа
Write-Host "`nЗапуск веб-интерфейса..." -ForegroundColor Green
Start-Process "http://localhost:8080"

# Запускаем Freqtrade UI
freqtrade webserver --config user_data/config_advanced.json

Write-Host "`nТест завершен! Откройте браузер для просмотра результатов" -ForegroundColor Green