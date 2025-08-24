# fix_logger_errors.ps1
$files = @(
    "src\bots\advanced_paper_bot.py",
    "src\data\collectors\historical_data.py",
    "src\data\collectors\websocket_client.py",
    "src\ml\models\ml_engine.py",
    "src\risk\risk_manager.py"
)

foreach ($file in $files) {
    if (Test-Path $file) {
        Write-Host "Fixing $file..." -ForegroundColor Yellow

        # Читаем файл построчно
        $lines = Get-Content $file
        $newLines = @()

        foreach ($line in $lines) {
            # Проверяем и исправляем проблемные строки
            if ($line -match 'logger\.log_error.*context.*Failed') {
                $newLines += '            logger.logger.error(f"Failed to initialize: {e}")'
            }
            elseif ($line -match 'logger\.log_error') {
                # Заменяем все вызовы log_error на logger.error
                $newLines += $line -replace 'logger\.log_error\(e,.*\)', 'logger.logger.error(f"Error: {e}")'
            }
            else {
                $newLines += $line
            }
        }

        # Сохраняем исправленный файл
        $newLines | Set-Content $file -Encoding UTF8

        Write-Host "Fixed $file" -ForegroundColor Green
    }
}

Write-Host "All files fixed!" -ForegroundColor Green