param(
    [ValidateRange(1, 7)][int]$Desde = 1,
    [ValidateRange(1, 7)][int]$Hasta = 7
)
$ErrorActionPreference = 'Stop'
if ($Desde -gt $Hasta) { throw 'Desde debe ser menor o igual que Hasta.' }
$pythonPath = Join-Path (Split-Path -Parent $PSScriptRoot) '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Crea .venv e instala requirements.txt primero.' }
$steps = @('01_limpieza_datos.py', '02_feature_engineering.py', '03_baselines.py', '04_dnn.py', '05_automl.py', '06_nas.py', '07_comparacion.py')
$logsPath = Join-Path $PSScriptRoot 'results\logs'
New-Item -ItemType Directory -Force -Path $logsPath | Out-Null
for ($step = $Desde; $step -le $Hasta; $step++) {
    $scriptPath = Join-Path $PSScriptRoot $steps[$step - 1]
    $logName = '{0}_{1}.log' -f $steps[$step - 1], (Get-Date -Format 'yyyyMMdd_HHmmss')
    & $pythonPath -X utf8 -u $scriptPath 2>&1 | Tee-Object -FilePath (Join-Path $logsPath $logName)
    if ($LASTEXITCODE -ne 0) { throw "Fallo $scriptPath. Revisa el log antes de continuar." }
}
