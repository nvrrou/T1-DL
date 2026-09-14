param(
    [ValidateRange(1, 7)][int]$Desde = 1,
    [ValidateRange(1, 7)][int]$Hasta = 7,
    [ValidateRange(1, 7)][int[]]$Omitir = @(3)
)
$ErrorActionPreference = 'Stop'
if ($Desde -gt $Hasta) { throw 'Desde debe ser menor o igual que Hasta.' }
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Crea .venv e instala requirements.txt primero.' }
$steps = @( '05_automl.py', '07_comparacion.py')
$logsPath = Join-Path $PSScriptRoot 'result_sintetico\logs'
New-Item -ItemType Directory -Force -Path $logsPath | Out-Null
for ($step = $Desde; $step -le $Hasta; $step++) {
    if ($step -in $Omitir) { continue }
    $scriptPath = Join-Path $PSScriptRoot $steps[$step - 1]
    $logName = '{0}_{1}.log' -f $steps[$step - 1], (Get-Date -Format 'yyyyMMdd_HHmmss')
    # Windows PowerShell trata stderr como NativeCommandError aunque sea un aviso.
    # Conservar el traceback completo y decidir por el codigo de salida de Python.
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $pythonPath -X utf8 -u $scriptPath 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath (Join-Path $logsPath $logName)
        $pythonExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($pythonExitCode -ne 0) { throw "Fallo $scriptPath (codigo $pythonExitCode). Revisa $logsPath\$logName." }
}
