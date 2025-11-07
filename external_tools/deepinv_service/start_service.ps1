param(
    [int]$Port = 6112,
    [switch]$Detach
)

$envRoot = "A:\Comfy25\envs\DeepInv"
$python = Join-Path $envRoot "Scripts\python.exe"
$service = Join-Path $PSScriptRoot "service.py"

if (-not (Test-Path $python)) {
    Write-Error "DeepInv environment python not found at $python"
    exit 1
}

$env:DEEPINV_SERVICE_PORT = $Port

if ($Detach.IsPresent) {
    Start-Process -FilePath $python -ArgumentList "`"$service`"" -WindowStyle Minimized
    Write-Host "DeepInv service started in background on port $Port" -ForegroundColor Green
} else {
    & $python $service
}
