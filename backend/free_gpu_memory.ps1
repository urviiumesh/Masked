param(
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if (-not $smi) {
    throw 'nvidia-smi.exe was not found. NVIDIA drivers are required for the CUDA launch path.'
}

$rows = & $smi.Source '--query-compute-apps=pid,name,used_memory' '--format=csv,noheader,nounits' 2>$null
$apps = @($rows | Where-Object { $_ -and $_.Trim() } | ForEach-Object {
    $parts = $_ -split ',', 3 | ForEach-Object { $_.Trim() }
    if ($parts.Count -ge 2 -and $parts[0] -match '^\d+$') {
        [pscustomobject]@{ Pid = [int]$parts[0]; Name = $parts[1]; UsedMiB = if ($parts.Count -ge 3) { $parts[2] } else { '?' } }
    }
})

if ($apps.Count -eq 0) {
    Write-Host '[GPU] No NVIDIA compute processes are using VRAM.'
    exit 0
}

Write-Host '[GPU] Processes using NVIDIA compute memory:'
$apps | Format-Table -AutoSize | Out-Host
if ($DryRun) { exit 0 }

foreach ($app in $apps) {
    if ($app.Pid -eq $PID) { continue }
    $process = Get-Process -Id $app.Pid -ErrorAction SilentlyContinue
    if ($process) {
        Write-Host "[GPU] Stopping PID $($app.Pid) ($($app.Name))..."
        Stop-Process -Id $app.Pid -Force -ErrorAction Stop
    }
}

Start-Sleep -Seconds 2
$remaining = @(& $smi.Source '--query-compute-apps=pid,name,used_memory' '--format=csv,noheader,nounits' 2>$null | Where-Object { $_ -and $_.Trim() })
if ($remaining.Count -gt 0) {
    Write-Warning "Some NVIDIA compute processes remain: $($remaining -join '; ')"
} else {
    Write-Host '[GPU] NVIDIA compute memory is clear.'
}
