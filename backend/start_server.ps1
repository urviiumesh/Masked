$ErrorActionPreference = 'Stop'
$venvPython = 'D:\rv_hack\Ayudh\FaceRecognitionSystem.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) { throw "Missing requested virtual environment: $venvPython" }

# Uvicorn's reload supervisor owns the listening socket on Windows. If its
# worker is killed to release GPU memory, the supervisor can keep accepting
# connections without ever answering them. Remove any previous DHRISHTI
# server tree before binding the port again.
$allProcesses = @(Get-CimInstance Win32_Process)
$serverRoots = @($allProcesses | Where-Object {
    $_.CommandLine -and
    $_.CommandLine -match '(?i)(?:-m\s+)?uvicorn\s+api\.main:app(?:\s|$)' -and
    $_.CommandLine -match '(?i)--port(?:=|\s+)8000(?:\s|$)'
})

if ($serverRoots.Count -gt 0) {
    $processIds = [System.Collections.Generic.HashSet[int]]::new()
    $pending = [System.Collections.Generic.Queue[int]]::new()
    foreach ($process in $serverRoots) {
        [void]$processIds.Add([int]$process.ProcessId)
        $pending.Enqueue([int]$process.ProcessId)
    }
    while ($pending.Count -gt 0) {
        $parentId = $pending.Dequeue()
        foreach ($child in $allProcesses | Where-Object { $_.ParentProcessId -eq $parentId }) {
            if ($processIds.Add([int]$child.ProcessId)) {
                $pending.Enqueue([int]$child.ProcessId)
            }
        }
    }
    Write-Host "[API] Stopping previous DHRISHTI server processes: $([string]::Join(', ', $processIds))"
    Stop-Process -Id @($processIds) -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 750
}

$remainingListener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($remainingListener) {
    $owners = ($remainingListener | Select-Object -ExpandProperty OwningProcess -Unique) -join ', '
    throw "Port 8000 is still occupied by process(es): $owners"
}

& "$PSScriptRoot\free_gpu_memory.ps1"
$env:DHRISHTI_ORT_PROVIDER = 'CUDAExecutionProvider'
$env:DHRISHTI_REQUIRE_GPU = '1'
$env:PYTHONPATH = $PSScriptRoot
$nvidiaRoot = Join-Path (Split-Path $venvPython -Parent | Split-Path -Parent) 'Lib\site-packages\nvidia'
$cudaBins = @(Get-ChildItem $nvidiaRoot -Directory -Recurse | Where-Object { $_.Name -eq 'bin' } | ForEach-Object FullName)
$env:PATH = ($cudaBins -join [IO.Path]::PathSeparator) + [IO.Path]::PathSeparator + $env:PATH
Set-Location $PSScriptRoot
& $venvPython -m uvicorn api.main:app --host 0.0.0.0 --port 8000
