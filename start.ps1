[CmdletBinding()]
param(
    [ValidateSet('start', 'stop', 'status', 'logs')]
    [string]$Action = 'start',
    [switch]$NoBuild
)
$ErrorActionPreference = 'Stop'
$backendPath = Join-Path $PSScriptRoot 'MediFlow'
$composeArgs = @('--context', 'desktop-linux', 'compose', '--project-directory', $backendPath,
    '-f', (Join-Path $backendPath 'docker-compose.yml'),
    '-f', (Join-Path $backendPath 'docker-compose.windows.yml'))

function Invoke-Docker {
    & docker @composeArgs @args
    if ($LASTEXITCODE -ne 0) { throw "Docker command failed (exit $LASTEXITCODE)." }
}

function Get-DockerEngine {
    # Windows PowerShell 5.1 treats native stderr as errors, even with 2>$null.
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $engine = & docker --context desktop-linux info --format '{{.OSType}}' 2>$null
        if ($LASTEXITCODE -eq 0) { return $engine }
    } finally { $ErrorActionPreference = $previousPreference }
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Install Docker Desktop, enable the WSL 2 backend, and use Linux containers.'
}
$engineType = Get-DockerEngine
if (-not $engineType) {
    $dockerBinary = (Get-Command docker).Source
    $desktopRoot = Split-Path (Split-Path (Split-Path $dockerBinary))
    $desktopExe = Join-Path $desktopRoot 'Docker Desktop.exe'
    if (-not (Test-Path -LiteralPath $desktopExe)) {
        throw 'Open Docker Desktop and wait until the engine is running, then try again.'
    }
    Start-Process -FilePath $desktopExe -WindowStyle Hidden
    Write-Host 'Waiting for Docker Desktop...'
    $ready = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        Start-Sleep -Seconds 2
        $engineType = Get-DockerEngine
        if ($engineType) { $ready = $true; break }
    }
    if (-not $ready) { throw 'Docker Desktop is not ready. Check its WSL 2 engine status.' }
}
if ($engineType -ne 'linux') { throw 'Switch Docker Desktop to Linux containers.' }

$envPath = Join-Path $backendPath '.env'
if (-not (Test-Path -LiteralPath $envPath)) {
    Copy-Item -LiteralPath (Join-Path $backendPath '.env.example') -Destination $envPath
    Write-Host 'Created MediFlow/.env. Set SILICONFLOW_API_KEY and SILICONFLOW_MODEL.'
}

switch ($Action) {
    'start' {
        $settingsText = Get-Content -LiteralPath $envPath -Raw
        $selectedProvider = if ($settingsText -match '(?m)^LLM_PROVIDER[ \t]*=[ \t]*([^\r\n]*)') {
            $Matches[1].Trim().Trim('"').Trim("'").ToLowerInvariant()
        } else { 'anthropic' }
        $keyName = switch ($selectedProvider) {
            'siliconflow' { 'SILICONFLOW_API_KEY' }
            'openai' { 'OPENAI_API_KEY' }
            'anthropic' { 'ANTHROPIC_API_KEY' }
            default { throw 'LLM_PROVIDER must be siliconflow, openai or anthropic.' }
        }
        $hasKey = $settingsText -match ('(?m)^' + $keyName + '[ \t]*=[ \t]*([^\r\n]*)')
        $keyValue = if ($hasKey) { $Matches[1].Trim().Trim('"').Trim("'") } else { '' }
        if ($keyValue -in @('', 'xxx', 'your-api-key', 'your-openai-api-key', 'your-siliconflow-api-key')) {
            if ($selectedProvider -in @('openai','siliconflow')) {
                Write-Host "$keyName is not configured. Emergency guidance remains available; AI consultation requires a key."
            } else {
                throw 'Set a valid ANTHROPIC_API_KEY in MediFlow/.env before starting.'
            }
        }
        Invoke-Docker config --quiet
        $upArgs = @('up', '-d', '--wait', '--wait-timeout', '300')
        if (-not $NoBuild) { $upArgs += '--build' }
        Invoke-Docker @upArgs
        Write-Host 'Web: http://localhost:5175'
        Write-Host 'API docs: http://localhost:8002/docs'
        Write-Host 'Prometheus: http://localhost:9090'
    }
    'stop' { Invoke-Docker stop }
    'status' { Invoke-Docker ps }
    'logs' { Invoke-Docker logs --tail 100 -f mediflow frontend }
}
