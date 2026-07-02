param(
    [string]$SourceRoot = "C:\Tesis\TPScouting",
    [string]$DeliveryRoot = "C:\Tesis\TPScouting-entrega"
)

$ErrorActionPreference = "Stop"

function Resolve-RequiredPath([string]$PathValue) {
    if (-not (Test-Path -LiteralPath $PathValue)) {
        throw "No existe la ruta requerida: $PathValue"
    }
    return (Resolve-Path -LiteralPath $PathValue).Path
}

function Assert-WithinRoot([string]$Root, [string]$PathValue) {
    $resolvedRoot = Resolve-RequiredPath $Root
    $resolvedPath = if (Test-Path -LiteralPath $PathValue) {
        (Resolve-Path -LiteralPath $PathValue).Path
    } else {
        [System.IO.Path]::GetFullPath($PathValue)
    }
    if (-not ($resolvedPath -eq $resolvedRoot -or $resolvedPath.StartsWith($resolvedRoot + [System.IO.Path]::DirectorySeparatorChar))) {
        throw "Ruta fuera del root permitido: $resolvedPath"
    }
}

function Copy-FileSafe([string]$RelativePath) {
    $src = Join-Path $SourceRoot $RelativePath
    $dst = Join-Path $DeliveryRoot $RelativePath
    if (-not (Test-Path -LiteralPath $src)) {
        Write-Host "Omitido, no existe: $RelativePath"
        return
    }
    Assert-WithinRoot $SourceRoot $src
    Assert-WithinRoot $DeliveryRoot (Split-Path -Parent $dst)
    $parent = Split-Path -Parent $dst
    if (-not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    Copy-Item -LiteralPath $src -Destination $dst -Force
}

function Copy-DirectoryFiltered([string]$RelativePath, [string[]]$IncludeExtensions = @()) {
    $srcDir = Join-Path $SourceRoot $RelativePath
    $dstDir = Join-Path $DeliveryRoot $RelativePath
    if (-not (Test-Path -LiteralPath $srcDir)) {
        Write-Host "Omitido, no existe: $RelativePath"
        return
    }
    Assert-WithinRoot $SourceRoot $srcDir
    if (-not (Test-Path -LiteralPath $dstDir)) {
        New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
    }

    Get-ChildItem -LiteralPath $srcDir -Recurse -File | ForEach-Object {
        $relative = $_.FullName.Substring($srcDir.Length + 1)
        $normalized = $relative -replace "\\", "/"

        if ($normalized -match "(^|/)__pycache__/") { return }
        if ($normalized -match "(^|/)\.pytest_cache/") { return }
        if ($normalized -match "(^|/)\.vscode/") { return }
        if ($_.Name -in @("players.db", "registro_mejoras.md", "dashboard.html.bak")) { return }
        if ($_.Extension -in @(".db", ".sqlite3", ".csv", ".bak", ".pyc", ".pyo")) { return }
        if ($_.Name -in @("experiments.csv", "training_metadata.json", "training_splits.json", "temporal_training_dataframe.joblib")) { return }
        if ($_.Name -like "*.before_*") { return }
        if ($_.Name -like "*.db.*") { return }

        if ($IncludeExtensions.Count -gt 0 -and ($IncludeExtensions -notcontains $_.Extension)) {
            return
        }

        $target = Join-Path $dstDir $relative
        $targetParent = Split-Path -Parent $target
        Assert-WithinRoot $DeliveryRoot $targetParent
        if (-not (Test-Path -LiteralPath $targetParent)) {
            New-Item -ItemType Directory -Path $targetParent -Force | Out-Null
        }
        Copy-Item -LiteralPath $_.FullName -Destination $target -Force
    }
}

$SourceRoot = Resolve-RequiredPath $SourceRoot
$DeliveryRoot = Resolve-RequiredPath $DeliveryRoot

if (-not (Test-Path -LiteralPath (Join-Path $DeliveryRoot ".git"))) {
    throw "El destino no parece ser un repo Git: $DeliveryRoot"
}

Write-Host "Sincronizando lista blanca desde:"
Write-Host "  Source:   $SourceRoot"
Write-Host "  Entrega:  $DeliveryRoot"

Copy-FileSafe ".github\workflows\ci.yml"
Copy-FileSafe "requirements.txt"
Copy-FileSafe "requirements-lock.txt"

Copy-DirectoryFiltered "scouting_app"
Copy-DirectoryFiltered "tests" @(".py")
Copy-FileSafe "scripts\smoke_render.py"

Write-Host ""
Write-Host "Sincronizacion terminada. Revisar ahora:"
Write-Host "  cd $DeliveryRoot"
Write-Host "  git status --short"
Write-Host "  python -m pytest -q"
Write-Host "  revisar manualmente docs, diagramas, render.yaml y requirements-dev.txt si cambiaron"
Write-Host ""
Write-Host "No se hizo commit ni push automaticamente."
