#!/usr/bin/env pwsh
# Master PowerShell script to download all FlowMDM dependencies on Windows

param(
    [switch]$PretrainedModels = $false,
    [switch]$SmplFiles = $false,
    [switch]$Glove = $false,
    [switch]$T2mEvaluators = $false,
    [switch]$All = $false,
    [switch]$Force = $false,
    [switch]$Verbose = $false
)

Write-Host "FlowMDM Download Manager for Windows" -ForegroundColor Magenta
Write-Host "=====================================" -ForegroundColor Magenta

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# If no specific downloads are requested and not All, show usage
if (!$PretrainedModels -and !$SmplFiles -and !$Glove -and !$T2mEvaluators -and !$All) {
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\download_all.ps1 -All                    # Download everything"
    Write-Host "  .\download_all.ps1 -PretrainedModels       # Download pretrained models only"
    Write-Host "  .\download_all.ps1 -SmplFiles              # Download SMPL files only"
    Write-Host "  .\download_all.ps1 -Glove                  # Download GloVe embeddings only"
    Write-Host "  .\download_all.ps1 -T2mEvaluators          # Download T2M evaluators only"
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -Force                                     # Overwrite existing files"
    Write-Host "  -Verbose                                   # Show detailed output"
    Write-Host ""
    exit 0
}

$downloadArgs = @()
if ($Force) { $downloadArgs += "-Force" }
if ($Verbose) { $downloadArgs += "-Verbose" }

try {
    if ($All -or $PretrainedModels) {
        Write-Host ""
        Write-Host "[1/4] Downloading Pretrained Models..." -ForegroundColor Cyan
        & "$ScriptDir\download_pretrained_models.ps1" @downloadArgs
        if ($LASTEXITCODE -ne 0) { throw "Pretrained models download failed" }
    }
    
    if ($All -or $SmplFiles) {
        Write-Host ""
        Write-Host "[2/4] Downloading SMPL Files..." -ForegroundColor Cyan
        & "$ScriptDir\download_smpl_files.ps1" @downloadArgs
        if ($LASTEXITCODE -ne 0) { throw "SMPL files download failed" }
    }
    
    if ($All -or $Glove) {
        Write-Host ""
        Write-Host "[3/4] Downloading GloVe Embeddings..." -ForegroundColor Cyan
        & "$ScriptDir\download_glove.ps1" @downloadArgs
        if ($LASTEXITCODE -ne 0) { throw "GloVe download failed" }
    }
    
    if ($All -or $T2mEvaluators) {
        Write-Host ""
        Write-Host "[4/4] Downloading T2M Evaluators..." -ForegroundColor Cyan
        & "$ScriptDir\download_t2m_evaluators.ps1" @downloadArgs
        if ($LASTEXITCODE -ne 0) { throw "T2M evaluators download failed" }
    }
    
    Write-Host ""
    Write-Host "All requested downloads completed successfully!" -ForegroundColor Green
    
} catch {
    Write-Error "Download process failed: $($_.Exception.Message)"
    exit 1
}