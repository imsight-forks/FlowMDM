#!/usr/bin/env pwsh
# PowerShell script to download SMPL files for FlowMDM on Windows
# Equivalent to runners/prepare/download_smpl_files.sh

param(
    [switch]$Force = $false,
    [switch]$Verbose = $false
)

Write-Host "Setting up SMPL/SMPLX files..." -ForegroundColor Green

$ErrorActionPreference = "Stop"

# Get script directory and project paths - work from script location, not current directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ScriptFileName = Split-Path -Leaf $MyInvocation.MyCommand.Path

if ($Verbose) {
    Write-Host "Script: $ScriptFileName" -ForegroundColor Yellow
    Write-Host "Script directory: $ScriptDir" -ForegroundColor Yellow
}

# Calculate FlowMDM root: script is in win64-scripts, so go up one level
$FlowMDMRoot = Split-Path -Parent $ScriptDir

# Verify we're in the right place by checking for characteristic files
if (!(Test-Path (Join-Path $FlowMDMRoot "runners")) -or !(Test-Path (Join-Path $FlowMDMRoot "model"))) {
    Write-Error "Cannot locate FlowMDM root directory. Expected to find 'runners' and 'model' directories in: $FlowMDMRoot"
    exit 1
}

if ($Verbose) {
    Write-Host "FlowMDM root: $FlowMDMRoot" -ForegroundColor Yellow
}

# Target directories in project
$BodyModelsDir = Join-Path $FlowMDMRoot "body_models"
$SmplTargetDir = Join-Path $BodyModelsDir "smpl"
$SmplhTargetDir = Join-Path $BodyModelsDir "smplh"

# Check if SMPL models already exist in target location
if ((Test-Path $SmplTargetDir) -and !$Force) {
    Write-Host "SMPL models already exist in body_models/smpl/. Use -Force to overwrite." -ForegroundColor Yellow
    Write-Host "Skipping download..." -ForegroundColor Yellow
    exit 0
}

if ((Test-Path $SmplhTargetDir) -and !$Force) {
    Write-Host "SMPLH models already exist in body_models/smplh/. Use -Force to overwrite." -ForegroundColor Yellow
    Write-Host "Skipping download..." -ForegroundColor Yellow
    exit 0
}

Write-Host "Downloading SMPL models..." -ForegroundColor Green

# Create temporary directory for downloads
$TempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("FlowMDM_SMPL_" + [System.Guid]::NewGuid().ToString("N").Substring(0,8))
New-Item -ItemType Directory -Path $TempDir | Out-Null

if ($Verbose) {
    Write-Host "Using temporary directory: $TempDir" -ForegroundColor Yellow
}

Write-Host "Downloading SMPL files to temporary location..." -ForegroundColor Cyan

# URLs for SMPL files
$SmplUrl = "https://drive.google.com/uc?id=1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2"
$SmplhUrl = "https://drive.google.com/file/d/1zHTQ1VrVgr-qGl_ahc0UDgHlXgnwx_lM/view"

try {
    # Change to temporary directory for downloads
    Set-Location $TempDir
    
    # Check if gdown is available
    $gdownAvailable = $false
    try {
        $null = Get-Command gdown -ErrorAction Stop
        $gdownAvailable = $true
    } catch {}
    
    if ($gdownAvailable) {
        Write-Host "Using gdown for downloads..." -ForegroundColor Cyan
        
        # Download SMPL files
        & gdown $SmplUrl
        & gdown --fuzzy $SmplhUrl
        
        if ($LASTEXITCODE -ne 0) {
            throw "gdown failed with exit code $LASTEXITCODE"
        }
    } else {
        Write-Host "gdown not available, using PowerShell download..." -ForegroundColor Yellow
        Write-Host "Note: Direct download may not work for all Google Drive files" -ForegroundColor Yellow
        
        # Download files using WebClient
        $webClient = New-Object System.Net.WebClient
        $webClient.Headers.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        Write-Host "Downloading SMPL..." -ForegroundColor Cyan
        $webClient.DownloadFile($SmplUrl, "smpl.zip")
        
        # For the second file, extract ID and use direct download
        $SmplhFileId = "1zHTQ1VrVgr-qGl_ahc0UDgHlXgnwx_lM"
        $SmplhDirectUrl = "https://drive.google.com/uc?export=download&id=$SmplhFileId"
        
        Write-Host "Downloading SMPLH..." -ForegroundColor Cyan
        $webClient.DownloadFile($SmplhDirectUrl, "smplh.zip")
        $webClient.Dispose()
    }
    
    # Extract files in temporary directory
    Write-Host "Extracting files in temporary directory..." -ForegroundColor Cyan
    
    # Remove existing directories if they exist
    if (Test-Path "smpl") { Remove-Item "smpl" -Recurse -Force }
    if (Test-Path "smplh") { Remove-Item "smplh" -Recurse -Force }
    
    # Extract files
    Expand-Archive -Path "smpl.zip" -DestinationPath "." -Force
    Expand-Archive -Path "smplh.zip" -DestinationPath "." -Force
    
    # Now create the target body_models directory structure
    Write-Host "Creating final directory structure..." -ForegroundColor Cyan
    if (!(Test-Path $BodyModelsDir)) {
        New-Item -ItemType Directory -Path $BodyModelsDir | Out-Null
    }
    
    # Remove existing target directories if they exist
    if (Test-Path $SmplTargetDir) { Remove-Item $SmplTargetDir -Recurse -Force }
    if (Test-Path $SmplhTargetDir) { Remove-Item $SmplhTargetDir -Recurse -Force }
    
    # Copy from temp to final location
    if (Test-Path "smpl") {
        Copy-Item "smpl" $SmplTargetDir -Recurse -Force
        Write-Host "Moved SMPL files to final location" -ForegroundColor Cyan
    }
    
    if (Test-Path "smplh") {
        Copy-Item "smplh" $SmplhTargetDir -Recurse -Force
        Write-Host "Moved SMPLH files to final location" -ForegroundColor Cyan
    }
    
    Write-Host "SMPL files download completed successfully!" -ForegroundColor Green
    
} catch {
    Write-Error "Download failed: $($_.Exception.Message)"
    exit 1
} finally {
    # Always clean up temporary directory
    if (Test-Path $TempDir) {
        Set-Location $FlowMDMRoot  # Make sure we're not in the temp dir
        Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue
        if ($Verbose) {
            Write-Host "Cleaned up temporary directory: $TempDir" -ForegroundColor Yellow
        }
    }
}