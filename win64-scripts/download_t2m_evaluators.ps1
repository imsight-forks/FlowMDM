#!/usr/bin/env pwsh
# PowerShell script to download T2M evaluators for FlowMDM on Windows
# Equivalent to runners/prepare/download_t2m_evaluators.sh

param(
    [switch]$Force = $false,
    [switch]$Verbose = $false
)

Write-Host "Downloading T2M evaluators..." -ForegroundColor Green

$ErrorActionPreference = "Stop"

# Get script directory and calculate FlowMDM root - work from script location, not current directory
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

# Google Drive URLs for T2M evaluators
$T2mUrl = "https://drive.google.com/file/d/1ZL81tHLaGA3D7ZhLcbc7JKEs40OgzLov/view"
$DatasetUrl = "https://drive.google.com/file/d/1nNZOSlYxDjyuUHAXzauSWsEFgRi0N5ON/view"
$T2mDir = Join-Path $FlowMDMRoot "t2m"

# Check if T2M directory already exists
if ((Test-Path $T2mDir) -and !$Force) {
    Write-Host "T2M evaluators directory already exists. Use -Force to overwrite." -ForegroundColor Yellow
    Write-Host "Skipping download..." -ForegroundColor Yellow
    exit 0
}

Write-Host "Downloading T2M evaluators..." -ForegroundColor Green

# Create temporary directory for downloads
$TempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("FlowMDM_T2M_" + [System.Guid]::NewGuid().ToString("N").Substring(0,8))
New-Item -ItemType Directory -Path $TempDir | Out-Null

if ($Verbose) {
    Write-Host "Using temporary directory: $TempDir" -ForegroundColor Yellow
}

try {
    # Change to temporary directory for downloads
    Set-Location $TempDir
    
    Write-Host "Downloading to temporary location..." -ForegroundColor Cyan
    
    # Check if gdown is available
    $gdownAvailable = $false
    try {
        $null = Get-Command gdown -ErrorAction Stop
        $gdownAvailable = $true
    } catch {}
    
    if ($gdownAvailable) {
        Write-Host "Using gdown for downloads..." -ForegroundColor Cyan
        
        Write-Host "Downloading T2M evaluator..." -ForegroundColor Cyan
        & gdown --fuzzy $T2mUrl
        
        Write-Host "Downloading dataset files..." -ForegroundColor Cyan
        & gdown --fuzzy $DatasetUrl
        
        if ($LASTEXITCODE -ne 0) {
            throw "gdown failed with exit code $LASTEXITCODE"
        }
    } else {
        Write-Host "gdown not available, using PowerShell download..." -ForegroundColor Yellow
        
        # Extract file IDs for direct download
        $T2mFileId = "1ZL81tHLaGA3D7ZhLcbc7JKEs40OgzLov"
        $DatasetFileId = "1nNZOSlYxDjyuUHAXzauSWsEFgRi0N5ON"
        
        $T2mDirectUrl = "https://drive.google.com/uc?export=download&id=$T2mFileId"
        $DatasetDirectUrl = "https://drive.google.com/uc?export=download&id=$DatasetFileId"
        
        $webClient = New-Object System.Net.WebClient
        $webClient.Headers.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        Write-Host "Downloading T2M evaluator from: $T2mDirectUrl" -ForegroundColor Cyan
        $webClient.DownloadFile($T2mDirectUrl, "t2m.zip")
        
        Write-Host "Downloading dataset from: $DatasetDirectUrl" -ForegroundColor Cyan
        $webClient.DownloadFile($DatasetDirectUrl, "dataset.zip")
        
        $webClient.Dispose()
    }
    
    # Extract the zip files in temporary directory
    Write-Host "Extracting files in temporary location..." -ForegroundColor Cyan
    
    Write-Host "Extracting t2m.zip..." -ForegroundColor Cyan
    Expand-Archive -Path "t2m.zip" -DestinationPath "." -Force
    
    Write-Host "Extracting dataset.zip..." -ForegroundColor Cyan
    Expand-Archive -Path "dataset.zip" -DestinationPath "." -Force
    
    # Remove existing t2m directory in target location
    if (Test-Path $T2mDir) {
        Write-Host "Removing existing t2m directory..." -ForegroundColor Yellow
        Remove-Item $T2mDir -Recurse -Force
    }
    
    # Move extracted files to final location
    Write-Host "Moving extracted files to final location..." -ForegroundColor Cyan
    if (Test-Path "t2m") {
        Move-Item "t2m" $T2mDir -Force
    } else {
        throw "Extraction failed - t2m directory not found after extraction"
    }
    
    Write-Host "T2M evaluators download completed successfully!" -ForegroundColor Green
    
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