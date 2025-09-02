#!/usr/bin/env pwsh
# PowerShell script to download GloVe embeddings for FlowMDM on Windows
# Equivalent to runners/prepare/download_glove.sh

param(
    [switch]$Force = $false,
    [switch]$Verbose = $false
)

Write-Host "Downloading GloVe embeddings (used by evaluators, not by MDM itself)..." -ForegroundColor Green

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

# Google Drive URL for GloVe
$GloveUrl = "https://drive.google.com/file/d/1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n/view?usp=sharing"
$ZipFileName = "glove.zip"
$GloveDir = Join-Path $FlowMDMRoot "glove"

# Check if GloVe directory already exists
if ((Test-Path $GloveDir) -and !$Force) {
    Write-Host "GloVe directory already exists. Use -Force to overwrite." -ForegroundColor Yellow
    Write-Host "Skipping download..." -ForegroundColor Yellow
    exit 0
}

Write-Host "Downloading GloVe embeddings..." -ForegroundColor Green

# Create temporary directory for downloads
$TempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("FlowMDM_GloVe_" + [System.Guid]::NewGuid().ToString("N").Substring(0,8))
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
        Write-Host "Using gdown for download..." -ForegroundColor Cyan
        & gdown --fuzzy $GloveUrl
        
        if ($LASTEXITCODE -ne 0) {
            throw "gdown failed with exit code $LASTEXITCODE"
        }
    } else {
        Write-Host "gdown not available, using PowerShell download..." -ForegroundColor Yellow
        
        # Extract file ID for direct download
        $FileId = "1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n"
        $DirectUrl = "https://drive.google.com/uc?export=download&id=$FileId"
        
        $webClient = New-Object System.Net.WebClient
        $webClient.Headers.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        Write-Host "Downloading GloVe from: $DirectUrl" -ForegroundColor Cyan
        $webClient.DownloadFile($DirectUrl, $ZipFileName)
        $webClient.Dispose()
    }
    
    # Extract the zip file in temporary directory
    Write-Host "Extracting $ZipFileName in temporary location..." -ForegroundColor Cyan
    Expand-Archive -Path $ZipFileName -DestinationPath "." -Force
    
    # Remove existing glove directory in target location
    if (Test-Path $GloveDir) {
        Write-Host "Removing existing glove directory..." -ForegroundColor Yellow
        Remove-Item $GloveDir -Recurse -Force
    }
    
    # Move extracted files to final location
    Write-Host "Moving extracted files to final location..." -ForegroundColor Cyan
    if (Test-Path "glove") {
        Move-Item "glove" $GloveDir -Force
    } else {
        throw "Extraction failed - glove directory not found after extraction"
    }
    
    Write-Host "GloVe download completed successfully!" -ForegroundColor Green
    
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