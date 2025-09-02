#!/usr/bin/env pwsh
# PowerShell script to download FlowMDM pretrained models on Windows
# Equivalent to runners/prepare/download_pretrained_models.sh

param(
    [switch]$Force = $false,
    [switch]$Verbose = $false
)

Write-Host "Downloading FlowMDM pretrained models..." -ForegroundColor Green

# Set error action preference
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

# Google Drive file ID for pretrained models
$GoogleDriveUrl = "https://drive.google.com/file/d/1fyx4rd6P_m26Vzb2xEvlkkSw9aDhT8rV/view"
$ZipFileName = "results.zip"
$ResultsDir = Join-Path $FlowMDMRoot "results"

# Check if results directory already exists
if ((Test-Path $ResultsDir) -and !$Force) {
    Write-Host "Results directory already exists. Use -Force to overwrite." -ForegroundColor Yellow
    Write-Host "Skipping download..." -ForegroundColor Yellow
    exit 0
}

# Create temporary directory for downloads
$TempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("FlowMDM_Models_" + [System.Guid]::NewGuid().ToString("N").Substring(0,8))
New-Item -ItemType Directory -Path $TempDir | Out-Null

if ($Verbose) {
    Write-Host "Using temporary directory: $TempDir" -ForegroundColor Yellow
}

try {
    # Change to temporary directory for downloads
    Set-Location $TempDir
    
    # Method 1: Try using gdown (if available in pixi environment)
    Write-Host "Downloading to temporary location..." -ForegroundColor Cyan
    
    # Check if gdown is available
    $gdownAvailable = $false
    try {
        $null = Get-Command gdown -ErrorAction Stop
        $gdownAvailable = $true
        if ($Verbose) { Write-Host "gdown found in PATH" -ForegroundColor Green }
    }
    catch {
        if ($Verbose) { Write-Host "gdown not found in PATH" -ForegroundColor Yellow }
    }
    
    if ($gdownAvailable) {
        # Use gdown with fuzzy matching (same as bash script)
        & gdown --fuzzy $GoogleDriveUrl
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Download completed successfully with gdown" -ForegroundColor Green
        } else {
            throw "gdown failed with exit code $LASTEXITCODE"
        }
    } else {
        # Method 2: Fallback to direct Google Drive download using PowerShell
        Write-Host "gdown not available, using PowerShell native download..." -ForegroundColor Cyan
        
        # Extract file ID from Google Drive URL
        $FileId = "1fyx4rd6P_m26Vzb2xEvlkkSw9aDhT8rV"
        $DirectDownloadUrl = "https://drive.google.com/uc?export=download&id=$FileId"
        
        # Create web client with proper headers
        $webClient = New-Object System.Net.WebClient
        $webClient.Headers.Add("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        Write-Host "Downloading from: $DirectDownloadUrl" -ForegroundColor Cyan
        $webClient.DownloadFile($DirectDownloadUrl, $ZipFileName)
        $webClient.Dispose()
        
        Write-Host "Download completed with PowerShell WebClient" -ForegroundColor Green
    }
    
    # Verify the zip file exists and has reasonable size
    if (!(Test-Path $ZipFileName)) {
        throw "Download failed - $ZipFileName not found"
    }
    
    $zipFileSize = (Get-Item $ZipFileName).Length
    if ($zipFileSize -lt 1MB) {
        throw "Download failed - $ZipFileName is too small ($zipFileSize bytes)"
    }
    
    if ($Verbose) {
        Write-Host "Downloaded file size: $($zipFileSize / 1MB) MB" -ForegroundColor Green
    }
    
    # Extract the zip file in temporary directory
    Write-Host "Extracting $ZipFileName in temporary location..." -ForegroundColor Cyan
    Expand-Archive -Path $ZipFileName -DestinationPath "." -Force
    
    # Verify extraction was successful
    if (!(Test-Path "results")) {
        throw "Extraction failed - results directory not found after extraction"
    }
    
    # Remove existing results directory if it exists in target location
    if (Test-Path $ResultsDir) {
        Write-Host "Removing existing results directory..." -ForegroundColor Yellow
        Remove-Item $ResultsDir -Recurse -Force
    }
    
    # Move extracted results to final location
    Write-Host "Moving extracted files to final location..." -ForegroundColor Cyan
    Move-Item "results" $ResultsDir -Force
    
    # Check the contents of results directory
    $resultsContent = Get-ChildItem $ResultsDir -Recurse
    Write-Host "Extraction completed. Found $($resultsContent.Count) items in results directory." -ForegroundColor Green
    
    if ($Verbose) {
        Write-Host "Contents of results directory:" -ForegroundColor Yellow
        Get-ChildItem $ResultsDir -Recurse | Select-Object Mode, LastWriteTime, Length, Name | Format-Table
    }
    
    Write-Host "Pretrained models download completed successfully!" -ForegroundColor Green
    Write-Host "Models are now available in the 'results' directory." -ForegroundColor Green

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