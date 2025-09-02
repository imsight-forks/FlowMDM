#!/usr/bin/env pwsh
# PowerShell script to generate simple walking animation using FlowMDM
# Task 1: Create a simple walking animation with forward and backward segments

param(
    [switch]$Verbose = $false
)

Write-Host "Generating simple walking animation with FlowMDM..." -ForegroundColor Green

$ErrorActionPreference = "Stop"

# Get script directory and project paths
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$FlowMDMRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $FlowMDMRoot))

if ($Verbose) {
    Write-Host "Script directory: $ScriptDir" -ForegroundColor Yellow
    Write-Host "FlowMDM root: $FlowMDMRoot" -ForegroundColor Yellow
    Write-Host "Project root: $ProjectRoot" -ForegroundColor Yellow
}

# Define paths - using Babel model instead of HumanML3D to avoid dataset normalization file issues
$ModelPath = Join-Path $FlowMDMRoot "results\babel\FlowMDM\model001300000.pt"
$InstructionsFile = Join-Path $ScriptDir "simple_walk_instructions.json"
$OutputDir = Join-Path $ProjectRoot "tmp\FlowMDM\simple-walk"

# Verify required files exist
if (!(Test-Path $ModelPath)) {
    Write-Error "Babel model not found at: $ModelPath"
    Write-Error "Please run the download_pretrained_models script first."
    exit 1
}

if (!(Test-Path $InstructionsFile)) {
    Write-Error "Instructions file not found at: $InstructionsFile"
    exit 1
}

# Create output directory if it doesn't exist
if (!(Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    Write-Host "Created output directory: $OutputDir" -ForegroundColor Green
}

Write-Host "Using model: $ModelPath" -ForegroundColor Cyan
Write-Host "Using instructions: $InstructionsFile" -ForegroundColor Cyan
Write-Host "Output directory: $OutputDir" -ForegroundColor Cyan

try {
    # Change to FlowMDM directory for execution
    Push-Location $FlowMDMRoot
    
    Write-Host "Starting FlowMDM generation..." -ForegroundColor Green
    Write-Host "This may take several minutes depending on your GPU..." -ForegroundColor Yellow
    
    # Run FlowMDM generation with local pixi environment (using Babel parameters)
    & pixi run python -m runners.generate `
        --model_path $ModelPath `
        --instructions_file $InstructionsFile `
        --num_repetitions 1 `
        --bpe_denoising_step 125 `
        --guidance_param 1.5 `
        --dataset babel
        
    if ($LASTEXITCODE -ne 0) {
        throw "FlowMDM generation failed with exit code $LASTEXITCODE"
    }
    
    Write-Host "Generation completed successfully!" -ForegroundColor Green
    
    # Move generated files to output directory
    Write-Host "Moving generated files to output directory..." -ForegroundColor Cyan
    
    # Look for generated files (typically in save/ subdirectory)
    $SaveDir = Join-Path $FlowMDMRoot "save"
    if (Test-Path $SaveDir) {
        $GeneratedFiles = Get-ChildItem $SaveDir -Recurse -File
        if ($GeneratedFiles.Count -gt 0) {
            foreach ($file in $GeneratedFiles) {
                $destPath = Join-Path $OutputDir $file.Name
                Copy-Item $file.FullName $destPath -Force
                Write-Host "Copied: $($file.Name)" -ForegroundColor Green
            }
            Write-Host "Moved $($GeneratedFiles.Count) files to output directory" -ForegroundColor Green
        } else {
            Write-Warning "No generated files found in save directory"
        }
    } else {
        Write-Warning "Save directory not found - files may be in different location"
    }
    
    # List what's in the output directory
    Write-Host "`nGenerated files in output directory:" -ForegroundColor Yellow
    $outputFiles = Get-ChildItem $OutputDir
    if ($outputFiles.Count -gt 0) {
        foreach ($file in $outputFiles) {
            Write-Host "  - $($file.Name) ($($file.Length) bytes)" -ForegroundColor White
        }
    } else {
        Write-Host "  (No files found - check FlowMDM output location)" -ForegroundColor Red
    }
    
    Write-Host "`nSimple walk animation generation completed!" -ForegroundColor Green
    Write-Host "Output location: $OutputDir" -ForegroundColor Green
    
} catch {
    Write-Error "Generation failed: $($_.Exception.Message)"
    exit 1
} finally {
    Pop-Location
}