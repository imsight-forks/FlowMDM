# FlowMDM Windows PowerShell Scripts

Windows PowerShell equivalents of the bash download scripts in `runners/prepare/`.

## Available Scripts

### Individual Download Scripts

- **`download_pretrained_models.ps1`** - Downloads the pretrained FlowMDM models
- **`download_smpl_files.ps1`** - Downloads SMPL body model files  
- **`download_glove.ps1`** - Downloads GloVe word embeddings (for evaluators)
- **`download_t2m_evaluators.ps1`** - Downloads T2M evaluation tools

### Master Script

- **`download_all.ps1`** - Unified script to download any combination of the above

## Usage

### Quick Start - Download Pretrained Models Only

```powershell
# Navigate to FlowMDM directory
cd model_zoo\FlowMDM

# Run in pixi environment (recommended)
pixi run -e rt-flowmdm powershell -ExecutionPolicy Bypass -File .\win64-scripts\download_pretrained_models.ps1

# Or directly with PowerShell
powershell -ExecutionPolicy Bypass -File .\win64-scripts\download_pretrained_models.ps1
```

### Download Everything

```powershell
# Download all dependencies
pixi run -e rt-flowmdm powershell -ExecutionPolicy Bypass -File .\win64-scripts\download_all.ps1 -All

# Or just specific components
pixi run -e rt-flowmdm powershell -ExecutionPolicy Bypass -File .\win64-scripts\download_all.ps1 -PretrainedModels -Glove
```

### Individual Script Usage

```powershell
# With options
.\win64-scripts\download_pretrained_models.ps1 -Force -Verbose

# Force overwrite existing files
.\win64-scripts\download_pretrained_models.ps1 -Force

# Verbose output for debugging  
.\win64-scripts\download_pretrained_models.ps1 -Verbose
```

## Options

All scripts support these common options:

- `-Force` - Overwrite existing files/directories
- `-Verbose` - Show detailed progress and debug information

## Requirements

### Method 1: gdown (Recommended)
If `gdown` is available in your environment (installed via pixi or pip), the scripts will use it for reliable Google Drive downloads.

```bash
# Install gdown if not available
pip install gdown
```

### Method 2: Native PowerShell (Fallback)
If `gdown` is not available, scripts fall back to PowerShell's `WebClient` for direct downloads. This may not work for all Google Drive files due to download restrictions.

## Temporary Directory Usage

All download scripts have been optimized to use the system temporary directory for downloads and extractions:

- **Downloads**: All files are downloaded to a unique temporary directory (e.g., `%TEMP%\FlowMDM_SMPL_a1b2c3d4`)
- **Extraction**: Zip files are extracted in the temporary directory
- **Final Placement**: Extracted content is moved to the final project location
- **Cleanup**: Temporary directories are automatically cleaned up, even if errors occur

This approach ensures:
- ✅ No temporary files left in the project directory
- ✅ Clean project structure
- ✅ Proper cleanup on both success and failure
- ✅ No interference with version control

## Skip Existing Downloads

All scripts automatically check if their target directories already exist:

- **`download_pretrained_models.ps1`** - Skips if `results/` directory exists
- **`download_smpl_files.ps1`** - Skips if `body_models/smpl/` or `body_models/smplh/` directories exist
- **`download_glove.ps1`** - Skips if `glove/` directory exists
- **`download_t2m_evaluators.ps1`** - Skips if `t2m/` directory exists

Use the `-Force` parameter to overwrite existing files and directories.

## File Sizes

- **Pretrained Models**: ~200-500MB
- **SMPL Files**: ~100-300MB  
- **GloVe Embeddings**: ~100MB
- **T2M Evaluators**: ~50-100MB

## Output Locations

- `results/` - Pretrained models
- `body_models/` - SMPL files
- `glove/` - GloVe embeddings
- `t2m/` - T2M evaluators

## Troubleshooting

### Common Issues

1. **Execution Policy Error**
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

2. **Google Drive Download Restrictions**
   - Use the pixi environment with gdown installed
   - Files may require manual download for very large files

3. **Network/Firewall Issues**
   - Ensure internet connectivity
   - Check corporate firewall settings
   - Try running with `-Verbose` for detailed error messages

4. **Disk Space Issues**
   - Ensure sufficient space in system temporary directory (`%TEMP%`)
   - Large downloads may require 1-2GB of temporary space
   - Temporary files are automatically cleaned up after completion

### Debug Mode

Run any script with `-Verbose` to see detailed download progress:

```powershell
.\win64-scripts\download_pretrained_models.ps1 -Verbose
```

## Equivalent Bash Scripts

These PowerShell scripts replicate the functionality of:

- `runners/prepare/download_pretrained_models.sh`
- `runners/prepare/download_smpl_files.sh` 
- `runners/prepare/download_glove.sh`
- `runners/prepare/download_t2m_evaluators.sh`