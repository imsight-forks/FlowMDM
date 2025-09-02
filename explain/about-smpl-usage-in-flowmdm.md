# SMPL Model Usage in FlowMDM

## HEADER
- **Purpose**: Document how SMPL body models are used in FlowMDM for different datasets and purposes
- **Status**: Active
- **Date**: 2025-01-02
- **Dependencies**: SMPL/SMPLH body models, smplx library
- **Target**: Developers understanding FlowMDM's body model integration

## Overview

FlowMDM uses SMPL (Skinned Multi-Person Linear) models in specific scenarios, primarily for:
1. **Babel dataset processing** - Converting SMPL parameters to joint positions
2. **Mesh visualization** - Optional conversion from joint positions to 3D meshes
3. **Data preprocessing** - AMASS dataset processing

## Body Models Directory Structure

```
model_zoo/FlowMDM/body_models/
├── smpl/
│   ├── SMPL_NEUTRAL.pkl     # Gender-neutral SMPL model (39MB)
│   ├── J_regressor_extra.npy # Additional joint regressors
│   ├── kintree_table.pkl     # Kinematic tree structure
│   └── smplfaces.npy         # Face indices for mesh
└── smplh/
    └── SMPLH_MALE.pkl        # Male SMPL+H model with hand joints (277MB)
```

## Dataset-Specific Usage

### HumanML3D Dataset
- **Does NOT require SMPL models during generation**
- Uses rotation-invariant coordinates (RIC) representation
- 263-dimensional features that directly encode joint positions
- Can optionally use SMPL for mesh visualization via SMPLify-3D

### Babel Dataset  
- **REQUIRES SMPL models for generation**
- Motion stored as 135-dimensional SMPL parameters
- Uses `SMPLH_MALE.pkl` for parameter-to-joint conversion
- Conversion happens via `SlimSMPLTransform` in `feats_to_xyz()`

## Key Usage Patterns

### 1. During Generation (Babel Only)

```python
# In runners/generate.py - feats_to_xyz()
if dataset == 'babel':
    from data_loaders.amass.transforms import SlimSMPLTransform
    transform = SlimSMPLTransform(
        batch_size=batch_size, 
        name='SlimSMPLTransform',
        ename='smplnh',  # SMPL with additional joints
        normalization=True
    )
    # Internally uses SMPLH model at ./body_models/smplh/
```

### 2. For Mesh Visualization (Optional)

```python
# In utils/visualize/simplify_loc2rot.py
smplmodel = smplx.create(
    "./body_models/",  # Points to body_models directory
    model_type="smpl", 
    gender="neutral",   # Uses SMPL_NEUTRAL.pkl
    ext="pkl",
    batch_size=num_frames
)
```

### 3. AMASS Data Processing

```python
# In data_loaders/amass/transforms/smpl.py
rots2joints = SMPLH(
    path='./body_models/smplh',
    jointstype='smplnh',
    input_pose_rep='matrix',
    batch_size=batch_size,
    gender='male',  # Uses SMPLH_MALE.pkl
    name='SMPLH'
)
```

## SMPLify-3D Integration

When converting joint positions to SMPL meshes for visualization:

1. **Input**: 22 joint positions from FlowMDM output
2. **Process**: SMPLify-3D optimization (150 iterations)
3. **Output**: SMPL pose parameters (72D) and shape parameters (10D)
4. **Result**: Full body mesh with 6890 vertices

### Configuration

```python
# From utils/visualize/joints2smpl/src/config.py
SMPL_MODEL_DIR = "./body_models/"
SMPL_MEAN_FILE = "./utils/visualize/joints2smpl/smpl_models/neutral_smpl_mean_params.h5"
```

## When SMPL Models Are Used

| Scenario | Dataset | Model Used | Required? |
|----------|---------|------------|-----------|
| Generation | HumanML3D | None | No |
| Generation | Babel | SMPLH_MALE.pkl | Yes |
| Mesh Visualization | Any | SMPL_NEUTRAL.pkl | Optional |
| AMASS Processing | N/A | SMPLH_MALE.pkl | Yes |
| Training | HumanML3D | None | No |
| Training | Babel | SMPLH_MALE.pkl | Yes |

## Important Notes

### Model Dependencies
- **smplx library**: Required for loading and using SMPL models
- **trimesh**: Required for mesh export and visualization
- **PyTorch**: Models are loaded as PyTorch modules

### Performance Considerations
- SMPLify-3D optimization is computationally expensive (~150 iterations per frame)
- Mesh generation is optional and primarily for high-quality visualization
- Joint positions (default output) are sufficient for most applications

### File Size Impact
- SMPL models add ~316MB to the repository
- Only required if using Babel dataset or mesh visualization
- Can be omitted for HumanML3D-only workflows

## Usage Examples

### Check if SMPL Models Exist

```python
import os
smpl_path = "./body_models/smpl/SMPL_NEUTRAL.pkl"
smplh_path = "./body_models/smplh/SMPLH_MALE.pkl"

has_smpl = os.path.exists(smpl_path)
has_smplh = os.path.exists(smplh_path)

if not has_smplh and dataset == "babel":
    raise FileNotFoundError("SMPLH model required for Babel dataset")
```

### Generate Mesh from Joints

```bash
# Convert stick figure to SMPL mesh
python -m runners.render_mesh \
    --input_path results/sample_rep0.mp4 \
    --cuda True \
    --device 0

# Output: 
# - sample_rep0_obj/: Directory with .obj files per frame
# - sample_rep0_smpl_params.npy: SMPL parameters
```

## Summary

SMPL models in FlowMDM serve three main purposes:
1. **Essential** for Babel dataset (SMPL parameter conversion)
2. **Optional** for mesh visualization (any dataset)
3. **Required** for AMASS data preprocessing

HumanML3D users can operate without SMPL models, as the dataset uses direct joint representations. Babel users must have the SMPLH model for proper motion generation.