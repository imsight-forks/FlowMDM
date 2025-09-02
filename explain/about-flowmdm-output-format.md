# FlowMDM Output Data Format Guide

## HEADER
- **Purpose**: Comprehensive guide to understanding and working with FlowMDM output data
- **Status**: Active
- **Date**: 2025-01-02
- **Dependencies**: FlowMDM model outputs, numpy, visualization tools
- **Target**: Developers working with FlowMDM-generated motion data

## Overview

FlowMDM generates human motion data as 3D joint coordinates over time. The output format is designed for easy visualization, analysis, and integration with animation pipelines.

## Primary Output Structure

### Main Output File: `results.npy`

The model saves outputs as a numpy dictionary with the following structure:

```python
{
    'motion': np.ndarray,      # Shape: [num_repetitions, 22, 3, seq_len]
    'text': list[str],          # Text descriptions for each motion
    'lengths': np.ndarray,      # Frame counts for each motion segment
    'num_samples': int,         # Number of unique text prompts
    'num_repetitions': int      # Number of generations per prompt
}
```

### Motion Data Dimensions

```python
motion.shape = [batch_size, n_joints, xyz, sequence_length]
```

- **batch_size**: Number of generated motion samples
- **n_joints**: 22 skeletal joints (fixed for T2M skeleton)
- **xyz**: 3D coordinates (X, Y, Z) in meters
- **sequence_length**: Total frames in the sequence

## Joint Configuration

### T2M Skeleton (22 Joints)

The model uses the Text-to-Motion (T2M) kinematic chain with the following joint ordering:

```python
# Joint indices and names
joints = {
    0: "Pelvis (Root)",
    1: "Left Hip", 2: "Right Hip", 3: "Spine1",
    4: "Left Knee", 5: "Right Knee", 6: "Spine2",
    7: "Left Ankle", 8: "Right Ankle", 9: "Spine3",
    10: "Left Foot", 11: "Right Foot", 12: "Neck",
    13: "Left Shoulder", 14: "Right Shoulder",
    15: "Head",
    16: "Left Elbow", 17: "Right Elbow",
    18: "Left Wrist", 19: "Right Wrist",
    20: "Left Hand", 21: "Right Hand"
}
```

### Kinematic Chain Definition

```python
# From model_zoo/FlowMDM/data_loaders/humanml/utils/paramUtil.py
t2m_kinematic_chain = [
    [0, 2, 5, 8, 11],      # Right leg chain
    [0, 1, 4, 7, 10],      # Left leg chain  
    [0, 3, 6, 9, 12, 15],  # Spine to head
    [9, 14, 17, 19, 21],   # Right arm chain
    [9, 13, 16, 18, 20]    # Left arm chain
]
```

## Loading and Processing Output Data

### Basic Loading

```python
import numpy as np

# Load results
data = np.load('results.npy', allow_pickle=True).item()
motions = data['motion']  # Shape: [batch, 22, 3, seq_len]
texts = data['text']       # List of text prompts
lengths = data['lengths']  # Frame counts per segment

# Access first generated motion
first_motion = motions[0]  # Shape: [22, 3, seq_len]
```

### Extracting Frame Data

```python
# Get joint positions at specific frame
frame_idx = 30
frame_joints = motions[0, :, :, frame_idx]  # Shape: [22, 3]

# Get trajectory of specific joint over time  
pelvis_trajectory = motions[0, 0, :, :]  # Shape: [3, seq_len]
left_hand_trajectory = motions[0, 20, :, :]
```

### Processing Multi-Segment Compositions

```python
# For compositions with multiple text segments
texts = data['text'][0].split(' /// ')  # Split concatenated descriptions
segment_lengths = data['lengths']

# Extract individual segments
start_frame = 0
segments = []
for i, length in enumerate(segment_lengths):
    segment = motions[0, :, :, start_frame:start_frame+length]
    segments.append({
        'motion': segment,
        'text': texts[i],
        'frames': length
    })
    start_frame += length
```

## Coordinate System Details

### Spatial Properties
- **Units**: Meters (human-scale measurements)
- **Coordinate Frame**: Y-up system (standard for motion capture)
- **Origin**: Pelvis/root joint at frame 0
- **Frame Rate**: 
  - HumanML3D: 20 FPS
  - Babel: 30 FPS

### Normalization
Motion data is denormalized during generation using dataset statistics:

```python
# Applied during feats_to_xyz conversion
# For HumanML3D dataset
mean = np.load('dataset/HML_Mean_Gen.npy')
std = np.load('dataset/HML_Std_Gen.npy')
denormalized = (normalized * std + mean)
```

## Visualization Examples

### Basic Skeleton Drawing

```python
# Define skeleton connections
skeleton_pairs = [
    # Spine
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    # Left leg
    (0, 1), (1, 4), (4, 7), (7, 10),
    # Right leg  
    (0, 2), (2, 5), (5, 8), (8, 11),
    # Left arm
    (9, 13), (13, 16), (16, 18), (18, 20),
    # Right arm
    (9, 14), (14, 17), (17, 19), (19, 21)
]

# Extract frame
frame = motions[0, :, :, 0]  # Shape: [22, 3]

# Plot skeleton (pseudocode)
for start_idx, end_idx in skeleton_pairs:
    start_joint = frame[start_idx]
    end_joint = frame[end_idx]
    draw_line(start_joint, end_joint)
```

### Animation Export

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_frame(frame_idx):
    joints = motions[0, :, :, frame_idx]
    # Update 3D scatter plot with joint positions
    # Draw skeleton connections
    return artists

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
anim = FuncAnimation(fig, animate_frame, frames=motions.shape[-1])
anim.save('motion.mp4', fps=20)
```

## Additional Output Files

FlowMDM generates several supplementary files:

| File | Content | Format |
|------|---------|--------|
| `results.txt` | Text prompts | One prompt per line |
| `results_len.txt` | Frame lengths | One number per line |
| `sample_rep*.mp4` | Individual visualizations | Video file |
| `sample_all.mp4` | Grid visualization | Combined video |

## Common Processing Tasks

### Convert to World Coordinates

```python
# Motions are already in world coordinates after feats_to_xyz
# To apply additional transformations:
def translate_motion(motion, offset):
    """Translate entire motion sequence"""
    translated = motion.copy()
    translated[:, :3, :] += offset.reshape(3, 1)
    return translated

# Center motion at origin
pelvis_start = motion[0, :3, 0]
centered = translate_motion(motion, -pelvis_start)
```

### Extract Motion Features

```python
# Calculate velocities
velocities = np.diff(motion, axis=-1)  # Frame-to-frame differences

# Get joint angles (requires inverse kinematics)
# Speed statistics
speeds = np.linalg.norm(velocities, axis=1)  # Joint speeds
mean_speed = np.mean(speeds, axis=-1)  # Average per joint
```

### Retargeting to Different Skeletons

```python
# Basic retargeting approach
def retarget_to_custom_skeleton(t2m_motion, mapping):
    """Map T2M joints to custom skeleton"""
    custom_motion = np.zeros((len(mapping), 3, t2m_motion.shape[-1]))
    for custom_idx, t2m_idx in mapping.items():
        if t2m_idx is not None:
            custom_motion[custom_idx] = t2m_motion[t2m_idx]
    return custom_motion
```

## Important Notes

### Memory Considerations
- Long sequences (>500 frames) may require significant memory
- Use `--use_chunked_att` flag during generation for memory optimization

### Quality vs. Smoothness
- Lower `bpe_denoising_step` (30-60): Smoother transitions
- Higher `bpe_denoising_step` (100-200): Better individual action quality

### Dataset Differences
- **HumanML3D**: 263-dim features, 20 FPS, detailed descriptions
- **Babel**: 135-dim SMPL params, 30 FPS, simple action labels

## References

- Source: [model_zoo/FlowMDM/runners/generate.py](../runners/generate.py)
- T2M Skeleton: [data_loaders/humanml/utils/paramUtil.py](../data_loaders/humanml/utils/paramUtil.py)
- Feature Conversion: `feats_to_xyz()` function in generate.py
- Original Repository: https://github.com/imsight-forks/FlowMDM