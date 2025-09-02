# 3D Model Keypoint Topology Guide for FlowMDM

## HEADER
- **Purpose**: Comprehensive documentation of keypoint topologies and joint names for all 3D models used in FlowMDM
- **Status**: Active
- **Date**: 2025-01-02
- **Dependencies**: SMPL, SMPLH, HumanML3D, Babel datasets
- **Target**: Developers working with different motion representations and body models

## Overview

FlowMDM uses multiple 3D body model representations depending on the dataset and purpose. This guide documents the joint names, ordering, and kinematic topology for each model type.

## T2M (Text-to-Motion) Model - 22 Joints

### Joint Names and Indices

```python
# T2M joint ordering (used by HumanML3D dataset)
t2m_joints = {
    0: "Pelvis (Root)",
    1: "Left Hip",
    2: "Right Hip", 
    3: "Spine1",
    4: "Left Knee",
    5: "Right Knee",
    6: "Spine2",
    7: "Left Ankle",
    8: "Right Ankle",
    9: "Spine3",
    10: "Left Foot",
    11: "Right Foot",
    12: "Neck",
    13: "Left Collar",
    14: "Right Collar",
    15: "Head",
    16: "Left Shoulder",
    17: "Right Shoulder",
    18: "Left Elbow",
    19: "Right Elbow",
    20: "Left Wrist",
    21: "Right Wrist"
}
```

### Kinematic Chain Structure

```python
# From data_loaders/humanml/utils/paramUtil.py
t2m_kinematic_chain = [
    [0, 2, 5, 8, 11],      # Right leg: Pelvis → R.Hip → R.Knee → R.Ankle → R.Foot
    [0, 1, 4, 7, 10],      # Left leg: Pelvis → L.Hip → L.Knee → L.Ankle → L.Foot
    [0, 3, 6, 9, 12, 15],  # Spine: Pelvis → Spine1 → Spine2 → Spine3 → Neck → Head
    [9, 14, 17, 19, 21],   # Right arm: Spine3 → R.Collar → R.Shoulder → R.Elbow → R.Wrist
    [9, 13, 16, 18, 20]    # Left arm: Spine3 → L.Collar → L.Shoulder → L.Elbow → L.Wrist
]
```

### Visual Topology

```
         Head(15)
           |
         Neck(12)
           |
    L.Collar(13)--Spine3(9)--R.Collar(14)
         |            |            |
   L.Shoulder(16)  Spine2(6)  R.Shoulder(17)
         |            |            |
    L.Elbow(18)    Spine1(3)   R.Elbow(19)
         |            |            |
    L.Wrist(20)   Pelvis(0)   R.Wrist(21)
                   /    \
            L.Hip(1)    R.Hip(2)
                |          |
           L.Knee(4)    R.Knee(5)
                |          |
          L.Ankle(7)   R.Ankle(8)
                |          |
           L.Foot(10)   R.Foot(11)
```

## SMPL Model - 24 Joints

### Joint Names and Indices

```python
# Standard SMPL joint ordering
smpl_joints = {
    0: "Pelvis",
    1: "Left Hip",
    2: "Right Hip",
    3: "Spine1",
    4: "Left Knee",
    5: "Right Knee",
    6: "Spine2",
    7: "Left Ankle",
    8: "Right Ankle",
    9: "Spine3",
    10: "Left Foot",
    11: "Right Foot",
    12: "Neck",
    13: "Left Collar",
    14: "Right Collar",
    15: "Head",
    16: "Left Shoulder",
    17: "Right Shoulder",
    18: "Left Elbow",
    19: "Right Elbow",
    20: "Left Wrist",
    21: "Right Wrist",
    22: "Left Hand",    # Additional in SMPL
    23: "Right Hand"    # Additional in SMPL
}
```

### Key Differences from T2M
- SMPL has 24 joints vs T2M's 22 joints
- Additional joints: Left Hand (22) and Right Hand (23)
- Used for mesh generation and visualization
- 6890 vertices in the mesh representation

## SMPLH Model - 52 Joints

### Core Body Joints (0-21)
Same as T2M/SMPL first 22 joints

### Hand Joint Extension (22-51)

```python
# SMPLH extends SMPL with detailed hand articulation
smplh_hand_joints = {
    # Left hand fingers (22-36)
    22: "left_index1", 23: "left_index2", 24: "left_index3",
    25: "left_middle1", 26: "left_middle2", 27: "left_middle3",
    28: "left_pinky1", 29: "left_pinky2", 30: "left_pinky3",
    31: "left_ring1", 32: "left_ring2", 33: "left_ring3",
    34: "left_thumb1", 35: "left_thumb2", 36: "left_thumb3",
    
    # Right hand fingers (37-51)
    37: "right_index1", 38: "right_index2", 39: "right_index3",
    40: "right_middle1", 41: "right_middle2", 42: "right_middle3",
    43: "right_pinky1", 44: "right_pinky2", 45: "right_pinky3",
    46: "right_ring1", 47: "right_ring2", 48: "right_ring3",
    49: "right_thumb1", 50: "right_thumb2", 51: "right_thumb3"
}
```

### SMPLH Features
- 52 joints total: 22 body + 30 hand joints
- Each finger has 3 joints (proximal, middle, distal)
- Supports detailed hand gestures and grasping
- Used primarily with Babel dataset

## AMASS Format - 22 Joints

### Joint Mapping

```python
# From data_loaders/amass/info/joints.py
amass_joints = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee",
    "right_knee", "spine2", "left_ankle", "right_ankle", "spine3",
    "left_foot", "right_foot", "neck", "left_collar", "right_collar",
    "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist"
]
```

### AMASS to SMPL Correspondence

```python
# AMASS uses first 22 joints of SMPL
AMASS_JOINT_MAP = {
    'MidHip': 0,
    'LHip': 1, 'LKnee': 4, 'LAnkle': 7, 'LFoot': 10,
    'RHip': 2, 'RKnee': 5, 'RAnkle': 8, 'RFoot': 11,
    'LShoulder': 16, 'LElbow': 18, 'LWrist': 20,  
    'RShoulder': 17, 'RElbow': 19, 'RWrist': 21,
    'spine1': 3, 'spine2': 6, 'spine3': 9, 'Neck': 12, 'Head': 15,
    'LCollar': 13, 'RCollar': 14
}
```

## Dataset-Model Relationships

| Dataset | Model Type | Joint Count | Features | Use Case |
|---------|------------|-------------|----------|----------|
| HumanML3D | T2M | 22 | Rotation-invariant coords | Text-to-motion generation |
| Babel | SMPLH | 52 (uses 22) | SMPL parameters | Action-focused generation |
| AMASS | AMASS/SMPLH | 22 | Motion capture data | Data preprocessing |
| Visualization | SMPL | 24 | Full mesh | 3D rendering |

## Joint Connectivity Patterns

### Common Kinematic Patterns
All models share similar connectivity patterns:

1. **Spine Chain**: Root → Spine1 → Spine2 → Spine3 → Neck → Head
2. **Left Leg**: Root → L.Hip → L.Knee → L.Ankle → L.Foot
3. **Right Leg**: Root → R.Hip → R.Knee → R.Ankle → R.Foot
4. **Left Arm**: Spine3/Neck → L.Collar → L.Shoulder → L.Elbow → L.Wrist (→ L.Hand)
5. **Right Arm**: Spine3/Neck → R.Collar → R.Shoulder → R.Elbow → R.Wrist (→ R.Hand)

### Parent-Child Relationships

```python
# Example parent mapping for T2M
t2m_parents = {
    0: -1,   # Root has no parent
    1: 0,    # L.Hip → Pelvis
    2: 0,    # R.Hip → Pelvis
    3: 0,    # Spine1 → Pelvis
    4: 1,    # L.Knee → L.Hip
    5: 2,    # R.Knee → R.Hip
    6: 3,    # Spine2 → Spine1
    7: 4,    # L.Ankle → L.Knee
    8: 5,    # R.Ankle → R.Knee
    9: 6,    # Spine3 → Spine2
    10: 7,   # L.Foot → L.Ankle
    11: 8,   # R.Foot → R.Ankle
    12: 9,   # Neck → Spine3
    13: 9,   # L.Collar → Spine3
    14: 9,   # R.Collar → Spine3
    15: 12,  # Head → Neck
    16: 13,  # L.Shoulder → L.Collar
    17: 14,  # R.Shoulder → R.Collar
    18: 16,  # L.Elbow → L.Shoulder
    19: 17,  # R.Elbow → R.Shoulder
    20: 18,  # L.Wrist → L.Elbow
    21: 19   # R.Wrist → R.Elbow
}
```

## Conversion Between Models

### T2M to SMPL
- Add joints 22, 23 (hands) with zero or interpolated values
- Maintain first 22 joints unchanged

### SMPL to T2M
- Simply truncate to first 22 joints
- No data loss for body motion

### SMPLH to T2M/SMPL
- Use first 22 joints for T2M
- Use first 24 joints for SMPL
- Discard finger articulation

## Coordinate Systems

### Common Properties
- **Origin**: Pelvis/Root joint
- **Orientation**: Y-up (standard for motion capture)
- **Units**: Meters
- **Frame Rate**: 20 FPS (HumanML3D) or 30 FPS (Babel)

### Joint Position Format
```python
# All models output 3D positions
joint_position = [x, y, z]  # in meters

# Motion data shape
motion.shape = [batch_size, n_joints, 3, sequence_length]
```

## Implementation Notes

### Memory Layout
- Joint data stored as contiguous arrays
- Efficient for batch processing
- Supports vectorized operations

### Normalization
- HumanML3D uses dataset-specific mean/std
- Babel uses SMPL parameter normalization
- Important for model training/inference

## References

- SMPL Paper: "SMPL: A Skinned Multi-Person Linear Model" (SIGGRAPH Asia 2015)
- HumanML3D: "Generating Diverse and Natural 3D Human Motions from Text" (CVPR 2022)
- SMPL-H/MANO: "Embodied Hands: Modeling and Capturing Hands and Bodies Together" (SIGGRAPH Asia 2017)
- Official SMPL Website: https://smpl.is.tue.mpg.de/
- SMPLX Repository: https://github.com/vchoutas/smplx
- HumanML3D Repository: https://github.com/EricGuo5513/HumanML3D