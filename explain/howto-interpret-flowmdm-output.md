## HEADER
- Title: How to Interpret FlowMDM Output (Skeleton, SMPL Mapping, Pose Components)
- Status: Draft
- Date: 2025-09-03
- Authors: automated-agent
- Purpose: Answer Task 1 questions about FlowMDM output coordinate system, skeleton identity, mapping to SMPL variants, and source code origins of translation & orientation.
- Depends: `runners/generate.py`, `data_loaders/humanml/scripts/motion_process.py`, `data_loaders/humanml/utils/paramUtil.py`, `data_loaders/amass/transforms/smpl.py`, existing doc `about-flowmdm-output-format.md`, `about-smpl-usage-in-flowmdm.md`
- Related: `about-flowmdm-output-format.md`, `about-smpl-usage-in-flowmdm.md`, `howto-export-smpl-params-and-convert-to-smplx.md`

---

### 1. Which SMPL variant (SMPL / SMPLH / SMPLX) matches FlowMDM output?
FlowMDM outputs a 22‑joint skeleton defined by the constant `t2m_kinematic_chain` (and associated joint order) in `data_loaders/humanml/utils/paramUtil.py`. HumanML3D samples are produced directly in this 22‑joint Rotation‑Invariant (RIC) space; Babel samples are first decoded via a slim SMPLH transform (`SlimSMPLTransform`) and then reduced to the same ordering.

Summary of requested points:
1. FlowMDM skeleton definition: A fixed 22‑joint ordering (pelvis root; spine chain to head; left/right hips→knees→ankles→feet; shoulders→elbows→wrists plus simple hand end points) implemented in code as `t2m_kinematic_chain` and used consistently for both datasets after conversion.
2. What is the Text2Motion (T2M) skeleton & first use: A community 22‑joint reduction of SMPL/AMASS body joints popularized by the HumanML3D preprocessing pipeline (GitHub reference – HumanML3D: https://github.com/EricGuo5513/HumanML3D). It was adopted to lower dimensionality while retaining motion semantics for text‑to‑motion tasks.
3. Relation to SMPL / SMPLH / SMPLX: It is a subset / re‑ordering of core SMPL body joints (pelvis, hips, knees, ankles, feet, spine levels, neck, head, shoulders, elbows, wrists + simple hand proxies). Fingers, toes, facial, and detailed hand joints present in SMPLH/SMPLX are omitted; thus FlowMDM outputs are joint positions only (no parametric pose rotations) and cannot directly reconstruct full SMPLX without additional estimation.

### 2. Why 22 joints when SMPL* models use many more? Mapping details

SMPL variants have these canonical joint counts:
- SMPL: 24 body joints (including pelvis root + limbs + spine + neck + head + feet + hands as wrist points) plus shape vertices.
- SMPLH: SMPL + 2×15 hand joints.
- SMPLX: SMPLH + facial joints (jaw, eyes) + fingers (already in H) + additional features (expression blend components) totaling 55+ joints.

FlowMDM adopts the reduced **T2M 22‑joint set** (common in text‑to‑motion works for efficiency). This is effectively a subset / remapping of SMPL body joints where long hand/finger chains, toe tips, and some spine intermediates are collapsed. The mapping (indices refer to SMPL base numbering; SMPL uses 0=Pelvis root):

| T2M Index | T2M Name         | Approx SMPL Joint Source | Notes |
|-----------|------------------|---------------------------|-------|
| 0         | Pelvis (Root)    | 0 Pelvis                  | Root translation & global orient applied here |
| 1         | Left Hip         | 1 L_Hip                   | |
| 2         | Right Hip        | 2 R_Hip                   | |
| 3         | Spine1           | 3 Spine1                  | (Sometimes called Spine) |
| 4         | Left Knee        | 4 L_Knee                  | |
| 5         | Right Knee       | 5 R_Knee                  | |
| 6         | Spine2           | 6 Spine2                  | |
| 7         | Left Ankle       | 7 L_Ankle                 | |
| 8         | Right Ankle      | 8 R_Ankle                 | |
| 9         | Spine3           | 9 Spine3 / Chest          | Chest or thorax |
| 10        | Left Foot        | 10 L_Foot                 | Heel/foot joint (no toes) |
| 11        | Right Foot       | 11 R_Foot                 | |
| 12        | Neck             | 12 Neck                   | |
| 13        | Left Shoulder    | 13 L_Collar / Shoulder    | |
| 14        | Right Shoulder   | 14 R_Collar / Shoulder    | |
| 15        | Head             | 15 Head                   | |
| 16        | Left Elbow       | 16 L_Elbow                | |
| 17        | Right Elbow      | 17 R_Elbow                | |
| 18        | Left Wrist       | 18 L_Wrist                | Fingers omitted |
| 19        | Right Wrist      | 19 R_Wrist                | Fingers omitted |
| 20        | Left Hand        | 20 L_Hand (proxy)         | Usually a duplicate / end effector placed from wrist offset |
| 21        | Right Hand       | 21 R_Hand (proxy)         | Same as above |

SMPL’s 22+ indexing diverges after wrists; the T2M skeleton purposely stops—representing hands as simple points (wrist and a distal proxy) instead of full articulated fingers.

Important: Exact joint label names in original AMASS/SMPL may differ slightly (e.g., `L_Collar` vs `LeftShoulder`). The table expresses conceptual alignment. The pipeline’s code path (`rots2joints/smplh.py`) handles internal mapping; T2M selection then slices to these 22.

#### Converting / Retargeting the 22‑Joint T2M Skeleton to SMPLX
You have two main pathways depending on what data you generated:

1. Babel generation with parameter export (preferred, no fitting needed).  
2. Pure 22‑joint positions only (HumanML3D output) – requires optimization (fitting) to infer SMPLX pose.

Recommended first: use existing FlowMDM extended generation script and flags so you obtain SMPL(H)/SMPLX‑aligned parameters directly.

##### 2.1 Direct Export Path (Babel)
Use `runners/generate-ex.py` with `--export-smplx` to automatically build a zero‑padded SMPLX axis‑angle pose tensor and optional validation:
```
pixi run -e latest python runners/generate-ex.py \
  --model_path results/babel/FlowMDM/model001300000.pt \
  --dataset babel --num_samples 2 --seed 42 \
  --export-smpl --export-smplx --smplx-model-path ../../data
```
Produces inside the run output directory (see also `explain/howto-export-smpl-params-and-convert-to-smplx.md`):
- `smpl_params.npy` (list of dicts: global_orient, body_pose, left/right_hand_pose, betas, transl)
- `smplx_pose.npy` (list of (T, D) axis‑angle pose arrays: global + body (+ hands) + padded jaw/eyes)
- `smplx_layout.json` (component ordering metadata)

Visualize with full mesh (vertices) afterwards:
```
pixi run -e latest python visualization/show-animation-smplx.py <RESULT_DIR> --smplx-model-path ../../data --autoplay
```
Relevant code links:
- Parameter extraction: `runners/generate-ex.py` (`feats_to_xyz_with_smpl`, `extract_smpl_params`, SMPLX export block)
- SMPLH→joints mapping: `data_loaders/amass/transforms/smpl.py` (`SlimSMPLDatastruct.joints`)
- SMPLX viewer: `visualization/show-animation-smplx.py`

##### 2.2 Fitting Path (Only 22 Joints Available)
If you only have `results.npy` joint positions (no pose parameters) you must fit an SMPLX model so that selected SMPLX joints match the 22 T2M coordinates frame by frame.

Core idea: optimize axis‑angle pose (and optionally shape & translation) to minimize L2 between a subset of SMPLX joints and your T2M joints. Fingers / facial remain unconstrained (can stay zero). This is a standard inverse kinematics / pose fitting loop.

Approximate SMPLX joint index correspondence (base SMPL part) used for fitting (SMPLX shares leading body joint ordering with SMPL):
```
T2M -> SMPLX (body) indices
0 Pelvis          -> 0
1 Left Hip        -> 1
2 Right Hip       -> 2
3 Spine1          -> 3
4 Left Knee       -> 4
5 Right Knee      -> 5
6 Spine2          -> 6
7 Left Ankle      -> 7
8 Right Ankle     -> 8
9 Spine3/Chest    -> 9
10 Left Foot      -> 10
11 Right Foot     -> 11
12 Neck           -> 12
13 Left Shoulder  -> 13
14 Right Shoulder -> 14
15 Head           -> 15
16 Left Elbow     -> 16
17 Right Elbow    -> 17
18 Left Wrist     -> 18
19 Right Wrist    -> 19
20 Left Hand EoS  -> 20 (treat as wrist proxy or ignore)
21 Right Hand EoS -> 21 (same)
```
You can exclude indices 20/21 or weight them lightly since they are end‑effectors without finger articulation.

Minimal fitting example (script sketch – put in `tmp/fit_t2m_to_smplx.py`):
```python
import torch, numpy as np, smplx

# Load T2M joints: shape [22, 3, T]
data = np.load('results.npy', allow_pickle=True).item()
J = data['motion'][0]  # [22,3,T]
J = np.transpose(J, (2,0,1))  # [T,22,3]
T = J.shape[0]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = smplx.create('data/smplx', model_type='smplx', gender='neutral', batch_size=T, use_pca=False).to(device)

# Optimize only body pose + global orient + transl (hands, face fixed zero)
global_orient = torch.zeros(T,3, requires_grad=True, device=device)
body_pose = torch.zeros(T, model.NUM_BODY_JOINTS*3, requires_grad=True, device=device)
transl = torch.zeros(T,3, requires_grad=True, device=device)
optimizer = torch.optim.Adam([global_orient, body_pose, transl], lr=0.05)

t2m_to_smplx = list(range(22))  # mapping above
target = torch.from_numpy(J).float().to(device)  # [T,22,3]

for it in range(300):
    out = model(global_orient=global_orient,
                body_pose=body_pose.view(T, -1, 3),
                transl=transl)
    model_joints = out.joints[:, t2m_to_smplx, :]  # [T,22,3]
    loss = ((model_joints - target)**2).mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if it % 50 == 0: print('iter', it, 'loss', loss.item())

# Save fitted SMPLX components
np.save('fitted_smplx_global_orient.npy', global_orient.detach().cpu().numpy())
np.save('fitted_smplx_body_pose.npy', body_pose.detach().cpu().numpy())
np.save('fitted_smplx_transl.npy', transl.detach().cpu().numpy())
```
Notes:
- Add bone length regularization or prior losses (e.g., pose prior) for better realism.
- If sequence long, optionally fit a smaller set of keyframes then interpolate.
- Hand & facial parameters remain zeros; for richer hands run a secondary refinement model.

##### 2.3 Choosing a Path
- If you control generation: prefer `generate-ex.py --export-smplx` (lossless wrt model outputs, fastest).
- If you only possess joints: perform fitting as above; quality depends on initialization and regularization.

Further reading / code: see `explain/howto-export-smpl-params-and-convert-to-smplx.md` for layout details and validation steps, plus `visualization/show-animation-smplx.py` for mesh playback once parameters are available.

### 3. Rigid Root Transform Reconstruction & Coordinate Systems

This section unifies how FlowMDM reconstructs the root rigid transform (rotation + translation) for both datasets, how to package these into 4x4 homogeneous matrices, and how to convert them into SMPLX-compatible parameters (including optional axis adjustments).

#### 3.1 HumanML3D Reconstruction Path
1. Diffusion output shape `[B, 263, 1, T]` (normalized).
2. Denormalize in `feats_to_xyz()` (`runners/generate.py`).
3. `recover_from_ric()` (`data_loaders/humanml/scripts/motion_process.py`):
  - Extract root quaternion + planar trajectory via `recover_root_rot_pos`.
  - Reassemble local joint offsets from RIC slice `data[..., 4:(joints_num - 1)*3 + 4]`.
  - Apply inverse root quaternion to local offsets to obtain world joints:
    ```python
    positions = qrot(qinv(r_rot_quat)[..., None, :], positions)
    ```
  - Add root translation (X,Z only explicitly; Y already encoded in joint heights):
    ```python
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    ```
  - Prepend root joint → final joints `[B, 22, 3, T]` after reordering.

Root rigid components after recovery:
- Rotation: quaternion `r_rot_quat` (shape `[B, T, 4]`, real-first order in code paths using quaternion utilities).
- Translation: planar trajectory `(x,z)` + implicit `y` from root joint (assemble as `[x, y, z]`).

Building per-frame 4x4 transform using recovered variables (consistent with code path variable semantics):
```python
import numpy as np
from scipy.spatial.transform import Rotation as Rsc

# Inputs per frame t (after running recover_from_ric):
# r_rot_quat[t] : quaternion (w,x,y,z)
# r_pos[t]      : planar translation (x, z)
# joints[t,0]   : root joint 3D position already containing y (height) -> joints shape [T,22,3]

def humanml_root_T(r_rot_quat_frame, r_pos_frame, root_joint_frame):
  # r_rot_quat_frame = (w,x,y,z); SciPy expects (x,y,z,w)
  qw, qx, qy, qz = r_rot_quat_frame
  R_mat = Rsc.from_quat([qx, qy, qz, qw]).as_matrix()
  # Assemble translation: x,z from r_pos; y from recovered root joint
  tx = r_pos_frame[0]
  tz = r_pos_frame[1]
  ty = root_joint_frame[1]  # Y-up height
  T = np.eye(4)
  T[:3, :3] = R_mat
  T[:3, 3] = [tx, ty, tz]
  return T

# Example loop
# Ts = [humanml_root_T(r_rot_quat[t], r_pos[t], joints[t,0]) for t in range(T)]
```

#### 3.2 Babel (SMPLH) Reconstruction Path
1. Diffusion output `[B, 135, 1, T]` (22 joints × 6D rotation + velocity/translation features).
2. `SlimSMPLTransform` (ename `smplnh`) in `feats_to_xyz()` converts 6D → rotation matrices via `rotation_6d_to_matrix` (see `data_loaders/amass/tools_teach/geometry.py`).
3. `SlimSMPLDatastruct.joints` triggers SMPLH forward pass generating full world-space joints (global rotation + translation already applied via root channel in rfeats).
4. Slice/reorder to T2M 22 joints.

Root rigid components:
- Rotation: first joint rotation matrix (6D rep → matrix → quaternion / axis-angle as needed).
- Translation: SMPLH `transl` parameter per frame.

Build per-frame 4x4 transform from exported root 6D rotation + translation variables:
```python
import numpy as np

# Inputs per frame t (from SlimSMPLTransform path):
# root_rot6d[t]  : 6D rotation representation for root (length 6)
# root_transl[t] : translation vector (x,y,z)

def rot6d_to_matrix(d6):
  a1, a2 = d6[:3], d6[3:]
  b1 = a1 / np.linalg.norm(a1)
  b2 = a2 - (b1 * a2).sum() * b1
  b2 = b2 / np.linalg.norm(b2)
  b3 = np.cross(b1, b2)
  return np.stack([b1, b2, b3], axis=1)  # (3,3)

def babel_root_T(root_rot6d_frame, root_transl_frame):
  R_mat = rot6d_to_matrix(root_rot6d_frame)
  T = np.eye(4)
  T[:3,:3] = R_mat
  T[:3,3] = root_transl_frame
  return T

# Example loop
# Ts = [babel_root_T(root_rot6d[t], root_transl[t]) for t in range(T)]
```

#### 3.3 Coordinate System Summary (Extended)
| Aspect | HumanML3D | Babel (SMPLH) | SMPLX (Repo Model Files) |
|--------|-----------|---------------|--------------------------|
| Up Axis | Y | Y | Y (empirically verified; historical docs sometimes cite Z-up) |
| Units | meters | meters | meters |
| Root Translation Source | RIC planar + root joint Y | SMPLH `transl` | `transl` parameter |
| Root Rotation Source | Quaternion from RIC | 6D → matrix (root) | `global_orient` (axis-angle) |
| Joint Set (post) | 22 T2M joints | 22 T2M joints | Full 55+ (body+hands+face) |
| Representation | RIC (rot, traj, local joints) | rot6d + velocity | Axis-angle (global/body) + PCA/non-PCA hands + facial |

#### 3.4 Converting Outputs to SMPLX Parameters
Goal: Populate `(global_orient, transl, body_pose)` for SMPLX.

HumanML3D:
1. Obtain root quaternion `q` (w,x,y,z). Convert to axis-angle:
  ```python
  from scipy.spatial.transform import Rotation as Rsc
  axis_angle = Rsc.from_quat([x,y,z,w]).as_rotvec()  # (3,)
  ```
2. Root translation: use recovered `(x,z)` plus root joint Y height: `transl = [x, y_root, z]`.
3. Body pose: Not directly available (only joint positions). If you need full SMPLX body_pose you must fit (see fitting subsection) or regress rotations via IK.

Babel:
1. Root rotation: First joint 6D → matrix → axis-angle (SciPy `as_rotvec`).
2. Translation: Use SMPLH `transl` per frame directly.
3. Body pose: If extended export path used (`--export-smplx`), already assembled. Otherwise convert each joint's 6D to axis-angle and stack:
  - Order must follow SMPLX body joint order (first 21 body joints after root). Hands/facial remain zero or predicted if features available.

#### 3.5 Up / Forward Axis Confirmation (No Conversion Needed)
Both the Text2Motion 22‑joint skeleton and the bundled SMPLX model files are Y‑up. Empirical check (`tmp/export_smplx_obj.py`) shows the raw SMPLX mesh requires no elevation rotation. Inspection of `t2m_raw_offsets` in `data_loaders/humanml/utils/paramUtil.py` indicates:
- Root at origin `[0,0,0]`.
- Left / Right hips offset along ±X.
- Upward spine offsets along +Y.
- Foot end offsets include `[0,0,1]` entries, establishing +Z as the anatomical forward (toes point +Z).

Resulting convention throughout this repo:
```
Up: +Y
Forward: +Z
Right: +X
Handedness: Right-handed
```

Guidance:
1. Do not apply any X-axis ±90° rotations between FlowMDM joints and SMPLX meshes—they already align.
2. When integrating into engines that assume +Y forward (some game engines): rotate motions/meshes -90° about Y (yaw) so that +Z_forward_here → +Y_forward_engine.
3. Simple forward-axis detection (for validation): compute average vector from ankle to toe proxy joints over T frames; its dominant component should lie on +Z.
4. Handedness check: cross(Forward (+Z), Up (+Y)) ≈ -X; since Right = +X, (Right, Up, Forward) forms a right-handed coordinate frame (X × Y = Z). This matches the assumed math libraries.

No up-axis issue remains; Section 3.5 is purely declarative for future readers to avoid unnecessary conversions.

#### 3.6 Summary
- HumanML3D: root transform = quaternion + planar trajectory + implicit Y; need IK for full body rotations.
- Babel: root transform directly recoverable as rot6d + translation; per-joint rotations convertible to SMPLX.
- 4x4 construction: SciPy `Rotation` utilities; ensure quaternion ordering when converting.
- Axes: Both T2M joints and bundled SMPLX mesh share a Y-up, +Z-forward, +X-right frame (right-handed). No axis conversion required.


### 5. Key Source Code Locations
- Skeleton definition: `data_loaders/humanml/utils/paramUtil.py` (`t2m_raw_offsets`, `t2m_kinematic_chain`).
- HumanML recovery: `recover_from_ric()` in `data_loaders/humanml/scripts/motion_process.py`.
- Babel SMPL transform: `SlimSMPLTransform` & `SlimSMPLDatastruct` in `data_loaders/amass/transforms/smpl.py`.
- Rotation/feature representation conversions: `Globalvelandy` (`rots2rfeats`), `SMPLH` (`rots2joints`).
- Public conversion entry point: `feats_to_xyz()` / extended `feats_to_xyz_with_smpl()` in `runners/generate.py` / `runners/generate-ex.py`.
- Axis verification helper (empirical Y-up, +Z forward confirmation): `tmp/export_smplx_obj.py` (produces raw OBJ + optional diagnostic spans).
- SMPLX visualization (already assumes Y-up, +Z forward): `visualization/show-animation-smplx.py`.

### 6. Practical Implications
- The 22‑joint output loses finger articulation; for full-body retargeting to SMPLX you must reconstruct or regress missing joints (see `howto-export-smpl-params-and-convert-to-smplx.md`).
- Global root yaw alignment is already applied; avoid re-rotating sequences around Y unless intentionally reorienting.
- Translational components (especially Y/root height) should not be zeroed unless producing in‑place locomotion—root Y encodes vertical movements (steps, jumps).
- Mixing HumanML and Babel outputs is consistent because both end in the same T2M space (frame rate differs; resample as needed).
- No up-axis or forward-axis conversion is required within this repo (see Section 3.5). Only add yaw adjustments when adapting to an external coordinate convention (e.g., +Y forward engines).

### 7. Verifying Interpretations Programmatically
Minimal checks you can run (HumanML example):
```python
# Pseudocode snippet – create a script in tmp/ instead of inline CLI
from model_zoo.FlowMDM.runners.generate import feats_to_xyz
import torch

fake = torch.randn(1, 263, 1, 60)
joints = feats_to_xyz(fake, 'humanml')
assert joints.shape[1] == 22 and joints.shape[2] == 3
```

For Babel you would mock `[1,135,1,T]` input and ensure `SlimSMPLTransform` is reachable (body model files present).

Forward axis sanity check (optional):
```python
import numpy as np
from model_zoo.FlowMDM.data_loaders.humanml.utils import paramUtil

offsets = np.array(paramUtil.t2m_raw_offsets)  # shape (22,3)
# Heuristic: feet / hand distal points should have positive Z component establishing +Z forward
forward_score = (offsets[:,2] > 0).sum()
assert forward_score > 0, 'Expected some joints with +Z offset to confirm +Z forward'
print('Forward axis = +Z (heuristic joints with +Z offsets:', forward_score, ')')
```

Root transform reconstruction tests (Section 3) can then be combined with this axis confirmation to ensure downstream retargeting doesn’t apply redundant rotations.

### 8. FAQ
Q: Are the 22 joints exactly the same numeric indices as SMPL’s first 22?  
A: Conceptually aligned but not guaranteed 1:1 with raw SMPL export order—FlowMDM enforces a T2M ordering tailored for text‑to‑motion tasks.

Q: Where is root velocity stored?  
A: Encoded inside the feature representation (HumanML: derived from root positional deltas in RIC; Babel: inside `Globalvelandy` rfeats which include global velocity and possibly canonicalized orientation). Not explicitly saved in `results.npy`—only integrated positions are saved.

Q: How to get SMPL parameters from Babel generations?  
A: Extend `feats_to_xyz` or use `generate-ex.py` which already offers an augmented `feats_to_xyz_with_smpl` path returning SMPL parameter datastruct.

Q: Do I ever need to apply a -90° X rotation (Z-up → Y-up) with the bundled assets?  
A: No. Both joints and SMPLX meshes ship already Y-up/+Z forward. Apply such a conversion only if you import an external Z-up model.

Q: What is the canonical forward axis for FlowMDM outputs?  
A: +Z (see Section 3.5 and the offsets heuristic in Section 7). Adjust only when embedding into systems that expect a different forward axis.

### 9. References
- Original HumanML3D / Text-to-Motion repository (skeleton & preprocessing conventions).  
- SMPL / SMPLH / SMPLX model definitions (SMPL-X project, MPI).  
- FlowMDM repo modules referenced: `runners/generate.py`, `runners/generate-ex.py`, `data_loaders/humanml/scripts/motion_process.py`, `data_loaders/humanml/utils/paramUtil.py`, `data_loaders/amass/transforms/smpl.py`.  
- Axis verification script: `tmp/export_smplx_obj.py` (empirical confirmation of Y-up/+Z forward).  
- Extended export guidance: `explain/howto-export-smpl-params-and-convert-to-smplx.md`.

---
End of document.
