# HowTo: Retain Root Translation in SMPLX Visualization

## HEADER
- **Purpose**: Explain why joint-based FlowMDM outputs show locomotion while the SMPLX mesh viewer stays fixed, and how to export & apply root translation so SMPLX meshes move identically.
- **Status**: Draft
- **Date**: 2025-09-03
- **Author**: automation
- **Depends**: `generate-ex.py`, `show-animation-smplx.py`, `SlimSMPLTransform`, `smplx` library
- **Target**: Developers integrating parametric (SMPL/SMPLX) outputs with FlowMDM visualizations

---
## 1. The Symptom
- Joint animation (original FlowMDM viewer / `plot_script.py`) shows the character walking or moving across space.
- SMPLX mesh animation (`visualization/show-animation-smplx.py`) appears static at origin (only subtle limb rotations if any), or completely frozen.

Result: Perceived loss of global locomotion/translation when switching to SMPLX visualization.

---
## 2. Root Cause Summary
| Layer | Data Available | What Happens | Outcome |
|-------|----------------|--------------|---------|
| FlowMDM joints (standard pipeline) | Absolute joint positions already include global root translation each frame | Viewer plots positions directly | Locomotion visible |
| SMPL(H) param export (intended in `generate-ex.py`) | Should contain `trans` (T,3) in `SlimSMPLDatastruct.rots.trans` | In current partial implementation `extract_smpl_params` captures `transl` but later logic to save / build SMPLX pose ignores it | Translation lost |
| SMPLX pose array (`smplx_pose.npy`) | Only concatenated axis-angle rotations (165 dims = 3 + 63 + 45 + 45 + 3 + 3 + 3) | No translation slot defined | All frames share origin |
| SMPLX viewer | Calls `smplx_model(..., transl=None)` | Model defaults to zeros | Mesh anchored at (0,0,0) |

The translation exists (it can be obtained from `SlimSMPLDatastruct.rots.trans`) but is never persisted into `smplx_pose.npy` nor passed to `smplx_model` during playback.

---
## 3. Where Translation Lives Internally
`SlimSMPLTransform` (and underlying AMASS-style datastructs) produce a structure where:
```python
rots_data = datastruct.rots          # container
rots_data.rots   # (T, J, 3, 3) rotation matrices
rots_data.trans  # (T, 3)       global translation of the root/pelvis
```
Your current helper `extract_smpl_params(datastruct)` already reads `trans` into the returned dict as `'transl'`.

However, subsequent code paths (saving `smpl_params.npy`, constructing `smplx_pose.npy`, and validation) either:
- Never invoked (incomplete code branches), or
- Do not propagate `transl` into a file consumed by the viewer.

---
## 4. Design Options to Preserve Translation
### Option A: Separate Translation File (Minimal Intrusion)
- Save `smplx_transl.npy` alongside `smplx_pose.npy` (array shape `(T,3)` or list per sample).
- Update viewer CLI to optionally load translation file and pass `transl=<tensor>`.

Pros: Backward compatible; keeps `smplx_pose.npy` unchanged.  
Cons: Two files must stay in sync.

### Option B: Extend Pose Layout
- Append 3 translation values per frame to pose vector → shape becomes `(T,168)`.
- Add metadata key `"has_translation": true` and new layout entry.
- Viewer checks layout, splits final 3 values into `transl` tensor (no change for legacy 165‑dim files).

Pros: Single file artifact; self-contained.  
Cons: Requires conditional parsing logic.

### Option C: Hybrid (Preferred for Clarity)
- Keep base `smplx_pose.npy` (165 dims) + `smplx_layout.json`.
- If translation present, write `smplx_transl.npy` and annotate layout json: `{ "translation_file": "smplx_transl.npy", "frames": T }`.

---
## 5. Minimal Code Adjustments (Conceptual Only)
Below are illustrative snippets—apply changes in new extension modules to avoid intrusive edits per repo guidelines.

### 5.1 During Generation (`generate-ex.py`)
```python
# After building each sample's smpl_params dict
if args.export_smpl:
    collected_smpl_params.append(smpl_params)  # includes 'transl'

# After loop
if args.export_smplx:
    smplx_pose_list = []
    transl_list = []
    for entry in collected_smpl_params:
        pose, layout, meta = build_smplx_pose(entry)
        smplx_pose_list.append(pose.astype(np.float32))
        transl_list.append(entry['transl'].astype(np.float32))  # (T,3)
    np.save(out_dir / 'smplx_pose.npy', smplx_pose_list, allow_pickle=True)
    np.save(out_dir / 'smplx_transl.npy', transl_list, allow_pickle=True)
    with open(out_dir / 'smplx_layout.json','w') as f:
        json.dump({
            'layout': layout,
            'dims': 165,
            'has_translation': True,
            'translation_file': 'smplx_transl.npy'
        }, f, indent=2)
```

### 5.2 In the Viewer (`show-animation-smplx.py`)
```python
# After loading smplx_pose_data
transl_path = result_dir / 'smplx_transl.npy'
transl_data = None
if transl_path.exists():
    transl_data = np.load(transl_path, allow_pickle=True)

# In render_frame / _build_smplx_mesh_polydata
if transl_data is not None:
    # per-frame translation for current sequence (select index 0 if list)
    frame_transl = transl_data[seq_index][frame_index]
    transl_tensor = torch.from_numpy(frame_transl).float().unsqueeze(0)
else:
    transl_tensor = None

output = self.smplx_model(
    global_orient=..., body_pose=..., left_hand_pose=..., right_hand_pose=...,
    transl=transl_tensor,   # <— add here
    betas=..., expression=..., jaw_pose=..., leye_pose=..., reye_pose=...)
```

### 5.3 Backward Compatibility
Wrap translation load so absence logs:
```python
print('[info] No smplx_transl.npy found; mesh will remain at origin.')
```

---
## 6. Diagnostics Checklist
Run these before assuming a model bug:
```python
import numpy as np
pose = np.load('smplx_pose.npy', allow_pickle=True)
print('Frames:', pose[0].shape[0])
print('Std first 30 dims:', pose[0][:, :30].std())
# Check translation existence in raw SMPL params if you saved them
a = np.load('smpl_params.npy', allow_pickle=True)
print('Has transl key?', 'transl' in a[0])
print('Root displacement magnitude:', np.linalg.norm(a[0]['transl'][-1] - a[0]['transl'][0]))
```
If `Root displacement magnitude` is large but mesh viewer is static → translation discard confirmed.

---
## 7. Edge Cases & Considerations
| Case | Handling |
|------|----------|
| Sequence length 1 | Log warning; translation irrelevant |
| Missing `transl` in some entries | Fallback to zeros for those frames |
| Very long sequences (>1500) | Consider stride-downsample in viewer or precompute vertex cache |
| Mixed batch export | Ensure consistent ordering between pose & transl arrays (save lists of equal length) |
| Non-walking prompts (stationary) | Low displacement is expected—different from missing translation |

---
## 8. Validation Strategy
1. Export with new translation file.
2. Load both viewers: old (joints) and new (SMPLX mesh). Visually verify path alignment of pelvis.
3. Numeric check: compute pelvis joint (index 0) from joint viewer output vs SMPLX root vertex (approx by model joints). Differences should be small (cm-level) after unit conversions.
4. Stress test with two prompts (walk forward + turn) ensuring translation + rotation both reflected.

---
## 9. Performance Notes
- Passing per-frame translation adds negligible overhead (one (1,3) tensor copy per frame).
- Precomputing full vertex sequence becomes more attractive once translation makes path longer (camera follow logic may be added later).

---
## 10. References (Context7)
- SMPLX model loading & directory structure: /vchoutas/smplx ("Model Loading Structure", "Model Loading and Switching").
- PyVista dynamic updates: /pyvista/pyvista (BackgroundPlotter update patterns).

---
## 11. Summary
The locomotion you observe in the skeleton viewer originates from absolute joint positions already containing global translation. The SMPLX visualization reconstructs vertices only from rotations (axis-angle) and omits the root translation parameter, fixing the mesh at the origin. Persist and feed `transl` to the SMPLX forward pass (Option A or B) to restore full-motion trajectories without altering legacy artifacts.

---
## 12. Future Extensions
- Add optional camera follow mode (centering root every N frames).
- Support extended pose vector (include translation + optional expression 30D) with automatic layout parsing.
- Provide utility to retro-fit translation into existing result directories if raw joints or smpl_params exist.
