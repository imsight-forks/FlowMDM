# HowTo: Export SMPL(H) Parameters & Convert to SMPLX Pose in FlowMDM

## HEADER
- **Purpose**: Implementation guide to modify FlowMDM generation so Babel runs can emit raw SMPL(H) pose/beta parameters and a SMPLX-compatible concatenated pose tensor.
- **Status**: Draft
- **Date**: 2025-09-03
- **Author**: automation
- **Depends**: `SlimSMPLTransform`, `generate.py`, optional `smplx` package
- **Target**: Developers extending FlowMDM outputs

---
## 1. Context & Motivation
Babel generations internally use SMPLH parameters which are converted to joints via `SlimSMPLTransform`. Downstream tasks (retargeting, editing, avatar rigs) often need those original parametric values or a SMPLX-aligned pose layout. This guide adds optional export without altering default behavior.

---
## 2. High-Level Design
Add three CLI flags to `runners/generate.py`:
- `--export-smpl` – Save original SMPL/SMPLH pose & shape parameters per sample.
- `--export-smplx` – Additionally build zero‑padded SMPLX axis‑angle pose array.
- `--smplx-model-path PATH` – (Optional) Run a validation forward pass with a SMPLX model.

Data is collected right after sampling and before final video/save steps. For HumanML3D (no SMPL params) flags are ignored with a warning.

---
## 3. Data Components & Shapes
| Name | Shape | Description |
|------|-------|-------------|
| global_orient | (T,3) | Root joint axis-angle |
| body_pose | (T,21*3) | 21 body joints (excluding root) |
| left_hand_pose | (T,15*3) | Hand joints (SMPLH) optional |
| right_hand_pose | (T,15*3) | Hand joints (SMPLH) optional |
| betas | (10,) | Shape (assumed constant) |

SMPLX expects additional (jaw, leye, reye, expression). We pad with zeros: jaw(3), leye(3), reye(3), optionally expression(30) if you want a placeholder (can omit for minimal variant). Document layout in metadata.

---
## 4. Code Touchpoints
1. `model_zoo/FlowMDM/runners/generate.py` – Extend argument parser & integrate export block.
2. `model_zoo/FlowMDM/data_loaders/amass/transforms/smpl.py` – Update `SlimSMPLTransform` (or wrapper) to optionally return raw SMPL parameters alongside joints. Add parameter `return_params: bool = False`.

Minimal invasive strategy: modify only the call site to request parameters; inside transform assemble a dict containing required tensors before final joint computation.

---
## 5. Parser Additions (generate.py)
```python
# ... inside build parser section
parser.add_argument('--export-smpl', action='store_true', help='Export raw SMPL/SMPLH parameters (Babel only).')
parser.add_argument('--export-smplx', action='store_true', help='Also export SMPLX-compatible pose vector.')
parser.add_argument('--smplx-model-path', type=str, default=None, help='Path to SMPLX model directory for validation forward pass.')
```

---
## 6. Transform Modification (SlimSMPLTransform)
Add a flag to its `__call__` or forward method:
```python
def __call__(self, smpl_params, return_params: bool = False):
    # existing processing ...
    joints = ...  # current result
    if return_params:
        return joints, {
            'global_orient': global_orient,     # (T,3)
            'body_pose': body_pose,             # (T,63)
            'left_hand_pose': left_hand_pose,   # (T,45) or None
            'right_hand_pose': right_hand_pose, # (T,45) or None
            'betas': betas                      # (10,)
        }
    return joints
```
Ensure types are detached or cloned only when converting to numpy to avoid unnecessary copies.

---
## 7. Export Logic (generate.py) – Pseudocode
```python
smpl_export = []
# after sampling per sample motion
if args.dataset == 'babel' and (args.export_smpl or args.export_smplx):
    joints, smpl_dict = transform(smpl_params, return_params=True)
    smpl_export.append({k: (v.cpu().numpy() if v is not None else None) for k, v in smpl_dict.items()})

# After loop over samples
if args.export_smml and len(smpl_export):  # (typo intentionally to show to fix) -> 'export_smpl'
    np.save(out_dir / 'smpl_params.npy', smpl_export, allow_pickle=True)
```
(When implementing fix the typo and variable scoping; keep original `results.npy` untouched.)

---
## 8. Building a SMPLX-Compatible Pose Vector
```python
def build_smplx_pose(entry):
    go = entry['global_orient']        # (T,3)
    body = entry['body_pose']          # (T,63)
    lh = entry.get('left_hand_pose')
    rh = entry.get('right_hand_pose')
    parts = [go, body]
    has_hands = lh is not None and rh is not None
    if has_hands:
        parts.extend([lh, rh])
    T = go.shape[0]
    def zeros(c): return np.zeros((T, c), dtype=np.float32)
    # Pad jaw, leye, reye
    parts.extend([zeros(3), zeros(3), zeros(3)])
    # (Optional) expression placeholder (30 zeros) – comment out if not needed
    # parts.append(zeros(30))
    pose = np.concatenate(parts, axis=1)  # (T, D)
    layout = ['global_orient','body','left_hand','right_hand','jaw','leye','reye']
    return pose, layout, {'has_hands': has_hands}
```

Invocation:
```python
if args.export_smplx:
    smplx_entries = []
    for e in smpl_export:
        pose, layout, meta = build_smplx_pose(e)
        smplx_entries.append(pose)
    np.save(out_dir / 'smplx_pose.npy', smplx_entries, allow_pickle=True)
    with open(out_dir / 'smplx_layout.json','w') as f:
        json.dump({'layout': layout, 'entries': len(smplx_entries)}, f, indent=2)
```

---
## 9. Optional SMPLX Validation Forward
```python
if args.smplx_model_path and args.export_smplx:
    import smplx, torch
    # Use first sample for quick check
    sample_pose = smpl_export[0]
    T = sample_pose['global_orient'].shape[0]
    model = smplx.create(args.smplx_model_path, model_type='smplx', gender='neutral', batch_size=T)
    # Reshape components to (T, N, 3)
    out = model(
        global_orient=torch.from_numpy(sample_pose['global_orient']).float(),
        body_pose=torch.from_numpy(sample_pose['body_pose']).float().reshape(T,21,3),
        left_hand_pose=(torch.from_numpy(sample_pose['left_hand_pose']).float().reshape(T,15,3)
                        if sample_pose['left_hand_pose'] is not None else None),
        right_hand_pose=(torch.from_numpy(sample_pose['right_hand_pose']).float().reshape(T,15,3)
                        if sample_pose['right_hand_pose'] is not None else None),
        jaw_pose=torch.zeros(T,1,3), leye_pose=torch.zeros(T,1,3), reye_pose=torch.zeros(T,1,3),
        expression=torch.zeros(T,10)
    )
    np.save(out_dir / 'smplx_vertices_preview.npy', out.vertices.detach().cpu().numpy()[:5])
```

---
## 10. Output Artifacts
```
<results_dir>/
  smpl_params.npy           # list[dict] per sample (allow_pickle)
  smplx_pose.npy            # list[np.ndarray] per sample (axis-angle padded)
  smplx_layout.json         # component ordering & count
  smplx_vertices_preview.npy (optional) # first frames vertices
```

---
## 11. Edge Cases
- HumanML3D: Log "SMPL export skipped: dataset has no SMPL params".
- Missing `smplx`: allow export of padded pose; skip validation.
- Long sequences: consider streaming write later; current in-memory fine for typical prompt lengths.

---
## 12. Changelog Entry Template
Append to (create if missing) `model_zoo/FlowMDM/explain/CHANGELOG-LOCAL.md`:
```
2025-09-03: Added optional SMPL / SMPLX export during Babel generation (flags: --export-smpl, --export-smplx, --smplx-model-path). Default behavior unchanged.
```

---
## 13. Testing Checklist
- Run a Babel generation with flags on; confirm artifact files exist.
- Verify `smpl_params.npy[0]['body_pose'].shape[1] == 63`.
- If hands present, length along axis-angle after concat with jaw/eyes: 3 + 63 + 45 + 45 + 3 + 3 + 3 = 165 (without expression placeholder).
- Load `smplx_pose.npy` and ensure time dimension matches T.

---
## 14. Leveraging Official SMPL-X Transfer Tools
The upstream `smplx` repository ships an official model transfer utility (see `transfer_model/` and its README) that can translate between SMPL, SMPL+H, SMPL-X (and provide correspondence docs). Rather than maintaining a custom ad‑hoc mapping, you may:

1. Vendor (or reference) `context/refcode/smplx/transfer_model` code (already added as a submodule at `context/refcode/smplx`).
2. Use its documented scripts to:
    - Convert legacy SMPL / SMPL+H parameters into SMPL-X format including facial & hand spaces.
    - Retrieve vertex / joint correspondences for higher‑fidelity alignment instead of zero padding (improves hand articulation and future face integration).
3. Replace the zero‑filled (jaw, eyes) placeholders by invoking the transfer code with neutral values; this preserves future extensibility (e.g., adding expressions) while keeping deterministic output.

### Recommended Integration Path
| Step | Action | Outcome |
|------|--------|---------|
| A | Export raw SMPLH params as in Sections 6‑8 | Local param cache |
| B | Call transfer utility (offline script) providing model folders | Produces SMPL-X param set |
| C | Store converted params in `smplx_converted_params.npy` | Reusable richer representation |
| D | (Optional) Run SMPL-X forward to validate joint alignment | Consistency assurance |

### When To Prefer Official Transfer
Use official transfer when:
- You require consistent vertex-level alignment across body families.
- Downstream uses rely on facial or more precise hand DOFs.
- You plan future expression / eye gaze modeling (zero padding would otherwise hide issues).

Keep lightweight zero‑padding path as a fast fallback for quick experiments or where only core body + (optional) hands are needed.

## 15. Future Improvements
- Provide inverse conversion utility (SMPLX → joints) for checks.
- Add unit test comparing reconstructed joints from SMPLH vs original pipeline.
- Offer choice of saving rotation matrices instead of axis-angle.
- Integrate optional call-out to official transfer script (auto-discover if submodule present).

---
## 16. Summary
This guide introduces a low-impact extension enabling FlowMDM to emit reusable parametric body model data (SMPLH + padded SMPLX) for Babel, enhancing interoperability with avatar pipelines and retargeting workflows.
