# BUG REPORT: Blank / Empty Matplotlib Motion Renderings

## Summary
FlowMDM's generated `sample_repXX.mp4` (and related matplotlib based visualizations) sometimes appear as blank / empty (only background, no skeleton). The PyVista-based script `tests/check-flowmdm-result-animation.py` renders the same `results.npy` correctly, proving the motion data is valid. The issue originates in the matplotlib rendering utilities (`data_loaders/humanml/utils/plot_script.py`).

## Impact
- Affected functions: `plot_3d_motion_mix`, `plot_3d_motion_multicolor`, `plot_3d_motion`, and their PDF variants.
- Output videos contain only the gray plane or even fully blank frames.
- Users may think generation failed, while data in `results.npy` is correct.

## Root Cause (Primary)
`matplotlib.use('Agg')` is invoked *after* `import matplotlib.pyplot as plt` in `plot_script.py`. Once `pyplot` is imported, the backend is already selected; calling `matplotlib.use()` afterwards has no effect or can cause inconsistent state (especially inside repeated function calls in multi-environment contexts like pixi). On some platforms/backends this results in an off-screen canvas that never draws 3D artists used by `Axes3D`, yielding blank frames on save.

### Evidence
Excerpt (current order):
```python
import matplotlib
import matplotlib.pyplot as plt  # backend chosen here already
from matplotlib.animation import FuncAnimation
...
matplotlib.use('Agg')  # too late
```
Matplotlib documentation: backend must be set *before* importing `pyplot`.

## Secondary Contributing Issues
| Issue | Effect |
|-------|--------|
| Repeated creation `ax = p3.Axes3D(fig)` instead of `fig.add_subplot(111, projection='3d')` | Works but less standard; some backends mishandle manual Axes3D construction under Agg override edge-cases |
| Agg vs interactive mismatch when run inside an environment expecting GUI | Can produce frames with zero drawn artists |
| Extreme camera elevation (`elev=120, azim=-90`) plus centering logic may hide skeleton if translation subtraction fails (e.g., if root joint is all zeros) | Can appear as "flat line" coincident with plane |
| In-place clearing by iterating `ax.lines[:]` and `ax.collections[:]` also removes the ground plane each frame; if plane removed before first draw, remaining list empty | Produces visually blank frame |

## Data Validity Confirmation
`results.npy` inspected:
- Shape: `(1, 22, 3, 120)`
- Value range approx `[-1.0, 1.53]`
- No NaNs.
Thus the problem is not with generated motion.

## Minimal Reproduction
1. Generate motion (or use existing results directory):
   ```powershell
   pixi run -e latest python -m runners.generate --model_path results/babel/FlowMDM/model001300000.pt --instructions_file runners/jsons/composition_babel.json --bpe_denoising_step 60 --guidance_param 2.0
   ```
2. Observe produced `sample_rep00.mp4` is blank (on affected systems).
3. Load `results.npy` with `tests/check-flowmdm-result-animation.py` and see correct skeleton.

## Proposed Fix
### Required Changes
1. Move backend selection to *top* of module (before importing pyplot):
```python
import matplotlib
matplotlib.use('Agg')          # must precede pyplot import
import matplotlib.pyplot as plt
```
2. Replace manual `Axes3D` construction with canonical API:
```python
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111, projection='3d')
```
3. Preserve ground plane: draw it once, then selectively remove only dynamic line objects each frame:
```python
# Tag dynamic lines
for line in list(ax.lines):
    if getattr(line, '_is_dynamic', False):
        line.remove()
...
ln, = ax.plot3D(...)
ln._is_dynamic = True
```
(or keep existing clearing but recreate plane each frame reliably.)
4. (Optional robustness) Guard against empty / zero root normalization:
```python
if np.allclose(data[:,0,0], 0) and np.allclose(data[:,0,2], 0):
    # skip subtracting trajectory to avoid collapsing motion
```

### Patch Sketch
```diff
-import matplotlib
-import matplotlib.pyplot as plt
+import matplotlib
+matplotlib.use('Agg')  # set backend before pyplot
+import matplotlib.pyplot as plt
@@
-fig = plt.figure(figsize=figsize)
-plt.tight_layout()
-ax = p3.Axes3D(fig)
+fig = plt.figure(figsize=figsize)
+ax = fig.add_subplot(111, projection='3d')
+fig.tight_layout()
```
(Apply similarly across plot functions.)

## Alternative / Quick Workaround
Avoid matplotlib pipeline entirely and use the working PyVista animator for inspection:
```powershell
pixi run -e latest python tests/check-flowmdm-result-animation.py
```

## Verification Plan
1. Apply patch.
2. Regenerate a short motion (or re-run visualization only) and confirm non-empty MP4.
3. Compare a few frames by sampling joint coordinates—ensure visual orientation unchanged.
4. Run on Windows (current environment) and a headless Linux container to ensure consistent output.

### Added Quick Backend Reproduction Scripts
Located under `tmp/`:
- `test_matplotlib_backend_order.py` (demonstrates late backend set — prints existing backend, attempts late `use('Agg')`).
- `test_matplotlib_backend_order_fixed.py` (sets backend early, saves PNG reliably).

Run (latest env example):
```
pixi run -e latest python tmp/test_matplotlib_backend_order.py
pixi run -e latest python tmp/test_matplotlib_backend_order_fixed.py
```
Compare reported backends and confirm both PNG files are produced (sizes may differ if first script fails to render fully on some systems).

## Risks
Low—change is confined to visualization utilities. Backend specification ordering aligns with official guidelines. Axes creation change is standard practice.

## References
- Matplotlib FAQ: "How do I select a backend?" (must set before pyplot import).
- Working comparison: `tests/check-flowmdm-result-animation.py` (PyVista, unaffected by matplotlib backend ordering).

## Recommendation
Implement the patch, re-render existing `results.npy` to rebuild `sample_rep00.mp4`, and update documentation to mention: "If visualization videos are blank, ensure `plot_script.py` sets backend before importing pyplot."

---
Prepared: 2025-09-03
