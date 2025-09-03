## HEADER
- Title: About adding 2D path-conditioned motion generation to FlowMDM
- Status: draft
- Created: 2025-09-03
- Updated: 2025-09-03
- Maintainer: survey-implementation
- Scope: model_zoo/FlowMDM (non-intrusive extension)
- Tags: guidance, conditioning, trajectory, path-following
---

## Short Answer
FlowMDM (as provided) does NOT natively accept an explicit ground‑plane 2D trajectory prior (x,z path) for root motion. Out‑of‑the‑box conditioning is limited to text (classifier‑free guidance) plus internal blended positional encodings. You can still obtain path‑following motion via lightweight extensions. Easiest: generate normally, then warp the root trajectory onto your target path while preserving local pose/velocity. Higher fidelity: inject the path as additional conditioning tokens or guidance loss during diffusion steps.

## Why There is No Built‑In Path Input
Observed code paths:
- `runners/generate.py` only builds `model_kwargs['y']` with `text`, `lengths`, `mask`, `scale`.
- `DiffusionWrapper_FlowMDM` augments with positional bias matrices and masks; no trajectory field.
- `FlowMDM.forward` consumes only embeddings from text + sequence position; no hook for external per‑frame root translation.

Therefore a path prior must be integrated externally or by minimally extending `model_kwargs` handling.

## Option Matrix
| Approach | Intrusiveness | Temporal Plausibility | Guarantees Path | Notes |
|----------|---------------|-----------------------|-----------------|-------|
| A. Post‑hoc root path retarget (trajectory warping) | None (new script) | High (if smooth reparam) | Yes | Fast baseline; may distort foot contacts (need IK / contact fix) |
| B. Hard override root each denoise step | Low (monkey patch loop) | Medium (pose may not anticipate turns) | Yes | Replace root (pelvis) translation in latent after each `p_sample` |
| C. Add path as extra conditioning sequence (embedding) | Medium (model edit) | High | Probabilistic | Concatenate learned embedding of (Δx, Δz, heading) per frame |
| D. Guidance loss (classifier‑free style) | Medium (wrapper edit) | High | Approx (tolerance window) | Add gradient pushing root towards path spline |
| E. Train / finetune with path channel | Highest | High | Yes (after training) | Requires dataset with ground truth root trajectories |

## Recommended Implementation Path (Incremental)
1. Start with A (fast proof): warp root to path; evaluate artifacts.
2. If turning anticipation needed, prototype B (override during sampling) to let kinematics adapt partially while preserving diffused joint offsets.
3. For production, implement C (embedding) or D (guidance) without retraining full network initially (can freeze base weights and learn small projection).

## A. Post‑Hoc Root Trajectory Warping
Steps:
1. Provide desired 2D path as polyline or spline P(s) = (x(s), z(s)), s∈[0,1].
2. Extract generated motion `M` shape `[1, 22, 3, T]` (T frames). Get original root translation R_orig[t] = M[0, 0, [0,2], t].
3. Compute cumulative horizontal arc length of original root; normalize to [0,1] to produce parameter u[t].
4. Sample path points P(u[t]). Replace root x,z with these values. Optionally align heading by rotating all joints each frame so that forward vector matches path tangent.
5. (Optional) Foot contact correction: detect foot joints with low vertical velocity + low horizontal velocity before warp; after warp + rotation, apply small IK (2‑bone) or velocity projection to reduce foot sliding.

Code Sketch (conceptual):
```python
def warp_motion_to_path(motion_xyz, path_xy):
    # motion_xyz: [J,3,T]; path_xy: function u->[x,z]
    import numpy as np
    J, _, T = motion_xyz.shape
    root = motion_xyz[0]  # [3,T]
    horiz = root[[0,2]]  # x,z
    d = np.linalg.norm(np.diff(horiz, axis=1), axis=0)
    s = np.concatenate([[0], np.cumsum(d)])
    u = s / s[-1] if s[-1] > 1e-6 else np.linspace(0,1,T)
    target = np.array([path_xy(ui) for ui in u]).T  # [2,T]
    # heading adjustment
    tangents = np.gradient(target, axis=1)
    headings = np.arctan2(tangents[1], tangents[0])
    root_y = root[1:2]
    # compute per-frame yaw needed to rotate original forward to path tangent
    # (Assume original forward along +Z; adapt as needed.)
    def rot_yaw(theta):
        c,s = np.cos(theta), np.sin(theta)
        R = np.array([[c,0,s],[0,1,0],[-s,0,c]])
        return R
    motion_out = motion_xyz.copy()
    for t, theta in enumerate(headings):
        R = rot_yaw(theta)
        motion_out[:, :, t] = R @ motion_out[:, :, t]
        motion_out[0,0,t] = target[0,t]
        motion_out[0,2,t] = target[1,t]
        motion_out[0,1,t] = root_y[0,t]
    return motion_out
```

Pros: zero model changes. Cons: no proactive anticipation (arms/torso may not lean into turns).

## B. Hard Root Override During Sampling
Hook inside diffusion loop (`DiffusionWrapper_FlowMDM.p_sample_loop_progressive`). After each `out = self.diffusion.p_sample(...)`, modify `out['sample'][:, root_joint, :3, t]` for all frames according to target path at final coordinate frame. Because denoising refines full sequence each step, you can override entire temporal root track every iteration.

Patch Example (pseudo):
```python
# inside for t in indices:
out = self.diffusion.p_sample(...)
with torch.no_grad():
    sample = out['sample']  # [B,J,F,T]
    target_xy = torch.from_numpy(path_xy_seq).to(sample.device)  # [2,T]
    sample[:,0,0,:] = target_xy[0]
    sample[:,0,2,:] = target_xy[1]
```
Add yaw reorientation by rotating all joints about Y each step (torch.matmul with per-frame rotation matrices). This enforces path precisely; minimal extra code.

## C. Extra Conditioning Tokens (No Full Retrain)
1. Precompute per-frame features: (x_path, z_path, dx, dz, heading, curvature).
2. Project to latent via small MLP: `CondMLP: R^K -> R^{latent_dim}`.
3. Concatenate to text embedding sequence before transformer OR supply as `cond_tokens` additional rows. Must also adjust mask.
4. Finetune only MLP (and optionally final layers) with reconstruction loss on original training motions where path features = ground truth root trajectory stats.

Minimal model edit location: after `emb = self.compute_embedding(...)` in `FlowMDM.forward` before `seqTransEncoder` call.

## D. Guidance Loss (At Inference Only)
Classifier‑free analogy: run two forward passes (with and without path). Instead, simpler: define differentiable loss L_path = Σ_t || root_xy_t - path_xy_t ||^2 (optionally + heading alignment). During each diffusion step, treat current noisy sample x_t as requiring gradient step: `x_t = x_t - η * ∇_{x_t} L_path`. Insert before advancing to next timestep.

Pros: No model weight changes, soft adherence (tunable). Cons: Extra backward pass per step → slower.

## E. Proper Training Extension
Add root trajectory channel or features to input/ output representation so model inherently predicts path‑aligned motion when provided path tokens. Requires dataset labels (already have root trajectory implicitly) so you can synthesize random target paths by spatially warping training motions to diversify.

## Edge Considerations
- Frame Rate: Ensure path sampling uses dataset FPS (20 HumanML3D, 30 Babel) for correct speed.
- Path Length vs Duration: If mismatch, reparameterize path by arc length to match total motion distance or insert idle frames.
- Velocity Continuity: If using post‑hoc warp, recompute velocities for any downstream physics or contact heuristics.
- Foot Contact: Consider simple heuristic (foot height < h_thresh & |Δpos| < v_thresh) to lock feet during warp (solve small IK to root).
- Orientation Discontinuities: Smooth heading with Savitzky–Golay or spline fit before applying per-frame rotation.

## Minimal Practical Recipe (Copy/Paste)
1. Generate motion normally (`runners.generate`).
2. Load `results.npy`; pick a sample motion `motion = data['motion'][i]  # [J,3,T]`.
3. Define path polyline `P = [(x0,z0), ...]`; build arc length parameterization.
4. Apply warping function (Section A). Save new `motion_warped`.
5. Visualize with existing plotting utilities (skip subtraction of original trajectory).

## When to Move Beyond Post‑Hoc
If you need believable anticipation (lean before turns, step placement aligning with curvature) or strict biomechanics (reduced foot sliding), prefer B or D; for large‑scale generation with varied paths, implement C/E.

## Summary
You cannot directly feed a 2D path to vanilla FlowMDM. However, precise path‑following is achievable today with a short post‑processing script, and tighter integration can be layered in gradually without retraining the core transformer (override) or with small auxiliary modules (embedding, guidance). Full retraining allows the richest motion quality but is optional.

## References / Further Reading
- FlowMDM paper & code (current repo) – baseline text conditioning only.
- Classifier‑Free Guidance: Ho & Salimans (2022) – template for guidance style edits.
- Trajectory conditioning precedents: MDM variants & motion diffusion works (e.g., MotionDiffuse) for design inspiration.
