# Related Works on Path-Retargeting of Human Motion Clips (Edit trajectory while preserving motion style)

## Overview

- Task: Given an existing human motion clip, modify its global trajectory/path explicitly while maintaining local pose dynamics and overall motion style (e.g., walk cycle, timing, contacts) as close as possible to the original.
- Core dimensions:
  - Path control: explicit root/pelvis trajectory constraints or waypoints.
  - Style preservation: minimal local changes, foot contact integrity, smoothness.
  - Editing vs. generation: ability to take an input motion and retarget its path vs. re-generate from scratch.
  - Implementation readiness: open-source code, maintenance, ease of integration into motion pipelines (e.g., BVH/SMPL(H/X), HumanML3D).
- Keywords: motion path editing, trajectory constraints, spacetime constraints, contact-preserving warping, motion inpainting, diffusion guidance, IK post-correction, retargeting.

## Related Papers

### Diffusion-based motion control/editing (Open-source)

#### Guided Motion Diffusion (GMD): Controllable Human Motion Synthesis (ICCV 2023)
- **Paper**: https://openaccess.thecvf.com/content/ICCV2023/papers/Karunratanakul_Guided_Motion_Diffusion_for_Controllable_Human_Motion_Synthesis_ICCV_2023_paper.pdf
- **Github Page**: https://github.com/korrawe/guided-motion-diffusion
- **Tutorial**: —
- **YouTube Demo**: Project page videos https://korrawe.github.io/gmd-project/
- **Summary**: Adds spatial constraints (e.g., predefined global trajectory and obstacles) at inference via feature projection and an imputation formulation, guiding a pretrained text-to-motion diffusion model to follow paths. Supports root-trajectory control without retraining for each task. Highly relevant for path retargeting; can be adapted to edit an existing clip by conditioning and imitation/inpainting setups.

#### OmniControl: Control Any Joint at Any Time for Human Motion Generation (ICLR 2024)
- **Paper**: https://arxiv.org/abs/2310.08580
- **Github Page**: https://github.com/neu-vi/omnicontrol
- **Tutorial**: —
- **YouTube Demo**: Project page videos https://neu-vi.github.io/omnicontrol/
- **Summary**: Analytic spatial guidance allows dense control over arbitrary joints and times, including pelvis/root trajectories. While framed as generation, it can be used to force a given motion to follow a new path combined with inpainting. Strong control fidelity; suitable for path retargeting with careful constraints.

#### PriorMDM: Human Motion Diffusion as a Generative Prior (arXiv 2023; with controls)
- **Paper**: https://arxiv.org/abs/2303.01418
- **Github Page**: https://github.com/priorMDM/priorMDM
- **Tutorial**: —
- **YouTube Demo**: —
- **Summary**: Treats motion diffusion as a prior to support tasks such as inpainting and root/joint control. The repo includes examples for horizontal root trajectory control and mixed joint control. Particularly useful to preserve style while adjusting root trajectory via inpainting of global motion frames.

#### InterControl: Zero-shot Human Interaction Generation by Controlling Every Joint (NeurIPS 2024)
- **Paper**: https://papers.nips.cc/paper_files/paper/2024/hash/ (see project/repo for details)
- **Github Page**: https://github.com/zhenzhiwang/intercontrol
- **Tutorial**: —
- **YouTube Demo**: —
- **Summary**: Builds on PriorMDM to enable versatile joint controls, including positional goals enforced via IK-like guidance during sampling. While focused on interactions, the joint-control interface covers pelvis/root trajectories and waypoints, enabling explicit path editing.

#### Flexible Motion In-betweening with Diffusion Models (CondMDI) (2024)
- **Paper/Project**: https://setarehc.github.io/CondMDI/
- **Github Page**: —
- **Tutorial**: —
- **YouTube Demo**: —
- **Summary**: Keyframe in-betweening with support for conditioning on root trajectory and joint trajectories. Shows accurate tracking and low foot skate compared to MDM/PriorMDM/OmniControl. If code becomes available, it would be useful for path-retargeting via sparse constraints.

#### MoLA: Motion Generation and Editing with Latent Diffusion (CVPRW 2025)
- **Paper**: https://openaccess.thecvf.com/content/CVPR2025W/HuMoGen/papers/Uchida_MoLA_Motion_Generation_and_Editing_with_Latent_Diffusion_Enhanced_by_CVPRW_2025_paper.pdf
- **Github Page**: —
- **Tutorial**: —
- **YouTube Demo**: —
- **Summary**: Presents training-free editing with demonstrated path-following tasks and comparisons vs. OmniControl/MotionLCM. Promising for minimally invasive path retargeting but code availability is unclear.

### Classic constraint-based motion editing (Foundational; no official code)

#### Motion Editing with Spacetime Constraints (SIGGRAPH I3D 1997)
- **Paper**: https://research.cs.wisc.edu/graphics/Papers/Gleicher/California/SpacetimeEditing.pdf
- **Github Page**: —
- **Tutorial**: —
- **YouTube Demo**: Author video page https://gleicher.sites.cs.wisc.edu/video/1997_st-moedit/
- **Summary**: Optimizes motion over a time window to satisfy spatial/temporal constraints while preserving original motion via displacement maps and band-limited changes. Directly addresses “edit path while preserving style” with global optimization.

#### Retargetting Motion to New Characters (SIGGRAPH 1998)
- **Paper**: https://icg.gwu.edu/sites/g/files/zaxdzs6126/files/downloads/Retargetting%20motion%20to%20new%20characters.pdf
- **Github Page**: —
- **Tutorial**: —
- **YouTube Demo**: —
- **Summary**: Casts retargeting as a spacetime optimization preserving constraints such as footplants; emphasizes smooth displacement maps. The methodology informs path retargeting with contact integrity.

#### Comparing Constraint-Based Motion Editing Methods (CVIU 2001)
- **Paper**: https://graphics.cs.wisc.edu/Papers/2001/Gle01/compare-journal.pdf
- **Github Page**: —
- **Tutorial**: —
- **YouTube Demo**: —
- **Summary**: Surveys IK-based, warping, and spacetime methods, emphasizing band-limited edits to preserve motion style while enforcing spatial constraints.

#### Footskate Cleanup for Motion Capture Editing (SCA 2002)
- **Paper**: https://graphics.cs.wisc.edu/Papers/2002/KSG02/cleanup.pdf
- **Github Page**: —
- **Tutorial**: —
- **YouTube Demo**: —
- **Summary**: Analytic IK and blending for footplant enforcement and continuity. Essential post-process after path retargeting to eliminate skate and root popping.

#### Motion Path Editing (Talk/Slides)
- **Slides**: https://pdfs.semanticscholar.org/8704/8d59940aa7e26c2de1d788a47edc57843316.pdf
- **Github Page**: —
- **Tutorial**: —
- **YouTube Demo**: —
- **Summary**: Factor motion into “path” and “detail,” reparameterize along arc length, then apply detail atop new path. Explicitly targets our goal: change trajectory while preserving local motion characteristics.

### Additional controllable motion diffusion works (Open-source where available)

#### Multi-Track Timeline Control for Text-Driven 3D Human Motion Generation (CVPR 2024 Workshop)
- **Paper**: https://openaccess.thecvf.com/content/CVPR2024W/HuMoGen/papers/Petrovich_Multi-Track_Timeline_Control_for_Text-Driven_3D_Human_Motion_Generation_CVPRW_2024_paper.pdf
- **Github Page**: — (builds on MDM; see below)
- **Tutorial**: —
- **YouTube Demo**: —
- **Summary**: Composition and control across timeline; includes keyframe/trajectory controls using guidance/inpainting. Applicable for path editing via timeline constraints.

#### Motion Diffusion Model (MDM) baseline (CVPR 2023)
- **Paper**: https://arxiv.org/abs/2209.14916
- **Github Page**: https://github.com/GuyTevet/motion-diffusion-model
- **Tutorial**: —
- **YouTube Demo**: —
- **Summary**: Strong base model with widespread forks for control/inpainting. Useful starting point if building custom path-guided editing via test-time guidance.

### Practitioner tools and repos relevant to path retargeting workflows

#### deep-motion-editing (retargeting + IK/contact utilities)
- **Repo**: https://github.com/DeepMotionEditing/deep-motion-editing
- **Paper**: —
- **Tutorial**: —
- **YouTube Demo**: —
- **Summary**: Utilities for BVH processing and IK-based foot-contact fixes (e.g., `fix_foot_contact`), which can be chained after path retargeting (diffusion- or optimization-based) to preserve contact realism.

#### ControlMM (trajectory and joint control utilities over motion diffusion)
- **Repo**: https://github.com/exitudio/ControlMM
- **Paper**: —
- **Tutorial**: —
- **YouTube Demo**: —
- **Summary**: Provides practical scripts for controlling pelvis/root and limb joint trajectories during motion generation; supports different control densities and ControlNet toggles. Useful engineering scaffold for path control experiments.

#### BVH retargeters for Blender (rig retargeting; complements path editing)
- **Retarget-bvh (MakeWalk)**: https://github.com/Diffeomorphic/retarget-bvh
- **Retarget_bvh (fork)**: https://github.com/jbellenger/retarget_bvh
- **Summary**: While these focus on skeleton retargeting rather than path editing, they are often used downstream once the path-retargeted motion is produced on a canonical rig.

---

## Ranking by Relevance to “Edit Existing Motion Path while Preserving Style”

1) Guided Motion Diffusion (GMD) — precise root trajectory conditioning; open-source; minimal architectural changes; strong fit for path retargeting on existing clips via imputation.
2) PriorMDM — supports root and joint control with inpainting; widely used; practical for editing scenarios to preserve style with explicit trajectory control.
3) OmniControl — dense control over any joints/time; effective for path-following with strong fidelity, though more generation-oriented.
4) InterControl — expands joint control with IK-like target enforcement; can implement waypoint paths via pelvis/root control.
5) Classic Spacetime Path Editing (Gleicher 1997/1998 + Motion Path Editing) — gold-standard formulation for preserving style; requires custom implementation but provides theoretical foundation for global path edits with constraints and band-limited changes.
6) MoLA (training-free editing; path-following demonstrated) — promising but code not confirmed.
7) CondMDI — root/joint trajectory conditioning in in-betweening; code status unclear but conceptually applicable to path edits of existing clips.
8) Engineering utilities (deep-motion-editing, ControlMM, Blender retargeters) — not path editing alone but essential components to build a robust pipeline (contact cleanup, rig retargeting, visualization).

## Implementation Readiness and Integration Notes

- Best immediate starting points with code:
  - GMD (ICCV’23): straightforward to constrain root trajectories; can adapt to “edit” mode via imputation using the source motion as partial observations.
  - PriorMDM: has ready-made scripts for horizontal root trajectory control; combine with classifier-free guidance and low-weight text condition to keep original style.
  - OmniControl: for strict path adherence, provide dense pelvis controls along the new spline; inpaint or reconstruct local motion to stay close to source.
- Contact preservation:
  - After retargeting the path, run a contact-aware IK pass to mitigate footskate/root popping. Reuse utilities like deep-motion-editing’s foot contact fix, and optionally reproject onto terrain if applicable.
- Practical path authoring:
  - Represent the desired trajectory as a 2D/3D parametric curve (spline), perform arc-length reparameterization to match source clip timing, and set dense waypoints for pelvis/root, leaving limb/joint motions largely unconstrained to preserve style.
- Data formats:
  - For SMPL/HumanML3D pipelines, ensure consistent global frame conventions (absolute vs. relative root). GMD and related repos often require absolute-root preprocessing for trajectory conditioning.
- Evaluation:
  - Quantify trajectory error (ADE/FDE or average deviation to curve), style preservation (feature distance to original pose sequence), and contact metrics (foot sliding/contact violations).

## Notes on Tutorials and Demos (Medium/Zhihu/YouTube)

- Medium/Zhihu tutorials specific to these motion-diffusion path-editing repos are scarce at the time of writing; most guidance is in official READMEs and project pages.
- Where available, project pages (OmniControl, GMD) include embedded demo videos; YouTube links are not always explicitly provided.
