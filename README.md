<h1 align="center">FlowMDM</h3>

<h3 align="center">Seamless Human Motion Composition with Blended Positional Encodings (CVPR'24)</h3>

  <p align="center">
    <a href="https://barquerogerman.github.io/FlowMDM/"><img alt="Project" src="https://img.shields.io/badge/-Project%20Page-lightgrey?logo=Google%20Chrome&color=informational&logoColor=white"></a>
    <a href="https://arxiv.org/abs/2402.15509"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2402.15509-b31b1b.svg"></a> 
    <img alt="visits" src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FBarqueroGerman%2FFlowMDM&count_bg=%2320AF15&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visits&edge_flat=false">
  </p>

<br>

Human Motion Composition             |  Human Motion Extrapolation
:-------------------------:|:-------------------------:
![](assets/example_composition.gif)  |  ![](assets/example_extrapolation.gif)


## 🔎 About
<div style="text-align: center;">
    <img src="assets/main_figure.png" align="center" width=100% >
</div>
</br>
Conditional human motion generation is an important topic with many applications in virtual reality, gaming, and robotics. 
While prior works have focused on generating motion guided by text, music, or scenes, these typically result in isolated motions confined to short durations. 
Instead, we address the generation of long, continuous sequences guided by a series of varying textual descriptions. In this context, we introduce FlowMDM, the first diffusion-based model that generates seamless Human Motion Compositions (HMC) without any postprocessing or redundant denoising steps. For this, we introduce the Blended Positional Encodings, a technique that leverages both absolute and relative positional encodings in the denoising chain. More specifically, global motion coherence is recovered at the absolute stage, whereas smooth and realistic transitions are built at the relative stage. As a result, we achieve state-of-the-art results in terms of accuracy, realism, and smoothness on the Babel and HumanML3D datasets. FlowMDM excels when trained with only a single description per motion sequence thanks to its Pose-Centric Cross-ATtention, which makes it robust against varying text descriptions at inference time. Finally, to address the limitations of existing HMC metrics, we propose two new metrics: the Peak Jerk and the Area Under the Jerk, to detect abrupt transitions.

<!--
## Running instructions
-->

## 📌 News
- [2024-05-13] Eval/Gen instructions updated (wrong value in --bpe_denoising_step fixed)
- [2024-03-18] Code + model weights released!
- [2024-02-27] FlowMDM is now accepted at CVPR 2024!
- [2024-02-26] Our paper is available in [Arxiv](https://arxiv.org/abs/2402.15509).

## 📝 TODO List
- [x] Release pretrained models.
- [x] Release generation (skeletons + blender support for meshes) + evaluation + training code.
- [ ] Release generation code for demo-style visualizations.

## 👩🏻‍🏫 Getting started

This code was originally tested on Ubuntu 20.04.6 LTS + Python 3.8 + PyTorch 1.13.0 (the "default" / legacy environment). A modern "latest" pixi environment is also provided with recent Python (3.11+) and PyTorch (2.7.x CUDA 12.6). For new work, prototyping, or interactive visualization we recommend using the latest environment; use the default environment only when strict reproduction of the original paper setup is required.

### Environments (legacy vs latest)

| Env | Purpose | Python | PyTorch | CUDA | Activate / Run Prefix |
|-----|---------|--------|---------|------|-----------------------|
| default | Reproduce paper, legacy deps | 3.8/3.9 | 1.13.0+cu117 | 11.7 | `pixi run <task>` |
| latest  | Modern development & visualization | 3.11+ | 2.7.1+cu126 | 12.6 | `pixi run -e latest <task>` |

Recommendations:
* Use `-e latest` for faster sampling, improved tooling, and interactive PyVista animations.
* Use plain tasks (no `-e latest`) only if you must replicate legacy numerical results.

### Prerequisites

[Pixi](https://pixi.sh/) - Modern package manager that replaces conda. Install from [pixi.sh](https://pixi.sh/)

### Installation

**Option 1: Quick Setup (Legacy / Paper Reproduction)**
```shell
# Install dependencies and setup environment
pixi run setup

# Test CUDA availability
pixi run test-cuda
```

**Option 2: Modern (Latest) Environment**
```shell
# Create & install latest environment (solves lock + installs)
pixi install -e latest

# One‑time setup: SpaCy model + chumpy (installed via task)
pixi run -e latest setup

# (Optional) Verify GPU
pixi run -e latest test-cuda
```

**Option 3: Manual Step-by-Step (Legacy)**
```shell
# Install pixi environment
pixi install

# Setup PyTorch with CUDA + additional dependencies  
pixi run setup

# Generate motion from text
pixi run generate-motion
```

> [!IMPORTANT]
> The setup automatically installs:
> - PyTorch 1.13.0 with CUDA 11.7 support
> - The `chumpy` package from git (PyPI version 0.70 has NumPy compatibility issues)
> - CLIP, SMPLX, and all other dependencies
> - SpaCy English model for text processing

### Available Commands (Legacy)

```shell
pixi run help              # Show generation options
pixi run generate-motion   # Generate sample walking motion  
pixi run pytorch-version   # Check PyTorch version
pixi run test-cuda        # Verify CUDA setup
```

### Available Commands (Latest)

```shell
pixi run -e latest help              # Show generation options
pixi run -e latest generate-motion   # Generate sample walking motion (modern stack)
pixi run -e latest pytorch-version   # Check PyTorch version (latest env)
pixi run -e latest test-cuda         # Verify CUDA setup (latest env)
```

This [README file](https://github.com/BarqueroGerman/FlowMDM/blob/main/runners/README.md) contains instructions on how to visualize, evaluate, and train the model.

## 🕹 Interactive Visualization

An interactive PyVista / Qt utility is provided to inspect generated motion results:

`visualization/show-animation.py`

Usage:

```shell
# After generating results (creates a directory with results.npy)
pixi run -e latest python visualization/show-animation.py results/babel/FlowMDM/<RESULT_DIR> --autoplay

# Or point directly to a results.npy file
pixi run -e latest python visualization/show-animation.py results/babel/FlowMDM/<RESULT_DIR>/results.npy

# Custom playback FPS (defaults: 30 for Babel, 20 for HumanML3D)
pixi run -e latest python visualization/show-animation.py <RESULT_DIR> --fps 24
```

Controls:
* Space: Play/Pause
* Left / Right: Step one frame
* r: Reset to frame 0
* q: Quit window

Notes:
* Provide either the directory containing `results.npy` or the file path itself.
* Prefer the `latest` environment for smoother real‑time interaction and modern PyVista.
* The utility performs in‑place point updates for efficient rendering of long sequences.

> [!NOTE]
> This repository inherits a lot of work from the original MDM and Guided-Diffusion repositories. Most of FlowMDM's contribution can be found in the `model/FlowMDM.py` and  `diffusion/diffusion_wrappers.py` files, and the `model/x_transformers` folder.

## 📚 Citation

If you find our work helpful, please cite:

```bibtex
@inproceedings{barquero2024seamless,
  title={Seamless Human Motion Composition with Blended Positional Encodings},
  author={Barquero, German and Escalera, Sergio and Palmero, Cristina},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```

## 🤝🏼 Acknowledgements
- [TEMOS](https://github.com/Mathux/TEMOS): We inherit a lot of the code from TEMOS.
- [TEACH](https://github.com/athn-nik/teach): We use TEACH in our work, and inherit part of the code from them.
- [MDM](https://guytevet.github.io/mdm-page/): We use MDM in our work, and inherit as well part of the code.
- [PriorMDM](https://github.com/priorMDM/priorMDM): We use PriorMDM in our work, and inherit as well part of the code.
- [x-transformers](https://github.com/lucidrains/x-transformers): BPEs are built on their transformers library.

## ⭐ Star History


<p align="center">
    <a href="https://star-history.com/#BarqueroGerman/FlowMDM&Date" target="_blank">
        <img width="500" src="https://api.star-history.com/svg?repos=BarqueroGerman/FlowMDM&type=Date" alt="Star History Chart">
    </a>
<p>
