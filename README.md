# HybridSplat-FlashGS

> **Modified version for [HybridSplat](https://github.com/Aetheryne/HybridSplat).**
> Original: [FlashGS](https://maxwellf1.github.io/flashgs_page) — real-time 3DGS rendering library.

This is the accelerated 2D Gaussian Splatting renderer used by HybridSplat in the final evaluation stage (`use_flashgs_rendering=True`). It is extended to handle the **dual-branch Gaussian representation** (base Gaussians G_base + reflective Gaussians G_ref) by loading per-Gaussian `specular` and `roughness` attributes from the saved PLY files and accumulating them alongside RGB during rendering.

## Modifications

### 1. PLY loader extended for `specular` and `roughness`

The original PLY loader reads `position`, `SH coefficients`, `opacity`, and `cov3D`. The modified loader additionally reads two scalar per-Gaussian attributes:

```
specular   — per-Gaussian baked reflection coefficient (r_REF / β weight)
roughness  — per-Gaussian roughness (auxiliary channel)
```

These are written to the PLY files by HybridSplat's pruning stage (`save_ply_at_end=True`).

### 2. `preprocess` packs `specular + roughness` into `other_features`

In `preprocess.cu`, `specular` and `roughness` are packed into a per-Gaussian `float2`:

```cpp
other_features[idx] = { specular[idx], roughness[idx] };
```

This `other_features` buffer is passed to the tile-based rendering kernel alongside RGB.

### 3. Render kernel accumulates `other_features` per pixel

The render kernel alpha-composites `other_features` in exactly the same way as RGB:

```cpp
O.x += other_feat.x * w;   // accumulated specular
O.y += other_feat.y * w;   // accumulated roughness
```

The output `out_other_features` is a per-pixel `float2` containing the accumulated specular and roughness maps, which HybridSplat uses for the final `c = (1-β)·c_base + β·c_ref` blending.

## Hardware Requirements

NVIDIA server-grade and consumer-grade GPUs (tested on A100, V100, RTX 2080Ti, RTX 3090, RTX 4090).

## Installation

```bash
pip install .
```

## Citation

If you use FlashGS, please cite the original work:

```bibtex
@article{chen2024flashgs,
    title={FlashGS: Efficient 3D Gaussian Splatting for Large-scale and High-resolution Rendering},
    author={Chen, Guofeng and Liu, Siyan and Gao, Wentao and Peng, Shenghua},
    journal={arXiv preprint arXiv:2408.07967},
    year={2024},
}
```
