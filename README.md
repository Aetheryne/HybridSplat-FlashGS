# HybridSplat-FlashGS

> **Modified version for [HybridSplat](https://github.com/Aetheryne/HybridSplat).**
> Original: [FlashGS](https://maxwellf1.github.io/flashgs_page) — real-time 3DGS rendering library.

This is the accelerated Gaussian Splatting renderer used by HybridSplat in the final evaluation stage (`use_flashgs_rendering=True`). It contains two categories of modifications over the original FlashGS:

1. **2DGS support**: the renderer is adapted from 3D Gaussian Splatting (volumetric ellipsoids) to **2D Gaussian Splatting** (flat disk primitives / surfels).
2. **Dual-branch attributes**: extended to load and render per-Gaussian `specular` and `roughness` attributes for HybridSplat's reflective G_ref branch.

---

## Modification 1 — 2DGS Support

The original FlashGS operates on 3DGS primitives using 2D screen-space Gaussian evaluation (`rho2d`). HybridSplat uses 2DGS (flat surfels), which requires a proper 3D ray–splat intersection model. The following changes implement this.

### `compute_transmat` — splat-space transformation matrix

A new `compute_transmat` function is added to `preprocess.cu`. Given a Gaussian's center, 2D scale `(s_u, s_v)`, and quaternion rotation, it constructs the local **UV-tangent-plane to world** transformation matrix `T`:

```
T = R · S    (3×3, rows: Tu, Tv, Tw)
```

where the third row `Tw` encodes the view-space depth direction. The surface normal is derived from the rotation and back-face corrected by checking `dot(p_view, normal)`.

The three rows `Tu`, `Tv`, `Tw` are stored in dedicated per-Gaussian buffers and passed to the render kernel.

### Ray–splat intersection in the render kernel

The original 3DGS alpha computation (commented out in `render.cu`):

```cpp
// 3DGS calculation
float2 d = { xy.x - pixf.x, xy.y - pixf.y };
float power = con_o.w + con_o.x*d.x*d.x + con_o.z*d.y*d.y + con_o.y*d.x*d.y;
```

is replaced by the 2DGS ray–surfel intersection from the [2DGS paper](https://surfsplatting.github.io/):

```cpp
// 2DGS calculation
float3 k = pixf.x * Tw - Tu;
float3 l = pixf.y * Tw - Tv;
float3 p = cross(k, l);
float2 s = { p.x / p.z, p.y / p.z };   // local UV coordinates on the disk
float rho3d = s.x*s.x + s.y*s.y;        // distance on the surfel disk

float2 d = { xy.x - pixf.x, xy.y - pixf.y };
float rho2d = FilterInvSquare * (d.x*d.x + d.y*d.y);  // 2D screen-space low-pass

float rho   = min(rho3d, rho2d);         // low-pass filter (2DGS paper §4.2)
float depth = (rho3d <= rho2d) ? (s.x*Tw.x + s.y*Tw.y) + Tw.z : Tw.z;
float alpha = min(0.99f, opacity * exp(-0.5f * rho));
```

The `rho = min(rho3d, rho2d)` combines the 3D disk distance with the 2D low-pass filter exactly as described in the 2DGS paper appendix.

### Normal map output

A `normal_opacity` buffer stores `(nx, ny, nz, opacity)` per Gaussian. The render kernel accumulates the normal during alpha compositing:

```cpp
N.x += nor_o.x * w;
N.y += nor_o.y * w;
N.z += nor_o.z * w;
```

producing a per-pixel surface normal map as an additional output.

---

## Modification 2 — Dual-branch Attributes (`specular` + `roughness`)

### PLY loader extended

The PLY loader reads two additional scalar per-Gaussian attributes that HybridSplat's pruning stage writes into the PLY files:

```
specular   — per-Gaussian baked reflection coefficient (r_REF / β weight)
roughness  — per-Gaussian roughness (auxiliary channel)
```

### `preprocess` packs them into `other_features`

```cpp
other_features[idx] = { specular[idx], roughness[idx] };
```

### Render kernel accumulates `other_features` per pixel

```cpp
O.x += other_feat.x * w;   // accumulated specular
O.y += other_feat.y * w;   // accumulated roughness
```

The output `out_other_features` (`float2` per pixel) carries the accumulated specular and roughness maps, used by HybridSplat for the final blending:

```
c = (1 - β) · c_base  +  β · c_ref
```

---

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

@inproceedings{Huang2DGS2024,
    title={2D Gaussian Splatting for Geometrically Accurate Radiance Fields},
    author={Huang, Binbin and Yu, Zehao and Chen, Anpei and Geiger, Andreas and Gao, Shenghua},
    booktitle={SIGGRAPH 2024 Conference Papers},
    year={2024},
}
```
