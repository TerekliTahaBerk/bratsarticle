# Third-party notices

The repository's original source code is licensed under Apache-2.0. It imports
third-party Python packages but does not vendor their source. Users must comply
with the licenses of the versions installed in their environment.

Direct runtime dependencies and the optional reportable baselines use
permissive licenses compatible with linking/importing from this Apache-2.0
project:

| Package or implementation | Recorded license | Use |
|---|---|---|
| PyTorch | Apache-2.0 and bundled permissive notices | Tensor execution |
| MONAI / Swin UNETR | Apache-2.0 | Modern 3D hybrid baseline |
| nnU-Net v2 | Apache-2.0 | Official 2D and 3D baselines |
| Google DeepMind `surface-distance` | Apache-2.0 | Surface metrics |
| NumPy, pandas, SciPy, scikit-image, scikit-learn, psutil | BSD-family/permissive | Numerical and scientific utilities |
| NiBabel and OmegaConf | MIT/BSD-family | NIfTI I/O and configuration |

BraTS-Africa processed images, labels and metadata are CC BY 4.0 and are not
redistributed by this repository. BraTS 2020 data are likewise not bundled;
authorized users obtain them under the original dataset terms.

Exact installed versions are recorded in the environment lock. The official
nnU-Net and MONAI implementations are called as dependencies; their code is not
copied into this repository.
