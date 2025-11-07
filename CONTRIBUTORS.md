# Contributors

This file acknowledges the contributors to Eric's Image Processing Nodes for ComfyUI.

## Project Lead & Primary Author

**Eric Hiss** (@EricRollei)
- Email: eric@historic.camera, eric@rollei.us
- Role: Project creator, maintainer, primary developer
- Contributions: All nodes, documentation, infrastructure

## Third-Party Code & Libraries

This project builds upon excellent work from the research and open-source communities:

### Algorithm Authors & Research Contributors

**Kai Zhang** (cszn)
- Algorithms: DnCNN, SCUNet, SwinIR (co-author)
- Source: https://github.com/cszn
- Contributions: Multiple state-of-the-art denoising algorithms

**Jingyun Liang** et al.
- Algorithm: SwinIR
- Source: https://github.com/JingyunLiang/SwinIR
- Paper: "SwinIR: Image Restoration Using Swin Transformer" (ICCV 2021)

**Liangyu Chen** et al.
- Algorithm: NAFNet
- Source: https://github.com/megvii-research/NAFNet
- Paper: "Simple Baselines for Image Restoration" (ECCV 2022)

**Kostadin Dabov**, **Alessandro Foi**, **Vladimir Katkovnik**, **Karen Egiazarian**
- Algorithm: BM3D
- Paper: "Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering" (IEEE TIP 2007)
- Seminal work in image denoising

**lizhihao6**
- Library: pytorch-bm3d (GPU-accelerated BM3D implementation)
- Source: https://github.com/lizhihao6/pytorch-bm3d
- License: MIT License
- Contribution: Makes BM3D practical for real-time use

**DeepInv Contributors**
- Library: DeepInv framework
- Source: https://github.com/deepinv/deepinv
- License: BSD 3-Clause License
- Contribution: Unified interface for multiple restoration models

**XPixelGroup**
- Library: DiffBIR (Diffusion-based Blind Image Restoration)
- Source: https://github.com/XPixelGroup/DiffBIR
- License: Apache 2.0
- Contribution: State-of-the-art diffusion-based restoration

### Open Source Library Contributors

**PyTorch Team**
- Library: PyTorch
- License: BSD 3-Clause
- Essential deep learning framework

**NumPy Developers**
- Library: NumPy
- License: BSD 3-Clause
- Fundamental numerical computing

**OpenCV Contributors**
- Library: OpenCV
- License: Apache 2.0
- Computer vision fundamentals

**PyWavelets Team**
- Library: PyWavelets
- License: MIT
- Wavelet transform implementation

**SciPy Developers**
- Library: SciPy
- License: BSD 3-Clause
- Scientific computing tools

**scikit-image Team**
- Library: scikit-image
- License: BSD 3-Clause
- Image processing algorithms

**CuPy Team**
- Library: CuPy
- License: MIT
- GPU-accelerated NumPy equivalent

**Hugging Face**
- Library: transformers
- License: Apache 2.0
- Pretrained model infrastructure

**Ross Wightman** (rwightman)
- Library: timm (PyTorch Image Models)
- License: Apache 2.0
- Vision model architectures

## Research Paper Citations

This project implements algorithms from numerous research papers. Complete citations are available in:
- `LICENSE` file (main credits)
- `Docs/README.md` (References section)
- Individual node file headers
- `CITATION.cff` (machine-readable format)

## Community Contributors

<!-- Future contributors will be listed here -->
<!-- Format:
**Name** (@github-username)
- Date: YYYY-MM
- Contribution: Description of contribution
- Email: email@example.com (optional)
-->

## How to Be Added

If you contribute to this project:

1. **Code Contributions**: Submit a pull request following the guidelines in `CONTRIBUTING.md`
2. **Bug Reports**: File detailed issues on GitHub
3. **Documentation**: Improve guides and examples
4. **Testing**: Help test and validate fixes

Significant contributors will be added to this file in recognition of their work.

## Special Thanks

- **ComfyUI Community**: For building an amazing platform
- **Research Community**: For publishing open-access papers and code
- **Open Source Contributors**: For the libraries that make this possible

## Contact

For questions about contributions or to report missing attributions:
- GitHub Issues: https://github.com/EricRollei/Eric_Image_Processing_Nodes/issues
- Email: eric@historic.camera

---

*Last Updated: November 6, 2025*
