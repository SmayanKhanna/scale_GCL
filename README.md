# Graph Contrastive Learning versus Untrained Baselines: The Role of Dataset Size

[![Paper](https://img.shields.io/badge/arXiv-24XX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/24XX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation for our paper, **Graph Contrastive Learning versus Untrained Baselines: The Role of Dataset Size**. We investigate when Graph Contrastive Learning (GCL) truly offers an advantage over simple, untrained baselines for graph classification.

Our key finding is that the benefit of GCL is strongly dependent on **dataset size**. On small, standard TU datasets, we find that simple baselines can often rival or even outperform GCL methods. However, on larger datasets like OGBG-MOLHIV, we observe a clear crossover effect where GCL's advantage emerges and then plateaus as the number of training graphs increases.

<img width="700" height="360" alt="molhiv_scaling_dual" src="https://github.com/user-attachments/assets/4dc6882c-4ebd-42b7-a374-a5fc8cda92cf" />

> **Figure:** Our central finding on OGBG-MOLHIV. At small scales (<4k graphs), an untrained GNN baseline is competitive. GCL's advantage only becomes apparent as the dataset size increases, though this gain eventually plateaus.

---
## Setup and Installation

To get started, clone the repository and set up the Conda environment. Our code relies on PyTorch, PyTorch Geometric (PyG) and networkX.

```bash
# Clone the repository
git clone [https://github.com/your-username/scale_GCL.git](https://github.com/your-username/scale_GCL.git)
cd scale_GCL

# Create Conda environment (recommended)
conda create -n scale_gcl python=3.10
conda activate scale_gcl

# Install PyTorch (select the command for your CUDA version)
# See: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
pip install torch torchvision torchaudio

# Install PyTorch Geometric
pip install torch_geometric

# Install other dependencies
pip install numpy
pip install networkx
```

## Citation

If you find our work useful, please consider citing our paper:

```bash
@article{Khanna2025GCLScale,
  title   = {Graph Contrastive Learning versus Untrained Baselines: The Role of Dataset Size},
  author  = {Khanna, Smayan and G{\"o}kmen, Doruk Efe and Kondor, Risi and Vitelli, Vincenzo},
  journal = {arXiv preprint arXiv:24XX.XXXXX},
  year    = {2025}
}
