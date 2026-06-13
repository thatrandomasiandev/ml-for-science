# ML for Science

A research benchmark suite for **scientific machine learning** across four domains where domain structure critically shapes model design: protein property prediction (equivariant GNNs), climate downscaling (physics-informed neural networks), single-cell genomics batch correction (optimal transport), and materials discovery (Gaussian process active learning). All experiments use synthetic DGPs with known physical invariants and ground-truth oracles.

The central research question: *how do we build ML models that respect scientific symmetries, physical laws, and experimental constraints while achieving predictive accuracy?*

---

## Research scope

| Module | Domain | Methods | Primary metrics |
|--------|--------|---------|-----------------|
| **Protein** | 3D structure → folding stability | EGNN, MLP baseline | RMSE, Spearman, rotation consistency |
| **Climate** | Coarse → fine temperature fields | Bicubic + residual PINN | RMSE, spectral bias, physics residual |
| **Genomics** | scRNA-seq batch correction | VAE + linear / Sinkhorn OT | Batch mixing, biological preservation |
| **Materials** | Property optimization | GP active learning (random, UCB, EI) | Regret, normalized score, top-k hit rate |

---

## Module 1: Protein property prediction

### Problem formulation

Protein function depends on **3D structure**, and predictions must be invariant to rotation and translation (SE(3) symmetry). Standard MLPs on flattened coordinates violate this symmetry, requiring **equivariant architectures** (Satorras et al., 2021; Thomas et al., 2018).

### Implemented methods

| Method | Approach | Reference |
|--------|----------|-----------|
| **EGNN** | E(n)-equivariant graph neural network on residue graphs | Satorras et al. (2021) |
| **MLP baseline** | Non-equivariant baseline on flattened coordinates | Comparison |

EGNN updates node features and coordinates via message passing that commutes with Euclidean transformations — predictions are identical regardless of how the protein is oriented in space (Satorras et al., 2021).

### Synthetic DGP (`data/protein_dgp.py`)

- Random-walk 3D protein chains
- Stability = f(radius of gyration, contact density) — **SE(3)-invariant** by construction
- Ground-truth stability available for every conformation

### Evaluation metrics

- **RMSE / Spearman:** Prediction accuracy and ranking quality
- **Rotation consistency:** |f(R·x) − f(x)| — should be ≈ 0 for equivariant models

---

## Module 2: Climate downscaling

### Problem formulation

**Climate downscaling** maps coarse-resolution temperature fields to fine-resolution grids. Pure interpolation (bicubic) ignores physical structure; **physics-informed neural networks (PINNs)** (Raissi et al., 2019) embed PDE constraints into the loss function.

### Implemented method

1. **Bicubic upsampling** of coarse field (baseline)
2. **Residual PINN** (`models/pinn.py`) learns a physics-aware correction:
   - Data loss: match fine-resolution observations
   - Physics loss: penalize PDE residual (smoothness / Laplacian constraint)

### Synthetic DGP (`data/climate_dgp.py`)

- Fine field: smooth 2D Fourier sum (physically plausible temperature surface)
- Coarse field: block-averaged downsampling
- Known fine-resolution ground truth for RMSE evaluation

### Evaluation metrics

- **RMSE:** Reconstruction error vs. fine-resolution truth
- **Spectral bias:** Frequency-domain error (PINNs are known to prefer low frequencies; Rahaman et al., 2019)
- **Physics residual:** Magnitude of PDE constraint violation

---

## Module 3: Single-cell genomics batch correction

### Problem formulation

Single-cell RNA sequencing (scRNA-seq) experiments exhibit **batch effects** — technical variation between experimental runs that confounds biological signal (Haghverdi et al., 2018). Correction must **mix batches** while **preserving cell-type structure**.

### Implemented methods

| Method | Approach | Reference |
|--------|----------|-----------|
| **VAE** | Learn low-dimensional latent representation of expression profiles | Kingma & Welling (2014) |
| **Linear correction** | Remove batch-specific linear shift in latent space | ComBat-style (Johnson et al., 2007) |
| **Sinkhorn OT** | Optimal transport alignment between batch distributions | Kassraee et al. (2022) |

Optimal transport batch correction (Kassraee et al., 2022) finds a coupling between batch distributions that minimizes transport cost while aligning shared cell types.

### Synthetic DGP (`data/genomics_dgp.py`)

- Expression = cell-type profile + batch shift + noise
- Known cell-type labels and batch assignments
- Ground-truth uncorrected and corrected profiles available

### Evaluation metrics

- **Batch mixing:** k-NN batch entropy in latent space (higher = better mixing)
- **Biological preservation:** Cell-type silhouette score (higher = structure preserved)

---

## Module 4: Materials active learning

### Problem formulation

Materials discovery requires evaluating expensive property oracles (DFT calculations, synthesis + assay). **Active learning** with Gaussian process surrogates (Jones et al., 1998) selects the most informative compositions to evaluate next.

### Implemented acquisition strategies

| Strategy | Criterion | Reference |
|----------|-----------|-----------|
| **Random** | Uniform sampling | Baseline |
| **Uncertainty** | Maximize posterior variance | GP uncertainty sampling |
| **Expected improvement** | Maximize E[max(f* − f(x), 0)] | Jones et al. (1998) |

### Synthetic DGP (`data/materials_dgp.py`)

- Property landscape smooth in a **low-dimensional active subspace**
- Known global optimum for regret computation
- Tunable evaluation budget

### Evaluation metrics

- **Regret:** f(x*) − f(x̂_best) after budget exhausted
- **Normalized score:** Improvement over random search
- **Top-k hit rate:** Fraction of true top-k compositions found

---

## Benchmark protocol

```bash
pip install -e ".[dev]"

python scripts/run_benchmark.py --config configs/protein_benchmark.yaml --module all
python scripts/run_benchmark.py --config configs/protein_benchmark.yaml --module protein
python scripts/run_benchmark.py --config configs/climate_benchmark.yaml --module climate
python scripts/run_benchmark.py --config configs/genomics_benchmark.yaml --module genomics
python scripts/run_benchmark.py --config configs/materials_benchmark.yaml --module materials

pytest
```

---

## Project layout

```
src/ml_sci/
├── data/              # Protein, climate, genomics, materials DGPs
├── models/            # EGNN, PINN, VAE building blocks
├── protein/           # Equivariant property prediction
├── climate/           # PINN downscaling over bicubic baseline
├── genomics/          # VAE + batch correction (linear, Sinkhorn OT)
├── materials/         # GP active learning strategies
└── evaluation/        # Benchmark runner and reporting
```

---

## Implementation notes

- EGNN is a **simplified** E(n)-equivariant GNN (Satorras et al., 2021); full implementation includes edge features and higher-order interactions
- PINN physics loss uses **smoothness regularization**, not full Navier-Stokes equations
- Batch correction operates on **VAE latents**, not raw expression space (following scVI; Lopez et al., 2018)
- GP surrogate uses **sklearn GaussianProcessRegressor** with RBF kernel

---

## References

- Haghverdi, L., Lun, A. T. L., Morgan, M. D., & Marioni, J. C. (2018). Batch effects in single-cell RNA-sequencing data are corrected by matching mutual nearest neighbors. *Nature Biotechnology*, 36(5), 421–427. [DOI](https://doi.org/10.1038/nbt.4091)
- Johnson, W. E., Li, C., & Rabinovic, A. (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics*, 8(1), 118–127. [DOI](https://doi.org/10.1093/biostatistics/kxj037)
- Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global optimization of expensive black-box functions. *Journal of Global Optimization*, 13(4), 455–492. [DOI](https://doi.org/10.1023/A:1008306431147)
- Kassraee, P., et al. (2022). Single-cell matching with optimal transport. [bioRxiv](https://doi.org/10.1101/2022.04.06.487174)
- Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. *ICLR*. [arXiv](https://arxiv.org/abs/1312.6114)
- Lopez, R., et al. (2018). Deep generative modeling for single-cell transcriptomics. *Nature Methods*, 15(12), 1053–1058. [DOI](https://doi.org/10.1038/s41592-018-0229-2)
- Rahaman, N., et al. (2019). On the spectral bias of neural networks. *ICML*. [arXiv](https://arxiv.org/abs/1806.08734)
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707. [DOI](https://doi.org/10.1016/j.jcp.2018.10.045)
- Satorras, V. G., Hoogeboom, E., & Welling, M. (2021). E(n) equivariant graph neural networks. *ICML*. [arXiv](https://arxiv.org/abs/2102.09844)
- Shahriari, B., et al. (2016). Taking the human out of the loop: A review of Bayesian optimization. *Proceedings of the IEEE*, 104(1), 148–175. [DOI](https://doi.org/10.1109/JPROC.2015.2494218)
- Thomas, N., et al. (2018). Tensor field networks: Rotation- and translation-equivariant neural networks for 3D point clouds. [arXiv](https://arxiv.org/abs/1802.08219)

---

## Future work

- Real protein structures (PDB) with ESM embeddings (Lin et al., 2023)
- ERA5 → regional downscaling with full PDE constraints
- scVI / scANVI integration on 10x Genomics datasets
- Multi-fidelity Bayesian optimization with DFT/MD simulators
