# ML for Science

PhD-level scientific ML suite covering **protein property prediction** (EGNN), **climate downscaling** (PINNs), **single-cell batch correction** (VAE + OT), and **materials active learning** — all evaluated on synthetic data with known ground truth.

## Modules

| Module | Description | Key metrics |
|--------|-------------|-------------|
| **Protein** | SE(3)-equivariant GNN vs MLP on folding stability from 3D chains | RMSE, Spearman, rotation consistency |
| **Climate** | Residual PINN refinement over bicubic on spectral temperature fields | RMSE, spectral bias, physics residual |
| **Genomics** | VAE reconstruction + linear / Sinkhorn batch correction | Batch mixing, biological preservation |
| **Materials** | GP-based active learning over expensive property oracle | Regret, normalized score, top-k hit rate |

## Assumptions

- **Protein:** Stability is a function of rotation/translation-invariant geometry (radius of gyration, contact density)
- **Climate:** Fine field is a smooth Fourier sum; coarse grid is block-averaged; PINN learns a physics-aware residual correction over bicubic upsampling
- **Genomics:** Expression = cell-type profile + batch shift + noise; correction should mix batches without destroying type structure
- **Materials:** Property landscape is smooth in a low-dimensional active subspace; GP surrogate guides acquisition

## Setup

```bash
cd 10-ml-for-science
pip install -e ".[dev]"
```

## Run benchmarks

```bash
# All modules
python scripts/run_benchmark.py --config configs/protein_benchmark.yaml --module all

# Individual modules
python scripts/run_benchmark.py --config configs/protein_benchmark.yaml --module protein
python scripts/run_benchmark.py --config configs/climate_benchmark.yaml --module climate
python scripts/run_benchmark.py --config configs/genomics_benchmark.yaml --module genomics
python scripts/run_benchmark.py --config configs/materials_benchmark.yaml --module materials
```

Results are written to `results/{timestamp}/metrics.json` and `summary.md`.

## Run tests

```bash
pytest
```

## Project layout

```
src/ml_sci/
├── data/              # Protein, climate, genomics, materials DGPs with ground-truth accessors
├── models/            # EGNN, PINN, VAE building blocks
├── protein/           # Equivariant property prediction
├── climate/           # PINN and bicubic downscaling
├── genomics/          # VAE + batch correction (linear, Sinkhorn OT)
├── materials/         # Active learning (random, uncertainty, EI)
└── evaluation/        # Benchmark runner and reporting
```

## Future work

- Real protein structures (PDB) with ESM embeddings and AlphaFold coordinates
- ERA5 → regional climate downscaling with full Navier-Stokes physics losses
- scVI / scANVI integration on real 10x datasets
- Bayesian optimization with multi-fidelity materials simulators (DFT, MD)
