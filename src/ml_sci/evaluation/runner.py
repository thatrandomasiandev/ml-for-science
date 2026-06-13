"""Benchmark runner for protein, climate, genomics, and materials modules."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ml_sci.climate.downscale import fit_bicubic_downscaler, fit_pinn_downscaler
from ml_sci.data.climate_dgp import ClimateDGPConfig, generate_climate_data
from ml_sci.data.genomics_dgp import GenomicsDGPConfig, generate_genomics_data
from ml_sci.data.materials_dgp import MaterialsDGPConfig, generate_materials_data
from ml_sci.data.protein_dgp import ProteinDGPConfig, generate_protein_data
from ml_sci.genomics.batch_correction import apply_batch_correction
from ml_sci.genomics.metrics import batch_mixing_score, biological_preservation
from ml_sci.genomics.vae import train_vae
from ml_sci.materials.active_learning import run_active_learning
from ml_sci.materials.metrics import normalized_score, regret, top_k_hit_rate
from ml_sci.protein.trainer import fit_protein_predictor
from ml_sci.utils.seed import config_hash


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _aggregate(results: list[dict]) -> dict[str, float]:
    if not results:
        return {}
    keys = results[0].keys()
    return {k: float(np.mean([r[k] for r in results])) for k in keys if isinstance(results[0][k], (int, float))}


def _aggregate_std(results: list[dict]) -> dict[str, float]:
    if not results:
        return {}
    keys = results[0].keys()
    return {
        k: float(np.std([r[k] for r in results]))
        for k in keys
        if isinstance(results[0][k], (int, float))
    }


def run_protein_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Train EGNN vs MLP on synthetic protein stability prediction."""
    seeds = config.get("seeds", [42])
    models = config.get("models", ["egnn", "mlp"])
    n_proteins_list = config.get("n_proteins_list", [200, 400])
    n_residues = config.get("n_residues", 32)
    epochs = config.get("epochs", 80)
    device = config.get("device", "cpu")

    all_results = []
    for n_proteins in n_proteins_list:
        for model_type in models:
            seed_results = []
            for seed in seeds:
                data = generate_protein_data(
                    ProteinDGPConfig(n_proteins=n_proteins, n_residues=n_residues, seed=seed)
                )
                result = fit_protein_predictor(
                    data,
                    model_type=model_type,
                    epochs=epochs,
                    hidden_dim=config.get("hidden_dim", 64),
                    n_layers=config.get("n_layers", 2),
                    dropout=config.get("dropout", 0.1),
                    lr=config.get("lr", 0.001),
                    seed=seed,
                    device=device,
                )
                seed_results.append(
                    {
                        "rmse": result.rmse,
                        "spearman": result.spearman,
                        "rotation_consistency": result.rotation_consistency,
                        "train_loss": result.train_loss,
                    }
                )
            mean = _aggregate(seed_results)
            std = _aggregate_std(seed_results)
            all_results.append(
                {
                    "model": model_type,
                    "n_proteins": n_proteins,
                    **{f"{k}_mean": v for k, v in mean.items()},
                    **{f"{k}_std": v for k, v in std.items()},
                }
            )
    return {"module": "protein", "results": all_results}


def run_climate_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Compare bicubic and PINN climate downscaling."""
    seeds = config.get("seeds", [42])
    factors = config.get("downscale_factors", [4, 8])
    noise_levels = config.get("noise_levels", [0.05, 0.15])
    epochs = config.get("epochs", 100)
    device = config.get("device", "cpu")

    all_results = []
    for factor in factors:
        for noise_std in noise_levels:
            for method in ["bicubic", "pinn"]:
                seed_results = []
                for seed in seeds:
                    data = generate_climate_data(
                        ClimateDGPConfig(
                            downscale_factor=factor,
                            noise_std=noise_std,
                            seed=seed,
                        )
                    )
                    if method == "bicubic":
                        result = fit_bicubic_downscaler(data)
                    else:
                        result = fit_pinn_downscaler(
                            data,
                            epochs=epochs,
                            hidden_dim=config.get("hidden_dim", 64),
                            seed=seed,
                            device=device,
                        )
                    seed_results.append(
                        {
                            "rmse": result.rmse,
                            "spectral_bias": result.spectral_bias,
                            "physics_residual": result.physics_residual,
                            "train_loss": result.train_loss,
                        }
                    )
                mean = _aggregate(seed_results)
                std = _aggregate_std(seed_results)
                all_results.append(
                    {
                        "method": method,
                        "downscale_factor": factor,
                        "noise_std": noise_std,
                        **{f"{k}_mean": v for k, v in mean.items()},
                        **{f"{k}_std": v for k, v in std.items()},
                    }
                )
    return {"module": "climate", "results": all_results}


def run_genomics_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate VAE + batch correction on synthetic scRNA-seq."""
    seeds = config.get("seeds", [42])
    n_batches_list = config.get("batch_sizes", [2, 4])
    methods = config.get("correction_methods", ["none", "linear", "sinkhorn"])
    epochs = config.get("epochs", 80)
    device = config.get("device", "cpu")

    all_results = []
    for n_batches in n_batches_list:
        for method in methods:
            seed_results = []
            for seed in seeds:
                data = generate_genomics_data(
                    GenomicsDGPConfig(n_batches=n_batches, seed=seed)
                )
                vae = train_vae(
                    data.expression,
                    epochs=epochs,
                    hidden_dim=config.get("hidden_dim", 64),
                    seed=seed,
                    device=device,
                )
                corrected = apply_batch_correction(
                    vae.reconstructed,
                    data.batch_labels,
                    method=method,
                )
                seed_results.append(
                    {
                        "batch_mixing": batch_mixing_score(
                            corrected.corrected, data.batch_labels
                        ),
                        "bio_preservation": biological_preservation(
                            corrected.corrected,
                            data.expression,
                            data.cell_types,
                        ),
                        "vae_loss": vae.train_loss,
                    }
                )
            mean = _aggregate(seed_results)
            std = _aggregate_std(seed_results)
            all_results.append(
                {
                    "method": method,
                    "n_batches": n_batches,
                    **{f"{k}_mean": v for k, v in mean.items()},
                    **{f"{k}_std": v for k, v in std.items()},
                }
            )
    return {"module": "genomics", "results": all_results}


def run_materials_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Compare active learning strategies on materials oracle."""
    seeds = config.get("seeds", [42])
    methods = config.get("methods", ["random", "uncertainty", "expected_improvement"])
    budgets = config.get("eval_budgets", [30, 60])

    all_results = []
    for budget in budgets:
        for method in methods:
            seed_results = []
            for seed in seeds:
                data = generate_materials_data(MaterialsDGPConfig(seed=seed))
                gt = data.ground_truth
                result = run_active_learning(
                    data.compositions,
                    budget,
                    gt["optimum"],
                    gt["active_dims"],
                    data.metadata["length_scale"],
                    method=method,
                    seed=seed,
                )
                random_baseline = float(np.mean(result.all_properties[: max(budget // 10, 5)]))
                seed_results.append(
                    {
                        "best_property": result.best_property,
                        "regret": regret(result.best_property, gt["oracle_value"]),
                        "normalized_score": normalized_score(
                            result.best_property,
                            random_baseline,
                            gt["oracle_value"],
                        ),
                        "top10_hit_rate": top_k_hit_rate(
                            result.all_properties,
                            gt["top_1_percent_threshold"],
                            k=10,
                        ),
                    }
                )
            mean = _aggregate(seed_results)
            std = _aggregate_std(seed_results)
            all_results.append(
                {
                    "method": method,
                    "eval_budget": budget,
                    **{f"{k}_mean": v for k, v in mean.items()},
                    **{f"{k}_std": v for k, v in std.items()},
                }
            )
    return {"module": "materials", "results": all_results}


def run_benchmark(
    config_path: str | Path,
    module: str = "all",
    output_dir: str | Path | None = None,
) -> Path:
    """Run benchmark(s) and write results."""
    config = load_config(config_path)
    default_path = Path(config_path).parent / "default.yaml"
    merged = {**load_config(default_path), **config} if default_path.exists() else config

    results: dict[str, Any] = {
        "config_hash": config_hash(merged),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "modules": {},
    }

    if module in ("protein", "all"):
        results["modules"]["protein"] = run_protein_benchmark(merged)
    if module in ("climate", "all"):
        results["modules"]["climate"] = run_climate_benchmark(merged)
    if module in ("genomics", "all"):
        results["modules"]["genomics"] = run_genomics_benchmark(merged)
    if module in ("materials", "all"):
        results["modules"]["materials"] = run_materials_benchmark(merged)

    out = Path(output_dir or "results")
    run_dir = out / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    from ml_sci.evaluation.report import write_report

    write_report(results, run_dir / "summary.md")

    return run_dir
