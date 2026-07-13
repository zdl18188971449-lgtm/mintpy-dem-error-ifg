#!/usr/bin/env python3
"""Run and summarize the synthetic DEM-error benchmark across terrain types."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

import mintpy_dem_error_ifg as correction
import run_synthetic_dem_benchmark as benchmark


METRIC_FIELDS = [
    "dem_rmse_m",
    "dem_mae_m",
    "truth_dem_std_m",
    "dem_nrmse",
    "dem_correlation",
    "dem_phase_rmse_rad",
    "corrected_truth_rmse_rad",
    "network_model_residual_rms_rad",
]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DEM-error correction over multiple synthetic terrain types"
    )
    parser.add_argument("-o", "--output", default="multi_terrain_benchmark")
    parser.add_argument(
        "--terrains", nargs="+", choices=benchmark.TERRAIN_TYPES, default=benchmark.TERRAIN_TYPES
    )
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=20260713)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--num-acquisitions", type=int, default=12)
    parser.add_argument("--max-neighbor", type=int, default=3)
    parser.add_argument("--baseline-time-correlation", type=float, default=0.0)
    parser.add_argument("--atmosphere-scale", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--acceleration-scale", type=float, default=1.0)
    parser.add_argument("--outlier-cycles", type=float, default=1.0)
    parser.add_argument("--adaptive-bic-margin", type=float, default=2.0)
    parser.add_argument("--graph-lambda", type=float, default=8.0)
    parser.add_argument("--graph-iterations", type=int, default=40)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(correction.MODEL_SPECS),
        default=benchmark.DEFAULT_MODELS,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run_cases(args: argparse.Namespace, root: Path) -> list[dict]:
    all_rows = []
    for terrain in args.terrains:
        for replicate in range(1, args.replicates + 1):
            seed = args.seed_base + replicate - 1
            run_dir = root / terrain / f"rep_{replicate:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            command = [
                "-o",
                str(run_dir),
                "--terrain-type",
                terrain,
                "--seed",
                str(seed),
                "--size",
                str(args.size),
                "--num-acquisitions",
                str(args.num_acquisitions),
                "--max-neighbor",
                str(args.max_neighbor),
                "--baseline-time-correlation",
                str(args.baseline_time_correlation),
                "--atmosphere-scale",
                str(args.atmosphere_scale),
                "--noise-scale",
                str(args.noise_scale),
                "--acceleration-scale",
                str(args.acceleration_scale),
                "--outlier-cycles",
                str(args.outlier_cycles),
                "--adaptive-bic-margin",
                str(args.adaptive_bic_margin),
                "--graph-lambda",
                str(args.graph_lambda),
                "--graph-iterations",
                str(args.graph_iterations),
                "--models",
                *args.models,
                "--overwrite",
            ]
            log_path = run_dir / "run.log"
            with log_path.open("w", encoding="utf-8") as log_file, contextlib.redirect_stdout(log_file):
                status = benchmark.main(command)
            if status != 0:
                raise RuntimeError(f"benchmark failed for {terrain} replicate {replicate}")

            with (run_dir / "benchmark_metrics.csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            best = min(rows, key=lambda row: float(row["dem_rmse_m"]))
            print(
                f"{terrain} replicate {replicate}: {best['model']} "
                f"RMSE={float(best['dem_rmse_m']):.3f} m"
            )
            for row in rows:
                converted = {
                    "terrain": terrain,
                    "replicate": replicate,
                    "seed": seed,
                    **row,
                }
                for field in METRIC_FIELDS:
                    converted[field] = float(converted[field])
                converted["valid_pixels"] = int(converted["valid_pixels"])
                all_rows.append(converted)
    return all_rows


def summarize(rows: list[dict], root: Path) -> list[dict]:
    detail_fields = ["terrain", "replicate", "seed", *[key for key in rows[0] if key not in {"terrain", "replicate", "seed"}]]
    with (root / "multi_terrain_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["terrain"], row["model"])].append(row)

    summary_rows = []
    for terrain in dict.fromkeys(row["terrain"] for row in rows):
        for model in dict.fromkeys(row["model"] for row in rows):
            group = grouped[(terrain, model)]
            summary = {"terrain": terrain, "model": model, "replicates": len(group)}
            for field in METRIC_FIELDS:
                values = [float(row[field]) for row in group]
                summary[f"{field}_mean"] = statistics.fmean(values)
                summary[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            summary_rows.append(summary)

    with (root / "multi_terrain_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def plot_rmse(summary: list[dict], terrains: list[str], models: list[str], root: Path) -> None:
    rmse_lookup = {
        (row["terrain"], row["model"]): float(row["dem_rmse_m_mean"])
        for row in summary
    }
    nrmse_lookup = {
        (row["terrain"], row["model"]): float(row["dem_nrmse_mean"])
        for row in summary
    }
    rmse_matrix = np.asarray(
        [[rmse_lookup[(terrain, model)] for terrain in terrains] for model in models]
    )
    nrmse_matrix = np.asarray(
        [[nrmse_lookup[(terrain, model)] for terrain in terrains] for model in models]
    )
    global_mean = np.mean(rmse_matrix, axis=1)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(max(17, 2.6 * len(terrains) + 8), max(6, 0.62 * len(models))),
        gridspec_kw={"width_ratios": [1.35, 1.35, 1.0]},
        constrained_layout=True,
    )

    for axis, matrix, title, colorbar_label in [
        (axes[0], rmse_matrix, "Absolute DEM error", "DEM RMSE (m)"),
        (axes[1], nrmse_matrix, "Relative DEM error", "DEM NRMSE"),
    ]:
        image = axis.imshow(matrix, cmap="YlGnBu", aspect="auto")
        axis.set_xticks(np.arange(len(terrains)), terrains)
        axis.set_yticks(
            np.arange(len(models)), [benchmark.display_model_name(name) for name in models]
        )
        axis.set_title(title)
        for row_index in range(len(models)):
            for column_index in range(len(terrains)):
                value = matrix[row_index, column_index]
                color = "white" if value > np.nanmedian(matrix) else "black"
                axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", color=color)
        figure.colorbar(image, ax=axis, label=colorbar_label, shrink=0.85)

    order = np.argsort(global_mean)
    axes[2].barh(np.arange(len(models)), global_mean[order], color="#287271")
    axes[2].set_yticks(
        np.arange(len(models)),
        [benchmark.display_model_name(models[index]) for index in order],
    )
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Mean DEM RMSE across terrains (m)")
    axes[2].set_title("Cross-terrain ranking")
    for index, value in enumerate(global_mean[order]):
        axes[2].text(value, index, f" {value:.2f}", va="center")
    axes[2].set_xlim(0, max(global_mean) * 1.15)

    figure.savefig(root / "multi_terrain_rmse.png", dpi=180)
    figure.savefig(root / "multi_terrain_rmse.pdf")
    plt.close(figure)


def plot_gallery(summary: list[dict], terrains: list[str], root: Path) -> None:
    best_model = {}
    for terrain in terrains:
        candidates = [row for row in summary if row["terrain"] == terrain]
        best_model[terrain] = min(candidates, key=lambda row: float(row["dem_rmse_m_mean"]))["model"]

    cases = []
    for terrain in terrains:
        run_dir = root / terrain / "rep_01"
        with h5py.File(run_dir / "inputs" / "geometryRadar.h5", "r") as file:
            height = np.asarray(file["height"][:], dtype=np.float64)
        with h5py.File(run_dir / "simulationTruth.h5", "r") as file:
            truth = np.asarray(file["demError"][:], dtype=np.float64)
            mask = np.asarray(file["validMask"][:], dtype=bool)
        model = best_model[terrain]
        with h5py.File(run_dir / "correction" / f"demComponent_{model}.h5", "r") as file:
            estimate = np.asarray(file["demError"][:], dtype=np.float64)
        cases.append((terrain, model, height, truth, estimate - truth, mask))

    truth_limit = max(float(np.nanpercentile(np.abs(case[3][case[5]]), 98.0)) for case in cases)
    error_limit = max(float(np.nanpercentile(np.abs(case[4][case[5]]), 98.0)) for case in cases)
    figure, axes = plt.subplots(3, len(terrains), figsize=(3.3 * len(terrains), 9.5), constrained_layout=True)
    for column, (terrain, model, height, truth, error, mask) in enumerate(cases):
        relief = height - np.nanmin(height)
        axes[0, column].imshow(relief, cmap="terrain")
        axes[0, column].set_title(f"{terrain}\nrelief={np.ptp(height):.0f} m")
        truth_image = axes[1, column].imshow(
            np.where(mask, truth, np.nan), cmap="RdBu_r", vmin=-truth_limit, vmax=truth_limit
        )
        axes[1, column].set_title("True DEM error")
        error_image = axes[2, column].imshow(
            np.where(mask, error, np.nan), cmap="RdBu_r", vmin=-error_limit, vmax=error_limit
        )
        axes[2, column].set_title(
            f"Best error\n{benchmark.display_model_name(model)}"
        )
        for row in range(3):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    figure.colorbar(truth_image, ax=axes[1, :].tolist(), label="DEM error (m)", shrink=0.75)
    figure.colorbar(error_image, ax=axes[2, :].tolist(), label="Estimation error (m)", shrink=0.75)
    figure.savefig(root / "multi_terrain_gallery.png", dpi=180)
    figure.savefig(root / "multi_terrain_gallery.pdf")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    if args.replicates < 1:
        raise ValueError("--replicates must be at least 1")
    root = Path(args.output).resolve()
    if root.exists() and any(root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty, use --overwrite: {root}")
    root.mkdir(parents=True, exist_ok=True)

    rows = run_cases(args, root)
    summary = summarize(rows, root)
    plot_rmse(summary, args.terrains, args.models, root)
    plot_gallery(summary, args.terrains, root)
    config = vars(args).copy()
    config["output"] = str(root)
    (root / "batch_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"\nCombined outputs: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
