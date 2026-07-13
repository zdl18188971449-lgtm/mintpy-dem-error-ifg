#!/usr/bin/env python3
"""Run repeated HOMA-DEM simulations, confidence intervals, and ablations."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import run_published_model_benchmark as benchmark


ABLATION_VARIANTS = {
    "homa_no_ica": {"enable_ica": False},
    "homa_no_dynamic": {"enable_dynamic": False},
    "homa_no_pgdc": {"enable_pgdc": False},
    "homa_no_igs": {"enable_igs": False},
    "homa_no_graph": {"enable_graph": False},
}

SCENARIOS = [
    "static_nonlinear",
    "wrapped_linear",
    "sparse_dem",
    "dynamic_height",
]

SCENARIO_LABELS = {
    "static_nonlinear": "Static nonlinear",
    "wrapped_linear": "Wrapped / cycle error",
    "sparse_dem": "Sparse DEM error",
    "dynamic_height": "Dynamic height",
}

PUBLISHED_METHODS = {
    "static_nonlinear": [
        "hybrid_optimal_2026",
        "adaptive_ht_2021",
        "linear_huber",
        "ica_2019",
        "adaptive_vce_huber_graph",
    ],
    "wrapped_linear": [
        "hybrid_optimal_2026",
        "linear_huber",
        "igs_cmaes_2021_equivalent",
    ],
    "sparse_dem": [
        "hybrid_optimal_2026",
        "adaptive_vce_huber_graph",
        "pgdc_2025",
    ],
    "dynamic_height": [
        "hybrid_optimal_2026",
        "dynamic_height_2025",
        "adaptive_vce_huber_graph",
    ],
}

METHOD_SHORT_LABELS = {
    "hybrid_optimal_2026": "HOMA-DEM",
    "adaptive_ht_2021": "Adaptive HT",
    "linear_huber": "Linear Huber",
    "ica_2019": "ICA",
    "adaptive_vce_huber_graph": "Current graph",
    "igs_cmaes_2021_equivalent": "IGS-CMAES",
    "pgdc_2025": "PGDC",
    "dynamic_height_2025": "Dynamic SBAS",
    "homa_no_ica": "No ICA",
    "homa_no_dynamic": "No dynamic",
    "homa_no_pgdc": "No PGDC",
    "homa_no_igs": "No IGS",
    "homa_no_graph": "No graph",
}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repeated synthetic validation and HOMA-DEM ablation study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-o", "--output-dir", default="homa_statistical_validation")
    parser.add_argument("--replicates", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=20260713)
    parser.add_argument("--size", type=int, default=18)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def _run_one(arguments: tuple[int, int]) -> list[dict]:
    size, seed = arguments
    rows, _ = benchmark.run_benchmark(size, seed, ABLATION_VARIANTS)
    selected = []
    for row in rows:
        if row["scenario"] not in SCENARIOS:
            continue
        selected.append({"seed": seed, **row})
    return selected


def _mean_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(finite))
    if finite.size < 2:
        return mean, np.nan, np.nan
    half_width = float(
        stats.t.ppf(0.5 + confidence / 2.0, finite.size - 1)
        * np.std(finite, ddof=1)
        / np.sqrt(finite.size)
    )
    return mean, mean - half_width, mean + half_width


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], row["method"])].append(row)
    summary = []
    for (scenario, method), records in sorted(grouped.items()):
        rmse = np.asarray([record["rmse_m"] for record in records])
        runtime = np.asarray([record["runtime_s"] for record in records])
        mean, ci_low, ci_high = _mean_ci(rmse)
        summary.append(
            {
                "scenario": scenario,
                "method": method,
                "label": records[0]["label"],
                "reproduction_status": records[0]["reproduction_status"],
                "n": int(np.sum(np.isfinite(rmse))),
                "mean_rmse_m": mean,
                "std_rmse_m": float(np.nanstd(rmse, ddof=1)),
                "median_rmse_m": float(np.nanmedian(rmse)),
                "q25_rmse_m": float(np.nanquantile(rmse, 0.25)),
                "q75_rmse_m": float(np.nanquantile(rmse, 0.75)),
                "mean_ci95_low_m": max(ci_low, 0.0),
                "mean_ci95_high_m": ci_high,
                "mean_runtime_s": float(np.nanmean(runtime)),
                "median_runtime_s": float(np.nanmedian(runtime)),
            }
        )
    return summary


def _paired_comparisons(rows: list[dict]) -> list[dict]:
    lookup = {
        (row["seed"], row["scenario"], row["method"]): row for row in rows
    }
    output = []
    methods_by_scenario: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        methods_by_scenario[row["scenario"]].add(row["method"])
    seeds = sorted({int(row["seed"]) for row in rows})
    for scenario in SCENARIOS:
        for method in sorted(methods_by_scenario[scenario]):
            if method == "hybrid_optimal_2026":
                continue
            differences = []
            wins = []
            for seed in seeds:
                homa = lookup.get((seed, scenario, "hybrid_optimal_2026"))
                other = lookup.get((seed, scenario, method))
                if homa is None or other is None:
                    continue
                difference = float(other["rmse_m"] - homa["rmse_m"])
                if np.isfinite(difference):
                    differences.append(difference)
                    wins.append(difference > 0)
            mean, ci_low, ci_high = _mean_ci(np.asarray(differences))
            output.append(
                {
                    "scenario": scenario,
                    "comparison_method": method,
                    "n": len(differences),
                    "mean_other_minus_homa_rmse_m": mean,
                    "mean_difference_ci95_low_m": ci_low,
                    "mean_difference_ci95_high_m": ci_high,
                    "homa_win_rate": float(np.mean(wins)) if wins else np.nan,
                }
            )
    return output


def _save_figure(figure: plt.Figure, path: Path, dpi: int) -> None:
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(
        path.with_suffix(".tiff"),
        dpi=dpi,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    figure.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _plot_error_bars(summary: list[dict], output_dir: Path, dpi: int) -> None:
    lookup = {(row["scenario"], row["method"]): row for row in summary}
    figure, axes = plt.subplots(2, 2, figsize=(7.16, 4.55))
    figure.subplots_adjust(
        left=0.16, right=0.985, bottom=0.12, top=0.95, wspace=0.55, hspace=0.60
    )
    for panel, (axis, scenario) in enumerate(zip(axes.flat, SCENARIOS)):
        methods = PUBLISHED_METHODS[scenario]
        positions = np.arange(len(methods))[::-1]
        means = []
        for position, method in zip(positions, methods):
            record = lookup[(scenario, method)]
            mean = record["mean_rmse_m"]
            low = record["mean_ci95_low_m"]
            high = record["mean_ci95_high_m"]
            means.append(mean)
            color = "#B23A2B" if method == "hybrid_optimal_2026" else "#777777"
            marker = "o" if method == "hybrid_optimal_2026" else "s"
            axis.errorbar(
                mean,
                position,
                xerr=np.asarray([[mean - low], [high - mean]]),
                fmt=marker,
                color=color,
                markersize=4.2,
                capsize=2.0,
                elinewidth=0.8,
                markeredgecolor="white",
                markeredgewidth=0.35,
            )
        axis.set_yticks(positions, [METHOD_SHORT_LABELS[m] for m in methods])
        axis.set_xscale("log")
        axis.set_xlim(min(means) / 1.8, max(means) * 2.3)
        axis.set_xlabel("Mean RMSE (m), 95% CI")
        axis.set_title(SCENARIO_LABELS[scenario], fontweight="bold")
        axis.grid(axis="x", which="both", color="#D8D8D8", linewidth=0.45)
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            -0.45,
            1.12,
            f"({chr(ord('a') + panel)})",
            transform=axis.transAxes,
            fontweight="bold",
        )
    _save_figure(figure, output_dir / "homa_repeated_error_bars", dpi)


def _plot_ablation(rows: list[dict], output_dir: Path, dpi: int) -> None:
    lookup = {
        (row["seed"], row["scenario"], row["method"]): float(row["rmse_m"])
        for row in rows
    }
    seeds = sorted({int(row["seed"]) for row in rows})
    methods = list(ABLATION_VARIANTS)
    values = np.full((len(methods), len(SCENARIOS)), np.nan)
    for method_index, method in enumerate(methods):
        for scenario_index, scenario in enumerate(SCENARIOS):
            changes = []
            for seed in seeds:
                full = lookup.get((seed, scenario, "hybrid_optimal_2026"))
                ablated = lookup.get((seed, scenario, method))
                if (
                    full is not None
                    and ablated is not None
                    and full > 0
                    and ablated > 0
                ):
                    changes.append(np.log10(ablated / full))
            if changes:
                values[method_index, scenario_index] = float(np.mean(changes))
    finite_limit = np.nanpercentile(np.abs(values), 90)
    limit = max(float(finite_limit), 1.0)
    figure, axis = plt.subplots(figsize=(7.16, 2.65))
    image = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(
        np.arange(len(SCENARIOS)), [SCENARIO_LABELS[s] for s in SCENARIOS]
    )
    axis.set_yticks(
        np.arange(len(methods)), [METHOD_SHORT_LABELS[m] for m in methods]
    )
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            ratio = 10.0**value
            axis.text(
                column_index,
                row_index,
                f"{ratio:.2f}x" if ratio < 10.0 else f"{ratio:.0f}x",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if abs(value) > 0.55 * limit else "black",
            )
    colorbar = figure.colorbar(image, ax=axis, pad=0.025)
    colorbar.set_label("Mean paired log10 RMSE ratio (ablated / full)")
    axis.set_title("HOMA-DEM module ablation", fontweight="bold", pad=5)
    figure.tight_layout()
    _save_figure(figure, output_dir / "homa_ablation_heatmap", dpi)


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    if args.replicates < 2:
        raise ValueError("--replicates must be at least 2 for error bars")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists, use --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    seeds = [args.seed_start + index for index in range(args.replicates)]
    work = [(args.size, seed) for seed in seeds]
    if args.workers == 1:
        batches = [_run_one(item) for item in work]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            batches = list(executor.map(_run_one, work))
    rows = [row for batch in batches for row in batch]
    summary = _summarize(rows)
    paired = _paired_comparisons(rows)
    _write_csv(output_dir / "repeated_metrics.csv", rows)
    _write_csv(output_dir / "repeated_summary.csv", summary)
    _write_csv(output_dir / "paired_comparisons.csv", paired)
    (output_dir / "validation_config.json").write_text(
        json.dumps(
            {
                "size": args.size,
                "replicates": args.replicates,
                "seeds": seeds,
                "workers": args.workers,
                "confidence_interval": "two-sided t interval for the mean, 95%",
                "paired_design": True,
                "ablation_variants": ABLATION_VARIANTS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _configure_style()
    _plot_error_bars(summary, output_dir, args.dpi)
    _plot_ablation(rows, output_dir, args.dpi)
    print(f"statistical validation written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
