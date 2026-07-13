#!/usr/bin/env python3
"""Draw IEEE TGRS-style HOMA-DEM comparison and performance figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from string import ascii_lowercase

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HOMA_COLOR = "#B23A2B"
CURRENT_COLOR = "#35618D"
PUBLISHED_COLOR = "#777777"
INPUT_COLOR = "#B8B8B8"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create TGRS-style HOMA-DEM figures from benchmark outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "benchmark_dir", nargs="?", default="published_model_benchmark"
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.0,
            "axes.titlesize": 7.2,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def save_figure(figure: plt.Figure, base_path: Path, dpi: int) -> None:
    figure.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(
        base_path.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"}
    )
    figure.savefig(base_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.025,
        0.975,
        f"({label})",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.8},
    )


def plot_spatial_plate(maps: dict[str, np.ndarray], output_dir: Path, dpi: int) -> None:
    static_limit = float(np.nanpercentile(np.abs(maps["static_truth"]), 99))
    wrapped_limit = float(np.nanpercentile(np.abs(maps["wrapped_truth"]), 99))
    sparse_limit = float(np.nanmax(np.abs(maps["sparse_truth"])))
    dynamic_limit = float(np.nanmax(np.abs(maps["dynamic_truth"])))
    rows = [
        {
            "label": "Static nonlinear",
            "unit": "DEM error (m)",
            "limit": static_limit,
            "panels": [
                ("static_truth", "Reference"),
                ("static_hybrid", "HOMA-DEM"),
                ("static_adaptive", "Adaptive HT [2021]"),
                ("static_ica", "ICA [2019]"),
            ],
        },
        {
            "label": "Wrapped / cycle error",
            "unit": "DEM error (m)",
            "limit": wrapped_limit,
            "panels": [
                ("wrapped_truth", "Reference"),
                ("wrapped_hybrid", "HOMA-DEM"),
                ("wrapped_linear", "Linear Huber"),
                ("wrapped_igs", "IGS-CMAES [2021]"),
            ],
        },
        {
            "label": "Sparse DEM error",
            "unit": "DEM error (m)",
            "limit": sparse_limit,
            "panels": [
                ("sparse_truth", "Reference"),
                ("sparse_hybrid", "HOMA-DEM"),
                ("sparse_graph", "Current graph model"),
                ("sparse_pgdc", "PGDC [2025]"),
            ],
        },
        {
            "label": "Dynamic height",
            "unit": "Height change (m)",
            "limit": dynamic_limit,
            "panels": [
                ("dynamic_truth", "Reference"),
                ("dynamic_hybrid", "HOMA-DEM"),
                ("dynamic_estimate", "Dynamic SBAS [2025]"),
                ("dynamic_static", "Static graph model"),
            ],
        },
    ]

    figure = plt.figure(figsize=(7.16, 7.05))
    grid = figure.add_gridspec(
        4,
        5,
        width_ratios=(1.0, 1.0, 1.0, 1.0, 0.045),
        left=0.075,
        right=0.965,
        bottom=0.045,
        top=0.985,
        wspace=0.10,
        hspace=0.25,
    )
    panel_index = 0
    for row_index, row in enumerate(rows):
        image = None
        for column_index, (key, title) in enumerate(row["panels"]):
            axis = figure.add_subplot(grid[row_index, column_index])
            image = axis.imshow(
                maps[key],
                cmap="RdBu_r",
                vmin=-row["limit"],
                vmax=row["limit"],
                interpolation="nearest",
                rasterized=True,
            )
            axis.set_title(title, pad=2.0)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.55)
                spine.set_color("black")
            _panel_label(axis, ascii_lowercase[panel_index])
            panel_index += 1
        color_axis = figure.add_subplot(grid[row_index, 4])
        colorbar = figure.colorbar(image, cax=color_axis)
        colorbar.ax.tick_params(labelsize=6.0, length=2.0, width=0.5)
        colorbar.outline.set_linewidth(0.5)
        colorbar.set_label(row["unit"], fontsize=6.4, labelpad=2.0)
        figure.text(
            0.015,
            0.875 - row_index * 0.237,
            row["label"],
            rotation=90,
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold",
        )

    save_figure(figure, output_dir / "tgrs_homa_spatial_comparison", dpi)
    plt.close(figure)


def _load_metrics(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["rmse_m"] = float(row["rmse_m"])
        row["runtime_s"] = float(row["runtime_s"])
    return rows


def plot_performance(metrics: list[dict], output_dir: Path, dpi: int) -> None:
    scenario_specs = [
        (
            "static_nonlinear",
            "Static nonlinear",
            ["hybrid_optimal_2026", "adaptive_ht_2021", "linear_huber", "ica_2019", "adaptive_vce_huber_graph"],
        ),
        (
            "wrapped_linear",
            "Wrapped / cycle error",
            ["hybrid_optimal_2026", "linear_huber", "igs_cmaes_2021_equivalent"],
        ),
        (
            "sparse_dem",
            "Sparse DEM error",
            ["hybrid_optimal_2026", "adaptive_vce_huber_graph", "pgdc_2025"],
        ),
        (
            "dynamic_height",
            "Dynamic height",
            ["hybrid_optimal_2026", "dynamic_height_2025", "adaptive_vce_huber_graph"],
        ),
    ]
    method_labels = {
        "hybrid_optimal_2026": "HOMA-DEM",
        "adaptive_ht_2021": "Adaptive HT [2021]",
        "linear_huber": "Linear Huber",
        "ica_2019": "ICA [2019]",
        "igs_cmaes_2021_equivalent": "IGS-CMAES [2021]",
        "pgdc_2025": "PGDC [2025]",
        "dynamic_height_2025": "Dynamic SBAS [2025]",
        "adaptive_vce_huber_graph": "Current graph model",
    }
    metric_lookup = {(row["scenario"], row["method"]): row for row in metrics}
    figure, axes = plt.subplots(2, 2, figsize=(7.16, 4.45))
    figure.subplots_adjust(left=0.175, right=0.985, bottom=0.12, top=0.96, wspace=0.58, hspace=0.62)

    for panel_index, (axis, (scenario, title, methods)) in enumerate(
        zip(axes.flat, scenario_specs)
    ):
        records = [metric_lookup[(scenario, method)] for method in methods]
        positions = np.arange(len(methods))[::-1]
        minimum = min(record["rmse_m"] for record in records)
        maximum = max(record["rmse_m"] for record in records)
        for position, method, record in zip(positions, methods, records):
            if method == "hybrid_optimal_2026":
                color, marker, size = HOMA_COLOR, "o", 26
            elif method == "adaptive_vce_huber_graph":
                color, marker, size = CURRENT_COLOR, "D", 22
            else:
                color, marker, size = PUBLISHED_COLOR, "s", 20
            axis.hlines(position, minimum / 1.7, record["rmse_m"], color="#C7C7C7", linewidth=0.7)
            axis.scatter(
                record["rmse_m"],
                position,
                color=color,
                marker=marker,
                s=size,
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
            axis.annotate(
                f"{record['rmse_m']:.3g}",
                (record["rmse_m"], position),
                xytext=(4, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=6.0,
            )
        labels = [
            f"{method_labels[method]}  ({record['runtime_s']:.2f} s)"
            for method, record in zip(methods, records)
        ]
        axis.set_yticks(positions, labels)
        axis.set_xscale("log")
        axis.set_xlim(max(minimum / 1.7, 1e-4), maximum * 2.2)
        axis.set_xlabel("RMSE (m), log scale")
        axis.set_title(title, fontweight="bold", pad=4.0)
        axis.grid(axis="x", which="both", color="#D8D8D8", linewidth=0.45)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.text(
            -0.47,
            1.12,
            f"({ascii_lowercase[panel_index]})",
            transform=axis.transAxes,
            fontsize=7.5,
            fontweight="bold",
            ha="left",
            va="top",
        )

    save_figure(figure, output_dir / "tgrs_homa_performance", dpi)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    configure_style()
    output_dir = Path(args.benchmark_dir).resolve()
    maps = dict(np.load(output_dir / "published_benchmark_maps.npz"))
    metrics = _load_metrics(output_dir / "published_method_metrics.csv")
    plot_spatial_plate(maps, output_dir, args.dpi)
    plot_performance(metrics, output_dir, args.dpi)
    print(f"TGRS figures written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
