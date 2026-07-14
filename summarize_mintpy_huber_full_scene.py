#!/usr/bin/env python3
"""Summarize a full-scene MintPy OLS versus compatible Huber run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


OLS_COLOR = "#4C78A8"
HUBER_COLOR = "#B64E3B"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def robust_limit(values: np.ndarray, percentile: float = 98.0) -> float:
    finite = np.abs(values[np.isfinite(values)])
    return float(np.percentile(finite, percentile)) if finite.size else 1.0


def collect(project_dir: Path):
    ols_dir = project_dir / "mintpy_ols_current_input"
    huber_dir = project_dir / "homa_huber_mintpy_compatible"
    with h5py.File(ols_dir / "demErr.h5") as ols_dem_file, h5py.File(
        huber_dir / "demErr.h5"
    ) as huber_dem_file, h5py.File(
        ols_dir / "timeseriesResidual.h5"
    ) as ols_res_file, h5py.File(
        huber_dir / "timeseriesResidual.h5"
    ) as huber_res_file, h5py.File(project_dir / "velocity.h5") as ols_vel_file, h5py.File(
        huber_dir / "velocity.h5"
    ) as huber_vel_file:
        ols_dem = ols_dem_file["dem"][:].astype(np.float64)
        huber_dem = huber_dem_file["dem"][:].astype(np.float64)
        valid = (ols_dem != 0) | (huber_dem != 0)
        ols_dem[~valid] = np.nan
        huber_dem[~valid] = np.nan
        ols_velocity = ols_vel_file["velocity"][:].astype(np.float64) * 1000.0
        huber_velocity = huber_vel_file["velocity"][:].astype(np.float64) * 1000.0
        ols_velocity[~valid] = np.nan
        huber_velocity[~valid] = np.nan

        num_date = ols_res_file["timeseries"].shape[0]
        dates = [value.decode() for value in ols_res_file["date"][:]]
        ols_sumsq = np.zeros(num_date)
        huber_sumsq = np.zeros(num_date)
        ols_abs_sample = []
        huber_abs_sample = []
        for y0 in range(0, valid.shape[0], 128):
            y1 = min(y0 + 128, valid.shape[0])
            block_mask = valid[y0:y1]
            if not np.any(block_mask):
                continue
            ols_res = ols_res_file["timeseries"][:, y0:y1].astype(np.float64)[:, block_mask]
            huber_res = huber_res_file["timeseries"][:, y0:y1].astype(np.float64)[:, block_mask]
            ols_sumsq += np.sum(ols_res**2, axis=1)
            huber_sumsq += np.sum(huber_res**2, axis=1)
            ols_abs_sample.append(np.abs(ols_res[:, ::50]).reshape(-1))
            huber_abs_sample.append(np.abs(huber_res[:, ::50]).reshape(-1))

    count = int(np.sum(valid))
    ols_abs = np.concatenate(ols_abs_sample)
    huber_abs = np.concatenate(huber_abs_sample)
    dem_difference = huber_dem - ols_dem
    velocity_difference = huber_velocity - ols_velocity
    dem_offset = float(np.nanmean(dem_difference))
    velocity_offset = float(np.nanmean(velocity_difference))
    metrics = {
        "valid_pixels": count,
        "num_dates": num_date,
        "dem_difference_mean_m": dem_offset,
        "dem_difference_rmse_m": float(np.sqrt(np.nanmean(dem_difference**2))),
        "dem_difference_centered_rmse_m": float(
            np.sqrt(np.nanmean((dem_difference - dem_offset) ** 2))
        ),
        "residual_rms_ols_m": float(np.sqrt(np.sum(ols_sumsq) / (count * num_date))),
        "residual_rms_huber_m": float(np.sqrt(np.sum(huber_sumsq) / (count * num_date))),
        "residual_abs_median_ols_m": float(np.median(ols_abs)),
        "residual_abs_median_huber_m": float(np.median(huber_abs)),
        "residual_abs_p95_ols_m": float(np.percentile(ols_abs, 95)),
        "residual_abs_p95_huber_m": float(np.percentile(huber_abs, 95)),
        "velocity_difference_mean_mm_per_year": velocity_offset,
        "velocity_difference_rmse_mm_per_year": float(
            np.sqrt(np.nanmean(velocity_difference**2))
        ),
        "velocity_difference_centered_rmse_mm_per_year": float(
            np.sqrt(np.nanmean((velocity_difference - velocity_offset) ** 2))
        ),
    }
    return {
        "metrics": metrics,
        "dates": dates,
        "ols_date_rms": np.sqrt(ols_sumsq / count),
        "huber_date_rms": np.sqrt(huber_sumsq / count),
        "ols_dem": ols_dem,
        "huber_dem": huber_dem,
        "dem_difference": dem_difference,
        "ols_velocity": ols_velocity,
        "huber_velocity": huber_velocity,
        "velocity_difference": velocity_difference,
    }


def save_outputs(result, output_dir: Path, label: str, dpi: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = result["metrics"]
    (output_dir / "mintpy_huber_full_scene_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="ascii"
    )
    with (output_dir / "mintpy_huber_per_date_residual.csv").open(
        "w", newline="", encoding="ascii"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["date", "ols_rms_m", "huber_rms_m"])
        writer.writerows(zip(result["dates"], result["ols_date_rms"], result["huber_date_rms"]))

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    dem_limit = max(robust_limit(result["ols_dem"]), robust_limit(result["huber_dem"]))
    dem_diff_limit = robust_limit(result["dem_difference"])
    vel_limit = max(robust_limit(result["ols_velocity"]), robust_limit(result["huber_velocity"]))
    vel_diff_limit = robust_limit(result["velocity_difference"])
    panels = [
        (result["ols_dem"], "a  MintPy OLS DEM error", "RdBu_r", -dem_limit, dem_limit, "m"),
        (result["huber_dem"], "b  Huber DEM error", "RdBu_r", -dem_limit, dem_limit, "m"),
        (result["dem_difference"], "c  Huber - OLS", "RdBu_r", -dem_diff_limit, dem_diff_limit, "m"),
        (result["ols_velocity"], "d  MintPy velocity", "RdBu_r", -vel_limit, vel_limit, "mm yr$^{-1}$"),
        (result["huber_velocity"], "e  Huber-corrected velocity", "RdBu_r", -vel_limit, vel_limit, "mm yr$^{-1}$"),
        (result["velocity_difference"], "f  Huber - OLS", "RdBu_r", -vel_diff_limit, vel_diff_limit, "mm yr$^{-1}$"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.5), constrained_layout=True)
    for axis, (data, title, cmap, vmin, vmax, unit) in zip(axes.flat, panels):
        image = axis.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xticks([])
        axis.set_yticks([])
        colorbar = fig.colorbar(image, ax=axis, fraction=0.045, pad=0.02)
        colorbar.set_label(unit)
    fig.suptitle(f"{label}: identical MintPy workflow, estimator-only comparison", fontsize=9)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"mintpy_huber_full_scene_spatial.{extension}", dpi=dpi)
    plt.close(fig)

    x = np.arange(len(result["dates"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    axes[0].plot(x, result["ols_date_rms"] * 1000, "o-", color=OLS_COLOR, label="MintPy OLS")
    axes[0].plot(x, result["huber_date_rms"] * 1000, "s-", color=HUBER_COLOR, label="Huber")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(result["dates"], rotation=45, ha="right", fontsize=6)
    axes[0].set_ylabel("Residual RMS (mm)")
    axes[0].set_title("a  Acquisition-wise residual", loc="left", fontweight="bold")
    axes[0].legend()
    categories = ["L2 RMS", "Median |r|", "95th |r|"]
    ols_values = np.asarray(
        [metrics["residual_rms_ols_m"], metrics["residual_abs_median_ols_m"], metrics["residual_abs_p95_ols_m"]]
    ) * 1000
    huber_values = np.asarray(
        [metrics["residual_rms_huber_m"], metrics["residual_abs_median_huber_m"], metrics["residual_abs_p95_huber_m"]]
    ) * 1000
    width = 0.36
    indices = np.arange(3)
    axes[1].bar(indices - width / 2, ols_values, width, color=OLS_COLOR, label="MintPy OLS")
    axes[1].bar(indices + width / 2, huber_values, width, color=HUBER_COLOR, label="Huber")
    axes[1].set_xticks(indices, categories)
    axes[1].set_ylabel("Residual magnitude (mm)")
    axes[1].set_title("b  Objective and robust summaries", loc="left", fontweight="bold")
    axes[1].legend()
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"mintpy_huber_full_scene_residuals.{extension}", dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    save_outputs(collect(args.project_dir), args.output_dir, args.label, args.dpi)


if __name__ == "__main__":
    main()
