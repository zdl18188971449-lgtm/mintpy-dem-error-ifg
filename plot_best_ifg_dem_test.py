#!/usr/bin/env python3
"""Run a multi-model DEM-error test on the highest-quality MintPy interferogram."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def load_dem_module(script: Path):
    spec = importlib.util.spec_from_file_location("mintpy_dem_error_ifg", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir", help="MintPy work directory containing inputs/")
    parser.add_argument("-o", "--output-dir", default="best_ifg_dem_test")
    parser.add_argument("--looks", type=int, default=4)
    parser.add_argument("--block-rows", type=int, default=16)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["linear_ols", "quadratic_ols", "linear_wls", "linear_huber"],
    )
    return parser.parse_args()


def read_quality_table(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        rows.append(
            {
                "pair": fields[0],
                "coherence": float(fields[1]),
                "btemp": float(fields[2]),
                "bperp": float(fields[3]),
            }
        )
    if not rows:
        raise ValueError(f"no interferogram quality records in {path}")
    return rows


def block_reduce_2d(data: np.ndarray, looks: int, reducer="mean") -> np.ndarray:
    rows = data.shape[-2] // looks
    cols = data.shape[-1] // looks
    trimmed = data[..., : rows * looks, : cols * looks]
    shaped = trimmed.reshape(*trimmed.shape[:-2], rows, looks, cols, looks)
    if reducer == "mean":
        return np.nanmean(shaped, axis=(-3, -1))
    if reducer == "sum":
        return np.nansum(shaped, axis=(-3, -1))
    raise ValueError(reducer)


def multilook_inputs(
    stack: h5py.File,
    geometry: h5py.File,
    mask: h5py.File,
    looks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phase_ds = stack["unwrapPhase"]
    coherence_ds = stack["coherence"]
    num_ifg, length, width = phase_ds.shape
    out_length = length // looks
    out_width = width // looks
    phase_out = np.full((num_ifg, out_length, out_width), np.nan, np.float32)
    coherence_out = np.full_like(phase_out, np.nan)
    incidence_out = np.full((out_length, out_width), np.nan, np.float32)
    range_out = np.full_like(incidence_out, np.nan)
    mask_out = np.zeros((out_length, out_width), bool)
    mask_name = "mask" if "mask" in mask else next(iter(mask.keys()))

    output_block_rows = 64
    for output_y0 in range(0, out_length, output_block_rows):
        output_y1 = min(out_length, output_y0 + output_block_rows)
        y0 = output_y0 * looks
        y1 = output_y1 * looks
        phase = np.asarray(phase_ds[:, y0:y1, : out_width * looks], np.float64)
        coherence = np.asarray(
            coherence_ds[:, y0:y1, : out_width * looks], np.float64
        )
        valid = np.isfinite(phase) & np.isfinite(coherence) & (coherence > 0)
        numerator = block_reduce_2d(
            np.where(valid, phase * coherence, np.nan), looks, "sum"
        )
        denominator = block_reduce_2d(
            np.where(valid, coherence, np.nan), looks, "sum"
        )
        phase_out[:, output_y0:output_y1, :] = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 0,
        )
        coherence_out[:, output_y0:output_y1, :] = block_reduce_2d(
            coherence, looks, "mean"
        )

        incidence = np.asarray(
            geometry["incidenceAngle"][y0:y1, : out_width * looks], np.float64
        )
        slant_range = np.asarray(
            geometry["slantRangeDistance"][y0:y1, : out_width * looks], np.float64
        )
        incidence_out[output_y0:output_y1, :] = block_reduce_2d(
            incidence, looks, "mean"
        )
        range_out[output_y0:output_y1, :] = block_reduce_2d(
            slant_range, looks, "mean"
        )
        mask_block = np.asarray(
            mask[mask_name][y0:y1, : out_width * looks], np.float64
        )
        mask_out[output_y0:output_y1, :] = block_reduce_2d(
            mask_block, looks, "mean"
        ) >= 0.5

    return phase_out, coherence_out, incidence_out, range_out, mask_out


def finite_percentile(arrays, percentile: float, minimum: float) -> float:
    values = np.concatenate(
        [np.abs(array[np.isfinite(array)]).reshape(-1) for array in arrays]
    )
    if values.size == 0:
        return minimum
    return max(float(np.nanpercentile(values, percentile)), minimum)


def main() -> int:
    args = parse_args()
    work_dir = Path(args.work_dir).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = work_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    module = load_dem_module(Path(__file__).with_name("mintpy_dem_error_ifg.py"))
    quality_rows = read_quality_table(work_dir / "coherenceSpatialAvg.txt")
    best = max(quality_rows, key=lambda row: row["coherence"])
    best_pair = tuple(best["pair"].split("_"))

    stack_path = work_dir / "inputs" / "ifgramStack.h5"
    geometry_path = work_dir / "inputs" / "geometryRadar.h5"
    mask_path = work_dir / "maskTempCoh.h5"
    with (
        h5py.File(stack_path, "r") as stack,
        h5py.File(geometry_path, "r") as geometry,
        h5py.File(mask_path, "r") as mask,
    ):
        date_pairs = module.read_date_pairs(stack["date"])
        try:
            target_index = date_pairs.index(best_pair)
        except ValueError as exc:
            raise ValueError(f"best pair {best['pair']} is absent from ifgramStack") from exc

        phase, coherence, incidence, slant_range, spatial_mask = multilook_inputs(
            stack, geometry, mask, args.looks
        )
        selected = (
            np.asarray(stack["dropIfgram"][:], bool)
            if "dropIfgram" in stack
            else np.ones(len(date_pairs), bool)
        )
        wavelength = float(stack.attrs["WAVELENGTH"])
        bperp = np.asarray(stack["bperp"][:], np.float64)

    out_length, out_width = spatial_mask.shape
    inverse_geometry = np.divide(
        1.0,
        slant_range * np.sin(np.deg2rad(incidence)),
        out=np.full_like(slant_range, np.nan, np.float64),
        where=(slant_range > 0) & np.isfinite(incidence),
    )
    dem_coefficient = (
        (-4.0 * np.pi / wavelength)
        * bperp[:, None, None]
        * inverse_geometry[None, :, :]
    )
    years = module.date_years(date_pairs)
    temporal_baseline = np.asarray(
        [years[slave] - years[master] for master, slave in date_pairs]
    )

    results = {}
    original = phase[target_index].copy()
    original[~spatial_mask] = np.nan
    for model_name in args.models:
        spec = module.MODEL_SPECS[model_name]
        deformation, labels = module.build_deformation_design(
            date_pairs, spec.poly_order
        )
        dem_error = np.full((out_length, out_width), np.nan, np.float32)
        dem_phase = np.zeros((out_length, out_width), np.float32)
        corrected = original.copy()
        residual_sum = 0.0
        residual_count = 0

        for y0 in range(0, out_length, args.block_rows):
            y1 = min(out_length, y0 + args.block_rows)
            phase_block = phase[:, y0:y1, :].reshape(len(date_pairs), -1)
            coefficient_block = dem_coefficient[:, y0:y1, :].reshape(
                len(date_pairs), -1
            )
            coherence_block = coherence[:, y0:y1, :].reshape(
                len(date_pairs), -1
            )
            mask_block = spatial_mask[y0:y1, :].reshape(-1)
            weights = np.ones_like(phase_block, np.float64)
            weights[:, ~mask_block] = 0.0
            if spec.solver == "coherence":
                coh = np.clip(coherence_block, 1e-3, 0.999)
                weights *= np.clip(
                    coh**2 / np.maximum(1.0 - coh**2, 1e-6), 0.0, 1e3
                )

            solution, valid = module.solve_model_block(
                phase_block,
                coefficient_block,
                deformation,
                selected,
                weights,
                phase_velocity=spec.phase_velocity,
                temporal_baseline=temporal_baseline,
                robust=spec.solver == "huber",
            )
            dem_values = solution[:, 0]
            target_component = np.zeros(phase_block.shape[1], np.float64)
            target_component[valid] = (
                coefficient_block[target_index, valid] * dem_values[valid]
            )
            target_corrected = phase_block[target_index] - target_component
            target_corrected[~valid] = np.nan

            prediction = np.zeros_like(phase_block)
            if np.any(valid):
                prediction[:, valid] = (
                    coefficient_block[:, valid] * dem_values[valid][None, :]
                    + deformation @ solution[valid, 1:].T
                )
            residual = phase_block - prediction
            fit_valid = np.isfinite(residual[selected][:, valid])
            residual_values = residual[selected][:, valid][fit_valid]
            residual_sum += float(np.sum(residual_values**2))
            residual_count += int(residual_values.size)

            dem_error[y0:y1] = dem_values.reshape(y1 - y0, out_width)
            dem_phase[y0:y1] = target_component.reshape(y1 - y0, out_width)
            corrected[y0:y1] = target_corrected.reshape(y1 - y0, out_width)

        valid_target = np.isfinite(original) & np.isfinite(corrected)
        results[model_name] = {
            "dem_error": dem_error,
            "dem_phase": dem_phase,
            "corrected": corrected,
            "valid_pixels": int(np.sum(valid_target)),
            "original_rms": float(np.sqrt(np.nanmean(original[valid_target] ** 2))),
            "corrected_rms": float(np.sqrt(np.nanmean(corrected[valid_target] ** 2))),
            "dem_phase_rms": float(np.sqrt(np.nanmean(dem_phase[valid_target] ** 2))),
            "model_residual_rms": float(np.sqrt(residual_sum / residual_count)),
            "terms": labels,
        }

    h5_path = output_dir / f"{best['pair']}_dem_model_comparison.h5"
    with h5py.File(h5_path, "w") as file:
        file.attrs["DATE12"] = best["pair"]
        file.attrs["MEAN_COHERENCE"] = best["coherence"]
        file.attrs["TEMPORAL_BASELINE_DAYS"] = best["btemp"]
        file.attrs["PERP_BASELINE_M"] = best["bperp"]
        file.attrs["LOOKS"] = args.looks
        file.create_dataset("originalPhase", data=original, compression="gzip")
        file.create_dataset(
            "coherence", data=coherence[target_index], compression="gzip"
        )
        file.create_dataset("mask", data=spatial_mask, compression="gzip")
        for model_name, result in results.items():
            group = file.create_group(model_name)
            group.create_dataset(
                "correctedPhase", data=result["corrected"], compression="gzip"
            )
            group.create_dataset(
                "demPhase", data=result["dem_phase"], compression="gzip"
            )
            group.create_dataset(
                "demError", data=result["dem_error"], compression="gzip"
            )
            for key in (
                "valid_pixels",
                "original_rms",
                "corrected_rms",
                "dem_phase_rms",
                "model_residual_rms",
            ):
                group.attrs[key.upper()] = result[key]

    csv_path = output_dir / f"{best['pair']}_model_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "model",
                "valid_pixels",
                "original_rms_rad",
                "corrected_rms_rad",
                "dem_phase_rms_rad",
                "all_ifg_model_residual_rms_rad",
            ]
        )
        for model_name, result in results.items():
            writer.writerow(
                [
                    model_name,
                    result["valid_pixels"],
                    result["original_rms"],
                    result["corrected_rms"],
                    result["dem_phase_rms"],
                    result["model_residual_rms"],
                ]
            )

    phase_limit = finite_percentile(
        [original] + [result["corrected"] for result in results.values()], 98, 0.1
    )
    component_limit = finite_percentile(
        [result["dem_phase"] for result in results.values()], 98, 0.05
    )
    dem_limit = finite_percentile(
        [result["dem_error"] for result in results.values()], 98, 1.0
    )

    model_names = list(results)
    fig, axes = plt.subplots(3, len(model_names) + 1, figsize=(20, 11), constrained_layout=True)
    phase_norm = TwoSlopeNorm(vmin=-phase_limit, vcenter=0.0, vmax=phase_limit)
    component_norm = TwoSlopeNorm(
        vmin=-component_limit, vcenter=0.0, vmax=component_limit
    )
    dem_norm = TwoSlopeNorm(vmin=-dem_limit, vcenter=0.0, vmax=dem_limit)

    phase_image = axes[0, 0].imshow(original, cmap="RdBu_r", norm=phase_norm)
    axes[0, 0].set_title("Original unwrapped phase")
    coherence_image = axes[1, 0].imshow(
        np.where(spatial_mask, coherence[target_index], np.nan),
        cmap="viridis",
        vmin=0.5,
        vmax=1.0,
    )
    axes[1, 0].set_title("Spatial coherence")
    corrected_values = [results[name]["corrected_rms"] for name in model_names]
    colors = ["#3B82F6", "#EAB308", "#10B981", "#EF4444"][: len(model_names)]
    axes[2, 0].barh(model_names, corrected_values, color=colors)
    axes[2, 0].invert_yaxis()
    axes[2, 0].set_xlabel("Corrected RMS (rad)")
    axes[2, 0].set_title("Selected interferogram")
    axes[2, 0].grid(axis="x", alpha=0.25)

    component_image = None
    dem_image = None
    for column, model_name in enumerate(model_names, start=1):
        result = results[model_name]
        axes[0, column].imshow(result["corrected"], cmap="RdBu_r", norm=phase_norm)
        axes[0, column].set_title(
            f"{model_name}\ncorrected RMS={result['corrected_rms']:.3f} rad"
        )
        component_image = axes[1, column].imshow(
            result["dem_phase"], cmap="PuOr_r", norm=component_norm
        )
        axes[1, column].set_title(
            f"DEM phase\nRMS={result['dem_phase_rms']:.3f} rad"
        )
        dem_image = axes[2, column].imshow(
            result["dem_error"], cmap="BrBG", norm=dem_norm
        )
        axes[2, column].set_title(
            f"DEM error\nmean={np.nanmean(result['dem_error']):.2f} m"
        )

    for axis in axes.flat:
        if axis is not axes[2, 0]:
            axis.set_xticks([])
            axis.set_yticks([])
    fig.colorbar(phase_image, ax=axes[0, :], shrink=0.72, label="Phase (rad)")
    fig.colorbar(coherence_image, ax=axes[1, 0], shrink=0.72, label="Coherence")
    fig.colorbar(component_image, ax=axes[1, 1:], shrink=0.72, label="DEM phase (rad)")
    fig.colorbar(dem_image, ax=axes[2, 1:], shrink=0.72, label="DEM error (m)")
    fig.suptitle(
        f"Highest-quality interferogram: {best['pair']} | "
        f"mean coherence={best['coherence']:.4f}, Btemp={best['btemp']:.0f} d, "
        f"Bperp={best['bperp']:.1f} m | {args.looks}x{args.looks} looks",
        fontsize=15,
    )

    png_path = output_dir / f"{best['pair']}_dem_correction_comparison.png"
    pdf_path = output_dir / f"{best['pair']}_dem_correction_comparison.pdf"
    fig.savefig(png_path, dpi=220, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)

    print(f"selected: {best}")
    for model_name, result in results.items():
        print(
            model_name,
            f"valid={result['valid_pixels']}",
            f"corrected_rms={result['corrected_rms']:.6f}",
            f"model_residual_rms={result['model_residual_rms']:.6f}",
        )
    for path in (h5_path, csv_path, png_path, pdf_path):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
