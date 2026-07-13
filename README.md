# MintPy interferogram DEM-error correction

`mintpy_dem_error_ifg.py` reads a MintPy `ifgramStack.h5`, estimates one
residual DEM-error value per pixel from the selected interferogram network,
reconstructs the DEM phase for every interferogram, and writes corrected
interferogram stacks without modifying the source file.

The HDF5/block-processing workflow follows the practical pattern used by
[vceSAR](https://github.com/ymcmrs/vceSAR). The DEM inversion is implemented
independently in the interferogram domain.

## Observation model

For interferogram `master_slave` and pixel `p`:

```text
unwrapPhase = K_dem * delta_h + G_deformation * parameters + residual

K_dem = (-4*pi/wavelength) * Bperp / (slantRange * sin(incidenceAngle))
```

The constant deformation term cancels between the two acquisitions. The tool
supports polynomial, periodic, and step deformation terms.

## Classic model presets

| Model | Temporal model | Estimator |
|---|---|---|
| `linear_ols` | linear velocity | equal-weight OLS |
| `quadratic_ols` | velocity + acceleration | equal-weight OLS, MintPy default class |
| `linear_velocity` | linear | phase-velocity OLS |
| `linear_wls` / `quadratic_wls` | linear / quadratic | coherence-derived WLS |
| `linear_huber` / `quadratic_huber` | linear / quadratic | Huber IRLS |
| `linear_vce` / `quadratic_vce` | linear / quadratic | external vceSAR variance WLS |

Coherence weights use the classic relative phase-variance relation
`weight proportional to coherence^2 / (1 - coherence^2)`. VCE models accept a
dataset shaped `(num_ifg,)`, `(num_ifg, length, width)`, or vceSAR-style
`model_parameters` shaped `(num_ifg, >=3)`, where sill plus nugget is used as
the variance.

## Preflight

Run a read-only check first:

```bash
/home/zdl/miniforge3/envs/SAR/bin/python \
  /mnt/d/WSL/ALOS/Scripts/mintpy_dem_error_ifg.py \
  inputs/ifgramStack.h5 \
  -g inputs/geometryRadar.h5 \
  --models linear_ols quadratic_ols linear_wls linear_huber \
  --dry-run
```

The preflight reports network components and the representative design-matrix
rank. A disconnected network is not silently rejected because a temporal model
can make the numerical system full rank, but the resulting cross-component
calibration is model-controlled and should not be treated as an independent
observation.

## Run

```bash
/home/zdl/miniforge3/envs/SAR/bin/python \
  /mnt/d/WSL/ALOS/Scripts/mintpy_dem_error_ifg.py \
  inputs/ifgramStack.h5 \
  -g inputs/geometryRadar.h5 \
  -o dem_error_ifg_compare \
  --models linear_ols quadratic_ols linear_velocity linear_wls linear_huber \
  --mask maskTempCoh.h5 \
  --min-coherence 0.3
```

For vceSAR weighting:

```bash
... --models linear_vce quadratic_vce \
    --variance-file variogramStackModel.h5 \
    --variance-dataset model_parameters
```

To compare the classic vceSAR covariance models, first generate separate
variance-model files with `vce_modeling.py --model spherical`, `gaussian`, or
`exponential`, then run the corresponding `*_vce` preset into separate output
directories. The DEM tool deliberately consumes the variance result instead of
duplicating vceSAR's variogram fitting code.

Optional deformation terms:

```bash
... --periodic 1.0 0.5 --step-date 20080512
```

## Outputs

Each model writes:

- `ifgramStack_demErr_MODEL.h5`: MintPy-compatible stack with corrected
  `unwrapPhase` and all other source datasets copied unchanged.
- `demComponent_MODEL.h5`: `demPhase` for every interferogram and the 2D
  `demError` map in meters.
- `ifgram_metrics_MODEL.csv`: per-interferogram RMS diagnostics.
- `model_comparison.csv`: model-level DEM statistics and residual RMS.

Invalid pixels retain their original phase in the corrected stack, receive a
zero DEM phase component, and are stored as `NaN` in the DEM-error map.

## Interpretation limits

- The method estimates relative DEM error in the spatial reference frame of
  the input interferograms.
- A connected, redundant interferogram network is required for defensible
  separation of deformation and DEM error.
- Atmospheric, ionospheric, orbit, and unwrapping residuals can leak into the
  DEM estimate. Correct these first or use robust/VCE weighting.
- A lower in-sample residual is not sufficient evidence of a better model.
  Compare held-out interferograms, baseline correlation, and external
  ICESat-2/LiDAR/GNSS elevations.
- OLS, coherence WLS, Huber, and VCE-WLS minimize different objective
  functions. Compare their DEM maps and independent validation errors; do not
  rank a weighted model only by its unweighted training RMS.

## Tests

```bash
/home/zdl/miniforge3/envs/SAR/bin/python \
  /mnt/d/WSL/ALOS/Scripts/test_mintpy_dem_error_ifg.py
```

The repository also includes a real `16 x 16` HFT-473 MintPy fixture and a
full-scene comparison for the highest-quality interferogram. See
[`TESTING.md`](TESTING.md) for reproducible commands and expected results.
