# MintPy-Compatible DEM-Error Comparison

## Why the earlier full-scene comparison was weaker

The earlier HOMA full-scene experiment estimated DEM error directly from
`ifgramStack.h5`. MintPy 1.6.2 instead estimates it from the displacement time
series after network inversion and all enabled corrections preceding
`correct_topography`. For HFT-473 and HFT-474 this input is
`timeseries_tropHgt.h5`.

The earlier comparison therefore changed the input domain, nuisance errors,
temporal model, weighting, spatial regularization, masking, and objective at
the same time. Its lower performance does not isolate the DEM-error estimator.
The sparse and disconnected HFT-474 interferogram network further weakens a
direct interferogram-domain separation.

## Strict comparison contract

`mintpy_dem_error_timeseries.py` delegates the following operations to the
installed MintPy code without modification:

- input time-series selection and date/reference metadata;
- template parsing, excluded dates, polynomial/step/periodic deformation terms;
- perpendicular baseline referencing, incidence angle, and slant range;
- pixel validity checks, memory-based row blocks, HDF5 layout, and metadata;
- DEM displacement subtraction and residual time-series generation.

Only the solution of MintPy's per-pixel system is selectable:

```text
d(t) = B_perp(t) / (R sin(theta)) * delta_h + G_defo(t) * x + residual(t)
```

`mintpy_ols` uses MintPy's original L2 least squares. `homa_huber` starts from
the same OLS solution and applies Huber iteratively reweighted least squares to
the same acquisitions and design matrix. No interferogram coherence weights,
graph regularization, adaptive temporal order, ICA, global amplitude scaling,
or post-hoc correction is introduced in this controlled comparison.

## HFT-47 commands

Run from each MintPy work directory so a relative `exclude_date.txt` is
resolved exactly as it is by `smallbaselineApp.py`. The current HFT projects do
not contain this file, so all acquisitions are used.

Native control for HFT-473:

```bash
mkdir -p estimator_comparison/mintpy_ols
python /mnt/d/WSL/ALOS/mintpy-dem-error-ifg/mintpy_dem_error_timeseries.py \
  timeseries_tropHgt.h5 -g inputs/geometryRadar.h5 \
  -t smallbaselineApp.cfg --estimator mintpy_ols \
  -o estimator_comparison/mintpy_ols/timeseries_tropHgt_demErr.h5 \
  --dem-err-file estimator_comparison/mintpy_ols/demErr.h5
```

Estimator replacement:

```bash
mkdir -p estimator_comparison/homa_huber
python /mnt/d/WSL/ALOS/mintpy-dem-error-ifg/mintpy_dem_error_timeseries.py \
  timeseries_tropHgt.h5 -g inputs/geometryRadar.h5 \
  -t smallbaselineApp.cfg --estimator homa_huber \
  -o estimator_comparison/homa_huber/timeseries_tropHgt_demErr.h5 \
  --dem-err-file estimator_comparison/homa_huber/demErr.h5
```

Use the same commands in HFT-474. Keep each estimator in a separate directory
because MintPy always writes `timeseriesResidual.h5` beside the corrected time
series. Do not use `--update` with `homa_huber`; MintPy's native configuration
metadata does not contain the replacement estimator name.

After correction, run the same native MintPy `reference_date`, `velocity`,
`geocode`, and plotting steps on each corrected time series. Do not compare the
new result with the earlier `ifgramStack_homaDem.h5` products as if only the
estimator differed.

## Required validation

The primary checks are held-out acquisition prediction, DEM agreement with
independent ICESat-2/LiDAR/control elevations after removal of a common datum,
and stability across date subsets. Training residual RMS is secondary because
a more flexible or robust objective is not optimized for the same unweighted
quantity as OLS.

## HFT-47 pilot result

A file-level test was run on valid 16 by 16 subsets selected from each current
project. The wrapper's `mintpy_ols` mode and a direct MintPy 1.6.2 command were
identical for the DEM, corrected time series, residual time series, and file
attributes; the maximum absolute array difference was zero.

| Project | Subset origin (x, y) | DEM difference RMSE, Huber vs OLS (m) | Residual RMS OLS / Huber (m) | Leave-one-date-out RMSE OLS / Huber (m) |
|---|---:|---:|---:|---:|
| HFT-473 | (256, 128) | 1.055 | 0.01204 / 0.01232 | 0.01938 / 0.01998 |
| HFT-474 | (1024, 512) | 6.495 | 0.01286 / 0.01366 | 0.02779 / 0.02700 |

Huber is slightly worse on the HFT-473 pilot and slightly better in held-out
prediction on HFT-474. HFT-474's residualized DEM sensitivity is about 3.8
times lower than HFT-473 because of its smaller perpendicular-baseline span.
Small displacement-fit changes can therefore produce large DEM changes. This
pilot supports estimator isolation, but it is not sufficient evidence for a
full-scene accuracy claim.
